import os
import tiktoken
import torch
import json
import math
import random
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union

from datasets import Dataset, IterableDataset, DatasetDict
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from helpers.hparams import IGNORE_INDEX
from helpers.dataset_template.template import get_template_and_fix_tokenizer
from helpers.log_helper import Logger
from helpers.sim_args import SimArguments
from helpers.io_helper import read_json
from helpers.hparams import SimcopilotConfig, INSERT_SIGN_MAP
from helpers.base_helper import get_interaction_move_from_object


def load_data_file(data_path) -> dict:
    import pickle 
    
    if data_path.endswith('.json'):
        data = read_json(data_path)
    elif data_path.endswith('.pkl'):
        with open (data_path, 'rb') as f:
            data = pickle.load(f)
    elif data_path.endswith('.pth'):
        data = torch.load(data_path)
    else:
        Logger.error("Currently only support [json, pkl, pth] format dataset!")
        raise ValueError("Currently only support [json, pkl, pth] format dataset!")
    
    return data

def get_dataset_from_json(args: SimArguments=None, data_json=None, with_learner=False) -> Dataset:
    """
    return:
        DatasetDict(train=result_data)
    """
    attrs = ['query', 'response']
    if args.additional_data_attr is not None:
        attrs.extend(args.additional_data_attr)
    
    data_dict: Dict[str, List[Any]] = {
        a: [] for a in attrs
    }
    
    for d in data_json:
        for a in attrs:
            if a not in d:
                Logger.error(f"Can't find {a} in {d}")
                raise ValueError(f"Can't find {a} in {d}")
            
            data_dict[a].append(d[a])
    
    result_data = Dataset.from_dict(data_dict)
    return result_data

def split_dataset(dataset: Dataset, args: SimArguments) -> Dict[str, "Dataset"]:
    if args.eval_nums > 0:
        if args.eval_nums > 1.0:
            args.eval_nums = math.ceil(args.eval_nums)
        else:
            args.eval_nums = math.ceil(args.eval_nums * len(dataset))
        
        if args.eval_nums >=  len(dataset):
            return {"train_dataset": None, "eval_dataset": dataset}
        else:
            train_val = dataset.train_test_split(
                test_size=args.eval_nums, shuffle=True
            )
            return {"train_dataset": train_val["train"], "eval_dataset": train_val["test"]}
    else:
        return {"train_dataset": dataset, "eval_dataset": None}

def preprocess_dataset(
    tokenizer: "PreTrainedTokenizer",
    data_args: "SimArguments",
) -> Union["Dataset", "IterableDataset"]:
    with_learner = data_args.is_learner
    
    if with_learner:
        assert data_args.data_path.endswith('.pkl') or data_args.data_path.endswith('.pth')
    
    data_json = load_data_file(data_args.data_path)
    if data_args.shuffle_data:
        random.shuffle(data_json)
    if data_args.max_samples > 0:
        data_json = data_json[:min(data_args.max_samples, len(data_json))]
    
    # Dataset
    dataset = get_dataset_from_json(args=data_args, data_json=data_json, with_learner=with_learner)
    
    # template, will add eos, pad token
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    
    # construct prompt
    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples['query'])):
            query = examples['query'][i]
            response = examples['response'][i]
            history = None
            system = None
            
            if data_args.additional_data_attr is not None:
                addition = {
                    a: examples[a][i]
                    for a in data_args.additional_data_attr
                }
            else:
                addition = {}
            
            yield query, response, history, system, addition

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        def tokenize(text: str, max_length: int):
            old_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"
            result = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=False,
            )['input_ids']
            tokenizer.padding_side = old_padding_side
            return result
        
        def tokenize_pure(text: str):
            result = tokenizer(
                text,
                add_special_tokens=False,
            )['input_ids']
            return result
        
        model_inputs = {
            "input_ids": [], 
            "attention_mask": [], 
            "labels": []
        }
        if data_args.additional_data_attr is not None:
            model_inputs.update({
                a: [] for a in data_args.additional_data_attr
            })

        for query, response, history, system, addition in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue
            
            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                
            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]
            
            if len(input_ids) > data_args.generation_max_length:
                input_ids = input_ids[:data_args.generation_max_length]
                labels = labels[:data_args.generation_max_length]
            
            attention_mask = [1] * len(input_ids)
            
            if not data_args.is_train:
                input_ids = source_ids
                attention_mask = [1] * len(input_ids)
                
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
            
            if data_args.additional_data_attr is not None:
                for a in data_args.encode_data_attr:
                    addition[a] = tokenize_pure(addition[a])
                
                for a,v in addition.items():
                    model_inputs[a].append(v)
            
        if not data_args.is_train and data_args.use_wp_token:
            model_inputs.pop('labels')


        return model_inputs


    def print_supervised_dataset_example(example: Dict[str, List[int]]) -> None:
        Logger.info(f"query_ids:\n{example['input_ids']}")
        Logger.info(f"query:\n{tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")
        if 'lbaels' in example:
            Logger.info(f"response_ids:\n{example['labels']}")
            Logger.info(f"response:\n{tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example['labels'])), skip_special_tokens=False)}")
        
        if data_args.additional_data_attr is not None:
            for a in data_args.additional_data_attr:
                if a in data_args.encode_data_attr:
                    Logger.info(f"{a}:\n{example[a]}")
                    Logger.info(f"{a}_ids:\n{tokenizer.decode(example[a], skip_special_tokens=False)}")
                else:
                    Logger.info(f"{a}:\n{example[a]}")

    seqargs = Seq2SeqTrainingArguments(output_dir='.')
    with seqargs.main_process_first(desc="dataset map pre-processing"):
        remove_columns = ['query', 'response']
            
        kwargs = {}
        kwargs = dict(
            num_proc=1,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_supervised_dataset,
            batched=True,
            remove_columns=remove_columns,
            **kwargs
        )
        
        try:
            print_supervised_dataset_example(next(iter(dataset)))
        except StopIteration:
            raise RuntimeError("Empty dataset!")

        return dataset
