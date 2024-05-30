import os
import json
import torch
import time
import numpy as np
from torch import nn
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput, PredictionOutput
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import DataCollatorForSeq2Seq

from helpers.hparams import IGNORE_INDEX
from helpers.log_helper import Logger
from helpers.test_helper import *
from helpers.sim_args import SimArguments

def pad_tensors_to_maxN_np(tensor_list, pad_value=-100):
    max_N = max(tensor.shape[0] for tensor in tensor_list)
    
    target_T = 30

    padded_tensors = np.full((len(tensor_list), max_N, target_T, 2), pad_value, dtype=np.float64)

    for i, tensor in enumerate(tensor_list):

        N, T, _ = tensor.shape
        
        if T > target_T:
            T = target_T
        
        padded_tensors[i, :N, :T, :] = tensor[:N, :T, :]
    
    return padded_tensors


@dataclass
class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    
    args: "SimArguments" = None

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        if self.args.encode_data_attr is not None:
            attr_max_len_in_batch = {
                a: max([len(f[a]) for f in features]) for a in self.args.encode_data_attr
            }
            for feature in features:
                for a in self.args.encode_data_attr:
                    remainder = [self.tokenizer.pad_token_id] * (attr_max_len_in_batch[a] - len(feature[a]))
                    feature[a] = np.concatenate([feature[a], remainder]).astype(np.int64)
            

        if self.args.map_mode == 'pointset' and self.args.additional_data_attr is not None and 'map_embeds' in self.args.additional_data_attr:
            if len(features[0]['map_embeds']) == 0 or len(features[0]['map_embeds'][0]) == 0:
                raise ValueError("map_embeds is empty")
            
            max_point_num = max([len(f['map_embeds']) for f in features])
            point_dim = len(features[0]['map_embeds'][0])
            packing_meta = [IGNORE_INDEX] * point_dim
            for f in features:
                remainder = [packing_meta] * (max_point_num - len(f['map_embeds']))
                f['map_embeds'] = f['map_embeds'] + remainder
                
        if self.args.additional_data_attr is not None and 'wp_token_mat' in self.args.additional_data_attr:
            if len(features[0]['wp_token_mat']) == 0 or len(features[0]['wp_token_mat'][0]) == 0:
                raise ValueError("wp_token_mat is empty")
            
            pad_result = pad_tensors_to_maxN_np([np.array(f['wp_token_mat']) for f in features])
            pad_result_num = pad_result.tolist()
            for i,f in enumerate(features):
                f['wp_token_mat'] = pad_result_num[i]
            pass

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


def decode_generation_seqeunces(tokenizer, token_sequences):
    token_sequences = np.where(
        token_sequences != -100, token_sequences, tokenizer.pad_token_id
    )
    return tokenizer.batch_decode(
        token_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        return_outputs = True
        loss, outputs = super().compute_loss(model, inputs, return_outputs)
        
        labels = inputs["labels"].detach().clone().cpu() if "labels" in inputs else None # inputs["labels"] is origin, less than outputs.logits, gap is 100
        first_non_idx = np.where(labels != IGNORE_INDEX)[1][0]+100
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        predictions = outputs.logits.argmax(dim=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions[:,first_non_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        Logger.logger.info(f"decoded_labels: {decoded_labels}")
        Logger.logger.info(f"decoded_preds: {decoded_preds}")
        Logger.logger.info(f"loss: {loss}")
        Logger.logger.info("")
        
        return loss
        

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None # backup labels
        if self.args.predict_with_generate: # True
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["labels"] = inputs["labels"][:, :prompt_len] # truncate the labels instead of padding the inputs

        t0 = time.time()
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        for i in range(generated_tokens.shape[0]):
            Logger.logger.info(f"generated_tokens: {self.tokenizer.decode(generated_tokens[i][prompt_len:])}")
        Logger.logger.info(f"prediction_step time: {time.time() - t0}")
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        Logger.logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
            
