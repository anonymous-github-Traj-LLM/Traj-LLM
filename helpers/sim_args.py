import os
import sys
from typing import Any, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from helpers.base_helper import get_current_time
from helpers.log_helper import Logger
from helpers.config_helper import ConfigHelper
from helpers.hparams import INSERT_SIGN_MAP

def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> "SimArguments":
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return ConfigHelper.load_config(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()[0]

def parse_train_args(
    args: Optional[Dict[str, Any]] = None
) -> "SimArguments":
    parser = HfArgumentParser(SimArguments)
    return _parse_args(parser, args)

@dataclass
class SimArguments:
    RUN_MODE_ENUM = [
        'train', 'test', 
        'train_w_learner', 'test_w_learner', 
        'train_w_map_embeds', 'test_w_map_embeds',
        'train_w_scorelearner', 'test_w_scorelearner'
    ]
    TRAIN_MODE_ENUM = ['train', 'train_w_learner', 'train_w_map_embeds', 'train_w_scorelearner']
    LEARNER_MODE_ENUM = [
        'train_w_learner', 'test_w_learner', 
        'train_w_map_embeds', 'test_w_map_embeds',
        'train_w_scorelearner', 'test_w_scorelearner'
    ]
    MAP_EMBEDS_MODE_ENUM = ['train_w_map_embeds', 'test_w_map_embeds']
    
    MAP_MODE_ENUM = ['img', 'pointset', 'text']
    
    exp_name: str = field(default="default")
    
    notes: Optional[str] = field(default='default')
    
    run_mode: Optional[str] = field(
        default='train'
    )
    map_mode: Optional[str] = field(
        default='text'
    )
    map_embed_dim: Optional[int] = field(
        default=4096
    )
    
    force_use_eval: Optional[bool] = field(default=False)
    force_use_cache: Optional[bool] = field(default=False)
    
    base_model: str = field(
        default="models/Llama-2-7b-chat-hf"
    )
    use_wp_token: Optional[bool] = field(default=False)
    data_path: Optional[str] = field(default=None)
    additional_data_attr: Optional[str] = field(default=None)
    encode_data_attr: Optional[str] = field(default=None)
    eval_nums: Optional[float] = field(default=0)
    template: Optional[str] = field(default="llama")
    output_dir: str = field(default="output")
    weight_dir: Optional[str] = field(default=None)
    do_sample: Optional[bool] = field(default=True)
    
    lora_target: Optional[str] = field(default='q_proj,v_proj,k_proj,o_proj')
    lora_rank: int = field(default=8)
    lora_alpha: float = field(default=32.0)
    lora_dropout: float = field(default=0.1)
    
    max_samples: Optional[int] = field(default=0)
    num_epochs: Optional[int] = field(default=1)
    batch_size: Optional[int] = field(default=8)
    micro_batch_size: Optional[int] = field(default=1)
    val_batch_size: Optional[int] = field(default=1)
    eval_steps: Optional[int] = field(default=10)
    learning_rate: Optional[float] = field(default=2e-5)
    
    generation_max_length: Optional[int] = field(default=4096)
    save_steps: Optional[int] = field(default=200)
    logging_steps: Optional[int] = field(default=10)

    tokenizer_init_padding_side: Optional[str] = field(default='right')
    
    loss_weight: Optional[Dict[str, float]] = field(default=None)
    shuffle_data: Optional[bool] = field(default=True)
    
    temperature: Optional[float] = field(
        default=1.1,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.75,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=40,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )

    def __post_init__(self):        
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        
        # output relate
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.output_dir = os.path.join(self.output_dir, get_current_time()+"_"+self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # output log
        self.log_path = os.path.join(self.output_dir, "record.log")
        Logger.get_logger(__name__, self.log_path)
        Logger.info("start logging...")
        
        # template check
        if self.template not in INSERT_SIGN_MAP:
            Logger.error(f"template {self.template} is not supported!")
            raise ValueError(f"template {self.template} is not supported!")

        # additional_data_attr reform
        if self.additional_data_attr is not None:
            if len(self.additional_data_attr) == 0:
                self.additional_data_attr = None
            else:
                self.additional_data_attr = list(map(lambda x:x.strip(), self.additional_data_attr.split(",")))
        
        # add wp_token_mat
        if self.use_wp_token and 'wp_token_mat' not in self.additional_data_attr:
            self.additional_data_attr = self.additional_data_attr + ['wp_token_mat']
        
        # encode_data_attr reform and check
        if self.encode_data_attr is not None:
            if len(self.encode_data_attr) == 0:
                self.encode_data_attr = []
            else:
                self.encode_data_attr = list(map(lambda x:x.strip(), self.encode_data_attr.split(",")))
        if self.additional_data_attr is not None and self.encode_data_attr is None:
            self.encode_data_attr = []
        
        # run mode check
        if self.run_mode not in SimArguments.RUN_MODE_ENUM:
            Logger.error(f"run_mode {self.run_mode} is not supported!")
            raise ValueError(f"run_mode {self.run_mode} is not supported!")
        
        # map mode check
        if self.map_mode not in SimArguments.MAP_MODE_ENUM:
            Logger.error(f"map_mode {self.map_mode} is not supported!")
            raise ValueError(f"map_mode {self.map_mode} is not supported!")
        
        # lora check
        if self.lora_target is not None:
            self.lora_target = list(map(lambda x:x.strip(), self.lora_target.split(",")))
        if 'qwen' in self.base_model.lower():
            for lt in self.lora_target:
                if lt not in ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"]:
                    Logger.error(f"lora_target {lt} is not supported in Qwen.")
                    raise ValueError(f"lora_target {lt} is not supported in Qwen.")
        elif 'llama' in self.base_model.lower():
            for lt in self.lora_target:
                if lt not in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                    Logger.error(f"lora_target {lt} is not supported in llama.")
                    raise ValueError(f"lora_target {lt} is not supported in llama.")
        
        # assist params
        self.is_train = self.run_mode in SimArguments.TRAIN_MODE_ENUM
        self.is_learner = self.run_mode in SimArguments.LEARNER_MODE_ENUM
        self.is_map_embeds = self.run_mode in SimArguments.MAP_EMBEDS_MODE_ENUM
        
        # print
        print_str = "SimArguments:\n"
        for k,v in vars(self).items():
            print_str += f"    {k} = {v}\n"
        Logger.info(print_str)