import os
import math
import torch
from types import MethodType
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    InfNanRemoveLogitsProcessor, 
    LogitsProcessorList
)
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)

from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled
    
try:
    from transformers.utils import (
        is_torch_bf16_cpu_available,
        is_torch_bf16_gpu_available,
        is_torch_cuda_available,
        is_torch_npu_available
    )
    _is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
    _is_bf16_available = is_torch_bf16_gpu_available() or is_torch_bf16_cpu_available
except ImportError:
    _is_fp16_available = torch.cuda.is_available()
    _is_bf16_available = torch.cuda.is_bf16_supported()
    
from helpers.log_helper import Logger
from finetune.learner_lm import LearnerLMLoRA
from finetune.llama_custom import LlamaCustom
from helpers.sim_args import SimArguments

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp", "ln_1", "ln_2"]

def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32

def setup_seed(seed = 42):
    import numpy as np
    import torch
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # may cause slower training


def get_logits_processor() -> LogitsProcessorList:
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor



def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def prepare_model_for_training(
    args,
    model
):
    output_layer_name = "lm_head"
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    Logger.info("Gradient checkpointing enabled. Turn off use_cache.")

    if hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
                return output_layer.__class__.forward(self, x.to(output_layer.weight.dtype)).to(torch.float32)

            output_layer.forward = MethodType(forward_in_fp32, output_layer)

    return model


def init_adapter(
    args: SimArguments,
    model,
    default_torch_type = torch.float16,
):
    if args.is_train:
        Logger.info("Fine-tuning method: LoRA")
    else:
        Logger.info("Testing LoRA")
    
    target_modules = args.lora_target
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # this can not change, will add unsupported param error
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    
    if args.is_learner:
        model = LearnerLMLoRA(model, lora_config, args)
        
        if args.weight_dir is not None:
            checkpoints_to_merge = args.weight_dir
            lora_weights = torch.load(os.path.join(checkpoints_to_merge, 'adapter_model.bin'))
            learner_weights = torch.load(os.path.join(checkpoints_to_merge, 'learner_encoder.pth'))
            out_embd_prj_weights = torch.load(os.path.join(checkpoints_to_merge, 'out_embd_prj.pth'))
            map_encoder_weights = torch.load(os.path.join(checkpoints_to_merge, 'map_encoder.pth'))
            lora_results = set_peft_model_state_dict(model, lora_weights)
            model.learner_encoder.load_state_dict(learner_weights)
            model.out_embd_prj.load_state_dict(out_embd_prj_weights)
            model.map_encoder.load_state_dict(map_encoder_weights)
            Logger.info(f"Load adapter_model.bin, learner_encoder.pth, out_embd_prj.pth, map_encoder.pth from {args.weight_dir}!")
            
        else:
            Logger.info("No weight_dir was given, use random init!")
        
        if args.force_use_cache:
            model.config.use_cache = True
            Logger.info("LearnerLMLoRA Turn on use_cache.")
        else:
            model.config.use_cache = False
            Logger.info("LearnerLMLoRA Turn off use_cache.")
        

    else: # no learner
        
        if args.weight_dir is not None:
            model = PeftModel.from_pretrained(model, args.weight_dir)
            model = model.merge_and_unload()
            Logger.info(f"Load adapter_model.bin from {args.weight_dir}!")
        else:
            model = get_peft_model(model, lora_config)
            if id(model.peft_config) != id(model.base_model.peft_config):
                model.base_model.peft_config = model.peft_config
            Logger.info("No weight_dir was given, use random init!")

    return model

    
def load_model_and_tokenizer(args: SimArguments):
    
    default_torch_type = torch.bfloat16
    
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "use_auth_token": None,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        split_special_tokens=False,
        padding_side=args.tokenizer_init_padding_side,
        **config_kwargs
    )
    if args.tokenizer_init_padding_side == "left":
        Logger.warning(f"In LLama-factory: training with left-padded tensors in fp16 precision may cause overflow!")
    Logger.info(f"tokenizer.padding_side: {tokenizer.padding_side}")

    
    config = AutoConfig.from_pretrained(args.base_model, **config_kwargs)
    
    if args.is_train:
        setattr(config, "torch_dtype", default_torch_type)
    else:
        default_torch_type = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
    Logger.info(f"Using torch dtype: {default_torch_type}")
    
    # Fix config (for Qwen)
    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)
    
    # load base model
    if not args.is_learner:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            torch_dtype=default_torch_type,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map="auto",
            **config_kwargs
        )
        
    else:            
        if getattr(config, "model_type", None) == "llama":
            model = LlamaCustom.from_pretrained(
                args.base_model,
                config=config,
                torch_dtype=default_torch_type,
                low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
                device_map="auto",
                **config_kwargs
            )
        else:
            Logger.error(f"Model type is not supported. {getattr(config, 'model_type', None)}")
            raise ValueError(f"Model type is not supported. {getattr(config, 'model_type', None)}")
    
    

    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)
        
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
        


    model = prepare_model_for_training(args, model=model) if args.is_train else model
    model = init_adapter(args, model, default_torch_type)

    model = model.train() if args.is_train else model.eval()

    if args.force_use_eval:
        model = model.eval()
        
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    if not args.is_train:
        model.requires_grad_(False)
        model = model.to(default_torch_type)
    else:
        if args.use_wp_token:
            model.base_model.state_judge_head.requires_grad_(True)
            model.base_model.score_head.requires_grad_(True)
            model.base_model.coord_head.requires_grad_(True)
            model.base_model.turn_flag_head.requires_grad_(True)
            model.base_model.param_head.requires_grad_(True)
            model.base_model.action_head.requires_grad_(True)
            model.base_model.numerical_coord_head.requires_grad_(True)
    
    trainable_params, all_param = count_parameters(model)
    Logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not args.is_train:
        Logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")
        
    for name, param in model.named_parameters():
        if param.requires_grad:
            Logger.info(f"{name} requires_grad: True")
            # break

    return model, tokenizer
    
