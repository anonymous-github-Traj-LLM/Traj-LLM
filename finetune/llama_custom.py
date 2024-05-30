from typing import List, Optional, Tuple, Union
import time
import torch
from peft import PeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from finetune.learner_lm import combines_learner_output_with_inputs
from helpers.log_helper import Logger

class LlamaCustom(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        # motion_embeds=None,
        # interaction_embeds=None,
        # object_embeds=None,
        # map_embeds=None,
        sim_embeds=None,
        **kwargs,
    ):
        t0 = time.time()
        if input_ids is not None:
            pass
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs.update(
            {
                "sim_embeds": sim_embeds,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sim_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if sim_embeds is not None and past_key_values is None:
            inputs_embeds, attention_mask, _ = combines_learner_output_with_inputs(
                input_ids,
                inputs_embeds,
                sim_embeds,
                attention_mask,
            )
            position_ids = None

        # from modeling_llama.py
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
