from typing import List, Optional, Tuple, Union
from peft.config import PeftConfig

import torch
import time
import torch.nn as nn
from peft import PeftModelForCausalLM
from dataclasses import dataclass
from finetune.modules import *
from finetune.map_encode.pointset_encoder import PointSetEncoderConfig, PointSetEncoder
from helpers.log_helper import Logger
from helpers.hparams import INSERT_SIGN_MAP, IGNORE_INDEX
from helpers.base_helper import split_subsequence, find_subsequence
from helpers.sim_args import SimArguments


@dataclass
class SimcopilotInput:
    map_embeds: torch.FloatTensor
    motion_ids: torch.FloatTensor
    interaction_ids: List[List[torch.FloatTensor]]
    object_ids: List[List[torch.FloatTensor]]
    

class LearnerEncoderConfig:
    out_fts: int = 2048
    
    d_word_vec: int = 512
    n_position: int = 4096
    
    d_model: int = 512
    d_inner: int = 1024
    n_head: int = 8
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1


class LearnerBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, dropout=0.1):
        super(LearnerBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q, k, v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class LearnerEncoder(nn.Module):
    def __init__(self,
                 n_src_vocab,
                 pad_idx,
                 learner_config: LearnerEncoderConfig
                 ) -> None:
        super().__init__()
        
        self.src_word_emb = nn.Embedding(n_src_vocab, learner_config.d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(learner_config.d_word_vec, n_position=learner_config.n_position)
        self.drop_out = nn.Dropout(learner_config.dropout)
        
        self.motion_learner = LearnerBlock(
            learner_config.d_model, 
            learner_config.d_inner, 
            learner_config.n_head, 
            learner_config.d_k, 
            learner_config.d_v, 
            learner_config.dropout
        )
        self.interaction_learner = LearnerBlock(
            learner_config.d_model, 
            learner_config.d_inner, 
            learner_config.n_head, 
            learner_config.d_k, 
            learner_config.d_v, 
            learner_config.dropout
        )
        self.road_learner = LearnerBlock(
            learner_config.d_model, 
            learner_config.d_inner, 
            learner_config.n_head, 
            learner_config.d_k, 
            learner_config.d_v, 
            learner_config.dropout
        )
        
        
        self.out_fts = learner_config.d_model

    
    def forward(self, input:SimcopilotInput):
        bs = input.map_embeds.shape[0]
        motion_ids = input.motion_ids
        motion_embeds = self.src_word_emb(motion_ids)
        motion_pos_embeds = self.drop_out(self.position_enc(motion_embeds))
        map_embeds = input.map_embeds
        map_embeds = map_embeds[:,None,:]
        
        res_Jns = []
        for i in range(bs):
            cur_motion_pos_embeds = motion_pos_embeds[i,...].unsqueeze(0)
            cur_map_embeds = map_embeds[i,...].unsqueeze(0)
            
            curobjects = input.object_ids[i]
            Mns = []
            for j in range(len(curobjects)):
                obj_fcs = self.drop_out(self.position_enc(self.src_word_emb(curobjects[j].unsqueeze(0))))
                
                Mi,_ = self.motion_learner(obj_fcs, cur_motion_pos_embeds, cur_motion_pos_embeds)
                Mns.append(Mi)

            curinteractions = input.interaction_ids[i]
            Ins = []
            for j in range(1, len(Mns)):
                Si_fcs = self.drop_out(self.position_enc(self.src_word_emb(curinteractions[j-1].unsqueeze(0))))
                M_cat = torch.cat([Mns[0], Mns[j]], dim=1)
                Ii,_ = self.interaction_learner(Si_fcs, M_cat, M_cat)
                Ins.append(Ii)
            
            # list
            Jns = []
            for j in range(len(Ins)):
                Ji,_ = self.road_learner(Ins[j], cur_map_embeds, cur_map_embeds)
                Jns.append(Ji)
            
            cat_Jns = torch.cat(Jns, dim=1)
            res_Jns.append(cat_Jns)
        
        return res_Jns
            
            
        
        
class LearnerLMLoRA(PeftModelForCausalLM):
    def __init__(self, 
                 model, 
                 peft_config: PeftConfig,
                 args: SimArguments=None,
                 adapter_name="default"
                 ):
        super().__init__(model, peft_config, adapter_name)
        
        self.args = args
        
        pointsetCfg = PointSetEncoderConfig()
        pointsetCfg.in_channels = args.map_embed_dim
        self.map_encoder = PointSetEncoder(pointsetCfg)
        
        self.learner_encoder = LearnerEncoder(
            n_src_vocab=self.config.vocab_size,
            pad_idx=self.config.pad_token_id,
            learner_config=LearnerEncoderConfig()
        )
        self.out_embd_prj = nn.Linear(self.learner_encoder.out_fts, self.config.hidden_size)
        self.to(model.device)
        self.modules_to_save = ["learner_encoder","out_embd_prj", "map_encoder"]
        self.tokenizer_padding_side = "left" 
        self.tokenizer_pad_token_id = 0 
        
        Logger.info(f"LearnerLMLoRA.modules_to_save: {self.modules_to_save}")
        Logger.info(f"LearnerLMLoRA.generation_config: {self.generation_config}")
    
    def encode_siminput(
        self,
        input_ids,
        attention_mask,
        labels,
        motion_ids,
        interaction_ids,
        object_ids,
        map_embeds,
        ):
        sep_ids = INSERT_SIGN_MAP[self.base_model.config.model_type]['pos_sign_ids']

        object_list = []
        interaction_list = []
        for i in range(input_ids.shape[0]):
            one_object_list = split_subsequence(object_ids[i], sep_ids)
            one_interaction_list = split_subsequence(interaction_ids[i], sep_ids)
            object_list.append(one_object_list)
            interaction_list.append(one_interaction_list)

        sim_input = SimcopilotInput(
            map_embeds=map_embeds,
            motion_ids=motion_ids,
            interaction_ids=interaction_list,
            object_ids=object_list
        )
        learner_encode_outputs = self.learner_encoder(sim_input)
        proj_output_list = []
        for i in range(len(learner_encode_outputs)):
            proj_output_list.append(self.out_embd_prj(learner_encode_outputs[i]))
        
        max_proj_length = max([proj_output.shape[1] for proj_output in proj_output_list])
        
        model_type = self.base_model.config.model_type
        
        embed_input_ids = []
        embed_input_embeds = []
        embed_attention_masks = []
        embed_labels = []
        for i in range(input_ids.shape[0]):
            diff = max_proj_length - proj_output_list[i].shape[1]
            if diff == 0:
                new_input_ids = input_ids[i,...]
                new_attention_mask = attention_mask[i,...]
                new_labels = labels[i,...] if labels is not None else None
            else:
                if self.tokenizer_padding_side == 'left':
                    new_input_ids = torch.cat([ torch.ones((diff), dtype=input_ids.dtype, device=input_ids.device) * self.tokenizer_pad_token_id, 
                                                input_ids[i,...]])
                    new_attention_mask = torch.cat([ torch.zeros((diff), dtype=attention_mask.dtype, device=attention_mask.device), 
                                                attention_mask[i,...]])
                    if labels is not None:
                        new_labels = torch.cat([ torch.ones((diff), dtype=labels.dtype, device=labels.device) * IGNORE_INDEX, 
                                                labels[i,...]])
                    else:
                        new_labels = None
                else:
                    new_input_ids = torch.cat([input_ids[i,...],
                                                torch.ones((diff), dtype=input_ids.dtype, device=input_ids.device) * self.tokenizer_pad_token_id])        
                    new_attention_mask = torch.cat([attention_mask[i,...],
                                                    torch.zeros((diff), dtype=attention_mask.dtype, device=attention_mask.device)])
                    if labels is not None:
                        new_labels = torch.cat([labels[i,...],
                                                torch.ones((diff), dtype=labels.dtype, device=labels.device) * IGNORE_INDEX])
                    else:
                        new_labels = None

            embed_input_ids.append(new_input_ids)
            embed_attention_masks.append(new_attention_mask)
            embed_labels.append(new_labels)
            
            if model_type == 'qwen':
                inputs_embeds = self.model.transformer.wte(new_input_ids)
            else:
                inputs_embeds = self.model.model.embed_tokens(new_input_ids)
            embed_input_embeds.append(inputs_embeds)
        
        new_inputs_embeds, new_attention_mask, new_labels = combines_learner_output_with_inputs(
            embed_input_ids, embed_input_embeds, proj_output_list, embed_attention_masks, embed_labels, model_type, max_proj_length
        )
        return new_inputs_embeds, new_attention_mask, new_labels
    
    
    
    def prepare_custom_embeddings(self, **kwargs) -> List[torch.Tensor]:
        
        if self.args.is_learner and not self.args.is_map_embeds:
            
            motion_ids = kwargs['initial']
            interaction_ids = kwargs['interaction']
            object_ids = kwargs['object']
            
            if self.args.map_mode == 'img':
                map_embeds = kwargs['map_embeds']
                bs = map_embeds.shape[0]
            elif self.args.map_mode == 'pointset':
                pointsets = kwargs['map_embeds']
                map_embeds = self.map_encoder(pointsets)
                bs = map_embeds.shape[0]

            elif self.args.map_mode == 'text':
                raise ValueError(f"map_mode must be img or pointset when is_learner is True")
            
            
            
            
            sep_ids = INSERT_SIGN_MAP[self.base_model.config.model_type]['pos_sign_ids']
            
            object_list = []
            interaction_list = []
            for i in range(bs):
                one_object_list = split_subsequence(object_ids[i], sep_ids)
                one_interaction_list = split_subsequence(interaction_ids[i], sep_ids)
                object_list.append(one_object_list)
                interaction_list.append(one_interaction_list)

            sim_input = SimcopilotInput(
                map_embeds=map_embeds,
                motion_ids=motion_ids,
                interaction_ids=interaction_list,
                object_ids=object_list
            )
            
            learner_encode_outputs = self.learner_encoder(sim_input)
            proj_output_list = []
            for i in range(len(learner_encode_outputs)):
                proj_output_list.append(self.out_embd_prj(learner_encode_outputs[i]))
            
            return proj_output_list
                

        elif self.args.is_map_embeds:
            if self.args.map_mode != 'pointset':
                raise ValueError(f"map_mode must be pointset when is_map_embeds is True")
            
            pointsets = kwargs['map_embeds']
            map_encodes = self.map_encoder(pointsets)
            map_embeds = self.out_embd_prj(map_encodes)
            
            return [i[None,...] for i in map_embeds]
    
    def pad_and_embed_into_inputs(self, input_ids, attention_mask, labels, embeds_list):
        
        max_proj_length = max([proj_output.shape[1] for proj_output in embeds_list])
        model_type = self.base_model.config.model_type
        
        embed_input_ids = []
        embed_input_embeds = []
        embed_attention_masks = []
        embed_labels = []
        for i in range(input_ids.shape[0]):
            diff = max_proj_length - embeds_list[i].shape[1]
            if diff == 0:
                new_input_ids = input_ids[i,...]
                new_attention_mask = attention_mask[i,...]
                new_labels = labels[i,...] if labels is not None else None
            else:
                if self.tokenizer_padding_side == 'left':
                    new_input_ids = torch.cat([ torch.ones((diff), dtype=input_ids.dtype, device=input_ids.device) * self.tokenizer_pad_token_id, 
                                                input_ids[i,...]])
                    new_attention_mask = torch.cat([ torch.zeros((diff), dtype=attention_mask.dtype, device=attention_mask.device), 
                                                attention_mask[i,...]])
                    if labels is not None:
                        new_labels = torch.cat([ torch.ones((diff), dtype=labels.dtype, device=labels.device) * IGNORE_INDEX, 
                                                labels[i,...]])
                    else:
                        new_labels = None
                else:
                    new_input_ids = torch.cat([input_ids[i,...],
                                                torch.ones((diff), dtype=input_ids.dtype, device=input_ids.device) * self.tokenizer_pad_token_id])        
                    new_attention_mask = torch.cat([attention_mask[i,...],
                                                    torch.zeros((diff), dtype=attention_mask.dtype, device=attention_mask.device)])
                    if labels is not None:
                        new_labels = torch.cat([labels[i,...],
                                                torch.ones((diff), dtype=labels.dtype, device=labels.device) * IGNORE_INDEX])
                    else:
                        new_labels = None

            embed_input_ids.append(new_input_ids)
            embed_attention_masks.append(new_attention_mask)
            if new_labels is not None:
                embed_labels.append(new_labels)
            
            if model_type == 'qwen':
                inputs_embeds = self.model.transformer.wte(new_input_ids)
            else:
                inputs_embeds = self.model.model.embed_tokens(new_input_ids)
            embed_input_embeds.append(inputs_embeds)
        
        if len(embed_labels) == 0:
            embed_labels = None
        
        new_inputs_embeds, new_attention_mask, new_labels = combines_learner_output_with_inputs(
            embed_input_ids, embed_input_embeds, embeds_list, embed_attention_masks, embed_labels, model_type, max_proj_length
        )
        
        return new_inputs_embeds, new_attention_mask, new_labels
        
    
    def forward(self, 
                input_ids=None,
                attention_mask=None, 
                inputs_embeds=None, 
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                task_ids=None,
                **kwargs):
        
        embeds_list = self.prepare_custom_embeddings(**kwargs)
        
        new_inputs_embeds, new_attention_mask, new_labels = self.pad_and_embed_into_inputs(input_ids, attention_mask, labels, embeds_list)

        wp_token_mat = kwargs.pop('wp_token_mat', None)
        if wp_token_mat is not None:
            wp_token_mat = wp_token_mat.to(torch.float16)
            
        for a in self.args.additional_data_attr:
            if a in kwargs:
                kwargs.pop(a, None)
        return super().forward(input_ids=None, 
                               attention_mask=new_attention_mask, 
                               inputs_embeds=new_inputs_embeds, 
                               labels=new_labels, 
                               output_attentions=output_attentions, 
                               output_hidden_states=output_hidden_states, 
                               return_dict=return_dict, 
                               task_ids=task_ids, 
                               wp_token_mat = wp_token_mat,
                               **kwargs)
    
    def generate(self, **kwargs):
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        labels = kwargs['labels'] if 'labels' in kwargs else None
        
        
        embeds_list = self.prepare_custom_embeddings(**kwargs)
        
        new_inputs_embeds, new_attention_mask, new_labels = self.pad_and_embed_into_inputs(input_ids, attention_mask, labels, embeds_list)
        
        
        kwargs['inputs_embeds'] = new_inputs_embeds
        kwargs['attention_mask'] = new_attention_mask
        if labels is not None:
            kwargs['labels'] = new_labels
        
        
        for a in self.args.additional_data_attr:
            if a in kwargs:
                if self.args.use_wp_token and a == 'wp_token_mat':
                    continue
                kwargs.pop(a, None)

        if self.args.use_wp_token:
            output = self.base_model.model.wp_generate(**kwargs)
        else:
            output = self.base_model.generate(**kwargs)
        
        return output
    

def combines_learner_output_with_inputs(
    input_ids, inputs_embeds, proj_output_list, attention_mask, labels=None, model_type='llama', max_proj_length=0
):
    bs = len(input_ids)
    
    insert_seq = INSERT_SIGN_MAP[model_type]['embed_sign_ids']
    insert_seq_length = len(insert_seq)
    
    new_inputs_embeds = []
    new_attention_masks = []
    new_labels = []
    
    for b in range(bs):
        cur_input_id = input_ids[b]
        cur_input_embed = inputs_embeds[b]
        cur_attention_mask = attention_mask[b]
        cur_label = labels[b] if labels is not None else None
        
        cur_proj_output = proj_output_list[b]
        cur_add_proj_len = cur_proj_output.shape[1]
        true_expand_len = cur_add_proj_len - insert_seq_length
        
        new_input_embed = torch.zeros([cur_input_embed.shape[0]+true_expand_len, cur_input_embed.shape[1]], dtype=cur_input_embed.dtype).to(cur_input_embed.device)        
        new_attention_mask = torch.zeros([cur_attention_mask.shape[0]+true_expand_len], dtype=cur_attention_mask.dtype).to(cur_attention_mask.device)
        new_label = torch.zeros([cur_label.shape[0]+true_expand_len], dtype=cur_label.dtype).to(cur_label.device) if cur_label is not None else None
        
        pos = find_subsequence(cur_input_id, INSERT_SIGN_MAP[model_type]['embed_sign_ids'])[0]
        
        new_input_embed[:pos, :] = cur_input_embed[:pos, :]
        new_input_embed[pos:pos+cur_add_proj_len, :] = cur_proj_output.squeeze(0)
        new_input_embed[pos+cur_add_proj_len:, :] = cur_input_embed[pos+insert_seq_length:, :]
        
        new_attention_mask[:pos] = cur_attention_mask[:pos]
        new_attention_mask[pos:pos+cur_add_proj_len] = 1
        new_attention_mask[pos+cur_add_proj_len:] = cur_attention_mask[pos+insert_seq_length:]
        
        if cur_label is not None:
            new_label[:pos] = cur_label[:pos]
            new_label[pos:pos+cur_add_proj_len] = -100
            new_label[pos+cur_add_proj_len:] = cur_label[pos+insert_seq_length:]
        
        new_inputs_embeds.append(new_input_embed)
        new_attention_masks.append(new_attention_mask)
        new_labels.append(new_label)
        
    res_inputs_embeds = torch.stack(new_inputs_embeds)
    res_attention_masks = torch.stack(new_attention_masks)
    res_labels = torch.stack(new_labels) if labels is not None else None
    return res_inputs_embeds, res_attention_masks, res_labels