#!/bin/bash

python main.py \
    --exp_name train_exp \
    --run_mode train_w_learner \
    --output_dir output \
    --base_model llama_model_path \
    --data_path dataset/trajllm_train_mini_10k.pkl \
    --max_samples 0 \
    --shuffle_data \
    --additional_data_attr object,interaction,initial,map_embeds \
    --encode_data_attr object,interaction,initial \
    --map_mode pointset \
    --map_embed_dim 2 \
    --template llama \
    --generation_max_length 2048 \
    --lora_target q_proj,v_proj \
    --lora_rank 8 \
    --lora_alpha 32.0 \
    --lora_dropout 0.1 \
    --tokenizer_init_padding_side right \
    --micro_batch_size 8 \
    --batch_size 16 \
    --val_batch_size 8 \
    --eval_steps 200 \
    --num_epochs 1 \
    --save_steps 1000 \
    --logging_steps 20  \