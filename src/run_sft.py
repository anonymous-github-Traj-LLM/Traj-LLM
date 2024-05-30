import os
os.environ['WANDB_MODE'] = 'offline'
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from helpers.dataset_template.preprocess import preprocess_dataset, split_dataset
from helpers.model_helper import load_model_and_tokenizer, get_logits_processor
from helpers.train_helper import CustomSeq2SeqTrainer, ComputeMetrics, CustomDataCollatorForSeq2Seq
from helpers.log_helper import Logger
from helpers.sim_args import SimArguments
from helpers.hparams import IGNORE_INDEX

def run_sft(args: SimArguments):
    model, tokenizer = load_model_and_tokenizer(args)
    model.to(torch.bfloat16)
    dataset = preprocess_dataset(tokenizer, args)

    if not args.is_train:
        tokenizer.padding_side = "left"
        Logger.info("Use left-padding in testing.")
    
    Logger.info(f"model.generation_config: {model.generation_config}")
    
    if args.is_learner:
        model.tokenizer_padding_side = tokenizer.padding_side
        model.tokenizer_pad_token_id = tokenizer.pad_token_id
    
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        label_pad_token_id = IGNORE_INDEX,
        padding=True,
        args=args
    )
    
    
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        warmup_ratio=0.04,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        evaluation_strategy="steps" if args.eval_nums > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if args.eval_nums > 0 else None,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        save_total_limit=4,
        label_names=args.additional_data_attr,
        
        load_best_model_at_end=True if args.eval_nums > 0 else False,
        ddp_find_unused_parameters=None,
        group_by_length=False,
        report_to=None,
        run_name=None,
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_config=model.generation_config,
    )
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer) if args.eval_nums else None,
        **split_dataset(dataset, args)
    )
    trainer.sim_args = args
    
    Logger.info(f'Output dir: {args.output_dir}')
    if args.is_train:
        Logger.info("Start training...")
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        Logger.info("Save model.")
        
        # save module
        if args.is_learner:
            torch.save(model.learner_encoder.state_dict(), os.path.join(args.output_dir, 'learner_encoder.pth'))
            torch.save(model.out_embd_prj.state_dict(), os.path.join(args.output_dir, 'out_embd_prj.pth'))
            torch.save(model.map_encoder.state_dict(), os.path.join(args.output_dir, 'map_encoder.pth'))
            Logger.info("Save learner_encoder, out_embd_prj, map_encoder.")
    
    if not args.is_train:
        if args.use_wp_token:
            Logger.info("Start predicting with waypoint token...")
            trainer.predict_with_waypoint_token(dataset)
            pass
        
        else:
        
            generation_config = {
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
                "max_new_tokens": args.generation_max_length,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "logits_processor": get_logits_processor(),
                "use_cache": args.force_use_cache
            }
            Logger.info(f"predict generation_config: {generation_config}")
            
            Logger.info("Start predicting...")
            predict_results = trainer.predict(dataset, metric_key_prefix="predict", **generation_config)
            predict_results.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(predict_results)
            Logger.info("Save predictions.")