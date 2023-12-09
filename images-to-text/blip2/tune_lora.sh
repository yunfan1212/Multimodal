#!/usr/bin/env bash
python tune_lora_trainer.py \
--output_dir output \
--model_name_or_path /home/blip2-opt-2.7b \
--dataset_name football-dataset \
--remove_unused_columns=False \
--do_train \
--do_eval \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--learning_rate="5e-5" \
--warmup_steps="0" \
--weight_decay 0.1 \
--overwrite_output_dir \
--max_steps 70 \
--logging_steps=5
