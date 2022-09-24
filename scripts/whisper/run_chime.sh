#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python run_speech_recognition_whisper.py \
	--model_name_or_path="medium.en" \
  --dataset_name="speech-seq2seq/chime4-raw" \
  --dataset_config_name="1-channel" \
  --train_split_name="train" \
  --eval_split_name="validation" \
  --test_split_name="test" \
  --text_column_name="text" \
	--max_steps="2500" \
	--output_dir="./" \
	--run_name="whisper-chime4" \
	--wandb_project="whisper" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-4" \
	--warmup_steps="500" \
	--report_to="wandb" \
	--preprocessing_num_workers="16" \
	--evaluation_strategy="steps" \
	--eval_steps="500" \
	--save_strategy="steps" \
	--save_steps="500" \
	--generation_max_length="128" \
	--length_column_name="input_lengths" \
	--do_lower_case="False" \
	--push_to_hub="False" \
	--gradient_checkpointing \
	--group_by_length \
	--freeze_encoder \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--do_predict \
	--predict_with_generate \
	--use_auth_token
