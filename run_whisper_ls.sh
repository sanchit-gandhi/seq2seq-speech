#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python run_speech_recognition_whisper.py \
	--model_name_or_path="small.en" \
	--dataset_name="librispeech_asr" \
	--max_steps="10000" \
	--dataset_config_name="all" \
	--train_split_name="train.clean.100+train.clean.360+train.other.500" \
	--eval_split_name="validation.clean" \
	--test_split_name="validation.other+test.clean+test.other" \
	--text_column_name="text" \
	--output_dir="./" \
	--run_name="whisper-ls-960h" \
	--wandb_project="whisper" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="50" \
	--learning_rate="1e-4" \
	--warmup_steps="500" \
	--report_to="wandb" \
	--preprocessing_num_workers="16" \
	--evaluation_strategy="steps" \
	--eval_steps="2500" \
	--save_strategy="steps" \
	--save_steps="2500" \
	--generation_max_length="256" \
	--length_column_name="input_lengths" \
	--do_lower_case="True" \
	--group_by_length \
	--freeze_encoder \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--do_predict \
	--predict_with_generate \
