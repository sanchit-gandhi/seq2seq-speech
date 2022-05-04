#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc.py \
	--model_name_or_path="/home/patrick_huggingface_co/flax-wav2vec2-large-lv60-scan" \
	--dataset_name="librispeech_asr" \
	--dataset_cache_dir="/mnt/disks/persist" \
	--text_column="text" \
	--dataset_config_name="all" \
	--train_split_name="train.clean.100" \
	--eval_split_name="validation.clean" \
	--preprocessing_num_workers="1" \
	--output_dir="train_clean_100" \
	--num_train_epochs="3" \
	--learning_rate="3e-4" \
	--logging_steps="1" \
	--warmup_steps="500" \
	--do_train \
	--do_eval \
	--overwrite_output_dir \
	--gradient_checkpointing \
	--freeze_feature_encoder \
