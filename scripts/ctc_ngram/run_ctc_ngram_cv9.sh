#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-cv9-black-box" \
	--tokenizer_name="/home/patrick/ngrams/common_voice_9_0" \
	--decoder_name="/home/patrick/ngrams/common_voice_9_0" \
	--dataset_cache_dir="/home/patrick/.cache/huggingface/datasets" \
	--dataset_name="mozilla-foundation/common_voice_9_0" \
	--dataset_config_name="en" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--text_column_name="sentence" \
	--preprocessing_num_workers="1" \
	--max_eval_duration_in_seconds="20.0" \
	--output_dir="/home/patrick/ngrams/common_voice_9_0/evaluation" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--use_auth_token
