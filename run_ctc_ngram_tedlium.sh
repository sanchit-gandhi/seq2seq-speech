#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-tedlium-black-box" \
	--tokenizer_name="/home/patrick/ngrams/tedlium" \
	--decoder_name="/home/patrick/ngrams/tedlium" \
	--dataset_cache_dir="/home/patrick/.cache/huggingface/datasets" \
	--dataset_name="LIUM/tedlium" \
	--dataset_config_name="release3" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--text_column="text" \
	--preprocessing_num_workers="1" \
	--output_dir="/home/patrick/ngrams/tedlium/evaluation" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--use_auth_token
