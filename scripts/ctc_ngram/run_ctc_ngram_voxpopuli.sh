#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-voxpopuli-black-box" \
	--tokenizer_name="/home/patrick/ngrams/voxpopuli" \
	--decoder_name="/home/patrick/ngrams/voxpopuli" \
	--dataset_cache_dir="/home/patrick/.cache/huggingface/datasets" \
	--dataset_name="polinaeterna/voxpopuli" \
	--dataset_config_name="en" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--text_column_name="normalized_text" \
	--preprocessing_num_workers="1" \
	--output_dir="/home/patrick/ngrams/voxpopuli/evaluation" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--max_label_length=2056 \
	--use_auth_token \
	--do_lower_case
