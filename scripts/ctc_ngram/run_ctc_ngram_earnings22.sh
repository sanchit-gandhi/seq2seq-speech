#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-earnings22-cased-hidden-activation-featproj-dropout-0.2" \
	--tokenizer_name="/home/patrick/ngrams/earnings22_robust_split" \
	--decoder_name="/home/patrick/ngrams/earnings22_robust_split" \
	--dataset_cache_dir="/home/patrick/.cache/huggingface/datasets" \
	--dataset_name="sanchit-gandhi/earnings22_robust_split" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--text_column="sentence" \
	--preprocessing_num_workers="1" \
	--output_dir="/home/patrick/ngrams/earnings22_robust_split/evaluation" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--use_auth_token
