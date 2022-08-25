#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-ls-960h-baseline" \
	--tokenizer_name="/home/patrick/ngrams/librispeech_asr" \
	--decoder_name="/home/patrick/ngrams/librispeech_asr" \
	--dataset_cache_dir="/home/patrick/.cache/huggingface/datasets" \
	--dataset_name="librispeech_asr" \
	--dataset_config_name="all" \
	--eval_split_name="validation.clean" \
	--test_split_name="validation.other+test.clean+test.other" \
	--text_column="text" \
	--preprocessing_num_workers="1" \
	--output_dir="/home/patrick/ngram_output_dir" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--use_auth_token
