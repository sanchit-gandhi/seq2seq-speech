#!/usr/bin/env bash
python ./run_flax_speech_recognition_ctc_ngram.py \
	--model_name_or_path="sanchit-gandhi/flax-wav2vec2-ctc-ls-960h-baseline" \
	--tokenizer_name="sanchit-gandhi/flax-wav2vec2-ctc-ls-960h-baseline" \
	--decoder_name="patrickvonplaten/wav2vec2-base-100h-with-lm" \
	--dataset_cache_dir="/home/sanchitgandhi/cache/huggingface/datasets" \
	--dataset_name="librispeech_asr" \
	--dataset_config_name="all" \
	--eval_split_name="validation.clean" \
	--test_split_name="validation.other+test.clean+test.other" \
	--text_column="text" \
	--preprocessing_num_workers="1" \
	--output_dir="./ngram_output_dir" \
	--max_steps="50000" \
	--eval_steps="10000" \
	--save_steps="10000" \
	--wandb_project="librispeech_960h" \
	--wandb_name="flax-wav2vec2-ctc-ls-960h-with-lm-baseline" \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--use_auth_token