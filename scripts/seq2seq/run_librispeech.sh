#!/usr/bin/env bash
python run_flax_speech_recognition_seq2seq.py \
        --dataset_name="librispeech_asr" \
        --model_name_or_path="sanchit-gandhi/flax-wav2vec2-2-bart-large-scan" \
        --dataset_config_name="all" \
        --train_split_name="train.clean.100+train.clean.360+train.other.500" \
        --eval_split_name="validation.clean" \
        --test_split_name="validation.other+test.clean+test.other" \
        --text_column_name="text" \
        --id_column_name="id" \
        --output_dir="./" \
        --wandb_project="librispeech_960h" \
        --wandb_name="flax-wav2vec2-2-bart-large-ls-960h-black-box" \
        --dataset_cache_dir="/home/sanchitgandhi/cache/huggingface/datasets" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="2" \
        --learning_rate="1e-4" \
        --warmup_steps="500" \
        --logging_steps="25" \
        --max_steps="50000" \
        --eval_steps="10000" \
        --save_steps="10000" \
        --generation_max_length="200" \
        --generation_num_beams="5" \
        --generation_length_penalty="1.2" \
        --hidden_dropout="0.2" \
        --activation_dropout="0.2" \
        --feat_proj_dropout="0.2" \
        --overwrite_output_dir \
        --gradient_checkpointing \
        --freeze_feature_encoder \
        --predict_with_generate \
        --do_lower_case \
        --do_eval \
        --do_train \
        --do_predict \
        --push_to_hub \
        --use_auth_token
