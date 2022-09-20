#!/usr/bin/env bash
python run_flax_speech_recognition_seq2seq.py \
        --dataset_name="kensho/spgispeech" \
        --model_name_or_path="sanchit-gandhi/flax-wav2vec2-2-bart-large-scan" \
        --dataset_config_name="L" \
        --train_split_name="train" \
        --eval_split_name="validation" \
        --test_split_name="test" \
        --text_column_name="transcript" \
        --id_column_name="wav_filename" \
        --output_dir="./" \
        --wandb_project="spgispeech" \
        --wandb_name="flax-wav2vec2-2-bart-large-spgispeech-black-box" \
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
        --do_lower_case="False" \
        --overwrite_output_dir \
        --gradient_checkpointing \
        --freeze_feature_encoder \
        --predict_with_generate \
        --do_eval \
        --do_train \
        --do_predict \
        --push_to_hub \
        --use_auth_token