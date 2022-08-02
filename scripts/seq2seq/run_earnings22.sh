#!/usr/bin/env bash
python run_flax_speech_recognition_seq2seq.py \
        --dataset_name="sanchit-gandhi/earnings22_robust_split" \
        --model_name_or_path="sanchit-gandhi/flax-wav2vec2-2-bart-large-scan" \
        --dataset_config_name="all" \
        --train_split_name="train" \
        --eval_split_name="validation" \
        --test_split_name="test" \
        --text_column_name="sentence" \
        --id_column_name="source_id" \
        --output_dir="./flax-wav2vec2-2-bart-large-earnings22-baseline" \
        --wandb_project="earnings22" \
        --wandb_name="flax-wav2vec2-2-bart-large-earnings22-baseline" \
        --dataset_cache_dir="/home/sanchitgandhi/cache/huggingface/datasets" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="4" \
        --logging_steps="25" \
        --max_steps="50000" \
        --eval_steps="10000" \
        --save_steps="10000" \
        --generation_max_length="40" \
        --generation_num_beams="1" \
        --generation_length_penalty="1.2" \
        --final_generation_max_length="200" \
        --final_generation_num_beams="5" \
        --learning_rate="1e-4" \
        --warmup_steps="500" \
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