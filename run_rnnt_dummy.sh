#!/usr/bin/env bash
python run_speech_recognition_rnnt.py \
        --model_name_or_path="config_rnnt_bpe.yaml" \
        --dataset_name="librispeech_asr" \
        --tokenizer_path="tokenizer" \
        --vocab_size="1024" \
        --num_train_epochs="20" \
        --evaluation_strategy="epoch" \
        --dataset_config_name="clean" \
        --train_split_name="train.100" \
        --eval_split_name="validation" \
        --test_split_name="test" \
        --text_column_name="text" \
        --file_column_name="file" \
        --output_dir="./" \
        --wandb_project="rnnt_debug" \
        --per_device_train_batch_size="16" \
        --per_device_eval_batch_size="8" \
        --logging_steps="25" \
        --learning_rate="1e-4" \
        --warmup_steps="500" \
        --overwrite_output_dir \
        --do_lower_case \
        --do_eval \
        --do_train \
        --push_to_hub="False"
