# Seq2Seq Speech in JAX
A JAX/Flax repository for combining a pre-trained speech encoder model (e.g. Wav2Vec2, HuBERT, WavLM) with a pre-trained text decoder model (e.g. GPT2, Bart) to yield a Speech Sequence-to-Sequence (Seq2Seq) model for automatic speech recognition. 

The script `run_flax_speech_recognition_seq2seq.py` can be used to fine-tune a Speech Seq2Seq model on one of the official speech recognition datasets or a custom dataset. It makes use of the `pmap` JAX operator to provide model parallelism accross GPU/TPU devices.

The modelling files are based very heavily on those from Hugging Face [Transformers 🤗](https://github.com/huggingface/transformers). This is a standalone repository to enable rapid prototyping and involvement with the community. The final modelling files and training script will be merged into Transformers 🤗 to be used with the rest of the open-source library. The final system weights will be made publicly available at [huggingface.co](huggingface.co) 🚀

![Seq2SeqModel](seq2seq.png)
**Figure 1:** Speech-encoder text-decoder style Seq2Seq model.

## Example Usage
To instantiate a _Wav2Vec2-2-Bart_ model with the `FlaxSpeechEncoderDecoderModel` framework, run the following Python script inside the cloned repo:
```python
from transformers import AutoFeatureExtractor, AutoTokenizer
from models.modeling_flax_speech_encoder_decoder import FlaxSpeechEncoderDecoderModel
import numpy as np

# checkpoints to leverage
encoder_id = "facebook/wav2vec2-large-lv60"
decoder_id = "facebook/bart-large"

model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_id, decoder_id, encoder_add_adapter=True)

model.config.decoder_start_token_id = model.config.decoder.bos_token_id
model.config.pad_token_id = model.config.decoder.pad_token_id
model.config.eos_token_id = model.config.decoder.eos_token_id
model.config.use_cache = False
model.config.processor_class = "Wav2Vec2Processor"

# check if generation works
out = model.generate(np.ones((1, 2000)))

model.save_pretrained("./")

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_extractor.save_pretrained("./")
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained("./")
```

To train the model on [Librispeech ASR](https://huggingface.co/datasets/librispeech_asr) in default precision, run the bash script provided below:
```bash
#!/usr/bin/env bash
python run_flax_speech_recognition_seq2seq.py \
        --dataset_name="librispeech_asr" \
        --model_name_or_path="./" \
        --dataset_config_name="clean" \
        --train_split_name="train.100" \
        --eval_split_name="validation" \
        --output_dir="./" \
        --preprocessing_num_workers="16" \
        --length_column_name="input_length" \
        --overwrite_output_dir \
        --num_train_epochs="5" \
        --per_device_train_batch_size="2" \
        --per_device_eval_batch_size="2" \
        --gradient_accumulation_steps="1" \
        --logging_steps="25" \
        --max_duration_in_seconds="15" \
        --max_target_length="128" \
        --generation_max_length="40" \
        --generation_num_beams="1" \
        --learning_rate="1e-4" \
        --warmup_steps="500" \
        --text_column_name="text" \
        --save_total_limit="1" \
        --freeze_feature_encoder \
        --predict_with_generate \
        --do_lower_case \
        --do_eval \
        --do_train
```
