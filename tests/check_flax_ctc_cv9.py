#!/usr/bin/env python3
import tempfile
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoFeatureExtractor
from models.modeling_flax_wav2vec2 import FlaxWav2Vec2ForCTC
import jax.numpy as jnp
from datasets import load_dataset
import datasets
from run_flax_speech_recognition_ctc import ctc_loss

model_id = "facebook/wav2vec2-large-lv60"
# model_id = "hf-internal-testing/tiny-random-wav2vec2"
tokenizer_id = "patrickvonplaten/wav2vec2_ctc_cv9_tokenizer"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id, return_attention_mask=True)

# in PyTorch we always use 'mean' by default.
# See: https://github.com/huggingface/transformers/blob/93b802c43e70f41edfb166ad6ead79de95a26c32/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py#L126
model_pt = Wav2Vec2ForCTC.from_pretrained(model_id, ctc_loss_reduction="mean")

with tempfile.TemporaryDirectory() as temp_folder:
    model_pt.save_pretrained(temp_folder)
    model_fx = FlaxWav2Vec2ForCTC.from_pretrained(temp_folder, from_pt=True)


# load CV9 dataset and read soundfiles
ds = load_dataset("mozilla-foundation/common_voice_9_0", "en", split="train[:1%]", cache_dir="/home/sanchitgandhi/cache/huggingface/datasets/")

# resample dataset to 16kHz on the fly
ds = ds.cast_column(
            "audio", datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

# inputs pt
samples = [d["array"] for d in ds[:4]["audio"]]
inputs_pt = feature_extractor(samples, return_tensors="pt", padding="longest", sampling_rate=16_000)

# inputs fx
inputs_fx = feature_extractor(samples, return_tensors="np", padding="longest", sampling_rate=16_000)

# labels
transcription = ds[:4]["sentence"]
labels_pt = tokenizer(transcription, return_tensors="pt", padding="longest")
labels_ids_pt = labels_pt["input_ids"].masked_fill(labels_pt.attention_mask.ne(1), -100)

labels_fx = tokenizer(transcription, return_tensors="np", padding="longest")
labels_ids_fx = jnp.where(labels_fx.attention_mask == 0, -100, labels_fx.input_ids)

# set pt model config to accommodate for CV9 tokenizer vocab size
model_pt.config.vocab_size = tokenizer.vocab_size

# pytorch
with torch.no_grad():
    outputs = model_pt(**inputs_pt, labels=labels_ids_pt)

logits_pt = outputs.logits
loss_pt = outputs.loss

# flax
logits_fx = model_fx(**inputs_fx).logits
logits_attention_mask = model_fx._get_feature_vector_attention_mask(logits_fx.shape[1], inputs_fx.attention_mask)

# Check logits the same
logits_diff = np.abs((logits_pt.detach().numpy() - np.asarray(logits_fx))).max()
assert logits_diff < 1e-3, "Logits don't match"

# flax loss
blank_id = model_fx.config.pad_token_id
loss_fx = ctc_loss(logits_fx, logits_attention_mask, labels_ids_fx, blank_id)

# Check loss is the same
loss_diff = np.asarray(loss_fx) - loss_pt.numpy()
assert loss_diff < 1e-3, "Loss doesn't match"