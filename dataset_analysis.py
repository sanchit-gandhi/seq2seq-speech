#!/usr/bin/env python3
from datasets import load_dataset

# Adapt
# ================================
dataset_name = "librispeech_asr"
dataset_config = "switchboard"
dataset_cache_dir = "/home/patrick/.cache/huggingface/datasets"
use_auth_token = True
# ================================

dataset = load_dataset(
    dataset_name,
#    dataset_config,
    use_auth_token=use_auth_token,
    cache_dir=dataset_cache_dir,
)

import ipdb; ipdb.set_trace()
