#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the Flax library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Optional, Dict, Union, Any, Tuple, List

import numpy as np
import torch

from omegaconf import OmegaConf
from models.rnnt_bpe_model import RNNTBPEModel

import datasets
from datasets import DatasetDict, load_dataset, Dataset
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    is_tensorboard_available,
    set_seed,
    Trainer,
)
from transformers.trainer_pt_utils import get_parameter_names, nested_detach
from transformers.trainer_utils import get_last_checkpoint, is_main_process, speed_metrics
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from process_asr_text_tokenizer import __process_data as nemo_process_data, \
    __build_document_from_manifests as nemo_build_document_from_manifests

import bitsandbytes as bnb

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    manifest_path: str = field(
        default="data",
        metadata={
            "help": "Manifest path."
        },
    )
    tokenizer_path: str = field(
        default="tokenizers",
        metadata={
            "help": "Tokenizer path."
        },
    )
    vocab_size: int = field(
        default=1024,
        metadata={"help": "Tokenizer vocab size."}
    )
    tokenizer_type: str = field(
        default="spe",
        metadata={
            "help": "Can be either spe or wpe. spe refers to the Google sentencepiece library tokenizer."
                    "wpe refers to the HuggingFace BERT Word Piece tokenizer."
        },
    )
    spe_type: str = field(
        default="unigram",
        metadata={
            "help": "Type of the SentencePiece model. Can be `bpe`, `unigram`, `char` or `word`."
                    "Used only if `tokenizer_type` == `spe`"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to cache directory for saving and loading datasets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Truncate audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_target_length: Optional[int] = field(
        default=2,
        metadata={
            "help": "The minimum total sequence length for target text after tokenization. Sequences shorter "
                    "than this will be filtered."
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
                    "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
                    "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
                    "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={"help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"},
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    wandb_project: str = field(
        default="speech-recognition-rnnt",
        metadata={"help": "The name of the wandb project."},
    )


def build_tokenizer(model_args, data_args, manifests):
    """
    Function to build a NeMo tokenizer from manifest file(s).
    Copied from https://github.com/NVIDIA/NeMo/blob/66c7677cd4a68d78965d4905dd1febbf5385dff3/scripts/tokenizers/process_asr_text_tokenizer.py#L268
    """
    data_root = model_args.tokenizer_path
    if isinstance(manifests, list):
        joint_manifests = ",".join(manifests)
    else:
        joint_manifests = manifests
    vocab_size = model_args.vocab_size
    tokenizer = model_args.tokenizer_type
    spe_type = model_args.spe_type

    logger.info("Building tokenizer...")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    else:
        os.system(f"rm -rf {data_root}")
        os.makedirs(data_root)

    text_corpus_path = nemo_build_document_from_manifests(data_root, joint_manifests)

    tokenizer_path = nemo_process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=data_args.do_lower_case,
        spe_character_coverage=1.0,
        spe_sample_size=-1,
        spe_train_extremely_large_corpus=False,
        spe_max_sentencepiece_length=-1,
        spe_bos=False,
        spe_eos=False,
        spe_pad=False,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logger.info('Done!')

    # Tokenizer path
    if tokenizer == 'spe':
        tokenizer_dir = os.path.join(data_root, f"tokenizer_spe_{spe_type}_v{vocab_size}")
        tokenizer_type_cfg = "bpe"
    else:
        tokenizer_dir = os.path.join(data_root, f"tokenizer_wpe_v{vocab_size}")
        tokenizer_type_cfg = "wpe"

    return tokenizer_dir, tokenizer_type_cfg


def NeMoDataCollator(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Data collator that will dynamically pad the inputs received.
    Since NeMo models don't have a HF processor defined (feature extractor + tokenizer), we'll pad by hand...
    The padding idx is arbitrary: we provide the model with the input lengths and label lengths, from which
    all the relevant padding information is inferred. Thus, we'll use the default np.pad padding idx (0).
    """
    # split inputs and labels since they have to be of different lengths
    # and need different padding methods
    input_ids = [feature["input_ids"] for feature in features]
    labels = [feature["labels"] for feature in features]

    # first, pad the audio inputs to max_len
    input_lengths = [len(inp_val) for inp_val in input_ids]
    max_input_len = max(input_lengths)
    input_ids = [np.pad(input_val, (0, max_input_len - input_len), 'constant') for input_val, input_len in
                    zip(input_ids, input_lengths)]

    # next, pad the target labels to max_len
    label_lengths = [len(lab) for lab in labels]
    max_label_len = max(label_lengths)
    labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant') for lab, lab_len in zip(labels, label_lengths)]

    batch = {"input_lengths": input_lengths, "labels": labels, "label_lengths": label_lengths}

    # return batch as a pt tensor (list -> np.array -> torch.tensor)
    batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

    # leave all ints as are, convert float64 to pt float
    batch["input_ids"] = torch.tensor(np.array(input_ids, dtype=np.float32), requires_grad=False)

    return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load the model config (discarding optimiser and trainer attributes)
    config = OmegaConf.load(model_args.model_name_or_path).model

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if training_args.do_predict:
        test_split = data_args.test_split_name.split("+")
        for split in test_split:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        raise ValueError(
            "Cannot not train, not do evaluation and not do prediction. At least one of "
            "training, evaluation or prediction has to be done."
        )

    # if not training, there is no need to run multiple epochs
    if not training_args.do_train:
        training_args.num_train_epochs = 1

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 6. Resample speech dataset ALWAYS
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=config.sample_rate)
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * config.sample_rate)
    min_input_length = int(data_args.min_duration_in_seconds * config.sample_rate)
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    do_lower_case = data_args.do_lower_case
    dataset_name = data_args.dataset_name
    # Define tokens to ignore/replace
    # TODO: clean-this up... It's currently a bit of a mess, sorry!
    gigaspeech_punctuation = {" <comma>": ",", " <period>": ".", " <questionmark>": "?", " <exclamationpoint>": "!"}
    gigaspeech_disfluencies = ["<other>", "<sil>"]
    swb_disfluencies = ["[noise]", "[laughter]", "[silence]", "<a_aside>", "<b_aside>", "<e_aside>", "[laughter-",
                        "[vocalized-noise]", "_1"]
    swb_punctuations = ["{", "}", "[", "]-", "]"]
    earnings_disfluencies = ["<crosstalk>", "<affirmative>", "<inaudible>", "inaudible", "<laugh>"]
    ignore_segments = ["ignore_time_segment_in_scoring", "<noise>", "<music>", "[noise]", "[laughter]", "[silence]",
                       "[vocalized-noise]", "<crosstalk>", "<affirmative>", "<inaudible>", "<laugh>", "<other>",
                       "<sil>", ""]

    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval and data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    if training_args.do_predict and data_args.max_test_samples is not None:
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    # filter data where the targets are ignored in scoring
    def is_target_labels(input_str):
        return input_str.lower() not in ignore_segments

    raw_datasets = raw_datasets.filter(
        is_target_labels,
        num_proc=num_workers,
        input_columns=[text_column_name],
        desc="filtering data where the targets are ignored in scoring",
    )

    def prepare_dataset(batch):
        # process audio
        try:
            sample = batch[audio_column_name]
        except ValueError:
            # E22: some samples are empty (no audio). Reading the empty audio array will trigger
            # a soundfile ValueError. For now, we'll manually set these arrays to a zero array.
            # They will be filtered in the subsequent filtering stage and so are
            # explicitly ignored during training.
            # TODO: clean-up E22 dataset to remove these samples
            sample = {"array": np.array([0.]), "sampling_rate": config.sample_rate}

        # NeMo RNNT model performs the audio preprocessing in the `.forward()` call
        # => we only need to supply it with the raw audio values
        batch["input_ids"] = sample["array"]
        batch["input_lengths"] = len(sample["array"])

        # Process targets. Note: this is quite lengthy as we perform any necessary processing
        # for each of the 8 datasets in the benchmark
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]

        if dataset_name == "google/xtreme_s":
            batch[text_column_name] = input_str
            return batch

        # Common Voice 9
        if input_str.startswith('"') and input_str.endswith('"'):
            # we can remove trailing quotation marks as they do not affect the transcription
            input_str = input_str[1:-1]
        # normalize quotation marks
        input_str = re.sub(r'["“”]', '"', input_str)
        # normalize apostrophes
        input_str = re.sub(r"[’']", "'", input_str)
        # normalize hyphens
        input_str = re.sub(r"[—–]", "-", input_str)
        # replace double quotation marks with single
        input_str = input_str.replace('""', '"')
        if dataset_name == "mozilla-foundation/common_voice_9_0" and len(input_str):
            # for CV9, we'll normalize the text to always finish with punctuation
            if input_str[-1] not in [".", "?", "!"]:
                input_str = input_str + "."

        # TEDLIUM-3
        # delete the <unk> token from the text and replace spaced apostrophes with un-spaced
        input_str = input_str.replace("<unk>", "").replace(" '", "'")

        # GigaSpeech
        for disfluency in gigaspeech_disfluencies:
            input_str = input_str.replace(disfluency, "")
        # convert spelled out punctuation to symbolic form
        for punctuation, replacement in gigaspeech_punctuation.items():
            input_str = input_str.replace(punctuation, replacement)
        if dataset_name == "speechcolab/gigaspeech" and len(input_str):
            # for GS, we'll normalize the text to always finish with punctuation
            if input_str[-1] not in [".", "?", "!"]:
                input_str = input_str + "."

        # SWB
        for disfluency in swb_disfluencies:
            input_str = input_str.replace(disfluency, "")
        # remove parenthesised text (test data only)
        input_str = re.sub("[\(].*?[\)]", "", input_str)
        for punctuation in swb_punctuations:
            input_str = input_str.replace(punctuation, "")
        # replace anomalous words with their correct transcriptions
        split_str = input_str.split("/")
        if len(split_str) > 1:
            input_str = " ".join(
                [" ".join([" ".join(i.split(" ")[:-1]) for i in split_str])] + [split_str[-1].split(" ")[-1]])

        # Earnings 22
        for disfluency in earnings_disfluencies:
            input_str = input_str.replace(disfluency, "")
        # replace mal-formatted ellipsis
        input_str = input_str.replace("…", ".")

        # JIWER compliance
        # remove multiple spaces
        input_str = re.sub(r"\s\s+", " ", input_str)
        # strip trailing spaces
        input_str = input_str.strip()

        # We can't currently tokenize the dataset... we need the pre-processed text data in order to
        # build our SPE tokenizer. Once we've defined our tokenizer, we can come back and
        # tokenize the text. For now, returned the pre-processed text data
        batch[text_column_name] = input_str
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter data with inputs shorter than min_input_length or longer than max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_lengths"],
    )

    # TODO: filter data with targets longer than max_target_length
    def is_labels_in_length_range(transcription):
        return len(transcription) > min_target_length

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_in_length_range,
        num_proc=num_workers,
        input_columns=[text_column_name],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # Function to build a NeMo tokenizer manifest from a HF dataset
    def build_manifest(ds, split, manifest_path):
        with open(manifest_path, 'w') as fout:
            for sample in tqdm(ds[split]):
                # Write the metadata to the manifest
                metadata = {
                    "text": sample[text_column_name]
                }
                json.dump(metadata, fout)
                fout.write('\n')

    if not os.path.exists(model_args.manifest_path):
        os.makedirs(model_args.manifest_path)

    config.train_ds = config.validation_ds = config.test_ds = None

    if training_args.do_train:
        TRAIN_MANIFEST = os.path.join(model_args.manifest_path, "train.json")
        logger.info(f"Building training manifest at {TRAIN_MANIFEST}")
        build_manifest(vectorized_datasets, "train", TRAIN_MANIFEST)
        # Only use the train transcripts to build the tokenizer
        manifest = TRAIN_MANIFEST

        tokenizer_dir, tokenizer_type_cfg = build_tokenizer(model_args, data_args, manifest)

        # generalise the script later to load a pre-built tokenizer for eval only
        config.tokenizer.dir = tokenizer_dir
        config.tokenizer.type = tokenizer_type_cfg

    model = RNNTBPEModel(cfg=config)

    tokenizer = model.tokenizer.tokenizer.encode_as_ids

    # now that we have our model and tokenizer defined, we can tokenize the text data
    def tokenize_transcripts(batch):
        batch["labels"] = tokenizer(batch[text_column_name])
        return batch

    vectorized_datasets = vectorized_datasets.map(tokenize_transcripts, num_proc=num_workers, desc="Tokenizing datasets...", remove_columns=next(iter(raw_datasets.values())).column_names)

    # bnb optimizer
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = bnb.optim.Adam8bit(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    optimizers = (optimizer, None)

    def compute_metrics(pred):
        # Tuple of WERs returned by the model during eval: (wer, wer_num, wer_denom)
        wer_num = pred.predictions[1]
        wer_denom = pred.predictions[2]
        # compute WERs over concat batches
        wer = sum(wer_num) / sum(wer_denom)
        return {"wer": wer}

    os.environ["WANDB_PROJECT"] = data_args.wandb_project

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets['train'] if training_args.do_train else None,
        eval_dataset=vectorized_datasets['eval'] if training_args.do_eval else None,
        data_collator=NeMoDataCollator,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Running Final Evaluation (beam search) ***")
        beam_decoding_config = copy.deepcopy(config.decoding)
        beam_decoding_config.strategy = "beam"  # Options are `greedy`, `greedy_batch`, `beam`, `tsd` and `alsd`
        beam_decoding_config.beam.beam_size = 4  # Increase beam size for better scores, but it will take much longer for transcription !

        trainer.model.change_decoding_strategy(beam_decoding_config)
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
