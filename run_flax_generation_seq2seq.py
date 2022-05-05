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

import logging
import os
import sys
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
from datasets import DatasetDict, load_dataset, load_metric
from tqdm import tqdm

import flax
import jax
import jax.numpy as jnp
import optax
import transformers
import wandb as wandb
from flax import core, jax_utils, struct
from flax.training.common_utils import get_metrics, onehot, shard
from models.modeling_flax_speech_encoder_decoder import FlaxSpeechEncoderDecoderModel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@flax.struct.dataclass
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
    length_penalty: float = field(
        default= 1,
        metadata={"help": ""}
    )


@flax.struct.dataclass
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
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
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
        default=0,
        metadata={
            "help": "The minimum total sequence length for target text after tokenization. Sequences shorter "
            "than this will be filtered."
        },
    )
    pad_input_to_multiple_of: Optional[int] = field(
        default=32000,
        metadata={
            "help": "If set will pad the input sequence to a multiple of the provided value. "
            "This is important to avoid triggering recompilations on TPU."
        },
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set will pad the target sequence to a multiple of the provided value. "
            "This is important to avoid triggering recompilations on TPU."
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
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    wandb_project: str = field(
        default="flax-speech-recognition-seq2seq",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="Seq2Seq",
        metadata={"help": "The name of the wandb job type."},
    )
    log_first_ids: bool = field(
        default=True,
        metadata={"help": "Whether to log the first id's from the dataset. Defaults to `True`. If `False`, will log the first id's returned by the grouped length sampler."}
    )

def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@flax.struct.dataclass
class FlaxDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The begin-of-sentence of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_input_length (:obj:`float`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
        pad_input_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the input sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        pad_target_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the target sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "longest"
    target_padding: Union[bool, str] = "max_length"
    max_input_length: Optional[float] = None
    max_target_length: Optional[int] = None
    pad_input_to_multiple_of: Optional[int] = None
    pad_target_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        input_ids = [feature["input_id"] for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            max_length=self.max_input_length,
            padding=self.input_padding,
            pad_to_multiple_of=self.pad_input_to_multiple_of,
            return_tensors="np",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            pad_to_multiple_of=self.pad_target_to_multiple_of,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["inputs"] = batch.pop("input_values")
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


def get_grouped_indices(
    dataset, batch_size: int, rng: Optional[List[int]] = None, mega_batch_mult: Optional[int] = None
) -> np.array:
    """
    Adapted from the `get_length_grouped_indices` function in the PyTorch Trainer utils file (https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L486)
    Function that returns a list of indices in which each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted (if a JAX rng is specified)
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    lengths = dataset["input_length"]

    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use JAX for the random permutation as the PRNG key will be set based on the seed outside of the sampler.
    num_samples = len(lengths)
    indices = jax.random.permutation(rng, np.arange(num_samples)) if rng is not None else np.arange(num_samples)

    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = np.argmax(megabatch_maximums).item()
    # Switch to put the longest batch in first position
    # (note that this is different to the PT grouped sampler in which we only put the longest element in the first position, and not its batch)
    megabatches[0], megabatches[max_idx] = megabatches[max_idx], megabatches[0]

    megabatches = np.array([i for megabatch in megabatches for i in megabatch])

    return megabatches


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size, the last incomplete batch is dropped."""
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    return samples_idx.reshape((sections_split, batch_size))


def data_loader(dataset, batch_size, rng, sampler, collator):
    samples_idx = sampler(dataset, batch_size, rng)

    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size

    batch_idx = np.split(samples_idx, sections_split)

    for idx in batch_idx:
        samples = dataset[idx]
        batch = collator(samples)
        batch = shard(batch.data)
        yield batch


def write_wandb_log(metrics, prefix=None):
    if jax.process_index() == 0:
        log_metrics = {}
        for k, v in metrics.items():
            if prefix:
                log_metrics[f"{prefix}/{k}"] = v
            else:
                log_metrics[k] = v
        wandb.log(log_metrics, step=1)


def write_wandb_pred(pred_str, label_str, eval_ids, prefix="eval", top_ids=None, final_step=True):
    if jax.process_index() == 0:
        top_ids = top_ids if top_ids else eval_ids
        num_beams = len(pred_str)
        # convert str data to a wandb compatible format
        str_data = []
        for id in top_ids:
            if id in eval_ids:
                idx = eval_ids.index(id)
                str_data.append(
                    [eval_ids[idx], label_str[idx]] + [pred_str[beam][idx] for beam in range(num_beams)])
        columns = ["id", "label_str"] + [f"beam_{i + 1}" for i in range(num_beams)]
        wandb.log(
            {f"{prefix}/predictions": wandb.Table(columns=columns, data=str_data[:50])}, step=1,
        )
        if final_step:
            str_data = np.array(str_data)
            str_data = str_data[str_data[:, 1] != str_data[:, 2]]
            wandb.log(
                {f"{prefix}/incorrect_predictions": wandb.Table(columns=columns, data=str_data[:200000])}, step=1
            )

def main():
    # 1. Parse input arguments
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

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set the verbosity to info of the Transformers logger.
    # We only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(project=data_args.wandb_project, job_type=data_args.wandb_job_type)

    logger.info("Training/evaluation parameters %s", training_args)

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
        )

    if training_args.do_predict:
        test_split = data_args.test_split_name.split("+")
        for split in test_split:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
            )

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
    if data_args.id_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--id_column_name {data_args.id_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--id_column_name` to the correct id column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = FlaxSpeechEncoderDecoderModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # 6. Resample speech dataset ALWAYS
    raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    pad_input_to_multiple_of = data_args.pad_input_to_multiple_of
    pad_target_to_multiple_of = data_args.pad_target_to_multiple_of
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    id_column_name = data_args.id_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    if data_args.max_test_samples is not None:
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["input_id"] = batch[id_column_name]

        # process targets
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        batch["labels_length"] = len(batch["labels"])
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="preprocess train dataset",
        )

    # filter data with inputs shorter than min_input_length or longer than max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # filter data with targets shorter than min_target_length or longer than max_target_length
    def is_labels_in_length_range(length):
        return length > min_target_length and length < max_target_length

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_in_length_range,
        num_proc=num_workers,
        input_columns=["labels_length"],
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

    # 8. Load Metric
    metric = load_metric("wer")

    def compute_metrics(pred_ids: List[List[int]], label_ids: List[List[int]]):
        padded_ids = np.where(np.asarray(label_ids) == -100, tokenizer.pad_token_id, np.asarray(label_ids))
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(padded_ids, skip_special_tokens=True)

        pred_ids = np.array(pred_ids)
        num_beams = pred_ids.shape[1]
        # decode on a beam-by-beam basis
        pred_str = [tokenizer.batch_decode(pred_ids[:, beam, :], skip_special_tokens=True) for beam in reversed(range(num_beams))]
        # compute wer for top beam
        wer = metric.compute(predictions=pred_str[0], references=label_str)

        return {"wer": wer}, pred_str, label_str

    # 9. Create a single speech processor
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_target_length,
        pad_input_to_multiple_of=pad_input_to_multiple_of,
        pad_target_to_multiple_of=pad_target_to_multiple_of,
    )

    # Store some constants
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    num_eval_samples = len(vectorized_datasets["eval"])
    num_test_samples = sum([len(vectorized_datasets[split]) for split in test_split])

    # Cross entropy loss
    def loss_fn(logits, labels):
        vocab_size = logits.shape[-1]
        onehot_targets = onehot(labels, vocab_size)
        loss = optax.softmax_cross_entropy(logits, onehot_targets)
        # ignore padded tokens from loss, i.e. where labels are not set to -100
        padding = labels >= 0
        loss = loss * padding
        loss = loss.sum()
        num_labels = padding.sum()
        return loss, num_labels

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss, num_labels = loss_fn(logits, labels)

        total_samples = jax.lax.psum(num_labels, "batch")
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_map(lambda l: l / total_samples, loss)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Define generation function
    gen_kwargs = {"max_length": training_args.generation_max_length, "num_beams": training_args.generation_num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["inputs"], **gen_kwargs, length_penalty=model_args.length_penalty)
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_eval_step = jax.pmap(eval_step, "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the model on each device
    params = jax_utils.replicate(model.params)

    logger.info("***** Running Evaluation *****")
    logger.info(f"  Eval Dataset : {data_args.eval_split_name}")
    logger.info(f"  Test Dataset : {data_args.test_split_name}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_eval_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {eval_batch_size}")
    logger.info(f"  Total evaluation steps = {num_eval_samples + num_test_samples}")

    # ======================== Evaluating ==============================
    eval_metrics = []
    eval_preds = []
    eval_ids = []
    eval_labels = []

    # Generate eval set by sequentially sampling indices from the eval dataset and grouping by length
    eval_samples_idx = get_grouped_indices(vectorized_datasets["eval"], eval_batch_size)
    eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

    for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc=f"Evaluating {data_args.eval_split_name}...", position=0)):
        samples = [vectorized_datasets["eval"][int(idx)] for idx in batch_idx]
        batch = data_collator(samples)
        eval_ids.extend(batch.pop("input_ids"))
        batch = shard(batch.data)
        labels = batch["labels"]

        metrics = p_eval_step(params, batch)
        eval_metrics.append(metrics)

        # generation
        if training_args.predict_with_generate:
            generated_ids = p_generate_step(params, batch)
            eval_preds.extend(
                    jax.device_get(generated_ids.reshape(-1, gen_kwargs["num_beams"], gen_kwargs["max_length"])))
            eval_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

    # normalize eval metrics
    eval_metrics = get_metrics(eval_metrics)
    eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

    # compute WER metric and get predicted string
    wer_desc = ""
    pred_str = []
    label_str = []
    if training_args.predict_with_generate:
        wer_metric, pred_str, label_str = compute_metrics(eval_preds, eval_labels)
        eval_metrics.update(wer_metric)
        wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])

    # Print metrics
    desc = f"Eval Loss: {eval_metrics['loss']} | {wer_desc})"
    logger.info(desc)

    # Save metrics
    write_wandb_log(eval_metrics, prefix="eval")
    write_wandb_pred(pred_str, label_str, eval_ids,
                     top_ids=vectorized_datasets["eval"]["input_id"] if data_args.log_first_ids else None)

    # ======================== Prediction ==============================
    if training_args.do_predict:
        for step_offset, split in enumerate(test_split, 1):
            pred_metrics = []
            pred_generations = []
            pred_ids = []
            pred_labels = []

            # Generate eval set by sequentially sampling indices from the eval dataset and grouping by length
            pred_samples_idx = get_grouped_indices(vectorized_datasets[split], eval_batch_size)
            pred_batch_idx = generate_batch_splits(pred_samples_idx, eval_batch_size)

            for i, batch_idx in enumerate(tqdm(pred_batch_idx, desc=f"Predicting {split}...", position=0)):
                samples = [vectorized_datasets[split][int(idx)] for idx in batch_idx]
                batch = data_collator(samples)
                pred_ids.extend(batch.pop("input_ids"))
                batch = shard(batch.data)
                labels = batch["labels"]

                metrics = p_eval_step(params, batch)
                pred_metrics.append(metrics)

                # generation
                if training_args.predict_with_generate:
                    generated_ids = p_generate_step(params, batch)
                    pred_generations.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["num_beams"], gen_kwargs["max_length"])))
                    pred_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

            # normalize eval metrics
            pred_metrics = get_metrics(pred_metrics)
            pred_metrics = jax.tree_map(jnp.mean, pred_metrics)

            # compute WER metric and get predicted string (for debugging)
            wer_desc = ""
            pred_str = []
            label_str = []
            if training_args.predict_with_generate:
                wer_metric, pred_str, label_str = compute_metrics(pred_generations, pred_labels)
                pred_metrics.update(wer_metric)
                wer_desc = " ".join([f"{split} {key}: {value} |" for key, value in wer_metric.items()])

            # Print metrics and update progress bar
            desc = f"{split} Loss: {pred_metrics['loss']} | {wer_desc})"
            logger.info(desc)

            # Save metrics
            write_wandb_log(pred_metrics, prefix=split)
            write_wandb_pred(pred_str, label_str, pred_ids, prefix=split,
                             top_ids=vectorized_datasets[split]["input_id"] if data_args.log_first_ids else None)

if __name__ == "__main__":
    main()
