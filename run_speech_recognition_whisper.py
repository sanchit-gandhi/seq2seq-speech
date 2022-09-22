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
Fine-tuning OpenAI Whisper models for speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# flake8: noqa: E501
import copy
import logging
import os
import re
import whisper
from transformers import GPT2TokenizerFast
import sys
import evaluate
from dataclasses import dataclass, field

from typing import Optional, Dict, Union, List

import numpy as np
import torch

import datasets
from datasets import DatasetDict, load_dataset
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from OpenAI Whisper NGC."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co or OpenAI Whisper NGC."},
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
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze the acoustic encoder of the model. Recommend when fine-tuning on small datasets."}
    )
    use_adam8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes 8bit AdamW optimiser."}
    )


class SuppressBlank:
    def __init__(self, tokenizer, sample_begin: int = 1):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def __call__(self, input_ids, scores):
        tokens = input_ids
        logits = scores
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
        return logits


class SuppressTokens:
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids, scores):
        logits = scores
        logits[:, self.suppress_tokens] = -np.inf
        return logits


class ApplyTimestampRules:
    def __init__(
        self, tokenizer, sample_begin: int = 1, max_initial_timestamp_index: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def __call__(self, input_ids, scores):
        tokens = input_ids
        logits = scores
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == self.sample_begin and self.max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf
        return logits




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
    max_predict_samples: Optional[int] = field(
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
            "help": "Truncate training audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    max_eval_duration_in_seconds: float = field(
        default=None,
        metadata={
            "help": "Truncate eval/test audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
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


def WhisperDataCollator(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Data collator that will dynamically pad the inputs received.
    Since Whisper models don't have a HF processor defined (feature extractor + tokenizer), we'll pad by hand...
    The padding idx is arbitrary: we provide the model with the input lengths and label lengths, from which
    all the relevant padding information is inferred. Thus, we'll use the default np.pad padding idx (0).
    """
    # split inputs and labels since they have to be of different lengths
    # and need different padding methods
    input_ids = [feature["input_ids"] for feature in features]
    labels = [feature["labels"] for feature in features]

    def transform(array):
        padded_input = whisper.pad_or_trim(np.asarray(array, dtype=np.float32))
        input_ids = whisper.log_mel_spectrogram(padded_input)
        return input_ids

    # first, pad the audio inputs to max_len
#    input_ids = np.asarray([transform(input_val) for input_val in input_ids], dtype=np.float32)
    input_ids = torch.concat([transform(input_val)[None,:] for input_val in input_ids])

    # next, pad the target labels to max_len
    label_lengths = [len(lab) for lab in labels]
    max_label_len = max(label_lengths)
    labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]

    batch = {"labels": labels}
    batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

    batch["input_ids"] = input_ids

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

    # Set wandb project ID before instantiating the Trainer
    os.environ["WANDB_PROJECT"] = data_args.wandb_project

    sample_rate = 16_000

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

    # load the model 
    model = whisper.load_model(model_args.model_name_or_path)

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
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=sample_rate)
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * sample_rate)
    min_input_length = min(int(data_args.min_duration_in_seconds * sample_rate), 1)
    max_eval_input_length = int(data_args.max_eval_duration_in_seconds * sample_rate) if data_args.max_eval_duration_in_seconds else None
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    do_lower_case = data_args.do_lower_case
    dataset_name = data_args.dataset_name

    # Define tokens to ignore/replace
    tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]
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

    if training_args.do_predict and data_args.max_predict_samples is not None:
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_predict_samples))

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
        # pre-process audio
        try:
            sample = batch[audio_column_name]
        except ValueError:
            # E22: some samples are empty (no audio). Reading the empty audio array will trigger
            # a soundfile ValueError. For now, we'll manually set these arrays to a zero array.
            # They will be filtered in the subsequent filtering stage and so are
            # explicitly ignored during training.
            sample = {"array": np.array([0.]), "sampling_rate": sampling_rate}

        # Whisper RNNT model performs the audio preprocessing in the `.forward()` call
        # => we only need to supply it with the raw audio values
        batch["input_ids"] = sample["array"]
        batch["input_lengths"] = len(sample["array"])

        # 'Error correction' of targets
        input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]

        # LibriSpeech ASR
        if dataset_name == "librispeech_asr":
            pass  # no error correction necessary

        # VoxPopuli
        if dataset_name == "google/xtreme_s":
            pass  # no error correction necessary

        # Common Voice 9
        if dataset_name == "mozilla-foundation/common_voice_9_0":
            if input_str.startswith('"') and input_str.endswith('"'):
                # we can remove trailing quotation marks as they do not affect the transcription
                input_str = input_str[1:-1]
            # replace double quotation marks with single
            input_str = input_str.replace('""', '"')

        # TED-LIUM (Release 3)
        if dataset_name == "LIUM/tedlium":
            # delete the <unk> token from the text
            input_str = input_str.replace("<unk>", "")
            # replace spaced apostrophes with un-spaced (it 's -> it's)
            for contraction in tedlium_contractions:
                input_str = input_str.replace(contraction, contraction[1:])

        # GigaSpeech
        if dataset_name == "speechcolab/gigaspeech":
            for disfluency in gigaspeech_disfluencies:
                input_str = input_str.replace(disfluency, "")
            # convert spelled out punctuation to symbolic form
            for punctuation, replacement in gigaspeech_punctuation.items():
                input_str = input_str.replace(punctuation, replacement)

        # SWB: hide the path to the private HF dataset
        if "switchboard" in dataset_name:
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

        # Earnings 22: still figuring out best segmenting method. Thus, dataset name subject to change
        if "earnings22" in dataset_name:
            for disfluency in earnings_disfluencies:
                input_str = input_str.replace(disfluency, "")

        # SPGISpeech
        if dataset_name == "kensho/spgispeech":
            pass  # no error correction necessary

        # JIWER compliance (for WER/CER calc.)
        # remove multiple spaces
        input_str = re.sub(r"\s\s+", " ", input_str)
        # strip trailing spaces
        input_str = input_str.strip()

        # We can't currently tokenize the dataset... we need the pre-processed text data in order to
        # build our SPE tokenizer. Once we've defined our tokenizer, we can come back and
        # tokenize the text. For now, just return the pre-processed text data
        batch[text_column_name] = input_str
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter training data with inputs shorter than min_input_length or longer than max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_lengths"],
        )

    if max_eval_input_length is not None:
        # filter training data with inputs longer than max_input_length
        def is_eval_audio_in_length_range(length):
            return min_input_length < length < max_eval_input_length

        vectorized_datasets = vectorized_datasets.filter(
            is_eval_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    def is_labels_non_zero(transcription):
        return len(transcription) > 0

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_non_zero,
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

    if model_args.freeze_encoder:
        model.freeze_encoder()
        logging.info("Model encoder has been frozen")

    # now that we have our model and tokenizer defined, we can tokenize the text data
    whisper_tok = whisper.tokenizer.get_tokenizer(False, task="transcribe", language="en")
    decoding_options = whisper.decoding.DecodingOptions(task="transcribe", language="en")
    task = whisper.decoding.DecodingTask(model, decoding_options)
    suppress_tokens = task._get_suppress_tokens()

    logits_processors = [SuppressBlank(whisper_tok), SuppressTokens(suppress_tokens), ApplyTimestampRules(whisper_tok)]
    tokenizer = whisper_tok.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_transcripts(batch):
        batch["labels"] = tokenizer(batch[text_column_name]).input_ids
        return batch

    vectorized_datasets = vectorized_datasets.map(tokenize_transcripts, num_proc=num_workers,
                                                  desc="Tokenizing datasets...",
                                                  remove_columns=next(iter(raw_datasets.values())).column_names)

    # 8. Load Metric
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.eos_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [x.lstrip().strip() for x in pred_str]

        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric_wer.compute(predictions=pred_str, references=label_str)
        cer = metric_cer.compute(predictions=pred_str, references=label_str)
        print(pred_str)

        return {"wer": wer, "cer": cer}

    class WhisperTrainer(Seq2SeqTrainer):
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            self.model.save_to(save_path=os.path.join(output_dir, model_args.model_name_or_path + ".whisper"))
            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # Initialize Trainer
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets['train'] if training_args.do_train else None,
        eval_dataset=vectorized_datasets['eval'] if training_args.do_eval else None,
        data_collator=WhisperDataCollator,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
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

    # Change decoding strategy for final eval/predict
#    if training_args.do_eval or training_args.do_predict:
#        trainer.model.num_beams = 2

    results = {}
    if training_args.do_eval:
        metrics = trainer.evaluate(logits_processor=logits_processors)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        for split in test_split:
            predict_results = trainer.predict(
                vectorized_datasets[split], metric_key_prefix=split, logits_processor=logits_processors)
            metrics = predict_results.metrics
            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(vectorized_datasets[split])
            )
            metrics[f"{split}_samples"] = min(max_predict_samples, len(vectorized_datasets[split]))

            trainer.log_metrics(split, metrics)
            trainer.save_metrics(split, metrics)

            if "wandb" in training_args.report_to:
                import wandb
                metrics = {os.path.join(split, k[len(split)+1:]): v for k, v in metrics.items()}
                wandb.log(metrics)

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

    return results


if __name__ == "__main__":
    main()
