#!/usr/bin/env python3
from datasets import load_dataset
from collections import Counter
import re
import os
from transformers import Wav2Vec2ProcessorWithLM
from transformers import AutoProcessor
from pathlib import Path
from pyctcdecode import build_ctcdecoder


# adapt to dataset
dataset_name = "LIUM/tedlium"
dataset_config = "release3"
split = "train"
text_column = "text"
tokenizer_name = "sanchit-gandhi/flax-wav2vec2-ctc-earnings22-cased-hidden-activation-featproj-dropout-0.2"
do_lower = False # only set to TRUE if dataset is NOT cased

# should be kept the same across datasets (except for ablation)
cutoff_freq = 0.01
do_upper = False  # only set to TRUE for ablation studies
preprocessing_chars_to_remove = []  # only remove chars for ablation studies
additional_chars_to_remove_regex = ""  # only set to something for ablation studies
remove_punctuation = additional_chars_to_remove_regex != ""
# dataset specific "error correction"
# For GigaSpeech, we need to convert spelled out punctuation to symbolic form
gigaspeech_punctuation = {"<comma>": ",", "<period>": ".", "<questionmark>": "?", "<exclamationpoint": "!"}

# Filenames
file_name = "5gram.arpa"
home_path = "/home/patrick"
dir_path = f"{home_path}/ngrams/{dataset_name.split('/')[-1]}"
text_save_path = f"{dir_path}/text.txt"
ngram_save_path = f"{dir_path}/{file_name}"
dataset_cache_dir = "/home/patrick/.cache/huggingface/datasets"

# in case the dataset requires access like CV9
use_auth_token = True

dataset = load_dataset(
    dataset_name,
    dataset_config,
    split=split,
    use_auth_token=use_auth_token,
    cache_dir=dataset_cache_dir,
)

# remove all data that is unnecessary to save RAM
dataset = dataset.remove_columns(list(set(dataset.column_names) - set([text_column])))

# define function to see stats about letters and to create vocab
# NOTE: this function has to be 1-to-1 aligned with:
# https://github.com/sanchit-gandhi/seq2seq-speech/blob/25d3af18d779d12cdb6c30040f30f51f5a6bb75b/get_ctc_tokenizer.py#L45
def process_text(dataset, word_delimiter_token="|", do_lower=False, do_upper=False, remove_punctuation=False, cutoff_freq=0.0):
    def extract_all_chars(batch):
        all_text = " ".join(batch[text_column])

        if do_lower and do_upper:
            raise ValueError("Cannot do uppercase and lowercase tokenization concurrently. Set at most one of `do_lower` or `do_upper` to `True`.")
        if do_lower:
            all_text = all_text.lower()
        if do_upper:
            all_text = all_text.upper()
        for punctuation, replacement in gigaspeech_punctuation.items():
            all_text = all_text.replace(punctuation.lower(), replacement)
            all_text = all_text.replace(punctuation.upper(), replacement)
        for char in preprocessing_chars_to_remove:
            all_text = all_text.replace(char, "")
        # only used for ablation studies
        if remove_punctuation:
            all_text = re.sub(additional_chars_to_remove_regex, '', all_text)

        count_chars_dict = Counter(list(all_text))
        # sort by freq
        count_chars_dict = sorted(count_chars_dict.items(), key=lambda item: (-item[1], item[0]))
        # retrieve dict, freq
        vocab, freqs = zip(*count_chars_dict)

        result = {"vocab": [list(vocab)], "freqs": [list(freqs)]}
        result[text_column] = [all_text]

        return result

    dataset = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        remove_columns=dataset.column_names,
    )

    vocab, freqs = dataset["vocab"][0], dataset["freqs"][0]
    total_num_chars = sum(freqs)
    chars_to_remove = []

    print("Character Occurences")
    print(f"Total characters in dataset: {total_num_chars}")
    print(50 * "-")
    print(f"{'Char'.rjust(5)} | {'Total occ'.rjust(10)} | {'% of total occ'.rjust(20)} |")
    print(50 * "-")
    for char, freq in zip(vocab, freqs):
        freq_in_percent = freq / total_num_chars * 100
        print(f"{char.rjust(5)} | {str(freq).rjust(10)} | {str(round(freq_in_percent, 3)).rjust(20)} |")
        if freq_in_percent < cutoff_freq:
            chars_to_remove.append(char)
    print(50 * "-")
    print(f"REMOVED CHARS: {chars_to_remove}")
    print(50 * "-")


    def remove_rare_occurence_chars(batch):
        all_text = " ".join(batch[text_column])
        for char in chars_to_remove:
            all_text = all_text.replace(char, "")

        result = {}
        result[text_column] = [all_text]

        return result

    dataset = dataset.map(
        remove_rare_occurence_chars,
        batched=True,
        batch_size=-1,
        remove_columns=dataset.column_names,
    )

    return dataset


# Cool, now let's remove very rare, "wrong" characters. Everything belowe 0.01% (note that's 1/10,000) seems like a good estimate
# It keeps all letters of the alphabet and some punctuation, but removes clearly all incorrect letters like 
# accentuated letters from German or French, Chinese letters, ...
# Running it once more and now keeping the dict

if not os.path.isfile(text_save_path):
    text_data = process_text(dataset, do_lower=do_lower, do_upper=do_upper, remove_punctuation=remove_punctuation, cutoff_freq=cutoff_freq)

    # save vocab dict to be loaded into tokenizer
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    with open(text_save_path, "w") as file:
        file.write(" ".join(text_data[text_column]))

if not os.path.isfile(ngram_save_path):
    ngram_command = f"/home/patrick/kenlm/build/bin/lmplz -o 5 < '{text_save_path}' > '{ngram_save_path}' --skip_symbols"
    os.system(ngram_command)

    # correct with "</s>"
    ngram_save_path_correct = ngram_save_path + "_correct.arpa"
    with open(ngram_save_path, "r") as read_file, open(ngram_save_path_correct, "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)

    os.system(f"mv {ngram_save_path_correct} {ngram_save_path}")


processor = AutoProcessor.from_pretrained(tokenizer_name)
vocab_dict = processor.tokenizer.get_vocab()
if do_lower:
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
else:
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

processor.tokenizer.encoder = sorted_vocab_dict
processor.tokenizer.decoder = {v: k for k, v in processor.tokenizer.encoder.items()}

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=ngram_save_path,
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)
processor_with_lm.save_pretrained(dir_path)

new_ngram_path = os.path.join(dir_path, "language_model", file_name)
bin_save_path = new_ngram_path.split(".")[0] + ".bin"
os.system(f"{home_path}/kenlm/build/bin/build_binary '{new_ngram_path}' '{bin_save_path}'")
os.system(f"mv '{bin_save_path}' '{new_ngram_path}'")



# CONFIGS
# 1. LIBRISPEECH:
#    dataset_name = "librispeech_asr"
#    dataset_config = None
#    split = "train.clean.100+train.clean.360+train.other.500"
#    text_column = "text"
#    tokenizer_name = "sanchit-gandhi/flax-wav2vec2-ctc-ls-960h-baseline"
#    do_lower = True  # only set to TRUE if dataset is NOT cased

#    => no REMOVED CHARS: [] -> all chars are used

# ========================================================================
# 2. Earnings 22:
#    dataset_name = "sanchit-gandhi/earnings22_robust_split"
#    dataset_config = None
#    split = "train"
#    text_column = "sentence"
#    tokenizer_name = "sanchit-gandhi/flax-wav2vec2-ctc-earnings22-cased-hidden-activation-featproj-dropout-0.2"
#    do_lower = False # only set to TRUE if dataset is NOT cased

#    => REMOVED CHARS: ['&', 'X', ';', '€', 'Z', '/', ':', '*', '£', '¥', 'ł', '₽', '!', '+', '–', '@', '¢', '₵', '\\', '_', '#', '=', 'ı', '₦', '[', 'ø', '₱']
