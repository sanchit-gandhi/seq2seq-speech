#!/usr/bin/env python3
from datasets import load_dataset
from collections import Counter
import json
import os
import re
import tempfile
from transformers import Wav2Vec2CTCTokenizer

# which dataset
dataset_name = "ldc/switchboard"
# which config
dataset_config = "switchboard"
# which split => @Sanchit, we should only use the train split for "fairness"
split = "train"
# in case the dataset requires access like CV9
use_auth_token = True
# name of the text data column
text_column = "text"
# name of tok to upload to the Hub
tokenizer_name = "wav2vec2_ctc_swb_tokenizer"
# dataset cache directory
dataset_cache_dir = "/home/sanchitgandhi/cache/huggingface/datasets"
# For GigaSpeech, we need to convert spelled out punctuation to symbolic form
gigaspeech_punctuation = {"<comma>": ",", "<period>": ".", "<questionmark>": "?", "<exclamationpoint": "!"}
# chars to remove - these junk tokens get filtered out in our data preprocessing script during training
preprocessing_chars_to_remove = ['{', '}', '<', '>', '_', '[', ']']
# additional chars to remove if `remove_punctuation` is set to True
additional_chars_to_remove_regex = '[,?.!-;:"“%‘”�{}()<>' + "']"

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
def create_vocabulary_from_data(dataset, word_delimiter_token="|", do_lower=False, do_upper=False, remove_punctuation=False, cutoff_freq=0.0):
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
        if remove_punctuation:
            all_text = re.sub(additional_chars_to_remove_regex, '', all_text)

        count_chars_dict = Counter(list(all_text))
        # sort by freq
        count_chars_dict = sorted(count_chars_dict.items(), key=lambda item: (-item[1], item[0]))
        # retrieve dict, freq
        vocab, freqs = zip(*count_chars_dict)

        return {"vocab": list(vocab), "freqs": list(freqs)}

    dataset = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        remove_columns=dataset.column_names,
    )

    vocab, freqs = dataset["vocab"], dataset["freqs"]
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

    vocab = list(set(vocab) - set(chars_to_remove))

    # Wav2Vec2CTC Tokenizers always have those as the first tokens (important for CTC)
    vocab = ["<pad>", "<s>", "</s>", "<unk>"] + vocab

    alphabet = list(map(chr, range(97, 123)))

    for char in alphabet:
        char = char.upper() if do_upper else char
        if char not in vocab:
            vocab.append(char)

    # create json dict
    vocab_dict = {v: k for k, v in enumerate(list(vocab))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    return vocab_dict

# Note that the functions accepts the following important args
# 1. --do_lower
# => whether to lowercase all letters or not.
# Note that if you lowercase letters for the vocab, then you also need to
# do so when preparing the data for the training, dev and test set
# 2. --cutoff_freq
# => This is very important! Lots of datasets will contain "wrong" characters in the training set, e.g.
# characters that just occur a couple of times.
# By default, the CTC vocab creation would just add them to the vocab even if their occurance is neglectible # compared to the "super frequent" letters. We can see such characters as "errors" or irrelevant in the
# dataset, so that we should delete them from the vocab. During training they would then just be classified
# unkown <unk> tokens which the model can handle.
# In this script, we deploy a mechanism to remove all chars whose freq in % is below a certain threshold.


# To begin with, let's take a look into the charecter distribution to decide whether to lowercase everything
# and how many "incorrect" chars are in the dataset

# do_lower = False
# cutoff_freq = 0.0
# create_vocabulary_from_data(dataset, do_lower=do_lower, cutoff_freq=cutoff_freq)

"""
Total characters in dataset: 57415071
--------------------------------------------------
 Char |  Total occ |       % of total occ |
--------------------------------------------------
      |    9158936 |               15.952 |
    e |    5656975 |                9.853 |
    a |    3843802 |                6.695 |
    t |    3612796 |                6.292 |
    i |    3362877 |                5.857 |
    o |    3275590 |                5.705 |
    n |    3208804 |                5.589 |
    s |    3155007 |                5.495 |
    r |    3065229 |                5.339 |
    h |    2033409 |                3.542 |
    l |    1985225 |                3.458 |
    d |    1727989 |                 3.01 |
    c |    1399592 |                2.438 |
    u |    1167110 |                2.033 |
    m |    1075733 |                1.874 |
    f |     884318 |                 1.54 |
    . |     881733 |                1.536 |
    p |     846057 |                1.474 |
    g |     809581 |                 1.41 |
    y |     740494 |                 1.29 |
    w |     722667 |                1.259 |
    b |     606687 |                1.057 |
    " |     571968 |                0.996 |
    v |     477330 |                0.831 |
    , |     345764 |                0.602 |
    T |     332058 |                0.578 |
    k |     284615 |                0.496 |
    S |     174125 |                0.303 |
    A |     157656 |                0.275 |
    H |     156398 |                0.272 |
    C |     143595 |                 0.25 |
    I |     141826 |                0.247 |
    M |     113026 |                0.197 |
    B |     102932 |                0.179 |
    ' |      88702 |                0.154 |
    - |      88461 |                0.154 |
    x |      85563 |                0.149 |
    P |      84495 |                0.147 |
    L |      67280 |                0.117 |
    R |      67254 |                0.117 |
    W |      66508 |                0.116 |
    D |      66094 |                0.115 |
    F |      61841 |                0.108 |
    G |      59031 |                0.103 |
    E |      54387 |                0.095 |
    N |      53495 |                0.093 |
    z |      47955 |                0.084 |
    O |      43305 |                0.075 |
    j |      41654 |                0.073 |
    q |      40510 |                0.071 |
    J |      39647 |                0.069 |
    K |      31204 |                0.054 |
    U |      23517 |                0.041 |
    V |      21380 |                0.037 |
    Y |      14863 |                0.026 |
    ? |       8574 |                0.015 |
    ! |       7327 |                0.013 |
    : |       5456 |                 0.01 |
    ’ |       4864 |                0.008 |
    Z |       4735 |                0.008 |
    Q |       4488 |                0.008 |
    ; |       2781 |                0.005 |
    X |       1391 |                0.002 |
    ” |       1374 |                0.002 |
    “ |       1344 |                0.002 |
    ‘ |       1100 |                0.002 |
    — |        690 |                0.001 |
    é |        297 |                0.001 |
    ü |        177 |                  0.0 |
    ) |        122 |                  0.0 |
    ( |        121 |                  0.0 |
    ä |        109 |                  0.0 |
    ...
   阪 |          1 |                  0.0 |
    ﬂ |          1 |                  0.0 |
"""
# All right, we see lots of "wrong" tokens and also see that there is a mix of upper-case and lower-case tokens
# Let's lower-case all tokens and take a look again

# do_lower = True
# cutoff_freq = 0.0

# create_vocabulary_from_data(dataset, do_lower=do_lower, cutoff_freq=cutoff_freq)


"""
Character Occurences
Total characters in dataset: 57415071
--------------------------------------------------
 Char |  Total occ |       % of total occ |
--------------------------------------------------
      |    9158936 |               15.952 |
    e |    5711362 |                9.947 |
    a |    4001458 |                6.969 |
    t |    3944854 |                6.871 |
    i |    3504703 |                6.104 |
    s |    3329132 |                5.798 |
    o |    3318895 |                5.781 |
    n |    3262299 |                5.682 |
    r |    3132483 |                5.456 |
    h |    2189807 |                3.814 |
    l |    2052505 |                3.575 |
    d |    1794083 |                3.125 |
    c |    1543187 |                2.688 |
    u |    1190627 |                2.074 |
    m |    1188759 |                 2.07 |
    f |     946159 |                1.648 |
    p |     930552 |                1.621 |
    . |     881733 |                1.536 |
    g |     868612 |                1.513 |
    w |     789175 |                1.375 |
    y |     755357 |                1.316 |
    b |     709619 |                1.236 |
    " |     571968 |                0.996 |
    v |     498710 |                0.869 |
    , |     345764 |                0.602 |
    k |     315819 |                 0.55 |
    ' |      88702 |                0.154 |
    - |      88461 |                0.154 |
    x |      86954 |                0.151 |
    j |      81301 |                0.142 |
    z |      52690 |                0.092 |
    q |      44998 |                0.078 |
    ? |       8574 |                0.015 |
    ! |       7327 |                0.013 |
    : |       5456 |                 0.01 |
    ’ |       4864 |                0.008 |
    ; |       2781 |                0.005 |
    ” |       1374 |                0.002 |
    “ |       1344 |                0.002 |
    ‘ |       1100 |                0.002 |
    — |        690 |                0.001 |
    é |        319 |                0.001 |
    ü |        182 |                  0.0 |
    ) |        122 |                  0.0 |
    ( |        121 |                  0.0 |
    ...
   阪 |          1 |                  0.0 |
    ﬂ |          1 |                  0.0 |
"""

# Cool, now let's remove very rare, "wrong" characters. Everything belowe 0.01% (note that's 1/10,000) seems like a good estimate
# It keeps all letters of the alphabet and some punctuation, but removes clearly all incorrect letters like
# accentuated letters from German or French, Chinese letters, ...
# Running it once more and now keeping the dict
do_lower = True
do_upper = False
remove_punctuation = False
cutoff_freq = 0.01

vocab_dict = create_vocabulary_from_data(dataset, do_lower=do_lower, do_upper=do_upper, remove_punctuation=remove_punctuation, cutoff_freq=cutoff_freq)

# save vocab dict to be loaded into tokenizer
with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, "vocab.json"), "w") as file:
        json.dump(vocab_dict, file)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tmp, do_lower_case=do_upper)

# push tokenizer to the Hub
# E.g. see: https://huggingface.co/patrickvonplaten/wav2vec2_ctc_cv9_tokenizer
tokenizer.push_to_hub(tokenizer_name)
