#!/usr/bin/env python3
# flake8: noqa: E501
from datasets import load_dataset
import os
import datasets
import torchaudio.functional as F
import numpy as np
import pandas as pd
import json
import webrtcvad
import struct

import sys

dataset_name, dataset_config, text_column = sys.argv[1:]
# 1. python dataset_analysis.py "speech-seq2seq/ami" "ihm" "text"
# 2. python dataset_analysis.py "librispeech_asr" "clean" "text"
# 3. python dataset_analysis.py "librispeech_asr" "other" "text"
# 4. python dataset_analysis.py "LIUM/tedlium" "release3" "text"
# 5. python dataset_analysis.py "polinaeterna/voxpopuli" "en" "normalized_text"
# 6. python dataset_analysis.py "kensho/spgispeech" "L" "transcript"
# 7. python dataset_analysis.py "speechcolab/gigaspeech" "l" "text"
# 8. python dataset_analysis.py "mozilla-foundation/common_voice_9_0" "en" "sentence"
# 9. python dataset_analysis.py "speech-seq2seq/chime4-raw" "1-channel" "text"


# Adapt
# ================================
use_auth_token = True
dataset_cache_dir = "/home/patrick/.cache/huggingface/datasets"

folder = f"{'_'.join(dataset_name.split('/'))}_{dataset_config}_stats_v3"
folder = os.path.join("/home/patrick/datasets_stats", folder)
stats_file = folder + ".txt"

int16_max = (2**15) - 1


def float_to_pcm16(wav):
    "pack a wav, in [-1, 1] numpy float array format, into 16 bit PCM bytes"
    return struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))


def separete_speech_non_speech(
    wav, sr, vad_aggressiveness=3, webrtc_sr=16_000, webrtc_window_ms=30, dilations=3
):
    """
    Args:
        wav: the waveform, a float numpy array with range [-1, 1].
        sr: sample rate of input wav.
        vad_aggressiveness: must be 1, 2 or 3. Higher is more strict about filtering nonspeech.
        webrtc_sr: must be 8000, 16000, 32000, or 48000. Use a value close to your input.
        webrtc_window_ms: must be 10, 20, or 30ms.
        dilations: number of windows to dilate VAD results. Increase if phonemes are leaking into output.
    Returns:
        wave of all regions with no speech content, concatenated together
    """
    # resample
    if sr != webrtc_sr:
        wav = librosa.core.resample(wav, sr, webrtc_sr)
        wav = F.resample(wav, sr, webrtc_sr)

    # init VAD
    vad = webrtcvad.Vad(vad_aggressiveness)

    # trim wav to integer number of windows
    W = webrtc_window_ms * webrtc_sr // 1000
    T = len(wav)
    rem = T % W
    wav = wav if rem == 0 else wav[:-rem]

    # run VAD
    windows = [wav[i * W : (i + 1) * W] for i in range(len(wav) // W)]
    va = [vad.is_speech(float_to_pcm16(win), webrtc_sr) for win in windows]

    # dilate VAD to adjacent frames
    va = np.array(va, dtype=bool)
    for _ in range(dilations):
        va[:-1] |= va[1:]
        va[1:] |= va[:-1]

    # collect all frames without VA, concat into background-audio wave
    bg = [win for win, is_speech in zip(windows, va) if not is_speech]
    speech = [win for win, is_speech in zip(windows, va) if is_speech]
    bg = bg if len(bg) == 0 else np.concatenate(bg)
    speech = speech if len(speech) == 0 else np.concatenate(speech)
    return bg, speech, va


def mean_noise_power(wav, sr, webrtc_window_ms):
    # normalize
    wav /= abs(wav).max().clip(1e-6)

    # obtain nonspeech samples
    bg, speech, vad = separete_speech_non_speech(wav, sr, webrtc_window_ms=webrtc_window_ms)

    bg_len = bg.shape[-1] if len(bg) > 0 else 0
    speech_len = speech.shape[-1] if len(speech) > 0 else 0

    dNoiseEng = (bg**2).mean() if len(bg) > 0 else 0

    try:
        dSigEng = (speech**2).mean() - dNoiseEng
        snr = 10 * np.log10(dSigEng / dNoiseEng)
    except:
        snr = None

    return vad, snr, bg_len, speech_len


def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9

    # init
    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array(
        [
            0.40974774,
            0.40986926,
            0.40998566,
            0.40969089,
            0.40986186,
            0.40999006,
            0.41027138,
            0.41052627,
            0.41101024,
            0.41143264,
            0.41231718,
            0.41337272,
            0.41526426,
            0.4178192,
            0.42077252,
            0.42452799,
            0.42918886,
            0.43510373,
            0.44234195,
            0.45161485,
            0.46221153,
            0.47491647,
            0.48883809,
            0.50509236,
            0.52353709,
            0.54372088,
            0.56532427,
            0.58847532,
            0.61346212,
            0.63954496,
            0.66750818,
            0.69583724,
            0.72454762,
            0.75414799,
            0.78323148,
            0.81240985,
            0.84219775,
            0.87166406,
            0.90030504,
            0.92880418,
            0.95655449,
            0.9835349,
            1.01047155,
            1.0362095,
            1.06136425,
            1.08579312,
            1.1094819,
            1.13277995,
            1.15472826,
            1.17627308,
            1.19703503,
            1.21671694,
            1.23535898,
            1.25364313,
            1.27103891,
            1.28718029,
            1.30302865,
            1.31839527,
            1.33294817,
            1.34700935,
            1.3605727,
            1.37345513,
            1.38577122,
            1.39733504,
            1.40856397,
            1.41959619,
            1.42983624,
            1.43958467,
            1.44902176,
            1.45804831,
            1.46669568,
            1.47486938,
            1.48269965,
            1.49034339,
            1.49748214,
            1.50435106,
            1.51076426,
            1.51698915,
            1.5229097,
            1.528578,
            1.53389835,
            1.5391211,
            1.5439065,
            1.54858517,
            1.55310776,
            1.55744391,
            1.56164927,
            1.56566348,
            1.56938671,
            1.57307767,
            1.57654764,
            1.57980083,
            1.58304129,
            1.58602496,
            1.58880681,
            1.59162477,
            1.5941969,
            1.59693155,
            1.599446,
            1.60185011,
            1.60408668,
            1.60627134,
            1.60826199,
            1.61004547,
            1.61192472,
            1.61369656,
            1.61534074,
            1.61688905,
            1.61838916,
            1.61985374,
            1.62135878,
            1.62268119,
            1.62390423,
            1.62513143,
            1.62632463,
            1.6274027,
            1.62842767,
            1.62945532,
            1.6303307,
            1.63128026,
            1.63204102,
        ]
    )

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + (v3 - g_vals[wav_snr_idx]) / (
            g_vals[wav_snr_idx + 1] - g_vals[wav_snr_idx]
        ) * (db_vals[wav_snr_idx + 1] - db_vals[wav_snr_idx])

    # Calculate SNR
    dEng = sum(wav**2)
    dFactor = 10 ** (wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor)  # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor)  # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr


def snr_map(batch):
    num_words = len(batch[text_column].split(" "))
    array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    webrtc_window_ms = 30
    length_in_s = array.shape[-1] / sampling_rate
    try:
        bg_len = speech_len = 1
#        vad, snr, bg_len, speech_len = mean_noise_power(array, sampling_rate, webrtc_window_ms)

#        is_speech_array = np.array(np.repeat(vad, webrtc_window_ms * sampling_rate // 1000), dtype=np.int8)
#        speech_array = array[:is_speech_array.shape[-1]][is_speech_array == 1]
#        snr = wada_snr(speech_array)
#        OR
        snr = wada_snr(array)
        if snr == -20.0 or snr == 100.0:
            snr = None
    except:
        snr = None
        bg_len = speech_len = 0

    batch["num_words"] = num_words
    batch["length_in_s"] = length_in_s
    batch["silence_length"] = bg_len
    batch["speech_length"] = speech_len
    batch["snr"] = snr

    return batch


# remove `"False"` 
if not os.path.exists(folder) or "gigaspeech" in dataset_name and False:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        use_auth_token=use_auth_token,
        cache_dir=dataset_cache_dir,
    )
    all_dataset = datasets.concatenate_datasets(dataset.values())

    length = {k: dataset[k].num_rows for k in dataset.keys()}
    cum_sum_length = {
        k: np.cumsum(list(length.values()))[i] for i, k in enumerate(dataset.keys())
    }

    if not os.path.exists(folder):
        out = all_dataset.map(
            snr_map, batch_size=1, remove_columns=all_dataset.features.keys()
        )
        out.save_to_disk(folder)
    else:
        out = datasets.load_from_disk(folder)

    sources = all_dataset["source"]
    source_kinds = list(set(sources))
    sources = np.asarray(sources, np.int32)
else:
    sources = None
    out = datasets.load_from_disk(folder)
    cum_sum_length = None


results = {}
# SNR
snrs = np.array(out["snr"], dtype=np.float32)

if sources is not None:
    for source_type in source_kinds:
        snrs_type = snrs[sources == source_type]
        snrs_type = snrs_type[~np.isnan(snrs_type)]
        snrs_type = snrs_type[~np.isinf(snrs_type)]
        snr = pd.DataFrame(snrs_type)
        results[f"snr_{source_type}"] = snr.describe().to_dict()[0]
else:
    snrs = snrs[~np.isnan(snrs)]
    snrs = snrs[~np.isinf(snrs)]
    snr = pd.DataFrame(snrs)
    results["snr"] = snr.describe().to_dict()[0]

# Hours
hours = {}
start = 0
if cum_sum_length is not None:
    for key, value in cum_sum_length.items():
        hours[key] = round(np.sum(out["length_in_s"][start:value]) / 3600, 2)
        start = value
    results["hours"] = hours

# Other Speech
results["mean_seconds"] = round(np.mean(out["length_in_s"]), 2)

if sources is not None:
    for source_type in source_kinds:
        words_per_min_list = np.array([60 * x / y for x, y, i in zip(out["num_words"], out["length_in_s"], sources) if y != 0 and i == source_type], dtype=np.float32)
        words_per_min_list = pd.DataFrame(words_per_min_list)
        results[f"words_per_min_{source_type}"] = words_per_min_list.describe().to_dict()[0]
else:
    words_per_min_list = np.array([60 * x / y for x, y in zip(out["num_words"], out["length_in_s"]) if y != 0], dtype=np.float32)
    words_per_min_list = pd.DataFrame(words_per_min_list)
    results["words_per_min"] = words_per_min_list.describe().to_dict()[0]

results["speech_to_silence"] = round(
    np.sum(out["speech_length"]) / np.sum(out["silence_length"]), 2
)

# Text
results["mean_words"] = round(np.mean(out["num_words"]), 2)
# results["num_samples"] = length


result_str = json.dumps(results, indent=2, sort_keys=True) + "\n"
print(result_str)
with open(stats_file, "w") as f:
    f.write(result_str)
