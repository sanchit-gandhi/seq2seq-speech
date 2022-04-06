# Seq2Seq Speech in JAX
A JAX/Flax repository for combining a pre-trained speech encoder model (e.g. Wav2Vec2, HuBERT, WavLM) with a pre-trained text decoder model (e.g. GPT2, Bart) to yield a Speech Sequence-to-Sequence (Seq2Seq) model for automatic speech recognition. 

The script `run_flax_speech_recognition_seq2seq.py` can be used to fine-tune a Speech Seq2Seq model on one of the official speech recognition datasets or a custom dataset. It makes use of the `pmap` JAX operator to provide model parallelism of transformers.

The modelling files are based very heavily on those at Hugging Face Transformers 🤗.

![Seq2SeqModel](seq2seq.png)