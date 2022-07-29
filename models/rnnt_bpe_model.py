from dataclasses import dataclass
from typing import Optional

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel
from omegaconf import DictConfig
from transformers.utils import ModelOutput


@dataclass
class RNNTOutput(ModelOutput):
    """
    Base class for RNNT outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    wer: Optional[float] = None
    wer_num: Optional[float] = None
    wer_denom: Optional[float] = None


class RNNTBPEModel(EncDecRNNTBPEModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, trainer=None)

    def encoding(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the acoustic model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `forward` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def forward(self, inputs, input_lengths=None, labels=None, label_lengths=None, compute_wer=False):
        # encoding() only performs encoder forward
        encoded, encoded_len = self.encoding(input_signal=inputs, input_signal_length=input_lengths)
        del inputs

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=labels, target_length=label_lengths)

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=labels, input_lengths=encoded_len, target_lengths=target_length
            )
            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)
            wer = wer_num = wer_denom = None
            if compute_wer:
                self.wer.update(encoded, encoded_len, labels, target_length)
                wer, wer_num, wer_denom = self.wer.compute()
                self.wer.reset()

        else:
            # If experimental fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=labels,
                transcript_lengths=label_lengths,
                compute_wer=compute_wer,
            )
            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

        return RNNTOutput(loss=loss_value, wer=wer, wer_num=wer_num, wer_denom=wer_denom)
