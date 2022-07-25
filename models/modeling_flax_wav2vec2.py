# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
""" Flax Wav2Vec2 model."""

from functools import partial
from typing import Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import ModelOutput

from models import Wav2Vec2Config

scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat


@flax.struct.dataclass
class FlaxWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`FlaxWav2Vec2BaseModelOutput`], with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`jnp.ndarray` of shape `(batch_size, sequence_length, last_conv_dim)`):
            Sequence of extracted feature vectors of the last convolutional layer of the model with `last_conv_dim`
            being the dimension of the last convolutional layer.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jnp.ndarray = None
    extract_features: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


WAV_2_VEC_2_START_DOCSTRING = r"""
    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""


WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *jnp.ndarray*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask) .. warning:: `attention_mask` should only be passed
            if the corresponding processor has `config.return_attention_mask == True`. For all models whose processor
            has `config.return_attention_mask == False`, such as
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), `attention_mask` should **not** be
            passed to avoid degraded performance when doing batched inference. For such models `input_values` should
            simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly
            different results depending on whether `input_values` is padded or not.
        mask_time_indices (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],
            kernel_size=(self.config.conv_kernel[self.layer_id],),
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxConvWithWeightNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(self.config.num_conv_pos_embeddings,),
            kernel_init=jax.nn.initializers.he_normal(),
            padding="VALID",
            feature_group_count=self.config.num_conv_pos_embedding_groups,
            dtype=self.dtype,
        )
        weight_shape = (
            self.conv.features,
            self.conv.features // self.conv.feature_group_count,
            self.conv.kernel_size[0],
        )
        self.weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)
        self.weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :])
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))
        self.prev_padding = self.conv.kernel_size[0] // 2

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        hidden_states = jnp.pad(hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0)))
        hidden_states = self.conv.apply({"params": {"kernel": kernel.T, "bias": self.bias}}, hidden_states)
        return hidden_states


class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = FlaxConvWithWeightNorm(self.config, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, hidden_states):
        hidden_states = hidden_states.transpose((0, 1, 2))

        hidden_states = self.conv(hidden_states)

        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose((0, 1, 2))
        return hidden_states


class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.feat_extract_norm == "layer":
            # note that we can't use scan on the conv layers as they differ on a layer-by-layer basis
            BlockLayer = remat(FlaxWav2Vec2LayerNormConvLayer) if self.config.gradient_checkpointing else FlaxWav2Vec2LayerNormConvLayer
            self.layers = [
                BlockLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_feat_extract_layers)
            ]
        elif self.config.feat_extract_norm == "group":
            raise NotImplementedError("At the moment only ``config.feat_extact_norm == 'layer'`` is supported")
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )

    def __call__(self, hidden_states):
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class FlaxWav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, input_values, freeze_feature_encoder=False):
        hidden_states = input_values[:, :, None]
        hidden_states = self.conv_layers(hidden_states)
        if freeze_feature_encoder:
            hidden_states = jax.lax.stop_gradient(hidden_states)
        return hidden_states


class FlaxWav2Vec2FeatureProjection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.projection = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.feat_proj_dropout)

    def __call__(self, hidden_states, deterministic=True):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states, norm_hidden_states


class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()

        self.fused_proj = nn.Dense(
            self.embed_dim * 3,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.out_proj = dense()

        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""

        if self.config.fuse_matmuls:
            attention_states = self.fused_proj(hidden_states)
            query_states, key_states, value_states = jnp.split(attention_states, 3, axis=-1)

        else:
            # get query proj
            query_states = self.q_proj(hidden_states)

            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, float("-inf")).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class FlaxWav2Vec2FeedForward(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.intermediate_dropout = nn.Dropout(rate=self.config.activation_dropout)

        self.intermediate_dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        if isinstance(self.config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_act_fn = self.config.hidden_act

        self.output_dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.output_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, deterministic=deterministic)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxWav2Vec2Attention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False):
        if self.config.use_scan:
            hidden_states = hidden_states[0]
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states), deterministic=deterministic
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if self.config.use_scan:
            outputs = (outputs, None)

        return outputs


class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        num_layers = self.config.num_hidden_layers
        BlockEncoderLayer = (
            remat(
                FlaxWav2Vec2EncoderLayerStableLayerNorm,
                static_argnums=(2, 3),
                prevent_cse=not self.config.use_scan,
            )
            if self.config.gradient_checkpointing
            else FlaxWav2Vec2EncoderLayerStableLayerNorm
        )

        if self.config.use_scan:
            # since all decoder layers are the same, we use nn.scan directly
            assert not output_attentions, "cannot use `scan` with `output_attentions` set to `True`"
            assert not output_hidden_states, "cannot use `scan` with `output_hidden_states` set to `True`"
            hidden_states = (hidden_states,)

            hidden_states, _ = scan_with_axes(
                BlockEncoderLayer,
                variable_axes={"params": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
                length=num_layers,
            )(self.config, dtype=self.dtype, name="FlaxWav2Vec2EncoderLayers",)(
                hidden_states, attention_mask, deterministic, output_attentions
            )
            hidden_states = hidden_states[0]

        else:
            for layer in range(num_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = BlockEncoderLayer(
                    self.config,
                    dtype=self.dtype,
                    name=str(layer),
                )(hidden_states, attention_mask, deterministic, output_attentions)

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pos_conv_embed = FlaxWav2Vec2PositionalConvEmbedding(self.config, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layers = FlaxWav2Vec2EncoderLayerStableLayerNormCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states = jnp.where(
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape), hidden_states, 0
            )

        position_embeddings = self.pos_conv_embed(hidden_states)

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = self.layer_norm(outputs[0])

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=outputs.attentions
        )


class FlaxWav2Vec2Adapter(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # hidden_states require down-projection if feature dims don't match
        if self.config.output_hidden_size != self.config.hidden_size:
            self.proj = nn.Dense(
                self.config.output_hidden_size,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                dtype=self.dtype,
            )
            self.proj_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = FlaxWav2Vec2AdapterLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        # down-project hidden_states if required
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = self.layers(hidden_states)

        return hidden_states


class FlaxWav2Vec2AdapterLayer(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=2 * self.config.output_hidden_size,
            kernel_size=(self.config.adapter_kernel_size,),
            strides=(self.config.adapter_stride,),
            padding=((1, 1),),
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.glu(hidden_states, axis=2)

        return hidden_states


class FlaxWav2Vec2AdapterLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        BlockAdapterLayer = remat(FlaxWav2Vec2AdapterLayer) if self.config.gradient_checkpointing else FlaxWav2Vec2AdapterLayer
        self.layers = [
            BlockAdapterLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_adapter_layers)
        ]

    def __call__(self, hidden_states):
        for conv_layer in self.layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix: str = "wav2vec2"
    main_input_name = "input_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Wav2Vec2Config,
        input_shape: Tuple = (1, 1024),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_values = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_values)
        params_rng, dropout_rng = jax.random.split(rng, 2)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_values, attention_mask, return_dict=False)["params"]

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        extract_features=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_features: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            batch_size, sequence_length = input_values.shape
            attention_mask = jnp.ones((batch_size, sequence_length))

        if extract_features is not None:
            extract_features = jnp.array(extract_features, dtype="f4")

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            extract_features,
            not train,
            output_attentions,
            output_hidden_states,
            output_features,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        return self.module._get_feature_vector_attention_mask(feature_vector_length, attention_mask, add_adapter=add_adapter)


class FlaxWav2Vec2Module(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(self.config, dtype=self.dtype)
        self.feature_projection = FlaxWav2Vec2FeatureProjection(self.config, dtype=self.dtype)
        self.masked_spec_embed = self.param(
            "masked_spec_embed", jax.nn.initializers.uniform(), (self.config.hidden_size,)
        )

        if self.config.do_stable_layer_norm:
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(self.config, dtype=self.dtype)
        else:
            raise NotImplementedError("``config.do_stable_layer_norm is False`` is currently not supported.")

        self.adapter = FlaxWav2Vec2Adapter(self.config, dtype=self.dtype) if self.config.add_adapter else None

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        extract_features=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        output_features=False,
        freeze_feature_encoder=False,
        return_dict=None,
    ):

        # forward pass through the feature extractor if features not specified
        if extract_features is None:
            extract_features = self.feature_extractor(input_values, freeze_feature_encoder=freeze_feature_encoder)

        if output_features:
            return extract_features

        # make sure that no loss is computed on padded inputs
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features, deterministic=deterministic)
        if mask_time_indices is not None:  # apply SpecAugment along time axis with given indices
            hidden_states = jnp.where(
                jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape),
                jnp.broadcast_to(self.masked_spec_embed[None, None, :], hidden_states.shape),
                hidden_states,
            )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return FlaxWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)

        batch_size = attention_mask.shape[0]

        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype("bool")
        return attention_mask


class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2Module


class FlaxWav2Vec2ForCTCModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.final_dropout)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        extract_features=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        output_features=False,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freeze_feature_encoder=freeze_feature_encoder,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxCausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)

        batch_size = attention_mask.shape[0]

        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype("bool")
        return attention_mask


class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel):
    module_class = FlaxWav2Vec2ForCTCModule
