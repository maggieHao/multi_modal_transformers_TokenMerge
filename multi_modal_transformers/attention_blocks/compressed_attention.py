"""
Attention Blocks with token compression methods.
"""
import functools
import warnings
from typing import Any, Callable, Optional, Union, overload

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.typing import ArrayLike

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
  DenseGeneral,
  default_kernel_init,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import LayerNorm
from flax.typing import (
  Array,
  PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

import flax
import flax.linen as nn
import einops as e

from omegaconf import DictConfig
from hydra.utils import call, instantiate

from multi_modal_transformers.tokenizers.token_compression import compute_top_k_tokens


class CompressedMultiHeadDotProductAttention(nn.Module):
  """
  Multi-head dot-product attention with token compression methods.

  Adapted from Flax's MultiHeadDotProductAttention module.
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  use_bias: bool = True
  prune_fn: Callable = None
  merge_fn: Callable = None
  decode: bool = False
  normalize_qk: bool = False
  # Deprecated, will be removed.
  qkv_dot_general: Optional[DotGeneralT] = None
  out_dot_general: Optional[DotGeneralT] = None
  qkv_dot_general_cls: Any = None
  out_dot_general_cls: Any = None

  @compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
      inputs_k: key of shape ``[batch_sizes..., length, features]``. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape ``[batch_sizes..., length, features]``. If None,
        inputs_v will copy the value of inputs_k.
      inputs_kv: key/values of shape ``[batch_sizes..., length, features]``. If
        None, inputs_kv will copy the value of inputs_q. This arg will be
        deprecated soon. Use inputs_k and inputs_v instead.
      mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
        key/value_length]``. Attention weights are masked out if their
        corresponding mask value is ``False``.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      dropout_rng: optional rng key to pass to the attention layer's dropout
        mask. Otherwise, self.make_rng('dropout') is used instead.

    Returns:
      output of shape ``[batch_sizes..., length, features]``.
    """
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. '
            'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
            'value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
      DenseGeneral,
      axis=-1,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      features=(self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
      dense(name='query')(inputs_q),
      dense(name='key')(inputs_k),
      dense(name='value')(inputs_v),
    )
    
    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(
        name='query_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(query)  # type: ignore[call-arg]
      key = LayerNorm(
        name='key_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[Union[int, jax.Array], ...] = (zero,) * len(
          batch_dims
        ) + (
          cur_index,
          zero,
          zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index,
            tuple(batch_dims) + (1, 1, max_length),
          ),
        )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # TODO: add ToMe here
    

    # apply attention
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), 'q, k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        module,
    )

    # return weighted sum over values for each query position
    x = jnp.einsum(
        '...hqk,...khd->...qhd', attn_weights, value, precision=precision
        )

    # calculate importance scores for token pruning
    importance_scores = jnp.mean( # mean across heads
            jnp.mean(attn_weights, axis=-1), # mean across sequence
            axis=-2) # batch, num_tokens

    # compress tokens with prune_fn
    x = self.prune_fn(x, importance_scores)

    # compress tokens with merge_fn
    #x = self.merge_fn()
    
    # output projection
    out = DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      name='out',  # type: ignore[call-arg]
    )(x)

    return out

class CompressedEncoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    layer_norm: DictConfig 
    dropout: DictConfig
    self_attention: DictConfig
    mlp_block: DictConfig
    train: Optional[bool] = None
    prune_fn: Optional[Callable] = None
    merge_fn: Optional[Callable] = None
    
    @nn.compact
    def __call__(self, inputs, mask=None, train=None):

        # Attention block.
        x = instantiate(self.layer_norm)(inputs)
        
        # require partial instantiation for compressed attention as we pass fn
        compressed_attn = instantiate(self.self_attention, _partial_=True)
        compressed_attn(prune_fn=self.prune_fn, merge_fn=self.merge_fn)(x, x, mask=mask, train=not train)
        
        x = instantiate(self.dropout)(x, not train)
        
        # skip connection
        x = x + inputs

        # MLP block.
        y = instantiate(self.layer_norm)(x)
        y = instantiate(self.mlp_block, _recursive_=False)(y, train)

        return x + y

class AddPositionEmbedding(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    posemb_init: Callable

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module."""
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pe

class StackedCompressedEncoder1DBlock(nn.Module):
    """Stacking Transformer encoder layers."""

    num_blocks: int
    encoder_1d_block: DictConfig
    prune_fns: Optional[Callable] = None
    merge_fns: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, masks, train=False):
        
        # apply learnt position embedding
        x = AddPositionEmbedding(
                posemb_init=nn.initializers.normal(stddev=0.02),
                name="posembed_input",
                )(x)

        # TODO: consider converting to scan later
        for layer_idx in range(self.num_blocks):
            encoder_block = instantiate(self.encoder_1d_block, _partial_=True)
            x = encoder_block(self.prune_fns[layer_idx], self.merge_fns[layer_idx])(
                    x, 
                    mask=masks[layer_idx],
                    train=train,
                    )
        
        return x


