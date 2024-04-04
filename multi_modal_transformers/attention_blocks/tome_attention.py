"""
Attention Block with ToMe.
"""

from typing import Optional, Callable
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import math
from flax.linen import initializers

from omegaconf import DictConfig
from hydra.utils import call, instantiate
from multi_modal_transformers.tokenizers.token_compression import bipartite_soft_matching, merge_wavg

class ToMeMultiHeadDotProductAttention(nn.Module):
  """
  Multi-head dot-product attention with ToMe.

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
  attention_fn: Callable[..., Array] = dot_product_attention
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
    sow_weights: bool = False,
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
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection. Remember to mark 'intermediates' as
        mutable via ``mutable=['intermediates']`` in order to have that
        collection returned.

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
    # should be defined early, from the attention configs etc.
    # size should be defined in the model and returned in the end
    r = 5
    keys_avg = jnp.sum(random_array, axis=2)
    merge = bipartite_soft_matching(keys_avg, r = r)
    # updated x after merging
    x, size = merge_wavg(merge, x, size) 

    # apply attention
    if sow_weights:
      x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        module=self,
      )  # pytype: disable=wrong-keyword-args
    else:
      x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
      )
    # back to the original inputs dimensions
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




class ToMeEncoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    layer_norm: DictConfig 
    dropout: DictConfig
    self_attention: DictConfig
    mlp_block: DictConfig
    train: Optional[bool] = None
    mask: Optional[ArrayLike] = None
    
    @nn.compact
    def __call__(self, inputs, mask=None, train=None):
        
        train = nn.merge_param('train', self.train, train)
        mask = nn.merge_param('mask', self.mask, mask)

        # Attention block.
        x = instantiate(self.layer_norm)(inputs)
        x = instantiate(self.self_attention)(x, x, mask=mask, deterministic = not train)
        x = instantiate(self.dropout)(x, not train)
        
        # skip connection
        x = x + inputs

        # MLP block.
        y = instantiate(self.layer_norm)(x)
        y = instantiate(self.mlp_block, _recursive_=False)(y, train)

        return x + y, None

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

class StackedEncoder1DBlock(nn.Module):
    """Stacking Transformer encoder layers."""

    num_blocks: int
    encoder_1d_block: DictConfig

    @nn.compact
    def __call__(self, x, train=False, mask=None):
        
        # apply learnt position embedding
        x = AddPositionEmbedding(
                posemb_init=nn.initializers.normal(stddev=0.02),
                name="posembed_input",
                )(x)

        # Use scan to iterate over ToMeEncoder1DBlock layers
        attention_stack = nn.scan(
            Encoder1DBlock,
            variable_axes={'params': 0},
            variable_broadcast=False, 
            split_rngs={'params': True, 'dropout': True},
            length=self.num_blocks,
            )

        x, _ = attention_stack(layer_norm=self.encoder_1d_block["layer_norm"],
                               dropout=self.encoder_1d_block["dropout"],
                               self_attention=self.encoder_1d_block["self_attention"],
                               mlp_block=self.encoder_1d_block["mlp_block"],
                               train=train,
                               mask=mask,
                              )(x, None)
        
        return x


if __name__=="__main__":

    # add basic tests for merging modules here

