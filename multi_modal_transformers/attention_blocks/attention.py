"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

from typing import Optional, Callable
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen import initializers

from omegaconf import DictConfig
from hydra.utils import call, instantiate


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    dense: DictConfig
    activation: DictConfig
    norm: DictConfig
    dense_out: DictConfig
    
    @nn.compact
    def __call__(self, inputs, train=False):
        """Apply MLPBlock module."""
        
        x = instantiate(self.dense)(inputs)
        x = call(self.activation)(x)
        x = instantiate(self.norm)(x, not train)
        
        x = instantiate(self.dense_out)(x)
        x = instantiate(self.norm)(x, not train)
        
        return x

class Encoder1DBlock(nn.Module):
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
        x = instantiate(self.self_attention)(x, mask, not train)
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

        # Use scan to iterate over Encoder1DBlock layers
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


class MultiHeadAttentionPooling(nn.Module):
  """Multihead Attention Pooling."""
  
  query_map_input: DictConfig
  dot_product_attention: DictConfig
  layer_norm: DictConfig
  mlp_block: DictConfig

  @nn.compact
  def __call__(self, x, train=False):

    # for now infer dimension from input
    batch_size, sequence_length, embedding_dim = x.shape

    # the pool operation uses a learnt input for generating query tensor
    # below we instantiate the parameters associated to this learnt input.
    query = self.param(
            "learnt_q_input", 
            call(self.query_map_input.kernel_init),
            (1, 1, embedding_dim), 
            )
    query = jnp.tile(query, [batch_size, 1, 1])

    # regular attention block with learnt inputs 
    x = instantiate(self.dot_product_attention)(query, x)
    y = instantiate(self.layer_norm)(x)
    y = instantiate(self.mlp_block, _recursive_=False)(y, train)

    return x + y
