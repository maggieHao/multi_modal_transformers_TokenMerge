"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

from typing import Optional
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

        return x + y
    

class StackedEncoder1DBlock(nn.Module):
    """Stacking Transformer encoder layers."""

    num_blocks: int
    encoder_1d_block: DictConfig

    @nn.compact
    def __call__(self, x, train=False, mask=None):

        # Use remat_scan to iterate over Encoder1DBlock
        attention_stack = nn.remat_scan(
            Encoder1DBlock,
            lengths=(1, self.num_blocks)
        )

        return attention_stack(layer_norm=self.encoder_1d_block["layer_norm"],
                               dropout=self.encoder_1d_block["dropout"],
                               self_attention=self.encoder_1d_block["self_attention"],
                               mlp_block=self.encoder_1d_block["mlp_block"],
                               train=train,
                               mask=mask,
                              )(x)

class MultiHeadAttentionPooling(nn.Module):
  """Multihead Attention Pooling."""
  
  query_map_input: DictConfig
  dot_product_attention: DictConfig
  layer_norm: DictConfig
  mlp_block: DictConfig

  @nn.compact
  def __call__(self, x, train=False):

    # for now infer dimension from input (TODO: consider refactoring this)
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
