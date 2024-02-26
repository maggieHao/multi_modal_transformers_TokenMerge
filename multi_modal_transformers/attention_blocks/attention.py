"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import flax
import flax.linen as nn
from flax.linen import initializers

from omegaconf import DictConfig
from hydra.utils import call, instantiate

###########################
# Decoder-only Transformer
###########################


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
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    layer_norm: DictConfig 
    dropout: DictConfig
    self_attention: DictConfig
    mlp_block: DictConfig

    @nn.compact
    def __call__(self, inputs, train=False, mask=None):
        """Apply Encoder1DBlock module.

        Args:
          inputs: input data.
          mask: self-attention mask.

        Returns:
          output after transformer encoder block.
        """
        # Attention block.
        x = instantiate(self.layer_norm)(inputs)
        x = instantiate(self.self_attention)(x, mask, not train)
        x = instantiate(self.dropout)(x, not train)
        
        x = x + inputs

        # MLP block.
        y = instantiate(self.layer_norm)(x)
        y = instantiate(self.mlp_block, _recursive_=False)(y, train)

        return x + y
