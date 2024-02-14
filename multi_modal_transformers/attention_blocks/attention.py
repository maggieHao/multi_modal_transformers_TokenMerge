"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import flax.linen as nn
from flax.linen import initializers

from hydra.utils import call, instantiate

###########################
# Decoder-only Transformer
###########################


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    config: dict

    @nn.compact
    def __call__(self, inputs, train=False):
        """Apply MLPBlock module."""
        x = instantiate(self.config["dense"])(inputs)
        x = call(self.config["activation"])(x)
        x = instantiate(self.config["norm"])(x, not train)
        x = instantiate(self.config["dense_out"])(x)
        x = instantiate(self.config["norm"])(x, not train)

        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: dict

    @nn.compact
    def __call__(self, inputs, train=False, mask=None):
        """Apply Encoder1DBlock module.

        Args:
          inputs: input data.
          mask: self-attention mask.

        Returns:
          output after transformer encoder block.
        """
        config = self.config

        # Attention block.
        x = instantiate(config.layer_norm)(inputs)
        x = instantiate(config.self_attention)(x, mask, not train)
        x = instantiate(config.dropout)(x, not train)
        
        x = x + inputs

        # MLP block.
        y = instantiate(config.layer_norm)(x)
        y = MLPBlock(config.mlp_block)(y, train)

        return x + y
