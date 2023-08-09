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
    def __call__(self, inputs):
        """Apply MLPBlock module."""
        x = instantiate(self.config["dense"])(inputs)
        x = call(self.config["activation"])(x)
        x = instantiate(self.config["norm"])(x)
        x = instantiate(self.config["dense_out"])(x)
        x = instantiate(self.config["norm"])(x)

        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: dict

    @nn.compact
    def __call__(self, inputs, deterministic=False, out=False, mask=None):
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
        x = instantiate(config.self_attention)(x, mask)
        x = instantiate(config.dropout)(x, deterministic=deterministic)
        
        x = x + inputs

        # MLP block.
        y = instantiate(config.layer_norm)(x)
        y = MLPBlock(config=config.mlp_block)(y)

        return x + y
