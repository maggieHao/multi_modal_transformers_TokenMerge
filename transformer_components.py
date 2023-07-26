"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import flax.linen as nn

###########################
# Decoder-only Transformer
###########################


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    config: dict

    @nn.compact
    def __call__(self, inputs):
        """Apply MLPBlock module."""
        x = nn.Dense(
            features=self.config["hidden_size"],
            use_bias=self.config["use_bias"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
        )(inputs)
        x = nn.relu(x)
        x = nn.Dropout(
            rate=self.config["dropout_rate"],
        )(x, deterministic=False)

        x = nn.Dense(
            features=self.config["hidden_size"],
            use_bias=self.config["use_bias"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        outputs = nn.Dropout(
            rate=self.config["dropout_rate"],
        )(x, deterministic=False)

        return outputs


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: dict

    @nn.compact
    def __call__(self, inputs, mask=None):
        """Apply Encoder1DBlock module.

        Args:
          inputs: input data.
          mask: self-attention mask.

        Returns:
          output after transformer encoder block.
        """
        config = self.config

        # Attention block.
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(num_heads=config.num_heads, qkv_features=config.qkv_dim)(
            x, mask
        )

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)
        y = MLPBlock(config=config)(y)

        return x + y
