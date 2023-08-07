"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import flax.linen as nn
from flax.linen import initializers

###########################
# Decoder-only Transformer
###########################


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    config: dict

    @nn.compact
    def __call__(self, inputs):
        """Apply MLPBlock module."""
        actual_out_dim = inputs.shape[-1] if self.config["out_dim"] is None else self.config["out_dim"]
        

        if self.config.train_parallel:
            x = nn.Dense(
                features=self.config["hidden_size"],
                use_bias=self.config["use_bias"],
                kernel_init=nn.with_partitioning(
                    nn.initializers.he_normal(),
                    (None, 'model'),
                    ),
                bias_init=nn.with_partitioning(
                    nn.initializers.constant(0.0),
                    (None, 'model'),
                    ),
            )(inputs)
            x = nn.relu(x)
            x = nn.Dropout(
                rate=self.config["dropout_rate"],
            )(x, deterministic=False)

            x = nn.Dense(
                features=actual_out_dim,
                use_bias=self.config["use_bias"],
                kernel_init=nn.with_partitioning(
                    nn.initializers.he_normal(),
                    (None, 'model'),
                    ),
                bias_init=nn.with_partitioning(
                    nn.initializers.constant(0.0),
                    (None, 'model'),
                    ),
            )(x)

            outputs = nn.Dropout(
                rate=self.config["dropout_rate"],
            )(x, deterministic=False)
        
        else:
            x = nn.Dense(
                features=self.config["hidden_size"],
                use_bias=self.config["use_bias"],
                kernel_init=nn.initializers.he_normal(),
                #bias_init=nn.initializers.he_normal(),
            )(inputs)
            x = nn.relu(x)
            x = nn.Dropout(
                rate=self.config["dropout_rate"],
            )(x, deterministic=False)

            x = nn.Dense(
                features=actual_out_dim,
                use_bias=self.config["use_bias"],
                kernel_init=nn.initializers.he_normal(),
                #bias_init=nn.initializers.he_normal(),
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
        if config.train_parallel:
            x = nn.SelfAttention(
                    num_heads=config.num_heads, 
                    qkv_features=config.qkv_dim,
                    kernel_init=nn.with_partitioning(
                    nn.initializers.he_normal(),
                    (None, 'model'),
                    ),
                bias_init=nn.with_partitioning(
                    nn.initializers.constant(0.0),
                    (None, 'model'),
                    ),
                    )(
                x, mask
            )
        else:
            x = nn.SelfAttention(
                    num_heads=config.num_heads, 
                    qkv_features=config.qkv_dim,
                    kernel_init=nn.initializers.he_normal(),
                    #bias_init=nn.initializers.he_normal()
                    )(
                x, mask
            )

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)
        y = MLPBlock(config=config)(y)
        
        # check if residual connection needs resizing
        if x.shape[-1] != y.shape[-1]:
            if config.train_parallel:
                x = nn.Dense(
                        y.shape[-1],
                        kernel_init=nn.with_partitioning(
                            initializers.he_normal(),
                            (None, 'model')
                            ),
                        bias_init=nn.with_partitioning(
                            initializers.he_normal(),
                            (None, 'model')
                            ),
                        )(x)
            else:
                x = nn.Dense(y.shape[-1])(x)

        return x + y
