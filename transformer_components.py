"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import jax
import chex
import flax.linen as nn

###########################
# Decoder-only Transformer
###########################

class MLPBlock(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, inputs):
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


class DecoderBlock(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, inputs):
        attention_config = self.config["self_attention_block"]
        dot_attention_config = self.config["multihead_dot_product_attention_block"]
        mlp_config = self.config["mlp_block"]

        # Encoder-Decoder Attention Block
        x = nn.MultiHeadDotProductAttention(
            num_heads=dot_attention_config["num_heads"],
            qkv_features=dot_attention_config["qkv_features"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
            use_bias=dot_attention_config["use_bias"],
            broadcast_dropout=dot_attention_config["broadcast_dropout"],
            dropout_rate=dot_attention_config["dropout_rate"],
            deterministic=dot_attention_config["deterministic"],
            decode=True,
        )(inputs_q=inputs, inputs_kv=inputs)
        x = nn.Dropout(attention_config["dropout_rate"])(x, deterministic=False)
        # residual connection
        x = x + inputs

        # Feed Forward Block
        y = nn.LayerNorm()(x)
        y = MLPBlock(config=mlp_config)(y)
        # residual connection
        outputs = y + x

        return outputs

