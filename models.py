"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import jax
import chex
import flax.linen as nn


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


class EncoderBlock(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, inputs):
        attention_config = self.config["self_attention_block"]
        mlp_config = self.config["mlp_block"]

        # Attention Block
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(
            num_heads=attention_config["num_heads"],
            qkv_features=attention_config["qkv_features"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
            use_bias=attention_config["use_bias"],
            broadcast_dropout=attention_config["broadcast_dropout"],
            dropout_rate=attention_config["dropout_rate"],
            deterministic=attention_config["deterministic"],
        )(x)
        x = nn.Dropout(attention_config["dropout_rate"])(x, deterministic=False)
        # residual connection
        x = x + inputs

        # Feed Forward Block
        y = nn.LayerNorm()(x)
        y = MLPBlock(config=mlp_config)(y)
        # residual connection
        outputs = y + inputs

        return outputs


class DecoderBlock(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, encoder_outputs, decoder_inputs):
        attention_config = self.config["self_attention_block"]
        dot_attention_config = self.config["multihead_dot_product_attention_block"]
        mlp_config = self.config["mlp_block"]

        # Decoder Block
        x = nn.LayerNorm()(decoder_inputs)
        x = nn.SelfAttention(
            num_heads=attention_config["num_heads"],
            qkv_features=attention_config["qkv_features"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
            use_bias=attention_config["use_bias"],
            broadcast_dropout=attention_config["broadcast_dropout"],
            dropout_rate=attention_config["dropout_rate"],
            deterministic=attention_config["deterministic"],
        )(x)
        x = nn.Dropout(attention_config["dropout_rate"])(x, deterministic=False)
        # residual connection
        x = x + decoder_inputs

        # Encoder-Decoder Attention Block
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=dot_attention_config["num_heads"],
            qkv_features=dot_attention_config["qkv_features"],
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
            use_bias=dot_attention_config["use_bias"],
            broadcast_dropout=dot_attention_config["broadcast_dropout"],
            dropout_rate=dot_attention_config["dropout_rate"],
            deterministic=dot_attention_config["deterministic"],
        )(y, encoder_outputs)
        y = nn.Dropout(attention_config["dropout_rate"])(y, deterministic=False)
        # residual connection
        y = y + x

        # Feed Forward Block
        z = nn.LayerNorm()(y)
        z = MLPBlock(config=mlp_config)(z)
        # residual connection
        outputs = z + y

        return outputs


class Encoder(nn.Module):
   """Encoder that accepts input embeddings"""

    config: dict

    def __call__(self, inputs):
        x = inputs
        for lyr in range(self.config["num_layers"]):
            x = EncoderBlock(config=config)(x)
        outputs = nn.LayerNorm()(x)

        return outputs


class Decoder(nn.Module):
    config: dict

    def __call__(self, encoder_outputs, decoder_inputs):
        x = decoder_inputs
        for lyr in range(self.config["num_layers"]):
            x = DecoderBlock(config=config)(encoder_outputs, x)
        outputs = nn.LayerNorm()(x)
        logits = nn.Dense()(outputs)

        return logits 
