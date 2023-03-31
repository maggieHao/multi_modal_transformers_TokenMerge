"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import jax
import chex
import flax.linen as nn


###########################
# Embeddings
###########################

class ResNetV2Block(nn.Module):
    """
    Note: fixing parameter defaults to match Gato.
    """
    features: int
    strides: Tuple[int, int] = (1, 1)
    kernel_size: Tuple[int, int] = (3, 3)
    padding: str = "SAME"
    weights: ModuleDef = nn.Conv
    normalization: ModuleDef = nn.GroupNorm
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # Important: I am uncertain about this function.
        #residual = x

        # its not possible to perform group norm with 32 groups on image
        # containing only 3 channels! Start with conv before first group norm.
        y = self.weights(features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding=self.padding)(x)
        
        residual = y

        y = self.normalization()(y)
        y = self.activation(y)
        y = self.weights(features=self.features, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        y = self.normalization()(y)
        y = self.activation(y)
        y = self.weights(features=self.features, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        return y+residual


class TokenLearner(nn.Module):
    raise NotImplementedError

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


class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""
    config: dict

    def __call__(self, input_seq):
        # split the input sequence into its components
        instruction_seq, image_seq, action_seq, concept_seq = input_seq

        # generate token embeddings and add positional encoding
        # TODO: add positional encoding
        
        ## instruction embeddings
        instruction_embeddings = nn.Embed()(instruction_seq)

        ## trajectory embeddings
        image_embeddings = ResNetV2Block()(image_seq)
        image_embeddings = TokenLearner()(image_embeddings)
        action_embeddings = nn.Embed()(action_seq)

        trajectory_embeddings = 

        ## concept embeddings
        concept_embeddings = nn.Embed()(concept_seq)
        

        embeddings = jnp.concatenate([instruction_embeddings, trajectory_embeddings, concept_embeddings], axis=1)

        # pass embeddings through transformer blocks
        for lyr in range(self.config["num_blocks"]):
            x = DecoderBlock(config=config)(x)

        # pass through final layer norm
        outputs = nn.LayerNorm()(x)

        # this should be a distribution over all tokens
        logits = nn.Dense()(outputs)

        return logits 
