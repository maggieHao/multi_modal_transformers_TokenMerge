"""Implementation of tokenizer for discrete/continuous values."""

import dataclasses

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from jax import random

############################
# Discrete Embedding
############################

class ActionTokenizer(nn.Module):
    """
    Tokenizes discrete actions and creates embedding.
    """

    config: dict

    def setup(self):
        if self.config.train_parallel:
            self.embedding = nn.Embed(
                num_embeddings=self.config["num_actions"],
                features=self.config["embedding_dim"],
                embedding_init=nn.with_partitioning(
                    initializers.variance_scaling(
                        1.0, 'fan_in', 'normal', out_axis=0
                        ), 
                    (None, 'model'))
            )
        else:
            self.embedding = nn.Embed(
                num_embeddings=self.config["num_actions"],
                features=self.config["embedding_dim"],
            )

    def __call__(self, action):
        return self.embedding(action)


############################
# Continuous Embedding
############################

def mu_law_encoder(x, mu=255):
    return jnp.sign(x) * jnp.log(1 + mu * jnp.abs(x)) / jnp.log(1 + mu)

