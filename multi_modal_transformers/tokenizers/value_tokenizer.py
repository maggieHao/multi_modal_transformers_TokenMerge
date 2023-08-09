"""Implementation of tokenizer for discrete/continuous values."""

import dataclasses

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from jax import random

from hydra.utils import instantiate

############################
# Discrete Embedding
############################

class ActionTokenizer(nn.Module):
    """
    Tokenizes discrete actions and creates embedding.
    """

    config: dict

    def setup(self):
        self.embedding = instantiate(self.config["action_embedding"])

    def __call__(self, action):
        return self.embedding(action)


############################
# Continuous Embedding
############################

def mu_law_encoder(x, mu=255):
    return jnp.sign(x) * jnp.log(1 + mu * jnp.abs(x)) / jnp.log(1 + mu)

