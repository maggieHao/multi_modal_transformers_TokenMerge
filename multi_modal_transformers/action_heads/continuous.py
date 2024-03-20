"""Variations on continuous action heads."""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops as e

from omegaconf import DictConfig
from hydra.utils import call, instantiate

class ContinuousActionHead(nn.Module):
    max_action: int
    attention_pooling: DictConfig
    dense: DictConfig

    @nn.compact
    def __call__(self, readouts):
        embeddings = instantiate(self.attention_pooling, _recursive_=False)(readouts)
        mean = instantiate(self.dense, _recursive_=True)(embeddings)
        #mean = jnp.squeeze(mean)
        return jnp.tanh(mean / self.max_action) * self.max_action

    def l2_loss(self, readouts, actions):
        mean = self(readouts)
        return jnp.mean(jnp.square(mean - actions))


