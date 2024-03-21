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
        embeddings = jnp.mean(readouts, axis=-2)
        #embeddings = instantiate(self.attention_pooling, _recursive_=False)(readouts)
        #embeddings = e.rearrange(embeddings, "batch seq embedding -> (batch seq) embedding")
        #jax.debug.print("action head post pooling shape: {}", embeddings.shape)
        
        mean = instantiate(self.dense, _recursive_=True)(embeddings)
        mean = e.rearrange(mean, "batch (seq mean) -> batch seq mean", seq = 1)
        #jax.debug.print("action head post mean projection shape: {}", mean.shape)

        return jnp.tanh(mean / self.max_action) * self.max_action

