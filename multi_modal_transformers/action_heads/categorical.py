"""Categorical output variations."""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops as e

from omegaconf import DictConfig
from hydra.utils import call, instantiate

class CategoricalActionHead(nn.Module): 
    dense: DictConfig

    def __call__(self, readouts):
        embeddings = e.rearrange(
                        readouts, 
                        "batch (timestep action) embeddings -> batch timestep action embeddings", 
                        action=dense.features
                        )
        embeddings = jnp.squeeze(jnp.mean(embeddings, axis=-2))
        logits = instantiate(self.dense, _recursive_=True)(embeddings)
        
        return logits
        

