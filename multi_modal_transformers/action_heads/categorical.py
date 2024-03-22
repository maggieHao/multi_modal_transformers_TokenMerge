"""Categorical output variations."""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops as e

from omegaconf import DictConfig
from hydra.utils import call, instantiate

def assign_bins(input_data, bounds, num_bins, bin_strategy="uniform"):
    """
    Assigns continuous values to discrete bins
    """
    if bin_strategy=="uniform":
        bins = jnp.linspace(bounds[0], bounds[1], num_bins + 1)
        binned_data = jnp.digitize(input_data, bins)
    else:
        raise NotImplementedError

    return binned_data

class CategoricalActionHead(nn.Module): 
    num_bins: int
    max_action: int
    action_space_dim: int
    dense: DictConfig

    @nn.compact
    def __call__(self, readouts):
        embeddings = e.rearrange(
                        readouts, 
                        "batch (action timestep) embeddings -> batch action timestep embeddings", 
                        action=self.action_space_dim,
                        )
        embeddings = jnp.squeeze(jnp.mean(embeddings, axis=-2))
        logits = instantiate(self.dense, _recursive_=True)(embeddings)
        
        return logits
        
