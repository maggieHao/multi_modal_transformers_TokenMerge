"""Diffusion action head."""

import jax
import jax.numpy as jnp
import flax 
import flax.linen as nn
from flax.linen import initializers

from omegaconf import DictConfig
from hydra.utils import call, instantiate

class DiffusionActionHead(nn.Module):
    """
    Unlike a regular diffuser this one is for predicting robot policy actions :).
    """
    attention_pooling: DictConfig

    def setup(self):
        self.pooling = instantiate(self.attention_pooling, _recursive_=False)


    def __call__(
            self,
            readouts,
            ):
        return self.pooling(readouts)


    def predict_action(self):
        raise NotImplementedError


    def loss(self):
        raise NotImplementedError
