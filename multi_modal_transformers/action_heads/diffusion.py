"""Diffusion action head."""

from typing import Callable

import jax
import jax.numpy as jnp
import flax 
import flax.linen as nn
from flax.linen import initializers

from omegaconf import DictConfig
from hydra.utils import call, instantiate


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class FourierFeatures(nn.Module):
    """
    Learnt Fourier Features.

    Source: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
    """
    output_size: int
    kernel_init: Callable
    mlp_block: DictConfig

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "fourier_kernel",
            call(self.kernel_init),
            (self.output_size // 2, x.shape[-1]),
        )
        x = 2 * jnp.pi * x @ w.T
        x = jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
        x = instantiate(self.mlp_block, _recursive_=False)(x) # consider reviewing compared to original implementation

        return x
        
class OctoDenoise(nn.Module):
    num_blocks: int
    time_encoder: DictConfig
    mlp_block: DictConfig

    @nn.compact
    def __call__(self, noisy_action, timestep, readoutembedding):
        time_embedding = instantiate(self.time_encoder_model, _recursive_=False)(timestep)
        x = jnp.concatenate([noisy_action, time_embedding, readout_embedding])
        for _ in range(self.num_blocks):
            instantiate(self.mlp_block, _recursive_=False)(x)

        return x

class DiffusionActionHead(nn.Module):
    """
    Unlike a regular diffuser this one is for predicting robot policy actions :).
    """
    attention_pooling: DictConfig
    denoising_model: DictConfig


    def setup(self):
        self.pooling = instantiate(self.attention_pooling, _recursive_=False)
        self.denoiser = instantiate(self.denoising_model, _recursive_=False)
        
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.array(
            [jnp.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)]
        )

    def __call__(
            self,
            readouts,
            time, 
            noisy_actions,
            ):
        embeddings = self.pooling(readouts)
        jax.debug.print("Pooled Readout Dim: {}", embeddings)
        
        embeddings = self.denoiser(noisy_actions, time, embeddings)
        jax.debug.print("Output Dim: {}", embeddings)

        return embeddings


    def predict_action(self):
        raise NotImplementedError


    def loss(self):
        raise NotImplementedError
