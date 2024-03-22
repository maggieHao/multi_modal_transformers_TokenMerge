"""Diffusion action head."""

from typing import Callable

import jax
import jax.numpy as jnp
import flax 
import flax.linen as nn
from flax.linen import initializers
import optax as opt
import einops as e

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
    output_dim: int
    kernel_init: Callable
    mlp_block: DictConfig

    @nn.compact
    def __call__(self, x):
        w = self.param(
            "fourier_kernel",
            call(self.kernel_init),
            (self.output_dim // 2, x.shape[-1]),
        )
        x = 2 * jnp.pi * x @ w.T
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)
        x = instantiate(self.mlp_block, _recursive_=False)(x) # consider reviewing compared to original implementation

        return x
        
class OctoDenoise(nn.Module):
    num_blocks: int
    time_encoder: DictConfig
    mlp_block: DictConfig

    @nn.compact
    def __call__(self, noisy_action, timestep, readout_embedding):
        time_embedding = instantiate(self.time_encoder, _recursive_=False)(timestep)
        x = jnp.concatenate([noisy_action, time_embedding, readout_embedding], axis=-1)
        for _ in range(self.num_blocks):
            x = instantiate(self.mlp_block, _recursive_=False)(x)

        return x

class DiffusionActionHead(nn.Module):
    """
    Unlike a regular diffuser this one is for predicting robot policy actions :).
    """
    diffusion_steps: int
    attention_pooling: DictConfig
    denoising_model: DictConfig
    rng_collection: str = "diffusion"

    def setup(self):
        # model architectures
        self.pooling = instantiate(self.attention_pooling, _recursive_=False)
        self.denoiser = instantiate(self.denoising_model, _recursive_=False)
        
        # diffusion process params
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.array(
            [jnp.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)]
        )
    
    def predict_denoise_term(
            self,
            readouts,
            time, 
            noisy_actions,
            train = True,
            ):
        """
        Predicts denoising term in diffusion process.
        """
        # pool the readouts into one embedding for conditioning
        #embeddings = self.pooling(readouts)
        #embeddings = e.rearrange(embeddings, "batch readout embed -> batch (readout embed)")
        
        embeddings = jnp.mean(readouts, axis=-2)
        
        # predict denoise term 
        denoise_term = self.denoiser(noisy_actions, time, embeddings)

        return denoise_term


    def denoise_loss(
            self,
            readouts, 
            actions, 
            train=True
            ):
        """
        Training loss for denoise term prediction.
        """
        batch_size = actions.shape[0]
        rng = self.make_rng("diffusion")
        time_key, noise_key = jax.random.split(rng)

        # sample a random time value
        time = jax.random.randint(time_key, (batch_size, 1), 0, self.diffusion_steps)

        # generate noisy action
        noise = jax.random.normal(noise_key, actions.shape)
        alpha_hat = self.alpha_hats[time]
        alpha_1 = jnp.sqrt(alpha_hat)
        alpha_2 = jnp.sqrt(1 - alpha_hat)
        noisy_action = alpha_1 * actions + alpha_2 * noise

        # predict denoise term
        predictions = self.predict_denoise_term(
                readouts, 
                time, 
                noisy_action, 
                )

        # compute mse loss 
        loss = opt.l2_loss(predictions, noise)
        loss = jnp.mean(jnp.sum(loss, axis=-1))
        return loss
        
    
    def predict_action(
            self,
            readouts,
            train=True,
            ):
        """
        Predicts actions using denoising process of diffusion policy.
        """
        # as we are using jax.lax.scan we need to unbind module vars
        module, variables = self.unbind()
      
        def denoise_step(carry, time):
            """
            Performs one step of denoising on a batch of data.
            """
            current_sample, keys = carry
            batch_size = current_sample.shape[0]
            

            # broadcast time term
            time_repeat = e.repeat(jnp.array([time]), "time -> repeats time", repeats=batch_size)

            # predict denoise term
            denoise_term = module.apply(
                    variables, 
                    readouts, 
                    time_repeat, 
                    current_sample, 
                    method="predict_denoise_term"
                    )

            # generate random gaussian noise (TODO: check keys here)
            random_noise = jax.vmap(jax.random.normal, (0, None), (0))(keys, (current_sample.shape[-1],))

            # compute denoised sample
            # see algorithm 2 from https://arxiv.org/abs/2006.11239
            coefficient_1 = 1 / jnp.sqrt(self.alphas[time])
            coefficient_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            coefficient_3 = jnp.sqrt(self.betas[time])
            denoised_sample =  (coefficient_1 * (current_sample - (coefficient_2 * denoise_term))) + coefficient_3 * random_noise

            # clip action to valid range
            denoised_sample = jnp.clip(denoised_sample, -5, 5) # TODO: read clipping values from config

            return (denoised_sample, keys), ()
        
            
            return actions
    
        batch_size, num_tokens = readouts.shape[:2]
        
        # generate noisy sample from gaussian
        rng = self.make_rng(self.rng_collection)
        keys = jax.random.split(rng, batch_size) # we want a rng for each sample in the batch
        noisy_samples = jax.vmap(jax.random.normal, (0, None), (0))(keys, (8,)) # generate noisy samples
       
        # perform diffusion process to generate action prediction
        (actions, _), _ = jax.lax.scan(
                denoise_step, # denoise update
                (noisy_samples, keys), # initial sample and rng for sampling
                jnp.arange(self.diffusion_steps-1, -1, -1), # diffusion timesteps
                )

        return actions

