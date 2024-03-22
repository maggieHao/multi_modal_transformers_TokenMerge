"""
Octo model architecture.
"""
from copy import copy, deepcopy
from typing import Any, Callable
from dataclasses import dataclass
from functools import partial

# linear algebra/deep learning frameworks
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax import struct
from flax.linen import initializers
from flax.training import train_state
import einops as e

# multi-modal architectures
import multi_modal_transformers
from multi_modal_transformers.tokenizers.token_sequencer import (
        TokenSequence, 
        TokenEmbeddings,
        TaskDescriptionPrefix,
        Text,
        Image,
        Readout,
        )
from multi_modal_transformers.tokenizers.readout.readout import AddPositionEmbedding
from multi_modal_transformers.tokenizers.images.image_tokenizer import ImageTokenizer, SingleImageTokenizer
from multi_modal_transformers.tokenizers.text.text_tokenizer import BasicTextTokenizer
from multi_modal_transformers.attention_blocks.attention import Encoder1DBlock, StackedEncoder1DBlock

# huggingface transformers
from transformers import AutoTokenizer

# model config
import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig

# model metrics
import wandb
from clu import metrics


## OCTO Module ##

class Octo(nn.Module):
    config: DictConfig

    def setup(self):
        # token sequence manager
        self.token_sequence = TokenSequence(self.config.input_sequence)
        
        # generate attention mask
        self.attention_mask = self.token_sequence.generate_attention_mask(
                repeats=self.config.attention_blocks.stacked_encoder_1d_block.encoder_1d_block.self_attention.num_heads
                ) 
        
        # generate assemble embeddings function (Revise this section for bugs)
        self.slice_idx = self.token_sequence.slice_idx
        self.assemble_embeddings = partial(self.token_sequence.assemble_embeddings, slice_idx=self.slice_idx)

        # modality encoders
        self.text_encoder = instantiate(self.config.tokenizers.text.encoder) 
        self.image_encoder = instantiate(self.config.tokenizers.images.encoder, _recursive_=False)
        self.readout_encoder = instantiate(self.config.tokenizers.readouts.encoder, _recursive_=True) 

        # attention blocks
        self.attention_blocks = instantiate(self.config.attention_blocks.stacked_encoder_1d_block, _recursive_=False)

        # action heads 
        self.action_space_dim = self.config.action_heads.action_space_dim
        for action_head in self.config.action_heads.heads:
            exec("self.{action_head} = instantiate(self.config.action_heads.{action_head}, _recursive_=False)")

    # transformer backbone

    def generate_readouts(self, text_tokens, images):
        """
        Generate readout embeddings for action heads.
        """
        batch_size = images.shape[0]

        # create embeddings for each modality
        text_embeddings = self.text_encoder(text_tokens)
        
        image_embeddings = self.image_encoder(images)
        image_embeddings = e.rearrange(image_embeddings, "batch history patch embedding -> batch (history patch) embedding")
        

        # TODO: inspect this method + its parameters
        readout_dummy = jnp.zeros((
            image_embeddings.shape[0], # batch dimension
            self.config.num_observation_blocks * self.config.tokens_per_readout,
            self.config.token_embedding_dim
            ))
        readout_embeddings = self.readout_encoder(readout_dummy)
        
        # assemble embeddings into sequence with appropriate masking
        embeddings = TokenEmbeddings(
                text = text_embeddings,
                images = image_embeddings,
                readouts = readout_embeddings,
                )
        embeddings = self.assemble_embeddings(embeddings)


        # apply attention blocks
        mask = jnp.repeat(jnp.expand_dims(self.attention_mask, axis=0), batch_size, axis=0) 
        embeddings = self.attention_blocks(embeddings, mask=mask, train=True)

        # filter for readout embeddings
        readout_idx = self.token_sequence.get_modality_idx("readouts")
        filtered_embeddings = jnp.take(embeddings, readout_idx, axis=1)

        return filtered_embeddings

    # diffusion action head
        
    def predict_diffusion_denoise_term(self, text_tokens, images, time, noisy_actions):
        """
        Predict denoising term for diffusion process.
        """
        readout_embeddings = self.generate_readouts(text_tokens, images)
        denoise_terms = self.diffusion_action_head.predict_denoise_term(readout_embeddings, time, noisy_actions)
        
        return denoise_terms

    def compute_diffusion_denoise_loss(self, text_tokens, images, actions):
        """
        Compute loss for denoise prediction.
        """
        readout_embeddings = self.generate_readouts(text_tokens, images)
        loss = self.diffusion_action_head.denoise_loss(readout_embeddings, actions)
        return loss

    def predict_diffusion_action(self, text_tokens, images):
        """
        Predict next action using diffusion process.
        """
        readout_embeddings = self.generate_readouts(text_tokens, images)
        action_prediction = self.diffusion_action_head.predict_action(readout_embeddings)

        return action_prediction

    # continuous action head

    def predict_continuous_action(self, text_tokens, images):
        """
        Predict next action using continuous action head.
        """
        readout_embeddings = self.generate_readouts(text_tokens, images)
        action_prediction = self.continuous_action_head(readout_embeddings)

        return action_prediction

    def compute_continuous_l2_loss(self, text_tokens, images, actions):
        """
        Compute l2 loss for continuous action head.
        """
        predictions = self.predict_continuous_action(text_tokens, images)
        predictions = jnp.squeeze(predictions)
        
        return jnp.sum(jnp.square(predictions - actions), axis=-1)

    # categorical action head

    def predict_categorical_action(self, text_tokens, images):
        """
        Predict next action using categorical action head.
        """
        readout_embeddings = self.generate_readouts(text_tokens, images)
        action_prediction = self.categorical_action_head(readout_embeddings)
        
        return action_prediction

    def compute_ce_loss(self, text_tokens, images, actions):
        """
        Compute cross-entropy loss for categorical action head.
        """

## Model Training State ##

def diffusion_train_step(model, train_state, text_tokens, images, actions):
    """
    Performs one step of diffusion process training on a batch of data.
    """
    
    # generate new random keys
    train_rngs = {}
    train_rngs["dropout"] = jax.random.fold_in(train_state.rngs["dropout"], train_state.step)
    train_rngs["patch_encoding"] = jax.random.fold_in(train_state.rngs["patch_encoding"], train_state.step)
    train_rngs["diffusion"] = jax.random.fold_in(train_state.rngs["diffusion"], train_state.step)

    # compute loss and gradient of loss
    loss, grads = jax.value_and_grad(
            train_state.apply_fn,
            argnums=0)(
                    {"params": train_state.params},
                    text_tokens, 
                    images,
                    actions,
                    rngs=train_rngs,
                    method="compute_diffusion_denoise_loss"
                    )

    # perform gradient descent using computed gradients
    train_state = train_state.apply_gradients(grads=grads["params"])
   
    # update metrics
    wandb.log({
            "loss": loss,
        })
    metric_updates = train_state.metrics.single_from_model_output(
            loss=loss,
            )
    metrics = train_state.metrics.merge(metric_updates)
    train_state = train_state.replace(metrics=metrics)

    return train_state, grads

def continuous_train_step(model, train_state, text_tokens, images, actions):
    """
    Performs one step of continuous action head training on a batch of data.
    """
    
    # generate new random keys
    train_rngs = {}
    train_rngs["dropout"] = jax.random.fold_in(train_state.rngs["dropout"], train_state.step)
    train_rngs["patch_encoding"] = jax.random.fold_in(train_state.rngs["patch_encoding"], train_state.step)

    # compute loss and gradient of loss
    def mse_loss(params):
        loss = train_state.apply_fn(
                        {"params": params},
                        text_tokens, 
                        images,
                        actions,
                        rngs=train_rngs,
                        method="compute_continuous_l2_loss"
                        )

        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(mse_loss)
    loss, grads = grad_fn(train_state.params)

    # perform gradient descent using computed gradients
    train_state = train_state.apply_gradients(grads=grads)
   
    # update metrics
    wandb.log({
            "loss": loss,
        })
    metric_updates = train_state.metrics.single_from_model_output(
            loss=loss,
            )
    metrics = train_state.metrics.merge(metric_updates)
    train_state = train_state.replace(metrics=metrics)

    return train_state, grads

@struct.dataclass
class OCTOMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")

class OCTOTrainState(train_state.TrainState):
    metrics: OCTOMetrics
    text_tokenize_fn: Callable
    rngs: dict
    continuous_train_step: Callable = continuous_train_step
    diffusion_train_step: Callable = diffusion_train_step

def create_octo_train_state(
        text, 
        images,
        text_tokenizer,
        diffusion_inputs, 
        rngs,
        model, 
        optimizer,
        method="predict_diffusion_denoise_term", # default to diffusion model
        ):
    """Create initial training state."""
    
    if method=="predict_diffusion_denoise_term":
        variables = model.init(
            rngs, 
            text,
            images,
            diffusion_inputs["time"],
            diffusion_inputs["noisy_actions"],
            method=method 
        )
    elif method=="predict_continuous_action":
        variables = model.init(
                rngs, 
                text, 
                images, 
                method=method
                )
    else:
        raise Exception("dude you used an unsupported method for model initialization")

    params = variables["params"]

    return OCTOTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=OCTOMetrics.empty(),
        text_tokenize_fn=partial(text_tokenizer, 
                               return_tensors="jax", 
                               max_length=16, # hardcode while debugging
                               padding="max_length", 
                               truncation=True
                               ),
        rngs=rngs,
    )

if __name__=="__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../../model_configs", job_name="octo")
    OCTO_CONFIG = compose(
            config_name="octo_base",
            )
    
    keys = jax.random.split(jax.random.PRNGKey(0), 4)

    # test inputs #
    instructions = [
            "Pick up the red block and put it on the green block",
            "Pick up the green block and put it on the red block",
            ]
    tokenizer = instantiate(OCTO_CONFIG.tokenizers.text.tokenizer)
    text_tokens = tokenizer(
            instructions, 
            return_tensors="jax", 
            max_length=16, # hardcode while debugging
            padding="max_length", 
            truncation=True,
            )["input_ids"]
    images = jnp.ones((2, 2, 280, 280, 3))
    time = jnp.ones((2, 1))
    actions = jnp.ones((2, 8))
    noisy_actions = jnp.ones((2, 8))

    # instantiate model using diffusion noise prediction method
    model = Octo(OCTO_CONFIG)
    variables = model.init(
            {"params": keys[0], 
             "patch_encoding": keys[1], 
             "dropout": keys[2],
             "diffusion": keys[3],
             }, 
            text_tokens, 
            images,
            time, 
            noisy_actions,
            method="predict_diffusion_denoise_term",
            )
    
    # test forward pass for action prediction
    outputs = model.apply(
            {
                "params": variables["params"],
            }, 
            text_tokens, 
            images,
            method="predict_diffusion_action",
            rngs={
                "dropout": keys[2],
                "patch_encoding": keys[2],
                "diffusion": keys[2],
                },
            )

    # test computation of loss value
    loss = model.apply(
            {
                "params": variables["params"],
            }, 
            text_tokens, 
            images,
            actions,
            method="compute_diffusion_denoise_loss",
            rngs={
                "dropout": keys[2],
                "patch_encoding": keys[2],
                "diffusion": keys[2],
                },
            )

