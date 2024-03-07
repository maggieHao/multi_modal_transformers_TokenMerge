"""
Octo model architecture.
"""
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
from multi_modal_transformers.attention_blocks.attention import Encoder1DBlock

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
from clu import metrics


## OCTO Module ##

class Octo(nn.Module):
    config: DictConfig

    def setup(self):
        # token sequence manager
        self.token_sequence = TokenSequence(self.config.input_sequence)
        
        # modality encoders
        self.text_encoder = instantiate(self.config.tokenizers.text.encoder) 
        self.image_encoder = instantiate(self.config.tokenizers.images.encoder, _recursive_=False)
        self.readout_encoder = instantiate(self.config.tokenizers.readouts.encoder, _recursive_=True) 
        
        # attention blocks
        self.attention_blocks = [instantiate(self.config.attention_blocks.encoder_1d_block, _recursive_=False) for _ in range(self.config.attention_blocks.num_blocks)] 

        # attention heads
        self.diffusion_action_head = instantiate(self.config.action_heads.diffusion_action_head, _recursive_=False)

    def generate_readouts(self, text_tokens, images):
        """
        Generate readout embeddings for action heads.
        """

        # create embeddings for each modality
        text_embeddings = self.text_encoder(text_tokens)
        
        image_embeddings = self.image_encoder(images)
        image_embeddings = e.rearrange(image_embeddings, "batch history patch embedding -> batch (history patch) embedding")
        
        readout_dummy = jnp.zeros((
            text_embeddings.shape[0], # batch dimension
            self.config.num_observation_blocks * self.config.tokens_per_readout,
            self.config.token_embedding_dim
            ))
        readout_embeddings = self.readout_encoder(readout_dummy)
        
        # assemble embeddings into sequence with appropriate masking
        # TODO: handle padding of missing observations
        embeddings = TokenEmbeddings(
                images = image_embeddings,
                text = text_embeddings,
                readouts = readout_embeddings,
                )
        embeddings = self.token_sequence.assemble_embeddings(embeddings)
        attention_mask = self.token_sequence.generate_attention_mask()
        
        # apply attention blocks
        for block in self.attention_blocks:
            embeddings = block(
                    embeddings,
                    mask=attention_mask,
                    train=True,
                    )

        # filter for readout embeddings
        readout_idx = self.token_sequence.get_modality_idx("readouts")     
        readout_embeddings = jnp.take(embeddings, readout_idx, axis=1)

        return readout_embeddings

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


## Model Training State ##

@struct.dataclass
class OCTOMetrics(metrics.Collection):
    denoise_loss: metrics.Average.from_output("loss")

class OCTOTrainState(train_state.TrainState):
    metrics: OCTOMetrics
    text_tokenize_fn: Callable

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
    variables = model.init(
        rngs, 
        text,
        images,
        diffusion_inputs["time"],
        diffusion_inputs["noisy_actions"],
        method=method 
    )

    params = variables["params"]

    return OCTOTrainState.create(
        apply_fn=getattr(model, method),
        params=params,
        tx=optimizer,
        metrics=OCTOMetrics.empty(),
        text_tokenize_fn=partial(text_tokenizer, 
                               return_tensors="jax", 
                               max_length=16, # hardcode while debugging
                               padding="max_length", 
                               truncation=True
                               )
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

