"""
Octo model architecture.
"""
from typing import Any
from dataclasses import dataclass

# deep learning framework
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen import initializers
import einops as e

# huggingface transformers
from transformers import AutoTokenizer

import multi_modal_transformers

# custom tokenizers
from multi_modal_transformers.tokenizers.token_sequencer import (
        TokenSequence, 
        TokenEmbeddings,
        TaskDescriptionPrefix,
        Text,
        Image,
        Readout,
        )

from multi_modal_transformers.tokenizers.readout.readout import AddPositionEmbedding
from multi_modal_transformers.tokenizers.numeric_values.value_tokenizer import ActionTokenizer
from multi_modal_transformers.tokenizers.images.image_tokenizer import ImageTokenizer, SingleImageTokenizer
from multi_modal_transformers.tokenizers.text.text_tokenizer import BasicTextTokenizer

# attention blocks
from multi_modal_transformers.attention_blocks.attention import Encoder1DBlock

# model config
import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig


# model definition
class Octo(nn.Module):
    config: DictConfig
        
    @nn.compact
    def __call__(self, text_tokens, images):

        ## Generating Embedding Sequence ##
        # embed text input
        text_encoder = instantiate(self.config.tokenizers.text.encoder)
        text_embeddings = text_encoder(text_tokens)

        # embed images
        image_encoder = instantiate(self.config.tokenizers.images.encoder, _recursive_=False)
        image_embeddings = image_encoder(images)
        image_embeddings = e.rearrange(image_embeddings, "batch history patch embedding -> batch (history patch) embedding")

        # embed readout
        readout_dummy = jnp.zeros((
            text_embeddings.shape[0], # batch dimension
            self.config.num_observation_blocks * self.config.tokens_per_readout,
            self.config.token_embedding_dim
            ))
        readout_encoder = instantiate(self.config.tokenizers.readouts.encoder, _recursive_=True)
        readout_embeddings = readout_encoder(readout_dummy)
        
        # assemble embedding sequence for attention layers
        embeddings = TokenEmbeddings(
                images = image_embeddings,
                text = text_embeddings,
                readouts = readout_embeddings,
                )
        sequence = TokenSequence(
                self.config.input_sequence
                )
        attention_mask = sequence.generate_attention_mask()
        #TODO: compute padding mask 
        embeddings = sequence.assemble_embeddings(embeddings)

        ## Apply Attention Mechanism ##
        for _ in range(self.config.attention_blocks.num_blocks):
            embeddings = instantiate(self.config.attention_blocks.encoder_1d_block, _recursive_=False)(
                    embeddings,
                    mask=attention_mask,
                    train=True,
                    )

        ## Generate Action Head Predictions ##
        # filter output embeddings for readout embeddings
        readout_idx = sequence.get_modality_idx("readouts")     

        # pass readout embeddings to action prediction head


        return embeddings


if __name__=="__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../../model_configs", job_name="octo")
    OCTO_CONFIG = compose(
            config_name="octo_base",
            )
    
    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    # test inputs #
    instructions = [
            "Pick up the red block and put it on the green block",
            "Pick up the green block and put it on the red block",
            ]
    tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=16)
    inputs = tokenizer(
            instructions, 
            return_tensors="jax", 
            max_length=16, # hardcode while debugging
            padding="max_length", 
            truncation=True,
            )
    text_tokens = inputs["input_ids"]
    images = jnp.ones((2, 2, 280, 280, 3))

    # instantiate model
    model = Octo(OCTO_CONFIG)
    variables = model.init({"params": keys[0], "patch_encoding": keys[1], "dropout": keys[1]}, text_tokens, images)
    
    # apply forward pass
    outputs = model.apply(
            {
                "params": variables["params"],
            }, 
            text_tokens, 
            images,
            rngs={
                "dropout": keys[2],
                "patch_encoding": keys[2]
                }
            )

    # check the ouputs
    print(outputs)            
