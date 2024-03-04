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

# custom tokenizers
from multi_modal_transformers.tokenizers.token_sequencer import TokenSequence
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
        # embed text input
        text_encoder = instantiate(self.config.tokenizers.text.encoder)
        text_embeddings = text_encoder(text_tokens)
        # text_embeddings = TextArray()

        # embed images
        image_encoder = instantiate(self.config.tokenizers.images.encoder, _recursive_=False)
        image_embeddings = image_encoder(images)
        # image_embeddings = ImageArray()
        
        # create param for readout
        readout_dummy = jnp.zeros((
            self.config.num_observation_blocks * self.config.tokens_per_readout,
            self.config.token_embedding_dim
            )) # used to infer shape of position encodings
        readout_encoder = instantiate(self.config.tokenizers.readouts.encoder, _recursive_=False)
        readout_embeddings = readout_encoder(readout_dummy)
        # readout_embeddings = ReadoutArray()

        # assemble sequence
        sequence = TokenSequence(self.config.input_sequence) # TODO: move to instantiate
        embeddings = sequence.assemble_embeddings([text_embeddings, image_embeddings, readout_embeddings])
        attention_mask = sequence.generate_attention_mask()
    

        # apply transformer attention
        for _ in range(self.config.attention_blocks.num_blocks):
            embeddings = instantiate(self.config.attention_blocks.encoder_1d_block, _recursive_=False)(
                    embeddings,
                    mask=attention_mask,
                    train=True,
                    )

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
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    inputs = tokenizer(instructions, return_tensors="jax", padding=True, truncation=True)
    print(inputs)
    text_tokens = inputs["input_ids"]
    images = jnp.ones((2, 3, 280, 280, 3))

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
