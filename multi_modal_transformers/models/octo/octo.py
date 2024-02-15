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

    def setup(self):
        self.text_encoder = instantiate(OCTO_CONFIG.tokenizers.text.encoder)
        self.image_encoder = instantiate(OCTO_CONFIG.tokenizers.images.encoder, _recursive_=False)

    def __call__(self, text_tokens, images):
        text_embeddings = self.text_encoder(text_tokens)
        image_embeddings = self.image_encoder(images)
        return text_embeddings, image_embeddings


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
    text_tokens = inputs["input_ids"]
    images = jnp.ones((2, 3, 280, 280, 3))

    # instantiate model
    model = Octo(OCTO_CONFIG)
    variables = model.init({"params": keys[0], "patch_encoding": keys[1]}, text_tokens, images)
    
    # apply forward pass
    outputs = model.apply(
            {
                "params": variables["params"],
            }, 
            text_tokens, 
            images,
            rngs={"patch_encoding": keys[2]}
            )

    # check the ouputs
    print(outputs)            
