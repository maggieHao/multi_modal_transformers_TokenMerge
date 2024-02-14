"""
Octo model architecture.
"""

# deep learning framework
import jax
import jax.numpy as jnp
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
from hydra.utils import instantiate
from omegaconf import DictConfig

class Octo(nn.Module):
    config: DictConfig

    def setup(self):
        self.text_encoder = instantiate(OCTO_CONFIG.tokenizers.text.encoder)
        self.image_encoder = instantiate(OCTO_CONFIG.tokenizers.images.encoder)

    def __call__(self, text_tokens):
        return self.text_encoder(text_tokens)

if __name__=="__main__":
     # clear hydra global state to avoid conflicts with other hydra instances
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../../model_configs", job_name="octo")
    OCTO_CONFIG = compose(
            config_name="octo_base",
            )
    
    # test text
    instructions = [
            "Pick up the red block and put it on the green block",
            "Pick up the green block and put it on the red block",
            ]
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    inputs = tokenizer(instructions, return_tensors="jax", padding=True, truncation=True)
    text_tokens = inputs["input_ids"]

    # test image input 
    images = jnp.ones((2, 3, 224, 224, 3))

    # tokenize instructions
    model = Octo(OCTO_CONFIG)
    params = model.init(jax.random.PRNGKey(0), text_tokens)
    output = model.apply(params, text_tokens)
    print(output)            
