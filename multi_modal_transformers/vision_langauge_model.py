"""
Vision Langauge Model (VLM) implementation.
"""

# deep learning framework
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
import einops as e

# custom tokenizers
from multi_modal_transformers.tokenizers.image_tokenizer import ImageTokenizer

# transformer modules
from multi_modal_transformers.transformer_components import Encoder1DBlock

from hydra.utils import instantiate

class ConceptPlanner(nn.Module):
    """A Vision Language Model that suggests concepts to execute."""

    config: dict

    @nn.compact
    def __call__(self, images):
        # resnet backbone
        image_tokenizer = SingleImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, num_tokens_per_image, _ = image_embeddings.shape

        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        for i in range(num_blocks):
            x = Encoder1DBlock(self.config.transformer)(
                x,
                mask=attention_mask,
                train=train,
            )
        
        # pass through final linear layer
        x = instantiate(self.config.transformer.output_dense)(x)
        
        # get next token logits
        # TODO: implement correct indexing to get next token
        next_token_logits = None

        return next_token_logits
