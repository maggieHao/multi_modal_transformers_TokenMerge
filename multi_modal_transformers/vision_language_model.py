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
    def __call__(self, images, text):
        # image tokenizer
        image_tokenizer = SingleImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, num_tokens_per_image, _ = image_embeddings.shape
        
        # text tokenizer
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # concatenate image and text embeddings
        token_embeddings, ps = e.pack((image_embeddings, text_embeddings), "batch * features")

        # create an attention mask for text embeddings
        image_mask = jnp.ones((batch_size, num_tokens_per_image))
        text_mask = jnp.where(text == 0, 0, 1)

        # concatenate image and text masks
        attention_mask_input = e.pack((image_mask, text_mask), "batch *")
        attention_mask = nn.make_attention_mask(
            attention_mask_input>0,
            attention_mask_input>0,
                )
        # 1D attention mask -> multi-head attention
        multi_head_attention_mask = e.repeat(
            attention_mask,
            "batch head_dim q k -> batch (repeats head_dim) q k",
            repeats=num_heads,
        )

        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        for i in range(num_blocks):
            x = Encoder1DBlock(self.config.transformer)(
                token_embeddings,
                mask=multi_head_attention_mask,
                train=train,
            )
        
        # pass through final linear layer
        x = instantiate(self.config.transformer.output_dense)(x)
        
        # get next token logits
        # TODO: implement correct indexing to get next token
        next_token_logits = None

        return next_token_logits
