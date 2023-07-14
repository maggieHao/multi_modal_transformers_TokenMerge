"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

# deep learning framework
import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
import einops as e

# custom tokenizers
from tokenizers.value_tokenizer import ActionTokenizer
from tokenizers.image_tokenizer import ImageTokenizer
from tokenizers.text_tokenizer import (
    BasicTokenizer,
    BasicTextTokenizer,
)

# transformer modules
from transformer_components import Encoder1DBlock

class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""
    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, actions):
        
        ### Tokenization + Input Embeddings ###

        ## text embeddings
        text_tokenizer = BasicTokenizer(
            vocab_dir=self.config.model.executor.text_tokenizer.vocab_dir
            )
        text_tokenizer = BasicTextTokenizer(
            config = self.config.model.executor.text_tokenizer, tokenizer=text_tokenizer
        )
        text_embeddings = text_tokenizer(text)

        ## image embeddings
        image_tokenizer = ImageTokenizer(config = self.config.model.executor.image_tokenizer)
        image_embeddings = image_tokenizer(images)

        ## action embeddings
        action_tokenizer = ActionTokenizer(config = self.config.model.executor.action_tokenizer)
        action_embeddings = action_tokenizer(actions)

        ## positional embeddings
        positional_embedding = nn.Embed(
            num_embeddings=21,
            features=self.config.model.executor.token_embedding_dim,
        )

        ## observation embeddings
        #observation_embeddings = self.positional_embedding(jnp.arange(21))

        # concatenate text, image, action and observation embeddings
        
        # interleave image and action embeddings such that image, action, image, action, ...
        def interweave_embeddings(image_embeddings, action_embeddings):
            batch_size = image_embeddings.shape[0]
            num_images = image_embeddings.shape[1]
            tokens_per_image = image_embeddings.shape[2]
            feature_size = image_embeddings.shape[-1]
            total_tokens = (image_embeddings.shape[1]*image_embeddings.shape[2]) + action_embeddings.shape[1]
            
            # interleave image and action embeddings
            embeddings = jax.lax.concatenate((image_embeddings, jnp.expand_dims(action_embeddings, axis=2)), dimension=2)
            embeddings = jnp.reshape(embeddings, (batch_size, total_tokens, feature_size))

            return embeddings

        # interleave image and action embeddings
        interleaved_embeddings = interweave_embeddings(image_embeddings, action_embeddings)

        # concatenate text and interleaved embeddings
        embeddings = jnp.concatenate((text_embeddings, interleaved_embeddings), axis=1)

        
        ### Transformer Self Attention ###

        # generate attention mask for padding tokens
        attention_mask_input = e.rearrange(
                e.repeat(
                    jnp.where(actions == 0, 0, 1), 
                    'batch seq -> batch seq repeats', 
                    repeats=5) # 5 patches per image + action (TODO: replace with config param)
               , 'batch seq repeats -> batch (seq repeats)')
        # add in 1's for text tokens
        attention_mask_input = jnp.concatenate((jnp.ones((attention_mask_input.shape[0], text_embeddings.shape[1])), attention_mask_input), axis=1)
        attention_mask = nn.make_attention_mask(attention_mask_input > 0, attention_mask_input > 0)
        attention_mask = e.repeat(attention_mask, 'batch heads q k -> batch (heads repeats) q k', repeats=4)

        # pass through self attention layer
        x = Encoder1DBlock(self.config.model.executor.self_attention_1)(embeddings, mask=attention_mask)

        print(x.shape)
        print(attention_mask.shape)

        return x
