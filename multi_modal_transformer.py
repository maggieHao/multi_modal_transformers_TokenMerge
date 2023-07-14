"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import jax
import jax.numpy as jnp
import chex
import flax.linen as nn

# import custom tokenizers
from tokenizers.value_tokenizer import ActionTokenizer
from tokenizers.image_tokenizer import ImageTokenizer
from tokenizers.text_tokenizer import (
    BasicTokenizer,
    BasicTextTokenizer,
)

class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""
    config: dict

    def setup(self):
        # text tokenizer
        text_tokenizer = BasicTokenizer(
            vocab_dir=self.config.model.executor.text_tokenizer.vocab_dir
            )
        self.text_tokenizer = BasicTextTokenizer(
            config = self.config.model.executor.text_tokenizer, tokenizer=text_tokenizer
        )

        # image tokenizer
        self.image_tokenizer = ImageTokenizer(config = self.config.model.executor.image_tokenizer)
        
        # action tokenizer
        self.action_tokenizer = ActionTokenizer(config = self.config.model.executor.action_tokenizer)

        # learnt positional embeddings for observations
        self.positional_embedding = nn.Embed(
            num_embeddings=21,
            features=self.config.model.executor.token_embedding_dim,
        )
    
    # TODO: move parameters to single batch parameter
    def __call__(self, text, images, actions, key):
        
        ### Tokenization + Input Embeddings ###

        ## text embeddings
        text_embeddings = self.text_tokenizer(text)

        ## image embeddings
        image_embeddings = self.image_tokenizer(images, key)

        ## action embeddings
        action_embeddings = self.action_tokenizer(actions)

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
        
        #mask = nn.make_attention_mask()

        return embeddings 
