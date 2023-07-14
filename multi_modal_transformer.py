"""
Transformer architecture implementation.
Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

import jax
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
            config=cfg.model.executor.text_tokenizer, tokenizer=text_tokenizer
        )

        # image tokenizer
        self.image_tokenizer = ImageTokenizer(config = self.config.model.executor.image_tokenizer)
        
        # action tokenizer
        self.action_tokenizer = ActionTokenizer(config = self.config.model.executor.action_tokenizer)

        # learnt positional embeddings for observations
        self.positional_embedding = nn.Embed(
            num_embeddings=21,
            features=self.config.model.executor.observation_embeddings.embedding_dim,
        )

        # padding token
        self.padding_embedding = nn.Embed(
            num_embeddings=1,
            features=self.config.model.executor.observation_embeddings.embedding_dim,
        )

    def __call__(self, text, images, actions):
        
        ### Tokenization + Input Embeddings ###

        ## text embeddings
        text_embeddings = self.text_tokenizer(text)

        ## image embeddings
        image_embeddings = self.image_tokenizer(images)

        ## action embeddings
        action_embeddings = self.action_tokenizer(actions)

        ## observation embeddings
        #observation_embeddings = self.positional_embedding(jnp.arange(21))

        # concatenate text, image, action and observation embeddings

        # interleave image and action embeddings such that image, action, image, action, ...
        def interweave_embeddings(image_embeddings, action_embeddings):
            batch_size = image_embeddings.shape[0]
            feature_size = image_embeddings.shape[-1]
            num_tokens = image_embeddings.shape[1] + action_embeddings.shape[1]

            # interleave image and action embeddings
            interleaved_embeddings = jnp.zeros((batch_size, num_tokens, feature_size))
            interleaved_embeddings[::2, :] = image_embeddings
            interleaved_embeddings[1::2, :] = action_embeddings

            return interleaved_embeddings

        # interleave image and action embeddings
        interleaved_embeddings = interweave_embeddings(image_embeddings, action_embeddings)

        # concatenate text and interleaved embeddings
        embeddings = jnp.concatenate((text_embeddings, interleaved_embeddings), axis=1)

        # add padding token
        pad_length = self.config.model.executor.max_seq_len - embeddings.shape[1]
        padding = jnp.zeros((embeddings.shape[0], pad_length))
        padding = self.padding_embedding(padding)
        embeddings = jnp.concatenate((embeddings, padding), axis=1)

        print(embeddings.shape)

        ### Transformer Self Attention ###

        return embeddings 
