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

    def __call__(self, text, images, actions):
        
        ## text embeddings
        text_embeddings = self.text_tokenizer(text)

        ## observation embeddings
        def embed_observation(image, action):
            # embed image
            image_embedding = self.image_tokenizer(image)

            # embed action
            action_embedding = self.action_tokenizer(action)
            
            # concatenate embeddings
            observation_embedding = jnp.concatenate([image_embedding, action_embedding], axis=1)

            # add learnable positional embeddings
            positional_embedding = self.positional_embedding(jnp.arange(21))
            observation_embedding = observation_embedding + positional_embedding

            return observation_embedding
        
        observation_embeddings = jax.vmap(embed_observation)(images, actions)

        # concatenate text and observation embeddings
        embeddings = jnp.concatenate([text_embeddings, observation_embeddings], axis=1)

        # pad embeddings to max length
        
        # pass embeddings through transformer blocks
        for lyr in range(self.config["num_blocks"]):
            embeddings = DecoderBlock(config=config)(embeddings)

        # pass through final layer norm
        outputs = nn.LayerNorm()(x)

        # this should be a distribution over all possible actions
        logits = nn.Dense()(outputs)

        return logits 
