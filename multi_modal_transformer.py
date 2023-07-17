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

def combine_embeddings(action_embeddings, image_embeddings, text_embeddings, obs_pos_embeddings):
    """
    Combines action, image, and text embeddings into a single embedding vector.
    """
    # get dimensions
    (batch_size, num_images, tokens_per_image, feature_size) = image_embeddings.shape
    num_actions = action_embeddings.shape[1]
    total_tokens = (num_images*tokens_per_image) + num_actions 

    # perform interleaving
    embeddings = jax.lax.concatenate((image_embeddings, jnp.expand_dims(action_embeddings, axis=2)), dimension=2)
    embeddings = jnp.reshape(embeddings, (batch_size, total_tokens, feature_size))
    
    # add positional embeddings
    embeddings = embeddings + obs_pos_embeddings

    # concatenate text and interleaved embeddings
    embeddings = jnp.concatenate((text_embeddings, embeddings), axis=1)
    
    return embeddings

def generate_attention_mask(actions, image_embeddings, text_embeddings, num_heads):
    """
    Generates multi-head attention mask for padding tokens.
    """
    # get dimensions
    (batch_size, num_images, tokens_per_image, feature_size) = image_embeddings.shape
    (_, num_actions) = actions.shape
    (_, text_tokens, _) = text_embeddings.shape

    # create binary mask for padding tokens in image + action sequence
    attention_mask_input = e.rearrange(
            e.repeat(
                jnp.where(actions == 0, 0, 1), 
                'batch seq -> batch seq repeats', 
                repeats=tokens_per_image + 1) # patches per image + action 
            , 'batch seq repeats -> batch (seq repeats)')
        
    # create binary mask for text tokens
    text_mask = jnp.ones((batch_size, text_tokens))
    attention_mask_input = jnp.concatenate(
            (text_mask, attention_mask_input), 
            axis=1,
            )
    
    # generate 1D attention mask
    attention_mask = nn.make_attention_mask(
            attention_mask_input > 0, 
            attention_mask_input > 0,
            )

    # 1D attention mask -> multi-head attention
    multi_head_attention_mask = e.repeat(attention_mask, 'batch head_dim q k -> batch (head_dim repeats) q k', repeats=num_heads)

    return multi_head_attention_mask


class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""
    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, actions):
        
        ### Tokenization + Generate Input Embeddings ###

        ## text embeddings ##
        text_tokenizer = BasicTokenizer(
            vocab_dir=self.config.text_tokenizer.vocab_dir
            )
        text_tokenizer = BasicTextTokenizer(
            config = self.config.text_tokenizer, tokenizer=text_tokenizer
        )
        text_embeddings = text_tokenizer(text)

        
        ## image embeddings ##
        image_tokenizer = ImageTokenizer(config = self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images)


        ## action embeddings ##
        action_tokenizer = ActionTokenizer(config = self.config.action_tokenizer)
        action_embeddings = action_tokenizer(actions)

        ## positional encoding of observation tokens ##
        (batch_size, num_images, tokens_per_image, _) = image_embeddings.shape
        num_actions = action_embeddings.shape[1]
        total_tokens = (num_images*tokens_per_image) + num_actions
        tokens_per_obs = tokens_per_image + 1

        obs_encoder = nn.Embed(
                tokens_per_obs,
                self.config.token_embedding_dim,
                )
        
        obs_positions = jnp.arange(tokens_per_obs)
        obs_positions_batch = e.repeat(obs_positions, 'observation -> batch (seq observation)', batch=batch_size, seq=total_tokens//tokens_per_obs) # one image per observation
        obs_pos_embeddings = obs_encoder(obs_positions_batch)

        
        ## combine embeddings ##
        combined_embeddings = combine_embeddings(action_embeddings, image_embeddings, text_embeddings, obs_pos_embeddings)

        
        ### Transformer Self Attention ###

        # generate attention mask for padding tokens
        attention_mask = generate_attention_mask(actions, image_embeddings, text_embeddings, self.config.self_attention.num_heads)

        # pass through self attention layer
        for attention_block in range(self.config.self_attention.num_blocks):
            x = Encoder1DBlock(self.config.self_attention)(
                    combined_embeddings, 
                    mask=attention_mask,
                    )

        print(x.shape)
        print(attention_mask.shape)

        return x
