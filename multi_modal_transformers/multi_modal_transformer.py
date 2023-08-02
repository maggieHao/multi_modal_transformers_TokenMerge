"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

# deep learning framework
import jax
import jax.numpy as jnp
import flax.linen as nn
import einops as e

# custom tokenizers
from multi_modal_transformers.tokenizers.value_tokenizer import ActionTokenizer
from multi_modal_transformers.tokenizers.image_tokenizer import ImageTokenizer
from multi_modal_transformers.tokenizers.text_tokenizer import BasicTextTokenizer

# transformer modules
from multi_modal_transformers.transformer_components import Encoder1DBlock


def combine_embeddings(
    text_embeddings, image_embeddings, action_embeddings, observation_position_embeddings
):
    """Combine action, image, and text embeddings into a single embedding vector."""
    # interleave images and actions to create observations
    action_embeddings = jnp.expand_dims(action_embeddings, axis=2) # add axis for tokens
    observation_embeddings, ps = e.pack((image_embeddings, action_embeddings), 'batch seq * embed')
    
    # add positional embeddings for observations
    observation_embeddings = observation_embeddings + observation_position_embeddings
    observation_embeddings = e.rearrange(observation_embeddings, 'batch seq tokens embed -> batch (seq tokens) embed')

    # concatenate text and observation embeddings
    embeddings, ps = e.pack((text_embeddings, observation_embeddings), 'batch * embed')

    return embeddings

def generate_attention_mask(actions, image_embeddings, text_embeddings, num_heads):
    """Generate multi-head attention mask for padding tokens."""
    # get dimensions
    (batch_size, num_images, tokens_per_image, feature_size) = image_embeddings.shape
    (_, num_actions) = actions.shape
    (_, text_tokens, _) = text_embeddings.shape

    # create binary mask for padding tokens in image + action sequence
    observation_mask = e.repeat(
            jnp.where(actions == 0, 0, 1),
            "batch seq -> batch (seq repeats)",
            repeats=tokens_per_image + 1, # image tokens + one action token
        )

    # create binary mask for text tokens
    text_mask = jnp.ones((batch_size, text_tokens))

    # pack text and observation mask
    attention_mask_input, ps = e.pack((text_mask, observation_mask), 'batch *')

    # generate 1D attention mask
    attention_mask = nn.make_attention_mask(
        attention_mask_input > 0,
        attention_mask_input > 0,
    )

    # 1D attention mask -> multi-head attention
    multi_head_attention_mask = e.repeat(
        attention_mask,
        "batch head_dim q k -> batch (head_dim repeats) q k",
        repeats=num_heads,
    )

    return multi_head_attention_mask

def slice_action_sequence(actions, embeddings, num_text_tokens, num_obs_tokens):
    """Retrieve action embeddings from sequence."""

    # TODO: provide this index in dataset
    # the first zero value in the action sequence is the target action index
    target_action_idx = jnp.argmax(actions == 0, axis=-1)
    target_action_idx = ((target_action_idx+1)*num_obs_tokens) + num_text_tokens
    target_action_idx = target_action_idx - 1 # correct for zero indexing

    # slice action embeddings
    action_logits = embeddings[jnp.arange(embeddings.shape[0]), target_action_idx, :]

    return action_logits
    


class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""

    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, actions, train=True):
        """Forward pass through the model."""
        # Tokenization + Generate Input Embeddings
        
        # text embeddings
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # image embeddings
        image_tokenizer = ImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, sequence_length, num_tokens_per_image, _ = image_embeddings.shape

        # action embeddings
        action_tokenizer = ActionTokenizer(config=self.config.action_tokenizer)
        action_embeddings = action_tokenizer(actions)

        # positional encoding of observation tokens
        observation_position_embeddings = self.param(
            "observation_embeddings",
            nn.initializers.normal(stddev=0.02),
            (num_tokens_per_image + 1, self.config.token_embedding_dim),
                )
        
        # combine embeddings
        x = combine_embeddings(
            text_embeddings,
            image_embeddings, 
            action_embeddings, 
            observation_position_embeddings,
        )

        # Transformer Self Attention
        # generate attention mask for padding tokens
        attention_mask = generate_attention_mask(
            actions,
            image_embeddings,
            text_embeddings,
            self.config.self_attention.num_heads,
        )

        # pass through self attention layer
        config_= self.config.self_attention.copy()
        config_.out_dim = None
        for i in range(self.config.self_attention.num_blocks):
            if i != self.config.self_attention.num_blocks - 1:
                x = Encoder1DBlock(config_)(
                    x,
                    mask=attention_mask,
                )
            else:
                x = Encoder1DBlock(self.config.self_attention)(
                    x,
                    mask=attention_mask,
                )

        # get action logits at appropriate timestep
        action_logits = slice_action_sequence(actions, x, text_embeddings.shape[1], num_tokens_per_image +1)

        return action_logits
