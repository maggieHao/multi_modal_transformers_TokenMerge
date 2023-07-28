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
from tokenizers.value_tokenizer import ActionTokenizer
from tokenizers.image_tokenizer import ImageTokenizer
from tokenizers.text_tokenizer import BasicTextTokenizer

# transformer modules
from transformer_components import Encoder1DBlock


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
    attention_mask_input = e.rearrange(
        e.repeat(
            jnp.where(actions == 0, 0, 1),
            "batch seq -> batch seq repeats",
            repeats=tokens_per_image + 1,
        ),  # patches per image + action
        "batch seq repeats -> batch (seq repeats)",
    )

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
    multi_head_attention_mask = e.repeat(
        attention_mask,
        "batch head_dim q k -> batch (head_dim repeats) q k",
        repeats=num_heads,
    )

    return multi_head_attention_mask


class ConceptLearner(nn.Module):
    """A multi-modal decoder-only Transformer architecture."""

    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, actions):
        """Forward pass through the model."""
        # Tokenization + Generate Input Embeddings

        # text embeddings
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # image embeddings
        image_tokenizer = ImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images)
        batch_size, sequence_length, num_tokens_per_image, _ = image_embeddings.shape

        # action embeddings
        action_tokenizer = ActionTokenizer(config=self.config.action_tokenizer)
        action_embeddings = action_tokenizer(actions)

        # positional encoding of observation tokens
        num_observation_tokens = num_tokens_per_image + 1  # image + action
        observation_position_tokens = jnp.arange(0, num_observation_tokens)
        observation_position_tokens = e.repeat(
                                    observation_position_tokens, 
                                    'tokens -> batch seq tokens', 
                                    batch=batch_size, 
                                    seq=self.config.max_seq_len, # TODO: this shouldn't be hardcoded!!!
                                    )
        
        observation_position_embeddings = nn.Embed(
            num_observation_tokens,
            self.config.token_embedding_dim,
        )(observation_position_tokens)
        
        # rearrange observation position embeddings
        #observation_position_embeddings = e.rearrange(
        #        observation_position_embeddings, 
        #        '(batch seq token) embed -> batch seq tokens embed')
        
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
        for _ in range(self.config.self_attention.num_blocks):
            x = Encoder1DBlock(self.config.self_attention)(
                x,
                mask=attention_mask,
            )

        # calculate logits with linear layer
        x = e.rearrange(x, "batch tokens features -> batch (tokens features)")
        x = nn.Dense(
            self.config.linear_out.output_dim,
        )(x)

        return x
