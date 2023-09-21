"""
Transformer architecture implementation.

Heavily inspired by: https://github.com/google/flax/blob/main/examples/wmt/models.py
"""

# deep learning framework
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
import einops as e

# custom tokenizers
from multi_modal_transformers.tokenizers.value_tokenizer import ActionTokenizer
from multi_modal_transformers.tokenizers.image_tokenizer import ImageTokenizer, SingleImageTokenizer
from multi_modal_transformers.tokenizers.text_tokenizer import BasicTextTokenizer

# transformer modules
from multi_modal_transformers.transformer_components import Encoder1DBlock


from hydra.utils import instantiate

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
        "batch head_dim q k -> batch (repeats head_dim) q k",
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
    

class ConceptLearner:
    """High-level wrapper for ConceptLearnerV1 and ConceptLearnerV2."""

    @classmethod
    def initialize_from_config(cls, cfg):
        """Initialize ConceptLearner."""
        if cfg.version == "v1":
            return ConceptLearnerV1(cfg)
        elif cfg.version == "v2":
            return ConceptLearnerV2(cfg)
        else:
            raise NotImplementedError(f"ConceptLearner version {cfg.version} not implemented.")
            

class ConceptLearnerV1(nn.Module):
    """A multi-modal decoder-only Transformer architecture, inspired by the GATO architecture."""

    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, actions, train=False):
        """Forward pass through the model."""
        # Tokenization + Generate Input Embeddings
        batch_size = text.shape[0]

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
        

        # TODO: investigate moving to config
        # positional encoding of observation tokens
        observation_position_embeddings = instantiate(self.config.observation_position_embedding) # partial
        observation_positions = jnp.arange(num_tokens_per_image + 1)
        observation_positions = e.repeat(observation_positions, "tokens -> batch seq tokens", batch=batch_size, seq=self.config.max_seq_len)
        observation_position_embeddings = observation_position_embeddings(num_embeddings=num_tokens_per_image + 1)(observation_positions)
        
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
            self.config.transformer.self_attention.num_heads,
        )

        # pass through self attention layer
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

        # get action logits at appropriate timestep
        action_logits = slice_action_sequence(actions, x, text_embeddings.shape[1], num_tokens_per_image +1)

        return action_logits



# TODO: refactor to support compute_attention_map method
class ConceptLearnerV2(nn.Module):
    """A multi-modal decoder-only Transformer architecture, that uses a single image token."""

    config: dict

    # TODO: move parameters to single batch parameter
    @nn.compact
    def __call__(self, text, images, train=False):
        """Forward pass through the model."""
        # Tokenization + Generate Input Embeddings
        batch_size = text.shape[0]

        # text embeddings
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # image embeddings
        image_tokenizer = SingleImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, num_tokens_per_image, _ = image_embeddings.shape

        # combine embeddings
        x, _ = e.pack((text_embeddings, image_embeddings), 'batch * embed')
        
        # pass through self attention layer
        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        for i in range(num_blocks):
            x = Encoder1DBlock(self.config.transformer)(
                x,
                mask=None,
                train=train,
            )
        
        # flatten embeddings
        x = e.rearrange(x, 'batch seq embed -> batch (seq embed)')
        # pass through final linear layer
        action_logits = instantiate(self.config.transformer.output_dense)(x)

        return action_logits


    # implement method to compute attention map
    def compute_attention_map(self, text, images, train=False, layer=0):
        """Compute attention map for a layer."""
        # Tokenization + Generate Input Embeddings
        batch_size = text.shape[0]

        # text embeddings
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # image embeddings
        image_tokenizer = SingleImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, num_tokens_per_image, _ = image_embeddings.shape

        # combine embeddings
        x, _ = e.pack((text_embeddings, image_embeddings), 'batch * embed')

        # pass through self attention layer
        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        for i in range(layer+1):
            x = Encoder1DBlock(self.config.transformer)(
                x,
                mask=None,
                train=train,
            )

            if i == layer:
                attention_block = Encoder1DBlock(self.config.transformer)
                QW = Encoder1DBlock.SelfAttention.query_kernel
                KW = Encoder1DBlock.SelfAttention.key_kernel

                # compute attention weights
                attn_weights = jnp.einsum('...qhd,...khd->...hqk', QW[0], KW[0])
                
                # take average across keys
                attn_weights = jnp.mean(attn_weights, axis=-1)

                # take the average across heads
                attn_weights = jnp.mean(attn_weights, axis=-1)

                # normalize
                attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)

        return attn_weights


class ConceptLearnerMetaLoss(nn.Module):
    """A multi-modal decoder-only Transformer architecture, that uses a single image token."""

    config: dict

    @nn.compact
    def __call__(self, text, images, actions, train=False):
        # Tokenization + Generate Input Embeddings
        batch_size = text.shape[0]

        # text embeddings
        text_tokenizer = BasicTextTokenizer(config=self.config.text_tokenizer)
        text_embeddings = text_tokenizer(text)

        # image embeddings
        image_tokenizer = SingleImageTokenizer(config=self.config.image_tokenizer)
        image_embeddings = image_tokenizer(images, train=train)
        batch_size, num_tokens_per_image, _ = image_embeddings.shape

        # action embeddings
        action_tokenizer = ActionTokenizer(config=self.config.action_tokenizer)
        action_embeddings = action_tokenizer(actions)
        # add a dimension for the action token
        action_embeddings = jnp.expand_dims(action_embeddings, axis=1)

         # combine embeddings
        x, _ = e.pack((text_embeddings, image_embeddings, action_embeddings), 'batch * embed')
        
        # pass through self attention layer
        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        for i in range(num_blocks):
            x = Encoder1DBlock(self.config.transformer)(
                x,
                mask=None,
                train=train,
            )
        
        # flatten embeddings
        x = e.rearrange(x, 'batch seq embed -> batch (seq embed)')

        # pass through final linear layer
        action_logits = instantiate(self.config.transformer.output_dense)(x)

        return action_logits

