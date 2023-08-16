"""
Vision Langauge Model (VLM) implementation.
"""
from functools import partial

# deep learning framework
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen import initializers
import einops as e

# custom tokenizers
from multi_modal_transformers.tokenizers.text_tokenizer import BasicTextTokenizer
from multi_modal_transformers.tokenizers.image_tokenizer import SingleImageTokenizer

# transformer modules
from multi_modal_transformers.transformer_components import Encoder1DBlock

from hydra.utils import instantiate


# model conponents

# tokenizers
class ImageTextTokenizer(nn.Module):
    """
    Tokenize  image + text to sequence of token embeddings.

    Example sequence: image_tok, image_tok, ..., image_tok, text_tok, text_tok, ..., text_tok
    """
    
    config: dict

    @nn.compact
    def __call__(self, images, text, train=False):
        """Tokenize text, image, and action embeddings."""
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
        attention_mask_input, ps = e.pack((image_mask, text_mask), "batch *")
        attention_mask = nn.make_attention_mask(
            attention_mask_input>0,
            attention_mask_input>0,
                )

        return token_embeddings, attention_mask


# decoder-only transformer
class DecoderTransformer(nn.Module):
    """Transformer decoder."""

    config: dict

    @nn.compact
    def __call__(self, token_embeddings, attention_mask, train=False):
        """Transformer decoder."""
        
        # 1D attention mask -> multi-head attention
        multi_head_attention_mask = e.repeat(
            attention_mask,
            "batch head_dim q k -> batch (repeats head_dim) q k",
            repeats=self.config.transformer.self_attention.num_heads,
        )
        
        num_blocks = self.config.transformer.num_blocks
        # TODO: replace for loop with flax.linen.scan
        

        for i in range(num_blocks):
            x = Encoder1DBlock(self.config.transformer)(
                token_embeddings,
                mask=multi_head_attention_mask,
                train=train,
            )
        
        return x


# network heads
class TokenLogitHead(nn.Module):
    """Token logits."""

    config: dict

    @nn.compact
    def __call__(self, contextual_embeddings, token_idx):
        """Token logits."""
        token_logits = instantiate(self.config.transformer.token_logit_head)(contextual_embeddings)
        return token_logits[jnp.arange(token_logits.shape[0]), token_idx.astype(int), :]


class StateValueHead(nn.Module):
    """State value."""

    config: dict

    @nn.compact
    def __call__(self, contextual_embeddings):
        """State value."""
        return instantiate(self.config.transformer.state_value_head)(contextual_embeddings)


# indexing functions
def get_imagetext_idx(text, num_tokens_per_image):
    """Retrieve next token index from text sequence."""
    text_idx = jnp.argmax(text == 0, axis=-1)
    return num_tokens_per_image + text_idx


# storing model parameters
@flax.struct.dataclass
class ConceptPlanAgent:
    # mapping from tokens to input embeddings
    tokenizer_params: dict

    # transformer self-attention layers
    transformer_params: dict

    # output prediction layers
    token_logits_params: dict
    state_value_params: dict

    @property
    def params(self):
        return {
                "params": {
            "tokenizer_params": self.tokenizer_params["params"],
            "transformer_params": self.transformer_params["params"],
            "token_logits_params": self.token_logits_params["params"],
            "state_value_params": self.state_value_params["params"],
            }
        }


# TODO: define a model class with multiple inference methods

# model inference
@partial(jax.jit, static_argnums=(3, 4))
def predict_next_token(agent_state, images, text, train=False, search="greedy"):
    """Predict next token."""
    # get index of next token
    next_token_idx = get_imagetext_idx(text, agent_state.params.tokenizer_params.num_tokens_per_image)
    
    # tokenize, embed, and predict next token
    input_token_embeddings, attention_mask = agent_state.tokenize(agent_state.params.tokenizer_params, images, text, train=train)
    contextual_embeddings = agent_state.embed(agent_state.params.transformer_params, input_token_embeddings, attention_mask, train=train)
    next_token_logits = agent_state.token_logits(agent_state.params.token_logits_params, contextual_embeddings, next_token_idx, train=train)

    # choose next token using search strategy
    if search == "greedy":
        # get next token and log probability
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token_log_prob = jax.nn.log_softmax(next_token_logits, axis=-1)[jnp.arange(next_token_logits.shape[0]), next_token]
    else:
        raise NotImplementedError

    return next_token, next_token_log_prob


@partial(jax.jit, static_argnums=(2, 3))
def predict_concept_and_value(agent_state, images, train=False, search="greedy"):
    """Autoregressively generate a sequence of tokens that defines a concept to execute."""
    # initialize text with padding token
    text = jnp.zeros((images.shape[0], 4), dtype=jnp.int32) # replace 4 with max sequence length from config
    text_log_probs = jnp.zeros((images.shape[0], 1), dtype=jnp.float32) # replace 4 with max sequence length from config
    terminate_mask = jnp.zeros((images.shape[0], 4), dtype=jnp.int32) # replace 4 with max sequence length from config
    
    for idx in range(4): # replace 4 with max sequence length from config:
        # predict state value before text generation
        if idx == 0:
            state_value = agent_state.state_value(agent_state.params.state_value_params, contextual_embeddings, train=train)
        
        # get index of next token
        next_token_idx = get_imagetext_idx(text, agent_state.params.tokenizer_params.num_tokens_per_image)
        
        # predict next token
        input_token_embeddings, attention_mask = agent_state.tokenize(agent_state.params.tokenizer_params, images, text, train=train)
        contextual_embeddings = agent_state.embed(agent_state.params.transformer_params, input_token_embeddings, attention_mask, train=train)
        next_token_logits = agent_state.token_logits(agent_state.params.token_logits_params, contextual_embeddings, next_token_idx, train=train)

        # choose next token using search strategy
        if search == "greedy":
            # get next token
            next_token = jnp.argmax(next_token_logits, axis=-1)
        else:
            raise NotImplementedError
        
        # get log probability of next token
        next_token_log_prob = jax.nn.log_softmax(next_token_logits, axis=-1)[jnp.arange(next_token_log_prob.shape[0]), next_token]
        
        # apply terminate mask to log probability
        next_token_log_prob = jnp.where(terminate_mask, 0, next_token_log_prob)
        
        # update text sequence log probability
        text_log_probs += next_token_log_prob
        
        # mask next token with terminate mask
        next_token = jnp.where(terminate_mask, 0, next_token)

        # set next token value in text
        def set_token_value(text, idx, token):
            text = text.at[idx].set(token)

        text = jax.vmap(set_token_value, in_axes=(0, None, None))(text, idx, next_token)
        
        # update terminate mask
        terminate_mask = jnp.logical_or(terminate_mask, next_token == 5)
        
    return text, text_log_probs, state_value
