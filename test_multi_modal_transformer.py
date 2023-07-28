"""Unit tests for multi_modal_transformer.py."""

# testing framework
from absl.testing import absltest
from absl.testing import parameterized
import chex

# multi modal transformer
from multi_modal_transformer import (
        combine_embeddings,
        generate_attention_mask,
        ConceptLearner,
        )

# deep learning framework
import jax
import jax.numpy as jnp
import einops as e

class MultiModalTransformerTest(parameterized.TestCase):
    """Unit tests for MultiModelTransformer."""

    def test_combine_embeddings(self):
        """Test combine_embeddings."""
        # dummy data
        batch_size = 10
        sequence_length = 20
        tokens_per_image = 10
        obs_dim = tokens_per_image + 1 # image + action
        embedding_dim = 512

        text_embedding = jnp.ones((batch_size, 5, embedding_dim))
        image_embedding = jnp.ones((batch_size, sequence_length, tokens_per_image, embedding_dim))
        action_embedding = jnp.ones((batch_size, sequence_length, embedding_dim))
        observation_position_embedding = jnp.ones((batch_size, sequence_length, obs_dim, embedding_dim)) 
        # compute embedding
        combined_embedding = combine_embeddings(
                text_embedding,
                image_embedding,
                action_embedding,
                observation_position_embedding,
                )

        # check shape
        chex.assert_shape(combined_embedding, (batch_size, 5 + (sequence_length*obs_dim), embedding_dim))

if __name__ == '__main__':
  absltest.main()
