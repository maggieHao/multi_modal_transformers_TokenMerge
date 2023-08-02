"""Unit tests for image_tokenizer.py."""

# testing framework
from absl.testing import absltest
from absl.testing import parameterized
import chex

# image tokenizer
from tokenizers.image_tokenizer import (
        image_to_patches,
        encode_patch_position,
        )

# deep learning framework
import jax
import jax.numpy as jnp
import einops as e

class ImageTokenizerTest(parameterized.TestCase):
    """Unit tests for ImageTokenizer."""

    def test_image_to_patches(self):
        """Test image_to_patches."""
        # create dummy image where each patch contains constant value 
        # values are assigned in raster scan order
        patches = jnp.ones((16, 70, 70, 3))
        value_assignment = jnp.arange(16) + 1 # 1, 2, 3, ..., 16
        patches = e.einsum(patches, value_assignment, 'b h w c, b -> b h w c') # assign values
        image = e.rearrange(patches, '(row col) h w c -> (row h) (col w) c', row=4, col=4)

        # generate patches
        result_patch = image_to_patches(image, patch_size=70, normalize=False)
        
        # check shape and values
        chex.assert_shape(result_patch, (16, 70, 70, 3))
        chex.assert_tree_all_close(result_patch, patches)


    def test_encode_patch_position(self):
        """Test encode_patch_position."""
        # deterministic case
        patch_encoding = jnp.arange(128)
        image = jnp.ones((128, 128, 3))
        patch_size = 1
        num_patches = 128**2
        row_encoding, col_encoding = encode_patch_position(
                image, 
                None,
                patch_size=patch_size, 
                num_tokens=128,
                train=False)
        chex.assert_shape(row_encoding, (num_patches,))
        chex.assert_tree_all_close(row_encoding[123], jnp.int32(122))


        # stochastic case
        key = jax.random.PRNGKey(0)
        patch_encoding = jnp.arange(280)
        image = jnp.ones((280, 280, 3))
        patch_size = 1
        num_patches = (280/patch_size) ** 2
        row_encoding, col_encoding = encode_patch_position(
                image, 
                key,
                patch_size=patch_size, 
                num_tokens=128,
                train=True)
        chex.assert_shape(row_encoding, (num_patches,))
        chex.assert_tree_all_close(row_encoding[123], jnp.int32(122), atol=70)

    # TODO: test the call method
    #def test_call(self):


if __name__ == '__main__':
  absltest.main()
