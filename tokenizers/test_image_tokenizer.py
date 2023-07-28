"""Unit tests for image_tokenizer.py."""

# testing framework
from absl.testing import absltest
from absl.testing import parameterized
import chex

# image tokenizer
from tokenizers.image_tokenizer import image_to_patches

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


if __name__ == '__main__':
  absltest.main()
