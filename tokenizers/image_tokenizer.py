"""
Image tokenizer implementation that aligns with Gato paper.
"""

import dataclasses
import warnings
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import einops
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any


############################
# Image Tokenizer
############################

# TODO(peterdavidfagan): verify this is row-major format.
def image_to_patches(image, patch_size, normalize):
    """
    Converts an image into patches, assuming square images.

    Args:
        image (jax.numpy.ndarray): the image to be converted into patches.
        patch_size (int): the size of the patch.
        normalize (bool): whether to normalize the pixel values of the patches.

    Returns:
        jax.numpy.ndarray: the patches of the image.
    """
    # check if the image is square
    c, h, w = image.shape
    chex.assert_equal(h, w)

    # check if the image is divisible by the patch size
    # automatically resize if not
    if h % patch_size != 0:
        warnings.warn(
            "The image is not divisible by the patch size. Automatically resizing image."
        )
        new_dim = h // patch_size
        image = jax.image.resize(image, (new_dim, new_dim), method="bilinear")

    # create an array of patches
    patches = einops.rearrange(
        image, "c (h p1) (w p2) -> (h w) (p1) (p2) (c)", p1=patch_size, p2=patch_size
    )

    # normalize pixel values
    if normalize:
        patches = (2*patches / 255.0) - 1.0
        patches = patches / jnp.sqrt(patch_size)

    return patches

# vectorise image patching function
image_to_patches_v = jax.vmap(image_to_patches, in_axes=(0, None, None), out_axes=(0))

# create datclass (inspired by: https://github.com/google/flax/blob/71e4432d62306afd0fd12f556ba077de1362eb46/examples/wmt/tokenizer.py#L146)
@dataclasses.dataclass
class ImageTokenizeOp:
    sp_tokenizer: Any

    def __call__(self, features):
        return self.sp_tokenizer(features)

########################
# Position Encoding
########################

class PositionEncoder(nn.Module):
    """
    Original Transformer position encoding scheme.

    Inspired by: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """

    config: dict

    def setup(self):
        self.encode_mat = jnp.zeros(
            (1, self.config["max_len"], self.config["embedding_dim"])
        )
        domain_values = jnp.arange(self.config["max_len"], dtype=jnp.float32).reshape(
            -1, 1
        ) / jnp.power(
            10000,
            jnp.arange(0, self.config["embedding_dim"], 2, dtype=jnp.float32)
            / self.config["embedding_dim"],
        )
        self.encode_mat = self.encode_mat.at[:, :, 0::2].set(jnp.sin(domain_values))
        self.encode_mat = self.encode_mat.at[:, :, 1::2].set(jnp.cos(domain_values))

    def __call__(self, sequence):
        return self.encode_mat[:, : sequence.shape[1], :]


# use resnet for patch embedding as in Gato paper.
# https://github.com/google/flax/blob/main/examples/imagenet/models.py

class ResNetV2Block(nn.Module):
    """
    Note: fixing parameter defaults to match Gato.
    """
    features: int
    strides: Tuple[int, int] = (1, 1)
    kernel_size: Tuple[int, int] = (3, 3)
    padding: str = "SAME"
    weights: ModuleDef = nn.Conv
    normalization: ModuleDef = nn.GroupNorm
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # Important: I am uncertain about this function.
        #residual = x

        # its not possible to perform group norm with 32 groups on image
        # containing only 3 channels! Start with conv before first group norm.
        y = self.weights(features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding=self.padding)(x)
        
        residual = y

        y = self.normalization()(y)
        y = self.activation(y)
        y = self.weights(features=self.features, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        y = self.normalization()(y)
        y = self.activation(y)
        y = self.weights(features=self.features, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        return y+residual


class ImageTokenizer(nn.Module):
    """
    Tokenizer for images.
    """

    config: dict

    @property
    def tokens_per_image(self):
        raise NotImplementedError

    def setup(self):
        self.patch_size = self.config["patch_size"]
        self.normalize = self.config["normalize"]
        self.embedding_function = ResNetV2Block(features = self.config["embedding_dim"])
        #self.position_encoder = PositionEncoder(self.config)

    def __call__(self, image):
        # convert image into patches
        patches = image_to_patches_v(image, self.patch_size, self.normalize)
        
        # create embeddings using ResNetV2
        patches = patches.reshape(-1, self.patch_size, self.patch_size, 3)
        embedding = self.embedding_function(patches) 
        print(embedding.shape)
        embedding = embedding.reshape(-1, 20, self.config["embedding_dim"])

        return embedding


if __name__ == "__main__":
    # for debugging purposes
    image = jnp.ones((15, 3, 280, 280))
    image_tokens = image_to_patches_v(image, 14, True)

    # instantiate tokenizer
    config = {
        "patch_size": 14,
        "normalize": True,
        "embedding_dim": 64,
            }
    tokenizer = ImageTokenizer(config)

    # initialize tokenizer
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = tokenizer.init(init_rng, image)
    print(params)
    print(image_tokens.shape)
