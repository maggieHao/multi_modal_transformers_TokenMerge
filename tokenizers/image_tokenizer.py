"""
A set of image tokenizers for use with MART.
"""
import warnings

import einops
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn


########################
# Helper Functions
########################


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
    if h % patch_size != 0:
        warnings.warn(
            "The image is not divisible by the patch size. Automatically resizing image."
        )
        new_dim = h // patch_size
        image = jax.image.resize(image, (new_dim, new_dim), method="bilinear")

    # create an array of patches
    patches = einops.rearrange(
        image, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )

    # normalize pixel values
    if normalize:
        patches = (patches + 1.0) / 255.0
        patches = patches / jnp.sqrt(patch_size)

    return patches


# vectorise image patching function
image_to_patches_v = jax.vmap(image_to_patches, in_axes=(0, None, None), out_axes=(0))

########################
# Position Encodings
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


# TODO: Create learnable position encoding scheme


########################
# Tokenizers
########################


class ViTTokenizer(nn.Module):
    """
    Generating image tokens in accordance with ViT (https://arxiv.org/abs/2010.11929).
    """

    config: dict
    dtype = jnp.float32

    def setup(self):
        self.embed_patch = nn.Dense(features=self.config["embedding_dim"])
        self.embed_position = PositionEncoder(self.config)

    def __call__(self, image):

        # convert the image into image patches
        image_patches = image_to_patches_v(image, self.config["patch_size"], True)

        # generate image patch embeddings
        image_patch_embeddings = self.embed_patch(image_patches)

        # add position embeddings
        position_embedding = self.embed_position()
        image_patch_embeddings = image_patch_embeddings + position_embedding

        return image_patch_embeddings


if __name__ == "__main__":
    # for debugging purposes
    config = {"image_shape": (224, 224), "patch_size": 16}

    tokenizer = ViTTokenizer(config)
    image = jnp.ones((1, 3, 224, 224))
    image_tokens = tokenizer(image)
    print(image_tokens.shape)
