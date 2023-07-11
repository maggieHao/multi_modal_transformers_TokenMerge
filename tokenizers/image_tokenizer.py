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
from jax import random

# import custom utils for logging
from ..utils.logger import get_logger

ModuleDef = Any
LOG = get_logger(__name__)

############################
# Image Preprocessing
############################

# TODO(peterdavidfagan): verify this is row-major format.
def image_to_patches(image, patch_size, normalize):
    """
    Converts an image into patches, assuming square images.
    Assumes HWC convention.

    Args:
        image (jax.numpy.ndarray): the image to be converted into patches.
        patch_size (int): the size of the patch.
        normalize (bool): whether to normalize the pixel values of the patches.

    Returns:
        jax.numpy.ndarray: the patches of the image.
    """
    # check if the image is square
    h, w, c = image.shape
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
        image, "(h p1) (w p2) c -> (h w) (p1) (p2) (c)", p1=patch_size, p2=patch_size
    )

    # normalize pixel values
    if normalize:
        patches = (2*patches / 255.0) - 1.0
        patches = patches / jnp.sqrt(patch_size)

    return patches


def encode_patch_position(image, patch_size, key, train=True):
    """
    Calculates the patch interval for an image.
    """
    h, w, c = image.shape
    
    def get_patch_position_encoding(interval_length, start_idx, stop_idx, key):
        # normalize patch interval
        start_idx = start_idx / interval_length
        stop_idx = stop_idx / interval_length

        # quantize patch interval
        start_idx = jnp.floor(start_idx * interval_length)
        stop_idx = jnp.ceil(stop_idx * interval_length)
    

        if train:
            # sample uniformly from the patch interval
            return random.randint(key, shape=(1,), minval=start_idx, maxval=stop_idx+1)
        else:
            # use the center of the patch interval
            return (start_idx + stop_idx) // 2

    row_intervals = jnp.arange(0, h+patch_size, patch_size)
    row_keys = random.split(key, h // patch_size)
    col_intervals = jnp.arange(0, w+patch_size, patch_size)
    col_keys = random.split(key, w // patch_size)
        
    row_position_encoding = jax.vmap(get_patch_position_encoding, in_axes=(None, 0, 0, 0))(
            h, row_intervals[:-1], row_intervals[1:], row_keys)
    row_position_encoding = jnp.squeeze(row_position_encoding)
    row_position_encoding = jnp.repeat(row_position_encoding, w // patch_size, axis=0)

    col_position_encoding = jax.vmap(get_patch_position_encoding, in_axes=(None, 0, 0, 0))(
            w, col_intervals[:-1], col_intervals[1:], col_keys)
    col_position_encoding = jnp.squeeze(col_position_encoding)
    col_position_encoding = jnp.repeat(col_position_encoding, h // patch_size, axis=0)

    return row_position_encoding, col_position_encoding



############################
# Image Embedding
############################


### GATO RESNET (Incomplete) ###

# https://github.com/google/flax/blob/main/examples/imagenet/models.py
class ResNetV2Block(nn.Module):
    """
    Note: fixing parameter defaults to match Gato.
    """
    features: int
    strides: Tuple[int, int] = (1, 1)
    kernel_size: Tuple[int, int] = (3, 3)
    padding: str = "SAME"
    conv: ModuleDef = nn.Conv
    normalization: ModuleDef = nn.GroupNorm
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # TODO: review groupnorm 
        residual = x

        y = self.normalization(num_groups=3)(x)
        y = self.activation(y)
        y = self.conv(features=self.features,
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        y = self.normalization()(y)
        y = self.activation(y)
        y = self.conv(features=3, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding)(y)

        out = y+residual

        #flatten output
        out = jnp.ravel(out)
        
        return out

### RobotCat VQ-GAN (Incomplete) ###



########################
# Image Tokenizer
########################

class ImageTokenizer(nn.Module):
    """
    Converts images into tokens.
    """

    config: dict

    @property
    def tokens_per_image(self):
        raise NotImplementedError

    def setup(self):
        self.patch_size = self.config["patch_size"]
        self.normalize = self.config["normalize"]
        self.embedding_function = ResNetV2Block(features = self.config["embedding_dim"])
        self.row_embeddings = nn.Embed(self.config["position_interval"], (self.patch_size**2)*3)
        self.col_embeddings = nn.Embed(self.config["position_interval"], (self.patch_size**2)*3)

    def __call__(self, image, key, train=True):
        """
        Args:
            images (jax.numpy.ndarray): the images to be tokenized (num_batches, num_sequences, num_images, H, W, C).
        """

        # convert image into patches
        patches = image_to_patches(image, self.patch_size, self.normalize)

        chex.assert_equal(
                patches.shape,
                (
                    (image.shape[0]//self.patch_size)**2, # patches per image
                    self.patch_size, # patch_dim
                    self.patch_size, # patch_dim
                    image.shape[-1] # channels_dim
                )
                )

        # create patch embeddings
        def embed_patch(patch):
            return self.embedding_function(patch)

        patch_embeddings = jax.vmap(embed_patch)(patches)
        
        chex.assert_equal(
                patch_embeddings.shape,
                (
                    (image.shape[0]//self.patch_size)**2, # patches per image
                    (self.patch_size**2)*image.shape[-1] # embedding_dim
                )
                )

        # create patch position embeddings
        row_positions, col_positions = encode_patch_position(image, self.patch_size, key, train=train)
        
        def embed_row(row):
            return self.row_embeddings(row)

        def embed_col(col):
            return self.col_embeddings(col)
        
        row_position_embeddings = jax.vmap(embed_row)(row_positions)
        col_position_embeddings = jax.vmap(embed_col)(col_positions)

        chex.assert_equal(
                row_position_embeddings.shape,
                patch_embeddings.shape
                )

        return patch_embeddings + row_position_embeddings + col_position_embeddings

