"""
Image tokenizer implementation that aligns with Gato paper.
"""
import sys

import dataclasses
import warnings
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import einops as e
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random

# import custom utils for logging
from utils.logger import get_logger

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
        image = jax.image.resize(image, (new_dim, new_dim), method="nearest")

    # create an array of patches
    patches = e.rearrange(
        image, "(h p1) (w p2) c -> (h w) (p1) (p2) (c)", p1=patch_size, p2=patch_size
    )

    # normalize pixel values
    if normalize:
        patches = (2*patches / 255.0) - 1.0
        patches = patches / jnp.sqrt(patch_size)

    return patches


def encode_patch_position(image, key, patch_size, num_tokens, train=True):
    """
    Calculates the patch position tokens for patches in a square image.

    For a given patch in an image we wish to generate a position index.
    To accomplish this we normalize and quantise the pixel indices 
    corresponding to the patch. During training we randomly sample from the
    quantised range while during evaluation we take the mean. 
    """
    # get image dimensions
    h, w, c = image.shape
    patches_per_dim = h // patch_size
    num_patches = patches_per_dim**2

    # get indices of patch intervals
    idx_vals = jnp.arange(0, h+patch_size, patch_size)
    idx_pairs, packed_shape = e.pack((idx_vals[:-1], idx_vals[1:]), 'idx *')
    row_idx = e.repeat(idx_pairs, 'row_idx row_interval -> (repeat row_idx) row_interval', repeat = patches_per_dim) # [patch_idx, row_interval]
    col_idx = e.repeat(idx_pairs, 'col_idx col_interval -> (col_idx repeat) col_interval', repeat = patches_per_dim) # [patch_idx, col_interval]
    patch_idx, ps = e.pack((row_idx, col_idx), 'patch_idx *') # [patch_idx, row_interval col_interval]

    def get_patch_position_encoding(idx, key, interval_length, num_tokens):
      """
      Generate position token for a given patch and dimension (row/col)
      """
      # normalize and quatise
      idx = jnp.floor((idx / interval_length) * (num_tokens-1))
      row_start_idx, row_stop_idx, col_start_idx, col_stop_idx = idx
      
      if train:
        # split the key
        key_1, key_2 = jax.random.split(key)
        # sample uniformly from the patch interval
        row_token = random.randint(key_1, shape=(1,), minval=row_start_idx, maxval=row_stop_idx)
        col_token = random.randint(key_2, shape=(1,), minval=col_start_idx, maxval=col_stop_idx)
      else:
        # use the center of the patch interval
        row_token = jnp.int32((row_start_idx + row_stop_idx) // 2)
        col_token = jnp.int32((col_start_idx + col_stop_idx) // 2)
        
      return row_token, col_token
      
    # generate tokens for patches in image
    if train:
      # generate random keys
      keys = random.split(key, num_patches) # key per patch

      # generate encodings (with sampling)
      row_tokens, col_tokens = jax.vmap(
          get_patch_position_encoding, 
          in_axes=(0, 0, None, None))(patch_idx, keys, h, num_tokens)

    else:
      # generate encodings (with sampling)
      row_tokens, col_tokens = jax.vmap(
          get_patch_position_encoding, 
          in_axes=(0, None, None, None))(patch_idx, None, h, num_tokens)

    return jnp.squeeze(row_tokens), jnp.squeeze(col_tokens)


############################
# Image Embedding
############################

# https://github.com/google/flax/blob/main/examples/imagenet/models.py
class ResNetV2Block(nn.Module):
    """
    Note: fixing parameter defaults to match Gato.
    """
    config: dict

    @nn.compact
    def __call__(self, x):
        # start with convolution projection
        x = nn.Conv(
                features=self.config.token_embedding.input_projection.features,
                kernel_size=self.config.token_embedding.input_projection.kernel_size,
                strides=self.config.token_embedding.input_projection.strides,
                padding=self.config.token_embedding.input_projection.padding,
                )(x)
        x = nn.GroupNorm()(x)
        x = nn.gelu(x)

        # resnetv2block
        residual = x

        y = nn.GroupNorm()(x)
        y = nn.gelu(y)
        y = nn.Conv(
                features=self.config.token_embedding.resnet_block.features,
                kernel_size=self.config.token_embedding.resnet_block.kernel_size,
                strides=self.config.token_embedding.resnet_block.strides,
                padding=self.config.token_embedding.resnet_block.padding,
                )(y)
        
        y = nn.GroupNorm()(y)
        y = nn.gelu(y)
        y = nn.Conv(
                features=self.config.token_embedding.resnet_block.features,
                kernel_size=self.config.token_embedding.resnet_block.kernel_size,
                strides=self.config.token_embedding.resnet_block.strides,
                padding=self.config.token_embedding.resnet_block.padding,
                )(y)
  
        out = y+residual
        
        # map to embedding dimension
        out = nn.GroupNorm()(out)
        out = nn.gelu(out)
        
        #flatten output
        out = jnp.reshape(out, (*out.shape[:2], -1))
        out = nn.Dense(features=self.config.embedding_dim)(out)
        
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

    def setup(self):
        self.image_size = self.config["image_size"]
        self.patch_size = self.config["patch_size"]
        self.position_interval = self.config["position_interval"]
        self.normalize = self.config["normalize"]
        self.embedding_function = ResNetV2Block(config=self.config)
        self.row_embeddings = nn.Embed(self.position_interval, self.config.embedding_dim)
        self.col_embeddings = nn.Embed(self.position_interval, self.config.embedding_dim)
        self.rng_collection = self.config["rng_collection"]

    def __call__(self, image, train=True):
        """
        Args:
            images (jax.numpy.ndarray): the images to be tokenized (num_batches, num_sequences, num_images, H, W, C).
        """
        # get dimensions
        batch_size, num_images, h, w, c = image.shape
        num_tokens = (h // self.patch_size) * (w // self.patch_size)

        # flatten batch and sequence dimensions for readability of vmapping
        image_flat = jnp.reshape(image, (-1, *image.shape[-3:]))
        
        # resize the image to the desired size
        if image_flat.shape[-3:] != self.image_size:
            print("image shape: ", image_flat.shape[-3:])
            print("image size: ", self.image_size)
            sys.exit("Input image is not the correct size.")

        # convert image into patches
        patches = jax.vmap(image_to_patches, in_axes=(0, None, None), out_axes=0)(image_flat, self.patch_size, self.normalize)
        
        chex.assert_equal(
                patches.shape[-3:],
                (
                    self.patch_size,
                    self.patch_size, 
                    c,
                )
                )

        # create patch embeddings
        patch_embeddings = self.embedding_function(patches)
        
        # TODO: add utility to check dimension and finite values with chex
        
        # create patch position embeddings
        key = self.make_rng(self.rng_collection)
        keys = jax.random.split(key, batch_size*num_images)
        row_positions, col_positions = jax.vmap(encode_patch_position, in_axes=(0, None, None, 0, None))(image_flat, self.patch_size, self.position_interval, keys, train)
        

        row_position_embeddings = self.row_embeddings(row_positions)
        col_position_embeddings = self.col_embeddings(col_positions)

        chex.assert_equal(
                row_position_embeddings.shape,
                patch_embeddings.shape
                )
        
        # add position embeddings to patch embeddings (broadcasting)
        patch_embeddings = patch_embeddings + row_position_embeddings + col_position_embeddings

        # reshape back to original shape
        patch_embeddings = jnp.reshape(patch_embeddings, (batch_size, num_images, num_tokens,-1))

        return patch_embeddings

