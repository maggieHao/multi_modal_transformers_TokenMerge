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
from flax.linen import initializers
from jax import random


from hydra.utils import call, instantiate
# OmegeConf
from omegaconf import OmegaConf, DictConfig

# import custom utils for logging
#from utils.logger import get_logger
#LOG = get_logger(__name__)

ModuleDef = Any

############################
# Image Preprocessing
############################

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
        patches = (2*(patches/255.0)) - 1.0
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
    num_blocks: int
    # input projection layer
    input_conv: DictConfig
    input_pool: DictConfig
    # resnet blocks
    resnet_norm: DictConfig
    resnet_activation: DictConfig
    resnet_conv: DictConfig
    # output_layer
    output_dense: DictConfig

    @nn.compact
    def __call__(self, x):
        # start with convolution projection
        x = instantiate(self.input_conv)(x)
        x = call(self.input_pool)(x)

        # resnetv2block
        residual = x
        
        for _ in range(self.num_blocks):
            x = instantiate(self.resnet_norm)(x)
            x = call(self.resnet_activation)(x)
            x = instantiate(self.resnet_conv)(x)

        if residual.shape != x.shape:
            residual = instantiate(self.resnet_conv)(residual)
  
        x = x+residual
        
        #flatten output
        x = jnp.reshape(x, (*x.shape[:3], -1))
        x = instantiate(self.output_dense)(x)

        return x

class ResNetV2Block_(nn.Module):
    """
    Note: fixing parameter defaults to match Gato.
    """
    config: DictConfig

    @nn.compact
    def __call__(self, x):
        # start with convolution projection
        x = instantiate(self.config.input_projection.conv)(x)
        x = call(self.config.input_projection.pool)(x)

        # resnetv2block
        residual = x
        
        for _ in range(self.config.num_blocks):
            x = instantiate(self.config.resnet_block.norm)(x)
            x = call(self.config.resnet_block.activation)(x)
            x = instantiate(self.config.resnet_block.conv)(x)

        if residual.shape != x.shape:
            residual = instantiate(self.config.resnet_block.conv)(residual)
  
        x = x+residual
        
        #flatten output
        x = jnp.reshape(x, (*x.shape[:2], -1))
        x = instantiate(self.config.output_projection.dense)(x)

        return x


########################
# Image Tokenizer
########################

class ImageTokenizer(nn.Module):
    """
    Converts images into tokens.
    """
    image_size: list
    patch_size: int
    normalize: bool 
    position_interval: int
    rng_collection: str
    embedding_dim: int
    # position embeddings
    row_position_embedding: DictConfig
    col_position_embedding: DictConfig
    # resnet block
    resnet: DictConfig

    def setup(self):
        self.embedding_function = instantiate(resnet)
        self.row_embeddings = instantiate(row_position_embedding)
        self.col_embeddings = instantiate(col_position_embedding)

    def __call__(self, image, train=True):
        """
        Args:
            images (jax.numpy.ndarray): the images to be tokenized (num_batches, num_images, H, W, C).
        """
        # get dimensions
        batch_size, num_images, h, w, c = image.shape
        num_tokens = (h // self.patch_size) * (w // self.patch_size)

        
        # exit if image is not the correct size
        if image.shape[-3:] != self.image_size:
            print(image.shape[-3:])
            print(self.image_size)
            sys.exit("Input image is not the correct size.")

        # convert image into patches
        patches = jax.vmap(
                jax.vmap(
                    image_to_patches, 
                    in_axes=(0, None, None), 
                    out_axes=0
                    ), 
                in_axes=(0, None, None), 
                out_axes=0)(
                        image, 
                        self.patch_size, 
                        self.normalize
                        )

        # create position encodings
        if train:
            key = self.make_rng(self.rng_collection)
            keys = jax.random.split(key, (batch_size,num_images))
            row_position_encoding, col_position_encoding = jax.vmap(
                    jax.vmap(
                        encode_patch_position, 
                        in_axes=(0, 0, None, None, None), 
                        out_axes=0),
                    in_axes=(0, 0, None, None, None),
                    out_axes=0)(
                            image,
                            keys,
                            self.patch_size, 
                            self.position_interval,
                            train
                            )
        else:
            keys = None
            row_position_encoding, col_position_encoding = jax.vmap(
                    jax.vmap(
                        encode_patch_position, 
                        in_axes=(0, None, None, None, None), 
                        out_axes=0),
                    in_axes=(0, None, None, None, None),
                    out_axes=0,
                    )(
                image,
                keys,
                self.patch_size, 
                self.position_interval,
                train
                )

        # create position embeddings 
        row_position_embeddings = self.row_embeddings(row_position_encoding)
        col_position_embeddings = self.col_embeddings(col_position_encoding)

        # create patch embeddings
        patch_embeddings = self.embedding_function(patches)

        # add position embeddings to patch embeddings (broadcasting)
        patch_embeddings = patch_embeddings + row_position_embeddings + col_position_embeddings

        return patch_embeddings


# for now write separate tokenizer for single images
class SingleImageTokenizer(nn.Module):
    """
    Converts images into tokens.
    """

    config: dict
    
    # TODO: move to @nn.compact
    def setup(self):
        self.image_size = self.config["image_size"]
        self.patch_size = self.config["patch_size"]
        self.position_interval = self.config["position_interval"]
        self.normalize = self.config["normalize"]
        self.embedding_function = ResNetV2Block_(config=self.config["resnet"]) # consider refactoring to use instantiate
        self.row_embeddings = instantiate(self.config["row_position_embedding"])
        self.col_embeddings = instantiate(self.config["col_position_embedding"])
        self.rng_collection = self.config["rng_collection"]

    def __call__(self, image, train=True):
        """
        Args:
            images (jax.numpy.ndarray): the images to be tokenized (num_batches, num_images, H, W, C).
        """
        # get dimensions
        batch_size, h, w, c = image.shape
        num_tokens = (h // self.patch_size) * (w // self.patch_size)

        
        # exit if image is not the correct size
        if image.shape[-3:] != self.image_size:
            print(image.shape[-3:])
            print(self.image_size)
            sys.exit("Input image is not the correct size.")

        # convert image into patches
        patches = jax.vmap(
                    image_to_patches, 
                    in_axes=(0, None, None), 
                    out_axes=0
                    )(
                    image, 
                    self.patch_size, 
                    self.normalize
                    )


        # create position encodings
        if train:
            key = self.make_rng(self.rng_collection)
            keys = jax.random.split(key, (batch_size,))
            row_position_encoding, col_position_encoding = jax.vmap(
                        encode_patch_position, 
                        in_axes=(0, 0, None, None, None), 
                        out_axes=0)(
                            image,
                            keys,
                            self.patch_size, 
                            self.position_interval,
                            train
                            )
        else:
            keys = None
            row_position_encoding, col_position_encoding = jax.vmap(
                        encode_patch_position, 
                        in_axes=(0, None, None, None, None), 
                        out_axes=0)(
                image,
                keys,
                self.patch_size, 
                self.position_interval,
                train
                )
        
        # create position embeddings 
        row_position_embeddings = self.row_embeddings(row_position_encoding)
        col_position_embeddings = self.col_embeddings(col_position_encoding)
        

        # create patch embeddings
        patch_embeddings = self.embedding_function(patches)

        # add position embeddings to patch embeddings (broadcasting)
        patch_embeddings = patch_embeddings + row_position_embeddings + col_position_embeddings
        
        return patch_embeddings

