"""Implementing text tokenizer."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple

import os
import sentencepiece as spm

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from jax import random
import numpy as np
import einops as e

from hydra.utils import instantiate, call

# multi-processing
import multiprocessing as mp
from multiprocessing import Pool


# Basic Tokenizer
class BasicTokenizer:
    """
    Basic tokenizer.
    """

    def __init__(self, vocab_dir):
        # read vocab text file to list
        with open(vocab_dir, "r") as f:
            vocab = f.read().split("\n")
            vocab = [word.strip() for word in vocab if word != ""]

        # create a dictionary mapping unique words to indices
        self.word2idx = {word: idx+1 for idx, word in enumerate(list(sorted(set(vocab))))}
        self.word2idx["pad"] = 0
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text):
        """Tokenize text."""
        # convert each token to index
        return np.array([self.word2idx[token] for token in text])

# Text Embedding
class BasicTextTokenizer(nn.Module):
    """
    Text embedding module.
    """
    config: dict

    def setup(self):
        
        self.embedding = instantiate(self.config["text_embedding"])
        self.position_embedding = instantiate(self.config["text_position_embedding"])

    def __call__(self, tokens):
        # embed text
        word_embeddings = self.embedding(tokens)
        
        # add position embedding
        positions = jnp.arange(tokens.shape[1])
        positions = e.repeat(positions, "pos -> batch pos", batch=tokens.shape[0])
        position_embeddings = self.position_embedding(positions)

        return word_embeddings + position_embeddings

