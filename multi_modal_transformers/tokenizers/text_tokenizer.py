"""Implementing text tokenizer."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple

import os
import sentencepiece as spm

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
import numpy as np
import einops as e

# multi-processing
import multiprocessing as mp
from multiprocessing import Pool


# Generate Corpus/Vocab

def generate_move_puzzle_corpus():
    """Generating move puzzle corpus."""
    colors = ["red", "green", "blue", "yellow", "cyan", "orange", "pink", "brown", "grey"]
    shapes = ["square", "rectangle", "L", "S", "T"]
    directions = ["left", "right", "up", "down"]
    os.makedirs("corpus", exist_ok=True)
    # generate corpus
    with open("corpus/move_puzzle.txt", "w") as f:
        for color in colors:
            for shape in shapes:
                for direction in directions:
                    f.write(f"move {color} {shape} {direction} \n")
    
    # generate vocab
    with open("corpus/move_puzzle_vocab.txt", "w") as f:
        for color in colors:
            f.write(f"{color} \n")
        for shape in shapes:
            f.write(f"{shape} \n")
        for direction in directions:
            f.write(f"{direction} \n")

        f.write("move \n")

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
        self.word2idx = {word: idx for idx, word in enumerate(set(vocab))}
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text):
        """Tokenize text."""
        # convert each token to index
        return np.array([self.word2idx[token] for token in text])


# Sentence Piece Tokenizer 

def train_sentencepiece_model(input_file, model_prefix, vocab_size):
    """Training sentencepiece model."""
    if "/" in model_prefix:
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="unigram",
            )

# Text Embedding

class BasicTextTokenizer(nn.Module):
    """
    Text embedding module.
    """
    config: dict

    def setup(self):
        self.embedding = nn.Embed(
                num_embeddings=self.config["vocab_size"],
                features=self.config["embedding_dim"],
                )
        
        self.position_embedding = self.param(
                'text_pos_embedding', 
                nn.initializers.normal(stddev=0.02), 
                (self.config["max_text_len"],
                self.config["embedding_dim"])
                )

        #self.position_embedding = nn.Embed(
        #        num_embeddings=self.config["max_text_len"],
        #        features=self.config["embedding_dim"],
        #        )

    def __call__(self, tokens):
        # embed text
        word_embeddings = self.embedding(tokens)
        
        # add position embedding
        positions = jnp.arange(tokens.shape[1])
        positions = e.repeat(positions, "pos -> batch pos", batch=tokens.shape[0])

        return word_embeddings + self.position_embedding


if __name__=="__main__":
    # generate corpus
    generate_move_puzzle_corpus()

    # train sentencepiece model
    train_sentencepiece_model("corpus/move_puzzle.txt", "spm_files/move_puzzle", 30)

    # test encoding
    print(spm.SentencePieceProcessor(model_file="spm_files/move_puzzle.model").encode("Move red square left"))
