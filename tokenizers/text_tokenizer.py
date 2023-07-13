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

# multi-processing
import multiprocessing as mp
from multiprocessing import Pool


############################
# Generate Corpus/Vocab
############################

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

############################
# Basic Tokenizer
############################

def _tokenize(text, word2idx):
    """Tokenize text."""
    # split by space
    tokens = text.decode().split(" ")
    # convert each token to index
    return jnp.array([word2idx[token] for token in tokens])

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


    def _tokenize(self, text):
        """Tokenize text."""
        # split by space
        tokens = text.decode().split(" ")
        # convert each token to index
        return jnp.array([self.word2idx[token] for token in tokens])
    
    def tokenize(self, text):
        """Tokenize text."""
        # for each text in list of texts
        # in parallel decode text to indices
        # use multi-processing
        with Pool(mp.cpu_count()) as p:
            indices = p.map(self._tokenize, text)
        return indices
        


@dataclasses.dataclass
class BasicTextTokenizeOp:
    tokenizer: BasicTokenizer

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

############################
# Sentence Piece Tokenizer 
############################

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

@dataclasses.dataclass
class SPTextTokenizeOp:
    tokenizer: spm.SentencePieceProcessor

    def __call__(self, text):
        return self.tokenizer.encode(text)


############################
# Text Embedding
############################

class BasicTextTokenizer(nn.Module):
    """
    Text embedding module.
    """

    config: dict
    tokenizer: BasicTokenizer

    def setup(self):
        self.embedding = nn.Embed(
                num_embeddings=self.tokenizer.vocab_size,
                features=self.config["embedding_dim"],
                )

    def __call__(self, text):
        
        # convert text to indices
        text = self.tokenizer.tokenize(text)
        text = jnp.stack(text)
        word_embeddings = self.embedding(text)

        return word_embeddings


if __name__=="__main__":
    # generate corpus
    generate_move_puzzle_corpus()

    # train sentencepiece model
    train_sentencepiece_model("corpus/move_puzzle.txt", "spm_files/move_puzzle", 30)

    # test encoding
    print(spm.SentencePieceProcessor(model_file="spm_files/move_puzzle.model").encode("Move red square left"))
