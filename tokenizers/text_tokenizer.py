"""Implementing text tokenizer."""

import dataclasses

import os
import sentencepiece as spm

def generate_move_puzzle_corpus():
    """Generating move puzzle corpus."""
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    shapes = ["square", "rectangle", "L", "S", "T"]
    directions = ["left", "right", "up", "down"]
    os.makedirs("corpus", exist_ok=True)
    with open("corpus/move_puzzle.txt", "w") as f:
        for color in colors:
            for shape in shapes:
                for direction in directions:
                    f.write(f"Move {color} {shape} {direction} \n")

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
class TextTokenizeOp:
    tokenizer: spm.SentencePieceProcessor

    def __call__(self, text):
        return self.tokenizer.encode(text)

if __name__=="__main__":
    # such a tokenizer may not even be necessary with such a small vocab

    # generate corpus
    generate_move_puzzle_corpus()

    # train sentencepiece model
    train_sentencepiece_model("corpus/move_puzzle.txt", "spm_files/move_puzzle", 29)

    # test encoding
    print(spm.SentencePieceProcessor(model_file="spm_files/move_puzzle.model").encode("Move red square left"))
