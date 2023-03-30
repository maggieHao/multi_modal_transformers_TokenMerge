"""Memory-Augmented Transformer Model."""

# custom imports
from MART.model.transformers.layers import (
        MLPBlock,
        EncoderBlock,
        DecoderBlock
        )

# deep-learning frameworks
import flax.linen as nn

class MemoryAugmentedEncoderBlock(nn.Module):

class MemoryAugmentedDecoderBlock(nn.Module):

class MemoryAugmentedTransformer(nn.Module):
    config: dict

    def setup(self):


    def memory_matrix(self, ):
        pass

    @nn.compact
    def __call__(self):
        pass
