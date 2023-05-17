"""
Implements offset attention 

source: https://arxiv.org/pdf/2012.09688.pdf
"""

from typing import Optional, Any

import flax
import flax.linen as nn
from flax.linen.attention import SelfAttention
import jax.numpy as jnp


class OffsetAttention(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, inputs_q: Any, mask: Optional[Any] = None,
                deterministic: Optional[bool] = None, is_training: bool = False):
        self_attention_output = nn.SelfAttention(
                    num_heads = self.config["num_heads"], 
                    qkv_features = self.config["qkv_features"], 
                    out_features = self.config["out_features"],
                )(inputs_q, mask=mask, deterministic=deterministic)
        offset = inputs_q - self_attention_output
        
        x = nn.Dense(self.config["embed_dim"])(offset)
        x = nn.BatchNorm(use_running_average=not is_training)(x)
        x = nn.relu(x)

        return x + inputs_q

