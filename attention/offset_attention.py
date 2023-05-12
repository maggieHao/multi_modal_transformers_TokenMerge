"""
Implements offset attention 

source: https://arxiv.org/pdf/2012.09688.pdf
"""

import flax
import flax.linen as nn
from flax.linen.attention import SelfAttention

class OffsetAttention(SelfAttention):
    """
    Implement multi-head attention using the offset mechanism
    """

    @nn.compact
    def __call__(self, inputs_q: Array, mask: Optional[Array] = None, 
                 deterministic: Optional[bool] = None):
        """
        """
        self_attention_output = super().__call__(inputs_q, inputs_q, mask=mask, 
                deterministic=deterministic)
        offset = inputs_q - self_attention_output

        x = nn.Linear(offset)
        x = nn.BatchNorm(x)
        x = nn.ReLU(x)

        return x + inputs_q

