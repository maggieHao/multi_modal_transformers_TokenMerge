"""Implementation of tokenizer for discrete/continuous values."""

import dataclasses

import chex
import jax.numpy as jnp

def mu_law_encoder(x, mu=255):
    return jnp.sign(x) * jnp.log(1 + mu * jnp.abs(x)) / jnp.log(1 + mu)

@dataclasses.dataclass
class DiscreteValueTokenizeOp:
    def __call__(self, x):
        chex.assert_type(x, jnp.array)
        chex.assert_rank(x, 1)
        return x

@dataclasses.dataclass
class ContinuousValueTokenizeOp:
    def __call__(self, x, mu_encode=True):
        if mu_encode:
            return mu_law_encoder(x)
        return x

if __name__ == "__main__":
    print("Testing tensor tokenizer...")
