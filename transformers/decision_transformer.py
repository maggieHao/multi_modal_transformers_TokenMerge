"""A decoder-only transformer model for sequence generation."""

from MART.model.layers import DecoderBlock

import flax.linen as nn


class DecisionTransformer(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, encoder_outputs, decoder_inputs):
        x = decoder_inputs
        for lyr in range(self.config["decoder"]["num_blocks"]):
            x = DecoderBlock(self.config["decoder"]["decoder_block"])(
                encoder_outputs, x
            )
        x = nn.LayerNorm()(x)
        logits = nn.Dense(
            features=self.config["decoder"]["vocab_size"],
            use_bias=True,
            kernel_init=nn.initializers.normal(),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        return logits
