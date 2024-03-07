import jax
import jax.numpy as jnp
from flax import linen as nn

from transformers import FlaxT5EncoderModel, AutoTokenizer, AutoConfig


class T5Tokenizer(nn.Module):

    def setup(self):
        self.config = AutoConfig.from_pretrained('t5-base')
        self.model = FlaxT5EncoderModel(self.config).module

    def __call__(self, input_ids):
        embeddings = self.model(input_ids).last_hidden_state
        embeddings = jax.lax.stop_gradient(embeddings)
        return embeddings

if __name__ == "__main__":
     
    instructions = [
            "Pick up the red block and put it on the green block",
            "Pick up the green block and put it on the red block",
            ]

    # tokenize instructions
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    inputs = tokenizer(instructions, return_tensors="jax", padding=True, truncation=True)
    
    # encode instructions
    T5Tokenizer = T5Tokenizer()
    params = T5Tokenizer.init(jax.random.PRNGKey(0), inputs['input_ids'])
    outputs = T5Tokenizer.apply(params, inputs['input_ids'])
