"""
Block attention implementation inspired by OCTO model.
"""

import abc
import re
from typing import List, Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops as e
import math

import multi_modal_transformers

class TokenSet(abc.ABC):
  """
  A class that encapsulates a set of tokens and their attention rule.

  Here attention rule refers to the logic for how a set of tokens attends
  to other tokens in a sequence of token sets.
  """

  def __init__(self, num_tokens, timestep, tokens_compressed_per_layer=0):
    self.num_tokens = num_tokens # number of tokens in the set
    self.timestep = timestep # subsequence timestep
    self.tokens_compressed_per_layer = tokens_compressed_per_layer # number of tokens compressed per layer
    self.modality_sequence_idx = None # the index of the modality in the sequence

  @abc.abstractmethod
  def intra_attention_rule(self):
    """
    Defines attention masking behaviour for tokens within the set.
    """
    raise NotImplemented

  @abc.abstractmethod
  def inter_attention_rule(self, token_set):
    """
    Defines attention making behaviour for tokens outside the set.
    """
    raise NotImplemented

  @abc.abstractmethod
  def attention_rule(self, token_sequence):
    """
    Defines overall attention mask for token set query.
    """
    raise NotImplemented


class Text(TokenSet):
  """
  A prefix token set for text descriptions.
  """

  def __init__(self, num_tokens, timestep):
    super().__init__(num_tokens, timestep)
    self.modality = "text"

  def inter_attention_rule(self, tokenset)->jax.typing.ArrayLike:
    """
    Text tokens attend causally to all past inputs by default.
    """
    if isinstance(tokenset, Readout): # do not attend to readout tokens
        return jnp.zeros((self.num_tokens, tokenset.num_tokens))
    elif tokenset.timestep <= self.timestep:
      return jnp.ones((self.num_tokens, tokenset.num_tokens))
    else:
      return jnp.zeros((self.num_tokens, tokenset.num_tokens))
    

  def intra_attention_rule(self)->jax.typing.ArrayLike:
    """
    By default text in token sets attend to tokens within the set
    in a causal fashion for the purpose of autoregressively generating
    text ouputs.
    """
    return jnp.squeeze(nn.make_causal_mask(jnp.zeros(self.num_tokens)))

  def attention_rule(self, token_sequence=List[TokenSet]):
    mask = []
    for tokenset in token_sequence:
      if (tokenset.timestep==self.timestep) and (isinstance(tokenset, self.__class__)):
        mask.append(self.intra_attention_rule())
      else:
        mask.append(self.inter_attention_rule(tokenset))
    return jnp.hstack(mask)


class TaskDescriptionPrefix(Text):
  """
  A task description prefix.
  """

  def __init__(self, num_tokens, timestep):
    super().__init__(num_tokens, timestep)
  

  def inter_attention_rule(self, tokenset):
    """
    Task description tokens do not attend to other observations.
    """
    return jnp.zeros((self.num_tokens, tokenset.num_tokens))

  def intra_attention_rule(self):
    """
    Tokens in the task description prefix attend to themselves
    """
    return jnp.ones((self.num_tokens, self.num_tokens))


class Image(TokenSet):
  """
  A token set for image observation.
  """
  def __init__(self, num_tokens, timestep):
    super().__init__(num_tokens, timestep)
    self.modality = "images"

  def inter_attention_rule(self, tokenset):
    """
    Image tokens attend causally to all past inputs by default.
    """
    if isinstance(tokenset, Readout): # do not attend to readout tokens
        return jnp.zeros((self.num_tokens, tokenset.num_tokens))
    elif tokenset.timestep <= self.timestep:
      return jnp.ones((self.num_tokens, tokenset.num_tokens))
    else:
      return jnp.zeros((self.num_tokens, tokenset.num_tokens))

  def intra_attention_rule(self)->jax.typing.ArrayLike:
    """
    Image patch tokens should attend to eachother. 
    """
    return jnp.ones((self.num_tokens, self.num_tokens))

  def attention_rule(self, token_sequence=List[TokenSet]):
    mask = []
    for tokenset in token_sequence:
      if (tokenset.timestep==self.timestep) and (isinstance(tokenset, self.__class__)):
        mask.append(self.intra_attention_rule())
      else:
        mask.append(self.inter_attention_rule(tokenset))
    return jnp.hstack(mask)


class Readout(TokenSet):
  """
  A token set for readout tokens.
  """
  def __init__(self, num_tokens, timestep):
    super().__init__(num_tokens, timestep)
    self.modality = "readouts"

  def inter_attention_rule(self, tokenset):
    """
    Readout attends to all past tokens except other readout tokens.
    """
    if isinstance(tokenset, self.__class__): # do not attend to readout tokens
        return jnp.zeros((self.num_tokens, tokenset.num_tokens))
    elif tokenset.timestep <= self.timestep: # attend causally to previous tokens
      return jnp.ones((self.num_tokens, tokenset.num_tokens))
    else: 
      return jnp.zeros((self.num_tokens, tokenset.num_tokens))

  def intra_attention_rule(self)->jax.typing.ArrayLike:
    """
    Readout tokens attend to themselves. 
    """
    return jnp.ones((self.num_tokens, self.num_tokens))

  def attention_rule(self, token_sequence=List[TokenSet]):
    mask = []
    for tokenset in token_sequence:
      if (tokenset.timestep==self.timestep) and (isinstance(tokenset, self.__class__)):
        mask.append(self.intra_attention_rule())
      else:
        mask.append(self.inter_attention_rule(tokenset))
    return jnp.hstack(mask)


class TokenSequence:
  """
  A class which encapsulates a particular token sequence.
  """

  def __init__(self, token_sequence: str, token_compression_sequence: str = None):
    self.token_sequence_str = token_sequence
    self.token_compression_sequence_str = token_compression_sequence
    self.token_sequence = self._parse()
    self.slice_idx = self._generate_embedding_slices()
    self.tokenset_slices = self._generate_embedding_subsets()
    self.assemble_embeddings = jax.jit(self._assemble_embeddings, static_argnames=["slice_idx"])

  def _parse(self, layer=0):
    """
    Parse the string representation of the token sequence.
    """
    # TODO: add assert statements to ensure correct representation syntax

    # parse seq string into timesteps blocks of tokens
    block = re.findall(r'\[(.*?)\]', self.token_sequence_str)

    # parse string into how many times to repeat a given timestep block
    pattern = r'(?<=\])(.*?)(?=\[|$)'
    repeat_matches = re.findall(pattern, self.token_sequence_str)
    repeats = []
    for repeat in repeat_matches:
      if repeat.strip() == '':
        repeats.append(1)
      else:
        pattern = r'\*(\d+)'
        repeats.append(int(re.findall(pattern, repeat)[0]))
    

    
    # parse compressed token sequence
    if self.token_compression_sequence_str is not None:
        compressed_block = re.findall(r'\[(.*?)\]', self.token_compression_sequence_str)
        parsed_iter = zip(block, compressed_block, repeats)
        
        sequence = []
        seq_timestep = 0
        for block_idx, (block, compressed_block, repeat) in enumerate(parsed_iter):
          token_groups = re.split(r';', block)
          compressed_token_groups = re.split(r';', compressed_block)
          for _ in range(repeat):
            for token_group, compressed_token_group in zip(token_groups, compressed_token_groups):
              token_group_name = re.search(r'^(.*?)\{', token_group).group(1)
              num_tokens = int(re.search(r'\d+', token_group).group())
              num_compressed_tokens = int(re.search(r'\d+', compressed_token_group).group())
              num_tokens = num_tokens - (layer * num_compressed_tokens)
              sequence.append(globals()[token_group_name](num_tokens, seq_timestep))
            seq_timestep += 1 # add timestep index for each block in seq
    
    else:
        parsed_iter = zip(block, repeats) # zip blocks and repeats for assembling sequence
        sequence = []
        seq_timestep = 0
        for block_idx, (block, repeat) in enumerate(parsed_iter):
          token_groups = re.split(r';', block)
          for _ in range(repeat):
            for token_group in token_groups:
              token_group_name = re.search(r'^(.*?)\{', token_group).group(1)
              num_tokens = int(re.search(r'\d+', token_group).group())
              sequence.append(globals()[token_group_name](num_tokens, seq_timestep))
            seq_timestep += 1 # add timestep index for each block in seq
    
    return sequence
    
  def _assemble_embeddings(self, embeddings, slice_idx):
    """
    This method accepts embeddings and assembles them into the TokenSequence
    based on the Token representation
    """
    embedding_seq = []
    for slice_id, token_group in zip(slice_idx, self.token_sequence):
        embedding_seq.append(jax.lax.dynamic_slice_in_dim(
          getattr(embeddings, token_group.modality),
          slice_id[0],
          slice_id[1],
          axis=1, # sequence dimension
          ))

    return jnp.concatenate(embedding_seq, axis=1)


  def _generate_embedding_slices(self):
    """
    Generate dynamic slices for assembling token modalities into a sequence.
    """
    modality_idx = {
            "images": 0,
            "text": 0,
            "readouts": 0,
            } 
    slice_idx = []

    for token_group in self.token_sequence:
        start_idx = modality_idx[token_group.modality]
        slice_idx.append(
                tuple([
                    start_idx, 
                    token_group.num_tokens,
                    ])
                )
        modality_idx[token_group.modality] = start_idx + token_group.num_tokens
    
    return iter(slice_idx)
  
  def _generate_embedding_subsets(self):
    """
    Generate indices for distinct tokensets in sequence.
    """
    slice_idx = []
    curr_idx = 0
    for token_group in self.token_sequence:
        start_idx = curr_idx
        slice_idx.append(
                tuple([
                    start_idx, 
                    token_group.num_tokens,
                    ])
                )
        curr_idx += token_group.num_tokens 

    return iter(slice_idx)

  def generate_attention_mask(self, repeats = 1, layer=None):
    """
    This method generates an attention mask for the given sequence.
    """ 
    token_sequence = self._parse(layer=layer)
    attention_mask = jnp.vstack([token_group.attention_rule(self.token_sequence) for token_group in token_sequence])
    attention_mask = jnp.asarray(attention_mask, dtype=bool)

    return e.repeat(attention_mask, "q k -> repeats q k", repeats=repeats)

  def get_modality_idx(self, modality):
      """
      Return the indices in the sequence corresponding to tokens of a given modality.
      """
      curr_idx = 0
      idx = []
      for token_group in self.token_sequence:
          if token_group.modality==modality:
              stop_idx = curr_idx + token_group.num_tokens
              idx.append(jnp.arange(curr_idx, stop_idx))
          curr_idx += token_group.num_tokens
      return jnp.ravel(jnp.array(idx))

  def generate_layer_token_sequence(self, layer):
    """
    Generate the token sequence.
    """
    return self._parse(layer=layer)

@flax.struct.dataclass
class TokenEmbeddings:
    text: jax.Array = jnp.array([])
    images: jax.Array = jnp.array([])
    readouts: jax.Array = jnp.array([])

if __name__=="__main__":
    # basic tests of token sequence
    multi_modal_seq = "[TaskDescriptionPrefix{20}] [Image{10};Readout{10}]*2"
    multi_modal_compressed_seq = "[TaskDescriptionPrefix{0}] [Image{2};Readout{0}]*2"
    seq = TokenSequence(multi_modal_seq, multi_modal_compressed_seq)
    attention_mask = seq.generate_attention_mask(layer=0)
    print(attention_mask.shape)
    print(attention_mask)
    
    print(seq.get_modality_idx("readouts"))

    # assemble text embeddings
    #dummy_embeddings = jnp.vstack([jnp.ones(10) * i for i in range(8)])
    #dummy_embeddings = TokenEmbeddings(dummy_embeddings, Text)
    #seq_embeddings = seq.assemble_embeddings([dummy_embeddings])
