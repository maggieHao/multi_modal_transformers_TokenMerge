import abc
import re
from typing import List, Tuple

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

# Idea: wish to dynamically generate attention masks based on token class
# assignment and specific rules

# tokens are often bunched into sets of a given class start through creating a
# class that enapsulates a given set type

class TokenSet(abc.ABC):
  """
  A class that encapsulates a set of tokens and their attention rule.

  Here attention rule refers to the logic for how a set of tokens attends
  to other tokens in a sequence of token sets.
  """

  def __init__(self, num_tokens, timestep):
    self.num_tokens = num_tokens # number of tokens in the set
    self.timestep = timestep # subsequence timestep
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


# definitions of token sets
class Text(TokenSet):
  """
  A prefix token set for text descriptions.
  """

  def __init__(self, num_tokens, timestep):
    super().__init__(num_tokens, timestep)
 
  def inter_attention_rule(self, tokenset):
    """
    Text tokens attend causally to all past inputs by default.
    """
    if tokenset.timestep < self.timestep:
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
    super.__init__(num_tokens, timestep)

  def attention_rule(self, token_sequence=List[TokenSet]):
    raise NotImplemented


class TokenSequence:
  """
  A class which encapsulates a particular token sequence.
  """

  def __init__(self, token_sequence: str):
    self.token_sequence_str = token_sequence
    self.token_sequence = self._parse()

  def _parse(self):
    """
    Parse the string representation of the token sequence.
    """
    # TODO: run assert statements to ensure correct representation syntax

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

    # zip blocks and repeats for assembling sequence
    parsed_iter = zip(block, repeats)

    # assemble the sequence
    sequence = []
    for timestep, (block, repeat) in enumerate(parsed_iter):
      token_groups = re.split(r';', block)
      for _ in range(repeat):
        for token_group in token_groups:
          token_group_name = re.search(r'^(.*?)\{', token_group).group(1)
          num_tokens = int(re.search(r'\d+', token_group).group())
          sequence.append(globals()[token_group_name](num_tokens, timestep))
          #sequence.append(getattr(module_name, token_group_name)(num_tokens, timestep)
    
    return sequence

  def assemble_embeddings(self, embeddings: List[TokenSet]):
    """
    This method accepts embeddings and assembles them into the TokenSequence
    based on the Token representation
    """
    embedding_seq = []  
    for token_group in self.token_sequence:
      for embedding_idx, embedding in enumerate(embeddings):
        if isinstance(token_group, embedding.__embedding_type__()):
          embedding_seq.append(jax.lax.dynamic_slice(
              embedding.embeddings, 
              jnp.zeros(2, dtype=int),
              jnp.array([int(token_group.num_tokens), embedding.embeddings.shape[-1]])))
          embeddings[embedding_idx].embeddings = jax.lax.dynamic_slice(
                                                  embedding.embeddings, 
                                                  jnp.array([int(token_group.num_tokens), 0]),
                                                  jnp.array([embedding.embeddings.shape[0] - int(token_group.num_tokens), embedding.embeddings.shape[1]]))

    return jnp.concatenate(embedding_seq)


  def generate_attention_mask(self):
    """
    This method generates an attention mask for the given sequence.
    """ 
    return jnp.vstack([token_group.attention_rule(self.token_sequence) for token_group in self.token_sequence])


  @abc.abstractmethod
  def applying_pruning(self):
    raise NotImplemented

  @abc.abstractmethod
  def apply_merging(self):
    raise NotImplemented

class TokenEmbeddings:

  def __init__(self, embeddings: jax.typing.ArrayLike, embedding_type: TokenSet):
    self.embeddings = embeddings
    self.embedding_type = embedding_type

  def __embedding_type__(self):
    return self.embedding_type

