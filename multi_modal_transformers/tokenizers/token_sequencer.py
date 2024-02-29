import abc
from typing import List, Tuple

import flax

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
    num_tokens = num_tokens # number of tokens in the set
    timestep = timestep # subsequence timestep
    modality_sequence_idx = None # the index of the modality in the sequence

  @abc.abstract_method
  def intra_attention_rule(self):
    """
    Defines attention masking behaviour for tokens within the set.
    """
    raise NotImplemented

  @abc.abstract_method
  def inter_attention_rule(self, token_set: TokenSet):
    """
    Defines attention making behaviour for other token sets
    """
    raise NotImplemented

  @abc.abstract_method
  def attention_rule(self, token_sequence = List[TokenSet]):
    """
    Assigns the set of nodes which a the given set will pay attention to in the
    token sequence.
    """
    raise NotImplemented


class TokenSequence():
  """
  A class which encapsulates a particular token sequence.
  """

  def __init__(self, token_sequence: str):
    self.token_sequence_str = token_sequence
    self._token_sequence = self._parse()
    self.num_blocks

  def _parse(self):
    """
    Parse the string representation of the token sequence.
    """

  def assemble_embeddings(self, embeddings:List(Tuple(TokenSet, Array))):
    """
    This method accepts a embeddings and assembles them into the TokenSequence
    based on the Token representation
    """


  def generate_attention_mask(token_sequence: str):
    """
    This method generates an attention mask for the given sequence.
    """

  @abc.abstract_method
  def applying_pruning(self):
    raise NotImplemented 

  @abc.abstract_method
  def apply_merging(self):
    raise NotImplemented



# definitions of token sets
class Text(TokenSet):
  """
  A prefix token set for text descriptions. 
  """

  def __init__(self, num_tokens, timestep):
    super.__init__(num_tokens, timestep)

  def attention_rule(self, token_sequence=List[TokenClass]):
    """
    - 
    """
    # intra token attention rule (it should pay attention to itself)

    # inter token attention rule

class PrefixText(Text):

class ClarificationText(Text):

class Image(TokenSet):
  """
  A token set for image observation.
  """


# test attention mask generation




# construct a sequence as a list of these basic set types

# generate attention masks through applying mask rules for each given class
