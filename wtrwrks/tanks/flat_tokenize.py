"""FlatTokenize tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import nltk
import tinysegmenter


class FlatTokenize(ta.Tank):
  """The defintion of the FlatTokenize tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'flat_tokenize'
  slot_keys = ['strings', 'tokenizer', 'detokenizer', 'ids']
  tube_keys = ['target', 'tokenizer', 'detokenizer', 'diff', 'shape', 'ids']
  pass_through_keys = ['tokenizer', 'detokenizer', 'ids']

  def _pour(self, strings, ids, tokenizer, detokenizer=lambda a: ' '.join(a)):
    """Execute the FlatTokenize tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of strings
      The array of strings to tokenize.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    ids: np.ndarray
      An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings since all information about axis is lost when flattened. Each id from ids must be unique.The array of is the same shape as strings

    Returns
    -------
    dict(
      target: np.ndarray
        A one dimensional array of tokens.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      diff: np.ndarray of strings
        The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
      shape: list of ints
        The shape of the inputted array.
      ids: np.ndarray
        An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings. The array of is the same shape as target
    )

    """
    strings = np.array(strings)
    # Guard for the empty array case
    if not strings.size:
      return {'target': ut.maybe_copy(strings), 'diff': ut.maybe_copy(strings), 'tokenizer': tokenizer, 'detokenizer': detokenizer}

    all_tokens = []
    all_diffs = []

    # Go through each element of the string array, and it's corresponding id.
    r_ids = []
    for string_id, string in zip(ids.flatten(), strings.flatten()):
      # Tokenize the string and add it to the long list of all the tokens.
      tokens = tokenizer(string)
      all_tokens.extend(tokens)

      # Copy the string id len(tokens) times so that the ids always have
      # the same length as the tokens. This makes it more suitable for breaking
      # up in downstream tanks.
      r_ids.extend([string_id] * len(tokens))

      # Find the string diff after detokenizing the tokens.
      processed = detokenizer(tokens)
      diff = di.get_diff_string(processed, string)

      # Copy the diff len(tokens) times so that it always has the same size
      # as tokens. This makes it more suitable for breaking up in downstream
      # tanks.
      all_diffs.extend([diff] * len(tokens))

    target = np.array(all_tokens).astype(strings.dtype)
    diff = np.array(all_diffs).astype(strings.dtype)
    r_ids = np.array(r_ids)

    return {'target': target, 'diff': diff, 'tokenizer': tokenizer, 'detokenizer': detokenizer, 'ids': r_ids, 'shape': strings.shape}

  def _pump(self, target, diff, tokenizer, detokenizer, ids, shape):
    """Execute the FlatTokenize tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      A one dimensional array of tokens.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
    shape: list of ints
      The shape of the inputted array.
    ids: np.ndarray
      An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings. The array of is the same shape as target

    Returns
    -------
    dict(
      strings: np.ndarray of strings
        The array of strings to tokenize.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      ids: np.ndarray
        An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings since all information about axis is lost when flattened. Each id from ids must be unique.The array of is the same shape as strings
    )

    """
    # Handle the empty array case
    if not ids.size:
      return {'strings': target.reshape([0] + list(shape[1:])), 'tokenizer': tokenizer, 'detokenizer': detokenizer}

    # Get all the unique ids, and the indices which point to that unique value
    # value. Sort them, so that the indices are pulled out in sequential order.
    unique_ids, indices = np.unique(ids, return_index=True)
    indices = np.sort(indices)

    all_strings = []
    r_ids = []
    for index in indices:

      # Pull out all the tokens belonging to a particular id.
      string_id = ids[index]
      mask = (ids == string_id)
      tokens = target[mask].flatten()

      # Pull out the corresponding diff_string. Since it was duplicated many
      # times we only need to pull out the first one.
      diff_string = diff[mask][0]

      # Detokenize the tokens and reconstruct the orignal string from the
      # diff_string

      string = detokenizer(tokens)
      string = di.reconstruct(string, diff_string)

      r_ids.append(string_id)
      all_strings.append(string)
    r_ids = np.reshape(r_ids, shape)
    strings = np.reshape(all_strings, shape)
    return {'strings': strings, 'tokenizer': tokenizer, 'detokenizer': detokenizer, 'ids': r_ids}
