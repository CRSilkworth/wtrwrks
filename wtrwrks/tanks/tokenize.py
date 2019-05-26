"""Tokenize tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import nltk
import tinysegmenter


class Tokenize(ta.Tank):
  """The defintion of the Tokenize tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  slot_keys = ['strings', 'tokenizer', 'detokenizer', 'max_len']
  tube_keys = ['target', 'tokenizer', 'detokenizer', 'diff']

  def _pour(self, strings, tokenizer, max_len, detokenizer=lambda a: ' '.join(a)):
    """Execute the Tokenize tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of strings
      The array of strings to tokenize.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    max_len: int
      The maximum number of tokens. Defines the size of the added dimension.

    Returns
    -------
    dict(
      target: np.ndarray
        The array of tokenized strings. Will have rank = rank('a') + 1 where the last dimesion will have size max_len.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      diff: np.ndarray of strings
        The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
    )

    """
    # Convert to a numpy array.
    strings = np.array(strings)

    # Handle the empty array case
    if not strings.size:
      return {'target': ut.maybe_copy(strings), 'diff': ut.maybe_copy(strings), 'tokenizer': tokenizer, 'detokenizer': detokenizer}

    all_tokens = []
    all_diffs = []


    for string in strings.flatten():
      # Tokenize the string, and regularize the length of the array by padding
      # with '' to fill out the array if it's too small or truncated if it's
      # too long.
      tokens = np.array(tokenizer(string))
      if tokens.size < max_len:
        num = max_len - tokens.size
        tokens = np.concatenate([tokens, np.full([num], '')])
      else:
        tokens = tokens[:max_len]

      all_tokens.append(tokens)

      # Detokenize the tokens and reconstruct the orignal string from the
      # diff_string
      processed = detokenizer(tokens)
      diff = di.get_diff_string(processed, string)
      all_diffs.append(diff)

    # Combine all the tokens arrays into a single array and reshape to the
    # shape of the original strings array with an additional dimesion of size
    # max_len.
    token_array = np.stack(all_tokens)
    target = np.reshape(token_array, list(strings.shape) + [max_len])

    # Keep all the string diffs and reshape it to match the original strings
    # array shape.
    diff_array = np.stack(all_diffs)
    diff = np.reshape(diff_array, strings.shape)

    return {'target': target.astype(strings.dtype), 'diff': diff, 'tokenizer': tokenizer, 'detokenizer': detokenizer}

  def _pump(self, target, diff, tokenizer, detokenizer):
    """Execute the Tokenize tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The array of tokenized strings. Will have rank = rank('a') + 1 where the last dimesion will have size max_len.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.

    Returns
    -------
    dict(
      strings: np.ndarray of strings
        The array of strings to tokenize.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      max_len: int
        The maximum number of tokens. Defines the size of the added dimension.
    )

    """
    max_len = target.shape[-1]
    all_strings = []

    # Flatten out all the dimensions aside from the token dimesion of target and
    # flatten out the diff array completely to iterate through them.
    for tokens, diff_string in zip(np.reshape(target, (-1, max_len)), diff.flatten()):

      # Detokenize and reconstruct the original string from the diff string.
      string = detokenizer(tokens)
      string = di.reconstruct(string, diff_string)
      all_strings.append(string)

    # Reshape to the original shape.
    strings = np.reshape(all_strings, target.shape[:-1])
    return {'strings': strings, 'tokenizer': tokenizer, 'max_len': max_len, 'detokenizer': detokenizer}
