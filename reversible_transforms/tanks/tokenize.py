import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import reversible_transforms.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import nltk
import tinysegmenter


class Tokenize(ta.Tank):
  """The CatToIndex class where the cats input is an numpy array. Handles any rank for 'cats'

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_keys : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """
  slot_keys = ['strings', 'tokenizer', 'max_len', 'delimiter']
  tube_keys = ['target', 'tokenizer', 'delimiter', 'diff']

  def _pour(self, strings, tokenizer, max_len, delimiter=' '):
    """Execute the mapping in the pour (forward) direction .

    Parameters
    ----------
    strings : np.ndarray
      The strings to tokenize
    language : str
      The language that the strings are in. Decides tokenizer.
    max_len : int
      The length of the outputed arrays.

    Returns
    -------
    dict(
      'target': int, float, other non array type
        The result of the sum of 'a' and 'b'.
      'diff': np.ndarray of strings
        The diff between the original strings and the tokenized and then untokenized strings.
      'language' : str
        The language that the strings are in. Decides tokenizer.
      'max_len' : int
        The length of the outputed arrays.
    )

    """
    strings = np.array(strings)

    all_tokens = []
    all_diffs = []
    for string in strings.flatten():
      tokens = np.array(tokenizer(string))
      if tokens.size < max_len:
        num = max_len - tokens.size
        tokens = np.concatenate([tokens, np.full([num], '')])
      else:
        tokens = tokens[:max_len]

      all_tokens.append(tokens)

      processed = delimiter.join(tokens)
      diff = di.get_diff_string(processed, string)
      all_diffs.append(diff)

    token_array = np.stack(all_tokens)
    target = np.reshape(token_array, list(strings.shape) + [max_len])

    diff_array = np.stack(all_diffs)
    diff = np.reshape(diff_array, strings.shape)

    return {'target': target.astype(strings.dtype), 'diff': diff, 'tokenizer': tokenizer, 'delimiter': delimiter}

  def _pump(self, target, diff, tokenizer, delimiter):
    """Execute the mapping in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the sum of 'a' and 'b'.
    missing_vals: list
      The list of all the cats that were not found in cat_to_index_map.
    cat_to_index_map : dict
      The map from categorical values to indices.


    Returns
    -------
    dict(
      'cats' : np.ndarray
        The categorical values to be mapped to indices
      'cat_to_index_map' : dict
        The map from categorical values to indices.
    )

    """
    max_len = target.shape[-1]
    all_strings = []
    for tokens, diff_string in zip(np.reshape(target, (-1, max_len)), diff.flatten()):
      string = delimiter.join(tokens)
      string = di.reconstruct(string, diff_string)
      all_strings.append(string)

    strings = np.reshape(all_strings, target.shape[:-1])
    return {'strings': strings, 'tokenizer': tokenizer, 'max_len': max_len, 'delimiter': delimiter}
