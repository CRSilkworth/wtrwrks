import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import reversible_transforms.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import nltk
import tinysegmenter


class FlatTokenize(ta.Tank):
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
  slot_keys = ['strings', 'tokenizer', 'detokenizer', 'ids']
  tube_keys = ['target', 'tokenizer', 'detokenizer', 'diff', 'shape', 'ids']

  def _pour(self, strings, ids, tokenizer, detokenizer=lambda a: ' '.join(a)):
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
    if not strings.size:
      return {'target': ut.maybe_copy(strings), 'diff': ut.maybe_copy(strings), 'tokenizer': tokenizer, 'detokenizer': detokenizer}

    all_tokens = []
    all_diffs = []

    r_ids = []
    for string_id, string in zip(ids.flatten(), strings.flatten()):
      tokens = tokenizer(string)
      all_tokens.extend(tokens)

      r_ids.extend([string_id] * len(tokens))

      processed = detokenizer(tokens)
      diff = di.get_diff_string(processed, string)
      all_diffs.extend([diff] * len(tokens))

    target = np.array(all_tokens).astype(strings.dtype)
    diff = np.array(all_diffs).astype(strings.dtype)
    r_ids = np.array(r_ids)

    return {'target': target, 'diff': diff, 'tokenizer': tokenizer, 'detokenizer': detokenizer, 'ids': r_ids, 'shape': strings.shape}

  def _pump(self, target, diff, tokenizer, detokenizer, ids, shape):
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
    if not ids.size:
      return {'strings': target.reshape([0] + list(shape[1:])), 'tokenizer': tokenizer, 'detokenizer': detokenizer}

    unique_ids, indices = np.unique(ids, return_index=True)
    indices = np.sort(indices)

    all_strings = []
    r_ids = []
    for index in indices:
      string_id = ids[index]
      mask = (ids == string_id)

      tokens = target[mask]
      diff_string = diff[mask][0]

      string = detokenizer(tokens)
      string = di.reconstruct(string, diff_string)

      r_ids.append(string_id)
      all_strings.append(string)
    r_ids = np.reshape(r_ids, shape)
    strings = np.reshape(all_strings, shape)
    return {'strings': strings, 'tokenizer': tokenizer, 'detokenizer': detokenizer, 'ids': r_ids}
