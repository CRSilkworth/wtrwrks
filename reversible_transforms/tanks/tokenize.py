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
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """
  slot_keys = ['strings', 'tokenizer', 'max_len', 'delimiter']
  tube_dict = {
    'target': None,
    'diff': (str, None),
    'tokenizer': None,
    'delimiter': (str, None)
  }

  # def __init__(self, *args, **kwargs):
  #   super(Tokenize, self).__init__(*args, **kwargs)
  #   if self.slots['language'] == 'en':
  #     self.tokenizer = np.vectorize(nltk.word_tokenize)
  #   elif self.slots['language'] == 'ja':
  #     tokenizer = tinysegmenter.TinySegmenter()
  #     self.tokenizer = np.vectorize(tokenizer.tokenize)
  #   elif self.slots['language'] == 'ko':
  #     self.tokenizer = np.vectorize(lambda s: s.split())
  #   elif self.slots['language'] == 'zh_hans':
  #     tokenizer = HMMTokenizer()
  #     self.tokenizer = np.vectorize(tokenizer.cut)
  #   elif self.slots['language'] == 'zh_hant':
  #     tokenizer = MMSEGTokenizer()
  #     self.tokenizer = np.vectorize(tokenizer.cut)
  #   else:
  #     self.tokenizer = np.vectorize(lambda s: s.split())

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

    return {'target': target, 'diff': diff, 'tokenizer': tokenizer, 'delimiter': delimiter}

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
    if self.tube_dict['target'][1] is not None:
      dtype = self.tube_dict['target'][1]
    else:
      dtype = self.tube_dict['target'][0]
    max_len = target.shape[-1]
    all_strings = []
    for tokens, diff_string in zip(np.reshape(target, (-1, max_len)), diff.flatten()):
      string = delimiter.join(tokens)
      string = di.reconstruct(string, diff_string)
      all_strings.append(string)

    strings = np.reshape(all_strings, target.shape[:-1]).astype(dtype)
    return {'strings': strings, 'tokenizer': tokenizer, 'max_len': max_len, 'delimiter': delimiter}
