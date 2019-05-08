import reversible_transforms.waterworks.tank as ta
import reversible_transforms.string_manipulations.diff as di
import numpy as np


class ReplaceSubstring(ta.Tank):
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
  slot_keys = ['strings', 'old_substring', 'new_substring']
  tube_dict = {
    'target': None,
    'diff': (str, None),
    'old_substring': None,
    'new_substring': None,
  }

  def _pour(self, strings, old_substring, new_substring):
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
    target = np.char.replace(strings, old_substring, new_substring)

    diff = np.vectorize(di.get_diff_string)(target, strings)

    return {'target': target, 'diff': diff, 'old_substring': old_substring, 'new_substring': new_substring}

  def _pump(self, target, diff, old_substring, new_substring):
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

    strings = np.vectorize(di.reconstruct)(target, diff)

    return {'strings': strings.astype(dtype), 'old_substring': old_substring, 'new_substring': new_substring}
