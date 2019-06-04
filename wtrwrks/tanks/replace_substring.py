"""ReplaceSubstring tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.string_manipulations.diff as di
import numpy as np


class ReplaceSubstring(ta.Tank):
  """The defintion of the ReplaceSubstring tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'replace_substring'
  slot_keys = ['strings', 'old_substring', 'new_substring']
  tube_keys = ['target', 'old_substring', 'new_substring', 'diff']
  pass_through_keys = ['old_substring', 'new_substring']

  def _pour(self, strings, old_substring, new_substring):
    """Execute the ReplaceSubstring tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of strings
      The array of strings that will have it's substrings replaced.
    old_substring: str or unicode
      The substring to be replaced.
    new_substring: str or unicode
      The substring to replace with.

    Returns
    -------
    dict(
      target: np.ndarray of same type as 'a'
        The of array of strings with the substrings replaced.
      old_substring: str or unicode
        The substring to be replaced.
      new_substring: str or unicode
        The substring to replace with.
      diff: np.ndarray of strings
        The diff of the strings caused by converting all old_substrings to new_substrings and back.
    )

    """
    strings = np.array(strings)
    target = np.char.replace(strings, old_substring, new_substring)

    diff = np.vectorize(di.get_diff_string)(target, strings)

    return {'target': target, 'diff': diff, 'old_substring': old_substring, 'new_substring': new_substring}

  def _pump(self, target, diff, old_substring, new_substring):
    """Execute the ReplaceSubstring tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of same type as 'a'
      The of array of strings with the substrings replaced.
    old_substring: str or unicode
      The substring to be replaced.
    new_substring: str or unicode
      The substring to replace with.
    diff: np.ndarray of strings
      The diff of the strings caused by converting all old_substrings to new_substrings and back.

    Returns
    -------
    dict(
      strings: np.ndarray of strings
        The array of strings that will have it's substrings replaced.
      old_substring: str or unicode
        The substring to be replaced.
      new_substring: str or unicode
        The substring to replace with.
    )

    """
    strings = np.vectorize(di.reconstruct)(target, diff)

    return {'strings': strings, 'old_substring': old_substring, 'new_substring': new_substring}
