"""LowerCase tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import nltk
import tinysegmenter


class LowerCase(ta.Tank):
  """The defintion of the LowerCase tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'lower_case'
  slot_keys = ['strings']
  tube_keys = ['target', 'diff']

  def _pour(self, strings):
    """Execute the LowerCase tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of strings
      The array of strings to lower case

    Returns
    -------
    dict(
      target: np.ndarray of strings
        The array of lower cased strings.
      diff: np.ndarray of strings
        The string difference between the original strings and the lower cased strings.
    )

    """
    strings = np.array(strings)
    target = np.char.lower(strings)

    diff = np.vectorize(di.get_diff_string)(target, strings)
    self.target = target
    self.diff = diff
    return {'target': target, 'diff': diff}

  def _pump(self, target, diff):
    """Execute the LowerCase tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of strings
      The array of lower cased strings.
    diff: np.ndarray of strings
      The string difference between the original strings and the lower cased strings.

    Returns
    -------
    dict(
      strings: np.ndarray of strings
        The array of strings to lower case
    )

    """

    strings = np.vectorize(di.reconstruct)(target, diff)

    return {'strings': strings}
