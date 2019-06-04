"""Lemmatize tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer


class Lemmatize(ta.Tank):
  """The defintion of the Lemmatize tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'lemmatize'
  slot_keys = ['strings', 'lemmatizer']
  tube_keys = ['target', 'lemmatizer', 'diff']
  pass_through_keys = ['lemmatizer']


  def _pour(self, strings, lemmatizer):
    """Execute the Lemmatize tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of strings
      The array of strings to be lemmatized.
    lemmatizer: func
      A function which takes in a string and outputs a standardized version of that string

    Returns
    -------
    dict(
      target: np.ndarray of strings
        The array of lemmatized strings
      lemmatizer: func
        A function which takes in a string and outputs a standardized version of that string.
      diff: np.ndarray of strings
        The array of strings which define the differences between the original string and the string that has been lemmatized.
    )

    """
    strings = np.array(strings)
    target = np.vectorize(lemmatizer)(strings)

    diff = np.vectorize(di.get_diff_string)(target, strings)

    return {'target': target, 'diff': diff, 'lemmatizer': lemmatizer}

  def _pump(self, target, diff, lemmatizer):
    """Execute the Lemmatize tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of strings
      The array of lemmatized strings
    lemmatizer: func
      A function which takes in a string and outputs a standardized version of that string.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been lemmatized.

    Returns
    -------
    dict(
      strings: np.ndarray of strings
        The array of strings to be lemmatized.
      lemmatizer: func
        A function which takes in a string and outputs a standardized version of that string
    )

    """
    strings = np.vectorize(di.reconstruct)(target, diff)

    return {'strings': strings, 'lemmatizer': lemmatizer}
