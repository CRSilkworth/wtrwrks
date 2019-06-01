"""HalfWidth tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
# from chop.mmseg import Tokenizer as MMSEGTokenizer
# from chop.hmm import Tokenizer as HMMTokenizer
import unicodedata


def _half_width(string):
  return unicodedata.normalize('NFKC', unicode(string))


class HalfWidth(ta.Tank):
  """The defintion of the HalfWidth tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'half_width'
  slot_keys = ['strings']
  tube_keys = ['target', 'diff']

  def _pour(self, strings):
    """Execute the HalfWidth tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    strings: np.ndarray of unicode
      The array of unicode characters to be converted to half width

    Returns
    -------
    dict(
      target: np.ndarray of unicode
        The array of half widthed strings.
      diff: np.darray of unicode
        The string difference between the original strings and the half widthed strings.
    )

    """
    strings = np.array(strings)
    target = np.vectorize(_half_width)(strings)

    diff = np.vectorize(di.get_diff_string)(target, strings)

    return {'target': target, 'diff': diff}

  def _pump(self, target, diff):
    """Execute the HalfWidth tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of unicode
      The array of half widthed strings.
    diff: np.darray of unicode
      The string difference between the original strings and the half widthed strings.

    Returns
    -------
    dict(
      strings: np.ndarray of unicode
        The array of unicode characters to be converted to half width
    )

    """

    strings = np.vectorize(di.reconstruct)(target, diff)

    return {'strings': strings}
