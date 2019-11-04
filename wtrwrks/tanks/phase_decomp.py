"""PhaseDecomp tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.string_manipulations.diff as di
import numpy as np
import nltk
import tinysegmenter


class PhaseDecomp(ta.Tank):
  """The defintion of the PhaseDecomp tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'phase_decomp'
  slot_keys = ['a', 'w_k']
  tube_keys = ['target', 'div', 'w_k']
  pass_through_keys = ['w_k']

  def _pour(self, a, w_k):
    """Execute the PhaseDecomp tank (operation) in the pour (forward) direction.

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
    exp = np.tensordot(a, w_k, [[], []])

    div = np.floor_divide(exp, 1)
    target = np.mod(exp, 1)

    return {'target': target, 'div': div, 'w_k': w_k}

  def _pump(self, target, div, w_k):
    """Execute the PhaseDecomp tank (operation) in the pump (backward) direction.

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
    # print target, div, w_k
    w_rank = len(w_k.shape)
    a_rank = len(target.shape) - w_rank

    tiled_w_k = np.reshape(w_k, [1] * a_rank + list(w_k.shape))
    tiled_w_k = np.tile(tiled_w_k, list(target.shape[: a_rank]) + [1] * w_rank)

    axes = np.arange(a_rank, a_rank + w_rank, dtype=np.int64)
    a = (target + div)/tiled_w_k

    a[np.isnan(a)] = 1.0
    for axis in xrange(w_rank):
      a = a[..., -1]

    return {'a': a, 'w_k': w_k}
