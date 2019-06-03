"""Simple functions that help out in tank definitions."""
import copy
import numpy as np


def maybe_copy(a):
  """Copy the object 'a' if it is mutable or possible to copy. Otherwise just return the original object.

  Parameters
  ----------
  a : object
      The object to copy

  Returns
  -------
  object
      The (possibly) copied object

  """
  if type(a) in (str, unicode, int, float):
    return a
  if type(a) is np.ndarray:
    return np.array(a, copy=True)
  if type(a) in (list, dict):
    return copy.copy(a)

  return a
