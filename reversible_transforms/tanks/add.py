import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


class Add(ta.Tank):
  """The tank used to add two numpy arrays together. The 'smaller_size_array' is the whichever of the two inputs has the fewer number of elements and 'a_is_smaller' is a bool which says whether 'a' is that array.

  Attributes
  ----------
  tube_keys : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'smaller_size_array', 'a_is_smaller']

  def _pour(self, a, b):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The first argment to be summed.
    b : np.ndarray
      The second argment to be summed.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the sum of 'a' and 'b'.
      'smaller_size_array': np.ndarray
        a or b depending on which has the fewer number of elements. defaults to b.
      'a_is_smaller': bool
        If a has a fewer number of elements then it's true, otherwise it's false.
    )

    """
    if type(a) is not np.ndarray:
      a = np.array(a)
    if type(b) is not np.ndarray:
      b = np.array(b)

    a_is_smaller = a.size < b.size
    if a_is_smaller:
      smaller_size_array = ut.maybe_copy(a)
    else:
      smaller_size_array = ut.maybe_copy(b)

    target = np.array(a + b)

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller}

  def _pump(self, target, smaller_size_array, a_is_smaller):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target : np.ndarray
      The result of the sum of 'a' and 'b'.
    smaller_size_array : np.ndarray
      The array that have the fewer number of elements
    a_is_smaller: bool
      If a is the array with the fewer number of elements

    Returns
    -------
    dict(
      'a' : np.ndarray
        The first argment.
      'b' : np.ndarray
        The second argment.
    )

    """
    if a_is_smaller:
      a = ut.maybe_copy(smaller_size_array)
      b = np.array(target - a)
    else:
      a = np.array(target - smaller_size_array)
      b = ut.maybe_copy(smaller_size_array)

    return {'a': a, 'b': b}
