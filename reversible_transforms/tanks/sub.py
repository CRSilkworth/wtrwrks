import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


# def sub(a, b, type_dict=None, waterwork=None, name=None):
#   """Sub two objects together in a reversible manner. This function selects out the proper Sub subclass depending on the types of 'a' and 'b'.
#
#   Parameters
#   ----------
#   a : Tube, type that can be summed or None
#       First object to be subed, or if None, a 'funnel' to fill later with data.
#   b : Tube, type that can be summed or None
#       Second object to be subed, or if None, a 'funnel' to fill later with data.
#   type_dict : dict({
#     keys - ['a', 'b']
#     values - type of argument 'a' type of argument 'b'.
#   })
#     The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).
#
#   waterwork : Waterwork or None
#     The waterwork to sub the tank (operation) to. Default's to the _default_waterwork.
#   name : str or None
#       The name of the tank (operation) within the waterwork
#
#   Returns
#   -------
#   Tank
#       The created sub tank (operation) object.
#
#   """
#   type_dict = ut.infer_types(type_dict, a=a, b=b)
#   target_dtype = ut.decide_dtype(np.array(a).dtype, np.array(b).dtype)
#
#   class SubTyped(Sub):
#     tube_keys = {
#       'target': (np.ndarray, target_dtype),
#       'smaller_size_array': (np.ndarray, target_dtype),
#       'a_is_smaller': (bool, None)
#     }
#
#   return SubTyped(a=a, b=b, waterwork=waterwork, name=name)


class Sub(ta.Tank):
  """The tank used to sub two numpy arrays together. The 'smaller_size_array' is the whichever of the two inputs has the fewer number of elements and 'a_is_smaller' is a bool which says whether 'a' is that array.

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
    """Execute the sub in the pour (forward) direction .

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

    target = np.array(a - b)

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller}

  def _pump(self, target, smaller_size_array, a_is_smaller):
    """Execute the sub in the pump (backward) direction .

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
      b = np.array(a - target)
    else:
      a = np.array(target + smaller_size_array)
      b = ut.maybe_copy(smaller_size_array)

    return {'a': a, 'b': b}
