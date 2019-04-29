import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def sum(a, axis=(), type_dict=None, waterwork=None, name=None):
  """Find the sum of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the sum of.
  axis : Tube, int, tuple or None
      The axis (axes) along which to take the sum.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  type_dict = ut.infer_types(type_dict, a=a, axis=axis)

  return Mean(a=a, axis=axis, waterwork=waterwork, name=name)


class Mean(ta.Tank):
  """The sum class. Handles 'a's of np.ndarray type.

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

  slot_keys = ['a', 'axis']
  tube_dict = {
    'target': np.ndarray,
    'a': np.ndarray,
    'axis': int
  }

  def _pour(self, a, axis):
    # Because 'None' is used to signify a funnel in this system, the empty
    # tuple is used to denote a sum along all axes.
    if axis == ():
      axis = None

    # Must just return 'a' as well since so much information is lost in a
    # sum
    return {'target': np.sum(a, axis=axis), 'a': ut.maybe_copy(a), 'axis': axis}

  def _pump(self, target, a, axis):
    if axis is None:
      axis = ()
    return {'a': ut.maybe_copy(a), 'axis': axis}