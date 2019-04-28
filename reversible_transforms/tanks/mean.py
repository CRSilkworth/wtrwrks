import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def mean(a, axis=(), type_dict=None, waterwork=None, name=None):
  """Multiply two objects together in a reversible manner. This function selects out the proper Mul subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be multiplied or None
      First object to be multiplied, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be multiplied or None
      Second object to be multiplied, or if None, a 'funnel' to fill later with data.
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
  slot_keys = ['a', 'axis']
  tube_dict = {
    'target': np.ndarray,
    'a': np.ndarray
  }

  def _pour(self, a, axis):
    if axis == ():
      axis = None
    return {'target': np.mean(a, axis=axis), 'a': ut.maybe_copy(a)}

  def _pump(self, target, a):
    return {'a': ut.maybe_copy(a)}
