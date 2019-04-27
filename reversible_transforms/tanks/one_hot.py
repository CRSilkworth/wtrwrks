import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut


def one_hot(a, depth, axis=-1, type_dict=None, waterwork=None, name=None):
  """OneHottiply two objects together in a reversible manner. This function selects out the proper OneHot subclass depending on the types of 'a' and 'b'.

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
  type_dict = ut.infer_types(type_dict, a=a, depth=depth)

  class OneHotTyped(OneHot):
    tube_dict = {
      'target': ut.decide_type(type_dict['a'], type_dict['b']),
      'a': type_dict['a'],
      'depth': dict
    }

  return OneHotTyped(a=a, depth, waterwork=waterwork, name=name)


class OneHot(ta.Tank):
  slot_keys = ['a', 'depth']
  tube_dict = {
    'target': np.ndarray,
    'index_to_cat': dict
  }

  def _pour(self, a, b):
    return {'target': a * b, 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  def _pump(self, a, b, target):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

class OneHotIndex(ta.Tank):
  slot_keys = ['a', 'depth']
  tube_dict = {
    'target': np.ndarray,
    'index_to_cat': dict
  }

  def _pour(self, a, b):
    return {'target': a * b, 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  def _pump(self, a, b, target):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}
