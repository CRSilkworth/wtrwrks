import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut


def div(a, b, type_dict=None, waterwork=None, name=None):
  """Divide one object by another in a reversible manner. This function selects out the proper Div subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be divided or None
      First object to be divided, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be divided or None
      Second object to be divided, or if None, a 'funnel' to fill later with data.
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
  type_dict = ut.infer_types(type_dict, a=a, b=b)

  class DivBasicTyped(DivBasic):
    tube_dict = {
      'target': ut.decide_type(type_dict['a'], type_dict['b']),
      'a': type_dict['a'],
      'b': type_dict['b']
    }

  return DivBasicTyped(a=a, b=b, waterwork=waterwork, name=name)


class DivBasic(ta.Tank):
  slot_keys = ['a', 'b']
  tube_dict = {
    'target': None,
    'a': None,
    'b': None
  }

  def _pour(self, a, b):
    return {'target': a / b, 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  def _pump(self, target, a, b):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}
