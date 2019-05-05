import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut


# def clone(a, type_dict=None, waterwork=None, name=None):
#   type_dict = ut.infer_types(type_dict, a=a)
#
#   class CloneTyped(Clone):
#     tube_dict = {
#       'a': type_dict['a'],
#       'b': type_dict['a']
#     }
#
#   return CloneTyped(a=a, waterwork=waterwork, name=name)


class Clone(ta.Tank):
  slot_keys = ['a']
  tube_dict = {
    'a': None,
    'b': None
  }

  def _pour(self, a):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(a)}

  def _pump(self, a, b):
    return {'a': ut.maybe_copy(a)}
