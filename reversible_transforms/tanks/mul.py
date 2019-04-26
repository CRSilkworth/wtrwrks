import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def mul(a, b, type_dict=None, waterwork=None, name=None):
  type_dict = ut.infer_types(type_dict, a=a, b=b)
  return Mul(a=a, b=b, waterwork=waterwork, name=name)


class Mul(ta.Tank):
  slot_keys = ['a', 'b']
  tube_keys = ['a', 'target']

  def _pour(self, a, b):
    return {'target': a * b, 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  def _pump(self, a, b, target):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}
