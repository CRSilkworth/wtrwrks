import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut


class Clone(ta.Tank):
  slot_keys = ['a']
  tube_keys = ['a', 'b']

  def _pour(self, a):
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(a)}

  def _pump(self, a, b):
    return {'a': ut.maybe_copy(a)}
