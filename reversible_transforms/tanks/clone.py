import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta


class Clone(ta.Tank):
  slot_keys = ['a']
  tube_keys = ['a', 'b']

  def _pour(self, a):
    return {'a': a, 'b': a}

  def _pump(self, a, b):
    return {'a': a}
