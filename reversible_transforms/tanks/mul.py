import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta


class Mul(ta.Tank):
  slot_keys = ['a', 'b']
  tube_keys = ['', 'data']

  def pour(self, a, b):
    return {'data': a + b, 'a': a}

  def pump(self, a, data):
    return {'a': a, 'b': data - a}
