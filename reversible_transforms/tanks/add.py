import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta

def add(a, b):
  pass

class Add(ta.Tank):
  slot_keys = ['a', 'b']
  tube_keys = ['a', 'data']

  def _pour(self, a, b):
    return {'data': a + b, 'a': a}

  def _pump(self, a, data):
    return {'a': a, 'b': data - a}
