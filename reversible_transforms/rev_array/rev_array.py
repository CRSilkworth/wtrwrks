import numpy as np

class OpCon(object):
  def __init__(self, op, key):
    self.out_key = (op, key)
    self.in_keys = []
    self.op = op
    self.key = key

  def __hash__(self):
    return hash(self.out_key)

  def __eq__(self, other):
    return self.out_key == other.out_key

  def __str__(self):
    return str((str(self.op), str(self.key)))

class RevArray(object):
  def __init__(self, input):
    if type(input) is np.ndarray:
      self._init_from_np(input)
    elif type(input) is RevArray:
      self._init_from_rev_array(input)
    elif type(input) is dict:
      self._init_from_dict(input)

    self.dtype = self.arrays['data'].dtype

  def _init_from_np(self, input):
    self.arrays = {'data': input}

  def _init_from_rev_array(self, input):
    self.arrays = {}
    for key in input.arrays:
      self.arrays[key] = np.copy(input[key])

  def _init_from_dict(self, input):
    self.arrays = {}
    for key in input:
      self.arrays[key] = np.copy(input[key])

  def _validate(self):
    if 'data' not in self.arrays:
      raise ValueError("RevArray must contain 'data' key")

    num_rows = self.arrays['data'].shape[0]
    for key in self.arrays:
      if self.arrays[key].shape[0] != num_rows:
        raise ValueError("All arrays in RevArray must have equal sized first dimension.")

  def __getitem__(self, key):
    return self.arrays[key]

  def __iter__(self):
    for row_num in xrange(self.arrays['data'].shape[0]):
      row_dict = {}
      for key in self.arrays:
        row_dict[key] = np.copy(self.arrays[key][row_num: row_num + 1])

      yield RevArray(row_dict)
