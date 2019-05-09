

class DatasetTransform(object):
  def __init__(self):
    self.transforms = {}
    self.transform_columns = {}

  def add_transform(self, name, columns, transform):
    self.transforms[name] = transform
    self.transform_columns[name] = columns

  def calc_global_values(self, array):
    subarrays = {}
    for key in self:
      cols = self.transform_columns[key]
      subarray = array[:, cols]

      self.transforms[key].calc_global_values(subarray)
  def pour(self, array):
    
  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)
