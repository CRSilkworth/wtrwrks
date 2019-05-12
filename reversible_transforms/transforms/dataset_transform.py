import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.waterworks.name_space as ns
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.transforms.transform as tr
from reversible_transforms.waterworks.empty import empty
import os


class DatasetTransform(tr.Transform):
  attribute_dict = {'name': '', 'transforms': None, 'transform_col_ranges': None}

  def _setattributes(self, **kwargs):
    super(DatasetTransform, self)._setattributes(**kwargs)
    if self.transforms is None:
      self.transforms = {}
      self.transform_col_ranges = {}

  def add_transform(self, col_ranges, transform):
    name = transform.name
    if name is None or name == '':
      raise ValueError("Transform must have it's name set, got: " + str(name))
    elif name in self.transforms:
      raise ValueError(str(name) + " already the name of a transform.")

    self.transforms[name] = transform
    self.transform_col_ranges[name] = col_ranges

  def calc_global_values(self, array):
    subarrays = {}
    for key in self:
      col_range = self.transform_col_ranges[key]
      subarray = array[:, col_range[0]: col_range[1]]

      self.transforms[key].calc_global_values(subarray)

  def define_waterwork(self, array=empty):
    with ns.NameSpace(self.name):
      indices = [self.transform_col_ranges[k] for k in sorted(self.transforms)]
      parts, parts_slots = td.partition(a=array, indices=indices)

      print parts['target']
      for name, part in zip(parts['target'], sorted(self.transforms)):
        with ns.NameSpace(name):
          self.transforms[name].define_waterwork(array=part)

  def get_waterwork(self, array=empty):
    with wa.Waterwork() as ww:
      self.define_waterwork(array)

    return ww

  def pour(self, array, **transform_kwargs):
    ww = self.get_waterwork()

    funnel_dict = {}
    for name in self.transforms:
      trans = self.transforms[name]
      col_range = self.transform_col_ranges[name]
      subarray = array[col_range[0]: col_range[1]]

      if name in transform_kwargs:
        kwargs = transform_kwargs[name]
      else:
        kwargs = {}

      funnel_dict.update(
        trans._get_funnel_dict(subarray, prefix=self.name, **kwargs)
      )

    tap_dict = ww.pour(funnel_dict, key_type=str)

    pour_outputs = {}
    for name in self.transforms:
      trans = self.transforms[name]

      temp_outputs = trans._extract_pour_outputs(tap_dict, prefix=self.name)
      for k, v in temp_outputs.iteritems():
        pour_outputs[os.path.join(self.name, name, k)] = v

    return pour_outputs

  def pump(self, **pour_outputs):
    ww = self.get_waterwork()

    tap_dict = {}
    for name in self.transforms:
      trans = self.transforms[name]
      kwargs = {}
      for output_name in pour_outputs:
        prefix = os.path.join(self.name, name) + '/'
        if not output_name.startswith(prefix):
          continue
        key = output_name.replace(prefix, '')
        kwargs[key] = pour_outputs[output_name]

      tap_dict.update(
        trans._get_tap_dict(prefix=self.name, **kwargs)
      )
    funnel_dict = ww.pump(tap_dict, key_type='str')

    return funnel_dict[os.path.join(self.name, 'partition/slots/a')]

  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)
