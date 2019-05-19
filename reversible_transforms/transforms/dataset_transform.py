import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.waterworks.name_space as ns
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.transforms.transform as tr
import reversible_transforms.transforms.cat_transform as ct
import reversible_transforms.transforms.datetime_transform as dt
import reversible_transforms.transforms.num_transform as nt
import reversible_transforms.transforms.string_transform as st
from reversible_transforms.waterworks.empty import empty
import os
import numpy as np
import tensorflow as tf


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
      transp, transp_slots = td.transpose(a=array, axes=[1, 0])
      parts, _ = td.partition(a=transp['target'], indices=indices)
      transp_slots['a'].set_name('input')

      parts_list, _ = td.iter_list(parts['target'], num_entries=len(self.transforms))
      for part, name in zip(parts_list, sorted(self.transforms)):
        trans = self.transforms[name]
        if isinstance(trans, nt.NumTransform):
          cast, _ = td.cast(part, np.float64, name='-'.join([name, 'cast']))
          part = cast['target']
        elif isinstance(trans, dt.DateTimeTransform):
          cast, _ = td.cast(part, np.datetime64, name='-'.join([name, 'cast']))
          part = cast['target']
        with ns.NameSpace(name):
          trans.define_waterwork(array=part)

  def get_waterwork(self, array=empty):
    with wa.Waterwork() as ww:
      self.define_waterwork(array)

    return ww

  def pour(self, array, transform_kwargs):
    ww = self.get_waterwork()

    funnel_dict = {'input': array}
    funnel_dict = self._add_name_to_dict(funnel_dict)
    for name in self.transforms:
      trans = self.transforms[name]
      if name in transform_kwargs:
        kwargs = transform_kwargs[name]
      else:
        kwargs = {}

      funnel_dict.update(
        trans._get_funnel_dict(prefix=self.name, **kwargs)
      )

    tap_dict = ww.pour(funnel_dict, key_type='str')

    keys_to_output = [
      self._add_name('Partition_0/tubes/missing_cols'),
      self._add_name('Partition_0/tubes/missing_array'),
    ]
    pour_outputs = {k: tap_dict[k] for k in keys_to_output}
    for name in self.transforms:
      trans = self.transforms[name]

      temp_outputs = trans._extract_pour_outputs(tap_dict, prefix=self.name)
      for k, v in temp_outputs.iteritems():
        pour_outputs[os.path.join(self.name, name, k)] = v

    return pour_outputs

  def pump(self, pour_outputs):
    ww = self.get_waterwork()

    keys_to_input = [
      self._add_name('Partition_0/tubes/missing_cols'),
      self._add_name('Partition_0/tubes/missing_array'),

    ]
    tap_dict = {k: pour_outputs[k] for k in keys_to_input}
    tap_dict[self._add_name('Partition_0/tubes/indices')] = np.array([self.transform_col_ranges[k] for k in sorted(self.transform_col_ranges)])
    tap_dict[self._add_name('Transpose_0/tubes/axes')] = [1, 0]

    for name in sorted(self.transforms):
      trans = self.transforms[name]
      if isinstance(trans, nt.NumTransform):
        input_dtype = trans.input_dtype
        tank_name = os.path.join(self.name, '-'.join([name, 'cast']))
        tap_dict[os.path.join(tank_name, 'tubes', 'diff')] = np.zeros((), input_dtype)
        tap_dict[os.path.join(tank_name, 'tubes', 'input_dtype')] = input_dtype
      elif isinstance(trans, dt.DateTimeTransform):
        input_dtype = trans.input_dtype
        tank_name = os.path.join(self.name, '-'.join([name, 'cast']))
        tap_dict[os.path.join(tank_name, 'tubes', 'diff')] = np.zeros((), dtype='timedelta64')
        tap_dict[os.path.join(tank_name, 'tubes', 'input_dtype')] = input_dtype
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

    return funnel_dict[os.path.join(self.name, 'input')]

  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)

  def write_tfrecord(self, file_name, array, transform_kwargs=None):
    pass
