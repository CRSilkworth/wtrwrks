import waterworks.tanks.tank_defs as td
import waterworks.waterworks.name_space as ns
import waterworks.waterworks.waterwork as wa
import waterworks.transforms.transform as tr
import waterworks.transforms.cat_transform as ct
import waterworks.transforms.datetime_transform as dt
import waterworks.transforms.num_transform as nt
import waterworks.transforms.string_transform as st
from waterworks.waterworks.empty import empty
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
    all_ranges = []
    for key in self:
      trans = self.transforms[key]

      col_range = self.transform_col_ranges[key]
      all_ranges.append(col_range[0])
      all_ranges.append(col_range[1])

      subarray = array[:, col_range[0]: col_range[1]]
      if isinstance(trans, nt.NumTransform):
        subarray = subarray.astype(np.float64)
      elif isinstance(trans, dt.DateTimeTransform):
        subarray = subarray.astype(np.datetime64)
      elif isinstance(trans, st.StringTransform):
        subarray = subarray.astype(np.unicode)
      elif isinstance(trans, ct.CatTransform):
        subarray = subarray.astype(np.unicode)

      self.transforms[key].calc_global_values(subarray)

    all_ranges = set(range(array.shape[1]))
    for key in self:
      col_range = self.transform_col_ranges[key]

      for index in xrange(col_range[0], col_range[1]):
        if index in all_ranges:
          all_ranges.remove(index)

    if all_ranges:
      raise ValueError("Must use all columns in array. Columns " + str(sorted(all_ranges)) + " are unused. Either remove them from the array or all additional transforms which use them.")
  def define_waterwork(self, array=empty):
    with ns.NameSpace(self.name):
      indices = [self.transform_col_ranges[k] for k in sorted(self.transforms)]
      transp, transp_slots = td.transpose(a=array, axes=[1, 0])
      parts, _ = td.partition(a=transp['target'], indices=indices)
      parts['missing_cols'].set_name('missing_cols')
      parts['missing_array'].set_name('missing_array')
      transp_slots['a'].set_name('input')

      parts_list, _ = td.iter_list(parts['target'], num_entries=len(self.transforms))
      for part, name in zip(parts_list, sorted(self.transforms)):
        trans = self.transforms[name]
        trans_back, _ = td.transpose(a=part, axes=[1, 0])
        part = trans_back['target']

        if isinstance(trans, nt.NumTransform):
          cast, _ = td.cast(part, np.float64, name='-'.join([name, 'cast']))
          part = cast['target']
        elif isinstance(trans, dt.DateTimeTransform):
          cast, _ = td.cast(part, np.datetime64, name='-'.join([name, 'cast']))
          part = cast['target']
        elif isinstance(trans, st.StringTransform):
          cast, _ = td.cast(part, np.unicode, name='-'.join([name, 'cast']))
          part = cast['target']
        elif isinstance(trans, ct.CatTransform):
          cast, _ = td.cast(part, np.unicode, name='-'.join([name, 'cast']))
          part = cast['target']
        with ns.NameSpace(name):
          trans.define_waterwork(array=part)

  def get_waterwork(self, array=empty):
    with wa.Waterwork() as ww:
      self.define_waterwork(array)

    return ww

  def pour(self, array):
    ww = self.get_waterwork()
    funnel_dict = {'input': array}
    funnel_dict = self._pre(funnel_dict)
    for name in self.transforms:
      trans = self.transforms[name]
      funnel_dict.update(
        trans._get_funnel_dict(prefix=self.name)
      )

    tap_dict = ww.pour(funnel_dict, key_type='str')

    pour_outputs = {}
    for name in self.transforms:
      trans = self.transforms[name]

      temp_outputs = trans._extract_pour_outputs(tap_dict, prefix=self.name)
      pour_outputs.update(temp_outputs)
    return pour_outputs

  def pump(self, pour_outputs):
    ww = self.get_waterwork()

    shape = (0, 1)
    tap_dict = {
      self._pre('missing_cols'): np.zeros(shape, dtype=np.object),
      self._pre('missing_array'): np.zeros(shape, dtype=np.object),
    }
    tap_dict[self._pre('Partition_0/tubes/indices')] = np.array([self.transform_col_ranges[k] for k in sorted(self.transform_col_ranges)])
    tap_dict[self._pre('Transpose_0/tubes/axes')] = [1, 0]
    for num, _ in enumerate(self):
      num += 1
      transp_key = 'Transpose_' + str(num) + '/tubes/axes'
      tap_dict[self._pre(transp_key)] = [1, 0]

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
      elif isinstance(trans, st.StringTransform):
        input_dtype = trans.input_dtype
        tank_name = os.path.join(self.name, '-'.join([name, 'cast']))
        tap_dict[os.path.join(tank_name, 'tubes', 'diff')] = np.array([], dtype=np.unicode)
        tap_dict[os.path.join(tank_name, 'tubes', 'input_dtype')] = input_dtype
      elif isinstance(trans, ct.CatTransform):
        input_dtype = trans.input_dtype
        tank_name = os.path.join(self.name, '-'.join([name, 'cast']))
        tap_dict[os.path.join(tank_name, 'tubes', 'diff')] = np.array([], dtype=np.unicode)
        tap_dict[os.path.join(tank_name, 'tubes', 'input_dtype')] = input_dtype
      kwargs = {}
      prefix = os.path.join(self.name, name) + '/'
      for output_name in pour_outputs:
        # if prefix == 'DT/STRING/':
          # print output_name, output_name.startswith(prefix)
        if not output_name.startswith(prefix):
          continue
        kwargs[output_name] = pour_outputs[output_name]
      tap_dict.update(
        trans._get_tap_dict(kwargs, prefix=self.name)
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

  def _get_example_dicts(self, pour_outputs):
    all_example_dicts = {}
    for key in self.transforms:
      trans = self.transforms[key]
      all_example_dicts[key] = trans._get_example_dicts(pour_outputs, prefix=self.name)

    example_dicts = []
    for row_num, trans_dicts in enumerate(zip(*[all_example_dicts[k] for k in self.transforms])):
      example_dict = {}
      for trans_dict in trans_dicts:
        example_dict.update(trans_dict)
      example_dicts.append(example_dict)
    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    pour_outputs = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_pour_outputs = trans._parse_example_dicts(example_dicts, prefix=self.name)
      pour_outputs.update(trans_pour_outputs)
    return pour_outputs

  def _feature_def(self, num_cols=None):
    feature_dict = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_feature_dict = trans._feature_def(prefix=self.name)
      feature_dict.update(trans_feature_dict)
    return feature_dict

  def _shape_def(self, num_cols=None):
    shape_dict = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_shape_dict = trans._shape_def(prefix=self.name)
      shape_dict.update(trans_shape_dict)
    return shape_dict
