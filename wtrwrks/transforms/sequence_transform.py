"""DatasetTransform definition."""
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.name_space as ns
import wtrwrks.waterworks.waterwork as wa
import wtrwrks.transforms.transform as tr
import wtrwrks.transforms.cat_transform as ct
import wtrwrks.transforms.datetime_transform as dt
import wtrwrks.transforms.num_transform as nt
import wtrwrks.transforms.string_transform as st
from wtrwrks.waterworks.empty import empty
import os
import numpy as np

class DatasetTransform(tr.Transform):
  """A Transform that is built out of other transform. Used to handle entire dataset which may have varied input types. Transforms an array normalized vectors and any information needed to fully reconstruct the original array.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object

  Attributes
  ----------
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  input_shape: list of ints
    The shape of the original inputted array.
  transforms : dict
    The dictionary which holds all the Transforms that the DatasetTransform is built from.
  transform_col_ranges : list of 2-tuples
    The slice defintions that split up an full dataset array into subarrays which are fed to the sub transforms found in the 'transforms' dictionary.

  """
  attribute_dict = {'name': '', 'transforms': None, 'transform_col_ranges': None}

  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)

  def _feature_def(self, num_cols=None):
    """Get the dictionary that contain the FixedLenFeature information for each key found in the example_dicts. Needed in order to build ML input pipelines that read tfrecords.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values equal the FixedLenFeature defintion of the example key.

    """
    feature_dict = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_feature_dict = trans._feature_def(prefix=self.name)
      feature_dict.update(trans_feature_dict)
    return feature_dict

  def _get_example_dicts(self, pour_outputs):
    """Create a list of dictionaries for each example from the outputs of the pour method.

    Parameters
    ----------
    pour_outputs : dict
      The outputs of the _extract_pour_outputs method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    list of dicts of features
      The example dictionaries which contain tf.train.Features.

    """
    # Pull out the example dicts from each of the transforms.
    all_example_dicts = {}
    for key in self.transforms:
      trans = self.transforms[key]
      all_example_dicts[key] = trans._get_example_dicts(pour_outputs, prefix=self.name)

    # Merge the dictionaries together so that there is one example_dict for
    # each example.
    example_dicts = []
    for row_num, trans_dicts in enumerate(zip(*[all_example_dicts[k] for k in self.transforms])):
      example_dict = {}
      for trans_dict in trans_dicts:
        example_dict.update(trans_dict)
      example_dicts.append(example_dict)
    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    """Convert the list of example_dicts into the original outputs that came from the pour method.

    Parameters
    ----------
    example_dicts: list of dicts of arrays
      The example dictionaries which the arrays associated with a single example.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary of all the values outputted by the pour method.

    """
    pour_outputs = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_pour_outputs = trans._parse_example_dicts(example_dicts, prefix=self.name)
      pour_outputs.update(trans_pour_outputs)
    return pour_outputs

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(DatasetTransform, self)._setattributes(**kwargs)
    if self.transforms is None:
      self.transforms = {}
      self.transform_col_ranges = {}

  def _shape_def(self, num_cols=None):
    """Get the dictionary that contain the original shapes of the arrays before being converted into tfrecord examples.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values are the shapes of the arrays of a single example.

    """
    shape_dict = {}
    for key in self.transforms:
      trans = self.transforms[key]
      trans_shape_dict = trans._shape_def(prefix=self.name)
      shape_dict.update(trans_shape_dict)
    return shape_dict

  def add_transform(self, col_ranges, transform):
    """Add another transform to the dataset transform. The added transform must have a unique name.

    Parameters
    ----------
    col_ranges : 2-tuple of ints
      The slice of the array that this transformation will operate on.
    transform : Transform
      The transform object that will operate on the subarray described by col_ranges.

    """
    name = transform.name
    if name is None or name == '':
      raise ValueError("Transform must have it's name set, got: " + str(name))
    elif name in self.transforms:
      raise ValueError(str(name) + " already the name of a transform.")

    self.transforms[name] = transform
    self.transform_col_ranges[name] = col_ranges

  def calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    verbose : bool
      Whether or not to print out warnings.

    """
    self.input_dtype = array.dtype
    self.input_shape = array.shape
    all_ranges = []
    for key in self:
      trans = self.transforms[key]

      # Get the slice definition of the transform.
      col_range = self.transform_col_ranges[key]
      all_ranges.append(col_range[0])
      all_ranges.append(col_range[1])

      # Get the subarrays and cast them to valid dtypes.
      subarray = array[:, col_range[0]: col_range[1]]
      if isinstance(trans, nt.NumTransform):
        subarray = subarray.astype(np.float64)
      elif isinstance(trans, dt.DateTimeTransform):
        subarray = subarray.astype(np.datetime64)
      elif isinstance(trans, st.StringTransform):
        subarray = subarray.astype(np.unicode)
      elif isinstance(trans, ct.CatTransform):
        subarray = subarray.astype(np.unicode)

      # Calculate the global values for this transform.
      self.transforms[key].calc_global_values(subarray)

    # Verify that all the columns were used from the array, otherwise throw
    # an error.
    all_ranges = set(range(array.shape[1]))
    for key in self:
      col_range = self.transform_col_ranges[key]
      for index in xrange(col_range[0], col_range[1]):
        if index in all_ranges:
          all_ranges.remove(index)

    if all_ranges:
      raise ValueError("Must use all columns in array. Columns " + str(sorted(all_ranges)) + " are unused. Either remove them from the array or all additional transforms which use them.")

  def define_waterwork(self, array=empty):
    """Get the waterwork that completely describes the pour and pump transformations.

    Parameters
    ----------
    array : np.ndarray or empty
      The array to be transformed.

    Returns
    -------
    Waterwork
      The waterwork with all the tanks (operations) added, and names set.

    """
    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")

    with ns.NameSpace(self.name):
      indices = [self.transform_col_ranges[k] for k in sorted(self.transforms)]

      # Can only partition along the 0th axis so transpose it so that the
      # 'column' dimension is the first
      perm = [1, 0] + list(self.input_shape[2:])
      transp, transp_slots = td.transpose(a=array, axes=perm)

      # Parition the full dataset array into subarrays so that the individual
      # transforms can handle them.
      parts, _ = td.partition(a=transp['target'], indices=indices)
      parts['missing_cols'].set_name('missing_cols')
      parts['missing_array'].set_name('missing_array')
      transp_slots['a'].set_name('input')

      # Split up the Tube object into a list of Tubes so they can each be fed
      # into individual transforms.
      parts_list, _ = td.iter_list(parts['target'], num_entries=len(self.transforms))
      for part, name in zip(parts_list, sorted(self.transforms)):
        trans = self.transforms[name]

        # Transpose it back to it's original orientation
        trans_back, _ = td.transpose(a=part, axes=perm)
        part = trans_back['target']

        # Depending on the type of transform, cast the subarray to its valid
        # type.
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

  def get_waterwork(self, array=empty, recreate=False):
    """Create the Transform's waterwork or return the one that was already created.

    Parameters
    ----------
    recreate : bool
      Whether or not to force the transform to create a new waterwork.

    Returns
    -------
    Waterwork
      The waterwork object that this transform creates.

    """
    if self.waterwork is not None and not recreate:
      return self.waterwork
    with wa.Waterwork() as ww:
      self.define_waterwork(array)
    self.waterwork = ww
    return ww

  def pour(self, array):
    """Execute the transformation in the pour (forward) direction.

    Parameters
    ----------
    array : np.ndarray
      The numpy array to transform.

    Returns
    -------
    dict
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original rate.

    """
    # Retrieve the waterwork and initate the funnel_dict
    ww = self.get_waterwork()
    funnel_dict = {'input': array}
    funnel_dict = self._pre(funnel_dict)

    # Get the funnel dicts from each of the transforms.
    for name in self.transforms:
      trans = self.transforms[name]
      funnel_dict.update(
        trans._get_funnel_dict(prefix=self.name)
      )

    # Run the waterwork in the pour direction
    tap_dict = ww.pour(funnel_dict, key_type='str')

    # Extract out the relevant pour outputs from each of the transforms.
    pour_outputs = {}
    for name in self.transforms:
      trans = self.transforms[name]

      temp_outputs = trans._extract_pour_outputs(tap_dict, prefix=self.name)
      pour_outputs.update(temp_outputs)
    return pour_outputs

  def pump(self, pour_outputs):
    """Execute the transformation in the pump (backward) direction.

    Parameters
    ----------
    kwargs: dict
      The dictionary all information needed to completely reconstruct the original rate.

    Returns
    -------
    array : np.ndarray
      The original numpy array that was poured.

    """
    ww = self.get_waterwork()

    # Define the parts of the tap dict that are not dependent on the transforms
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

      # Depending on the type of Transform, create a default empty array of
      # the inputted dtype. This is needed for cast tank.
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

      # Add in the tap dicts from each of transforms
      for output_name in pour_outputs:
        if not output_name.startswith(prefix):
          continue
        kwargs[output_name] = pour_outputs[output_name]
      tap_dict.update(
        trans._get_tap_dict(kwargs, prefix=self.name)
      )

    # Run the waterwork in the pump direction.
    funnel_dict = ww.pump(tap_dict, key_type='str')

    return funnel_dict[os.path.join(self.name, 'input')]
