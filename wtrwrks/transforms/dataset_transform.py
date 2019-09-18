"""DatasetTransform definition."""
import transform as n
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.name_space as ns
import wtrwrks.waterworks.waterwork as wa
import wtrwrks.transforms.transform as tr
from wtrwrks.waterworks.empty import empty
import os
import numpy as np
import pandas as pd
import itertools
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
import pprint

class DatasetTransform(tr.Transform):
  """A Transform that is built out of other transform. Used to handle entire dataset which may have varied input types. Transforms an array normalized vectors and any information needed to fully reconstruct the original array.

  Parameters
  ----------
  name : str
    The name of the transform.
  dtype : numpy dtype
    The data type the transformed data should have. Defaults to np.float64.
  input_dtype: numpy dtype
    The datatype of the original inputted array.

  Attributes
  ----------
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  input_shape: list of ints
    The shape of the original inputted array.
  transforms : dict
    The dictionary which holds all the Transforms that the DatasetTransform is built from.
  transform_names : list of strs
    The names of all the subtransforms
  transform_cols : dict
    A mapping from the transform name to a list of strs which define the columns of the datafram that the subtransform acts on.
  params : dict
    A dicitonary which holds any additional information about the dataset the user wants to save. (e.g. num val examples)

  """

  attribute_dict = {'transforms': None, 'transform_cols': None, 'transform_names': None, 'params': None}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set()
  required_params.update(n.Transform.required_params)

  def __init__(self, from_file=None, save_dict=None, **kwargs):
    """Define a transform using a dictionary, file, or by setting the attribute values in kwargs.

    Parameters
    ----------
    from_file : None or str
      The file path of the Tranform that was written to disk.
    save_dict : dict or None
      The dictionary of attributes that completely define a Transform.
    name : str
      The name of the transform.
    dtype : numpy dtype
      The data type the transformed data should have. Defaults to np.float64.
    input_dtype: numpy dtype
      The datatype of the original inputted array.
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(DatasetTransform, self).__init__(from_file, save_dict, **kwargs)

    if self.transforms is None:
      self.transforms = {}
      self.transform_cols = {}
      self.transform_names = []

    if self.params is None:
      self.params = {}

  def __getitem__(self, key):
    """Return the transform corresponding to key."""
    return self.transforms[key]

  def __iter__(self):
    """Iterate transform set. It is just the iterator of the transforms dictionary."""
    return iter(self.transforms)

  def _calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      Some of the data that will be transformed.

    """
    df = pd.DataFrame(
      data=array,
      index=range(array.shape[0]),
      columns=self.cols
    )
    for key in self.transform_names:
      trans = self.transforms[key]
      cols = self.transform_cols[key]

      if np.array(cols).dtype.type not in (np.dtype('O'), np.dtype('S'), np.dtype('U')):
        cols = [self.name + '_' + str(dim) for dim in cols]

      self.transform_cols[key] = cols
      self.transforms[key].cols = cols

      # Get the subarrays and cast them to valid dtypes.
      subarray = df[cols].values
      if trans.input_dtype is None:
        raise ValueError("Must explicitly set the input dtype if using the transform as part of a dataset_transform.")

      subarray = subarray.astype(trans.input_dtype)

      # Calculate the global values for this transform.
      self.transforms[key]._calc_global_values(subarray)
      self.transforms[key].is_calc_run = True

  def _finish_calc(self):
    """Finish up the calc global value process."""
    # Run the finsh calc of it's constituents
    for key in self:
      trans = self.transforms[key]
      trans._finish_calc()

    # Verify that all the columns were used from the array, otherwise throw
    # an error.
    all_ranges = set(range(len(self.cols)))
    for key in self:
      cols = self.transform_cols[key]
      if np.array(cols).dtype.type in (np.dtype('O'), np.dtype('S'), np.dtype('U')):
        cols = [list(self.cols).index(c) for c in cols]
      for index in cols:
        if index in all_ranges:
          all_ranges.remove(index)

    if all_ranges:
      raise ValueError("Must use all columns in array. Columns " + str(sorted([self.cols[i] for i in all_ranges])) + " are unused. Either remove them from the array or all additional transforms which use them.")

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    import wtrwrks.transforms.transform_defs as trd
    for key in self.attribute_dict:
      if key == 'transforms':
        continue
      setattr(self, key, save_dict[key])

    transforms = {}
    for key in save_dict['transforms']:
      trans_save_dict = save_dict['transforms'][key]
      trans = eval('trd.' + trans_save_dict['__class__'])(save_dict=trans_save_dict)
      transforms[key] = trans
    setattr(self, 'transforms', transforms)

  def _get_array_attributes(self, prefix=''):
    """Get the dictionary that contain the original shapes of the arrays before being converted into tfrecord examples.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    array_attributes : dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values are the shapes of the arrays of a single example.

    """
    att_dict = {}
    for key in self.transform_names:
      trans = self.transforms[key]
      trans_att_dict = trans._get_array_attributes(prefix=os.path.join(prefix, self.name))
      att_dict.update(trans_att_dict)
    return att_dict

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      if key == 'transforms':
        continue
      save_dict[key] = getattr(self, key)

    save_dict['__class__'] = str(self.__class__.__name__)
    save_dict['__module__'] = str(self.__class__.__module__)

    save_dict['transforms'] = {}
    for key in self.transform_names:
      save_dict['transforms'][key] = self.transforms[key]._save_dict()
    return save_dict

  def _start_calc(self):
    """Start the calc global value process."""
    for key in self:
      trans = self.transforms[key]
      trans._start_calc()
    self.num_examples = 0.

  def add_transform(self, cols, transform):
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

    self.transform_names.append(name)
    self.transforms[name] = transform
    self.transform_cols[name] = cols
    if len(cols) == 0 or type(cols) is not list:
      raise ValueError("Must pass a non empty list of columns to cols.")
    elif np.array(cols).dtype.type in (np.dtype('O'), np.dtype('S'), np.dtype('U')):
      transform.cols = cols[:]
    else:
      transform.cols = [transform.name + '_' + str(col) for col in cols]

  def define_waterwork(self, array=empty, return_tubes=None, prefix=''):
    """Get the waterwork that completely describes the pour and pump transformations.

    Parameters
    ----------
    array : np.ndarray or empty
      The array to be transformed.
    return_tubes : list of str or None
      Tube objects to be returned from the Waterwork object. Only needed if Waterworks are being stiched together.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    Waterwork
      The waterwork with all the tanks (operations) added, and names set.

    """
    assert self.is_calc_run, ("Run calc_global_values before running the transform")

    with ns.NameSpace(self.name):
      indices = []

      all_cols = []
      for name in self.transform_names:
        trans_cols = self.transform_cols[name]
        all_cols.extend(trans_cols)

      for name in self.transform_names:
        trans_cols = self.transform_cols[name]
        indices.append([list(all_cols).index(c) for c in trans_cols])

      # Can only partition along the 0th axis so transpose it so that the
      # 'column' dimension is the first
      perm = [1, 0]
      transp, transp_slots = td.transpose(a=array, axes=perm)

      # Parition the full dataset array into subarrays so that the individual
      # transforms can handle them.
      parts, _ = td.partition_by_index(
        a=transp['target'], indices=indices,
        tube_plugs={
          'missing_cols': np.zeros((0, 1), dtype=self.input_dtype),
          'missing_array': np.zeros((0, 1), dtype=self.input_dtype)
        }
      )
      parts['missing_cols'].set_name('missing_cols')
      parts['missing_array'].set_name('missing_array')
      transp_slots['a'].set_name('array')

      # Split up the Tube object into a list of Tubes so they can each be fed
      # into individual transforms.
      parts_list, _ = td.iter_list(parts['target'], num_entries=len(self.transforms))
      for part, name in zip(parts_list, self.transform_names):
        trans = self.transforms[name]

        # Transpose it back to it's original orientation
        trans_back, _ = td.transpose(a=part, axes=perm, name=name + '-Trans')
        part = trans_back['target']

        # Depending on the type of transform, cast the subarray to its valid
        # type.
        cast, _ = td.cast(
          part, trans.input_dtype,
          tube_plugs={
            'input_dtype': self.input_dtype,
            'diff': np.array([], dtype=self.input_dtype)
          },
          name=name + '-Cast'
        )
        # if isinstance(trans, nt.NumTransform):
        #   cast, _ = td.cast(part, np.float64, name='-'.join([name, 'cast']))
        #   part = cast['target']
        # elif isinstance(trans, dt.DateTimeTransform):
        #   cast, _ = td.cast(part, np.datetime64, name='-'.join([name, 'cast']))
        #   part = cast['target']
        # elif isinstance(trans, st.StringTransform):
        #   cast, _ = td.cast(part, np.unicode, name='-'.join([name, 'cast']))
        #   part = cast['target']
        # elif isinstance(trans, mlst.MultiLingualStringTransform):
        #   cast, _ = td.cast(part, np.unicode, name='-'.join([name, 'cast']))
        #   part = cast['target']
        # elif isinstance(trans, ct.CatTransform):
        #   cast, _ = td.cast(part, np.unicode, name='-'.join([name, 'cast']))
        #   part = cast['target']
        with ns.NameSpace(name):
          trans.define_waterwork(array=cast['target'], prefix=os.path.join(prefix, self.name))

    if return_tubes is not None:
      ww = parts['missing_array'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes

  def pour(self, data=None, data_iter=None):
    """Execute the transformation in the pour (forward) direction.

    Parameters
    ----------
    data : np.ndarray
      The numpy array to transform.
    data_iter : iterator of np.array or pd.DataFrame
      The entire dataset in the form of an iterator of numpy array or a pandas DataFrame. Needed if the dataset is too large to fit in memory. Should have the same columns as the arrays that will be fed to the pour method. Can only use if 'data' is not being used
    Returns
    -------
    tap_dict : dict (or iterator of dicts)
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original data. Returns an iterator of dicts if something is passed to 'data_iter' rather than the 'data' argument.

    """

    all_cols = []
    for name in self.transform_names:
      all_cols.extend(self.transform_cols[name])

    # if dealing with an array use numerical columns
    if not type(data) is pd.DataFrame:
      new_all_cols = []
      for col in all_cols:
        new_all_cols.append(self.cols.index(col))
      all_cols = new_all_cols

    def normalize(data):
      if type(data) is pd.DataFrame:
        data = data[all_cols]
      else:
        data = data[:, all_cols]
      return data

    if data is not None and data_iter is None:
      data = normalize(data)
      return super(DatasetTransform, self).pour(data=data)
    elif data_iter is not None and data is None:
      data_iter = itertools.imap(normalize, data_iter)
      return super(DatasetTransform, self).pour(data_iter=data_iter)
    else:
      raise ValueError("Must supply exactly one data or data_iter.")

  def pump(self, tap_dict, df=False, index=None):
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
    funnel_dict = ww.pump(tap_dict, key_type='str')
    array = funnel_dict[self._pre('array')].astype(self.input_dtype)
    if df:
      if index is None:
        index = np.arange(array.shape[0])
      data = pd.DataFrame(
        data=array,
        index=index,
        columns=self.cols
      )
      for name in self.transform_names:
        trans = self.transforms[name]
        data[trans.cols] = data[trans.cols].astype(trans.input_dtype)
    else:
      data = array

    return data

  def get_eval_class(self, table_name):
    """Create the sqlalchemy class that defines one row of data. Useful for writing evaluation data to a sql database.

    Parameters
    ----------
    table_name : str
      The name of the table that the class will be defined for.

    Returns
    -------
    EvalClass : class
      The class that defines a row of data.

    """
    Base = declarative_base()

    class EvalClass(Base):
      __tablename__ = table_name
      example_id = sa.Column(sa.Integer, primary_key=True)

    already_added_cols = set(['example_id'])
    for name in self.transform_names:
      trans = self.transforms[name]
      for col in trans.cols:

        if col in already_added_cols:
          continue
        already_added_cols.add(col)

        if trans.input_dtype in (np.int64, np.int32, int):
          db_dtype = sa.Integer
        elif trans.input_dtype in (np.float64, np.float32, float):
          db_dtype = sa.Float
        elif trans.input_dtype in (np.bool, bool):
          db_dtype = sa.Boolean
        elif trans.input_dtype in (np.dtype('S'), np.dtype('U'), np.dtype('O')):
          db_dtype = sa.String
        elif trans.input_dtype in (np.datetime64,):
          db_dtype = sa.DateTime
        else:
          raise ValueError("{} is not a supported type to be used on the database.".format())

        setattr(EvalClass, col, sa.Column(db_dtype))

    return EvalClass

  def get_schema_dict(self, var_lim=None):
    """Create a dictionary which defines the proper fields which are needed to store the untransformed data in a (postgres) SQL database.

    Parameters
    ----------
    var_lim : int or dict
      The maximum size of strings. If an int is passed then all VARCHAR fields have the same limit. If a dict is passed then each field gets it's own limit. Defaults to 255 for all string fields.

    Returns
    -------
    schema_dict : dict
      Dictionary where the keys are the field names and the values are the SQL data types.

    """
    schema_dict = {}
    for name in self.transform_names:
      trans = self.transforms[name]

      if type(var_lim) is dict and name in var_lim:
        schema_dict.update(trans.get_schema_dict(var_lim[name]))
      elif type(var_lim) is dict:
        schema_dict.update(trans.get_schema_dict(None))
      else:
        schema_dict.update(trans.get_schema_dict(var_lim))

    return schema_dict

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
