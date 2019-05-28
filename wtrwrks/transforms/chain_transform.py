"""ChainTransform definition."""
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.name_space as ns
import wtrwrks.waterworks.waterwork as wa
import wtrwrks.transforms.transform as tr
from wtrwrks.waterworks.empty import empty
import os

class ChainTransform(tr.Transform):
  """A transform that is built out of other transforms. The transforms are stitched together in a chain such that one of the pour outputs from the preceding transform is fed as an argument into the next.

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
  transform_order : list of str
    The list that stores the order in which the transforms operate on the data.
  tap_keys : list of str and None
    A list that stores which of the preceding transforms' outputs will be fed as an input into the next transform. The first value is always 'None'.

  """
  attribute_dict = {'name': '', 'transforms': None, 'transform_order': None, 'tap_keys': None}

  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)

  def _extract_pour_outputs(self, tap_dict, prefix='', keep_connected=False, **kwargs):
    """Pull out all the values from tap_dict that cannot be explicitly reconstructed from the transform itself. These are the values that will need to be fed back to the transform into run the tranform in the pump direction.

    Parameters
    ----------
    tap_dict : dict
      The dictionary outputted by the pour (forward) transform.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      Dictionay of only those tap dict values which are can't be inferred from the Transform itself.

    """
    pour_outputs = {}
    for trans_num, trans_key in enumerate(self.transform_order):
      trans = self.transforms[trans_key]
      pour_outputs.update(trans._extract_pour_outputs(tap_dict, prefix=os.path.join(prefix, self.name)))

      if trans_num >= len(self.transform_order) - 1:
        continue

      tap_key = self._pre(self.tap_keys[trans_num + 1], prefix)

      if tap_key in pour_outputs and not keep_connected:
        del pour_outputs[tap_key]

    return pour_outputs

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    """Pull out the original array from the funnel_dict which was produced by running pump.

    Parameters
    ----------
    funnel_dict : dict
      The dictionary outputted by running the transform's pump method. The keys are the names of the funnels and the values are the values of the tubes.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    np.ndarray
      The array reconstructed from the pump method.

    """

    trans = self.transforms[self.transform_order[0]]
    return trans._extract_pump_outputs(funnel_dict, prefix=os.path.join(prefix, self.name))

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

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      if key == 'transforms':
        continue
      save_dict[key] = getattr(self, key)

    save_dict['transforms'] = {}
    for key in self.transforms:
      save_dict['transforms'][key] = self.transforms[key]._save_dict()
    return save_dict

  def pump_examples(self, example_dicts, prefix=''):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    example_dicts: list of dicts of arrays
      The example dictionaries which the arrays associated with a single example.

    Returns
    -------
    np.ndarray
      The numpy array to transform into examples.

    """
    if type(example_dicts) is not dict:
      pour_outputs = self._array_dicts_to_array_dict(example_dicts)
    att_dict = self._get_array_attributes(prefix)
    array = None

    for trans_num, trans_key in enumerate(self.transform_order[::-1]):
      trans = self.transforms[trans_key]

      prefix = os.path.join(self.name, trans_key) + '/'

      # Build the pour outputs for this transform from the previous transform
      # array and any other relevant values in pour_outputs.
      kwargs = {}
      if trans_num != 0:
        tap_key = self._pre(self.tap_keys[-trans_num])
        kwargs[tap_key] = array
        # print tap_key, array.shape
      for output_name in pour_outputs:
        if not output_name.startswith(prefix):
          continue
        kwargs[output_name] = pour_outputs[output_name]
        kwargs[output_name] = kwargs[output_name].reshape([-1] + att_dict[output_name]['shape'])
        kwargs[output_name] = kwargs[output_name].astype(att_dict[output_name]['np_type'])
        # print output_name, kwargs[output_name].shape
      kwargs = self._nopre(kwargs)
      array = trans.pump_examples(kwargs)

    return array

  def _alter_pour_outputs(self, pour_outputs, prefix=''):
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
    r_pour_outputs = {}
    for key in self.transforms:
      trans = self.transforms[key]

      trans_pour_outputs = {k: v for k, v in pour_outputs.iteritems() if k.startswith(os.path.join(prefix, self.name, trans.name))}

      trans_pour_outputs = trans._alter_pour_outputs(trans_pour_outputs, prefix=os.path.join(prefix, self.name))
      r_pour_outputs.update(trans_pour_outputs)

    return r_pour_outputs

  def _parse_examples(self, arrays_dict, prefix=''):
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
    for trans_num, trans_key in enumerate(self.transform_order.reverse()):
      trans = self.transforms[key]
      # if array is not None:
      #   tap_key = self.tap_keys[-(trans_num + 1)]
      #   trans_arrays_dict[tap_key] = array
      trans_pour_outputs = trans._parse_examples(arrays_dict, prefix=os.path.join(prefix, self.name))
      pour_outputs.update(trans_pour_outputs)
    return pour_outputs

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(ChainTransform, self)._setattributes(**kwargs)
    if self.transforms is None:
      self.transforms = {}
      self.transform_order = []
      self.tap_keys = []

  def _get_array_attributes(self, prefix=''):
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
    att_dict = {}
    for key_num, key in enumerate(self.transform_order):
      trans = self.transforms[key]
      trans_att_dict = trans._get_array_attributes(prefix=os.path.join(prefix, self.name))

      if key_num != len(self.transform_order) - 1:
        tap_key = self._pre(self.tap_keys[key_num + 1], prefix='')
        del trans_att_dict[tap_key]

      att_dict.update(trans_att_dict)

    return att_dict

  def _get_funnel_dict(self, array=None, prefix=''):
    """Construct a dictionary where the keys are the names of the slots, and the values are either values from the Transform itself, or are taken from the supplied array.

    Parameters
    ----------
    array : np.ndarray
      The inputted array of raw information that is to be fed through the pour method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with all funnels filled with values necessary in order to run the pour method.

    """
    funnel_dict = {}
    # Get the funnel dicts from each of the transforms.
    for trans_num, name in enumerate(self.transform_order):
      trans = self.transforms[name]
      if trans_num != 0:
        array = None

      funnel_dict.update(
        trans._get_funnel_dict(array, prefix=self.name)
      )
    return funnel_dict

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
    array = None

    for trans_num, trans_key in enumerate(self.transform_order[::-1]):
      trans = self.transforms[trans_key]

      prefix = os.path.join(self.name, trans_key) + '/'

      # Build the pour outputs for this transform from the previous transform
      # array and any other relevant values in pour_outputs.
      kwargs = {}
      if trans_num != 0:
        tap_key = self._pre(self.tap_keys[-trans_num])
        kwargs[tap_key] = array
        # print tap_key, array.shape
      for output_name in pour_outputs:
        if not output_name.startswith(prefix):
          continue
        kwargs[output_name] = pour_outputs[output_name]
        # print output_name, kwargs[output_name].shape
      kwargs = self._nopre(kwargs)
      array = trans.pump(kwargs)

    return array

  def add_transform(self, transform, tap_key=None):
    """Add another transform to the dataset transform. The added transform must have a unique name.

    Parameters
    ----------
    transform : Transform
      The transform object that will operate on the subarray described by col_ranges.
    tap_key : str
      The key of the previous transform's tap that will be inputted as an array into
    """
    name = transform.name
    if name is None or name == '':
      raise ValueError("Transform must have it's name set, got: " + str(name))
    elif name in self.transforms:
      raise ValueError(str(name) + " already the name of a transform.")

    if self.transform_order and tap_key is None:
      raise ValueError(str(name) + " is not the first transform of the chain. Must specify a tap_key from the previous tranform as input into this transform.")

    self.transforms[name] = transform
    self.transform_order.append(name)
    self.tap_keys.append(tap_key)

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
    for trans_num, trans_key in enumerate(self.transform_order):
      trans = self.transforms[trans_key]

      # Calculate the global values for this transform.
      trans.calc_global_values(array)
      if trans_num >= len(self.transform_order) - 1:
        continue

      trans_outputs = trans.pour(array)

      tap_key = self.tap_keys[trans_num + 1]
      array = trans_outputs[tap_key]

  def define_waterwork(self, array=empty, return_tubes=None):
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
      for trans_num, trans_key in enumerate(self.transform_order):
        trans = self.transforms[trans_key]

        with ns.NameSpace(trans.name):
          if trans_num < len(self.transform_order) - 1:
            tap_key = self.tap_keys[trans_num + 1]
            return_tubes = [self._pre(tap_key)]
          else:
            return_tubes = None

          tubes = trans.define_waterwork(array, return_tubes)

          if tubes is None:
            continue

          old_name = tubes[0].name
          tubes[0].set_name("to_be_cloned")

          tube_dict, _ = td.clone(tubes[0])
          array = tube_dict['a']

          tube_dict['b'].set_name(old_name)

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
