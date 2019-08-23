import wtrwrks.tanks.utils as ut
import wtrwrks.tanks.bool as bo
import wtrwrks.tanks.clone as cl
import wtrwrks.tanks.add as ad
import wtrwrks.tanks.sub as su
import wtrwrks.tanks.mul as mu
import wtrwrks.tanks.div as dv
import wtrwrks.tanks.cat_to_index as cti
import wtrwrks.tanks.cast as ct
import wtrwrks.tanks.concatenate as co
import wtrwrks.tanks.reduce as rd
import wtrwrks.tanks.replace as rp
import wtrwrks.tanks.random_replace as rr
import wtrwrks.tanks.one_hot as oh
import wtrwrks.tanks.transpose as tr
import wtrwrks.tanks.pack as pc
import wtrwrks.tanks.pack_with_row_map as pw
import wtrwrks.tanks.datetime_to_num as dtn
import wtrwrks.tanks.tokenize as to
import wtrwrks.tanks.lower_case as lc
import wtrwrks.tanks.half_width as hw
import wtrwrks.tanks.lemmatize as lm
import wtrwrks.tanks.split as sp
import wtrwrks.tanks.partition as pa
import wtrwrks.tanks.iterate as it
import wtrwrks.tanks.flatten as fl
import wtrwrks.tanks.flat_tokenize as ft
import wtrwrks.tanks.replace_substring as rs
import wtrwrks.tanks.random_choice as rc
import wtrwrks.tanks.shape as sh
import wtrwrks.tanks.getitem as gi
import wtrwrks.tanks.reshape as rh
import wtrwrks.tanks.remove as rm
import wtrwrks.tanks.bert_random_insert as be
import wtrwrks.tanks.tile as tl
import wtrwrks.tanks.dim_size as ds
import wtrwrks.tanks.tube_iterables as ti
import wtrwrks.tanks.effective_length as el
import wtrwrks.tanks.print_val as pr
from wtrwrks.waterworks.empty import empty
import numpy as np


def add(a=empty, b=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None):
  """Add two objects together while returning extra information in order to be able to undo the operation. 'a' and 'b' must be able to be converted into numpy arrays.

  Parameters
  ----------
  a: np.ndarray
    The first object to add. Must be able to be converted into a numpy array.
  b: np.ndarray
    The second object to add. Must be able to be converted into a numpy array.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The result of a+b.
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether or not 'a' is the smaller size array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Add tank.
  slots: dict(
      a: np.ndarray
        The first object to add. Must be able to be converted into a numpy array.
      b: np.ndarray
        The second object to add. Must be able to be converted into a numpy array.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Add tank.

  """
  tank = ad.Add(a=a, b=b, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])

  return tank.get_tubes(), tank.get_slots()


def bert_random_insert(a=empty, ends=empty, num_tries=empty, random_seed=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """A specialized tank built specifically for the BERT ML model. Randomly inserts a [SEP] token at the end of some sentence in a row, then with some probability overwrites the latter part of the string with a randomly selected sentence. For more information or motivation look up the bert model.

  Parameters
  ----------
  a: np.ndarray
    The array that will have the [SEP] and [CLS] tags inserted as well as randomly setting half of the rows to having random sentences after the first [SEP] tag.
  ends: np.ndarray of bools
    An array of the same shape as 'a' which marks the end of a sentence with a True.
  num_tries: int
    The number of times to try and find a random sentences to replace the second part of the 'a' array.
  random_seed: int
    The random seed.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The array a with the [SEP] and [CLS] tags as well a some randomly overwritten second sentences.
    removed: np.ndarray
      A array with the same size as target that contains all the substrings that were overwritten.
    ends: np.ndarray of bools
      An array of the same shape as 'a' which marks the end of a sentence with a True.
    num_tries: int
      The number of times to try and find a random sentences to replace the second part of the 'a' array.
    segment_ids: np.ndarray
      An array of zeros and ones with the same shape as 'a' which says whether the token is part of the first sentence or the second.
    is_random_next: np.ndarray
      An array of bools which says whether the second sentence was replaced with a random sentence.
    random_seed: int
      The random seed.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Add tank.
  slots: dict(
      a: np.ndarray
        The array that will have the [SEP] and [CLS] tags inserted as well as randomly setting half of the rows to having random sentences after the first [SEP] tag.
      ends: np.ndarray of bools
        An array of the same shape as 'a' which marks the end of a sentence with a True.
      num_tries: int
        The number of times to try and find a random sentences to replace the second part of the 'a' array.
      random_seed: int
        The random seed.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Add tank.

  """
  tank = be.BertRandomInsert(a=a, ends=ends, num_tries=num_tries, random_seed=random_seed, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def cast(a=empty, dtype=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Cast an object to another dtype while saving the original dtype and any lost information.

  Parameters
  ----------
  a: np.ndarray
    The object to be casted to a new type. Must be able to be converted into a numpy array.
  dtype: a numpy dtype
    The type to cast the object to.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: The type specified by the dtype slot
      The result of casting 'a' to the new dtype.
    input_dtype: a numpy dtype
      The dtype of the original array.
    diff: The datatype of the original 'a' array
      The difference between the original 'a' and the casted array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Cast tank.
  slots: dict(
      a: np.ndarray
        The object to be casted to a new type. Must be able to be converted into a numpy array.
      dtype: a numpy dtype
        The type to cast the object to.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Cast tank.

  """
  tank = ct.Cast(a=a, dtype=dtype, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def cat_to_index(cats=empty, cat_to_index_map=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Convert an array of values drawn from a set of categories into an index according to the map cat_to_index_map, while keeping track of the values that aren't found in the map. Any values not found in the map are given -1 as an index.

  Parameters
  ----------
  cats: np.ndarray
    The array with all the category values to map to indices.
  cat_to_index_map: dict
    The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of ints
      The indices of all the corresponding category values from 'cats'.
    cat_to_index_map: dict
      The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
    missing_vals: list of category values
      All the category values from 'cats' which were not found in cat_to_index_map.
    input_dtype: a numpy dtype
      The dtype of the inputted 'cats' array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the CatToIndex tank.
  slots: dict(
      cats: np.ndarray
        The array with all the category values to map to indices.
      cat_to_index_map: dict
        The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the CatToIndex tank.

  """
  tank = cti.CatToIndex(cats=cats, cat_to_index_map=cat_to_index_map, waterwork=waterwork, name=name)
  # return tank['target'], tank['cat_to_index_map'], tank['missing_vals'], tank['input_dtype'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def clone(a=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Copy an object in order to send it to two different tanks. Usually not performed explicitly but rather when a tube is put as input into two different slots. A clone operation is automatically created.

  Parameters
  ----------
  a: object
    The object to be cloned into two.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    a: type of slot 'a'
      The first of the two cloned objects.
    b: type of slot 'a'
      The second of the two cloned objects.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Clone tank.
  slots: dict(
      a: object
        The object to be cloned into two.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Clone tank.

  """
  tank = cl.Clone(a=a, waterwork=waterwork, name=name)

  # return tank['a'], tank['b'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def clone_many(a=empty, num=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Copy an object in order to send it to two different tanks. Usually not performed explicitly but rather when a tube is put as input into two different slots. A clone operation is automatically created.

  Parameters
  ----------
  a: object
    The object to be cloned.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    a0: object
      zeroth clone
    a1: object
      first clone,
    .
    .
    .
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Clone tank.
  slots: dict(
      a: object
        The object to be cloned.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Clone tank.

  """
  # return tank['a'], tank['b'], tank.get_slots()
  keys = ['a' + str(i) for i in xrange(num)]

  class CloneManyTyped(cl.CloneMany):
    tube_keys = keys

  tank = CloneManyTyped(a=a, num=num, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])

  tubes = tank.get_tubes()
  return [tubes[tube_key] for tube_key in keys], tank.get_slots()


def concatenate(a_list=empty, axis=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Concatenate a np.array from subarrays along one axis while saving the indices of the places where they were merged so that the process can be reversed.

  Parameters
  ----------
  a_list: list of arrays
    The list of arrays to concatenate.
  axis: int
    The axis along which to concatenate.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The concatenation of 'a_list' along 'axis'
    axis: int
      The axis along which to concatenate.
    indices: np.ndarray
      The indices that mark the separation of arrays.
    dtypes: list of dtypes
      The dtypes of the original elements of 'a_list'. Must be of the same length as the orignal 'a_list.'
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Concatenate tank.
  slots: dict(
      a_list: list of arrays
        The list of arrays to concatenate.
      axis: int
        The axis along which to concatenate.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Concatenate tank.

  """
  tank = co.Concatenate(a_list=a_list, axis=axis, waterwork=waterwork, name=name)
  # return tank['target'], tank['axis'], tank['indices'], tank['dtypes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def datetime_to_num(a=empty, zero_datetime=empty, num_units=empty, time_unit=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Converts a datetime64 array to an array of numbers.

  Parameters
  ----------
  a: np.ndarray of datetime64
    The array of datetimes to be converted to numbers.
  zero_datetime: datetime64
    The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
  num_units: int
    This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
  time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
    This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of
      The array of datetimes that were converted to numbers.
    zero_datetime: datetime64
      The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
    num_units: int
      This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
      This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    diff: np.ndarray of timedelta64
      The difference between the original array 'a' and the array which lost information from taking everything according to a finite time resolution. e.g. For num_units=1 and time_unit='M' datetime(2000, 3, 4) gets mapped to datetime(2000, 3) so the diff would be a timedelta64 of 4 days.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the DatetimeToNum tank.
  slots: dict(
      a: np.ndarray of datetime64
        The array of datetimes to be converted to numbers.
      zero_datetime: datetime64
        The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
      num_units: int
        This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
      time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
        This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the DatetimeToNum tank.

  """
  tank = dtn.DatetimeToNum(a=a, zero_datetime=zero_datetime, num_units=num_units, time_unit=time_unit, waterwork=waterwork, name=name)

  # return tank['target'], tank['zero_datetime'], tank['num_units'], tank['time_unit'], tank['diff'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def dim_size(a=empty, axis=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Get the size of a dimension of an array.

  Parameters
  ----------
  a: np.ndarray
    The array to get the shape of
  axis: int
    The axis to get the size of.
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: list of ints
      The shape of the array.
    a: np.ndarray
      The array to get the shape of
    axis: int
      The axis to get the dim_size from.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to get the shape of
      axis: int
        The axis to get the dim_size from.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = ds.DimSize(a=a, axis=axis, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def div(a=empty, b=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Divide two objects together while returning extra information in order to be able to undo the operation. 'a' and 'b' must be able to be converted into numpy arrays.

  Parameters
  ----------
  a: np.ndarray
    The numerator array.
  b: np.ndarray
    The denominator array

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The result of a/b
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether a is the smaller sized array.
    missing_vals: np.ndarray
      The values from either 'a' or 'b' that were lost when the other array had a zero in that location.
    remainder: np.ndarray
      The remainder of a/b in the case that 'a' and 'b' are of integer type.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Div tank.
  slots: dict(
      a: np.ndarray
        The numerator array.
      b: np.ndarray
        The denominator array
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Div tank.

  """
  tank = dv.Div(a=a, b=b, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def do_nothing(a=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Copy an object in order to send it to two different tanks. Usually not performed explicitly but rather when a tube is put as input into two different slots. A clone operation is automatically created.

  Parameters
  ----------
  a: object
    The object to be cloned into two.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    a: type of slot 'a'
      The first of the two cloned objects.
    b: type of slot 'a'
      The second of the two cloned objects.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Clone tank.
  slots: dict(
      a: object
        The object to be cloned into two.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Clone tank.

  """
  tank = cl.DoNothing(a=a, waterwork=waterwork, name=name)

  # return tank['a'], tank['b'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def effective_length(a=empty, default_val=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Get the length of the last dimension, not including the default_val.

  Parameters
  ----------
  a: np.ndarray
    The array to get the effective length of.
  default_val:
    The value to not count
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    a: np.ndarray
      The array to get the effective length of.
    default_val:
      The value to not count
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      target: np.ndarray
        An array of the same shape as 'a' except missing the last dimension. The values are effective lengths of the last dimesion of a.
      a: np.ndarray
        The array to get the effective length of.
      default_val:
        The value to not count
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = el.EffectiveLength(a=a, default_val=default_val, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def equals(a=empty, b=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Get the length of the last dimension, not including the default_val.

  Parameters
  ----------
  a: np.ndarray
    The array to get the effective length of.
  default_val:
    The value to not count
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    a: np.ndarray
      The array to get the effective length of.
    default_val:
      The value to not count
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      target: np.ndarray
        An array of the same shape as 'a' except missing the last dimension. The values are effective lengths of the last dimesion of a.
      a: np.ndarray
        The array to get the effective length of.
      default_val:
        The value to not count
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = bo.Equals(a=a,  b=b, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def flat_tokenize(strings=empty, ids=empty, tokenizer=empty, detokenizer=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Tokenize an array of strings according to the supplied tokenizer function, but instead of keeping the original structure of the inputted array, flatten the array and put all tokens from all strings on the same axis.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings to tokenize.
  tokenizer: func
    Function which converts a string into a list of strings.
  detokenizer: func
    Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
  ids: np.ndarray
    An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings since all information about axis is lost when flattened. Each id from ids must be unique.The array of is the same shape as strings

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      A one dimensional array of tokens.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
    shape: list of ints
      The shape of the inputted array.
    ids: np.ndarray
      An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings. The array of is the same shape as target
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the FlatTokenize tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings to tokenize.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      ids: np.ndarray
        An array of ids which uniquely identify each element of 'strings'. Necessary in order to reconstruct strings since all information about axis is lost when flattened. Each id from ids must be unique.The array of is the same shape as strings
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the FlatTokenize tank.

  """
  tank = ft.FlatTokenize(strings=strings, tokenizer=tokenizer, detokenizer=detokenizer, ids=ids)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def flatten(a=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Flatten a multidimensional array into 1D.

  Parameters
  ----------
  a: np.ndarray
    The array to be flattened

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: dtype of 'a'
      The flattened array.
    shape: list of ints.
      The original shape of 'a'.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Flatten tank.
  slots: dict(
      a: np.ndarray
        The array to be flattened
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Flatten tank.

  """
  tank = fl.Flatten(a=a)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def getitem(a=empty, key=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Run __getitem__ on some object (e.g. a dictionary) to return some value.

  Parameters
  ----------
  a: object
    The object to getitem from.
  key: hashable
    The key to pass to the getitem
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: object
      The value returned from the __getitem__ call to 'a'.
    a: object
      The object to getitem from.
    key: hashable
      The key to pass to the getitem
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: object
        The object to getitem from.
      key: hashable
        The key to pass to the getitem
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = gi.GetItem(a=a, key=key, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def half_width(strings=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Convert any unicode (e.g. chinese, japanese, korean, etc.) characters in a array from full width to half width.

  Parameters
  ----------
  strings: np.ndarray of unicode
    The array of unicode characters to be converted to half width

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of unicode
      The array of half widthed strings.
    diff: np.darray of unicode
      The string difference between the original strings and the half widthed strings.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the HalfWidth tank.
  slots: dict(
      strings: np.ndarray of unicode
        The array of unicode characters to be converted to half width
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the HalfWidth tank.

  """
  tank = hw.HalfWidth(strings=strings, waterwork=waterwork, name=name)
  # return tank['target'], tank['diff'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def iter_dict(a=empty, keys=None, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Create a dictionary of tubes from a tube which is dictionary valued. Necessary if one wants to operate on the individual values of a dictionary rather then the entire dictionary.

  Parameters
  ----------
  a: dict
    The tube dictionary. whose values will be converted to tubes.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(

  )
    A dictionary where the keys are the tube names and the values are the tube objects of the IterDict tank.
  slots: dict(
      a: dict
        The tube dictionary. whose values will be converted to tubes.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the IterDict tank.

  """
  if keys is None:
    raise ValueError("Must explicitly set keys.")

  class IterDictTyped(it.IterDict):
    tube_keys = keys
  tank = IterDictTyped(a=a, waterwork=waterwork, name=name)

  tubes = tank.get_tubes()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  return {tube_key: tubes[tube_key] for tube_key in keys}, tank.get_slots()


def _iter_dict(a=empty, keys=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Used internally. Do not use."""

  class IterDictTyped(it.IterDict):
    tube_keys = keys
  tank = IterDictTyped(a=a, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def iter_list(a=empty, num_entries=None, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Create a list of tubes from a tube which is list valued. Necessary if one wants to operate on the individual elements of a list rather then the entire list.

  Parameters
  ----------
  a: list
    The tube list whose elements will be converted to tubes.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(

  )
    A dictionary where the keys are the tube names and the values are the tube objects of the IterList tank.
  slots: dict(
      a: list
        The tube list whose elements will be converted to tubes.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the IterList tank.

  """
  if num_entries is None:
    raise ValueError("Must explicitly set num_entries.")

  keys = ['a' + str(i) for i in xrange(num_entries)]

  class IterListTyped(it.IterList):
    tube_keys = keys
  tank = IterListTyped(a=a, waterwork=waterwork, name=name)

  tubes = tank.get_tubes()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  return [tubes[tube_key] for tube_key in keys], tank.get_slots()


def _iter_list(a=empty, num_entries=None, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Use only for internal purposes. Do not use."""
  keys = ['a' + str(i) for i in xrange(num_entries)]

  class IterListTyped(it.IterList):
    tube_keys = keys
  tank = IterListTyped(a=a, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def lemmatize(strings=empty, lemmatizer=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Standardize the strings of an array according to the supplied lemmatizer function.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings to be lemmatized.
  lemmatizer: func
    A function which takes in a string and outputs a standardized version of that string

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of strings
      The array of lemmatized strings
    lemmatizer: func
      A function which takes in a string and outputs a standardized version of that string.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been lemmatized.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Lemmatize tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings to be lemmatized.
      lemmatizer: func
        A function which takes in a string and outputs a standardized version of that string
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Lemmatize tank.

  """
  tank = lm.Lemmatize(strings=strings, lemmatizer=lemmatizer)
  # return tank['target'], tank['lemmatizer'], tank['diff'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def logical_not(a=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Take the 'not' of boolean array.

  Parameters
  ----------
  a: np.ndarray of bools
    The array to take the logical not of.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of bools.
      The negated array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the LogicalNot tank.
  slots: dict(
      a: np.ndarray of bools
        The array to take the logical not of.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the LogicalNot tank.

  """
  tank = bo.LogicalNot(a=a, waterwork=waterwork, name=name)
  # return tank['target'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def lower_case(strings=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Convert any upper case substrings of an array into their lower case counterparts.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings to lower case

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of strings
      The array of lower cased strings.
    diff: np.ndarray of strings
      The string difference between the original strings and the lower cased strings.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the LowerCase tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings to lower case
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the LowerCase tank.

  """
  tank = lc.LowerCase(strings=strings, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def merge_equal(*args, **kwargs):
  """Merge several equal objects into a single object. All tubes must have equal values, otherwise you will get unexpected results. (Opposite of clone_many)

  Parameters
  ----------
  args: list
    list of tube objects
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: object
      The merged object. Simply takes the value of the first in the list.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the IterList tank.
  slots: dict(
    args: list
      list of tube objects
  )

  """
  waterwork = None
  if 'waterwork' in kwargs:
    waterwork = kwargs['waterwork']
    del kwargs['waterwork']
  name = None
  if 'name' in kwargs:
    name = kwargs['name']
    del kwargs['name']

  temp_test_equal = False
  if 'test_equal' in kwargs:
    temp_test_equal = kwargs['test_equal']
    del kwargs['test_equal']

  slot_plugs = None
  if 'slot_plugs' in kwargs:
    slot_plugs = kwargs['slot_plugs']
    del kwargs['slot_plugs']

  tube_plugs = None
  if 'tube_plugs' in kwargs:
    tube_plugs = kwargs['tube_plugs']
    del kwargs['tube_plugs']

  slot_names = None
  if 'slot_names' in kwargs:
    slot_names = kwargs['slot_names']
    del kwargs['slot_names']

  tube_names = None
  if 'tube_names' in kwargs:
    tube_names = kwargs['tube_names']
    del kwargs['tube_names']

  keys = ['a' + str(i) for i in xrange(len(args))]
  kwargs = {}
  for num, key in enumerate(keys):
    kwargs[key] = args[num]

  class MergeEqualTyped(cl.MergeEqual):
    slot_keys = keys
    test_equal = temp_test_equal

  tank = MergeEqualTyped(waterwork=waterwork, name=name, **kwargs)
  tank.waterwork.merge_tubes(tank.get_tubes()['target'], *args)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])

  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def mul(a=empty, b=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Multiply two objects together while returning extra information in order to be able to undo the operation. 'a' and 'b' must be able to be converted into numpy arrays.

  Parameters
  ----------
  a: np.ndarray
    The first array to be multiplied
  b: np.ndarray
    The second array to be multiplied

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The result of a*b
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether a is the smaller sized array.
    missing_vals: np.ndarray
      The values from either 'a' or 'b' that were lost when the other array had a zero in that location.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Mul tank.
  slots: dict(
      a: np.ndarray
        The first array to be multiplied
      b: np.ndarray
        The second array to be multiplied
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Mul tank.

  """

  tank = mu.Mul(a=a, b=b, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def multi_cat_to_index(cats=empty, selector=empty, cat_to_index_maps=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Convert an array of values drawn from a set of categories into an index according to the map cat_to_index_map, while keeping track of the values that aren't found in the map. Any values not found in the map are given -1 as an index.

  Parameters
  ----------
  cats: np.ndarray
    The array with all the category values to map to indices.
  cat_to_index_map: dict
    The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of ints
      The indices of all the corresponding category values from 'cats'.
    cat_to_index_map: dict
      The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
    missing_vals: list of category values
      All the category values from 'cats' which were not found in cat_to_index_map.
    input_dtype: a numpy dtype
      The dtype of the inputted 'cats' array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the CatToIndex tank.
  slots: dict(
      cats: np.ndarray
        The array with all the category values to map to indices.
      cat_to_index_map: dict
        The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the CatToIndex tank.

  """
  tank = cti.MultiCatToIndex(cats=cats, selector=selector, cat_to_index_maps=cat_to_index_maps, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def multi_isin(a=empty, bs=empty, selector=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  tank = bo.MultiIsIn(a=a, bs=bs, selector=selector, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  return tank.get_tubes(), tank.get_slots()


def multi_tokenize(strings=empty, selector=empty, tokenizers=empty, max_len=empty, detokenizers=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Tokenize an array of strings according to the supplied tokenizer function, keeping the original shape of the array of strings but adding an additional 'token' dimension.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings to tokenize.
  tokenizer: func
    Function which converts a string into a list of strings.
  detokenizer: func
    Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
  max_len: int
    The maximum number of tokens. Defines the size of the added dimension.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The array of tokenized strings. Will have rank = rank('a') + 1 where the last dimesion will have size max_len.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Tokenize tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings to tokenize.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      max_len: int
        The maximum number of tokens. Defines the size of the added dimension.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Tokenize tank.

  """
  tank = to.MultiTokenize(strings=strings, selector=selector, tokenizers=tokenizers, max_len=max_len, detokenizers=detokenizers, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def one_hot(indices=empty, depth=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Convert an array of indices of rank n to an array of 1's and 0's of rank n+1, where there is a 1 in the location specified by the index and zeros everywhere else.

  Parameters
  ----------
  indices: np.ndarray of ints
    The array of indices to be one hotted.
  depth: int
    The maximum allowed index value and the size of the n + 1 dimension of the outputted array.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The one hotted array.
    missing_vals: list of ints
      The indices which were not in the range of 0 <= i < depth
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the OneHot tank.
  slots: dict(
      indices: np.ndarray of ints
        The array of indices to be one hotted.
      depth: int
        The maximum allowed index value and the size of the n + 1 dimension of the outputted array.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the OneHot tank.

  """
  tank = oh.OneHot(indices=indices, depth=depth, waterwork=waterwork, name=name)
  # return tank['target'], tank['missing_vals'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def pack(a=empty, lengths=empty, default_val=empty, max_group=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """More efficiently pack in the data of an array by overwriting the default_val's. The array must have rank at least equal to 2 The last dimension will be packed so that fewer default vals appear, and the next to last dimension with be shortened, any other dimensions are left unchanged.
  e.g.

  default_val = 0
  a = np.array([
    [1, 1, 1, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [3, 3, 0, 3, 3, 0],
    [4, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 0, 0, 0, 0],
    [7, 7, 0, 0, 0, 0]
  ])

  target = np.array([
    [1, 1, 1, 2, 2, 0],
    [3, 3, 3, 3, 4, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 7, 7, 0, 0]
  ])

  Parameters
  ----------
  a : np.ndarray
    The array to pack
  lengths: np.ndarray
    The of lengths of 'valid' data. The not valid data will be overwritten when it's packed together.
  max_group: int
    Maximum number of original rows of data packed into a single row.
  default_val: np.ndarray.dtype
    The value that will be allowed to be overwritten in the packing process.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
    ends: np.ndarray
      The endpoints of all the original rows within the packed array.
    row_map: np.ndarray
      A mapping from the new rows to the original rows.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    max_group: int
      Maximum number of original rows of data packed into a single row.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to pack
      lengths: np.ndarray
        The of lengths of 'valid' data. The not valid data will be overwritten when it's packed together.
      max_group: int
        Maximum number of original rows of data packed into a single row.
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = pc.Pack(a=a, default_val=default_val, lengths=lengths, max_group=max_group, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def pack_with_row_map(a=empty, row_map=empty, default_val=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """More efficiently pack in the data of an array by overwriting the default_val's. The array must have rank at least equal to 2 The last dimension will be packed so that fewer default vals appear, and the next to last dimension with be shortened, any other dimensions are left unchanged.
  e.g.

  default_val = 0
  a = np.array([
    [1, 1, 1, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [3, 3, 0, 3, 3, 0],
    [4, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 0, 0, 0, 0],
    [7, 7, 0, 0, 0, 0]
  ])

  target = np.array([
    [1, 1, 1, 2, 2, 0],
    [3, 3, 3, 3, 4, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 7, 7, 0, 0]
  ])

  Parameters
  ----------
  a: np.ndarray
    The array to pack
  default_val: np.ndarray.dtype
    The value that will be allowed to be overwritten in the packing process.
  row_map: np.ndarray
    A mapping from the new rows to the original rows.
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    row_map: np.ndarray
      A mapping from the new rows to the original rows.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to pack
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
      row_map: np.ndarray
        A mapping from the new rows to the original rows.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = pw.PackWithRowMap(a=a, default_val=default_val, row_map=row_map, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def partition(a=empty, ranges=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Create a list of array slices according to the ranges. All slices are ranges of the 0th axis of the array.

  Parameters
  ----------
  a: np.ndarray
    The array to take slices from.
  ranges: np.ndarray
    The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: list of arrays
      The list of array slices. The length of the list is equal to the size of the first dimension of 'ranges'.
    ranges: np.ndarray of ints
      The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.
    missing_cols: np.ndarray of ints
      The columns of the array that were not selected by the slices defined by 'ranges'
    missing_array: np.ndarray
      The slices of array that were not selected by the slices defined by 'ranges'.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Partition tank.
  slots: dict(
      a: np.ndarray
        The array to take slices from.
      ranges: np.ndarray
        The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Partition tank.

  """
  tank = pa.Partition(a=a, ranges=ranges, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def partition_by_index(a=empty, indices=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Create a list of array slices according to the indices. All slices are ranges of the 0th axis of the array.

  Parameters
  ----------
  a: np.ndarray
    The array to take slices from.
  indices: np.ndarray
    The indices of all the slices.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: list of arrays
      The list of array slices. The length of the list is equal to the size of the first dimension of 'indices'.
    indices: np.ndarray of ints
      The indices of all the slices.
    missing_cols: np.ndarray of ints
      The columns of the array that were not selected by the slices defined by 'indices'
    missing_array: np.ndarray
      The slices of array that were not selected by the slices defined by 'indices'.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Partition tank.
  slots: dict(
      a: np.ndarray
        The array to take slices from.
      indices: np.ndarray
        The indices of all the slices.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Partition tank.

  """
  tank = pa.PartitionByIndex(a=a, indices=indices, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def print_val(**kwargs):
  """Convert a list of Tubes into a single Tube. Usually called by the waterwork object.

  Parameters
  ----------
  kwargs:
    The arguments to print.
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    kwargs:
      The orignal arguments.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the IterList tank.
  slots: dict(
    kwargs:
      The arguments to print.
  )

  """
  waterwork = None
  if 'waterwork' in kwargs:
    waterwork = kwargs['waterwork']
    del kwargs['waterwork']
  name = None
  if 'name' in kwargs:
    name = kwargs['name']
    del kwargs['name']

  keys = sorted(kwargs)

  class PrintTyped(pr.Print):
    slot_keys = keys
    tube_keys = keys
    pass_through_keys = keys
  tank = PrintTyped(waterwork=waterwork, name=name, **kwargs)

  return tank.get_tubes(), tank.get_slots()


def random_choice(a=empty, shape=empty, p=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Randomly select values from a list to fill some array.

  Parameters
  ----------
  a: 1D np.ndarray or int
    The allowed values for to randomly select from
  shape: list of ints
    The shape of the outputted array of random values.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The randomly selected values
    a: 1D np.ndarray or int
      The allowed values for to randomly select from
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: 1D np.ndarray or int
        The allowed values for to randomly select from
      shape: list of ints
        The shape of the outputted array of random values.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = rc.RandomChoice(a=a, shape=shape, p=p, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def random_replace(a=empty, replace_with=empty, prob=empty, max_replace=None, do_not_replace_vals=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Replace the values of an array with some other values specified by replace_with.

  Parameters
  ----------
  a: np.ndarray
    The array which has values that are to be replaced.
  replace_with: np.ndarray
    The value to be used to replace the corresponding values in 'a'.
  prob: 0 <= float <= 1
    The probability that each value is replaced
  max_replace: int <= a.shape[-1]
    The maximum allowed replacements along the last dimension.
  do_not_replace_vals: np.ndarray
    Values to skip when randomly replacing.
  max_replace: int
    The maximum number of allowed replacements in the last dimension.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of same type as 'a'
      The array with the necessary values replaced.
    mask_mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values were replaced.
    mask_positions: np.ndarray of bools
      The positions of the masked values
    prob: 0 <= float <= 1
      The probability that each value is replaced
    replaced_vals: np.ndarray of same type as 'a'
      The values that were overwritten when they were replaced by the replace_with values.
    do_not_replace_vals: np.ndarray
      Values to skip when randomly replacing.
    max_replace: int <= a.shape[-1]
      The maximum allowed replacements along the last dimension.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Replace tank.
  slots: dict(
      a: np.ndarray
        The array which has values that are to be replaced.
      replace_with: np.ndarray
        The value to be used to replace the corresponding values in 'a'.
      prob: 0 <= float <= 1
        The probability that each value is replaced
      max_replace: int <= a.shape[-1]
        The maximum allowed replacements along the last dimension.
      do_not_replace_vals: np.ndarray
        Values to skip when randomly replacing.
      max_replace: int
        The maximum number of allowed replacements in the last dimension.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Replace tank.

  """
  if do_not_replace_vals is None:
    do_not_replace_vals = []

  tank = rr.RandomReplace(a=a, replace_with=replace_with, prob=prob, max_replace=max_replace, do_not_replace_vals=do_not_replace_vals, waterwork=waterwork, name=name)
  # return tank['target'], tank['mask'], tank['replaced_vals'], tank['replace_with_shape'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def replace(a=empty, mask=empty, replace_with=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Replace the values of an array with some other values specified by replace_with.

  Parameters
  ----------
  a: np.ndarray
    The array which has values that are to be replaced.
  mask: np.ndarray of bools
    An array of booleans whose True values denote which of array 'a's values are to be replaced.
  replace_with: np.ndarray
    The values to be used to replace the corresponding values in 'a'.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of same type as 'a'
      The array with the necessary values replaced.
    mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values are to be replaced.
    replaced_vals: np.ndarray of same type as 'a'
      The values that were overwritten when they were replaced by the replace_with values.
    replace_with_shape: list of ints
      The original shape of the replace_with array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Replace tank.
  slots: dict(
      a: np.ndarray
        The array which has values that are to be replaced.
      mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values are to be replaced.
      replace_with: np.ndarray
        The values to be used to replace the corresponding values in 'a'.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Replace tank.

  """
  tank = rp.Replace(a=a, mask=mask, replace_with=replace_with, waterwork=waterwork, name=name)
  # return tank['target'], tank['mask'], tank['replaced_vals'], tank['replace_with_shape'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def replace_substring(strings=empty, old_substring=empty, new_substring=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Replace old_substring in an array with a new_substring.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings that will have it's substrings replaced.
  old_substring: str or unicode
    The substring to be replaced.
  new_substring: str or unicode
    The substring to replace with.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray of same type as 'a'
      The of array of strings with the substrings replaced.
    old_substring: str or unicode
      The substring to be replaced.
    new_substring: str or unicode
      The substring to replace with.
    diff: np.ndarray of strings
      The diff of the strings caused by converting all old_substrings to new_substrings and back.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the ReplaceSubstring tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings that will have it's substrings replaced.
      old_substring: str or unicode
        The substring to be replaced.
      new_substring: str or unicode
        The substring to replace with.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the ReplaceSubstring tank.

  """
  tank = rs.ReplaceSubstring(strings=strings, old_substring=old_substring, new_substring=new_substring, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def remove(a=empty, mask=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Remove elements of an array according to a mask.

  Parameters
  ----------
  a: np.ndarray
    The array to execute the remove on
  mask: np.ndarray
    An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The array with elements removed
    removed: np.ndarray
      The remove elements of the array
    mask: np.ndarray
      An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to execute the remove on
      mask: np.ndarray
        An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = rm.Remove(a=a, mask=mask, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def reshape(a=empty, shape=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Get the shape of an array.

  Parameters
  ----------
  a: np.ndarray
    The array to reshape
  shape : list of ints
    The new shape of the array.
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The reshaped array
    old_shape: list of ints
      The old shape of the array
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to reshape
      shape : list of ints
        The new shape of the array.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = rh.Reshape(a=a, shape=shape, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def shape(a=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Get the shape of an array.

  Parameters
  ----------
  aa: np.ndarray
    The array to get the shape of
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: list of ints
      The shape of the array.
    a: np.ndarray
      The array to get the shape of
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to get the shape of
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = sh.Shape(a=a, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def split(a=empty, indices=empty, axis=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Split up an array along an axis according to provided indices. For a more detailed description look at the documentation for the corresponding numpy function.

  Parameters
  ----------
  a: np.ndarray
    The array to split up.
  indices: np.ndarray
    The indices of the points to split up the array.
  axis: int
    The axis along which to split up the array.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: list of arrays
      The list of split up arrays.
    indices: np.ndarray
      The indices of the points to split up the array.
    axis: int
      The axis along which to split up the array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Split tank.
  slots: dict(
      a: np.ndarray
        The array to split up.
      indices: np.ndarray
        The indices of the points to split up the array.
      axis: int
        The axis along which to split up the array.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Split tank.

  """
  tank = sp.Split(a=a, indices=indices, axis=axis, waterwork=waterwork, name=name)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def sub(a=empty, b=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Subtract two objects together while returning extra information in order to be able to undo the operation. 'a' and 'b' must be able to be converted into numpy arrays.

  Parameters
  ----------
  a: np.ndarray
    The object to subtract something from.
  b: np.ndarray
    The object which substracts from something else.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The result of a-b.
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether or not 'a' is the smaller size array.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Sub tank.
  slots: dict(
      a: np.ndarray
        The object to subtract something from.
      b: np.ndarray
        The object which substracts from something else.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Sub tank.

  """
  tank = su.Sub(a=a, b=b, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def tile(a=empty, reps=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Tile the elements of an array into an array with a shape defined by reps.

  Parameters
  ----------
  a: np.ndarray
    The array to reshape
  reps : list of ints
    The number of times to tile the array in each dimension

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The reshaped array
    old_shape: list of ints
      The old shape of the array
    reps : list of ints
      The number of times to tile the array in each dimension
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Sub tank.
  slots: dict(
      a: np.ndarray
        The array to reshape
      reps : list of ints
        The number of times to tile the array in each dimension
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Sub tank.

  """
  tank = tl.Tile(a=a, reps=reps, waterwork=waterwork, name=name)
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def tokenize(strings=empty, tokenizer=empty, max_len=empty, detokenizer=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Tokenize an array of strings according to the supplied tokenizer function, keeping the original shape of the array of strings but adding an additional 'token' dimension.

  Parameters
  ----------
  strings: np.ndarray of strings
    The array of strings to tokenize.
  tokenizer: func
    Function which converts a string into a list of strings.
  detokenizer: func
    Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
  max_len: int
    The maximum number of tokens. Defines the size of the added dimension.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The array of tokenized strings. Will have rank = rank('a') + 1 where the last dimesion will have size max_len.
    tokenizer: func
      Function which converts a string into a list of strings.
    detokenizer: func
      Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
    diff: np.ndarray of strings
      The array of strings which define the differences between the original string and the string that has been tokenized then detokenized.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Tokenize tank.
  slots: dict(
      strings: np.ndarray of strings
        The array of strings to tokenize.
      tokenizer: func
        Function which converts a string into a list of strings.
      detokenizer: func
        Function which takens in a list of tokens and returns a string. Not strictly necessary but it makes the tube 'diff' much smaller if it's close to the real method of detokenizing.
      max_len: int
        The maximum number of tokens. Defines the size of the added dimension.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Tokenize tank.

  """
  tank = to.Tokenize(strings=strings, tokenizer=tokenizer, max_len=max_len, detokenizer=detokenizer, waterwork=waterwork, name=name)
  # return tank['target'], tank['tokenizer'], tank['delimiter'], tank['diff'], tank.get_slots()

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def transpose(a=empty, axes=empty, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
  """Permute the dimensions of array while saving the permutation so that it can be undone.

  Parameters
  ----------
  a: np.ndarray
    The array to be transposed.
  axes: list of ints
    The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target: np.ndarray
      The transposed array.
    axes: list of ints
      The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the Transpose tank.
  slots: dict(
      a: np.ndarray
        The array to be transposed.
      axes: list of ints
        The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Transpose tank.

  """
  tank = tr.Transpose(a=a, axes=axes, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def tube_list(*args, **kwargs):
  """Convert a list of Tubes into a single Tube. Usually called by the waterwork object.

  Parameters
  ----------
  l: list
    list of tube objects
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  tubes: dict(
    target:
      The tube of the list.
  )
    A dictionary where the keys are the tube names and the values are the tube objects of the IterList tank.
  slots: dict(
    l: list
      list of tube objects
  )

  """
  waterwork = None
  if 'waterwork' in kwargs:
    waterwork = kwargs['waterwork']
    del kwargs['waterwork']
  name = None
  if 'name' in kwargs:
    name = kwargs['name']
    del kwargs['name']

  slot_plugs = None
  if 'slot_plugs' in kwargs:
    slot_plugs = kwargs['slot_plugs']
    del kwargs['slot_plugs']

  tube_plugs = None
  if 'tube_plugs' in kwargs:
    tube_plugs = kwargs['tube_plugs']
    del kwargs['tube_plugs']

  slot_names = None
  if 'slot_names' in kwargs:
    slot_names = kwargs['slot_names']
    del kwargs['slot_names']

  tube_names = None
  if 'tube_names' in kwargs:
    tube_names = kwargs['tube_names']
    del kwargs['tube_names']

  keys = ['a' + str(i) for i in xrange(len(args))]
  kwargs = {}
  for num, key in enumerate(keys):
    kwargs[key] = args[num]

  class TubeListTyped(ti.TubeList):
    slot_keys = keys
  tank = TubeListTyped(waterwork=waterwork, name=name, **kwargs)

  if slot_plugs is not None:
    for key in slot_plugs:
      tank.get_slots()[key].set_plug(slot_plugs[key])
  if tube_plugs is not None:
    for key in tube_plugs:
      tank.get_tubes()[key].set_plug(tube_plugs[key])
  if slot_names is not None:
    for key in slot_names:
      tank.get_slots()[key].set_name(slot_names[key])
  if tube_names is not None:
    for key in tube_names:
      tank.get_tubes()[key].set_name(tube_names[key])
  return tank.get_tubes(), tank.get_slots()


def _tube_list(**kwargs):
  """Used internally. Do not use."""
  waterwork = None
  if 'waterwork' in kwargs:
    waterwork = kwargs['waterwork']
    del kwargs['waterwork']
  name = None
  if 'name' in kwargs:
    name = kwargs['name']
    del kwargs['name']
  keys = ['a' + str(i) for i in xrange(len(kwargs))]

  class TubeListTyped(ti.TubeList):
    slot_keys = keys
  tank = TubeListTyped(waterwork=waterwork, name=name, **kwargs)

  return tank.get_tubes(), tank.get_slots()


isnan = bo.create_one_arg_bool_tank(np.isnan, class_name='IsNan', func_name='isnan')
isnat = bo.create_one_arg_bool_tank(np.isnat, class_name='IsNat', func_name='isnat')
# equal = bo.create_two_arg_bool_tank(np.equal, class_name='Equals', func_name='equal')
greater = bo.create_two_arg_bool_tank(np.greater, class_name='Greater', func_name='greater')
greater_equal = bo.create_two_arg_bool_tank(np.greater_equal, class_name='GreaterEqual', func_name='greater_equal')
less = bo.create_two_arg_bool_tank(np.less, class_name='Less', func_name='less')
less_equal = bo.create_two_arg_bool_tank(np.less_equal, class_name='LessEqual', func_name='less_equal')
isin = bo.create_two_arg_bool_tank(np.isin, class_name='IsIn', func_name='isin')

max = rd.create_one_arg_reduce_tank(np.max, class_name='Max', func_name='max')
min = rd.create_one_arg_reduce_tank(np.min, class_name='Min', func_name='min')
sum = rd.create_one_arg_reduce_tank(np.sum, class_name='Sum', func_name='sum')
mean = rd.create_one_arg_reduce_tank(np.mean, class_name='Mean', func_name='mean')
std = rd.create_one_arg_reduce_tank(np.std, class_name='Std', func_name='std')
all = rd.create_one_arg_reduce_tank(np.all, class_name='All', func_name='all')
any = rd.create_one_arg_reduce_tank(np.any, class_name='Any', func_name='any')
