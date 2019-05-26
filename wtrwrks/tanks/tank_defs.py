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
import wtrwrks.tanks.one_hot as oh
import wtrwrks.tanks.transpose as tr
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
from wtrwrks.waterworks.empty import empty
import numpy as np


def add(a=empty, b=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def cast(a=empty, dtype=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def cat_to_index(cats=empty, cat_to_index_map=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def clone(a=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def concatenate(a_list=empty, axis=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()

def datetime_to_num(a=empty, zero_datetime=empty, num_units=empty, time_unit=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def div(a=empty, b=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def flat_tokenize(strings=empty, ids=empty, tokenizer=empty, detokenizer=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def flatten(a=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def half_width(strings=empty, waterwork=None, name=None):
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
  tank = hw.HalfWidth(strings=strings)
  # return tank['target'], tank['diff'], tank.get_slots()
  return tank.get_tubes(), tank.get_slots()


def iter_dict(a=empty, keys=None, type_dict=None, waterwork=None, name=None):
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
    raise ValueError("Must explicitly set num_entries.")

  class IterDictTyped(it.IterDict):
    tube_keys = keys
  tank = IterDictTyped(a=a, waterwork=waterwork, name=name)

  tubes = tank.get_tubes()
  return {tube_key: tubes[tube_key] for tube_key in keys}, tank.get_slots()
  return {tube_key: tubes[tube_key] for tube_key in keys}, tank.get_slots(), tank


def iter_list(a=empty, num_entries=None, type_dict=None, waterwork=None, name=None):
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
  return [tubes[tube_key] for tube_key in keys], tank.get_slots()


def lemmatize(strings=empty, lemmatizer=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def logical_not(a=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def lower_case(strings=empty, waterwork=None, name=None):
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
  tank = lc.LowerCase(strings=strings)
  return tank.get_tubes(), tank.get_slots()


def mul(a=empty, b=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def one_hot(indices=empty, depth=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def partition(a=empty, indices=empty, type_dict=None, waterwork=None, name=None):
  """Create a list of array slices according to the indices. All slices are ranges of the 0th axis of the array.

  Parameters
  ----------
  a: np.ndarray
    The array to take slices from.
  indices: np.ndarray
    The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.

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
      The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.
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
        The index ranges of each slice. The first dimension can be of any size and the second dimension must be two.
  )
    A dictionary where the keys are the slot names and the values are the slot objects of the Partition tank.

  """
  tank = pa.Partition(a=a, indices=indices, waterwork=waterwork, name=name)

  return tank.get_tubes(), tank.get_slots()


def replace(a=empty, mask=empty, replace_with=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


def replace_substring(strings=empty, old_substring=empty, new_substring=empty, waterwork=None, name=None):
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
  tank = rs.ReplaceSubstring(strings=strings, old_substring=old_substring, new_substring=new_substring)
  # return tank['target'], tank['old_substring'], tank['new_substring'], tank['diff'], tank.get_slots()
  return tank.get_tubes(), tank.get_slots()


def split(a=empty, indices=empty, axis=empty, type_dict=None, waterwork=None, name=None):
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

  return tank.get_tubes(), tank.get_slots()


def sub(a=empty, b=empty, waterwork=None, name=None):
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
  # return tank['target'], tank['smaller_size_array'], tank['a_is_smaller'], tank.get_slots()
  return tank.get_tubes(), tank.get_slots()


def tokenize(strings=empty, tokenizer=empty, max_len=empty, detokenizer=empty, waterwork=None, name=None):
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
  tank = to.Tokenize(strings=strings, tokenizer=tokenizer, max_len=max_len, detokenizer=detokenizer)
  # return tank['target'], tank['tokenizer'], tank['delimiter'], tank['diff'], tank.get_slots()
  return tank.get_tubes(), tank.get_slots()


def transpose(a=empty, axes=empty, waterwork=None, name=None):
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
  return tank.get_tubes(), tank.get_slots()


isnan = bo.create_one_arg_bool_tank(np.isnan, class_name='IsNan')
isnat = bo.create_one_arg_bool_tank(np.isnat, class_name='IsNat')
equal = bo.create_two_arg_bool_tank(np.equal, class_name='Equals')
greater = bo.create_two_arg_bool_tank(np.greater, class_name='Greater')
greater_equal = bo.create_two_arg_bool_tank(np.greater_equal, class_name='GreaterEqual')
less = bo.create_two_arg_bool_tank(np.less, class_name='Less')
less_equal = bo.create_two_arg_bool_tank(np.less_equal, class_name='LessEqual')
isin = bo.create_two_arg_bool_tank(np.isin, class_name='IsIn')

max = rd.create_one_arg_reduce_tank(np.max, class_name='Max')
min = rd.create_one_arg_reduce_tank(np.min, class_name='Min')
sum = rd.create_one_arg_reduce_tank(np.sum, class_name='Sum')
mean = rd.create_one_arg_reduce_tank(np.mean, class_name='Mean')
std = rd.create_one_arg_reduce_tank(np.std, class_name='Std')
