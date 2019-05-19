import reversible_transforms.tanks.utils as ut
import reversible_transforms.tanks.bool as bo
import reversible_transforms.tanks.clone as cl
import reversible_transforms.tanks.add as ad
import reversible_transforms.tanks.sub as su
import reversible_transforms.tanks.mul as mu
import reversible_transforms.tanks.div as dv
import reversible_transforms.tanks.cat_to_index as cti
import reversible_transforms.tanks.cast as ct
import reversible_transforms.tanks.concatenate as co
import reversible_transforms.tanks.reduce as rd
import reversible_transforms.tanks.replace as rp
import reversible_transforms.tanks.one_hot as oh
import reversible_transforms.tanks.transpose as tr
import reversible_transforms.tanks.datetime_to_num as dtn
import reversible_transforms.tanks.tokenize as to
import reversible_transforms.tanks.lower_case as lc
import reversible_transforms.tanks.half_width as hw
import reversible_transforms.tanks.lemmatize as lm
import reversible_transforms.tanks.split as sp
import reversible_transforms.tanks.partition as pa
import reversible_transforms.tanks.iterate as it
import reversible_transforms.tanks.replace_substring as rs
from reversible_transforms.waterworks.empty import empty
import numpy as np


def clone(a=empty, waterwork=None, name=None, return_tank=False):
  """Copy an object in order to send it to two different tanks. Usually not performed explicitly but rather when a tube is put as input into two different slots. A clone operation is automatically created.

  Parameters
  ----------
  a : object
      The object to be cloned into two.

  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = cl.Clone(a=a, waterwork=waterwork, name=name)

  # return tank['a'], tank['b'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def add(a=empty, b=empty, waterwork=None, name=None, return_tank=False):
  """Add two objects together in a reversible manner. This function selects out the proper Add subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be summed or None
      First object to be added, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be summed or None
      Second object to be added, or if None, a 'funnel' to fill later with data.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = ad.Add(a=a, b=b, waterwork=waterwork, name=name)
  # return tank['target'], tank['smaller_size_array'], tank['a_is_smaller'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def sub(a=empty, b=empty, waterwork=None, name=None, return_tank=False):
  """Sub two objects together in a reversible manner. This function selects out the proper Sub subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be summed or None
      First object to be subed, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be summed or None
      Second object to be subed, or if None, a 'funnel' to fill later with data.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to sub the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created sub tank (operation) object.

  """

  tank = su.Sub(a=a, b=b, waterwork=waterwork, name=name)
  # return tank['target'], tank['smaller_size_array'], tank['a_is_smaller'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def mul(a=empty, b=empty, waterwork=None, name=None, return_tank=False):
  """Mul two objects together in a reversible manner. This function selects out the proper Mul subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be summed or None
      First object to be muled, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be summed or None
      Second object to be muled, or if None, a 'funnel' to fill later with data.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to mul the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created mul tank (operation) object.

  """

  tank = mu.Mul(a=a, b=b, waterwork=waterwork, name=name)
  # return tank['target'], tank['smaller_size_array'], tank['a_is_smaller'], tank['missing_vals'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def div(a=empty, b=empty, waterwork=None, name=None, return_tank=False):
  """Div two objects together in a reversible manner. This function selects out the proper Div subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be summed or None
      First object to be dived, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be summed or None
      Second object to be dived, or if None, a 'funnel' to fill later with data.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to div the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created div tank (operation) object.

  """

  tank = dv.Div(a=a, b=b, waterwork=waterwork, name=name)
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank
  # return tank['target'], tank['smaller_size_array'], tank['a_is_smaller'], tank['missing_vals'], tank['remainder'], tank.get_slots()


def cast(a=empty, dtype=empty, waterwork=None, name=None, return_tank=False):
  """Find the min of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  dtype : Tube, int, tuple or None
      The dtype (axes) along which to take the min.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = ct.Cast(a=a, dtype=dtype, waterwork=waterwork, name=name)
  # return tank['target'], tank['input_dtype'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def cat_to_index(cats=empty, cat_to_index_map=empty, waterwork=None, name=None, return_tank=False):
  """Convert categorical values to index values according to cat_to_index_map. Handles the case where the categorical value is not in cat_to_index by mapping to -1.

  Parameters
  ----------
  cats : int, str, unicode, flot, numpy array or None
      The categorical values to be mapped to indices
  cat_to_index_map : dictionary
      A mapping from categorical values to indices
  type_dict : dict({
    keys - ['cats', 'cat_to_index_map']
    values - type of argument 'cats' type of argument 'cat_to_index_map'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = cti.CatToIndex(cats=cats, cat_to_index_map=cat_to_index_map, waterwork=waterwork, name=name)
  # return tank['target'], tank['cat_to_index_map'], tank['missing_vals'], tank['input_dtype'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def concatenate(a_list=empty, axis=empty, waterwork=None, name=None, return_tank=False):
  """Concatenate a np.array from subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = co.Concatenate(a_list=a_list, axis=axis, waterwork=waterwork, name=name)
  # return tank['target'], tank['axis'], tank['indices'], tank['dtypes'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def tokenize(strings=empty, tokenizer=empty, max_len=empty, delimiter=empty, waterwork=None, name=None, return_tank=False):
  """Adds another dimension to the array 'strings' of size max_len which the elements of strings split up into tokens (e.g. words).

  Parameters
  ----------
  strings : Tube, np.ndarray or None
    The array of strings to tokenize
  tokenizer : Tube, function or None
    The function that converts a string into a list of tokens.
  max_len : Tube, int or None
    The max number of allowed tokens to be created from one string. I.e.  the size of the last dimension of the target array.
  delimiter : Tube, str, unicode or None
    The delimiter used to join string back together from tokens.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = to.Tokenize(strings=strings, tokenizer=tokenizer, max_len=max_len, delimiter=delimiter)
  # return tank['target'], tank['tokenizer'], tank['delimiter'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def lower_case(strings=empty, waterwork=None, name=None, return_tank=False):
  """Adds another dimension to the array 'strings' of size max_len which the elements of strings split up into tokens (e.g. words).

  Parameters
  ----------
  strings : Tube, np.ndarray or None
    The array of strings to lower_case
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = lc.LowerCase(strings=strings)
  # return tank['target'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def half_width(strings=empty, waterwork=None, name=None, return_tank=False):
  """Add another dimension to the array 'strings' of size max_len which the elements of strings split up into tokens (e.g. words).

  Parameters
  ----------
  strings : Tube, np.ndarray or None
    The array of strings to half_width
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = hw.HalfWidth(strings=strings)
  # return tank['target'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def lemmatize(strings=empty, lemmatizer=empty, waterwork=None, name=None, return_tank=False):
  """Add another dimension to the array 'strings' of size max_len which the elements of strings split up into tokens (e.g. words).

  Parameters
  ----------
  strings : Tube, np.ndarray or None
    The array of strings to lemmatize
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = lm.Lemmatize(strings=strings, lemmatizer=lemmatizer)
  # return tank['target'], tank['lemmatizer'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def replace_substring(strings=empty, old_substring=empty, new_substring=empty, waterwork=None, name=None, return_tank=False):
  """Add another dimension to the array 'strings' of size max_len which the elements of strings split up into tokens (e.g. words).

  Parameters
  ----------
  strings : Tube, np.ndarray or None
    The array of strings to half_width
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = rs.ReplaceSubstring(strings=strings, old_substring=old_substring, new_substring=new_substring)
  # return tank['target'], tank['old_substring'], tank['new_substring'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def one_hot(indices=empty, depth=empty, waterwork=None, name=None, return_tank=False):
  """One hotify an index or indices, keeping track of the missing values (i.e. indices outside of range 0 <= index < depth) so that it can be undone. A new dimension will be added to the indices dimension so that 'target' with have a rank one greater than 'indices'. The new dimension is always added to the end. So indices.shape == target.shape[:-1] and target.shape[-1] == depth.

  Parameters
  ----------
  indices : int or numpy array of dtype int
      The indices to be one hotted.
  depth : int
      The length of the one hotted dimension.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = oh.OneHot(indices=indices, depth=depth, waterwork=waterwork, name=name)
  # return tank['target'], tank['missing_vals'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def replace(a=empty, mask=empty, replace_with=empty, waterwork=None, name=None, return_tank=False):
  """Find the min of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to replace the values of.
  mask : Tube, np.ndarray or None
      An array of booleans which define which values of a are to be replaced.
  replace_with : Tube, np.ndarray or None
    The values to replace those values of 'a' which have a corresponding 'True' in the mask.
  axis : Tube, int, tuple or None
      The axis (axes) along which to take the min.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = rp.Replace(a=a, mask=mask, replace_with=replace_with, waterwork=waterwork, name=name)
  # return tank['target'], tank['mask'], tank['replaced_vals'], tank['replace_with_shape'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def transpose(a=empty, axes=empty, waterwork=None, name=None, return_tank=False):
  """Find the min of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  axes : Tube, tuple or None
      A permutation of axes.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = tr.Transpose(a=a, axes=axes, waterwork=waterwork, name=name)
  # return tank['target'], tank['axes'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def datetime_to_num(a=empty, zero_datetime=empty, num_units=empty, time_unit=empty, waterwork=None, name=None, return_tank=False):
  """Add two objects together in a reversible manner. This function selects out the proper Add subclass depending on the types of 'a' and 'b'.

  Parameters
  ----------
  a : Tube, type that can be summed or None
      First object to be added, or if None, a 'funnel' to fill later with data.
  b : Tube, type that can be summed or None
      Second object to be added, or if None, a 'funnel' to fill later with data.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = dtn.DatetimeToNum(a=a, zero_datetime=zero_datetime, num_units=num_units, time_unit=time_unit, waterwork=waterwork, name=name)

  # return tank['target'], tank['zero_datetime'], tank['num_units'], tank['time_unit'], tank['diff'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def logical_not(a=empty, waterwork=None, name=None, return_tank=False):
  """Create a function which generates the tank instance corresponding to some single argument, boolean valued numpy function. (e.g. np.isnan). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

  Parameters
  ----------
  np_func : numpy function
      A numpy function which operates on an array to give another array.
  class_name : str
      The name you'd like to have the Tank class called.

  Returns
  -------
  func
      A function which outputs a tank instance which behaves like the np_func but is also reversible

  """
  tank = bo.LogicalNot(a=a, waterwork=waterwork, name=name)
  # return tank['target'], tank.get_slots()
  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def split(a=empty, indices=empty, axis=empty, type_dict=None, waterwork=None, name=None, return_tank=False):
  """Split a np.array into subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = sp.Split(a=a, indices=indices, axis=axis, waterwork=waterwork, name=name)

  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def partition(a=empty, indices=empty, type_dict=None, waterwork=None, name=None, return_tank=False):
  """Split a np.array into subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  tank = pa.Partition(a=a, indices=indices, waterwork=waterwork, name=name)

  if not return_tank:
    return tank.get_tubes(), tank.get_slots()
  return tank.get_tubes(), tank.get_slots(), tank


def iter_list(a=empty, num_entries=None, type_dict=None, waterwork=None, name=None, return_tank=False):
  """Split a np.array into subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  if num_entries is None:
    raise ValueError("Must explicitly set num_entries.")

  keys = ['a' + str(i) for i in xrange(num_entries)]

  class IterListTyped(it.IterList):
    tube_keys = keys
  tank = IterListTyped(a=a, waterwork=waterwork, name=name)

  tubes = tank.get_tubes()
  if not return_tank:
    return [tubes[tube_key] for tube_key in keys], tank.get_slots()
  return [tubes[tube_key] for tube_key in keys], tank.get_slots(), tank


def iter_dict(a=empty, keys=None, type_dict=None, waterwork=None, name=None, return_tank=False):
  """Split a np.array into subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  if keys is None:
    raise ValueError("Must explicitly set num_entries.")

  class IterDictTyped(it.IterDict):
    tube_keys = keys
  tank = IterDictTyped(a=a, waterwork=waterwork, name=name)

  tubes = tank.get_tubes()
  if not return_tank:
    return {tube_key: tubes[tube_key] for tube_key in keys}, tank.get_slots()
  return {tube_key: tubes[tube_key] for tube_key in keys}, tank.get_slots(), tank

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
