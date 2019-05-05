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

import numpy as np


def clone(a, type_dict=None, waterwork=None, name=None):
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

  type_dict = ut.infer_types(type_dict, a=a)

  class CloneTyped(cl.Clone):
    tube_dict = {
      'a': type_dict['a'],
      'b': type_dict['a']
    }

  return CloneTyped(a=a, waterwork=waterwork, name=name)


def add(a, b, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, b=b)
  target_dtype = ut.decide_dtype(np.array(a).dtype, np.array(b).dtype)

  class AddTyped(ad.Add):
    tube_dict = {
      'target': (np.ndarray, target_dtype),
      'smaller_size_array': (np.ndarray, target_dtype),
      'a_is_smaller': (bool, None)
    }

  return AddTyped(a=a, b=b, waterwork=waterwork, name=name)


def sub(a, b, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, b=b)
  target_dtype = ut.decide_dtype(np.array(a).dtype, np.array(b).dtype)

  class SubTyped(su.Sub):
    tube_dict = {
      'target': (np.ndarray, target_dtype),
      'smaller_size_array': (np.ndarray, target_dtype),
      'a_is_smaller': (bool, None)
    }

  return SubTyped(a=a, b=b, waterwork=waterwork, name=name)


def mul(a, b, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, b=b)
  target_dtype = ut.decide_dtype(np.array(a).dtype, np.array(b).dtype)

  class MulTyped(mu.Mul):
    tube_dict = {
      'target': (np.ndarray, target_dtype),
      'smaller_size_array': (np.ndarray, target_dtype),
      'a_is_smaller': (bool, None),
      'missing_vals': (np.ndarray, target_dtype),
    }

  return MulTyped(a=a, b=b, waterwork=waterwork, name=name)


def div(a, b, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, b=b)
  target_dtype = ut.decide_dtype(np.array(a).dtype, np.array(b).dtype)

  class DivTyped(dv.Div):
    tube_dict = {
      'target': (np.ndarray, target_dtype),
      'smaller_size_array': (np.ndarray, target_dtype),
      'a_is_smaller': (bool, None),
      'missing_vals': (np.ndarray, target_dtype),
      'remainder': (np.ndarray, target_dtype)
    }

  return DivTyped(a=a, b=b, waterwork=waterwork, name=name)


def cast(a, dtype, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, dtype=dtype)

  class CastTyped(ct.Cast):
    slot_keys = ['a', 'dtype']
    tube_dict = {
      'target': (np.ndarray, dtype),
      'input_dtype': (type, None),
      'diff': (np.ndarray, np.array(a).dtype)
    }

  return CastTyped(a=a, dtype=dtype, waterwork=waterwork, name=name)


def cat_to_index(cats, cat_to_index_map, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, cats=cats, cat_to_index_map=cat_to_index_map)

  if type_dict['cat_to_index_map'] != (dict, None):
    raise TypeError("cat_to_index_map must be of type dict.")

  return cti.CatToIndex(cats=cats, cat_to_index_map=cat_to_index_map, waterwork=waterwork, name=name)

def concatenate(a_list, axis, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a_list=a_list, axis=axis)
  dtypes = [np.array(a).dtype for a in a_list]
  target_dtype = ut.decide_dtype(*dtypes)

  class ConcatenateTyped(co.Concatenate):
    tube_dict = {
      'target': (np.ndarray, target_dtype),
      'indices': (np.ndarray, np.int64),
      'axis': (int, None),
      'dtypes': (list, None)
    }
  return ConcatenateTyped(a_list=a_list, axis=axis, waterwork=waterwork, name=name)


def one_hot(indices, depth, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, indices=indices, depth=depth)

  class OneHotTyped(oh.OneHot):
    tube_dict = {
      'target': (np.ndarray, np.float64),
      'missing_vals': type_dict['indices']
    }

  return OneHotTyped(indices=indices, depth=depth, waterwork=waterwork, name=name)



def replace(a, mask, replace_with, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, mask=mask, replace_with=replace_with)

  class ReplaceTyped(rp.Replace):
    tube_dict = {
      'target': type_dict['a'],
      'replaced_vals': type_dict['a'],
      'mask': type_dict['mask'],
      'replace_with_shape': (tuple, None)
    }

  return ReplaceTyped(a=a, mask=mask, replace_with=replace_with, waterwork=waterwork, name=name)


def transpose(a, axes, type_dict=None, waterwork=None, name=None):
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
  type_dict = ut.infer_types(type_dict, a=a, axes=axes)

  class TransposeTyped(tr.Transpose):
    tube_dict = {
      'target': type_dict['a'],
      'axes': type_dict['axes']
    }

  return TransposeTyped(a=a, axes=axes, waterwork=waterwork, name=name)


isnan = bo.create_one_arg_bool_tank(np.isnan, class_name='IsNan')

equal = bo.create_two_arg_bool_tank(np.equal, class_name='Equals')
greater = bo.create_two_arg_bool_tank(np.greater, class_name='Greater')
greater_equal = bo.create_two_arg_bool_tank(np.greater_equal, class_name='GreaterEqual')
less = bo.create_two_arg_bool_tank(np.less, class_name='Less')
less_equal = bo.create_two_arg_bool_tank(np.less_equal, class_name='LessEqual')

max = rd.create_one_arg_reduce_tank(np.max, class_name='Max')
min = rd.create_one_arg_reduce_tank(np.min, class_name='Min')
sum = rd.create_one_arg_reduce_tank(np.sum, class_name='Sum')
mean = rd.create_one_arg_reduce_tank(np.mean, class_name='Mean')
std = rd.create_one_arg_reduce_tank(np.std, class_name='Std')
