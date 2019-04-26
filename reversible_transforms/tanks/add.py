import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def add(a, b, type_dict=None, waterwork=None, name=None):
  type_dict = ut.infer_types(type_dict, a=a, b=b)

  missing_keys = set(['a', 'b']) - set(type_dict.keys())
  if missing_keys:
    raise ValueError("Type(s) of " + str(sorted(missing_keys)) + " not known. Must define in type_dict if nothing is passed to a, b")
  if type_dict['a'] is np.ndarray and type_dict['b'] is np.ndarray:
    return AddNP(a=a, b=b, waterwork=waterwork, name=name)
  elif type_dict['a'] is np.ndarray or type_dict['b'] is np.ndarray:
    return AddMixNP(a=a, b=b, waterwork=waterwork, name=name)

  return AddBasic(a=a, b=b, waterwork=waterwork, name=name)


class Add(ta.Tank):
  """Add base class. All subclasses must have exactly the same slot_keys, but can differ in the tube_keys. _pour and _pump must be defined"""
  slot_keys = ['a', 'b']
  tube_keys = None

  def _pour(self, a, b):
    raise ValueError("_pour not defined!")

  def _pump(self, *args, **kwargs):
    raise ValueError("_pump not defined!")


class AddBasic(Add):
  tube_keys = ['a', 'target']

  def _pour(self, a, b):
    return {'target': a + b, 'a': ut.maybe_copy(a)}

  def _pump(self, a, target):
    return {'a': ut.maybe_copy(a), 'b': target - a}


class AddNP(Add):
  tube_keys = ['target', 'smaller', 'a_smaller']

  def _pour(self, a, b):
    a_smaller = a.size < b.size
    if a_smaller:
      smaller = ut.maybe_copy(a)
    else:
      smaller = ut.maybe_copy(b)
    return {'target': a + b, 'smaller': smaller, 'a_smaller': a_smaller}

  def _pump(self, target, smaller, a_smaller):
    if a_smaller:
      a = ut.maybe_copy(smaller)
      b = target - smaller
    else:
      a = target - smaller
      b = ut.maybe_copy(smaller)

    return {'a': a, 'b': b}


class AddMixNP(Add):
  tube_keys = ['target', 'non_array', 'a_array']

  def _pour(self, a, b):
    a_array = type(a) is np.ndarray
    if a_array:
      non_array = ut.maybe_copy(b)
    else:
      non_array = ut.maybe_copy(a)
    return {'target': a + b, 'non_array': non_array, 'a_array': a_array}

  def _pump(self, target, non_array, a_array):
    if a_array:
      a = target - non_array
      b = ut.maybe_copy(non_array)
    else:
      a = ut.maybe_copy(non_array)
      b = target - non_array

    return {'a': a, 'b': b}
