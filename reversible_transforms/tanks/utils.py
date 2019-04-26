import copy
import numpy as np


def infer_types(type_dict, **input_dict):
  if type_dict is None:
    type_dict = {}

  r_dict = {}
  r_dict.update(type_dict)

  for key in input_dict:
    r_dict[key] = type(input_dict[key])

  return r_dict


def maybe_copy(a):
  if type(a) in (str, unicode, int, float):
    return a
  if type(a) is np.ndarray:
    return np.array(a, copy=True)
  if type(a) in (list, dict):
    return a.copy()

  return copy.copy(a)
