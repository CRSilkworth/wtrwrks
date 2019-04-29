import copy
import numpy as np
import reversible_transforms.waterworks.tube as tu


def infer_types(type_dict, **input_dict):
  if type_dict is None:
    type_dict = {}

  r_dict = {}
  r_dict.update(type_dict)

  for key in input_dict:
    if type(input_dict[key]) is tu.Tube:
      r_dict[key] = input_dict[key].val_type
    else:
      r_dict[key] = type(input_dict[key])

  return r_dict


def decide_type(*types):
  out_type = None
  for t in types:
    if out_type is None:
      out_type = t
    else:
      if out_type is int:
        out_type = t
      elif out_type is float and t is np.ndarray:
        out_type = t
  return out_type


def maybe_copy(a):
  if type(a) in (str, unicode, int, float):
    return a
  if type(a) is np.ndarray:
    return np.array(a, copy=True)
  if type(a) in (list, dict):
    return copy.copy(a)

  return copy.copy(a)
