import numpy as np


def empty_array_like(a, default_val=None):
  """Create an array of the same shape and type as 'a', with all default values.

  Parameters
  ----------
  a : np.ndarray of bools
    The array shape and dtype to copy


  Returns
  -------
  np.ndarray
    An array of all default values.

  """
  dtype = a.dtype

  # Choose different default values, and type for the returned array depending
  # on the input_dtype
  if dtype.type in (np.string_, np.unicode_):
    if default_val is None:
      default_val = ''
  elif dtype in (np.int64, np.int32, np.float64, np.float32):
    if default_val is None:
      default_val = 0
  elif dtype.type == np.timedelta64:
    default_val = 0
  elif dtype.type == np.datetime64:
    default_val = '1970-01-01'
  else:
    raise TypeError("Only string and number types are supported. Got " + str(dtype))

  full_array = np.full(a.shape, default_val, dtype=dtype)
  return full_array


def empty_array(shape, dtype, default_val=None):
  """Create an array of the same shape and type as 'a', with all default values.

  Parameters
  ----------
  a : np.ndarray of bools
    The array shape and dtype to copy


  Returns
  -------
  np.ndarray
    An array of all default values.

  """
  # Choose different default values, and type for the returned array depending
  # on the input_dtype
  if dtype.type in (np.string_, np.unicode_):
    if default_val is None:
      default_val = ''
  elif dtype in (np.int64, np.int32, np.float64, np.float32):
    if default_val is None:
      default_val = 0
  elif dtype.type == np.timedelta64:
    default_val = 0
  elif dtype.type == np.datetime64:
    default_val = '1970-01-01'
  elif dtype.type == np.dtype('O'):
    pass
  else:
    raise TypeError("Only string and number types are supported. Got " + str(dtype))

  full_array = np.full(shape, default_val, dtype=dtype)
  return full_array


def get_default_val(a):
  dtype = a.dtype
  # Choose different default values, and type for the returned array depending
  # on the input_dtype
  if dtype.type in (np.string_, np.unicode_):
    default_val = ''
  elif dtype in (np.int64, np.int32, np.float64, np.float32, np.timedelta64):
    default_val = 0
  elif dtype.type == np.datetime64:
    default_val = '1970-01-01'
  else:
    default_val = None

  return default_val
