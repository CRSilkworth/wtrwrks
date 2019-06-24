import numpy as np
import tensorflow as tf
import codecs

def _int_feat(value):
    """Wrapper for inserting int64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feat(value):
    """Wrapper for inserting float64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feat(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, np.ndarray):
      dtype = value.dtype
      if dtype.char == 'U':
        value = np.char.encode(value, encoding='utf-8').tolist()
  elif not isinstance(value, list):
    if type(value) in (unicode, np.unicode_):
      value = np.char.encode(value, encoding='utf-8')
    value = [value]
  else:
    if dtype.char == 'U':
      value = np.char.encode(value, encoding='utf-8').tolist()

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def select_feature_func(dtype):
  """Choose a feature function based on the inputted datatype."""
  if dtype in (np.int32, np.int64, np.bool):
    return _int_feat
  elif dtype in (np.float32, np.float64):
    return _float_feat
  elif dtype.type in (np.string_, np.unicode_):
    return _bytes_feat
  else:
    raise TypeError("Only string and number types are supported. Got " + str(dtype))


def select_tf_dtype(dtype):
  """Choose a tensorflow type based on the inputted numpy datatype."""
  if dtype in (np.int32, np.int64, np.bool):
    return tf.int64
  elif dtype in (np.float32, np.float64):
    return tf.float32
  elif dtype.type in (np.string_, np.unicode_):
    return tf.string
  else:
    raise TypeError("Only string and number types are supported. Got " + str(dtype))


def size_from_shape(shape):
  """Calculate the size of the array from it's shape."""
  if not shape:
    return 1

  return np.prod(shape)
