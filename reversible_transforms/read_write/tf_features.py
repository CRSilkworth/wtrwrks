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
      value = value.tolist()
      if dtype.char == 'U':
        value = [v.encode('utf-8') for v in value]
  if not isinstance(value, list):
    if type(value) in (unicode, np.unicode_):
      value = value.encode('utf-8')
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
