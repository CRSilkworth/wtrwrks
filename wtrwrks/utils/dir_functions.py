import json
import numpy as np
import pandas as pd
import dill as pickle
import os
import subprocess as sp
import tensorflow as tf

def untar_dir(file_to_untar, dir_to_save_to, verbose=False):
  """Untar a tarball and save it to another directory

  Parameters
  ----------
  file_to_untar : str
    The file name of the tarball
  dir_to_save_to : str
      The directory to save the untarred directory

  """
  if verbose:
    tar_str = "tar zxvf "
  else:
    tar_str = "tar zxf "
  command = tar_str + file_to_untar + " -C " + dir_to_save_to
  stdout = sp.call(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)


def tar_dir(dir_to_tar, dir_to_save_to, ignore_patterns=None, verbose=False):
  """Tar up a directory and save to a particular location.

  Parameters
  ----------
  dir_to_tar : str
    Directory to tar up
  dir_to_save_to : str
    The directory to save the tarred directory

  """
  tar_file_name = dir_to_tar.rstrip('/').split('/')[-1] + '.tar.gz'
  tar_file_name = os.path.join(dir_to_save_to, tar_file_name)

  if verbose:
    tar_str = "tar zcvf "
  else:
    tar_str = "tar zcf "

  if ignore_patterns is None:
    command = tar_str + tar_file_name + " " + dir_to_tar
  else:
    excludes = ' '
    for pat in ignore_patterns:
      excludes += "--exclude='" + pat + "' "
    command = tar_str + tar_file_name + excludes + dir_to_tar

  stdout = sp.call(command, shell=True, stdout=sp.PIPE)


def create_dirs(dir):
  """Create directories and all intermediary directories if they don't exist.

  Parameters
  ----------
  dir : str
    Name of the directory

  """
  if not os.path.exists(dir):
    os.makedirs(dir)


def maybe_create_dir(*args):
  """Create directories and all intermediary directories if they don't exist.

  Parameters
  ----------
  *args:
      The path of the directories to maybe create.

  Returns
  -------
  str
    The new directory path
  """
  if not args:
    return './'

  full_dir = os.path.join(*args)
  if not os.path.isdir(full_dir):
      os.makedirs(full_dir)
  return full_dir


def save_to_file(obj, file_name):
  """Wrapper to automatically find the proper way to save a file

  Parameters
  ----------
  obj :
    Object to save to file
  file_name : str
    File name to save the file to.


  """
  # Create the directory path
  dir = file_name.split('/')[:-1]
  if dir:
    maybe_create_dir(*dir)

  # Find file type and then save using appropriate function.
  file_type = file_name.split('.')[-1]
  with open(file_name, 'w') as obj_file:
    if file_type == 'json':
      json.dump(obj, obj_file)
    elif file_type == 'npy':
      np.save(obj_file, obj)
    elif file_type == 'pickle':
      pickle.dump(obj, obj_file)
  if file_type == 'h5':
    store = pd.HDFStore(file_name)
    store['df'] = obj


def read_from_file(file_name):
  """Wrapper to automatically find the proper way to read a file

  Parameters
  ----------
  file_name : str
    File name of file to read

  Returns
  -------
  obj :
    Object that was read from file
  """

  # Find file type from the extension and then read using appropriate
  # function.
  file_type = file_name.split('.')[-1]
  with open(file_name, 'r') as obj_file:
    if file_type == 'json':
      obj = json.load(obj_file)
    elif file_type == 'npy':
      obj = np.load(obj_file)
    elif file_type == 'pickle':
      obj = pickle.load(obj_file)
  if file_type == 'h5':
    store = pd.HDFStore(file_name)
    obj = store['df']

  return obj
