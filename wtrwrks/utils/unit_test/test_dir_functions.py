import shutil
import tempfile
import unittest
import wtrwrks.utils.dir_functions as d
import wtrwrks.utils.test_helpers as th
import os
import numpy as np


class TestDirFunctions(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_create_dirs(self):
    dir = os.path.join(self.temp_dir, 'test', 'test')
    d.create_dirs(
        dir=dir
    )

  def test_maybe_create_dir(self):
    out_dir = d.maybe_create_dir(
        self.temp_dir,
        'test',
        'test'
    )

  def test_read_from_file(self):
    temp_json = os.path.join(self.temp_dir, 'temp.json')
    d.save_to_file([], temp_json)

    outputs = d.read_from_file(
        file_name=temp_json
    )

    temp_npy = os.path.join(self.temp_dir, 'temp.npy')
    d.save_to_file(np.zeros(dtype=np.int64, shape=[1]), temp_npy)

    array = d.read_from_file(
        file_name=temp_npy
    )
    th.assert_arrays_equal(self, array, np.zeros(dtype=np.int64, shape=[1]))

  def test_tar_dir(self):
    tar_dir = os.path.join(self.temp_dir, 'temp')
    tar_dir = d.maybe_create_dir(
        tar_dir
    )

    d.tar_dir(
        dir_to_tar=tar_dir,
        dir_to_save_to=self.temp_dir
    )

    d.untar_dir(
        file_to_untar=tar_dir.rstrip('/') + '.tar.gz',
        dir_to_save_to=self.temp_dir
    )


if __name__ == "__main__":
    unittest.main()
