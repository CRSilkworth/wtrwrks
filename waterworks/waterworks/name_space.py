import waterworks.waterworks.globs as gl
import os


class NameSpace(object):
  def __init__(self, name):
    self.name = name
    self.all_name_spaces = []

  def __enter__(self):
    """When entering, set the global _name_space."""
    if gl._name_space is None:
      self.all_name_spaces = [self]
    else:
      self.all_name_spaces = gl._name_space.all_name_spaces + [self]

    gl._name_space = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """When exiting, set the global _default_waterwork back to None."""
    if len(self.all_name_spaces[:-1]):
      gl._name_space = self.all_name_spaces[-2]
    else:
      gl._name_space = None

  def _get_name_string(self):
    name_strings = [ns.name for ns in self.all_name_spaces]
    name_string = os.path.join(*name_strings)
    return name_string

  def __str__(self):
    return self._get_name_string()
