"""WaterworkPart definition."""
import wtrwrks.waterworks.globs as gl
import os


class WaterworkPart(object):
  """Base class for the components of a Waterwork. Primarly used to set the waterwork the part belongs to as well as its name within that waterwork.

  Attributes
  ----------
  waterwork : Waterwork or None
    The waterwork that the part will be added to. If None, it is assinged to the _default_waterwork.
  name : str or None
    The name of the part within the waterwork. Must be unique. If None, it will be set to a default value depending on the subclass.

  """

  def __init__(self, waterwork, name):
    """Define a WaterworkPart, assign it to a Waterwork and set its name within the Waterwork.

    Parameters
    ----------
    waterwork : Waterwork or None
      The waterwork that the part will be added to. If None, it is assinged to the _default_waterwork.
    name : str or None
      The name of the part within the waterwork. Must be unique. If None, it will be set to a default value depending on the subclass.

    """
    # Set the waterwork to add the tank to
    self.waterwork = waterwork
    if waterwork is None and gl._default_waterwork is None:
      raise ValueError("Must define op within 'with' statement of waterwork or pass a waterwork as an argument.")
    elif waterwork is None:
      self.waterwork = gl._default_waterwork

    # Set the namespace the part was created in
    self.name_space = gl._name_space

    # Set the tank name. Check and make sure it's valid.
    self.name = name
    pre = '' if self.name_space is None else self.name_space._get_name_string()

    if name is None:
      self.name = self._get_default_name(prefix=pre)
    elif type(name) not in (str, unicode):
      raise TypeError("'name' must be of type str or unicode. Got " + str(type(name)))
    elif not self.name.startswith(pre):
      self.name = os.path.join(pre, self.name)

  def _get_default_name(self):
    """Set a default name. Must be defined by subclass."""
    pass
