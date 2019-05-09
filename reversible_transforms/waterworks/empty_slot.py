import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.globs as gl
import numpy as np
import os


def placeholder(val_type, val_dtype=None, waterwork=None, name=None):

  if val_type is np.ndarray and val_dtype is None:
    raise ValueError("Must give a val_dtype if val_type is np.ndarray")

  return Placeholder(val_type=val_type, val_dtype=val_dtype, waterwork=waterwork, name=name)


class EmptySlot(wp.WaterworkPart):
  """A special type of waterwork part, used to start waterworks without having the data at defition time.

  Attributes
  ----------
  tank : Tank
    The tank that this tube is a part of.
  key : str
    The string to identify the tube within the tank. Must be unique among all other tubes of this tank.
  val : some data type or None
    The value last outputed by the tank (i.e. operation), if applicable.
  slot : Slot or None
    The slot from the other tank this tube is connected to, if applicable.
  name : str
    The string used to identify the tube within the entire waterwork. Must be unique among all other tubes of this waterwork.
  """
  def __init__(self, val_type=None, val_dtype=None, val=None, slot=None, waterwork=None, name=None):
    """Initialize the tube.

    Parameters
    ----------
    tank : Tank
      The tank that this tube is a part of.
    key : str
      The string to identify the tube within the tank. Must be unique among all other tubes of this tank.
    val : some data type or None
      The value last outputed by the tank (i.e. operation), if applicable.
    slot : Slot or None
      The slot from the other tank this tube is connected to, if applicable.
    """
    self.slot = slot
    self.name = name
    self.val = val
    self.val_type = val_type
    self.val_dtype = val_dtype
    self.waterwork = waterwork

    super(Placeholder, self).__init__(waterwork, self.name)
    if self.name in self.waterwork.placeholders:
      raise ValueError(self.name + " already defined as placeholder. Choose a different name.")

    self.waterwork.placeholders[self.name] = self

  def __add__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.add(a=self, b=other, waterwork=self.waterwork)

  def __div__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.div(a=self, b=other, waterwork=self.waterwork)

  def __eq__(self, other):
    """Determine whether two tubes are the same within one waterwork."""
    return self.name == other.name

  def __hash__(self):
    """Determine whether two tubes are the same within one waterwork."""
    return hash(self.name)

  def __mul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.mul(a=self, b=other, waterwork=self.waterwork)

  def __radd__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.add(a=other, b=self, waterwork=self.waterwork)

  def __rdiv__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.div(a=other, b=self, waterwork=self.waterwork)

  def __rmul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.mul(a=other, b=self, waterwork=self.waterwork)

  def __rsub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.sub(a=other, b=self, waterwork=self.waterwork)

  # def __str__(self):
  #   """Get a string of the name of the tube."""
  #   return self.name

  def __sub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import reversible_transforms.tanks.tank_defs as td
    return td.sub(a=self, b=other, waterwork=self.waterwork)

  def _get_default_name(self, prefix=''):
    """Create the default name of the placeholder, of the form 'Placeholder_<num>'.


    Returns
    -------
    str
        The name of the tank.

    """
    num = 0
    # Start with the name being 'Placeholder_0'. If that is already taken,
    # keep increasing the number from 0 until an unused name is found.
    full_name = os.path.join(prefix, 'Placeholder_' + str(num))
    while full_name in gl._default_waterwork.placeholders:
      num += 1
      full_name = os.path.join(prefix, 'Placeholder_' + str(num))

    return full_name

  def get_tuple(self):
    """Get a tuple that describes the tube."""
    return (self.tank.name, self.key)

  def get_val(self):
    """Get the value stored in the tube."""
    return self.val

  def set_val(self, val):
    """Set the value stored in the tube."""
    self.val = val

  def set_name(self, name):
    """Set the name of the placeholder within the waterwork."""
    old_name = self.name
    self.name = name
    if type(name) not in (str, unicode):
      raise TypeError("'name' must be of type str or unicode. Got " + str(type(name)))
    elif not self.name.startswith(self.name_space._get_name_string()):
      self.name = os.path.join(self.name_space._get_name_string(), self.name)
    if self.name in self.waterwork.placeholders:
      raise ValueError(self.name + " already defined as placeholder. Choose a different name.")

    del self.waterwork.placeholders[old_name]
    self.waterwork.placeholders[self.name] = self
