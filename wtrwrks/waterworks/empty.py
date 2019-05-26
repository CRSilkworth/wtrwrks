"""Empty definition."""
import wtrwrks.waterworks.waterwork_part as wp


class Empty(wp.WaterworkPart):
  """A special type of waterwork part, used to start waterworks without having the data at definition time.

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

  def __init__(self):
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

  def __add__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.add(a=self, b=other)

  def __div__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.div(a=self, b=other)

  def __eq__(self, other):
    """Determine whether two tubes are the same within one waterwork."""
    return type(self) is Empty and type(other) is Empty

  def __ne__(self, other):
    """Determine whether two tubes are the same within one waterwork."""
    return not (self == other)

  def __mul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.mul(a=self, b=other)

  def __radd__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.add(a=other, b=self)

  def __rdiv__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.div(a=other, b=self)

  def __rmul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.mul(a=other, b=self)

  def __rsub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.sub(a=other, b=self)

  def __str__(self):
    """Get a string of the name of the tube."""
    return "EmptySlot"

  def __sub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.sub(a=self, b=other)


empty = Empty()
