import reversible_transforms.waterworks.waterwork_part as wp

class Tube(wp.WaterworkPart):
  def __init__(self, tank, key,  slot=None, name=None):
    self.key = key
    self.tank = tank
    self.tube = tube

  def __hash__(self):
    return hash((self.tank, self.key))

  def __eq__(self, other):
    return (self.tank, self.key) == (other.tank, other.key)

  def __str__(self):
    return str((str(self.tank), str(self.key)))
