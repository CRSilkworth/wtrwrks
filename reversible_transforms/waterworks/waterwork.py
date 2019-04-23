import reversible_transforms.waterworks.globs as gl
import reversible_transforms.waterworks.waterwork_part as wp

class Waterwork(object):
  def __init__(self, name=None):
    self.funnels = {}
    self.tubes = {}
    self.slots = {}
    self.tanks = {}
    self.taps = {}

  def __enter__(self):
    gl._default_waterwork = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    gl._default_waterwork = None

  def pour(self, funnel_dict):
    pass

  def pump(self, tap_dict):
    pass
