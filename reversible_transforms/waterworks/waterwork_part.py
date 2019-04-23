import reversible_transforms.waterworks.globals as gl

class WaterworkPart(object):
  def __init__(self, waterwork, name):
    # Set the tank name
    self.name = name
    if name is None:
      self.name = self._get_default_name()

    # Set the waterwork to add the tank to
    self.waterwork = waterwork
    if waterwork is None and gl._default_waterwork is None:
      raise ValueError("Must define op within 'with' statement of waterwork or pass a waterwork as an argument.")
    elif waterwork is None:
      self.waterwork = gl._default_waterwork

  def _get_default_name(self):
    pass
