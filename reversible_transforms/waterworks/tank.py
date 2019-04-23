import reversible_transforms.waterworks.globals as gl
import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.slot as sl
import reversible_transforms.waterworks.tube as tu


class Tank(wp.WaterworkPart):
  slot_keys = []
  tube_keys = []

  def __init__(self, waterwork=None, name=None, **input_dict):
    super(self.__class__, self).__init__(waterwork, name)

    for key in input_dict:
      if key not in self.slot_keys:
        raise TypeError(key + ' not a valid argument for ' + str(type(self)))

    self.slots = {}
    self.tubes = {}

    self._create_slots(self.slot_keys, waterwork)
    self._create_tubes(self.tube_keys, waterwork)
    self._join_tubes_to_slots(input_dict, waterwork)

  def __hash__(self):
    return hash(self.name)

  def __getitem__(self, key):
    return self.tubes[key]

  def __str__(self):
    return str(self.name)

  def get_tube(self, key):
    return self.tubes[key]

  def get_slot(self, key):
    return self.slots[key]

  def get_tubes(self):
    tubes = {}
    tubes.update(self.tubes)
    return tubes

  def get_slots(self):
    slots = {}
    slots.update(self.slots)
    return slots

  def _get_default_name(self):
    num = 0
    cls_name = self.__class__.__name__

    full_name = cls_name + '_' + str(num)
    while full_name in gl._default_waterwork.tanks:
      num += 1
      full_name = cls_name + '_' + str(num)

    return full_name

  def _create_slots(self, slot_keys, waterwork):
    for key in slot_keys:
      slot = sl.Slot(self, key)
      self.slots[key]

      waterwork.slots[slot.name] = slot
      waterwork.funnels[slot.name] = slot

  def _create_tubes(self, tube_keys, waterwork):
    for key in tube_keys:
      tube = tu.Tube(self, key)
      self.tubes[key] = tube

      waterwork.tubes[tube.name] = tube
      waterwork.taps[tube.name] = tube

  def _join_tubes_to_slots(self, input_dict, waterwork):
    for key in input_dict:
      slot = self.slots[key]
      tube = input_dict[key]

      tube.slot = slot
      slot.tube = tube

      del waterwork.funnels[slot.name]
      del waterwork.taps[tube.name]

  def pour(self, *args, **kwargs):
    raise ValueError("'pour' method not defined for " + str(type(self)))

  def pump(self, *args, **kwargs):
    raise ValueError("'pump' method not defined for " + str(type(self)))
