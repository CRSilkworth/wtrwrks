import reversible_transforms.waterworks.tank as ta

def create_new_tank(self, class_name, slot_keys, tube_keys, pour_func, pump_func):

  class TankSubclass(ta.Tank):
    slot_keys = slot_keys.copy()
    tube_keys = tube_keys.copy()

    def __init__(self,  waterwork=None, name=None):
      super(self.__class__, self).__init__(waterwork, name)

      self._create_slots(input_dict.keys(), waterwork)
      self._create_tubes(tube_keys, waterwork)

    def pour(self, a, b):
      return {'data': a + b, 'a': a}

    def pump(self, a, data):
      return {'a': a, 'b': data - a}
