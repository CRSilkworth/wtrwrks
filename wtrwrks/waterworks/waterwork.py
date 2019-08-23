import wtrwrks.waterworks.globs as gl
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.name_space as ns
import wtrwrks.utils.dir_functions as d
import wtrwrks.utils.multi as mu
import wtrwrks.utils.batch_functions as b
from wtrwrks.waterworks.empty import Empty, empty
import wtrwrks.read_write.tf_features as feat
import os
import pprint
import importlib
import dill as pickle
import tensorflow as tf
import numpy as np
import logging
import pathos.multiprocessing as mp
import glob
import re
import itertools
import traceback
import jpype


class Waterwork(object):
  """The full graph of tanks (i.e. operations) on the data, along with all slots and tubes which define the inputs/outputs of operations and hold their values. Can be thought of as a larger reversible operation that are composed of many smaller reversible operations.

  Attributes
  ----------
  funnels : dict(
    keys - strs. Names of the funnels.
    values - Slot objects.
  )
    All of the slots defined within the waterwork which are not connected to some other tube. i.e. the 'open' slots that need data in order to produce an output in the pour direction.
  taps : dict(
    keys - strs. Names of the taps.
    values - Tube objects.
  )
    All of the tubes defined within the waterwork which are not connected to some other slot. i.e. the 'open' tubes that need data in order to produce an output in the pump direction.
  slots : dict(
    keys - strs. Names of the slots.
    values - Slot objects.
  )
    All of the slots defined within the waterwork.
  tubes : dict(
    keys - strs. Names of the tubes.
    values - Tube objects.
  )
    All of the tubes defined within the waterwork.
  tanks : dict(
    keys - strs. Names of the tanks.
    values - Tube objects.
  )
    All of the tanks (or operations) defined within the waterwork.
  """

  def __init__(self, name='', from_file=None):
    """Initialize the waterwork to have empty funnels, slots, tanks, and taps."""
    self.funnels = {}
    self.tubes = {}
    self.slots = {}
    self.tanks = {}
    self.taps = {}
    self.merged = {}
    self.name = name

    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)

  def __enter__(self):
    """When entering, set the global _default_waterwork to this waterwork."""
    if gl._default_waterwork is not None:
      raise ValueError("_default_waterwork is already set. Cannot be reset until context is exitted. Are you within the with statement of another waterwork?")

    # Create a new namespace for this waterwork
    self.name_space = ns.NameSpace(self.name)
    self.name_space.__enter__()

    gl._default_waterwork = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """When exiting, set the global _default_waterwork back to None."""
    gl._default_waterwork = None
    self.name_space.__exit__(exc_type, exc_val, exc_tb)

  def _from_save_dict(self, save_dict):
    import wtrwrks.tanks.tank_defs as td
    with ns.NameSpace(''):
      self.name = save_dict['name']
      for tank_name in save_dict['tanks']:
        tank_dict = save_dict['tanks'][tank_name]
        func = getattr(td, tank_dict['func_name'])

        kwargs = {}
        if 'kwargs' in tank_dict:
          kwargs = tank_dict['kwargs']

        args = []
        if 'args' in tank_dict:
          args = tank_dict['args']

        tubes, slots = func(name=tank_name, waterwork=self, *args, **kwargs)
        tank = tubes[tubes.keys()[0]].tank
        self.tanks[tank_name] = tank

      for slot_name in save_dict['slots']:
        slot_dict = save_dict['slots'][slot_name]
        tank = self.tanks[slot_dict['tank']]

        slot = tank.get_slot(slot_dict['key'])
        slot.plug = slot_dict['plug']

        # Set to proper name
        del self.slots[slot.name]
        slot.name = slot_name
        self.slots[slot_name] = slot

      for tube_name in save_dict['tubes']:
        tube_dict = save_dict['tubes'][tube_name]
        tank = self.tanks[tube_dict['tank']]

        tube = tank.get_tube(tube_dict['key'])
        tube.plug = tube_dict['plug']

        # Set to proper name
        del self.tubes[tube.name]
        tube.name = tube_name
        self.tubes[tube_name] = tube

      for tube_name in save_dict['tubes']:
        tube = self.tubes[tube_name]
        downstream_tube_name = save_dict['tubes'][tube_name]['downstream_tube']

        if downstream_tube_name is not None:
          tube.downstream_tube = self.tubes[downstream_tube_name]

      for slot_name in self.slots:
        slot_dict = save_dict['slots'][slot_name]
        slot = self.slots[slot_name]

        if slot_dict['tube'] is not None:
          tube = self.tubes[slot_dict['tube']]
          tube.slot = slot
        else:
          tube = empty

        slot.tube = tube

      self.funnels = {}
      for funnel_name in save_dict['funnels']:
        self.funnels[funnel_name] = self.slots[funnel_name]

      self.taps = {}
      for tap_name in save_dict['taps']:
        self.taps[tap_name] = self.tubes[tap_name]

    for key in save_dict['merged']:
      tube = self.tubes[key]
      self.merged[tube] = set([self.tubes[k] for k in save_dict['merged'][key]])

  def _pour_tank_order(self):
    """Get the order to calculate the tanks in the pour direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks if not self.tanks[t].get_slot_tanks()])
    visited = set(tanks)
    sorted_tanks = []

    while tanks:
      tank = tanks.pop(0)
      sorted_tanks.append(tank)

      child_tanks = tank.get_tube_tanks() - visited
      for child_tank in child_tanks:
        # If there are any tanks child_tank depnends on , that haven't been
        # visited, then continue.

        if child_tank.get_slot_tanks() - visited:
          continue

        tanks.append(child_tank)
        visited.add(child_tank)

    return sorted_tanks

  def _pump_tank_order(self):
    """Get the order to calculate the tanks in the pump direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks if not self.tanks[t].get_tube_tanks()])
    visited = set(tanks)
    sorted_tanks = []

    while tanks:
      tank = tanks.pop(0)
      sorted_tanks.append(tank)

      child_tanks = tank.get_slot_tanks() - visited
      for child_tank in child_tanks:
        # If there are any tanks child_tank depnends on , that haven't been
        # visited, then continue.
        if child_tank.get_tube_tanks() - visited:
          continue

        tanks.append(child_tank)
        visited.add(child_tank)
    # tanks = sorted([self.tanks[t] for t in self.tanks])
    # return sorted(tanks, cmp=lambda a, b: 0 if b in a.get_pump_dependencies() else -1)
    return sorted_tanks

  def _save_dict(self):
    save_dict = {}
    save_dict['name'] = self.name
    save_dict['funnels'] = sorted(self.funnels.keys())
    save_dict['taps'] = sorted(self.taps.keys())

    save_dict['tanks'] = {}
    save_dict['slots'] = {}
    save_dict['tubes'] = {}
    save_dict['merged'] = {}

    for key in self.tanks:
      tank = self.tanks[key]
      save_dict['tanks'][key] = tank._save_dict()

    for key in self.slots:
      slot = self.slots[key]
      save_dict['slots'][key] = slot._save_dict()

    for key in self.tubes:
      tube = self.tubes[key]
      save_dict['tubes'][key] = tube._save_dict()

    for tube in self.merged:
      key = tube.name
      save_dict['merged'][key] = sorted([t.name for t in self.merged[tube]])

    return save_dict

  def clear_vals(self):
    """Set all the slots, tubes and placeholder values back to None """
    for d in [self.slots, self.tubes]:
      for key in d:
        d[key].set_val(None)

  def maybe_get_slot(self, *args):
    """Get a particular tank's if it exists, otherwise return None. Can take a variety input types.

    Parameters
    ----------
    *args: Slot, str or tuple
      Either the slot itself, the name of the slot or the (tank name, slot key) tuple.

    Returns
    -------
    Slot or None
      The slot object if found, otherwise a None

    """
    import wtrwrks.waterworks.slot as sl
    # Pull out the tank and key depending on the type and number of inputs.
    if len(args) == 2:
      tank = args[0]
      key = args[1]
    elif len(args) == 1 and type(args[0]) is tuple:
      tank = args[0][0]
      key = args[0][1]
    elif len(args) == 1 and type(args[0]) in (str, unicode) and args[0] in self.slots:
      return self.slots[args[0]]
    elif len(args) == 1 and isinstance(args[0], sl.Slot):
      return args[0]
    else:
      return None

    # Pull out the relevant tank object.
    if type(tank) in (str, unicode) and tank in self.tanks:
      tank = self.tanks[tank]
    elif isinstance(tank, sl.Slot):
      pass
    else:
      return None

    # Get the slot
    if key in tank.slots:
      return tank.slots[key]

    return None

  def maybe_get_tube(self, *args):
    """Get a particular tank's if it exists, otherwise return None. Can take a variety input types.

    Parameters
    ----------
    *args: tube, str or tuple
      Either the tube itself, the name of the tube or the (tank name, tube key) tuple.

    Returns
    -------
    tube or None
      The tube object if found, otherwise a None

    """
    import wtrwrks.waterworks.tube as tu
    # Pull out the tank and key depending on the type and number of inputs.
    if len(args) == 2:
      tank = args[0]
      key = args[1]
    elif len(args) == 1 and type(args[0]) is tuple:
      tank = args[0][0]
      key = args[0][1]
    elif len(args) == 1 and type(args[0]) in (str, unicode) and args[0] in self.tubes:
      return self.tubes[args[0]]
    elif len(args) == 1 and isinstance(args[0], tu.Tube):
      return args[0]
    else:
      return None

    # Pull out the relevant tank object.
    if type(tank) in (str, unicode) and tank in self.tanks:
      tank = self.tanks[tank]
    elif isinstance(tank, tu.Tube):
      pass
    else:
      return None

    # Get the tube
    if key in tank.tubes:
      return tank.tubes[key]

    return None

  def get_slot(self, tank, key):
    """Get a particular tank's slot.

    Parameters
    ----------
    tank : Tank or str
        Either the tank object or the name of the tank.
    key : str
        The slot key of the slot for that tank.

    Returns
    -------
    Slot
        The slot object

    """
    if type(tank) in (str, unicode):
      tank = self.tanks[tank]

    return self.tanks[tank.name].slots[key]

  def get_tube(self, tank, key):
    """Get a particular tank's tube.

    Parameters
    ----------
    tank : Tank or str
        Either the tank object or the name of the tank.
    key : str
        The tube key of the tube for that tank.

    Returns
    -------
    Tube
        The tube object.

    """
    if type(tank) in (str, unicode):
      tank = self.tanks[tank]

    return self.tanks[tank.name].tubes[key]

  def get_taps(self, return_plugged=False):
    """Get a dictionary of all the taps. Optionally choose to return the taps which are plugged or not."""
    r_dict = {k: v for k, v in self.taps.iteritems() if v.plug is None}
    return r_dict

  def get_funnels(self, return_plugged=False):
    """Get a dictionary of all the funnels. Optionally choose to return the funnels which are plugged or not."""
    r_dict = {k: v for k, v in self.funnels.iteritems() if v.plug is None}
    return r_dict

  def merge_tubes(self, target, *args):
    self.merged[target] = set()
    for arg in args:
      if type(arg) is Empty:
        continue
      if arg in self.merged:
        for other_arg in self.merged[arg]:
          other_arg.downstream_tube = target
          self.merged[target].add(other_arg)
        del self.merged[arg]

      elif arg.downstream_tube is not None and arg.downstream_tube != target:
        self.merged[target].add(arg.downstream_tube)
        arg.downstream_tube.downstream_tube = target

        if arg.downstream_tube in self.merged:
          for equal_arg in self.merged[arg.downstream_tube]:
            equal_arg.downstream_tube = target
            self.merged[target].add(equal_arg)

          if arg.downstream_tube != target:
            del self.merged[arg.downstream_tube]

      self.merged[target].add(arg)
      arg.downstream_tube = target

  def multi_pour(self, funnel_dict_iter, key_type='tube', return_plugged=False, num_threads=1, use_threading=False, batch_size=None, pour_func=None):
    if pour_func is None:
      save_dict = self._save_dict()

      def pour_func(funnel_dict):
        ww = Waterwork()
        ww._from_save_dict(save_dict)

        tap_dict = ww.pour(funnel_dict, key_type, return_plugged)
        return tap_dict

    tap_dicts = []
    for batch_num, batch in enumerate(b.batcher(funnel_dict_iter, batch_size)):
      tap_dicts.extend(
        mu.multi_map(pour_func, batch, num_threads, use_threading)
      )

    return tap_dicts

  def multi_pump(self, tap_dict_iter, key_type='slot', return_plugged=False, num_threads=1, use_threading=False, batch_size=None, pump_func=None):
    if pump_func is None:
      save_dict = self._save_dict()

      def pump_func(tap_dict):
        ww = Waterwork()
        ww._from_save_dict(save_dict)

        funnel_dict = ww.pump(tap_dict, key_type, return_plugged)
        return funnel_dict

    funnel_dicts = []
    for batch_num, batch in enumerate(b.batcher(tap_dict_iter, batch_size)):
      funnel_dicts.extend(
        mu.multi_map(pump_func, batch, num_threads, use_threading)
      )

    return funnel_dicts

  def multi_write_examples(self, funnel_dict_iter, file_name, num_threads=1, use_threading=False, batch_size=None, file_num_offset=0, skip_fails=False, skip_keys=None, serialize_func=None):
    if serialize_func is None:
      save_dict = self._save_dict()

      def serialize_func(funnel_dict):
        jpype.attachThreadToJVM()
        ww = Waterwork()
        ww._from_save_dict(save_dict)

        tap_dict = ww.pour(funnel_dict, 'str', False)
        feature_dict, func_dict = self._get_feature_dicts(tap_dict)

        serial = ww._serialize_tap_dict(tap_dict, func_dict)
        return serial

    if type(funnel_dict_iter) in (list, tuple):
      funnel_dict_iter = (i for i in funnel_dict_iter)

    file_names = []
    if not file_name.endswith('.tfrecord'):
      raise ValueError("file_name must end in '.tfrecord'")

    dir = file_name.split('/')[:-1]
    d.maybe_create_dir(*dir)

    for batch_num, batch in enumerate(b.batcher(funnel_dict_iter, batch_size)):
      if batch_num == 0:
        tap_dict = self.pour(batch[0], key_type='str', return_plugged=False)
        feature_dict, func_dict = self._get_feature_dicts(tap_dict)
        feature_dict_fn = re.sub(r'_?[0-9]*.tfrecord', '.pickle', file_name)
        d.save_to_file(feature_dict, feature_dict_fn)

      logging.info("Serializing batch %s", batch_num)
      if skip_fails:
        try:
          all_serials = mu.multi_map(serialize_func, batch, num_threads, use_threading)
        except Exception:
          logging.warn("Batched %s failed. Skipping.", batch_num)
          continue
      else:
        all_serials = mu.multi_map(serialize_func, batch, num_threads, use_threading)
      logging.info("Finished serializing batch %s", batch_num)

      file_num = file_num_offset + batch_num
      fn = file_name.replace('.tfrecord', '_' + str(file_num) + '.tfrecord')
      file_names.append(fn)

      logging.info("Writing batch %s", batch_num)
      writer = tf.python_io.TFRecordWriter(fn)
      for serials in all_serials:
        for serial in serials:
          writer.write(serial)
      logging.info("Finished writing batch %s", batch_num)
      writer.close()

    return file_names

  def multi_read_examples(self, file_name_iter, num_threads=1, key_type='slot', return_plugged=False, use_threading=False, skip_fails=False):
    """Pours the arrays then writes the examples to tfrecords in a multithreading manner. It creates one example per 'row', i.e. axis=0 of the arrays. All arrays must have the same axis=0 dimension and must be of a type that can be written to a tfrecord

    Parameters
    ----------
    funnel_dicts : list of dicts(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
      The inputs to the waterwork's full pour functions. There is exactly one funnel_dict for every process.
    file_name : str
      The name of the tfrecord file to write to. An extra '_<num>' will be added to the name.
    file_num_offset : int
      A number that controls what number will be appended to the file name (so that files aren't overwritten.)

    """
    # tap_dicts = []
    file_names = []
    for file_name in file_name_iter:
      file_names.append(file_name)
    tap_dicts = self._files_to_tap_dicts(file_names)

    funnel_dicts = []
    if skip_fails:
      try:
        funnel_dicts = self.multi_pump(tap_dicts, num_threads, key_type, return_plugged, use_threading)
      except Exception:
        logging.warn("Batched %s failed. Skipping.", file_names)
        funnel_dicts = []
    else:
      funnel_dicts = self.multi_pump(tap_dicts, num_threads, key_type, return_plugged, use_threading)

    return funnel_dicts

  def pour(self, funnel_dict=None, key_type='tube', return_plugged=False):
    """Run all the operations of the waterwork in the pour (or forward) direction.

    Parameters
    ----------
    funnel_dict : dict(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The inputs to the waterwork's full pour function.
    key_type : str ('tube', 'tuple', 'name')
      The type of keys to return in the return dictionary. Can either be the tube objects themselves (tube), the tank, output key pair (tuple) or the name (str) of the tube.

    Returns
    -------
    dict(
      keys - Tube objects, (or tuples if tuple_keys set to True). The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The outputs of the waterwork's full pour function

    """

    if funnel_dict is None:
      funnel_dict = {}

    # Set all the values of the funnels from the inputted arguments.
    evaled_dict = {}
    logging.debug("%s's funnel_dict - %s", self.name, sorted(funnel_dict))
    for ph, val in funnel_dict.iteritems():
      sl_obj = self.maybe_get_slot(ph)
      if sl_obj is not None:
        if sl_obj.plug is not None:
          raise ValueError(str(sl_obj) + ' has a plug. If you want to set the value dynamically then do funnel.unplug().')

        evaled_dict[sl_obj.name] = val
        sl_obj.set_val(val)
        if type(sl_obj.tube) is not Empty:
          sl_obj.tube.set_val(val)
      else:
        raise ValueError(str(ph) + ' is not a supported input into pour function')

    # Check that all funnels have a value
    logging.debug("%s's funnels - %s", self.name, sorted(self.funnels))
    for funnel_key in self.funnels:
      logging.debug("funnel_key - %s", funnel_key)
      funnel = self.funnels[funnel_key]
      if funnel.plug is not None:
        logging.debug("Plugging - %s", funnel_key)
        # funnel.set_val(funnel.plug(stand_funnel_dict))
      elif funnel.get_val() is None:
        raise ValueError("All funnels must have a set value. " + str(funnel) + " is not set.")

    # Run all the tanks (operations) in the pour direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pour_tank_order()
    logging.debug("%s's pour_tank_order - %s", self.name, [t.name for t in tanks])
    for tank in tanks:
      logging.debug("Pouring tank - %s", tank.name)

      # Fill any of the slots that are to be plugged
      kwargs = {}
      for slot_name in tank.slots:
        slot = tank.slots[slot_name]
        if slot.plug is not None:
          slot.set_val(slot.plug(evaled_dict))

      kwargs = {k: tank.slots[k].get_val() for k in tank.slots}

      logging.debug("Inputting kwargs to pour - %s", {k: v for k, v in kwargs.iteritems()})

      try:
        tube_dict = tank.pour(**kwargs)
      except:
        logging.exception("Failure in pour of tank %s", tank.name)
        raise

      for key in tube_dict:
        slot = tank.tubes[key].slot

        if type(slot) is not Empty:
          slot.set_val(tube_dict[key])
          evaled_dict[slot.name] = tube_dict[key]
    # Create the dictionary to return
    r_dict = {}
    logging.debug("%s's taps - %s", self.name, sorted(self.taps))
    for tap_name in self.taps:
      logging.debug("setting tap_key - %s", tap_name)
      tap = self.taps[tap_name]

      if tap.plug is not None and not return_plugged:
        continue

      if key_type == 'tube':
        r_dict[tap] = tap.get_val()
      elif key_type == 'tuple':
        r_dict[tap.get_tuple()] = tap.get_val()
      elif key_type == 'str':
        r_dict[tap.name] = tap.get_val()
      else:
        raise ValueError(str(key_type) + " is an invalid key_type.")

    return r_dict

  def pump(self, tap_dict=None, key_type='slot', return_plugged=False):
    """Run all the operations of the waterwork in the pump (or backward) direction.

    Parameters
    ----------
    tap_dict : dict(
      keys - Tube objects. The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The inputs of the waterwork's full pump function
    key_type : str ('tube', 'tuple', 'name')
      The type of keys to return in the return dictionary. Can either be the tube objects themselves (tube), the tank, output key pair (tuple) or the name (str) of the tube.

    Returns
    -------
    dict(
      keys - Slot objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The outputs to the waterwork's full pump function.

    """
    if tap_dict is None:
      tap_dict = {}

    evaled_dict = {}
    # Set all the values of the taps from the inputted arguments.
    logging.debug("%s's tap_dict - %s", self.name, sorted(tap_dict))
    for tap, val in tap_dict.iteritems():
      tu_obj = self.maybe_get_tube(tap)
      if tu_obj is not None:
        if tu_obj.downstream_tube is not None:
          logging.warn("%s has downstream_tube %s. Setting that value instead.", tu_obj.name, tu_obj.downstream_tube.name)
          tu_obj = tu_obj.downstream_tube
        if tu_obj.plug is not None:
          raise ValueError(str(tu_obj) + ' has a plug. Cannot set the value of a tap which is plugged.')

        evaled_dict[tu_obj.name] = val
        tu_obj.set_val(val)
      else:
        raise ValueError(str(tap) + ' is not a supported form of input into pump function')

    # Check that all funnels have a value
    logging.debug("%s's taps - %s", self.name, sorted(self.taps))
    for tap_key in self.taps:
      logging.debug("tap_key - %s", tap_key)
      tap = self.taps[tap_key]
      if tap.plug is not None:
        logging.debug("Plugging - %s", tap_key)
        tap.set_val(tap.plug(evaled_dict))
      elif tap.get_val() is None:
        raise ValueError("All taps must have a set value. " + str(tap) + " is not set.")

    # Run all the tanks (operations) in the pump direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pump_tank_order()
    logging.debug("%s's pump_tank_order - %s", self.name, [t.name for t in tanks])
    for tank in tanks:
      logging.debug("Pumping tank - %s", tank.name)
      for tube_name in tank.tubes:
        tube = tank.tubes[tube_name]
        if tube.plug is not None:
          tube.set_val(tube.plug(evaled_dict))
      kwargs = {k: tank.tubes[k].get_val() for k in tank.tubes}

      logging.debug("Inputting kwargs to pour - %s", {k: v for k, v in kwargs.iteritems()})

      try:
        slot_dict = tank.pump(**kwargs)
      except:
        logging.exception("Failure in pump of tank %s", tank.name)
        raise

      for key in slot_dict:
        tube = tank.slots[key].tube

        if type(tube) is not Empty:
          tube.set_val(slot_dict[key])
          evaled_dict[tube.name] = slot_dict[key]
    # Create the dictionary to return
    r_dict = {}
    logging.debug("%s's funnels - %s", self.name, sorted(self.funnels))
    for funnel_name in self.funnels:
      funnel = self.funnels[funnel_name]

      if funnel.plug is not None and not return_plugged:
        continue

      if key_type == 'slot':
        r_dict[funnel] = funnel.get_val()
      elif key_type == 'tuple':
        r_dict[funnel.get_tuple()] = funnel.get_val()
      elif key_type == 'str':
        r_dict[funnel.name] = funnel.get_val()
      else:
        raise ValueError(str(key_type) + " is an invalid key_type.")

    return r_dict

  def read_and_decode(self, serialized_example, feature_dict, prefix=''):
    """Convert a serialized example created from an example dictionary from this transform into a dictionary of shaped tensors for a tensorflow pipeline.

    Parameters
    ----------
    serialized_example : tfrecord serialized example
      The serialized example to read and convert to a dictionary of tensors.
    feature_dict :
      A dictionary with the keys being the keys of the tap dict (and their shapes) and the values being the FixedLenFeature's with the proper values all filled in

    Returns
    -------
    dict of tensors
      The tensors created by decoding the serialized example and reshaping them.

    """
    feat_dict = {}
    for key in feature_dict:
      shape = feature_dict[key]['shape']
      tf_dtype = feature_dict[key]['tf_dtype']
      feat_dict[key] = tf.FixedLenFeature(shape, tf_dtype)

    features = tf.parse_single_example(
      serialized_example,
      features=feat_dict
    )

    return features

  def read_examples(self, file_name, key_type='slot', return_plugged=False):
    tap_dicts = self._files_to_tap_dicts([file_name])
    return self.pump(tap_dicts[0], key_type=key_type, return_plugged=return_plugged)

  def _files_to_tap_dicts(self, file_names):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    batched_arrays: dict of arrays
      The keys

    Returns
    -------
    np.ndarray
      The numpy array to transform into examples.

    """
    if not file_names:
      return []

    feature_dict_fn = re.sub(r'_?[0-9]*.tfrecord', '.pickle', file_names[0])

    if not os.path.isfile(feature_dict_fn):
      raise ValueError("Expected a feature_dict file named", feature_dict_fn)

    feature_dict = d.read_from_file(feature_dict_fn)

    tap_dicts = []
    for file_name in file_names:
      dataset = tf.data.TFRecordDataset(file_name)
      dataset = dataset.map(lambda a: self.read_and_decode(a, feature_dict))

      iter = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
      )

      init = iter.make_initializer(dataset)
      features = iter.get_next()

      with tf.Session() as sess:
        sess.run(init)

        tap_dict = {}
        try:
          while True:
            example_dict = sess.run(features)
            for key in example_dict:
              tap_dict.setdefault(key, [])
              tap_dict[key].append(example_dict[key])

        except tf.errors.OutOfRangeError:
          pass

      for key in tap_dict:
        tap_dict[key] = np.stack(tap_dict[key], axis=0)

      for key in sorted(feature_dict, reverse=True):
        np_dtype = feature_dict[key]['np_dtype']

        if np_dtype.char == 'U':
          tap_dict[key] = tap_dict[key].astype(str)
          tap_dict[key] = np.char.decode(tap_dict[key], encoding='utf-8')
        elif np_dtype.char == 'S':
          tap_dict[key] = tap_dict[key].astype(np.str)
        else:
          tap_dict[key] = tap_dict[key].astype(np_dtype)
      tap_dicts.append(tap_dict)

    return tap_dicts

  def save_to_file(self, file_name):
    if not file_name.endswith('pickle') and not file_name.endswith('pkl') and not file_name.endswith('dill'):
      raise ValueError("Waterwork can only be saved as a pickle.")
    save_dict = self._save_dict()
    d.save_to_file(save_dict, file_name)

  def _write_tap_dict(self, writer, tap_dict, func_dict, skip_keys=None):
    if skip_keys is None:
      skip_keys = []

    keys_to_write = [k for k in tap_dict if k not in skip_keys]

    num_examples = None
    for key in keys_to_write:
      array = tap_dict[key]
      num_examples = array.shape[0]
      break

    for row_num in xrange(num_examples):
      example_dict = {}
      for key in keys_to_write:
        flat = tap_dict[key][row_num].flatten()
        example_dict[key] = func_dict[key](flat)

      example = tf.train.Example(
        features=tf.train.Features(feature=example_dict)
      )
      writer.write(example.SerializeToString())

  def _serialize_tap_dict(self, tap_dict, func_dict, skip_keys=None):
    if skip_keys is None:
      skip_keys = []

    keys_to_write = [k for k in tap_dict if k not in skip_keys]

    num_examples = None
    for key in keys_to_write:
      array = tap_dict[key]
      num_examples = array.shape[0]
      break

    serials = []
    for row_num in xrange(num_examples):
      example_dict = {}
      for key in keys_to_write:
        flat = tap_dict[key][row_num].flatten()
        example_dict[key] = func_dict[key](flat)

      example = tf.train.Example(
        features=tf.train.Features(feature=example_dict)
      )
      serials.append(example.SerializeToString())
    return serials

  def _get_feature_dicts(self, tap_dict):
    func_dict = {}
    feature_dict = {}
    for key in tap_dict:
      array = tap_dict[key]
      num_examples = array.shape[0]

      if array.shape[0] != num_examples:
        raise ValueError("All arrays must have the same size first dimesion in order to split them up into individual examples")

      if array.dtype not in (np.int32, np.int64, np.bool, np.float32, np.float64) and array.dtype.type not in (np.string_, np.unicode_):
        raise TypeError("Only string and number types are supported. Got " + str(array.dtype))
      feature_dict[key] = {}
      feature_dict[key]['np_dtype'] = array.dtype
      feature_dict[key]['tf_dtype'] = feat.select_tf_dtype(array.dtype)
      feature_dict[key]['shape'] = array.shape[1:]
      func_dict[key] = feat.select_feature_func(array.dtype)

    return feature_dict, func_dict

  def write_tap_dicts(self, tap_dicts, file_name, skip_keys=None):

    feature_dict, func_dict = self._get_feature_dicts(tap_dicts[0])

    writer = tf.python_io.TFRecordWriter(file_name)
    for tap_dict in tap_dicts:
      if not tap_dict:
        continue
      self._write_tap_dict(writer, tap_dict, func_dict, skip_keys)

    feature_dict_fn = re.sub(r'_?[0-9]*.tfrecord', '.pickle', file_name)
    d.save_to_file(feature_dict, feature_dict_fn)

    writer.close()

  def write_examples(self, funnel_dict, file_name):
    """Pours the array then writes the examples to tfrecords. It creates one example per 'row', i.e. axis=0 of the arrays. All arrays must have the same axis=0 dimension and must be of a type that can be written to a tfrecord

    Parameters
    ----------
    funnel_dict : dict(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The inputs to the waterwork's full pour function.
    file_name : str
      The name of the tfrecord file to write to.

    """
    if not file_name.endswith('.tfrecord'):
      raise ValueError("file_name must end in '.tfrecord'. Got: ", file_name)

    dir = file_name.split('/')[:-1]
    d.maybe_create_dir(*dir)

    writer = tf.python_io.TFRecordWriter(file_name)
    tap_dict = self.pour(funnel_dict, key_type='str')

    feature_dict, func_dict = self._get_feature_dicts(tap_dict)

    self._write_tap_dict(writer, tap_dict, func_dict)

    feature_dict_fn = re.sub(r'_?[0-9]*.tfrecord', '.pickle', file_name)
    d.save_to_file(feature_dict, feature_dict_fn)
    writer.close()

    return file_name


def recon_and_pump(args):
  # jpype.attachThreadToJVM()
  save_dict = args[0]
  tap_dict = args[1]
  key_type = args[2]
  return_plugged = args[3]

  ww = Waterwork()
  ww._from_save_dict(save_dict)

  funnel_dict = ww.pump(tap_dict, key_type, return_plugged)
  return funnel_dict


def inf_gen(val):
  while True:
    yield val
