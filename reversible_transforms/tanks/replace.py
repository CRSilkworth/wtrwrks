import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


class Replace(ta.Tank):
  """The Replace class. Handles 'a's of np.ndarray type.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  slot_keys = ['a', 'mask', 'replace_with']
  tube_dict = {
    'target': np.ndarray,
    'replaced_vals': np.ndarray,
    'mask': int,
    'replace_with_shape': tuple
  }

  def _pour(self, a, mask, replace_with):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to replace the values of.
    mask : np.ndarray
      An array of booleans which define which values of a are to be replaced.
    replace_with : np.ndarray
      The values to replace those values of 'a' which have a corresponding 'True' in the mask.

    Returns
    -------
    dict(
      'target': np.ndarray
        The inputted array 'a' with all the proper values replaced.
      'mask' : np.ndarray
        An array of booleans which define which values of a are to be replaced.
      'replaced_vals' : np.ndarray
        The values that were overwritten.
      'replace_with_shape' : tuple
        The shape of the inputted 'replaced_with' array.
    )

    """
    target = ut.maybe_copy(a)
    replaced_vals = target[mask]
    target[mask] = replace_with
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'mask': mask, 'replaced_vals': replaced_vals.flatten(), 'replace_with_shape': replace_with.shape}

  def _pump(self, target, mask, replaced_vals, replace_with_shape):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The inputted array a with all the proper values replaced.
    mask : np.ndarray
      An array of booleans which define which values of a are to be replaced.
    replaced_vals : np.ndarray
      The values that were overwritten.
    replace_with_shape : tuple
      The shape of the inputted replaced_with array.

    Returns
    -------
    dict(
      'a' : np.ndarray
        The array to replace the values of.
      'mask' : np.ndarray
        An array of booleans which define which values of a are to be replaced.
      'replace_with' : np.ndarray
        The values to replace those values of 'a' which have a corresponding 'True' in the mask.
    )

    """
    a = ut.maybe_copy(target)
    replace_with = a[mask]

    masked_shape = a[mask].shape
    a[mask] = replaced_vals.reshape(masked_shape)

    if replace_with_shape:
      num_elements = np.prod(replace_with_shape)
    else:
      num_elements = 1

    if num_elements == 1:
      replace_with = replace_with.flatten()[0].reshape(replace_with_shape)

    a = a.astype(replaced_vals.dtype)
    return {'a': a, 'mask': mask, 'replace_with': replace_with}
