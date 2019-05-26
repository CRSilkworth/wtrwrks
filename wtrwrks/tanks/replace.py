"""Replace tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Replace(ta.Tank):
  """The defintion of the Replace tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  slot_keys = ['a', 'mask', 'replace_with']
  tube_keys = ['target', 'mask', 'replaced_vals', 'replace_with_shape']

  def _pour(self, a, mask, replace_with):
    """Execute the Replace tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array which has values that are to be replaced.
    mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values are to be replaced.
    replace_with: np.ndarray
      The values to be used to replace the corresponding values in 'a'.

    Returns
    -------
    dict(
      target: np.ndarray of same type as 'a'
        The array with the necessary values replaced.
      mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values are to be replaced.
      replaced_vals: np.ndarray of same type as 'a'
        The values that were overwritten when they were replaced by the replace_with values.
      replace_with_shape: list of ints
        The original shape of the replace_with array.
    )

    """
    # Cast the replace_with values to an array.
    replace_with = np.array(replace_with)
    target = ut.maybe_copy(a)

    # Save the values that are going to be replaced.
    replaced_vals = target[mask]

    # Replace the values with the values found in replace_with.
    target[mask] = replace_with

    # If the mask is all false then save the actual replace_with values, since
    # that information would otherwise be lost. Otherwise just save the shape.
    if mask.any():
      replace_with_shape = replace_with.shape
    else:
      replace_with_shape = (replace_with,)

    return {'target': target, 'mask': mask, 'replaced_vals': replaced_vals.flatten(), 'replace_with_shape': replace_with_shape}

  def _pump(self, target, mask, replaced_vals, replace_with_shape):
    """Execute the Replace tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of same type as 'a'
      The array with the necessary values replaced.
    mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values are to be replaced.
    replaced_vals: np.ndarray of same type as 'a'
      The values that were overwritten when they were replaced by the replace_with values.
    replace_with_shape: list of ints
      The original shape of the replace_with array.

    Returns
    -------
    dict(
      a: np.ndarray
        The array which has values that are to be replaced.
      mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values are to be replaced.
      replace_with: np.ndarray
        The values to be used to replace the corresponding values in 'a'.
    )

    """
    a = ut.maybe_copy(target)
    replace_with = a[mask]

    masked_shape = a[mask].shape
    a[mask] = replaced_vals.reshape(masked_shape)

    if mask.any():
      # If the replace_with had any shape then find the number of elements.
      # Otherwise it's just a scalar and has one element
      if replace_with_shape:
        num_elements = np.prod(replace_with_shape)
      else:
        num_elements = 1

      # If there was only one element then just save the replace_with value
      # as the first element. Reshape it so it matches it's former shape.
      if num_elements == 1:
        replace_with = replace_with.flatten()[0].reshape(replace_with_shape)
    else:
      # Otherwise the replace_with_shape is actually the replace_with values.
      replace_with = replace_with_shape[0]
    a = a.astype(replaced_vals.dtype.type)
    return {'a': a, 'mask': mask, 'replace_with': replace_with}
