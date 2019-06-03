"""Shape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut
import random


class BertRandomInsert(ta.Tank):
  """A specialized tank built specifically for the BERT ML model. Randomly inserts a [SEP] token at the end of some sentence in a row, then with some probability overwrites the latter part of the string with a randomly selected sentence. For more information or motivation look up the bert model.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'bert_random_insert'
  slot_keys = ['a', 'ends', 'num_tries', 'random_seed']
  tube_keys = ['target', 'removed', 'num_tries', 'ends', 'random_seed']

  def _pour(self, a, ends, num_tries, random_seed):
    """

    Parameters
    ----------
    a: np.ndarray
      The array that will have the [SEP] and [CLS] tags inserted as well as randomly setting half of the rows to having random sentences after the first [SEP] tag.
    ends: np.ndarray of bools
      An array of the same shape as 'a' which marks the end of a sentence with a True.
    num_tries: int
      The number of times to try and find a random sentences to replace the second part of the 'a' array.
    Returns
    -------
    dict(
      target: np.ndarray
        The array a with the [SEP] and [CLS] tags as well a some randomly overwritten second sentences.
      removed: np.ndarray
        A array with the same size as target that contains all the substrings that were overwritten.
      ends: np.ndarray of bools
        An array of the same shape as 'a' which marks the end of a sentence with a True.
      num_tries: int
        The number of times to try and find a random sentences to replace the second part of the 'a' array.
    )

    """
    self.temp_ends = ends
    np.random.seed(random_seed)
    sepped_array = []
    max_row_len = a.shape[1] + 3
    for row_num in xrange(a.shape[0]):
      sep_index = np.random.choice(np.where(ends[row_num])[0])
      last_sent_index = np.where(ends[row_num])[0][-1]
      row = a[row_num].tolist()
      row.insert(last_sent_index + 1, '[SEP]')
      row.insert(sep_index + 1, '[SEP]')
      row.insert(0, '[CLS]')
      sepped_array.append(row)

    target = []
    removed = []
    for row_num in xrange(len(sepped_array)):
      row = sepped_array[row_num]

      removed_chunk = ['[NA]'] * max_row_len
      if np.random.rand() < 0.5:
        sep_index = row.index('[SEP]')

        space_left = max_row_len - (sep_index + 1)
        chosen_chunk = None
        for _ in xrange(num_tries):
          rand_row = sepped_array[np.random.choice(a.shape[0])]
          chunk = rand_row[rand_row.index('[SEP]') + 1:]
          chunk = chunk[:chunk.index('[SEP]') + 1]
          if len(chunk) < space_left:
            if chosen_chunk is None or len(chunk) > len(chosen_chunk):
              chosen_chunk = chunk
        if chosen_chunk is not None:
          removed_chunk = row[sep_index + 1:]
          removed_chunk = ['[NA]'] * (max_row_len - len(removed_chunk)) + removed_chunk
          row = row[:sep_index + 1] + chosen_chunk
          row = row + [''] * (max_row_len - len(row))

      target.append(row)
      removed.append(removed_chunk)

    target = np.array(target)
    removed = np.array(removed)
    # if self.name == 'Pack_1':

    return {'target': target, 'removed': removed, 'num_tries': num_tries, 'ends': ends, 'random_seed': random_seed}

  def _pump(self, target, removed, num_tries, ends, random_seed):
    """Execute the Shape tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The array a with the [SEP] and [CLS] tags as well a some randomly overwritten second sentences.
    removed: np.ndarray
      A array with the same size as target that contains all the substrings that were overwritten.
    ends: np.ndarray of bools
      An array of the same shape as 'a' which marks the end of a sentence with a True.
    num_tries: int
      The number of times to try and find a random sentences to replace the second part of the 'a' array.

    Returns
    -------
    dict(
      a: np.ndarray
        The array that will have the [SEP] and [CLS] tags inserted as well as randomly setting half of the rows to having random sentences after the first [SEP] tag.
      ends: np.ndarray of bools
        An array of the same shape as 'a' which marks the end of a sentence with a True.
      num_tries: int
        The number of times to try and find a random sentences to replace the second part of the 'a' array.
    )

    """
    mask = removed != '[NA]'
    target[mask] = removed[mask]
    a = []
    for row_num in xrange(target.shape[0]):
      row = target[row_num]
      row = row[(row != '[CLS]') & (row != '[SEP]')]
      a.append(row)
    a = np.stack(a)
    return {'a': a, 'num_tries': num_tries, 'ends': ends, 'random_seed': random_seed}
