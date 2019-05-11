import transform as n
import numpy as np
import reversible_transforms.tanks.tank_defs as td
from reversible_transforms.waterworks.empty import empty
import os

class StringTransform(n.Transform):
  """Class used to create mappings from a raw string to a vector and back.

  Parameters
  ----------
  df : pd.DataFrame
    The dataframe with all the data used to define the mappings.
  columns : list of strs
    The column names of all the relevant columns that make up the data to be taken from the dataframe
  language: str
    The language the strings are in.
  lower: bool
    Whether or not to lowercase all the strings.
  half_width: bool
    Whether or not to convert all full width characters to half width.
  lemmatize: bool
    Whether or not to stem or lemmatize the words.
  remove_stopwords: bool
    Whether or not to remove all the commonly occuring words.
  max_vocab_size:
    The maximum number of words allowed to be a part of the known vocabulary.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to rereate the transform object

  Attributes
  ----------
  attribute_list : list of strs
    The list of attributes that need to be saved in order to fully reconstruct the transform object.

  """
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'index_to_word': None, 'word_to_index': None, 'max_sent_len': 20, 'tokenizer': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'unk_index': None, 'delimiter': ' '}

  def _setattributes(self, **kwargs):
    super(StringTransform, self)._setattributes(self.attribute_dict, **kwargs)

    if self.index_to_word is None:
      raise ValueError("Must supply index_to_word mapping.")
    # if self.index_to_word is not None and self.index_to_word[0] != '__UNK__':
    #   raise ValueError("First element of the index_to_word map must be the default '__UNK__' token")


  def calc_global_values(self, array, verbose=True):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe
    language: str
      The language the strings are in.
    lower: bool
      Whether or not to lowercase all the strings.
    half_width: bool
      Whether or not to convert all full width characters to half width.
    lemmatize: bool
      Whether or not to stem or lemmatize the words.
    remove_stopwords: bool
      Whether or not to remove all the commonly occuring words.
    max_vocab_size:
      The maximum number of words allowed to be a part of the known vocabulary.
    """
    self.input_dtype = array.dtype
    self.word_to_index = {
      word: num for num, word in enumerate(self.index_to_word)
    }

  def define_waterwork(self, array=empty):
    tokens, tokens_slots = td.tokenize(strings=array, max_len=self.max_sent_len)
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['strings'].set_name('input')
    tokens_slots['tokenizer'].set_name('tokenizer')
    tokens_slots['delimiter'].set_name('delimiter')

    if self.lower_case:
      tokens, tokens_slots = td.lower_case(tokens['target'])
      tokens['diff'].set_name('lower_case_diff')
    if self.half_width:
      tokens, tokens_slots = td.half_width(tokens['target'])
      tokens['diff'].set_name('half_width_diff')
    if self.lemmatize:
      tokens, tokens_slots = td.lemmatize(tokens['target'])
      tokens['diff'].set_name('lemmatize_diff')
      tokens_slots['lemmatizer'].set_name('lemmatizer')
    if self.unk_index is not None:
      isin, _ = td.isin(tokens['target'], self.index_to_word + [''])
      mask, _ = td.logical_not(isin['target'])
      tokens, _ = td.replace(isin['a'], mask['target'], self.index_to_word[self.unk_index])
      tokens['replaced_vals'].set_name('missing_vals')

    indices, _ = td.cat_to_index(tokens['target'], self.word_to_index)
    indices['target'].set_name('indices')

    if self.unk_index is None:
      indices['missing_vals'].set_name('missing_vals')

  def _get_funnel_dict(self, array, tokenizer=None, delimiter=None, lemmatizer=None, prefix=''):
    funnel_dict = {'input': array}

    if tokenizer is None and self.tokenizer is None:
      raise ValueError("No tokenizer set for this Transform. Must supply one as input into pour.")
    elif tokenizer is not None:
      funnel_dict['tokenizer'] = tokenizer
    else:
      funnel_dict['tokenizer'] = self.tokenizer

    if delimiter is not None:
      funnel_dict['delimiter'] = delimiter
    else:
      funnel_dict['delimiter'] = self.delimiter

    if self.lemmatize and lemmatizer is None and self.lemmatizer is None:
      raise ValueError("No lemmatizer set for this Transform. Must supply one as input into pour.")
    elif self.lemmatize and lemmatizer is not None:
      funnel_dict['lemmatizer'] = lemmatizer
    elif self.lemmatize:
      funnel_dict['lemmatizer'] = self.lemmatizer

    return self._add_name_to_dict(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = {k: tap_dict[self._add_name(k, prefix)] for k in ['indices', 'missing_vals', 'tokenize_diff']}
    if self.lower_case:
      r_dict['lower_case_diff'] = tap_dict['lower_case_diff']
    if self.half_width:
      r_dict['half_width_diff'] = tap_dict['half_width_diff']
    if self.lemmatize:
      r_dict['lemmatize_diff'] = tap_dict['lemmatize_diff']

    return r_dict

  def _get_tap_dict(self, indices, missing_vals, tokenize_diff, lower_case_diff=None, half_width_diff=None, lemmatize_diff=None, delimiter=None, prefix=''):
    if delimiter is None:
      delimiter = self.delimiter

    tap_dict = {
      'indices': indices,
      'missing_vals': missing_vals,
      ('CatToIndex_0', 'cat_to_index_map'): self.word_to_index,
      ('CatToIndex_0', 'input_dtype'): self.input_dtype,
      ('Tokenize_0', 'delimiter'): delimiter,
      ('Tokenize_0', 'tokenizer'): self.tokenizer,
      ('Tokenize_0', 'diff'): tokenize_diff
    }
    if self.lower_case:
      u_dict = {('LowerCase_0', 'diff'): lower_case_diff}
      tap_dict.update(u_dict)
    if self.half_width:
      u_dict = {('HalfWidth_0', 'diff'): half_width_diff}
      tap_dict.update(u_dict)
    if self.lemmatize:
      u_dict = {
        ('Lemmatize_0', 'diff'): lemmatize_diff,
        ('Lemmatize_0', 'lemmatizer'): self.lemmatizer
      }
      tap_dict.update(u_dict)
    if self.unk_index is not None:
      u_dict = {
        ('CatToIndex_0', 'missing_vals'): [''] * indices[indices == -1].size,
        ('Replace_0', 'mask'): indices == self.unk_index,
        ('Replace_0', 'replace_with_shape'): (1,),
        ('IsIn_0', 'b'): self.index_to_word[self.unk_index]
      }
      tap_dict.update(u_dict)

    return self._add_name_to_dict(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._add_name('input', prefix)]
  # def forward_transform(self, array, verbose=True):
  #   """Convert a row in a dataframe to a vector.
  #
  #   Parameters
  #   ----------
  #   row : pd.Series
  #     A row in a dataframe where the index is the column name and the value is the column value.
  #   verbose : bool
  #     Whether or not to print out warnings.
  #
  #   Returns
  #   -------
  #   np.array(
  #     shape=[len(self)],
  #     dtype=np.float64
  #   )
  #     The vectorized and normalized data.
  #
  #   """
  #   # Define the key word arguments for the normalizer, ensuring that inverse
  #   # is set to false.
  #   kwargs = {}
  #   kwargs.update(self.normalizer_kwargs)
  #   kwargs['inverse'] = False
  #
  #   # Find the indices for each word, filling with -1 if the vector is
  #   # longer than the number of tokens.
  #   indices = -1 * np.ones([array.shape[0], self.max_sent_len], dtype=np.int64)
  #   diff_string = np.empty([array.shape[0]], dtype=object)
  #   unks = np.empty([array.shape[0], self.max_sent_len], dtype=object)
  #   unks.fill('')
  #   for row_num, string in enumerate(array[:, self.col_index]):
  #     r_dict = self.normalizer(string, self.max_sent_len, **kwargs)
  #     tokens = r_dict['tokens']
  #
  #     for token_num, token in enumerate(tokens):
  #       # If the word isn't known, fill it with the 'UNK' index.
  #       # Otherwise pull out the relevant index.
  #       if token not in self.word_to_index:
  #         index = self.word_to_index['__UNK__']
  #         unks[row_num, token_num] = token
  #       else:
  #         index = self.word_to_index[token]
  #
  #       # Fill the array and increase the token number.
  #       indices[row_num, token_num] = index
  #     diff_string[row_num] = r_dict['diff_string']
  #   return {'data': indices, 'diff_string': diff_string.astype(np.unicode), 'unknowns': unks}
  #
  # def backward_transform(self, arrays_dict, verbose=True):
  #   """Convert the vectorized and normalized data back into it's raw form. Although a lot of information is lost and so it's best to also keep some outside reference to the original row in the dataframe.
  #
  #   Parameters
  #   ----------
  #   vector : np.array(
  #     shape=[len(self)],
  #     dtype=np.int64
  #   )
  #     The vectorized and normalized data.
  #   verbose : bool
  #     Whether or not to print out warnings.
  #
  #   Returns
  #   -------
  #   row : pd.Series
  #     A row in a dataframe where the index is the column name and the value is the column value.
  #
  #   """
  #   # Fill a dictionary representing the original row. Convert the index to
  #   # a category value
  #   kwargs = {}
  #   kwargs.update(self.normalizer_kwargs)
  #   kwargs['inverse'] = True
  #
  #   array = np.empty([arrays_dict['data'].shape[0], 1], dtype=object)
  #   for row_num, (indices, diff_string, unks) in enumerate(zip(arrays_dict['data'], arrays_dict['diff_string'], arrays_dict['unknowns'])):
  #
  #     tokens = [self.index_to_word[i] if i != 0 else unks[num] for num, i in enumerate(indices) if i != -1]
  #
  #     string = self.normalizer({'tokens': tokens, 'diff_string': diff_string}, self.max_sent_len, **kwargs)
  #
  #     array[row_num] = string
  #
  #   return array.astype(self.input_dtype)
  #
  # def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len
