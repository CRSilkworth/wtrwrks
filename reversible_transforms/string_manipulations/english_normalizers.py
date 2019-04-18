import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.stem.wordnet as wordnet
import reversible_transforms.string_manipulations.diff as df


word_net_lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))


def default_normalizer(string_or_dict, lemmatize=False, lowercase=False, remove_stopwords=False, inverse=False):
  """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize an English string in a consistent manner. In the case that inverse is True then it undoes everything to give back the original string.

  Parameters
  ----------
  string : str
    The string to be tokenized and normalized

  Returns
  -------
  list of strs
    The tokenized and normalized strings

  """
  if not inverse:
    string = string_or_dict
    r_tokens = []

    # Split the string into individual words/punctuation.
    tokens = nltk.word_tokenize(string)

    # If lemmatize, then stem the words according to their part of speech.
    if lemmatize:
      pos_tags = nltk.pos_tag(tokens)

    for token_num, token in enumerate(tokens):
      # If you're lower casing everything then do it here.
      if lowercase:
        token = token.lower()

      # If you're removing stop words and this is a stop word continue.
      if remove_stopwords and token in stopwords:
        continue

      # If lemmatize, then lemmatize it
      if lemmatize:
        pos_tag = get_wordnet_pos(pos_tags[token_num][1])
        token = word_net_lemmatizer.lemmatize(token, pos_tag)

      r_tokens.append(token)

    diff_string = df.get_diff_string(' '.join(r_tokens), string)

    return {'tokens': r_tokens, 'diff_string': diff_string}
  else:
    string = ' '.join(string_or_dict['tokens'])
    string = df.reconstruct(string, string_or_dict['diff_string'])

    return string


def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN
