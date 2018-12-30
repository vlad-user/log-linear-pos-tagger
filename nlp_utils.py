import re
import string
import collections
import itertools
import operator
import pickle
import gc
import os
import random

import numpy as np
from scipy.misc import logsumexp
import scipy.optimize as optimize

NUMBER = 'NUMBER'
START = 'START'
STOP = 'STOP'

SPECIAL_WORDS = [NUMBER, START, STOP]

class NotFitError(Exception):
    pass

    def __init__(self):
        self.msg = "The object is not fit. Call fit() method."

    def __repr__(self):
        return self.msg

class Counter(collections.Counter):
    """Helper for counting elements in iterable objects."""

    def __init__(self, items):
        super(Counter, self).__init__(items)

    def least_common(self, n=10):
        return heapq.nsmallest(
                n, self.items(), key=operator.itemgetter(1))

class TextProcessor:
    """Helper class for processing text.

    This class should be used for processing general text.
    For PenTreebank standard TextProcessor_v2 should be used."""

    def sent_tokenize(self, text, sent_pattern=None):
        """Tokenizes text to sentences based on `sent_pattern`.

        Args:
            `text`: A string of text that needs to be tokenized.
            `sent_pattern`: (Optional) A regex pattern based on which a
                text needs to be tokenized. If None, uses the default
                pattern.
        Returns:
            A list of tokenized to sentences text.
        """
        
        sent_pattern = (
                r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s"
                if sent_pattern is None else sent_pattern)
        compiled = re.compile(sent_pattern)
        return re.split(compiled, text)
        
    def word_tokenize(self, text, word_pattern=None):
        """Tokenizes a text to a words.

        **NOTE:** The text must be first tokenzed with 'sent_tokenize'
        function with default pattern.

        Args:
            `text`: A string that needs to be tokenized to words.
            `word_pattern`: (Optional) A regex pattern based on which a text
                needs to be tokenized. If None, uses the default pattern.

        Returns:
            A list of tokenized text.
        """

        word_pattern = (r"(\W+)" if word_pattern is None else word_pattern)
        compiled = re.compile(word_pattern)
        return re.split(compiled, text)

        return text.split(' ')

    def word_tokenize_clean(self, text):
        """Tokenizes text and removes everything except words from `text`."""
        words = self.word_tokenize(text)
        words = [w.replace(' ', '') for w in words]
        words = [w for w in words if w not in string.punctuation]

        return words

    def word_pos_tokenize(self, text):
        """Tokenizes a sentence `sent` to a word and its part of speech tag.

        Args:
            `text`: A sentence tokenized by 'sent_tokenize' function.

        Returns:
            A list of tuples (word, POS)
        """

        tokenized = [x.split('_') for x in self.word_tokenize_clean(text)]
        tokenized = [tuple(x) for x in tokenized
                     if (x[1] not in string.punctuation
                        and x[0] != ''
                        and x[1] != '')]
        return tokenized

    def remove_special_characters(
            self, text,
            characters=None):
        """Removes all special characters from a `text` string.
        
        Args:
            `text`: A text that needs to be cleaned from special characters.
            `characters`: A string with all characters that need to be
                removed. Default characters are all punctuation without
                the underscores, hyphens and dots.

        Returns:
            A `text` without special `characters`.
        """
        if characters is None:
            characters = string.punctuation.replace('_', '')
            characters = characters.replace('-', '')
            characters = characters.replace('.', '')
            characters = characters.replace(',', '')
        tokens = [w for s in self.sent_tokenize(text)
                  for w in self.word_tokenize(text)]

        pattern = re.compile("(\b*[{}]\b*)".format(re.escape(characters)))
        return " ".join([pattern.sub('  ', t) for t in tokens])

    def replace_numbers(self, text):
        """Replaces numbers in text by a global variable `NUMBER`."""
        num_pattern = re.compile(r"[0-9]+")
        return re.sub(num_pattern, NUMBER, text)

    def pipe_text_processing(self, text, func_list=None):
        """Applies functions in `func_list` on `text` one after the other.

        Args:
            `text`: A string for which a processing should be applied.
            `func_list`: (Optional) A list of functions that accept a
                string as input and output string. The last function
                in `func_list` may output not a text (e.g. tokens).
                Default order is 
                1. Removes special characters.
                2. Tokenizes text to a list of tuples (word, POS).

        Returns:
            Processed text.
        """

        if func_list is None:
            #text = self.remove_special_characters(text)
            result = self.word_pos_tokenize(text)

        else:
            for f in func_list:
                text = f(text)

            result = text
        return result

    def merge_words_and_tags(self, raw_words, words, tags):
        """Merges words and tags."""
        result = []
        i = 0
        for w in raw_words:
            if i < len(words) and w == words[i]:
                result.append('_'.join([w, tags[i]]))
                i += 1
            else:
                result.append(w)

        sent = ' '.join(result).replace(' .', ' ._.').replace(',', ',_,')
        return sent

class TextProcessor_v2:
    """Helper for processing PenTreebank text."""
    def sent_tokenize(self, text):
        """Splits text based on new line character."""
        sents = text.split('\n')
        return [s for s in sents if s != '']

    def word_tokenize(self, text):
        """Tokenizes sentence based on space character."""
        return text.split(' ')

    def word_pos_tokenize(self, text):
        """Splits PenTreebank text into a tuples of (word, tag)."""
        words_and_tags = [tuple(w.split('_')) for w in self.word_tokenize(text)]
        return words_and_tags

    def merge_words_and_tags(self, words, tags):
        """Merges list of words and tags into a PenTreebank sentence.

        Args:
            `words`: A list of words.
            `tags`: A list of tags.
        Returns:
            Merged to words and tags sentence.
        """
        result = []
        for (w, t) in zip(words, tags):
            result.append('_'.join([w, t]))
        return ' '.join(result)

class Vectorizer:
    """Vectorizes a tokenized sentences to binary feature vectores.

    The indicator features are implemented by storing a lists of
    indices. This is by far the fastest and efficient options to
    implement. Storing features as numpy array is impossible in terms
    of memory since the sizes of vectors may be tens or even hundred
    of thousands and each sentence has such vectors as the number of
    words and punctuation symbols in a sentence. The option of using
    sparse vectors is better but still takes more memory and
    significantly more time to train."""
    
    def __init__(self, rare_word_thresh=2, raw_feature_converter=None):
        """Creates a new Vectorizer object.

        Args:
            `rare_word_thresh`: (Optional) A integer indicating the
                minimum number of appearances for a feature to be
                used in a model. Default value is 10.
            `raw_feature_converter`: (Optional) A function that accepts
                a list of words in a sentence, `k` - index of a
                current word and tags that previously appeared in a
                sentence (prior to index `k`) and returns a dictionary
                with names of a features as keys and feature values as
                values. For example, see `_convert_to_raw_hist2()` 
                method (this is a default converter).
        """
        
        self._rare_word_thresh = rare_word_thresh
        if raw_feature_converter is None:
            self.convert_to_raw_hist = self._convert_to_raw_hist2
        else:
            self.convert_to_raw_hist = raw_feature_converter

    def fit(self, train_data):
        """Fits the vectorizer.
        
        Args:
            `train_data`: A list of lists of tokenized sentences s.t.
                each tokenized sentence consists of a
                tuple (word, POS tag)
        """

        raw_features = []
        tags = []
        for words_and_tags in train_data:
            h_and_t = self.words_and_tags_to_hists_and_tags(
                    words_and_tags)

            for hist, t in h_and_t:
                raw_features = (raw_features
                                + [(h, t) for h in hist])
                tags.append(t)
            tags = list(set(tags))


        feature_count = Counter(raw_features)
        filtered_features = [x for x in feature_count.keys()
                             if feature_count[x] >= self._rare_word_thresh]

        self._tags = list(set(tags))
        self._n_features = len(filtered_features)
        self._mapping = {}
        for f in filtered_features:
            self._mapping[f[0][0], f[0][1], f[1]] = len(self._mapping)

    def transform(self, data):
        """Returns a list of transformed binary features."""

        transformed = []
        for words_and_tags in data:
            hists_and_tags = self.words_and_tags_to_hists_and_tags(
                    words_and_tags)
            
            for hist, tag in hists_and_tags:
                transformed_dict = {
                    'linear_term': self.hist_and_tag_to_binary((hist, tag)),
                    'log_inner_sum': [self.hist_and_tag_to_binary((hist, t))
                                      for t in self.list_available_tags()]
                    }

                transformed.append(transformed_dict)

        return transformed

    def words_and_tags_to_hists_and_tags(self, words_and_tags):
        """Converts a list of words and tags to a list of histories and tags."""

        hists_and_tags = []
        words = [w[0] for w in words_and_tags]
        prev_tags = []
        try:
            for idx, (w, t) in enumerate(words_and_tags):
                dict_ = self.convert_to_raw_hist(
                        words, idx, prev_tags)
                prev_tags.append(t)
                raw_hist = [(name, val) for name, val in dict_.items()]
                hists_and_tags.append((raw_hist, t))
        except ValueError:

            print(len(words_and_tags))
            print(words_and_tags)
            raise
        return hists_and_tags

    def hist_and_tag_to_binary(self, hist_and_tag):
        """Converts `hist_and_tag` raw feature to its vectorized form.

        Args:
            hist_and_tag: A tuple (history, tag) where history is
            a a dict return by `_convert_to_raw_hist()` or similar
            methods. A tag is a POS tag.

        Returns:
            Vectorized numpy array.

        Raises:
            NotFitError: If vectorizer hasn't been fit. 
        """

        try:
            mapping = self._mapping
        except AttributeError as exc:
            raise NotFitError() from exc

        hist = hist_and_tag[0]
        tag = hist_and_tag[1]
        res = []
        for h in hist:
            if (h[0], h[1], tag) in self._mapping:
                res.append(self._mapping[h[0], h[1], tag])

        return res

    def list_available_tags(self):
        return self._tags

    @staticmethod
    def _convert_to_raw_hist_model2(words, idx, prev_tags):
        """See `_convert_to_raw_hist()`."""
        def _is_first(idx): return 0 == idx
        def _is_second(idx): return 1 == idx
        def _is_last(words, idx): return len(words) <= idx
        def _contains_digits(word): return any(x.isdigit() for x in word)

        dict_ = {}

        dict_['w_i'] = words[idx] # 100
        
        if _is_first(idx):
            dict_.update({
                    'w_i-1': START, # 106
                    't_i-1': START, # 104
                    't_i-1t_i-2': START # 103
                })
        elif _is_second(idx):
            dict_.update({
                    'w_i-1': words[idx-1], # 106
                    't_i-1': prev_tags[idx-1], # 104
                    't_i-1t_i-2': prev_tags[idx-1] + ' ' + START # 103
                })
        else:
            dict_.update({
                    'w_i-1': words[idx-1], # 106
                    't_i-1': prev_tags[idx-1], # 104
                    't_i-1t_i-2': prev_tags[idx-1] + ' ' + prev_tags[idx-2] # 103
                })

        dict_['w_i+1'] = (STOP # 107
                          if _is_last(words, idx+1)
                          else words[idx+1])
        
        if _contains_digits(words[idx]):
            dict_['has_digits'] = 1

        if '-' in words[idx]:
            dict_['has_hyphen'] = 1

        if any(w.isupper() for w in words[idx]):
            dict_['has_upper'] = 1

        if len(words[idx]) >= 4:
            dict_['prefix4'] = words[idx][:4]
            dict_['suffix4'] = words[idx][-4:]

        if len(words[idx]) >= 3:
            dict_['prefix3'] = words[idx][:3]
            dict_['suffix3'] = words[idx][-3:]

        if len(words[idx]) >= 2:
            dict_['prefix2'] = words[idx][:2] # 101
            dict_['suffix2'] = words[idx][-2:] # 102

        if len(words[idx]) >= 1:
            dict_['prefix1'] = words[idx][0] # 101
            dict_['suffix1'] = words[idx][-1] # 102


        return dict_

    def _convert_to_raw_hist2(self, words, idx, prev_tags):
        """See `_convert_to_raw_hist()`"""
        def _is_first(idx): return 0 == idx
        def _is_second(idx): return 1 == idx
        def _is_last(words, idx): return len(words) <= idx
        def _contains_digits(word): return any(x.isdigit() for x in word)
        dict_ = {}
        if words[idx] == NUMBER:
            dict_['is_number'] = 1
        else:
            dict_['w_i'] = words[idx] # 100
            
            if _is_first(idx):
                dict_.update({
                        'w_i-1': START, # 106
                        't_i-1': START, # 104
                        't_i-1t_i-2': START # 103
                    })
            elif _is_second(idx):
                dict_.update({
                        'w_i-1': words[idx-1], # 106
                        't_i-1': prev_tags[idx-1], # 104
                        't_i-1t_i-2': prev_tags[idx-1] + ' ' + START # 103
                    })
            else:
                dict_.update({
                        'w_i-1': words[idx-1], # 106
                        't_i-1': prev_tags[idx-1], # 104
                        't_i-1t_i-2': prev_tags[idx-1] + ' ' + prev_tags[idx-2] # 103
                    })

            dict_['w_i+1'] = (STOP # 107
                              if _is_last(words, idx+1)
                              else words[idx+1])
            if _contains_digits(words[idx]):
                dict_['has_digits'] = 1

            if '-' in words[idx]:
                dict_['has_hyphen'] = 1

            if any(w.isupper() for w in words[idx]):
                dict_['has_upper'] = 1



            if len(words[idx]) >= 3:
                dict_['prefix3'] = words[idx][:3]
                dict_['suffix3'] = words[idx][-3:]

            if len(words[idx]) >= 2:
                dict_['prefix2'] = words[idx][:2] # 101
                dict_['suffix2'] = words[idx][-2:] # 102

            if len(words[idx]) >= 1:
                dict_['prefix1'] = words[idx][0] # 101
                dict_['suffix1'] = words[idx][-1] # 102


        return dict_


    def _convert_to_raw_hist(self, words, idx, prev_tags):
        """Converts a history to a dict of features.

        Each dict has feature names as keys and feature values
        as values:
        for all w_i:
            w_i-1: preceding word to word w_i
            w_i-2: a word before the word w_i-1
            t_i-1: a tag of w_i-1vectorizer
            t_i-2: a tag of w_i-2
            t_i-2t_i-1: tags t_i-1 and t_i-2 together
            w_i+1: a word that follows current word w_i
            w_i+2: a word that follows w_i+1
        if w_i is not rare:
            w_i: a word
        else:
            is_number: indicator \in {0, 1} whether a word is a number
            has_upper: indicator \in {0, 1} whether a word has uppercase
                character
            has_hyphen: indicator \in {0, 1} whether a word has hyphen
            prefix3: first three letters of a word w_i
            suffix3: three last letters of a word w_i
            prefix2: ...
            prefix1: ...
            suffix2: ...
            suffix1: ...

        Args:
            words: A list of sentence splitted to words.
            idx: The index within a sentence of current word.
            prev_tags: A list of tags that precede current index.

        Returns:
            A dictionary object with feature names as keys and
            raw feature values as values.
        """

        def _is_first(idx): return 0 == idx
        def _is_second(idx): return 1 == idx
        def _is_last(words, idx): return len(words) <= idx
        def _is_not_rare(word): return word in self._vocabulary

        dict_ = {}
        if _is_first(idx):
            dict_.update({
                    'w_i-1': START,
                    'w_i-2': START,
                    't_i-1': START,
                    't_i-2': START,
                    't_i-1t_i-2': START
                })
        elif _is_second(idx):
            dict_.update({
                    'w_i-1': words[idx-1],
                    'w_i-2': START,
                    't_i-1': prev_tags[idx-1],
                    't_i-2': START,
                    't_i-1t_i-2': prev_tags[idx-1] + ' ' + START
                })
        else:
            dict_.update({
                    'w_i-1': words[idx-1],
                    'w_i-2': words[idx-2],
                    't_i-1': prev_tags[idx-1],
                    't_i-2': prev_tags[idx-2],
                    't_i-1t_i-2': prev_tags[idx-1] + ' ' + prev_tags[idx-2]
                })

        dict_['w_i+2'] = (STOP
                          if _is_last(words, idx+2)
                          else words[idx+2])
        dict_['w_i+1'] = (STOP
                          if _is_last(words, idx+1)
                          else words[idx+1])
        

        if _is_not_rare(words[idx]):
            dict_['w_i'] = words[idx]
        else: 
            if words[idx] == NUMBER:
                dict_['is_number'] = 1
            else:
                dict_['has_hyphen'] = (1
                                       if '-' in words[idx]
                                       else 0)
                dict_['has_upper'] = (1
                                      if any(x.isupper() for x in words[idx])
                                      else 0)
                if len(words[idx]) >= 4:
                    dict_['prefix4'] = words[idx][:4]
                    dict_['suffix4'] = words[idx][-4:]

                if len(words[idx]) >= 3:
                    dict_['prefix3'] = words[idx][:3]
                    dict_['suffix3'] = words[idx][-3:]
                
                if len(words[idx]) >= 2:
                    dict_['prefix2'] = words[idx][:2]
                    dict_['suffix2'] = words[idx][-2:]
                
                if len(words[idx]) >= 1:
                    dict_['prefix1'] = words[idx][0]
                    dict_['suffix1'] = words[idx][-1]


        return dict_




class LogLinearModel:
    """A class that represents Log Linear Model.
    
    This class should be used for training the model and storing
    the vectorizer and weights. It uses BFGS algorithm to find optimal
    weights, but regular SGD algorithm with mini-batches could also do
    the job since the probelem is log-concave. Depending on CPU, the
    size of feature space and number of samples it may take days to
    train. For instance, to fit the model with ~45000 features and 5000
    sentences takes about 20 hours. 
    """
    def __init__(self, vectorizer, datapath):
        """Creates a new LogLinearModel object.

        Args:
            `vectorizer`: A fitted Vectorizer object.
            `datapath`: A path where all intermediate files are
                stored.

        """
        
        self.weights = np.zeros(shape=vectorizer._n_features)
        self.vectorizer = vectorizer
        self._datapath = os.path.abspath(datapath)

    def minimize(self,
                 data,
                 n_features=None,
                 datapath=None,
                 lambda_=0.005,
                 init=None,
                 verbose=False,
                 store_intermediate=False,
                 weights_fname='x.pkl'):
        """Minimizer for weights.

        Args:
            `data`: Transformed data.
            `n_features`: Number of features (default value is the
                same as in `vectorizer` that model had been
                initialized with).
            `datapath`: A directory where all files are stored
                (train data file etc.). Default is the one where
                this file is located.
            `lambda_`: A regularization parameter. Default is `0.005`.
            `init`: Initialization weights. Should be numpy array of
                same shape as `n_features`.
            `verbose`: If `True`, will print the loss function after
                each calculation. Default is `False`.
            `store_intermediate`: If `True`, will store weights after
                each loss calculation. Can be used for calculating
                inference accuracy during training. Default is `False`.
            `weights_fname`: A name of weights file. Ignored if
                `store_intermediate` is `False`.
        
        Raises:
            `ValueError`: If `n_features` is not equal to `init`s shape.
        """

        if n_features is None:
            n_features = self.vectorizer._n_features

        if datapath is None:
            datapath = self._datapath

        if init is not None and init.shape[0] != n_features:
            raise ValueError('`init` array must be of shape ',
                             n_features,
                             'but it is',
                             init.shape)

        res = scipy_minimize(data,
                             n_features=n_features,
                             datapath=datapath,
                             lambda_=lambda_,
                             init=init,
                             verbose=verbose,
                             store_intermediate=store_intermediate,
                             weights_fname=weights_fname)
        self.weights = res.x
        if verbose: print('Done')


def scipy_minimize(data, n_features, datapath, lambda_=0.005, init=None,
                   verbose=False, store_intermediate=True,
                   weights_fname='x.pkl'):
    """Wrapper for `scipy.optimize.minimize(method='L-BFGS-B')`"""

    init = (init if init is not None else np.zeros(n_features))
    args = (data,
            n_features,
            lambda_,
            datapath,
            verbose,
            store_intermediate,
            weights_fname)
    return optimize.minimize(fun=objective,
                             x0=init,
                             method='L-BFGS-B',
                             jac=jacobian,
                             args=args,
                             options={'disp':verbose})

def objective(x, *args):
    """Calculates log linear loss."""

    def calculate_loss(data, x, lambda_):
        linear_list = []
        log_list = []

        for feat_dict in data:

            linear_list.append(dot_prod(x, feat_dict['linear_term']))

            inner_sum = [dot_prod(x, indices)
                         for indices in feat_dict['log_inner_sum']]

            log_list.append(logsumexp(inner_sum))

        return (np.sum(log_list)
                - np.sum(linear_list)
                + (lambda_/2)*np.sum(np.square(x)))

    data, n_features, lambda_, datapath, verbose, store_intermediate, weights_fname = args
    transformed = data
    losses = []

    res = calculate_loss(transformed, x, lambda_)

    if verbose: print(res)

    if store_intermediate:
        fname = os.path.join(datapath, weights_fname)
        with open(fname, 'wb') as fo:
            pickle.dump(x, fo, protocol=pickle.HIGHEST_PROTOCOL)
    return res

def jacobian(x, *args):
    """Calculates jacobian of log-linear model."""

    data, n_features, lambda_, datapath, verbose, store_intermediate, weights_fname = args
    empirical_counts = np.zeros(shape=n_features)
    expected_counts = np.zeros(shape=n_features)


    transformed = data

    for feat_dict in transformed:
        empirical_counts[feat_dict['linear_term']] += 1.0
        numer_list = [np.exp(dot_prod(x, indices))
                      for indices in feat_dict['log_inner_sum']]
        
        denom = np.sum(numer_list)
        numer_list = [n/denom for n in numer_list]

        for i, indices in enumerate(feat_dict['log_inner_sum']):
            expected_counts[indices] += numer_list[i]

    return (expected_counts
            - empirical_counts
            + lambda_*x)

def dot_prod(x, indices):
    """Calculates dot product of `x` with vector of indicators
    `indices` represented by a list of indices."""

    if not indices:
        return 0.0
    else:
        return np.sum([x[i] for i in indices])