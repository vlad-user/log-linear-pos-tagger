import operator
import itertools

import numpy as np

from nlp_utils import START
from nlp_utils import dot_prod

class ViterbiInference:
    """Implements Viterbi Dynamic Programming algorithm."""
    def __init__(self, model):
        """Instantiates a new object.

        Args:
            `model`: A fitted model having fitted `vectorizer`
                and `weights`.
        """
        self.model = model
        self.vectorizer = model.vectorizer
        self.tags = self.vectorizer.list_available_tags()
        self.weights = model.weights
        self.sparse_weights = self.weights
        self._normalization_cache = {}
        self._proba_cache = {}

    def predict(self, words):
        """Predicts tags for `words` using Viterbi algorithm.

        Args:
            words: A list of words.

        Returns:
            A list of tags.
        """

        N = len(words)
        pred = {}
        bp = {}
        pred[-1, START, START] = 1
        bp[-1, START, START] = -1
        tags = []

        for k in range(N):
            for (u, v) in self._get_cross_prod_tags(k):
                options = [(pred[k-1, w, u] * self._proba(words, k, [w, u, v]), w)
                            for w in self._get_tags(k-2)]
                res = max(options, key=operator.itemgetter(0))
                pred[k, u, v] = res[0]
                bp[k, u, v] = res[1]
        
        options = [(pred[N-1, u, v], u, v) for (u, v) in self._get_cross_prod_tags(N-1)]
        res = max(options, key=operator.itemgetter(0))
        tags.append(res[2])
        tags.append(res[1])

        for k in reversed(list(range(N-2))):
            tags.append(bp[k+2, tags[-1], tags[-2]])

        self._normalization_cache = {}
        tags = list(reversed(tags))
        if len(words) == 1:
            return [tags[-1]]
        elif len(words) == 0:
            return []
        return tags
    
    def compute_accuracy(self, words_and_tags):
        """Computes accuracy for provided tags vs predicted tags.
        
        ## Usage:
        ```python
        import numpy as np
        import nlp_utils
        # ... train `model` here
        
        processor = nlp_utils.TextProcessor()
        with open('test.wtag') as fo:
            text = fo.read()
        test_data = []
        
        for sent in processor.sent_tokenize(text):
            words_and_tags = processor.word_pos_tokenize(sent)
            test_data.append(words_and_tags)

        accuracies = []
        for words_and_tags in test_data[:10]:
            accuracies.append(model.compute_accuracy(words_and_tags))
            print(np.mean(accuracies))
        ```

        Args:
            words_and_tags: A list of tuples (word, POS tag).

        Returns:
            Accuracy.
        """

        words = [w[0] for w in words_and_tags]
        tags = [w[1] for w in words_and_tags]
        acc = [t1==t2 for t1, t2 in zip(tags, self.predict(words))]

        return sum(acc)/len(acc)

    def _proba(self, words, k, tags):
        """Computes probability of tag given history.

        Args:
            `words`: A list of words.
            k: index of a current word in a sentence.
            tags: A list of following tags [t_k-2, t_k-1, t_k]
        
        Returns:
            Probability P(t_k|t_k-1, t_k-2, words)=P(t_k|history).
        """

        w, u, v = tags[0], tags[1], tags[2]

        if k == 0:
            prev_tags = []
        elif k == 1:
            prev_tags = [u]
        elif k == 2:
            prev_tags = [w, u]
        else:
            prev_tags = list(range(len(words)))
            for i, idx in enumerate([k-2, k-1, k]):
                prev_tags[idx] = tags[i]

        raw_hist = self.vectorizer.convert_to_raw_hist(words, k, prev_tags)
        hist = [(name, val) for name, val in raw_hist.items()]
        indices = self.vectorizer.hist_and_tag_to_binary((hist, v))
        numer = np.exp(dot_prod(self.weights, indices))
        denom = self._partition_func(hist, k)
        result = numer/denom

        return result

    def _partition_func(self, hist, k):
        """Returns normalization partition function.

        Args:
            hist: A list of tuples (<feature name>, <feature value>).
            k: An index of a current word in a sentence.

        Returns:
            Normalization partition function.
        """
        if k >= 2:
            hash_k = 2
        else:
            hash_k = k
        hash_key = tuple(sorted(hist)) + (hash_k, )
        
        if hash_key in self._normalization_cache:
            res = self._normalization_cache[hash_key]
        else:
            index_list = [self.vectorizer.hist_and_tag_to_binary((hist, t))
                          for t in self._get_tags(k)]

            res = np.sum([np.exp(dot_prod(self.weights, v)) for v in index_list])
            self._normalization_cache[hash_key] = res
        return res

    def _get_tags(self, k):
        """Returns a list of possible tags for a location `k` in a sentence."""

        return [START] if k < 0 else self.vectorizer.list_available_tags()

    def _get_cross_prod_tags(self, k):
        """Returns a list of possible cross product for pair of tags."""

        return itertools.product(self._get_tags(k-1),
                                 self._get_tags(k))
