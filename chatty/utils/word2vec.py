from gensim.models import KeyedVectors
import numpy as np
import os

from chatty.utils import contractions, cleaning
from conf import ROOT_PATH


GOOG_NEWS_PATH = os.path.join(ROOT_PATH,
                              'research',
                              'daily_dialogue',
                              'data',
                              'GoogleNews-vectors-negative300.bin.gz')


class _DailyDialogueWordVectors(object):
    def __init__(self, strings):
        self.model = KeyedVectors.load_word2vec_format(GOOG_NEWS_PATH, binary=True)  
        self.strings = strings

    def _get_vector(self, word):
        if word in self.model.vocab:
            return self.model.get_vector(word)
        return None

    def _get_vectors(self, utter):
        words = utter.split(' ')
        return list(filter(lambda x: x is not None, map(self._get_vector, words)))

    def _avg_vectors(self, vectors):
        if not vectors:
            return np.zeros((1, 300))
        vectors = [vec.reshape(1, -1) for vec in vectors]
        concated = np.concatenate(vectors, axis=0)
        return np.mean(concated, axis=0, keepdims=True)

    def _clean(self, word):
        word = cleaning.lower(word)
        return word

    def _utter_to_vec(self, utter):
        utter = self._clean(utter)
        utter = contractions.replace_all(utter)
        utter_vec = self._get_vectors(utter)
        utter_vec = self._avg_vectors(utter_vec)
        return utter_vec

    def _corpus_to_vecs(self, utterances):
        return np.concatenate(list(map(self._utter_to_vec, utterances)), axis=0)

    def load_daily_dialogue_vectors(self):
        "takes a lot of memory so only loading model temporarily"
        return self._corpus_to_vecs(self.strings)


def vectorize(strings):
    "A one time vectorization tool to vectorize strings according to Google News word2vec"
    vectorizer = _DailyDialogueWordVectors(strings)
    return vectorizer.load_daily_dialogue_vectors()
