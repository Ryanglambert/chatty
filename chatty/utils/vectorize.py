import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse


class CountVectorizerColumnN(object):
    """Just like CountVectorizer plus a colnum to grab from X
    This makes things convenient for using Feature Union with pipeline and sampling
    """
    def __init__(self, *args, colnum=None, normalize=False, **kwargs):
        if not normalize:
            self.countvec = CountVectorizer(*args, **kwargs)
        else:
            self.countvec = TfidfVectorizer(*args, use_idf=False, **kwargs)
        self.colnum = colnum
        
        
    def fit(self, raw_documents, y=None):
        raw_documents = raw_documents[:, self.colnum]
        self.countvec.fit_transform(raw_documents)
        return self

    def get_feature_names(self):
        return self.countvec.get_feature_names()
        
    def transform(self, raw_documents):
        # only interested in one column
        raw_documents = raw_documents[:, self.colnum]
        X = self.countvec.transform(raw_documents)
        return X


class ColumnGetter(object):
    def __init__(self, colnum_range=(0, 10)):
        self.colnum_range = colnum_range

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        start, end = self.colnum_range
        return sparse.csr_matrix(X[:, start: end].astype('float64'))


class Cosiner(object):
    def __init__(self, colrange_pair):
        """
        colrange_pair = ((2, 302), (302, 602))
        """
        v1_bounds, v2_bounds = colrange_pair
        self.v1_start, self.v1_end = v1_bounds
        self.v2_start, self.v2_end = v2_bounds

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return ['cosine_similarity']

    def transform(self, X):
        "broadcast cosine distance (not pairwise which is slower)"
        v1_vecs = X[:, self.v1_start: self.v1_end].astype('float64')
        v2_vecs = X[:, self.v2_start: self.v2_end].astype('float64')
        v1_norms = np.linalg.norm(v1_vecs, axis=1)
        v2_norms = np.linalg.norm(v2_vecs, axis=1)
        bcast_cosines = (v1_vecs * v2_vecs).sum(axis=1) / (v1_norms * v2_norms)
        bcast_cosines = bcast_cosines.reshape(-1, 1)
        bcast_cosines = np.nan_to_num(bcast_cosines)
        return sparse.csr_matrix(bcast_cosines, dtype='float32')

