from sklearn.feature_extraction.text import CountVectorizer


class CountVectorizerColumnN(object):
    """Just like CountVectorizer plus a colnum to grab from X
    This makes things convenient for using Feature Union with pipeline and sampling
    """
    def __init__(self, *args, colnum=None, **kwargs):
        self.countvec = CountVectorizer(*args, **kwargs)
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