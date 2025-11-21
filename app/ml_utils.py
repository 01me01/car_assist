from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key].astype(str).values

class NumSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.keys].fillna(0).values
