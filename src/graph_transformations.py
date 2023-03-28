
from sklearn.base import BaseEstimator, TransformerMixin


class GSPTransformer(BaseEstimator, TransformerMixin):
    def fit(self, adj_matrix, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X