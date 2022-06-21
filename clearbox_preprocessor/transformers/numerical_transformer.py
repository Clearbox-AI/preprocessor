import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer


class NumericalTransformer(BaseEstimator, TransformerMixin):
    imputer: SimpleImputer
    scaler: MinMaxScaler
    est: KBinsDiscretizer

    def __init__(self, n_bins: int = 0) -> None:
        """
        Initialize the transformer. The transformer is initialized with a number of bins for discretization. If the number of
        bins is zero, the transformer will use a MinMaxScaler instead of KBinsDiscretizer.
        """
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame):
        """
        Fit the transformer. Firstly, the imputer is fitted to the data. Then, the scaler or the discretizer is fitted to the
        imputed data.
        """
        data = X.copy()

        self.imputer = SimpleImputer(strategy="most_frequent")
        self.imputer.fit(data)
        data = self.imputer.transform(data)

        if self.n_bins > 0:
            self.est = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="kmeans"
            )
            self.est.fit(data)
            data = self.est.transform(data)
        else:
            self.scaler = MinMaxScaler()

            self.scaler.fit(data)
            data = self.scaler.transform(data)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data. The data is transformed by the imputer, then the scaler or the discretizer.
        """
        X = X.copy()

        preprocessed_X: np.ndarray = self.imputer.transform(X)
        if self.n_bins > 0:
            preprocessed_X = self.est.transform(X)
        else:
            preprocessed_X = self.scaler.transform(X)

        return preprocessed_X
