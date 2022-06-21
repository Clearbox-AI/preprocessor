import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    imputer: SimpleImputer
    encoder: OneHotEncoder

    def fit(self, X: pd.DataFrame):
        """
        Fit the transformer. Firstly, the imputer is fitted to the data. Then, the one hot encoder is fitted to the imputed
        data.
        """
        data = X.copy().astype(str)

        self.imputer = SimpleImputer(strategy="most_frequent", add_indicator=False)
        self.imputer.fit(data)

        data = self.imputer.transform(data)

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder.fit(data)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data. The data is transformed by the imputer, then the one hot encoder.
        """
        X = X.copy().astype(str)

        preprocessed_X: np.ndarray = self.imputer.transform(X)
        preprocessed_X = self.encoder.transform(X)

        return preprocessed_X
