import pandas as pd
import numpy as np
import sklearn.compose

from typing import List

from transformers.categorical_transformer import CategoricalTransformer
from transformers.numerical_transformer import NumericalTransformer


class Preprocessor:
    numerical_features: List[str]
    categorical_features: List[str]
    discarded_columns: List[str]
    transformer: sklearn.compose.ColumnTransformer

    def __init__(
        self,
        data: pd.DataFrame,
        discarding_threshold: float = 0.9,
    ):
        """
        Initialize the preprocessor. The preprocessor is initialized with a DataFrame and a threshold for discarding features.
        The preliminary operations deal with distinguishing numerical columns from categorical columns, discarding the columns
        that do not carry significant information. A column transformer is defined below for each type of column.
        """
        X = data.copy()

        self._infer_feature_types(X)

        X = self._feature_selection(X, discarding_threshold)

        transformers_list = list()
        if len(self.numerical_features) > 0:
            transformers_list.append(
                (
                    "ordinal_transformer",
                    NumericalTransformer(),
                    self.numerical_features,
                )
            )
        if len(self.categorical_features) > 0:
            transformers_list.append(
                (
                    "categorical_transformer",
                    CategoricalTransformer(),
                    self.categorical_features,
                )
            )

        self.transformer = sklearn.compose.ColumnTransformer(
            transformers=transformers_list
        )

    def _infer_feature_types(self, data: pd.DataFrame) -> None:
        """
        Infer the type of each feature in the DataFrame. The type is either numerical or categorical. DateTime and Boolean
        features are converted to numerical by default.
        """
        boolean_features = list(data.select_dtypes(include=["bool"]).columns)
        data[boolean_features] = data[boolean_features].astype(int)

        datetime_features = list(
            data.select_dtypes(include=["datetime", "timedelta"]).columns
        )
        data[datetime_features] = data[datetime_features].astype(int)

        self.numerical_features = list(
            data.select_dtypes(include=["number", "datetime"]).columns
        )

        self.categorical_features = list(
            data.select_dtypes(include=["object", "category"]).columns
        )

    def _feature_selection(
        self,
        data: pd.DataFrame,
        discarding_threshold: float,
    ) -> pd.DataFrame:
        """
        Perform a selection of the most useful columns for a given DataFrame, ignorig the other features. The selection is
        performed in two steps:
        1. The columns with more than 50% of missing values are discarded.
        2. The columns containing only one value or, conversely, a large number of different values are discarded. In the latter
        case the default threshold is equal to 90%, ie if more than 90% of the instances have different values then the entire
        column is discarded.
        """
        cat_features_stats = [
            (
                i,
                data[i].value_counts(),
                data[i].isnull().sum(),
            )
            for i in self.get_categorical_features()
        ]

        num_features_stats = [
            (
                i,
                data[i].value_counts(),
                data[i].isnull().sum(),
            )
            for i in self.get_categorical_features()
        ]

        self.discarded_columns = []

        for column_stats in cat_features_stats:
            if column_stats[2] > 0.5 * len(data):
                self.discarded_columns.append(column_stats[0])

            if (column_stats[1].shape[0] == 1) or (
                column_stats[1].shape[0] >= (len(data) * discarding_threshold)
            ):
                self.discarded_columns.append(column_stats[0])

        for column_stats in num_features_stats:
            if column_stats[2] > 0.5 * len(data):
                self.discarded_columns.append(column_stats[0])

            if column_stats[1].shape[0] <= 1:
                self.discarded_columns.append(column_stats[0])

        data.drop(self.discarded_columns, axis=1, inplace=True)

        self.numerical_features = list(
            set(self.numerical_features) - set(self.discarded_columns)
        )
        self.categorical_features = list(
            set(self.categorical_features) - set(self.discarded_columns)
        )

        return data

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the preprocessor to the initial DataFrame.
        """
        X = X.copy()

        X.drop(self.discarded_columns, axis=1, inplace=True)

        self.transformer.fit(X)

        return

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the DataFrame using the preprocessor.
        """
        X = X.copy()

        X.drop(self.discarded_columns, axis=1, inplace=True)

        for transformer in self.transformer.transformers_:
            if "categorical_transformer" in transformer:
                categories = [cat for cat in transformer[2]]

                X[categories] = X[categories].astype(str)

        preprocessed_X: np.ndarray = self.transformer.transform(X)

        return preprocessed_X

    def get_numerical_features(self) -> List[str]:
        """
        Return the list of numerical features.
        """
        return self.numerical_features

    def get_categorical_features(self) -> List[str]:
        """
        Return the list of categorical features.
        """
        return self.categorical_features
