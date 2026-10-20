from __future__ import annotations

from typing import Dict, List, Sequence

import polars as pl

# Label given to missing values (nulls and blank strings), so they get their own one-hot column
NULL_LABEL  = "None"
# Label rare values are aggregated into
OTHER_LABEL = "other"
# Prefix of the temporary columns used by inverse_transform
_MAX_PREFIX = "__clearbox_preprocessor_max__"


def clean_labels(col: str) -> pl.Expr:
    """Cast a column to string and replace nulls and blank strings with ``NULL_LABEL``."""
    x = pl.col(col).cast(pl.String)
    return (
        pl.when(x.is_null() | (x.str.strip_chars() == ""))
        .then(pl.lit(NULL_LABEL))
        .otherwise(x)
        .alias(col)
    )


class CategoricalTransformer:
    """
    One-hot encode categorical columns with a vocabulary fixed at fit time.

    Every call to ``transform`` returns the same columns in the same order, whatever labels
    the data contains. Rare labels seen at fit time are encoded as ``"other"``.

    Parameters
    ----------
    categorical_features : Sequence[str]
        Columns to encode.
    vocabularies : Dict[str, List[str]]
        Labels of each column, in one-hot column order.
    rare_labels : Dict[str, List[str]]
        Labels of each column that are encoded as ``"other"``.
    unseen_labels : str
        "ignore" encodes labels not seen at fit time as all zeros; "error" raises a ValueError.
    """
    def __init__(
        self,
        categorical_features: Sequence[str],
        vocabularies: Dict[str, List[str]],
        rare_labels: Dict[str, List[str]],
        unseen_labels: str = "ignore",
    ):
        self.categorical_features = tuple(categorical_features)
        self.vocabularies  = {col: list(vocabularies[col]) for col in self.categorical_features}
        self.unseen_labels = unseen_labels
        self.original_encoded_columns = {
            col: [f"{col}_{label}" for label in self.vocabularies[col]]
            for col in self.categorical_features
        }
        # Label -> one-hot position, including rare labels mapped to the position of "other"
        self.codes = {}
        for col in self.categorical_features:
            codes = {label: i for i, label in enumerate(self.vocabularies[col])}
            for label in rare_labels.get(col, []):
                codes[label] = codes[OTHER_LABEL]
            self.codes[col] = codes

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Replace each categorical column with its one-hot (UInt8) columns, appended at the end.

        Raises
        ------
        ValueError
            If ``unseen_labels="error"`` and the data contains labels not seen at fit time.
        """
        features = self.categorical_features
        codes = data.select([
            clean_labels(col).replace_strict(self.codes[col], default=None, return_dtype=pl.UInt32)
            for col in features
        ])
        if self.unseen_labels == "error":
            unseen = {}
            for col in features:
                if codes[col].null_count() > 0:
                    values = data.select(clean_labels(col)).to_series().filter(codes[col].is_null())
                    unseen[col] = values.unique().head(5).to_list()
            if unseen:
                raise ValueError(f"The data contains labels not seen when the Preprocessor was initialized: {unseen}")

        one_hot = codes.select([
            (pl.col(col) == i).fill_null(False).cast(pl.UInt8).alias(name)
            for col in features
            for i, name in enumerate(self.original_encoded_columns[col])
        ])
        return pl.concat([data.drop(features), one_hot], how="horizontal")

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Rebuild each categorical column from its one-hot columns, which are dropped.

        The label of each row is the one with the highest score, so the one-hot columns can
        hold probabilities or other soft scores (e.g. the output of a generative model).
        ``"None"`` is decoded to null.
        """
        columns  = set(data.columns)
        features = []
        for col in self.categorical_features:
            names = self.original_encoded_columns[col]
            missing = [name for name in names if name not in columns]
            if len(missing) == len(names):
                continue
            if missing:
                raise ValueError(f"One-hot columns of '{col}' missing from the data: {missing}")
            features.append(col)
        if not features:
            return data

        schema = data.schema

        def score(name: str) -> pl.Expr:
            dtype = schema[name]
            if dtype.is_integer() or dtype == pl.Boolean:
                return pl.col(name)
            return pl.col(name).cast(pl.Float64).fill_nan(None)

        # The row maximum of each block is materialized first: inside a single expression it
        # would be recomputed for every one-hot column
        maxima = {col: f"{_MAX_PREFIX}{col}" for col in features}
        data = data.with_columns([
            pl.max_horizontal([score(name) for name in self.original_encoded_columns[col]]).alias(maxima[col])
            for col in features
        ])
        decoded = []
        for col in features:
            # Position of the first column holding the row maximum (null if every score is null)
            argmax = pl.coalesce([
                pl.when(score(name) == pl.col(maxima[col])).then(pl.lit(i, dtype=pl.UInt32))
                for i, name in enumerate(self.original_encoded_columns[col])
            ])
            labels = pl.Series([None if label == NULL_LABEL else label for label in self.vocabularies[col]], dtype=pl.String)
            decoded.append(pl.lit(labels).gather(argmax).alias(col))
        encoded = [name for col in features for name in self.original_encoded_columns[col]]
        encoded += list(maxima.values())
        return data.with_columns(decoded).drop(encoded)
