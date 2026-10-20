from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from .utils.categorical_transformer import NULL_LABEL, OTHER_LABEL, CategoricalTransformer, clean_labels
from .utils.datetime_transformer    import DatetimeTransformer, infer_string_datetime_format, is_temporal
from .utils.numerical_transformer   import NumericalTransformer

# Share of rows above which a single label makes a categorical column uninformative
DOMINANT_LABEL_THRESHOLD = 0.98

# Datetime columns are never discretized or quantile-transformed: those scalings fall back to min-max
DATETIME_SCALING = {
    "none":        "none",
    "normalize":   "normalize",
    "standardize": "standardize",
    "quantile":    "normalize",
    "kbins":       "normalize",
}

# Name of the label-count column in feature selection, chosen not to clash with column names
_COUNT = "__clearbox_preprocessor_count__"

_INTEGER_BOUNDS = {
    pl.Int8:   np.iinfo(np.int8),
    pl.Int16:  np.iinfo(np.int16),
    pl.Int32:  np.iinfo(np.int32),
    pl.Int64:  np.iinfo(np.int64),
    pl.UInt8:  np.iinfo(np.uint8),
    pl.UInt16: np.iinfo(np.uint16),
    pl.UInt32: np.iinfo(np.uint32),
    pl.UInt64: np.iinfo(np.uint64),
}


def _to_polars(data) -> Tuple[pl.DataFrame, bool]:
    """Return ``data`` as an eager Polars DataFrame, and whether it was a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data), True
    if isinstance(data, pl.LazyFrame):
        return data.collect(), False
    if isinstance(data, pl.DataFrame):
        return data, False
    raise TypeError(f"The datatype provided ({type(data)}) is not supported by the Preprocessor.")


class Preprocessor:
    """
    A class for preprocessing datasets based on polars, including feature selection, handling missing values, scaling,
    and time-series feature extraction.

    All the statistics used by ``transform`` (fill values, scaling parameters, bins, categories) are computed once,
    on the dataset passed to the constructor. The same preprocessor can then be applied to other datasets
    (e.g. a test set or synthetic data) consistently, and ``inverse_transform`` maps transformed data, or data
    generated in the transformed space, back to the original format.

    Parameters
    ----------
    data : pl.LazyFrame or pl.DataFrame or pd.DataFrame
        The dataset to be processed. It can be a Polars LazyFrame, Polars DataFrame, or Pandas DataFrame.

    cat_labels_threshold : float, optional, default=0.02
        A float value between 0 and 1 that sets the threshold for discarding categorical features.
        It defines a minimum frequency threshold for keeping a label as a separate category. If a label appears
        in less than :code:`cat_labels_threshold * 100%` of the total occurrences in a categorical column, it is grouped
        into a generic ``"other"`` category. Missing values are never grouped.

        For instance, if ``cat_labels_threshold=0.02`` and a label appears less than 2% in the dataset, that label will be converted to `"other"`.

    excluded_col : List, optional, default=None
        A list of column names to be excluded from processing. These columns will be returned in the
        final DataFrame without being modified.

    time_id : str, optional, default=None
        The time series index. It identifies unique time series events.

    missing_values_threshold : float, optional, default=0.999
        Columns with a share of missing values greater than this threshold are discarded.

    n_bins : int, default=0
        Number of bins to discretize numerical features into when ``scaling="kbins"``. The bins are quantile-based
        and are computed on the dataset passed to the constructor.

    scaling : str, default="none"
        The method used to scale numerical features:

        - "none"        : No scaling is applied
        - "normalize"   : Normalizes numerical features to the [0, 1] range.
        - "standardize" : Standardizes numerical features to have a mean of 0 and a standard deviation of 1.
        - "quantile"    : Transforms numerical features using quantiles information.
        - "kbins"       : Converts continuous numerical data into discrete bins (Int64 bin indices). The number of bins is defined by the parameter n_bins

        Datetime features are scaled with the same method, except for "quantile" and "kbins", for which they are normalized.

    num_fill_null : FillNullStrategy or str, default="none"
        Strategy or value used to fill null values in numerical and datetime features:

        - "none"        : Fills null values with a value just below the column minimum, which ``inverse_transform`` turns back into nulls.
        - "mean"        : Fills null values with the mean of the column.
        - "interpolate" : Fills null values using interpolation.
        - "forward"     : Fills null values using the previous non-null value.
        - "backward"    : Fills null values using the next non-null value.
        - "min"         : Fills null values with the minimum value of the column.
        - "max"         : Fills null values with the maximum value of the column.
        - "zero"        : Fills null values with zeros.
        - "one"         : Fills null values with ones.
        - value         : Fills null values with the specified value.

        Mean, minimum and maximum are those of the dataset passed to the constructor. Nulls that "interpolate",
        "forward" and "backward" cannot fill (e.g. leading nulls) are filled with the mean.

    unseen_labels : str, default="ignore"
        - "ignore"        : If new data contains labels unseen during fit one hot encoding contains 0 in every column.
        - "error"         : Raise an error if new data contains labels unseen during fit.

    ml_task : ["classification", "regression"], default=None
        The Machie Learning task the dataset is used for. It can be either classification or regression.

    target_column : str, default=None
        The target column for Machine Learning tasks. If the task is classification, the target column is encoded using LabelEncoder, while if it is regression the target column is normalized.

    Attributes
    ----------
    numerical_features : Tuple[str]
        Names of the numerical features in the dataset.

    categorical_features : Tuple[str]
        Names of the categorical features in the dataset.

    datetime_features : Tuple[str]
        Names of the date, datetime and time features in the dataset, including string columns holding dates.

    boolean_features : Tuple[str]
        Names of the boolean features in the dataset.

    discarded_features : List[str]
        Features that were discarded during preprocessing.

    discarded_reasons : Dict[str, str]
        Reason each feature was discarded.

    single_value_columns : Dict[str, Any]
        Discarded columns containing a single value, along with that value. ``inverse_transform`` restores them.

    output_columns : List[str]
        Columns returned by ``transform`` for a dataset with the same columns as the one passed to the constructor.

    feature_columns : List[str]
        The subset of ``output_columns`` produced by the transformations, i.e. without excluded and unsupported columns.

    Raises
    ------
    ValueError
        If ``cat_labels_threshold`` is not between 0 and 1.

    Notes
    -----
    The constructor transforms Pandas DataFrames into Polars DataFrames for more efficient processing.
    Features keep the order of the columns of the dataset, so the output layout does not depend on the run.
    """
    ML_TASKS = {"classification", "regression", None}
    NUM_FILL_NULL_STRATEGIES = {"none", "interpolate","forward", "backward", "min", "max", "mean", "zero", "one"}
    SCALING_STRATEGIES = {"none", "normalize", "standardize", "quantile", "kbins"}
    UNSEEN_LABELS_STRATEGIES = {"ignore", "error"}

    def __init__(
            self,
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
            cat_labels_threshold: float = 0.02,
            excluded_col: Optional[List] = None,
            time_id: str = None,
            missing_values_threshold: float = 0.999,
            n_bins: int = 0,
            scaling: Literal["none", "normalize", "standardize", "quantile", "kbins"] = "none",
            num_fill_null : Literal["none", "interpolate","forward", "backward", "min", "max", "mean", "zero", "one"] = "none",
            unseen_labels: Literal["ignore", "error"] = "ignore",
            ml_task: Literal["classification", "regression", None] = None,
            target_column: str = None,
        ):
        df, self.data_was_pd = _to_polars(data)
        excluded_col = list(excluded_col) if excluded_col is not None else []

        # Argument values check
        if cat_labels_threshold>1 or cat_labels_threshold<0:
            raise ValueError("Invalid value for cat_labels_threshold")
        if missing_values_threshold > 1 or missing_values_threshold < 0:
            raise ValueError("Invalid value for missing_values_threshold")
        if ml_task not in self.ML_TASKS:
            raise ValueError("Invalid value for ml_task")
        if target_column is not None and target_column not in df.columns:
            raise ValueError("The target column is not present in the dataset")
        for col in excluded_col:
            if col not in df.columns:
                raise ValueError(f"The excluded column {col} is not present in the dataset")
        if time_id is not None and time_id not in df.columns:
            raise ValueError(f"The time_id column {time_id} is not present in the dataset")
        if scaling not in self.SCALING_STRATEGIES:
            raise ValueError("Invalid value for scaling")
        if scaling == "kbins" and n_bins <= 0:
            raise ValueError("Specify a number of bins (n_bins) greater than 0.")
        if isinstance(num_fill_null, str):
            if num_fill_null not in self.NUM_FILL_NULL_STRATEGIES:
                raise ValueError("Invalid value for num_fill_null")
        elif not isinstance(num_fill_null, (int, float)) or isinstance(num_fill_null, bool):
            raise ValueError("Invalid value for num_fill_null")
        if unseen_labels not in self.UNSEEN_LABELS_STRATEGIES:
            raise ValueError("Invalid value for unseen_labels")
        if df.height == 0:
            raise ValueError("The dataset is empty")

        if ml_task is not None and target_column is None:
            warnings.warn('The Machine Learning task was specified but the target column was not specified.')
        if target_column is not None and ml_task is None:
            warnings.warn('The target column was specified but the Machine Learning task was not specified.')
        if target_column is not None and target_column not in excluded_col:
            excluded_col.append(target_column)

        self.schema                 = df.schema
        self.discarded_info         = []
        self.missing_threshold      = missing_values_threshold
        self.excluded_col           = excluded_col
        self.time_id                = time_id
        self.n_bins                 = n_bins
        self.num_fill_null          = num_fill_null
        self.scaling                = scaling
        self.cat_labels_threshold   = cat_labels_threshold
        self.unseen_labels          = unseen_labels
        self.target_column          = target_column
        self.ml_task                = ml_task

        self._infer_feature_types(df)
        rare_labels, vocabularies = self._feature_selection(df)

        self.datetime_transformer = DatetimeTransformer(self.schema, self.datetime_features, self.datetime_formats)
        numerical_scaling = {col: scaling for col in self.numerical_features}
        numerical_scaling.update({col: DATETIME_SCALING[scaling] for col in self.datetime_features})
        if numerical_scaling:
            self.numerical_transformer = NumericalTransformer(
                self.datetime_transformer.transform(df.select(list(numerical_scaling))),
                features      = [col for col in self.schema if col in numerical_scaling],
                scaling       = numerical_scaling,
                num_fill_null = num_fill_null,
                n_bins        = n_bins,
            )
        self.n_bins_labels = self.numerical_transformer.n_bins_labels if hasattr(self, "numerical_transformer") else None
        if self.categorical_features:
            self.categorical_transformer = CategoricalTransformer(
                self.categorical_features, vocabularies, rare_labels, unseen_labels,
            )
        self.categorical_features_sizes = [len(vocabularies[col]) for col in self.categorical_features]

        dropped = set(self.discarded_features) | set(self.categorical_features)
        encoded_columns = [
            name for col in self.categorical_features
            for name in self.categorical_transformer.original_encoded_columns[col]
        ]
        self.output_columns  = [col for col in self.schema if col not in dropped] + encoded_columns
        model_features       = set(self.datetime_features) | set(self.numerical_features) | set(self.boolean_features)
        self.feature_columns = [col for col in self.output_columns if col in model_features] + encoded_columns
        duplicated = sorted(col for col, count in Counter(self.output_columns).items() if count > 1)
        if duplicated:
            raise ValueError(f"One-hot encoded column names clash with other columns: {duplicated}. Rename the columns involved.")

        if target_column is not None:
            target = df.get_column(target_column)
            if ml_task == "classification":
                self.target_col_encoder = LabelEncoder()
                self.target_col_encoder.fit(target.to_numpy())
            elif ml_task == "regression":
                self.target_col_encoder = [df.select(pl.col(target_column).min()),
                                           df.select(pl.col(target_column).max())]

    def _infer_feature_types(
            self,
            df: pl.DataFrame,
        ) -> None:
        """
        Infer the type of each feature. The type is either numerical, categorical, datetime or boolean.
        String columns whose values all parse as dates or times with one of the supported formats are datetime features.
        Columns of other types (e.g. lists or structs) are left untouched.
        """
        boolean, datetime, numerical, categorical = [], [], [], []
        self.datetime_formats = {}
        for col, dtype in self.schema.items():
            if col in self.excluded_col:
                continue
            if dtype == pl.Boolean:
                boolean.append(col)
            elif is_temporal(dtype):
                datetime.append(col)
            elif dtype.is_numeric():
                numerical.append(col)
            elif dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
                fmt = infer_string_datetime_format(df.get_column(col).cast(pl.String))
                if fmt is not None:
                    datetime.append(col)
                    self.datetime_formats[col] = fmt
                else:
                    categorical.append(col)
        self.boolean_features     = tuple(boolean)
        self.datetime_features    = tuple(datetime)
        self.numerical_features   = tuple(numerical)
        self.categorical_features = tuple(categorical)

    def _feature_selection(
            self,
            df: pl.DataFrame,
        ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Perform feature selection to retain the most informative columns in a DataFrame while discarding redundant features.

        The selection process follows these steps:

        1. **Missing Values Filtering**:
            Columns with a share of missing values greater than ``missing_values_threshold`` are discarded.
        2. **Low-Variance Filtering**:
            Columns containing only one value are discarded. Their value is stored in ``single_value_columns``.
        3. **High Cardinality Filtering**:
            Categorical columns in which a single unique value appears in more than 98% of the records are discarded.
        4. **Rare Label Aggregation**:
            In categorical columns, labels appearing in less than a specified proportion (``cat_labels_threshold``) of instances are aggregated into a single category `"other"`.
            Columns left with a single label are discarded.

        Warnings are issued for discarded features, and the remaining features are updated accordingly.

        Parameters
        ----------
        df : pl.DataFrame
            The dataset used for feature selection.

        Returns
        -------
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]
            The rare labels and the one-hot vocabulary of each kept categorical feature.
        """
        n_rows = df.height
        typed    = set(self.boolean_features + self.datetime_features + self.numerical_features + self.categorical_features)
        features = [col for col in self.schema if col in typed]
        stats = df.select(
            [pl.col(col).null_count().alias(f"null_count:{col}") for col in features]
            + [pl.col(col).n_unique().alias(f"n_unique:{col}") for col in features]
        ).row(0, named=True)

        cleaned_categoricals = df.lazy().select([clean_labels(col) for col in self.categorical_features])
        label_counts = dict(zip(self.categorical_features, pl.collect_all([
            cleaned_categoricals.group_by(col).agg(pl.len().alias(_COUNT)).sort([_COUNT, col], descending=[True, False])
            for col in self.categorical_features
        ])))

        reasons = {}
        self.single_value_columns = {}
        rare_labels, vocabularies = {}, {}
        for col in features:
            if stats[f"null_count:{col}"] / n_rows > self.missing_threshold:
                reasons[col] = f"{col} has more than {self.missing_threshold:.1%} missing values"
                continue
            if col not in label_counts:
                if stats[f"n_unique:{col}"] <= 1:
                    reasons[col] = f"{col} contains a single value"
                    self.single_value_columns[col] = df.get_column(col)[0]
                continue

            counts = label_counts[col]
            labels = counts.get_column(col).to_list()
            if len(labels) == 1:
                reasons[col] = f"{col} contains a single value"
                self.single_value_columns[col] = None if labels[0] == NULL_LABEL else labels[0]
                continue
            if counts.get_column(_COUNT)[0] >= n_rows * DOMINANT_LABEL_THRESHOLD:
                reasons[col] = f"{col} contains a single value in more than {DOMINANT_LABEL_THRESHOLD:.0%} of the records"
                continue

            rare = []
            if len(labels) > 2:
                rare = [
                    label for label, count in counts.iter_rows()
                    if count / n_rows < self.cat_labels_threshold and label != NULL_LABEL
                ]
            rare_set = set(rare)
            kept = [label for label in labels if label not in rare_set]
            if rare and OTHER_LABEL not in kept:
                kept.append(OTHER_LABEL)
            if len(kept) <= 1:
                reasons[col] = f"{col} contains only rare labels"
                continue
            rare_labels[col]  = [label for label in rare if label != OTHER_LABEL]
            vocabularies[col] = sorted(kept)

        self.discarded_features = list(reasons)
        self.discarded_reasons  = reasons
        self.discarded_info     = [f"\n{reason}" for reason in reasons.values()]
        self.discarded          = (list(reasons), {col: labels for col, labels in rare_labels.items() if labels})
        if reasons:
            warnings.warn("The following columns were discarded:\n" + "\n".join(f"- {reason}" for reason in reasons.values()))
        if self.discarded[1]:
            warnings.warn("Some rare labels have been aggregated into the 'other' category.\nNote: you can view the discarded labels using the 'discarded' attribute of your Preprocessor class.\nIf certain labels were unintentionally discarded, try adjusting the 'cat_labels_threshold' parameter, but keep in mind that rare labels are at risk of being re-identified in case of privacy attack.\n")

        # Update the feature lists removing the discarded columns
        self.boolean_features     = tuple(col for col in self.boolean_features     if col not in reasons)
        self.numerical_features   = tuple(col for col in self.numerical_features   if col not in reasons)
        self.categorical_features = tuple(col for col in self.categorical_features if col not in reasons)
        self.datetime_features    = tuple(col for col in self.datetime_features    if col not in reasons)
        self.datetime_formats     = {col: fmt for col, fmt in self.datetime_formats.items() if col not in reasons}
        return rare_labels, vocabularies

    def transform(
            self,
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
        ) -> pl.DataFrame | pd.DataFrame:
        """
        Transform the input dataset by processing numerical, time, and categorical columns.
        This includes filling null values, scaling or discretizing numerical features, and encoding
        categorical features.

        Parameters
        ----------
        data : pl.LazyFrame or pl.DataFrame or pd.DataFrame
            The input dataset to be transformed. It can be a Polars LazyFrame, Polars DataFrame,
            or a Pandas DataFrame. It must contain every feature kept by the Preprocessor; excluded
            and discarded columns are optional.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            The transformed dataset, returned as a Pandas DataFrame if the Preprocessor was initialized with
            a Pandas DataFrame (keeping the index of ``data`` if it is a Pandas DataFrame too), and as a Polars DataFrame otherwise.

        Raises
        ------
        TypeError
            If the input data type is not supported.
        ValueError
            If features are missing from the data, or if ``unseen_labels="error"`` and the data contains unseen labels.

        Notes
        -----
        - Rows keep their order.
        - Columns are returned in the order of the dataset the Preprocessor was initialized with:
          first the non-categorical columns, then the one-hot encoded categorical features.
        - Categorical features are one-hot encoded with the categories seen at initialization. Missing values are
          encoded in a ``<column>_None`` column and rare labels in a ``<column>_other`` column.
        - Numerical features can be normalized, standardized, quantile-transformed or discretized based on the specified parameters.
        - Datetime features are converted to seconds since the Unix epoch (seconds since midnight for times) and then filled and scaled like numerical features.
        - Boolean features are converted to 0/1 integers.

        Example:
        --------
        .. code-block:: python

            preprocessor = Preprocessor(real_data, scaling="standardize")
            transformed_data = preprocessor.transform(real_data)
        """
        index = data.index if isinstance(data, pd.DataFrame) else None
        data, _ = _to_polars(data)

        features = self.boolean_features + self.datetime_features + self.numerical_features + self.categorical_features
        columns = set(data.columns)
        missing = [col for col in features if col not in columns]
        if missing:
            raise ValueError(f"Columns missing from the data: {missing}")
        data = data.drop([col for col in self.discarded_features if col in columns])

        if hasattr(self, "numerical_transformer"):
            data = self.datetime_transformer.transform(data)
            data = self.numerical_transformer.transform(data)
        data = data.with_columns([pl.col(col).cast(pl.UInt8) for col in self.boolean_features])
        if hasattr(self, "categorical_transformer"):
            data = self.categorical_transformer.transform(data)

        # Handling the target column
        if self.target_column is not None and self.target_column in data.columns:
            if self.ml_task == "classification":
                y_encoded = self.target_col_encoder.transform(data.get_column(self.target_column).to_numpy())
                data = data.with_columns(pl.Series(self.target_column, y_encoded))
            elif self.ml_task == "regression":
                col_min = self.target_col_encoder[0][self.target_column].item()
                col_max = self.target_col_encoder[1][self.target_column].item()
                data = data.with_columns((pl.col(self.target_column) - col_min) / ((col_max - col_min) or 1))

        data = data.select(self._ordered_columns(data.columns, self.output_columns))
        return self._to_output(data, index)

    def inverse_transform(
            self,
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Reverse the transformations applied during the ``preprocessor.transform(data)`` phase.

        This method performs the inverse transformations on numerical, datetime, boolean and categorical
        features to restore the original dataset format. The data does not need to be an exact output of
        ``transform``: it can come from a generative model trained on transformed data.

        Parameters:
        ----------
        data : pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray
            The transformed dataset. A NumPy array must have the columns of ``output_columns`` or of ``feature_columns``, in that order.

        Returns:
        -------
        pl.DataFrame or pd.DataFrame
            A DataFrame with all transformations reversed (a Pandas DataFrame if the Preprocessor was initialized with one), including:
            - Restored numerical features (inverse normalization, standardization, or quantile transformation;
              bin midpoints for discretized features). Integer columns are rounded.
            - Restored datetime features, in their original type or string format.
            - Boolean features, thresholded at 0.5.
            - Categorical features reconstructed from the one-hot encoding, picking the highest-scoring category of each row.
            - Single-valued discarded columns, filled with their value.
            Columns are returned in the order of the dataset the Preprocessor was initialized with.

        Raises:
        ------
        TypeError
            If the provided data type is not supported.

        Notes:
        ------
        - With ``num_fill_null="none"``, the values encoding the nulls are converted back to nulls.
        - One-hot encoded categorical columns are reconstructed into their original categorical format.

        Example:
        --------
        .. code-block:: python

            preprocessor = Preprocessor(real_data, scaling="standardize")
            transformed_data = preprocessor.transform(real_data)

            # Reverse the transformations
            original_data = preprocessor.inverse_transform(transformed_data)
        """
        index = data.index if isinstance(data, pd.DataFrame) else None
        if isinstance(data, np.ndarray):
            data = self._array_to_polars(data)
        else:
            data, _ = _to_polars(data)

        if hasattr(self, "categorical_transformer"):
            data = self.categorical_transformer.inverse_transform(data)
        if hasattr(self, "numerical_transformer"):
            data = self.numerical_transformer.inverse_transform(data)
            data = self.datetime_transformer.inverse_transform(data)
        columns = set(data.columns)
        data = data.with_columns([
            (pl.col(col).cast(pl.Float64) >= 0.5).alias(col)
            for col in self.boolean_features if col in columns
        ])

        if self.target_column is not None and self.target_column in data.columns:
            if self.ml_task == "classification":
                labels = np.clip(np.rint(data.get_column(self.target_column).to_numpy()), 0, len(self.target_col_encoder.classes_) - 1)
                data = data.with_columns(pl.Series(self.target_column, self.target_col_encoder.inverse_transform(labels.astype(np.int64))))
            elif self.ml_task == "regression":
                col_min = self.target_col_encoder[0][self.target_column].item()
                col_max = self.target_col_encoder[1][self.target_column].item()
                data = data.with_columns(pl.col(self.target_column) * ((col_max - col_min) or 1) + col_min)

        data = data.with_columns([
            pl.lit(value, dtype=self.schema[col]).alias(col)
            for col, value in self.single_value_columns.items() if col not in columns
        ])

        # Make sure the column data types match the ones of the original dataframe
        data = data.with_columns([
            self._cast_expr(col, self.schema[col])
            for col, dtype in data.schema.items() if col in self.schema and dtype != self.schema[col]
        ])
        data = data.select(self._ordered_columns(data.columns, list(self.schema)))
        return self._to_output(data, index)

    @staticmethod
    def _cast_expr(col: str, dtype: pl.DataType) -> pl.Expr:
        expr = pl.col(col)
        if dtype in _INTEGER_BOUNDS:
            bounds = _INTEGER_BOUNDS[dtype]
            expr = expr.cast(pl.Float64).round().clip(float(bounds.min), float(bounds.max))
        return expr.cast(dtype, strict=False).alias(col)

    @staticmethod
    def _ordered_columns(columns: List[str], reference: List[str]) -> List[str]:
        """``columns`` sorted as in ``reference``, followed by the columns not in ``reference``."""
        present = set(columns)
        known   = set(reference)
        return [col for col in reference if col in present] + [col for col in columns if col not in known]

    def _array_to_polars(self, array: np.ndarray) -> pl.DataFrame:
        if array.ndim != 2:
            raise ValueError("The array must be 2-dimensional")
        for columns in (self.output_columns, self.feature_columns):
            if array.shape[1] == len(columns):
                return pl.DataFrame(array, schema=columns, orient="row")
        raise ValueError(
            f"The array has {array.shape[1]} columns, expected {len(self.feature_columns)} "
            f"(feature_columns) or {len(self.output_columns)} (output_columns)"
        )

    def _to_output(self, data: pl.DataFrame, index: Optional[pd.Index]) -> pl.DataFrame | pd.DataFrame:
        if not self.data_was_pd:
            return data
        data = data.to_pandas()
        if index is not None and len(index) == len(data):
            data.index = index
        return data

    def extract_ts_features(
            self,
            data:      pl.LazyFrame | pd.DataFrame,
            y:         pl.Series | pd.Series = None,
            time:      str = None,
            column_id: str = None,
        ) -> pd.DataFrame:
        """
        Extract relevant time-series features from the provided data.

        Parameters
        ----------
        data : pl.LazyFrame or pd.DataFrame
            The input dataset containing the time-series data. It can be a Polars LazyFrame
            or a Pandas DataFrame.
        y : pl.Series or pd.Series
            The label series associated with the data. It can be a Polars Series or a Pandas Series.
        time : str, optional
            The name of the time column used to sort the data. If not provided, the method
            will try to use ``self.time_id`` if available.
        column_id : str, optional
            The name of the ID column, if present in the data. This is used to distinguish
            different time-series within the same dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the extracted and filtered relevant time-series features.

        Raises
        ------
        TypeError
            If the provided data is not a Polars LazyFrame or a Pandas or Polars DataFrame.
        TypeError
            If the provided label series is not a Polars Series or a Pandas Series.
        ValueError
            If the time column name is not provided and ``self.time_id`` is not available.

        Notes
        -----
        - The function uses the ``extract_relevant_features`` method from the ``tsfresh`` library
        to extract features from the time-series data.
        - The method stores the filtered features in ``self.features_filtered`` for further use.
        """
        # tsfresh is slow to import, so it is only imported when needed
        from tsfresh import extract_features, extract_relevant_features

        # Check if time column is provided, else fallback to self.time_id if available
        if time is None:
            if self.time_id is not None:
                time = self.time_id
            else:
                raise ValueError("Time column name is required for time-series feature extraction.")

        # Transform input dataframe into Pandas.DataFrame
        if isinstance(data, pl.LazyFrame):
            data_pd = data.collect().to_pandas()
        elif isinstance(data, pl.DataFrame):
            data_pd = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            data_pd = data
        else:
            raise TypeError("The dataframe must be a Polars LazyFrame, a Polars DataFrame or a Pandas DataFrame")

        if isinstance(y, pl.Series):
            y = y.to_pandas()
        elif not isinstance(y, pd.Series):
            raise TypeError("The labels series must be a Polars Series or a Pandas Series")

        # First try to extract relevant features
        features_filtered = extract_relevant_features(data_pd, y, column_sort=time, column_id=column_id)
        self.features_filtered = features_filtered

        # If no relevant features were found, fall back to extracting all features
        if features_filtered.shape[1] == 0:
            # Extract all features without filtering for relevance
            all_features = extract_features(data_pd, column_sort=time, column_id=column_id)
            return all_features

        return features_filtered

    def get_features_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Gets the sizes of ordinal and categorical features after transformation.

        The ordinal features are the numerical, datetime and boolean features, which come first in the
        transformed data; each categorical feature is then encoded in as many columns as its size.

        Returns:
            Tuple: Sizes of ordinal and categorical features.
        """
        n_ordinal = len(self.numerical_features) + len(self.datetime_features) + len(self.boolean_features)
        numerical_sizes = [n_ordinal] if n_ordinal > 0 else []
        return numerical_sizes, list(self.categorical_features_sizes)

    def get_numerical_features(self) -> Tuple[str]:
        """
        Return the list of numerical features.
        """
        return self.numerical_features

    def get_categorical_features(self) -> Tuple[str]:
        """
        Return the list of categorical features.
        """
        return self.categorical_features
