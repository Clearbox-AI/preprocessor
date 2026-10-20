from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Sequence

import numpy as np
import polars as pl
from scipy.special import ndtr, ndtri
from sklearn.preprocessing import QuantileTransformer

# Offset, as a fraction of the column range, of the value used to encode nulls when
# ``num_fill_null="none"``. Nulls are encoded slightly below the column minimum and every
# value below half that offset is decoded back to null.
NULL_SENTINEL_OFFSET = 0.01


# Same bounds as sklearn's QuantileTransformer, so that the mapping below matches it exactly
_BOUNDS_THRESHOLD = 1e-7
_QUANTILE_CLIP_MIN = ndtri(_BOUNDS_THRESHOLD - np.spacing(1))
_QUANTILE_CLIP_MAX = ndtri(1 - (_BOUNDS_THRESHOLD - np.spacing(1)))


def _quantile_to_normal(x: np.ndarray, quantiles: np.ndarray, references: np.ndarray) -> np.ndarray:
    """
    Map one column (without NaN) to a standard normal distribution, as
    ``QuantileTransformer(output_distribution="normal").transform`` does.
    """
    lower = x - _BOUNDS_THRESHOLD < quantiles[0]
    upper = x + _BOUNDS_THRESHOLD > quantiles[-1]
    # Interpolating in both directions and averaging handles repeated quantiles
    y = 0.5 * (np.interp(x, quantiles, references) - np.interp(-x, -quantiles[::-1], -references[::-1]))
    y[upper] = 1
    y[lower] = 0
    return np.clip(ndtri(y), _QUANTILE_CLIP_MIN, _QUANTILE_CLIP_MAX)


def _normal_to_quantile(z: np.ndarray, quantiles: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Inverse of ``_quantile_to_normal``, as ``QuantileTransformer.inverse_transform`` does."""
    u = ndtr(z)
    lower = u - _BOUNDS_THRESHOLD < 0
    upper = u + _BOUNDS_THRESHOLD > 1
    x = np.interp(u, references, quantiles)
    x[upper] = quantiles[-1]
    x[lower] = quantiles[0]
    return x


def _map_columns(function: Callable[[int], np.ndarray], n_columns: int) -> List[np.ndarray]:
    """Call ``function`` on every column index, in threads (NumPy and SciPy release the GIL)."""
    workers = min(n_columns, os.cpu_count() or 1)
    if workers <= 1:
        return [function(i) for i in range(n_columns)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, range(n_columns)))


def _finite(col: str) -> pl.Expr:
    """Cast to Float64 and turn NaN and +/-inf into nulls."""
    x = pl.col(col).cast(pl.Float64)
    return pl.when(x.is_finite()).then(x).otherwise(None).alias(col)


class NumericalTransformer:
    """
    Fill nulls in and scale numerical columns (including datetime columns already encoded
    as float seconds).

    Every statistic used by ``transform`` (fill values, scaling parameters, bin edges, the
    quantile mapping) is computed once on the data passed to the constructor, so
    transforming a test set never uses statistics of the test set itself.

    Parameters
    ----------
    data : pl.DataFrame
        Data the transformer is fitted on.
    features : Sequence[str]
        Columns to transform.
    scaling : Dict[str, str]
        Scaling method of each column: "none", "normalize", "standardize", "quantile" or "kbins".
    num_fill_null : str or float
        Strategy or value used to fill nulls (see ``Preprocessor``).
    n_bins : int
        Number of bins for the columns scaled with "kbins".
    """
    def __init__(
        self,
        data: pl.DataFrame,
        features: Sequence[str],
        scaling: Dict[str, str],
        num_fill_null,
        n_bins: int = 0,
    ):
        self.numerical_features = tuple(features)
        self.scaling            = dict(scaling)
        self.num_fill_null      = num_fill_null
        self.n_bins             = n_bins
        self.n_bins_labels      = None

        data  = data.select([_finite(col) for col in self.numerical_features])
        stats = data.select([
            pl.struct(
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).null_count().alias("null_count"),
            ).alias(col)
            for col in self.numerical_features
        ]).row(0, named=True)

        self.params = {}
        for col, s in stats.items():
            col_min = s["min"] if s["min"] is not None else 0.0
            col_max = s["max"] if s["max"] is not None else col_min
            col_range = col_max - col_min
            self.params[col] = {
                "min":       col_min,
                "max":       col_max,
                "mean":      s["mean"] if s["mean"] is not None else col_min,
                # Constant columns are scaled by 1 instead of dividing by zero
                "range":     col_range if col_range > 0 else 1.0,
                "std":       s["std"] if s["std"] else 1.0,
                "has_nulls": s["null_count"] > 0,
            }
        self.fill_values = {col: self._fill_value(col) for col in self.numerical_features}

        filled = data.with_columns([self._fill_expr(col) for col in self.numerical_features])

        self.quantile_features = tuple(col for col in self.numerical_features if self.scaling[col] == "quantile")
        if self.quantile_features:
            self.scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(1, min(1000, filled.height)),
                random_state=0,
            ).fit(filled.select(self.quantile_features).to_numpy())

        self.kbins_features = tuple(col for col in self.numerical_features if self.scaling[col] == "kbins")
        if self.kbins_features:
            if n_bins <= 0:
                raise ValueError("Specify a number of bins (n_bins) greater than 0.")
            self.n_bins_labels = list(map(str, range(n_bins)))
            quantiles = np.linspace(0, 1, n_bins + 1)
            self.bin_edges     = {}
            self.bin_midpoints = {}
            for col in self.kbins_features:
                # Duplicate edges (heavily repeated values) collapse into fewer bins
                edges = np.unique(np.quantile(filled[col].to_numpy(), quantiles))
                self.bin_edges[col]     = edges[1:-1].tolist()
                self.bin_midpoints[col] = ((edges[:-1] + edges[1:]) / 2).tolist() if edges.size > 1 else edges.tolist()

    def _fill_value(self, col: str) -> float:
        strategy = self.num_fill_null
        p = self.params[col]
        if not isinstance(strategy, str):
            return float(strategy)
        if strategy == "none":
            return p["min"] - NULL_SENTINEL_OFFSET * p["range"]
        if strategy in ("min", "max"):
            return p[strategy]
        if strategy == "zero":
            return 0.0
        if strategy == "one":
            return 1.0
        # "mean", and the fallback for nulls that "forward", "backward" and "interpolate" cannot fill
        return p["mean"]

    def _fill_expr(self, col: str) -> pl.Expr:
        x = pl.col(col)
        if self.num_fill_null == "interpolate":
            x = x.interpolate().forward_fill().backward_fill()
        elif self.num_fill_null == "forward":
            x = x.forward_fill().backward_fill()
        elif self.num_fill_null == "backward":
            x = x.backward_fill().forward_fill()
        return x.fill_null(self.fill_values[col]).alias(col)

    def _scale_expr(self, col: str) -> pl.Expr:
        x = pl.col(col)
        p = self.params[col]
        scaling = self.scaling[col]
        if scaling == "normalize":
            x = (x - p["min"]) / p["range"]
        elif scaling == "standardize":
            x = (x - p["mean"]) / p["std"]
        elif scaling == "kbins":
            # Bin index = number of inner edges strictly below the value, i.e. bins are (a, b]
            x = pl.lit(pl.Series(self.bin_edges[col], dtype=pl.Float64)).search_sorted(x, side="left").cast(pl.Int64)
        return x.alias(col)

    def _unscale_expr(self, col: str) -> pl.Expr:
        x = pl.col(col).cast(pl.Float64)
        p = self.params[col]
        scaling = self.scaling[col]
        if scaling == "normalize":
            x = x * p["range"] + p["min"]
        elif scaling == "standardize":
            x = x * p["std"] + p["mean"]
        elif scaling == "kbins":
            midpoints = self.bin_midpoints[col]
            index = x.round().clip(0, len(midpoints) - 1).cast(pl.Int64)
            x = pl.lit(pl.Series(midpoints, dtype=pl.Float64)).gather(index)
        if self.num_fill_null == "none" and p["has_nulls"]:
            threshold = p["min"] - NULL_SENTINEL_OFFSET / 2 * p["range"]
            x = pl.when(x < threshold).then(None).otherwise(x)
        return x.alias(col)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Fill nulls and scale the numerical columns of ``data``.

        Parameters
        ----------
        data : pl.DataFrame
            Data containing every column in ``numerical_features``.

        Returns
        -------
        pl.DataFrame
            ``data`` with the numerical columns transformed (Float64, or Int64 bin indices
            for the "kbins" columns).
        """
        features = self.numerical_features
        data = data.with_columns([_finite(col) for col in features])
        data = data.with_columns([self._fill_expr(col) for col in features])
        data = data.with_columns([self._scale_expr(col) for col in features])
        if self.quantile_features:
            data = data.with_columns(self._quantile_map(data, _quantile_to_normal))
        return data

    def _quantile_map(self, data: pl.DataFrame, function) -> List[pl.Series]:
        """Apply ``function`` with the fitted quantiles to every quantile-scaled column of ``data``."""
        quantiles  = self.scaler.quantiles_
        references = self.scaler.references_
        def column(i: int) -> np.ndarray:
            values = data.get_column(self.quantile_features[i]).to_numpy().astype(np.float64)
            return function(values, quantiles[:, i], references)
        values = _map_columns(column, len(self.quantile_features))
        return [pl.Series(col, values[i]) for i, col in enumerate(self.quantile_features)]

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Reverse scaling and, with ``num_fill_null="none"``, restore the nulls.

        "kbins" columns are mapped back to the midpoint of their bin. Values do not need to
        be exact transform outputs (e.g. they can come from a generative model).
        """
        columns  = set(data.columns)
        features = [col for col in self.numerical_features if col in columns]
        quantile_features = [col for col in self.quantile_features if col in columns]
        if quantile_features:
            if len(quantile_features) != len(self.quantile_features):
                missing = sorted(set(self.quantile_features) - set(quantile_features))
                raise ValueError(f"Quantile-scaled columns missing from the data: {missing}")
            data = data.with_columns([_finite(col).fill_null(0.0) for col in self.quantile_features])
            data = data.with_columns(self._quantile_map(data, _normal_to_quantile))
        return data.with_columns([self._unscale_expr(col) for col in features])
