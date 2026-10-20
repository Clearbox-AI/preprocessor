from __future__ import annotations

from typing import Dict, Optional, Sequence

import polars as pl

# Formats tried, in order, when checking whether a string column holds dates or times.
# Formats without separators (bare "%Y", "%H", ...) are deliberately left out: they would
# turn numeric codes such as "10" or "2023" into dates.
# Formats without fractional seconds come first so that their values are decoded back
# without a spurious fractional part.
STRING_DATETIME_FORMATS = (
    ("%Y-%m-%d %H:%M:%S",     pl.Datetime),
    ("%Y-%m-%dT%H:%M:%S",     pl.Datetime),
    ("%Y-%m-%dT%H:%M:%SZ",    pl.Datetime),
    ("%Y-%m-%d %H:%M:%S%.f",  pl.Datetime),
    ("%Y-%m-%dT%H:%M:%S%.f",  pl.Datetime),
    ("%Y-%m-%dT%H:%M:%S%.fZ", pl.Datetime),
    ("%Y-%m-%d %H:%M",        pl.Datetime),
    ("%Y-%m-%dT%H:%M",        pl.Datetime),
    ("%Y-%m-%d",              pl.Date),
    ("%Y/%m/%d",              pl.Date),
    ("%d/%m/%Y",              pl.Date),
    ("%m/%d/%Y",              pl.Date),
    ("%H:%M:%S",              pl.Time),
    ("%H:%M:%S%.f",           pl.Time),
    ("%H:%M",                 pl.Time),
)

_SECONDS_PER_DAY = 86_400
_NS_PER_DAY      = _SECONDS_PER_DAY * 10**9


def is_temporal(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) or dtype in (pl.Date, pl.Time)


def infer_string_datetime_format(series: pl.Series, sample_size: int = 1000) -> Optional[str]:
    """
    Return the first format in ``STRING_DATETIME_FORMATS`` that parses every value of a
    sample of ``series``, or ``None`` if the column does not look like dates/times.
    """
    sample = series.drop_nulls().head(sample_size)
    if sample.len() == 0:
        return None
    for fmt, dtype in STRING_DATETIME_FORMATS:
        if sample.str.strptime(dtype, fmt, strict=False).null_count() == 0:
            return fmt
    return None


def _format_dtype(fmt: str) -> pl.DataType:
    return dict(STRING_DATETIME_FORMATS)[fmt]


def _format_resolution(fmt: str) -> Optional[int]:
    """Smallest step, in seconds, that ``fmt`` can represent (``None`` if sub-second)."""
    if "%.f" in fmt:
        return None
    if "%S" in fmt:
        return 1
    if "%M" in fmt:
        return 60
    return _SECONDS_PER_DAY


class DatetimeTransformer:
    """
    Encode date, datetime and time columns as float seconds (seconds since the Unix epoch
    for dates and datetimes, seconds since midnight for times) and decode them back.

    Columns stored as strings are parsed with the format inferred at fit time and are
    formatted back to strings with the same format on decode.

    Parameters
    ----------
    schema : pl.Schema or dict
        Original dtype of each column.
    datetime_features : Sequence[str]
        Columns to encode.
    datetime_formats : Dict[str, str]
        Parsing format of the datetime columns that are stored as strings.
    """
    def __init__(
        self,
        schema,
        datetime_features: Sequence[str],
        datetime_formats: Dict[str, str],
    ):
        self.datetime_features = tuple(datetime_features)
        self.datetime_formats  = dict(datetime_formats)
        # Temporal dtype each column is decoded to (before string formatting, if any)
        self.temporal_dtypes = {
            col: _format_dtype(self.datetime_formats[col]) if col in self.datetime_formats else schema[col]
            for col in self.datetime_features
        }

    def _encode_expr(self, col: str, dtype: pl.DataType) -> pl.Expr:
        expr = pl.col(col)
        if not is_temporal(dtype):
            fmt  = self.datetime_formats[col]
            expr = expr.cast(pl.String).str.strptime(_format_dtype(fmt), fmt, strict=False)
            dtype = _format_dtype(fmt)
        if dtype == pl.Time:
            return (expr.to_physical().cast(pl.Float64) / 1e9).alias(col)
        if dtype == pl.Date:
            expr = expr.cast(pl.Datetime("us"))
        return (expr.dt.epoch("us").cast(pl.Float64) / 1e6).alias(col)

    def _decode_expr(self, col: str) -> pl.Expr:
        seconds = pl.col(col).cast(pl.Float64)
        dtype   = self.temporal_dtypes[col]
        if col in self.datetime_formats:
            # Round to what the string format can show, so that e.g. 10:00:59.7 becomes
            # 10:01:00 instead of being truncated to 10:00:59
            resolution = _format_resolution(self.datetime_formats[col])
            if resolution is not None:
                seconds = (seconds / resolution).round() * resolution
        if dtype == pl.Time:
            expr = (seconds * 1e9).round().cast(pl.Int64, strict=False).clip(0, _NS_PER_DAY - 1).cast(pl.Time)
        elif dtype == pl.Date:
            expr = pl.from_epoch((seconds / _SECONDS_PER_DAY).round().cast(pl.Int64, strict=False), time_unit="d")
        else:
            expr = pl.from_epoch((seconds * 1e6).round().cast(pl.Int64, strict=False), time_unit="us")
            if isinstance(dtype, pl.Datetime):
                if dtype.time_zone is not None:
                    expr = expr.dt.replace_time_zone("UTC").dt.convert_time_zone(dtype.time_zone)
                expr = expr.dt.cast_time_unit(dtype.time_unit or "us")
        if col in self.datetime_formats:
            expr = expr.dt.strftime(self.datetime_formats[col])
        return expr.alias(col)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Replace the datetime columns of ``data`` with their float-seconds encoding."""
        schema = data.schema
        return data.with_columns([
            self._encode_expr(col, schema[col]) for col in self.datetime_features
        ])

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Convert float-seconds columns back to their original date/time representation."""
        columns = set(data.columns)
        return data.with_columns([
            self._decode_expr(col) for col in self.datetime_features if col in columns
        ])
