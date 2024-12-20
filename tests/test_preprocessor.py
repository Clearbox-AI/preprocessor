import pytest
import pandas as pd
import polars as pl
from preprocessor import Preprocessor
import numpy as np


@pytest.fixture
def sample_pandas_df():
    # Create a simple DataFrame with numeric, categorical, temporal and boolean features
    np.random.seed(0)
    data = {
        "time": pd.date_range("2021-01-01", periods=10, freq="D"),
        "numeric_feature": np.random.randn(10),
        "categorical_feature": ["A", "A", "B", "B", "C", "C", "C", "C", "D", ""] ,
        "bool_feature": [True, False, True, False, True, False, True, False, True, False]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_polars_df(sample_pandas_df):
    return pl.from_pandas(sample_pandas_df)


def test_preprocessor_initialization_pandas(sample_pandas_df):
    # Test initialization with a pandas DataFrame
    preprocessor = Preprocessor(sample_pandas_df, discarding_threshold=0.9, get_discarded_info=False, time="time")
    assert isinstance(preprocessor, Preprocessor)
    assert preprocessor.time == "time"


def test_preprocessor_initialization_polars(sample_polars_df):
    # Test initialization with a Polars DataFrame
    preprocessor = Preprocessor(sample_polars_df, discarding_threshold=0.9, get_discarded_info=False, time="time")
    assert isinstance(preprocessor, Preprocessor)
    assert preprocessor.time == "time"


def test_preprocessor_discarding_threshold_error(sample_pandas_df):
    # Test that invalid discarding_threshold raises ValueError
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, discarding_threshold=1.1)
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, discarding_threshold=-0.1)


def test_transform_no_bins(sample_pandas_df):
    # Test the transform method without binning (scaling=normalize by default)
    preprocessor = Preprocessor(sample_pandas_df, discarding_threshold=0.9, time="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert isinstance(transformed, (pl.DataFrame, pd.DataFrame))
    assert isinstance(transformed, pd.DataFrame)
    for col in preprocessor.discarded_features:
        assert col not in transformed.columns


def test_transform_with_bins(sample_pandas_df):
    # Test transform with discretization (n_bins > 0)
    preprocessor = Preprocessor(sample_pandas_df, n_bins=3, discarding_threshold=0.9, time="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert isinstance(transformed, pd.DataFrame)
    numeric_bins = [c for c in transformed.columns if c.startswith("numeric_feature_")]
    assert len(numeric_bins) > 0


def test_transform_quantile_scaling(sample_pandas_df):
    # Test transform with quantile scaling
    preprocessor = Preprocessor(sample_pandas_df, scaling="quantile", time="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert "numeric_feature" in transformed.columns


def test_transform_unknown_scaling_method(sample_pandas_df):
    # Test that unknown scaling method raises ValueError
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, scaling="unknown_method")


def test_transform_rare_labels(sample_pandas_df):
    # Setup a scenario where certain categorical values are rare
    preprocessor = Preprocessor(sample_pandas_df, cat_labels_threshold=0.2, time="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert "categorical_feature_other" in transformed.columns or (transformed["categorical_feature"].str.contains("other").any())


def test_get_numerical_features(sample_pandas_df):
    preprocessor = Preprocessor(sample_pandas_df, time="time")
    numerical = preprocessor.get_numerical_features()
    assert "numeric_feature" in numerical


def test_get_categorical_features(sample_pandas_df):
    preprocessor = Preprocessor(sample_pandas_df, time="time")
    categorical = preprocessor.get_categorical_features()
    assert "categorical_feature" in categorical


@pytest.mark.parametrize("data_type", ["polars_df", "pandas_df"])
def test_data_type_mismatch(sample_pandas_df, sample_polars_df, data_type):
    if data_type == "polars_df":
        preprocessor = Preprocessor(sample_pandas_df)
        with pytest.raises(SystemExit):
            preprocessor.transform(sample_polars_df)
    else:
        preprocessor = Preprocessor(sample_polars_df)
        with pytest.raises(SystemExit):
            preprocessor.transform(sample_pandas_df)


@pytest.mark.skipif("tsfresh" not in [pkg.key for pkg in pytest.importorskip("pkg_resources").working_set],
                    reason="tsfresh not installed")
def test_extract_ts_features(sample_pandas_df):
    # Test tsfresh feature extraction
    # Requires a target label and time column
    # Create a simple y
    y = pd.Series([0,1,0,1,0,1,0,1,0,1], name="target")
    preprocessor = Preprocessor(sample_pandas_df, time="time")
    features = preprocessor.extract_ts_features(sample_pandas_df, y, time="time")
    assert isinstance(features, pd.DataFrame)
    assert features.shape[1] > 0
