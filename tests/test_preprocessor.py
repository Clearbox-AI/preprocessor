import pytest
import pandas as pd
import polars as pl
from clearbox_preprocessor import Preprocessor
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
    preprocessor = Preprocessor(sample_pandas_df, missing_values_threshold=0.9, time_id="time")
    assert isinstance(preprocessor, Preprocessor)
    assert preprocessor.time_id == "time"


def test_preprocessor_initialization_polars(sample_polars_df):
    # Test initialization with a Polars DataFrame
    preprocessor = Preprocessor(sample_polars_df, missing_values_threshold=0.9, time_id="time")
    assert isinstance(preprocessor, Preprocessor)
    assert preprocessor.time_id == "time"


def test_preprocessor_discarding_threshold_error(sample_pandas_df):
    # Test that invalid discarding_threshold raises ValueError
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, missing_values_threshold=1.1)
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, missing_values_threshold=-0.1)


def test_transform_no_bins(sample_pandas_df):
    # Test the transform method without binning (scaling=normalize by default)
    preprocessor = Preprocessor(sample_pandas_df, missing_values_threshold=0.9, time_id="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert isinstance(transformed, (pl.DataFrame, pd.DataFrame))
    assert isinstance(transformed, pd.DataFrame)
    for col in preprocessor.discarded_features:
        assert col not in transformed.columns


def test_transform_with_bins(sample_pandas_df):
    # Test transform with discretization (n_bins > 0)
    preprocessor = Preprocessor(sample_pandas_df, n_bins=3, missing_values_threshold=0.9, time_id="time", scaling="kbins")
    transformed = preprocessor.transform(sample_pandas_df)
    assert isinstance(transformed, pd.DataFrame)
    # Check if numeric features have been binned into categories
    assert "numeric_feature" in transformed.columns
    unique_values = transformed["numeric_feature"].unique()
    # There should be at most n_bins unique values after binning
    assert len(unique_values) <= 3


def test_transform_quantile_scaling(sample_pandas_df):
    # Test transform with quantile scaling
    preprocessor = Preprocessor(sample_pandas_df, scaling="quantile", time_id="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert "numeric_feature" in transformed.columns


def test_transform_unknown_scaling_method(sample_pandas_df):
    # Test that unknown scaling method raises ValueError
    with pytest.raises(ValueError):
        Preprocessor(sample_pandas_df, scaling="unknown_method")


def test_transform_rare_labels(sample_pandas_df):
    # Setup a scenario where certain categorical values are rare
    preprocessor = Preprocessor(sample_pandas_df, cat_labels_threshold=0.2, time_id="time")
    transformed = preprocessor.transform(sample_pandas_df)
    assert "categorical_feature_other" in transformed.columns or (transformed["categorical_feature"].str.contains("other").any())


def test_get_numerical_features(sample_pandas_df):
    preprocessor = Preprocessor(sample_pandas_df, time_id="time")
    numerical = preprocessor.get_numerical_features()
    assert "numeric_feature" in numerical


def test_get_categorical_features(sample_pandas_df):
    preprocessor = Preprocessor(sample_pandas_df, time_id="time")
    categorical = preprocessor.get_categorical_features()
    assert "categorical_feature" in categorical


# def test_data_type_mismatch_polars(sample_pandas_df, sample_polars_df):
#     preprocessor = Preprocessor(sample_pandas_df)
#     with pytest.raises(SystemExit):
#         preprocessor.transform(sample_polars_df)


# def test_data_type_mismatch_pandas(sample_pandas_df, sample_polars_df):
#     preprocessor = Preprocessor(sample_polars_df)
#     with pytest.raises(SystemExit):
#         preprocessor.transform(sample_pandas_df)


@pytest.mark.skipif("tsfresh" not in [pkg.key for pkg in pytest.importorskip("pkg_resources").working_set],
                    reason="tsfresh not installed")
def test_extract_ts_features(sample_pandas_df):
    # Test tsfresh feature extraction
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create time series data in the format expected by tsfresh
    # tsfresh expects:
    # 1. A column for entity/id
    # 2. A column for time (must be sortable)
    # 3. One or more value columns 
    # 4. Target variable indexed by entity/id
    time_series = pd.DataFrame({
        'id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'time': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],  # Numeric time index
        'value1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Simple increasing value for id 1 and id 2
        'value2': np.sin(np.linspace(0, 10, 10))  # Sine wave pattern
    })
    
    # Create target variable with matching indices (must be a Series indexed by id values)
    y = pd.Series({1: 0, 2: 1}, name='target')
    
    # Initialize preprocessor
    preprocessor = Preprocessor(time_series, time_id="time")
    
    # Extract features
    features = preprocessor.extract_ts_features(time_series, y, time="time", column_id="id")
    
    # Verify extraction worked
    assert isinstance(features, pd.DataFrame)
    assert features.shape[1] > 0  # Should have extracted some features
    
    # Optional: Check feature values make sense
    if 'value1__mean' in features.columns:
        # id 1 should have values 0-4 with mean 2
        # id 2 should have values 5-9 with mean 7
        # Allow some floating-point precision issues
        assert abs(features.loc[1, 'value1__mean'] - 2.0) < 1e-5
        assert abs(features.loc[2, 'value1__mean'] - 7.0) < 1e-5


def test_categorical_transformer(sample_pandas_df):
    # Test categorical feature transformation
    preprocessor = Preprocessor(sample_pandas_df, cat_labels_threshold=0.2, time_id="time")
    
    # First check that categorical features are recognized correctly
    assert "categorical_feature" in preprocessor.get_categorical_features()
    
    # Check that the categorical transformer was initialized
    assert hasattr(preprocessor, "categorical_transformer")
    
    # Check categorical original_encoded_columns
    assert "categorical_feature" in preprocessor.categorical_transformer.original_encoded_columns
    
    # Transform the data
    transformed = preprocessor.transform(sample_pandas_df)
    
    # Verify transformed data is not empty
    assert len(transformed) > 0
    assert transformed.shape[0] == sample_pandas_df.shape[0]
    
    # Get one-hot encoded columns
    categorical_columns = [col for col in transformed.columns if col.startswith("categorical_feature_")]
    
    # Make sure we have one-hot encoded columns
    assert len(categorical_columns) > 0
    
    # Check that "C" (most frequent category) is preserved
    assert "categorical_feature_C" in categorical_columns
    
    # Check that rare categories are grouped to "other"
    # D is rare with only 1 occurrence out of 10 (10%) which is below the 20% threshold
    assert "categorical_feature_other" in categorical_columns
    
    # Check the sum of one-hot encoded columns
    one_hot_sum = transformed[categorical_columns].sum(axis=1)
    assert one_hot_sum.sum() > 0  # Make sure there's at least some one-hot encoding
    assert len(one_hot_sum) == len(sample_pandas_df)  # Check that row count matches
