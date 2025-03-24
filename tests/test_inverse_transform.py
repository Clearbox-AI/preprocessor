import pytest
import pandas as pd
import polars as pl
import numpy as np
from clearbox_preprocessor import Preprocessor

def test_inverse_transform_basic():
    # Create a simple dataset with numerical and categorical features
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [10, 20, 30, 40, 50],
        'cat1': ['A', 'B', 'A', 'C', 'B'],
        'cat2': ['X', 'Y', 'X', 'Z', 'Y']
    })
    
    prepro = Preprocessor(df)
    transformed = prepro.transform(df)
    inverse_transformed = prepro.inverse_transform(transformed)
    
    # Check if the inverse transform restores the original data
    assert (df!=inverse_transformed[df.columns]).sum().sum()==0

def test_inverse_transform_with_scaling():
    # Create a dataset with numerical features
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [10, 20, 30, 40, 50]
    })
    
    # Test with different scaling methods
    scaling_methods = ['normalize', 'standardize', 'quantile']
    for scaling in scaling_methods:
        prepro = Preprocessor(df, scaling=scaling)
        transformed = prepro.transform(df)
        inverse_transformed = prepro.inverse_transform(transformed)
        
        # Check if the inverse transform restores the original data
        assert (np.isclose(df,inverse_transformed)==False, 1e-3)[0].sum().sum()==0

def test_inverse_transform_with_categorical():
    # Create a dataset with categorical features
    df = pd.DataFrame({
        'cat1': ['A', 'B', 'A', 'C', 'B'],
        'cat2': ['X', 'Y', 'X', 'Z', 'Y']
    })
    
    prepro = Preprocessor(df)
    transformed = prepro.transform(df)
    inverse_transformed = prepro.inverse_transform(transformed)
    
    # Check if the inverse transform restores the original data
    assert (df!=inverse_transformed[df.columns]).sum().sum()==0

def test_inverse_transform_with_missing_values():
    # Create a dataset with missing values
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4, 5],
        'cat1': ['A', 'B', None, 'C', 'B']
    })
    
    # Test with different fill strategies
    fill_strategies = ['none']
    for strategy in fill_strategies:
        prepro = Preprocessor(df, num_fill_null=strategy)
        transformed = prepro.transform(df)
        inverse_transformed = prepro.inverse_transform(transformed)
        
        # Check if the inverse transform restores the original data
        pd.testing.assert_frame_equal(df.replace(np.nan, None).replace('null', None), inverse_transformed.replace(np.nan, None).replace('null', None))
