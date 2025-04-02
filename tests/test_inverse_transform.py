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
    
    # Initialize preprocessor with custom settings to avoid any discarding
    prepro = Preprocessor(df, cat_labels_threshold=0.1)
    
    # Verify initial state
    assert "cat1" in prepro.get_categorical_features()
    assert "cat2" in prepro.get_categorical_features()
    
    # Transform data
    transformed = prepro.transform(df)
    
    # Verify transformed data is not empty
    assert len(transformed) == len(df)
    
    # Check for one-hot encoded columns
    cat1_columns = [col for col in transformed.columns if col.startswith("cat1_")]
    cat2_columns = [col for col in transformed.columns if col.startswith("cat2_")]
    assert len(cat1_columns) > 0
    assert len(cat2_columns) > 0
    
    # Inverse transform
    inverse_transformed = prepro.inverse_transform(transformed)
    
    # Verify inverse has same shape
    assert len(inverse_transformed) == len(df)
    
    # Check columns are restored
    assert set(inverse_transformed.columns) == set(df.columns)
    
    # Compare values with specific debug info
    for col in df.columns:
        if col in inverse_transformed.columns:
            mismatches = (df[col] != inverse_transformed[col]).sum()
            assert mismatches == 0, f"Column {col} has {mismatches} mismatched values"

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
        
        # Verify transformed data is not empty
        assert len(transformed) == len(df)
        
        # Inverse transform
        inverse_transformed = prepro.inverse_transform(transformed)
        
        # Verify inverse has same shape
        assert len(inverse_transformed) == len(df)
        
        # Check columns are restored
        assert set(inverse_transformed.columns) == set(df.columns)
        
        # Compare values with tolerance for floating point
        for col in df.columns:
            # For numerical features with scaling, use np.isclose
            if col in inverse_transformed.columns:
                diff = np.abs(df[col].values - inverse_transformed[col].values)
                max_diff = np.max(diff)
                assert max_diff < 1e-3, f"Column {col} has max difference of {max_diff}"

def test_inverse_transform_with_categorical():
    # Create a dataset with categorical features
    df = pd.DataFrame({
        'cat1': ['A', 'B', 'A', 'C', 'B'],
        'cat2': ['X', 'Y', 'X', 'Z', 'Y']
    })
    
    # Initialize preprocessor with custom settings to avoid any discarding
    prepro = Preprocessor(df, cat_labels_threshold=0.1)
    
    # Verify initial state
    assert "cat1" in prepro.get_categorical_features()
    assert "cat2" in prepro.get_categorical_features()
    
    # Transform data
    transformed = prepro.transform(df)
    
    # Verify transformed data is not empty
    assert len(transformed) == len(df)
    
    # Check for one-hot encoded columns
    cat1_columns = [col for col in transformed.columns if col.startswith("cat1_")]
    cat2_columns = [col for col in transformed.columns if col.startswith("cat2_")]
    assert len(cat1_columns) > 0
    assert len(cat2_columns) > 0
    
    # Inverse transform
    inverse_transformed = prepro.inverse_transform(transformed)
    
    # Verify inverse has same shape
    assert len(inverse_transformed) == len(df)
    
    # Check columns are restored
    assert set(inverse_transformed.columns) == set(df.columns)
    
    # Compare values with specific debug info
    for col in df.columns:
        if col in inverse_transformed.columns:
            mismatches = (df[col] != inverse_transformed[col]).sum()
            assert mismatches == 0, f"Column {col} has {mismatches} mismatched values"

def test_inverse_transform_with_missing_values():
    # Create a dataset with missing values
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4, 5],
        'cat1': ['A', 'B', None, 'C', 'B']
    })
    
    # Use a lenient fill strategy
    prepro = Preprocessor(df, num_fill_null="mean", cat_labels_threshold=0.1)
    
    # Verify initial state
    assert "num1" in prepro.get_numerical_features()
    assert "cat1" in prepro.get_categorical_features()
    
    # Transform data
    transformed = prepro.transform(df)
    
    # Verify transformed data is not empty
    assert len(transformed) == len(df)
    
    # Inverse transform
    inverse_transformed = prepro.inverse_transform(transformed)
    
    # Verify inverse has same shape
    assert len(inverse_transformed) == len(df)
    
    # Check columns are restored
    assert set(inverse_transformed.columns) == set(df.columns)
    
    # For missing values, we compare non-NaN values only
    for col in df.columns:
        if col in inverse_transformed.columns:
            # Get non-NaN indices in original data
            valid_idx = df[col].notna()
            
            # Compare only those indices
            if valid_idx.any():
                orig_vals = df.loc[valid_idx, col].fillna("")
                inv_vals = inverse_transformed.loc[valid_idx, col].fillna("")
                
                mismatches = (orig_vals != inv_vals).sum()
                assert mismatches == 0, f"Column {col} has {mismatches} mismatched non-NaN values"
