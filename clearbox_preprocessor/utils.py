import polars as pl
import numpy as np
from scipy.stats import norm


def calculate_quantile_mappings(data, n_quantiles=1000):
    """
    Calculates quantile mappings for numerical columns in a DataFrame.

    Parameters:
        data (pl.DataFrame): Input data to compute mappings.
        n_quantiles (int): Number of quantile levels.

    Returns:
        dict: A dictionary mapping column names to (values, quantiles).
    """
    quantile_maps = {}
    for col in data.columns:
        if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            values = data[col].to_numpy()
            sorted_values = np.sort(values)
            quantiles = np.linspace(0, 1, len(sorted_values))
            quantile_maps[col] = (sorted_values, quantiles)
    return quantile_maps


def transform_with_quantiles(new_data, quantile_maps, output_distribution="uniform"):
    """
    Transforms a new DataFrame using precomputed quantile mappings into the target distribution.

    Parameters:
        new_data (pl.DataFrame): Data to be transformed.
        quantile_maps (dict): Precomputed quantile mappings.
        output_distribution (str): Target distribution ("uniform" or "normal").

    Returns:
        pl.DataFrame: Transformed DataFrame.
    """
    transformed_data = {}
    for col in new_data.columns:
        if col in quantile_maps:
            train_values, train_quantiles = quantile_maps[col]
            new_values = new_data[col].to_numpy()
            
            quantiles = np.interp(new_values, train_values, train_quantiles)
            
            if output_distribution == "uniform":
                transformed_values = quantiles
            elif output_distribution == "normal":
                transformed_values = norm.ppf(quantiles)
                transformed_values = np.clip(transformed_values, -5, 5)
            else:
                raise ValueError(f"Unsupported output distribution: {output_distribution}")
            
            transformed_data[col] = transformed_values
        else:
            transformed_data[col] = new_data[col].to_numpy()
    
    return pl.DataFrame(transformed_data)