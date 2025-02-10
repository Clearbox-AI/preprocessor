import polars as pl
import numpy as np
from scipy.stats import norm


def _calculate_quantile_mappings(data, n_quantiles=1000):
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

def _transform_with_quantiles(new_data, quantile_maps, output_distribution="uniform"):
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
        if col in quantile_maps.keys():
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

def _inverse_transform_with_quantiles(transformed_data, quantile_maps, input_distribution="uniform"):
    """
    Performs the inverse transformation of quantile-transformed data back to the original values.

    Parameters:
        transformed_data (pl.DataFrame): Transformed data to be inverse-transformed.
        quantile_maps (dict): Precomputed quantile mappings.
        input_distribution (str): Input distribution ("uniform" or "normal").

    Returns:
        pl.DataFrame: Inverse-transformed DataFrame.
    """
    inverse_transformed_data = {}
    for col in transformed_data.columns:
        if col in quantile_maps.keys():
            train_values, train_quantiles = quantile_maps[col]
            transformed_values = transformed_data[col].to_numpy()

            if input_distribution == "uniform":
                quantiles = transformed_values
            elif input_distribution == "normal":
                quantiles = norm.cdf(transformed_values)
            else:
                raise ValueError(f"Unsupported input distribution: {input_distribution}")

            original_values = np.interp(quantiles, train_quantiles, train_values)
            inverse_transformed_data[col] = original_values
        else:
            inverse_transformed_data[col] = transformed_data[col].to_numpy()

    return pl.DataFrame(inverse_transformed_data)

class NumericalTransformer:
    def __init__(
        self,
        data: pl.LazyFrame,
        preprocessor,
    ):
        """
        """
        scaling = preprocessor.scaling
        self.scaling = scaling
        numerical_features = preprocessor.numerical_features
        self.numerical_features = numerical_features
        n_bins = preprocessor.n_bins
        self.n_bins = n_bins
        self.n_bins_labels = None

        match scaling:
            case "none":
                pass
            case "normalize":
                # Normalization parameters initialization
                self.numerical_parameters = [data.select(numerical_features).min().collect(), 
                                                data.select(numerical_features).max().collect()]
            case "standardize":
                # Standardization parameters initialization
                self.numerical_parameters = [data.select(numerical_features).mean().collect(), 
                                                data.select(numerical_features).std().collect()] 
            case "quantile":
                # Quantile transformation parameters initialization
                self.numerical_parameters = _calculate_quantile_mappings(data.select(numerical_features).collect())      
            case "kbins":
                # K-bins discretizer parameters initialization
                if n_bins==0:
                    raise ValueError("Specify a number of bins (n_bins) greater than 0.")
                else:
                    self.n_bins_labels=list(map(str, list(range(0, n_bins)))) 
            case _:
                raise ValueError(f"Unknown scaling method: {scaling}.")

    def transform(self, data: pl.DataFrame):
        """
        """
        scaling = self.scaling
        numerical_features = self.numerical_features

        match scaling:
            case "none":
                pass
            case "normalize":
                # Normalization of numerical features
                for col in data.select(numerical_features).columns:
                    col_min = self.numerical_parameters[0][col].item()
                    col_max = self.numerical_parameters[1][col].item()
                    data = data.with_columns((pl.col(col) - col_min) / (col_max - col_min))
            case "standardize":
                # Standardization of numerical features
                for col in data.select(numerical_features).columns:
                    col_mean = self.numerical_parameters[0][col].item()
                    col_std  = self.numerical_parameters[1][col].item()
                    data = data.with_columns((pl.col(col) - col_mean) /  col_std) 
            case "quantile":
                # Quantile transformation of numerical features
                num_data = _transform_with_quantiles(data.select(numerical_features), 
                                                    self.numerical_parameters, 
                                                    output_distribution="normal")    
                for col in num_data.columns:
                    data = data.with_columns(num_data[col].alias(col))
            case "kbins":
                # KBinsDiscretizer applied to numerical features
                data = data.with_columns(numerical_features.qcut(self.n_bins, labels=self.n_bins_labels))

        return data


    def inverse_transform(self, data: pl.DataFrame):
        """
        """
        numerical_features  = self.numerical_features
        
        # Numerical features
        match self.scaling:
            case "none":
                pass
            case "normalize":
                # Inverse normalization
                for col in data.select(numerical_features).columns:
                    col_min = self.numerical_parameters[0][col].item()
                    col_max = self.numerical_parameters[1][col].item()
                    data = data.with_columns(pl.col(col) * (col_max - col_min) + col_min)
            case "standardize":
                # Inverse standardization
                for col in data.select(numerical_features).columns:
                    col_mean = self.numerical_parameters[0][col].item()
                    col_std  = self.numerical_parameters[1][col].item()
                    data = data.with_columns(pl.col(col) *  col_std + col_mean) 
            case "quantile":
                # Inverse quantile transformation
                num_data = _inverse_transform_with_quantiles(data.select(numerical_features), 
                                                            self.numerical_parameters, 
                                                            input_distribution="normal")    
                for col in num_data.columns:
                    data = data.with_columns(num_data[col].alias(col))
            
        return data



    