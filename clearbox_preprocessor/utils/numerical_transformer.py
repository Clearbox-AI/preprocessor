import polars as pl
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

class NumericalTransformer:
    def __init__(
        self,
        data: pl.LazyFrame,
        preprocessor,
    ):
        scaling                 = preprocessor.scaling
        self.scaling            = scaling
        numerical_features      = preprocessor.numerical_features
        self.numerical_features = numerical_features
        n_bins                  = preprocessor.n_bins
        self.n_bins             = n_bins
        self.n_bins_labels      = None
        num_fill_null           = preprocessor.num_fill_null
        self.num_fill_null      = num_fill_null

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
                self.numerical_parameters = []
                self.scaler = QuantileTransformer(output_distribution="normal", random_state=0).fit(data.select(numerical_features).collect())
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
        Apply numerical transformations to the dataset.

        This method fills null values and applies scaling techniques such as normalization, standardization, 
        quantile transformation, or k-bins discretization to numerical features.

        Parameters
        ----------
        data : pl.DataFrame
            A Polars DataFrame containing numerical features to be transformed.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame with numerical features transformed based on the selected 
            scaling method.

        Raises
        ------
        ValueError
            If an invalid scaling method is provided.
        """
        scaling = self.scaling
        numerical_features = self.numerical_features
        num_fill_null = self.num_fill_null

        # Scale numerical features with the specified method
        match scaling:
            case "none":
                pass
            case "normalize":
                # Normalization of numerical features
                for col in numerical_features:
                    col_min = self.numerical_parameters[0][col].item()
                    col_max = self.numerical_parameters[1][col].item()
                    data = data.with_columns((pl.col(col) - col_min) / (col_max - col_min))
            case "standardize":
                # Standardization of numerical features
                for col in numerical_features:
                    col_mean = self.numerical_parameters[0][col].item()
                    col_std  = self.numerical_parameters[1][col].item()
                    data = data.with_columns((pl.col(col) - col_mean) /  col_std) 
            case "quantile": 
                num_data = pl.DataFrame(self.scaler.transform(data.select(numerical_features)),
                                        schema = self.numerical_features)
                for col in num_data.columns:
                    data = data.with_columns(num_data[col].alias(col))
            case "kbins":
                # KBinsDiscretizer applied to numerical features
                data = data.with_columns(numerical_features.qcut(self.n_bins, labels=self.n_bins_labels))

        # Fill null values with the specified strategy
        col_num = pl.col(numerical_features)
        if isinstance(num_fill_null, str):
            if num_fill_null == "interpolate":
                data = data.with_columns(col_num).interpolate()
            elif num_fill_null == "none":
                if scaling in ["quantile", "normalize"]:
                    data = data.with_columns(col_num.fill_null(-0.01))
                else:
                    data = data.with_columns(col_num.fill_null(-10))
            else:
                data = data.with_columns(col_num.fill_null(strategy=num_fill_null))
        else:
            data = data.with_columns(col_num.fill_null(num_fill_null))

        return data


    def inverse_transform(self, data: pl.DataFrame):
        """
        Reverse the transformations applied to numerical features.

        This method restores the original numerical values by applying the inverse of 
        the normalization, standardization, or quantile transformation that was applied 
        during the preprocessing phase.

        Parameters
        ----------
        data : pl.DataFrame
            A Polars DataFrame containing transformed numerical features.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame where numerical features have been reverted 
            to their original scale.

        Raises
        ------
        ValueError
            If an invalid scaling method is provided.
        """
        numerical_features  = self.numerical_features
        num_fill_null = self.num_fill_null
        scaling = self.scaling

        # If num_fill_null is "none", convert very negative values to NaN
        if num_fill_null=="none":
            if scaling in ["quantile", "normalize"]:
                for col in numerical_features:
                    data = data.with_columns(
                        pl.when(pl.col(col) <= -0.01)
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
            else:
                for col in numerical_features:
                    data = data.with_columns(
                        pl.when(pl.col(col) <= -10)
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )

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
                num_data = pl.DataFrame(self.scaler.inverse_transform(data.select(numerical_features)),
                                        schema=self.numerical_features)
                for col in num_data.columns:
                    data = data.with_columns(num_data[col].alias(col))
            
        return data



    