import sys

import pandas as pd
import polars as pl
import polars.selectors as cs

from tsfresh import extract_relevant_features

from typing import List, Dict, Tuple, Union, TypeAlias, Literal
import bisect
from clearbox_preprocessor.utils.numerical_transformers import calculate_quantile_mappings, transform_with_quantiles
import warnings


class Preprocessor:
    numerical_features   : Tuple[str]
    categorical_features : Tuple[str]
    temporal_features    : Tuple[str]
    discarded_features   : Union[List[str], Dict[str, str]]
    single_value_columns : Dict[str, str]
    FillNullStrategy     : TypeAlias = Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"]
    Scaling              : TypeAlias = Literal["normalize", "standardize", "quantile"]
    """
    A class for preprocessing datasets, including feature selection, handling missing values, scaling, 
    and time-series feature extraction.

    Attributes
    ----------
    numerical_features : Tuple[str]
        Names of the numerical features in the dataset.
    categorical_features : Tuple[str]
        Names of the categorical features in the dataset.
    temporal_features : Tuple[str]
        Names of the temporal features in the dataset.
    discarded_features : Union[List[str], Dict[str, str]]
        Features that were discarded during preprocessing, along with reasons if available.
    single_value_columns : Dict[str, str]
        Dictionary storing columns with only one unique value, along with the unique value.
    """
    def __init__(
            self, 
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame, 
            discarding_threshold: float = 0.9, 
            get_discarded_info: bool = False,
            excluded_col: List = [],
            time: str = None,
            missing_values_threshold: float = 0.999,
            n_bins: int = 0,
            scaling: str = "normalize", 
            num_fill_null : FillNullStrategy = "mean",
            cat_labels_threshold: float = 0.01,
            unseen_labels = 'ignore',
            target_columns = None,
        ):
        """
        Initialize the Preprocessor class.

        Parameters
        ----------
        data : pl.LazyFrame or pl.DataFrame or pd.DataFrame
            The dataset to be processed. It can be a Polars LazyFrame, Polars DataFrame, or Pandas DataFrame.

        discarding_threshold : float, optional, default=0.9
            A float value between 0 and 1 that sets the threshold for discarding categorical features.
            If more than `discarding_threshold * 100%` of values in a categorical feature are unique,
            the column is discarded. For instance, if `discarding_threshold=0.9`, a column will be
            discarded if more than 90% of its values are unique.

        get_discarded_info : bool, optional, default=False
            If set to `True`, the preprocessor will feature the method `get_discarded_features_reason`,
            which provides information on which columns were discarded and the reason for discarding.
            Note that enabling this option may significantly slow down the processing operation.
            The list of discarded columns is available even when `get_discarded_info=False`, so consider
            setting this flag to `True` only if you need to know why a column was discarded or, in the case
            of columns containing only one unique value, what that value was.

        excluded_col : List, optional, default=[]
            A list of column names to be excluded from processing. These columns will be returned in the
            final DataFrame without being modified.

        time : str, optional, default=None
            The name of the time column to sort the DataFrame in case of time series data.

       scaling : str, default="normalize"
            The method used to scale numerical features:
            - "normalize"   : Normalizes numerical features to the [0, 1] range.
            - "standardize" : Standardizes numerical features to have a mean of 0 and a standard deviation of 1.
            - "quantile" : Transforms numerical features using quantiles information.


        num_fill_null : FillNullStrategy or str, default="mean"
            Strategy or value used to fill null values in numerical features:
            - "mean"        : Fills null values with the mean of the column.
            - "interpolate" : Fills null values using interpolation.
            - "forward"     : Fills null values using the previous non-null value.
            - "backward"    : Fills null values using the next non-null value.
            - "min"         : Fills null values with the minimum value of the column.
            - "max"         : Fills null values with the maximum value of the column.
            - "zero"        : Fills null values with zeros.
            - "one"         : Fills null values with ones.
            - value         : Fills null values with the specified value.

        n_bins : int, default=0
            Number of bins to discretize numerical features. If set to a value greater than 0,
            numerical features are discretized into the specified number of bins using quantile-based
            binning. If 0, the scaling method specified in `scaling` is applied instead.

        unseen_labels : str, default="ignore"
            - "ignore"        : If new data contains labels unseen during fit one hot encoding contains 0 in every column.
            - "error"         : Raise an error if new data contains labels unseen during fit.

        Raises
        ------
        ValueError
            If `discarding_threshold` is not between 0 and 1.

        Notes
        -----
        - The constructor transforms Pandas DataFrames into Polars LazyFrames for more efficient processing.
        - The methods `_infer_feature_types` and `_feature_selection` are called to handle feature type
        inference and feature selection.
        """
        # Transform data from Pandas or Polars DataFrame to Polars LazyFrame
        if isinstance(data, pd.DataFrame):
            self.data_was_pd = True
            data = pl.from_pandas(data).lazy()
        elif isinstance(data, pl.DataFrame):
            data = data.lazy()
            self.data_was_pd = False
        else:
            self.data_was_pd = False

        if discarding_threshold>1 or discarding_threshold<0:
            raise ValueError("Invalid value for discarding_threshold")
    
        self.discarding_threshold   = discarding_threshold
        self.discarded_info = []
        self.missing_threshold      = missing_values_threshold
        self.get_discarded_info     = get_discarded_info
        self.excluded_col           = excluded_col
        self.time                   = time

        self._infer_feature_types(data)
        self._feature_selection(data)
        self.nbins_labels = None
        self.nbins = n_bins
        self.numerical_parameters = None
        self.num_fill_null = num_fill_null
        self.scaling = scaling
        col_num = cs.numeric()-cs.by_name(self.excluded_col) 
        col_str = cs.string()-cs.by_name(self.excluded_col) 
        data_shape = data.select(pl.len()).collect()['len'][0]
        self.cat_labels_threshold = cat_labels_threshold
        rare_labels_dict = {}
        self.unseen_labels = unseen_labels

        for col in data.select(cs.string()-cs.by_name(self.excluded_col)).columns:
            freq = (
                data
                .group_by(col)
                .agg(pl.len().alias("frequency"))
            )

            rare_labels = (
                freq
                .filter(pl.col("frequency") < self.cat_labels_threshold * data_shape)
                .select(col)
                .collect()  
            )
            if rare_labels.height > 0:
                rare_values_list = rare_labels.get_column(col).to_list()
            else:
                rare_values_list = []
            rare_labels_dict[col] = rare_values_list
        
        
        self.rare_labels = rare_labels_dict
        for col in self.rare_labels.keys():
            data = data.with_columns(
                pl.when(pl.col(col).is_in(self.rare_labels[col]))
                .then(pl.lit("other"))
                .otherwise(pl.col(col))
                .alias(col)
            )
        self.one_hot_encoded_columns = data.select(col_str).collect().to_dummies().columns
        
        if self.nbins > 0:
            # KBinsDiscretizer applied to numerical features to discretize continuous numerical features into a specified number of bins.
            # This is useful for transforming continuous data into categorical data, which can be beneficial for certain types of analysis or models.
            self.nbins_labels=list(map(str, list(range(0, n_bins))))     
        else:
            match scaling:
                case "normalize":
                    # Normalization parameters of numerical features
                    self.numerical_parameters = [data.select(col_num).min().collect(), 
                                                 data.select(col_num).max().collect()]
                case "standardize":
                    # Standardization parameters of numerical features
                    self.numerical_parameters = [data.select(col_num).mean().collect(), 
                                                 data.select(col_num).std().collect()] 
                case "quantile":
                    quantile_maps = calculate_quantile_mappings(data.select(col_num).collect())      
                    self.numerical_parameters = quantile_maps
                case _:
                    raise ValueError(f"Unknown scaling method: {scaling}")


    def _infer_feature_types(self, data: pl.LazyFrame) -> None:
        """
        Infer the type of each feature in the LazyFrame. The type is either numerical, categorical, temporal or boolean. 
        """
        # Store the names of boolean columns into 'boolean_features'
        self.boolean_features = cs.expand_selector(data, cs.boolean())

        # Store the names of temporal columns into 'temporal_features'
        self.temporal_features = cs.expand_selector(data, cs.temporal())

        # Store the names of numerical columns into 'numerical_features'
        self.numerical_features = cs.expand_selector(data, cs.numeric())

        # Store the names of string columns into 'categorical_features'
        self.categorical_features = cs.expand_selector(data, cs.string())

    def _feature_selection(
            self,
            data: pl.LazyFrame,
        ) -> None:
        """
        Perform a selection of the most useful columns for a given DataFrame, ignoring the other features. The selection is
        performed in two steps:
            1. The columns with more than 90% of missing values are discarded.
            2. The columns containing only one value or, conversely, a large number of different values are discarded. In the latter
               case the default threshold is equal to 90%, i.e. if more than 90% of the instances have different values then the entire
               column is discarded.
        """
        # Replace empty strings ("") with None value
        col = cs.string()-cs.by_name(self.excluded_col)
        data = data.with_columns(col.replace("",None)) 

        col_all = cs.all()-cs.by_name(self.excluded_col)
        col_cat = cs.by_name(self.categorical_features)-cs.by_name(self.excluded_col)
        self.discarded_features = []
        # All feature types - Discard columns if more than missing_threshold% of values is null or all values are equal (only one value in the column)
        lf_ = data.select(pl.any_horizontal(col_all.drop_nulls().value_counts().count() == 1, 
                                               )).collect()
            
        # Categorical features - Discard columns that contain a large number of different values (more than discarding_threshold % of values are diffent from each other)
        lf_cat = data.select(col_cat.value_counts().count()>pl.len()*self.discarding_threshold).collect()

        for col in lf_.columns: 
            if lf_.select(pl.col(col)).item() == True:
                warning_message = f"{col} contains a unique value"
                warnings.warn(warning_message)
                self.discarded_info.append(warning_message)
            elif col in lf_cat.columns and lf_cat.select(pl.col(col)).item() == True:
                warning_message = f"{col} contains too many labels"                
                self.discarded_features.append(col)
                warnings.warn(warning_message)
                self.discarded_info.append(warning_message)

        # Update the numerical_features, categorical_features and temporal_features lists removing the discarded columns
        self.boolean_features     = tuple(set(self.boolean_features)     - set(self.discarded_features))
        self.numerical_features   = tuple(set(self.numerical_features)   - set(self.discarded_features))
        self.categorical_features = tuple(set(self.categorical_features) - set(self.discarded_features))
        self.temporal_features    = tuple(set(self.temporal_features)    - set(self.discarded_features))
    
    def transform(
            self, 
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame, 
        ) -> pl.DataFrame | pd.DataFrame:
        """
        Transform the input dataset by processing numerical, temporal, and categorical columns.
        This includes filling null values, scaling or discretizing numerical features, and encoding
        categorical features.

        Parameters
        ----------
        data : pl.LazyFrame or pl.DataFrame or pd.DataFrame
            The input dataset to be transformed. It can be a Polars LazyFrame, Polars DataFrame,
            or a Pandas DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            The transformed dataset, returned as a Polars DataFrame or a Pandas DataFrame,
            depending on the input data type.

        Raises
        ------
        SystemExit
            If the input data type does not match the data type used when the Preprocessor was initialized.

        Notes
        -----
        - The method identifies and processes numerical, temporal, and categorical features separately.
        - Categorical features are filled with the most frequent value and then one-hot encoded.
        - Numerical features can be normalized, standardized, or discretized based on the specified parameters.
        - Temporal features are filled using interpolation and reordered to the beginning of the dataset.
        """
        # Transform data from Pandas or Polars DataFrame to Polars LazyFrame
        if isinstance(data, pd.DataFrame) and self.data_was_pd == True:
            data = pl.from_pandas(data).lazy()
        elif isinstance(data, pl.DataFrame) and self.data_was_pd == False:
            data = data.lazy()
        elif isinstance(data, pl.LazyFrame) and self.data_was_pd == False:
            pass
        else:
            sys.exit('ErrorType\nThe datatype provided does not not match with the datatype of the dataset provided when the Preprocessor was initialized.')

        # Replace empty strings ("") with None value
        col_str = cs.string()-cs.by_name(self.excluded_col)
        data = data.with_columns(col_str.replace("",None)) 

        # Drop discarded columns, previously defined in _feature_selection()
        if isinstance(self.discarded_features, dict):
            data = data.drop(self.discarded_features.keys())
        else:
            data = data.drop(self.discarded_features)

    # Temporal features processing
        # Fill Null values by interpolation and reorder columns such that temporal ones are positioned at the beginning of the LazyFrame 
        col = cs.temporal()-cs.by_name(self.excluded_col)   
        data = data.select(col.interpolate(), cs.all()-col)

    # Numerical features processing
        # Fill Null values with the selcted strategy or value (default: "mean")
        col_num = cs.numeric()-cs.by_name(self.excluded_col) 
        if isinstance(self.num_fill_null, str):
            if self.num_fill_null == "interpolate":
                data = data.with_columns(col_num).interpolate()
            else:
                data = data.with_columns(col_num.fill_null(strategy=self.num_fill_null))
        else:
            data = data.with_columns(col_num.fill_null(self.num_fill_null))
    
    
        # OneHotEncoding and collect the pl.LazyFrame into a pl.Dataframe
        # The Dataframe is sorted according to "time" column if present
        col_str = cs.string()-cs.by_name(self.excluded_col) 
        data = data.collect()
        
        for col in self.rare_labels.keys():
            data = data.with_columns(
                pl.when(pl.col(col).is_in(self.rare_labels[col]))
                .then(pl.lit("other"))
                .otherwise(pl.col(col))
                .alias(col)
            )
        
        if self.nbins > 0:
            # KBinsDiscretizer applied to numerical features
            data = data.with_columns(col_num.qcut(self.n_bins, labels=self.nbins_labels))
        else:
            match self.scaling:
                case "normalize":
                    # Normalization of numerical features
                    for col in data.select(col_num).columns:
                        col_min = self.numerical_parameters[0][col].item()
                        col_max = self.numerical_parameters[1][col].item()
                        data = data.with_columns((pl.col(col) - col_min) / (col_max - col_min))
                case "standardize":
                    # Standardization of numerical features
                    for col in data.select(col_num).columns:
                        col_mean = self.numerical_parameters[0][col].item()
                        col_std  = self.numerical_parameters[1][col].item()
                        data = data.with_columns((pl.col(col) - col_mean) /  col_std)    
                case "quantile":
                    # Quantile transformation of numerical features
                    num_data = transform_with_quantiles(data.select(col_num), 
                                                        self.numerical_parameters, 
                                                        output_distribution="normal")    
                    for col in num_data.columns:
                        data = data.with_columns(num_data[col].alias(col))

        if self.time:
            df = data.sort(self.time).to_dummies(col_str)
        else:
            df = data.to_dummies(col_str)


        if self.unseen_labels == 'error' and len([i for i in df.columns if i not in self.one_hot_encoded_columns]):
            raise ValueError("New data contains unseen labels")
        
        not_in_new_data = [i for i in self.one_hot_encoded_columns if i not in data.columns]
        for i in not_in_new_data:
            df = df.with_columns(pl.lit(0).alias(i))
        df = df[list(self.numerical_features)+
                    self.one_hot_encoded_columns+
                    list(self.boolean_features)]

        if self.data_was_pd:
            df = df.to_pandas()        
        return df

    def extract_ts_features(
            self,
            data: pl.LazyFrame | pd.DataFrame,
            y: pl.Series | pd.Series,
            time: str = None,
            column_id:str = None,
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
            will try to use `self.time` if available.
        column_id : str, optional
            The name of the ID column, if present in the data. This is used to distinguish 
            different time-series within the same dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the extracted and filtered relevant time-series features.

        Raises
        ------
        ValueError
            If the provided data is not a Polars LazyFrame or a Pandas or Polars DataFrame.
        ValueError
            If the provided label series is not a Polars Series or a Pandas Series.
        ValueError
            If the time column name is not provided and `self.time` is not available.

        Notes
        -----
        - The function uses the `extract_relevant_features` method from the `tsfresh` library 
        to extract features from the time-series data.
        - The method stores the filtered features in `self.features_filtered` for further use.
        """
        if isinstance(data, pl.LazyFrame):
            data_pd = data.collect().to_pandas()
        elif isinstance(data, pl.DataFrame):
            data_pd = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            data_pd = data
        else:
            print("The dataframe must be a Polars LazyFrame or a Pandas DataFrame")
            return

        if isinstance(y, pl.Series):
            y = y.to_pandas()
        elif isinstance(y, pd.Series):
            pass
        else:
            print("The labels series must be a Polars Series or a Pandas Series")
            return

        if not self.time and not time:
            print("Please enter a name for the time column")
            return
        elif self.time and not time:
            time = self.time
        
        features_filtered = extract_relevant_features(data_pd, y, column_sort=time, column_id=column_id)
        self.features_filtered = features_filtered
        
        return features_filtered

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
    
    def help(self):
        """
        Print some guidelines and working principles of the Preprocessor
        """
        print("""       Initializze the Preprocessor as: \n
            preprocessor = Preprocessor(your_dataframe, discarding_threshold: float = 0.9, get_discarded_info = False)\n
        'your_dataframe' can be a Polars LazyFrame or a Pandas DataFrame.
        Below are listed all the possible values for the arguments of the Preprocessor():\n
        'discarding_threshold': (default = 0.9)
            Float number between 0 and 1 to set the threshold for discarding categorical features. 
            If more than discarding_threshold * 100 % of values in a categorical feature are different from each other, then the column is discarded. 
            For example, if discarding_threshold=0.9, a column will be discarded if more than 90% of its values are unique.\n
        'get_discarded_info': (defatult = False)
            When set to 'True', the preprocessor will feature the methods preprocessor.get_discarded_features_reason, which provides information on which columns were discarded and the reason why.
            Note that setting get_discarded_info=True will considerably slow down the processing operation!
            The list of discarded columns will be available even if get_discarded_info=False, so consider setting this flag to True only if you need to know why a column was discarded or, if it contained just one value, what that value was.\n\n
        After having initialized the preprocessor, call the following method to start the processing: \n
            preprocessor.transform(your_dataframe, num_fill_null_strat="mean", n_bins : int  = 0)\n
        Below are listed all the possible values for the arguments of the method .transform():\n
        'scaling': (default="normalize")
            Specifies the scaling operation to perform on numerical features.
            - "normalize"   : applies normalization to numerical features
            - "standardize" : applies standardization to numerical features\n
        'num_fill_null': (default = "mean")
            Specifies the value to fill null values with or the strategy for filling null values in numerical features.
            - value      : fills null values with the specified value  
            - "mean"     : fills null values with the average of the column
            - "forward"  : fills null values with the previous non-null value in the column
            - "backward" : fills null values with the following non-null value in the column
            - "min"      : fills null values with the minimum value of the column
            - "max"      : fills null values with the maximum value of the column
            - "zero"     : fills null values with zeros
            - "one"      : fills null values with ones
        'n_bins': (default = 0)
            Integer number that determines the number of bins to which numerical features are discretized. When set to 0, the preprocessing step defaults to the scaling method specified in the 'scaling' atgument instead of discretization.
            Note that if n_bins is different than 0, discretization will take place instead of scaling, regardless of whether the 'scaling' argument is specified.
            """)