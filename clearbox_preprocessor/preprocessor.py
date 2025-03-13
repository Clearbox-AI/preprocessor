import sys

import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import LabelEncoder

from tsfresh import extract_relevant_features

from typing import List, Tuple, Literal, Dict
import warnings
import numpy as np

from .utils.numerical_transformer import NumericalTransformer
from .utils.categorical_transformer import CategoricalTransformer
from .utils.datetime_transformer import DatetimeTransformer

# UNCOMMENT FOR DEBUGGING
# from utils.numerical_transformer import NumericalTransformer
# from utils.categorical_transformer import CategoricalTransformer
# from utils.datetime_transformer import DatetimeTransformer

class Preprocessor:
    ML_TASKS = {"classification", "regression", None}
    NUM_FILL_NULL_STRATEGIES = {"interpolate","forward", "backward", "min", "max", "mean", "zero", "one"}
    SCALING_STRATEGIES = {"none", "normalize", "standardize", "quantile"}
    """
    A class for preprocessing datasets based on polars, including feature selection, handling missing values, scaling, 
    and time-series feature extraction.

    Parameters
    ----------
    data : pl.LazyFrame or pl.DataFrame or pd.DataFrame
        The dataset to be processed. It can be a Polars LazyFrame, Polars DataFrame, or Pandas DataFrame.

    cat_labels_threshold : float, optional, default=0.02
        A float value between 0 and 1 that sets the threshold for discarding categorical features.
        It defines a minimum frequency threshold for keeping a label as a separate category. If a label appears 
        in less than :code:`cat_labels_threshold * 100%` of the total occurrences in a categorical column, it is grouped 
        into a generic ``"other"`` category. 

        For instance, if ``cat_labels_threshold=0.02`` and a label appears less than 2% in the dataset, that label will be converted to `"other"`.

    excluded_col : List, optional, default=[]
        A list of column names to be excluded from processing. These columns will be returned in the
        final DataFrame without being modified.

    time_id : str, optional, default=None
        The time series index. It identifies unique time series events.

    scaling : str, default="none"
        The method used to scale numerical features:

        - "none"        : No scaling is applied   
        - "normalize"   : Normalizes numerical features to the [0, 1] range.
        - "standardize" : Standardizes numerical features to have a mean of 0 and a standard deviation of 1.
        - "quantile"    : Transforms numerical features using quantiles information.
        - "kbins"       : Converts continuous numerical data into discrete bins. The number of bins is defined by the parameter n_bin

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
        Number of bins to discretize numerical features. If set to a value greater than 0 and if scaling=="kbins",
        numerical features are discretized into the specified number of bins using quantile-based
        binning.

    unseen_labels : str, default="ignore"
        - "ignore"        : If new data contains labels unseen during fit one hot encoding contains 0 in every column.
        - "error"         : Raise an error if new data contains labels unseen during fit.

    ml_task : ["classification", "regression"], default=None
        The Machie Learning task the dataset is used for. It can be either classification or regression.

    target_column : str, default=None
        The target coluymn for Machine Learning tasks. If the task is classification, the target column is encoded using LabelEncoder, while if it is regression the target column is normalized.
    
    Attributes
    ----------
    numerical_features : Tuple[str]
        Names of the numerical features in the dataset.

    categorical_features : Tuple[str]
        Names of the categorical features in the dataset.

    datetime_features : Tuple[str]
        Names of the time features in the dataset.

    discarded_features : Union[List[str], Dict[str, str]]
        Features that were discarded during preprocessing, along with reason they were discarded, if available.

    single_value_columns : Dict[str, str]
        Dictionary storing columns with only one unique value, along with the unique value.

    Raises
    ------
    ValueError
        If ``cat_labels_threshold`` is not between 0 and 1.

    Notes
    -----
    The constructor transforms Pandas DataFrames into Polars LazyFrames for more efficient processing.
    """
    def __init__(
            self, 
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame, 
            cat_labels_threshold: float = 0.02,
            excluded_col: List = [],
            time_id: str = None,
            missing_values_threshold: float = 0.999,
            n_bins: int = 0,
            scaling: Literal["none", "normalize", "standardize", "quantile"] = "none", 
            num_fill_null : Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"] = "mean",
            unseen_labels = 'ignore',
            ml_task: Literal["classification", "regression", None] = None,
            target_column: str = None,
        ):
        # Argument values check
        if cat_labels_threshold>1 or cat_labels_threshold<0:
            raise ValueError("Invalid value for cat_labels_threshold")
        if ml_task not in self.ML_TASKS:
            raise ValueError("Invalid value for ml_task")
        if target_column is not None and target_column not in data.columns:
            raise ValueError("The target column is not present in the dataset")
        for col in excluded_col:
            if col not in data.columns:
                raise ValueError(f"The excluded column {col} is not present in the dataset")
        if scaling not in self.SCALING_STRATEGIES:
            raise ValueError("Invalid value for scaling")
        if num_fill_null not in self.NUM_FILL_NULL_STRATEGIES:
            raise ValueError("Invalid value for num_fill_null")
        
        # Transform data from Pandas or Polars DataFrame to Polars LazyFrame
        if isinstance(data, pd.DataFrame):
            self.data_was_pd = True
            data = pl.from_pandas(data).lazy()
        elif isinstance(data, pl.DataFrame):
            data = data.lazy()
            self.data_was_pd = False
        else:
            self.data_was_pd = False

        self.discarded_info         = []
        self.missing_threshold      = missing_values_threshold
        self.excluded_col           = excluded_col
        self.time_id                = time_id
        self.n_bins_labels          = None
        self.n_bins                 = n_bins
        self.num_fill_null          = num_fill_null
        self.scaling                = scaling
        self.cat_labels_threshold   = cat_labels_threshold
        self.unseen_labels          = unseen_labels
        self.target_column          = target_column
        self.ml_task                = ml_task

        if ml_task is not None and target_column is None:
            warnings.warn('The Machine Learning task was specified but the target column was not specified.')
        if target_column is not None and ml_task is None:
            warnings.warn('The target column was not specified but the Machine Learning task was not specified.')
        if target_column is not None:
            self.excluded_col.append(target_column)

        self._infer_feature_types(data)
        self.datetime_transformer   = DatetimeTransformer(self)
        data                        = self.datetime_transformer.fit(data.collect()) # Return data with time columns transformed into timestamp integer
        self.datetime_features          = self.datetime_transformer.datetime_features
        self.categorical_features   = tuple(set(self.categorical_features) - set(self.datetime_features))
        self._feature_selection(data)

        # Initialization of NumericalTransformer and CategoricalTransformer
        if len(self.numerical_features) > 0:
            self.numerical_transformer   = NumericalTransformer(data, self)
        if len(self.categorical_features) > 0:
            self.categorical_transformer = CategoricalTransformer(data, self)

        if target_column is not None:
            match ml_task:
                case "classification":
                    self.target_col_encoder = LabelEncoder()
                    self.target_col_encoder.fit(data.select(pl.col(target_column)).collect().to_series())
                case "regression":
                    self.target_col_encoder = [data.select(pl.col(target_column)).min().collect(), 
                                            data.select(pl.col(target_column)).max().collect()]
                case None:
                    pass
            
    def _infer_feature_types(
            self, 
            data: pl.LazyFrame
        ) -> None:
        """
        Infer the type of each feature in the LazyFrame. The type is either numerical, categorical, time or boolean. 
        """
        # Collect the schema to get column names and their data types
        schema = data.collect_schema()

        # Store the names of boolean columns into 'boolean_features'
        boolean_columns = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype == pl.Boolean]
        self.boolean_features = tuple(set(boolean_columns) - set(self.excluded_col))

        # Store the names of time columns into 'datetime_features'
        datetime_columns = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype in (pl.Date, pl.Datetime)]
        self.datetime_features = tuple(set(datetime_columns) - set(self.excluded_col))

        # Store the names of numerical columns into 'numerical_features'
        numerical_columns = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype in (pl.Int64, pl.Float64)]
        self.numerical_features = tuple(set(numerical_columns) - set(self.excluded_col))

        # Store the names of categorical columns into 'categorical_features'
        categorical_columns = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype == pl.Utf8]
        self.categorical_features = tuple(set(categorical_columns) - set(self.excluded_col))

    def _shrink_labels(
            self, 
            instance: pl.DataFrame, 
            too_much_info: Dict[str, List[str]]
        ) -> pl.DataFrame:
        """
        Shrinks labels in the dataset by replacing rare labels with a generic category.

        Parameters
        ----------
        instance : pl.DataFrame
            The Polars DataFrame containing the dataset to modify.
        too_much_info : dict[str, list[str]]
            Dictionary where keys are column names and values are lists of labels to be replaced.

        Returns
        -------
        pl.DataFrame
            A modified DataFrame where specified labels are replaced.

        """
        expressions = []
        schema = instance.collect_schema()

        for column_name, values_to_shrink in too_much_info.items():
            if schema[column_name] == pl.String:
                # Convert null values in string "None" and substyitute rare categorical labels with "other"
                expr = (pl.col(column_name).
                        fill_null("None").
                        replace(values_to_shrink,['other']))
            expressions.append(expr)

        # Apply all transformations in one go
        instance = instance.with_columns(expressions)
        return instance

    def _feature_selection(
            self,
            data: pl.LazyFrame,
        ) -> None:
        """
        Perform feature selection to retain the most informative columns in a DataFrame while discarding redundant features.

        The selection process follows these steps:
        
        1. **Low-Variance Filtering**:
            Columns containing only one value are discarded.
        2. **High Cardinality Filtering**:
            Categorical columns in which a single unique value appears in more than 98% of the records are discarded.
        3. **Rare Label Aggregation**: 
            In categorical columns, labels appearing in less than a specified proportion (``cat_labels_threshold``) of instances are aggregated into a single category `"other"`.

        Warnings are issued for discarded features, and the remaining features are updated accordingly.

        Parameters
        ----------
        data : pl.LazyFrame
            The input Polars LazyFrame containing the dataset for feature selection.

        Warnings
        --------
        - Columns that contain only one unique value are discarded.
        - Categorical columns in which a single unique value appears in more than 98% of the records are discarded.
        - Categorical columns with rare labels are modified by aggregating them into ``"other"``.
        """
        self.discarded_features = []

        col_cat = cs.by_name(self.categorical_features)-cs.by_name(self.excluded_col)
        data = data.with_columns(col_cat.replace({"":None, " ":None})) 
        data = data.collect()

        cat_features_stats = [
            (i, data[i].value_counts(), data[i].n_unique(), data.columns.index(i))
            for i in self.categorical_features
        ]

        ord_features_stats = [
            (i, data[i].value_counts(), data[i].unique(), data.columns.index(i))
            for i in self.numerical_features
        ]

        no_info = []
        too_much_info = {}
        # Categorical features
        for column_stats in cat_features_stats:
            if (column_stats[1].shape[0] == 1) or (column_stats[1].shape[0] >= (data.shape[0] * 0.98)):
                no_info.append(column_stats[0])
                warning_message = f"\n{column_stats[0]} contains a single value"
                warnings.warn(warning_message+' and was discarded')
                self.discarded_features.append(column_stats[0])
                self.discarded_info.append(warning_message)
            else:
                counts = column_stats[1].select("count").to_numpy() / column_stats[1].select("count").sum()
                values_to_shrink_indices = np.where(counts < self.cat_labels_threshold)[0]
                if values_to_shrink_indices.shape[0] > 0 and column_stats[1].shape[0] > 2:
                    too_much_info[column_stats[0]] = [column_stats[1][column_stats[0]].to_list()[i] for i in values_to_shrink_indices]

        # Numerical features
        for column_stats in ord_features_stats:
            if column_stats[1].shape[0] <= 1:
                no_info.append(column_stats[0])
                warning_message = f"\n{column_stats[0]} contains a single value"  
                warnings.warn(warning_message+' and was discarded')
                self.discarded_features.append(column_stats[0])
                self.discarded_info.append(warning_message)

        if len(too_much_info)>0:
            warnings.warn(f"Some rare labels have been aggregated into the 'other' category. You can view the discarded labels using the 'discarded' attribute of your Preprocessor class.\nIf certain labels were unintentionally discarded, try adjusting the 'cat_labels_threshold' parameter, but keep in mind that rare labels are at risk of being re-identified in case of privacy attack.")

        data = self._shrink_labels(data, too_much_info)
        self.discarded = (no_info, too_much_info)

        # Update the numerical_features, categorical_features and datetime_features lists removing the discarded columns
        self.boolean_features     = tuple(set(self.boolean_features)     - set(self.discarded_features))
        self.numerical_features   = tuple(set(self.numerical_features)   - set(self.discarded_features))
        self.categorical_features = tuple(set(self.categorical_features) - set(self.discarded_features))
        self.datetime_features    = tuple(set(self.datetime_features)    - set(self.discarded_features))
    
    def transform(
            self, 
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame, 
        ) -> pl.DataFrame | pd.DataFrame:
        """
        Transform the input dataset by processing numerical, time, and categorical columns.
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
        - The method identifies and processes numerical, time, and categorical features separately.
        - Categorical features are filled with the most frequent value and then one-hot encoded.
        - Numerical features can be normalized, standardized, or discretized based on the specified parameters.
        - time features are filled using interpolation and reordered to the beginning of the dataset.

        Example:
        --------
        .. code-block:: python

            preprocessor = Preprocessor(real_data, scaling="standardize")
            transformed_data = preprocessor.transform(real_data)
        """
        # Transform data from Pandas.DataFrame or Polars.DataFrame to Polars.LazyFrame
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data).lazy()
        elif isinstance(data, pl.DataFrame):
            data = data.lazy()
        elif isinstance(data, pl.LazyFrame):
            pass
        else:
            sys.exit(f'ErrorType\nThe datatype provided ({type(data)}) is not supported by the Preprocessor.')
        
        # Replace empty strings ("") with None value
        col_str = pl.col(self.categorical_features)
        data = data.with_columns(col_str.replace({"":None, " ":None})) 

        # Substitute rare lables with "other" in categorical features
        data = self._shrink_labels(data, self.discarded[1])

        # Drop discarded columns, previously defined in _feature_selection()
        if isinstance(self.discarded_features, dict):
            data = data.drop(self.discarded_features.keys())
        else:
            data = data.drop(self.discarded_features)

        # Time features processing
        # Convert time columns to timestamp integers, fill null values by linear interpolation and scale time columns
        if len(self.datetime_features)>0:
            data = self.datetime_transformer.transform(data)
            
        # Numerical features processing
        # Fill Null values with the selcted strategy or value (default: "mean")
        # Scale numerical features if scaling parameter was specified
        if hasattr(self, "numerical_transformer"):
            data = self.numerical_transformer.transform(data)

        # Categorical features processing
        # OneHotEncoding and collect the pl.LazyFrame into a pl.Dataframe
        # The Dataframe is sorted according to "time" column if present
        if hasattr(self, "categorical_transformer"):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            data, new_encoded_columns = self.categorical_transformer.transform(data)

            self.categorical_features_sizes = []
            for values in new_encoded_columns.values():
                    self.categorical_features_sizes.append(len(values))

            # Raise an Error if a column in the new dataframe was not present in the encoded original datframe
            if self.unseen_labels == 'error':
                unseen = [col for col in new_encoded_columns if col not in self.categorical_transformer.original_encoded_columns]
                if unseen:
                    warnings.warn(f"New data contains unseen categorical columns: {unseen}", UserWarning)
        
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Handling the target column
        if self.target_column is not None:
            match self.ml_task:
                case "classification":
                    y_encoded = self.target_col_encoder.transform(data.select(pl.col(self.target_column)).to_series())
                    data = data.with_columns(pl.Series(y_encoded).alias(self.target_column))
                case "regression":
                    col_min = self.target_col_encoder[0][self.target_column].item()
                    col_max = self.target_col_encoder[1][self.target_column].item()
                    data = data.with_columns((pl.col(self.target_column) - col_min) / (col_max - col_min))
                case None:
                    pass
            
        if self.data_was_pd:
            data = data.to_pandas()   
            
        return data


    def inverse_transform(
            self,
            data: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    ) -> pl.DataFrame:
        """
        Reverse the transformations applied during the ``preprocessor.transform(data)`` phase.

        This method performs the inverse transformations on numerical and categorical
        features to restore the original dataset format.

        Parameters:
        ----------
        data : pl.LazyFrame | pl.DataFrame | pd.DataFrame
            The input dataset in either Polars LazyFrame, Polars DataFrame, or Pandas DataFrame format.
            The format must match the dataset type initially provided when the Preprocessor was initialized.

        Returns:
        -------
        pl.DataFrame
            A Polars DataFrame with all transformations reversed, including:
            - Restored numerical features (inverse normalization, standardization, or quantile transformation).
            - Reconstructed categorical features from one-hot encoding.

        Raises:
        ------
        SystemExit
            If the provided data type does not match the originally initialized dataset type.

        Notes:
        ------
        - If ``data_was_pd`` is ``True``, the method expects and processes a Pandas DataFrame.
        - If ``data_was_pd`` is ``False``, it expects and processes a Polars DataFrame or LazyFrame.
        - The numerical features are reversed based on the stored transformation method (``normalize``, ``standardize``, ``quantile``).
        - One-hot encoded categorical columns are reconstructed into their original categorical format.

        Example:
        --------
        .. code-block:: python

            preprocessor = Preprocessor(real_data, scaling="standardize")
            transformed_data = preprocessor.transform(real_data)
            
            # Reverse the transformations
            original_data = preprocessor.inverse_transform(transformed_data)
        """
        # Transform data from Pandas.DataFrame or Polars.LazyFrame to Polars.DataFrame
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            pass
        elif isinstance(data, pl.LazyFrame):
            data = data.collect()
        else:
            sys.exit(f'ErrorType\nThe datatype provided ({type(data)}) is not supported by the Preprocessor.')

        # Inverse transofmration of numerical and categorical features
        if len(self.datetime_features)>0:
            data = self.datetime_transformer.inverse_transform(data)
        if len(self.numerical_features)>0:
            data = self.numerical_transformer.inverse_transform(data)
        if len(self.categorical_features)>0:
            data = self.categorical_transformer.inverse_transform(data)
        if self.target_column is not None:
            match self.ml_task:
                case "classification":
                    y_original = self.target_col_encoder.inverse_transform(data.select(pl.col(self.target_column)).to_series())
                    data = data.with_columns(pl.Series(y_original).alias(self.target_column))
                case "regression":
                    col_min = self.numerical_parameters[0][self.target_column].item()
                    col_max = self.numerical_parameters[1][self.target_column].item()
                    data = data.with_columns(pl.col(self.target_column) * (col_max - col_min) + col_min)
                
        if self.data_was_pd:
            data = data.to_pandas()        
        return data
     
    def extract_ts_features(
            self,
            data:      pl.LazyFrame | pd.DataFrame,
            y:         pl.Series | pd.Series = None,
            time:      str = None,
            column_id: str = None,
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
            will try to use ``self.time`` if available.
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
            If the time column name is not provided and ``self.time`` is not available.

        Notes
        -----
        - The function uses the ``extract_relevant_features`` method from the ``tsfresh`` library 
        to extract features from the time-series data.
        - The method stores the filtered features in ``self.features_filtered`` for further use.
        """
        # Transform input dataframe into Pandas.DataFrame
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

    def get_features_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Gets the sizes of ordinal and categorical features after transformation.

        Returns:
            Tuple: Sizes of ordinal and categorical features.
        """
        numerical_sizes   = []
        categorical_sizes = []

        if hasattr(self, "numerical_transformer"):
            numerical_sizes.append(len(self.numerical_features))
        
        if hasattr(self, "categorical_features_sizes"):
            categorical_sizes = self.categorical_features_sizes
        elif hasattr(self, "categorical_transformer"):
            for values in self.categorical_transformer.original_encoded_columns.values():
                categorical_sizes.append(len(values))

        return numerical_sizes, categorical_sizes

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

if __name__=="__main__":
    #######################################################################################################
    ## DEBUGGING                                                                                         ##
    ## To run this part remove the dot from the import lines at the beginning of this file as following: ##
    ## from utils.numerical_transformer import NumericalTransformer                                      ##
    ## from utils.categorical_transformer import CategoricalTransformer                                  ##
    #######################################################################################################
    import os
    import pandas as pd
    import polars as pl

    # Tabular data
    # file_path = "https://raw.githubusercontent.com/Clearbox-AI/SURE/main/examples/data/census_dataset"
    # real_data = pl.read_csv(os.path.join(file_path,"census_dataset_training.csv"))

    # Time series
    # file_path = "https://raw.githubusercontent.com/Clearbox-AI/clearbox-synthetic-kit/main/tutorials/time_series/data/daily_delhi_climate"
    # path=os.path.join(file_path, "DailyDelhiClimateTrain.csv")
    # real_data = pl.read_csv(path)

    real_data= pd.read_csv('MEP/MEP_Pettinari_2024W37.csv')

    # file_path = "https://raw.githubusercontent.com/Clearbox-AI/clearbox-synthetic-kit/main/tests/resources/uci_adult_dataset"
    # real_data = pd.read_csv(os.path.join(file_path,"dataset.csv"))
    # # real_data["income"]      = real_data["income"].map({"<=50K": 0, ">50K": 1})

    preprocessor            = Preprocessor(real_data, num_fill_null='forward', scaling='standardize')
    real_data_preprocessed  = preprocessor.transform(real_data)
    df_inverse              = preprocessor.inverse_transform(real_data_preprocessed)