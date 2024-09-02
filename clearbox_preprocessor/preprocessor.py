import sys

import pandas as pd
import polars as pl
import polars.selectors as cs

from tsfresh import extract_relevant_features

from typing import List, Dict, Tuple, Union, TypeAlias, Literal

class Preprocessor:
    numerical_features   : Tuple[str]
    categorical_features : Tuple[str]
    temporal_features    : Tuple[str]
    discarded_features   : Union[List[str], Dict[str, str]]
    single_value_columns : Dict[str, str]
    
    FillNullStrategy    : TypeAlias = Literal["interpolate","forward", "backward", "min", "max", "mean", "zero", "one"]
    Scaling             : TypeAlias = Literal["normalize", "standardize"]

    def __init__(
        self, 
        data: pl.LazyFrame | pd.DataFrame, 
        discarding_threshold: float = 0.9, 
        get_discarded_info: bool = False,
        excluded_col: List = [],
        time: str = None
    ):
        """
        Initialize the preprocessor.

        Below are listed all the possible values for the arguments of the Preprocessor():
        
        'data':
            The dataset passed to the Preprocessor can be a Polars LazyFrame or a Pandas DataFrame.

        'discarding_threshold': (default = 0.9)
            Float number between 0 and 1 to set the threshold for discarding categorical features. 
            If more than discarding_threshold * 100 % of values in a categorical feature are different from each other, then the column is discarded. 
            For example, if discarding_threshold=0.9, a column will be discarded if more than 90% of its values are unique.

        'get_discarded_info': (defatult = False)
            When set to 'True', the preprocessor will feature the methods preprocessor.get_discarded_features_reason, which provides information on which columns were discarded and the reason why.
            Note that setting get_discarded_info=True will considerably slow down the processing operation!
            The list of discarded columns will be available even if get_discarded_info=False, so consider setting this flag to True only if you need to know why a column was discarded or, if it contained just one value, what that value was.

        'excluded_col': (default = [])
            List containing the names of the columns to be excluded from processing. These columns will be returned in the final dataframe without being manipulated.    

        'time': (default = None)
            String name of the time column by which to sort the dataframe in case of time series.
        """

        # Transform data from Pandas DataFrame to Polars LazyFrame
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
        self.get_discarded_info     = get_discarded_info
        self.excluded_col           = excluded_col
        self.time                   = time

        self._infer_feature_types(data)
        self._feature_selection(data)

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
            1. The columns with more than 50% of missing values are discarded.
            2. The columns containing only one value or, conversely, a large number of different values are discarded. In the latter
               case the default threshold is equal to 90%, i.e. if more than 90% of the instances have different values then the entire
               column is discarded.
        """
        # Replace empty strings ("") with None value
        col = cs.string()-cs.by_name(self.excluded_col)
        data = data.with_columns(col.replace("",None)) 

        col_all = cs.all()-cs.by_name(self.excluded_col)
        col_cat = cs.by_name(self.categorical_features)-cs.by_name(self.excluded_col)
        if self.get_discarded_info == False:
            self.discarded_features = []
            # All feature types - Discard columns if more than 50% of values is null or all values are equal (only one value in the column)
            lf_ = data.select(pl.any_horizontal(col_all.count()/pl.len() < 0.5, 
                                                col_all.drop_nulls().value_counts().count() == 1, 
                                               )).collect()
            
            # Categorical features - Discard columns that contain a large number of different values (more than discarding_threshold % of values are diffent from each other)
            lf_cat = data.select(col_cat.value_counts().count()>pl.len()*self.discarding_threshold).collect()

            for col in lf_.columns: 
                if lf_.select(pl.col(col)).item() == True:
                    self.discarded_features.append(col)
                elif col in lf_cat.columns and lf_cat.select(pl.col(col)).item() == True:
                    self.discarded_features.append(col)
        else:
            self.discarded_features    = dict()
            self.single_value_columns  = dict()

            # All feature types - Discard columns if more than 50% of values is null or all values are equal (only one value in the column)
            df_50perc_null  = data.select(col_all.count()/pl.len() < 0.5).collect()
            df_only1value  = data.select(col_all.drop_nulls().value_counts().count() == 1 ).collect()

            # Categorical features - Discard columns that contain a large number of different values (more than discarding_threshold % of values are diffent from each other)
            lf_cat = data.select(col_cat.value_counts().count()>pl.len()*self.discarding_threshold).collect()
        
            for col in df_50perc_null.columns: 
                if df_50perc_null.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "More than 50% of the values is null or empty"
                elif df_only1value.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "All vales are equal"
                    self.single_value_columns[col] = data.select(pl.col(col).first()).collect().item()
                elif col in lf_cat.columns and lf_cat.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "More than discarding_threshold % of values are different from each other"

        # Update the numerical_features, categorical_features and temporal_features lists removing the discarded columns
        self.boolean_features     = tuple(set(self.boolean_features)     - set(self.discarded_features))
        self.numerical_features   = tuple(set(self.numerical_features)   - set(self.discarded_features))
        self.categorical_features = tuple(set(self.categorical_features) - set(self.discarded_features))
        self.temporal_features    = tuple(set(self.temporal_features)    - set(self.discarded_features))
    
    def collect(self, 
                data: pl.LazyFrame | pd.DataFrame, 
                scaling: str = "normalize", 
                num_fill_null : FillNullStrategy = "mean",
                n_bins: int = 0
    ) -> pl.DataFrame | pd.DataFrame:
        """
        The preliminary operations deal with distinguishing numerical columns from categorical columns and discarding the columns
        that do not carry significant information. Then the processing steps are defined and carried out for each type of column.

        Below are listed all the possible values for the arguments of the method .collect()

        'data':
            The dataset passed to the Preprocessor can be a Polars LazyFrame or a Pandas DataFrame.

        'scaling': (default="normalize")
            Specifies the scaling operation to perform on numerical features.
            - "normalize"   : applies normalization to numerical features
            - "standardize" : applies standardization to numerical features

        'num_fill_null': (default = "mean")
            Specifies the value to fill null values with or the strategy for filling null values in numerical features.
            - value         : fills null values with the specified value  
            - "mean"        : fills null values with the average of the column
            - "interpolate" : fills null values by interpolation
            - "forward"     : fills null values with the previous non-null value in the column
            - "backward"    : fills null values with the following non-null value in the column
            - "min"         : fills null values with the minimum value of the column
            - "max"         : fills null values with the maximum value of the column
            - "zero"        : fills null values with zeros
            - "one"         : fills null values with ones

        'n_bins': (default = 0)
            Integer number that determines the number of bins into which numerical features are discretized. When set to 0, the preprocessing step defaults to the scaling method specified in the 'scaling' atgument instead of discretization.
            Note that if n_bins is different than 0, discretization will take place instead of scaling, regardless of whether the 'scaling' argument is specified.
        """
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
        if isinstance(num_fill_null, str):
            if num_fill_null == "interpolate":
                data = data.with_columns(col_num).interpolate()
            else:
                data = data.with_columns(col_num.fill_null(strategy=num_fill_null))
        else:
            data = data.with_columns(col_num.fill_null(num_fill_null))

        if n_bins > 0:
            # KBinsDiscretizer applied to numerical features
            labels=list(map(str, list(range(0, n_bins))))
            data = data.with_columns(col_num.qcut(n_bins, labels=labels))
        else:
            match scaling:
                case "normalize":
                    # Normalization of numerical features
                    data = data.with_columns((col_num - col_num.min()) / (col_num.max() - col_num.min()))
                case "standardize":
                    # Standardization of numerical features
                    data = data.with_columns((col_num - col_num.mean()) /  col_num.std())    

    # Categorical features processing
        # Fill Null values with the most frequent value
        col_cat = set(self.categorical_features) - set(self.excluded_col)
        for col in col_cat:
            freq_val = data.select(pl.col(col).drop_nulls().mode().first()).collect().item()
            data = data.with_columns(pl.col(col).fill_null(freq_val))
        
        # OneHotEncoding and collect the pl.LazyFrame into a pl.Dataframe
        # The Dataframe is sorted according to "time" column if present
        col_str = cs.string()-cs.by_name(self.excluded_col) 
        if self.time:
            df = data.sort(self.time).collect().to_dummies(col_str)
        else:
            df = data.collect().to_dummies(col_str)

        if self.data_was_pd:
            df = df.to_pandas()

        return df

    def extract_ts_features(self,
                            data: pl.LazyFrame | pd.DataFrame,
                            y: pl.Series | pd.Series,
                            time: str = None,
                            column_id:str = None,
                            ) -> pd.DataFrame:
        """
        Extract relevant time-series features. 
        Input arguments:
            data: pl.LazyFrame | pd.DataFrame   (Dataframe)
            y: pl.Series | pd.DataFrame         (Label series)
            time: str                           (Name of the time column)
            column_id = None                    (Name of the id column if present)
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
    
    def get_discarded_features_reason(self) -> None:
        """
        Print the discarded columns and the reason why they were discarded
        """
        try:
            getattr(self,"discarded_features")
            print('1. The following columns are discarded because all vales are equal (the value is reported next to the feature):')
            for key,value in self.discarded_features.items():
                if value =='All vales are equal':
                    print("    ", key, ":", self.single_value_columns[key])
            print('\n\n2. The following columns are discarded because more than 50% of the values is null or empty:')
            for key,value in self.discarded_features.items():
                if value =='More than 50% of the values is null or empty':
                    print("    ", key)
            print('\n\n3. The following columns are discarded because more than discarding_threshold % of values are different from each other:')
            for key,value in self.discarded_features.items():
                if value =='More than discarding_threshold % of values are different from each other':
                    print("    ", key)
        except AttributeError:
            print("AttributeError\nThe preprocessor has no attribute 'discarded_features'.\nMake sure you called the method 'preprocessor.collect(your_LazyFrame)' or you set the argument 'get_discarded_info=True' when initializing the Prprocessor to assess the discarded features.")

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
            preprocessor.collect(your_dataframe, num_fill_null_strat="mean", n_bins : int  = 0)\n
        Below are listed all the possible values for the arguments of the method .collect():\n
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