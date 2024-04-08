import sys

import pandas as pd
import polars as pl
import polars.selectors as cs

from typing import List, Dict, Tuple, Union, TypeAlias, Literal

class Preprocessor:
    numerical_features   : Tuple[str]
    categorical_features : Tuple[str]
    temporal_features    : Tuple[str]
    discarded_features   : Union[List[str], Dict[str, str]]
    single_value_columns : Dict[str, str]
    
    FillNullStrategy    : TypeAlias = Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    Scaling             : TypeAlias = Literal["normalize", "standardize"]

    def __init__(
        self, data: pl.LazyFrame | pd.DataFrame, 
        discarding_threshold: float = 0.9, 
        get_discarded_info = False
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
            When set to 'True', the preprocessor will feature the methods preprocessor.get_discarded_features_reason, which provides information on which columns were discarded and the reason why, and preprocessor.get_single_valued_columns, which provides the values of the single-valued discarded columns.
            Note that setting get_discarded_info=True will considerably slow down the processing operation!
            The list of discarded columns will be available even if get_discarded_info=False, so consider setting this flag to True only if you need to know why a column was discarded or, if it contained just one value, what that value was.
        """

        # Transform data from Pandas DataFrame to Polars LazyFrame
        if isinstance(data, pd.DataFrame):
            self.data_was_pd = True
            data = pl.from_pandas(data).lazy()
        else:
            self.data_was_pd = False

        self.discarding_threshold   = discarding_threshold
        self.get_discarded_info     = get_discarded_info

        self._infer_feature_types(data)
        self._feature_selection(data)

    def _infer_feature_types(self, data: pl.LazyFrame) -> None:
        """
        Infer the type of each feature in the LazyFrame. The type is either numerical or categorical. 
        DateTime and Boolean features are converted to numerical by default.
        """
        # Transform boolean into int
        data = data.with_columns(pl.col(pl.Boolean).cast(pl.UInt8))

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
        Perform a selection of the most useful columns for a given DataFrame, ignorig the other features. The selection is
        performed in two steps:
        1. The columns with more than 50% of missing values are discarded.
        2. The columns containing only one value or, conversely, a large number of different values are discarded. In the latter
        case the default threshold is equal to 90%, ie if more than 90% of the instances have different values then the entire
        column is discarded.
        """
        # Replace empty strings ("") with None value
        data = data.with_columns(cs.string().replace("",None)) 

        if self.get_discarded_info == False:
            self.discarded_features = []
            # All feature types - Discard columns if more than 50% of values is null or all values are equal (only one value in the column)
            lf_ = data.select(pl.any_horizontal(pl.all().count()/pl.len() < 0.5, 
                                                pl.all().drop_nulls().value_counts().count() == 1, 
                                               )).collect()
            
            # Categorical features - Discard columns that contain a large number of different values (more than discarding_threshold % of values are diffent from each other)
            lf_cat = data.select(pl.col(self.categorical_features).value_counts().count()>pl.len()*self.discarding_threshold).collect()

            for col in lf_.columns: 
                if lf_.select(pl.col(col)).item() == True:
                    self.discarded_features.append(col)
                elif col in lf_cat.columns and lf_cat.select(pl.col(col)).item() == True:
                    self.discarded_features.append(col)
        else:
            self.discarded_features    = dict()
            self.single_value_columns = dict()

            # All feature types - Discard columns if more than 50% of values is null or all values are equal (only one value in the column)
            df_50perc_null  = data.select(pl.all().count()/pl.len() < 0.5).collect()
            df_only1value  = data.select(pl.all().drop_nulls().value_counts().count() == 1 ).collect()

            # Categorical features - Discard columns that contain a large number of different values (more than discarding_threshold % of values are diffent from each other)
            lf_cat = data.select(pl.col(self.categorical_features).value_counts().count()>pl.len()*self.discarding_threshold).collect()
        
            for col in df_50perc_null.columns: 
                if df_50perc_null.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "More than 50% of the values is null or empty"
                elif df_only1value.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "All vales are equal"
                    self.single_value_columns[col] = data.select(pl.col(col).first()).collect().item()
                elif col in lf_cat.columns and lf_cat.select(pl.col(col)).item() == True:
                    self.discarded_features[col] = "More than discarding_threshold % of values are different from each other"

        # Update the numerical_features, categorical_features and temporal_features lists removing the discarded columns
        self.numerical_features   = tuple(set(self.numerical_features)   - set(self.discarded_features))
        self.categorical_features = tuple(set(self.categorical_features) - set(self.discarded_features))
        self.temporal_features    = tuple(set(self.temporal_features)    - set(self.discarded_features))

        return data
    
    def collect(self, 
                data: pl.LazyFrame | pd.DataFrame, 
                scaling = "normalize", 
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
        """
        if isinstance(data, pd.DataFrame) and self.data_was_pd == True:
            data = pl.from_pandas(data).lazy()
        elif isinstance(data, pl.LazyFrame) and self.data_was_pd == False:
            pass
        else:
            sys.exit('ErrorType\nThe datatype provided does not not match with the datatype of the dataset provided when the Preprocessor was initialized.')

        # Drop discarded columns, previously defined in _feature_selection()
        if isinstance(self.discarded_features, dict):
            data = data.drop(self.discarded_features.keys())
        else:
            data = data.drop(self.discarded_features)

    # Temporal features processing
        # Fill Null values by interpolation and reorder columns such that temporal ones are positioned at the beginning of the LazyFrame    
        data = data.select(cs.temporal().interpolate(), cs.all()-cs.temporal())

    # Numerical features processing
        # Fill Null values with the selcted strategy or value (default: "mean")
        if isinstance(num_fill_null, str):
            data = data.with_columns(cs.numeric().fill_null(strategy=num_fill_null))
        else:
            data = data.with_columns(cs.numeric().fill_null(num_fill_null))

        if n_bins > 0:
            # KBinsDiscretizer applied to numerical features
            labels=list(map(str, list(range(0, n_bins))))
            data = data.with_columns(cs.numeric().qcut(n_bins, labels=labels))
        else:
            match scaling:
                case "normalize":
                    # Normalization of numerical features
                    data = data.with_columns((cs.numeric() - cs.numeric().min()) / (cs.numeric().max() - cs.numeric().min()))
                case "standardize":
                    # Standardization of numerical features
                    data = data.with_columns((cs.numeric() - cs.numeric().mean()) /  cs.numeric().std())    

    # Categorical features processing
        # Fill Null values with the most frequent value
        for col in self.categorical_features:
            freq_val = data.select(pl.col(col).drop_nulls().mode().first()).collect().item()
            data = data.with_columns(pl.col(col).fill_null(freq_val))
        
        # OneHotEncoding and collect the pl.LazyFrame into a pl.Dataframe
        df = data.collect().to_dummies(cs.string())

        if self.data_was_pd:
            df = df.to_pandas()

        return df

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
            print('1. The following columns are discarded because all vales are equal:')
            for key,value in self.discarded_features.items():
                if value =='All vales are equal':
                    print("    ", key)
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

    def get_single_valued_columns(self) -> None:
        """
        Print the single-values columns and their value
        """
        try:
            getattr(self,"single_value_columns")
            if self.single_value_columns:
                print('Discarded single-valued columns and the value contained:')
                for key, value in self.single_value_columns.items():
                    print("    ", key, ":", value)
            else:
                print("No single-valued columns were discardedd.")
            
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
            When set to 'True', the preprocessor will feature the methods preprocessor.get_discarded_features_reason, which provides information on which columns were discarded and the reason why, and preprocessor.get_single_valued_columns, which provides the values of the single-valued discarded columns.
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