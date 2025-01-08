[![Documentation Status](https://readthedocs.org/projects/clearbox-preprocessor/badge/?version=latest)](https://clearbox-sure.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/clearbox-preprocessor.svg)](https://badge.fury.io/py/clearbox-preprocessor)
[![Downloads](https://pepy.tech/badge/clearbox-preprocessor)](https://pepy.tech/project/clearbox-preprocessor)
[![GitHub stars](https://img.shields.io/github/stars/Clearbox-AI/preprocessor?style=social)](https://github.com/Clearbox-AI/preprocessor)

# Clearbox AI Preprocessor

This repository contains the continuation of the work presented in our series of blogposts *"The whys and hows of data preparation"* ([part 1](https://www.clearbox.ai/blog/2022-01-25-the-whys-and-hows-of-data-preparation), [part 2](https://www.clearbox.ai/blog/2022-02-22-the-whys-and-hows-of-data-preparation-part-2), [part 3](https://www.clearbox.ai/blog/2022-07-05-the-whys-and-hows-of-data-preparation-part-3)). 

The new version of Preprocessor exploits Polars library's features to achieve blazing fast tabular data manipulation.

It is possible to input the Preprocessor a `Pandas.DataFrame` or a `Polars.LazyFrame`.


## Preprocessing customization
A bunch of options are available to customize the preprocessing. 

The `Preprocessor` class features the following input arguments, besides the input dataset:
   - `discarding_threshold: float (default = 0.9)`
      
        Float number between 0 and 1 to set the threshold for discarding categorical features. 
        If more than discarding_threshold * 100 % of values in a categorical feature are different from each other, then the column is discarded. 
        For example, if discarding_threshold=0.9, a column will be discarded if more than 90% of its values are unique.
    
   - `get_discarded_info: bool (defatult = False)`
        
        When set to 'True', the preprocessor will feature the methods preprocessor.get_discarded_features_reason, which provides information on which columns were discarded and the reason why, and preprocessor.get_single_valued_columns, which provides the values of the single-valued discarded columns.
        Note that setting get_discarded_info=True will considerably slow down the processing operation!
        The list of discarded columns will be available even if get_discarded_info=False, so consider setting this flag to True only if you need to know why a column was discarded or, if it contained just one value, what that value was.
      
   - `excluded_col: (default = [])`
      
        List containing the names of the columns to be excluded from processing. These columns will be returned in the final dataframe withouth being manipulated. 
   - `time: (default = None)`
  
        String name of the time column by which to sort the dataframe in case of time series.
      
   - `scaling: (default="normalize")`
    
        Specifies the scaling operation to perform on numerical features.
        - "normalize"   : applies normalization to numerical features
        - "standardize" : applies standardization to numerical features
    
   - `num_fill_null: (default = "mean")`
    
        Specifies the value to fill null values with or the strategy for filling null values in numerical features.
        - value      : fills null values with the specified value  
        - "mean"     : fills null values with the average of the column
        - "forward"  : fills null values with the previous non-null value in the column
        - "backward" : fills null values with the following non-null value in the column
        - "min"      : fills null values with the minimum value of the column
        - "max"      : fills null values with the maximum value of the column
        - "zero"     : fills null values with zeros
        - "one"      : fills null values with ones

   - `n_bins: (default = 0)`
  
        Integer number that determines the number of bins into which numerical features are discretized. When set to 0, the preprocessing step defaults to the scaling method specified in the 'scaling' atgument instead of discretization.
      
        Note that if n_bins is different than 0, discretization will take place instead of scaling, regardless of whether the 'scaling' argument is specified.

### Timeseries
The Prperocessor also features a timeseries manipulation and feature extraction method called `extract_ts_features()`.

This method takes as input:
- the preprocessed dataframe
- the target vector in the form of a `Pandas.Series` or a `Polars.Series`
- the name of the time column
- the name of the id column to group by
  
It returns the most relevant features selected among a wide range of features.


## Installation

You can install the preprocessor by running the following command:

```shell
$ pip install clearbox-preprocessor
```

## Usage
You can start using the Preprocessor by importing it and creating a `Pandas.DataFrame` or a `Polars.LazyFrame`:

```python
import polars as pl
from clearbox_preprocessor import Preprocessor

q = pl.LazyFrame(
    {
        "cha": ["x", None, "z", "w", "x", "k"],
        "int": [123, 124, 223, 431, 435, 432],
        "dat": ["2023-1-5T00:34:12.000Z", "2023-2-3T04:31:45.000Z", "2023-2-4T04:31:45.000Z", None, "2023-5-12T21:41:58.000Z", "2023-6-1T17:52:22.000Z"],
        "boo": [True, False, None, True, False, False],
        "equ": ["a", "a", "a", "a", None, "a"],
        "flo": [43.034, 343.1, 224.23, 75.3, None, 83.2],
        "str": ["asd", "fgh", "fgh", "", None, "cvb"]
    }
).with_columns(pl.col('dat').str.to_datetime("%Y-%m-%dT%H:%M:%S.000Z"))

q.collect()
```
<img src="https://github.com/Clearbox-AI/preprocessor/assets/152599516/c2e1878d-6af3-4157-8a7d-61fdccfde270" alt="image" width="40%" height="auto">

At this point, you can initialize the Preprocessor by passing the LazyFrame or DataFrame created to it and then calling the `transform()` method to materialize the processed dataframe.

Note that if no argument is specified beyond the dataframe *q*, the default settings are employed for preprocessing:

```python
preprocessor = Preprocessor(q)
df = preprocessor.transform(q)
df
```
<img src="https://github.com/Clearbox-AI/preprocessor/assets/152599516/7cd5b6f6-26f9-4af9-8250-751f43cac7d5" alt="image" width="70%" height="auto">


### Customization example
In the following example, when the Preprocessor is initialized:
1. The discarding threshold is lowered from 90% to 80% (a column will be discarded if more than 80% of its values are unique).
2. The discarding featrues informations are stored in the `preprocessor` instance.
3. The column "boo" is excluded from the preprocessing and is preserved unchanged.
4. The scaling method of the numerical features chosen is standardization
5. The fill null strategy for numerical features is "forward".

```python
preprocessor    = Preprocessor(q, 
                               get_discarded_info=True, 
                               discarding_threshold = 0.8, 
                               excluded_col = ["boo"], 
                               scaling = "standardize", 
                               num_fill_null = "forward"
                            )
df = preprocessor.transform(q)
df
```
<img src="https://github.com/Clearbox-AI/preprocessor/assets/152599516/ba61531d-a462-4d58-847a-47127d6050fd" alt="image" width="52%" height="auto">

If the Processor's argument `get_discarded_info` is set to `True` during initialization, it is possible to call the method `get_discarded_features_reason()` to display the discarded features. 
In the case of discarded single-valued columns, the value contained is also displayed and is available in a dictionary called `single_value_columns`, stored in the Preprocessor instance, and can be used as metadata.

```python
preprocessor.get_discarded_features_reason()
```
<img src="https://github.com/Clearbox-AI/preprocessor/assets/152599516/5c5d7a42-d52d-4d32-862d-fc3c95c31d67" alt="image" width="90%" height="auto">


## To do
- [ ] Implement unit tests
