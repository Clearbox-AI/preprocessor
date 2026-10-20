[![Documentation Status](https://readthedocs.org/projects/clearbox-preprocessor/badge/?version=latest)](https://clearbox-preprocessor.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/clearbox-preprocessor.svg)](https://badge.fury.io/py/clearbox-preprocessor)
[![Downloads](https://pepy.tech/badge/clearbox-preprocessor)](https://pepy.tech/project/clearbox-preprocessor)
[![GitHub stars](https://img.shields.io/github/stars/Clearbox-AI/preprocessor?style=social)](https://github.com/Clearbox-AI/preprocessor)

# Clearbox AI Preprocessor

This repository contains the continuation of the work presented in our series of blogposts *"The whys and hows of data preparation"* ([part 1](https://www.clearbox.ai/blog/2022-01-25-the-whys-and-hows-of-data-preparation), [part 2](https://www.clearbox.ai/blog/2022-02-22-the-whys-and-hows-of-data-preparation-part-2), [part 3](https://www.clearbox.ai/blog/2022-07-05-the-whys-and-hows-of-data-preparation-part-3)). 

The new version of Preprocessor exploits Polars library's features to achieve blazing fast tabular data manipulation.

It is possible to input the Preprocessor a `Pandas.DataFrame`, a `Polars.DataFrame` or a `Polars.LazyFrame`.

## Installation

You can install the preprocessor by running the following command:

```shell
$ pip install clearbox-preprocessor
```

## Preprocessing customization
A bunch of options are available to customize the preprocessing. 

The `Preprocessor` class features the following input arguments, besides the input dataset:
   - `cat_labels_threshold: float (default = 0.02)`

        Float number between 0 and 1. In categorical features, labels appearing in less than `cat_labels_threshold * 100 %` of the rows are aggregated into a single `"other"` category.
        For example, if `cat_labels_threshold=0.02`, a label that appears in less than 2% of the rows is converted to `"other"`. Missing values are never aggregated.
        Keep in mind that rare labels are at risk of being re-identified in case of privacy attack.

   - `missing_values_threshold: float (default = 0.999)`

        Float number between 0 and 1. Columns with a share of missing values greater than this threshold are discarded.

   - `excluded_col: (default = None)`
      
        List containing the names of the columns to be excluded from processing. These columns will be returned in the final dataframe without being manipulated. 

   - `time_id: (default = None)`
  
        String name of the time column of time series, used by `extract_ts_features()`.
      
   - `scaling: (default="none")`
    
        Specifies the scaling operation to perform on numerical features.
        - "none"        : no scaling is applied
        - "normalize"   : applies normalization to numerical features
        - "standardize" : applies standardization to numerical features
        - "quantile"    : transforms numerical features to a normal distribution using quantiles information
        - "kbins"       : discretizes numerical features into `n_bins` quantile-based bins

        Datetime features are converted to seconds since the Unix epoch and scaled with the same method (they are normalized when `scaling` is "quantile" or "kbins").
    
   - `num_fill_null: (default = "none")`
    
        Specifies the value to fill null values with or the strategy for filling null values in numerical and datetime features.
        - "none"        : fills null values with a value just below the minimum of the column, which `inverse_transform()` turns back into nulls
        - value         : fills null values with the specified value  
        - "mean"        : fills null values with the average of the column
        - "interpolate" : fills null values by linear interpolation
        - "forward"     : fills null values with the previous non-null value in the column
        - "backward"    : fills null values with the following non-null value in the column
        - "min"         : fills null values with the minimum value of the column
        - "max"         : fills null values with the maximum value of the column
        - "zero"        : fills null values with zeros
        - "one"         : fills null values with ones

   - `n_bins: (default = 0)`
  
        Integer number of bins into which numerical features are discretized when `scaling="kbins"`.

   - `unseen_labels: (default = "ignore")`

        What to do with categorical labels that were not seen when the Preprocessor was initialized: "ignore" encodes them as all zeros, "error" raises an error.

   - `ml_task: (default = None)` and `target_column: (default = None)`

        The Machine Learning task ("classification" or "regression") and its target column. The target is label-encoded for classification and normalized for regression.

All the statistics used by the preprocessing (fill values, scaling parameters, bins and categories) are computed on the dataset the Preprocessor is initialized with.

### Feature selection
When it is initialized, the Preprocessor discards:
- columns with more missing values than `missing_values_threshold`,
- columns containing a single value,
- categorical columns in which a single value appears in more than 98% of the rows,
- categorical columns left with a single category after aggregating the rare labels.

The discarded columns are listed in `preprocessor.discarded_features`, along with the reason in `preprocessor.discarded_reasons`. The value of the single-valued columns is stored in `preprocessor.single_value_columns`, and `inverse_transform()` restores these columns.

### Timeseries
The Preprocessor also features a timeseries manipulation and feature extraction method called `extract_ts_features()`.

This method takes as input:
- the preprocessed dataframe
- the target vector in the form of a `Pandas.Series` or a `Polars.Series`
- the name of the time column
- the name of the id column to group by
  
It returns the most relevant features selected among a wide range of features.

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
```
┌──────┬─────┬─────────────────────┬───────┬──────┬────────┬──────┐
│ cha  ┆ int ┆ dat                 ┆ boo   ┆ equ  ┆ flo    ┆ str  │
│ ---  ┆ --- ┆ ---                 ┆ ---   ┆ ---  ┆ ---    ┆ ---  │
│ str  ┆ i64 ┆ datetime[μs]        ┆ bool  ┆ str  ┆ f64    ┆ str  │
╞══════╪═════╪═════════════════════╪═══════╪══════╪════════╪══════╡
│ x    ┆ 123 ┆ 2023-01-05 00:34:12 ┆ true  ┆ a    ┆ 43.034 ┆ asd  │
│ null ┆ 124 ┆ 2023-02-03 04:31:45 ┆ false ┆ a    ┆ 343.1  ┆ fgh  │
│ z    ┆ 223 ┆ 2023-02-04 04:31:45 ┆ null  ┆ a    ┆ 224.23 ┆ fgh  │
│ w    ┆ 431 ┆ null                ┆ true  ┆ a    ┆ 75.3   ┆      │
│ x    ┆ 435 ┆ 2023-05-12 21:41:58 ┆ false ┆ null ┆ null   ┆ null │
│ k    ┆ 432 ┆ 2023-06-01 17:52:22 ┆ false ┆ a    ┆ 83.2   ┆ cvb  │
└──────┴─────┴─────────────────────┴───────┴──────┴────────┴──────┘
```

At this point, you can initialize the Preprocessor by passing the LazyFrame or DataFrame created to it and then calling the `transform()` method to materialize the processed dataframe.

Note that if no argument is specified beyond the dataframe *q*, the default settings are employed for preprocessing:

```python
preprocessor = Preprocessor(q)
df = preprocessor.transform(q)
df
```
```
┌───────┬──────────┬──────┬──────────┬──────────┬───────┬───────┬───────┬───────┬──────────┬───────┬──────────┬─────────┬─────────┬─────────┐
│ int   ┆ dat      ┆ boo  ┆ flo      ┆ cha_None ┆ cha_k ┆ cha_w ┆ cha_x ┆ cha_z ┆ equ_None ┆ equ_a ┆ str_None ┆ str_asd ┆ str_cvb ┆ str_fgh │
│ ---   ┆ ---      ┆ ---  ┆ ---      ┆ ---      ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---      ┆ ---   ┆ ---      ┆ ---     ┆ ---     ┆ ---     │
│ f64   ┆ f64      ┆ u8   ┆ f64      ┆ u8       ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8       ┆ u8    ┆ u8       ┆ u8      ┆ u8      ┆ u8      │
╞═══════╪══════════╪══════╪══════════╪══════════╪═══════╪═══════╪═══════╪═══════╪══════════╪═══════╪══════════╪═════════╪═════════╪═════════╡
│ 123.0 ┆ 1.6729e9 ┆ 1    ┆ 43.034   ┆ 0        ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 1       ┆ 0       ┆ 0       │
│ 124.0 ┆ 1.6754e9 ┆ 0    ┆ 343.1    ┆ 1        ┆ 0     ┆ 0     ┆ 0     ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 0       ┆ 0       ┆ 1       │
│ 223.0 ┆ 1.6755e9 ┆ null ┆ 224.23   ┆ 0        ┆ 0     ┆ 0     ┆ 0     ┆ 1     ┆ 0        ┆ 1     ┆ 0        ┆ 0       ┆ 0       ┆ 1       │
│ 431.0 ┆ 1.6728e9 ┆ 1    ┆ 75.3     ┆ 0        ┆ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 0        ┆ 1     ┆ 1        ┆ 0       ┆ 0       ┆ 0       │
│ 435.0 ┆ 1.6839e9 ┆ 0    ┆ 40.03334 ┆ 0        ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 1        ┆ 0     ┆ 1        ┆ 0       ┆ 0       ┆ 0       │
│ 432.0 ┆ 1.6856e9 ┆ 0    ┆ 83.2     ┆ 0        ┆ 1     ┆ 0     ┆ 0     ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 0       ┆ 1       ┆ 0       │
└───────┴──────────┴──────┴──────────┴──────────┴───────┴───────┴───────┴───────┴──────────┴───────┴──────────┴─────────┴─────────┴─────────┘
```
The missing values of `flo` and `dat` are encoded just below the minimum of the column, missing and empty labels get their own `<column>_None` column, and the datetime column is converted to seconds since the Unix epoch.


### Customization example
In the following example, when the Preprocessor is initialized:
1. The rare labels threshold is raised from 2% to 20% (labels appearing in less than 20% of the rows are aggregated into "other").
2. The column "boo" is excluded from the preprocessing and is preserved unchanged.
3. The scaling method of the numerical features chosen is standardization
4. The fill null strategy for numerical features is "forward".

```python
preprocessor    = Preprocessor(q, 
                               cat_labels_threshold = 0.2, 
                               excluded_col = ["boo"], 
                               scaling = "standardize", 
                               num_fill_null = "forward"
                            )
df = preprocessor.transform(q)
df
```
```
┌───────────┬───────────┬───────┬───────────┬──────────┬───────────┬───────┬──────────┬───────┬──────────┬─────────┬───────────┐
│ int       ┆ dat       ┆ boo   ┆ flo       ┆ cha_None ┆ cha_other ┆ cha_x ┆ equ_None ┆ equ_a ┆ str_None ┆ str_fgh ┆ str_other │
│ ---       ┆ ---       ┆ ---   ┆ ---       ┆ ---      ┆ ---       ┆ ---   ┆ ---      ┆ ---   ┆ ---      ┆ ---     ┆ ---       │
│ f64       ┆ f64       ┆ bool  ┆ f64       ┆ u8       ┆ u8        ┆ u8    ┆ u8       ┆ u8    ┆ u8       ┆ u8      ┆ u8        │
╞═══════════╪═══════════╪═══════╪═══════════╪══════════╪═══════════╪═══════╪══════════╪═══════╪══════════╪═════════╪═══════════╡
│ -1.104093 ┆ -1.012748 ┆ true  ┆ -0.87403  ┆ 0        ┆ 0         ┆ 1     ┆ 0        ┆ 1     ┆ 0        ┆ 0       ┆ 1         │
│ -1.097662 ┆ -0.571811 ┆ false ┆ 1.494306  ┆ 1        ┆ 0         ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 1       ┆ 0         │
│ -0.460932 ┆ -0.556692 ┆ null  ┆ 0.556099  ┆ 0        ┆ 1         ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 1       ┆ 0         │
│ 0.876843  ┆ -0.556692 ┆ true  ┆ -0.619364 ┆ 0        ┆ 1         ┆ 0     ┆ 0        ┆ 1     ┆ 1        ┆ 0       ┆ 0         │
│ 0.902569  ┆ 0.920643  ┆ false ┆ -0.619364 ┆ 0        ┆ 0         ┆ 1     ┆ 1        ┆ 0     ┆ 1        ┆ 0       ┆ 0         │
│ 0.883275  ┆ 1.220608  ┆ false ┆ -0.557011 ┆ 0        ┆ 1         ┆ 0     ┆ 0        ┆ 1     ┆ 0        ┆ 0       ┆ 1         │
└───────────┴───────────┴───────┴───────────┴──────────┴───────────┴───────┴──────────┴───────┴──────────┴─────────┴───────────┘
```

It is possible to inverse transform the processed dataframe with the method `preprocessor.inverse_transform()`.

```python
preprocessor = Preprocessor(q)
df = preprocessor.transform(q)
inverse_df = preprocessor.inverse_transform(df)
```

### Preprocessing for generative models
`inverse_transform()` does not need an exact output of `transform()`, so it can decode the samples of a generative model (e.g. a VAE or a diffusion model) trained on the transformed data:
- categorical features are decoded to the category with the highest score in their one-hot columns, so the columns can hold probabilities or logits,
- boolean features are thresholded at 0.5 and integer features are rounded,
- datetime features are converted back to their original type, or string format,
- with `num_fill_null="none"`, values below the minimum of a column that had missing values are decoded as missing values.

The input can also be a NumPy array, with the columns of `preprocessor.feature_columns` (or `preprocessor.output_columns`, which also includes the excluded columns). `preprocessor.get_features_sizes()` returns the number of ordinal (numerical, datetime and boolean) columns, which come first, and the number of one-hot columns of each categorical feature.

```python
preprocessor = Preprocessor(real_data, scaling="quantile")
X = preprocessor.transform(real_data).select(preprocessor.feature_columns).to_numpy()

model.fit(X)                                         # your generative model
synthetic_data = preprocessor.inverse_transform(model.sample(1000))
```

### Preprocessing in Machine Learning pipelines
Initialize the Preprocessor on the training data and use it to transform any other dataset (test set, synthetic data, attack datasets). Every transformed dataset gets the same columns in the same order, unseen labels are encoded as all zeros (or raise an error with `unseen_labels="error"`) and the rows keep their order.

```python
preprocessor = Preprocessor(synthetic_train, ml_task="classification", target_column="income")
X_train = preprocessor.transform(synthetic_train)
X_test  = preprocessor.transform(real_test)          # train on synthetic, test on real
```

The Preprocessor can be pickled to reuse it later.
