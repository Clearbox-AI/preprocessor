# Clearbox AI Preprocessor

This repository represents the third part of our series of blogposts "The whys and hows of data preparation" ([part 1](https://www.clearbox.ai/blog/2022-01-25-the-whys-and-hows-of-data-preparation), [part 2](https://www.clearbox.ai/blog/2022-02-22-the-whys-and-hows-of-data-preparation-part-2)). The idea is to provide a practical example to the previous posts, releasing a simple implementation of a preprocessor for tabular data, usable for educational purposes.

## INSTALLATION

You can install the preprocessor by running the following command:

`pip install clearbox_preprocessor`

## USAGE

You can start using the preprocessor by importing it:

```python
from clearbox_preprocessor import Preprocessor
```
Then you just have to create a DataFrame and pass it to the preprocessor to instantiate it:

```python
import pandas as pd

df = pd.read_csv('tests/resources/uci_adult_dataset/dataset.csv')
preprocessor = Preprocessor(df)
```
At this point, you can use the preprocessor to preprocess the data. It is sufficient to call the `fit` method to fit preprocessor and the `transform` method to transform the data:

```python
preprocessor.fit()
ds = preprocessor.transform(df)
```

And it's done! Now you have your preprocessed data as a Numpy ndarray, ready to be used to train your ML model.

## TODO

- [ ] Add a transformer for the following column types:
    - [ ] Datetime
    - [ ] Boolean
- [ ] Implement unit tests
