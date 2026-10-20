# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`clearbox-preprocessor` (PyPI) is a Polars-based tabular preprocessor for ML datasets. It fills nulls, scales or discretizes numerical columns, one-hot encodes categoricals, converts datetimes to numbers, and inverts all of it. Its main uses are:
- encoding data before synthesis with a generative model (VAE, diffusion), then decoding the samples with `inverse_transform`;
- consistent train/test encoding in ML pipelines (TSTR evaluation, privacy attacks).

The single public class is `clearbox_preprocessor.Preprocessor`.

## Commands

```bash
pip install -r requirements.txt          # runtime deps (pyarrow is needed for pandas <-> polars)
pip install -e ".[dev]"                  # editable install + pytest
pip install -r docs/requirements.txt     # Sphinx deps, docs only

python -m pytest                         # all tests
python -m pytest tests/test_regressions.py::test_nulls_roundtrip   # single test

make html                                # Sphinx docs -> docs/build/html (Makefile wraps sphinx-build)
```

tsfresh is imported only inside `extract_ts_features`, because importing it takes about 2 s. tsfresh 0.20.3 also needs `pkg_resources` (setuptools < 81), and its numba dependency needs NumPy ≤ 1.26. The tsfresh test is skipped when tsfresh can't be imported.

## Release

Pushing to `main` triggers `.github/workflows/pypi_publishing.yaml`. It builds and uploads to PyPI **only if the version in `setup.py` is not already on PyPI**, so a release means bumping `version=` in `setup.py`. `docs/source/conf.py` has its own `release`; bump it together with `setup.py`. Read the Docs builds from `.readthedocs.yaml`: it installs `docs/requirements.txt` plus the package. `requirements.txt` feeds `install_requires`, so keep it runtime-only.

## Architecture

Fitting happens entirely in `Preprocessor.__init__`. `transform` and `inverse_transform` only apply what was learned at fit time: fill values, scaling parameters, bin edges, quantiles and one-hot vocabularies. The train/test consistency that TSTR and privacy attacks rely on depends on this, so never compute statistics from the data being transformed.

Fit order in `__init__`:
1. Input is converted to an eager `pl.DataFrame`. `data_was_pd` makes `transform`/`inverse_transform` return pandas, keeping the input's pandas index.
2. `_infer_feature_types` walks the schema **in column order**, and all feature tuples keep that order. This is what makes the output layout deterministic, so don't use `set()` to order features.
   - `Boolean` → boolean
   - `Date`/`Datetime`/`Time` → datetime
   - any numeric dtype → numerical
   - `String`/`Categorical`/`Enum` → datetime if `infer_string_datetime_format` matches a sample, otherwise categorical
   - anything else passes through untouched
3. `_feature_selection` does the following:
   - Drops columns that are mostly missing, single-valued, dominated by one categorical label (≥98%), or left with a single category after rare-label aggregation. Single values go into `single_value_columns`, and `inverse_transform` restores them.
   - Builds the categorical vocabularies. Nulls and blank strings become the `"None"` label, and rare labels map to `"other"`. The returned `discarded = (dropped_cols, {col: rare_labels})` keeps its legacy shape.
4. Transformers in `utils/`:
   - `DatetimeTransformer` only encodes and decodes: datetimes become float seconds since the epoch, times become seconds since midnight. String columns are re-formatted with their fitted format.
   - `NumericalTransformer` fills and scales **both numerical and encoded datetime columns**. `DATETIME_SCALING` maps quantile/kbins to normalize for datetimes.
   - `CategoricalTransformer` one-hot encodes against the fixed vocabulary. Unseen labels become all zeros, or raise with `unseen_labels="error"`.

   A transformer attribute exists only if its feature group is non-empty, so code checks `hasattr(self, "numerical_transformer")`.
5. `output_columns` holds the non-categorical columns in schema order (excluded columns included), then the one-hot blocks. `feature_columns` is the same without excluded and passthrough columns. `get_features_sizes()` describes the `feature_columns` layout for generative models.

`inverse_transform` must accept **approximate** inputs (samples from a generative model, or a NumPy array laid out like `feature_columns`/`output_columns`):
- categoricals decode by row-wise argmax over their one-hot block;
- booleans are thresholded at 0.5;
- integer columns are rounded and clipped to their dtype before the final cast back to `self.schema`.

### Null encoding with `num_fill_null="none"` (the default)

Nulls are filled with `min - 0.01 * range`. On inverse, values below `min - 0.005 * range` become null again, but only for columns that had nulls at fit time. `NULL_SENTINEL_OFFSET` is shared by both directions, so change them together.

### Performance notes

- Build transformations as one list of Polars expressions per stage. Avoid per-column `with_columns` loops and repeated `collect()` calls.
- Read `df.schema` and `df.columns` once and outside loops. Each access rebuilds the object, which is very slow on wide frames.
- Polars does not reliably reuse a repeated sub-expression within one `with_columns`. The categorical decode therefore materializes each block's `max_horizontal` as a temporary column first.
- Quantile scaling is fitted with sklearn's `QuantileTransformer`. The forward and inverse mappings are re-implemented in `numerical_transformer.py` (`_quantile_to_normal`/`_normal_to_quantile`) and run column-parallel in threads. `test_quantile_matches_sklearn` pins the equivalence.

## Docs

Docstrings in `preprocessor.py` are the API reference (Sphinx autodoc + numpydoc, NumPy docstring style). `README.md` has the user-facing parameter docs and usage examples. Keep both in sync with the constructor arguments.
