import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import QuantileTransformer

from clearbox_preprocessor import Preprocessor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

ADULT = Path(__file__).parent / "resources" / "uci_adult_dataset" / "dataset.csv"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def adult():
    return pl.read_csv(ADULT)


def assert_frames_equal(left: pl.DataFrame, right: pl.DataFrame):
    assert left.columns == right.columns
    assert left.schema == right.schema
    for col in left.columns:
        if left[col].dtype.is_float():
            np.testing.assert_allclose(left[col].to_numpy(), right[col].to_numpy(), err_msg=f"Column {col} differs")
        else:
            assert left[col].equals(right[col]), f"Column {col} differs"


# ---------------------------------------------------------------------------
# Round trips (preprocessing for generative models and reconstruction)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scaling", ["none", "normalize", "standardize"])
def test_adult_roundtrip_is_exact(adult, scaling):
    p = Preprocessor(adult, scaling=scaling, cat_labels_threshold=0)
    assert_frames_equal(p.inverse_transform(p.transform(adult)), adult)


def test_adult_quantile_roundtrip(adult):
    p = Preprocessor(adult, scaling="quantile", cat_labels_threshold=0)
    inv = p.inverse_transform(p.transform(adult))
    assert inv.schema == adult.schema
    # The quantile mapping is lossy in sparse tails (e.g. capital_loss): match sklearn's own round trip
    numerical = adult.select(p.numerical_features).to_numpy().astype(float)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(numerical)
    expected = np.rint(qt.inverse_transform(qt.transform(numerical)))
    np.testing.assert_array_equal(inv.select(p.numerical_features).to_numpy(), expected)
    for col in p.categorical_features:
        assert inv[col].equals(adult[col])


def test_rare_labels_roundtrip_to_other(adult):
    p = Preprocessor(adult, cat_labels_threshold=0.05)
    inv = p.inverse_transform(p.transform(adult))
    for col, rare in p.discarded[1].items():
        assert set(inv.filter(adult[col].is_in(rare))[col].unique()) == {"other"}


def test_nulls_roundtrip(rng):
    n = 300
    x = rng.normal(-20, 5, n)
    x[:30] = np.nan
    c = rng.choice(list("abc"), n).astype(object)
    c[10:40] = None
    df = pd.DataFrame({"x": x, "y": rng.normal(size=n), "c": c})
    for scaling in ["none", "normalize", "standardize", "quantile"]:
        p = Preprocessor(df, scaling=scaling, num_fill_null="none")
        out = p.transform(df)
        assert not out.isna().any().any(), scaling
        inv = p.inverse_transform(out)
        pd.testing.assert_series_equal(inv["x"].isna(), df["x"].isna(), check_names=False)
        pd.testing.assert_series_equal(inv["c"].isna(), df["c"].isna(), check_names=False)
        if scaling != "quantile":
            np.testing.assert_allclose(inv["x"].dropna(), df["x"].dropna())


def test_no_spurious_nulls_without_nulls_at_fit(rng):
    # Values far below the column minimum must not be decoded as nulls if the column had no nulls
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    p = Preprocessor(df, scaling="normalize")
    out = p.transform(df)
    out["x"] = out["x"] - 10
    assert not p.inverse_transform(out)["x"].isna().any()


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%H:%M:%S", "%Y-%m-%dT%H:%M:%S%.fZ"])
def test_string_datetime_roundtrip(rng, fmt):
    n = 200
    values = pl.Series(rng.integers(1_500_000_000, 1_700_000_000, n) * 1_000_000).cast(pl.Datetime("us"))
    df = pl.DataFrame({"d": values.dt.strftime(fmt), "x": rng.normal(size=n)})
    p = Preprocessor(df, scaling="normalize")
    assert p.datetime_features == ("d",)
    assert_frames_equal(p.inverse_transform(p.transform(df)), df)


def test_native_temporal_roundtrip(rng):
    n = 200
    start = dt.datetime(2020, 1, 1)
    df = pl.DataFrame({
        "date":  pl.date_range(start.date(), start.date() + dt.timedelta(days=n - 1), eager=True),
        "ts_ns": pl.datetime_range(start, start + dt.timedelta(hours=n - 1), "1h", time_unit="ns", eager=True),
        "ts_tz": pl.datetime_range(start, start + dt.timedelta(hours=n - 1), "1h", time_zone="Europe/Rome", eager=True),
        "time":  pl.Series([dt.time(i % 24, i % 60, i % 60) for i in range(n)]),
        "x":     rng.normal(size=n),
    })
    for scaling in ["none", "normalize", "standardize", "quantile", "kbins"]:
        p = Preprocessor(df, scaling=scaling, n_bins=5)
        inv = p.inverse_transform(p.transform(df))
        assert inv.schema == df.schema
        for col in ["date", "ts_ns", "ts_tz", "time"]:
            assert inv[col].equals(df[col]), (scaling, col)


def test_generated_data_is_decoded(rng):
    # Data produced by a generative model is not an exact transform output
    n = 500
    df = pd.DataFrame({
        "age":  rng.integers(18, 90, n),
        "flag": rng.random(n) > 0.5,
        "c":    rng.choice(list("abc"), n),
        "y":    rng.normal(size=n),
    })
    p = Preprocessor(df, scaling="normalize")
    out = p.transform(df).astype(float)
    noisy = out + rng.normal(0, 0.01, out.shape)
    one_hot = p.categorical_transformer.original_encoded_columns["c"]
    noisy[one_hot] = out[one_hot] * 0.6 + 0.2  # soft scores, argmax unchanged
    inv = p.inverse_transform(noisy)
    assert (inv["c"] == df["c"]).all()
    assert (inv["flag"] == df["flag"]).all()
    assert inv["age"].dtype == df["age"].dtype
    assert (inv["age"] - df["age"]).abs().max() <= 2


def test_inverse_transform_from_numpy(rng):
    df = pl.DataFrame({"x": rng.normal(size=50), "c": rng.choice(list("ab"), 50), "keep": np.arange(50)})
    p = Preprocessor(df, excluded_col=["keep"], scaling="standardize")
    out = p.transform(df)
    assert out.columns == p.output_columns
    features = out.select(p.feature_columns).to_numpy()
    inv = p.inverse_transform(features)
    assert inv.columns == ["x", "c"]
    np.testing.assert_allclose(inv["x"].to_numpy(), df["x"].to_numpy())
    assert inv["c"].equals(df["c"])
    with pytest.raises(ValueError):
        p.inverse_transform(features[:, :1])


def test_single_value_columns_are_restored(rng):
    df = pl.DataFrame({"x": rng.normal(size=20), "const": [7] * 20, "label": ["a"] * 20})
    p = Preprocessor(df)
    assert set(p.discarded_features) == {"const", "label"}
    assert p.single_value_columns == {"const": 7, "label": "a"}
    assert_frames_equal(p.inverse_transform(p.transform(df)), df)


# ---------------------------------------------------------------------------
# Train/test consistency (ML pipelines, TSTR, privacy attacks)
# ---------------------------------------------------------------------------

def test_statistics_come_from_fit_data():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"x": [100.0, None, 300.0], "y": [1.0, 2.0, 3.0]})
    p = Preprocessor(train, num_fill_null="mean", scaling="normalize")
    out = p.transform(test)
    assert out["x"].tolist() == pytest.approx([25.0, 0.5, 75.0])


def test_kbins_edges_come_from_fit_data():
    train = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0) % 7})
    p = Preprocessor(train, scaling="kbins", n_bins=4)
    assert p.transform(train)["x"].value_counts().to_dict() == {0: 25, 1: 25, 2: 25, 3: 25}
    test = pd.DataFrame({"x": [-5.0, 1000.0], "y": [0.0, 1.0]})
    assert p.transform(test)["x"].tolist() == [0, 3]
    # Bins are decoded to their midpoint
    assert p.inverse_transform(p.transform(test))["x"].tolist() == pytest.approx([12.375, 86.625])


def test_output_layout_is_stable(adult):
    p = Preprocessor(adult)
    full = p.transform(adult)
    subset = p.transform(adult.filter(pl.col("native_country") == "United-States").head(10))
    assert full.columns == subset.columns == p.output_columns
    assert full.schema == subset.schema


def test_features_keep_column_order(adult):
    p = Preprocessor(adult)
    assert list(p.categorical_features) == [col for col in adult.columns if col in p.categorical_features]
    one_hot = [col for col in p.output_columns if col not in adult.columns]
    expected = [name for col in p.categorical_features for name in p.categorical_transformer.original_encoded_columns[col]]
    assert one_hot == expected


def test_row_order_is_preserved(rng):
    n = 100
    start = dt.datetime(2021, 1, 1)
    df = pl.DataFrame({
        "t": pl.datetime_range(start, start + dt.timedelta(days=n - 1), "1d", eager=True).shuffle(seed=1),
        "x": rng.normal(size=n),
    })
    out = Preprocessor(df).transform(df)
    assert out["x"].equals(df["x"])


def test_pandas_index_is_preserved(rng):
    df = pd.DataFrame({"x": rng.normal(size=10), "c": list("ababababab")}, index=np.arange(100, 110))
    p = Preprocessor(df)
    out = p.transform(df)
    assert (out.index == df.index).all()
    assert (p.inverse_transform(out).index == df.index).all()


def test_unseen_labels():
    train = pd.DataFrame({"c": list("aabbcc") * 10, "x": np.arange(60.0)})
    test = pd.DataFrame({"c": ["a", "zzz"], "x": [1.0, 2.0]})
    out = Preprocessor(train).transform(test)
    assert out[["c_a", "c_b", "c_c"]].values.tolist() == [[1, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="zzz"):
        Preprocessor(train, unseen_labels="error").transform(test)


@pytest.mark.parametrize("ml_task", ["classification", "regression"])
def test_target_column(rng, ml_task):
    n = 100
    target = rng.choice(["no", "yes"], n) if ml_task == "classification" else rng.normal(50, 10, n)
    df = pd.DataFrame({"x": rng.normal(size=n), "target": target})
    p = Preprocessor(df, ml_task=ml_task, target_column="target")
    out = p.transform(df)
    if ml_task == "classification":
        assert set(out["target"]) == {0, 1}
    else:
        assert out["target"].min() == 0 and out["target"].max() == 1
    inv = p.inverse_transform(out)
    if ml_task == "classification":
        assert (inv["target"] == df["target"]).all()
    else:
        np.testing.assert_allclose(inv["target"], df["target"])
    # Data without the target (e.g. at inference time) can still be transformed
    assert "target" not in p.transform(df.drop(columns="target")).columns


def test_excluded_col_default_is_not_shared(rng):
    df = pd.DataFrame({"x": rng.normal(size=20), "t": rng.normal(size=20)})
    Preprocessor(df, ml_task="regression", target_column="t")
    p = Preprocessor(pd.DataFrame({"x": rng.normal(size=20), "z": rng.normal(size=20)}))
    assert p.excluded_col == []


def test_preprocessor_is_picklable(adult):
    p = Preprocessor(adult, scaling="quantile")
    restored = pickle.loads(pickle.dumps(p))
    assert_frames_equal(restored.transform(adult), p.transform(adult))


# ---------------------------------------------------------------------------
# Type inference and feature selection
# ---------------------------------------------------------------------------

def test_all_numeric_and_categorical_dtypes_are_detected(rng):
    n = 50
    df = pl.DataFrame({
        "i32":  pl.Series(rng.integers(0, 50, n), dtype=pl.Int32),
        "u8":   pl.Series(rng.integers(0, 50, n), dtype=pl.UInt8),
        "f32":  pl.Series(rng.normal(size=n), dtype=pl.Float32),
        "cat":  pl.Series(rng.choice(list("abc"), n)).cast(pl.Categorical),
        "enum": pl.Series(rng.choice(list("xy"), n)).cast(pl.Enum(["x", "y"])),
    })
    p = Preprocessor(df, scaling="normalize")
    assert p.numerical_features == ("i32", "u8", "f32")
    assert p.categorical_features == ("cat", "enum")
    inv = p.inverse_transform(p.transform(df))
    assert inv.schema == df.schema
    for col in ["i32", "u8", "cat", "enum"]:
        assert inv[col].equals(df[col]), col


@pytest.mark.parametrize("values", [["May", "June", "July"], ["10", "20", "30"], ["2021", "2022", "2023"]])
def test_strings_that_are_not_dates_stay_categorical(rng, values):
    df = pd.DataFrame({"s": rng.choice(values, 100), "x": rng.normal(size=100)})
    p = Preprocessor(df)
    assert p.categorical_features == ("s",)
    assert p.datetime_features == ()


def test_missing_values_threshold(rng):
    x = np.full(100, np.nan)
    x[:2] = [1.0, 2.0]
    df = pd.DataFrame({"mostly_null": x, "y": rng.normal(size=100)})
    assert Preprocessor(df, missing_values_threshold=0.5).discarded_features == ["mostly_null"]
    assert Preprocessor(df).discarded_features == []


def test_columns_with_only_rare_labels_are_discarded(rng):
    df = pd.DataFrame({"id": [f"id{i}" for i in range(200)], "x": rng.normal(size=200)})
    p = Preprocessor(df)
    assert p.discarded_features == ["id"]
    assert p.transform(df).columns.tolist() == ["x"]


@pytest.mark.parametrize("strategy", ["interpolate", "forward", "backward", "mean", "min", "max", "zero", "one", 3.5])
def test_fill_strategies_leave_no_nulls(rng, strategy):
    x = rng.normal(size=50)
    x[[0, 10, 49]] = np.nan
    df = pd.DataFrame({"x": x, "c": rng.choice(list("ab"), 50)})
    out = Preprocessor(df, num_fill_null=strategy).transform(df)
    assert not out["x"].isna().any()


def test_quantile_matches_sklearn(rng):
    n = 2000
    df = pl.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 4, n).astype(float), "c": rng.exponential(size=n)})
    p = Preprocessor(df, scaling="quantile")
    expected = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit_transform(df.to_numpy())
    np.testing.assert_allclose(p.transform(df).to_numpy(), expected)


def test_get_features_sizes_match_layout(rng):
    n = 50
    df = pd.DataFrame({
        "d": pd.date_range("2020-01-01", periods=n),
        "b": rng.random(n) > 0.5,
        "x": rng.normal(size=n),
        "c": rng.choice(list("abc"), n),
        "e": rng.choice(list("de"), n),
    })
    p = Preprocessor(df)
    out = p.transform(df)
    numerical_sizes, categorical_sizes = p.get_features_sizes()
    assert numerical_sizes == [3]
    assert categorical_sizes == [3, 2]
    assert out.shape[1] == sum(numerical_sizes) + sum(categorical_sizes)


def test_invalid_arguments(rng):
    df = pd.DataFrame({"x": rng.normal(size=10), "y": rng.normal(size=10)})
    with pytest.raises(ValueError):
        Preprocessor(df, scaling="kbins")
    with pytest.raises(ValueError):
        Preprocessor(df, unseen_labels="raise")
    with pytest.raises(ValueError):
        Preprocessor(df, num_fill_null="median")
    with pytest.raises(TypeError):
        Preprocessor(df.to_numpy())
    with pytest.raises(ValueError, match="missing"):
        Preprocessor(df).transform(df.drop(columns="x"))
