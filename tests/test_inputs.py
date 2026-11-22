"""Input handling: every input form gives LightGBM's result for the same input."""

import functools
import itertools
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import lightgbm
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fastleaves
from cases import (
    DATASETS,
    assert_bit_identical,
    chain_model,
    complete_model,
    edge_rows,
    edge_values,
    features,
    split_categories,
    split_thresholds,
    splits,
    thresholds_model,
    train,
)

MODEL = "poisson", "with NaN", "default"
PRED_LEAF = pytest.mark.parametrize("pred_leaf", [False, True], ids=["predictions", "pred_leaf"])


def check(model, data, **options):
    """fastleaves predicts on `data` exactly what LightGBM predicts."""
    expected = model.predict(data, **options)
    assert_bit_identical(fastleaves.Booster.from_lightgbm(model).predict(data, **options), expected)


def rows(n, dataset="with NaN"):
    return features(np.random.default_rng(2), n, DATASETS[dataset])


@PRED_LEAF
@pytest.mark.parametrize("n", [1, 15, 16, 17, 63, 64, 65, 4095, 4096, 4097, 20_000])
def test_batch_sizes(n, pred_leaf):
    check(train(*MODEL), rows(n), pred_leaf=pred_leaf)


@PRED_LEAF
@pytest.mark.parametrize("n", [1, 17, 4095, 4096, 4097, 20_000])
def test_batch_sizes_100_features(n, pred_leaf):
    check(train("poisson", "100 features", "deep trees"), rows(n, "100 features"), pred_leaf=pred_leaf)


def test_unused_features_do_not_count():
    """A model with 1,000 features that splits on only a few of them (limit: 256 used)."""
    x = np.random.default_rng(0).normal(size=(500, 1000))
    model = lightgbm.train({"verbose": -1}, lightgbm.Dataset(x, x[:, 990:].sum(axis=1)), 5)
    assert (model.feature_importance() > 0).sum() <= 256
    check(model, np.random.default_rng(1).normal(size=(5000, 1000)))


def test_no_rows():
    model = train(*MODEL)
    check(model, rows(0))
    # LightGBM fails here (ZeroDivisionError when it reshapes); fastleaves returns no rows
    leaves = fastleaves.Booster.from_lightgbm(model).predict(rows(0), pred_leaf=True)
    assert_bit_identical(leaves, np.zeros((0, model.num_trees()), dtype=np.int32))


def read_only(x):
    x = x.copy()
    x.flags.writeable = False
    return x


def unaligned(x):
    """`x` in memory that is not 8-byte aligned."""
    y = np.empty(x.nbytes + 1, dtype=np.uint8)[1:].view(np.float64).reshape(x.shape)
    y[...] = x
    assert not y.flags.aligned
    return y


INPUT_FORMS = {
    "float64": lambda x: x,
    "float32": lambda x: x.astype(np.float32),
    "Fortran order": lambda x: np.asfortranarray(x),
    "every second row": lambda x: np.repeat(x, 2, axis=0)[::2],
    "read-only": read_only,
    "unaligned": unaligned,
    "list of lists": lambda x: x.tolist(),
    "int64 (LightGBM reads it as float32)": lambda x: (x * 1e7).astype(np.int64),
    "int32": lambda x: x.astype(np.int32),
    "bool": lambda x: x > 0,
    "DataFrame": lambda x: pd.DataFrame(x),
    "DataFrame of float32 and int16": lambda x: pd.DataFrame(x).astype({0: np.float32, 4: np.int16}),
}


@pytest.mark.parametrize("form", INPUT_FORMS)
def test_input_forms(form):
    check(train(*MODEL), INPUT_FORMS[form](rows(300, "numeric")))


def test_integers_are_read_as_float32():
    """LightGBM converts integer arrays to float32, so 2**24 + 1 is read as 2**24."""
    x = (2**24 + np.arange(8)).repeat(50).reshape(-1, 1)
    model = lightgbm.train({"verbose": -1, "min_data_in_leaf": 5}, lightgbm.Dataset(x.astype(float), x[:, 0] % 2), 5)
    check(model, x)


@PRED_LEAF
@pytest.mark.parametrize("threads", [1, 2, 3, os.cpu_count() + 1, 0, -1])
def test_thread_counts(threads, pred_leaf):
    check(train(*MODEL), rows(10_000), num_threads=threads, pred_leaf=pred_leaf)


def test_model_file_and_string(tmp_path):
    model = train(*MODEL)
    model.save_model(tmp_path / "model.txt")
    x, expected = rows(500), model.predict(rows(500))
    for booster in [fastleaves.Booster(model_file=tmp_path / "model.txt"),
                    fastleaves.Booster(model_str=model.model_to_string())]:
        assert_bit_identical(booster.predict(x), expected)
        assert booster.num_feature() == model.num_feature()
        assert booster.num_trees() == model.num_trees()
        assert booster.feature_name() == model.feature_name()


def test_isa_names_the_kernel():
    """`isa()` is "avx512" or "scalar", and "scalar" when FASTLEAVES_ISA=scalar."""
    assert fastleaves.isa() in ("avx512", "scalar")
    if os.environ.get("FASTLEAVES_ISA") == "scalar":
        assert fastleaves.isa() == "scalar"


def test_best_iteration_from_early_stopping():
    x = rows(2000, "numeric")
    y = np.exp(x[:, 0])
    valid = lightgbm.Dataset(x[1000:], y[1000:])
    model = lightgbm.train(
        {"objective": "poisson", "verbose": -1, "learning_rate": 0.5},
        lightgbm.Dataset(x[:1000], y[:1000]),
        num_boost_round=200,
        valid_sets=[valid],
        callbacks=[lightgbm.early_stopping(3, verbose=False)],
    )
    assert 0 < model.best_iteration < 200
    check(model, x)


@PRED_LEAF
def test_255_categories_on_one_feature(pred_leaf):
    """The most categories supported, next to a feature with few (set kept in the node)."""
    rng = np.random.default_rng(5)
    n = 50_000
    x = np.column_stack([rng.integers(0, 256, n), rng.integers(0, 8, n), rng.normal(size=n)]).astype(float)
    y = rng.normal(size=256)[x[:, 0].astype(int)] + rng.normal(size=8)[x[:, 1].astype(int)] + x[:, 2]
    params = {"verbose": -1, "min_data_per_group": 5, "cat_smooth": 1, "max_cat_threshold": 128, "max_bin": 1023}
    model = lightgbm.train(params, lightgbm.Dataset(x, y, categorical_feature=[0, 1]), 20)
    assert len(split_categories(model, 0)) == 255
    assert len(split_categories(model, 1)) <= 32
    test = np.tile(x[:1], (2 * 260 + 1, 1))
    test[:, 0] = [*(c + d for c in range(-1, 259) for d in (0.0, 0.5)), np.nan]  # every category, unseen ones
    check(model, np.vstack([test, x[:1000]]), pred_leaf=pred_leaf)


@PRED_LEAF
@pytest.mark.parametrize("missing", ["None", "Zero", "NaN"])
def test_254_thresholds_on_one_feature(missing, pred_leaf):
    """The most thresholds supported, in each missing-value mode, next to a feature with 255
    categories: missing values and unseen categories share a bin value."""
    model = thresholds_model(missing, 254)
    assert len(split_thresholds(model, 0)) == 254
    assert len(split_categories(model, 1)) == 255
    if missing != "None":  # missing values go left at some splits and right at others
        assert {s.default_left for s in splits(model) if s.threshold is not None} == {False, True}
    x = np.random.default_rng(7).integers(0, 257, size=(1000, 2)).astype(float)
    x[::10, 0] = np.nan
    edges = []
    for v0 in edge_values(model)[0]:
        for v1 in [0.0, 100.0, 254.0, 255.0, 256.0, -1.0, np.nan]:  # known and unseen categories
            edges.append([v0, v1])
    check(model, np.vstack([np.array(edges), x]), pred_leaf=pred_leaf)


def on_small_stack(f):
    """f() on a thread with a 256 KB stack, where recursing once per tree level would crash."""
    default = threading.stack_size(256 * 1024)
    try:
        with ThreadPoolExecutor(1) as thread:
            return thread.submit(f).result()
    finally:
        threading.stack_size(default)


LARGEST_TREES = {"chain": lambda: chain_model(32_768), "complete": lambda: complete_model(15)}


@pytest.mark.parametrize("shape", LARGEST_TREES)
def test_largest_tree(shape):
    """The most leaves supported in a tree, 32,768 (65,535 nodes): a chain 32,767 levels deep,
    and a complete tree 15 levels deep. fastleaves loads it on a small stack. Rows: each edge
    value of each feature, and every row of 0s and 1s (each leaf of the complete tree)."""
    text = LARGEST_TREES[shape]()
    model, booster = lightgbm.Booster(model_str=text), on_small_stack(lambda: fastleaves.Booster(model_str=text))
    n = model.num_feature()
    x = np.vstack([edge_rows(model, np.zeros(n)), list(itertools.product([0.0, 1.0], repeat=n))])
    for options in [{}, {"pred_leaf": True}]:
        assert_bit_identical(booster.predict(x, **options), model.predict(x, **options))


# How LightGBM writes sigmoids (6 significant digits), plus values it reads one double off
# with its own number parser, longer fractions, exponents and signs. The two longest
# fractions (26 and 39 digits) are where the order of LightGBM's Pow(10, n) changes the result.
SIGMOIDS = ["1", "0.7", "2.3892", "1.22951", "17.9733", "0.123457", "3.14159", "0.001", "1e-07", "123456",
            "2.5E+3", "+0.5", "0.30000000000000004", "1.2345678901234567890123456789", "7e-300",
            "0.29141777631706690743915000", "0.806360837783533740681241586834497869073"]


@pytest.mark.parametrize("sigmoid", SIGMOIDS)
def test_binary_sigmoid(sigmoid):
    """Any sigmoid in the model file, read as LightGBM reads it."""
    text = train("binary", "with NaN", "default").model_to_string()
    assert "objective=binary sigmoid:1\n" in text
    text = text.replace("objective=binary sigmoid:1\n", f"objective=binary sigmoid:{sigmoid}\n")
    x = rows(2000)
    expected = lightgbm.Booster(model_str=text).predict(x)
    assert_bit_identical(fastleaves.Booster(model_str=text).predict(x), expected)


@PRED_LEAF
def test_pandas_categorical_columns(pred_leaf):
    """Categories are matched by value, in the order seen in training, as LightGBM does."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=1000), "c": pd.Categorical(rng.choice(["x", "y", "z", "w"], 1000))})
    model = lightgbm.train({"verbose": -1, "min_data_per_group": 5}, lightgbm.Dataset(df, df["a"] + (df["c"] == "y")), 10)
    test = pd.DataFrame({
        "a": rng.normal(size=8),
        "c": pd.Categorical(["y", "x", "unseen", None, "w", "z", "y", "x"], categories=["y", "unseen", "x", "w", "z"]),
    })
    check(model, test, pred_leaf=pred_leaf)


def named(columns):
    """50 rows with the given column names, one column per name (repeating features)."""
    x = rows(50, "numeric")
    return pd.DataFrame(x[:, [i % 6 for i in range(len(columns))]], columns=list(columns))


NAMES = ["a", "b", "c", "d", "e", "f"]
FEATURE_NAMES = {
    "array (no names to check, as in LightGBM)": rows(50, "numeric"),
    "same names": named(NAMES),
    "other names": named(["a", "b", "c", "d", "e", "g"]),
    "swapped": named(["b", "a", "c", "d", "e", "f"]),
    "integer names": named(range(6)),
    "too few": named(NAMES[:5]),
    "too many": named(NAMES + ["g"]),
}


@functools.cache
def named_model():
    x = rows(1000, "numeric")
    return lightgbm.train({"verbose": -1}, lightgbm.Dataset(pd.DataFrame(x, columns=NAMES), x[:, 0]), 5)


@pytest.mark.parametrize("data", FEATURE_NAMES)
def test_validate_features(data):
    """Same names pass; otherwise LightGBM's message (as ValueError, not LightGBMError)."""
    model, df = named_model(), FEATURE_NAMES[data]
    try:
        expected = model.predict(df, validate_features=True)
    except lightgbm.basic.LightGBMError as e:
        with pytest.raises(ValueError, match=f"^{re.escape(str(e))}$"):
            fastleaves.Booster.from_lightgbm(model).predict(df, validate_features=True)
    else:
        assert_bit_identical(fastleaves.Booster.from_lightgbm(model).predict(df, validate_features=True), expected)


DATASET_MODELS = [("regression", d, "deep trees") for d in DATASETS]


@pytest.mark.parametrize("objective,dataset,training", DATASET_MODELS, ids=[m[1] for m in DATASET_MODELS])
@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_generated_rows(objective, dataset, training, data):
    """Rows built from edge values, thresholds and their neighbors, and arbitrary doubles."""
    model = train(objective, dataset, training)
    columns = [st.sampled_from(values) | st.floats() for values in edge_values(model)]
    x = data.draw(st.lists(st.tuples(*columns), min_size=1, max_size=40))
    check(model, np.array(x, dtype=np.float64))
