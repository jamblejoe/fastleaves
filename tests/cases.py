"""Building blocks of the parity tests: objectives, datasets, training settings, test inputs
and the bit-for-bit comparison. The matrix that combines them is in test_parity.py."""

import functools
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

N_TRAIN = 1000  # rows to train on
N_TREES = 20

# name: (LightGBM parameters, kind of labels)
OBJECTIVES = {
    "regression": ({"objective": "regression"}, "real"),
    "regression_l1": ({"objective": "regression_l1"}, "real"),
    "huber": ({"objective": "huber"}, "real"),
    "fair": ({"objective": "fair"}, "real"),
    "quantile": ({"objective": "quantile", "alpha": 0.8}, "real"),
    "mape": ({"objective": "mape"}, "positive"),
    "poisson": ({"objective": "poisson"}, "count"),
    "gamma": ({"objective": "gamma"}, "positive"),
    "tweedie": ({"objective": "tweedie", "tweedie_variance_power": 1.3}, "count"),
    "binary": ({"objective": "binary"}, "binary"),
    "cross_entropy": ({"objective": "cross_entropy"}, "probability"),
    "sqrt regression": ({"objective": "regression", "reg_sqrt": True}, "positive"),
    "multiclass": ({"objective": "multiclass", "num_class": 3}, "class"),
}


@dataclass(frozen=True)
class Data:
    """Six features: 0-3 numerical, 4-5 integers (categorical if listed in `categorical`)."""

    nan: float = 0.0  # share of NaN values
    zeros: float = 0.0  # share of exact zeros
    categorical: tuple = ()
    category_step: int = 1  # feature 5 takes the values 0, step, 2 * step, ...
    params: dict = field(default_factory=dict)  # LightGBM parameters this data needs


DATASETS = {
    "numeric": Data(),
    "with NaN": Data(nan=0.1),
    "NaN, use_missing=False": Data(nan=0.1, params={"use_missing": False}),
    "zero as missing": Data(zeros=0.2, params={"zero_as_missing": True}),
    "categorical": Data(nan=0.05, categorical=(4, 5)),
    "large category IDs": Data(nan=0.05, categorical=(4, 5), category_step=1000),
}

TRAINING = {
    "default": {},
    "deep trees": {"num_leaves": 63, "min_data_in_leaf": 3},
    "dart": {"boosting": "dart"},
    "goss": {"data_sample_strategy": "goss"},
    "extra trees": {"extra_trees": True},
    "no boost from average": {"boost_from_average": False},
    "linear trees": {"linear_tree": True},
    "random forest": {"boosting": "rf", "bagging_freq": 1, "bagging_fraction": 0.5},
}


def features(rng, n, data):
    """`n` rows of the dataset `data` (few distinct values, so models stay in v0.1 limits)."""
    x = np.column_stack(
        [
            rng.normal(size=n).round(1),
            rng.normal(size=n).round(1),
            rng.uniform(-1, 1, n).round(2),
            rng.exponential(size=n).round(1),
            rng.integers(0, 10, n),
            rng.integers(0, 30, n),
        ]
    ).astype(float)
    x[:, 5] *= data.category_step
    x[rng.random(x.shape) < data.zeros] = 0.0
    x[rng.random(x.shape) < data.nan] = np.nan
    return x


def labels(rng, x, kind, data):
    x = np.nan_to_num(x)
    score = 0.5 * x[:, 0] - 0.3 * x[:, 1] + np.sin(3 * x[:, 2]) + 0.1 * x[:, 4] - 0.02 * x[:, 5] / data.category_step
    score += rng.normal(scale=0.3, size=len(x))
    return {
        "real": lambda: score,
        "positive": lambda: np.exp(score),
        "count": lambda: rng.poisson(np.exp(score)),
        "binary": lambda: (score > 0).astype(float),
        "probability": lambda: 1 / (1 + np.exp(-score)),
        "class": lambda: np.digitize(score, [-0.5, 0.5]),
    }[kind]()


@functools.cache
def train(objective, dataset, training):
    """A LightGBM model trained on `dataset` with `objective` and `training` settings."""
    params, label_kind = OBJECTIVES[objective]
    data = DATASETS[dataset]
    rng = np.random.default_rng(0)
    x = features(rng, N_TRAIN, data)
    train_set = lgb.Dataset(x, labels(rng, x, label_kind, data), categorical_feature=list(data.categorical) or "auto")
    params = {"verbose": -1, "num_leaves": 15, "seed": 1, **params, **data.params, **TRAINING[training]}
    return lgb.train(params, train_set, num_boost_round=N_TREES)


# Values where implementations tend to differ.
ZERO = float(np.float32(1e-35))  # LightGBM reads inputs within ±ZERO as 0
EDGE_VALUES = [np.nan, np.inf, -np.inf, 0.0, -0.0, ZERO, -ZERO, np.nextafter(ZERO, 1), np.nextafter(-ZERO, -1),
               1e-36, -1e-36, 5e-324, -0.5, -0.99, -1.0, 0.5, 31.0, 31.5, 32.0, 2.0**31, -(2.0**31) - 1, 1e300, -1e300]
CATEGORIES = [c + d for c in range(-1, 34) for d in (0.0, 0.5)]


def edge_values(model):
    """Per feature: EDGE_VALUES, every category up to 33, each split threshold with its
    neighboring doubles, and each category of a split with its neighbors."""
    values = [EDGE_VALUES + CATEGORIES for _ in range(model.num_feature())]
    for block in model.model_to_string().split("\n\n"):
        tree = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "split_feature" not in tree:
            continue
        for f, t, d in zip(*(tree[k].split() for k in ("split_feature", "threshold", "decision_type"))):
            if int(d) & 1 == 0:  # numerical split
                t = float(t)
                values[int(f)] += [t, np.nextafter(t, -np.inf), np.nextafter(t, np.inf)]
            else:  # categorical: the threshold indexes a bitset of the categories that go left
                b = [int(v) for v in tree["cat_boundaries"].split()][int(t) : int(t) + 2]
                words = [int(w) for w in tree["cat_threshold"].split()[b[0] : b[1]]]
                cats = [32 * i + k for i, w in enumerate(words) for k in range(32) if w >> k & 1]
                values[int(f)] += [c + d for c in cats for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    return [sorted(set(v), key=repr) for v in values]


def input_rows(dataset, model):
    """1000 unseen rows, plus copies of the first one with a single feature set to each
    edge value of that feature."""
    x = features(np.random.default_rng(1), 1000, DATASETS[dataset])
    edges = []
    for f, values in enumerate(edge_values(model)):
        for v in values:
            row = x[0].copy()
            row[f] = v
            edges.append(row)
    return np.vstack([x, edges])


def assert_bit_identical(actual, expected):
    """Same dtype, shape and bits (NaN == NaN, -0.0 != 0.0)."""
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    bits = f"u{actual.itemsize}"
    differ = np.argwhere(actual.view(bits) != expected.view(bits))
    if differ.size:
        i = tuple(differ[0])
        raise AssertionError(
            f"{len(differ)} of {actual.size} predictions differ; "
            f"first at {i}: fastleaves {actual[i]!r}, LightGBM {expected[i]!r}"
        )
