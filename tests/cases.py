"""Building blocks of the parity tests: objectives, datasets, training settings, test inputs
and the bit-for-bit comparison. The matrix that combines them is in test_parity.py."""

import functools
import itertools
from dataclasses import dataclass, field
from typing import NamedTuple

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
    """Six features: 0-3 numerical, 4-5 integers (categorical if listed in `categorical`),
    then `extra` numerical features, every tenth of them constant."""

    nan: float = 0.0  # share of NaN values
    zeros: float = 0.0  # share of exact zeros
    categorical: tuple = ()
    categories: int = 30  # feature 5 takes `categories` values: 0, step, 2 * step, ...
    category_step: int = 1
    extra: int = 0  # more numerical features
    params: dict = field(default_factory=dict)  # LightGBM parameters this data needs


DATASETS = {
    "numeric": Data(),
    "with NaN": Data(nan=0.1),
    "NaN, use_missing=False": Data(nan=0.1, params={"use_missing": False}),
    "zero as missing": Data(zeros=0.2, params={"zero_as_missing": True}),
    "categorical": Data(nan=0.05, categorical=(4, 5)),
    "large category IDs": Data(nan=0.05, categorical=(4, 5), category_step=1000),
    # feature 4 keeps its set in the node, feature 5 (about 170 categories in splits) in a table
    "many categories": Data(
        nan=0.05, categorical=(4, 5), categories=200, params={"min_data_per_group": 5, "cat_smooth": 1}
    ),
    "100 features": Data(nan=0.05, categorical=(4, 5), extra=94),
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
    """`n` rows of the dataset `data`."""
    x = np.column_stack(
        [
            rng.normal(size=n),
            rng.normal(size=n),
            rng.uniform(-1, 1, n),
            rng.exponential(size=n),
            rng.integers(0, 10, n),
            rng.integers(0, data.categories, n),
            *(np.full(n, 1.0) if i % 10 == 0 else rng.normal(size=n) for i in range(data.extra)),
        ]
    ).astype(float)
    x[:, 5] *= data.category_step
    x[rng.random(x.shape) < data.zeros] = 0.0
    x[rng.random(x.shape) < data.nan] = np.nan
    return x


def labels(rng, x, kind, data):
    x = np.nan_to_num(x)
    score = 0.5 * x[:, 0] - 0.3 * x[:, 1] + np.sin(3 * x[:, 2]) + 0.1 * x[:, 4] - 0.02 * x[:, 5] / data.category_step
    score += 0.05 * x[:, 6:].sum(axis=1)
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
CATEGORIES = [c + d for c in [*range(-1, 34), 199, 200, 254, 255, 256] for d in (0.0, 0.5)]


class Split(NamedTuple):
    feature: int
    threshold: float | None  # numerical splits
    categories: list | None  # categorical splits: the categories that go left
    default_left: bool  # missing values go left


def splits(model):
    """Every split of the model, as a Split."""
    for block in model.model_to_string().split("\n\n"):
        tree = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "split_feature" not in tree:
            continue
        boundaries = [int(v) for v in tree.get("cat_boundaries", "").split()]
        all_words = [int(w) for w in tree.get("cat_threshold", "").split()]
        for f, t, d in zip(*(tree[k].split() for k in ("split_feature", "threshold", "decision_type"))):
            default_left = bool(int(d) & 2)
            if int(d) & 1 == 0:  # numerical split
                yield Split(int(f), float(t), None, default_left)
            else:  # categorical: the threshold indexes a bitset of the categories that go left
                b = boundaries[int(t) : int(t) + 2]
                words = all_words[b[0] : b[1]]
                cats = [32 * i + k for i, w in enumerate(words) if w for k in range(32) if w >> k & 1]
                yield Split(int(f), None, cats, default_left)


def split_categories(model, feature):
    """The categories in any of the model's splits on `feature`."""
    return {c for s in splits(model) if s.feature == feature and s.categories for c in s.categories}


def split_thresholds(model, feature):
    """The numerical thresholds in any of the model's splits on `feature`."""
    return {s.threshold for s in splits(model) if s.feature == feature and s.threshold is not None}


# (missing, n_thresholds): (seed, number of integers) that thresholds_model's search finds
# with LightGBM 4.7, tried first so the search does not retrain up to 11 models
THRESHOLDS_SEEDS = {("None", 254): (6, 255), ("Zero", 254): (8, 255), ("NaN", 254): (6, 253), ("None", 255): (6, 256)}


@functools.cache
def thresholds_model(missing, n_thresholds):
    """A model with exactly `n_thresholds` thresholds on numerical feature 0 and 255
    categories on categorical feature 1. Feature 0 takes integers 1, 2, ..., each with its
    own effect; LightGBM may skip a split or add one at its zero bin boundary, so the
    number of integers and the seed are the first that give `n_thresholds`. `missing` is LightGBM's
    missing type of feature 0: in "Zero" and "NaN" one row in ten is 0 or NaN."""
    search = itertools.product(range(6, 16), range(n_thresholds - 1, n_thresholds + 3))
    known = THRESHOLDS_SEEDS.get((missing, n_thresholds))
    for seed, values in itertools.chain([known] if known else [], search):
        rng = np.random.default_rng(seed)
        n = 50_000
        num = rng.integers(1, values + 1, n).astype(float)
        cat = rng.integers(0, 256, n).astype(float)
        y = rng.normal(size=values + 1)[num.astype(int)] + 0.3 * rng.normal(size=256)[cat.astype(int)]
        hole = rng.random(n) < 0.1
        num, params = {
            "None": (num, {"use_missing": False}),
            "Zero": (np.where(hole, 0.0, num), {"zero_as_missing": True}),
            "NaN": (np.where(hole, np.nan, num), {}),
        }[missing]
        params = {"verbose": -1, "num_leaves": 255, "min_data_in_leaf": 2, "learning_rate": 0.5, "max_bin": 1023,
                  "min_data_per_group": 5, "cat_smooth": 1, "max_cat_threshold": 128, **params}
        model = lgb.train(params, lgb.Dataset(np.column_stack([num, cat]), y, categorical_feature=[1]), 100)
        if len(split_thresholds(model, 0)) == n_thresholds:
            return model
    raise AssertionError(f"no model with {n_thresholds} thresholds ({missing})")


def edge_values(model):
    """Per feature: EDGE_VALUES, every category up to 33 and some above, each split threshold
    with its neighboring doubles, and each category of a split with its neighbors."""
    values = [EDGE_VALUES + CATEGORIES for _ in range(model.num_feature())]
    for s in splits(model):
        if s.threshold is not None:
            t = s.threshold
            values[s.feature] += [t, np.nextafter(t, -np.inf), np.nextafter(t, np.inf)]
        else:
            values[s.feature] += [c + d for c in s.categories for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    return [sorted(set(v), key=repr) for v in values]


def input_rows(dataset, model):
    """1000 unseen rows, plus copies of the first one with a single feature set to each
    edge value of that feature. A fresh copy each call, built once per model."""
    return _input_rows(dataset, model).copy()


@functools.cache
def _input_rows(dataset, model):
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


def random_model(seed):
    """A LightGBM model file with random trees, written directly instead of trained, so it has
    tree shapes, thresholds, category sets and leaf values that training rarely produces. It
    stays within the README's scope: at most 256 features used in splits, 254 thresholds and
    255 categories per feature, and one kind of split and missing type per feature."""
    rng = np.random.default_rng(seed)
    n_features = int(rng.choice([1, 3, 20, 300]))
    used = rng.permutation(n_features)[: rng.integers(1, min(n_features, 256) + 1)]
    kind = {f: rng.choice(["num", "cat"]) for f in used}  # kind of split
    missing = {f: rng.integers(0, 3) for f in used}  # missing type: None, Zero, NaN
    pool = {}  # the thresholds or categories of each feature's splits
    for f in used:
        if kind[f] == "num":
            edges = [v for v in EDGE_VALUES if np.isfinite(v)]
            values = [*edges, *(rng.normal(size=300) * 10.0 ** rng.integers(-5, 6, 300))]
            pool[f] = rng.choice(values, rng.integers(1, 255), replace=False)
        else:  # category IDs below 40, 256 or 30,000
            ids = int(rng.choice([40, 256, 30_000]))
            pool[f] = rng.choice(ids, rng.integers(1, min(ids, 255) + 1), replace=False)
    trees = []
    for t in range(rng.integers(1, 60)):
        n_leaves = 1 if rng.random() < 0.1 else int(2 ** rng.uniform(1, 10))
        trees.append(random_tree(rng, t, n_leaves, kind, missing, pool))
    return random_model_file(n_features, trees, rng.choice(["regression", "poisson"]))


def random_model_file(n_features, trees, objective="regression"):
    """A model file with `n_features` columns and the given tree blocks (see random_tree)."""
    header = [
        "tree", "version=v4", "num_class=1", "num_tree_per_iteration=1", "label_index=0",
        f"max_feature_idx={n_features - 1}", f"objective={objective}",
        "feature_names=" + " ".join(f"Column_{f}" for f in range(n_features)),
        "feature_infos=" + " ".join(["none"] * n_features),
    ]
    return "\n\n".join(["\n".join(header), *trees, "end of trees"]) + "\n"


def random_tree(rng, number, n_leaves, kind, missing, pool, chain=None):
    """Tree `number` of random_model, grown as LightGBM grows trees: each split turns a leaf
    into a node whose left child is that leaf and whose right child is a new leaf. `chain` is
    the share of splits of the newest leaf (random if None): near 1, leaves lie at very
    different depths; at 1 the tree is a chain as deep as it has splits."""
    chain = rng.random() if chain is None else chain
    child = [[], []]  # the left and right child of each node: ~leaf or a later node
    parent = {0: None}  # leaf: (node, side) that points to it
    features, thresholds, types, cat_sets = [], [], [], []
    for node in range(n_leaves - 1):
        leaf = node if rng.random() < chain else int(rng.integers(node + 1))
        if parent[leaf] is not None:
            child[parent[leaf][1]][parent[leaf][0]] = node
        child[0].append(~leaf)
        child[1].append(~(node + 1))
        parent[leaf], parent[node + 1] = (node, 0), (node, 1)
        f = rng.choice(list(pool))
        default_left = int(rng.integers(2))
        if kind[f] == "num":
            thresholds.append(repr(float(rng.choice(pool[f]))))
            types.append(missing[f] << 2 | default_left << 1)
        else:  # the threshold indexes the set of categories that go left
            thresholds.append(str(len(cat_sets)))
            cat_sets.append(rng.choice(pool[f], rng.integers(1, len(pool[f]) + 1), replace=False))
            types.append(missing[f] << 2 | default_left << 1 | 1)
        features.append(int(f))
    leaves = rng.normal(size=n_leaves) * 10.0 ** rng.integers(-6, 7, n_leaves)
    leaves[rng.random(n_leaves) < 0.05] = rng.choice([0.0, -0.0])
    words = []  # each category set as a bitset of 32-bit words
    for s in cat_sets:
        words.append([0] * (max(s) // 32 + 1))
        for c in s:
            words[-1][c // 32] |= 1 << (c % 32)
    join = lambda values: " ".join(str(v) for v in values)
    lines = [
        f"Tree={number}", f"num_leaves={n_leaves}", f"num_cat={len(cat_sets)}",
        f"split_feature={join(features)}", f"threshold={join(thresholds)}", f"decision_type={join(types)}",
        f"left_child={join(child[0])}", f"right_child={join(child[1])}",
        f"leaf_value={join(repr(float(v)) for v in leaves)}",
    ]
    if cat_sets:
        lines += [f"cat_boundaries={join(np.cumsum([0, *map(len, words)]))}",
                  f"cat_threshold={join(w for ws in words for w in ws)}"]
    return "\n".join(lines + ["is_linear=0", "shrinkage=1"])


def random_rows(rng, model, n):
    """`n` rows; in each, most values are edge values of their feature (see edge_values), the
    others random doubles of any magnitude."""
    values = edge_values(model)
    x = rng.normal(size=(n, len(values))) * 10.0 ** rng.integers(-5, 6, (n, len(values)))
    for f, v in enumerate(values):
        edge = rng.random(n) < 0.7
        x[edge, f] = rng.choice(v, edge.sum())
    return x
