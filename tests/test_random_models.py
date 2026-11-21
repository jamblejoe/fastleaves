"""Bit parity with LightGBM on model files with random trees (random_model below), written
directly instead of trained: tree shapes, thresholds, category sets and leaf values that
training rarely produces. LightGBM loads the same file and is the reference."""

import lightgbm
import numpy as np
import pytest

import fastleaves
from cases import EDGE_VALUES, assert_bit_identical, edge_values, model_file

N_MODELS = 200


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
    return model_file(n_features, trees, rng.choice(["regression", "poisson"]))


def random_tree(rng, number, n_leaves, kind, missing, pool):
    """Tree `number` of random_model, grown as LightGBM grows trees: each split turns a leaf
    into a node whose left child is that leaf and whose right child is a new leaf. Some trees
    split mostly their newest leaf, so their leaves lie at very different depths."""
    chain = rng.random()
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


@pytest.mark.parametrize("seed", range(N_MODELS))
def test_random_model(seed):
    text = random_model(seed)
    model, booster = lightgbm.Booster(model_str=text), fastleaves.Booster(model_str=text)
    rng = np.random.default_rng(seed)
    x = random_rows(rng, model, 5000)  # more than one block of 4096 rows
    n_trees = model.num_trees()
    start, count = rng.integers(0, n_trees + 1), rng.integers(1, n_trees + 1)
    for options in [{}, {"raw_score": True}, {"pred_leaf": True},
                    {"start_iteration": start, "num_iteration": count},
                    {"pred_leaf": True, "start_iteration": start, "num_iteration": count}]:
        assert_bit_identical(booster.predict(x, **options), model.predict(x, **options))
