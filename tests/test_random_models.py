"""Bit parity with LightGBM on model files with random trees (cases.random_model), written
directly instead of trained: tree shapes, thresholds, category sets and leaf values that
training rarely produces. LightGBM loads the same file and is the reference."""

import threading
from concurrent.futures import ThreadPoolExecutor

import lightgbm
import numpy as np
import pytest

import fastleaves
from cases import assert_bit_identical, random_model, random_model_file, random_rows, random_tree

N_MODELS = 200


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


def on_small_stack(f):
    """f() on a thread with a 256 KB stack, where recursing once per tree level would crash."""
    default = threading.stack_size(256 * 1024)
    try:
        with ThreadPoolExecutor(1) as thread:
            return thread.submit(f).result()
    finally:
        threading.stack_size(default)


@pytest.mark.parametrize("chain", [0.0, 0.5, 1.0], ids=["random shape", "half chain", "chain"])
def test_largest_tree(chain):
    """A tree with 32,768 leaves (the most supported) and 65,535 nodes; as a chain, 32,767
    levels deep. fastleaves loads it on a small stack."""
    rng = np.random.default_rng(0)
    pool = {0: np.arange(254.0), 1: np.arange(255)}
    tree = random_tree(rng, 0, 32_768, {0: "num", 1: "cat"}, {0: 2, 1: 0}, pool, chain)
    text = random_model_file(2, [tree])
    model, booster = lightgbm.Booster(model_str=text), on_small_stack(lambda: fastleaves.Booster(model_str=text))
    x = random_rows(rng, model, 5000)
    for options in [{}, {"pred_leaf": True}]:
        assert_bit_identical(booster.predict(x, **options), model.predict(x, **options))
