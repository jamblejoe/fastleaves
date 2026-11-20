"""Bit parity with LightGBM on model files with random trees (cases.random_model), written
directly instead of trained: tree shapes, thresholds, category sets and leaf values that
training rarely produces. LightGBM loads the same file and is the reference."""

import lightgbm
import numpy as np
import pytest

import fastleaves
from cases import assert_bit_identical, random_model, random_rows

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
