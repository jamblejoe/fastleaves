"""Bit parity with LightGBM over a matrix of objectives, datasets, training settings and
predict options. Every combination is one test. It either predicts exactly what LightGBM
predicts, or fastleaves refuses it with UnsupportedError, as the README's scope table says.

The test inputs are unseen rows plus edge values: NaN, ±inf, ±0, values around LightGBM's
zero threshold, categories out of range, and every split threshold of the model together
with its neighboring doubles (see cases.py).
"""

import lightgbm
import pytest

import fastleaves
from cases import DATASETS, OBJECTIVES, TRAINING, assert_bit_identical, input_rows, train

PREDICT = {
    "default": {},
    "raw_score": {"raw_score": True},
    "first 5 iterations": {"num_iteration": 5},
    "iterations 3 to 9": {"start_iteration": 3, "num_iteration": 7},
    "start after the end": {"start_iteration": 100},
    "1 thread": {"num_threads": 1},
    "pred_leaf": {"pred_leaf": True},
    "pred_leaf, iterations 3 to 9": {"pred_leaf": True, "start_iteration": 3, "num_iteration": 7},
}


def why_unsupported(objective, training, options):
    """Why fastleaves must refuse a case (README scope table), or None if it must match."""
    if objective == "multiclass":
        return "multiclass"
    if training in ("linear trees", "random forest"):
        return training
    raw = options.get("raw_score") or options.get("pred_leaf")
    if objective == "sqrt regression" and not raw:
        return f"{objective} output (only raw_score=True or pred_leaf=True)"
    return None


@pytest.mark.parametrize("predict", PREDICT)
@pytest.mark.parametrize("training", TRAINING)
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_parity(objective, dataset, training, predict):
    try:
        model = train(objective, dataset, training)
    except lightgbm.basic.LightGBMError as e:
        pytest.skip(f"LightGBM cannot train this: {e}")
    x, options = input_rows(dataset, model), PREDICT[predict]
    if why_unsupported(objective, training, options):
        with pytest.raises(fastleaves.UnsupportedError):
            fastleaves.Booster.from_lightgbm(model).predict(x, **options)
    else:
        assert_bit_identical(fastleaves.Booster.from_lightgbm(model).predict(x, **options), model.predict(x, **options))
