"""scikit-learn models of LightGBM: `Booster.from_lightgbm(model)` predicts what `model` predicts."""

import functools

import lightgbm
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import fastleaves
from cases import assert_bit_identical
from fastleaves import UnsupportedError


def frame(n, seed):
    """Two numerical columns (one with NaN) and a pandas categorical column."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    df.loc[::7, "a"] = np.nan
    target = df["a"].fillna(0) + (df["c"] == "y") + rng.normal(size=n)
    return df, target.to_numpy()


def early_stopped(model, x, y, x_valid, y_valid):
    return model.fit(x, y, eval_set=[(x_valid, y_valid)], callbacks=[lightgbm.early_stopping(5, verbose=False)])


@functools.cache
def fitted(name):
    """The model named `name` and rows it has not seen."""
    (x, y), (x_valid, y_valid), (rows, _) = frame(2000, 0), frame(500, 1), frame(1000, 2)
    params = dict(n_estimators=200, verbose=-1)
    if name == "regressor, early stopping":
        model = early_stopped(lightgbm.LGBMRegressor(**params), x, y, x_valid, y_valid)
    elif name == "classifier, string labels, early stopping":
        label = lambda y: np.where(y > 0.5, "yes", "no")  # noqa: E731
        model = early_stopped(lightgbm.LGBMClassifier(**params), x, label(y), x_valid, label(y_valid))
    elif name == "classifier, integer labels, numpy":
        x, rows = x[["a", "b"]].to_numpy(), rows[["a", "b"]].to_numpy()
        model = lightgbm.LGBMClassifier(n_estimators=20, verbose=-1).fit(x, np.where(y > 0.5, 7, 3))
    return model, rows


MODELS = ["regressor, early stopping", "classifier, string labels, early stopping", "classifier, integer labels, numpy"]
OPTIONS = {
    "default": {},
    "raw_score": dict(raw_score=True),
    "pred_leaf": dict(pred_leaf=True),
    "num_iteration": dict(num_iteration=5),
    "start_iteration + num_iteration": dict(start_iteration=3, num_iteration=4),
    "validate_features": dict(validate_features=True),
}


def assert_identical(actual, expected):
    if expected.dtype.kind in "OSU":  # class labels
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
        assert np.array_equal(actual, expected)
    else:
        assert_bit_identical(actual, expected)


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("name", MODELS)
def test_predictions(name, method, options):
    model, rows = fitted(name)
    if not hasattr(model, method):
        pytest.skip(f"{type(model).__name__} has no {method}")
    fast = fastleaves.Booster.from_lightgbm(model)
    expected = getattr(model, method)(rows, **OPTIONS[options])
    assert_identical(getattr(fast, method)(rows, **OPTIONS[options]), expected)


@pytest.mark.parametrize("name", MODELS)
def test_positional_arguments(name):
    """The model's order: X, raw_score, start_iteration, num_iteration, pred_leaf."""
    model, rows = fitted(name)
    fast = fastleaves.Booster.from_lightgbm(model)
    for args in [(True,), (False, 2, 3), (False, 0, None, True)]:
        assert_identical(fast.predict(rows, *args), model.predict(rows, *args))


def test_types():
    regressor, _ = fitted("regressor, early stopping")
    classifier, _ = fitted("classifier, string labels, early stopping")
    assert type(fastleaves.Booster.from_lightgbm(regressor)) is fastleaves.Estimator
    assert type(fastleaves.Booster.from_lightgbm(regressor.booster_)) is fastleaves.Booster
    fast = fastleaves.Booster.from_lightgbm(classifier)
    assert type(fast) is fastleaves.Classifier
    assert fast.best_iteration == classifier.best_iteration_ > 0
    assert list(fast.classes_) == ["no", "yes"] and fast.classes_ is not classifier.classes_


def test_not_fitted():
    with pytest.raises(NotFittedError):
        fastleaves.Booster.from_lightgbm(lightgbm.LGBMClassifier())


def test_multiclass_classifier_refused():
    x, y = frame(300, 0)
    model = lightgbm.LGBMClassifier(n_estimators=3, verbose=-1).fit(x, np.digitize(y, [0, 1]))
    with pytest.raises(UnsupportedError):
        fastleaves.Booster.from_lightgbm(model)
