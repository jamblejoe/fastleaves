"""Copies of LightGBM's scikit-learn models (`LGBMRegressor`, `LGBMClassifier`, ...)."""

import numpy as np

from ._booster import Booster


def from_sklearn(model):
    """Copy of a fitted scikit-learn model of LightGBM: a `Classifier` for an `LGBMClassifier`,
    an `Estimator` for others (`LGBMRegressor`, ...)."""
    if not hasattr(type(model), "booster_"):
        hint = ": use fastleaves.Booster.from_lightgbm" if hasattr(model, "model_to_string") else ""
        raise TypeError(f"not a scikit-learn model of LightGBM: {type(model).__name__}{hint}")
    return (Classifier if hasattr(type(model), "classes_") else Estimator)(model)


class Estimator(Booster):
    """A scikit-learn model of LightGBM (`LGBMRegressor`, ...) for prediction, made by
    `from_sklearn`. `predict` takes the model's arguments in its order and returns what its
    `predict` returns."""

    def __init__(self, model):
        booster = model.booster_  # raises if the model is not fitted
        super().__init__(model_str=booster.model_to_string(num_iteration=-1))
        self.best_iteration = booster.best_iteration

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        num_threads=0,
    ):
        return super().predict(
            X,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            num_threads=num_threads,
        )


class Classifier(Estimator):
    """A binary `lightgbm.LGBMClassifier` for prediction, made by `from_sklearn`: `predict`
    returns its class labels and `predict_proba` its class probabilities, as the model's do.
    `classes_` are its labels."""

    def __init__(self, model):
        super().__init__(model)
        self.classes_ = model.classes_.copy()

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        num_threads=0,
    ):
        proba = self.predict_proba(
            X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, num_threads
        )
        if raw_score or pred_leaf:
            return proba
        return self.classes_[np.argmax(proba, axis=1)]  # as LightGBM

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        num_threads=0,
    ):
        p = super().predict(
            X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, num_threads
        )
        if raw_score or pred_leaf:
            return p
        return np.vstack((1.0 - p, p)).transpose()  # as LightGBM

