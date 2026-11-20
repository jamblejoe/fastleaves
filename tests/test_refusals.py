"""Every refusal in the README's scope table. Unsupported models raise UnsupportedError
when they are loaded, unsupported inputs and options when predict is called. (Unsupported
objectives and training modes are also part of the matrix in test_parity.py.)"""

import lightgbm
import numpy as np
import pytest
import scipy.sparse

import fastleaves
from cases import split_categories, split_thresholds, thresholds_model, train

UnsupportedError = fastleaves.UnsupportedError


def wide_model():
    x = np.random.default_rng(0).normal(size=(2000, 300))
    params = {"verbose": -1, "num_leaves": 63, "min_data_in_leaf": 5}
    model = lightgbm.train(params, lightgbm.Dataset(x, x.sum(axis=1)), 15)
    assert (model.feature_importance() > 0).sum() > 256
    return model


def many_categories_model():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 257, size=(50_000, 1)).astype(float)
    y = rng.normal(size=257)[x[:, 0].astype(int)]  # an unrelated effect per category
    params = {"verbose": -1, "min_data_per_group": 5, "cat_smooth": 1, "max_cat_threshold": 128, "max_bin": 1023}
    model = lightgbm.train(params, lightgbm.Dataset(x, y, categorical_feature=[0]), 5)
    assert len(split_categories(model, 0)) == 256  # one more than supported
    return model


def many_thresholds_model():
    model = thresholds_model("None", 255)
    assert len(split_thresholds(model, 0)) == 255  # one more than supported
    return model


def old_version_model():
    return train("regression", "numeric", "default").model_to_string().replace("version=v4", "version=v3")


MODELS = {
    "more than 256 features used in splits": wide_model,
    "more than 255 categories on one feature": many_categories_model,
    "more than 254 thresholds per feature": many_thresholds_model,
    "multiclass": lambda: train("multiclass", "numeric", "default"),
    "random forest mode": lambda: train("regression", "numeric", "random forest"),
    "linear trees": lambda: train("regression", "numeric", "linear trees"),
    "model file version 3": old_version_model,
}


@pytest.mark.parametrize("model", MODELS)
def test_unsupported_model_fails_to_load(model):
    model = MODELS[model]()
    with pytest.raises(UnsupportedError):
        if isinstance(model, str):
            fastleaves.Booster(model_str=model)
        else:
            fastleaves.Booster.from_lightgbm(model)


def cross_entropy_lambda_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 6))
    return lightgbm.train({"verbose": -1, "objective": "cross_entropy_lambda"}, lightgbm.Dataset(x, rng.random(1000)), 5)


OUTPUTS = {
    "sqrt regression": lambda: train("sqrt regression", "numeric", "default"),
    "cross_entropy_lambda": cross_entropy_lambda_model,
}


@pytest.mark.parametrize("objective", OUTPUTS)
def test_unsupported_output_needs_raw_score(objective):
    booster = fastleaves.Booster.from_lightgbm(OUTPUTS[objective]())
    x = np.zeros((3, 6))
    with pytest.raises(UnsupportedError):
        booster.predict(x)
    booster.predict(x, raw_score=True)


def test_unsupported_option():
    booster = fastleaves.Booster.from_lightgbm(train("regression", "numeric", "default"))
    with pytest.raises(UnsupportedError):
        booster.predict(np.zeros((3, 6)), pred_contrib=True)


BAD_INPUTS = {
    # input: error, as LightGBM raises (ValueError) or refused by fastleaves (UnsupportedError)
    "sparse matrix": (scipy.sparse.csr_matrix(np.ones((3, 6))), UnsupportedError),
    "1D array": (np.zeros(6), ValueError),
    "too few columns": (np.zeros((3, 5)), ValueError),
    "too many columns": (np.zeros((3, 7)), ValueError),
}


@pytest.mark.parametrize("data", BAD_INPUTS)
def test_bad_input(data):
    booster = fastleaves.Booster.from_lightgbm(train("regression", "numeric", "default"))
    x, error = BAD_INPUTS[data]
    with pytest.raises(error):
        booster.predict(x)
