"""`Booster`: a LightGBM model, `from_lightgbm`, and its inputs converted as LightGBM converts them."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from ._core import Model as _Model
from ._core import UnsupportedError


class Booster:
    """A LightGBM model for prediction, from a model file or string as `lightgbm.Booster`
    takes them; `from_lightgbm` and `from_sklearn` copy a LightGBM object in memory.
    Unsupported models raise `UnsupportedError`."""

    def __init__(self, model_file=None, model_str=None):
        if (model_file is None) == (model_str is None):
            raise TypeError("pass exactly one of model_file and model_str")
        if model_file is not None:
            model_str = Path(model_file).read_text(encoding="utf-8")  # as LightGBM writes it
        self._model = _Model(model_str)
        self.pandas_categorical = _pandas_categorical(model_str)
        self.best_iteration = -1

    @staticmethod
    def from_lightgbm(booster):
        """Deprecated: use `fastleaves.from_lightgbm`."""
        warnings.warn(
            "Booster.from_lightgbm is deprecated; use fastleaves.from_lightgbm", FutureWarning, stacklevel=2
        )
        return from_lightgbm(booster)

    def num_feature(self):
        return self._model.num_features

    def num_trees(self):
        return self._model.num_trees

    def feature_name(self):
        return self._model.feature_names

    def predict(
        self,
        data,
        start_iteration=0,
        num_iteration=None,
        raw_score=False,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        num_threads=0,
    ):
        """Same arguments and results as `lightgbm.Booster.predict`: float64 predictions, or
        with `pred_leaf=True` an int32 matrix of leaf indices, one column per tree.

        `data` is a 2D numpy array, list of lists or pandas DataFrame. `validate_features`
        checks a DataFrame's column names against the model's. `num_threads` <= 0 uses all
        cores.
        """
        if pred_contrib:
            raise UnsupportedError("not supported: pred_contrib=True")
        if validate_features and _is_dataframe(data):
            self._validate_features(data)
        if num_iteration is None:  # as LightGBM
            num_iteration = self.best_iteration if start_iteration <= 0 else -1
        n = self._model.num_trees  # one tree per iteration
        start = min(max(start_iteration, 0), n)
        count = n - start if num_iteration <= 0 else min(num_iteration, n - start)
        x = _to_float64(data, self.pandas_categorical)
        threads = max(int(num_threads), 0)
        if pred_leaf:
            out = np.empty((len(x), count), dtype=np.int32)
            self._model.predict_leaf(x, out, start, start + count, threads)
        else:
            out = np.empty(len(x), dtype=np.float64)
            self._model.predict(x, out, start, start + count, bool(raw_score), threads)
        return out

    def _validate_features(self, df):
        """LightGBM's `ValidateFeatureNames`, with its messages."""
        names, expected = [str(c) for c in df.columns], self.feature_name()
        if len(names) != len(expected):
            raise ValueError(
                f"Model was trained on {len(expected)} features, but got {len(names)} input features to predict."
            )
        for i, (e, name) in enumerate(zip(expected, names)):
            if e != name:
                raise ValueError(f"Expected '{e}' at position {i} but found '{name}'")

def from_lightgbm(booster):
    """Copy of a `lightgbm.Booster`."""
    if hasattr(type(booster), "booster_"):
        raise TypeError("a scikit-learn model: use fastleaves.from_sklearn(model)")
    copy = Booster(model_str=booster.model_to_string(num_iteration=-1))
    copy.best_iteration = booster.best_iteration
    return copy


def _pandas_categorical(model_str):
    """The category lists LightGBM saved for pandas categorical columns, or None."""
    tail = model_str[model_str.rfind("\npandas_categorical:") :]
    if not tail.startswith("\npandas_categorical:"):
        return None
    return json.loads(tail.split(":", 1)[1].strip().splitlines()[0])


def _to_float64(data, pandas_categorical):
    """`data` as an aligned, C-contiguous float64 matrix, converted exactly as LightGBM converts it."""
    if _is_dataframe(data):
        data = _from_pandas(data, pandas_categorical)
    elif isinstance(data, list):
        data = np.array(data)
    if not isinstance(data, np.ndarray):
        raise UnsupportedError(f"not supported: input of type {type(data).__name__}")
    if data.ndim != 2:
        raise ValueError("Input numpy.ndarray or list must be 2 dimensional")
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)  # LightGBM reads other dtypes as float32
    return np.require(data, np.float64, ["C", "A"])


def _is_dataframe(data):
    """Whether `data` is a pandas DataFrame, without importing pandas."""
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(data, pd.DataFrame)


def _from_pandas(df, pandas_categorical):
    """LightGBM's `_data_from_pandas`: categorical columns become their codes, -1 becomes NaN."""
    pd = sys.modules["pandas"]
    if df.ndim != 2 or df.shape[0] < 1:
        raise ValueError("Input data must be 2 dimensional and non empty.")
    df = df.copy(deep=False)
    cat_cols = [c for c, t in zip(df.columns, df.dtypes) if isinstance(t, pd.CategoricalDtype)]
    if pandas_categorical is not None:
        if len(cat_cols) != len(pandas_categorical):
            raise ValueError("train and valid dataset categorical_feature do not match.")
        for col, categories in zip(cat_cols, pandas_categorical):
            if list(df[col].cat.categories) != list(categories):
                df[col] = df[col].cat.set_categories(categories)
    if cat_cols:
        df[cat_cols] = df[cat_cols].apply(lambda c: c.cat.codes).replace({-1: np.nan})
    dtype = np.result_type(*[t.type for t in df.dtypes], np.float32)
    return df.to_numpy(dtype=dtype)
