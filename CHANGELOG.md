# Changelog

All notable changes to fastleaves. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.7.0] - 2026-11-28

- Calls with more than 4,096 rows split them evenly among all threads, in blocks of 256 to
  4,096 rows. Before, threads took whole blocks of 4,096 rows, so 8,192 rows used only 2 threads.
- `fastleaves.from_sklearn(model)` copies a fitted scikit-learn model of LightGBM. An
  `LGBMClassifier` (binary) becomes a `fastleaves.Classifier`, whose `predict` returns class
  labels and `predict_proba` class probabilities; others (`LGBMRegressor`, ...) become a
  `fastleaves.Estimator`. Both take `predict`'s arguments in the model's order.
  The README's first guarantee (identical results) covers their `predict` and `predict_proba`.
- `fastleaves.from_lightgbm(booster)` copies a `lightgbm.Booster`. It refuses scikit-learn
  models and points to `from_sklearn`.
- Deprecated: `Booster.from_lightgbm`. It still works, with a `FutureWarning`; use
  `fastleaves.from_lightgbm`. It will be removed in a later release.
- The README's scope table states an existing input: a list of lists.

## [0.6.0] - 2026-11-24

- `binary` and `cross_entropy` models predict probabilities (were raw scores only). A
  `binary` model's sigmoid is read from the model file exactly as LightGBM reads it; a sigmoid
  with more than 22 digits after the decimal point is refused (raw scores only).

## [0.5.0] - 2026-11-23

- Windows x86-64 is supported: a wheel for Windows is published, and CI runs all tests on
  Windows too.
- Model files are read as UTF-8, as LightGBM writes them. Before, feature names with
  non-ASCII characters were garbled on systems whose default encoding is not UTF-8
  (Windows), so `validate_features` rejected the matching DataFrame columns.
- Loading a model with a very deep tree (up to 32,767 levels, as LightGBM allows) no longer
  crashes: model trees are read and freed with loops instead of recursion.
- The README's scope table states two existing refusals: trees with more than 32,768 leaves,
  and features whose splits mix kinds or missing-value modes (from continued training on
  different data).
- Batches of up to 4096 rows are predicted on the calling thread, without starting threads.
- Thread pools are kept between calls instead of being started on every call (one per
  `num_threads` value). Calls from several Python threads at once share a pool.

## [0.4.0] - 2026-11-20

- Numerical features may have up to 254 thresholds in the model's splits (was 253), so
  models trained with LightGBM's default `max_bin=255` load. Inputs are still binned to one
  byte: a missing value now shares its bin value with an unseen category.

## [0.3.0] - 2026-11-20

- Categorical features may have up to 255 categories in the model's splits (was 32).
  Features with up to 32 keep the previous fast path; larger category sets take one more
  memory lookup per split.

## [0.2.0] - 2026-11-20

- Models may use up to 256 features in their splits (was 12 features in total). Features
  the model never splits on are not read and do not count toward the limit.

## [0.1.0] - 2026-11-20

First release.

- `Booster` loads LightGBM 4.x models from a file, a string or a `lightgbm.Booster`.
- `predict` returns the same float64 values as `lightgbm.Booster.predict`, bit for bit,
  for regression-type objectives and any single-output objective with `raw_score=True`.
- Numerical and categorical splits (up to 32 categories per feature, any category IDs),
  up to 12 features and 253 thresholds per feature, all missing-value modes.
- Inputs: 2D numpy arrays (float64, float32, integer) and pandas DataFrames, including
  categorical columns, read through the buffer protocol.
- Options: `start_iteration`, `num_iteration`, `raw_score`, `pred_leaf`,
  `validate_features`, `num_threads`.
- AVX-512 kernel with a scalar fallback; `predict` releases the GIL.
- Unsupported models, inputs and options raise an error.
- One abi3 wheel for Linux x86-64, Python 3.11+.
