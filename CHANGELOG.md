# Changelog

All notable changes to fastleaves. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
