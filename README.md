# fastleaves

Fast batch prediction for LightGBM models on CPU, with results identical to LightGBM.

```bash
pip install fastleaves
```

```python
import fastleaves

booster = fastleaves.Booster(model_file="model.txt")   # or model_str=..., or Booster.from_lightgbm(b)
pred = booster.predict(X, num_threads=8)                # X: numpy array or pandas DataFrame
```

`predict` takes the same arguments as `lightgbm.Booster.predict` and returns the same
float64 values, bit for bit (int32 leaf indices with `pred_leaf=True`). It is built for
thousands to millions of rows per call; small batches work but are not optimized. Refusals
(see the scope below) raise `fastleaves.UnsupportedError`, a `ValueError`; inputs that
LightGBM rejects too raise LightGBM's `ValueError`.

### scikit-learn models

`from_lightgbm` also copies a fitted `LGBMRegressor` or `LGBMClassifier` (binary). The copy's
`predict` and `predict_proba` take the model's arguments and return what the model's return:

```python
fast = fastleaves.Booster.from_lightgbm(model)
fast.predict(X)         # == model.predict(X), class labels for a classifier
fast.predict_proba(X)   # == model.predict_proba(X), classifiers only
```

The early-stopping iteration carries over; `n_jobs` does not, so pass `num_threads`.
A copy saved with `model.booster_.save_model` and loaded with `Booster` predicts like
`model.booster_`: the probability of the second class, no labels.

## Guarantees

1. **Identical results.** Output equals `lightgbm.Booster.predict` (`numpy.array_equal`)
   for every supported model, input, option, thread count and CPU. Reference: LightGBM 4.x.
2. **No silent fallback.** Unsupported models raise an error at load time; unsupported
   inputs and options raise at call time. LightGBM is not needed at runtime.

## Scope

| | Supported | Raises an error |
|---|---|---|
| Model files | LightGBM 4.x | older versions |
| Splits | numerical; categorical with up to 255 distinct categories per feature (any category IDs) | more than 255 categories on one feature; numerical and categorical splits on one feature (continued training, see below) |
| Size | up to 256 features used in splits (any number of input columns), 254 thresholds per feature (enough for LightGBM's default `max_bin=255`), 32,768 leaves per tree | larger models |
| Missing values | all modes (`None`, `NaN`, `Zero`) | different modes on one feature (continued training, see below) |
| Output | `regression`, `regression_l1`, `huber`, `fair`, `quantile`, `mape`, `poisson`, `gamma`, `tweedie`, `binary`, `cross_entropy`; any single-output objective with `raw_score=True` | `sqrt` regression and `cross_entropy_lambda` outputs, a `binary` sigmoid with more than 22 digits after the decimal point (LightGBM writes at most 9), random-forest mode, multiclass, linear trees |
| Input | 2D numpy (float64, float32, integer); pandas DataFrame incl. categorical columns | sparse matrices, 1D input |
| `predict` options | `start_iteration`, `num_iteration`, `raw_score`, `pred_leaf`, `validate_features`, `num_threads` (default: all cores) | `pred_contrib`, other options |
| Platform | Linux and Windows x86-64, Python 3.11+; fast path on AVX-512 CPUs, slower scalar path elsewhere | macOS |

Continued training (`init_model`) on data that differs from the first run can give one
feature splits of two kinds or missing-value modes, for example when the new data has NaN
where the old had none. fastleaves refuses such models.

NaN, ±inf, ±0 and unseen categories behave as in LightGBM. `predict` releases the GIL.

Not planned: training, SHAP, GPU, XGBoost/CatBoost models, single-row latency.
Next: models with more thresholds, then multiclass.

## Performance

[LGBMBench](https://github.com/jamblejoe/LGBMBench) run of 2026-11-18
([results/fremtpl2.json](https://github.com/jamblejoe/LGBMBench/blob/main/results/fremtpl2.json),
LGBMBench 36b711d, a pre-release build of fastleaves): a 1,500-tree Poisson model with 9 features on a
4-core AVX-512 Xeon. Median time per call:

| Engine | Bit-exact | 1M rows, 1 thread | 1M rows, 4 threads | 10k rows, 4 threads | 1 row, 1 thread | Setup |
|---|---|---:|---:|---:|---:|---:|
| **fastleaves** | yes | **11.5 s** | **3.07 s** | 58.2 ms | 469 µs | 80 ms |
| lignum | yes | 16.3 s | 4.19 s | 45.6 ms | 53 µs | 19.6 s |
| lleaves | no (≤ 2e-14) | 18.3 s | 4.57 s | 63.8 ms | 91 µs | compile, cached |
| lgbmjit | yes | 20.5 s | 5.23 s | 52.6 ms | 89 µs | 2.7 s |
| onnxruntime | no (≤ 4e-7) | 70.7 s | 27.5 s | 312 ms | 164 µs | conversion, cached |
| tl2cgen | yes | 140 s | 35.9 s | 380 ms | 258 µs | compile, cached |
| LightGBM | — | 228 s | 56.7 s | 610 ms | 323 µs | 41 ms |

## How it works

Each input value becomes its position among the model's split thresholds, so comparisons
are small integer compares. Only the columns the model splits on are read. AVX-512 runs 16 rows through a tree at once without branching;
a lane takes a new row as soon as its row reaches a leaf. Rows are processed in cache-sized
blocks, one tree at a time, and summed in LightGBM's order.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
