# Tests

Every test compares fastleaves with LightGBM bit for bit (`assert_bit_identical`: same
dtype, shape and bits), or checks that fastleaves refuses what the README's scope table
refuses.

```bash
maturin develop --release && pytest            # fast path if the CPU has AVX-512
FASTLEAVES_ISA=scalar pytest                    # scalar path
cargo test --release                            # kernel against a direct tree walk
```

The report header shows which kernel ran (`fastleaves kernel: avx512` or `scalar`).

GitHub's runners may lack AVX-512, so CI also runs the fast kernel on Intel's emulator (SDE,
installed by `petarpetrovt/setup-sde`): the kernel tests on every push, the whole suite nightly
(`.github/workflows/nightly.yml`). The release workflow runs the whole suite on both paths against the
installed wheel. `FASTLEAVES_REQUIRE_AVX512=1` makes both suites fail if the
fast kernel is not used.

| File | What it checks |
|---|---|
| `test_parity.py` | The parity matrix: objective × dataset × training × predict options. |
| `test_inputs.py` | Input forms (dtypes, layouts, lists, DataFrames, pandas categories), 255 categories on one feature (every category and unseen ones), 254 thresholds on one feature next to 255 categories (each missing-value mode, every threshold and its neighbors), batch sizes (also with 100 features), models with many unused features, thread counts (also with `pred_leaf`), model files, early stopping, `validate_features`, generated rows (hypothesis). |
| `test_refusals.py` | One test for each refusal in the scope table. |
| `cases.py` | The datasets, objectives, training settings and test rows the tests share. |

## The parity matrix

Each test trains a small LightGBM model (1,000 rows, 20 trees, cached). It then predicts
on 1,000 unseen rows plus edge rows. In an edge row, one feature takes one of: NaN, ±inf,
±0, values around LightGBM's zero threshold (±1e-35f), categories −1 to 33, 199, 200 and
254 to 256 and fractions of them, huge values, **every split threshold of the model along with the next double
below and above it**, and every category in the model's splits with its neighbors (±0.5, ±1). `why_unsupported` in `test_parity.py` decides from the scope table
whether a case must match or be refused. Combinations LightGBM cannot train are skipped
with LightGBM's message.

### Implemented now

| Dimension | Values |
|---|---|
| Objective | regression, regression_l1, huber, fair, quantile, mape, poisson, gamma, tweedie; binary, cross_entropy, sqrt regression (raw scores only); multiclass (refused) |
| Dataset | numeric; with NaN (missing type NaN); NaN with `use_missing=False` (type None); zero as missing (type Zero); categorical columns with NaN; large category IDs (0 to 29,000); many categories (about 170 in one feature's splits, next to one with 10); 100 features (two categorical, every tenth of the others constant; most are used in splits) |
| Training | default; deep trees; dart; goss; extra trees; no boost from average; linear trees (refused); random forest (refused) |
| Predict options | default; raw_score; num_iteration; start_iteration + num_iteration; start after the last iteration; 1 thread; pred_leaf; pred_leaf with start_iteration + num_iteration |

13 × 8 × 8 × 8 = 6,656 cases, of which 160 are combinations LightGBM cannot train.
They run in about 100 s. `validate_features` is tested in `test_inputs.py` (it only checks
DataFrame column names).

### Planned

To be added to the matrix as features land or as more coverage is wanted.

| Dimension | Planned values |
|---|---|
| Objective | lambdarank, rank_xendcg (need query groups); binary/cross_entropy probabilities; multiclass, multiclassova; custom objective (no transformation) |
| Model shape | more than 254 thresholds (16-bit bins); single-leaf trees |
| Training | monotone and interaction constraints; forced splits; path smoothing; `max_bin` 2 / 15 / 1023; categorical parameters (`max_cat_threshold`, `cat_smooth`, `max_cat_to_onehot`); `init_model` (continued training); `refit` |
| Predict options | `pred_early_stop`; `predict_disable_shape_check` |
| Inputs | sparse matrices; pyarrow and polars tables |
| Environment | each LightGBM 4.x release; model files saved by older 4.x releases; CPUs without AVX-512 in CI |
