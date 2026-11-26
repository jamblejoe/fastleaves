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

`pytest` runs on one worker per core (pytest-xdist, `-n auto` in `pyproject.toml`); `-n 0`
runs it in one process, which is easier to debug and is how CI runs it on Intel's emulator
(SDE cannot follow the workers). `conftest.py` sets `OMP_NUM_THREADS=1`
unless it is already set, so the workers' LightGBM thread pools do not compete for the cores;
LightGBM's predictions do not depend on its thread count, and tests that pass `num_threads`
still set it. Each worker trains a model when it first needs it and caches it with its test
rows (`functools.cache` in `cases.py`).

CI runs the tests on Linux and Windows. GitHub's runners may lack AVX-512, so CI also runs the fast kernel on Intel's emulator (SDE,
installed by `petarpetrovt/setup-sde`): the kernel tests on every push, the whole suite nightly
(`.github/workflows/nightly.yml`). The release workflow runs the whole suite on both paths against the
installed wheel, on Linux and Windows (the emulated run on Linux only). `FASTLEAVES_REQUIRE_AVX512=1` makes both suites fail if the
fast kernel is not used.

A mutation test (`.github/workflows/mutants.yml`) checks the tests themselves: nightly for
all the Rust code, and on pull requests for the lines they change. cargo-mutants changes
the code in one place at a time (`<` to `<=`, a function body to a constant, ...) and runs
the Rust tests, the kernel on the emulator; `tests/mutants.py` runs pytest (scalar path)
on each change they miss. Some test must fail for every change, except those listed with
the reason in `tests/mutants-survivors.txt` (equivalent changes and unreachable checks). A
failing nightly run opens an issue "Nightly mutation test failed", or comments on the open
one, with the new survivors; the maintainer decides for each: a test that catches it, or
an entry in the list. Locally (the first command fails when it misses mutants; the second
judges them):

```bash
cargo mutants --cargo-arg=--release
```

```bash
python tests/mutants.py mutants.out/outcomes.json
```

| File | What it checks |
|---|---|
| `test_parity.py` | The parity matrix: objective × dataset × training × predict options. |
| `test_inputs.py` | Input forms (dtypes, layouts, lists, DataFrames, pandas categories), 255 categories on one feature (every category and unseen ones), 254 thresholds on one feature next to 255 categories (each missing-value mode, every threshold and its neighbors), the largest tree (32,768 leaves: a chain 32,767 levels deep and a complete tree, loaded on a 256 KB stack), `binary` sigmoids in the model file (including ones LightGBM's number parser reads one double off, and 22 digits after the decimal point), batch sizes (also with 100 features), models with many unused features, thread counts (also with `pred_leaf`), calls from several Python threads at once, model files (with `num_feature`, `num_trees`, `feature_name`; non-ASCII feature names), `isa()`, early stopping, `validate_features`, generated rows (hypothesis). |
| `test_sklearn.py` | scikit-learn models (`LGBMRegressor`, `LGBMClassifier` with string and integer labels): `predict` and `predict_proba` with each predict option and positional arguments, early stopping, unfitted and multiclass models, a scikit-learn model passed to `from_lightgbm` and a `Booster` to `from_sklearn`, the deprecated `Booster.from_lightgbm`. |
| `test_refusals.py` | One test for each refusal in the scope table. |
| `test_random_models.py` | 200 model files with random trees, written directly instead of trained (see below). |
| `cases.py` | The datasets, objectives, training settings, test rows and hand-written model files (a chain, a complete tree) the tests share. |
| `mutants.py`, `mutants-survivors.txt` | Not tests: the second stage of the nightly mutation test, and the mutants no test can catch (see above). |

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
| Objective | regression, regression_l1, huber, fair, quantile, mape, poisson, gamma, tweedie, binary, cross_entropy; sqrt regression (raw scores only); multiclass (refused) |
| Dataset | numeric; with NaN (missing type NaN); NaN with `use_missing=False` (type None); zero as missing (type Zero); categorical columns with NaN; large category IDs (0 to 29,000); many categories (about 170 in one feature's splits, next to one with 10); 100 features (two categorical, every tenth of the others constant; most are used in splits) |
| Training | default; deep trees; dart; goss; extra trees; no boost from average; linear trees (refused); random forest (refused) |
| Predict options | default; raw_score; num_iteration; start_iteration + num_iteration; start after the last iteration; 1 thread; pred_leaf; pred_leaf with start_iteration + num_iteration |

13 × 8 × 8 × 8 = 6,656 cases, of which 160 are combinations LightGBM cannot train.
`validate_features` is tested in `test_inputs.py` (it only checks
DataFrame column names).

### Planned

To be added to the matrix as features land or as more coverage is wanted.

| Dimension | Planned values |
|---|---|
| Objective | lambdarank, rank_xendcg (need query groups); multiclass, multiclassova; custom objective (no transformation) |
| Model shape | more than 254 thresholds (16-bit bins); single-leaf trees in trained models (random model files have them) |
| Training | monotone and interaction constraints; forced splits; path smoothing; `max_bin` 2 / 15 / 1023; categorical parameters (`max_cat_threshold`, `cat_smooth`, `max_cat_to_onehot`); `init_model` (continued training); `refit` |
| Predict options | `pred_early_stop`; `predict_disable_shape_check` |
| Inputs | sparse matrices; pyarrow and polars tables |
| Environment | each LightGBM 4.x release; model files saved by older 4.x releases; CPUs without AVX-512 in CI |

## Random model files

Trained models only have the shapes training produces. `random_model` in `test_random_models.py` writes
model files directly: 1 to 300 columns with up to 256 used, numerical features with up to
254 thresholds (edge values among them) and any missing type, categorical features with up
to 255 categories (IDs below 40, 256 or 30,000), 1 to 59 trees of 1 to 1,023 leaves, some
nearly chains, mixed `default_left`, leaf values of magnitude 1e-6 to 1e6 and ±0. LightGBM loads the
same file. Each model predicts 5,000 rows (more than one block), most values edge values of
their feature, with default options, `raw_score`, `pred_leaf` and a random iteration range.
Every model is within the scope table, so a refusal fails the test. The seed is the test ID.
