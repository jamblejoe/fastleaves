# Contributing to fastleaves

fastleaves is Rust (PyO3, maturin, rayon) with Python tests. The README's guarantees and
scope table are the contract; everything here serves them.

## Setup

```bash
pip install maturin && pip install -e ".[test]"   # or: maturin develop --release
pytest                          # one worker per core; -n 0 for one process
FASTLEAVES_ISA=scalar pytest    # force the scalar path
cargo test --release            # kernel tests
```

On a CPU without AVX-512, `pytest` runs the scalar path only; CI covers the fast kernel on
Intel's emulator (see [tests/README.md](tests/README.md)).

## Code layout

- `src/model.rs` reads LightGBM model files.
- `src/forest.rs` bins inputs and holds the scalar kernel.
- `src/avx512.rs` is the SIMD kernel.
- `src/python.rs` and `python/fastleaves` are the Python API.
- `tests/` holds the Python tests, described in [tests/README.md](tests/README.md): every
  prediction against LightGBM's, and every refusal. The Rust tests, next to the code in
  `src/`, check what needs no LightGBM: the SIMD kernel against the scalar one, and the
  model reader.

## Rules

- The tests define correctness: comparison with LightGBM bit for bit, never approximate;
  generated cases (hypothesis) for edge-case inputs; and a test for every error in the
  scope table. Never weaken a test to make it pass.
- A mutation test checks that the tests catch changes to the Rust code: nightly for all of
  it, on pull requests for the changed lines. A new surviving mutant gets a test that
  catches it, or its code is removed. Only equivalent changes and checks that cannot
  trigger go in `tests/mutants-survivors.txt`, each with the reason.
- No tolerances, fast-math or changed summation order.
- Unsupported models, inputs and options raise an error; never fall back to LightGBM
  or return approximate numbers.
- Changes to the README's guarantees or scope need the maintainer's approval. A scope
  change updates the scope table, the tests and the changelog together.
- Speed claims cite an [LGBMBench](https://github.com/jamblejoe/LGBMBench) run.
- Prefer less code. No new dependencies without a reason in the commit message.
  `unsafe` only in the SIMD kernel and in reading the input buffer (`src/python.rs`).
- The Rust crate is not published; its only user is the Python extension. Code that
  neither the Python API nor the tests use is removed (add it back when something needs
  it). Checks that guard an invariant stay, even when no test can reach them.

## Commits and pull requests

- Before committing: `pytest`, `FASTLEAVES_ISA=scalar pytest` and `cargo test --release` pass.
- New features come with tests, including tests for what is refused.
- Commit subjects are short and imperative ("Add pred_leaf and validate_features").
- Add user-visible changes to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md).

## Releasing

1. Move the `Unreleased` entries in `CHANGELOG.md` under the new version.
2. Set the version in `Cargo.toml` (the only place it is written).
3. Re-run LGBMBench on the release commit and update the README's performance table.
4. Push a tag `vX.Y.Z`.

The release workflow (`.github/workflows/publish.yml`) builds one abi3 wheel for
Python 3.11+, checks that it installs and imports with only numpy, runs all tests on both
CPU paths against the installed wheel (AVX-512 on Intel's emulator), and publishes to PyPI
by trusted publishing. Run it by hand for a dry run without the upload.
