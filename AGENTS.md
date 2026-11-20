# Instructions for coding agents

Read [README.md](README.md) (guarantees and scope) and [CONTRIBUTING.md](CONTRIBUTING.md)
(setup, layout, rules) before changing anything. Their rules apply to you in full; this
file only adds what is specific to agents.

## Stop and ask the maintainer

- Before changing the README's guarantees or scope table.
- When a test looks wrong. Never weaken, skip or delete it to get green.
- Before adding a dependency or `unsafe` outside the places CONTRIBUTING.md allows.

## Working style

- The maintainer reviews the Python tests closely, the Rust less so. Keep Rust simple and
  make every behavior change visible in a Python test.
- Before declaring work done, run `pytest`, `FASTLEAVES_ISA=scalar pytest`,
  `cargo test --release` and `cargo fmt --check`, and report failures as they are.
- The local CPU may lack AVX-512: say so when only the scalar path was tested.
- Never write performance numbers you did not take from an LGBMBench run.
- Keep README, CONTRIBUTING.md, tests/README.md and CHANGELOG.md in sync with the change.
