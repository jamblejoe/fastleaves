"""Second stage of the mutation test (nightly.yml). `cargo mutants` runs only the Rust tests,
which do not build src/python.rs and check less than the Python suite. Each mutant they miss
is applied to the source tree, the extension is rebuilt and pytest runs. The mutants pytest
misses too are printed; the script fails if one of them is not in mutants-survivors.txt.

Usage: python .github/mutants.py mutants.out/outcomes.json [pytest arguments]
"""

import json
import subprocess
import sys
from pathlib import Path


def apply(mutant):
    """Replaces the mutant's span (1-based lines and columns) by its replacement."""
    path = Path(mutant["file"])
    lines = path.read_text().split("\n")
    start, end = mutant["span"]["start"], mutant["span"]["end"]
    before = lines[start["line"] - 1][: start["column"] - 1]
    after = lines[end["line"] - 1][end["column"] - 1 :]
    lines[start["line"] - 1 : end["line"]] = [before + mutant["replacement"] + after]
    path.write_text("\n".join(lines))


def run(*command):
    return subprocess.run(command).returncode == 0


outcomes, pytest_args = json.loads(Path(sys.argv[1]).read_text())["outcomes"], sys.argv[2:]
missed = [o["scenario"]["Mutant"] for o in outcomes if o["summary"] == "MissedMutant"]
known = Path(__file__).with_name("mutants-survivors.txt").read_text().splitlines()  # `#`: comments
install = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."]

survivors = []
for i, mutant in enumerate(missed, 1):
    apply(mutant)
    try:
        if not run(*install):
            result = "unviable"
        # a crash counts as caught; a mutant that makes a loop endless times out (caught)
        elif not run("timeout", "1200", sys.executable, "-m", "pytest", "-x", "-q", *pytest_args):
            result = "caught"
        else:
            result = "MISSED"
            survivors.append(mutant["name"])
    finally:
        run("git", "checkout", "--", "src")
    print(f"[{i}/{len(missed)}] {result}: {mutant['name']}", flush=True)
run(*install)  # the unmutated extension again

print(f"\n{len(survivors)} of the {len(missed)} mutants the Rust tests miss survive pytest too:")
print("\n".join(survivors))
new = [s for s in survivors if s not in known]
if new:
    sys.exit(f"\n{len(new)} of them are not in .github/mutants-survivors.txt:\n" + "\n".join(new))
