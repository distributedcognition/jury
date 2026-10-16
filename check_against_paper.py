"""Compare the tables the scripts wrote to tables/ with the paper's copies in
reference/tables. Prints one line per table and exits non-zero on any difference
larger than a tenth of a point.

    python check_against_paper.py

A table is SAME when it matches line for line. It is CLOSE when every line has
the same text and the same numbers up to at most TOL in absolute value: that is
the footprint of a different CPU architecture or BLAS (see README.md, "Checking
against the paper"), where the iterative solvers behind MoRE, LLM-TOPLA and the
answer scorer, and a jury posterior that sits on a threshold, can move one
question in 1560. Anything else is DIFFERS.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 0.1 + 1e-6            # the paper's tables are printed to one decimal
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
PRODUCED_BY = {
    "experts": "run_paper_results.py", "reachability": "run_paper_results.py",
    "main": "run_paper_results.py", "bysource": "run_paper_results.py",
    "decomposition": "run_paper_results.py", "specialisation": "run_paper_results.py",
    "override": "run_paper_results.py", "testimony": "testimony_structure.py",
    "support": "support_control.py",
    "refit": "refit_bootstrap.py", "skillaggregation": "skillaggregation_mc.py",
    "pool_replication": "pool_replication.py", "multistart": "em_multistart.py",
}


def max_numeric_gap(a: list[str], b: list[str]) -> float | None:
    """Largest absolute difference between corresponding numbers, or None when
    the two tables differ in anything other than the numbers' values."""
    if len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if NUMBER.sub("#", x) != NUMBER.sub("#", y):
            return None
        for u, v in zip(NUMBER.findall(x), NUMBER.findall(y)):
            worst = max(worst, abs(float(u) - float(v)))
    return worst


def main() -> int:
    same, close, bad = 0, 0, 0
    for name, script in PRODUCED_BY.items():
        ref = ROOT / "reference" / "tables" / f"{name}.tex"
        got = ROOT / "tables" / f"{name}.tex"
        if not got.exists():
            print(f"  MISSING  {name:<18} run scripts/{script}")
            bad += 1
            continue
        a, b = ref.read_text().splitlines(), got.read_text().splitlines()
        if a == b:
            print(f"  SAME     {name:<18} {script}")
            same += 1
            continue
        gap = max_numeric_gap(a, b)
        if gap is not None and gap <= TOL:
            print(f"  CLOSE    {name:<18} {script}   (numbers within {gap:.1f} of the paper)")
            close += 1
        else:
            print(f"  DIFFERS  {name:<18} {script}")
            bad += 1
        for line in difflib.unified_diff(a, b, "reference", "produced", lineterm="", n=0):
            if not line.startswith(("---", "+++", "@@")):
                print(f"           {line}")
    n = len(PRODUCED_BY)
    print(f"\n{same}/{n} tables identical to the paper"
          + (f", {close} within {TOL:.1f} of it (platform rounding, see README.md)" if close else "")
          + (f", {bad} differ" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
