"""End-to-end demo.

Runs the judge against four example submissions and prints a verdict matrix.
Each submission is designed to trigger a specific failure mode predicted by
the writeup's reward-hacking analysis (section 5):

  real_prm                    -> passes everything
  outcome_classifier          -> fails step-localization (Hack #2)
  uniform                     -> fails capability floor
  threshold_gamer             -> fails BoN-32 lift (Hack #3)

The demo exercises the same judge code that production would use; the
short_circuit=False flag is the only difference, and exists so the verdict
table can show every gate's outcome instead of stopping at the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prm_env.data.splits import load_splits  # noqa: E402
from prm_env.judge import Judge  # noqa: E402


SUBMISSIONS = [
    ("real PRM",                        REPO / "examples" / "real_prm_submission.py"),
    ("outcome-classifier-in-disguise",  REPO / "examples" / "outcome_classifier_submission.py"),
    ("uniform 0.5 scorer",              REPO / "examples" / "uniform_submission.py"),
    ("threshold-shape gamer",           REPO / "examples" / "threshold_gamer_submission.py"),
]

GATE_ORDER = [
    "existence_and_signature",
    "disallowed_checkpoints",
    "flop_budget",
    "capability_floor",
    "bon32_lift",
    "step_localization_f1",
    "calibration_ece",
]


def _row(label: str, result) -> str:
    by_name = {g.name: g for g in result.gates}
    cells: list[str] = []
    for name in GATE_ORDER:
        g = by_name.get(name)
        if g is None:
            cells.append("  -  ")
        else:
            cells.append(" PASS" if g.passed else " FAIL")
    return f"{label:<35} | {' | '.join(cells)} | {result.final_score:.3f}"


def main() -> int:
    judge = Judge(splits=load_splits())

    header_cells = [n[:8] for n in GATE_ORDER]
    print()
    print(f"{'submission':<35} | " + " | ".join(f"{h:>5}" for h in header_cells) + " | score")
    print("-" * (37 + len(GATE_ORDER) * 8 + 8))

    results = []
    for label, path in SUBMISSIONS:
        result = judge.run(path, short_circuit=False)
        results.append((label, result))
        print(_row(label, result))
    print()

    # Per-submission diagnostic detail.
    for label, result in results:
        print(f"== {label} ==")
        for g in result.gates:
            flag = "PASS" if g.passed else "FAIL"
            print(f"  [{flag}] {g.name}: {g.detail}")
        print(f"  Final continuous score: {result.final_score:.4f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
