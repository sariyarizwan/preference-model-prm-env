"""OutcomeRM baseline.

Conceptually: a classifier head trained on (trajectory, final-correctness)
pairs that predicts whether the trajectory ends in a correct answer, then
exposes that single trajectory-level score uniformly across all step
positions.

For the laptop-runnable demo we model this as an outcome classifier with
calibrated accuracy: with probability `accuracy` it correctly classifies a
candidate trajectory's final correctness, and with probability 1 - accuracy
it does not. Light Gaussian jitter is layered on top so ties don't dominate
the BoN ranker. The default `accuracy=0.75` is set so the OutcomeRM
baseline lands in the realistic 65-75% BoN-32 band on the toy splits,
leaving room for real PRMs to demonstrate measurable lift.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List

from prm_env.symbolic_check import equivalent


_FINAL_ANSWER_RE = re.compile(r"=\s*(-?\d+(?:\.\d+)?)\s*\.?\s*$")


def _extract_claimed_answer(step_texts: List[str]) -> str | None:
    if not step_texts:
        return None
    last = step_texts[-1].strip()
    m = _FINAL_ANSWER_RE.search(last)
    return m.group(1) if m else None


class OutcomeRM:
    """Calibrated outcome classifier; uniform scores across steps."""

    def __init__(self,
                 gold_table: Dict[str, str],
                 accuracy: float = 0.75,
                 noise: float = 0.05,
                 seed: int = 17):
        self._gold = dict(gold_table)
        self._rng = random.Random(seed)
        self._accuracy = accuracy
        self._noise = noise

    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        gold = self._gold.get(problem_statement)
        claimed = _extract_claimed_answer(step_texts)
        actual_correct = bool(gold and claimed and equivalent(claimed, gold))
        # Bounded-accuracy classifier: occasionally wrong, simulating the
        # imperfection of a learned outcome head.
        if self._rng.random() < self._accuracy:
            predicted_correct = actual_correct
        else:
            predicted_correct = not actual_correct
        base = 0.92 if predicted_correct else 0.08
        jitter = self._rng.gauss(0.0, self._noise)
        s = max(0.0, min(1.0, base + jitter))
        return [s] * len(step_texts)
