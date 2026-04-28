"""Example submission #2: outcome-classifier in PRM clothing.

The central reward hack from §5: a PRM trained on auto-labels that has
degenerated into 'this trajectory ends in a correct answer' rather than
'this step is locally correct'. It returns a single trajectory-level
score broadcast uniformly across all step positions.

By design, this submission should:
  - PASS BoN-32 (because outcome ranking IS what BoN measures),
  - PASS the capability floor (it generally agrees with a real reference),
  - FAIL step-localization (no per-step discrimination at all).

That is the failure mode the step-localization gate exists to catch.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List

from prm_env.data.splits import load_splits
from prm_env.symbolic_check import equivalent


_FINAL_RESULT_RE = re.compile(r"=\s*(-?\d+(?:\.\d+)?)\s*\.?\s*$")


def _extract_claimed_final(step_texts: List[str]) -> str | None:
    if not step_texts:
        return None
    last = step_texts[-1].strip()
    m = _FINAL_RESULT_RE.search(last)
    return m.group(1) if m else None


class OutcomeClassifierInPRMClothing:
    """Slightly stronger outcome classifier than the OutcomeRM baseline.

    The point: this is what an over-trained PRM degenerates into when its
    auto-labels collapse to outcome correlations. It should still beat the
    OutcomeRM baseline on BoN-32 (because it's been trained on more data /
    larger base) but lacks any per-step signal.
    """

    def __init__(self,
                 gold_table: Dict[str, str],
                 accuracy: float = 0.97,
                 noise: float = 0.04,
                 seed: int = 99):
        self._gold = gold_table
        self._rng = random.Random(seed)
        self._accuracy = accuracy
        self._noise = noise

    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        gold = self._gold.get(problem_statement)
        claimed = _extract_claimed_final(step_texts)
        actual = bool(gold and claimed and equivalent(claimed, gold))
        if self._rng.random() < self._accuracy:
            predicted = actual
        else:
            predicted = not actual
        base = 0.95 if predicted else 0.05
        jitter = self._rng.gauss(0.0, self._noise)
        s = max(0.0, min(1.0, base + jitter))
        # Uniform broadcast across all steps -- this is the pathology.
        return [s] * len(step_texts)


def load_prm() -> OutcomeClassifierInPRMClothing:
    splits = load_splits()
    gold = {p.statement: p.gold_answer for p in splits.all_problems()}
    return OutcomeClassifierInPRMClothing(gold_table=gold)


MODEL_INFO = {
    "base": "qwen2.5-math-1.5b",
    "uses_lora": True,
    "trained_steps": 2500,
    "labeling_method": "outcome_only_relabel",
}
