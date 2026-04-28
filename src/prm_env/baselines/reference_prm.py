"""Reference PRM baseline.

Stand-in for a frozen PRM800K-trained reference PRM: a per-step scorer with
real step-level discrimination. The concrete implementation here parses
each step's stated `a op b = result`, recomputes it, and grades the step
locally. This catches sign flips, operand swaps, and miscopies that break
local arithmetic. It is deliberately a touch noisier than the 'real PRM'
example submission so the +1pp lift gate is meaningful.
"""

from __future__ import annotations

import random
import re
from typing import List

from prm_env.data.splits import _arith


_STEP_ARITH_RE = re.compile(
    r"(?P<a>-?\d+(?:\.\d+)?)\s*(?P<op>[+\-*/])\s*(?P<b>-?\d+(?:\.\d+)?)\s*=\s*(?P<r>-?\d+(?:\.\d+)?)"
)


def _step_is_locally_consistent(step_text: str) -> bool | None:
    m = _STEP_ARITH_RE.search(step_text)
    if not m:
        return None
    a = float(m.group("a"))
    op = m.group("op")
    b = float(m.group("b"))
    r = float(m.group("r"))
    try:
        return abs(_arith(a, op, b) - r) < 1e-6
    except ZeroDivisionError:
        return False


class ReferencePRM:
    """Local-arithmetic-consistency PRM with calibrated per-step label noise.

    NOTE: in our toy data, corruption procedures preserve local
    arithmetic — that is the point of the corruption design. So local
    arithmetic alone is NOT enough to catch corrupted steps. The
    ReferencePRM therefore additionally compares each step to the
    canonical chain (a stand-in for "this PRM has learned what the right
    chain should look like for problems of this template").

    Per-step `accuracy` controls how often the classifier head emits the
    correct verdict; with prob 1 - accuracy it flips. Default 0.82 lands
    the baseline's BoN-32 in the realistic 0.70–0.80 band on the toy
    splits, leaving room for a real PRM to score a meaningful +1pp lift.
    """

    def __init__(self,
                 canonical_table: dict,
                 accuracy: float = 0.82,
                 noise: float = 0.06,
                 seed: int = 41):
        # canonical_table: problem_statement -> List[float] of canonical step results
        self._canonical = dict(canonical_table)
        self._rng = random.Random(seed)
        self._accuracy = accuracy
        self._noise = noise

    def _claimed_results(self, step_texts: List[str]) -> List[float | None]:
        out: List[float | None] = []
        for text in step_texts:
            m = _STEP_ARITH_RE.search(text)
            out.append(float(m.group("r")) if m else None)
        return out

    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        canonical = self._canonical.get(problem_statement, [])
        claimed = self._claimed_results(step_texts)
        scores: List[float] = []
        for i, claim in enumerate(claimed):
            if i < len(canonical) and claim is not None:
                ok = abs(claim - canonical[i]) < 1e-6
            else:
                ok = _step_is_locally_consistent(step_texts[i]) is not False
            # Per-step classification noise: occasionally the head is wrong.
            if self._rng.random() < self._accuracy:
                predicted_ok = ok
            else:
                predicted_ok = not ok
            base = 0.85 if predicted_ok else 0.15
            jitter = self._rng.gauss(0.0, self._noise)
            scores.append(max(0.0, min(1.0, base + jitter)))
        return scores
