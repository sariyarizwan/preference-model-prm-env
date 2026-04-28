"""Example submission #3: uniform 0.5 scorer.

Trivially-bad submission. Designed to fail the capability floor (which
exists precisely to stop submissions like this from reaching the more
sensitive evals downstream).
"""

from __future__ import annotations

from typing import List


class UniformScorer:
    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        return [0.5] * len(step_texts)


def load_prm() -> UniformScorer:
    return UniformScorer()


MODEL_INFO = {
    "base": "qwen2.5-math-1.5b",
    "uses_lora": False,
    "trained_steps": 0,
    "labeling_method": "none",
}
