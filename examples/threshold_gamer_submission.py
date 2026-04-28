"""Example submission #4: threshold-shape gamer.

This is the third reward hack from the writeup: if the agent reverse-engineers
the 'first step below 0.5' rule for step-localization, it could train the
PRM to drop sharply at every step boundary regardless of correctness, faking
the localization shape. As predicted in section 5, this kills BoN-32 because
candidate trajectories become indistinguishable under the aggregation step
of the BoN ranker.
"""

from __future__ import annotations

from typing import List


class ThresholdGamer:
    """Always drops below 0.5 at step index 1 regardless of step content."""

    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        # Pretty pattern that fakes step-localization: high, then a sharp
        # drop after step 0, regardless of whether anything is wrong.
        out: List[float] = []
        for i in range(len(step_texts)):
            out.append(0.9 if i == 0 else 0.1)
        return out


def load_prm() -> ThresholdGamer:
    return ThresholdGamer()


MODEL_INFO = {
    "base": "qwen2.5-math-1.5b",
    "uses_lora": True,
    "trained_steps": 1200,
    "labeling_method": "step_shape_fake",   # honest naming
}
