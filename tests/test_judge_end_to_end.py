"""End-to-end tests: each example submission triggers the predicted gate.

These tests are the empirical proof that the judge's gates do what the
writeup claims. If any of these tests start failing it means either a
gate has lost its teeth or a submission has accidentally become a real
PRM, both of which warrant investigation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prm_env.data.splits import load_splits
from prm_env.judge import Judge


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def judge():
    return Judge(splits=load_splits())


def _gate(result, name):
    matches = [g for g in result.gates if g.name == name]
    return matches[0] if matches else None


def test_real_prm_passes_all_gates(judge):
    result = judge.run(EXAMPLES / "real_prm_submission.py", short_circuit=False)
    failed = [g.name for g in result.gates if not g.passed]
    assert not failed, f"real PRM unexpectedly failed: {failed}"
    assert result.final_score > 0.0


def test_outcome_classifier_fails_step_localization(judge):
    result = judge.run(EXAMPLES / "outcome_classifier_submission.py", short_circuit=False)
    g = _gate(result, "step_localization_f1")
    assert g is not None
    assert not g.passed, f"step-loc unexpectedly passed for outcome classifier: {g.detail}"
    # And it should *pass* BoN-32 -- that's the whole point of the hack.
    bon_gate = _gate(result, "bon32_lift")
    assert bon_gate is not None and bon_gate.passed, (
        f"outcome classifier failed BoN-32 unexpectedly: {bon_gate.detail if bon_gate else 'missing'}"
    )
    assert result.final_score == 0.0


def test_uniform_fails_capability_floor(judge):
    result = judge.run(EXAMPLES / "uniform_submission.py", short_circuit=False)
    g = _gate(result, "capability_floor")
    assert g is not None and not g.passed
    assert result.final_score == 0.0


def test_threshold_gamer_fails_bon32(judge):
    result = judge.run(EXAMPLES / "threshold_gamer_submission.py", short_circuit=False)
    bon_gate = _gate(result, "bon32_lift")
    assert bon_gate is not None and not bon_gate.passed
    assert result.final_score == 0.0
