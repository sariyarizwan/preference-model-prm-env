"""Tests for the step-corruption probe generator."""

from __future__ import annotations

import random

import pytest

from prm_env.corruptions import CORRUPTION_TYPES, build_probe_set, corrupt
from prm_env.data.splits import load_splits


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_corruption_changes_target_step_only_at_first(splits):
    sol = splits.labeling_pool[0].canonical_solution
    rng = random.Random(0)
    for kind in CORRUPTION_TYPES:
        c = corrupt(sol, step_idx=1, kind=kind, rng=rng)
        # Steps before corruption are byte-identical
        for i in range(1):
            assert c.step_texts[i] == sol.steps[i].render()
        # The corruption step itself differs
        assert c.step_texts[1] != sol.steps[1].render()


def test_correctness_labels_have_right_shape(splits):
    sol = splits.labeling_pool[0].canonical_solution
    rng = random.Random(0)
    c = corrupt(sol, step_idx=1, kind="sign_flip", rng=rng)
    assert len(c.correctness) == len(c.step_texts)
    assert all(c.correctness[:1])
    assert not any(c.correctness[1:])


def test_off_by_one_changes_only_result(splits):
    sol = splits.labeling_pool[0].canonical_solution
    target = sol.steps[1]
    c = corrupt(sol, step_idx=1, kind="off_by_one", rng=random.Random(0))
    # The corrupted step text should still mention the original a and b.
    txt = c.step_texts[1]
    assert f"{int(target.a) if target.a == int(target.a) else target.a}" in txt
    assert f"{int(target.b) if target.b == int(target.b) else target.b}" in txt


def test_sign_flip_actually_flips(splits):
    sol = splits.labeling_pool[0].canonical_solution
    target = sol.steps[1]
    c = corrupt(sol, step_idx=1, kind="sign_flip", rng=random.Random(0))
    txt = c.step_texts[1]
    # Original op should not appear in same position; the opposite op should.
    flipped_lookup = {"+": "-", "-": "+", "*": "/", "/": "*"}
    assert f" {flipped_lookup[target.op]} " in txt


def test_propagation_changes_downstream_results(splits):
    sol = splits.labeling_pool[0].canonical_solution
    if len(sol.steps) < 3:
        pytest.skip("need >=3 steps for propagation test")
    c = corrupt(sol, step_idx=1, kind="off_by_one", rng=random.Random(0))
    # Step 2 should differ from canonical because step 1's result propagated.
    assert c.step_texts[2] != sol.steps[2].render()


def test_build_probe_set_round_trip(splits):
    canonical_solutions = [p.canonical_solution for p in splits.step_corruption]
    probes = build_probe_set(canonical_solutions, seed=42)
    assert len(probes) == len(canonical_solutions)
    for probe, sol in zip(probes, canonical_solutions):
        assert len(probe.step_texts) == len(sol.steps)
        assert 0 <= probe.corruption_step < len(sol.steps)
        assert probe.corruption_kind in CORRUPTION_TYPES


def test_probe_set_is_deterministic(splits):
    sols = [p.canonical_solution for p in splits.step_corruption]
    a = build_probe_set(sols, seed=42)
    b = build_probe_set(sols, seed=42)
    for x, y in zip(a, b):
        assert x.step_texts == y.step_texts
        assert x.corruption_step == y.corruption_step
        assert x.corruption_kind == y.corruption_kind
