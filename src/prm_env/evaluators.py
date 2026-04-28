"""Evaluators for the three judge gates that produce numeric scores.

Each evaluator consumes a PRMScorer (the agent's submission, the OutcomeRM
baseline, or the reference PRM) and a slice of the data splits, and returns
a small structured result the judge can consume.

The PRMScorer interface intentionally takes only the problem statement and
the step texts. It does NOT receive the gold answer or the canonical chain.
Submissions that rely on internal lookups must construct those at load
time, simulating a learned model's frozen parameters.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Sequence

from prm_env.corruptions import CORRUPTION_TYPES, CorruptedSolution, corrupt
from prm_env.data.splits import Problem, Solution
from prm_env.symbolic_check import equivalent


# ---------------------------------------------------------------------------
# The submission interface
# ---------------------------------------------------------------------------


class PRMScorer(Protocol):
    """The contract every submission must satisfy."""

    def score(self, problem_statement: str, step_texts: List[str]) -> List[float]:
        """Per-step scalar scores in [0, 1]; higher means correct prefix."""
        ...


# ---------------------------------------------------------------------------
# BoN-32
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    step_texts: List[str]
    final_answer: str
    is_correct: bool


@dataclass(frozen=True)
class BoNResult:
    accuracy: float
    per_problem: List[bool]
    aggregation_used: str


def generate_candidates(problem: Problem, n: int = 32, seed: int = 0) -> List[Candidate]:
    """Deterministic mix of correct and corrupted candidates.

    In production, candidates come from sampling the weak generator at fixed
    seed and temperature. Here we synthesize them by reusing the canonical
    solution and applying our corruption procedures, which keeps the toy
    eval CPU-friendly while preserving the structure of the BoN task.
    """
    # Deterministic per-problem seed; Python's builtin hash() is randomized
    # per-process so we use sha1 of the problem id for stability.
    pid_digest = int.from_bytes(
        hashlib.sha1(problem.problem_id.encode("utf-8")).digest()[:4], "big"
    )
    rng = random.Random(seed ^ pid_digest)
    out: List[Candidate] = []

    # 1/4 correct candidates (the canonical chain).
    canonical = problem.canonical_solution
    for _ in range(n // 4):
        out.append(Candidate(
            step_texts=[s.render() for s in canonical.steps],
            final_answer=canonical.final_answer,
            is_correct=True,
        ))

    # 3/4 corrupted candidates spread evenly across the four corruption kinds.
    n_wrong = n - len(out)
    for i in range(n_wrong):
        kind = CORRUPTION_TYPES[i % len(CORRUPTION_TYPES)]
        if len(canonical.steps) >= 2:
            step_idx = 1 + rng.randrange(len(canonical.steps) - 1)
        else:
            step_idx = 0
        c = corrupt(canonical, step_idx, kind, rng)
        out.append(Candidate(
            step_texts=list(c.step_texts),
            final_answer=c.final_answer,
            is_correct=equivalent(c.final_answer, problem.gold_answer),
        ))

    rng.shuffle(out)
    return out


def _aggregate(scores: Sequence[float], how: str) -> float:
    finite = [s for s in scores if not math.isnan(s)]
    if not finite:
        return float("nan")
    if how == "min":
        return min(finite)
    if how == "mean":
        return statistics.fmean(finite)
    if how == "last":
        return finite[-1]
    raise ValueError(how)


def evaluate_bon32(
    scorer: PRMScorer,
    problems: List[Problem],
    n: int = 32,
    seed: int = 1234,
) -> BoNResult:
    """Run BoN-N (default 32) and report best-of-three aggregations.

    Best-of-three across {min, mean, last} is returned. The judge uses this
    so that a PRM correct under one literature-standard aggregation but not
    another isn't unfairly penalized.
    """
    by_agg: Dict[str, List[bool]] = {"min": [], "mean": [], "last": []}
    for prob in problems:
        candidates = generate_candidates(prob, n=n, seed=seed)
        # Score every candidate once; reuse for all aggregations.
        per_candidate = [scorer.score(prob.statement, c.step_texts) for c in candidates]
        for how in by_agg:
            agg = [_aggregate(scores, how) for scores in per_candidate]
            top_idx = max(range(len(candidates)), key=lambda i: (agg[i], -i))
            by_agg[how].append(candidates[top_idx].is_correct)
    accuracies = {k: sum(v) / len(v) if v else 0.0 for k, v in by_agg.items()}
    best_how = max(accuracies, key=accuracies.get)
    return BoNResult(
        accuracy=accuracies[best_how],
        per_problem=by_agg[best_how],
        aggregation_used=best_how,
    )


# ---------------------------------------------------------------------------
# Step localization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepLocResult:
    f1: float
    precision: float
    recall: float
    n_probes: int
    n_localized: int


def _first_transition(scores: Sequence[float], threshold: float) -> int | None:
    """Return the index of the first above->below threshold transition.

    Crucially, we require that some earlier step was ABOVE the threshold.
    A uniformly-low scorer (every step under threshold from index 0) is NOT
    localizing anything, it is just saying 'this trajectory is bad' — that
    is exactly the outcome-classifier shortcut, and it should produce a
    null prediction here, not a free hit at index 0.
    """
    seen_above = False
    for i, s in enumerate(scores):
        if math.isnan(s):
            continue
        if s >= threshold:
            seen_above = True
            continue
        if seen_above:
            return i
    return None


def evaluate_step_localization(
    scorer: PRMScorer,
    probes: List[CorruptedSolution],
    problems: List[Problem],
    threshold: float = 0.5,
    tolerance: int = 1,
) -> StepLocResult:
    """Compute precision/recall/F1 of 'first step below threshold' against
    the ground-truth corruption position, with ±tolerance step slack.

    A probe is a hit when the PRM's first sub-threshold step is within
    `tolerance` of the corruption position. This is what catches the
    outcome-correlation shortcut described in section 5 of the writeup.
    """
    assert len(probes) == len(problems)
    n_predictions = 0
    n_relevant = len(probes)
    n_hits = 0

    for probe, prob in zip(probes, problems):
        scores = scorer.score(prob.statement, list(probe.step_texts))
        predicted = _first_transition(scores, threshold)
        if predicted is None:
            continue
        n_predictions += 1
        if abs(predicted - probe.corruption_step) <= tolerance:
            n_hits += 1

    precision = n_hits / n_predictions if n_predictions else 0.0
    recall = n_hits / n_relevant if n_relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return StepLocResult(
        f1=f1,
        precision=precision,
        recall=recall,
        n_probes=n_relevant,
        n_localized=n_predictions,
    )


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    n_steps: int


def evaluate_calibration(
    scorer: PRMScorer,
    problems: List[Problem],
    n_bins: int = 10,
    seed: int = 7,
) -> CalibrationResult:
    """ECE on a calibration split. We build per-step ground truth by mixing
    canonical-correct chains (all steps True) with corrupted variants
    (steps after the corruption are False). The PRM's predicted scores are
    binned into `n_bins` and ECE is the bin-weighted absolute gap between
    average predicted score and empirical accuracy.
    """
    rng = random.Random(seed)
    all_pred: List[float] = []
    all_correct: List[bool] = []

    for prob in problems:
        # Half correct, half corrupted variants.
        canonical = prob.canonical_solution
        all_pred += scorer.score(prob.statement, [s.render() for s in canonical.steps])
        all_correct += [True] * len(canonical.steps)
        if len(canonical.steps) >= 2:
            step_idx = 1 + rng.randrange(len(canonical.steps) - 1)
        else:
            step_idx = 0
        kind = CORRUPTION_TYPES[rng.randrange(len(CORRUPTION_TYPES))]
        corrupted = corrupt(canonical, step_idx, kind, rng)
        all_pred += scorer.score(prob.statement, list(corrupted.step_texts))
        all_correct += corrupted.correctness

    # Drop NaNs (from padding past true step count, defensive).
    pairs = [(p, c) for p, c in zip(all_pred, all_correct) if not math.isnan(p)]
    if not pairs:
        return CalibrationResult(ece=1.0, n_steps=0)

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_total: List[int] = [0] * n_bins
    bin_correct: List[int] = [0] * n_bins
    bin_score_sum: List[float] = [0.0] * n_bins

    for p, c in pairs:
        b = min(int(p * n_bins), n_bins - 1)
        bin_total[b] += 1
        bin_correct[b] += int(c)
        bin_score_sum[b] += p

    n = len(pairs)
    ece = 0.0
    for b in range(n_bins):
        if bin_total[b] == 0:
            continue
        acc = bin_correct[b] / bin_total[b]
        conf = bin_score_sum[b] / bin_total[b]
        ece += (bin_total[b] / n) * abs(acc - conf)

    return CalibrationResult(ece=ece, n_steps=n)


# ---------------------------------------------------------------------------
# RewardBench-Math style preference agreement (capability floor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityResult:
    agreement: float
    n_pairs: int


def evaluate_capability_agreement(
    scorer: PRMScorer,
    reference: PRMScorer,
    problems: List[Problem],
    n_pairs_per_problem: int = 4,
    seed: int = 11,
) -> CapabilityResult:
    """Build (chosen, rejected) preference pairs from canonical-correct
    chains vs corrupted chains, and measure how often the candidate PRM
    agrees with the reference PRM on which one is preferred."""
    rng = random.Random(seed)
    agreements = 0
    total = 0
    for prob in problems:
        canonical = prob.canonical_solution
        canonical_steps = [s.render() for s in canonical.steps]
        for _ in range(n_pairs_per_problem):
            kind = CORRUPTION_TYPES[rng.randrange(len(CORRUPTION_TYPES))]
            if len(canonical.steps) >= 2:
                step_idx = 1 + rng.randrange(len(canonical.steps) - 1)
            else:
                step_idx = 0
            corrupted = corrupt(canonical, step_idx, kind, rng)
            corrupted_steps = list(corrupted.step_texts)

            def aggregate_min(scorer_fn, steps):
                s = scorer_fn(prob.statement, steps)
                finite = [v for v in s if not math.isnan(v)]
                return min(finite) if finite else float("nan")

            scorer_pref = aggregate_min(scorer.score, canonical_steps) > aggregate_min(scorer.score, corrupted_steps)
            ref_pref = aggregate_min(reference.score, canonical_steps) > aggregate_min(reference.score, corrupted_steps)
            agreements += int(scorer_pref == ref_pref)
            total += 1
    return CapabilityResult(
        agreement=agreements / total if total else 0.0,
        n_pairs=total,
    )
