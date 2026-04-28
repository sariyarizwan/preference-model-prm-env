"""Mechanical step-corruption procedures used to build the step-localization probe set.

The judge's central defense against the outcome-correlation shortcut is the
step-localization gate. To run that gate, we need a probe set of (correct
solution, corrupted step k) pairs where the corruption is local to step k
and propagates forward. A real PRM should drop its score at step k and
keep it low; an outcome-classifier in PRM clothing cannot do this.

Four corruption types are implemented:
  - sign_flip       : rewrite + as -, * as /, etc., on step k
  - operand_swap    : replace an operand of step k with a nearby wrong value
  - miscopy         : on step k, miscopy a previous step's result
  - off_by_one      : add or subtract a small delta from step k's result

Each corruption returns a CorruptedSolution with:
  - the rendered step texts (what the PRM scores)
  - the corruption position (ground-truth k)
  - per-step correctness labels (the calibration target)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from prm_env.data.splits import Solution, Step, _arith, _fmt


CORRUPTION_TYPES = ("sign_flip", "operand_swap", "miscopy", "off_by_one")


@dataclass(frozen=True)
class CorruptedSolution:
    """A solution where step `corruption_step` has been modified locally.

    Errors propagate: any subsequent step that consumed the original step's
    `result` as an operand is recomputed with the new (wrong) result.
    """
    step_texts: List[str]
    correctness: List[bool]    # per-step ground-truth correctness
    corruption_step: int       # 0-indexed position of the first wrong step
    corruption_kind: str
    final_answer: str          # what the corrupted solution claims as the final answer


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _opposite_op(op: str, rng: random.Random) -> str:
    if op == "+": return "-"
    if op == "-": return "+"
    if op == "*": return "/"
    if op == "/": return "*"
    raise ValueError(op)


def _nudge(x: float, rng: random.Random) -> float:
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    candidate = x + delta
    # Avoid producing the original value again
    if candidate == x:
        candidate += 1
    return candidate


def _render_step(idx: int, narrative: str, a: float, op: str, b: float, result: float) -> str:
    return f"Step {idx}: {narrative} {_fmt(a)} {op} {_fmt(b)} = {_fmt(result)}."


def _narrative_of(step: Step) -> str:
    """Recover the leading 'narrative' phrase from a Step's rendered text.

    The renderer in splits.py uses the form
        f"Step {idx}: {narrative} {a} {op} {b} = {result}."
    so we parse off the prefix and the trailing arithmetic."""
    text = step.text
    # Strip "Step N: " prefix
    after_colon = text.split(":", 1)[1].strip()
    # Strip trailing arithmetic " a op b = result."
    arithmetic = f"{_fmt(step.a)} {step.op} {_fmt(step.b)} = {_fmt(step.result)}."
    if after_colon.endswith(arithmetic):
        return after_colon[: -len(arithmetic)].strip()
    # Fallback: best-effort prefix
    return after_colon


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def corrupt(solution: Solution, step_idx: int, kind: str,
            rng: Optional[random.Random] = None) -> CorruptedSolution:
    """Return a CorruptedSolution where step `step_idx` is locally wrong.

    Errors propagate forward: any later step whose `a` or `b` equals the
    original step's `result` has that operand replaced with the corrupted
    `result`, and the later step is recomputed. Other later steps remain
    untouched.
    """
    if rng is None:
        rng = random.Random(0)
    if kind not in CORRUPTION_TYPES:
        raise ValueError(f"unknown corruption kind: {kind}")
    if step_idx < 0 or step_idx >= len(solution.steps):
        raise ValueError(f"step_idx out of range: {step_idx}")

    steps: List[Step] = list(solution.steps)
    target = steps[step_idx]
    narrative = _narrative_of(target)

    # Apply the corruption to the target step.
    if kind == "sign_flip":
        new_op = _opposite_op(target.op, rng)
        new_result = _arith(target.a, new_op, target.b)
        wrong_step = Step(
            text=_render_step(step_idx + 1, narrative, target.a, new_op, target.b, new_result),
            a=target.a, op=new_op, b=target.b, result=new_result,
        )

    elif kind == "operand_swap":
        # Wrong operand value, same operator. Flip a coin to choose a or b.
        if rng.random() < 0.5:
            new_a = _nudge(target.a, rng)
            new_result = _arith(new_a, target.op, target.b)
            wrong_step = Step(
                text=_render_step(step_idx + 1, narrative, new_a, target.op, target.b, new_result),
                a=new_a, op=target.op, b=target.b, result=new_result,
            )
        else:
            new_b = _nudge(target.b, rng)
            new_result = _arith(target.a, target.op, new_b)
            wrong_step = Step(
                text=_render_step(step_idx + 1, narrative, target.a, target.op, new_b, new_result),
                a=target.a, op=target.op, b=new_b, result=new_result,
            )

    elif kind == "miscopy":
        # The step copies the wrong intermediate forward: change `a` away from
        # whatever it currently is (this is a no-op at index 0 since there is
        # no prior result, so we fall back to operand_swap there).
        if step_idx == 0:
            return corrupt(solution, step_idx, "operand_swap", rng)
        new_a = _nudge(target.a, rng)
        new_result = _arith(new_a, target.op, target.b)
        wrong_step = Step(
            text=_render_step(step_idx + 1, narrative, new_a, target.op, target.b, new_result),
            a=new_a, op=target.op, b=target.b, result=new_result,
        )

    elif kind == "off_by_one":
        delta = rng.choice([-1, 1])
        new_result = target.result + delta
        wrong_step = Step(
            text=_render_step(step_idx + 1, narrative, target.a, target.op, target.b, new_result),
            a=target.a, op=target.op, b=target.b, result=new_result,
        )

    else:  # pragma: no cover
        raise AssertionError(kind)

    # Insert the wrong step.
    new_steps: List[Step] = steps[:step_idx] + [wrong_step]

    # Propagate the corrupted result forward through downstream steps that
    # actually depended on the original result.
    propagated_result = wrong_step.result
    propagated_from = target.result

    for j in range(step_idx + 1, len(steps)):
        s = steps[j]
        new_a, new_b = s.a, s.b
        if s.a == propagated_from:
            new_a = propagated_result
        if s.b == propagated_from:
            new_b = propagated_result
        new_result = _arith(new_a, s.op, new_b)
        # Re-render with the new operands and result, preserving the original narrative.
        narrative_j = _narrative_of(s)
        new_steps.append(Step(
            text=_render_step(j + 1, narrative_j, new_a, s.op, new_b, new_result),
            a=new_a, op=s.op, b=new_b, result=new_result,
        ))
        # If this step's result was itself consumed downstream, propagate its
        # change forward too.
        if new_result != s.result:
            propagated_from = s.result
            propagated_result = new_result
        else:
            # downstream chain heals; later steps remain unchanged
            propagated_from = float("nan")

    correctness = [True] * step_idx + [False] * (len(new_steps) - step_idx)
    final_answer = _fmt(new_steps[-1].result)
    return CorruptedSolution(
        step_texts=[s.render() for s in new_steps],
        correctness=correctness,
        corruption_step=step_idx,
        corruption_kind=kind,
        final_answer=final_answer,
    )


def build_probe_set(solutions: List[Solution], seed: int = 7) -> List[CorruptedSolution]:
    """Produce one corrupted variant per solution, cycling through corruption kinds.

    This is what the judge's step-localization gate iterates over.
    """
    rng = random.Random(seed)
    probes: List[CorruptedSolution] = []
    for i, sol in enumerate(solutions):
        kind = CORRUPTION_TYPES[i % len(CORRUPTION_TYPES)]
        # Corrupt a non-trivial step (avoid index 0 when possible so the
        # PRM has at least one correct prefix to score).
        if len(sol.steps) >= 2:
            step_idx = 1 + rng.randrange(len(sol.steps) - 1)
        else:
            step_idx = 0
        probes.append(corrupt(sol, step_idx, kind, rng))
    return probes
