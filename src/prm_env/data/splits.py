"""Toy MATH-style problems, structured so step corruption is mechanical.

Real production: replace with the MATH dataset partitioned by problem id.
Here we generate a small deterministic set procedurally so the demo runs
on CPU in seconds.

Each Step carries (a, op, b, result) so corruption procedures can find
and rewrite the arithmetic in the rendered step text. The rendered text
is what a PRM actually scores.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(frozen=True)
class Step:
    text: str         # what the PRM sees, e.g. "Step 1: ... 5 * 3 = 15"
    a: float
    op: str           # one of: + - * /
    b: float
    result: float

    def render(self) -> str:
        return self.text


@dataclass(frozen=True)
class Solution:
    """A single fully correct step-by-step solution."""
    steps: List[Step]
    final_answer: str

    def render(self) -> str:
        return "\n".join(s.render() for s in self.steps) + f"\nFinal answer: {self.final_answer}"


@dataclass(frozen=True)
class Problem:
    problem_id: str
    statement: str
    gold_answer: str
    canonical_solution: Solution


@dataclass(frozen=True)
class Splits:
    """The judge consumes these splits read-only."""
    labeling_pool: List[Problem] = field(default_factory=list)
    capability: List[Problem] = field(default_factory=list)         # for the RewardBench-Math floor
    bon_eval: List[Problem] = field(default_factory=list)           # held-out MATH-Hard for BoN-32
    step_corruption: List[Problem] = field(default_factory=list)    # source for the step-loc probe
    calibration: List[Problem] = field(default_factory=list)

    def all_problems(self) -> List[Problem]:
        return [
            *self.labeling_pool,
            *self.capability,
            *self.bon_eval,
            *self.step_corruption,
            *self.calibration,
        ]


# ---------------------------------------------------------------------------
# Procedural problem generators. Each returns (statement, gold_answer, steps).
# Templates are deliberately simple so corruption is mechanical and the toy
# data remains parseable.
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    """Render a float as int when integral, else trimmed float."""
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def _arith(a: float, op: str, b: float) -> float:
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b
    raise ValueError(op)


def _step(idx: int, narrative: str, a: float, op: str, b: float) -> Step:
    result = _arith(a, op, b)
    text = f"Step {idx}: {narrative} {_fmt(a)} {op} {_fmt(b)} = {_fmt(result)}."
    return Step(text=text, a=a, op=op, b=b, result=result)


def _gen_shop(rng: random.Random, pid: str) -> Problem:
    n_w = rng.randint(2, 9)
    p_w = rng.randint(2, 12)
    n_g = rng.randint(2, 7)
    p_g = rng.randint(3, 15)
    statement = (
        f"A shop sells widgets and gizmos. Today it sold {n_w} widgets at ${p_w} each "
        f"and {n_g} gizmos at ${p_g} each. What was the total revenue, in dollars?"
    )
    s1 = _step(1, "Compute revenue from widgets:", n_w, "*", p_w)
    s2 = _step(2, "Compute revenue from gizmos:", n_g, "*", p_g)
    s3 = _step(3, "Add the two:", s1.result, "+", s2.result)
    answer = _fmt(s3.result)
    return Problem(pid, statement, answer, Solution([s1, s2, s3], answer))


def _gen_train(rng: random.Random, pid: str) -> Problem:
    v1 = rng.randint(20, 80)
    t1 = rng.randint(1, 5)
    v2 = rng.randint(20, 80)
    t2 = rng.randint(1, 5)
    statement = (
        f"A train travels at {v1} mph for {t1} hours, then at {v2} mph for {t2} hours. "
        "What is the total distance traveled, in miles?"
    )
    s1 = _step(1, "Distance for the first leg:", v1, "*", t1)
    s2 = _step(2, "Distance for the second leg:", v2, "*", t2)
    s3 = _step(3, "Total distance:", s1.result, "+", s2.result)
    answer = _fmt(s3.result)
    return Problem(pid, statement, answer, Solution([s1, s2, s3], answer))


def _gen_alice(rng: random.Random, pid: str) -> Problem:
    start = rng.randint(20, 80)
    given = rng.randint(3, start - 1)
    received = rng.randint(2, 25)
    extra = rng.randint(2, 15)
    statement = (
        f"Alice has {start} marbles. She gives {given} marbles to Bob, receives "
        f"{received} from Carol, then loses {extra} on the way home. How many marbles does she have?"
    )
    s1 = _step(1, "After giving to Bob:", start, "-", given)
    s2 = _step(2, "After receiving from Carol:", s1.result, "+", received)
    s3 = _step(3, "After losing on the way home:", s2.result, "-", extra)
    answer = _fmt(s3.result)
    return Problem(pid, statement, answer, Solution([s1, s2, s3], answer))


def _gen_runner(rng: random.Random, pid: str) -> Problem:
    base = rng.randint(2, 9)
    days_a = rng.randint(2, 6)
    factor = rng.choice([2, 3])
    days_b = rng.randint(2, 5)
    statement = (
        f"Bob runs {base} miles per day for {days_a} days, then {factor} times that pace for "
        f"{days_b} days. What is his total mileage?"
    )
    s1 = _step(1, "Miles in the first stretch:", base, "*", days_a)
    s2 = _step(2, "Pace in the second stretch:", base, "*", factor)
    s3 = _step(3, "Miles in the second stretch:", s2.result, "*", days_b)
    s4 = _step(4, "Total mileage:", s1.result, "+", s3.result)
    answer = _fmt(s4.result)
    return Problem(pid, statement, answer, Solution([s1, s2, s3, s4], answer))


def _gen_class(rng: random.Random, pid: str) -> Problem:
    classes = rng.randint(3, 8)
    students = rng.randint(15, 30)
    absent_per = rng.randint(1, 4)
    statement = (
        f"A school has {classes} classes, each with {students} enrolled students. "
        f"Today {absent_per} students were absent from each class. How many students were present in total?"
    )
    s1 = _step(1, "Total enrolled:", classes, "*", students)
    s2 = _step(2, "Total absent:", classes, "*", absent_per)
    s3 = _step(3, "Total present:", s1.result, "-", s2.result)
    answer = _fmt(s3.result)
    return Problem(pid, statement, answer, Solution([s1, s2, s3], answer))


def _generate_problems(seed: int, n: int) -> List[Problem]:
    rng = random.Random(seed)
    generators = [_gen_shop, _gen_train, _gen_alice, _gen_runner, _gen_class]
    out: List[Problem] = []
    for i in range(n):
        gen = generators[i % len(generators)]
        out.append(gen(rng, f"p{i:04d}"))
    return out


def load_splits(seed: int = 1729) -> Splits:
    """Generate the toy splits deterministically.

    Production: replace this function with one that loads MATH partitions
    from disk by problem id. The Splits dataclass contract is unchanged.
    """
    # 200 problems total. The eval splits are 40-each so BoN-32 has
    # 0.025 resolution (1/40), enough to see the +1pp lift gate without
    # making the demo slow.
    pool = _generate_problems(seed, 200)
    return Splits(
        labeling_pool=pool[:80],
        capability=pool[80:120],
        bon_eval=pool[120:160],
        step_corruption=pool[160:180],
        calibration=pool[180:200],
    )


def iter_step_texts(problems: Iterable[Problem]) -> Iterable[str]:
    for p in problems:
        for s in p.canonical_solution.steps:
            yield s.render()
