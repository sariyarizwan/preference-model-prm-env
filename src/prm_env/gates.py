"""Hard gates the judge applies before any continuous scoring.

A gate either passes or hard-fails the submission (score = 0). Gates run
in declared order so that cheap checks short-circuit the expensive ones.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# Gate 1: existence and signature
# ---------------------------------------------------------------------------


def gate_existence_and_signature(submission_path: Path) -> tuple[GateResult, Optional[Any], Optional[dict]]:
    """Check that the submission exposes load_prm() and MODEL_INFO with the
    right shape, and that load_prm() returns an object with a `score` method
    that takes (problem_statement, step_texts).
    """
    if not submission_path.exists():
        return GateResult("existence_and_signature", False, f"file not found: {submission_path}"), None, None

    try:
        spec = importlib.util.spec_from_file_location("submission", submission_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover - defensive
        return GateResult("existence_and_signature", False, f"failed to import: {e!r}"), None, None

    load_prm = getattr(module, "load_prm", None)
    if not callable(load_prm):
        return GateResult("existence_and_signature", False, "missing load_prm()"), None, None
    if any(p.default is inspect.Parameter.empty for p in inspect.signature(load_prm).parameters.values()):
        return GateResult("existence_and_signature", False, "load_prm() takes no required args"), None, None

    model_info = getattr(module, "MODEL_INFO", None)
    if not isinstance(model_info, dict):
        return GateResult("existence_and_signature", False, "missing MODEL_INFO dict"), None, None
    required_keys = {"base", "uses_lora", "trained_steps", "labeling_method"}
    missing = required_keys - set(model_info)
    if missing:
        return GateResult("existence_and_signature", False, f"MODEL_INFO missing keys: {sorted(missing)}"), None, None

    try:
        scorer = load_prm()
    except Exception as e:
        return GateResult("existence_and_signature", False, f"load_prm() raised: {e!r}"), None, None

    score_fn = getattr(scorer, "score", None)
    if not callable(score_fn):
        return GateResult("existence_and_signature", False, "scorer missing .score(...)"), None, None

    return GateResult("existence_and_signature", True, "ok"), scorer, model_info


# ---------------------------------------------------------------------------
# Gate 2: disallowed checkpoints (static AST analysis)
# ---------------------------------------------------------------------------


_ALWAYS_DISALLOWED_SUBSTRINGS = (
    "skywork-prm",
    "rlhflow-prm",
    "math-shepherd-mistral",
    "qwen2.5-math-prm",
    "qwen2.5-math-7b-prm",
)


def gate_disallowed_checkpoints(submission_path: Path,
                                disallowed: tuple[str, ...] = _ALWAYS_DISALLOWED_SUBSTRINGS) -> GateResult:
    """Best-effort static check: scan source for any string literal that
    matches a disallowed checkpoint substring (case-insensitive).

    A real production gate would also weight-hash the loaded module and
    compare to a ban-list of public PRM weights. We document that here
    and implement only the static scan, which catches the majority of
    naive cheats.
    """
    try:
        source = submission_path.read_text()
        tree = ast.parse(source)
    except Exception as e:  # pragma: no cover
        return GateResult("disallowed_checkpoints", False, f"parse error: {e!r}")

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lc = node.value.lower()
            for needle in disallowed:
                if needle in lc:
                    found.append(needle)
    if found:
        return GateResult("disallowed_checkpoints", False, f"references to disallowed checkpoint(s): {sorted(set(found))}")
    return GateResult("disallowed_checkpoints", True, "no disallowed references found")


# ---------------------------------------------------------------------------
# Gate 3: FLOP cap
# ---------------------------------------------------------------------------


def gate_flop_budget(flops_used: float, flop_cap: float) -> GateResult:
    if flops_used > flop_cap:
        return GateResult("flop_budget", False, f"used {flops_used:.2e} > cap {flop_cap:.2e}")
    return GateResult("flop_budget", True, f"used {flops_used:.2e} of {flop_cap:.2e}")
