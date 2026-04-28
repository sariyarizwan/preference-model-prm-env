"""Symbolic answer-equivalence check, modeled on the PRM800K evaluator.

The PRM800K paper noted that string-equality on math final answers
under-counts correct responses (e.g. "1/2" vs "0.5", "\\frac{3}{4}" vs "0.75",
"7" vs "7.0"). This is the most common false-negative source in math evals
and is the reason the assessment writeup pins to the latest PRM800K-style
checker.

This implementation is a small but real subset of that behavior:
  - LaTeX \\frac stripping
  - whitespace, $ and braces stripping
  - decimal vs fraction equivalence via sympy
  - unit-suffix tolerance (drops trailing words)
"""

from __future__ import annotations

import re
from typing import Optional

import sympy


_UNIT_SUFFIXES = re.compile(
    r"\s*(dollars?|cents?|miles?|feet|inches|years?|months?|days?|"
    r"hours?|minutes?|seconds?|cm|m|km|kg|g|lbs?|%)\s*$",
    flags=re.IGNORECASE,
)
_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_BRACES_RE = re.compile(r"[{}]")


def _normalize(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = _FRAC_RE.sub(r"(\1)/(\2)", s)
    s = _BRACES_RE.sub("", s)
    s = _UNIT_SUFFIXES.sub("", s)
    s = s.replace("\\,", "")
    s = s.replace(",", "")  # drop thousands separators
    return s.strip()


def _to_sympy(s: str) -> Optional[sympy.Expr]:
    s = _normalize(s)
    if not s:
        return None
    try:
        return sympy.sympify(s, rational=True)
    except (sympy.SympifyError, SyntaxError, TypeError):
        return None


def equivalent(predicted: str, gold: str, tolerance: float = 1e-9) -> bool:
    """Return True iff predicted and gold answers are mathematically equivalent.

    Falls back to normalized string equality if symbolic comparison cannot
    parse one of the inputs.
    """
    if predicted is None or gold is None:
        return False
    p_norm = _normalize(predicted)
    g_norm = _normalize(gold)
    if p_norm == g_norm:
        return True
    p_expr = _to_sympy(predicted)
    g_expr = _to_sympy(gold)
    if p_expr is None or g_expr is None:
        return p_norm == g_norm
    try:
        diff = sympy.simplify(p_expr - g_expr)
        if diff == 0:
            return True
        # Numeric fallback for irrationals etc.
        try:
            return abs(float(diff)) <= tolerance
        except (TypeError, ValueError):
            return False
    except Exception:
        return p_norm == g_norm
