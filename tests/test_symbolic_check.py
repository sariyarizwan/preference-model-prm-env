"""Tests for the symbolic answer-equivalence checker."""

from __future__ import annotations

import pytest

from prm_env.symbolic_check import equivalent


@pytest.mark.parametrize("a, b", [
    ("7", "7"),
    ("7", "7.0"),
    ("0.5", "1/2"),
    ("\\frac{3}{4}", "0.75"),
    ("$42$", "42"),
    ("42 dollars", "42"),
    ("1,200", "1200"),
])
def test_equivalent_pairs(a, b):
    assert equivalent(a, b)
    assert equivalent(b, a)


@pytest.mark.parametrize("a, b", [
    ("7", "8"),
    ("0.5", "0.4"),
    ("\\frac{1}{2}", "\\frac{1}{3}"),
])
def test_inequivalent_pairs(a, b):
    assert not equivalent(a, b)
    assert not equivalent(b, a)


def test_unparseable_falls_back_to_string_eq():
    assert equivalent("the cat", "the cat")
    assert not equivalent("the cat", "the dog")


def test_none_inputs():
    assert not equivalent(None, "5")
    assert not equivalent("5", None)
    assert not equivalent(None, None)
