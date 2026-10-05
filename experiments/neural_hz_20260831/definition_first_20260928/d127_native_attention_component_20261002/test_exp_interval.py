"""Six rational controls; no floating-point exponential serves as an oracle."""
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ex


def _check(interval):
    lo, hi = interval
    assert isinstance(lo, F) and isinstance(hi, F)
    assert F(0) < lo <= hi
    assert max(lo.numerator.bit_length(), lo.denominator.bit_length(),
               hi.numerator.bit_length(), hi.denominator.bit_length()) <= 512
    return lo, hi


def test_exp_zero_and_known_brackets():
    assert ex.exp_bounds(F(0)) == (F(1), F(1))
    # These loose rational brackets follow from the positive Taylor series
    # and a geometric bound on its tail, independently of floating point.
    for x, lower, upper in ((F(1), F(2718, 1000), F(2719, 1000)),
                            (F(1, 2), F(16487, 10000), F(16488, 10000)),
                            (F(-1), F(3678, 10000), F(3679, 10000))):
        lo, hi = _check(ex.exp_bounds(x))
        assert lower < lo <= hi < upper


def test_exp_negative_reciprocal():
    for x in (F(1, 8), F(1, 2), F(1), F(8)):
        lo, hi = _check(ex.exp_bounds(x))
        nlo, nhi = _check(ex.exp_bounds(-x))
        # Reciprocation must reverse endpoints, and final outward rounding
        # must not shrink the reciprocal enclosure supplied by the positive arm.
        assert nlo <= F(1)/hi <= F(1)/lo <= nhi
        assert nlo*lo <= 1 <= nhi*hi


def test_exp_monotone_enclosures():
    previous = None
    for x in (F(-2), F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(2)):
        interval = _check(ex.exp_bounds(x))
        if previous is not None:
            assert previous[1] < interval[0]
        previous = interval
    lo, hi = ex.exp_bounds(F(1, 3))
    assert F(1) + F(1, 3) < lo <= hi


def test_exp_dyadic_rounding():
    for x in (F(-1), F(-1, 3), F(0), F(1, 7), F(1)):
        lo, hi = _check(ex.exp_bounds(x))
        bits = 168 if x < 0 else 72
        for endpoint in (lo, hi):
            denominator = endpoint.denominator
            assert denominator & (denominator-1) == 0
            assert denominator <= 1 << bits
            assert endpoint*(1 << bits) == int(endpoint*(1 << bits))
        assert hi-lo < F(1, 1 << 60)
    # Exercise the directed-rounding primitive itself on a negative rational:
    # truncation toward zero would make its lower endpoint unsound.
    lo = ex._round(F(-1, 3), 72, False, ex.Budget())
    hi = ex._round(F(-1, 3), 72, True, ex.Budget())
    assert lo < F(-1, 3) < hi and hi-lo == F(1, 1 << 72)
    assert ex.exp_bounds(F(-1, 3))[1] < 1 < ex.exp_bounds(F(1, 3))[0]


def test_exp_range_reduction():
    lo, hi = _check(ex.exp_bounds(F(8)))
    assert F(2980) < lo <= hi < F(2982)
    nlo, nhi = _check(ex.exp_bounds(F(-8)))
    assert F(1, 2982) < nlo <= nhi < F(1, 2980)
    # e lies strictly between 2 and 3; raising these positive bounds
    # proves this wide independent bracket without a float oracle.
    lo, hi = _check(ex.exp_bounds(F(64)))
    assert F(2**64) < lo <= hi < F(3**64)
    nlo, nhi = _check(ex.exp_bounds(F(-64)))
    assert F(0) < nlo <= F(1)/hi <= F(1)/lo <= nhi


def test_exp_fail_closed():
    for bad in (0, 1, True, 0.5, float('inf'), float('nan'), None):
        with pytest.raises(ValueError):
            ex.exp_bounds(bad)
    for bad in (F(65), F(-65), F(1 << 512), F(1, 1 << 512)):
        with pytest.raises(ValueError):
            ex.exp_bounds(bad)
    for bad in (0, -1, True, 256_000_001):
        with pytest.raises(ValueError):
            ex.Budget(max_work=bad)
    budget = ex.Budget(max_work=1)
    with pytest.raises(ValueError):
        ex.exp_bounds(F(1), budget=budget)
    with pytest.raises(ValueError):
        ex.exp_bounds(F(1), budget=object())
