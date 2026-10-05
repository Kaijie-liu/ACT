"""Exact finite controls; the universal argument is in THEORY.md, not sampling."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys

import pytest

_spec = importlib.util.spec_from_file_location('d024_capacity_reference', Path(__file__).with_name('capacity.py'))
cap = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cap
_spec.loader.exec_module(cap)


def _frame():
    return cap.Frame('xy', (F(-1), F(-1)), (F(1), F(1)), (
        cap.AffineRow('xy', (F(1), F(1, 4)), F(0), 'beta'),
        cap.AffineRow('xy', (F(1), F(-1, 4)), F(0), 'eta')))


def _compile(**kwargs):
    return cap.compile_capacities(_frame(), (F(1), F(-11, 10)), F(1, 50), enabled=True, **kwargs)


def _relu(value):
    return max(F(0), value)


def _evaluate(row, point):
    return row.bias + sum(a * b for a, b in zip(row.coefficients, point))


def _mixture(row, points, weight):
    values = tuple(_evaluate(row, point) for point in points)
    assert values[0] > 0 > values[1]
    assert all(-1 < value < 1 for point in points for value in point)
    mean = tuple(weight * a + (1 - weight) * b for a, b in zip(*points))
    return mean, weight * _relu(values[0]) + (1 - weight) * _relu(values[1])


def _three_layer_control():
    first = _frame()
    beta, eta = F(7, 8), F(9, 10)
    mean_f, r = _mixture(first.rows[0], ((F(69, 70), F(-26, 35)), (F(-9, 10), F(-2, 5))), beta)
    mean_h, p = _mixture(first.rows[1], ((F(41, 45), F(-4, 5)), (F(-7, 10), F(1, 5))), eta)
    assert mean_f == mean_h == (F(3, 4), F(-7, 10))
    assert (r, p) == (F(7, 10), F(1))
    delta = _evaluate(first.rows[0], mean_f) - _evaluate(first.rows[1], mean_f)
    assert -F(1, 2) * eta <= r - p <= F(1, 2) * beta
    assert -F(1, 2) * (1 - eta) <= r - p - delta <= F(1, 2) * (1 - beta)
    second = cap.Frame('rp', (F(0), F(0)), (F(5, 4), F(5, 4)), (
        cap.AffineRow('rp', (F(1), F(-11, 10)), F(1, 50), 'alpha1'),
        cap.AffineRow('rp', (F(1), F(-6, 5)), F(3, 100), 'alpha2')))
    points = (
        ((F(59, 100), F(1, 10)), (F(291, 400), F(49, 40))),
        ((F(12, 25), F(1, 10)), (F(151, 200), F(49, 40))),
    )
    alpha = F(1, 5)
    outputs, preactivations = [], []
    for row, constituents in zip(second.rows, points):
        assert all(0 < coordinate < F(5, 4) for point in constituents for coordinate in point)
        assert all(abs(a - b) < F(1, 2) for a, b in constituents)
        assert tuple(alpha * a + (1 - alpha) * b for a, b in zip(*constituents)) == (r, p)
        values = tuple(_evaluate(row, point) for point in constituents)
        assert values[0] > 0 > values[1]
        t = alpha * _relu(values[0]) + (1 - alpha) * _relu(values[1])
        g = _evaluate(row, (r, p))
        old_capacity = cap.compile_capacities(first, row.coefficients, row.bias, enabled=True)
        assert t <= old_capacity.positive_constant + sum(a * b for a, b in zip(old_capacity.positive, (beta, eta)))
        assert t - g <= old_capacity.negative_constant + sum(a * b for a, b in zip(old_capacity.negative, (beta, eta)))
        outputs.append(t)
        preactivations.append(g)
    t1, t2 = outputs
    g1, g2 = preactivations
    assert (t1, t2, g1, g2) == (F(1, 10), F(39, 500), F(-19, 50), F(-47, 100))
    # Identical tighter scalar bounds from Q, on both sides of the comparison.
    for t, g, lo, hi in ((t1, g1, F(-121, 200), F(13, 25)),
                          (t2, g2, F(-18, 25), F(53, 100))):
        assert 0 <= t <= hi * alpha and 0 <= t - g <= -lo * (1 - alpha)
    # The stronger comparator also retains the entire current parent pair factor.
    parent_delta, parent_difference = g1 - g2, t1 - t2
    assert -F(1, 100) * alpha <= parent_difference <= F(23, 200) * alpha
    assert -F(23, 200) * (1 - alpha) <= parent_difference - parent_delta <= F(1, 100) * (1 - alpha)
    third_points = ((F(23, 200), F(1, 50)), (F(131, 1400), F(18, 175)))
    gamma = F(3, 10)
    assert all(0 < a < F(13, 25) and 0 < b < F(53, 100)
               and F(-1, 100) < a - b < F(23, 200) for a, b in third_points)
    assert tuple(gamma * a + (1 - gamma) * b for a, b in zip(*third_points)) == (t1, t2)
    kvals = tuple(F(-1, 100) + a - b for a, b in third_points)
    assert kvals == (F(17, 200), F(-27, 1400))
    k = gamma * kvals[0] + (1 - gamma) * kvals[1]
    v = gamma * _relu(kvals[0]) + (1 - gamma) * _relu(kvals[1])
    assert (k, v) == (F(3, 250), F(51, 2000))
    assert 0 <= v <= F(21, 200) * gamma and 0 <= v - k <= F(1, 50) * (1 - gamma)
    assert v <= F(13, 25) * alpha and v - k <= F(1, 100) + F(53, 100) * alpha
    newest = cap.compile_capacities(second, (F(1), F(-1)), F(-1, 100), enabled=True)
    assert newest.positive == (F(23, 200), F(0))
    assert newest.negative == (F(0), F(1, 100))
    assert newest.positive_constant == 0 and newest.negative_constant == F(1, 100)
    assert newest.pairs[0].difference_lower == F(-1, 100)
    assert newest.pairs[0].difference_upper == F(23, 200)
    assert newest.positive[0] * alpha < v
    ratio = F(23, 121)
    assert newest.positive[0] - ratio * F(121, 200) == 0
    assert ratio * F(121, 200) == F(23, 200) < F(29, 250)
    assert v + ratio * (t1 - g1) == F(28251, 242000) > F(29, 250)
    # Actual maximum attains the symbolic bound at x=1,y=-1.
    actual_r, actual_p = F(3, 4), F(5, 4)
    actual_g1, actual_g2 = (_evaluate(row, (actual_r, actual_p)) for row in second.rows)
    actual_v = _relu(F(-1, 100) + _relu(actual_g1) - _relu(actual_g2))
    assert actual_v + ratio * (_relu(actual_g1) - actual_g1) == F(23, 200)


def test_explicit_opt_in_and_unsupported_premises():
    with pytest.raises(cap.Rejected, match='enabled'):
        cap.compile_capacities(_frame(), (F(1), F(-1)), F(0))
    frame = _frame()
    stable = replace(frame, rows=(replace(frame.rows[0], bias=F(3)), frame.rows[1]))
    with pytest.raises(cap.Rejected, match='predecessor'):
        cap.compile_capacities(stable, (F(1), F(-1)), F(0), enabled=True)
    duplicate = replace(frame, rows=(frame.rows[0], replace(frame.rows[0], phase_id='other')))
    with pytest.raises(cap.Rejected, match='difference'):
        cap.compile_capacities(duplicate, (F(1), F(-1)), F(0), enabled=True)


def test_exact_common_source_certificates():
    result = _compile()
    assert result.predecessor_bounds == ((F(-5, 4), F(5, 4)),) * 2
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert (pair.difference_lower, pair.difference_upper) == (F(-1, 2), F(1, 2))
    assert (pair.positive_capacity, pair.negative_capacity) == (F(1, 2), F(1, 2))
    assert result.frame_id == 'xy' and result.phase_ids == ('beta', 'eta')


def test_static_pairing_and_unmatched_magnitudes():
    result = _compile()
    assert result.positive == (F(1, 2), F(0))
    assert result.negative == (F(0), F(5, 8))
    assert result.pairs[0].amount == 1
    assert result.added_nnz == 5
    frame = _frame()
    three = replace(frame, rows=(*frame.rows, replace(frame.rows[0], phase_id='third')))
    result = cap.compile_capacities(three, (F(1), F(2), F(-3)), F(0), enabled=True)
    assert not result.pairs  # No global matching or solver-driven pairing.
    assert result.positive == (F(5, 4), F(5, 2), F(0))
    assert result.negative == (F(0), F(0), F(15, 4))


def test_complete_old_hull_control_and_new_terminal_bound():
    result = _compile()
    frame = _frame()
    beta = eta = F(1, 5)
    mean_f, r = _mixture(frame.rows[0], ((F(24, 25), F(24, 25)), (F(-371, 400), F(-6, 25))), beta)
    mean_h, p = _mixture(frame.rows[1], ((F(19, 20), F(0)), (F(-37, 40), F(0))), eta)
    assert mean_f == mean_h == (F(-11, 20), F(0))
    assert (r, p) == (F(6, 25), F(19, 100))
    child = ((F(47, 100), F(1, 100)), (F(1, 100), F(37, 100)))
    assert all(0 < value < F(5, 4) for point in child for value in point)
    assert all(abs(a - b) < F(1, 2) for a, b in child)
    values = tuple(F(1, 50) + a - F(11, 10) * b for a, b in child)
    assert values == (F(479, 1000), F(-377, 1000))
    assert tuple((a + b) / 2 for a, b in zip(*child)) == (r, p)
    g, t = sum(values) / 2, sum(_relu(value) for value in values) / 2
    assert (g, t) == (F(51, 1000), F(479, 2000))
    delta = _evaluate(frame.rows[0], mean_f) - _evaluate(frame.rows[1], mean_f)
    difference = r - p
    assert -F(1, 2) * eta <= difference <= F(1, 2) * beta
    assert -F(1, 2) * (1 - eta) <= difference - delta <= F(1, 2) * (1 - beta)
    assert t <= F(1, 50) + F(5, 4) * beta  # Unpaired capacity passes.
    assert t - g <= F(11, 8) * eta
    positive_cap = result.positive_constant + result.positive[0] * beta
    negative_cap = result.negative_constant + result.negative[1] * eta
    assert positive_cap == F(3, 25) < t and negative_cap == F(1, 8) < t - g
    negative_f = r - _evaluate(frame.rows[0], mean_f)
    assert t + F(2, 5) * negative_f == F(1111, 2000) > F(11, 20)
    # Symbolic cancellation in t+(2/5)n_f gives a constant bound for every beta.
    assert result.positive[0] - F(2, 5) * F(5, 4) == 0
    bound = result.positive_constant + F(2, 5) * F(5, 4)
    assert bound == F(13, 25) < F(11, 20)
    x, y = F(1, 4), F(1)
    f, h = x + y / 4, x - y / 4
    actual = _relu(F(1, 50) + _relu(f) - F(11, 10) * _relu(h)) + F(2, 5) * (_relu(f) - f)
    assert actual == bound
    _three_layer_control()


def test_zero_preactivations_preserve_all_original_bits():
    result = _compile()
    # Both old preactivations are exactly zero: every original bit assignment is legal.
    for beta, eta in ((F(0), F(0)), (F(0), F(1)), (F(1), F(0)), (F(1), F(1))):
        t, g = F(1, 50), F(1, 50)
        assert t <= result.positive_constant + result.positive[0] * beta
        assert t - g <= result.negative_constant + result.negative[1] * eta
    assert result.phase_ids == ('beta', 'eta')


def test_negative_bias_and_reversed_sign_orientation():
    result = cap.compile_capacities(_frame(), (F(-11, 10), F(1)), F(-1, 50), enabled=True)
    assert (result.positive_constant, result.negative_constant) == (F(0), F(1, 50))
    assert result.positive == (F(0), F(1, 2))
    assert result.negative == (F(5, 8), F(0))
    assert (result.pairs[0].positive, result.pairs[0].negative) == (1, 0)
    for point in ((F(1, 2), F(1, 2)), (F(-1, 2), F(-1, 2)), (F(0), F(1, 2))):
        f, h = (_evaluate(row, point) for row in _frame().rows)
        beta, eta = F(int(f > 0)), F(int(h > 0))
        g = F(-1, 50) - F(11, 10) * _relu(f) + _relu(h)
        assert _relu(g) <= result.positive[1] * eta
        assert _relu(-g) <= result.negative_constant + result.negative[0] * beta


def test_work_bit_shape_and_identity_fail_closed():
    with pytest.raises(cap.Rejected, match='work'):
        _compile(work_limit=0)
    with pytest.raises(cap.Rejected, match='work'):
        _compile(work_limit=cap.WORK_CAP + 1)
    frame = _frame()
    broken_frames = (
        replace(frame, upper=(F(1),)),
        replace(frame, lower=(F(2), F(-1))),
        replace(frame, rows=(replace(frame.rows[0], frame_id='other'), frame.rows[1])),
        replace(frame, rows=(replace(frame.rows[0], frame_id=1), frame.rows[1])),
        replace(frame, rows=(frame.rows[0], replace(frame.rows[1], phase_id='beta'))),
        replace(frame, rows=(replace(frame.rows[0], bias=F(1 << 512)), frame.rows[1])),
        replace(frame, rows=(replace(frame.rows[0], coefficients=(F(1), 0.25)), frame.rows[1])),
        replace(frame, lower=(F(-1),) * 129, upper=(F(1),) * 129),
    )
    for broken in broken_frames:
        with pytest.raises(cap.Rejected):
            cap.compile_capacities(broken, (F(1), F(-1)), F(0), enabled=True)
    with pytest.raises(cap.Rejected, match='bit'):
        cap.compile_capacities(frame, (F((1 << 512) - 1), F(0)), F(0), enabled=True)
    assert _compile().reference_retained_entries < cap.ENTRY_CAP
