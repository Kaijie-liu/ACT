"""Six fixed rational controls; no model, solver, random sample or phase search.

Hull mixtures are explicit mathematical witnesses, not a verification algorithm.
Masks mean unresolved by sufficient redundancy checks, never actual usefulness.
"""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d038_descendant_census_20260930 import amplitude as cap


def _iv(lower, upper=None):
    return F(lower), F(lower if upper is None else upper)


def _relu(value):
    return max(F(0), value)


def _compile(weights, bias, bounds, real_slots, **kwargs):
    return cap.compile_receiver(weights, bias, bounds, real_slots, enabled=True, **kwargs)


def _parent_hull(x, q, alpha):
    assert -1 <= x <= 1 and 0 <= alpha <= 1
    assert q >= 0 and q >= x and q <= alpha and q <= x + 1 - alpha


def _gate_state(preactivation, activation, phase):
    assert phase in (F(0), F(1)) and activation == _relu(preactivation)
    if phase == 0:
        assert preactivation <= 0 and activation == 0
    else:
        assert preactivation >= 0 and activation == preactivation


def _interval_rows(weight, residual, actual_weight, g, alpha, v):
    alo, ahi = weight
    lower, upper = residual
    assert alo <= actual_weight <= ahi and lower <= v <= upper
    q = _relu(g)
    _gate_state(g, q, alpha)
    h = actual_weight * q + v
    r, low, neg = _relu(h), _relu(lower), _relu(-upper)
    assert r >= low * (1 - alpha)
    assert r >= alo * q + lower * alpha + low * (1 - alpha)
    assert r - h >= neg * (1 - alpha)
    assert r - h >= -ahi * q - upper * alpha + neg * (1 - alpha)
    return h, r


def test_descendant_control_separates_complete_local_hulls():
    a, lower, upper = F(3, 2), F(-3, 5), F(-2, 5)
    compiled = _compile((_iv(a),), _iv(lower, upper), (_iv(-1, 1),), (True,))
    assert compiled['ordinary_bounds'] == _iv(F(-3, 5), F(11, 10))
    assert compiled['source_caps'] == (F(1),)
    assert compiled['residual_bounds'] == (_iv(lower, upper),)
    assert compiled['row_masks'] == (6,)
    assert compiled['row_unresolved_counts'] == (0, 1, 1, 0)
    assert compiled['potential_rows'] == 2 and compiled['potential_coordinate_nnz'] == 6

    # The parent label hull P includes the first old point by this true mixture.
    x, y, q, alpha = F(1, 10), F(0), F(2, 5), F(1, 2)
    assert (F(4, 5) + F(-3, 5)) / 2 == x
    assert (_relu(F(4, 5)) + _relu(F(-3, 5))) / 2 == q
    _parent_hull(x, q, alpha)

    # Each child source point is itself in FULL P, not merely the (q,y) box.
    # The two nested parent mixtures have strict signs and interior sources.
    source_points = (
        (F(8, 15), F(3, 10), F(7, 10), F(16, 21), F(-7, 9)),
        (F(4, 15), F(-1, 10), F(3, 10), F(8, 9), F(-11, 21)),
    )
    child_h, child_r = [], []
    for qi, xi, ai, active_x, inactive_x in source_points:
        assert 0 < active_x < 1 and -1 < inactive_x < 0
        assert ai * active_x + (1 - ai) * inactive_x == xi
        assert ai * _relu(active_x) + (1 - ai) * _relu(inactive_x) == qi
        _parent_hull(xi, qi, ai)
        child_h.append(a * qi - F(1, 2))
        child_r.append(_relu(child_h[-1]))
    assert tuple(child_h) == (F(3, 10), F(-1, 10))
    assert tuple(child_r) == (F(3, 10), F(0))
    assert sum(p[0] for p in source_points) / 2 == q
    assert sum(p[1] for p in source_points) / 2 == x
    assert sum(p[2] for p in source_points) / 2 == alpha
    h, r, beta = sum(child_h) / 2, sum(child_r) / 2, F(1, 2)
    assert h == a * q - F(1, 2) + y / 10 == F(1, 10)
    assert r == F(3, 20)

    # The unscaled cross-layer phase-difference factor also accepts the point.
    d, delta = q - r, x - h
    e = d - delta
    assert d == e == F(1, 4) and delta == 0
    assert -F(3, 5) * beta <= d <= F(3, 5) * alpha
    assert -F(3, 5) * (1 - beta) <= e <= F(3, 5) * (1 - alpha)
    # Existing single-edge D029 observations over the old affine source.
    source_lower, source_upper = F(-21, 10), F(11, 10)
    assert (source_lower - upper) * alpha <= a * q <= (source_upper - lower) * alpha
    assert lower + (source_lower - lower) * alpha <= h
    assert h <= upper + (source_upper - upper) * alpha
    assert r <= a * alpha and r - h <= -lower

    assert r < a * q + lower * alpha == F(3, 10)
    projected_lower = F(21, 10) * q - F(3, 5) * x - F(3, 5)
    assert r < projected_lower == F(9, 50)
    z = F(21, 10) * q - F(3, 5) * x - r
    assert z == F(63, 100) > F(123, 200) > F(3, 5)
    assert z - F(123, 200) == F(123, 200) - F(3, 5) == F(3, 200)
    # A true input attains the new bound; no bit is removed or identified.
    true_q = _relu(F(1))
    true_r = _relu(a * true_q - F(1, 2) - F(1, 10))
    assert F(21, 10) * true_q - F(3, 5) - true_r == F(3, 5)


def test_interval_contributions_and_all_but_one_residuals():
    weights = (_iv(1, 2), _iv(-3, -1), _iv(-1, 2), _iv(0), _iv(F(3, 2)))
    bounds = (_iv(-1, 2), _iv(F(1, 4), F(3, 2)), _iv(-2, F(3, 4)),
              _iv(-1, 1), _iv(-3, -1))
    bias = _iv(F(-1, 3), F(2, 5))
    result = _compile(weights, bias, bounds, (True,) * 5)
    # Independent four-product interval reference, not the optimized kernel rule.
    contributions = []
    for weight, bound in zip(weights, bounds):
        activation = (_relu(bound[0]), _relu(bound[1]))
        products = tuple(w * q for w in weight for q in activation)
        contributions.append((min(products), max(products)))
    expected = (bias[0] + sum(item[0] for item in contributions),
                bias[1] + sum(item[1] for item in contributions))
    assert expected == _iv(F(-67, 12), F(113, 20)) == result['ordinary_bounds']
    reference_residuals = tuple(
        (bias[0] + sum(item[0] for j, item in enumerate(contributions) if j != i),
         bias[1] + sum(item[1] for j, item in enumerate(contributions) if j != i))
        for i in range(5)
    )
    assert result['residual_bounds'] == reference_residuals
    assert reference_residuals[:3] == (
        _iv(F(-67, 12), F(33, 20)), _iv(F(-13, 12), F(59, 10)),
        _iv(F(-29, 6), F(83, 20)),
    )
    assert result['source_caps'] == (F(2), F(3, 2), F(3, 4), F(1), F(0))
    assert result['canonical_slots'] == result['valid_slots'] == 5
    assert result['padding_slots'] == 0
    assert result['zero_coeff_slots'] == result['interval_cross_zero_slots'] == 1
    empty = _compile((), bias, (), ())
    assert empty['ordinary_bounds'] == bias
    assert empty['source_caps'] == empty['residual_bounds'] == empty['row_masks'] == ()
    assert empty['canonical_slots'] == empty['potential_rows'] == empty['potential_coordinate_nnz'] == 0
    assert empty['row_unresolved_counts'] == (0, 0, 0, 0)


def test_padding_stable_sources_and_zero_phase_choices():
    result = _compile((_iv(2), _iv(-3), _iv(1), _iv(-2), _iv(0)),
        _iv(F(-1, 2), F(1, 2)),
        (_iv(1, 2), _iv(-3, -1), _iv(0), _iv(0), _iv(-2, 3)),
        (True, True, True, False, True))
    assert result['ordinary_bounds'] == _iv(F(3, 2), F(9, 2))
    assert result['source_caps'] == (F(2), F(0), F(0), F(0), F(3))
    assert result['canonical_slots'] == 5 and result['valid_slots'] == 4
    assert result['padding_slots'] == 1 and len(result['row_masks']) == 5
    assert result['row_masks'][3] == 0 and result['zero_coeff_slots'] == 1
    assert result['interval_cross_zero_slots'] == 0
    _gate_state(F(3, 2), F(3, 2), F(1))
    _gate_state(F(-2), F(0), F(0))
    # Real source exactly zero keeps its original two choices; padding has none.
    for alpha in (F(0), F(1)):
        _interval_rows(_iv(1), result['residual_bounds'][2], F(1), F(0), alpha, F(3))
    zero = _compile((_iv(1),), _iv(0), (_iv(0),), (True,))
    assert zero['source_caps'] == (F(0),) and zero['row_masks'] == (0,)
    # Four named zero-gate witnesses, not a search over network phases.
    for alpha, beta in ((F(0), F(0)), (F(0), F(1)), (F(1), F(0)), (F(1), F(1))):
        h, r = _interval_rows(_iv(1), _iv(0), F(1), F(0), alpha, F(0))
        _gate_state(h, r, beta)


def test_dense_fanin_redundancy_and_negative_edge():
    dense = _compile((_iv(1), _iv(1), _iv(-1), _iv(-1)), _iv(0),
                     (_iv(-1, 1),) * 4, (True,) * 4)
    assert dense['ordinary_bounds'] == _iv(-2, 2)
    assert dense['residual_bounds'] == (_iv(-2, 1), _iv(-2, 1), _iv(-1, 2), _iv(-1, 2))
    assert dense['row_masks'] == (0, 0, 0, 0)
    assert dense['potential_edges'] == dense['potential_rows'] == dense['potential_coordinate_nnz'] == 0
    negative = _compile((_iv(-2),), _iv(F(1, 4), F(1, 2)), (_iv(-1, 1),), (True,))
    assert negative['ordinary_bounds'] == _iv(F(-7, 4), F(1, 2))
    assert negative['row_masks'] == (9,) and negative['row_unresolved_counts'] == (1, 0, 0, 1)
    assert negative['potential_edges'] == 1 and negative['potential_rows'] == 2
    assert negative['potential_coordinate_nnz'] == 6
    positive = _compile((_iv(F(3, 2)),), _iv(F(-3, 5), F(-2, 5)), (_iv(-1, 1),), (True,))
    assert positive['row_masks'] == (6,) and positive['row_unresolved_counts'] == (0, 1, 1, 0)
    assert positive['potential_coordinate_nnz'] == 6
    inactive = _compile((_iv(F(1, 2)),), _iv(-1, F(-3, 4)), (_iv(-1, 1),), (True,))
    active = _compile((_iv(F(-1, 2)),), _iv(F(3, 4), 1), (_iv(-1, 1),), (True,))
    assert inactive['ordinary_bounds'][1] < 0 and inactive['row_masks'] == (0,)
    assert active['ordinary_bounds'][0] > 0 and active['row_masks'] == (0,)


def test_projected_extrema_and_interval_row_soundness():
    # Predetermined witnesses for both extrema of the three-row (q,alpha,r)
    # projection.  These convex mixtures are proofs, not a phase-split routine.
    regimes = ((F(3, 2), F(-3, 5), F(-2, 5), F(1)),
               (F(-2), F(1, 4), F(1, 2), F(1)),
               (F(0), F(-3, 10), F(2, 5), F(2)))
    for a, lower, upper, cap_q in regimes:
        l, u = _relu(lower), _relu(upper)
        slope = (_relu(a * cap_q + upper) - u) / cap_q
        for alpha, normalized_q in ((F(0), F(0)), (F(1, 2), F(1, 4)), (F(1), F(3, 4))):
            q = normalized_q * cap_q
            assert 0 <= q <= cap_q * alpha
            active_mean = q / alpha if alpha else F(0)
            low_witness = (1 - alpha) * l + alpha * _relu(a * active_mean + lower)
            endpoint_mass = q / cap_q
            high_witness = ((1 - alpha) * u + (alpha - endpoint_mass) * u
                            + endpoint_mass * _relu(a * cap_q + upper))
            assert low_witness == max(l * (1 - alpha), a * q + lower * alpha + l * (1 - alpha))
            assert high_witness == u + slope * q and low_witness <= high_witness
            middle = (low_witness + high_witness) / 2
            assert middle >= l * (1 - alpha)
            assert middle >= a * q + lower * alpha + l * (1 - alpha)
            assert middle <= u + slope * q

    fixtures = (
        (_iv(1, 2), _iv(F(-7, 10), F(-1, 5)), F(3, 2), F(3, 5), F(1), F(-2, 5)),
        (_iv(1, 2), _iv(F(-7, 10), F(-1, 5)), F(3, 2), F(-3, 5), F(0), F(-2, 5)),
        (_iv(-3, -1), _iv(F(1, 5), F(3, 5)), F(-2), F(1, 5), F(1), F(1, 2)),
        (_iv(-3, -1), _iv(F(1, 5), F(3, 5)), F(-2), F(3, 5), F(1), F(1, 2)),
        (_iv(-2, 3), _iv(F(-1, 2), F(3, 4)), F(-1), F(2, 5), F(1), F(1, 2)),
        (_iv(-2, 3), _iv(F(-1, 2), F(3, 4)), F(2), F(-3, 10), F(0), F(-1, 4)),
        (_iv(-3, -1), _iv(F(1, 5), F(3, 5)), F(-2), F(0), F(0), F(2, 5)),
        (_iv(-3, -1), _iv(F(1, 5), F(3, 5)), F(-2), F(0), F(1), F(2, 5)),
    )
    for weight, residual, actual_weight, g, alpha, v in fixtures:
        compiled = _compile((weight,), residual, (_iv(-1, 1),), (True,))
        assert compiled['residual_bounds'] == (residual,)
        h, _ = _interval_rows(weight, residual, actual_weight, g, alpha, v)
        assert compiled['ordinary_bounds'][0] <= h <= compiled['ordinary_bounds'][1]


def test_default_types_shapes_work_and_bits_fail_closed():
    disabled = cap.WorkBudget()
    assert cap.compile_receiver(None, None, None, None, budget=disabled) is None
    assert disabled.used == 0
    assert cap.compile_receiver(object(), object(), object(), object(), budget=object()) is None
    weights, bias, bounds, slots = (_iv(1),), _iv(0), (_iv(-1, 1),), (True,)
    with pytest.raises(cap.KernelError, match='enabled'):
        cap.compile_receiver(weights, bias, bounds, slots, enabled=1)
    with pytest.raises(cap.KernelDisabled):
        _compile(weights, bias, bounds, slots, budget=disabled)
    with pytest.raises(cap.KernelError, match='WorkBudget'):
        _compile(weights, bias, bounds, slots, budget=object())
    with pytest.raises(cap.BudgetExceeded):
        _compile(weights, bias, bounds, slots, budget=cap.WorkBudget(enabled=True, limit=0))
    invalid = (
        ([weights[0]], bias, bounds, slots),
        (weights, bias, [bounds[0]], slots),
        (weights, bias, bounds, [True]),
        (weights, bias, (), slots),
        (weights, bias, bounds, ()),
        (weights, bias, bounds, (1,)),
        (weights, bias, bounds, (False,)),
        (((F(1), 1.0),), bias, bounds, slots),
        (((1, F(1)),), bias, bounds, slots),
        ((_iv(2, 1),), bias, bounds, slots),
        (weights, _iv(1, -1), bounds, slots),
        (weights, bias, (_iv(1, -1),), slots),
        ((_iv(F(1 << 512)),), bias, bounds, slots),
        (weights, _iv(F(1, 1 << 512)), bounds, slots),
        (weights * (cap.MAX_SLOTS + 1), bias, bounds, slots),
    )
    for args in invalid:
        with pytest.raises(cap.KernelError):
            _compile(*args)
    # Valid 512-bit inputs can still produce an invalid product or total.
    with pytest.raises(cap.KernelError, match='bit bound'):
        _compile((_iv(F(1 << 511)),), bias, (_iv(-1, 2),), slots)
    with pytest.raises(cap.KernelError, match='bit bound'):
        _compile((_iv(F(1 << 511)),) * 2, bias, (_iv(-1, 1),) * 2, (True, True))
    with pytest.raises(cap.KernelError, match='bit bound'):
        _compile(weights, _iv(F(1 << 16)), bounds, slots,
                 budget=cap.WorkBudget(enabled=True, max_bits=16))
    inconsistent = cap.WorkBudget(enabled=True)
    inconsistent.used = -1
    with pytest.raises(cap.KernelError, match='ledger'):
        _compile(weights, bias, bounds, slots, budget=inconsistent)
    inconsistent.used, inconsistent.enabled = 0, 1
    with pytest.raises(cap.KernelError, match='ledger'):
        _compile(weights, bias, bounds, slots, budget=inconsistent)
    budget = cap.WorkBudget(enabled=True)
    budget.charge(11)
    _compile(weights, bias, bounds, slots, budget=budget)
    first = budget.used
    _compile(weights, bias, bounds, slots, budget=budget)
    assert 11 < first < budget.used <= budget.limit
    assert first - 11 <= 180 * len(weights) + 160
    assert budget.used - first <= 180 * len(weights) + 160
