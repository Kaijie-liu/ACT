"""Nine bounded rational controls; no solver, network run, or random search."""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d030_joint_support_reference_20260930 import joint_support as support


def _iv(lower, upper=None):
    return F(lower), F(lower if upper is None else upper)


def _form(constant=0, coefficients=None):
    return _iv(constant), {key: _iv(value) for key, value in (coefficients or {}).items()}


def _compile(forms, weights, affine, box, **kwargs):
    return support.compile_support(forms, weights, affine, box, enabled=True, **kwargs)


def _point_form(form):
    assert form[0][0] == form[0][1]
    assert all(value[0] == value[1] for value in form[1].values())
    return form[0][0], {key: value[0] for key, value in form[1].items()}


def _scale(form, scalar):
    return form[0] * scalar, {key: value * scalar for key, value in form[1].items()}


def _add(left, right):
    coefficients = dict(left[1])
    for key, value in right[1].items():
        coefficients[key] = coefficients.get(key, F(0)) + value
    return left[0] + right[0], coefficients


def _direct_support(form, box):
    # Independent full-form reference: no candidate cache, sparse delta update,
    # inherited arithmetic primitive, or optimization routine is reused.
    return form[0] + sum(max(value * box[key][0], value * box[key][1])
                         for key, value in form[1].items())


def _direct_upper(forms, weights, affine, box, paired):
    base = affine
    positive = []
    for form, weight in zip(forms, weights):
        # Padding's identically zero nominal form has no nonlinear term;
        # interval uncertainty, if any, belongs to the separate error bound.
        if form[0] == 0 and all(value == 0 for value in form[1].values()):
            continue
        if weight < 0:
            base = _add(base, _scale(form, weight / 2))
        elif weight > 0:
            positive.append(_scale(form, weight))
    if not positive:
        return _direct_support(base, box)
    group_count = (len(positive) + 1) // 2
    shared = _scale(base, F(1, group_count))
    answer = F(0)
    for index in range(0, len(positive), 2):
        left = positive[index]
        if index + 1 == len(positive):
            answer += max(_direct_support(shared, box),
                          _direct_support(_add(shared, left), box))
        elif paired:
            right = positive[index + 1]
            answer += max(_direct_support(shared, box),
                          _direct_support(_add(shared, left), box),
                          _direct_support(_add(shared, right), box),
                          _direct_support(_add(_add(shared, left), right), box))
        else:
            half = _scale(shared, F(1, 2))
            for term in (left, positive[index + 1]):
                answer += max(_direct_support(half, box),
                              _direct_support(_add(half, term), box))
    return answer


def _rows_hold(rows, q, v, alpha):
    return all(cq * q + cv * v + ca * alpha <= rhs
               for cq, cv, ca, rhs in rows)


def _plain_evidence(value):
    if type(value) is tuple:
        for child in value:
            _plain_evidence(child)
    else:
        assert type(value) in (F, int)


def test_d027_joint_pair_and_unpaired_singleton_comparison():
    box = {0: _iv(-1, 1), 1: _iv(-1, 1)}
    f = _form(0, {0: 1, 1: F(1, 4)})
    h = _form(0, {0: 1, 1: F(-1, 4)})
    result = _compile((f, h), (_iv(1), _iv(1)),
                      _form(F(1, 10), {0: -1}), box)
    assert result["upper"] == F(11, 10)
    assert result["separate_upper"] == F(8, 5)
    assert result["lower"] == result["separate_lower"] == F(1, 10)
    assert result["error"] == 0
    assert (result["upper_pairs"], result["upper_singletons"]) == (1, 0)
    assert (result["lower_pairs"], result["lower_singletons"]) == (0, 0)
    singleton = _compile((h,), (_iv(F(3, 2)),), _form(0, {0: -1}), box)
    assert singleton["upper"] == singleton["separate_upper"] == 1
    assert singleton["lower"] == singleton["separate_lower"]
    assert (singleton["upper_pairs"], singleton["upper_singletons"]) == (0, 1)
    # All returned bounds are exact rational certificates, not a solved verdict.
    for key in ("lower", "upper", "separate_lower", "separate_upper", "error"):
        assert type(result[key]) is F


def test_signed_zero_all_negative_and_empty_populations():
    box = {0: _iv(-1, 1), 1: _iv(-1, 1)}
    forms = (_form(0, {0: 1}), _form(0, {1: 1}))
    affine = _form(F(3, 7), {0: 2, 1: -1})
    empty = _compile((), (), affine, box)
    zero = _compile(forms, (_iv(0), _iv(0)), affine, box)
    for result in (empty, zero):
        assert result["lower"] == result["separate_lower"] == F(-18, 7)
        assert result["upper"] == result["separate_upper"] == F(24, 7)
        assert result["error"] == 0
        assert all(result[key] == 0 for key in
                   ("upper_pairs", "lower_pairs", "upper_singletons", "lower_singletons"))
    padding_only = _compile((_form(), _form(0, {0: 0})),
                            (_iv(2), _iv(-3)), affine, box)
    assert padding_only == empty
    ordinary = _compile(forms, (_iv(1), _iv(2)), affine, box)
    padded = _compile((forms[0], _form(), forms[1], _form(0, {1: 0})),
                      (_iv(1), _iv(7), _iv(2), _iv(-4)), affine, box)
    assert padded == ordinary
    # Nominally zero is not permission to discard its uncertain true value.
    uncertain_zero = _compile(((_iv(-1, 1), {}),), (_iv(2),), _form(), {})
    assert uncertain_zero["error"] == 2
    assert uncertain_zero["lower"] == -2 and uncertain_zero["upper"] == 2
    assert all(uncertain_zero[key] == 0 for key in
               ("upper_pairs", "lower_pairs", "upper_singletons", "lower_singletons"))
    negative = _compile(forms, (_iv(-1), _iv(-2)), _form(2), box)
    assert negative["upper"] == negative["separate_upper"] == F(7, 2)
    assert negative["lower"] == negative["separate_lower"] == -1
    assert negative["upper_pairs"] == negative["upper_singletons"] == 0
    assert negative["lower_pairs"] == 1
    # The all-negative majorant is intentionally loose: do not report its
    # affine support as exact support of the nonlinear expression (maximum 2).
    assert negative["upper"] > 2
    signed = _compile((_form(0, {0: 1}), _form(0, {0: -1})),
                      (_iv(1), _iv(-1)), _form(F(1, 3)), box)
    assert signed["upper"] == signed["separate_upper"]
    assert signed["lower"] == signed["separate_lower"]
    assert signed["upper_pairs"] == signed["lower_pairs"] == 0
    assert signed["upper_singletons"] == signed["lower_singletons"] == 1
    for x in (F(-1), F(0), F(1)):
        concrete = F(1, 3) + max(F(0), x) - max(F(0), -x)
        assert signed["lower"] <= concrete <= signed["upper"]
    constant = _compile((), (), _form(F(-2, 5)), {})
    assert constant["lower"] == constant["upper"] == F(-2, 5)


def test_shared_source_cancellation_is_not_independent_coordinates():
    box = {0: _iv(-1, 1), 1: _iv(-1, 1)}
    first = _form(0, {0: 1})
    opposite = _form(0, {0: -1})
    distinct = _form(0, {1: -1})
    shared = _compile((first, opposite), (_iv(1), _iv(1)), _form(), box)
    independent = _compile((first, distinct), (_iv(1), _iv(1)), _form(), box)
    assert shared["upper"] == 1 and shared["separate_upper"] == 2
    assert independent["upper"] == independent["separate_upper"] == 2
    assert shared["lower"] == 0
    assert shared["error"] == independent["error"] == 0
    assert first == _form(0, {0: 1}) and opposite == _form(0, {0: -1})
    assert shared["upper_pairs"] == independent["upper_pairs"] == 1


def test_interval_cross_sign_has_explicit_error_and_endpoint_soundness():
    box = {0: _iv(-1, 1)}
    form = (_iv(F(-1, 4), F(1, 4)), {0: _iv(F(3, 4), F(5, 4))})
    affine = (_iv(F(1, 4), F(3, 4)), {0: _iv(F(-1, 8), F(1, 8))})
    result = _compile((form,), (_iv(-1, 1),), affine, box)
    # eps_f=1/2, M=3/2, rho_w=1, w0=0, eps_affine=3/8.
    assert result["error"] == F(15, 8)
    assert result["lower"] == result["separate_lower"] == F(-11, 8)
    assert result["upper"] == result["separate_upper"] == F(19, 8)
    assert result["upper_pairs"] == result["lower_pairs"] == 0
    assert result["upper_singletons"] == result["lower_singletons"] == 0
    # Fixed small arithmetic fixtures, not attacks or runtime phase searches.
    endpoints = ((F(-1), F(1)), form[0], form[1][0], affine[0], affine[1][0],
                 (F(-1), F(0), F(1)))
    for weight, bias, coefficient, offset, slope, x in product(*endpoints):
        concrete = offset + slope * x + weight * max(F(0), bias + coefficient * x)
        assert result["lower"] <= concrete <= result["upper"]
    nonzero_midpoint = _compile((form,), (_iv(1, 3),), affine, box)
    assert nonzero_midpoint["error"] == F(23, 8)
    assert nonzero_midpoint["upper"] <= nonzero_midpoint["separate_upper"]
    assert nonzero_midpoint["lower"] >= nonzero_midpoint["separate_lower"]


def test_cached_support_matches_independent_direct_form_reference():
    box = {2: _iv(-2, 1), 5: _iv(F(1, 4), F(5, 4)), 9: _iv(F(-1, 2), F(3, 2))}
    forms = (
        _form(F(1, 7), {2: F(2, 3), 5: F(-1, 4)}),
        _form(F(-2, 5), {2: F(-3, 4), 9: F(1, 2)}),
        _form(0, {5: F(1, 3), 9: -2}),
        _form(F(1, 11), {2: F(1, 5), 5: F(2, 7), 9: F(-1, 3)}),
        _form(F(-1, 6), {2: F(-1, 2)}),
    )
    affine = _form(F(-2, 7), {2: F(5, 6), 5: F(-3, 8), 9: F(1, 4)})
    point_forms, point_affine = tuple(map(_point_form, forms)), _point_form(affine)
    for weights in ((F(1), F(-2), F(3, 2), F(0), F(2, 3)),
                    (F(-1), F(-2), F(0), F(-3, 2), F(1, 3)),
                    (F(1), F(1, 2), F(3, 2), F(2), F(1, 4))):
        result = _compile(forms, tuple(_iv(value) for value in weights), affine, box)
        negative_weights = tuple(-value for value in weights)
        assert result["upper"] == _direct_upper(point_forms, weights, point_affine, box, True)
        assert result["separate_upper"] == _direct_upper(point_forms, weights, point_affine, box, False)
        assert result["lower"] == -_direct_upper(point_forms, negative_weights, _scale(point_affine, F(-1)), box, True)
        assert result["separate_lower"] == -_direct_upper(point_forms, negative_weights, _scale(point_affine, F(-1)), box, False)
        assert result["error"] == 0
        assert result["lower"] >= result["separate_lower"]
        assert result["upper"] <= result["separate_upper"]
        for prefix, population in (("upper", sum(value > 0 for value in weights)),
                                   ("lower", sum(value < 0 for value in weights))):
            assert type(result[prefix + "_pairs"]) is int
            assert type(result[prefix + "_singletons"]) is int
            assert result[prefix + "_pairs"] == population // 2
            assert result[prefix + "_singletons"] == population % 2
    prepared = support.prepare_source(forms, box, enabled=True)
    point_weights = tuple(_iv(value) for value in weights)
    assert support.compile_prepared(prepared, point_weights, affine, enabled=True) == result
    roots = prepared.evidence_roots()
    _plain_evidence(roots)
    # This is an owned immutable snapshot, not a cache keyed by mutable dict
    # identities. It must preserve both the source coefficients and the box.
    forms[0][1][2] = _iv(17)
    box[2] = _iv(-3, 2)
    assert prepared.evidence_roots() == roots
    assert support.compile_prepared(prepared, point_weights, affine, enabled=True) == result


def test_four_rows_equal_feasible_lift_interval_for_fractional_phases():
    for a in (F(-3, 2), F(0), F(2)):
        for lower, upper, active_lower, active_upper in (
                (F(-2), F(3), F(-1), F(4)),
                (F(1, 3), F(5, 3), F(-4, 3), F(2, 3))):
            rows = support.project_observation(a, lower, upper, active_lower, active_upper, enabled=True)
            assert type(rows) is tuple and len(rows) == 4
            assert all(type(row) is tuple and len(row) == 4
                       and all(type(value) is F for value in row) for row in rows)
            expected = {
                (a, F(0), lower - active_upper, F(0)),
                (-a, F(0), active_lower - upper, F(0)),
                (a, F(1), upper - active_upper, upper),
                (-a, F(-1), active_lower - lower, -lower),
            }
            assert set(rows) == expected
            for alpha, q, v in product(
                    (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)),
                    (F(-1), F(0), F(1, 2), F(1), F(2)),
                    (lower, (lower + upper) / 2, upper)):
                # This is the six-row linear relaxation, not t=alpha*v when
                # alpha is fractional. No ReLU sign assumption is needed.
                t_lower = max(lower * alpha, v - upper * (1 - alpha), active_lower * alpha - a * q)
                t_upper = min(upper * alpha, v - lower * (1 - alpha), active_upper * alpha - a * q)
                assert _rows_hold(rows, q, v, alpha) == (t_lower <= t_upper)
                if t_lower <= t_upper:
                    assert lower * alpha <= t_lower <= upper * alpha
                    assert v - upper * (1 - alpha) <= t_lower <= v - lower * (1 - alpha)
                    assert active_lower * alpha <= a * q + t_lower <= active_upper * alpha


def test_zero_preactivation_keeps_both_original_phase_choices():
    lower, upper, active_lower, active_upper = F(-1), F(2), F(-3, 4), F(3, 2)
    for a in (F(-2), F(0), F(3, 2)):
        rows = support.project_observation(a, lower, upper, active_lower, active_upper, enabled=True)
        for v, alpha in product((F(-1, 2), F(0), F(5, 4)), (F(0), F(1))):
            g = q = F(0)
            assert active_lower <= a * g + v <= active_upper
            assert _rows_hold(rows, q, v, alpha)
            t = max(lower * alpha, v - upper * (1 - alpha), active_lower * alpha - a * q)
            assert t == alpha * v
            assert lower * alpha <= t <= upper * alpha
            assert v - upper * (1 - alpha) <= t <= v - lower * (1 - alpha)
            assert active_lower * alpha <= a * q + t <= active_upper * alpha


def test_default_off_and_type_shape_source_identity_fail_closed():
    disabled = support.WorkBudget()
    assert support.compile_support(None, None, None, None, budget=disabled) is None
    assert support.project_observation(None, None, None, None, None, budget=disabled) is None
    assert support.prepare_source(None, None, budget=disabled) is None
    assert support.compile_prepared(None, None, None, budget=disabled) is None
    assert disabled.used == 0
    forms, weights, affine, box = (_form(0, {0: 1}),), (_iv(1),), _form(), {0: _iv(-1, 1)}
    with pytest.raises(support.KernelError):
        support.compile_support(forms, weights, affine, box, enabled=1)
    with pytest.raises(support.KernelError):
        support.project_observation(F(1), F(-1), F(1), F(-2), F(2), enabled=1)
    invalid = (
        (list(forms), weights, affine, box),
        (forms, list(weights), affine, box),
        (forms, (), affine, box),
        (([_iv(0), {0: _iv(1)}],), weights, affine, box),
        (forms, ((F(1), 1.0),), affine, box),
        (forms, (_iv(2, 1),), affine, box),
        (forms, weights, (_iv(0), []), box),
        (forms, weights, affine, []),
        (((_iv(0), {1: _iv(1)}),), weights, affine, box),
        (((_iv(0), {-1: _iv(1)}),), weights, affine, {-1: _iv(-1, 1)}),
        (((_iv(0), {True: _iv(1)}),), weights, affine, {True: _iv(-1, 1)}),
        (((_iv(0), {"0": _iv(1)}),), weights, affine, {"0": _iv(-1, 1)}),
        (forms, weights, affine, {0: _iv(1, -1)}),
        (forms, weights, affine, {0: (F(-1), float("inf"))}),
    )
    for args in invalid:
        with pytest.raises(support.KernelError):
            _compile(*args)
    for unowned in ({}, (), object()):
        with pytest.raises(support.KernelError):
            support.compile_prepared(unowned, weights, affine, enabled=True)
    for args in ((1, F(-1), F(1), F(-2), F(2)),
                 (F(1), F(2), F(1), F(-2), F(2)),
                 (F(1), F(-1), F(1), F(3), F(2))):
        with pytest.raises(support.KernelError):
            support.project_observation(*args, enabled=True)


def test_work_bit_caps_and_invalid_ledgers_fail_before_unpaid_work():
    forms, weights, affine, box = (_form(0, {0: 1}),), (_iv(1),), _form(), {0: _iv(-1, 1)}
    disabled = support.WorkBudget()
    with pytest.raises(support.KernelDisabled):
        _compile(forms, weights, affine, box, budget=disabled)
    with pytest.raises(support.KernelDisabled):
        support.project_observation(F(1), F(-1), F(1), F(-2), F(2), enabled=True, budget=disabled)
    for ledger in (object(),):
        with pytest.raises(support.KernelError):
            _compile(forms, weights, affine, box, budget=ledger)
    tiny = support.WorkBudget(enabled=True, limit=0)
    with pytest.raises(support.BudgetExceeded):
        _compile(forms, weights, affine, box, budget=tiny)
    assert tiny.used == 0
    with pytest.raises(support.BudgetExceeded):
        support.project_observation(F(1), F(-1), F(1), F(-2), F(2), enabled=True,
                                    budget=support.WorkBudget(enabled=True, limit=0))
    for key, value in (("enabled", 1), ("limit", -1), ("used", -1),
                       ("used", 256_000_001), ("used", True), ("max_bits", 0)):
        ledger = support.WorkBudget(enabled=True)
        setattr(ledger, key, value)
        with pytest.raises(support.KernelError):
            _compile(forms, weights, affine, box, budget=ledger)
    for bad in (F(1 << 512), F(1, 1 << 512)):
        with pytest.raises(support.KernelError):
            _compile(forms, (_iv(bad),), affine, box)
    with pytest.raises(support.KernelError):
        _compile((_form(2),), (_iv(F(1 << 511)),), affine, {})
    with pytest.raises(support.KernelError):
        _compile(forms, weights, _form(F(1 << 16)), box,
                 budget=support.WorkBudget(enabled=True, max_bits=16))
    with pytest.raises(support.KernelError):
        support.project_observation(F(1 << 512), F(-1), F(1), F(-2), F(2), enabled=True)
    ledger = support.WorkBudget(enabled=True)
    ledger.charge(7)
    _compile(forms, weights, affine, box, budget=ledger)
    first = ledger.used
    _compile(forms, weights, affine, box, budget=ledger)
    second = ledger.used
    support.project_observation(F(1), F(-1), F(1), F(-2), F(2), enabled=True, budget=ledger)
    assert 7 < first < second < ledger.used <= ledger.limit
