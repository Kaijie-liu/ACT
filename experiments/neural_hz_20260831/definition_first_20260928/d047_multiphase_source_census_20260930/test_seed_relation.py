"""Four preregistered source-seed tests; no candidate import before freeze.

Finite rational grids are independent mathematical fixtures, not runtime
phase search.  Ordinals name caller-certified original gates in one frame;
neither integer labels nor equal interval endpoints certify network identity.
No native HZ binding, model result, or formal score follows from these tests.
"""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d046_multiphase_envelopes_20260930 import multiphase as mp
from experiments.neural_hz_20260831.definition_first_20260928.d047_multiphase_source_census_20260930 import seed_relation as sr


FORM_NAMES = ("delta_lower", "delta_upper", "w_lower", "w_upper",
              "difference_lower", "difference_upper", "companion_lower",
              "companion_upper")
AFTER_NAMES = FORM_NAMES[4:]


def _point(value):
    value = F(value)
    return value, value


def _value(form, assignment):
    bias, terms = form
    return bias + sum((coefficient * assignment[ordinal]
                       for ordinal, coefficient in terms), F(0))


def _phases(preactivation):
    if preactivation < 0:
        return (F(0),)
    if preactivation > 0:
        return (F(1),)
    return F(0), F(1)


def _assert_forms(result):
    for name in FORM_NAMES:
        bias, terms = result[name]
        assert type(bias) is F and type(terms) is tuple
        ordinals = tuple(ordinal for ordinal, _ in terms)
        assert ordinals == tuple(sorted(set(ordinals)))
        assert all(type(ordinal) is int and ordinal >= 0
                   and type(coefficient) is F and coefficient != 0
                   for ordinal, coefficient in terms)
        assert result["supports"][name] == len(terms)
    assert result["nonconstant_relations"] == sum(bool(result[name][1]) for name in AFTER_NAMES)
    assert result["hinge_crossing_relations"] == sum(result["hinge_crossing"].values())
    assert result["last_anchor_strict_relations"] == sum(result["last_anchor_strict"].values())
    assert result["potential_coordinate_nnz"] == 6 + sum(len(result[name][1]) for name in AFTER_NAMES)
    assert result["binding_mathematical_only"] is True
    assert result["actual_phase_column_binding_verified"] is False
    assert result["work_used"] > 0


def _contains(result, h, w, assignment):
    r, t = max(F(0), h), max(F(0), w)
    for lower, upper, actual in (
            ("delta_lower", "delta_upper", h - w),
            ("w_lower", "w_upper", w),
            ("difference_lower", "difference_upper", r - t),
            ("companion_lower", "companion_upper", t)):
        assert _value(result[lower], assignment) <= actual
        assert actual <= _value(result[upper], assignment)
    assert result["ordinary_h"][0] <= h <= result["ordinary_h"][1]
    assert result["ordinary_w"][0] <= w <= result["ordinary_w"][1]
    assert result["plain_shared_difference"][0] <= h - w <= result["plain_shared_difference"][1]


def _control():
    # Z=ReLU(z+1)=z+1 is an actual stable source, not a signed ReLU output.
    # h=z-2q/5-7p/10+1/4; w=z+q/10-p/5; q=R(x),p=R(y).
    return sr.compile_pair(
        (_point(F(-2, 5)), _point(F(-7, 10)), _point(1)),
        _point(F(-3, 4)),
        (_point(F(1, 10)), _point(F(-1, 5)), _point(1)),
        _point(-1),
        ((F(-1), F(1)), (F(-1), F(1)), (F(0), F(2))),
        (0, 1, 2), enabled=True)


def test_interval_seed_soundness():
    # These intervals enclose fixed actual coefficients (e.g. outward BN
    # arithmetic).  Endpoint choices below test containment, not new model
    # parameters or a candidate enumeration path.
    left = ((F(-2, 5), F(1, 5)), (F(1, 2), F(3, 4)))
    right = ((F(-1, 2), F(-1, 4)), (F(-1, 5), F(2, 5)))
    left_bias, right_bias = (F(-1, 5), F(1, 10)), (F(-1, 10), F(1, 4))
    bounds = ((F(-1), F(2)), (F(-2), F(1)))
    ordinals = (7, 3)  # Formula order is original ordinal, not slot order.
    budget = sr.WorkBudget(enabled=True)
    result = sr.compile_pair(left, left_bias, right, right_bias,
                             bounds, ordinals, enabled=True, budget=budget)
    assert budget.used > 0
    _assert_forms(result)
    assert result["source_counts"]["crossing_slots"] == 2
    assert result["source_counts"]["distinct_crossing_phases"] == 2
    for coefficients in product(*(left + right + (left_bias, right_bias))):
        a0, a1, b0, b1, ca, cb = coefficients
        for g0, g1 in product((F(-1), F(0), F(2)), (F(-2), F(0), F(1))):
            q0, q1 = max(F(0), g0), max(F(0), g1)
            h, w = ca + a0 * q0 + a1 * q1, cb + b0 * q0 + b1 * q1
            for alpha, eta in product(_phases(g0), _phases(g1)):
                _contains(result, h, w, {7: alpha, 3: eta})


def test_stable_padding_and_identity():
    bounds = ((F(1, 2), F(2)), (F(-2), F(-1, 4)),
              (F(0), F(2)), (F(-2), F(0)), _point(0), _point(0),
              (F(-1), F(1)))
    ordinals = (10, 11, 12, 13, 14, None, 15)
    left = (_point(1), _point(-2), _point(F(1, 2)), _point(1),
            _point(3), _point(-7), _point(F(-1, 3)))
    right = tuple(_point(0) for _ in bounds)
    result = sr.compile_pair(left, _point(F(1, 4)), right, _point(0),
                             bounds, ordinals, enabled=True)
    _assert_forms(result)
    assert result["source_counts"] == {
        "canonical_slots": 7, "padding_slots": 1, "crossing_slots": 1,
        "distinct_crossing_phases": 1, "stable_active_slots": 1,
        "stable_inactive_slots": 1, "zero_boundary_slots": 3}
    assert all(set(dict(result[name][1])) <= {15} for name in FORM_NAMES)
    # Every stable boundary gate still has both legal original zero phases.
    # The bound may omit these phases without deleting/fixing the original bit.
    actual = (F(1), F(-1), F(0), F(0), F(0), F(0), F(0))
    h = F(1, 4) + sum((weight[0] * max(F(0), value)
                       for weight, value in zip(left, actual)), F(0))
    for zero_bits in product((F(0), F(1)), repeat=4):
        assignment = {10: F(1), 11: F(0), **dict(zip((12, 13, 14, 15), zero_bits))}
        _contains(result, h, F(0), assignment)
    # Distinct canonical occurrences may refer to the SAME original source.
    # They reuse its phase, rather than creating independent copies.
    shared = sr.compile_pair((_point(1), _point(-1)), _point(0),
                             (_point(0), _point(0)), _point(0),
                             ((F(-1), F(1)),) * 2, (4, 4), enabled=True)
    assert shared["source_counts"]["crossing_slots"] == 2
    assert shared["source_counts"]["distinct_crossing_phases"] == 1
    assert all(len(shared[name][1]) <= 1 for name in FORM_NAMES)
    for bit in (F(0), F(1)):
        _contains(shared, F(0), F(0), {4: bit})
    uncertain = ((F(1), F(2)),)
    uncertain_bias = (F(0), F(1, 3))
    args = (uncertain, uncertain_bias, uncertain, uncertain_bias,
            ((F(-1), F(1)),), (0,))
    distinct = sr.compile_pair(*args, enabled=True)
    assert distinct["delta_lower"] == (F(-1, 3), ((0, F(-1)),))
    assert distinct["delta_upper"] == (F(1, 3), ((0, F(1)),))
    assert distinct["plain_shared_difference"] == (F(-4, 3), F(4, 3))
    # A true identity claim is an explicit caller premise, never inferred from
    # equality of the two coefficient-interval arrays.
    identical = sr.compile_pair(*args, same_consumer=True, enabled=True)
    for name in ("delta_lower", "delta_upper", "difference_lower", "difference_upper"):
        assert identical[name] == (F(0), ())
    assert identical["plain_shared_difference"] == _point(0)
    assert identical["ordinary_h"] == identical["ordinary_w"]


def test_multiphase_transfer_and_metrics():
    result = _control()
    expected = {
        "delta_lower": (F(1, 4), ((0, F(-1, 2)), (1, F(-1, 2)))),
        "delta_upper": (F(1, 4), ()),
        "w_lower": (F(-1), ((1, F(-1, 5)),)),
        "w_upper": (F(1), ((0, F(1, 10)),)),
        "difference_lower": (F(0), ((0, F(-1, 4)), (1, F(-1, 2)))),
        "difference_upper": (F(1, 4), ()),
        "companion_lower": (F(0), ()),
        "companion_upper": (F(1), ((0, F(1, 10)),))}
    for name, form in expected.items():
        assert result[name] == form
    _assert_forms(result)
    assert result["ordinary_h"] == (F(-37, 20), F(5, 4))
    assert result["ordinary_w"] == (F(-6, 5), F(11, 10))
    assert result["plain_shared_difference"] == (F(-3, 4), F(1, 4))
    assert result["hinge_crossing"]["difference_lower"] is True
    assert result["last_anchor_gap"]["difference_lower"] == F(1, 4)
    assert result["last_anchor_ordinal"]["difference_lower"] == 1
    assert result["last_anchor_strict"]["difference_lower"] is True
    assert all(result["last_anchor_gap"][name] == 0 for name in AFTER_NAMES[1:])
    assert all(result["last_anchor_strict"][name] is False for name in AFTER_NAMES[1:])
    # Link the interval source adapter to the already-qualified D046 formula,
    # retaining exact expected rows above as a separate arithmetic oracle.
    frame = mp.make_frame("D047 fixed source fixture", enabled=True)
    phases = {i: mp.make_phase(mp.make_value(frame, str(i), enabled=True),
                              i, str(i), enabled=True) for i in (0, 1)}
    seed_names = ("delta_lower", "delta_upper", "w_lower", "w_upper")
    for index, (seed_name, after_name) in enumerate(zip(seed_names, AFTER_NAMES)):
        sign = F(-1) if index == 0 else F(1)
        bias, terms = result[seed_name]
        source = mp.form(frame, "phase", sign * bias,
                         tuple((phases[i], sign * coefficient) for i, coefficient in terms),
                         enabled=True)
        transformed = (mp.positive_minorant(source, enabled=True) if index == 2
                       else mp.positive_majorant(source, enabled=True))
        reverse = {phase: i for i, phase in phases.items()}
        plain = (sign * transformed.bias,
                 tuple((reverse[phase], sign * coefficient)
                       for phase, coefficient in transformed.terms))
        assert result[after_name] == plain
    old = {0: F(3, 4), 1: F(1, 4)}
    assert -_value(result["difference_lower"], old) == F(5, 16) < F(7, 20)
    assert F(7, 20) + F(1, 4) / 4 + F(3, 4) / 2 == F(63, 80) > F(3, 4)
    for x, y, z in product((F(-1), F(0), F(1)), repeat=3):
        q, p = max(F(0), x), max(F(0), y)
        h = z - F(2, 5) * q - F(7, 10) * p + F(1, 4)
        w = z + q / 10 - p / 5
        for alpha, eta in product(_phases(x), _phases(y)):
            _contains(result, h, w, {0: alpha, 1: eta})
        physical = max(F(0), w) - max(F(0), h) + (q - x) / 4 + (p - y) / 2
        assert physical <= F(3, 4)


def test_disabled_and_rejected_premises():
    assert sr.compile_pair(None, None, None, None, None, None) is None
    assert sr.compile_pair(None, None, None, None, None, None,
                           budget=object(), same_consumer="not inspected") is None
    args = ((_point(1),), _point(0), (_point(0),), _point(0),
            ((F(-1), F(1)),), (0,))
    with pytest.raises(sr.KernelError):
        sr.compile_pair(*args, enabled=1)
    with pytest.raises(sr.KernelError):
        sr.compile_pair(*args, enabled=True, same_consumer=1)
    with pytest.raises(sr.KernelError):
        sr.compile_pair(*args, enabled=True, same_consumer=True)
    for bad_ordinals in ((None,), (-1,), (True,), ("original bit",), (0, 1)):
        with pytest.raises(sr.KernelError):
            sr.compile_pair(*args[:-1], bad_ordinals, enabled=True)
    with pytest.raises(sr.KernelError):
        sr.compile_pair((_point(1),) * 2, _point(0), (_point(0),) * 2,
                        _point(0), ((F(-1), F(1)), (F(-2), F(1))),
                        (0, 0), enabled=True)
    for bad_weights in (list(args[0]), (), ((F(1), F(-1)),), ((0.0, F(1)),),
                        ((F(1 << 512), F(1 << 512)),)):
        with pytest.raises(sr.KernelError):
            sr.compile_pair(bad_weights, *args[1:], enabled=True)
    with pytest.raises(sr.KernelError):
        sr.compile_pair(*args, enabled=True, budget=object())
    with pytest.raises(sr.KernelDisabled):
        sr.compile_pair(*args, enabled=True, budget=sr.WorkBudget())
    exhausted = sr.WorkBudget(enabled=True, limit=0)
    with pytest.raises(sr.BudgetExceeded):
        sr.compile_pair(*args, enabled=True, budget=exhausted)
    assert exhausted.used == 0
    invalid = sr.WorkBudget(enabled=True)
    invalid.used = -1
    with pytest.raises(sr.KernelError):
        sr.compile_pair(*args, enabled=True, budget=invalid)
    count = sr.MAX_SLOTS + 1
    with pytest.raises(sr.KernelError):
        sr.compile_pair((_point(0),) * count, _point(0), (_point(0),) * count,
                        _point(0), (_point(0),) * count, (None,) * count,
                        enabled=True)
