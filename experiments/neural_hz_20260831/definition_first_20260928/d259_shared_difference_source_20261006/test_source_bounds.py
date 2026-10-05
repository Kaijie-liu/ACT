"""Eight fixed, solver-free source-bound tests; no model is executed here.

Fraction is used only by the independent small mathematical oracles.  The
candidate receives float64 arrays or its explicitly supported rational range
endpoints, never a Fraction-decoded model or a source selected by a query.
"""
from fractions import Fraction as F
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as accounting
from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as sb
from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_observer as observer


RUN = Path(__file__).resolve().parents[2] / "results/d259_shared_difference_source_20261006_v1"
_NAMES = (
    "default_off_and_limits", "outward_arithmetic", "complete_channel_conv",
    "bn_whole_carrier", "additive_peeling_not_arbitrary_bounds",
    "difference_payment_controls", "complete_padding_classes", "summary_boundary",
)
_EVIDENCE = {}
_BUDGET = None


class _Meter:
    """Test-only signature adapter to the existing shared accounting object."""

    def __init__(self, *, max_work=256_000_000, max_entries=64_000_000):
        self.owner = accounting.Budget(max_work=max_work, max_entries=max_entries)
        self.meter = self.owner._branch()
        self.max_bits = self.owner.max_bits

    def charge(self, amount, entries=0):
        self.meter.charge(work=amount, entries=entries)


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = _Meter()
    return _BUDGET


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _iv(lo, hi):
    # These small declared inputs are not candidate-produced certificates.
    return sb.Interval(np.asarray(lo, dtype=np.float64),
                       np.asarray(hi, dtype=np.float64))


def _point(values):
    return sb.point(np.asarray(values, dtype=np.float64),
                    budget=_budget(), enabled=True)


def _call(name, *args, **kwargs):
    return getattr(sb, name)(*args, budget=_budget(), enabled=True, **kwargs)


def _exact(value):
    return F.from_float(float(value))


def _ends(value):
    assert value.lo.dtype == np.float64 and value.hi.dtype == np.float64
    assert value.lo.shape == value.hi.shape
    assert np.isfinite(value.lo).all() and np.isfinite(value.hi).all()
    assert (value.lo <= value.hi).all()
    return tuple((_exact(lo), _exact(hi))
                 for lo, hi in zip(value.lo.flat, value.hi.flat))


def _contains(value, intervals):
    actual = _ends(value)
    expected = tuple(intervals)
    assert len(actual) == len(expected)
    for (lo, hi), (wanted_lo, wanted_hi) in zip(actual, expected):
        assert lo <= wanted_lo <= wanted_hi <= hi
    return actual


def _tight_small(value, intervals):
    expected = tuple(intervals)
    actual = _contains(value, expected)
    # A nonvacuity check for these ordinary small, fixed arithmetic controls;
    # correctness above uses exact comparisons, not this accuracy allowance.
    allowance = F(1, 10**9)
    for (lo, hi), (wanted_lo, wanted_hi) in zip(actual, expected):
        assert wanted_lo - lo < allowance and hi - wanted_hi < allowance


def _linear(bounds, coefficients):
    terms = _call("mul", _point(coefficients), bounds)
    return _call("sum_axis", terms, axis=0)


def _mask(height, width, y, x):
    return tuple(tuple(0 <= y + ky - 1 < height and 0 <= x + kx - 1 < width
                       for kx in range(3)) for ky in range(3))


def _conv_oracle(weights, bias, source, mask):
    """Exact small affine-box oracle, independent of the candidate ledger."""
    lo_hi = _ends(source)
    result = []
    for co in range(weights.shape[0]):
        lower = upper = _exact(bias[co])
        for ci in range(weights.shape[1]):
            for ky in range(weights.shape[2]):
                for kx in range(weights.shape[3]):
                    if not mask[ky][kx]:
                        continue
                    coefficient = _exact(weights[co, ci, ky, kx])
                    a, b = coefficient * lo_hi[ci][0], coefficient * lo_hi[ci][1]
                    lower += min(a, b)
                    upper += max(a, b)
        result.append((lower, upper))
    return tuple(result)


def _conv(weights, bias, source, *, padding=0, valid_mask=None):
    return sb.conv_channel(np.asarray(weights, dtype=np.float64),
                           np.asarray(bias, dtype=np.float64), source, padding,
                           _budget(), enabled=True, valid_mask=valid_mask)


def _payment(e):
    ((lower, upper),) = _ends(e)
    return 2 * max(upper, F(0)), -2 * min(lower, F(0))


def _fused_oracle(weights, bias, source, selected, h, carrier, skip):
    """Exact independent box arithmetic for the fixed 1x1 fused controls."""
    assert weights.shape[2:] == (1, 1)
    sources, hs, skips = _ends(source), _ends(h), _ends(skip)
    rest_lo = rest_hi = F(0)
    coefficient_bounds = [[F(0), F(0)] for _ in selected]
    for co in range(weights.shape[0]):
        lo = hi = _exact(bias[co])
        for ci in range(weights.shape[1]):
            if (ci, 0, 0) in selected:
                continue
            w = _exact(weights[co, ci, 0, 0])
            products = tuple(w * endpoint for endpoint in sources[ci])
            lo += min(products)
            hi += max(products)
        a, b, error = (_exact(value[co]) for value in
                       (carrier.nominal_a, carrier.nominal_b, carrier.error))
        scaled = (a * lo, a * hi)
        rest = (min(scaled) + b - error + skips[co][0],
                max(scaled) + b + error + skips[co][1])
        products = tuple(left * right for left in hs[co] for right in rest)
        rest_lo += min(products)
        rest_hi += max(products)
        for index, descriptor in enumerate(selected):
            w = F(0) if descriptor is None else _exact(weights[(co,) + descriptor])
            values = tuple(endpoint * a * w for endpoint in hs[co])
            coefficient_bounds[index][0] += min(values)
            coefficient_bounds[index][1] += max(values)
    return ((rest_lo, rest_hi), tuple(tuple(pair) for pair in coefficient_bounds))


def test_01_default_off_and_limits():
    meter = _budget()
    before = (meter.owner.work, meter.owner.entries)
    assert sb.point(np.asarray([1.0]), budget=meter) is None
    assert sb.add(None, None, budget=meter) is None
    assert sb.fused_h(None, None, budget=meter) is None
    assert sb.fused_offset(None, None, None, None, None, budget=meter) is None
    assert observer.geometry_classes(32, 32, budget=meter) is None
    assert (meter.owner.work, meter.owner.entries) == before

    low_work = _Meter(max_work=16)
    with pytest.raises(accounting.Rejected):
        sb.point(np.zeros(32, dtype=np.float64), budget=low_work, enabled=True)
    assert low_work.owner.failed
    with pytest.raises(accounting.Rejected):
        sb.point(np.asarray([0.0]), budget=low_work, enabled=True)

    low_entries = _Meter(max_entries=4)
    with pytest.raises(accounting.Rejected):
        sb.point(np.zeros(8, dtype=np.float64), budget=low_entries, enabled=True)
    assert low_entries.owner.failed

    for invalid in (np.asarray([np.nan]), np.asarray([np.inf])):
        with pytest.raises(sb.Rejected):
            sb.point(invalid, budget=_Meter(), enabled=True)
    with pytest.raises(sb.Rejected):
        sb.add(_iv([1.0], [0.0]), _iv([0.0], [0.0]),
               budget=_Meter(), enabled=True)
    with pytest.raises(sb.Rejected):
        sb.from_rationals(((F(1 << 512), F(1 << 512)),),
                          budget=_Meter(), enabled=True)
    _record(1, default_off_no_work=True, shared_budget_sticky=True,
            finite_and_order_checked=True, rational_bits_limit=512,
            huge_array_or_model_fixture=False)


def test_02_outward_arithmetic():
    a, b = _iv([-1.25, 0.1], [0.5, 0.3]), _iv([0.25, -0.2], [0.75, 0.4])
    ae, be = _ends(a), _ends(b)
    expected_add = tuple((x[0] + y[0], x[1] + y[1]) for x, y in zip(ae, be))
    expected_sub = tuple((x[0] - y[1], x[1] - y[0]) for x, y in zip(ae, be))
    expected_mul = []
    for x, y in zip(ae, be):
        candidates = tuple(v * w for v in x for w in y)
        expected_mul.append((min(candidates), max(candidates)))
    _tight_small(_call("add", a, b), expected_add)
    _tight_small(_call("sub", a, b), expected_sub)
    _tight_small(_call("mul", a, b), expected_mul)
    _tight_small(_call("neg", a), tuple((-hi, -lo) for lo, hi in ae))
    _tight_small(_call("neg", _point(0.25)), ((F(-1, 4), F(-1, 4)),))
    _tight_small(_call("scale_half", a), tuple((lo / 2, hi / 2) for lo, hi in ae))
    _tight_small(_call("relu", a), tuple((max(F(0), lo), max(F(0), hi))
                                          for lo, hi in ae))
    _tight_small(_call("take", a, (1, 0), axis=0), (ae[1], ae[0]))

    values = np.asarray(((1.0, 0.1, -1.0), (-0.5, 1.25, 0.125)), dtype=np.float64)
    rows = _point(values)
    totals = tuple(sum((_exact(v) for v in row), F(0)) for row in values)
    _tight_small(_call("sum_axis", rows, axis=1), tuple((v, v) for v in totals))
    kept = _call("sum_axis", rows, axis=1, keepdims=True)
    assert kept.lo.shape == (2, 1)
    _contains(kept, tuple((v, v) for v in totals))

    rational = ((F(1, 3), F(1, 3)), (F(-7, 10), F(2, 3)), (F(5, 8), F(5, 8)))
    _tight_small(_call("from_rationals", rational), rational)
    _record(2, fraction_oracle=True, sum_uses_certified_absolute_error=True,
            sum_cancellation_control=True, rational_input_intervals=3,
            original_float_values_interpreted_as_exact_dyadics=True)


def test_03_complete_channel_conv():
    weights = np.asarray((
        (((0.25, -0.5, 0.75), (-1.0, 0.5, 0.25), (0.5, -0.25, 0.125)),
         ((-0.25, 0.5, 0.125), (0.75, -0.5, 0.25), (-0.125, 0.5, -0.75))),
        (((-0.5, 0.25, 0.125), (0.5, -0.25, 0.75), (0.25, 0.5, -0.125)),
         ((0.5, -0.25, 0.75), (-0.125, 0.5, -0.5), (0.25, -0.75, 0.125))),
    ), dtype=np.float64)
    original = weights.copy()
    bias = np.asarray((0.125, -0.25), dtype=np.float64)
    source = _iv((-1.0, 0.25), (0.5, 1.25))
    ledger = _conv(weights, bias, source, padding=1)
    assert ledger.mass_positive.lo.shape == weights.shape[:2]
    assert ledger.mass_negative.lo.shape == weights.shape[:2]
    position_bounds = []
    for y in range(3):
        for x in range(4):
            expected = _conv_oracle(weights, bias, source, _mask(3, 4, y, x))
            _contains(ledger.bounds, expected)
            position_bounds.append(expected)
    corner_mask = np.asarray(_mask(3, 4, 0, 0), dtype=bool)
    corner = _conv(weights, bias, source, padding=1, valid_mask=corner_mask)
    _contains(corner.bounds, position_bounds[0])
    empty = _conv(weights, bias, source, padding=1,
                  valid_mask=np.zeros((3, 3), dtype=bool))
    _tight_small(empty.bounds, tuple((_exact(v), _exact(v)) for v in bias))
    assert np.array_equal(weights, original)
    _record(3, outputs_checked=24, spatial_positions_checked=12,
            input_channels=2, output_channels=2, all_kernel_positions=True,
            real_padding_zero_not_free_source=True, source_weights_unchanged=True)


def test_04_bn_whole_carrier():
    source = _iv((0.0, -2.0), (1.0, 3.0))
    a = _iv((0.25, -0.75), (0.5, -0.5))
    b = _iv((-0.125, 0.125), (0.375, 0.25))
    carrier = sb.bn_carrier(source, a, b, budget=_budget(), enabled=True)
    se, ae, be = _ends(source), _ends(a), _ends(b)
    for index in range(2):
        physical = tuple(x * scale + offset for x in se[index]
                         for scale in ae[index] for offset in be[index])
        _contains(_call("take", carrier.bounds, (index,), axis=0),
                  ((min(physical), max(physical)),))
        nominal_a = _exact(carrier.nominal_a[index])
        nominal_b = _exact(carrier.nominal_b[index])
        error = _exact(carrier.error[index])
        assert error >= 0
        declared = tuple(nominal_a * x + nominal_b + sign * error
                         for x in se[index] for sign in (F(-1), F(1)))
        _contains(_call("take", carrier.bounds, (index,), axis=0),
                  ((min(declared), max(declared)),))
        # The same physical source/error value must be used by both readers.
        for x in se[index]:
            for sign in (F(-1), F(1)):
                shared = nominal_a * x + nominal_b + sign * error
                assert 3 * shared - 2 * shared == shared
    # Raw A/B trajectories start at -1/8 in channel 0, whereas the declared
    # independent error H also contains nominal_b - E <= -1/4.
    assert _ends(carrier.bounds)[0][0] <= F(-1, 4) < F(-1, 8)
    _record(4, original_parameter_box_covered=True,
            entire_nominal_plus_error_H_covered=True, channels=2,
            independent_error_per_actual_scalar_required=True,
            observation_does_not_construct_native_error_identity=True)


def test_05_additive_peeling_not_arbitrary_bounds():
    source = _iv((0.0, -0.5, -1.0), (1.0, 0.75, 0.25))
    weights = np.asarray(((((1.0,),), ((2.0,),), ((-0.5,),)),), dtype=np.float64)
    ledger = _conv(weights, (0.125,), source)
    one = sb.peel_conv_selected(ledger, ((0, 0, 0),), budget=_budget(), enabled=True)
    two = sb.peel_conv_selected(ledger, ((0, 0, 0), (2, 0, 0)),
                                budget=_budget(), enabled=True)
    _tight_small(ledger.bounds, ((F(-1), F(25, 8)),))
    _tight_small(one, ((F(-1), F(17, 8)),))
    _tight_small(two, ((F(-7, 8), F(13, 8)),))
    # Ordinary decimal payloads make tap products inexact in binary64.  The
    # oracle removes coefficients before exact arithmetic, independently of
    # the grouped-mass ledger's outward subtraction implementation.
    rounded_source = _iv((0.1, 0.2, -0.6), (0.3, 0.7, 0.4))
    rounded_weights = np.asarray(((((0.1,),), ((-0.3,),), ((0.2,),)),))
    rounded = _conv(rounded_weights, (0.1,), rounded_source)
    rounded_rest = sb.peel_conv_selected(rounded, ((0, 0, 0), (1, 0, 0)),
                                         budget=_budget(), enabled=True)
    oracle_weights = rounded_weights.copy()
    oracle_weights[:, :2] = 0.0
    expected_rest = _conv_oracle(oracle_weights, (0.1,), rounded_source, ((True,),))
    _tight_small(rounded_rest, expected_rest)
    with pytest.raises(sb.Rejected):
        sb.peel_conv_selected(ledger, ((0, 0, 0), (0, 0, 0)),
                              budget=_Meter(), enabled=True)

    # A correlated, already tightened sum cannot replace the additive ledger:
    # R(x)+R(-x) <= 1, but subtracting the positive endpoint of R(x) from
    # that bound would incorrectly say R(-x) <= 0.
    correlated_sum_upper, selected_upper = F(1), F(1)
    x = F(-1)
    q1, q2 = max(x, F(0)), max(-x, F(0))
    assert q1 + q2 <= correlated_sum_upper
    assert correlated_sum_upper - selected_upper == 0 < q2
    honest = _conv(np.asarray(((((1.0,),), ((1.0,),)),)), (0.0,),
                   _iv((0.0, 0.0), (1.0, 1.0)))
    safe = sb.peel_conv_selected(honest, ((0, 0, 0),),
                                 budget=_budget(), enabled=True)
    _tight_small(safe, ((F(0), F(1)),))
    _record(5, removable_terms_checked=2, duplicate_term_rejected=True,
            arbitrary_tight_sum_is_not_a_ledger=True,
            preserved_remaining_source_not_new_independent_graph=True)


def test_06_difference_payment_controls():
    # Ordinary asymmetric dyadic neighbour of the paper control.  All three
    # Add-frontier readers refer to the same q1,q2,t in this exact identity.
    source = _iv((0.0, 0.0, -1.0), (1.0, 1.0, 1.0))
    weights = np.asarray((
        (((1.0,),), ((0.0,),), ((0.25,),)),
        (((0.0,),), ((1.0,),), ((0.25,),)),
        (((0.0,),), ((0.0,),), ((1.0,),)),
    ), dtype=np.float64)
    ledger = _conv(weights, (0.0, 0.0, 0.0), source)
    rest = sb.peel_conv_selected(ledger, ((0, 0, 0), (1, 0, 0)),
                                 budget=_budget(), enabled=True)
    child1, child2 = (1.25, -0.75, 0.375), (-0.8125, 1.1875, 0.40625)
    child_bounds = (_linear(ledger.bounds, child1), _linear(ledger.bounds, child2))
    difference = tuple((_exact(a) - _exact(b)) / 2 for a, b in zip(child1, child2))
    assert difference == (F(33, 32), F(-31, 32), F(-1, 64))
    # Exercise the actual fused source path, not only the compositional API.
    pair_kernel = np.asarray((child1, child2), dtype=np.float64).reshape(2, 3, 1, 1)
    merged_h = _call("fused_h", pair_kernel, np.ones(2, dtype=np.float64))
    _tight_small(merged_h, tuple((value, value) for value in difference))
    h = sb.Interval(merged_h.lo[:, 0, 0], merged_h.hi[:, 0, 0])
    carrier = _call("bn_carrier", ledger.bounds, _point((1.0, 1.0, 1.0)),
                    _point((0.0, 0.0, 0.0)))
    skip = _point((0.0, 0.0, 0.0))
    selected = ((0, 0, 0), (1, 0, 0))
    fused_rest, fused_coefficients = _call("fused_offset", ledger, selected, h, carrier, skip)
    oracle_rest, oracle_coefficients = _fused_oracle(
        weights, (0.0, 0.0, 0.0), source, selected, h, carrier, skip)
    _tight_small(fused_rest, (oracle_rest,))
    for value, expected in zip(fused_coefficients, oracle_coefficients):
        _tight_small(value, (expected,))
    # This second ordinary control has signed cancellation and an absent tap.
    # The oracle uses the whole declared BN error carrier in every reduction.
    cancelling_weights = np.ones((3, 1, 1, 1), dtype=np.float64)
    cancelling_source = _iv((0.0,), (1.0,))
    cancelling = _conv(cancelling_weights, (0.0, 0.0, 0.0), cancelling_source)
    cancelling_carrier = _call("bn_carrier", cancelling.bounds,
                               _point((1.0, 1.0, 1.0)), _point((0.0, 0.0, 0.0)))
    cancelling_h, cancelling_skip = _point((1.0, -1.0, 0.25)), _point((1.0, 1.0, 1.0))
    cancelling_selected = ((0, 0, 0), None)
    cancelling_rest, cancelling_coefficients = _call(
        "fused_offset", cancelling, cancelling_selected, cancelling_h,
        cancelling_carrier, cancelling_skip)
    cancellation_oracle = _fused_oracle(
        cancelling_weights, (0.0, 0.0, 0.0), cancelling_source,
        cancelling_selected, cancelling_h, cancelling_carrier, cancelling_skip)
    _tight_small(cancelling_rest, (cancellation_oracle[0],))
    for value, expected in zip(cancelling_coefficients, cancellation_oracle[1]):
        _tight_small(value, (expected,))
    _contains(cancelling_rest, ((F(1, 4), F(1, 4)),))
    _contains(cancelling_coefficients[0], ((F(1, 4), F(1, 4)),))
    _contains(cancelling_coefficients[1], ((F(0), F(0)),))
    tau, mu = (difference[0] - difference[1]) / 2, (difference[0] + difference[1]) / 2
    assert tau == 1 and mu == F(1, 32)
    actual_coefficients = tuple(float(value) for value in difference)
    delta = _linear(ledger.bounds, actual_coefficients)
    parent_difference = _linear(source, (1.0, -1.0, 0.0))
    coarse = _call("sub", delta, parent_difference)
    peeled = _call("add", _linear(rest, actual_coefficients),
                   _linear(source, (0.03125, 0.03125, 0.0)))
    _tight_small(coarse, ((F(-159, 64), F(163, 64)),))
    _tight_small(peeled, ((F(-33, 64), F(37, 64)),))
    old_upper = tuple(_ends(value)[0][1] for value in child_bounds)
    thresholds = (old_upper[0] + 2, old_upper[1] + 2)
    coarse_payment, peeled_payment = _payment(coarse), _payment(peeled)
    assert all(payment >= threshold for payment, threshold in zip(coarse_payment, thresholds))
    assert all(payment < threshold for payment, threshold in zip(peeled_payment, thresholds))
    for value, payments, excluded in ((coarse, coarse_payment, True),
                                       (peeled, peeled_payment, False)):
        classification = observer.payment_classification(
            (float(value.lo), float(value.hi)), float(tau), (1.0, 1.0),
            tuple(float(item) for item in old_upper), budget=_budget(), enabled=True)
        assert classification["excluded_by_d258"] is excluded
        exact_rows = tuple((item.numerator, item.denominator)
                           for item in payments + thresholds)
        assert classification["exact_payment_and_threshold"] == exact_rows
    asymmetric_thresholds = observer.payment_classification(
        (-1.0, 1.0), 0.5, (0.5, 0.75), (1.25, 1.5),
        budget=_budget(), enabled=True)
    assert asymmetric_thresholds["excluded_by_d258"] is True
    assert asymmetric_thresholds["exact_payment_and_threshold"] == ((2, 1),) * 4
    # Exact source algebra, not a fitted auxiliary point or a solver query.
    for q1, q2, t in ((F(0), F(1), F(-1)), (F(1), F(0), F(1)),
                      (F(1, 3), F(2, 3), F(1, 2))):
        s = (q1 + t / 4, q2 + t / 4, t)
        delta_exact = sum((weight * value for weight, value in zip(difference, s)), F(0))
        assert delta_exact - tau * (q1 - q2) == (q1 + q2) / 32

    # Negative control: after selected-q removal, the two appearances of 2T
    # still share one source.  Bounding them independently loses cancellation.
    shared_source = _iv((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    shared_weights = np.asarray((
        (((1.0,),), ((0.0,),), ((2.0,),)),
        (((0.0,),), ((1.0,),), ((2.0,),)),
    ), dtype=np.float64)
    shared = _conv(shared_weights, (0.0, 0.0), shared_source)
    shared_rest = sb.peel_conv_selected(shared, ((0, 0, 0), (1, 0, 0)),
                                        budget=_budget(), enabled=True)
    residual_box = _linear(shared_rest, (1.0, -1.0))
    _tight_small(residual_box, ((F(-2), F(2)),))
    # Here the strong old single-child bounds use the valid same-source
    # identity g1=q1-q2+Z, g2=-q1+q2+Z, Z in [-1/4,1/4].
    z = _iv((-0.25,), (0.25,))
    cancelled_child = _call("add", _linear(shared_source, (1.0, -1.0, 0.0)), z)
    _tight_small(cancelled_child, ((F(-5, 4), F(5, 4)),))
    strong_threshold = _ends(cancelled_child)[0][1] + 2
    assert all(value >= strong_threshold for value in _payment(residual_box))
    assert F(2) * F(3, 5) - F(2) * F(3, 5) == 0
    _record(6, ordinary_asymmetric_dyadic_control=True,
            actual_fused_h_and_offset_fraction_oracles=True,
            fused_signed_cancellation_and_absent_tap=True,
            coarse_e=["-159/64", "163/64"], peeled_e=["-33/64", "37/64"],
            complete_source_e="(q1+q2)/32", escaped_sufficient_condition=True,
            strict_QG_improvement_proved=False,
            shared_2T_limit_preserved=True,
            negative_reference="same-H source-cancelled complete single-child bounds",
            new_component_lp_calls=0)


def test_07_complete_padding_classes():
    populations = []
    class_counts = []
    for channels, height in ((64, 32), (128, 8), (128, 14)):
        classes = observer.geometry_classes(height, height, budget=_budget(), enabled=True)
        expected = {}
        for y in range(height):
            for x in range(height):
                mask = _mask(height, height, y, x)
                expected.setdefault(mask, []).append((y, x))
        assert len(classes) == len(expected) == 9
        assert len({item["class_id"] for item in classes}) == len(classes)
        assert {item["mask"] for item in classes} == set(expected)
        for item in classes:
            positions = expected[item["mask"]]
            assert item["population"] == len(positions)
            assert tuple(item["representative"]) in positions
            assert _mask(height, height, *item["representative"]) == item["mask"]
        assert sum(item["population"] for item in classes) == height * height
        populations.append((channels - 1) * height * height)
        class_counts.append(len(classes))
    assert tuple(populations) == (64512, 8128, 24892)
    assert sum(populations) == 97532
    _record(7, complete_position_masks=True, three_spatial_sizes=[32, 8, 14],
            class_counts=class_counts, registered_family_populations=populations,
            registered_total=97532, actual_model_or_parameter_scan=False,
            local_mask_classification_not_two_layer_kernel_fusion=True)


def test_08_summary_boundary():
    meter = _budget()
    assert not meter.owner.failed
    assert 0 < meter.owner.work <= meter.owner.max_work
    assert 0 < meter.owner.entries <= meter.owner.max_entries
    assert tuple(_EVIDENCE) == _NAMES[:-1]
    _record(8, all_ordinary_positive_calls_share_budget=True,
            inherited_tests_not_invoked_by_this_module=True,
            source_bound_math_only=True, no_model_no_solver=True)
    assert tuple(_EVIDENCE) == _NAMES
    _record_file("summary.json", dict(
        schema="d259_shared_difference_source_v1",
        local_source_bound_math_completed=True,
        tests=8, inherited_mathematical_population=4269,
        required_tests=4277, required_test_files=229,
        records=_EVIDENCE, whole_work_used=meter.owner.work,
        numeric_entries=meter.owner.entries,
        accounting_scope="candidate scalar-work/entries through one shared test adapter; not full source/physical cost",
        new_component_solver_free=True, new_component_lp_calls=0,
        source_component_qualified=False, source_census_completed=False,
        source_census_qualified=False, actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        online_lifecycle_qualified=False, new_set_class=False,
        new_domain_qualified=False, new_capability_qualified=False,
        strict_QG_improvement_proved=False, formal_gain=0,
        baseline_solved=1870, independent_solved=61,
    ))
