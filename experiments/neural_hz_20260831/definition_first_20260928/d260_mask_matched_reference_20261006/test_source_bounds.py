"""Four fixed mathematical tests of the same-H, mask-matched reference.

Fraction appears only in independent small test oracles.  These tests neither
read models nor install QG, change its residual, or establish native binding.
"""
from fractions import Fraction as F
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as accounting
from experiments.neural_hz_20260831.definition_first_20260928.d260_mask_matched_reference_20261006 import source_bounds as sb
from experiments.neural_hz_20260831.definition_first_20260928.d260_mask_matched_reference_20261006 import source_observer as observer


RUN = Path(__file__).resolve().parents[2] / "results/d260_mask_matched_reference_20261006_v1"
_NAMES = ("masked_fraction_oracle", "original_bn_carrier_preserved",
          "fair_reference_and_population", "default_off_budget_and_summary")
_EVIDENCE = {}
_BUDGET = None
_CASES = {}


class _Meter:
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
    return sb.Interval(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))


def _exact(value):
    return F.from_float(float(value))


def _ends(value):
    assert value.lo.dtype == value.hi.dtype == np.float64
    assert value.lo.shape == value.hi.shape
    assert np.isfinite(value.lo).all() and np.isfinite(value.hi).all()
    assert (value.lo <= value.hi).all()
    return tuple((_exact(lo), _exact(hi)) for lo, hi in zip(value.lo.flat, value.hi.flat))


def _tight(value, expected):
    actual = _ends(value)
    expected = tuple(expected)
    assert len(actual) == len(expected)
    for (lo, hi), (wanted_lo, wanted_hi) in zip(actual, expected):
        assert lo <= wanted_lo <= wanted_hi <= hi
        # Only a nonvacuity check; the enclosing inequalities above are exact.
        assert wanted_lo - lo < F(1, 10**9) and hi - wanted_hi < F(1, 10**9)


def _mask(height, width, y, x):
    return tuple(tuple(0 <= y + ky - 1 < height and 0 <= x + kx - 1 < width
                       for kx in range(3)) for ky in range(3))


def _make_case(weights, bias, source, a, b):
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    ledger = sb.conv_channel(weights, bias, source, 1, _budget(), enabled=True)
    carrier = sb.bn_carrier(ledger.bounds, a, b, budget=_budget(), enabled=True)
    before = tuple(array.copy() for array in
                   (weights, bias, source.lo, source.hi, carrier.nominal_a,
                    carrier.nominal_b, carrier.error, carrier.bounds.lo, carrier.bounds.hi))
    result = sb.masked_child_bounds(ledger, source, carrier, budget=_budget(), enabled=True)
    after = (weights, bias, source.lo, source.hi, carrier.nominal_a,
             carrier.nominal_b, carrier.error, carrier.bounds.lo, carrier.bounds.hi)
    assert all(np.array_equal(old, current) for old, current in zip(before, after))
    return dict(weights=weights, bias=bias, source=source, a=a, b=b,
                ledger=ledger, carrier=carrier, result=result)


def _signed_case():
    if "signed" not in _CASES:
        weights = (
            (((1, -0.5, 0.25), (0.5, -1, 0.25), (1, 0.25, -0.75)),
             ((-0.25, 0.5, -1), (1, 0.25, -0.5), (0.25, -0.75, 0.5))),
            (((-0.5, 0.25, 0.75), (1, -0.25, 0.5), (-1, 0.5, 0.25)),
             ((0.5, -0.25, 1), (-0.75, 0.5, -0.25), (1, -0.5, 0.25))),
        )
        _CASES["signed"] = _make_case(
            weights, (0.1875, -0.15625), _iv((0.25, -0.5), (0.75, -0.125)),
            _iv((0.5, -1.25), (0.75, -0.75)), _iv((-0.125, 0.0625), (0.25, 0.1875)))
    return _CASES["signed"]


def _positive_case():
    if "positive" not in _CASES:
        _CASES["positive"] = _make_case(
            np.ones((2, 1, 3, 3), dtype=np.float64), (0.0, 0.0),
            _iv((1.0,), (2.0,)), _iv((1.0, 1.0), (2.0, 2.0)),
            _iv((0.0, 0.0), (0.0, 0.0)))
    return _CASES["positive"]


def _oracle(case, mask):
    """Independent exact per-tap support, Conv bias, then ORIGINAL BN E."""
    weights, bias = case["weights"], case["bias"]
    source = _ends(case["source"])
    carrier = case["carrier"]
    pre = []
    for co in range(weights.shape[0]):
        lo = hi = _exact(bias[co])
        for ky in range(3):
            for kx in range(3):
                if not mask[ky][kx]:
                    continue
                for ci in range(weights.shape[1]):
                    coefficient = _exact(weights[co, ci, ky, kx])
                    products = tuple(coefficient * endpoint for endpoint in source[ci])
                    lo += min(products)
                    hi += max(products)
        a, b, error = (_exact(v[co]) for v in
                       (carrier.nominal_a, carrier.nominal_b, carrier.error))
        scaled = (a * lo, a * hi)
        pre.append((min(scaled) + b - error, max(scaled) + b + error))
    return tuple(pre)


def test_01_masked_fraction_oracle():
    case = _signed_case()
    result = case["result"]
    weights, source = case["weights"], _ends(case["source"])
    assert result.tap_bounds.lo.shape == (2, 3, 3)
    assert result.preactivation.lo.shape == result.upper.shape == (9, 2)
    assert set(result.masks) == {_mask(3, 3, y, x) for y in range(3) for x in range(3)}
    assert len(set(result.masks)) == 9
    assert _exact(case["carrier"].nominal_a[1]) < 0
    tap_oracle = []
    for co in range(2):
        for ky in range(3):
            for kx in range(3):
                lo = hi = F(0)
                for ci in range(2):
                    products = tuple(_exact(weights[co, ci, ky, kx]) * endpoint
                                     for endpoint in source[ci])
                    lo += min(products)
                    hi += max(products)
                tap_oracle.append((lo, hi))
    _tight(result.tap_bounds, tap_oracle)
    for index, mask in enumerate(result.masks):
        expected = _oracle(case, mask)
        _tight(sb.Interval(result.preactivation.lo[index], result.preactivation.hi[index]), expected)
        for co, (_, exact_upper) in enumerate(expected):
            assert _exact(result.upper[index, co]) >= max(F(0), exact_upper)
    assert not result.upper.flags.writeable and not result.old_upper.flags.writeable
    assert not result.preactivation.lo.flags.writeable and not result.tap_bounds.lo.flags.writeable
    _record(1, exact_fraction_oracle=True, full_masks=9, complete_output_channels=2,
            complete_input_channels=2, signed_weights=True, negative_bn_scale=True,
            original_conv_and_bn_bias_preserved=True, inputs_unchanged=True)


def test_02_original_bn_carrier_preserved():
    case = _positive_case()
    carrier, result = case["carrier"], case["result"]
    corner = result.masks.index(_mask(3, 3, 0, 0))
    for co in range(2):
        assert _exact(carrier.nominal_a[co]) == F(3, 2)
        assert _exact(carrier.nominal_b[co]) == 0
        original_error = _exact(carrier.error[co])
        assert original_error >= 9
        # Four valid inputs all at 2 give Conv=8.  The original per-output
        # epsilon=+1 is still legal even after a tighter spatial range.
        same_h_value = F(12) + original_error
        assert _exact(result.preactivation.hi[corner, co]) >= same_h_value
        assert _exact(result.upper[corner, co]) >= same_h_value
        assert same_h_value > 16  # Wrongly re-estimating E as 4 would give 16.
        assert original_error > F(1, 2) * 8
    _tight(sb.Interval(result.preactivation.lo[corner], result.preactivation.hi[corner]),
           _oracle(case, result.masks[corner]))
    assert np.array_equal(result.old_upper, np.maximum(carrier.bounds.hi, 0.0))
    _record(2, original_bn_error_retained=True, no_mask_error_reestimation=True,
            original_error_lower_bound="9", wrong_reestimated_error="4",
            same_h_corner_upper_at_least="21", wrong_upper="16",
            original_per_output_error_identity_required=True,
            no_native_error_identity_constructed=True)


def test_03_fair_reference_and_population():
    result = _positive_case()["result"]
    expected = np.minimum(np.maximum(result.preactivation.hi, 0.0), result.old_upper[None, :])
    assert np.array_equal(result.upper, expected)
    assert (result.upper <= result.old_upper[None, :]).all()
    e_bounds, tau, parent_u = (-15.0, 15.0), 1.0, (1.0, 1.0)
    old_upper = tuple(float(v) for v in result.old_upper)
    excluded_masks = set()
    for index, mask in enumerate(result.masks):
        fair_upper = tuple(float(v) for v in result.upper[index])
        answer = observer.payment_classification(
            e_bounds, tau, parent_u, old_upper, fair_child_upper=fair_upper,
            budget=_budget(), enabled=True)
        fair = answer["fair_reference"]
        assert not answer["excluded_by_d258"]
        assert not answer["excluded_by_d258"] or fair["excluded_by_d258"]
        assert answer["exact_payment_and_threshold"][:2] == fair["exact_payment_and_threshold"][:2]
        expected_thresholds = tuple(_exact(value) + 2 for value in fair_upper)
        assert fair["exact_payment_and_threshold"][2:] == tuple(
            (value.numerator, value.denominator) for value in expected_thresholds)
        if fair["excluded_by_d258"]:
            excluded_masks.add(mask)
        assert fair["excluded_by_d258"] == (not all(all(row) for row in mask))
        already = observer.payment_classification(
            (-25.0, 25.0), tau, parent_u, old_upper, fair_child_upper=fair_upper,
            budget=_budget(), enabled=True)
        assert already["excluded_by_d258"] and already["fair_reference"]["excluded_by_d258"]
    assert len(excluded_masks) == 8
    populations = []
    for channels, side in ((64, 32), (128, 8), (128, 14)):
        classes = observer.geometry_classes(side, side, budget=_budget(), enabled=True)
        exact_population = {}
        for y in range(side):
            for x in range(side):
                mask = _mask(side, side, y, x)
                exact_population[mask] = exact_population.get(mask, 0) + 1
        assert len(classes) == len(exact_population) == 9
        assert {item["mask"] for item in classes} == set(result.masks)
        assert all(item["population"] == exact_population[item["mask"]] for item in classes)
        assert sum(item["population"] for item in classes) == side * side
        populations.append((channels - 1) * side * side)
    assert populations == [64512, 8128, 24892] and sum(populations) == 97532
    _record(3, fair_upper_is_minimum=True, same_e_tau_parent_bounds=True,
            old_exclusion_preserved=True, synthetic_newly_excluded_masks=8,
            complete_mask_classes=9, registered_populations=populations,
            registered_total=97532, actual_model_classification=False,
            escaped_condition_not_strict_improvement=True)


def test_04_default_off_budget_and_summary():
    meter = _budget()
    before = (meter.owner.work, meter.owner.entries)
    assert sb.masked_child_bounds(None, None, None, budget=meter) is None
    assert observer.payment_classification(None, None, None, None,
                                           fair_child_upper=None, budget=meter) is None
    assert (meter.owner.work, meter.owner.entries) == before
    case = _positive_case()
    negative = _Meter()
    with pytest.raises(sb.Rejected):
        sb.masked_child_bounds(case["ledger"], _iv((1.0,), (3.0,)), case["carrier"],
                               budget=negative, enabled=True)
    with pytest.raises(sb.Rejected):
        sb.masked_child_bounds(case["ledger"], case["source"], object(),
                               budget=negative, enabled=True)
    with pytest.raises(sb.Rejected):
        sb.masked_child_bounds(case["ledger"], case["source"], case["carrier"],
                               budget=negative, enabled=1)
    for low in (_Meter(max_work=32), _Meter(max_entries=16)):
        with pytest.raises(accounting.Rejected):
            sb.masked_child_bounds(case["ledger"], case["source"], case["carrier"],
                                   budget=low, enabled=True)
        assert low.owner.failed
        with pytest.raises(accounting.Rejected):
            sb.masked_child_bounds(case["ledger"], case["source"], case["carrier"],
                                   budget=low, enabled=True)
    assert not meter.owner.failed
    assert 0 < meter.owner.work <= meter.owner.max_work
    assert 0 < meter.owner.entries <= meter.owner.max_entries
    assert tuple(_EVIDENCE) == _NAMES[:-1]
    _record(4, default_off_no_work=True, shared_budget_sticky=True,
            incompatible_zero_hull_rejected=True, owned_carrier_required=True,
            ordinary_positive_cases_share_budget=True,
            numerical_compatibility_not_source_identity=True)
    assert tuple(_EVIDENCE) == _NAMES
    _record_file("summary.json", dict(
        schema="d260_mask_matched_reference_v1", tests=4,
        local_mask_matched_math_completed=True, mask_matched_reference_math_passed=False,
        inherited_mathematical_population=4277, required_tests=4281, required_test_files=230,
        records=_EVIDENCE, whole_work_used=meter.owner.work, numeric_entries=meter.owner.entries,
        accounting_scope="shared component scalar-work/entries; not full source or physical cost",
        new_component_solver_free=True, new_component_lp_calls=0,
        source_component_qualified=False, source_census_completed=False,
        source_census_qualified=False, actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        online_lifecycle_qualified=False, new_set_class=False, new_domain_qualified=False,
        new_capability_qualified=False, strict_QG_improvement_proved=False,
        formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        baseline_solved=1870, independent_solved=61,
    ))
