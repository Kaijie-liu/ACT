"""Four fixed arithmetic tests; no source worker, solver or model is run.

The Fraction oracle is independent of the candidate's overlap-credit formula.
It takes exact supports of four affine functions and, only in these small
tests, enumerates coefficient-interval endpoints.  It is not a candidate
phase, input or property search.  Equal array shapes do not certify that two
readouts were extracted from the same native source axis.
"""
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as accounting
from experiments.neural_hz_20260831.definition_first_20260928.d264_joint_support_kernel_20261006 import joint_support as js


RUN = Path(__file__).resolve().parents[2] / "results/d264_joint_support_kernel_20261006_v1"
_NAMES = (
    "exact_and_interval_fraction_oracle", "shared_source_control_and_bias",
    "default_off_identity_and_rejection", "shared_budget_and_summary",
)
_EVIDENCE = {}
_BUDGET = None


class _Meter:
    """Test-only signature adapter; ordinary positive calls share one owner."""

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


def _iv(lo, hi=None):
    if hi is None:
        hi = lo
    return js.Interval(np.asarray(lo, dtype=np.float64),
                       np.asarray(hi, dtype=np.float64))


def _rat(values):
    return js.sb.from_rationals(tuple((value, value) for value in values),
                                budget=_budget(), enabled=True)


def _exact(value):
    return F.from_float(float(value))


def _ends(value):
    assert value.lo.dtype == np.float64 and value.hi.dtype == np.float64
    assert value.lo.shape == value.hi.shape
    assert np.isfinite(value.lo).all() and np.isfinite(value.hi).all()
    assert (value.lo <= value.hi).all()
    return tuple((_exact(lo), _exact(hi))
                 for lo, hi in zip(value.lo.flat, value.hi.flat))


def _snapshot(values):
    return tuple((id(value), id(array), array.shape, array.strides,
                  array.dtype.str, array.flags.writeable, array.tobytes())
                 for value in values for array in (value.lo, value.hi))


def _oracle(a, b, delta, e0):
    """Exact robust four-plane supports, without the credit identity."""
    av, bv, dv, ev = _ends(a), _ends(b), _ends(delta), _ends(e0)
    assert len(av) == len(bv) and len(dv) == len(ev) == 1
    n = len(av)
    axes = tuple(tuple(dict.fromkeys(pair)) for pair in av + bv + dv + ev)
    directions = None
    largest_ra = largest_rb = largest_constant = F(0)
    smallest_same = smallest_opp = None
    endpoint_cases = 0
    for parameters in product(*axes):
        aa, bb = parameters[:n], parameters[n:2 * n]
        dd, ee = parameters[-2:]
        planes = (
            dd + sum((abs(v) for v in aa), F(0)),
            -dd + sum((abs(v) for v in aa), F(0)),
            dd + 2 * ee + sum((abs(x + 2 * y) for x, y in zip(aa, bb)), F(0)),
            -dd + 2 * ee + sum((abs(-x + 2 * y) for x, y in zip(aa, bb)), F(0)),
        )
        directions = planes if directions is None else tuple(
            max(old, new) for old, new in zip(directions, planes))
        largest_ra = max(largest_ra, sum((abs(v) for v in aa), F(0)))
        largest_rb = max(largest_rb, sum((abs(v) for v in bb), F(0)))
        largest_constant = max(largest_constant, abs(dd))
        same = sum((min(abs(x), 2 * abs(y)) for x, y in zip(aa, bb)
                    if x * y > 0), F(0))
        opp = sum((min(abs(x), 2 * abs(y)) for x, y in zip(aa, bb)
                   if x * y < 0), F(0))
        smallest_same = same if smallest_same is None else min(smallest_same, same)
        smallest_opp = opp if smallest_opp is None else min(smallest_opp, opp)
        endpoint_cases += 1
    assert directions is not None
    independent = (largest_ra + largest_constant
                   + 2 * max(F(0), ev[0][1] + largest_rb))
    return dict(upper=max(directions), directions=directions,
                independent=independent, radius_a=largest_ra,
                radius_b=largest_rb, constant=largest_constant,
                same=smallest_same, opp=smallest_opp,
                endpoint_cases=endpoint_cases)


def _verify_result(result, oracle, dimension, *, tight=False):
    fields = ("upper", "independent_upper", "raw_upper", "radius_a_upper",
              "radius_b_upper", "credit_same_lower", "credit_opp_lower",
              "constant_abs_upper")
    assert type(result.dimension) is int and result.dimension == dimension
    for name in fields:
        value = getattr(result, name)
        assert type(value) is float and np.isfinite(value)
        assert value >= 0.0
    assert _exact(result.upper) >= oracle["upper"]
    assert _exact(result.raw_upper) >= oracle["upper"]
    assert _exact(result.independent_upper) >= oracle["independent"]
    assert result.upper <= result.raw_upper
    assert result.upper <= result.independent_upper
    assert _exact(result.radius_a_upper) >= oracle["radius_a"]
    assert _exact(result.radius_b_upper) >= oracle["radius_b"]
    assert _exact(result.constant_abs_upper) >= oracle["constant"]
    assert _exact(result.credit_same_lower) <= oracle["same"]
    assert _exact(result.credit_opp_lower) <= oracle["opp"]
    if tight:
        # Soundness uses exact comparisons above.  This separate small-control
        # check prevents a vacuous huge envelope from masquerading as success.
        allowance = F(1, 10**9)
        assert _exact(result.upper) - oracle["upper"] < allowance
        assert _exact(result.independent_upper) - oracle["independent"] < allowance


def _call(a, b, delta, e0, *, tight=False):
    values = (a, b, delta, e0)
    before = _snapshot(values)
    oracle = _oracle(*values)
    meter = _budget()
    old_work, old_entries = meter.owner.work, meter.owner.entries
    result = js.joint_support(*values, budget=meter, enabled=True)
    assert 0 < meter.owner.work - old_work <= 68 * a.lo.size + 2279
    assert 0 < meter.owner.entries - old_entries <= 26 * a.lo.size + 1676
    assert _snapshot(values) == before
    _verify_result(result, oracle, a.lo.size, tight=tight)
    return result, oracle


def test_01_exact_and_interval_fraction_oracle():
    point = (_iv((0.5, -0.25, 0.125)), _iv((-0.125, 0.375, 0.25)),
             _iv(0.25), _iv(-0.125))
    result, exact = _call(*point, tight=True)
    assert exact["endpoint_cases"] == 1
    with pytest.raises(AttributeError):
        result.upper = 0.0

    interval = (_iv((0.25, -0.5), (0.5, -0.25)),
                _iv((0.125, 0.125), (0.25, 0.375)),
                _iv(-0.125, 0.25), _iv(-0.25, 0.125))
    uncertain, expanded = _call(*interval)
    assert expanded["endpoint_cases"] == 64
    assert uncertain.credit_same_lower > 0.0 and uncertain.credit_opp_lower > 0.0

    # A coefficient interval crossing zero cannot acquire a definite-sign
    # credit just because its midpoint happens to be positive.
    crossing = (_iv((-0.25, 0.125), (0.5, 0.25)),
                _iv((0.25, -0.375), (0.5, -0.125)), _iv(0.0), _iv(0.0))
    crossed, crossing_oracle = _call(*crossing)
    assert crossed.credit_same_lower == 0.0
    assert crossed.credit_opp_lower > 0.0
    _record(1, point_four_supports=[str(value) for value in exact["directions"]],
            interval_upper=str(expanded["upper"]), interval_endpoint_cases=64,
            crossing_endpoint_cases=crossing_oracle["endpoint_cases"],
            all_comparisons_use_exact_input_binary_values=True,
            interval_credit_is_only_certified_lower_bound=True,
            input_arrays_unchanged=True, result_statistics_immutable=True,
            endpoint_enumeration_is_test_oracle_only=True)


def test_02_shared_source_control_and_bias():
    a = _rat((F(1, 20), F(-1, 20)))
    b = _rat((F(1, 40), F(1, 40)))
    for interval, targets in ((a, (F(1, 20), F(-1, 20))),
                              (b, (F(1, 40), F(1, 40)))):
        assert all(lo <= value <= hi for (lo, hi), value in zip(_ends(interval), targets))
    shared, shared_oracle = _call(a, b, _iv(0.0), _iv(0.0), tight=True)
    assert F(1, 10) <= _exact(shared.upper) < F(1, 10) + F(1, 10**9)
    assert F(1, 5) <= _exact(shared.independent_upper) < F(1, 5) + F(1, 10**9)
    assert shared.upper < shared.independent_upper
    assert shared.credit_same_lower > 0.0 and shared.credit_opp_lower > 0.0

    biased, bias_oracle = _call(_iv((0.5, -0.25)), _iv((0.125, 0.375)),
                               _iv(0.25), _iv(-0.125), tight=True)
    assert bias_oracle["upper"] == F(5, 4)
    assert bias_oracle["independent"] == F(7, 4)
    negative, negative_oracle = _call(_iv((0.5, -0.25)), _iv((0.125, 0.375)),
                                     _iv(0.25), _iv(-2.0), tight=True)
    assert negative_oracle["upper"] == F(1)
    assert negative_oracle["independent"] == F(1)

    same, same_oracle = _call(_iv((0.5, 0.25)), _iv((0.125, 0.25)),
                              _iv(0.0), _iv(0.0), tight=True)
    assert same_oracle["upper"] == same_oracle["independent"] == F(3, 2)
    assert same.credit_same_lower > 0.0 and same.credit_opp_lower == 0.0
    assert abs(_exact(same.upper) - _exact(same.independent_upper)) < F(1, 10**9)
    zero, zero_oracle = _call(_iv((0.0, 0.0)), _iv((0.0, 0.0)),
                              _iv(0.0), _iv(0.0), tight=True)
    assert zero_oracle["upper"] == zero_oracle["independent"] == F(0)
    assert zero.credit_same_lower == zero.credit_opp_lower == 0.0
    empty, empty_oracle = _call(_iv(()), _iv(()), _iv(-0.25), _iv(0.125), tight=True)
    assert empty.dimension == 0 and empty_oracle["upper"] == F(1, 2)
    _record(2, d263_rational_control_upper=shared.upper,
            d263_independent_upper=shared.independent_upper,
            exact_control_joint="1/10", exact_control_independent="1/5",
            input_rational_enclosure_verified=True,
            rounded_input_oracle=str(shared_oracle["upper"]),
            biased_upper=biased.upper, negative_e_upper=negative.upper,
            only_same_sign_no_extra_credit=True, zero_and_empty_sources=True,
            no_jp_row_or_network_installed=True, model_source_identity_certified=False)


def test_03_default_off_identity_and_rejection():
    meter = _budget()
    before = (meter.owner.work, meter.owner.entries)
    assert js.joint_support(None, None, None, None, budget=meter) is None
    assert (meter.owner.work, meter.owner.entries) == before

    # The API deliberately does not certify source identity.  Permuting only
    # one coefficient row changes the declared common-axis problem even when
    # its shape and marginal coefficient multiset are identical.
    aa = _iv((2.0, 1.0, 0.25))
    aligned, aligned_oracle = _call(aa, _iv((0.5, 0.5, -0.5)),
                                    _iv(0.0), _iv(0.0), tight=True)
    permuted, permuted_oracle = _call(aa, _iv((-0.5, 0.5, 0.5)),
                                      _iv(0.0), _iv(0.0), tight=True)
    assert aligned_oracle["upper"] == F(23, 4)
    assert permuted_oracle["upper"] == F(17, 4)
    assert aligned.upper > permuted.upper

    a, b, delta, e0 = _iv((0.5, -0.25)), _iv((0.125, 0.25)), _iv(0.0), _iv(0.0)
    invalid = (
        (_iv(((0.5, -0.25),)), b, delta, e0),
        (a, _iv((0.125,)), delta, e0),
        (a, b, _iv((0.0,)), e0),
        (a, b, delta, _iv((0.0,))),
        (_iv((np.nan, -0.25)), b, delta, e0),
        (_iv((np.inf, -0.25)), b, delta, e0),
        (_iv((1.0, -0.25), (0.5, -0.25)), b, delta, e0),
        (js.Interval(np.asarray((0.5, -0.25), dtype=np.float32),
                     np.asarray((0.5, -0.25), dtype=np.float32)), b, delta, e0),
    )
    for values in invalid:
        private = _Meter()
        old = _snapshot(values)
        with pytest.raises(js.Rejected):
            js.joint_support(*values, budget=private, enabled=True)
        assert _snapshot(values) == old
        assert not private.owner.failed
    with pytest.raises(js.Rejected):
        js.joint_support(a, b, delta, e0, budget=_Meter(), enabled=1)
    assert not meter.owner.failed
    _record(3, default_off_no_work=True, exact_shapes_no_broadcast=True,
            finite_order_dtype_and_scalar_constants_checked=True,
            invalid_inputs_unchanged=True, ordinary_rejection_not_resource_poison=True,
            source_axis_identity_is_caller_contract=True,
            same_shape_not_source_identity=True,
            changed_axis_exact_uppers=["23/4", "17/4"],
            actual_model_binding_qualified=False)


def test_04_shared_budget_and_summary():
    values = (_iv((0.5, -0.25)), _iv((0.125, 0.25)), _iv(0.0), _iv(0.0))
    for limited in (_Meter(max_work=16), _Meter(max_entries=8)):
        old = _snapshot(values)
        with pytest.raises(accounting.Rejected):
            js.joint_support(*values, budget=limited, enabled=True)
        assert limited.owner.failed
        with pytest.raises(accounting.Rejected):
            js.joint_support(*values, budget=limited, enabled=True)
        assert _snapshot(values) == old
    meter = _budget()
    assert not meter.owner.failed
    assert 0 < meter.owner.work <= meter.owner.max_work
    assert 0 < meter.owner.entries <= meter.owner.max_entries
    assert meter.owner.max_bits == 512
    assert tuple(_EVIDENCE) == _NAMES[:-1]
    _record(4, ordinary_positive_cases_share_budget=True,
            work_and_entries_failures_sticky=True, failed_calls_leave_inputs_unchanged=True,
            rational_bit_contract=512, no_extra_extreme_case_population=True,
            fixed_arithmetic_upper_bounds=dict(work="68*n+2279", entries="26*n+1676",
                scope="one enabled kernel call; excludes input construction and test oracle"),
            no_solver_model_or_gpu_calls=True)
    assert tuple(_EVIDENCE) == _NAMES
    _record_file("summary.json", dict(
        schema="d264_joint_support_kernel_v1", tests=4,
        local_joint_support_math_completed=True, joint_support_kernel_math_passed=False,
        inherited_mathematical_population=4281, required_tests=4285, required_test_files=231,
        records=_EVIDENCE, whole_work_used=meter.owner.work, numeric_entries=meter.owner.entries,
        accounting_scope="shared component scalar-work/entries; not full source or physical cost",
        new_component_solver_free=True, new_component_lp_calls=0,
        source_audit_stage_registered=False, source_component_qualified=False,
        source_census_completed=False, source_census_qualified=False,
        actual_model_binding_qualified=False, actual_phase_column_binding_verified=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        complete_physical_qualification=False, online_lifecycle_qualified=False,
        new_set_class=False, new_domain_qualified=False, new_capability_qualified=False,
        strict_QG_improvement_proved=False, formal_gain=0,
        independent_e0_gain=0, new_benchmark_solves=0,
        baseline_solved=1870, independent_solved=61,
    ))
