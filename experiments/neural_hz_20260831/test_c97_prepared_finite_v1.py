"""Equivalent complete preparation, original failures and absent implied scan."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831 import c69_prepared_row_v1 as old
from experiments.neural_hz_20260831 import c97_prepared_row_v1 as new
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c69_birth_emission_v1 import lift as reference_lift
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import same_matrix, equal


@pytest.mark.parametrize('powers', [[0, 0], [3, -4], [-20, 12]])
@pytest.mark.parametrize('sign', [-1, 1])
def test_complete_preparation_same_bits_window_and_exact_inverse(powers, sign):
    cv = np.array([sign * .75, -.125]); cp = np.array(powers, np.int64)
    bv = np.array([.5]); bp = np.array([2], np.int64)
    a = old._prepare(cv, cp, bv, bp, .125, pool=WorkPool(256_000_000))
    b = new._prepare(cv, cp, bv, bp, .125, pool=WorkPool(256_000_000))
    assert a.shift == b.shift and a.head == b.head and a.rhs == b.rhs
    assert equal(a.continuous, b.continuous) and equal(a.binary, b.binary)
    for x, k, y in zip(cv, cp, b.continuous):
        assert F(float(y)) == F(float(x)) * F(2) ** (int(k) + b.shift)


@pytest.mark.parametrize('bad', ['nan', 'inf', 'zero', 'power', 'rhs'])
def test_complete_original_input_failures_remain_fail_closed(bad):
    cv = np.array([.75, -.125]); cp = np.zeros(2, np.int64); rhs = .125
    if bad == 'nan': cv[0] = np.nan
    elif bad == 'inf': cv[0] = np.inf
    elif bad == 'zero': cv[0] = 0.
    elif bad == 'power': cp[0] = 4097
    else: rhs = np.nan
    for module in (old, new):
        with pytest.raises((ValueError, FloatingPointError)):
            module._prepare(cv, cp, np.array([.5]), np.array([0], np.int64), rhs, pool=WorkPool(256_000_000))


def test_retained_original_finite_inverse_checks_and_no_duplicate_output_scan(monkeypatch):
    finite, inverse = [], []
    original_finite, original_equal = np.isfinite, np.array_equal
    def scan(value, *a, **kw):
        finite.append(np.asarray(value).size)
        return original_finite(value, *a, **kw)
    def compare(a, b, *args, **kw):
        inverse.append(np.asarray(a).size)
        return original_equal(a, b, *args, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(np, 'isfinite', scan); patch.setattr(np, 'array_equal', compare)
        got = new._prepare(np.array([.75, -.125]), np.zeros(2, np.int64),
            np.array([.5]), np.zeros(1, np.int64), .125, pool=WorkPool(256_000_000))
    # All three original operands are classified; RHS keeps its own finite
    # check. Both coefficient vectors AND the RHS retain inverse equality.
    assert finite == [3, 1] and inverse == [2, 1, 1]
    assert got is not None


def test_coefficient_inverse_still_rejects_lost_subnormal_bits():
    with pytest.raises(ValueError, match='exactly reversible'):
        new._scale_finite_prepared_coefficients(np.array([np.nextafter(0., 1.)]), np.array([-1]))


@pytest.mark.parametrize('kind', ['chain', 'shared', 'conv_disjoint'])
def test_new_original_source_same_HZ_and_exact_two_comparison_credit(kind):
    _, saved = complete(kind)
    before = reference_lift(saved['expression'], saved['keep'], enabled=True)
    after = lift(saved['expression'], saved['keep'], enabled=True)
    old_fields, fields = before['fields'], after['fields']
    for k in ('Ac', 'Ab', 'Auc', 'Aub', 'Gc', 'Gb'):
        assert same_matrix(getattr(old_fields['hz'], k), getattr(fields['hz'], k))
    for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows', 'owners', 'radix_gauges', 'uid_slabs'):
        assert equal(old_fields[k], fields[k])
    old_report, report = old_fields['report'], fields['report']
    emitted = report['prepared_encoding']
    count = emitted['logical_input_coefficients']
    rows = emitted['logical_input_rows']
    assert report['total_work_upper'] == old_report['total_work_upper'] - 2*count + 128 + rows
    assert report['largest_branch_work_upper'] == old_report['largest_branch_work_upper'] + 128 + rows
    physical = sum(getattr(saved['hz'], k).nnz for k in ('Ac', 'Ab', 'Auc', 'Aub'))
    assert emitted['omitted_post_finite_coefficient_checks'] == physical >= count
    assert emitted['original_input_finite_and_complete_inverse_checks_retained']
    assert emitted['generic_RHS_finite_check_retained']
    assert emitted['once_checked_logical_power_elements'] == count
    assert emitted['original_power_bounds_checked_before_signed_copy']
