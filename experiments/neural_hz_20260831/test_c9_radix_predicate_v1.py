from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import c9_radix_predicate_v1 as candidate
from experiments.neural_hz_20260831.c9_radix_predicate_audit_v1 import audit, recover_row
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import balance_row, scaled_exact


def fixture(gap=100, false_constant=False):
    tiny = 2.**-gap
    return SparseHZono(np.array([.1, -.2, .3]), sp.eye(3, format='csr'), sp.csr_matrix((3, 2)),
        sp.csr_matrix([[1., tiny, 0.], [0., 0., 0.]]), sp.csr_matrix([[-1., 0.], [0., 0.]]),
        np.array([0., float(false_constant)]),
        sp.csr_matrix([[1., -.1, tiny * .13], [-1., .1, -tiny * .13]]),
        sp.csr_matrix([[0., tiny * .7], [0., -tiny * .7]]), np.ones(2), frame_id=91, exact=True)


def fraction(value):
    return F.from_float(float(value))


def rational_rows(matrix, binary, rhs, definitions, nc):
    width = nc + binary.shape[1]
    result = []
    for row in range(matrix.shape[0]):
        values = [F(0)] * width
        start, stop = matrix.indptr[row:row + 2]
        for offset in range(start, stop):
            col, coef = int(matrix.indices[offset]), fraction(matrix.data[offset])
            if col < nc:
                values[col] += coef
            else:
                for j, value in enumerate(definitions[col - nc]):
                    values[j] += coef * value
        start, stop = binary.indptr[row:row + 2]
        for offset in range(start, stop):
            values[nc + int(binary.indices[offset])] += fraction(binary.data[offset])
        result.append((values, fraction(rhs[row])))
    return result


def rational_definitions(packed):
    hz, nc, nb = packed.hz, packed.original.n_cont, packed.original.n_bin
    definitions = []
    for slot, row in enumerate(packed.def_rows):
        values = [F(0)] * (nc + nb)
        start, stop = hz.Ac.indptr[row:row + 2]
        pivot = fraction(hz.Ac.data[stop - 1])
        for offset in range(start, stop - 1):
            col, coef = int(hz.Ac.indices[offset]), fraction(hz.Ac.data[offset])
            if col < nc:
                values[col] -= coef / pivot
            else:
                for j, value in enumerate(definitions[col - nc]):
                    values[j] -= coef / pivot * value
        start, stop = hz.Ab.indptr[row:row + 2]
        for offset in range(start, stop):
            values[nc + int(hz.Ab.indices[offset])] -= fraction(hz.Ab.data[offset]) / pivot
        assert sum(abs(value) for value in values) <= 1
        definitions.append(values)
    return definitions


@pytest.mark.parametrize('gap', [20, 60, 100, 300, 900])
@pytest.mark.parametrize('false_constant', [False, True])
def test_fraction_elimination_restores_every_equality_and_inequality(gap, false_constant):
    old = fixture(gap, false_constant)
    packed = candidate.pack(old, enabled=True)
    definitions = rational_definitions(packed)
    for c, b, rhs, roots, scales in ((packed.hz.Ac, packed.hz.Ab, packed.hz.b, packed.eq_roots, packed.eq_scales),
                                    (packed.hz.Auc, packed.hz.Aub, packed.hz.ub, packed.ineq_roots, packed.ineq_scales)):
        recovered = rational_rows(c[roots], b[roots], rhs[roots], definitions, old.n_cont)
        original_c, original_b, original_rhs = (old.Ac, old.Ab, old.b) if c is packed.hz.Ac else (old.Auc, old.Aub, old.ub)
        expected = rational_rows(original_c, original_b, original_rhs, [], old.n_cont)
        for (coefficients, value), scale, original in zip(recovered, scales, expected):
            assert ([coef / F(2)**int(scale) for coef in coefficients], value / F(2)**int(scale)) == original
    proof = audit(packed)
    assert proof['exact_two_way_predicate_equivalence']
    if gap >= 100:
        assert packed.report['packed_logical_rows'] > 0 and packed.report['scale_relays'] > 0


def test_forward_feasibility_and_nonconvex_phase_prefix_are_retained():
    old = fixture()
    packed = candidate.pack(old, enabled=True)
    definitions = rational_definitions(packed)
    original_eq = rational_rows(old.Ac, old.Ab, old.b, [], old.n_cont)
    original_ineq = rational_rows(old.Auc, old.Aub, old.ub, [], old.n_cont)
    physical_eq = rational_rows(packed.hz.Ac, packed.hz.Ab, packed.hz.b, definitions, old.n_cont)
    physical_ineq = rational_rows(packed.hz.Auc, packed.hz.Aub, packed.hz.ub, definitions, old.n_cont)
    feasible = infeasible = 0
    for xi in product((-1, 0, 1), repeat=3):
        for z in product((-1, 1), repeat=2):
            assignment = list(map(F, (*xi, *z)))
            extension = [sum(v * a for v, a in zip(row, assignment)) for row in definitions]
            assert all(abs(v) <= 1 for v in extension)
            def check(eq, ineq):
                return (all(sum(v * a for v, a in zip(row, assignment)) == rhs for row, rhs in eq)
                    and all(sum(v * a for v, a in zip(row, assignment)) <= rhs for row, rhs in ineq))
            before, after = check(original_eq, original_ineq), check(physical_eq, physical_ineq)
            assert before == after
            feasible += before
            infeasible += not before
    assert feasible and infeasible and packed.hz.n_bin == old.n_bin


@pytest.mark.parametrize('values,expected', [([1., .1], 0), ([2.**-60], 40), ([2.**60], -20), ([1., 2.**-100], None), ([], 0)])
def test_closest_zero_row_shift_and_fixed_window(values, expected):
    assert candidate.row_shift(values) == expected


def test_one_combined_emission_matches_defined_c8_two_step_scaling():
    raw, binary, rhs, exponent = np.array([-.1, 1e-14]), np.array([-.7e-12]), .13, 3
    old_c, old_b, old_rhs, old_pivot = balance_row(scaled_exact(raw, -exponent), scaled_exact(binary, -exponent), scaled_exact([rhs], -exponent)[0])
    encoder = candidate.RowEncoder(3, 1, 5)
    row, shift = encoder.encode(np.array([0, 1, 2]), np.append(raw, 1.), np.array([0]), binary, rhs,
        cp=np.array([0, 0, exponent]))
    actual = encoder.eq[row]
    assert np.array_equal(actual[1], np.append(old_c, old_pivot))
    assert np.array_equal(actual[3], old_b) and actual[4] == old_rhs


@pytest.mark.parametrize('field', ['original', 'matrix', 'rhs', 'map', 'hidden'])
def test_seal_detects_all_retained_numeric_mutations(field):
    packed = candidate.pack(fixture(), enabled=True)
    if field == 'original':
        packed.original.b[0] += 1.
    elif field == 'matrix':
        packed.hz.Ac.data[-1] += 1.
    elif field == 'rhs':
        packed.hz.ub[0] += 1.
    elif field == 'map':
        packed.eq_roots[0] += 1
    else:
        packed.hidden = np.ones(100)
    with pytest.raises(ValueError):
        packed.numeric_roots()


@pytest.mark.parametrize('option,value', [('max_aux', 0), ('max_extra_work', 0), ('max_extra_entries', 0),
    ('max_work', 0), ('max_entries', 0), ('max_aux', 16_385), ('max_extra_work', 16_000_001)])
def test_reserves_and_global_caps_fail_closed(option, value):
    with pytest.raises((ValueError, MemoryError)):
        candidate.pack(fixture(), enabled=True, **{option: value})


@pytest.mark.parametrize('kind', ['coefficient', 'pivot', 'rhs', 'missing', 'bound'])
def test_independent_audit_rejects_resealed_corruption(kind):
    packed = candidate.pack(fixture(), enabled=True)
    if kind == 'coefficient':
        packed.hz.Ac.data[0] = np.nextafter(packed.hz.Ac.data[0], np.inf)
    elif kind == 'pivot':
        row = packed.def_rows[0]
        packed.hz.Ac.data[packed.hz.Ac.indptr[row + 1] - 1] *= 1.5
    elif kind == 'rhs':
        packed.hz.ub[0] += 1.
    elif kind == 'missing':
        packed.eq_roots[1] = packed.eq_roots[0]
    else:
        row = packed.def_rows[0]
        packed.hz.Ac.data[packed.hz.Ac.indptr[row]] *= 16.
    packed.seal = packed.fingerprint()
    with pytest.raises(ValueError):
        audit(packed)


def test_native_backend_retains_mixed_radix_predicates_without_options_change():
    old = fixture()
    assert not inspect(old)['passed']
    packed = candidate.pack(old, enabled=True)
    assert audit(packed)['all_redundant_boxes_proved']
    result = inspect(packed.hz)
    assert result['passed'] and result['different_coefficients'] == 0
    assert result['native_thresholds'] == {'small_matrix_value': 1e-9, 'large_matrix_value': 1e15}
    assert not result['solve_called']


def test_default_off_has_no_side_effects(monkeypatch):
    def forbidden(*args):
        raise AssertionError('default-off inspected source')
    monkeypatch.setattr(candidate, 'source_digest', forbidden)
    assert candidate.pack(None) is None


def test_direct_row_reuses_its_range_calculation(monkeypatch):
    calls = []
    native = candidate.row_shift
    def counted(*args):
        calls.append(1)
        return native(*args)
    monkeypatch.setattr(candidate, 'row_shift', counted)
    encoder = candidate.RowEncoder(2, 1, 4)
    encoder.encode(np.array([0, 1]), np.array([.1, -.2]), np.array([0]), np.array([.3]), 0.)
    assert len(calls) == 1


@pytest.mark.parametrize('power', [.5, np.array([.5]), np.iinfo(np.int64).min, 4097])
def test_fractional_or_unbounded_powers_reject(power):
    with pytest.raises(ValueError):
        candidate.exponent_data(np.ones(1), power)


def test_fractional_coordinate_indices_reject():
    encoder = candidate.RowEncoder(2, 1, 4)
    with pytest.raises(ValueError):
        encoder.encode(np.array([.5]), np.array([.1]), np.array([0]), np.array([.3]), 0.)


def test_scalar_integer_power_broadcast_preserves_original_bits():
    encoder = candidate.RowEncoder(2, 1, 4)
    index, scale = encoder.encode(np.array([0, 1]), np.array([.1, -.2]), np.array([0]), np.array([.3]), .4, cp=3)
    assert scale == 0
    assert np.array_equal(encoder.eq[index][1], np.ldexp(np.array([.1, -.2]), 3))
