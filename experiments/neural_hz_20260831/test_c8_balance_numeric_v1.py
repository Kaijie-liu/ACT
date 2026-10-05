from fractions import Fraction as F

import numpy as np
import pytest

from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831 import c8_dyadic_balance_v1 as candidate
from experiments.neural_hz_20260831.c8_dyadic_balance_audit_v1 import audit
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
from experiments.neural_hz_20260831.test_c8_dyadic_balance_v1 import rational_definitions, rational_original


@pytest.mark.parametrize('values,parents', [
    ([1., 2.**-50, .125], [0, 1, 3]),
    ([.1, -.2, .3], [15, 0, -20]),
    ([2.**-1000, 2.**-600, 2.**-50], [0, 0, 0]),
    ([2.**100, 2.**-100, 1.], [0, 100, 0]),
    ([.5] * 1000, [0] * 1000),
])
def test_box_contains_exact_real_sum_for_mixed_exponents(values, parents):
    values, parents = np.asarray(values), np.asarray(parents)
    exponent = candidate.box_exponent(values, parents)
    exact = sum(abs(F.from_float(float(v))) * F(2)**int(p) for v, p in zip(values, parents))
    assert exact <= F(2)**exponent and exponent >= 0


def test_sum_bound_improves_on_maximum_times_count():
    values = np.array([1.] + [2.**-20] * 1023)
    old = int(np.frexp(values)[1].max()) + (values.size - 1).bit_length()
    new = candidate.box_exponent(values)
    assert new < old and new == 2


@pytest.mark.parametrize('values', [[], [0.], [np.inf], [np.nan]])
def test_invalid_box_rejects(values):
    with pytest.raises(ValueError):
        candidate.box_exponent(values)


def test_all_coefficient_and_rhs_bits_survive_nontrivial_row_scaling():
    coefficients, binary, constant = np.array([-.1, 1e-14]), np.array([-.7e-12]), .13
    c, b, rhs, pivot = candidate.balance_row(coefficients, binary, constant)
    assert pivot > 1 and np.frexp(pivot)[0] == .5
    assert min(abs(np.concatenate((c, b, [pivot])))) >= 2.**-20
    assert max(abs(np.concatenate((c, b, [pivot])))) <= 2.**40
    assert all(F.from_float(float(v)) / F.from_float(pivot) == F.from_float(float(o)) for v, o in zip(c, coefficients))
    assert all(F.from_float(float(v)) / F.from_float(pivot) == F.from_float(float(o)) for v, o in zip(b, binary))
    assert F.from_float(rhs) / F.from_float(pivot) == F.from_float(constant)


def test_no_permitted_row_window_rejects_without_dropping_a_term():
    with pytest.raises(ValueError, match='window'):
        candidate.balance_row(np.array([2.**-100]), np.empty(0), 0.)


def scaled_fixture():
    s = source(8)
    s.Gc.data[0] = 1e-14
    s.Gb.data[0] = -.7e-12
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, ()),), np.zeros(8), 8, s.frame_id)
    return candidate.lift(expr, np.ones(8, dtype=bool), enabled=True)


def test_scaled_nonconvex_hz_has_exact_rational_output_and_native_matrix_retention():
    lifted = scaled_fixture()
    assert any(lifted.hz.Ac[i, lifted.old_n_cont + i - lifted.old_n_eq] > 1
               for i in range(lifted.old_n_eq, lifted.hz.n_eq))
    assert rational_definitions(lifted)[1] == rational_original(lifted.expression, lifted.old_n_cont, lifted.old_n_bin, lifted.keep)
    assert audit(lifted)['all_positive_dyadic_row_scales_verified']
    native = inspect(lifted.hz)
    assert native['passed'] and native['different_coefficients'] == 0
    assert native['native_thresholds'] == {'small_matrix_value': 1e-9, 'large_matrix_value': 1e15}
    assert not native['solve_called'] and not native['presolve_called']
    assert native['lowered_n_bin'] == lifted.old_n_bin


def test_resealed_non_dyadic_pivot_is_rejected():
    lifted = scaled_fixture()
    row = lifted.old_n_eq
    lifted.hz.Ac.data[lifted.hz.Ac.indptr[row + 1] - 1] *= 1.5
    lifted.seal = lifted.fingerprint()
    with pytest.raises(ValueError, match='pivot'):
        audit(lifted)


def test_old_predicates_are_not_silently_repaired_or_dropped():
    lifted = scaled_fixture()
    lifted.hz.Ac.data[0] = 1e-14
    native = inspect(lifted.hz)
    assert not native['passed'] and native['different_coefficients'] > 0
    assert native['checks']['input_hz_unchanged']


def test_enlarged_output_coordinates_reject_before_publication():
    s = source(8)
    s.c[0] = 2.**50
    expr = cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(s, ()),), np.zeros(8), 8, s.frame_id)
    with pytest.raises(ValueError, match='output factor scale'):
        candidate.lift(expr, np.ones(8, dtype=bool), enabled=True)
