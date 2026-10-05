from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c10_predicate_census_v1 import census, exact_products
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('left,right', [(1., .3), (.3, .7), (-.3, .5), (2.**-1074, 2.),
    (2.**-1074, .5), (2.**1023, 2.), (1. + 2.**-52, 1. + 2.**-52),
    (float(2**26 + 1), float(2**26 + 1)), (float(2**27 - 1), float(2**27 - 1))])
def test_exact_product_criterion_against_fraction(left, right):
    exact, product = exact_products(np.array([left]), right)
    expected = np.isfinite(product[0]) and product[0] != 0. and Fraction(float(product[0])) == Fraction(left) * Fraction(right)
    assert bool(exact[0]) == expected


def test_random_normal_and_subnormal_products_match_exact_rationals():
    rng = np.random.default_rng(20260908)
    with np.errstate(over='ignore', under='ignore'):
        a = np.ldexp(rng.uniform(-1., 1., 2048), rng.integers(-1070, 1024, 2048))
        b = np.ldexp(rng.uniform(-1., 1., 2048), rng.integers(-1070, 1024, 2048))
    valid = np.isfinite(a) & np.isfinite(b) & (a != 0.) & (b != 0.)
    a, b = a[valid], b[valid]
    exact, product = exact_products(a, b)
    expected = [np.isfinite(p) and p != 0. and Fraction(float(p)) == Fraction(float(x)) * Fraction(float(y))
                for x, y, p in zip(a, b, product)]
    assert np.array_equal(exact, expected)


@pytest.mark.parametrize('value', [0., np.inf, np.nan])
def test_invalid_product_operand_is_rejected(value):
    with pytest.raises(ValueError):
        exact_products(np.array([value]), .5)


def fixture():
    ac = sp.csr_matrix([[1., 0., 0., 0., 0.], [-.5, 0., 1., 0., 0.],
                       [0., 0., -.3, 1., 0.], [0., 0., -.2, -.7, 1.]])
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        ac, sp.csr_matrix([[-1.], [0.], [0.], [0.]]), np.zeros(4),
        sp.csr_matrix((0, 5)), sp.csr_matrix((0, 1)), np.zeros(0), frame_id=7, exact=True)
    kw = {'old_n_cont': 2, 'logical_n_cont': 5, 'old_n_eq': 1,
        'eq_roots': np.arange(4, dtype=np.int64), 'def_rows': np.zeros(0, dtype=np.int64)}
    return hz, kw


def test_complete_structural_counts_and_no_input_mutation():
    hz, kw = fixture()
    before = source_digest(hz)
    report, table = census(hz, **kw)
    assert source_digest(hz) == before
    assert report['main_factors'] == 3 and report['main_value_dead'] == 2
    assert report['main_single_consumer'] == 1
    assert report['homogeneous_aliases'] == 2 and report['power_two_aliases'] == 1
    assert report['aliases_all_incident_products_exact'] == 1
    assert report['aliases_all_incident_products_window_safe'] == 2
    assert report['aliases_exact_and_window_safe'] == 1
    assert report['alias_incident_products_checked'] == 3
    assert table['all_products_exact'].tolist() == [True, False, False]
    assert table['single_elimination_nnz_upper'][:2].tolist() == [-2, -2]
    assert not report['transformation_constructed'] and not report['solver_executed']
    assert not report['predicate_addition_collision_exactness_proved']


def test_redundant_box_and_native_window_are_separate_guards():
    hz, kw = fixture()
    hz.Ac.data[hz.Ac.indptr[1]] = -2.
    report, table = census(hz, **kw)
    assert report['homogeneous_aliases'] == 2 and report['local_box_redundant_aliases'] == 1
    hz, kw = fixture()
    hz.Ac.data[hz.Ac.indptr[1]] = -2.**-50
    report, table = census(hz, **kw)
    assert report['aliases_all_incident_products_exact'] == 1
    assert report['aliases_exact_and_window_safe'] == 0


@pytest.mark.parametrize('change', ['row_map', 'missing_row', 'old_prefix', 'cap', 'increased_cap', 'entry_cap'])
def test_invalid_map_or_budget_rejects_without_mutation(change):
    hz, kw = fixture()
    if change == 'row_map':
        kw['eq_roots'][1] = 0
    elif change == 'missing_row':
        kw['def_rows'] = np.array([1], dtype=np.int64)
    elif change == 'old_prefix':
        kw['old_n_cont'] = 6
    elif change == 'cap':
        kw['max_work'] = 0
    elif change == 'increased_cap':
        kw['max_work'] = 256_000_001
    else:
        kw['max_entries'] = 0
    before = source_digest(hz)
    with pytest.raises((ValueError, MemoryError)):
        census(hz, **kw)
    assert source_digest(hz) == before
