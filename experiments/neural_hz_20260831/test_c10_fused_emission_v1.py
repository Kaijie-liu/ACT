from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool, FusedRowEncoder, fold_rows, window_products, aliases
from experiments.neural_hz_20260831.c10_predicate_census_v1 import exact_products, odd_significand
from experiments.neural_hz_20260831.c10_fused_emission_v1 import lift
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c10_fused_emission_audit_v1 import audit
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect


def expr_fixture(wide=False):
    hz = SparseHZono(np.zeros(4), sp.csr_matrix([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
        sp.csr_matrix((4, 1)), sp.csr_matrix([[1., 1e-40 if wide else .5]]),
        sp.csr_matrix([[-1.]]), np.zeros(1), sp.csr_matrix([[.5, 0.]]),
        sp.csr_matrix([[.5]]), np.ones(1), frame_id=45)
    d = sp.diags([.3, -.75, .5, .25], format='csr')
    reduce = sp.csr_matrix([[1., 1., 1., 0.], [0., 0., 0., 1.]])
    return cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(hz, (d, reduce)),), np.array([.125, -.25]), 2, hz.frame_id)


@pytest.mark.parametrize('a,b', [(1., .3), (.3, .7), (-.3, .5), (2.**-20, 2.**-60),
    (2.**40, 2.**-60), (1. + 2.**-52, 1. - 2.**-52),
    (float(2**26 + 1), float(2**26 + 1) * 2.**-28)])
def test_normal_product_matches_fraction_and_general_routine(a, b):
    good, result = window_products([a], [b])
    exact, general = exact_products([a], [b])
    oracle = Fraction(float(result[0])) == Fraction(a) * Fraction(b) and 2.**-20 <= abs(result[0]) <= 2.**40
    assert bool(good[0]) == oracle == bool(exact[0] and 2.**-20 <= abs(general[0]) <= 2.**40)


def test_normal_window_arithmetic_corpus_and_precomputed_ratios():
    rng = np.random.default_rng(20260908)
    a = np.ldexp(rng.uniform(.5, 1., 2048), rng.integers(-19, 41, 2048))
    b = np.ldexp(rng.uniform(.5, 1., 2048), rng.integers(-59, 1, 2048))
    a[::3] *= -1.
    odd, bits = odd_significand(b)
    good, result = window_products(a, b, odd, bits)
    expected = [Fraction(float(r)) == Fraction(float(x)) * Fraction(float(y))
                and 2.**-20 <= abs(r) <= 2.**40 for x, y, r in zip(a, b, result)]
    assert np.array_equal(good, expected)


@pytest.mark.parametrize('a,b', [(0., .5), (np.inf, .5), (1., np.nan), (2.**-21, .5), (1., 2.**-61), (1., 2.)])
def test_specialized_products_reject_outside_proof_domain(a, b):
    with pytest.raises(ValueError):
        window_products([a], [b])


def test_tagged_single_emission_and_exact_quotient_of_original_dag():
    expr = expr_fixture()
    candidate = lift(expr, np.ones(2, bool), enabled=True)
    old = original_lift(expr, np.ones(2, bool), enabled=True)
    maps = {k: getattr(old, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    proof = audit(candidate, old.hz, maps)
    assert proof['tagged_physical_lineage_checked']
    cols, parents, ratios, tagged = aliases(candidate)
    assert cols.size and candidate.hz.n_cont == old.hz.n_cont and candidate.hz.n_bin == old.hz.n_bin
    assert not np.isin(parents, cols).any()
    assert not np.all(np.frexp(np.abs(ratios))[0] == .5)
    assert candidate.report['alias_quotient']['persistent_extra_reconstruction_arrays'] == 0
    assert candidate.report['alias_quotient']['original_completed_hz_constructed'] is False
    before = collect(SimpleNamespace(), old.numeric_roots()).measure()
    after = collect(SimpleNamespace(), candidate.numeric_roots()).measure()
    assert after.resident_entries < before.resident_entries and after.resident_bytes < before.resident_bytes
    assert inspect(candidate.hz)['passed']


def test_wide_original_predicate_radix_and_shared_prefix_preserved():
    expr = expr_fixture(True)
    candidate = lift(expr, np.ones(2, bool), enabled=True, frame_widths=(8, 4))
    old = original_lift(expr, np.ones(2, bool), enabled=True, frame_widths=(8, 4))
    maps = {k: getattr(old, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    assert audit(candidate, old.hz, maps)['original_affine_proof']['all_old_predicates_preserved']
    assert candidate.def_rows.size and candidate.hz.n_bin == 4 and candidate.old_n_cont == 8


def row_encoder(ratio=.3, weight=.3):
    pool = WorkPool(0, 0)
    enc = FusedRowEncoder(3, 1, 100, pool=pool)
    roots, scales = [], []
    for cc, cv in (([0, 1], [-ratio, 1.]), ([0, 1, 2], [-weight, -1., 1.])):
        r, s = enc.encode(np.array(cc), np.array(cv), np.empty(0, np.int64), np.empty(0), 0.)
        roots.append(r)
        scales.append(s)
    return enc, roots, scales


@pytest.mark.parametrize('weight', [.3, -.3])
def test_owned_row_collision_and_cancellation(weight):
    enc, roots, scales = row_encoder(weight=weight)
    roots, scales, report = fold_rows(enc, roots, scales, old_nc=1, old_eq=0, output_slots=[2])
    assert report['selected_aliases'] == 1 and report['collision_groups'] == 1
    assert roots[0] == -1 and scales.view(np.float64)[0] == .3
    ac, ab, rhs = enc.matrices(enc.eq)
    assert ac.has_canonical_format and not np.any(ac.data == 0.)
    assert ac.nnz == (1 if weight < 0 else 2)


def test_inexact_collision_rejects_without_returning_alternate_path():
    enc, roots, scales = row_encoder(weight=.7)
    with pytest.raises(ValueError, match='collision sum'):
        fold_rows(enc, roots, scales, old_nc=1, old_eq=0, output_slots=[2])


def test_shared_pool_charges_radix_and_alias_without_double_reservation():
    pool = WorkPool(255_999_950, 199_999_950)
    enc = FusedRowEncoder(3, 1, 100, pool=pool)
    enc.charge(20)
    pool.charge('aliases', 30)
    assert pool.used == 50 and enc.extra_work == 20
    with pytest.raises(MemoryError):
        pool.charge('aliases', 1)
    assert pool.used == 50


@pytest.mark.parametrize('kwargs', [{'max_work': 0}, {'max_branch_work': 0}, {'max_entries': 0},
                                  {'max_work': 256_000_001}, {'frame_widths': (1, 1)}])
def test_fixed_complete_construction_guards(kwargs):
    with pytest.raises((ValueError, MemoryError)):
        lift(expr_fixture(), np.ones(2, bool), enabled=True, **kwargs)


@pytest.mark.parametrize('field', ['tag', 'ratio', 'hidden', 'phase'])
def test_tag_and_numeric_metadata_mutations_rejected(field):
    candidate = lift(expr_fixture(), np.ones(2, bool), enabled=True)
    tagged = np.flatnonzero(candidate.eq_roots < 0)
    if field == 'tag':
        candidate.eq_roots[tagged[0]] -= 1
    elif field == 'ratio':
        candidate.eq_scales.view(np.float64)[tagged[0]] *= .5
    elif field == 'hidden':
        candidate.report['hidden'] = np.zeros(5)
    else:
        candidate.hz.Ab.data[0] *= -1.
    with pytest.raises((ValueError, TypeError)):
        candidate.numeric_roots()


def test_default_off_does_not_touch_input():
    assert lift(object(), object()) is None


def test_no_hit_and_radix_individual_budget_are_fail_closed():
    with pytest.raises(ValueError, match='no locally'):
        lift(expr_fixture(), np.zeros(2, bool), enabled=True)
    enc = FusedRowEncoder(2, 1, 100, pool=WorkPool(0, 0), max_extra_work=10)
    with pytest.raises(MemoryError, match='radix'):
        enc.charge(11)


def test_independent_audit_rejects_resealed_ratio_corruption():
    expr = expr_fixture()
    candidate = lift(expr, np.ones(2, bool), enabled=True)
    original = original_lift(expr, np.ones(2, bool), enabled=True)
    tag = np.flatnonzero(candidate.eq_roots < 0)[0]
    candidate.eq_scales.view(np.float64)[tag] *= .5
    candidate.seal = candidate.fingerprint()
    maps = {k: getattr(original, k) for k in ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')}
    with pytest.raises(ValueError, match='ratio'):
        audit(candidate, original.hz, maps)
