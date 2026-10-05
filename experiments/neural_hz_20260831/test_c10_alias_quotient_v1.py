from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, _lower_hz_milp
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import quotient, exact_sum, frontier, storage
from experiments.neural_hz_20260831.c10_alias_quotient_audit_v1 import audit
from experiments.neural_hz_20260831.c10_predicate_census_v1 import census
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def fixture():
    ac = sp.csr_matrix([[1., 0., 0., 0., 0., 0.],
        [-.5, 0., 1., 0., 0., 0.], [0., 0., -.25, 1., 0., 0.],
        [.75, 0., 0., 0., 1., 0.], [0., 0., -1., -1., -.5, 1.]])
    hz = SparseHZono(np.array([.125]), sp.csr_matrix([[0., 0., 0., 0., 0., 1.]]), sp.csr_matrix([[.25]]),
        ac, sp.csr_matrix([[-1.], [0.], [0.], [0.], [0.]]), np.zeros(5),
        sp.csr_matrix([[0., 0., 0., 1., 1., 0.]]), sp.csr_matrix([[.5]]), np.ones(1),
        frame_id=17, exact=True)
    kw = dict(old_n_cont=2, logical_n_cont=6, old_n_eq=1,
              eq_roots=np.arange(5, dtype=np.int64), def_rows=np.zeros(0, dtype=np.int64))
    return hz, kw


def simple(ratio=.3, weight=.5):
    hz = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-ratio, 1., 0.], [-weight, -1., 1.]]), sp.csr_matrix((2, 1)), np.zeros(2),
        frame_id=8, exact=True)
    return hz, dict(old_n_cont=1, logical_n_cont=3, old_n_eq=0,
                   eq_roots=np.arange(2, dtype=np.int64), def_rows=np.zeros(0, dtype=np.int64))


@pytest.mark.parametrize('values', [[.5, .25], [.3, -.3], [.3, .3], [-1., .25, .25],
                                  [2.**40, -2.**40, 2.**-20], []])
def test_exact_sum_fraction_identity(values):
    assert Fraction(exact_sum(values)) == sum(map(Fraction, values), Fraction(0))


@pytest.mark.parametrize('values', [[.3, .7], [2.**40, 2.**40], [2.**-21],
                                  [np.inf], [np.nan], [1., -1. + 2.**-30]])
def test_inexact_or_window_collision_rejected(values):
    with pytest.raises(ValueError):
        exact_sum(values)


def test_default_off_does_not_inspect_input():
    assert quotient(object()) is None


def test_independent_frontier_signed_ratios_and_all_predicate_kinds():
    old, kw = fixture()
    before = source_digest(old)
    reduced = quotient(old, enabled=True, **kw)
    assert reduced.columns.tolist() == [3, 4]
    assert reduced.parents.tolist() == [2, 0]
    assert reduced.ratios.tolist() == [.25, -.75]
    report = audit(old, reduced)
    assert report['all_definitions_checked'] == 2
    assert report['fraction_changed_rows_checked'] == 2
    assert reduced.hz.n_cont == old.n_cont and reduced.hz.n_bin == old.n_bin
    assert reduced.hz.frame_id == old.frame_id and source_digest(old) == before
    assert reduced.report['quotient_with_certificate_bytes'] < reduced.report['original_component_bytes']
    assert reduced.report['certificate_array_bytes'] == 64
    assert set(reduced.numeric_roots()) == {'hz', 'columns', 'parents', 'ratios', 'defining_rows'}
    assert not any(value is old for value in vars(reduced).values())
    lowered = _lower_hz_milp(reduced.hz, prune_unused=True, coalesce_rows=True,
                             project_inactive_cont=False, fix_implied_binary=False)
    assert lowered.n_cont == 3 and lowered.n_bin == 1


def test_frontier_deterministic_and_maximal_across_chains():
    # Independent oracle greedily removes candidates adjacent to each selected
    # vertex, in the same declared descending order, without production masks.
    table = {'column': np.arange(2, 10), 'parent': np.array([0, 2, 3, 2, 5, 0, 7, 8]),
             'all_products_exact': np.ones(8, bool), 'products_window_safe': np.ones(8, bool)}
    selected, blocked = [], set()
    for col, parent in reversed(list(zip(table['column'], table['parent']))):
        if int(col) not in blocked:
            selected.append(int(col))
            blocked.add(int(parent))
    actual = table['column'][frontier(table, 10)].tolist()
    assert actual == sorted(selected)
    assert all(int(p) not in actual for c, p in zip(table['column'], table['parent']) if int(c) in actual)
    assert all(c in actual or c in blocked for c in table['column'])


def test_non_power_two_and_exact_sibling_collision():
    old, kw = simple(.3, .3)
    reduced = quotient(old, enabled=True, **kw)
    assert reduced.ratios.tolist() == [.3]
    assert audit(old, reduced)['fraction_changed_rows_checked'] == 1
    assert Fraction(float(reduced.hz.Ac.data[0])) == -2 * Fraction(.3)


def test_two_selected_siblings_merge_to_one_parent_exactly():
    old = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-.3, 1., 0., 0.], [-.3, 0., 1., 0.], [0., -1., -1., 1.]]),
        sp.csr_matrix((3, 1)), np.zeros(3), frame_id=31)
    reduced = quotient(old, enabled=True, old_n_cont=1, logical_n_cont=4, old_n_eq=0,
        eq_roots=np.arange(3, dtype=np.int64), def_rows=np.zeros(0, dtype=np.int64))
    assert reduced.columns.tolist() == [1, 2] and reduced.parents.tolist() == [0, 0]
    assert audit(old, reduced)['all_definitions_checked'] == 2
    assert reduced.hz.Ac.nnz == 2


def test_radix_and_later_phase_slots_remain_protected_even_when_aliases():
    old = SparseHZono(np.zeros(1), sp.csr_matrix([[0., 0., 0., 0., 1.]]), sp.csr_matrix((1, 1)),
        sp.csr_matrix([[-.3, 1., 0., 0., 0.], [-.3, -1., 1., 0., 0.],
                       [0., 0., -.5, 1., 0.], [0., 0., 0., -.5, 1.]]),
        sp.csr_matrix((4, 1)), np.zeros(4), frame_id=32)
    reduced = quotient(old, enabled=True, old_n_cont=1, logical_n_cont=3, old_n_eq=0,
        eq_roots=np.arange(2, dtype=np.int64), def_rows=np.array([2], dtype=np.int64))
    assert reduced.columns.tolist() == [1]
    assert audit(old, reduced)['all_definitions_checked'] == 1
    assert reduced.hz.Ac[:, 3:].nnz == 3


def test_exact_cancellation_erases_zero_without_retaining_oversized_owner():
    old, kw = simple(.3, -.3)
    reduced = quotient(old, enabled=True, **kw)
    assert audit(old, reduced)['all_definitions_checked'] == 1
    assert reduced.hz.Ac.nnz == 1 and reduced.hz.Ac.indices.tolist() == [2]
    assert not np.any(reduced.hz.Ac.data == 0.)
    actual = storage(reduced.numeric_roots())
    assert actual.resident_bytes == reduced.report['quotient_with_certificate_bytes']


def test_inexact_collision_rejects_whole_candidate_without_input_change():
    old, kw = simple(.3, .7)
    before = source_digest(old)
    with pytest.raises(ValueError, match='collision sum is not exactly'):
        quotient(old, enabled=True, **kw)
    assert source_digest(old) == before


def feasible(hz, x, z):
    if any(abs(v) > 1 for v in x) or any(v not in (-1, 1) for v in z):
        return False
    for continuous, binary, rhs, equality in ((hz.Ac, hz.Ab, hz.b, True), (hz.Auc, hz.Aub, hz.ub, False)):
        for row in range(len(rhs)):
            total = Fraction(0)
            for matrix, point in ((continuous, x), (binary, z)):
                a, b = matrix.indptr[row:row + 2]
                total += sum((Fraction(float(v)) * point[int(c)] for c, v in
                              zip(matrix.indices[a:b], matrix.data[a:b])), Fraction(0))
            if (total != Fraction(float(rhs[row]))) if equality else (total > Fraction(float(rhs[row]))):
                return False
    return True


def test_exact_two_way_sets_and_input_reconstruction_on_toy_grid():
    old, kw = fixture()
    reduced = quotient(old, enabled=True, **kw)
    retained = [0, 1, 2, 5]
    # A deterministic finite mathematical fixture check, not benchmark sampling.
    checked, feasible_count = 0, 0
    grid = list(map(Fraction, [-1., -.5, 0., .5, 1.]))
    output_grid = list(map(Fraction, [-.5, -.25, 0., .25, .5]))
    for values in product(grid, grid, grid, output_grid):
        x = [Fraction(0)] * old.n_cont
        for c, v in zip(retained, values):
            x[c] = v
        extended = reduced.reconstruct_fraction(x)
        assert extended[:2] == x[:2]
        for z in ((-1,), (1,)):
            accepted = feasible(reduced.hz, x, z)
            assert feasible(old, extended, z) == accepted
            assert feasible(reduced.hz, extended, z) == accepted
            checked += 1
            feasible_count += accepted
    assert checked == 1250 and feasible_count > 0
    # Every original feasible point satisfies the defining equations, hence its
    # selected coordinates are exactly this unique extension, not just a subset.
    for value in map(Fraction, [-1., -.5, 0., .5, 1.]):
        x = [value, Fraction(0), value / 2, value / 8, -3 * value / 4, value / 4]
        for z in ((-1,), (1,)):
            if feasible(old, x, z):
                assert reduced.reconstruct_fraction(x) == x
                assert feasible(reduced.hz, x, z)


@pytest.mark.parametrize('mutation', ['ratio', 'parent', 'column', 'row', 'hz', 'report', 'extra', 'hidden'])
def test_mutated_or_hidden_certificate_rejects(mutation):
    old, kw = fixture()
    reduced = quotient(old, enabled=True, **kw)
    if mutation in ('ratio', 'parent', 'column', 'row'):
        key = {'ratio': 'ratios', 'parent': 'parents', 'column': 'columns', 'row': 'defining_rows'}[mutation]
        getattr(reduced, key)[0] += 1
    elif mutation == 'hz':
        reduced.hz.Ac.data[0] += .25
    elif mutation == 'report':
        reduced.report['formal_gain'] = 1
    elif mutation == 'extra':
        reduced.original = old
    else:
        reduced.report['hidden'] = np.zeros(4)
    with pytest.raises(ValueError):
        reduced.numeric_roots()


@pytest.mark.parametrize('change', ['map', 'outside_prefix', 'work', 'increased', 'entries', 'nonfinite'])
def test_fail_closed_preconditions(change):
    old, kw = fixture()
    if change == 'map':
        kw['eq_roots'][1] = 0
    elif change == 'outside_prefix':
        kw['old_n_cont'] = old.n_cont + 1
    elif change == 'work':
        kw['max_work'] = 0
    elif change == 'increased':
        kw['max_work'] = 256_000_001
    elif change == 'entries':
        kw['max_entries'] = 0
    else:
        old.b[0] = np.nan
    with pytest.raises((ValueError, MemoryError)):
        quotient(old, enabled=True, **kw)


def test_independent_audit_rejects_resealed_incorrect_transformation():
    old, kw = fixture()
    reduced = quotient(old, enabled=True, **kw)
    reduced.hz.Ac.data[-1] *= 2.
    reduced.seal = reduced.fingerprint()
    with pytest.raises(ValueError, match='substitution mismatch'):
        audit(old, reduced)


def test_no_hit_guard_and_input_outside_box():
    old, kw = fixture()
    reduced = quotient(old, enabled=True, **kw)
    with pytest.raises(ValueError, match='box'):
        reduced.reconstruct_fraction([2.] * old.n_cont)
    with pytest.raises(ValueError, match='width'):
        reduced.reconstruct_fraction([0.])
    nohit, nkw = simple(2., .5)
    with pytest.raises(ValueError, match='no eligible'):
        quotient(nohit, enabled=True, **nkw)
