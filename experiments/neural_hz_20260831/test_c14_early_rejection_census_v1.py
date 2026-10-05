from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import c14_early_rejection_census_v1 as early
from experiments.neural_hz_20260831.c13_separated_affine_census_v1 import census as eager
from experiments.neural_hz_20260831.test_c11_single_use_census_v1 import fixture, small
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('case', ['eq', 'ineq', 'collision', 'product', 'window', 'rhs',
    'box', 'zero', 'cancel', 'offset', 'new_exact_box'])
def test_every_complete_decision_matches_eager_exact_conjunction(case):
    if case in ('eq', 'ineq'):
        hz, kw = fixture(case == 'ineq')
    else:
        options = {'collision': {}, 'product': dict(parent=.3, consumer=-.3, overlap=0.),
            'window': dict(parent=2.**-20, consumer=-.5, overlap=0.),
            'rhs': dict(parent=.125, constant=.3, consumer_rhs=.7, overlap=0.),
            'box': dict(parent=.75, constant=.5, overlap=0.), 'zero': dict(parent=0., overlap=0.),
            'cancel': dict(parent=.25, overlap=.25), 'offset': dict(parent=.125, constant=.3, consumer=-.3, overlap=0.),
            'new_exact_box': dict(parent=.6, constant=.25, overlap=0.)}
        hz, kw = small(**options[case])
    before = source_digest(hz)
    _, full = eager(hz, **kw)
    report, actual = early.census(hz, **kw)
    assert np.array_equal(actual['individually_admissible'], full['individually_admissible'])
    for key in (*early.GUARDS, 'all_products_exact'):
        evaluated = actual[key] != -1
        assert np.array_equal(actual[key][evaluated].astype(bool), full[key][evaluated]), key
    accepted = actual['individually_admissible']
    assert np.array_equal(actual['individual_nnz_delta'][accepted], full['individual_nnz_delta'][accepted])
    assert report['completed_population'] == len(actual['column'])
    assert sum(report['decision_counts'].values()) == len(actual['column'])
    assert all(np.all(actual[key][accepted] == 1) for key in early.GUARDS)
    assert source_digest(hz) == before


@pytest.mark.parametrize('change,reason,unknown', [
    (dict(parent=.125, constant=.3, consumer=-.3, overlap=0.), 1, 'coefficient_products_exact'),
    (dict(parent=.125, constant=.3, consumer_rhs=.7, overlap=0.), 2, 'redundant_box'),
    (dict(parent=.3, consumer=-.3, overlap=0.), 3, 'redundant_box'),
    (dict(parent=2.**-20, consumer=-.5, overlap=0.), 4, 'coefficient_products_exact'),
    (dict(parent=.75, constant=.5, overlap=0.), 5, 'collision_sums_exact')])
def test_first_failure_never_marks_remaining_guards_passed(change, reason, unknown):
    hz, kw = small(**change)
    _, table = early.census(hz, **kw)
    assert table['first_failure'].tolist() == [reason]
    assert table[unknown].tolist() == [-1]
    assert not table['delta_evaluated'][0] and not table['individually_admissible'][0]


def wide_fixture(definitions=12, width=64, fail_at=0):
    # Independent toy affine definitions with distinct latents and output rows.
    old = width
    nc = old + 2 * definitions
    ac = sp.lil_matrix((2 * definitions, nc))
    gc = sp.lil_matrix((definitions, nc))
    for k in range(definitions):
        col, out = old + 2 * k, old + 2 * k + 1
        values = np.full(width, -.25)
        if fail_at is not None:
            values[fail_at] = -.3
        ac[2 * k, :old] = values
        ac[2 * k, col] = 64.
        ac[2 * k + 1, col] = -19.2  # exact multiplier is float64 .3
        ac[2 * k + 1, out] = 1.
        gc[k, out] = 1.
    hz = SparseHZono(np.zeros(definitions), gc.tocsr(), sp.csr_matrix((definitions, 1)),
        ac.tocsr(), sp.csr_matrix((2 * definitions, 1)), np.zeros(2 * definitions), frame_id=91)
    return hz, dict(old_n_cont=old, logical_n_cont=nc, old_n_eq=0,
        eq_roots=np.arange(2 * definitions, dtype=np.int64), eq_scales=np.zeros(2 * definitions, np.int64),
        def_rows=np.zeros(0, np.int64))


@pytest.mark.parametrize('position,checked', [(0, 8), (7, 8), (8, 16), (63, 64)])
def test_fixed_order_and_whole_tested_blocks_are_charged(position, checked):
    hz, kw = wide_fixture(definitions=1, fail_at=position)
    report, table = early.census(hz, **kw)
    assert table['first_failure'].tolist() == [3]
    assert table['failed_term'].tolist() == [position]
    assert table['products_checked'].tolist() == [checked]
    assert report['work_parts']['coefficient_products'] == 64 * checked
    assert 'exact_coefficient_norm' not in report['work_parts']


def test_full_population_can_complete_below_eager_upper_without_partial_acceptance():
    hz, kw = wide_fixture()
    first, _ = early.census(hz, **kw)
    assert first['logical_work_used'] < first['eager_all_guard_work_upper']
    report, table = early.census(hz, **kw, max_work=first['logical_work_used'])
    assert report['completed_population'] == 12 and report['decision_counts']['coefficient_product_inexact'] == 12
    assert not table['individually_admissible'].any()
    assert not report['partial_population_acceptance']


def test_capacity_exhaustion_discards_complete_prefix_instead_of_returning_it():
    hz, kw = wide_fixture()
    baseline, _ = early.census(hz, **kw)
    seen = []
    with pytest.raises(MemoryError, match='whole census budget exhausted'):
        early.census(hz, **kw, max_work=baseline['logical_work_used'] - 1, observe=seen.append)
    rejected = seen[-1]
    assert rejected['event'] == 'whole_population_budget_rejected'
    assert rejected['processed'] == 11 and rejected['total'] == 12
    assert not rejected['acceptance_published'] and not rejected['partial_table_returned']


def test_capacity_exhaustion_also_discards_a_prefix_containing_admissible_definitions():
    hz, kw = wide_fixture()
    hz.Ac.data[0] = -.25  # The first full definition now passes every guard.
    report, table = early.census(hz, **kw)
    assert table['individually_admissible'][0] and report['individually_admissible'] == 1
    seen = []
    with pytest.raises(MemoryError):
        early.census(hz, **kw, max_work=report['logical_work_used'] - 1, observe=seen.append)
    assert seen[-1]['processed'] == 11 and not seen[-1]['acceptance_published']
    assert not seen[-1]['partial_table_returned']


def test_expensive_guards_cannot_run_after_a_proved_necessary_failure(monkeypatch):
    hz, kw = wide_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('later guard executed after proved failure')
    monkeypatch.setattr(early, 'coefficient_l1_numerator', forbidden)
    monkeypatch.setattr(early, 'overlaps', forbidden)
    report, _ = early.census(hz, **kw)
    assert report['completed_population'] == 12


def test_full_passing_wide_definitions_get_all_checks_and_exact_delta():
    hz, kw = wide_fixture(definitions=2, fail_at=None)
    report, table = early.census(hz, **kw)
    assert report['individually_admissible'] == 2
    assert table['individual_nnz_delta'].tolist() == [-2, -2]
    assert table['products_checked'].tolist() == [64, 64]
    assert report['work_parts']['exact_coefficient_norm'] == 16 * 128
    assert report['work_parts']['accepted_product_workspace'] == 2 * 128
    assert report['work_parts']['consumer_collision_checks'] == 32 * 4


def test_charge_refuses_before_changing_usage_or_calling_arithmetic(monkeypatch):
    pool = early.WorkPool(63)
    with pytest.raises(MemoryError):
        pool.charge('one_product', 64)
    assert pool.used == 0 and pool.parts == {}
    hz, kw = small(parent=.125, constant=.25, overlap=0.)
    report, _ = early.census(hz, **kw)
    def forbidden(*args, **kwargs):
        raise AssertionError('uncharged product')
    monkeypatch.setattr(early, 'exact_products', forbidden)
    structural = report['precharged_structural_work']
    with pytest.raises(MemoryError):
        early.census(hz, **kw, max_work=structural + 32 + 63)


@pytest.mark.parametrize('change', ['map', 'tag', 'shape', 'window', 'nonfinite', 'inexact', 'work', 'entries'])
def test_inherited_structure_and_resource_guards_still_apply(change):
    hz, kw = fixture()
    if change == 'map': kw['eq_roots'][1] = 0
    elif change == 'tag': kw['eq_roots'][0] = -1
    elif change == 'shape': kw['eq_roots'] = kw['eq_roots'].reshape(2, 2)
    elif change == 'window': hz.Ac.data[0] = 2.**-21
    elif change == 'nonfinite': hz.b[0] = np.nan
    elif change == 'inexact': hz.exact = False
    elif change == 'work': kw['max_work'] = 256_000_001
    else: kw['max_entries'] = 64_000_001
    with pytest.raises(ValueError):
        early.census(hz, **kw)
