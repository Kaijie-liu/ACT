from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import c13_separated_affine_census_v1 as split
from experiments.neural_hz_20260831.c11_single_use_census_v1 import census as generic, redundant_box
from experiments.neural_hz_20260831.test_c11_single_use_census_v1 import fixture, small
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('values', [[], [.3, -.7], [2.**-20, 2.**40],
    [np.nextafter(2.**-20, 1.), np.nextafter(2.**40, 0.)],
    [(-1.)**i * (i + .25) * 2.**-10 for i in range(1, 129)]])
def test_exact_grid_l1_matches_independent_fraction_sum(values):
    actual = split.coefficient_l1_numerator(values)
    assert F(actual, 1 << 72) == sum((abs(F(float(v))) for v in values), F(0))


@pytest.mark.parametrize('value', [0., np.nan, np.inf, 2.**-21, 2.**41])
def test_norm_grid_guards_are_not_relaxed(value):
    with pytest.raises(ValueError):
        split.coefficient_l1_numerator([value])


@pytest.mark.parametrize('values,rhs,pivot', [([.6], .25, 1.), ([.3, -.125], .125, 1.),
    ([.75], .5, 1.), ([], 2.**-100, 2.**-20), ([2.**40], 0., 2.**40)])
def test_exact_box_is_fraction_necessary_and_sufficient_for_full_independent_box(values, rhs, pivot):
    norm = split.coefficient_l1_numerator(values)
    assert split.exact_box(norm, rhs, pivot) == (sum((abs(F(v)) for v in values), F(0)) + abs(F(rhs)) <= F(pivot))
    if redundant_box(values, [], rhs, pivot):
        assert split.exact_box(norm, rhs, pivot)


@pytest.mark.parametrize('case', ['eq', 'ineq', 'collision', 'product', 'window', 'rhs', 'box', 'zero', 'cancel'])
def test_other_guards_match_generic_and_exact_box_never_loses_an_old_pass(case):
    if case in ('eq', 'ineq'):
        hz, kw = fixture(case == 'ineq')
    else:
        options = {'collision': {}, 'product': dict(parent=.3, consumer=-.3, overlap=0.),
            'window': dict(parent=2.**-20, consumer=-.5, overlap=0.),
            'rhs': dict(parent=.125, constant=.3, consumer_rhs=.7, overlap=0.),
            'box': dict(parent=.75, constant=.5, overlap=0.), 'zero': dict(parent=0., overlap=0.),
            'cancel': dict(parent=.25, overlap=.25)}
        hz, kw = small(**options[case])
    before = source_digest(hz)
    _, old = generic(hz, **kw)
    _, actual = split.census(hz, **kw)
    for key in old:
        if key in ('redundant_box', 'individually_admissible'):
            assert np.all(actual[key][old[key]])
        else:
            assert np.array_equal(actual[key], old[key]), key
    assert source_digest(hz) == before


def test_new_box_acceptance_is_exact_not_a_tolerance_change():
    hz, kw = small(parent=.6, constant=.25, overlap=0.)
    _, old = generic(hz, **kw)
    _, actual = split.census(hz, **kw)
    assert not old['redundant_box'][0] and actual['individually_admissible'][0]
    assert F(.6) + F(.25) < 1


def test_shared_coefficients_do_not_share_pivot_rhs_or_box_decisions():
    hz = SparseHZono(np.zeros(2), sp.csr_matrix([[0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.]]), sp.csr_matrix((2, 1)),
        sp.csr_matrix([[-.25, -.125, 1., 0., 0., 0.], [0., 0., -.5, 1., 0., 0.],
                       [-.25, -.125, 0., 0., 2., 0.], [0., 0., 0., 0., -1., 1.]]),
        sp.csr_matrix((4, 1)), np.array([.125, .25, 1.75, .75]), frame_id=34)
    kw = dict(old_n_cont=2, logical_n_cont=6, old_n_eq=0, eq_roots=np.arange(4, dtype=np.int64),
        eq_scales=np.zeros(4, np.int64), def_rows=np.zeros(0, np.int64))
    before = source_digest(hz)
    report, table = split.census(hz, **kw)
    assert report['coefficient_certificate_groups'] == 1 and report['unique_coefficient_terms'] == 2
    assert report['all_coefficient_terms'] == 4 and report['nonzero_definition_offsets'] == 2
    assert table['certificate_group'].tolist() == [0, 0]
    assert table['column'].tolist() == [2, 4] and table['consumer_row'].tolist() == [1, 3]
    assert table['redundant_box'].tolist() == [True, False]
    assert table['coefficient_products_exact'].tolist() == [True, True]
    assert table['individually_admissible'].tolist() == [True, False]
    assert source_digest(hz) == before


def test_offset_product_is_checked_separately_for_every_nonzero_constant():
    hz, kw = small(parent=.125, constant=.3, consumer=-.3, overlap=0.)
    _, table = split.census(hz, **kw)
    assert table['coefficient_products_exact'][0] and not table['offset_product_exact'][0]
    assert not table['all_products_exact'][0] and not table['individually_admissible'][0]


def test_partition_ratio_and_actual_bytes_guard_certificate_sharing(monkeypatch):
    monkeypatch.setattr(split, 'payload_digest', lambda _: b'forced')
    groups = split.CertificateGroups()
    assert groups.intern(np.array([.25]), np.array([.125]), .5) == 0
    assert groups.intern(np.array([.25]), np.array([.125]), .25) == 1
    assert groups.intern(np.array([.25, .125]), np.zeros(0), .5) == 2
    assert groups.intern(np.array([.25]), np.array([.125]), .5) == 0
    assert groups.intern(np.array([.125]), np.array([.125]), .5) == 3
    groups.prove()
    assert groups.extra_collision_work > 0
    assert F(groups.groups[0]['norm_numerator'], 1 << 72) == F(3, 8)


def test_all_costs_are_charged_before_arithmetic_and_group_allocation():
    hz, kw = fixture()
    report, _ = split.census(hz, **kw)
    seen = []
    with pytest.raises(MemoryError, match='complete individual arithmetic'):
        split.census(hz, **kw, max_work=report['logical_work_upper'] - 1, observe=seen.append)
    assert len(seen) == 1 and not seen[0]['arithmetic_cap_fits']
    assert report['unique_product_and_exact_norm_work'] == 80 * report['unique_coefficient_terms']
    groups = split.CertificateGroups(max_work=23)
    with pytest.raises(MemoryError):
        groups.intern(np.array([.25]), np.zeros(0), .5)
    assert not groups.groups and groups.signature_work == 0


@pytest.mark.parametrize('change', ['map', 'tag', 'window', 'inexact', 'work', 'entries'])
def test_fixed_structure_domain_and_resource_guards(change):
    hz, kw = fixture()
    if change == 'map': kw['eq_roots'][1] = 0
    elif change == 'tag': kw['eq_roots'][0] = -1
    elif change == 'window': hz.Ac.data[0] = 2.**-21
    elif change == 'inexact': hz.exact = False
    elif change == 'work': kw['max_work'] = 256_000_001
    else: kw['max_entries'] = 64_000_001
    with pytest.raises(ValueError):
        split.census(hz, **kw)
