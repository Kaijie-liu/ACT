from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono

from experiments.neural_hz_20260831 import c12_shared_affine_census_v1 as shared
from experiments.neural_hz_20260831.c11_single_use_census_v1 import census as generic, overlaps as generic_overlaps
from experiments.neural_hz_20260831.test_c11_single_use_census_v1 import fixture, small
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


@pytest.mark.parametrize('case', ['eq', 'ineq', 'inexact', 'product', 'window', 'rhs', 'box', 'zero', 'cancel'])
def test_all_individual_results_equal_frozen_generic_census(case):
    if case in ('eq', 'ineq'):
        hz, kw = fixture(case == 'ineq')
    else:
        changes = {'inexact': {}, 'product': dict(parent=.3, consumer=-.3, overlap=0.),
            'window': dict(parent=2.**-20, consumer=-.5, overlap=0.),
            'rhs': dict(parent=.125, constant=.3, consumer_rhs=.7, overlap=0.),
            'box': dict(parent=.75, constant=.5, overlap=0.),
            'zero': dict(parent=0., overlap=0.), 'cancel': dict(parent=.25, overlap=.25)}
        hz, kw = small(**changes[case])
    before = source_digest(hz)
    original_report, original = generic(hz, **kw)
    report, actual = shared.census(hz, **kw)
    for key in original:
        assert np.array_equal(actual[key], original[key]), key
    assert report['individually_admissible'] == original_report['individually_admissible']
    assert source_digest(hz) == before


def intern(group, coefficient=.125, binary=.25, constant=.25, pivot=2., ratio=.5):
    return group.intern(np.array([coefficient]), np.array([binary]), constant, pivot, ratio)


def test_full_byte_identity_shares_proof_without_merging_coordinates():
    groups = shared.CertificateGroups()
    for _ in range(50):
        assert intern(groups) == 0
    assert len(groups.groups) == 1 and groups.unique_terms == 3 and groups.replacement_terms == 150
    assert groups.signature_work == 8 * 150 + 16 * 50
    groups.prove()
    certificate = groups.groups[0]
    assert certificate['box'] and certificate['precise'] and certificate['window']
    assert list(map(lambda x: Fraction(float(x)), certificate['products'])) == [Fraction(1, 16), Fraction(1, 8), Fraction(1, 8)]


@pytest.mark.parametrize('field', ['coefficient', 'binary', 'constant', 'pivot', 'ratio'])
def test_every_scalar_certificate_dependency_is_in_identity(field):
    groups = shared.CertificateGroups()
    assert intern(groups) == 0
    assert intern(groups, **{field: .375}) == 1
    assert len(groups.groups) == 2


def test_forced_hash_collision_requires_actual_complete_byte_equality(monkeypatch):
    monkeypatch.setattr(shared, 'payload_digest', lambda payload: b'collision')
    groups = shared.CertificateGroups()
    assert intern(groups, ratio=.5) == 0
    assert intern(groups, ratio=.25) == 1
    assert intern(groups, ratio=.125) == 2
    assert intern(groups, ratio=.125) == 2
    assert groups.extra_collision_work == 3 * (4 * 3 + 16)
    groups.prove()
    assert [g['products'].tolist() for g in groups.groups] == [
        [.0625, .125, .125], [.03125, .0625, .0625], [.015625, .03125, .03125]]


def test_grouping_cap_charged_before_new_comparison_or_allocation():
    groups = shared.CertificateGroups(max_work=39)
    with pytest.raises(MemoryError):
        intern(groups)
    assert not groups.groups and groups.signature_work == 0


@pytest.mark.parametrize('definition,products,consumer', [
    ([1, 3, 7], [.125, .25, -.5], ([0, 2, 4], [.3, .4, .5])),
    ([1, 3, 7], [.125, .25, -.5], ([1, 3, 9], [.125, -.25, .5])),
    ([1], [.3], ([1], [.7])), ([], [], ([1], [.5]))])
def test_consumer_side_collision_probe_matches_original(definition, products, consumer):
    args = (np.asarray(definition, np.int64), np.asarray(products),
        (np.asarray(consumer[0], np.int64), np.asarray(consumer[1])))
    assert shared.overlaps(*args) == generic_overlaps(*args)


def test_full_census_cap_not_relaxed_by_certificate_sharing():
    hz, kw = fixture()
    seen = []
    report, _ = shared.census(hz, **kw)
    with pytest.raises(MemoryError, match='complete individual arithmetic'):
        shared.census(hz, **kw, max_work=report['logical_work_upper'] - 1, observe=seen.append)
    assert len(seen) == 1 and not seen[0]['arithmetic_cap_fits']
    with pytest.raises(ValueError):
        shared.census(hz, **kw, max_work=256_000_001)


def test_repeated_scalar_proof_does_not_merge_latents_or_rhs_consumers():
    hz = SparseHZono(np.zeros(2), sp.csr_matrix([[0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1.]]), sp.csr_matrix((2, 1)),
        sp.csr_matrix([[-.25, -.125, 1., 0., 0., 0.], [0., 0., -.5, 1., 0., 0.],
                       [-.25, -.125, 0., 0., 1., 0.], [0., 0., 0., 0., -.5, 1.]]),
        sp.csr_matrix((4, 1)), np.array([.125, .25, .125, .75]), frame_id=34)
    kw = dict(old_n_cont=2, logical_n_cont=6, old_n_eq=0, eq_roots=np.arange(4, dtype=np.int64),
        eq_scales=np.zeros(4, np.int64), def_rows=np.zeros(0, np.int64))
    before = source_digest(hz)
    report, table = shared.census(hz, **kw)
    _, original = generic(hz, **kw)
    assert report['coefficient_certificate_groups'] == 1 and report['individually_admissible'] == 2
    assert report['unique_replacement_and_rhs_terms'] == 3
    assert report['replacement_and_rhs_terms'] == 6
    assert table['column'].tolist() == [2, 4] and table['consumer_row'].tolist() == [1, 3]
    assert table['certificate_group'].tolist() == [0, 0]
    for key in original:
        assert np.array_equal(original[key], table[key])
    assert source_digest(hz) == before


def test_continuous_binary_partition_is_part_of_certificate_identity():
    groups = shared.CertificateGroups()
    assert groups.intern(np.array([.25]), np.array([.125]), 0., 1., .5) == 0
    assert groups.intern(np.array([.25, .125]), np.zeros(0), 0., 1., .5) == 1
