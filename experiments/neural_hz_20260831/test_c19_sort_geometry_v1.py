import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c19_sort_geometry_v1 import census
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_cost, merge_cost


def fixture():
    a = sp.lil_matrix((2, 64))
    a[0, [3, 50]] = [-.5, 1.]
    a[1, :] = np.ones(64)
    return a.tocsr(), sp.csr_matrix((0, 64)), 32, np.array([50]), np.array([3]), np.array([0])


def test_complete_census_without_sorting_or_source_mutation(monkeypatch):
    args = fixture()
    original = [a.copy() for a in (args[0].data, args[0].indices, args[0].indptr)]
    def forbidden(*args, **kwargs): raise AssertionError('diagnostic executed row sort')
    monkeypatch.setattr(np, 'argsort', forbidden)
    report = census(*args)
    assert report['counts']['rows'] == 1 and report['counts']['occurrences'] == 1
    assert report['modes'] == {'merge_ordered': 1}
    assert report['same_rows_C17_sort_work'] == sort_cost(64)
    assert report['same_rows_C18_normalization_work'] == 8 + merge_cost(64, 1)
    for a, b in zip((args[0].data, args[0].indices, args[0].indptr), original):
        assert a.tobytes() == b.tobytes()


def test_already_ordered_small_row():
    a = sp.csr_matrix([[-.5, 1., 0.], [0., -.5, 1.]])
    result = census(a, sp.csr_matrix((0, 3)), 1, [1], [0], [0])
    assert result['modes'] == {'already_ordered': 1}
    assert result['same_rows_C17_sort_work'] == result['same_rows_C18_normalization_work'] == 0


@pytest.mark.parametrize('budget', [0, 256_000_001])
def test_census_caps_before_mapping_allocation(budget, monkeypatch):
    args = fixture()
    def forbidden(*args, **kwargs): raise AssertionError('mapping allocated before cap')
    monkeypatch.setattr(np, 'arange', forbidden)
    with pytest.raises((MemoryError, ValueError)): census(*args, max_work=budget)


@pytest.mark.parametrize('corrupt', ['parent', 'definition', 'protected'])
def test_invalid_lineage(corrupt):
    args = list(fixture())
    if corrupt == 'parent': args[4] = np.array([50])
    if corrupt == 'definition': args[5] = np.array([2])
    if corrupt == 'protected': args[2] = 51
    with pytest.raises(ValueError): census(*args)
