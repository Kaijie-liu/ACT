import json

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831 import c118_column_orbits_v1 as module


def run(matrix):
    pool = WorkPool(256_000_000)
    report, evidence = module.census(matrix, pool=pool, enabled=True)
    assert json.loads(json.dumps(report)) == report
    assert all(type(a) is np.ndarray and a.dtype.kind in 'fi' for a in evidence.values())
    assert report['formal_gain'] == 0 and not report['source_or_live_admission']
    assert not report['candidate_elimination']
    assert int(evidence['group_sizes'].sum()) == matrix.shape[1]
    assert int((evidence['group_sizes'] * evidence['group_degrees']).sum()) == matrix.nnz
    assert len(evidence['group_sizes']) == report['class_count']
    return report, evidence, pool


def test_complete_exact_cloned_columns():
    matrix = sp.csr_matrix([[1., 1., 7.], [2., 2., 0.], [3., 3., 0.], [4., 4., 0.]])
    report, evidence, _ = run(matrix)
    assert evidence['class_ids'].tolist() == [0, 0, 1]
    assert evidence['group_offsets'].tolist() == [0, 2, 3]
    assert evidence['group_members'].tolist() == [0, 1, 2]
    assert report['duplicate_column_group_count'] == 1
    assert report['candidate_conditional_nnz_saving'] == 1


def test_equal_coefficients_do_not_identify_or_remove_distinct_latents():
    report, evidence, _ = run(sp.csr_matrix(np.ones((4, 2))))
    assert evidence['group_members'].tolist() == [0, 1]
    assert report['column_equivalence_is_not_latent_identity']
    assert report['original_input_coordinates_retained']
    assert report['old_output_pivots_retained']
    assert not report['normalized_sum_native_representability_proved']


def test_same_values_at_different_row_positions_are_distinct():
    report, evidence, _ = run(sp.csr_matrix([[1., 0.], [0., 1.], [2., 0.], [0., 2.]]))
    assert evidence['class_ids'].tolist() == [0, 1]
    assert report['duplicate_column_group_count'] == 0


def test_column_input_order_changes_members_not_equivalence():
    dense = np.array([[1., 8., 1., 8.], [2., 0., 2., 0.]])
    _, first, _ = run(sp.csr_matrix(dense))
    _, second, _ = run(sp.csr_matrix(dense[:, [3, 0, 1, 2]]))
    assert first['class_ids'].tolist() == [0, 1, 0, 1]
    assert second['class_ids'].tolist() == [0, 1, 0, 1]
    assert second['group_members'].tolist() == [0, 2, 1, 3]
    assert second['csc_data'].tolist() == [8., 1., 2., 8., 1., 2.]


def test_sparse_and_empty_columns_are_all_inspected():
    report, evidence, _ = run(sp.csr_matrix([[0., 1., 0., 0.], [0., 0., 0., 2.]]))
    assert evidence['class_ids'].tolist() == [0, 1, 0, 2]
    assert report['empty_column_count'] == 2
    assert report['total_columns'] == 4 and report['class_count'] == 3
    assert report['duplicate_column_group_count'] == 1
    assert report['strict_nnz_positive_group_count'] == 0


@pytest.mark.parametrize('shape', [(0, 3), (3, 0), (3, 3)])
def test_empty_matrix_populations(shape):
    report, evidence, _ = run(sp.csr_matrix(shape, dtype=np.float64))
    assert report['old_coefficient_nnz'] == 0
    assert report['total_columns'] == shape[1]
    assert report['empty_column_count'] == shape[1]
    assert evidence['group_members'].tolist() == list(range(shape[1]))
    assert report['candidate_conditional_nnz_saving'] == 0


def test_near_float_difference_is_not_an_exact_match():
    dense = np.ones((4, 2))
    dense[2, 1] = np.nextafter(1., 2.)
    report, evidence, _ = run(sp.csr_matrix(dense))
    assert evidence['class_ids'].tolist() == [0, 1]
    assert report['duplicate_column_group_count'] == 0


def test_proportional_columns_are_not_matches():
    report, evidence, _ = run(sp.csr_matrix([[1., 2.], [2., 4.], [3., 6.], [4., 8.]]))
    assert evidence['class_ids'].tolist() == [0, 1]
    assert report['candidate_conditional_nnz_saving'] == 0


def test_default_off_does_not_inspect_arguments():
    assert module.census(object(), pool=object()) is None


def test_validation_budget_precedes_numeric_observation(monkeypatch):
    matrix = sp.eye(3, format='csr', dtype=np.float64)
    def forbidden(*args):
        raise AssertionError('canonical bytes read before validation prepayment')
    monkeypatch.setattr(module, 'csr_has_canonical_format', forbidden)
    with pytest.raises(MemoryError, match='csr_validation_payload'):
        module.census(matrix, pool=WorkPool(0), enabled=True)


def test_conversion_budget_precedes_new_csc(monkeypatch):
    matrix = sp.eye(3, format='csr', dtype=np.float64)
    validation = 16 * (matrix.data.size + matrix.indices.size + matrix.indptr.size)
    def forbidden(*args, **kwargs):
        raise AssertionError('CSC built before conversion prepayment')
    monkeypatch.setattr(sp.csr_matrix, 'tocsc', forbidden)
    with pytest.raises(MemoryError, match='csc_conversion'):
        module.census(matrix, pool=WorkPool(int(validation)), enabled=True)


def test_column_key_budget_precedes_payload_copy(monkeypatch):
    matrix = sp.eye(3, format='csr', dtype=np.float64)
    paid = 16 * (2 * matrix.nnz + matrix.shape[0] + 1)
    paid += 16 * (2 * matrix.nnz + matrix.shape[1] + 1)
    def forbidden(*args):
        raise AssertionError('column bytes copied before complete column prepayment')
    monkeypatch.setattr(module, '_column_key', forbidden)
    with pytest.raises(MemoryError, match='complete_column_keys'):
        module.census(matrix, pool=WorkPool(int(paid)), enabled=True)


def test_input_bytes_flags_and_array_identities_are_unchanged():
    matrix = sp.csr_matrix([[1., 1., 0.], [2., 2., 3.]])
    vars(matrix).pop('_has_canonical_format', None)
    vars(matrix).pop('_has_sorted_indices', None)
    attributes = vars(matrix).copy()
    arrays = matrix.data, matrix.indices, matrix.indptr
    payloads = [a.tobytes() for a in arrays]
    _, evidence, _ = run(matrix)
    assert set(vars(matrix)) == set(attributes)
    assert all(vars(matrix)[key] is value for key, value in attributes.items())
    assert [a.tobytes() for a in arrays] == payloads
    assert all(not np.shares_memory(a, evidence[name]) for a, name in
               zip(arrays, ('csc_data', 'csc_indices', 'csc_indptr')))


@pytest.mark.parametrize('size,degree,saving', [(2, 4, 1), (3, 3, 2), (2, 3, 0), (2, 1, -2)])
def test_exact_q_times_d_formula_and_strict_guard(size, degree, saving):
    report, _, _ = run(sp.csr_matrix(np.ones((degree, size))))
    group, = report['duplicate_column_groups']
    assert group['old_coefficient_nnz'] == size * degree
    assert group['conditional_new_coefficient_nnz'] == size + 1 + degree
    assert group['conditional_nnz_saving'] == saving
    assert group['strict_nnz_positive'] == (saving > 0)
    assert report['candidate_conditional_nnz_saving'] == max(saving, 0)
    assert group['normalization_shift_if_all_parent_exponents_zero'] == (size - 1).bit_length()


def test_explicit_zero_is_rejected_without_silent_cleanup():
    matrix = sp.csr_matrix((np.array([1., 0.]), np.array([0, 1]), np.array([0, 2])), shape=(1, 2))
    before = matrix.data.tobytes(), matrix.indices.tobytes(), matrix.indptr.tobytes()
    with pytest.raises(ValueError, match='explicit stored zeros'):
        module.census(matrix, pool=WorkPool(256_000_000), enabled=True)
    assert before == (matrix.data.tobytes(), matrix.indices.tobytes(), matrix.indptr.tobytes())


def test_noncanonical_csr_is_rejected_without_cache_changes():
    matrix = sp.csr_matrix((np.array([1., 2.]), np.array([1, 0]), np.array([0, 2])), shape=(1, 2))
    attributes = vars(matrix).copy()
    with pytest.raises(ValueError, match='canonical'):
        module.census(matrix, pool=WorkPool(256_000_000), enabled=True)
    assert set(vars(matrix)) == set(attributes)
    assert all(vars(matrix)[key] is value for key, value in attributes.items())


def test_frozen_fee_formula_covers_complete_population():
    matrix = sp.csr_matrix([[1., 1., 0.], [2., 2., 3.]])
    _, evidence, pool = run(matrix)
    assert pool.parts == {
        'c118_csr_validation_payload': 16 * (2 * matrix.nnz + matrix.shape[0] + 1),
        'c118_csc_conversion': 16 * (2 * matrix.nnz + matrix.shape[1] + 1),
        'c118_complete_column_keys': 16 * (2 * matrix.nnz + matrix.shape[1]) + 256 * matrix.shape[1],
        'c118_complete_group_aggregation': 16 * matrix.shape[1]}
    rebuilt = sp.csc_matrix((evidence['csc_data'], evidence['csc_indices'], evidence['csc_indptr']),
                            shape=matrix.shape)
    assert (rebuilt != matrix).nnz == 0
