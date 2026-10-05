"""Read-only exact column-equivalence census; never an HZ simplifier.

Equality requires the complete CSC row-index and float64 payload bytes. A
repeated coefficient column is not a repeated latent: every original input
coordinate, old predicate, and output pivot remains distinct and untouched.
The reported q*d versus q+1+d saving is conditional symbolic accounting only;
no normalized sum is emitted and native representability is not established.
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse._sparsetools import csr_has_canonical_format


def _validate(matrix, *, pool):
    if type(matrix) is not sp.csr_matrix:
        raise ValueError('finite canonical float64 CSR matrix required')
    arrays = matrix.data, matrix.indices, matrix.indptr
    pool.charge('c118_csr_validation_payload', 16 * sum(int(a.size) for a in arrays))
    data, indices, indptr = arrays
    rows, columns = map(int, matrix.shape)
    if (data.dtype != np.dtype(np.float64)
            or any(a.ndim != 1 for a in arrays)
            or indices.dtype.kind != 'i' or indptr.dtype.kind != 'i'
            or indices.dtype.itemsize not in (4, 8) or indptr.dtype.itemsize not in (4, 8)
            or len(indices) != len(data) or len(indptr) != rows + 1
            or indptr[0] != 0 or indptr[-1] != len(data)
            or np.any(indptr[1:] < indptr[:-1])
            or np.any(indices < 0) or np.any(indices >= columns)):
        raise ValueError('well-formed finite canonical float64 CSR required')
    # The public has_canonical_format / has_sorted_indices properties lazily
    # change SciPy cache flags. Their array predicate is read-only instead.
    if (not csr_has_canonical_format(rows, indptr, indices)
            or not np.isfinite(data).all()):
        raise ValueError('finite canonical float64 CSR required')
    if np.any(data == 0):
        raise ValueError('explicit stored zeros are not allowed; no silent canonicalization')
    return rows, columns, int(data.size)


def _column_key(csc, column):
    start, stop = csc.indptr[column:column + 2]
    # Each matrix uses one fixed native float64 dtype and one fixed index dtype.
    # Dictionary hashing is an accelerator only: tuple equality compares every
    # byte, including the complete row support, before a class can be reused.
    return csc.indices[start:stop].tobytes(), csc.data[start:stop].tobytes()


def census(matrix, *, pool, enabled=False):
    """Return a JSON-safe complete report and ordinary numeric-array evidence.

    Default-off does not inspect any input. Work is prepaid at the unchanged
    16-per-numeric-element fee, plus 256 per complete column observation. CSC
    is materialized exactly once in a new allocation. Input arrays and SciPy
    cache flags are never changed. Empty columns remain in the population.
    """
    if not enabled:
        return None
    rows, columns, nnz = _validate(matrix, pool=pool)
    pool.charge('c118_csc_conversion', 16 * (2 * nnz + columns + 1))
    csc = matrix.tocsc(copy=True)
    pool.charge('c118_complete_column_keys', 16 * (2 * nnz + columns) + 256 * columns)
    keys, members, degrees = {}, [], []
    class_ids = np.empty(columns, dtype=np.int64)
    for column in range(columns):
        key = _column_key(csc, column)
        class_id = keys.get(key)
        if class_id is None:
            class_id = len(members)
            keys[key] = class_id
            members.append([])
            degrees.append(int(csc.indptr[column + 1] - csc.indptr[column]))
        members[class_id].append(column)
        class_ids[column] = class_id
    pool.charge('c118_complete_group_aggregation', 16 * columns)
    duplicate_reports, sizes, offsets, flat_members = [], [], [0], []
    strict_saving = strict_old = strict_new = strict_groups = duplicate_groups = 0
    covered_nnz = empty_columns = 0
    group_histogram = Counter()
    for class_id, (group, degree) in enumerate(zip(members, degrees)):
        size = len(group)
        old_nnz, new_nnz = size * degree, size + 1 + degree
        saving = old_nnz - new_nnz
        duplicate = size > 1
        strict = duplicate and saving > 0
        duplicate_groups += int(duplicate)
        if strict:
            strict_groups += 1
            strict_saving += saving
            strict_old += old_nnz
            strict_new += new_nnz
        if duplicate:
            duplicate_reports.append({'class_id': class_id, 'size': size, 'degree': degree,
                'duplicate_columns': True, 'old_coefficient_nnz': old_nnz,
                'conditional_new_coefficient_nnz': new_nnz,
                'conditional_nnz_saving': saving, 'strict_nnz_positive': strict,
                'normalization_shift_if_all_parent_exponents_zero': (size - 1).bit_length()})
        covered_nnz += old_nnz
        empty_columns += size if degree == 0 else 0
        sizes.append(size)
        group_histogram[f'{size},{degree}'] += 1
        flat_members.extend(group)
        offsets.append(len(flat_members))
    if covered_nnz != nnz:
        raise ValueError('complete column population does not cover all stored coefficients')
    report = {'schema': 'c118_exact_column_orbits_v1', 'shape': [rows, columns],
        'old_coefficient_nnz': nnz, 'total_columns': columns,
        'class_count': len(sizes), 'duplicate_column_groups': duplicate_reports,
        'all_group_sizes_degrees_and_membership_in_evidence': True,
        'group_size_degree_histogram': dict(group_histogram),
        'duplicate_column_group_count': duplicate_groups,
        'empty_column_count': empty_columns,
        'strict_nnz_positive_group_count': strict_groups,
        'conditional_positive_old_coefficient_nnz': strict_old,
        'conditional_positive_new_coefficient_nnz': strict_new,
        'candidate_conditional_nnz_saving': strict_saving,
        'conditional_total_coefficient_nnz': nnz - strict_saving,
        'full_column_population_inspected': True,
        'full_row_index_and_float64_bytes_compared': True,
        'hash_only_equivalence': False, 'approximate_or_proportional_matching': False,
        'explicit_zeros_rejected_without_input_canonicalization': True,
        'column_equivalence_is_not_latent_identity': True,
        'original_input_coordinates_retained': True, 'old_output_pivots_retained': True,
        'normalized_sum_native_representability_proved': False,
        'conditional_formula': 'q*d old versus q+1+d new; only q>1 and strict decrease counted',
        'conditional_formula_scope': 'coefficient rows only; all original variables retained',
        'diagnostic_only': True, 'candidate_elimination': False,
        'source_or_live_admission': False, 'formal_gain': 0}
    evidence = {'class_ids': class_ids,
        'group_offsets': np.asarray(offsets, dtype=np.int64),
        'group_members': np.asarray(flat_members, dtype=np.int64),
        'group_sizes': np.asarray(sizes, dtype=np.int64),
        'group_degrees': np.asarray(degrees, dtype=np.int64),
        'csc_data': csc.data, 'csc_indices': csc.indices, 'csc_indptr': csc.indptr}
    return report, evidence
