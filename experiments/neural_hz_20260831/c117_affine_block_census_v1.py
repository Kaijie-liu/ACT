"""Complete, bounded, read-only affine-program census; never a simplifier.

Content classes use full immutable payload byte tuples, not content hashes.
Program classes additionally preserve original source identity and every sum
multiplicity. Equality is deliberately sufficient, not algebraically complete.
Fresh C24 graph construction/authentication is a separate paid prerequisite.
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse._sparsetools import csr_has_canonical_format

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c24_dense_ownership_v1 import RADIX, validate_words


def _array_key(array):
    return array.dtype.str, tuple(map(int, array.shape)), array.tobytes(order='C')


def _operator(op, *, pool):
    """Read all semantic payload only after its full element charge."""
    if type(op) is sp.csr_matrix:
        arrays = (op.data, op.indices, op.indptr)
        pool.charge('c117_complete_operator_payload', 16 * sum(int(a.size) for a in arrays))
        # The public property lazily writes _has_canonical_format and
        # _has_sorted_indices. Call its same read-only array predicate instead.
        if (op.dtype != np.dtype(np.float64)
                or not csr_has_canonical_format(len(op.indptr) - 1, op.indptr, op.indices)
                or not np.isfinite(op.data).all()):
            raise ValueError('finite canonical float64 CSR required')
        geometry = ('csr', tuple(map(int, op.shape)))
        details = {'type': 'csr', 'shape': list(map(int, op.shape)),
                   'stored_nnz': int(op.nnz), 'nonzero_coefficients': int(np.count_nonzero(op.data))}
    elif type(op) is ImplicitConv2DOp:
        arrays = (op._kernel,) if op._row_mask is None else (op._kernel, op._row_mask)
        pool.charge('c117_complete_operator_payload', 16 * sum(int(a.size) for a in arrays))
        if (op._kernel.dtype != np.dtype(np.float64) or not np.isfinite(op._kernel).all()
                or (op._row_mask is not None and (op._row_mask.dtype != np.dtype(bool)
                    or op._row_mask.shape != (op.shape[0],)))):
            raise ValueError('finite Conv payload and full-width boolean mask required')
        geometry = ('implicit_conv2d', tuple(op.shape), tuple(op.input_shape), tuple(op.output_shape),
                    tuple(op._stride), tuple(op._padding), tuple(op._dilation), int(op._groups),
                    op._row_mask is not None)
        details = {'type': 'implicit_conv2d', 'shape': list(map(int, op.shape)),
                   'input_shape': list(op.input_shape), 'output_shape': list(op.output_shape),
                   'kernel_shape': list(op._kernel.shape), 'stride': list(op._stride),
                   'padding': list(op._padding), 'dilation': list(op._dilation),
                   'groups': int(op._groups), 'row_mask_present': op._row_mask is not None,
                   'enabled_operator_rows': int(op.shape[0]) if op._row_mask is None
                   else int(np.count_nonzero(op._row_mask)),
                   'nonzero_kernel_coefficients': int(np.count_nonzero(op._kernel))}
    else:
        raise ValueError('unsupported affine operator')
    # Python dictionary lookup hashes these tuples, but equality compares ALL
    # geometry, dtype, shape and payload bytes before a class is reused.
    key = geometry, tuple(_array_key(a) for a in arrays)
    return key, details, arrays


def _histogram(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(keys, counts)}


def classify(nodes, counts, owners, *, pool, enabled=False):
    """Return JSON report and numeric evidence; no graph/HZ mutation or admission.

    ``counts`` accepts C24's whole report or its complete ``node_counts`` list.
    Returned mask/owner/payload arrays refer to the read-only input graph. The
    caller must retain/authenticate that graph and include it in resource bills.
    Consumer degrees exclude each factor's own defining equation, as C24 does.
    """
    if not enabled:
        return None
    records = counts['node_counts'] if isinstance(counts, dict) else counts
    if not (len(nodes) == len(records) == len(owners)):
        raise ValueError('complete graph/count/ownership population required')
    node_reports, degrees = [], []
    operator_keys, program_keys = {}, {}
    operator_members, program_members = [], []
    operator_classes, program_classes = [], []
    payloads, needed_masks, support_masks, packed_owners = [], [], [], []
    global_histogram, needed_histogram = Counter(), Counter()
    consumer_total = non_source_edges = 0
    for ni, (node, record, owned) in enumerate(zip(nodes, records, owners)):
        width, parents = node['width'], node['parents']
        if type(width) is not int or width < 0:
            raise ValueError('nonnegative integer node width required')
        pool.charge('c117_node_masks_and_ownership', 256 + 16 * (3 * width + len(parents)))
        if any(type(p) is not int or not 0 <= p < ni for p in parents):
            raise ValueError('complete topological parent identities required')
        needed, support = node['needed'], node['support']
        if (type(needed) is not np.ndarray or needed.dtype != np.dtype(bool)
                or needed.shape != (width,) or type(support) is not np.ndarray
                or support.dtype != np.dtype(bool) or support.shape != (width,)
                or np.any(needed & ~support)):
            raise ValueError('full-width supported needed mask required')
        validate_words(owned)
        if owned.shape != (width,):
            raise ValueError('complete per-node ownership width required')
        degree = owned // RADIX
        if np.any((degree != 0) & ~needed):
            raise ValueError('unneeded coordinate has a consumer')
        auxiliary_count = int(np.count_nonzero(needed))
        if (record['kind'] != node['kind'] or record['width'] != width
                or record['auxiliaries'] != auxiliary_count):
            raise ValueError('complete node count disagrees with masks')
        edge_fields = ('continuous_edges', 'binary_edges', 'center_edges')
        if any(not isinstance(record[k], (int, np.integer)) or record[k] < 0 for k in edge_fields):
            raise ValueError('nonnegative complete edge counts required')
        kind = node['kind']
        op_class, source_identity, geometry = -1, None, None
        if kind == 'source':
            if parents or node['source'].n_out != width:
                raise ValueError('source identity/width mismatch')
            source_identity = id(node['source'])
            program_key = ('source_identity', source_identity, width)
            payloads.append(())
        elif kind == 'op':
            if len(parents) != 1 or node['op'].shape != (width, nodes[parents[0]]['width']):
                raise ValueError('operator/parent geometry mismatch')
            operator_key, geometry, arrays = _operator(node['op'], pool=pool)
            if operator_key not in operator_keys:
                operator_keys[operator_key] = len(operator_members)
                operator_members.append([])
            op_class = operator_keys[operator_key]
            operator_members[op_class].append(ni)
            program_key = ('op', op_class, program_classes[parents[0]], width)
            payloads.append(arrays)
        elif kind == 'sum':
            if not parents or any(nodes[p]['width'] != width for p in parents):
                raise ValueError('sum parent widths mismatch')
            multiplicities = Counter(program_classes[p] for p in parents)
            program_key = ('sum', width, tuple(sorted(multiplicities.items())))
            payloads.append(())
        else:
            raise ValueError('unsupported graph node kind')
        if program_key not in program_keys:
            program_keys[program_key] = len(program_members)
            program_members.append([])
        program_class = program_keys[program_key]
        program_members[program_class].append(ni)
        operator_classes.append(op_class)
        program_classes.append(program_class)
        all_hist, needed_hist = _histogram(degree), _histogram(degree[needed])
        global_histogram.update(all_hist)
        needed_histogram.update(needed_hist)
        consumer_total += int(degree.sum())
        if kind != 'source':
            non_source_edges += int(record['continuous_edges'])
        node_reports.append({'node': ni, 'kind': kind, 'width': width,
            'parents': list(parents), 'parent_program_classes': [program_classes[p] for p in parents],
            'parent_multiplicities': {str(p): n for p, n in sorted(Counter(parents).items())},
            'source_object_identity': source_identity, 'operator_geometry': geometry,
            'operator_content_class': op_class, 'complete_program_class': program_class,
            'support_width': int(np.count_nonzero(support)), 'needed_width': auxiliary_count,
            **{k: int(record[k]) for k in edge_fields},
            'consumer_occurrences': int(degree.sum()), 'full_consumer_degree_histogram': all_hist,
            'needed_consumer_degree_histogram': needed_hist,
            'needed_zero_use': int(needed_hist.get('0', 0)),
            'needed_one_use': int(needed_hist.get('1', 0)),
            'needed_multi_use': auxiliary_count - needed_hist.get('0', 0) - needed_hist.get('1', 0)})
        degrees.append(degree)
        needed_masks.append(needed)
        support_masks.append(support)
        packed_owners.append(owned)
    if consumer_total != non_source_edges:
        raise ValueError('complete consumer occurrence sum differs from non-source continuous edges')
    repeated_programs, overlaps = [], []
    for cid, members in enumerate(program_members):
        if len(members) < 2:
            continue
        width = nodes[members[0]]['width']
        pool.charge('c117_complete_equivalent_needed_overlap', 16 * width * len(members))
        coverage = np.zeros(width, dtype=np.int64)
        for ni in members:
            coverage += needed_masks[ni]
        union = int(np.count_nonzero(coverage))
        repeated_programs.append({'program_class': cid, 'nodes': members,
            'kind': nodes[members[0]]['kind'], 'width': width,
            'needed_row_occurrences': int(coverage.sum()), 'needed_row_union': union,
            'rows_needed_by_multiple_nodes': int(np.count_nonzero(coverage > 1)),
            'duplicate_needed_row_occurrences': int(coverage.sum()) - union,
            'candidate_only': True})
        overlaps.append({'program_class': cid, 'coverage': coverage})
    repeated_operators = [{'operator_content_class': cid, 'nodes': members,
        'complete_program_classes': sorted({program_classes[n] for n in members}),
        'content_equality_does_not_prove_equal_inputs': True}
        for cid, members in enumerate(operator_members) if len(members) > 1]
    report = {'schema': 'c117_complete_affine_program_census_v1', 'complete_node_count': len(nodes),
        'nodes': node_reports, 'operator_content_class_count': len(operator_members),
        'complete_program_class_count': len(program_members),
        'repeated_operator_content': repeated_operators, 'repeated_complete_programs': repeated_programs,
        'all_consumer_occurrences': consumer_total, 'all_non_source_continuous_edges': non_source_edges,
        'complete_consumer_occurrence_equality': True,
        'full_consumer_degree_histogram': dict(global_histogram),
        'needed_consumer_degree_histogram': dict(needed_histogram),
        'needed_one_use': int(needed_histogram.get('1', 0)),
        'needed_multi_use': sum(n for k, n in needed_histogram.items() if int(k) > 1),
        'full_payload_bytes_compared': True, 'hash_only_equivalence': False,
        'source_identity_not_source_value_or_frame': True, 'sum_multiplicities_preserved': True,
        'own_defining_occurrences_excluded': True, 'diagnostic_only': True,
        'candidate_elimination': False, 'source_or_live_admission': False, 'formal_gain': 0}
    evidence = {'operator_classes': np.asarray(operator_classes, dtype=np.int64),
        'program_classes': np.asarray(program_classes, dtype=np.int64),
        'consumer_degrees': degrees, 'packed_consumer_owners': packed_owners,
        'needed_masks': needed_masks, 'support_masks': support_masks,
        'complete_operator_payload_arrays': payloads, 'equivalent_program_needed_coverage': overlaps}
    return report, evidence
