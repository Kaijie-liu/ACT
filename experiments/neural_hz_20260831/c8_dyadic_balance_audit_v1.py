"""Independent complete definition audit and same-comparator subset witness."""

from collections import Counter
from dataclasses import asdict
import hashlib

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import equal_payload
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_reference_lower_bound_v1 import reachable_operators
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import operator_digest


def row(matrix, index):
    start, stop = matrix.indptr[index:index + 2]
    return matrix.indices[start:stop], matrix.data[start:stop]


def audit(lifted):
    """Verify every stored definition against ORIGINAL unfused row payloads."""
    lifted.validate()
    expr, hz, nodes = lifted.expression, lifted.hz, lifted.nodes
    expanded = []
    for n in nodes:
        if n['kind'] == 'source':
            paths = [(id(n['source']), ())]
        elif n['kind'] == 'op':
            paths = [(source, (*ops, id(n['op']))) for source, ops in expanded[n['parents'][0]]]
        else:
            paths = [path for parent in n['parents'] for path in expanded[parent]]
        if len(paths) > len(expr.terms):
            raise ValueError('factor graph has extra expression terms')
        expanded.append(paths)
    expected = [(id(t.source), tuple(id(op) for op in t.operators)) for t in expr.terms]
    if Counter(expanded[lifted.root]) != Counter(expected):
        raise ValueError('factoring changed source/operator term multiplicity')
    old = zero_transfer_reference(expr)
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        original, actual = getattr(old, name), getattr(hz, name)
        relevant = actual[:original.shape[0]]
        if not equal_payload(relevant[:, :original.shape[1]], original) or relevant[:, original.shape[1]:].nnz:
            raise ValueError('old predicate changed or acquired auxiliary coupling')
    if (hz.n_bin != lifted.old_n_bin or hz.frame_id != old.frame_id
            or hz.n_ineq != old.n_ineq or not equal_payload(hz.b[:old.n_eq], old.b) or not equal_payload(hz.ub, old.ub)):
        raise ValueError('old widths/predicate right hand sides changed')
    count = original_row_entries = 0
    for n in nodes:
        for coordinate in np.flatnonzero(n['needed']):
            slot = int(n['slots'][coordinate])
            eq = lifted.old_n_eq + slot - lifted.old_n_cont
            columns, values = row(hz.Ac, eq)
            bcols, bvals = row(hz.Ab, eq)
            if not columns.size or columns[-1] != slot or values[-1] <= 0. or np.any(columns[:-1] >= slot):
                raise ValueError('fresh factor is not uniquely triangularly defined')
            pivot_mantissa, pivot_exponent = np.frexp(values[-1])
            shift = int(pivot_exponent) - 1
            if pivot_mantissa != .5 or shift < 0:
                raise ValueError('fresh pivot is not a positive permitted power of two')
            magnitudes = np.abs(np.concatenate((values, bvals)))
            nonzero = magnitudes[magnitudes != 0.]
            if nonzero.min() < 2.**-20 or nonzero.max() > 2.**40:
                raise ValueError('new row violates preregistered coefficient window')
            # Verify minimal nonnegative row scaling, independently of emitter.
            raw_min_exponent = int(np.frexp(nonzero.min())[1]) - shift
            if shift != max(0, -20 - (raw_min_exponent - 1)):
                raise ValueError('row scaling is not the registered minimal dyadic shift')
            exponent = int(n['exponents'][coordinate])
            constant, parent_exponents = 0., 0
            expected_bcols, expected_bvals = np.empty(0, dtype=np.int64), np.empty(0)
            if n['kind'] == 'source':
                s = n['source']
                expected_cols, weights = row(s.Gc, coordinate)
                enabled = weights != 0.
                expected_cols, weights = expected_cols[enabled], weights[enabled]
                expected_bcols, expected_bvals = row(s.Gb, coordinate)
                enabled = expected_bvals != 0.
                expected_bcols, expected_bvals = expected_bcols[enabled], expected_bvals[enabled]
                constant = float(s.c[coordinate])
                bound_values = np.concatenate((weights, expected_bvals, [constant] if constant != 0. else []))
                bound_exponents = np.frexp(np.abs(bound_values))[1]
            elif n['kind'] == 'op':
                p = nodes[n['parents'][0]]
                op = n['op']
                # Deliberately not the candidate's support-sliced row emitter.
                coords, weights = op._row(int(coordinate)) if type(op) is ImplicitConv2DOp else row(op, coordinate)
                original_row_entries += int(weights.size)
                if original_row_entries > 256_000_000:
                    raise MemoryError('independent streaming row audit cap exceeded')
                enabled = p['needed'][coords] & (weights != 0.)
                coords, weights = coords[enabled], weights[enabled]
                expected_cols = p['slots'][coords]
                parent_exponents = p['exponents'][coords].astype(np.int64)
                bound_values = weights
                bound_exponents = np.frexp(np.abs(weights))[1] + parent_exponents
            else:
                parents = [(nodes[p]['slots'][coordinate], int(nodes[p]['exponents'][coordinate]), multiplicity)
                    for p, multiplicity in sorted(Counter(n['parents']).items()) if nodes[p]['support'][coordinate]]
                expected_cols = np.array([p[0] for p in parents], dtype=np.int64)
                parent_exponents = np.array([p[1] for p in parents], dtype=np.int64)
                weights = np.array([p[2] for p in parents], dtype=np.float64)
                bound_values = weights
                bound_exponents = np.frexp(weights)[1] + parent_exponents
            if not np.array_equal(columns[:-1], expected_cols) or not np.array_equal(bcols, expected_bcols):
                raise ValueError('defining equation has incorrect original factor coordinates')
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                restored = np.ldexp(-values[:-1], exponent - shift - parent_exponents)
                restored_binary = np.ldexp(-bvals, exponent - shift)
                restored_constant = np.ldexp(hz.b[eq], exponent - shift)
            if not np.array_equal(restored, weights) or not np.array_equal(restored_binary, expected_bvals) or restored_constant != constant:
                raise ValueError('defining coefficients are not exact original dyadics')
            if not bound_values.size or exponent < 0 or exponent > 1023:
                raise ValueError('new continuous factor box is not proved redundant')
            # Independent arbitrary-integer sum, with NO candidate's 26-bit clipping.
            minimum_exponent = int(bound_exponents.min())
            exact_envelope = sum(1 << (int(e) - minimum_exponent) for e in bound_exponents)
            if minimum_exponent + (exact_envelope - 1).bit_length() > exponent:
                raise ValueError('new continuous factor box is not proved redundant')
            count += 1
    if count != hz.n_cont - lifted.old_n_cont or hz.n_eq != old.n_eq + count:
        raise ValueError('missing or extra defining equalities')
    root = nodes[lifted.root]
    selected = lifted.keep & root['support']
    rows = np.flatnonzero(selected)
    reference_gc = sp.csr_matrix((np.ldexp(np.ones(rows.size), root['exponents'][rows]),
        (rows, root['slots'][rows])), shape=hz.Gc.shape)
    if not equal_payload(reference_gc, hz.Gc) or hz.Gb.nnz or not equal_payload(hz.c, expr.bias):
        raise ValueError('root value map or full bias changed')
    lifted.validate()
    return {'all_defining_rows_checked': count, 'original_streamed_row_entries': original_row_entries,
        'all_original_dyadic_coefficients_exact': True, 'all_auxiliary_boxes_proved': True,
        'all_old_predicates_preserved': True, 'all_positive_dyadic_row_scales_verified': True, 'exact_original_term_multiset': True,
        'triangular_unique_extension_and_prefix_projection': True,
        'rounded_composed_matrix_byte_identity_claimed': False}


def reference_subset(roots, net, *, max_nnz=64_000_000):
    if type(max_nnz) is not int or not 0 <= max_nnz <= 64_000_000:
        raise ValueError('invalid reference entry ceiling')
    distinct = {}
    for op in reachable_operators(roots):
        distinct.setdefault(operator_digest(op), op)
    selected = sorted(distinct.values(), key=lambda op: -op.logical_expanded_nnz)[:2]
    if not selected:
        raise ValueError('no reachable reference leaves')
    total = sum(op.logical_expanded_nnz for op in selected)
    if total > max_nnz or sum(16 * op.logical_expanded_nnz + 8 * (op.shape[0] + 1) for op in selected) > 1024**3:
        raise MemoryError('aggregate reference subset exceeds unchanged ceiling')
    graph_layers = {_op(layer).content_key: layer for layer in net.layers if layer.kind == 'CONV2D'}
    witness, records = {}, []
    for index, op in enumerate(selected):
        plain = ImplicitConv2DOp(op._kernel, op.input_shape, stride=op._stride, padding=op._padding,
            dilation=op._dilation, groups=op._groups)
        layer = graph_layers.get(plain.content_key)
        if layer is None:
            raise ValueError('unmatched original reference geometry')
        (matrix, unused_bias), construction = measured_build(lambda: cnn.sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=op._row_mask))
        if matrix.shape != op.shape or matrix.nnz != op.logical_expanded_nnz:
            raise ValueError('reference subset shape or count mismatch')
        for value in (matrix.data, matrix.indices, matrix.indptr):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if owner.base is not None or not owner.flags.owndata:
                raise ValueError('reference subset owner is not a fresh native allocation')
            if any(np.shares_memory(value, other._kernel) for other in distinct.values()):
                raise ValueError('reference subset aliases an original kernel')
        digest = hashlib.sha256(str(matrix.shape).encode())
        for row_id in range(op.shape[0]):
            columns, values = op._row(row_id)
            actual_columns, actual_values = row(matrix, row_id)
            if not np.array_equal(columns, actual_columns) or values.tobytes() != actual_values.tobytes():
                raise ValueError('reference subset coefficient mismatch')
        for value in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(value.tobytes())
        witness[f'reference_leaf_{index}'] = matrix
        records.append({'layer_for_provenance_only': layer.id, 'nnz': matrix.nnz,
            'all_rows_bitwise_checked': True, 'sha256': digest.hexdigest(), 'construction': construction})
    roots.measure(witness)  # Validate compatibility, not candidate accounting.
    lower = snapshot_partial_csr_owners(WholeStateRoots(active=witness, consumer_gc_enabled=False))
    if lower.resident_bytes > 1024**3:
        raise MemoryError('joint reference witness payload exceeds 1 GiB')
    return witness, {'reference_lower_bound': asdict(lower), 'leaves': records,
        'selection': 'two largest distinct-content reachable leaves; first occurrence ties',
        'aggregate_nnz': total, 'same_comparator': 'phase_selective_expanded_v1',
        'complete_reference_materialized': False}

