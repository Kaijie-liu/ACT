"""Complete-candidate < verified reference-subset implies whole-state reduction.

No approximate byte counts or candidate-root deletion. This constructs one
real CSR leaf of the unchanged reference and explicitly returns a LOWER BOUND.
"""

from dataclasses import asdict
import hashlib

import numpy as np

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op


def reachable_operators(roots):
    found = {}
    for value in roots.numeric.values():
        if type(value) is ImplicitConv2DOp:
            found[id(value)] = value
        elif type(value) in (cnn.SparseHZAffineExpr, cnn.SparseHZAffineTerm):
            terms = value.terms if type(value) is cnn.SparseHZAffineExpr else (value,)
            for term in terms:
                for op in term.operators:
                    if type(op) is ImplicitConv2DOp:
                        found[id(op)] = op
    return tuple(found.values())


def lower_bound_roots(roots, net, *, max_nnz=64_000_000):
    if type(max_nnz) is not int or not 0 <= max_nnz <= 64_000_000:
        raise ValueError("reference witness cannot increase fixed nnz cap")
    ops = reachable_operators(roots)
    if not ops:
        raise ValueError("reference witness has no reachable operator")
    selected = max(ops, key=lambda op: op.logical_expanded_nnz)
    count = selected.logical_expanded_nnz
    if count <= 0 or count > max_nnz:
        raise MemoryError("single reference witness preflight cap exceeded")
    # Native builder uses at most float64/int64 CSR buffers. Actual temporary
    # peak is measured independently; this is an additional early rejection.
    if 16 * count + 8 * (selected.shape[0] + 1) > 1024**3:
        raise MemoryError("single reference witness payload exceeds 1 GiB")
    plain = ImplicitConv2DOp(selected._kernel, selected.input_shape, stride=selected._stride,
        padding=selected._padding, dilation=selected._dilation, groups=selected._groups)
    graph = {_op(layer).content_key: layer for layer in net.layers if layer.kind == "CONV2D"}
    layer = graph.get(plain.content_key)
    if layer is None:
        raise ValueError("unmatched reference witness geometry/payload")
    (matrix, unused_bias), construction = measured_build(
        lambda: cnn.sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=selected._row_mask))
    if matrix.shape != selected.shape or matrix.nnz != count:
        raise ValueError("reference witness geometry/nnz mismatch")
    for buf in (matrix.data, matrix.indices, matrix.indptr):
        owner = buf
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        if owner.base is not None or not owner.flags.owndata:
            raise ValueError("reference witness does not have native allocation owners")
        for op in ops:
            if np.shares_memory(buf, op._kernel) or (op._row_mask is not None and np.shares_memory(buf, op._row_mask)):
                raise ValueError("reference witness aliases input operator storage")
    for row in range(selected.shape[0]):
        columns, values = selected._row(row)
        start, stop = matrix.indptr[row:row + 2]
        if not np.array_equal(columns, matrix.indices[start:stop]) or values.tobytes() != matrix.data[start:stop].tobytes():
            raise ValueError("reference witness scalar coefficient mismatch")
    proof_roots = {"verified_reference_operator_subset": matrix}
    # With W temporarily added, the existing visitor also rejects ambiguous
    # or incompatible original-root/CSR aliases. This extra union is a test,
    # not the candidate state used for the strict inequality.
    roots.measure(proof_roots)
    lower = snapshot_partial_csr_owners(WholeStateRoots(active=proof_roots, consumer_gc_enabled=False))
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode())
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    record = {"proof_kind": "whole_candidate_vs_verified_reference_subset_v1",
        "selected_layer_for_provenance_only": layer.id, "selection": "largest_logical_nnz_first_occurrence",
        "reachable_implicit_objects": len(ops), "all_reference_logical_nnz": sum(op.logical_expanded_nnz for op in ops),
        "constructed_reference_nnz": matrix.nnz, "reference_subset_sha256": digest.hexdigest(),
        "every_row_bitwise_verified": True, "reference_subset": asdict(lower),
        "reference_construction": construction, "complete_reference_materialized": False,
        "full_candidate_required": True, "same_comparator_and_cache_policy": True,
        "metric_proof": "M(complete_candidate) < M(verified_subset + retained_outputs) <= M(complete_reference)"}
    return proof_roots, [record]
