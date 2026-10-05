"""Capture the native expression boundary and audit its explicit live roots."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.interval_tf.tf_cnn import tf_conv2d
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831.c5_ordered_union_contraction_v3 import materialize
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import reference_materialize, compare_hz
from experiments.neural_hz_20260831.run_c5_ordered_full_hz_shadow_v3 import SNAPSHOT, SNAPSHOT_SHA
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots, snapshot_whole_state
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / "evidence/c5_native_boundary_ledger_20260905_v1.json"


class Captured(Exception):
    pass


def capture(saved):
    net = saved["net"]
    tf = SimpleNamespace(_net=net, _sparse_precomputed_relu=dict(saved["precomputed_relu"]),
                         _SPARSE_MAX_AFFINE_CELLS=64_000_000, _neural_hz_sparse_implicit_conv_dag=True)
    captured = {}

    def observe(expr, bounds, tf_arg, relu):
        captured.update(expr=expr, bounds=bounds, relu=relu)
        raise Captured()

    original = cnn._try_phase_selective_exact_relu
    cnn._try_phase_selective_exact_relu = observe
    try:
        result = tf_conv2d(net.by_id[17], saved["terminal_bounds"])
        cnn._try_deferred_expr_conv_relu(net.by_id[17], saved["expr_cache"][16], result, tf)
    except Captured:
        pass
    finally:
        cnn._try_phase_selective_exact_relu = original
    if not captured:
        raise ValueError("native boundary not reached")
    return captured


def parameter_roots(net):
    roots = {}

    def visit(value, path):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            roots[path] = value
        elif isinstance(value, dict):
            for name, item in value.items():
                visit(item, f"{path}.{name}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        elif value is not None and not isinstance(value, (int, float, bool, str, np.generic, torch.dtype, torch.device)):
            raise ValueError(f"unknown parameter payload: {path}: {type(value).__name__}")

    for layer in net.layers:
        visit(layer.params, f"network.{layer.id}")
    return roots


def ledgers(saved, boundary, selected, probe):
    if saved["precomputed_relu"]:
        raise ValueError("nonempty precomputed cache requires an explicit comparator transformation")
    expressions = dict(saved["expr_cache"])
    expressions["active_current"] = boundary["expr"]
    operators = {id(op): op for expr in expressions.values() for term in expr.terms
                 for op in term.operators if type(op) is ImplicitConv2DOp}
    total = sum(op.logical_expanded_nnz for op in operators.values())
    if total > 64_000_000:
        raise MemoryError("complete expanded operator preflight cap exceeded")
    layers = {}
    for layer in saved["net"].layers:
        if layer.kind == "CONV2D":
            layers[_op(layer).content_key] = layer
    expanded, operator_records = {}, []
    for key, op in operators.items():
        plain = ImplicitConv2DOp(op._kernel, op.input_shape, stride=op._stride,
                                padding=op._padding, dilation=op._dilation, groups=op._groups)
        layer = layers.get(plain.content_key)
        if layer is None:
            raise ValueError("reachable Conv has no matching graph operator")
        matrix, _ = cnn.sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=op._row_mask)
        if matrix.shape != op.shape or matrix.nnz != op.logical_expanded_nnz:
            raise ValueError("expanded operator geometry/count mismatch")
        # Validate every emitted coefficient and source index against the row oracle.
        for row in range(op.shape[0]):
            cols, vals = op._row(row)
            start, stop = matrix.indptr[row:row + 2]
            if not np.array_equal(cols, matrix.indices[start:stop]) or not np.array_equal(vals, matrix.data[start:stop]):
                raise ValueError("expanded comparator differs from exact row oracle")
        expanded[key] = matrix
        operator_records.append({"layer": layer.id, "logical_expanded_nnz": matrix.nnz,
                                 "masked_rows": 0 if op._row_mask is None else int(np.count_nonzero(~op._row_mask))})
    expanded_expr = {
        key: cnn.SparseHZAffineExpr(tuple(cnn.SparseHZAffineTerm(term.source,
            tuple(expanded.get(id(op), op) for op in term.operators)) for term in expr.terms),
            expr.bias, expr.n_out, expr.frame_id)
        for key, expr in expressions.items()
    }
    bounds = dict(saved["bounds"])
    bounds.update(terminal=saved["terminal_bounds"], prospective=boundary["bounds"])
    active = parameter_roots(saved["net"])
    active["selected_rows"] = selected
    results = {}
    for name, extra in (("entry", {}), ("completed_probe", {"probe": probe})):
        pair = {}
        for arm, exprs in (("candidate_implicit", expressions), ("phase_selective_expanded_v1", expanded_expr)):
            roots = WholeStateRoots(sparse_hz={k: v for k, v in saved["hz_cache"].items() if v is not None},
                affine_expr=exprs, precomputed_relu=saved["precomputed_relu"], phase_bounds=bounds,
                active={**active, **extra}, consumer_gc_enabled=False)
            pair[arm] = asdict(snapshot_whole_state(roots))
        candidate, baseline = pair["candidate_implicit"], pair["phase_selective_expanded_v1"]
        pair["numeric_bytes_strictly_reduced"] = candidate["resident_bytes"] < baseline["resident_bytes"]
        pair["numeric_entries_strictly_reduced"] = candidate["resident_entries"] < baseline["resident_entries"]
        results[name] = pair
    return {"boundaries": results, "complete_unique_conv_operators": operator_records,
            "logical_expanded_nnz_total": total, "network_numeric_parameter_roots": len(active) - 1,
            "gc_applied": False, "runtime_heap_completeness_proved": False,
            "construction_transient_gate_proved": False}


def main():
    if os.path.lexists(OUTPUT):
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    start = time.monotonic()
    if _sha256(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError("snapshot drift")
    with SNAPSHOT.open("rb") as stream:
        saved = pickle.load(stream)
    if saved["provenance"] != worker._provenance(ROOT):
        raise ValueError("production drift")
    boundary = capture(saved)
    expr, bounds = boundary["expr"], boundary["bounds"]
    negative, positive, unstable = cnn._phase_selective_masks(bounds, expr.n_out)
    selected = np.flatnonzero(unstable | (positive & (np.cumsum(positive) <= 8)))
    tick = time.monotonic()
    actual, stats = materialize(expr, selected)
    materialize_s = time.monotonic() - tick
    probe_rows = np.flatnonzero(positive)[:8]
    probe_nnz = actual.Gc[probe_rows].nnz + actual.Gb[probe_rows].nnz
    probe_threshold = expr.n_out + int(positive.sum())
    print(json.dumps({"stage": "native_candidate", "s": materialize_s,
                      "positive_probe_nnz": probe_nnz, "threshold": probe_threshold}), flush=True)
    tick = time.monotonic()
    reference = reference_materialize(expr, selected)
    checks = compare_hz(actual, reference)
    record = {"schema": "c5_native_boundary_ledger_v1", "formal_gain": 0,
        "snapshot_sha256": SNAPSHOT_SHA, "provenance": saved["provenance"],
        "native_masked_expression_captured": True, "relu_layer": boundary["relu"].id,
        "selected_rows": selected.size, "stable_negative": int(negative.sum()), "positive": int(positive.sum()),
        "all_hz_fields_bitwise_checks": checks, "bitwise_passed": all(checks.values()),
        "candidate_materialization_s": materialize_s, "oracle_s": time.monotonic() - tick, "term_stats": stats,
        "positive_probe_nnz": probe_nnz, "positive_probe_threshold": probe_threshold,
        "native_phase_selective_probe_admitted": probe_nnz > probe_threshold,
        "complete_runtime_prefix": False, "promotion_passed": False, "whole_runtime_gate_passed": False,
        "solver_calls": 0}
    print(json.dumps({"stage": "native_oracle", "bitwise_passed": record["bitwise_passed"]}), flush=True)
    if record["bitwise_passed"]:
        try:
            record["ledger"] = ledgers(saved, boundary, selected, actual)
        except Exception as exc:
            record["ledger_failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    record.update(wall_s=time.monotonic() - start,
                  max_rss_kib_including_expanded_oracle=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256=_sha256(Path(__file__)))
    _atomic_exclusive_json(OUTPUT, record)
    print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT),
                      "ledger_failure": record.get("ledger_failure")}), flush=True)


if __name__ == "__main__":
    main()
