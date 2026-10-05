"""One complete native probe/follow-up shadow with cumulative accounting."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_explicit_schema_ledger_v2 import parameter_roots, snapshot_known_buffers
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import reference_materialize, compare_hz, equal_payload
from experiments.neural_hz_20260831.run_c5_native_boundary_ledger_v1 import capture
from experiments.neural_hz_20260831.run_c5_ordered_full_hz_shadow_v3 import SNAPSHOT, SNAPSHOT_SHA
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / "evidence/c5_full_native_transaction_20260905_v1.json"


def representation_ledgers(saved, boundary, masks, hz_results):
    if saved["precomputed_relu"]:
        raise ValueError("nonempty precomputed cache requires explicit transformation")
    active = parameter_roots(saved["net"])
    active.update(masks)
    expressions = {**saved["expr_cache"], "current": boundary["expr"]}
    operators = {id(op): op for expr in expressions.values() for term in expr.terms
                 for op in term.operators if type(op) is ImplicitConv2DOp}
    total_entries = sum(op.logical_expanded_nnz for op in operators.values())
    if total_entries > 64_000_000:
        raise MemoryError("expanded operator preflight cap exceeded")
    graph_ops = {_op(layer).content_key: layer for layer in saved["net"].layers if layer.kind == "CONV2D"}
    expanded, records = {}, []
    for identity, op in operators.items():
        plain = ImplicitConv2DOp(op._kernel, op.input_shape, stride=op._stride,
                                padding=op._padding, dilation=op._dilation, groups=op._groups)
        layer = graph_ops.get(plain.content_key)
        if layer is None:
            raise ValueError("unmatched live Conv")
        matrix, _ = cnn.sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=op._row_mask)
        if matrix.shape != op.shape or matrix.nnz != op.logical_expanded_nnz:
            raise ValueError("expanded geometry/count mismatch")
        for row in range(op.shape[0]):
            columns, values = op._row(row)
            start, stop = matrix.indptr[row:row + 2]
            if not np.array_equal(columns, matrix.indices[start:stop]) or not np.array_equal(values, matrix.data[start:stop]):
                raise ValueError("expanded scalar row mismatch")
        expanded[identity] = matrix
        records.append({"layer": layer.id, "logical_expanded_nnz": matrix.nnz,
                        "masked_rows": 0 if op._row_mask is None else int((~op._row_mask).sum())})
    baseline = {key: cnn.SparseHZAffineExpr(tuple(cnn.SparseHZAffineTerm(term.source,
        tuple(expanded.get(id(op), op) for op in term.operators)) for term in expr.terms),
        expr.bias, expr.n_out, expr.frame_id) for key, expr in expressions.items()}
    bounds = {**saved["bounds"], "terminal": saved["terminal_bounds"], "prospective": boundary["bounds"]}
    results = {}
    stages = [("entry", {}), ("completed_probe", {"probe": hz_results["probe"]}),
              ("completed_followup", {"followup": hz_results["followup"]}),
              ("conservative_both_retained", hz_results)]
    for name, retained in stages:
        pair = {}
        for arm, exprs in (("candidate_implicit", expressions), ("phase_selective_expanded_v1", baseline)):
            roots = WholeStateRoots(sparse_hz={key: value for key, value in saved["hz_cache"].items() if value is not None},
                affine_expr=exprs, phase_bounds=bounds, precomputed_relu=saved["precomputed_relu"],
                active={**active, **retained}, consumer_gc_enabled=False)
            pair[arm] = asdict(snapshot_known_buffers(roots))
        cand, base = pair["candidate_implicit"], pair["phase_selective_expanded_v1"]
        pair["numeric_bytes_strictly_reduced"] = cand["resident_bytes"] < base["resident_bytes"]
        pair["numeric_entries_strictly_reduced"] = cand["resident_entries"] < base["resident_entries"]
        results[name] = pair
        print(json.dumps({"event": "ledger_boundary", "boundary": name,
                          "candidate_bytes": cand["resident_bytes"], "baseline_bytes": base["resident_bytes"],
                          "bytes_gate": pair["numeric_bytes_strictly_reduced"],
                          "entries_gate": pair["numeric_entries_strictly_reduced"]}), flush=True)
    return {"boundaries": results, "unique_conv_operators": records, "expanded_entries_total": total_entries,
            "network_numeric_parameter_roots": len(active) - len(masks), "gc_applied": False,
            "registered_loaded_snapshot_only": True, "original_runtime_ownership_proved": False,
            "construction_transient_gate_proved": False, "python_heap_completeness_proved": False}


def main():
    if os.path.lexists(OUTPUT):
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    start = time.monotonic()
    record = {"schema": "c5_full_native_transaction_v1", "formal_gain": 0, "stages": [],
              "complete_runtime_prefix": False, "relu_transform_executed": False, "solver_calls": 0,
              "whole_runtime_gate_passed": False, "promotion_passed": False}
    try:
        if _sha256(SNAPSHOT) != SNAPSHOT_SHA:
            raise ValueError("snapshot drift")
        with SNAPSHOT.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["provenance"] != worker._provenance(ROOT):
            raise ValueError("production drift")
        boundary = capture(saved)
        expr, bounds = boundary["expr"], boundary["bounds"]
        hashes = [source_digest(term.source) for term in expr.terms]
        negative, positive, unstable = cnn._phase_selective_masks(bounds, expr.n_out)
        probe_rows = np.flatnonzero(unstable | (positive & (np.cumsum(positive) <= 8)))
        followup_rows = np.flatnonzero(~negative)
        record.update(provenance=saved["provenance"], snapshot_sha256=SNAPSHOT_SHA,
                      source_sha256=hashes, probe_rows=probe_rows.size, followup_rows=followup_rows.size)
        budget = BudgetedMaterializer(len(expr.terms))
        results = {}
        for name, rows in (("probe", probe_rows), ("followup", followup_rows)):
            matrices = {}

            def observe(index, bound):
                matrices[index] = bound.matrix
                print(json.dumps({"event": "candidate_branch", "stage": name, "branch": index, "stats": bound.stats}), flush=True)

            tick = time.monotonic()
            actual, stats = budget.run(expr, rows, observe=observe)
            candidate_s = time.monotonic() - tick
            operator_checks = {}

            def oracle_observe(index, matrix):
                operator_checks[index] = equal_payload(matrices[index], matrix[rows])
                print(json.dumps({"event": "oracle_branch", "stage": name, "branch": index,
                                  "bitwise": operator_checks[index]}), flush=True)

            tick = time.monotonic()
            reference = reference_materialize(expr, rows, observe=oracle_observe)
            checks = compare_hz(actual, reference)
            checks["all_operators"] = len(operator_checks) == len(expr.terms) and all(operator_checks.values())
            checks["sources_unchanged"] = hashes == [source_digest(term.source) for term in expr.terms]
            entry = {"name": name, "rows": rows.size, "candidate_s": candidate_s,
                     "oracle_s": time.monotonic() - tick, "term_stats": stats, "bitwise_checks": checks,
                     "bitwise_passed": all(checks.values()), "remaining_whole_products": budget.remaining_whole,
                     "remaining_branch_products": list(budget.remaining_branches),
                     "hz_dimensions": {key: getattr(actual, key) for key in ("n_out", "n_cont", "n_bin", "n_eq", "n_ineq", "frame_id")}}
            record["stages"].append(entry)
            if not entry["bitwise_passed"]:
                raise ValueError("full native stage bitwise comparison failed")
            results[name] = actual
            if name == "probe":
                positive_rows = np.flatnonzero(positive)[:8]
                nnz = actual.Gc[positive_rows].nnz + actual.Gb[positive_rows].nnz
                threshold = expr.n_out + int(positive.sum())
                record["probe_admission"] = {"generator_nnz": nnz, "threshold": threshold, "admitted": nnz > threshold}
                if nnz > threshold:
                    raise ValueError("registered native follow-up is not requested")
            # Test-only coefficient/oracle buffers are not part of registered runtime roots.
            del matrices, reference
        record["complete_sequence_products"] = 256_000_000 - budget.remaining_whole
        record["branch_sequence_products"] = [200_000_000 - value for value in budget.remaining_branches]
        record["both_native_stages_passed"] = True
        masks = {"negative": negative, "positive": positive, "unstable": unstable,
                 "probe_rows": probe_rows, "followup_rows": followup_rows}
        try:
            record["ledger"] = representation_ledgers(saved, boundary, masks, results)
        except Exception as exc:
            record["ledger_failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    finally:
        record.update(wall_s=time.monotonic() - start, worker_sha256=_sha256(Path(__file__)),
                      max_rss_kib_including_test_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(OUTPUT, record)
        print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT),
                          "failure": record.get("failure"), "ledger_failure": record.get("ledger_failure")}), flush=True)
    if "failure" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
