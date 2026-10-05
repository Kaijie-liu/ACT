"""Qualify first matching affine island on actual live prefix state, then stop."""

from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_live_roots_v1 import collect
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import transaction
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import reference_materialize, compare_hz, equal_payload
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / "evidence/c5_live_transaction_20260905_v1.json"


class LiveQualificationStop(BaseException):
    pass


def matches(expr):
    return bool(expr.terms) and all(len(term.operators) == 4
        and type(term.operators[0]) is ImplicitConv2DOp and type(term.operators[2]) is ImplicitConv2DOp
        and sp.isspmatrix_csr(term.operators[1]) and sp.isspmatrix_csr(term.operators[3])
        for term in expr.terms)


def caller_roots():
    frame = sys._getframe()
    try:
        while frame is not None:
            if frame.f_code.co_filename == prefix.worker.__file__ and frame.f_code.co_name == "main":
                local = frame.f_locals
                return {"model_registered_state": local["model"].state_dict(keep_vars=True),
                        "input_spec": local["input_spec"], "output_spec": local["output_spec"],
                        "labeled": local["labeled"], "input_bounds": local["bounds"]}
            frame = frame.f_back
    finally:
        del frame
    raise ValueError("original worker roots unavailable")


def expanded_roots(roots, net):
    operators = {}
    for value in roots.numeric.values():
        if type(value) is ImplicitConv2DOp:
            operators[id(value)] = value
        elif type(value) in (cnn.SparseHZAffineExpr, cnn.SparseHZAffineTerm):
            terms = value.terms if type(value) is cnn.SparseHZAffineExpr else (value,)
            for term in terms:
                for op in term.operators:
                    if type(op) is ImplicitConv2DOp:
                        operators[id(op)] = op
    total = sum(op.logical_expanded_nnz for op in operators.values())
    if total > 64_000_000:
        raise MemoryError("live expanded preflight cap exceeded")
    graph = {_op(layer).content_key: layer for layer in net.layers if layer.kind == "CONV2D"}
    replacements, records = {}, []
    for identity, op in operators.items():
        plain = ImplicitConv2DOp(op._kernel, op.input_shape, stride=op._stride, padding=op._padding,
                                dilation=op._dilation, groups=op._groups)
        layer = graph.get(plain.content_key)
        if layer is None:
            raise ValueError("unmatched live Conv payload")
        matrix, _ = cnn.sparse_conv2d_matrix_from_layer_csr(layer, keep_rows=op._row_mask)
        if matrix.shape != op.shape or matrix.nnz != op.logical_expanded_nnz:
            raise ValueError("live expanded geometry mismatch")
        for row in range(op.shape[0]):
            cols, data = op._row(row)
            start, stop = matrix.indptr[row:row + 2]
            if not np.array_equal(cols, matrix.indices[start:stop]) or not np.array_equal(data, matrix.data[start:stop]):
                raise ValueError("live expanded scalar mismatch")
        replacements[identity] = matrix
        records.append({"layer": layer.id, "expanded_entries": matrix.nnz})

    def term_copy(term):
        return cnn.SparseHZAffineTerm(term.source, tuple(replacements.get(id(op), op) for op in term.operators))

    mapped, object_cache = {}, {}
    for key, value in roots.numeric.items():
        if id(value) not in object_cache:
            if type(value) is cnn.SparseHZAffineExpr:
                replacement = cnn.SparseHZAffineExpr(tuple(term_copy(term) for term in value.terms),
                                                     value.bias, value.n_out, value.frame_id)
            elif type(value) is cnn.SparseHZAffineTerm:
                replacement = term_copy(value)
            else:
                replacement = replacements.get(id(value), value)
            object_cache[id(value)] = replacement
        mapped[key] = object_cache[id(value)]
    return mapped, records


def qualify(expr, bounds, tf, relu, producer, directory):
    started = time.monotonic()
    record = {"schema": "c5_live_transaction_v1", "formal_gain": 0, "relu_layer": relu.id,
              "stages": [], "relu_executed": False, "solver_calls": 0, "cache_publication_executed": False,
              "promotion_passed": False, "provenance": prefix.worker._provenance(ROOT)}
    try:
        extra = {**caller_roots(), **producer, "current_expr": expr, "relu_input_bounds": bounds}
        _atomic_exclusive_json(directory / "live_schema_census.json", {
            "tf_fields": {name: type(value).__name__ for name, value in vars(tf).items()},
            "extra_fields": {name: type(value).__name__ for name, value in extra.items()}, "formal_gain": 0})
        roots = collect(tf, extra)
        # Read the actual owner surface before any candidate/oracle allocation.
        entry = roots.measure()
        record.update(registered_numeric_roots=len(roots.numeric), schema_counts=roots.schema_counts,
                      python_shallow_bytes=roots.python_shallow_bytes, live_entry=asdict(entry),
                      source_sha256=[source_digest(term.source) for term in expr.terms])
        print(json.dumps({"event": "live_roots", "roots": len(roots.numeric), "bytes": entry.resident_bytes}), flush=True)
        negative, positive, unstable = cnn._phase_selective_masks(bounds, expr.n_out)
        selected = {"probe": np.flatnonzero(unstable | (positive & (np.cumsum(positive) <= 8))),
                    "followup": np.flatnonzero(~negative)}
        budget = BudgetedMaterializer(len(expr.terms))
        outputs, operators = {}, {}
        for name, rows in selected.items():
            operators[name] = {}

            def observe(index, bound):
                operators[name][index] = bound.matrix

            (output, stats), guard = transaction(tf, extra,
                lambda: budget.run(expr, rows, observe=observe), measure=True)
            outputs[name] = output
            record["stages"].append({"name": name, "rows": rows.size, "term_stats": stats, "guard": guard})
            print(json.dumps({"event": "live_candidate", "stage": name, "rows": rows.size, "guard": guard}), flush=True)
            if name == "probe":
                p = np.flatnonzero(positive)[:8]
                nnz = output.Gc[p].nnz + output.Gb[p].nnz
                threshold = expr.n_out + int(positive.sum())
                record["probe_admission"] = {"nnz": nnz, "threshold": threshold, "admitted": nnz > threshold}
                if nnz > threshold:
                    raise ValueError("native full follow-up not requested")
        record["sequence_products"] = 256_000_000 - budget.remaining_whole
        for stage in record["stages"]:
            name, checks = stage["name"], {}

            def observe_oracle(index, matrix):
                checks[index] = equal_payload(operators[name][index], matrix[selected[name]])

            oracle = reference_materialize(expr, selected[name], observe=observe_oracle)
            fields = compare_hz(outputs[name], oracle)
            fields["all_operators"] = len(checks) == len(expr.terms) and all(checks.values())
            stage["bitwise_checks"] = fields
            if not all(fields.values()):
                raise ValueError("live numerical comparison failed")
            print(json.dumps({"event": "live_oracle", "stage": name, "bitwise": True}), flush=True)
        expanded, op_records = expanded_roots(roots, tf._net)
        record["expanded_operators"] = op_records
        record["boundaries"] = {}
        for name, retained in (("entry", {}), ("probe", {"result": outputs["probe"]}),
                               ("followup", {"result": outputs["followup"]}), ("both_retained", outputs)):
            candidate = roots.measure(retained)
            baseline = snapshot_partial_csr_owners(WholeStateRoots(active={**expanded, **retained}, consumer_gc_enabled=False))
            passed = candidate.resident_bytes < baseline.resident_bytes and candidate.resident_entries < baseline.resident_entries
            record["boundaries"][name] = {"candidate": asdict(candidate), "baseline": asdict(baseline), "passed": passed}
            if not passed:
                raise ValueError("live registered physical state not strictly reduced")
        if collect(tf, extra).fingerprint != roots.fingerprint:
            raise ValueError("incoming live roots changed during qualification")
        record["live_registered_gates_passed"] = True
        record["input_roots_unchanged"] = True
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    finally:
        record.update(wall_s=time.monotonic() - started, worker_sha256=_sha256(Path(__file__)),
                      max_rss_kib_including_test_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(OUTPUT, record)
        print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT), "failure": record.get("failure"),
                          "live_registered_gates_passed": record.get("live_registered_gates_passed", False)}), flush=True)
    raise LiveQualificationStop()


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != EXPERIMENT / "results" or OUTPUT.exists():
        raise ValueError("unsafe/occupied live diagnostic output")
    original_phase, original_deferred = cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu
    context = {}

    def deferred(layer, expr, result, tf):
        previous = dict(context)
        context.update(producer_fact=result, incoming_expr=expr)
        try:
            return original_deferred(layer, expr, result, tf)
        finally:
            context.clear()
            context.update(previous)

    def phase(expr, bounds, tf, relu):
        if not matches(expr):
            return original_phase(expr, bounds, tf, relu)
        return qualify(expr, bounds, tf, relu, dict(context), directory)

    cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = phase, deferred
    try:
        prefix.main()
    except LiveQualificationStop:
        print(json.dumps({"status": "qualification_stopped_before_relu", "formal_gain": 0}), flush=True)
    finally:
        cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = original_phase, original_deferred


if __name__ == "__main__":
    main()
