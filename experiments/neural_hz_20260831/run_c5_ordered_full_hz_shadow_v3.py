"""All-fields real HZ shadow; no production dispatch or verdicts."""

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
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import _lazy_append_linear, _lazy_add_const
from act.back_end.interval_tf.tf_cnn import tf_conv2d
from act.back_end.interval_tf.tf_mlp import tf_bias, tf_scale
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_ordered_union_contraction_v3 import materialize
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import reference_materialize, compare_hz, equal_payload
from experiments.neural_hz_20260831.run_corrected_phase_support_census_v1 import phase_support
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import affine_source_paths
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
SNAPSHOT = EXPERIMENT / "results/c5_corrected_prefix_20260905_v2/layer16.pickle"
SNAPSHOT_SHA = "3cb6d229300aaa8755cfcf6794f5671f9b0486fc50d0375e4d106a2b1757af8d"
OUTPUT = EXPERIMENT / "evidence/c5_ordered_full_hz_shadow_20260905_v3.json"


def main():
    if os.path.lexists(OUTPUT):
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    start = time.monotonic()
    seal = json.loads(SNAPSHOT.with_name("layer16.snapshot.json").read_text())
    if seal["pickle_sha256"] != SNAPSHOT_SHA or _sha256(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError("trusted snapshot mismatch")
    with SNAPSHOT.open("rb") as stream:
        saved = pickle.load(stream)
    if saved["provenance"] != worker._provenance(ROOT):
        raise ValueError("production provenance drift")
    net, expr = saved["net"], saved["expr_cache"][16]
    paths = affine_source_paths(net.layers, net.preds, 16)
    if len(paths) != len(expr.terms) or len(paths) != 2:
        raise ValueError("complete graph/runtime term mismatch")
    sources = []
    for path, term in zip(paths, expr.terms, strict=True):
        if saved["hz_cache"].get(path.source) is not term.source or term.source.frame_id != expr.frame_id:
            raise ValueError("source identity/frame mismatch")
        if [net.by_id[lid].kind for lid in path.events] != ["CONV2D", "SCALE", "BIAS", "ADD"]:
            raise ValueError("unexpected complete path")
        if len(term.operators) != 2 or type(term.operators[0]) is not ImplicitConv2DOp:
            raise ValueError("unexpected operator tuple")
        if term.operators[0].content_key != _op(net.by_id[path.events[0]]).content_key:
            raise ValueError("inner operator mismatch")
        diagonal = net.by_id[path.events[1]].params["a"].numpy().reshape(-1)
        if not equal_payload(term.operators[1], sp.diags(diagonal, format="csr")):
            raise ValueError("inner diagonal mismatch")
        sources.append({"source_layer": path.source, "events": list(path.events),
                        "source_sha256": source_digest(term.source)})
    outer_layer = net.by_id[17]
    outer = _op(outer_layer)
    bounds = tf_conv2d(outer_layer, saved["terminal_bounds"]).bounds
    bounds = tf_bias(net.by_id[19], tf_scale(net.by_id[18], bounds).bounds).bounds
    lo, hi = bounds.lb.numpy().reshape(-1), bounds.ub.numpy().reshape(-1)
    negative, positive = hi <= 0., (hi > 0.) & (lo >= 0.)
    selected = np.sort(np.concatenate((np.flatnonzero(~(negative | positive)), np.flatnonzero(positive)[:8])))
    bias = outer_layer.params.get("bias")
    vector = None if bias is None else np.repeat(bias.numpy().reshape(-1), outer.output_shape[2] * outer.output_shape[3])
    # Precisely the native bias path used by the implicit materializer.
    expr = _lazy_append_linear(expr, outer, vector, 64_000_000)
    output_scale = net.by_id[18].params["a"].numpy().reshape(-1)
    expr = _lazy_append_linear(expr, sp.diags(output_scale, format="csr"), None, 64_000_000)
    expr = _lazy_add_const(expr, net.by_id[19].params["c"].numpy().reshape(-1))
    operators = {}

    def candidate_observe(index, bound):
        operators[index] = bound.matrix
        print(json.dumps({"event": "candidate_term", "term": index, "stats": bound.stats}), flush=True)

    tick = time.monotonic()
    actual, stats = materialize(expr, selected, observe=candidate_observe)
    candidate_s = time.monotonic() - tick
    comparisons = {}

    def oracle_observe(index, matrix):
        comparisons[index] = equal_payload(operators[index], matrix[selected])
        print(json.dumps({"event": "oracle_term", "term": index, "all_retained_coefficients_bitwise": comparisons[index]}), flush=True)

    tick = time.monotonic()
    reference = reference_materialize(expr, selected, observe=oracle_observe)
    oracle_s = time.monotonic() - tick
    checks = compare_hz(actual, reference)
    outside = np.ones(expr.n_out, dtype=bool)
    outside[selected] = False
    checks["unselected_bias_preserved"] = equal_payload(actual.c[outside], expr.bias[outside])
    checks["unselected_value_maps_zero"] = actual.Gc[outside].nnz == actual.Gb[outside].nnz == 0
    checks["source_payloads_unchanged"] = all(source_digest(term.source) == rec["source_sha256"]
                                             for term, rec in zip(expr.terms, sources, strict=True))
    checks["complete_operator_comparison"] = len(comparisons) == len(expr.terms) and all(comparisons.values())
    paths = [Path(__file__), EXPERIMENT / "c5_ordered_union_contraction_v3.py",
             EXPERIMENT / "c5_ordered_row_oracle_v3.py", EXPERIMENT / "test_c5_ordered_union_contraction_v3.py",
             EXPERIMENT / "S0_C5_ORDERED_UNION_V3_PREREG_20260905.md"]
    record = {
        "schema": "c5_ordered_full_hz_shadow_v3", "formal_gain": 0,
        "scope": "ALL two ADD16 source terms through Conv17/SCALE18/BIAS19, complete selected HZ materialization",
        "snapshot_sha256": SNAPSHOT_SHA, "provenance": worker._provenance(ROOT),
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in paths},
        "frontier": phase_support(lo, hi, outer.output_shape[1]), "sources": sources,
        "term_stats": stats, "all_hz_fields_bitwise_checks": checks, "bitwise_passed": all(checks.values()),
        "candidate_materialization_s": candidate_s, "oracle_s": oracle_s, "wall_s": time.monotonic() - start,
        "actual_products_total": sum(s["actual_channel_products"] for s in stats),
        "channel_product_upper_bound_total": sum(s["channel_product_upper_bound"] for s in stats),
        "unrestricted_products_total": sum(s["unrestricted_spatial_channel_products"] for s in stats),
        "both_quarter_product_gates_passed": all(s["quarter_product_gate"] for s in stats),
        "joined_hz": {name: getattr(actual, name) for name in ("n_out", "n_cont", "n_bin", "n_eq", "n_ineq", "frame_id", "exact")},
        "max_rss_kib_including_oracle_and_observation": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mathematical_rounding_certificate": False, "whole_state_gate_passed": False,
        "complete_runtime_prefix": False, "promotion_passed": False, "solver_calls": 0,
    }
    _atomic_exclusive_json(OUTPUT, record)
    print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT), "bitwise_passed": record["bitwise_passed"],
                      "quarter_gates": record["both_quarter_product_gates_passed"]}), flush=True)
    if not record["bitwise_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
