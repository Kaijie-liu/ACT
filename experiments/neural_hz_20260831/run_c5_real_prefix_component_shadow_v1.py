"""Complete actual-source coefficient shadow, isolated and verdict-free."""

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
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.tf_cnn import sparse_conv2d_matrix_from_layer_csr
from act.back_end.interval_tf.tf_cnn import tf_conv2d
from act.back_end.interval_tf.tf_mlp import tf_bias, tf_scale
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import contract, live_value_rows, source_digest
from experiments.neural_hz_20260831.run_corrected_phase_support_census_v1 import phase_support
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import affine_source_paths, concrete_forward
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import _op
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
SNAPSHOT = EXPERIMENT / "results/c5_corrected_prefix_20260905_v2/layer16.pickle"
OUTPUT = EXPERIMENT / "evidence/c5_real_prefix_component_shadow_20260905_v1.json"


def maxabs(value):
    array = value.data if sp.issparse(value) else np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.


def main():
    if os.path.lexists(OUTPUT):
        raise FileExistsError(OUTPUT)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    seal = json.loads(SNAPSHOT.with_name("layer16.snapshot.json").read_text())
    if _sha256(SNAPSHOT) != seal["pickle_sha256"]:
        raise ValueError("trusted local snapshot hash mismatch")
    with SNAPSHOT.open("rb") as stream:
        saved = pickle.load(stream)
    net, expr = saved["net"], saved["expr_cache"][16]
    paths = affine_source_paths(net.layers, net.preds, 16)
    if len(paths) != len(expr.terms):
        raise ValueError("runtime terms differ from complete graph paths")
    outer_layer = net.by_id[17]
    outer = _op(outer_layer)
    before = tf_conv2d(outer_layer, saved["terminal_bounds"]).bounds
    before = tf_bias(net.by_id[19], tf_scale(net.by_id[18], before).bounds).bounds
    lo, hi = before.lb.numpy().reshape(-1), before.ub.numpy().reshape(-1)
    negative, positive = hi <= 0., (hi > 0.) & (lo >= 0.)
    selected = np.sort(np.concatenate((np.flatnonzero(~(negative | positive)), np.flatnonzero(positive)[:8])))
    keep = np.zeros(outer.shape[0], dtype=bool)
    keep[selected] = True
    output_scale = net.by_id[18].params["a"].numpy().reshape(-1)[selected]
    with torch.inference_mode():
        propagated_bias = F.conv2d(torch.from_numpy(expr.bias).reshape(outer.input_shape),
            outer_layer.params["weight"], outer_layer.params.get("bias"),
            stride=outer._stride, padding=outer._padding, dilation=outer._dilation).reshape(-1).numpy()
        global_bias = (propagated_bias * net.by_id[18].params["a"].numpy().reshape(-1)
                       + net.by_id[19].params["c"].numpy().reshape(-1))[selected]
        input_shape = tuple(net.by_id[2].params["input_shape"])
        concrete = concrete_forward(net.layers, net.preds, torch.from_numpy(saved["hz_cache"][0].c).reshape(input_shape))
    records, actual_sum, reference_sum, center_sum = [], None, None, np.zeros(selected.size)
    total_products = 0
    for index, (path, term) in enumerate(zip(paths, expr.terms, strict=True)):
        source = saved["hz_cache"].get(path.source)
        if source is None or source is not term.source or source.frame_id != expr.frame_id:
            raise ValueError("source/frame runtime identity mismatch")
        if [net.by_id[lid].kind for lid in path.events] != ["CONV2D", "SCALE", "BIAS", "ADD"]:
            raise ValueError("unexpected full path shape")
        if len(term.operators) != 2 or type(term.operators[0]) is not ImplicitConv2DOp or not sp.issparse(term.operators[1]):
            raise ValueError("unexpected live operator tuple")
        inner = term.operators[0]
        if inner.content_key != _op(net.by_id[path.events[0]]).content_key:
            raise ValueError("runtime inner differs from graph payload")
        scale = net.by_id[path.events[1]].params["a"].numpy().reshape(-1)
        difference = term.operators[1] - sp.diags(scale, format="csr")
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError("runtime diagonal differs from graph SCALE")
        live = live_value_rows(source)
        compiled = contract(source, inner, scale, outer, selected,
                            max_products=min(200_000_000, 256_000_000 - total_products))
        total_products += compiled.stats["actual_channel_products"]
        candidate = sp.diags(output_scale, format="csr") @ compiled.matrix
        actual_c, actual_gc, actual_gb = candidate @ source.c, candidate @ source.Gc, candidate @ source.Gb
        tick = time.monotonic()
        # The expanded matrices are an independent test oracle, not candidate storage.
        a, _ = sparse_conv2d_matrix_from_layer_csr(net.by_id[path.events[0]])
        b, _ = sparse_conv2d_matrix_from_layer_csr(outer_layer, keep_rows=keep)
        oracle_buffers = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in (a, b))
        restricted = (b[selected] @ sp.diags(scale) @ a[:, live]).tocsr()
        restricted = sp.diags(output_scale, format="csr") @ restricted
        del a, b
        reference_c = restricted @ source.c[live]
        reference_gc, reference_gb = restricted @ source.Gc[live], restricted @ source.Gb[live]
        errors = {"restricted_operator": maxabs(candidate[:, live] - restricted),
                  "center": maxabs(actual_c - reference_c), "continuous_map": maxabs(actual_gc - reference_gc),
                  "binary_map": maxabs(actual_gb - reference_gb)}
        relative_scale = max(1., maxabs(reference_c), maxabs(reference_gc), maxabs(reference_gb), maxabs(restricted))
        if candidate[:, ~live].nnz:
            raise ValueError("compiled matrix retained dead source columns")
        source_value = concrete[path.source].numpy()
        if np.any(source_value[~live] != 0.):
            raise ValueError("actual source-zero support disagrees with concrete network center")
        center_sum += candidate @ source_value
        result_hz = compiled.apply()
        predicates_equal = all(np.array_equal(getattr(result_hz, name).toarray(), getattr(source, name).toarray())
                               for name in ("Ac", "Ab", "Auc", "Aub"))
        predicates_equal &= all(np.array_equal(getattr(result_hz, name), getattr(source, name)) for name in ("b", "ub"))
        records.append({"term": index, "source_layer": path.source, "events": list(path.events),
                        "source_sha256": source_digest(source), "source_n_cont": source.n_cont, "source_n_bin": source.n_bin,
                        "source_n_eq": source.n_eq, "source_n_ineq": source.n_ineq,
                        "stats": compiled.stats, "oracle_numeric_buffer_bytes": oracle_buffers,
                        "oracle_s": time.monotonic() - tick, "all_coefficient_max_abs_errors": errors,
                        "floating_diagnostic_close": max(errors.values()) <= 1e-10 * relative_scale,
                        "bitwise_identical": all(v == 0. for v in errors.values()),
                        "predicates_and_rhs_equal": bool(predicates_equal),
                        "factor_widths_and_frame_equal": result_hz.n_cont == source.n_cont and result_hz.n_bin == source.n_bin and result_hz.frame_id == source.frame_id})
        print(json.dumps(records[-1]), flush=True)
    center_error = maxabs(center_sum + global_bias - concrete[19].numpy()[selected])
    sources = [Path(__file__), EXPERIMENT / "c5_live_value_contraction_v1.py",
               EXPERIMENT / "test_c5_live_value_contraction_v1.py",
               EXPERIMENT / "C5_REAL_PREFIX_COMPONENT_SHADOW_PREREG_20260905.md",
               EXPERIMENT / "S0_C5_LIVE_VALUE_SUPPORT_PREREG_20260905.md"]
    record = {"schema": "c5_real_prefix_component_shadow_v1", "formal_gain": 0,
              "scope": "complete ADD16 two-source coefficient shadow through Conv17/SCALE18/BIAS19; not runtime integration",
              "snapshot_sha256": seal["pickle_sha256"], "provenance": saved["provenance"],
              "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
              "frontier": phase_support(lo, hi, outer.output_shape[1]), "records": records,
              "actual_products_total": total_products,
              "unrestricted_products_total": sum(r["stats"]["unrestricted_spatial_channel_products"] for r in records),
              "fixed_center_network_max_abs_error": center_error,
              "max_rss_kib_including_oracle": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mathematical_rounding_certificate": False, "whole_state_gate_passed": False,
              "promotion_passed": False, "solver_calls": 0}
    _atomic_exclusive_json(OUTPUT, record)
    print(json.dumps({"output": str(OUTPUT), "sha256": _sha256(OUTPUT), "products": total_products,
                      "network_center_max_abs_error": center_error}), flush=True)


if __name__ == "__main__":
    main()
