"""Complete nested-ADD shape and support-independent cost screen, not authority."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import affine_source_paths, load_registered_graph
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
PREREG = EXPERIMENT / "S0_C4_NESTED_ADDITIVE_CORE_PREREG_20260905.md"
LIMITS = {"coefficient_entries_per_descriptor": 2_000_000,
          "contraction_products_per_descriptor": 200_000_000,
          "descriptor_bytes": 64 * 1024 * 1024,
          "total_contraction_products": 256_000_000}


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _op(layer):
    p = layer.params
    return ImplicitConv2DOp(_array(p["weight"]), tuple(p["input_shape"]),
                            stride=p.get("stride", 1), padding=p.get("padding", 0),
                            dilation=p.get("dilation", 1), groups=p.get("groups", 1))


def stationary_scale(value, shape):
    array = _array(value)
    if array.size != int(np.prod(shape)) or not np.isfinite(array).all():
        raise ValueError("invalid_scale_payload")
    shaped = array.reshape(shape)
    channels = shaped[0, :, 0, 0]
    if not np.array_equal(shaped, np.broadcast_to(channels[None, :, None, None], shape)):
        raise ValueError("nonstationary_scale")
    return channels


def geometry_cost(inner, outer):
    """Exact descriptor arithmetic counts, not physical-state or emission cost."""
    if inner.output_shape != outer.input_shape:
        raise ValueError("intermediate_shape_mismatch")
    ci, cm, co = inner.input_shape[1], inner.output_shape[1], outer.output_shape[1]
    gi, go = inner._groups, outer._groups
    ki, ko = int(np.prod(inner._kernel.shape[2:])), int(np.prod(outer._kernel.shape[2:]))
    entries = products = intersections = 0
    for outer_group in range(go):
        for inner_group in range(gi):
            lo = max(outer_group * (cm // go), inner_group * (cm // gi))
            hi = min((outer_group + 1) * (cm // go), (inner_group + 1) * (cm // gi))
            if hi <= lo:
                continue
            intersections += 1
            block_entries = ki * ko * (co // go) * (ci // gi)
            entries += block_entries
            products += block_entries * (hi - lo)
    formula = ki * ko * co * (cm // go) * (ci // gi)
    if products != formula:
        raise ValueError("group_intersection_formula_mismatch")
    return {"coefficient_entries": entries, "coefficient_bytes_lower_bound": entries * 8,
            "contraction_products": products, "group_intersections": intersections,
            "input_shape": list(inner.input_shape), "middle_shape": list(inner.output_shape),
            "output_shape": list(outer.output_shape), "inner_groups": gi, "outer_groups": go}


def screen(layers, preds, terminal):
    by_id = {layer.id: layer for layer in layers}
    paths = affine_source_paths(layers, preds, terminal, max_paths=4096)
    occurrences, operators, common_outer = [], {}, None
    for path in paths:
        events = path.events
        convs = [lid for lid in events if by_id[lid].kind == "CONV2D"]
        if len(convs) != 2:
            raise ValueError("each_complete_path_requires_two_convs")
        inner_id, outer_id = convs
        if common_outer is not None and common_outer != outer_id:
            raise ValueError("different_outer_conv_occurrence")
        common_outer = outer_id
        inner_pos, outer_pos = events.index(inner_id), events.index(outer_id)
        adds = [lid for lid in events if by_id[lid].kind == "ADD"]
        if not adds or any(not inner_pos < events.index(lid) < outer_pos for lid in adds):
            raise ValueError("add_outside_complete_inner_outer_core")
        for lid in convs:
            if lid not in operators:
                operators[lid] = _op(by_id[lid])
        inner, outer = operators[inner_id], operators[outer_id]
        scale_ids, bias_ids = [], []
        for position, lid in enumerate(events):
            layer = by_id[lid]
            if layer.kind == "SCALE":
                if position < inner_pos:
                    raise ValueError("scale_before_inner_conv")
                shape = inner.output_shape if position < outer_pos else outer.output_shape
                stationary_scale(layer.params["a"], shape)
                scale_ids.append(lid)
            if layer.kind == "BIAS":
                value = _array(layer.params["c"])
                if value.size != len(layer.out_vars) or not np.isfinite(value).all():
                    raise ValueError("invalid_explicit_bias")
                bias_ids.append(lid)
        occurrences.append({"source": path.source, "events": list(events),
                            "inner": inner_id, "outer": outer_id, "adds": adds,
                            "scale_events": scale_ids, "explicit_bias_events": bias_ids,
                            "geometry_cost": geometry_cost(inner, outer)})
    if not occurrences:
        raise ValueError("empty_core")
    violations = []
    for index, row in enumerate(occurrences):
        cost = row["geometry_cost"]
        for field, limit in (("coefficient_entries", LIMITS["coefficient_entries_per_descriptor"]),
                             ("contraction_products", LIMITS["contraction_products_per_descriptor"]),
                             ("coefficient_bytes_lower_bound", LIMITS["descriptor_bytes"])):
            if cost[field] > limit:
                violations.append({"occurrence": index, "field": field, "value": cost[field], "limit": limit})
    total = sum(r["geometry_cost"]["contraction_products"] for r in occurrences)
    if total > LIMITS["total_contraction_products"]:
        violations.append({"field": "total_contraction_products", "value": total,
                           "limit": LIMITS["total_contraction_products"]})
    return {"complete_path_count": len(paths), "common_outer": common_outer,
            "source_multiplicities": dict(Counter(p.source for p in paths)),
            "occurrences": occurrences, "structural_pass": True,
            "necessary_resource_caps_pass": not violations, "violations": violations,
            "total_contraction_products_no_cache_discount": total,
            "coefficient_bytes_lower_bound_no_cache_discount": sum(r["geometry_cost"]["coefficient_bytes_lower_bound"] for r in occurrences),
            "all_reachable_storage_accounted": False, "selected_support_emission_accounted": False,
            "runtime_lineage_proved": False, "positive_authorization": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != EXPERIMENT / "evidence" or output.name != "s0_c4_nested_add_preflight_20260905_v1.json":
        raise ValueError("unexpected exclusive evidence path")
    if os.path.lexists(output):
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.monotonic()
    anchor, model, net, clone, center = load_registered_graph()
    del model, center
    preds = clone.predecessor_dict()
    if len(preds[36]) != 1:
        raise ValueError("registered terminal arity changed")
    try:
        result = screen(net.layers, preds, preds[36][0])
    except ValueError as exc:
        result = {"structural_pass": False, "reason": str(exc), "positive_authorization": False}
    sources = [Path(__file__), PREREG, EXPERIMENT / "test_s0_c4_nested_add_preflight_v1.py",
               EXPERIMENT / "run_s0_c3_graph_preflight_v1.py",
               EXPERIMENT / "bn_graph_faithfulness_certificate_prototype.py",
               ROOT / "act/back_end/hybridz_tf/exact_linear_op.py"]
    record = {"schema": "s0_c4_nested_add_necessary_preflight_v1", "formal_gain": 0,
              "formal_baseline": "1870/2413", "e0": "61/400", "limits": LIMITS,
              "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "target": anchor["target"], "source_graph_sha256": anchor["source_certificate"]["graph_sha256"],
              "corrected_graph_sha256": anchor["candidate_certificate"]["graph_sha256"],
              "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
              "measured_target_relu": 36, "screen": result, "elapsed_s": time.monotonic() - started,
              "solver_calls": 0, "abstract_propagation": False, "production_option_enabled": False}
    _atomic_exclusive_json(output, record)
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "screen": result}))


if __name__ == "__main__":
    main()
