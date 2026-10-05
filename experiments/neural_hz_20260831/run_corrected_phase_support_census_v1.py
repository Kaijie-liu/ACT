"""Source-vs-corrected cheap interval-support diagnostic, never HZ authority."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from act.back_end.core import Bounds
from act.back_end.interval_tf.interval_tf import IntervalTF
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import load_registered_graph, concrete_forward
from experiments.neural_hz_20260831.run_s0_c4_nested_add_preflight_v1 import screen
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent


def phase_support(lb, ub, channels):
    lower, upper = np.asarray(lb).reshape(-1), np.asarray(ub).reshape(-1)
    if lower.shape != upper.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("invalid finite interval")
    if channels <= 0 or len(lower) % channels:
        raise ValueError("invalid output channel geometry")
    negative = upper <= 0.
    positive = ~negative & (lower >= 0.)
    unstable = ~(negative | positive)
    selected = np.flatnonzero(unstable).tolist() + np.flatnonzero(positive)[:8].tolist()
    spatial = len(lower) // channels
    selected_channels = sorted({r // spatial for r in selected})
    return {"negative": int(negative.sum()), "positive": int(positive.sum()),
            "unstable": int(unstable.sum()), "selected_rows": len(selected),
            "selected_channels": selected_channels, "selected_channel_count": len(selected_channels),
            "output_channels": channels, "total_rows": len(lower),
            "selected_rows_sha256": hashlib.sha256(np.asarray(sorted(selected), dtype="<i8").tobytes()).hexdigest(),
            "bounds_sha256": hashlib.sha256(lower.astype("<f8").tobytes() + upper.astype("<f8").tobytes()).hexdigest()}


def census(net, bounds, concrete):
    transfer, after, relus, centers = IntervalTF(), {}, [], []
    channels = None
    for layer in net.layers:
        preds = net.preds.get(layer.id, [])
        incoming = bounds if not preds else after[preds[0]].bounds
        if layer.kind == "CONV2D":
            channels = int(layer.params["weight"].shape[0])
        if layer.kind == "RELU":
            support = phase_support(incoming.lb.numpy(), incoming.ub.numpy(), channels)
            relus.append({"layer": layer.id, **support})
        fact = transfer.apply(layer, incoming, net, {}, after)
        after[layer.id] = fact
        value = concrete[layer.id].reshape(-1)
        lb, ub = fact.bounds.lb.reshape(-1), fact.bounds.ub.reshape(-1)
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all() or torch.any(lb > ub):
            raise ValueError(f"invalid_interval_at_{layer.id}")
        contained = bool(torch.all(value >= lb) and torch.all(value <= ub))
        centers.append({"layer": layer.id, "center_contained": contained})
        if layer.id == 36:
            break
    return {"relus": relus, "fixed_center_checks": centers,
            "all_fixed_centers_contained": all(r["center_contained"] for r in centers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != EXPERIMENT / "evidence" or output.name != "corrected_phase_support_census_20260905_v1.json":
        raise ValueError("unexpected exclusive output")
    if os.path.lexists(output):
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.monotonic()
    anchor, model, net, clone, center = load_registered_graph()
    del model
    query = parse_vnnlib_queries(Path(anchor["target"]["converted_spec"]),
        labeled_tensor=LabeledInputTensor(torch.zeros_like(center), torch.tensor([0])))
    if len(query) != 1:
        raise ValueError("query arity changed")
    spec = query[0][0]
    bounds = Bounds(spec.lb.to(center), spec.ub.to(center))
    candidate = copy.copy(net)
    candidate.preds = clone.predecessor_dict()
    candidate.succs = clone.successor_dict()
    arms = {}
    with torch.inference_mode():
        for name, graph in (("source", net), ("corrected", candidate)):
            concrete = concrete_forward(graph.layers, graph.preds, center)
            arms[name] = census(graph, bounds, concrete)
    core = screen(candidate.layers, candidate.preds, candidate.preds[36][0])
    support = next(r for r in arms["corrected"]["relus"] if r["layer"] == 36)
    # All actual target kernels have groups=1; no grouped inference is made.
    if any(r["geometry_cost"]["outer_groups"] != 1 for r in core["occurrences"]):
        raise ValueError("prospective channel slicing needs grouped accounting")
    full = core["total_contraction_products_no_cache_discount"]
    products = sum((r["geometry_cost"]["contraction_products"] // support["output_channels"])
                   * support["selected_channel_count"] for r in core["occurrences"])
    sources = [Path(__file__), EXPERIMENT / "CORRECTED_PHASE_SUPPORT_CENSUS_PREREG_20260905.md",
               EXPERIMENT / "test_corrected_phase_support_census_v1.py",
               EXPERIMENT / "run_s0_c3_graph_preflight_v1.py",
               EXPERIMENT / "run_s0_c4_nested_add_preflight_v1.py",
               ROOT / "act/back_end/interval_tf/interval_tf.py", ROOT / "act/back_end/interval_tf/tf_cnn.py",
               ROOT / "act/back_end/interval_tf/tf_mlp.py", ROOT / "act/back_end/utils.py"]
    record = {"schema": "corrected_phase_support_census_v1", "formal_gain": 0,
              "scope": "existing IntervalTF only; not HZ-tightened or independent outward-rounding authority",
              "target": anchor["target"], "source_graph_sha256": anchor["source_certificate"]["graph_sha256"],
              "corrected_graph_sha256": anchor["candidate_certificate"]["graph_sha256"],
              "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
              "arms": arms, "full_channel_contraction_products": full,
              "prospective_selected_channel_contraction_products": products,
              "prospective_within_contraction_cap": products <= 256_000_000,
              "solver_calls": 0, "hz_verifier_run": False, "positive_authorization": False,
              "seconds": time.monotonic() - started}
    _atomic_exclusive_json(output, record)
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "corrected_frontier": support,
                      "prospective_channel_products": products}))


if __name__ == "__main__":
    main()
