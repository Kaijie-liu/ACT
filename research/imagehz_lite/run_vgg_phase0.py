"""Phase 0 sentinel driver — VGG/Tiny conv-body representation experiment.

Per §9R-4/§9R-7/§9R-8 of `research/imagehz_vgg_prototype_plan.md`:
- runs ImageHZ-lite forward on 9 VGG canonical sentinels
- collects representation metrics only (no verifier, no LP, no FAL)
- evaluates the §9R-7 hard gate
- writes per-iid JSON + an aggregate summary + the gate result

NO production code is touched. The structural gate (§9R-5) is checked
on every iid; iids that fail it are skipped (with a recorded reason).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import torch

ACT_ROOT = Path(__file__).resolve().parents[2]
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT, build_provenance, load_instance,
)
from research.imagehz_lite.budget import Budget, BudgetExceeded
from research.imagehz_lite.domain import (
    ImageHZLite, Phase0FlattenSnapshot, TileBlock,
)
from research.imagehz_lite.metrics import (
    ALL_SENTINELS, Phase0Metrics, evaluate_phase0_gate,
)
from research.imagehz_lite.ops import (
    apply_conv2d, apply_flatten, apply_maxpool2d, apply_relu_triangle,
    structural_gate_passes,
)


import re

_VGG_LABEL_RE = re.compile(r"image\s+(\d+)|label[:\s]+(\d+)", re.I)


# 2026-06-04 hygiene: read per-iid Girard evidence from the merged atlas
# JSON instead of hardcoding True. This makes the structural gate honest:
# an iid not in the atlas can't be claimed as having Girard fires.
_MERGED_ATLAS_GLOB = "vgg_mini_atlas_canonical_plus_missing_*"
_GIRARD_TARGET_LAYERS = frozenset({11, 18, 25, 32, 17, 29, 35})


def _load_merged_atlas_girard_evidence() -> Dict[int, bool]:
    """Return a dict iid -> bool, True if the merged mini-atlas shows
    at least one Girard fire at one of the §6b target layers on that iid.

    Raises if no merged atlas dir is found, so the driver fails-closed
    rather than silently treating every iid as eligible.
    """
    audit_root = ACT_ROOT / "audit_results"
    candidates = sorted(audit_root.glob(_MERGED_ATLAS_GLOB),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(
            f"no merged atlas dir matching {_MERGED_ATLAS_GLOB} under "
            f"{audit_root}; cannot evaluate structural gate without "
            f"trace evidence"
        )
    latest = candidates[-1]
    agg_path = latest / "vgg_mini_atlas_merged.json"
    if not agg_path.exists():
        raise RuntimeError(
            f"merged atlas {latest} missing vgg_mini_atlas_merged.json"
        )
    data = json.loads(agg_path.read_text())
    out: Dict[int, bool] = {}
    for e in data.get("entries", []):
        iid = int(e.get("iid", -1))
        if iid < 0:
            continue
        fires = e.get("girard_fires") or []
        has_target_fire = any(
            int(f.get("layer_id", -1)) in _GIRARD_TARGET_LAYERS
            for f in fires
        )
        out[iid] = has_target_fire
    return out


def parse_input_box(vnn_path: Path, n_in: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quick per-X_<i> box parser shared with the rest of the project."""
    lb = np.full(n_in, -np.inf, dtype=np.float64)
    ub = np.full(n_in, np.inf, dtype=np.float64)
    pat_geq = re.compile(
        r"\(assert\s+\(>=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    pat_leq = re.compile(
        r"\(assert\s+\(<=\s+X_(\d+)\s+([-0-9eE.+]+)\s*\)\s*\)")
    with open(vnn_path) as f:
        text = f.read()
    for m in pat_geq.finditer(text):
        i = int(m.group(1)); v = float(m.group(2))
        lb[i] = max(lb[i], v) if np.isfinite(lb[i]) else v
    for m in pat_leq.finditer(text):
        i = int(m.group(1)); v = float(m.group(2))
        ub[i] = min(ub[i], v) if np.isfinite(ub[i]) else v
    if np.isinf(lb).any() or np.isinf(ub).any():
        raise RuntimeError(f"input box has unbounded dims: {vnn_path}")
    return lb, ub


def build_initial_imagehz_lite(
    lb: np.ndarray, ub: np.ndarray, shape_chw: Tuple[int, int, int],
) -> Tuple[ImageHZLite, int]:
    """One root TileBlock per non-degenerate input pixel.

    Each input pixel gets its own factor id 0..n-1; the generator value
    is the pixel's half-width. Phase 0 keeps tile shape (1, 1, 1) at the
    input — convs will grow them.
    """
    C, H, W = shape_chw
    n = C * H * W
    center = ((lb + ub) / 2.0).reshape(C, H, W)
    radius = ((ub - lb) / 2.0).reshape(C, H, W)
    c_t = torch.from_numpy(center).to(torch.float64)
    tiles: List[TileBlock] = []
    fid = 0
    for ci in range(C):
        for hi in range(H):
            for wi in range(W):
                r = float(radius[ci, hi, wi])
                if r == 0.0:
                    fid += 1
                    continue
                G = torch.tensor(
                    [[[[r]]]], dtype=torch.float64,
                )
                tiles.append(TileBlock(
                    origin_chw=(ci, hi, wi),
                    shape=(1, 1, 1),
                    G_tile=G,
                    factor_ids=(fid,),
                    aux_meta={
                        "kind": "root", "spawn_layer": 0,
                        "spawn_op": "input", "parent_block": None,
                    },
                ))
                fid += 1
    return ImageHZLite(c=c_t, tiles=tiles), fid


def _conv_attrs(node) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stride": 1, "padding": 0}
    for a in node.attribute:
        if a.name == "strides":
            out["stride"] = int(a.ints[0])
        elif a.name == "pads":
            out["padding"] = int(a.ints[0])
        elif a.name == "kernel_shape":
            out["kernel_shape"] = list(a.ints)
    return out


def _maxpool_attrs(node) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stride": None, "kernel_size": 2}
    for a in node.attribute:
        if a.name == "strides":
            out["stride"] = int(a.ints[0])
        elif a.name == "kernel_shape":
            out["kernel_size"] = int(a.ints[0])
    if out["stride"] is None:
        out["stride"] = out["kernel_size"]
    return out


def _peak_rss_bytes() -> int:
    """Linux ru_maxrss is kB; on macOS it's bytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = ru.ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _layer_op_kind(node) -> str:
    return node.op_type


def run_one_iid(
    iid: int, budget_cap: int = 20_000_000,
    layer_cap: Optional[int] = None,
    record_layer_trace: bool = True,
    girard_evidence: Optional[Dict[int, bool]] = None,
) -> Phase0Metrics:
    """Run ImageHZ-lite forward on one VGG canonical iid.

    Returns a Phase0Metrics with completed=True if the FLATTEN was
    reached and the structural gate passed, else completed=False with
    a fail_closed_reason.

    Memory note: VGG is large; this Phase 0 runs CPU only. ``layer_cap``
    optionally limits how many ops to step through (useful for debugging).
    """
    onnx_path, vnn_path = load_instance("vggnet16_2022", iid)
    prov = build_provenance("vggnet16_2022", iid).as_dict()
    m_onnx = onnx.load(str(onnx_path))
    in_dims = [d.dim_value for d in m_onnx.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W
    lb, ub = parse_input_box(Path(vnn_path), n_in)

    # Structural gate (§9R-5).
    op_names: List[str] = []
    flatten_idx = -1
    for i, node in enumerate(m_onnx.graph.node):
        op_names.append(_layer_op_kind(node))
        if node.op_type == "Flatten" and flatten_idx < 0:
            flatten_idx = i
            break  # we only need the prefix
    if flatten_idx < 0:
        return Phase0Metrics(
            iid=iid, completed=False,
            fail_closed_reason="no FLATTEN op in graph",
            root_ng_at_flatten=0, total_aux_count=0, wall_s=0.0,
            peak_memory_bytes=0, per_layer_l32_provenance_share=0.0,
            per_layer_l35_total_aux=0, layer_trace=[],
        )
    # 2026-06-04 hygiene: per-iid Girard evidence from merged atlas,
    # not a hardcoded True. If the iid is absent from the atlas, we
    # fail-closed (return False) rather than assume eligibility.
    if girard_evidence is None:
        girard_evidence = _load_merged_atlas_girard_evidence()
    iid_has_targeted_girard = girard_evidence.get(iid, False)
    gate_pass = structural_gate_passes(
        op_names, flatten_idx,
        trace_has_girard_root_loss_at_maxpool_or_relu=iid_has_targeted_girard,
    )
    if not gate_pass:
        return Phase0Metrics(
            iid=iid, completed=False,
            fail_closed_reason=(
                f"structural gate FAIL on iid {iid}: prefix ops={op_names[:flatten_idx]}"
            ),
            root_ng_at_flatten=0, total_aux_count=0, wall_s=0.0,
            peak_memory_bytes=0, per_layer_l32_provenance_share=0.0,
            per_layer_l35_total_aux=0, layer_trace=[],
        )

    # Initial ImageHZ-lite.
    hz, n_root = build_initial_imagehz_lite(lb, ub, (C, H, W))
    next_aux_id = n_root + 1
    budget = Budget(max_relu_aux_per_image=budget_cap)
    inits = {t.name: numpy_helper.to_array(t) for t in m_onnx.graph.initializer}

    girard_fires: List[Dict[str, Any]] = []
    layer_trace: List[Dict[str, Any]] = []
    l32_provenance_share = 0.0
    l35_total_aux = 0

    t0 = time.perf_counter()
    fail_closed_reason: Optional[str] = None
    flattened: Optional[Phase0FlattenSnapshot] = None
    try:
        layer_id = 0
        for node in m_onnx.graph.node:
            if layer_cap is not None and layer_id >= layer_cap:
                fail_closed_reason = f"layer_cap reached ({layer_cap})"
                break
            op = node.op_type
            if op == "Conv":
                w = torch.from_numpy(inits[node.input[1]].astype(np.float64))
                b = (torch.from_numpy(inits[node.input[2]].astype(np.float64))
                     if len(node.input) > 2 else None)
                attrs = _conv_attrs(node)
                ng_pre = hz.total_generator_count
                hz = apply_conv2d(
                    hz, w, bias=b,
                    stride=attrs.get("stride", 1),
                    padding=attrs.get("padding", 0),
                )
                ng_post = hz.total_generator_count
                layer_trace.append({
                    "layer_id": layer_id, "op": "Conv2D",
                    "ng_pre": ng_pre, "ng_post": ng_post,
                    "shape": tuple(hz.shape),
                })
            elif op in ("Relu", "ReLU"):
                ng_pre = hz.total_generator_count
                hz, next_aux_id = apply_relu_triangle(
                    hz, budget,
                    layer_id=layer_id, next_aux_id=next_aux_id,
                )
                ng_post = hz.total_generator_count
                layer_trace.append({
                    "layer_id": layer_id, "op": "ReLU",
                    "ng_pre": ng_pre, "ng_post": ng_post,
                    "shape": tuple(hz.shape),
                })
                # L35 RELU recording.
                if layer_id == 35:
                    l35_total_aux = sum(
                        t.n_gen_tile for t in hz.tiles
                        if t.aux_meta.get("kind") == "relu_aux"
                    )
            elif op in ("MaxPool", "MaxPool2D"):
                attrs = _maxpool_attrs(node)
                ng_pre = hz.total_generator_count
                hz, next_aux_id, stats = apply_maxpool2d(
                    hz, kernel_size=attrs["kernel_size"], stride=attrs["stride"],
                    budget=budget, layer_id=layer_id, next_aux_id=next_aux_id,
                )
                ng_post = hz.total_generator_count
                layer_trace.append({
                    "layer_id": layer_id, "op": "MaxPool2D",
                    "ng_pre": ng_pre, "ng_post": ng_post,
                    "shape": tuple(hz.shape),
                    **stats,
                })
                if ng_post < ng_pre:
                    girard_fires.append({
                        "layer_id": layer_id, "op": "MaxPool2D",
                        "ng_pre": ng_pre, "ng_post": ng_post,
                    })
                # L32 provenance share recording.
                if layer_id == 32:
                    n_total = stats["n_output_positions"]
                    n_root = stats["n_output_positions_with_root_provenance"]
                    l32_provenance_share = (
                        n_root / n_total if n_total > 0 else 0.0
                    )
            elif op == "Flatten":
                t_flatten = time.perf_counter()
                wall = t_flatten - t0
                rss = _peak_rss_bytes()
                flattened = apply_flatten(
                    hz, girard_fires=girard_fires,
                    peak_memory_bytes=rss, wall_s=wall,
                )
                break
            else:
                fail_closed_reason = (
                    f"structural drift at layer {layer_id}: op={op} "
                    f"not in allow-list"
                )
                break
            layer_id += 1
    except BudgetExceeded as e:
        fail_closed_reason = f"BudgetExceeded: {e}"

    wall = time.perf_counter() - t0
    rss = _peak_rss_bytes()

    if flattened is None:
        return Phase0Metrics(
            iid=iid, completed=False,
            fail_closed_reason=fail_closed_reason or "did not reach FLATTEN",
            root_ng_at_flatten=0, total_aux_count=0, wall_s=wall,
            peak_memory_bytes=rss,
            per_layer_l32_provenance_share=l32_provenance_share,
            per_layer_l35_total_aux=l35_total_aux,
            layer_trace=layer_trace,
        )

    return Phase0Metrics(
        iid=iid, completed=True, fail_closed_reason=None,
        root_ng_at_flatten=flattened.root_ng_at_flatten,
        total_aux_count=flattened.total_aux_count,
        wall_s=wall, peak_memory_bytes=rss,
        per_layer_l32_provenance_share=l32_provenance_share,
        per_layer_l35_total_aux=l35_total_aux,
        layer_trace=layer_trace,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--iids", type=str,
        default=",".join(str(i) for i in ALL_SENTINELS),
        help="comma-separated iids; default = §9R-7 sentinels",
    )
    ap.add_argument("--budget", type=int, default=20_000_000,
                    help="max_relu_aux_per_image budget cap")
    ap.add_argument(
        "--out", type=str,
        default="",
        help="output dir; defaults to audit_results/imagehz_lite_phase0_<STAMP>",
    )
    ap.add_argument("--layer-cap", type=int, default=None,
                    help="optional cap on operator steps per iid (debug)")
    args = ap.parse_args()

    iids = [int(x) for x in args.iids.split(",") if x.strip()]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = (
        Path(args.out) if args.out else
        ACT_ROOT / "audit_results" / f"imagehz_lite_phase0_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    per_iid_dir = out_root / "per_iid"
    per_iid_dir.mkdir(exist_ok=True)

    # Load merged atlas girard evidence ONCE; pass per-iid into the runner.
    try:
        girard_evidence = _load_merged_atlas_girard_evidence()
        print(f"[driver] loaded girard evidence for "
              f"{sum(1 for v in girard_evidence.values() if v)} iids "
              f"of {len(girard_evidence)} in merged atlas", flush=True)
    except RuntimeError as e:
        print(f"[driver] FAIL-CLOSED on atlas load: {e}", flush=True)
        return 2

    results: List[Phase0Metrics] = []
    for iid in iids:
        print(f"--- ImageHZ-lite Phase 0 iid {iid} ---", flush=True)
        try:
            m = run_one_iid(iid, budget_cap=args.budget, layer_cap=args.layer_cap,
                            girard_evidence=girard_evidence)
        except Exception as e:
            m = Phase0Metrics(
                iid=iid, completed=False,
                fail_closed_reason=f"unhandled exception: {type(e).__name__}: {e}",
                root_ng_at_flatten=0, total_aux_count=0, wall_s=0.0,
                peak_memory_bytes=0, per_layer_l32_provenance_share=0.0,
                per_layer_l35_total_aux=0, layer_trace=[],
            )
        results.append(m)
        with open(per_iid_dir / f"iid{iid:03d}.json", "w") as f:
            json.dump({
                "iid": m.iid,
                "completed": m.completed,
                "fail_closed_reason": m.fail_closed_reason,
                "root_ng_at_flatten": m.root_ng_at_flatten,
                "total_aux_count": m.total_aux_count,
                "wall_s": m.wall_s,
                "peak_memory_bytes": m.peak_memory_bytes,
                "l32_provenance_share": m.per_layer_l32_provenance_share,
                "l35_total_aux": m.per_layer_l35_total_aux,
                "layer_trace": m.layer_trace,
            }, f, indent=2, default=float)
        print(
            f"  completed={m.completed}  wall={m.wall_s:.1f}s  "
            f"root_ng={m.root_ng_at_flatten}  total_aux={m.total_aux_count}  "
            f"l32_root_share={m.per_layer_l32_provenance_share:.2f}  "
            f"l35_aux={m.per_layer_l35_total_aux}  "
            f"reason={m.fail_closed_reason}",
            flush=True,
        )
        gc.collect()

    gate = evaluate_phase0_gate(results)
    with open(out_root / "phase0_gate.json", "w") as f:
        json.dump({
            "stamp": stamp,
            "iids_attempted": iids,
            "gate": gate,
        }, f, indent=2, default=float)
    print()
    print(f"=== Phase 0 gate ===")
    print(f"  overall_pass: {gate['overall_pass']}")
    for fail in gate["failures"]:
        print(f"  FAIL: {fail}")
    print(f"  out: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
