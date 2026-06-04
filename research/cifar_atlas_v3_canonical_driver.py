"""CIFAR atlas v3 (canonical-rooted) — per-iid diagnostics builder.

Per advisor 2026-06-03 directive after A++ closed-negative on canonical
CIFAR. Atlas v3 records, for each of the 185 canonical atlas-UNK iids:

- final_lp_margin           — tight-LP top-1 UB on (Y[r] - Y[t])
- final_box_margin          — closed-form box-LP top-1 UB on (Y[r] - Y[t])
- lp_over_box_ratio         — lp / box  (≤ 1; smaller = LP tighter)
- top_rival_lp              — argmax rival from LP
- top_rival_box             — argmax rival from box
- lp_candidate_ort_margin   — ORT replay margin under LP's xi_root
- box_candidate_ort_margin  — ORT replay margin under box's xi_root
- phantom_lp                — lp_margin - lp_candidate_ort_margin
- phantom_box               — box_margin - box_candidate_ort_margin
- final_relu_unstable_count — # unstable neurons at hidden layer
- final_relu_total_slack    — Σ mu_i (DeepZ triangle constant)
- root_ng_to_input_ratio    — root_ng / n_input (1.0 = full pixel correlation)
- ng_total                  — total generators at snapshot
- nc_at_snapshot, nb_at_snapshot
- boxhz_collapse_flag       — nc == 0 AND nb == 0 (no constraints)
- provenance                — canonical_root, vnnlib_sha256, etc.

This is NOT a verifier. It runs the production HZ pipeline as a
subprocess per iid (to get a per-iid FLATTEN snapshot), then loads
the snapshot and runs the diagnostic computation. Output goes under
``audit_results/cifar_unknown_margin_atlas_canonical_<STAMP>/``.

Principle compliance: forward-only diagnostics. No CROWN, no PGD,
no MILP. ORT replay is deterministic single-input CPU.

Usage:
    python research/cifar_atlas_v3_canonical_driver.py
    python research/cifar_atlas_v3_canonical_driver.py --iids 0,1,3 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parent.parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

HYZOR_ROOT = Path("/data1/Kane/HyZor")
if str(HYZOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HYZOR_ROOT))

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT, build_provenance, load_instance,
)


ATLAS_V2_PATH = (
    "/data1/Kane/ACT/audit_results/cifar_unknown_margin_atlas_20260603/"
    "atlas_v2_bucketed.json"
)
PY = "/data1/Kane/miniconda3/envs/act-py312/bin/python"

LABEL_RE = re.compile(r";\s*CIFAR100\s+property\s+with\s+label:\s*(\d+)")


def parse_y_true(vnn_path: Path) -> int:
    with open(vnn_path) as f:
        first = f.readline()
    m = LABEL_RE.search(first)
    if not m:
        raise RuntimeError(
            f"could not parse y_true from {vnn_path}: {first!r}")
    return int(m.group(1))


def parse_input_box(
    vnn_path: Path, n_in: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
        raise RuntimeError(f"vnnlib parse left unbounded inputs at {vnn_path}")
    return lb, ub


# ── Production pipeline subprocess ────────────────────────────────


def run_production_for_snapshot(
    iid: int, snap_dir: Path, out_dir: Path, *,
    timeout_s: int = 240,
) -> Dict[str, Any]:
    """Drive the production HZ pipeline for a single canonical CIFAR iid.
    Writes the L*_FLATTEN.pkl snapshot into ``snap_dir``. Returns
    a metadata dict (production verdict, wall, etc.).
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/data1/Kane/ACT"
    env["ACT_VNNLIB_ROOT"] = str(CANONICAL_ROOT)
    env["ACT_HZ_ENDCAP_SNAPSHOT_DIR"] = str(snap_dir)
    env["ACT_HZ_ENDCAP_SNAPSHOT_KIND"] = "FLATTEN"
    env["ACT_HZ_CIFAR_ENDCAP_PROFILE"] = "1"
    # Disable the production witness sidecar for atlas; we're only
    # here for the snapshot + diagnostics, not for FAL receipts.
    env["ACT_HZ_CIFAR_ENDCAP_WITNESS"] = "0"
    env["ACT_HZ_TOPK_RIVAL_WITNESS"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    cmd = [
        PY, "-m", "act.pipeline.watchdog_runner",
        "--benchmark", "cifar100_2024",
        "--instance-ids", str(iid),
        "--wall-s", str(timeout_s),
        "--device", "cuda",
        "--dtype", "float64",
        "--rss-cap-gb", "32",
        "--out-dir", str(out_dir),
        "--canonical-root", str(CANONICAL_ROOT),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, env=env, cwd=str(ACT_ROOT),
        capture_output=True, text=True, timeout=timeout_s + 60,
    )
    wall = time.perf_counter() - t0
    return {
        "returncode": proc.returncode,
        "wall_s": wall,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
    }


def find_snapshot(snap_dir: Path) -> Optional[Path]:
    cands = sorted(snap_dir.glob("L*_FLATTEN.pkl"))
    return cands[0] if cands else None


# ── Diagnostic computation on a snapshot ──────────────────────────


def load_snapshot(snap_path: Path) -> Dict[str, Any]:
    with open(snap_path, "rb") as f:
        return pickle.load(f)


def compute_tail_output(
    c_in: np.ndarray, Gc_in: np.ndarray,
    W39: np.ndarray, b39: np.ndarray,
    W41: np.ndarray, b41: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Forward c, Gc through tail (Gemm + DeepZ triangle + Gemm).
    Returns (c_y, Gc_y_augmented, relu_stats).
    """
    c_h = W39 @ c_in + b39
    Gc_h = W39 @ Gc_in
    radius_h = np.abs(Gc_h).sum(axis=1)
    lb_h = c_h - radius_h
    ub_h = c_h + radius_h
    active = lb_h >= 0.0
    inactive = ub_h <= 0.0
    unstable = ~(active | inactive)
    unstable_idx = np.where(unstable)[0]
    k = int(unstable_idx.size)

    c_h_post = np.zeros_like(c_h)
    c_h_post[active] = c_h[active]
    Gc_h_post_base = np.zeros_like(Gc_h)
    Gc_h_post_base[active, :] = Gc_h[active, :]

    total_mu = 0.0
    if k > 0:
        l_u = lb_h[unstable_idx]
        u_u = ub_h[unstable_idx]
        lam = u_u / (u_u - l_u)
        mu = -l_u * u_u / (2.0 * (u_u - l_u))
        total_mu = float(np.sum(mu))
        c_h_post[unstable_idx] = lam * c_h[unstable_idx] + mu
        Gc_h_post_base[unstable_idx, :] = lam[:, None] * Gc_h[unstable_idx, :]
        Gc_aux = np.zeros((c_h.shape[0], k), dtype=Gc_h.dtype)
        for j, i in enumerate(unstable_idx):
            Gc_aux[i, j] = mu[j]
        Gc_h_post = np.concatenate([Gc_h_post_base, Gc_aux], axis=1)
    else:
        Gc_h_post = Gc_h_post_base

    c_y = W41 @ c_h_post + b41
    Gc_y = W41 @ Gc_h_post
    relu_stats = {
        "n_active": int(active.sum()),
        "n_inactive": int(inactive.sum()),
        "n_unstable": k,
        "total_mu_sum": total_mu,
    }
    return c_y, Gc_y, relu_stats


def lp_top1_box(
    c_y: np.ndarray, Gc_y: np.ndarray, y_true: int,
) -> Tuple[int, float, np.ndarray]:
    """Closed-form box LP across all rivals. Returns
    (top_rival, top_margin_ub, top_xi_star) where top_margin_ub is the
    LP upper bound on (Y[r] - Y[t]) at the top rival, and top_xi_star
    is the corner sign vector.
    """
    n_y = int(c_y.shape[0])
    diff_c = c_y - c_y[y_true]
    diff_G = Gc_y - Gc_y[y_true:y_true + 1, :]
    ubs = diff_c + np.abs(diff_G).sum(axis=1)
    ubs[y_true] = -np.inf
    top_r = int(np.argmax(ubs))
    top_ub = float(ubs[top_r])
    xi_star = np.sign(diff_G[top_r, :])
    xi_star[xi_star == 0.0] = 1.0
    return top_r, top_ub, xi_star


def lp_top1_tight(
    snap: Dict[str, Any], onnx_path: Path, y_true: int,
    rivals: List[int],
) -> Tuple[int, float, np.ndarray]:
    """Tight-triangle LP top-1 across rivals (uses production LP path).
    Returns (top_rival, top_margin_ub, xi_root_lp).
    """
    from receipt_factor_aware_endcap_lp import (
        _load_classifier_weights, _build_h39_affine, _h39_bounds,
    )
    from pilot_cifar_endcap_diagnose import _solve_endcap_lp_with_solution
    c = snap["c"].numpy() if hasattr(snap["c"], "numpy") else np.asarray(snap["c"])
    Gc = snap["Gc"].numpy() if hasattr(snap["Gc"], "numpy") else np.asarray(snap["Gc"])
    root_ng = int(snap.get("root_ng", 0))
    wts = _load_classifier_weights(str(onnx_path))
    W39, b39 = wts["linear1.weight"], wts["linear1.bias"]
    W41, b41 = wts["linear2.weight"], wts["linear2.bias"]
    c39, G39 = _build_h39_affine(W39, b39, c, Gc)
    l39, u39 = _h39_bounds(c39, G39)
    best_r, best_ub, best_xi = -1, -np.inf, None
    for j in rivals:
        s = _solve_endcap_lp_with_solution(
            W41, b41, c39, G39, l39, u39, y_true, j, time_limit_s=8.0,
        )
        lp_min = s.get("lp_min")
        if lp_min is None or not np.isfinite(float(lp_min)):
            continue
        ub = -float(lp_min)
        if ub > best_ub:
            best_ub = ub
            best_r = int(j)
            best_xi = np.asarray(s["xi_star"])[:root_ng].copy()
    if best_xi is None:
        # all LPs failed → return inf bound to flag
        return -1, float("nan"), np.zeros(root_ng, dtype=np.float64)
    return best_r, best_ub, best_xi


def ort_replay(
    onnx_path: Path, x_cand: np.ndarray, in_shape: Tuple[int, int, int],
) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    C, H, W = in_shape
    x_in = x_cand.astype(np.float32).reshape(1, C, H, W)
    y = sess.run(None, {inp.name: x_in})[0]
    return np.asarray(y, dtype=np.float64).reshape(-1)


# ── Per-iid diagnostic ───────────────────────────────────────────


def diagnose_one_iid(
    iid: int, snap_dir: Path, out_dir: Path,
) -> Dict[str, Any]:
    onnx_p, vnn_p = load_instance("cifar100_2024", iid)
    prov = build_provenance("cifar100_2024", iid).as_dict()
    y_true = parse_y_true(vnn_p)

    # Input dims
    import onnx as _onnx
    m_onnx = _onnx.load(str(onnx_p))
    in_dims = [d.dim_value for d in m_onnx.graph.input[0].type.tensor_type.shape.dim]
    C, H, W = int(in_dims[1]), int(in_dims[2]), int(in_dims[3])
    n_in = C * H * W
    lb, ub = parse_input_box(vnn_p, n_in)
    c_box = (lb + ub) / 2.0
    half = (ub - lb) / 2.0

    # 1) Run production HZ pipeline → snapshot
    prod_info = run_production_for_snapshot(iid, snap_dir, out_dir)
    snap_path = find_snapshot(snap_dir)
    if snap_path is None:
        return {
            "iid": iid, "y_true": y_true,
            "verdict": "ERROR_NO_SNAPSHOT",
            "production": prod_info,
            "provenance": prov,
        }
    snap = load_snapshot(snap_path)

    # 2) Forward tail to get c_y, Gc_y
    from receipt_factor_aware_endcap_lp import _load_classifier_weights
    wts = _load_classifier_weights(str(onnx_p))
    W39, b39 = wts["linear1.weight"], wts["linear1.bias"]
    W41, b41 = wts["linear2.weight"], wts["linear2.bias"]
    c_in = snap["c"].numpy() if hasattr(snap["c"], "numpy") else np.asarray(snap["c"])
    Gc_in = snap["Gc"].numpy() if hasattr(snap["Gc"], "numpy") else np.asarray(snap["Gc"])
    c_in = c_in.reshape(-1).astype(np.float64)
    Gc_in = Gc_in.reshape(c_in.shape[0], -1).astype(np.float64)
    c_y, Gc_y, relu_stats = compute_tail_output(c_in, Gc_in, W39, b39, W41, b41)

    # 3) Box-LP top-1 + xi_star → reconstruct → ORT.
    # 2026-06-04 fix: when snapshot's root_ng < n_input, reduction has
    # merged multiple input pixels into shared root factors, so we
    # cannot recover per-pixel xi from the compressed factors. The
    # production sidecar fails-closed here. Atlas v3 records the
    # compression event in the diag and skips the replay; the LP/box
    # margins are still valid as upper bounds on the OUTPUT space.
    box_r, box_ub, box_xi_full = lp_top1_box(c_y, Gc_y, y_true)
    root_ng = int(snap.get("root_ng", 0))
    root_ng_eq_input = (root_ng == n_in)

    if root_ng_eq_input:
        box_xi_root = box_xi_full[:root_ng]
        x_cand_box = np.clip(c_box + half * box_xi_root, lb, ub)
        y_ort_box = ort_replay(onnx_p, x_cand_box, in_shape=(C, H, W))
        box_replay_margin = float(y_ort_box[box_r] - y_ort_box[y_true])
        box_replay_argmax = int(np.argmax(y_ort_box))
    else:
        box_replay_margin = float("nan")
        box_replay_argmax = -1

    # 4) Tight-LP top-1 + xi_root_LP → reconstruct → ORT
    rivals = [j for j in range(int(c_y.shape[0])) if j != y_true]
    lp_r, lp_ub, lp_xi_root = lp_top1_tight(snap, onnx_p, y_true, rivals)
    if lp_r >= 0 and root_ng_eq_input:
        x_cand_lp = np.clip(c_box + half * np.clip(lp_xi_root, -1, 1), lb, ub)
        y_ort_lp = ort_replay(onnx_p, x_cand_lp, in_shape=(C, H, W))
        lp_replay_margin = float(y_ort_lp[lp_r] - y_ort_lp[y_true])
        lp_replay_argmax = int(np.argmax(y_ort_lp))
    else:
        lp_replay_margin = float("nan")
        lp_replay_argmax = -1

    # 5) Snapshot structural fields
    snap_ng = int(snap.get("ng", Gc_in.shape[1]))
    snap_nc = int(snap.get("nc", 0))
    snap_nb = int(snap.get("nb", 0))
    boxhz_collapse = bool(snap_nc == 0 and snap_nb == 0)

    # 6) Roll up
    return {
        "iid": iid,
        "y_true_vnnlib": int(y_true),
        "n_input": int(n_in),
        # margins
        "final_lp_margin": lp_ub,
        "final_box_margin": box_ub,
        "lp_over_box_ratio": (
            float(lp_ub / box_ub) if (np.isfinite(lp_ub) and box_ub != 0)
            else float("nan")
        ),
        # top rivals
        "top_rival_lp": int(lp_r),
        "top_rival_box": int(box_r),
        # ORT replays
        "lp_candidate_ort_margin": lp_replay_margin,
        "box_candidate_ort_margin": box_replay_margin,
        "lp_candidate_ort_argmax": lp_replay_argmax,
        "box_candidate_ort_argmax": box_replay_argmax,
        # phantom (LP UB - ORT actual)
        "phantom_lp": (
            float(lp_ub - lp_replay_margin)
            if np.isfinite(lp_ub) and np.isfinite(lp_replay_margin)
            else float("nan")
        ),
        "phantom_box": float(box_ub - box_replay_margin),
        # FAL-by-strict-replay flags
        "lp_replay_is_fal": bool(
            np.isfinite(lp_replay_margin)
            and lp_replay_argmax != y_true
            and lp_replay_margin > 0.0
        ),
        "box_replay_is_fal": bool(
            box_replay_argmax != y_true and box_replay_margin > 0.0
        ),
        # ReLU slack
        "final_relu_n_unstable": int(relu_stats["n_unstable"]),
        "final_relu_n_active": int(relu_stats["n_active"]),
        "final_relu_n_inactive": int(relu_stats["n_inactive"]),
        "final_relu_total_mu_sum": float(relu_stats["total_mu_sum"]),
        # correlation
        "root_ng": int(root_ng),
        "ng_at_snapshot": snap_ng,
        "root_ng_eq_input": bool(root_ng_eq_input),
        "root_ng_to_input_ratio": float(root_ng / n_in) if n_in > 0 else float("nan"),
        "ng_to_input_ratio": float(snap_ng / n_in) if n_in > 0 else float("nan"),
        # constraints
        "nc_at_snapshot": snap_nc,
        "nb_at_snapshot": snap_nb,
        "boxhz_collapse": boxhz_collapse,
        # provenance + production state
        "provenance": prov,
        "production_returncode": prod_info["returncode"],
        "production_wall_s": prod_info["wall_s"],
        "snapshot_path": str(snap_path),
    }


# ── Driver ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iids", type=str, default="",
                    help="comma-separated iids; empty = all atlas-UNK")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--keep-snapshots", action="store_true",
                    help="don't delete per-iid snap dirs after diagnostics")
    args = ap.parse_args()

    if args.iids:
        iids = [int(x) for x in args.iids.split(",") if x.strip()]
    else:
        with open(ATLAS_V2_PATH) as f:
            a = json.load(f)
        iids = sorted(int(e["iid"]) for e in a)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_base = (
        Path(args.out) if args.out else
        Path(f"/data1/Kane/ACT/audit_results/"
             f"cifar_unknown_margin_atlas_canonical_{stamp}")
    )
    out_base.mkdir(parents=True, exist_ok=True)
    per_iid_dir = out_base / "per_iid"
    per_iid_dir.mkdir(parents=True, exist_ok=True)
    snap_root = out_base / "snapshots"
    snap_root.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    t_all0 = time.perf_counter()
    for k, iid in enumerate(iids):
        print(f"--- [{k+1:>3}/{len(iids)}] iid {iid} ---", flush=True)
        snap_dir = snap_root / f"iid{iid:03d}"
        run_out = out_base / "run_outputs" / f"iid{iid:03d}"
        try:
            diag = diagnose_one_iid(iid, snap_dir, run_out)
        except Exception as e:
            diag = {
                "iid": iid, "verdict": "ERROR",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:500],
            }
        # Drop the big snapshot file unless --keep-snapshots
        if not args.keep_snapshots and snap_dir.exists():
            shutil.rmtree(snap_dir, ignore_errors=True)
        with open(per_iid_dir / f"iid{iid:03d}.json", "w") as f:
            json.dump(diag, f, indent=2, default=float)
        summary.append(diag)
        # progress (safe formatting; ERROR rows have no fields)
        if diag.get("verdict", "").startswith("ERROR"):
            print(
                f"  ERROR: {diag.get('verdict')} "
                f"{diag.get('error_type','')}: "
                f"{diag.get('error_msg','')[:120]}",
                flush=True,
            )
        else:
            def _fmt(v, w=8, p=4):
                try: return f"{float(v):>+{w}.{p}f}"
                except Exception: return f"{'N/A':>{w}}"
            print(
                f"  lp_ub={_fmt(diag.get('final_lp_margin'))}  "
                f"box_ub={_fmt(diag.get('final_box_margin'))}  "
                f"top_lp_rival={diag.get('top_rival_lp')}  "
                f"top_box_rival={diag.get('top_rival_box')}  "
                f"phantom_lp={_fmt(diag.get('phantom_lp'))}  "
                f"root_ratio={_fmt(diag.get('root_ng_to_input_ratio'), w=5, p=3)}  "
                f"BoxHZ={diag.get('boxhz_collapse')}",
                flush=True,
            )

    t_all = time.perf_counter() - t_all0

    # Roll-up
    with open(out_base / "atlas_v3.json", "w") as f:
        json.dump({
            "stamp": stamp,
            "canonical_root": str(CANONICAL_ROOT),
            "n_iids": len(summary),
            "wall_s_total": t_all,
            "entries": summary,
        }, f, indent=2, default=float)

    csv_path = out_base / "summary.csv"
    keys = [
        "iid", "y_true_vnnlib",
        "final_lp_margin", "final_box_margin", "lp_over_box_ratio",
        "top_rival_lp", "top_rival_box",
        "lp_candidate_ort_margin", "box_candidate_ort_margin",
        "phantom_lp", "phantom_box",
        "lp_replay_is_fal", "box_replay_is_fal",
        "final_relu_n_unstable", "final_relu_total_mu_sum",
        "root_ng", "ng_at_snapshot", "root_ng_to_input_ratio",
        "nc_at_snapshot", "nb_at_snapshot", "boxhz_collapse",
        "production_returncode", "production_wall_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for d in summary:
            w.writerow([d.get(k, "") for k in keys])

    print()
    print(f"=== atlas v3 done in {t_all:.1f}s ({t_all/60:.1f} min) ===")
    print(f"  atlas_v3.json: {out_base / 'atlas_v3.json'}")
    print(f"  summary.csv:   {csv_path}")
    print(f"  per_iid:       {per_iid_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
