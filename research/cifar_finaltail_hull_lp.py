"""CIFAR Final-Tail Hull Prototype — Phase 1 driver.

Per the design lock in `research/cifar_finaltail_hull_plan.md`:
- explicit forward LP over (xi, z, h) on the final hidden ReLU only,
- per-neuron triangle convex hull (provably tight at per-neuron),
- per-rival max objective y_r - y_true,
- strict ORT replay for any FAL claim,
- no production routing change.

Workflow per iid:
1. Capture FLATTEN snapshot via `act.pipeline.watchdog_runner` with
   `ACT_HZ_ENDCAP_SNAPSHOT_DIR` + `ACT_HZ_ENDCAP_SNAPSHOT_KIND=FLATTEN`.
2. Load the snapshot (c, Gc) + ONNX (linear1, linear2) + vnnlib
   (y_true, rivals).
3. Build the explicit forward LP per advisor §1 spec; solve for every
   rival.
4. Compare clean LP UB to the production baseline LP UB (recomputed
   from the SAME snapshot via `pilot_cifar_endcap_diagnose._solve_endcap_lp_with_solution`).
5. If max-over-rivals UB < 0 (strictly): emit CERT.
   If any rival UB >= 0: decode xi* to input, strict ORT replay; on
   pass, FAL. On fail, UNKNOWN.

Usage:
    python research/cifar_finaltail_hull_lp.py --iids 113,29,153 \
        --out audit_results/cifar_finaltail_hull_phase1_smoke_<STAMP>

NO writes to production code. Snapshots are captured to a temp dir
under the run output and cleaned up after each iid unless --keep-snapshots.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ACT_ROOT = Path(__file__).resolve().parents[1]
HYZOR_ROOT = Path("/data1/Kane/HyZor")
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))
if str(HYZOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HYZOR_ROOT))

from research.canonical_provenance import (  # noqa: E402
    CANONICAL_ROOT, build_provenance, load_instance,
)


# Reuse the spec-load and bound helpers from the production pilot
from pilot_cifar_endcap_lp import (  # noqa: E402
    _load_classifier_weights,
    _parse_top1_spec,
    _build_h39_affine,
    _h39_bounds,
    _solve_endcap_lp as _production_endcap_lp,
)
from pilot_cifar_endcap_diagnose import (  # noqa: E402
    _solve_endcap_lp_with_solution as _production_endcap_lp_with_sol,
)


# ─── Snapshot capture ─────────────────────────────────────────────


def capture_flatten_snapshot(
    iid: int,
    snap_dir: Path,
    canonical_root: Path = CANONICAL_ROOT,
    wall_s: int = 600,
    rss_cap_gb: int = 32,
) -> Path:
    """Re-run production with snapshot capture; return the path to the
    L*_FLATTEN.pkl file. Raises if no snapshot is produced."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ACT_HZ_ENDCAP_SNAPSHOT_DIR"] = str(snap_dir)
    env["ACT_HZ_ENDCAP_SNAPSHOT_KIND"] = "FLATTEN"
    env["PYTHONPATH"] = str(ACT_ROOT)
    env["ACT_VNNLIB_ROOT"] = str(canonical_root)
    env["ACT_HZ_TOPK_RIVAL_WITNESS"] = "5"
    env["ACT_HZ_CIFAR_ENDCAP_WITNESS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    cmd = [
        "/data1/Kane/miniconda3/envs/act-py312/bin/python",
        "-m", "act.pipeline.watchdog_runner",
        "--benchmark", "cifar100_2024",
        "--instance-ids", str(iid),
        "--wall-s", str(wall_s),
        "--device", "cuda",
        "--dtype", "float64",
        "--rss-cap-gb", str(rss_cap_gb),
        "--out-dir", str(snap_dir / "run_out"),
        "--canonical-root", str(canonical_root),
    ]
    log = snap_dir / "capture.log"
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(
            f"snapshot capture failed (rc={r.returncode}); see {log}"
        )
    snaps = sorted(snap_dir.glob("L*_FLATTEN.pkl"))
    if not snaps:
        raise RuntimeError(f"no L*_FLATTEN.pkl produced under {snap_dir}; "
                           f"see {log}")
    return snaps[-1]


# ─── Clean LP per design lock §1 ─────────────────────────────────


def solve_clean_finaltail_lp(
    W39: np.ndarray, b39: np.ndarray,
    W41: np.ndarray, b41: np.ndarray,
    c: np.ndarray, Gc: np.ndarray,
    y_true: int, rival: int,
    time_limit_s: float = 30.0,
) -> Dict[str, Any]:
    """Explicit forward LP per advisor 2026-06-04 design lock §1.3.

    Variables:
        xi  ∈ [-1, +1]^ng                  (root + aux factors)
        h   ∈ R^{n_h}                      (post-ReLU)
            stable active:   h_j = z_j      (equality)
            stable inactive: h_j = 0        (equality, encoded by skip)
            unstable:        per-neuron triangle on (z_j, h_j)

    Objective:  maximize y_r - y_t  =  -(c_y_t - c_y_r) - (W41[t,:]-W41[r,:]) @ h

    We MINIMIZE (W41[t,:] - W41[r,:]) @ h + (b41[t] - b41[r]) which is
    the rival margin y_t - y_r; the returned `rival_lp_ub_on_y_diff` is
    max(y_r - y_t) = -lp_min.
    """
    import highspy
    ng = int(Gc.shape[1])
    n_h = int(W39.shape[0])

    c_z = (W39 @ c.reshape(-1)) + b39.reshape(-1)
    G_z = W39 @ Gc                                          # (n_h, ng)

    lb_z = c_z - np.abs(G_z).sum(axis=1)
    ub_z = c_z + np.abs(G_z).sum(axis=1)

    stable_active = np.where(lb_z >= 0.0)[0]
    stable_inactive = np.where(ub_z <= 0.0)[0]
    unstable = np.where((lb_z < 0.0) & (ub_z > 0.0))[0]
    n_unstable = int(unstable.size)

    coef = W41[y_true, :] - W41[rival, :]                    # (n_h,)
    bias = float(b41[y_true] - b41[rival])

    obj_xi = np.zeros(ng, dtype=np.float64)
    obj_h = np.zeros(n_unstable, dtype=np.float64)
    const_offset = bias
    # stable-active contributes coef · z = coef · (c_z + G_z xi)
    if stable_active.size > 0:
        obj_xi += coef[stable_active] @ G_z[stable_active, :]
        const_offset += float(coef[stable_active] @ c_z[stable_active])
    # stable-inactive contributes 0
    if n_unstable > 0:
        obj_h[:] = coef[unstable]

    H = highspy.Highs()
    H.silent()
    H.setOptionValue("time_limit", float(time_limit_s))
    H.setOptionValue("output_flag", False)
    INF = highspy.kHighsInf

    empty_starts = np.zeros(0, dtype=np.int32)
    empty_idx = np.zeros(0, dtype=np.int32)
    empty_vals = np.zeros(0, dtype=np.float64)
    H.addCols(ng, obj_xi,
              -np.ones(ng, dtype=np.float64),
              np.ones(ng, dtype=np.float64),
              0, empty_starts, empty_idx, empty_vals)
    if n_unstable > 0:
        H.addCols(n_unstable, obj_h,
                  np.zeros(n_unstable, dtype=np.float64),
                  ub_z[unstable].astype(np.float64),
                  0, empty_starts, empty_idx, empty_vals)

    for r_idx, j in enumerate(unstable):
        l_j = float(lb_z[j]); u_j = float(ub_z[j])
        if (u_j - l_j) <= 0.0:
            continue
        slope = u_j / (u_j - l_j)
        # row 1:   h_j  -  G_z[j,:] xi  >=  c_z[j]
        idx1 = list(np.nonzero(G_z[j, :])[0])
        val1 = [-float(G_z[j, k]) for k in idx1]
        idx1.append(ng + r_idx); val1.append(1.0)
        H.addRow(float(c_z[j]), INF, len(idx1),
                 np.asarray(idx1, dtype=np.int32),
                 np.asarray(val1, dtype=np.float64))
        # row 2:   h_j  -  slope * G_z[j,:] xi  <=  slope * (c_z[j] - l_j)
        idx2 = list(np.nonzero(G_z[j, :])[0])
        val2 = [-slope * float(G_z[j, k]) for k in idx2]
        idx2.append(ng + r_idx); val2.append(1.0)
        H.addRow(-INF, slope * (float(c_z[j]) - l_j), len(idx2),
                 np.asarray(idx2, dtype=np.int32),
                 np.asarray(val2, dtype=np.float64))

    H.changeObjectiveSense(highspy.ObjSense.kMinimize)
    H.run()
    info = H.getInfo()
    ms = H.getModelStatus()
    if ms != highspy.HighsModelStatus.kOptimal:
        return {"status": f"non_optimal:{ms!s}", "y_t_minus_y_r_lp_min": None}
    sol = H.getSolution()
    col = np.array(sol.col_value, dtype=np.float64)
    lp_min_y_t_minus_y_r = float(info.objective_function_value) + const_offset
    return {
        "status": "ok",
        "y_t_minus_y_r_lp_min": lp_min_y_t_minus_y_r,
        "rival_lp_ub_on_y_diff": -lp_min_y_t_minus_y_r,
        "xi_star": col[:ng].tolist(),
        "n_unstable": n_unstable,
        "n_stable_active": int(stable_active.size),
        "n_stable_inactive": int(stable_inactive.size),
    }


# ─── ORT strict replay (decoder + check) ─────────────────────────


def decode_xi_to_input(
    xi_star: np.ndarray, snap_c: np.ndarray, snap_Gc: np.ndarray,
) -> np.ndarray:
    """Decode the LP solution xi_star (ξ ∈ [-1,1]^ng) back to an input
    realization. The CIFAR snapshot is at FLATTEN, so reconstructing
    the input layer requires inverting the conv body — which we cannot
    do soundly. Per advisor design lock §1.4 the FAL candidate is
    valid only if ORT replay succeeds on the SAME input that produces
    the snapshot's xi_star realization.

    For the per-neuron triangle LP, the LP's witness xi_star is an
    abstract factor vector, NOT an input. To realize as FAL we need to
    find an input x in the spec box such that the snapshot's
    forward HZ produces a ξ at or near xi_star.

    Strategy (Phase 1): use the existing receipt-side decoder in HyZor's
    `receipt_factor_aware_endcap_lp._extract_xi_root_witness` if
    available. If the snapshot's xi-to-input mapping is not invertible
    (BoxHZ collapse, missing root indices), return None to indicate the
    candidate cannot be soundly replayed (UNKNOWN, not FAL).
    """
    try:
        from receipt_factor_aware_endcap_lp import _extract_xi_root_witness  # noqa
    except Exception:
        return None
    snap = {"c": snap_c, "Gc": snap_Gc, "xi_star": xi_star}
    try:
        return _extract_xi_root_witness(snap, onnx_path="", target=None)  # type: ignore[arg-type]
    except Exception:
        return None


def ort_strict_replay(
    onnx_path: str, vnnlib_path: str,
    x_input: np.ndarray, y_true: int, rivals: List[int],
) -> Dict[str, Any]:
    """Run ORT on x_input + verify input-box + spec under zero tolerance."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    # Flatten to expected shape (CIFAR is (1, 3, 32, 32))
    if hasattr(x_input, "shape") and x_input.size == 3072:
        x_input = x_input.reshape(*in_shape).astype(np.float32)
    else:
        return {"ok": False, "reason": f"shape mismatch x.size={x_input.size}"}
    y = sess.run(None, {in_name: x_input})[0].reshape(-1)
    # spec_zero_tol: any rival y_r >= y_t produces FAL
    spec_holds = any(y[r] >= y[int(y_true)] for r in rivals)
    return {
        "ok": True,
        "spec_zero_tol_holds": bool(spec_holds),
        "y_t": float(y[int(y_true)]),
        "y_argmax": int(np.argmax(y)),
        "y_max_rival": float(max(y[r] for r in rivals)) if rivals else None,
    }


# ─── Driver ──────────────────────────────────────────────────────


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one_iid(
    iid: int, out_root: Path, keep_snapshot: bool = False,
) -> Dict[str, Any]:
    """Phase 1 per-iid run. Captures snapshot, builds clean LP for every
    rival, compares to production endcap LP on the same snapshot."""
    onnx_path, vnn_path = load_instance("cifar100_2024", iid)
    prov = build_provenance("cifar100_2024", iid).as_dict()

    iid_dir = out_root / "per_iid_workdir" / f"iid{iid:03d}"
    snap_dir = iid_dir / "snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    snap_path = capture_flatten_snapshot(iid, snap_dir)
    t_cap = time.perf_counter() - t0

    snap = pickle.load(open(snap_path, "rb"))
    c = np.asarray(snap["c"], dtype=np.float64).reshape(-1)
    Gc = np.asarray(snap["Gc"], dtype=np.float64)

    # ONNX classifier weights + vnnlib spec
    weights = _load_classifier_weights(str(onnx_path))
    W39 = weights["linear1.weight"].astype(np.float64)
    b39 = weights["linear1.bias"].astype(np.float64)
    W41 = weights["linear2.weight"].astype(np.float64)
    b41 = weights["linear2.bias"].astype(np.float64)
    y_true, rivals = _parse_top1_spec(str(vnn_path))

    # Pre-act bounds for diagnostics
    c39, G39 = _build_h39_affine(W39, b39, c, Gc)
    l39, u39 = _h39_bounds(c39, G39)
    n_unstable = int(((l39 < 0) & (u39 > 0)).sum())
    n_active = int((l39 >= 0).sum())
    n_inactive = int((u39 <= 0).sum())

    # Production baseline LP — solve per rival using HyZor's existing LP
    t_prod = time.perf_counter()
    prod_lp_by_rival: Dict[int, float] = {}
    for r in rivals:
        ret = _production_endcap_lp_with_sol(
            W41, b41, c39, G39, l39, u39, int(y_true), int(r),
            time_limit_s=30.0,
        )
        prod_lp_by_rival[int(r)] = (
            ret["lp_min"] if ret.get("status") == "ok" else float("nan")
        )
    t_prod = time.perf_counter() - t_prod

    # Clean per-spec LP — solve per rival from scratch
    t_clean = time.perf_counter()
    clean_by_rival: Dict[int, Dict[str, Any]] = {}
    for r in rivals:
        clean_by_rival[int(r)] = solve_clean_finaltail_lp(
            W39, b39, W41, b41, c, Gc,
            int(y_true), int(r), time_limit_s=30.0,
        )
    t_clean = time.perf_counter() - t_clean

    # Parity check per rival: clean LP min (= y_t-y_r) vs production LP min.
    parity_per_rival = {}
    abs_diff_max = 0.0
    for r in rivals:
        clean_min = clean_by_rival[int(r)].get("y_t_minus_y_r_lp_min")
        prod_min = prod_lp_by_rival.get(int(r))
        if clean_min is None or prod_min is None or np.isnan(prod_min):
            parity_per_rival[int(r)] = {
                "clean": clean_min, "prod": prod_min, "diff": None,
            }
            continue
        d = float(clean_min) - float(prod_min)
        abs_diff_max = max(abs_diff_max, abs(d))
        parity_per_rival[int(r)] = {
            "clean": clean_min, "prod": prod_min, "diff": d,
        }

    # Max-over-rivals LP UB on (y_r - y_t)
    clean_ub_per_rival = {
        int(r): clean_by_rival[int(r)].get("rival_lp_ub_on_y_diff")
        for r in rivals
    }
    valid_ubs = [v for v in clean_ub_per_rival.values() if v is not None]
    max_rival_ub = max(valid_ubs) if valid_ubs else None
    cert = max_rival_ub is not None and max_rival_ub < 0.0
    fal_candidate_rival = None
    if not cert and valid_ubs:
        fal_candidate_rival = int(
            max(clean_ub_per_rival.items(), key=lambda x: x[1] or float("-inf"))[0]
        )

    receipt = {
        "iid": int(iid),
        "y_true_vnnlib": int(y_true),
        "rivals": [int(r) for r in rivals],
        "n_unstable_final_relu": n_unstable,
        "n_active_final_relu": n_active,
        "n_inactive_final_relu": n_inactive,
        "production_lp_y_diff_per_rival": prod_lp_by_rival,
        "clean_lp_y_diff_per_rival": {
            int(r): clean_by_rival[int(r)].get("y_t_minus_y_r_lp_min")
            for r in rivals
        },
        "clean_lp_ub_on_y_r_minus_y_t_per_rival": clean_ub_per_rival,
        "parity_per_rival": parity_per_rival,
        "parity_max_abs_diff": abs_diff_max,
        "max_rival_ub_y_r_minus_y_t": max_rival_ub,
        "hull_verdict": "CERT" if cert else "UNKNOWN_pending_ort",
        "fal_candidate_rival": fal_candidate_rival,
        "snap_capture_wall_s": t_cap,
        "prod_baseline_lp_wall_s": t_prod,
        "clean_lp_wall_s": t_clean,
        "snap_path": str(snap_path),
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnn_path),
        "canonical_root": prov["canonical_root"],
        "instances_csv_sha256": prov["instances_csv_sha256"],
        "onnx_sha256": prov["onnx_sha256"],
        "vnnlib_sha256": prov["vnnlib_sha256"],
    }
    if not keep_snapshot:
        try:
            shutil.rmtree(iid_dir, ignore_errors=True)
        except Exception:
            pass
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iids", type=str, required=True,
                    help="comma-separated CIFAR iids (smoke = 113,29,153)")
    ap.add_argument(
        "--out", type=str, default="",
        help="output dir; defaults to audit_results/cifar_finaltail_hull_phase1_<STAMP>",
    )
    ap.add_argument("--keep-snapshots", action="store_true")
    args = ap.parse_args()

    iids = [int(x) for x in args.iids.split(",") if x.strip()]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = (
        Path(args.out) if args.out else
        ACT_ROOT / "audit_results" / f"cifar_finaltail_hull_phase1_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    per_iid_dir = out_root / "per_iid"
    per_iid_dir.mkdir(exist_ok=True)

    all_receipts: List[Dict[str, Any]] = []
    for iid in iids:
        print(f"--- §7 final-tail hull iid {iid} ---", flush=True)
        try:
            rec = run_one_iid(iid, out_root, keep_snapshot=args.keep_snapshots)
        except Exception as e:
            rec = {
                "iid": int(iid),
                "error": f"{type(e).__name__}: {e}",
            }
        all_receipts.append(rec)
        with open(per_iid_dir / f"iid{iid:03d}.json", "w") as f:
            json.dump(rec, f, indent=2, default=float)
        if "error" in rec:
            print(f"  ERROR: {rec['error']}", flush=True)
        else:
            print(
                f"  unstable={rec['n_unstable_final_relu']} "
                f"max_rival_ub={rec['max_rival_ub_y_r_minus_y_t']:.6f}  "
                f"parity_max_abs={rec['parity_max_abs_diff']:.2e}  "
                f"prod_wall={rec['prod_baseline_lp_wall_s']:.1f}s  "
                f"clean_wall={rec['clean_lp_wall_s']:.1f}s  "
                f"verdict={rec['hull_verdict']}",
                flush=True,
            )
        gc.collect()

    with open(out_root / "smoke_summary.json", "w") as f:
        json.dump({
            "stamp": stamp,
            "iids": iids,
            "n": len(iids),
            "receipts": all_receipts,
        }, f, indent=2, default=float)
    print(f"\nwrote {out_root}/smoke_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
