#!/usr/bin/env python3
"""FAL-only output-halfspace end-cap feasibility LP.

This pilot generalizes the generic MLP end-cap witness from top-1
``Y_r >= Y_true`` constraints to arbitrary VNNLIB unsafe disjuncts made
of output halfspaces:

    Y_i >= c, Y_i <= c, Y_i >= Y_j, Y_i <= Y_j

Scope:
  - snapshot at a FLATTEN layer
  - single affine tail: FLATTEN -> Gemm/MatMul -> outputs
  - continuous LP only (HiGHS LP API), no MILP, no branching, no gradients
  - FAL only: every candidate is replayed through raw ONNX at zero tolerance

The LP maximizes a common slack rho for one unsafe disjunct:

    A xi + rho <= b,  -1 <= xi <= 1

If rho >= threshold, the root part of xi is converted back to an input
box point and strict ORT replay decides whether the candidate is a real
counterexample. No CERT is claimed by this script.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


HYZOR_DIR = Path("/data1/Kane/HyZor")
if str(HYZOR_DIR) not in sys.path:
    sys.path.insert(0, str(HYZOR_DIR))

from receipt_factor_aware_endcap_lp import _parse_vnnlib_full  # noqa: E402
from generic_mlp_endcap_reuse import (  # noqa: E402
    _extract_tail_dense_layers,
    _strict_replay,
)


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _rows_for_disjunct(
    disjunct: Iterable[Dict[str, Any]],
    y0: np.ndarray,
    YG: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``A, b`` for unsafe constraints in ``A xi <= b`` form."""
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for c in disjunct:
        kind = c.get("kind")
        if kind == "Yj_ge":
            j = int(c["j"])
            # y_j >= c  ->  -YG_j xi <= y0_j - c
            rows.append(-YG[j, :])
            rhs.append(float(y0[j]) - float(c["c"]))
        elif kind == "Yj_le":
            j = int(c["j"])
            # y_j <= c  ->  YG_j xi <= c - y0_j
            rows.append(YG[j, :])
            rhs.append(float(c["c"]) - float(y0[j]))
        elif kind == "YjYt":
            j = int(c["j"])
            t = int(c["t"])
            # y_j >= y_t -> -(YG_j - YG_t) xi <= y0_j - y0_t
            rows.append(-(YG[j, :] - YG[t, :]))
            rhs.append(float(y0[j]) - float(y0[t]))
        elif kind == "YjYt_le":
            j = int(c["j"])
            t = int(c["t"])
            # y_j <= y_t -> (YG_j - YG_t) xi <= y0_t - y0_j
            rows.append(YG[j, :] - YG[t, :])
            rhs.append(float(y0[t]) - float(y0[j]))
        else:
            raise RuntimeError(f"unsupported output halfspace kind={kind!r}")
    if not rows:
        raise RuntimeError("empty output disjunct")
    return np.vstack(rows).astype(np.float64), np.asarray(rhs, dtype=np.float64)


def _solve_max_common_slack(
    A: np.ndarray,
    b: np.ndarray,
    *,
    time_limit_s: float,
) -> Dict[str, Any]:
    """Maximize rho subject to A xi + rho <= b and xi in [-1,1]."""
    import highspy

    n_xi = int(A.shape[1])
    rho_idx = n_xi
    n_vars = n_xi + 1
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit_s))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    cost = np.zeros(n_vars, dtype=np.float64)
    cost[rho_idx] = -1.0
    lower = np.concatenate([
        -np.ones(n_xi, dtype=np.float64),
        np.asarray([-highspy.kHighsInf], dtype=np.float64),
    ])
    upper = np.concatenate([
        np.ones(n_xi, dtype=np.float64),
        np.asarray([highspy.kHighsInf], dtype=np.float64),
    ])
    h.addCols(
        n_vars,
        cost,
        lower,
        upper,
        0,
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float64),
    )
    INF = highspy.kHighsInf
    for i in range(A.shape[0]):
        row = A[i, :]
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        idx = np.concatenate([nz.astype(np.int32), np.asarray([rho_idx], dtype=np.int32)])
        val = np.concatenate([row[nz].astype(np.float64), np.asarray([1.0])])
        h.addRow(-INF, float(b[i]), int(idx.size), idx, val)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        return {"status": f"model_status:{status!s}", "rho": None}
    sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
    return {
        "status": "ok",
        "rho": float(sol[rho_idx]),
        "xi": sol[:n_xi],
        "objective": float(h.getObjectiveValue()),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)
    c = _as_numpy(snap["c"]).reshape(-1)
    Gc_full = _as_numpy(snap["Gc"])
    nb = int(snap.get("nb", 0))
    nc = int(snap.get("nc", 0))
    root_ng = int(snap.get("root_ng", 0))
    lb_x, ub_x, disjuncts = _parse_vnnlib_full(args.vnnlib)
    input_dim = int(lb_x.size)
    if nb != 0 or nc != 0:
        raise RuntimeError(f"snapshot must be pure HZono, got nb={nb} nc={nc}")
    if root_ng != input_dim:
        raise RuntimeError(
            f"root_ng={root_ng} != input_dim={input_dim}; cannot replay xi_root"
        )
    if root_ng > int(Gc_full.shape[1]):
        raise RuntimeError(
            f"root_ng={root_ng} > snapshot ng={Gc_full.shape[1]}; root factors compressed"
        )

    tail = _extract_tail_dense_layers(args.onnx)
    if len(tail) != 1:
        raise RuntimeError(
            f"output-halfspace feasibility currently supports single Dense tails only; got {len(tail)}"
        )
    W, b_tail = tail[0]
    if int(W.shape[1]) != int(c.size):
        raise RuntimeError(
            f"tail input dim {W.shape[1]} != snapshot dim {c.size}"
        )
    if args.mode == "root":
        Gc = Gc_full[:, :root_ng]
    elif args.mode == "full":
        Gc = Gc_full
    else:
        raise RuntimeError(f"unsupported mode={args.mode!r}")
    y0 = W @ c + b_tail
    YG = W @ Gc

    per_disjunct: Dict[int, Dict[str, Any]] = {}
    fal_receipt = None
    max_rho = None
    best_disjunct = None
    limit = len(disjuncts) if args.max_disjuncts <= 0 else min(
        len(disjuncts), int(args.max_disjuncts)
    )
    for d_idx, disjunct in enumerate(disjuncts[:limit]):
        A, b = _rows_for_disjunct(disjunct, y0, YG)
        res = _solve_max_common_slack(A, b, time_limit_s=args.time_limit_s)
        entry: Dict[str, Any] = {
            "status": res["status"],
            "n_constraints": int(A.shape[0]),
            "rho": res.get("rho"),
        }
        if res["status"] == "ok":
            rho = float(res["rho"])
            if max_rho is None or rho > max_rho:
                max_rho = rho
                best_disjunct = int(d_idx)
            if rho >= float(args.replay_rho_threshold):
                xi = np.asarray(res["xi"], dtype=np.float64)
                xi_root = xi[:root_ng] if args.mode == "full" else xi
                replay = _strict_replay(
                    args.onnx, args.vnnlib, lb_x, ub_x, disjuncts, xi_root
                )
                entry["replay"] = replay
                if replay["all_checks_pass"] and fal_receipt is None:
                    fal_receipt = {
                        "disjunct": int(d_idx),
                        "rho": rho,
                        "mode": args.mode,
                        **replay,
                    }
        per_disjunct[int(d_idx)] = entry
        if fal_receipt is not None and not args.keep_going:
            break
    verdict = "FAL" if fal_receipt else "UNKNOWN"
    return {
        "source": "output_halfspace_endcap_feas",
        "snapshot": args.snapshot,
        "onnx": args.onnx,
        "vnnlib": args.vnnlib,
        "mode": args.mode,
        "snapshot_shape": {
            "dim": int(c.size),
            "ng": int(Gc_full.shape[1]),
            "nb": nb,
            "nc": nc,
            "root_ng": root_ng,
        },
        "tail": {
            "n_dense_layers": 1,
            "W": list(W.shape),
        },
        "n_disjuncts": int(len(disjuncts)),
        "n_checked": int(len(per_disjunct)),
        "max_rho": max_rho,
        "best_disjunct": best_disjunct,
        "verdict": verdict,
        "fal_receipt": fal_receipt,
        "per_disjunct": per_disjunct,
        "wall_s": float(time.time() - t0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vnnlib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("root", "full"), default="root")
    ap.add_argument("--time-limit-s", type=float, default=10.0)
    ap.add_argument("--max-disjuncts", type=int, default=0)
    ap.add_argument("--replay-rho-threshold", type=float, default=0.0)
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()
    out = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(
        f"verdict={out['verdict']} mode={out['mode']} "
        f"max_rho={out['max_rho']} checked={out['n_checked']}/"
        f"{out['n_disjuncts']} wall={out['wall_s']:.2f}s out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
