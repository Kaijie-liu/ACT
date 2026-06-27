#!/usr/bin/env python
"""Exact projected-HZ MILP for one-hidden-layer ReLU MLPs.

This is an experimental solver-side projection for networks of the form

    InputSpec -> Dense -> ReLU -> Dense -> Assert

It keeps the same exact ReLU semantics used by Hybrid Zonotopes, but eliminates
the HZ slack variables before the MILP reaches HiGHS.  No sampling, CROWN,
input split, or triangle relaxation is used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from act.pipeline.hybridz_projected_utils import build_net_and_interval
from act.pipeline.hybridz_spec_utils import (
    check_real_unsafe,
    flatten_query_specs,
    interval_hard_rivals_from_specs,
)
from act.pipeline.hybridz_option_utils import parse_key_value_options


def _as_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _layers_one_relu_mlp(net):
    layers = net.layers
    if len(layers) != 6:
        raise ValueError(f"expected 6 layers, got {len(layers)}")
    if "lb" not in layers[1].params or "ub" not in layers[1].params:
        raise ValueError("layer 1 is not an INPUT_SPEC-like layer")
    if "weight" not in layers[2].params or "bias" not in layers[2].params:
        raise ValueError("layer 2 is not a dense affine layer")
    if "weight" not in layers[4].params or "bias" not in layers[4].params:
        raise ValueError("layer 4 is not a dense affine layer")
    return layers[1], layers[2], layers[3], layers[4], layers[5]


def _add_row(rows, lo, hi, coeffs: Dict[int, float], row_lo: float, row_hi: float) -> None:
    clean = {int(k): float(v) for k, v in coeffs.items() if abs(float(v)) > 1e-15}
    rows.append(clean)
    lo.append(float(row_lo))
    hi.append(float(row_hi))


def _solve_one_margin(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    pre_lb: np.ndarray,
    pre_ub: np.ndarray,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    highs_options: Dict[str, object],
    mip_start: str = "none",
    mip_start_timeout: float = 2.0,
) -> Tuple[str, float | None, np.ndarray | None, dict]:
    try:
        import highspy
    except Exception as exc:
        return f"no_highspy:{exc}", None, None, {}

    Cmat = C.reshape(C.shape[0], -1).astype(np.float64)
    tvec = t.reshape(-1).astype(np.float64)
    if Cmat.shape[0] != 1 or tvec.size != 1:
        raise ValueError(f"projected one-ReLU solver expects one margin row, got {Cmat.shape} and {tvec.shape}")
    c_row = Cmat[0]
    target = float(tvec[0])

    n_in = int(W1.shape[1])
    n_hidden = int(W1.shape[0])
    if W2.shape[1] != n_hidden:
        raise ValueError(f"dense shape mismatch W1={W1.shape} W2={W2.shape}")

    active = pre_lb >= -1e-12
    inactive = pre_ub <= 1e-12
    unstable = ~(active | inactive)
    unstable_idx = np.nonzero(unstable)[0]
    z_pos = {int(i): int(j) for j, i in enumerate(unstable_idx)}

    n_bin = int(unstable_idx.size)
    n_cols = n_in + n_hidden + n_bin
    lb = np.concatenate(
        [
            x_lb.astype(np.float64),
            np.where(inactive, 0.0, np.maximum(pre_lb, 0.0)).astype(np.float64),
            np.zeros(n_bin, dtype=np.float64),
        ]
    )
    ub = np.concatenate(
        [
            x_ub.astype(np.float64),
            np.where(inactive, 0.0, np.maximum(pre_ub, 0.0)).astype(np.float64),
            np.ones(n_bin, dtype=np.float64),
        ]
    )

    cost = np.zeros(n_cols, dtype=np.float64)
    cost[n_in:n_in + n_hidden] = c_row @ W2
    const = float(c_row @ b2 - target)
    obj_thr = -const

    rows: List[Dict[int, float]] = []
    row_lo: List[float] = []
    row_hi: List[float] = []
    for i in range(n_hidden):
        wx = {j: float(W1[i, j]) for j in range(n_in) if abs(float(W1[i, j])) > 1e-15}
        if active[i]:
            # h_i = W_i x + b_i  ->  W_i x - h_i = -b_i
            coeff = dict(wx)
            coeff[n_in + i] = coeff.get(n_in + i, 0.0) - 1.0
            _add_row(rows, row_lo, row_hi, coeff, -float(b1[i]), -float(b1[i]))
        elif inactive[i]:
            # h_i is fixed to zero by bounds.
            continue
        else:
            z_col = n_in + n_hidden + z_pos[int(i)]
            # h_i >= W_i x + b_i  ->  W_i x - h_i <= -b_i
            coeff = dict(wx)
            coeff[n_in + i] = coeff.get(n_in + i, 0.0) - 1.0
            _add_row(rows, row_lo, row_hi, coeff, -1e30, -float(b1[i]))
            # h_i <= U_i z_i
            _add_row(rows, row_lo, row_hi, {n_in + i: 1.0, z_col: -float(pre_ub[i])}, -1e30, 0.0)
            # h_i <= W_i x + b_i - L_i (1-z_i)
            coeff = {j: -float(W1[i, j]) for j in range(n_in) if abs(float(W1[i, j])) > 1e-15}
            coeff[n_in + i] = coeff.get(n_in + i, 0.0) + 1.0
            coeff[z_col] = coeff.get(z_col, 0.0) - float(pre_lb[i])
            _add_row(rows, row_lo, row_hi, coeff, -1e30, float(b1[i] - pre_lb[i]))

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for row in rows:
        for j, v in sorted(row.items()):
            indices.append(int(j))
            data.append(float(v))
        indptr.append(len(indices))

    int_idx = np.arange(n_in + n_hidden, n_cols, dtype=np.int32)

    def _add_model(h, *, integer: bool, model_time_limit: float) -> None:
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(model_time_limit))
        h.setOptionValue("mip_rel_gap", 1e-9)
        h.setOptionValue("objective_target", float(obj_thr))
        h.setOptionValue("objective_bound", float(obj_thr))
        h.setOptionValue("presolve", "on")
        for key, value in highs_options.items():
            h.setOptionValue(str(key), value)
        h.addCols(
            n_cols,
            cost.astype(np.float64),
            lb.astype(np.float64),
            ub.astype(np.float64),
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
        if integer and int_idx.size:
            h.changeColsIntegrality(
                int_idx.size,
                int_idx,
                np.array([highspy.HighsVarType.kInteger] * int_idx.size),
            )
        if rows:
            h.addRows(
                len(rows),
                np.asarray(row_lo, dtype=np.float64),
                np.asarray(row_hi, dtype=np.float64),
                len(data),
                np.asarray(indptr, dtype=np.int32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(data, dtype=np.float64),
            )

    mip_start_status = "off"
    mip_start_indices: Optional[np.ndarray] = None
    mip_start_values: Optional[np.ndarray] = None
    if mip_start != "none":
        try:
            lp = highspy.Highs()
            _add_model(
                lp,
                integer=False,
                model_time_limit=max(0.01, min(float(mip_start_timeout), float(time_limit))),
            )
            lp.run()
            lp_status = lp.modelStatusToString(lp.getModelStatus())
            lp_sol = np.asarray(lp.getSolution().col_value, dtype=np.float64)
            if lp_sol.size == n_cols and np.all(np.isfinite(lp_sol)):
                start = np.clip(lp_sol, lb, ub)
                if int_idx.size:
                    start[int_idx] = (start[int_idx] >= 0.5).astype(np.float64)
                if mip_start == "lp-repair" and int_idx.size:
                    repair = highspy.Highs()
                    _add_model(
                        repair,
                        integer=False,
                        model_time_limit=max(0.01, min(float(mip_start_timeout), float(time_limit))),
                    )
                    fixed = start[int_idx].astype(np.float64)
                    repair.changeColsBounds(
                        int(int_idx.size),
                        int_idx.astype(np.int32),
                        fixed,
                        fixed,
                    )
                    repair.run()
                    repair_status = repair.modelStatusToString(repair.getModelStatus())
                    repair_sol = np.asarray(repair.getSolution().col_value, dtype=np.float64)
                    if repair_sol.size == n_cols and np.all(np.isfinite(repair_sol)):
                        start = np.clip(repair_sol, lb, ub)
                        start[int_idx] = fixed
                        mip_start_indices = np.arange(n_cols, dtype=np.int32)
                        mip_start_values = start.copy()
                        mip_start_status = f"prepared_repair:{lp_status}->{repair_status}"
                    else:
                        mip_start_indices = int_idx.copy()
                        mip_start_values = fixed.copy()
                        mip_start_status = f"prepared_repair_fallback:{lp_status}->{repair_status}"
                elif mip_start == "lp-binary-round":
                    mip_start_indices = int_idx.copy()
                    mip_start_values = start[int_idx].copy()
                else:
                    mip_start_indices = np.arange(n_cols, dtype=np.int32)
                    mip_start_values = start.copy()
                if not mip_start_status.startswith("prepared_repair"):
                    mip_start_status = f"prepared:{lp_status}"
            else:
                mip_start_status = f"skipped:no_lp_solution:{lp_status}"
        except Exception as exc:
            mip_start_status = f"error:{type(exc).__name__}:{str(exc)[:80]}"

    h = highspy.Highs()
    _add_model(h, integer=True, model_time_limit=float(time_limit))
    if mip_start_indices is not None and mip_start_values is not None:
        try:
            ret = h.setSolution(
                int(mip_start_indices.size),
                mip_start_indices.astype(np.int32),
                mip_start_values.astype(np.float64),
            )
            mip_start_status = f"{mip_start_status};set:{ret}"
        except Exception as exc:
            mip_start_status = f"{mip_start_status};set_error:{type(exc).__name__}:{str(exc)[:80]}"
        print(
            f"  highs_mip_start status={mip_start_status} "
            f"entries={int(mip_start_indices.size)} mode={mip_start}",
            flush=True,
        )
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()

    obj = getattr(info, "objective_function_value", None)
    try:
        margin = None if obj is None or not np.isfinite(obj) else const + float(obj)
    except TypeError:
        margin = None
    sol = np.asarray(h.getSolution().col_value, dtype=np.float64)
    x = sol[:n_in].copy() if sol.size >= n_in else None

    def stat(name: str):
        val = getattr(info, name, None)
        try:
            return None if val is None else float(val)
        except (TypeError, ValueError):
            return None

    stats = {
        "status": status,
        "nodes": int(getattr(info, "mip_node_count", 0) or 0),
        "gap": stat("mip_gap"),
        "dual_bound": stat("mip_dual_bound"),
        "obj": stat("objective_function_value"),
        "margin": margin,
        "obj_thr": obj_thr,
        "const": const,
        "n_vars": n_cols,
        "n_bin": n_bin,
        "n_rows": len(rows),
        "nnz": len(data),
        "active": int(active.sum()),
        "inactive": int(inactive.sum()),
        "unstable": int(unstable.sum()),
        "mip_start": mip_start,
        "mip_start_status": mip_start_status,
    }
    return status, margin, x, stats


def _solve_one_margin_scip(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    pre_lb: np.ndarray,
    pre_ub: np.ndarray,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    relu_encoding: str,
    scip_threads: int,
    scip_params: Dict[str, object],
    scip_aggressive: bool,
    scip_cutoff_feas: bool,
    scip_emphasis: str,
) -> Tuple[str, float | None, np.ndarray | None, dict]:
    """Exact one-hidden ReLU feasibility branch using open-source SCIP.

    This solves the same projected ReLU graph as the HiGHS backend, but asks
    SCIP for a primal point satisfying the unsafe cutoff.  It is mainly useful
    as an ADV-search portfolio branch; ORT replay remains only an audit of the
    SCIP-produced exact MILP witness.
    """
    try:
        from pyscipopt import Model, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING, quicksum
    except Exception as exc:
        return f"no_pyscipopt:{exc}", None, None, {}

    Cmat = C.reshape(C.shape[0], -1).astype(np.float64)
    tvec = t.reshape(-1).astype(np.float64)
    if Cmat.shape[0] != 1 or tvec.size != 1:
        raise ValueError(
            f"projected one-ReLU SCIP solver expects one margin row, got {Cmat.shape} and {tvec.shape}"
        )
    c_row = Cmat[0]
    target = float(tvec[0])

    n_in = int(W1.shape[1])
    n_hidden = int(W1.shape[0])
    if W2.shape[1] != n_hidden:
        raise ValueError(f"dense shape mismatch W1={W1.shape} W2={W2.shape}")

    active = pre_lb >= -1e-12
    inactive = pre_ub <= 1e-12
    unstable = ~(active | inactive)

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", float(time_limit))
    m.setParam("numerics/feastol", 1e-7)
    try:
        m.setParam("parallel/maxnthreads", max(1, int(scip_threads)))
    except Exception:
        pass
    if scip_aggressive:
        try:
            m.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
            m.setSeparating(SCIP_PARAMSETTING.AGGRESSIVE)
            m.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        except Exception as exc:
            return f"bad_scip_aggressive:{exc}", None, None, {}
    if scip_emphasis:
        try:
            emph = getattr(SCIP_PARAMEMPHASIS, scip_emphasis.upper())
            m.setEmphasis(emph)
        except Exception as exc:
            return f"bad_scip_emphasis:{scip_emphasis}:{exc}", None, None, {}
    for name, value in scip_params.items():
        try:
            m.setParam(name, value)
        except Exception as exc:
            return f"bad_scip_param:{name}:{exc}", None, None, {}

    x_vars = [
        m.addVar(lb=float(x_lb[i]), ub=float(x_ub[i]), vtype="C", name=f"x{i}")
        for i in range(n_in)
    ]
    h_vars = []
    n_bin = 0
    n_rows = 0
    nnz_est = 0
    for i in range(n_hidden):
        wx_terms = [
            float(W1[i, j]) * x_vars[j]
            for j in range(n_in)
            if abs(float(W1[i, j])) > 1e-15
        ]
        pre_expr = quicksum(wx_terms) + float(b1[i])
        nnz_est += len(wx_terms)
        li, ui = float(pre_lb[i]), float(pre_ub[i])
        if active[i]:
            yi = m.addVar(lb=max(0.0, li), ub=max(ui, li, 0.0), vtype="C", name=f"h{i}")
            m.addCons(yi == pre_expr)
            n_rows += 1
        elif inactive[i]:
            yi = m.addVar(lb=0.0, ub=0.0, vtype="C", name=f"h{i}")
        else:
            yi = m.addVar(lb=0.0, ub=max(ui, 0.0), vtype="C", name=f"h{i}")
            z = m.addVar(vtype="B", name=f"z{i}")
            n_bin += 1
            m.addCons(yi >= pre_expr)
            if relu_encoding == "indicator":
                m.addConsIndicator(-pre_expr <= 0, z)
                m.addConsIndicator(yi - pre_expr <= 0, z)
                m.addConsIndicator(pre_expr <= 0, z, activeone=False)
                m.addConsIndicator(yi <= 0, z, activeone=False)
                n_rows += 5
            else:
                m.addCons(yi <= ui * z)
                m.addCons(yi <= pre_expr - li * (1 - z))
                n_rows += 3
        h_vars.append(yi)

    out_coeff = (c_row @ W2).astype(np.float64)
    const = float(c_row @ b2 - target)
    margin_expr = quicksum(
        float(out_coeff[j]) * h_vars[j]
        for j in range(n_hidden)
        if abs(float(out_coeff[j])) > 1e-15
    ) + const
    if scip_cutoff_feas:
        # Exact CERT can be checked as infeasibility of the unsafe cutoff.
        # This avoids proving the full minimum margin when all we need is
        # emptiness of {margin <= 0}; a feasible point is still an exact ADV
        # candidate and is replayed by the caller when requested.
        m.addCons(margin_expr <= 0.0)
        try:
            m.setParam("limits/solutions", 1)
        except Exception:
            pass
        m.setObjective(quicksum([]), "minimize")
    else:
        # Minimize the exact unsafe margin.  A negative incumbent is already a
        # valid exact MILP witness for ADV; optimality is needed only for CERT.
        # The objective-stop value asks SCIP to stop after a non-boundary witness.
        try:
            m.setParam("limits/objectivestop", -1e-8)
        except Exception:
            pass
        m.setObjective(margin_expr, "minimize")

    t_solve = time.time()
    m.optimize()
    solve_s = time.time() - t_solve
    status = str(m.getStatus())
    x = None
    margin = None
    if m.getNSols() > 0:
        x = np.asarray([m.getVal(v) for v in x_vars], dtype=np.float64)
        h = np.asarray([m.getVal(v) for v in h_vars], dtype=np.float64)
        margin = const + float(out_coeff @ h)

    if x is not None and margin is not None and margin <= 1e-9:
        out_status = "Target"
    elif status == "optimal":
        out_status = "Optimal"
    elif status == "infeasible":
        out_status = "Infeasible"
    else:
        out_status = status

    stats = {
        "status": status,
        "nodes": int(m.getNNodes()),
        "gap": None,
        "dual_bound": None,
        "obj": None,
        "margin": margin,
        "obj_thr": 0.0,
        "const": const,
        "n_vars": int(n_in + n_hidden + n_bin),
        "n_bin": int(n_bin),
        "n_rows": int(n_rows),
        "nnz": int(nnz_est),
        "active": int(active.sum()),
        "inactive": int(inactive.sum()),
        "unstable": int(unstable.sum()),
        "mip_start": "none",
        "mip_start_status": "not_supported",
        "mip_solver": f"scip-{relu_encoding}",
        "scip_threads": int(scip_threads),
        "scip_params": dict(scip_params),
        "scip_aggressive": bool(scip_aggressive),
        "scip_cutoff_feas": bool(scip_cutoff_feas),
        "scip_emphasis": scip_emphasis,
        "solve_s": solve_s,
    }
    return out_status, margin, x, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="safenlp_2024")
    ap.add_argument("--iid", type=int, required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lp-queries", type=int, default=99)
    ap.add_argument("--query-indices", default="")
    ap.add_argument("--milp-timeout", type=float, default=18.0)
    ap.add_argument("--mip-start", choices=["none", "lp-round", "lp-binary-round", "lp-repair"], default="none",
                    help="use rounded LP-relaxation solution only as a HiGHS MIP start")
    ap.add_argument("--mip-start-timeout", type=float, default=2.0)
    ap.add_argument("--mip-solver", choices=["highs", "scip-bigm", "scip-indicator"], default="highs",
                    help="open-source MILP backend for the exact projected one-ReLU model")
    ap.add_argument("--check-witness", action="store_true")
    ap.add_argument("--stop-on-unsafe", action="store_true")
    ap.add_argument("--highs-option", action="append", default=[])
    ap.add_argument("--scip-threads", type=int, default=1,
                    help="SCIP parallel/maxnthreads for the exact projected one-ReLU backend")
    ap.add_argument("--scip-param", action="append", default=[],
                    help="extra SCIP parameter name=value for diagnostics")
    ap.add_argument("--scip-aggressive", action="store_true",
                    help="diagnostic: use SCIP aggressive presolve/separating/heuristics")
    ap.add_argument("--scip-cutoff-feas", action="store_true",
                    help="SCIP exact feasibility form: prove CERT by infeasibility of margin <= 0")
    ap.add_argument("--scip-emphasis", default="",
                    help="SCIP emphasis enum name, e.g. optimality, hardlp, phaseproof")
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    highs_options = parse_key_value_options(args.highs_option, "highs-option")
    scip_params = parse_key_value_options(args.scip_param, "scip-param")
    t0 = time.time()
    onnx_path, vnnlib_path, input_shape, queries, net, before, after, interval_s = build_net_and_interval(
        args.bench, args.iid, args.device
    )
    input_layer, dense1, relu_layer, dense2, assert_layer = _layers_one_relu_mlp(net)
    del input_layer, relu_layer, assert_layer

    W1 = _as_np(dense1.params["weight"])
    b1 = _as_np(dense1.params["bias"]).reshape(-1)
    W2 = _as_np(dense2.params["weight"])
    b2 = _as_np(dense2.params["bias"]).reshape(-1)
    x_lb = _as_np(after[1].bounds.lb).reshape(-1)
    x_ub = _as_np(after[1].bounds.ub).reshape(-1)
    pre_lb = _as_np(after[2].bounds.lb).reshape(-1)
    pre_ub = _as_np(after[2].bounds.ub).reshape(-1)

    flat_specs = flatten_query_specs(queries, W2.shape[0])
    final_bounds = after[4].bounds
    hard, lows = interval_hard_rivals_from_specs(flat_specs, final_bounds)
    if args.query_indices.strip():
        order = np.asarray([int(x) for x in args.query_indices.split(",") if x.strip()], dtype=int)
    else:
        order = np.argsort(np.asarray(lows)) if lows else np.arange(len(flat_specs))
    lp_n = min(int(args.lp_queries), len(order))

    print(
        f"bench={args.bench} iid={args.iid} onnx={onnx_path.name} vnnlib={vnnlib_path.name}",
        flush=True,
    )
    print(
        f"projected_one_relu input={W1.shape[1]} hidden={W1.shape[0]} output={W2.shape[0]} "
        f"queries={len(flat_specs)} interval_hard={hard}/{len(flat_specs)}",
        flush=True,
    )

    cert = hz_unsafe = real_adv = unknown = checked = 0
    query_results = []
    for rank, qidx in enumerate(order[:lp_n]):
        checked += 1
        C, t, kind = flat_specs[int(qidx)]
        ts = time.time()
        if args.mip_solver == "highs":
            status, margin, x, stats = _solve_one_margin(
                W1,
                b1,
                W2,
                b2,
                x_lb,
                x_ub,
                pre_lb,
                pre_ub,
                C,
                t,
                args.milp_timeout,
                highs_options,
                mip_start=args.mip_start,
                mip_start_timeout=args.mip_start_timeout,
            )
        else:
            status, margin, x, stats = _solve_one_margin_scip(
                W1,
                b1,
                W2,
                b2,
                x_lb,
                x_ub,
                pre_lb,
                pre_ub,
                C,
                t,
                args.milp_timeout,
                "indicator" if args.mip_solver == "scip-indicator" else "bigm",
                args.scip_threads,
                scip_params,
                args.scip_aggressive,
                args.scip_cutoff_feas,
                args.scip_emphasis,
            )
        sec = time.time() - ts
        real_bad = False
        cy = None
        if (status.startswith("Target") or (margin is not None and margin <= 1e-9)) and x is not None:
            hz_unsafe += 1
            if args.check_witness:
                real_bad, cy = check_real_unsafe(
                    onnx_path,
                    input_shape,
                    x.reshape(input_shape),
                    C.reshape(C.shape[0], -1),
                    t.reshape(-1),
                )
                if real_bad:
                    real_adv += 1
        elif status.startswith("Infeasible") or (
            status.startswith("Optimal") and margin is not None and margin > 1e-9
        ):
            cert += 1
        else:
            unknown += 1
        verdict = "unknown"
        if real_bad:
            verdict = "adv"
        elif status.startswith("Target") or (margin is not None and margin <= 1e-9):
            verdict = "hz_unsafe"
        elif status.startswith("Infeasible") or (
            status.startswith("Optimal") and margin is not None and margin > 1e-9
        ):
            verdict = "cert"
        rec = {
            "rank": int(rank),
            "q": int(qidx),
            "kind": kind,
            "status": status,
            "margin": None if margin is None else float(margin),
            "sec": sec,
            "verdict": verdict,
            "witness_checked": bool(args.check_witness and x is not None),
            "real_unsafe": bool(real_bad),
            "cy": None if cy is None else np.asarray(cy).reshape(-1).astype(float).tolist(),
            "milp_stats": stats,
        }
        query_results.append(rec)
        print(
            f"q={int(qidx)} status={status} margin={margin} verdict={verdict} "
            f"real={real_bad} sec={sec:.2f} nodes={stats.get('nodes')}",
            flush=True,
        )
        if args.stop_on_unsafe and real_bad:
            break

    instance_status = "unknown"
    if real_adv:
        instance_status = "adv"
    elif hz_unsafe:
        instance_status = "hz_unsafe"
    elif checked == len(flat_specs) and cert == len(flat_specs):
        instance_status = "verified"

    summary = {
        "bench": args.bench,
        "iid": args.iid,
        "onnx": onnx_path.name,
        "vnnlib": vnnlib_path.name,
        "queries": len(flat_specs),
        "checked": checked,
        "cert": cert,
        "hz_unsafe": hz_unsafe,
        "real_adv": real_adv,
        "unknown": unknown,
        "instance_status": instance_status,
        "interval_hard": hard,
        "low_margin_min": None if not lows else float(min(lows)),
        "highs_options": args.highs_option,
        "mip_solver": args.mip_solver,
        "mip_start": args.mip_start,
        "mip_start_timeout": args.mip_start_timeout,
        "projected_solver": "one_hidden_relu_exact_hz_projection",
        "rule": "Exact ReLU MILP projection of the forward Hybrid-Zonotope semantics; no triangle/CROWN/sampling/input split.",
        "query_results": query_results,
        "wall_s": time.time() - t0,
    }
    verdict = "UNKNOWN"
    if real_adv:
        verdict = "ADV"
    elif instance_status == "verified":
        verdict = "CERT"
    summary["verdict"] = verdict
    if query_results:
        first_stats = query_results[0].get("milp_stats", {})
        summary["nb"] = first_stats.get("n_bin")
        summary["ng"] = first_stats.get("n_vars")
        summary["nc"] = first_stats.get("n_rows")
    summary["time_s"] = summary["wall_s"]
    summary["verify_s"] = sum(float(q.get("sec", 0.0)) for q in query_results)
    summary["p0"] = False
    summary["err"] = None
    if args.summary_json:
        p = Path(args.summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"verdict_summary checked={checked} cert={cert} hz_unsafe={hz_unsafe} "
        f"real_adv={real_adv} unknown={unknown} instance={instance_status} "
        f"wall={summary['wall_s']:.2f}",
        flush=True,
    )
    print(json.dumps({
        "bench": args.bench,
        "iid": args.iid,
        "verdict": verdict,
        "margin": query_results[0].get("margin") if query_results else None,
        "n_queries": len(flat_specs),
        "nc": summary.get("nc"),
        "ng": summary.get("ng"),
        "nb": summary.get("nb"),
        "time_s": summary["time_s"],
        "verify_s": summary["verify_s"],
        "device": args.device,
        "gt_cex": bool(real_adv),
        "p0": False,
        "err": None,
        "projected_one_relu": True,
        "engine_adv": bool(real_adv),
    }), flush=True)


if __name__ == "__main__":
    main()
