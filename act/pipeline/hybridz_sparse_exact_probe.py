#!/usr/bin/env python
"""Sparse numeric exact-HZ probe used by the packaged HybridZ portfolio."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.sparse_hz import SparseHZono as SparseHZ  # noqa: E402
from act.back_end.solver.solver_hz_verdict import (  # noqa: E402
    _milp_cutoff_highs,
    _milp_cutoff_scip,
    hz_base_feasibility as _solver_hz_base_feasibility,
    sparse_highs_relaxation_empty_precheck as _highs_relaxation_empty_precheck,
    sparse_fbbt_tighten_bounds as _fbbt_tighten_bounds,
    sparse_lp_min_margin as _lp_min_margin,
    sparse_row_bound_infeasible as _row_bound_infeasible,
    sparse_solver_start_from_xi as _solver_start_from_xi,
)
from act.pipeline.hybridz_option_utils import parse_key_value_options  # noqa: E402
from act.pipeline.hybridz_spec_utils import (  # noqa: E402
    check_real_unsafe as _check_real_unsafe,
    flatten_query_specs as _flatten_query_specs,
    input_center_radius_indices as _input_center_rad,
    interval_hard_rivals_from_specs as _interval_hard_rivals_from_specs,
)
import act.back_end.hybridz_tf.tf_mlp as hz_mlp  # noqa: E402
from act.back_end.hybridz_tf.sparse_ops import (  # noqa: E402
    _scurve_domain_cut_matrices,
    _scurve_graph_cut_matrices,
    _scurve_range_cut_matrices,
    _sigmoid_deriv_np,
    _sigmoid_np,
    _tanh_deriv_np,
    _tanh_np,
    sparse_avgpool2d_matrix_from_layer as _avgpool2d_matrix,
    sparse_conv2d_matrix_from_layer as _conv2d_matrix,
    sparse_convtranspose2d_matrix_from_layer as _convtranspose2d_matrix,
    sparse_dense_matrix_from_layer as _dense_matrix,
    sparse_empty as _empty,
    sparse_hz_add_const as _bias_apply,
    sparse_hz_add_same_frame as _add_same_frame,
    sparse_hz_concat,
    sparse_hz_gather_rows as _gather_hz_rows,
    sparse_hz_gather_rows_like as _gather_hz_rows_like,
    sparse_hz_apply_relu_exact as _backend_relu_exact,
    sparse_hz_apply_scurve_piecewise as _backend_scurve_piecewise,
    sparse_hz_apply_softmax_simplex_layer as _softmax_simplex_hz,
    sparse_hz_apply_matmul_product_interval_layer as _matmul_product_interval_hz,
    sparse_hz_from_bounds,
    sparse_hz_linear as _linear_apply,
    sparse_hz_pad_frame as _pad_hz,
    sparse_hz_scale as _scale_apply,
    sparse_hz_sub_same_frame as _sub_same_frame,
    sparse_pad_cols as _pad_cols,
    sparse_maxpool2d_candidate_rows as _maxpool2d_candidate_rows,
    sparse_hz_tighten_relu_bounds as _tighten_relu_bounds,
)


REPO = Path(__file__).resolve().parents[2]

SPARSE_SUPPORTED_KINDS = {
    "INPUT",
    "INPUT_SPEC",
    "CONV2D",
    "CONVTRANSPOSE2D",
    "AVGPOOL2D",
    "DENSE",
    "UPSAMPLE",
    "MAXPOOL2D",
    "SCALE",
    "BIAS",
    "FLATTEN",
    "RESHAPE",
    "SQUEEZE",
    "UNSQUEEZE",
    "TRANSPOSE",
    "BN",
    "ASSERT",
    "SLICE",
    "GATHER",
    "CONCAT",
    "ADD",
    "SUB",
    "RELU",
    "SIGMOID",
    "TANH",
    "MATMUL",
    "SOFTMAX",
}

from act.pipeline.hybridz_sparse_census import (  # noqa: E402
    _build_net_and_interval,
    _format_big,
)


def _slice_row_idx_np(L, n: int) -> Optional[np.ndarray]:
    rows = hz_mlp._slice_row_idx(L, int(n))
    return None if rows is None else rows.detach().cpu().numpy().astype(np.int64).reshape(-1)


def _gather_row_idx_np(L, n: int) -> Optional[np.ndarray]:
    rows = hz_mlp._gather_row_idx(L, int(n))
    return None if rows is None else rows.detach().cpu().numpy().astype(np.int64).reshape(-1)


def _relu_exact(
    hz: SparseHZ,
    pre_bounds,
    *,
    tight_lp_limit: int = 0,
    tight_lp_timeout: float = 0.0,
    tight_fix_mode: str = "both",
    add_cuts: bool = False,
    compressed: bool = False,
) -> Tuple[SparseHZ, Tuple[int, int, int], Tuple[int, int, int, int]]:
    base_lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    base_ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if tight_lp_limit != 0:
        lb, ub, tight_stats = _tighten_relu_bounds(
            hz, pre_bounds, limit=tight_lp_limit, time_limit=tight_lp_timeout
        )
        if tight_fix_mode == "off-only":
            was_unstable = (base_lb < 0.0) & (base_ub > 0.0)
            active_by_tight = was_unstable & (lb >= 0.0) & (ub > 0.0)
            if np.any(active_by_tight):
                # Active ReLUs copy the full sparse preactivation row forward.
                # For sparse exact-HZ this can densify later affine layers; keep
                # those phases exact-unstable unless the interval pass already
                # proved them active. Inactive fixes remain sparsity-reducing.
                lb[active_by_tight] = base_lb[active_by_tight]
    else:
        lb = base_lb
        ub = base_ub
        tight_stats = (0, 0, 0, 0)
    class ArrayBounds:
        def __init__(self, lb_arr: np.ndarray, ub_arr: np.ndarray):
            import torch

            self.lb = torch.as_tensor(lb_arr, dtype=torch.float64)
            self.ub = torch.as_tensor(ub_arr, dtype=torch.float64)

    out, counts, meta = _backend_relu_exact(
        hz,
        ArrayBounds(lb, ub),
        compressed=compressed,
        valid_cuts=add_cuts,
        return_info=True,
    )
    _relu_exact.last_meta = meta
    return out, counts, tight_stats


def _scurve_breakpoints(
    lo: float,
    hi: float,
    K: int,
    *,
    grid: str,
    func,
    dfunc,
) -> np.ndarray:
    """Segment breakpoints for one convex/concave side of an S-curve.

    ``uniform`` preserves the original equal-width grid. ``curvature`` keeps the
    exact same number of segments but puts more cuts where |f''| is larger.  The
    downstream HZ encoding remains the same sound per-segment overapproximation;
    only the chosen segment endpoints change.
    """
    K = max(1, int(K))
    lo = float(lo)
    hi = float(hi)
    if K == 1 or hi - lo <= 1e-12 or grid == "uniform":
        return np.linspace(lo, hi, K + 1, dtype=np.float64)
    if grid != "curvature":
        raise ValueError(f"unknown S-curve grid {grid!r}")

    xs = np.linspace(lo, hi, max(33, 32 * K + 1), dtype=np.float64)
    h = 1e-4 * (1.0 + np.abs(xs))
    with np.errstate(all="ignore"):
        curv = np.abs((dfunc(xs + h) - dfunc(xs - h)) / (2.0 * h))
    curv = np.where(np.isfinite(curv), curv, 0.0)
    dens = np.sqrt(np.maximum(curv, 0.0)) + 1e-9
    dx = np.diff(xs)
    area = 0.5 * (dens[:-1] + dens[1:]) * dx
    cum = np.concatenate([[0.0], np.cumsum(area)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 1e-12:
        return np.linspace(lo, hi, K + 1, dtype=np.float64)
    cuts = np.interp(np.linspace(0.0, total, K + 1), cum, xs).astype(np.float64)
    cuts[0] = lo
    cuts[-1] = hi
    # Guard against tiny interpolation inversions from flat curvature tails.
    for i in range(1, cuts.size):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]
    return cuts


def _sigmoid_piecewise(
    hz: SparseHZ,
    pre_bounds,
    *,
    K: int = 2,
    compressed: bool = False,
    drop_degenerate: bool = False,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
    func=_sigmoid_np,
    dfunc=_sigmoid_deriv_np,
) -> Tuple[SparseHZ, Tuple[int, int]]:
    """Sparse copy of dense S-shape K-per-side parallelogram encoding."""
    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin
    lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    wide = (ub - lb) > 1e-12
    wide_idx = np.nonzero(wide)[0].astype(np.int32)
    narrow_idx = np.nonzero(~wide)[0].astype(np.int32)
    m = int(wide_idx.size)
    _sigmoid_piecewise.last_meta = {"wide_idx": wide_idx, "m": m, "compressed": compressed}

    out_c = np.zeros(n, dtype=np.float64)
    if narrow_idx.size:
        out_c[narrow_idx] = func(hz.c[narrow_idx])
    if m == 0:
        return SparseHZ(
            out_c,
            _empty(n, ng),
            _empty(n, nb),
            hz.Ac,
            hz.Ab,
            hz.b,
            hz.Auc,
            hz.Aub,
            hz.ub,
        ), (0, int(narrow_idx.size))
    if drop_degenerate:
        if not compressed:
            raise ValueError("drop_degenerate sigmoid is currently implemented for compressed mode only")
        _sigmoid_piecewise.last_meta = {"wide_idx": wide_idx, "m": m, "compressed": compressed, "pruned": True}
        if grid == "uniform":
            out, counts, meta = _backend_scurve_piecewise(
                hz,
                pre_bounds,
                K=K,
                func=func,
                dfunc=dfunc,
                inflection=0.0,
                domain_cuts=domain_cuts,
                graph_cuts=graph_cuts,
                return_info=True,
            )
            _sigmoid_piecewise.last_meta = meta
            return out, counts
        return _sigmoid_piecewise_pruned(
            hz,
            pre_bounds,
            K=K,
            domain_cuts=domain_cuts,
            graph_cuts=graph_cuts,
            grid=grid,
            func=func,
            dfunc=dfunc,
        )

    K_side = max(1, int(K))
    lb_w = lb[wide_idx]
    ub_w = ub[wide_idx]
    p = np.maximum(np.minimum(0.0, ub_w), lb_w)
    if grid == "uniform":
        sid = np.arange(K_side, dtype=np.float64).reshape(-1, 1)
        wL = (p - lb_w).reshape(1, -1) / K_side
        wR = (ub_w - p).reshape(1, -1) / K_side
        aL = lb_w.reshape(1, -1) + sid * wL
        aR = p.reshape(1, -1) + sid * wR
        a = np.vstack([aL, aR])
        b_seg = np.vstack([aL + wL, aR + wR])
    else:
        a = np.zeros((2 * K_side, m), dtype=np.float64)
        b_seg = np.zeros((2 * K_side, m), dtype=np.float64)
        for j in range(m):
            br_l = _scurve_breakpoints(lb_w[j], p[j], K_side, grid=grid, func=func, dfunc=dfunc)
            br_r = _scurve_breakpoints(p[j], ub_w[j], K_side, grid=grid, func=func, dfunc=dfunc)
            a[:K_side, j] = br_l[:-1]
            b_seg[:K_side, j] = br_l[1:]
            a[K_side:, j] = br_r[:-1]
            b_seg[K_side:, j] = br_r[1:]
    S = int(2 * K_side)

    fa = func(a)
    fb = func(b_seg)
    la = dfunc(a)
    lb_slope = dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = np.abs(la - lb_slope) < 1e-10

    denom = lb_slope - la
    safe_denom = np.where(nearly_linear, 1.0, denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = np.linspace(0.0, 1.0, 50, dtype=np.float64).reshape(50, 1, 1)
    pts = a.reshape(1, S, m) + t_pts * (b_seg - a).reshape(1, S, m)
    f_pts = func(pts)
    resid = f_pts - (slope.reshape(1, S, m) * pts + (fa - slope * a).reshape(1, S, m))
    max_err = np.max(np.abs(resid), axis=0)
    g1x_lin = hw
    g1y_lin = slope * hw
    g2x_lin = np.zeros_like(hw)
    g2y_lin = max_err

    g1_x = np.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = np.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = np.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = np.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.reshape(1, S, m)
    dy = f_pts - centers_y.reshape(1, S, m)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = np.where(np.abs(det) < 1e-30, 1.0, det)
    xi1 = (dy * g2_x.reshape(1, S, m) - dx * g2_y.reshape(1, S, m)) / safe_det.reshape(1, S, m)
    xi2 = (dy * g1_x.reshape(1, S, m) - dx * g1_y.reshape(1, S, m)) / (-safe_det.reshape(1, S, m))
    max_xi = np.maximum(np.max(np.abs(xi1), axis=0), np.max(np.abs(xi2), axis=0))
    scale_factor = np.where(max_xi > 1.0, max_xi * 1.01, 1.0)
    scale_factor = np.where(np.abs(det) < 1e-30, 1.0, scale_factor)
    g1_x *= scale_factor
    g1_y *= scale_factor
    g2_x *= scale_factor
    g2_y *= scale_factor

    _sigmoid_piecewise.last_meta = {
        "wide_idx": wide_idx,
        "m": m,
        "S": S,
        "a": a,
        "b_seg": b_seg,
        "centers_x": centers_x,
        "centers_y": centers_y,
        "g1_x": g1_x,
        "g1_y": g1_y,
        "g2_x": g2_x,
        "g2_y": g2_y,
        "compressed": compressed,
        "grid": grid,
    }

    out_c[wide_idx] = np.sum(centers_y, axis=0) / 2.0
    seg = np.arange(S * m, dtype=np.int32).reshape(S, m)
    flat_seg = seg.reshape(-1)
    wide_rows = np.broadcast_to(wide_idx.reshape(1, -1), (S, m)).reshape(-1)
    n_real = 2 * S * m
    n_slack = 0 if compressed else 4 * S * m
    ng_total = ng + n_real + n_slack
    nb_total = nb + S * m

    g1_cols = ng + flat_seg
    g2_cols = ng + S * m + flat_seg
    z_cols = nb + flat_seg
    owner_arr = np.broadcast_to(np.arange(m, dtype=np.int32).reshape(1, -1), (S, m)).reshape(-1)
    out_Gc = sp.coo_matrix(
        (
            np.concatenate([g1_y.reshape(-1), g2_y.reshape(-1)]),
            (
                np.concatenate([wide_rows, wide_rows]),
                np.concatenate([g1_cols, g2_cols]),
            ),
        ),
        shape=(n, ng_total),
    ).tocsr()
    out_Gb = sp.coo_matrix(
        (-centers_y.reshape(-1) / 2.0, (wide_rows, z_cols)),
        shape=(n, nb_total),
    ).tocsr()
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    n_box = 0 if compressed else 4 * S * m
    n_eq_new = n_box + m + m
    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []

    def add_c(rows, cols, data):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size:
            rr_c.append(np.asarray(rows, dtype=np.int32).reshape(-1))
            cc_c.append(np.asarray(cols, dtype=np.int32).reshape(-1))
            dd_c.append(arr)

    def add_b(rows, cols, data):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size:
            rr_b.append(np.asarray(rows, dtype=np.int32).reshape(-1))
            cc_b.append(np.asarray(cols, dtype=np.int32).reshape(-1))
            dd_b.append(arr)

    flat_rows = np.zeros(0, dtype=np.int32)
    if not compressed:
        row_grid = 4 * seg
        flat_rows = row_grid.reshape(-1)
        slack_base = ng + n_real + 4 * flat_seg
        add_c(flat_rows, g1_cols, np.ones(S * m))
        add_c(flat_rows, slack_base, np.ones(S * m))
        add_b(flat_rows, z_cols, -0.5 * np.ones(S * m))
        add_c(flat_rows + 1, g1_cols, -np.ones(S * m))
        add_c(flat_rows + 1, slack_base + 1, np.ones(S * m))
        add_b(flat_rows + 1, z_cols, -0.5 * np.ones(S * m))
        add_c(flat_rows + 2, g2_cols, np.ones(S * m))
        add_c(flat_rows + 2, slack_base + 2, np.ones(S * m))
        add_b(flat_rows + 2, z_cols, -0.5 * np.ones(S * m))
        add_c(flat_rows + 3, g2_cols, -np.ones(S * m))
        add_c(flat_rows + 3, slack_base + 3, np.ones(S * m))
        add_b(flat_rows + 3, z_cols, -0.5 * np.ones(S * m))

    link_rows = n_box + np.arange(m, dtype=np.int32)
    link_grid = np.broadcast_to(link_rows.reshape(1, -1), (S, m)).reshape(-1)
    add_c(link_grid, g1_cols, -g1_x.reshape(-1))
    add_c(link_grid, g2_cols, -g2_x.reshape(-1))
    add_b(link_grid, z_cols, centers_x.reshape(-1) / 2.0)
    pre_gc = hz.Gc[wide_idx].tocoo()
    if pre_gc.nnz:
        add_c(link_rows[pre_gc.row], pre_gc.col, pre_gc.data)
    if nb:
        pre_gb = hz.Gb[wide_idx].tocoo()
        if pre_gb.nnz:
            add_b(link_rows[pre_gb.row], pre_gb.col, pre_gb.data)

    sum_rows = n_box + m + np.arange(m, dtype=np.int32)
    sum_grid = np.broadcast_to(sum_rows.reshape(1, -1), (S, m)).reshape(-1)
    add_b(sum_grid, z_cols, np.ones(S * m))

    eq_b = np.zeros(n_eq_new, dtype=np.float64)
    if not compressed:
        eq_b[flat_rows] = 0.5
        eq_b[flat_rows + 1] = 0.5
        eq_b[flat_rows + 2] = 0.5
        eq_b[flat_rows + 3] = 0.5
    eq_b[link_rows] = np.sum(centers_x, axis=0) / 2.0 - hz.c[wide_idx]
    eq_b[sum_rows] = float(S - 2)

    eq_Ac = sp.coo_matrix(
        (np.concatenate(dd_c), (np.concatenate(rr_c), np.concatenate(cc_c))),
        shape=(n_eq_new, ng_total),
    ).tocsr()
    eq_Ab = sp.coo_matrix(
        (np.concatenate(dd_b), (np.concatenate(rr_b), np.concatenate(cc_b))),
        shape=(n_eq_new, nb_total),
    ).tocsr()
    old_Ac = _pad_cols(hz.Ac, ng_total)
    old_Ab = _pad_cols(hz.Ab, nb_total)
    Ac = sp.vstack([old_Ac, eq_Ac], format="csr")
    Ab = sp.vstack([old_Ab, eq_Ab], format="csr")
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()
    Auc_base = _pad_cols(hz.Auc if hz.Auc is not None else _empty(0, hz.n_cont), ng_total)
    Aub_base = _pad_cols(hz.Aub if hz.Aub is not None else _empty(0, hz.n_bin), nb_total)
    ub_base = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    if compressed:
        # Exact projection of the per-segment slack boxes:
        #   g + s - 0.5 z = 0.5, -g + s' - 0.5 z = 0.5, s,s' in [-1,1]
        # is equivalent to g + 0.5 z <= 0.5 and -g + 0.5 z <= 0.5.
        box_rows = 4 * np.arange(S * m, dtype=np.int32)
        rr_c = np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3])
        cc_c = np.concatenate([g1_cols, g1_cols, g2_cols, g2_cols])
        dd_c = np.concatenate([
            np.ones(S * m),
            -np.ones(S * m),
            np.ones(S * m),
            -np.ones(S * m),
        ])
        rr_b = rr_c.copy()
        cc_b = np.concatenate([z_cols, z_cols, z_cols, z_cols])
        dd_b = 0.5 * np.ones(4 * S * m, dtype=np.float64)
        box_Auc = sp.coo_matrix((dd_c, (rr_c, cc_c)), shape=(4 * S * m, ng_total)).tocsr()
        box_Aub = sp.coo_matrix((dd_b, (rr_b, cc_b)), shape=(4 * S * m, nb_total)).tocsr()
        Auc = sp.vstack([Auc_base, box_Auc], format="csr")
        Aub = sp.vstack([Aub_base, box_Aub], format="csr")
        ub_rhs = np.concatenate([ub_base, 0.5 * np.ones(4 * S * m, dtype=np.float64)])
    else:
        Auc = Auc_base
        Aub = Aub_base
        ub_rhs = ub_base
    if domain_cuts:
        dom_Auc, dom_Aub, dom_rhs = _scurve_domain_cut_matrices(
            hz,
            wide_idx,
            a.reshape(-1),
            b_seg.reshape(-1),
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            lb_w,
            ub_w,
        )
        rng_Auc, rng_Aub, rng_rhs = _scurve_range_cut_matrices(
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            fa.reshape(-1),
            fb.reshape(-1),
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            func(lb_w),
            func(ub_w),
        )
        Auc = sp.vstack([Auc, dom_Auc], format="csr")
        Aub = sp.vstack([Aub, dom_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, dom_rhs])
        Auc = sp.vstack([Auc, rng_Auc], format="csr")
        Aub = sp.vstack([Aub, rng_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, rng_rhs])
    if graph_cuts:
        graph_Auc, graph_Aub, graph_rhs = _scurve_graph_cut_matrices(
            hz,
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            a.reshape(-1),
            b_seg.reshape(-1),
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            lb_w,
            ub_w,
            func=func,
            dfunc=dfunc,
        )
        Auc = sp.vstack([Auc, graph_Auc], format="csr")
        Aub = sp.vstack([Aub, graph_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, graph_rhs])
    return SparseHZ(
        out_c,
        out_Gc,
        out_Gb,
        Ac,
        Ab,
        np.concatenate([hz.b, eq_b]),
        Auc,
        Aub,
        ub_rhs,
    ), (m, int(narrow_idx.size))


def _sigmoid_piecewise_pruned(
    hz: SparseHZ,
    pre_bounds,
    *,
    K: int = 2,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
    func=_sigmoid_np,
    dfunc=_sigmoid_deriv_np,
) -> Tuple[SparseHZ, Tuple[int, int]]:
    """Compressed sigmoid encoding with exact removal of zero-width inflection-side segments."""
    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin
    lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    wide = (ub - lb) > 1e-12
    wide_idx = np.nonzero(wide)[0].astype(np.int32)
    narrow_idx = np.nonzero(~wide)[0].astype(np.int32)
    m = int(wide_idx.size)

    out_c = np.zeros(n, dtype=np.float64)
    if narrow_idx.size:
        out_c[narrow_idx] = func(hz.c[narrow_idx])
    if m == 0:
        return SparseHZ(
            out_c,
            _empty(n, ng),
            _empty(n, nb),
            hz.Ac,
            hz.Ab,
            hz.b,
            hz.Auc,
            hz.Aub,
            hz.ub,
        ), (0, int(narrow_idx.size))

    K_side = max(1, int(K))
    seg_a: List[float] = []
    seg_b: List[float] = []
    owner: List[int] = []
    seg_count = np.zeros(m, dtype=np.int32)
    for j, idx in enumerate(wide_idx):
        lo = float(lb[idx])
        hi = float(ub[idx])
        pivot = min(max(0.0, lo), hi)
        start = len(owner)
        if pivot - lo > 1e-12:
            cuts = _scurve_breakpoints(lo, pivot, K_side, grid=grid, func=func, dfunc=dfunc)
            for s in range(K_side):
                seg_a.append(float(cuts[s]))
                seg_b.append(float(cuts[s + 1]))
                owner.append(j)
        if hi - pivot > 1e-12:
            cuts = _scurve_breakpoints(pivot, hi, K_side, grid=grid, func=func, dfunc=dfunc)
            for s in range(K_side):
                seg_a.append(float(cuts[s]))
                seg_b.append(float(cuts[s + 1]))
                owner.append(j)
        seg_count[j] = len(owner) - start

    a = np.asarray(seg_a, dtype=np.float64)
    b_seg = np.asarray(seg_b, dtype=np.float64)
    owner_arr = np.asarray(owner, dtype=np.int32)
    r = int(a.size)
    if r == 0:
        raise ValueError("wide sigmoid neuron produced no nondegenerate segment")

    fa = func(a)
    fb = func(b_seg)
    la = dfunc(a)
    lb_slope = dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = np.abs(la - lb_slope) < 1e-10

    denom = lb_slope - la
    safe_denom = np.where(nearly_linear, 1.0, denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = np.linspace(0.0, 1.0, 50, dtype=np.float64).reshape(50, 1)
    pts = a.reshape(1, r) + t_pts * (b_seg - a).reshape(1, r)
    f_pts = func(pts)
    resid = f_pts - (slope.reshape(1, r) * pts + (fa - slope * a).reshape(1, r))
    max_err = np.max(np.abs(resid), axis=0)
    g1x_lin = hw
    g1y_lin = slope * hw
    g2x_lin = np.zeros_like(hw)
    g2y_lin = max_err

    g1_x = np.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = np.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = np.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = np.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.reshape(1, r)
    dy = f_pts - centers_y.reshape(1, r)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = np.where(np.abs(det) < 1e-30, 1.0, det)
    xi1 = (dy * g2_x.reshape(1, r) - dx * g2_y.reshape(1, r)) / safe_det.reshape(1, r)
    xi2 = (dy * g1_x.reshape(1, r) - dx * g1_y.reshape(1, r)) / (-safe_det.reshape(1, r))
    max_xi = np.maximum(np.max(np.abs(xi1), axis=0), np.max(np.abs(xi2), axis=0))
    scale_factor = np.where(max_xi > 1.0, max_xi * 1.01, 1.0)
    scale_factor = np.where(np.abs(det) < 1e-30, 1.0, scale_factor)
    g1_x *= scale_factor
    g1_y *= scale_factor
    g2_x *= scale_factor
    g2_y *= scale_factor

    _sigmoid_piecewise.last_meta = {
        "wide_idx": wide_idx,
        "m": m,
        "compressed": True,
        "pruned": True,
        "r": r,
        "owner_arr": owner_arr,
        "seg_count": seg_count,
        "a": a,
        "b_seg": b_seg,
        "centers_x": centers_x,
        "centers_y": centers_y,
        "g1_x": g1_x,
        "g1_y": g1_y,
        "g2_x": g2_x,
        "g2_y": g2_y,
        "grid": grid,
    }

    out_c[wide_idx] = np.bincount(owner_arr, weights=centers_y, minlength=m) / 2.0
    n_real = 2 * r
    ng_total = ng + n_real
    nb_total = nb + r
    g1_cols = ng + np.arange(r, dtype=np.int32)
    g2_cols = ng + r + np.arange(r, dtype=np.int32)
    z_cols = nb + np.arange(r, dtype=np.int32)
    wide_rows = wide_idx[owner_arr]

    out_Gc = sp.coo_matrix(
        (
            np.concatenate([g1_y, g2_y]),
            (
                np.concatenate([wide_rows, wide_rows]),
                np.concatenate([g1_cols, g2_cols]),
            ),
        ),
        shape=(n, ng_total),
    ).tocsr()
    out_Gb = sp.coo_matrix((-centers_y / 2.0, (wide_rows, z_cols)), shape=(n, nb_total)).tocsr()
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    n_eq_new = 2 * m
    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []

    def add_c(rows, cols, data):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size:
            rr_c.append(np.asarray(rows, dtype=np.int32).reshape(-1))
            cc_c.append(np.asarray(cols, dtype=np.int32).reshape(-1))
            dd_c.append(arr)

    def add_b(rows, cols, data):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size:
            rr_b.append(np.asarray(rows, dtype=np.int32).reshape(-1))
            cc_b.append(np.asarray(cols, dtype=np.int32).reshape(-1))
            dd_b.append(arr)

    link_rows = np.arange(m, dtype=np.int32)
    sum_rows = m + link_rows
    add_c(link_rows[owner_arr], g1_cols, -g1_x)
    add_c(link_rows[owner_arr], g2_cols, -g2_x)
    add_b(link_rows[owner_arr], z_cols, centers_x / 2.0)
    pre_gc = hz.Gc[wide_idx].tocoo()
    if pre_gc.nnz:
        add_c(link_rows[pre_gc.row], pre_gc.col, pre_gc.data)
    if nb:
        pre_gb = hz.Gb[wide_idx].tocoo()
        if pre_gb.nnz:
            add_b(link_rows[pre_gb.row], pre_gb.col, pre_gb.data)
    add_b(sum_rows[owner_arr], z_cols, np.ones(r, dtype=np.float64))

    eq_b = np.zeros(n_eq_new, dtype=np.float64)
    eq_b[link_rows] = np.bincount(owner_arr, weights=centers_x, minlength=m) / 2.0 - hz.c[wide_idx]
    eq_b[sum_rows] = seg_count.astype(np.float64) - 2.0
    eq_Ac = sp.coo_matrix(
        (np.concatenate(dd_c), (np.concatenate(rr_c), np.concatenate(cc_c))),
        shape=(n_eq_new, ng_total),
    ).tocsr()
    eq_Ab = sp.coo_matrix(
        (np.concatenate(dd_b), (np.concatenate(rr_b), np.concatenate(cc_b))),
        shape=(n_eq_new, nb_total),
    ).tocsr()

    old_Ac = _pad_cols(hz.Ac, ng_total)
    old_Ab = _pad_cols(hz.Ab, nb_total)
    Ac = sp.vstack([old_Ac, eq_Ac], format="csr")
    Ab = sp.vstack([old_Ab, eq_Ab], format="csr")
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()

    Auc_base = _pad_cols(hz.Auc if hz.Auc is not None else _empty(0, hz.n_cont), ng_total)
    Aub_base = _pad_cols(hz.Aub if hz.Aub is not None else _empty(0, hz.n_bin), nb_total)
    ub_base = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    box_rows = 4 * np.arange(r, dtype=np.int32)
    rr_uc = np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3])
    cc_uc = np.concatenate([g1_cols, g1_cols, g2_cols, g2_cols])
    dd_uc = np.concatenate([np.ones(r), -np.ones(r), np.ones(r), -np.ones(r)])
    rr_ub = rr_uc.copy()
    cc_ub = np.concatenate([z_cols, z_cols, z_cols, z_cols])
    dd_ub = 0.5 * np.ones(4 * r, dtype=np.float64)
    box_Auc = sp.coo_matrix((dd_uc, (rr_uc, cc_uc)), shape=(4 * r, ng_total)).tocsr()
    box_Aub = sp.coo_matrix((dd_ub, (rr_ub, cc_ub)), shape=(4 * r, nb_total)).tocsr()
    Auc = sp.vstack([Auc_base, box_Auc], format="csr")
    Aub = sp.vstack([Aub_base, box_Aub], format="csr")
    ub_rhs = np.concatenate([ub_base, 0.5 * np.ones(4 * r, dtype=np.float64)])
    if domain_cuts:
        dom_Auc, dom_Aub, dom_rhs = _scurve_domain_cut_matrices(
            hz,
            wide_idx,
            a,
            b_seg,
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            lb[wide_idx],
            ub[wide_idx],
        )
        rng_Auc, rng_Aub, rng_rhs = _scurve_range_cut_matrices(
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            fa,
            fb,
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            func(lb[wide_idx]),
            func(ub[wide_idx]),
        )
        Auc = sp.vstack([Auc, dom_Auc], format="csr")
        Aub = sp.vstack([Aub, dom_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, dom_rhs])
        Auc = sp.vstack([Auc, rng_Auc], format="csr")
        Aub = sp.vstack([Aub, rng_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, rng_rhs])
    if graph_cuts:
        graph_Auc, graph_Aub, graph_rhs = _scurve_graph_cut_matrices(
            hz,
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            a,
            b_seg,
            owner_arr,
            z_cols,
            ng_total,
            nb_total,
            lb[wide_idx],
            ub[wide_idx],
            func=func,
            dfunc=dfunc,
        )
        Auc = sp.vstack([Auc, graph_Auc], format="csr")
        Aub = sp.vstack([Aub, graph_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, graph_rhs])
    return SparseHZ(
        out_c,
        out_Gc,
        out_Gb,
        Ac,
        Ab,
        np.concatenate([hz.b, eq_b]),
        Auc,
        Aub,
        ub_rhs,
    ), (m, int(narrow_idx.size))


def _propagate_sparse(
    net,
    queries,
    before,
    after,
    *,
    tight_relu_lp_limit: int = 0,
    tight_relu_lp_timeout: float = 0.0,
    tight_relu_fix_mode: str = "both",
    tight_schedule: Optional[Dict[int, int]] = None,
    relu_cuts: bool = False,
    compressed_relu: bool = False,
    compressed_sigmoid: bool = False,
    sigmoid_prune_degenerate: bool = False,
    scurve_domain_cuts: bool = False,
    scurve_graph_cuts: bool = False,
    sigmoid_k: int = 2,
    tanh_k: int = 1,
    scurve_grid: str = "uniform",
) -> SparseHZ:
    states: Dict[int, SparseHZ] = {}
    var_owner: Dict[int, Tuple[int, int]] = {}
    global_c = 0
    global_b = 0
    final: Optional[SparseHZ] = None
    witness_ok = True
    witness_msg = "constructive_center"
    xi_c = np.zeros(0, dtype=np.float64)
    xi_b = np.zeros(0, dtype=np.float64)
    layer_by_id = {int(L.id): L for L in net.layers}
    static_var_owner: Dict[int, int] = {}
    for layer in net.layers:
        for var in getattr(layer, "out_vars", []) or []:
            static_var_owner[int(var)] = int(layer.id)
    layer_consumes: Dict[int, set[int]] = {}
    use_count: Dict[int, int] = {}
    for layer in net.layers:
        consumed = {int(p) for p in net.preds.get(layer.id, [])}
        for var in getattr(layer, "in_vars", []) or []:
            owner = static_var_owner.get(int(var))
            if owner is not None and owner != int(layer.id):
                consumed.add(owner)
        layer_consumes[int(layer.id)] = consumed
        for owner in consumed:
            use_count[owner] = use_count.get(owner, 0) + 1

    class ArrayBounds:
        def __init__(self, lb_arr: np.ndarray, ub_arr: np.ndarray):
            import torch

            self.lb = torch.as_tensor(lb_arr, dtype=torch.float64)
            self.ub = torch.as_tensor(ub_arr, dtype=torch.float64)

    def mark_witness_bad(msg: str) -> None:
        nonlocal witness_ok, witness_msg
        if witness_ok:
            witness_ok = False
            witness_msg = msg

    def extend_relu_witness(prev: SparseHZ, out: SparseHZ, meta: dict, layer_id: int) -> None:
        nonlocal xi_c, xi_b
        if not witness_ok:
            return
        if xi_c.size != prev.n_cont or xi_b.size != prev.n_bin:
            mark_witness_bad(
                f"layer {layer_id}: witness size mismatch "
                f"{xi_c.size}+{xi_b.size} vs {prev.n_cont}+{prev.n_bin}"
            )
            return
        unstable_idx = np.asarray(meta.get("unstable_idx", []), dtype=np.int64)
        k = int(unstable_idx.size)
        if k == 0:
            return
        lb = np.asarray(meta["lb"], dtype=np.float64)
        ub = np.asarray(meta["ub"], dtype=np.float64)
        compressed = bool(meta.get("compressed", False))
        pre = prev.c + np.asarray(prev.Gc @ xi_c).reshape(-1)
        if prev.n_bin:
            pre = pre + np.asarray(prev.Gb @ xi_b).reshape(-1)
        add_c = np.zeros((2 if compressed else 4) * k, dtype=np.float64)
        add_b = np.zeros(k, dtype=np.float64)
        tol = 1e-4
        for j, row in enumerate(unstable_idx):
            alpha = float(lb[row])
            beta = float(ub[row])
            x = float(pre[row])
            if x < alpha - tol or x > beta + tol:
                mark_witness_bad(
                    f"layer {layer_id}: center preactivation outside ReLU bounds "
                    f"row={int(row)} x={x:.8g} lb={alpha:.8g} ub={beta:.8g}"
                )
                return
            x = min(max(x, alpha), beta)
            if x <= 0.0:
                xi1 = -1.0 if abs(alpha) < 1e-12 else 2.0 * x / alpha - 1.0
                xi2 = 1.0
                xi3 = -xi1
                xi4 = 1.0
                z = 1.0
            else:
                xi2 = 1.0 if abs(beta) < 1e-12 else 1.0 - 2.0 * x / beta
                xi1 = 1.0
                xi3 = 1.0
                xi4 = -xi2
                z = -1.0
            vals = np.array([xi1, xi2, xi3, xi4, z], dtype=np.float64)
            if np.max(np.abs(vals)) > 1.0 + 1e-5:
                mark_witness_bad(
                    f"layer {layer_id}: ReLU witness variable out of box "
                    f"row={int(row)} vals={vals.tolist()}"
                )
                return
            add_c[j] = np.clip(xi1, -1.0, 1.0)
            add_c[k + j] = np.clip(xi2, -1.0, 1.0)
            if not compressed:
                add_c[2 * k + j] = np.clip(xi3, -1.0, 1.0)
                add_c[3 * k + j] = np.clip(xi4, -1.0, 1.0)
            add_b[j] = z
        xi_c = np.concatenate([xi_c, add_c])
        xi_b = np.concatenate([xi_b, add_b])
        if xi_c.size != out.n_cont or xi_b.size != out.n_bin:
            mark_witness_bad(
                f"layer {layer_id}: extended witness size mismatch "
                f"{xi_c.size}+{xi_b.size} vs {out.n_cont}+{out.n_bin}"
            )

    def extend_scurve_witness(prev: SparseHZ, out: SparseHZ, meta: dict, layer_id: int, func) -> None:
        nonlocal xi_c, xi_b
        if not witness_ok:
            return
        if xi_c.size != prev.n_cont or xi_b.size != prev.n_bin:
            mark_witness_bad(
                f"layer {layer_id}: witness size mismatch "
                f"{xi_c.size}+{xi_b.size} vs {prev.n_cont}+{prev.n_bin}"
            )
            return
        wide_idx = np.asarray(meta.get("wide_idx", []), dtype=np.int64)
        m = int(wide_idx.size)
        if m == 0:
            return
        if meta.get("pruned"):
            a_seg = np.asarray(meta.get("a", []), dtype=np.float64)
            b_seg = np.asarray(meta.get("b_seg", []), dtype=np.float64)
            centers_x = np.asarray(meta.get("centers_x", []), dtype=np.float64)
            centers_y = np.asarray(meta.get("centers_y", []), dtype=np.float64)
            g1_x = np.asarray(meta.get("g1_x", []), dtype=np.float64)
            g1_y = np.asarray(meta.get("g1_y", []), dtype=np.float64)
            g2_x = np.asarray(meta.get("g2_x", []), dtype=np.float64)
            g2_y = np.asarray(meta.get("g2_y", []), dtype=np.float64)
            owner_arr = np.asarray(meta.get("owner_arr", []), dtype=np.int64)
            r = int(meta.get("r", owner_arr.size))
            if any(arr.size != r for arr in (a_seg, b_seg, centers_x, centers_y, g1_x, g1_y, g2_x, g2_y, owner_arr)):
                mark_witness_bad(f"layer {layer_id}: pruned S-curve witness metadata size mismatch")
                return
            pre = prev.c + np.asarray(prev.Gc @ xi_c).reshape(-1)
            if prev.n_bin:
                pre = pre + np.asarray(prev.Gb @ xi_b).reshape(-1)
            add_c = np.zeros(2 * r, dtype=np.float64)
            add_b = np.ones(r, dtype=np.float64)
            tol = 1e-7
            for j, row in enumerate(wide_idx):
                owned = np.nonzero(owner_arr == j)[0]
                if owned.size == 0:
                    mark_witness_bad(f"layer {layer_id}: pruned S-curve row={int(row)} has no segment")
                    return
                x = float(pre[int(row)])
                lo = float(np.min(a_seg[owned]))
                hi = float(np.max(b_seg[owned]))
                if x < lo - 1e-5 or x > hi + 1e-5:
                    mark_witness_bad(
                        f"layer {layer_id}: center preactivation outside pruned S-curve bounds "
                        f"row={int(row)} x={x:.8g} lb={lo:.8g} ub={hi:.8g}"
                    )
                    return
                x = min(max(x, lo), hi)
                widths = b_seg[owned] - a_seg[owned]
                ok_local = np.where(
                    (widths > 1e-12) & (x >= a_seg[owned] - tol) & (x <= b_seg[owned] + tol)
                )[0]
                if ok_local.size == 0:
                    ok_local = np.where((x >= a_seg[owned] - tol) & (x <= b_seg[owned] + tol))[0]
                if ok_local.size == 0:
                    mark_witness_bad(
                        f"layer {layer_id}: no pruned S-curve segment contains row={int(row)} x={x:.8g}"
                    )
                    return
                flat = int(owned[int(ok_local[0])])
                y = float(func(np.asarray([x], dtype=np.float64))[0])
                dx = x - float(centers_x[flat])
                dy = y - float(centers_y[flat])
                det = float(g1_y[flat] * g2_x[flat] - g1_x[flat] * g2_y[flat])
                if abs(det) < 1e-30:
                    xi1 = 0.0
                    xi2 = 0.0
                else:
                    xi1 = (dy * float(g2_x[flat]) - dx * float(g2_y[flat])) / det
                    xi2 = (dy * float(g1_x[flat]) - dx * float(g1_y[flat])) / (-det)
                if max(abs(xi1), abs(xi2)) > 1.0 + 1e-5:
                    mark_witness_bad(
                        f"layer {layer_id}: pruned S-curve witness variable out of box "
                        f"row={int(row)} xi1={xi1:.8g} xi2={xi2:.8g}"
                    )
                    return
                add_c[flat] = float(np.clip(xi1, -1.0, 1.0))
                add_c[r + flat] = float(np.clip(xi2, -1.0, 1.0))
                add_b[flat] = -1.0
            xi_c = np.concatenate([xi_c, add_c])
            xi_b = np.concatenate([xi_b, add_b])
            if xi_c.size != out.n_cont or xi_b.size != out.n_bin:
                mark_witness_bad(
                    f"layer {layer_id}: extended pruned S-curve witness size mismatch "
                    f"{xi_c.size}+{xi_b.size} vs {out.n_cont}+{out.n_bin}"
                )
            return
        S = int(meta["S"])
        compressed = bool(meta.get("compressed", False))
        a_seg = np.asarray(meta["a"], dtype=np.float64)
        b_seg = np.asarray(meta["b_seg"], dtype=np.float64)
        centers_x = np.asarray(meta["centers_x"], dtype=np.float64)
        centers_y = np.asarray(meta["centers_y"], dtype=np.float64)
        g1_x = np.asarray(meta["g1_x"], dtype=np.float64)
        g1_y = np.asarray(meta["g1_y"], dtype=np.float64)
        g2_x = np.asarray(meta["g2_x"], dtype=np.float64)
        g2_y = np.asarray(meta["g2_y"], dtype=np.float64)

        pre = prev.c + np.asarray(prev.Gc @ xi_c).reshape(-1)
        if prev.n_bin:
            pre = pre + np.asarray(prev.Gb @ xi_b).reshape(-1)
        n_real = 2 * S * m
        n_slack = 0 if compressed else 4 * S * m
        add_c = np.zeros(n_real + n_slack, dtype=np.float64)
        add_b = np.ones(S * m, dtype=np.float64)
        if not compressed:
            # Unselected segments have z=+1 and g1=g2=0. Their four box
            # equalities are s - 0.5*z = 0.5, hence each slack must be +1.
            # The selected segment's slacks are overwritten below.
            add_c[n_real:] = 1.0
        tol = 1e-7
        for j, row in enumerate(wide_idx):
            x = float(pre[int(row)])
            lo = float(np.min(a_seg[:, j]))
            hi = float(np.max(b_seg[:, j]))
            if x < lo - 1e-5 or x > hi + 1e-5:
                mark_witness_bad(
                    f"layer {layer_id}: center preactivation outside S-curve bounds "
                    f"row={int(row)} x={x:.8g} lb={lo:.8g} ub={hi:.8g}"
                )
                return
            x = min(max(x, lo), hi)
            widths = b_seg[:, j] - a_seg[:, j]
            ok = np.where((widths > 1e-12) & (x >= a_seg[:, j] - tol) & (x <= b_seg[:, j] + tol))[0]
            if ok.size == 0:
                ok = np.where((x >= a_seg[:, j] - tol) & (x <= b_seg[:, j] + tol))[0]
            if ok.size == 0:
                mark_witness_bad(f"layer {layer_id}: no S-curve segment contains row={int(row)} x={x:.8g}")
                return
            s = int(ok[0])
            flat = s * m + j
            y = float(func(np.asarray([x], dtype=np.float64))[0])
            dx = x - float(centers_x[s, j])
            dy = y - float(centers_y[s, j])
            det = float(g1_y[s, j] * g2_x[s, j] - g1_x[s, j] * g2_y[s, j])
            if abs(det) < 1e-30:
                xi1 = 0.0
                xi2 = 0.0
            else:
                xi1 = (dy * float(g2_x[s, j]) - dx * float(g2_y[s, j])) / det
                xi2 = (dy * float(g1_x[s, j]) - dx * float(g1_y[s, j])) / (-det)
            if max(abs(xi1), abs(xi2)) > 1.0 + 1e-5:
                mark_witness_bad(
                    f"layer {layer_id}: S-curve witness variable out of box "
                    f"row={int(row)} xi1={xi1:.8g} xi2={xi2:.8g}"
                )
                return
            xi1 = float(np.clip(xi1, -1.0, 1.0))
            xi2 = float(np.clip(xi2, -1.0, 1.0))
            add_c[flat] = xi1
            add_c[S * m + flat] = xi2
            add_b[flat] = -1.0
            if not compressed:
                slack = n_real + 4 * flat
                add_c[slack] = -xi1
                add_c[slack + 1] = xi1
                add_c[slack + 2] = -xi2
                add_c[slack + 3] = xi2
        xi_c = np.concatenate([xi_c, add_c])
        xi_b = np.concatenate([xi_b, add_b])
        if xi_c.size != out.n_cont or xi_b.size != out.n_bin:
            mark_witness_bad(
                f"layer {layer_id}: extended S-curve witness size mismatch "
                f"{xi_c.size}+{xi_b.size} vs {out.n_cont}+{out.n_bin}"
            )

    def mark_owner(L) -> None:
        for row, var in enumerate(getattr(L, "out_vars", []) or []):
            var_owner[int(var)] = (int(L.id), int(row))

    def source_hz(L, pred_idx: int = 0) -> SparseHZ:
        preds = net.preds.get(L.id, [])
        if preds:
            return states[preds[pred_idx]]
        in_vars = list(getattr(L, "in_vars", []) or [])
        if in_vars:
            owners = [var_owner.get(int(v)) for v in in_vars]
            if all(o is not None for o in owners):
                lid = owners[0][0]  # type: ignore[index]
                if all(o[0] == lid for o in owners):  # type: ignore[index]
                    rows = np.asarray([o[1] for o in owners], dtype=np.int64)  # type: ignore[index]
                    return _gather_hz_rows(states[lid], rows)
        raise KeyError(f"cannot find sparse source for layer {L.id} ({L.kind})")

    def release_sources(L) -> None:
        for lid in layer_consumes.get(int(L.id), set()):
            if lid not in use_count:
                continue
            use_count[lid] -= 1
            if use_count[lid] <= 0:
                states.pop(lid, None)

    for idx, L in enumerate(net.layers):
        kind = str(L.kind).upper()
        t0 = time.time()
        if kind == "INPUT":
            hz = SparseHZ(
                c=np.zeros(len(L.out_vars), dtype=np.float64),
                Gc=_empty(len(L.out_vars), global_c),
                Gb=_empty(len(L.out_vars), global_b),
                Ac=_empty(0, global_c),
                Ab=_empty(0, global_b),
                b=np.zeros(0, dtype=np.float64),
                Auc=_empty(0, global_c),
                Aub=_empty(0, global_b),
                ub=np.zeros(0, dtype=np.float64),
            )
        elif kind == "INPUT_SPEC":
            hz = sparse_hz_from_bounds(queries[0][0])
            if hz.n_out != len(L.out_vars):
                raise ValueError(f"input spec size mismatch: hz={hz.n_out}, layer={len(L.out_vars)}")
            global_c, global_b = hz.n_cont, hz.n_bin
            xi_c = np.zeros(global_c, dtype=np.float64)
            xi_b = np.zeros(global_b, dtype=np.float64)
        elif kind == "CONV2D":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            W, bvec = _conv2d_matrix(L)
            hz = _linear_apply(prev, W, bvec)
            del W
        elif kind == "CONVTRANSPOSE2D":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            W, bvec = _convtranspose2d_matrix(L)
            hz = _linear_apply(prev, W, bvec)
            del W
        elif kind == "DENSE":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            W, bvec = _dense_matrix(L)
            hz = _linear_apply(prev, W, bvec)
        elif kind == "MATMUL":
            preds = net.preds.get(L.id, [])
            if len(preds) != 2:
                raise NotImplementedError(f"unsupported sparse MATMUL routing at {L.id}")
            x = _pad_hz(source_hz(L, 0), global_c, global_b)
            y = _pad_hz(source_hz(L, 1), global_c, global_b)
            xb = after[preds[0]].bounds
            yb = after[preds[1]].bounds
            hz, xi_extra = _matmul_product_interval_hz(
                x,
                y,
                L,
                xb.lb.detach().cpu().numpy().reshape(-1).astype(np.float64),
                xb.ub.detach().cpu().numpy().reshape(-1).astype(np.float64),
                yb.lb.detach().cpu().numpy().reshape(-1).astype(np.float64),
                yb.ub.detach().cpu().numpy().reshape(-1).astype(np.float64),
                xi_c,
                xi_b,
            )
            xi_c = np.concatenate([xi_c, xi_extra])
            global_c = hz.n_cont
        elif kind == "UPSAMPLE":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            rows_t = hz_mlp._upsample_nearest_row_idx(
                L, prev.n_out, len(L.out_vars), before[L.id].lb.device)
            if rows_t is None:
                raise NotImplementedError(f"unsupported sparse UPSAMPLE shape at {L.id}")
            rows = rows_t.detach().cpu().numpy().astype(np.int64).reshape(-1)
            hz = _gather_hz_rows(prev, rows)
        elif kind == "AVGPOOL2D":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            W, bvec = _avgpool2d_matrix(L)
            hz = _linear_apply(prev, W, bvec)
            del W
        elif kind == "MAXPOOL2D":
            base = _pad_hz(source_hz(L), global_c, global_b)
            cands = _maxpool2d_candidate_rows(L)
            in_lb = before[L.id].lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
            in_ub = before[L.id].ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
            fill_value = float(np.min(in_lb) - 1.0) if in_lb.size else -1.0

            def gather_bounds(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                rows = np.asarray(rows, dtype=np.int64).reshape(-1)
                valid = rows >= 0
                lb = np.full(rows.size, fill_value, dtype=np.float64)
                ub = np.full(rows.size, fill_value, dtype=np.float64)
                if np.any(valid):
                    lb[valid] = in_lb[rows[valid]]
                    ub[valid] = in_ub[rows[valid]]
                return lb, ub

            hz = _gather_hz_rows_like(base, cands[0], fill_value=fill_value)
            cur_lb, cur_ub = gather_bounds(cands[0])
            mp_active = mp_inactive = mp_unstable = 0
            for fold_i, rows in enumerate(cands[1:], start=1):
                nxt = _gather_hz_rows_like(base, rows, fill_value=fill_value, template=hz)
                nxt_lb, nxt_ub = gather_bounds(rows)
                diff = _sub_same_frame(hz, nxt)
                relu, counts, _ = _relu_exact(
                    diff,
                    ArrayBounds(cur_lb - nxt_ub, cur_ub - nxt_lb),
                    compressed=compressed_relu,
                )
                extend_relu_witness(diff, relu, getattr(_relu_exact, "last_meta", {}), int(L.id))
                hz = _add_same_frame(nxt, relu)
                cur_lb = np.maximum(cur_lb, nxt_lb)
                cur_ub = np.maximum(cur_ub, nxt_ub)
                global_c, global_b = hz.n_cont, hz.n_bin
                mp_active += int(counts[0])
                mp_inactive += int(counts[1])
                mp_unstable += int(counts[2])
            print(
                f"layer {L.id:>2} MAXPOOL2D folds={max(0, len(cands) - 1)} "
                f"active/off/unstable={mp_active}/{mp_inactive}/{mp_unstable} "
                f"vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
                f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
                f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
                f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}",
                flush=True,
            )
            states[L.id] = hz
            mark_owner(L)
            final = hz
            release_sources(L)
            continue
        elif kind == "SCALE":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            hz = _scale_apply(prev, L.params["a"])
        elif kind == "BIAS":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            hz = _bias_apply(prev, L.params["c"])
        elif kind in {"FLATTEN", "RESHAPE", "SQUEEZE", "UNSQUEEZE", "TRANSPOSE", "BN", "ASSERT"}:
            hz = _pad_hz(source_hz(L), global_c, global_b)
        elif kind == "SLICE":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            rows = _slice_row_idx_np(L, prev.n_out)
            if rows is None or rows.size != len(L.out_vars):
                raise NotImplementedError(f"unsupported sparse SLICE shape at {L.id}")
            hz = _gather_hz_rows(prev, rows)
        elif kind == "GATHER":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            rows = _gather_row_idx_np(L, prev.n_out)
            if rows is None or rows.size != len(L.out_vars):
                raise NotImplementedError(f"unsupported sparse GATHER shape at {L.id}")
            hz = _gather_hz_rows(prev, rows)
        elif kind == "CONCAT":
            preds = net.preds.get(L.id, [])
            if not preds:
                pred_indices = L.params.get("preds_indices", [])
                preds = [net.layers[int(i)].id for i in pred_indices] if pred_indices else []
            if not preds:
                raise NotImplementedError(f"unsupported sparse CONCAT routing at {L.id}")
            hz = sparse_hz_concat([_pad_hz(states[p], global_c, global_b) for p in preds])
        elif kind in {"ADD", "SUB"}:
            a = _pad_hz(source_hz(L, 0), global_c, global_b)
            b = _pad_hz(source_hz(L, 1), global_c, global_b)
            hz = _add_same_frame(a, b) if kind == "ADD" else _sub_same_frame(a, b)
        elif kind == "RELU":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            relu_limit = tight_relu_lp_limit
            if tight_schedule is not None:
                relu_limit = int(tight_schedule.get(int(L.id), 0))
            hz, counts, tight_stats = _relu_exact(
                prev,
                before[L.id],
                tight_lp_limit=relu_limit,
                tight_lp_timeout=tight_relu_lp_timeout,
                tight_fix_mode=tight_relu_fix_mode,
                add_cuts=relu_cuts,
                compressed=compressed_relu,
            )
            extend_relu_witness(prev, hz, getattr(_relu_exact, "last_meta", {}), int(L.id))
            global_c, global_b = hz.n_cont, hz.n_bin
            tnote = ""
            if tight_stats[0]:
                tnote = (
                    f" tightLP solved/improved/fix_on/fix_off="
                    f"{tight_stats[0]}/{tight_stats[1]}/{tight_stats[2]}/{tight_stats[3]}"
                )
            print(
                f"layer {L.id:>2} RELU active/off/unstable={counts[0]}/{counts[1]}/{counts[2]} "
                f"vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
                f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
                f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
                f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}{tnote}",
                flush=True,
            )
            states[L.id] = hz
            mark_owner(L)
            final = hz
            release_sources(L)
            continue
        elif kind == "SIGMOID":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            hz, counts = _sigmoid_piecewise(
                prev,
                before[L.id],
                K=sigmoid_k,
                compressed=compressed_sigmoid,
                drop_degenerate=sigmoid_prune_degenerate,
                domain_cuts=scurve_domain_cuts,
                graph_cuts=scurve_graph_cuts,
                grid=scurve_grid,
            )
            extend_scurve_witness(prev, hz, getattr(_sigmoid_piecewise, "last_meta", {}), int(L.id), _sigmoid_np)
            global_c, global_b = hz.n_cont, hz.n_bin
            print(
                f"layer {L.id:>2} SIGMOID wide/narrow={counts[0]}/{counts[1]} K={sigmoid_k} "
                f"compressed={int(compressed_sigmoid)} pruned={int(sigmoid_prune_degenerate)} "
                f"domain_cuts={int(scurve_domain_cuts)} graph_cuts={int(scurve_graph_cuts)} "
                f"grid={scurve_grid} "
                f"vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
                f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
                f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
                f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}",
                flush=True,
            )
            states[L.id] = hz
            mark_owner(L)
            final = hz
            release_sources(L)
            continue
        elif kind == "TANH":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            hz, counts = _sigmoid_piecewise(
                prev,
                before[L.id],
                K=tanh_k,
                compressed=compressed_sigmoid,
                drop_degenerate=False,
                domain_cuts=scurve_domain_cuts,
                graph_cuts=scurve_graph_cuts,
                grid=scurve_grid,
                func=_tanh_np,
                dfunc=_tanh_deriv_np,
            )
            extend_scurve_witness(prev, hz, getattr(_sigmoid_piecewise, "last_meta", {}), int(L.id), _tanh_np)
            global_c, global_b = hz.n_cont, hz.n_bin
            print(
                f"layer {L.id:>2} TANH wide/narrow={counts[0]}/{counts[1]} K={tanh_k} "
                f"compressed={int(compressed_sigmoid)} domain_cuts={int(scurve_domain_cuts)} "
                f"graph_cuts={int(scurve_graph_cuts)} grid={scurve_grid} "
                f"vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
                f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
                f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
                f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}",
                flush=True,
            )
            states[L.id] = hz
            mark_owner(L)
            final = hz
            release_sources(L)
            continue
        elif kind == "SOFTMAX":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            preds = net.preds.get(L.id, [])
            input_shape = tuple(int(v) for v in L.params.get("input_shape", ()))
            if not input_shape and preds:
                pred_layer = layer_by_id.get(int(preds[0]))
                if pred_layer is not None:
                    input_shape = tuple(int(v) for v in pred_layer.params.get("output_shape", ()))
            if not input_shape:
                input_shape = (prev.n_out,)
            bnd = before[L.id]
            hz, xi_extra = _softmax_simplex_hz(
                prev,
                L,
                input_shape,
                bnd.lb.detach().cpu().numpy().reshape(-1).astype(np.float64),
                bnd.ub.detach().cpu().numpy().reshape(-1).astype(np.float64),
                xi_c,
                xi_b,
            )
            xi_c = np.concatenate([xi_c, xi_extra])
            global_c = hz.n_cont
            print(
                f"layer {L.id:>2} SOFTMAX simplex shape={input_shape} "
                f"vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
                f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
                f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
                f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}",
                flush=True,
            )
            states[L.id] = hz
            mark_owner(L)
            final = hz
            release_sources(L)
            continue
        else:
            raise NotImplementedError(f"unsupported layer kind {kind} at {L.id}")

        states[L.id] = hz
        mark_owner(L)
        final = hz
        release_sources(L)
        print(
            f"layer {L.id:>2} {kind:<8} out={hz.n_out:>5} vars={_format_big(hz.n_cont)}+{_format_big(hz.n_bin)} "
            f"eq={_format_big(hz.n_eq)} val_nnz={_format_big(hz.value_nnz)} "
            f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
            f"ub_nnz={_format_big(hz.ub_nnz)} sec={time.time() - t0:.2f}",
            flush=True,
        )
        if idx % 4 == 0:
            gc.collect()
    assert final is not None
    if witness_ok:
        if xi_c.size != final.n_cont or xi_b.size != final.n_bin:
            witness_ok = False
            witness_msg = (
                f"final witness size mismatch {xi_c.size}+{xi_b.size} "
                f"vs {final.n_cont}+{final.n_bin}"
            )
        else:
            if final.n_eq:
                eq_resid = (
                    np.asarray(final.Ac @ xi_c).reshape(-1)
                    + np.asarray(final.Ab @ xi_b).reshape(-1)
                    - final.b
                )
                max_eq = float(np.max(np.abs(eq_resid)))
            else:
                max_eq = 0.0
            if final.n_ub:
                ub_resid = (
                    np.asarray(final.Auc @ xi_c).reshape(-1)
                    + np.asarray(final.Aub @ xi_b).reshape(-1)
                    - final.ub
                )
                max_ub = float(np.max(ub_resid))
            else:
                max_ub = -np.inf
            witness_msg = f"constructive_center max_eq={max_eq:.3g} max_ub={max_ub:.3g}"
            if max_eq > 1e-4 or max_ub > 1e-4:
                witness_ok = False
    _propagate_sparse.last_base_witness = {
        "ok": bool(witness_ok),
        "msg": witness_msg,
        "xi": np.concatenate([xi_c, xi_b]) if witness_ok else None,
    }
    return final


def _self_test() -> None:
    hz = SparseHZ(
        c=np.asarray([2.0, -1.0], dtype=np.float64),
        Gc=_empty(2, 0),
        Gb=_empty(2, 0),
        Ac=_empty(0, 0),
        Ab=_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
    )
    margin, msg = _lp_min_margin(
        hz,
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        np.asarray([1.5], dtype=np.float64),
        1.0,
    )
    assert msg == "fixed"
    assert abs(float(margin) - 0.5) <= 1e-12

    margin, msg = _lp_min_margin(
        hz,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([1.5, -2.5], dtype=np.float64),
        1.0,
    )
    assert msg == "fixed"
    assert abs(float(margin) - 1.5) <= 1e-12

    bad_eq = SparseHZ(
        c=np.zeros(1, dtype=np.float64),
        Gc=_empty(1, 0),
        Gb=_empty(1, 0),
        Ac=_empty(1, 0),
        Ab=_empty(1, 0),
        b=np.ones(1, dtype=np.float64),
    )
    margin, msg = _lp_min_margin(
        bad_eq,
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        1.0,
    )
    assert margin is None and msg == "fixed_hz_infeasible_eq"

    bad_ub = SparseHZ(
        c=np.zeros(1, dtype=np.float64),
        Gc=_empty(1, 0),
        Gb=_empty(1, 0),
        Ac=_empty(0, 0),
        Ab=_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=_empty(1, 0),
        Aub=_empty(1, 0),
        ub=-np.ones(1, dtype=np.float64),
    )
    margin, msg = _lp_min_margin(
        bad_ub,
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        1.0,
    )
    assert margin is None and msg == "fixed_hz_infeasible_ub"

    import torch
    from act.back_end.core import Bounds

    scurve_bounds = Bounds(
        lb=torch.tensor([[-2.0, -0.5, 0.25]], dtype=torch.float64),
        ub=torch.tensor([[0.75, 1.5, 2.0]], dtype=torch.float64),
    )
    scurve_hz = sparse_hz_from_bounds(scurve_bounds)
    backend_hz, backend_counts = _sigmoid_piecewise(
        scurve_hz,
        scurve_bounds,
        K=2,
        compressed=True,
        drop_degenerate=True,
        domain_cuts=True,
        graph_cuts=True,
    )
    backend_meta = dict(getattr(_sigmoid_piecewise, "last_meta", {}))
    fallback_hz, fallback_counts = _sigmoid_piecewise_pruned(
        scurve_hz,
        scurve_bounds,
        K=2,
        domain_cuts=True,
        graph_cuts=True,
    )
    fallback_meta = dict(getattr(_sigmoid_piecewise, "last_meta", {}))
    assert backend_counts == fallback_counts == (3, 0)
    assert int(backend_meta["r"]) == int(fallback_meta["r"])
    assert np.array_equal(np.asarray(backend_meta["wide_idx"]), np.asarray(fallback_meta["wide_idx"]))
    assert np.array_equal(
        np.bincount(np.asarray(backend_meta["owner_arr"], dtype=np.int64), minlength=3),
        np.bincount(np.asarray(fallback_meta["owner_arr"], dtype=np.int64), minlength=3),
    )
    for meta in (backend_meta, fallback_meta):
        assert np.asarray(meta["owner_arr"]).size == int(meta["r"])
        assert np.asarray(meta["a"]).size == int(meta["r"])
        assert np.asarray(meta["b_seg"]).size == int(meta["r"])
    from act.back_end.solver.solver_hz_verdict import hz_base_feasibility, hz_row_max

    assert hz_base_feasibility(backend_hz, time_limit=2.0)[0] == "FEASIBLE"
    assert hz_base_feasibility(fallback_hz, time_limit=2.0)[0] == "FEASIBLE"
    for row in (
        np.array([1.0, -0.5, 0.25], dtype=np.float64),
        np.array([-0.25, 1.0, 1.5], dtype=np.float64),
        np.array([0.3, 0.7, -1.0], dtype=np.float64),
    ):
        bmx = hz_row_max(backend_hz, row, integer=True, time_limit=2.0)
        fmx = hz_row_max(fallback_hz, row, integer=True, time_limit=2.0)
        assert bmx is not None and fmx is not None
        assert abs(float(bmx) - float(fmx)) <= 1e-8
    print("PASS hybridz_sparse_exact_probe self-test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--bench", default="cifar100_2024")
    ap.add_argument("--iid", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lp-queries", type=int, default=5)
    ap.add_argument("--query-indices", default="")
    ap.add_argument("--lp-timeout", type=float, default=20.0)
    ap.add_argument("--skip-lp-before-milp", action="store_true",
                    help="experiment: for each selected query, skip the LP precheck and run the exact MILP directly")
    ap.add_argument("--check-witness", action="store_true")
    ap.add_argument("--milp-all", action="store_true")
    ap.add_argument("--cutoff-as-row", action="store_true",
                    help="encode margin<=0 as an explicit exact feasibility row instead of objective-bound")
    ap.add_argument("--milp-timeout", type=float, default=30.0)
    ap.add_argument("--stop-on-unsafe", action="store_true",
                    help="stop this instance once an exact-HZ unsafe feasible rival is found")
    ap.add_argument("--mip-solver", choices=["highs", "scip"], default="highs")
    ap.add_argument("--mip-start", choices=["none", "base", "base-binary"], default="none",
                    help="give exact HiGHS MILP a constructive-base start; verdicts still require exact MILP status")
    ap.add_argument("--elim-singletons", action="store_true",
                    help="exactly project objective-free singleton continuous columns from the MILP")
    ap.add_argument("--elim-eq-subst", action="store_true",
                    help="experiment: exact equality substitution for objective-free continuous singleton columns")
    ap.add_argument("--connected-presolve", action="store_true",
                    help="experimental exact MILP presolve: keep only the constraint component connected to the margin objective/cutoff")
    ap.add_argument("--fbbt-passes", type=int, default=0,
                    help="run N passes of exact linear feasibility-based bound tightening before MILP")
    ap.add_argument("--relax-precheck-timeout", type=float, default=0.0,
                    help="run a continuous-relaxation EMPTY-only precheck before exact MILP")
    ap.add_argument("--highs-threads", type=int, default=0)
    ap.add_argument("--highs-parallel", default="")
    ap.add_argument("--highs-heuristic-effort", type=float, default=None)
    ap.add_argument("--highs-option", action="append", default=[],
                    help="extra HiGHS option as name=value; may be repeated")
    ap.add_argument("--scip-threads", type=int, default=0,
                    help="SCIP parallel/maxnthreads and LP threads for exact MILP runs")
    ap.add_argument("--scip-option", action="append", default=[],
                    help="extra SCIP option as name=value; may be repeated")
    ap.add_argument("--tight-relu-lp-limit", type=int, default=0,
                    help="0=off, -1=all interval-unstable ReLUs, N=tighten N widest per ReLU")
    ap.add_argument("--tight-relu-lp-timeout", type=float, default=2.0)
    ap.add_argument("--tight-relu-fix-mode", choices=["both", "off-only"], default="both",
                    help="with tight ReLU LPs, optionally only phase-fix inactive ReLUs to preserve sparse exact-HZ rows")
    ap.add_argument("--tight-schedule", default="",
                    help="comma list layer_id:limit; overrides --tight-relu-lp-limit for ReLU layers")
    ap.add_argument("--sigmoid-k", type=int, default=2,
                    help="K segments per side for sparse sigmoid piecewise encoding")
    ap.add_argument("--tanh-k", type=int, default=1,
                    help="K segments per side for sparse tanh piecewise encoding")
    ap.add_argument("--relu-cuts", action="store_true",
                    help="add redundant exact-valid ReLU graph cuts while keeping exact binary eq_lagr")
    ap.add_argument("--compressed-relu", action="store_true",
                    help="experiment: exact projection of eq_lagr xi3/xi4 slack variables into inequality rows")
    ap.add_argument("--compressed-sigmoid", action="store_true",
                    help="experiment: exact projection of sigmoid segment box slacks into inequality rows")
    ap.add_argument("--sigmoid-prune-degenerate", action="store_true",
                    help="with --compressed-sigmoid, exactly remove zero-width sigmoid segments on one-sided intervals")
    ap.add_argument("--scurve-domain-cuts", action="store_true",
                    help="add exact-valid segment-domain cuts for sigmoid/tanh piecewise HZ encodings")
    ap.add_argument("--scurve-graph-cuts", action="store_true",
                    help="add exact-valid conditional secant/tangent graph cuts for sigmoid/tanh segments")
    ap.add_argument("--scurve-grid", choices=["uniform", "curvature"], default="uniform",
                    help="S-curve segment grid; curvature keeps K fixed but allocates segments by |f''|")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--base-feas-timeout", type=float, default=10.0,
                    help="time limit for the mandatory base-HZ nonemptiness guard")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    extra_highs_options = parse_key_value_options(args.highs_option, "highs-option")
    for key in extra_highs_options:
        if key in {"user_objective_scale", "user_bound_scale"}:
            raise SystemExit(
                f"--highs-option {key}=... is disabled for exact-HZ proof runs; "
                "it changes HiGHS-reported objective/bound semantics used by cutoff certificates"
            )
    extra_scip_options = parse_key_value_options(args.scip_option, "scip-option")
    highs_profiles: List[Tuple[str, Dict[str, object]]] = [("cli", dict(extra_highs_options))]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    sys.path.insert(0, str(REPO))

    t0 = time.time()
    onnx_path, vnnlib_path, input_shape, queries, net, before, after, interval_s = _build_net_and_interval(
        args.bench, args.iid, args.device
    )
    print(f"bench={args.bench} iid={args.iid} onnx={onnx_path.name} vnnlib={vnnlib_path.name}")
    print(f"input_shape={input_shape} queries={len(queries)} layers={len(net.layers)} interval_s={interval_s:.2f}")
    unsupported = [
        (int(L.id), str(L.kind).upper())
        for L in net.layers
        if str(L.kind).upper() not in SPARSE_SUPPORTED_KINDS
    ]
    if unsupported:
        lid, kind = unsupported[0]
        msg = f"unsupported layer kind {kind} at {lid}"
        print(f"EARLY_STOP unsupported_layer {msg}", flush=True)
        print("verdict_summary checked=0 cert=0 hz_unsafe=0 unknown=1 real_adv=0", flush=True)
        print(f"total_wall_s={time.time() - t0:.2f}", flush=True)
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_json, "w") as f:
                json.dump({
                    "bench": args.bench,
                    "iid": args.iid,
                    "unsupported_layer": {"id": lid, "kind": kind},
                    "verdict_summary": {
                        "checked": 0,
                        "cert": 0,
                        "hz_unsafe": 0,
                        "unknown": 1,
                        "real_adv": 0,
                    },
                }, f, indent=2, sort_keys=True)
        return
    schedule = None
    if args.tight_schedule.strip():
        schedule = {}
        for item in args.tight_schedule.split(","):
            if not item.strip():
                continue
            k, v = item.split(":", 1)
            schedule[int(k)] = int(v)

    hz = _propagate_sparse(
        net,
        queries,
        before,
        after,
        tight_relu_lp_limit=args.tight_relu_lp_limit,
        tight_relu_lp_timeout=args.tight_relu_lp_timeout,
        tight_relu_fix_mode=args.tight_relu_fix_mode,
        tight_schedule=schedule,
        relu_cuts=args.relu_cuts,
        compressed_relu=args.compressed_relu,
        compressed_sigmoid=args.compressed_sigmoid,
        sigmoid_prune_degenerate=args.sigmoid_prune_degenerate,
        scurve_domain_cuts=args.scurve_domain_cuts,
        scurve_graph_cuts=args.scurve_graph_cuts,
        sigmoid_k=args.sigmoid_k,
        tanh_k=args.tanh_k,
        scurve_grid=args.scurve_grid,
    )
    final_layer = net.layers[-1]
    final_pred = net.preds.get(final_layer.id, [None])[0]
    final_bounds = after[final_pred].bounds if final_pred in after else after[final_layer.id].bounds
    unsupported_spec = ""
    flat_specs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    try:
        flat_specs = _flatten_query_specs(queries, hz.n_out)
        hard, lows = _interval_hard_rivals_from_specs(flat_specs, final_bounds)
    except Exception as exc:
        hard, lows = 0, []
        unsupported_spec = str(exc)
    print(
        "final sparse HZ: "
        f"n={hz.n_out} ng={_format_big(hz.n_cont)} nb={_format_big(hz.n_bin)} "
        f"nc={_format_big(hz.n_eq)} value_nnz={_format_big(hz.value_nnz)} "
        f"eq_nnz={_format_big(hz.eq_nnz)} ub={_format_big(hz.n_ub)} "
        f"ub_nnz={_format_big(hz.ub_nnz)}"
    )
    if unsupported_spec:
        print(f"interval_hard=unsupported_spec error={unsupported_spec}", flush=True)
    else:
        print(
            f"interval_hard={hard}/{len(flat_specs)} "
            f"low_margin_min={min(lows):.6g} median={np.median(lows):.6g}"
        )

    base_witness = getattr(_propagate_sparse, "last_base_witness", {})
    if base_witness.get("ok"):
        base_hz_feasible = True
        base_hz_feas_msg = str(base_witness.get("msg", "constructive_center"))
        base_xi = base_witness.get("xi")
    else:
        base_status, base_msg = _solver_hz_base_feasibility(
            hz, time_limit=float(args.base_feas_timeout)
        )
        base_hz_feasible = base_status == "FEASIBLE"
        base_hz_feas_msg = f"{base_status}:{base_msg}"
        base_xi = None
        if base_witness:
            base_hz_feas_msg = f"{base_hz_feas_msg}; witness={base_witness.get('msg')}"
    print(
        f"base_hz_feasible={base_hz_feasible} msg={base_hz_feas_msg[:160]}",
        flush=True,
    )

    if args.query_indices.strip():
        order = np.asarray([int(x) for x in args.query_indices.split(",") if x.strip()], dtype=int)
    else:
        order = np.argsort(np.asarray(lows)) if lows else np.arange(len(queries))
    lp_n = min(args.lp_queries, len(order))
    center, rad, input_idx = _input_center_rad(queries[0][0])

    def witness_input(xi: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if xi is None:
            return None
        if xi.size < input_idx.size:
            return None
        x = center.copy()
        vals = np.clip(xi[:input_idx.size], -1.0, 1.0)
        x[input_idx] = center[input_idx] + rad[input_idx] * vals
        return x

    def check_witness(xi: Optional[np.ndarray], C: np.ndarray, t: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if not args.check_witness:
            return False, None
        x = witness_input(xi)
        if x is None:
            return False, None
        try:
            return _check_real_unsafe(onnx_path, input_shape, x, C, t)
        except Exception as exc:
            print(f"  WITNESS_CHECK_ERROR {type(exc).__name__}:{str(exc)[:120]}", flush=True)
            return False, None

    cert_count = hz_unsafe_count = unknown_count = real_adv_count = 0
    checked_count = 0
    query_results: List[dict] = []
    if unsupported_spec:
        print(f"  EARLY_STOP unsupported_spec {unsupported_spec}", flush=True)
        lp_n = 0
        unknown_count = 1
    if not base_hz_feasible:
        print(f"  EARLY_STOP base_hz_infeasible {base_hz_feas_msg[:160]}", flush=True)
        lp_n = 0
        unknown_count = 1
    for rank, qidx in enumerate(order[:lp_n]):
        checked_count += 1
        C, t, _ = flat_specs[int(qidx)]
        record = {
            "rank": int(rank),
            "q": int(qidx),
        }
        ts = time.time()
        if args.skip_lp_before_milp:
            margin, msg = None, "skipped"
            lp_sec = 0.0
        else:
            margin, msg = _lp_min_margin(hz, C, t, args.lp_timeout)
            lp_sec = time.time() - ts
        record.update({
            "lp_margin": None if margin is None else float(margin),
            "lp_status": msg,
            "lp_sec": lp_sec,
        })
        print(f"LP rank={rank} q={int(qidx)} margin={margin} msg={msg} sec={lp_sec:.2f}", flush=True)
        lp_safe = (margin is not None) and (margin > 1e-9)
        if lp_safe:
            cert_count += 1
            print(f"  LP-SAFE q={int(qidx)} margin={margin}", flush=True)
            record.update({"verdict": "cert", "cert_source": "lp_safe"})
            query_results.append(record)
            continue
        real_bad = False
        run_milp = args.milp_all
        if run_milp:
            ts = time.time()
            stop_after_query = False
            if args.mip_solver == "scip":
                status, mi, xi_mi = _milp_cutoff_scip(
                    hz, C, t, args.milp_timeout,
                    elim_singletons=args.elim_singletons,
                    cutoff_as_row=args.cutoff_as_row,
                    fbbt_passes=args.fbbt_passes,
                    scip_threads=args.scip_threads,
                    scip_options=extra_scip_options,
                )
                milp_stats = dict(getattr(_milp_cutoff_scip, "last_stats", {}))
                attempts = [{
                    "profile": "scip",
                    "status": status,
                    "margin": None if mi is None else float(mi),
                    "sec": time.time() - ts,
                    "milp_stats": milp_stats,
                }]
            else:
                attempts = []
                status = ""
                mi = None
                xi_mi = None
                milp_stats = {}
                for profile_name, profile_options in highs_profiles:
                    ats = time.time()
                    status, mi, xi_mi = _milp_cutoff_highs(
                        hz, C, t, args.milp_timeout,
                        elim_singletons=args.elim_singletons,
                        highs_threads=args.highs_threads,
                        highs_parallel=args.highs_parallel,
                        highs_heuristic_effort=args.highs_heuristic_effort,
                        cutoff_as_row=args.cutoff_as_row,
                        highs_options=profile_options,
                        mip_start_xi=(base_xi if args.mip_start.startswith("base") else None),
                        mip_start_binary_only=(args.mip_start == "base-binary"),
                        connected_presolve=args.connected_presolve,
                        base_xi=base_xi,
                        elim_eq_subst=args.elim_eq_subst,
                        fbbt_passes=args.fbbt_passes,
                        relax_precheck_timeout=args.relax_precheck_timeout,
                    )
                    milp_stats = dict(getattr(_milp_cutoff_highs, "last_stats", {}))
                    attempt = {
                        "profile": profile_name,
                        "status": status,
                        "margin": None if mi is None else float(mi),
                        "sec": time.time() - ats,
                        "milp_stats": milp_stats,
                    }
                    attempts.append(attempt)
                    print(
                        f"  highs_profile profile={profile_name} status={status} "
                        f"margin={mi} sec={attempt['sec']:.2f}",
                        flush=True,
                    )
                    if status.startswith("TARGET") or (status.startswith("OPTIMAL") and mi is not None and mi <= 1e-9):
                        break
                    if status.startswith("EMPTY:") or (status.startswith("OPTIMAL") and mi is not None and mi > 1e-9):
                        break
            milp_sec = time.time() - ts
            print(f"MILP-CUTOFF q={int(qidx)} status={status} margin={mi} sec={time.time() - ts:.2f}", flush=True)
            record.update({
                "milp_status": status,
                "milp_margin": None if mi is None else float(mi),
                "milp_sec": milp_sec,
                "milp_stats": milp_stats,
            })
            if args.mip_solver in {"highs", "scip"}:
                record["milp_attempts"] = attempts
            if status.startswith("TARGET") or (status.startswith("OPTIMAL") and mi is not None and mi <= 1e-9):
                print("  HZ_TARGET: exact-HZ unsafe feasible point found", flush=True)
                real_bad, cy = check_witness(xi_mi, C, t)
                if real_bad:
                    real_adv_count += 1
                    print(
                        f"  REAL_ADV q={int(qidx)} source=milp cy={cy.tolist() if cy is not None else None}",
                        flush=True,
                    )
                hz_unsafe_count += 1
                record.update({
                    "verdict": "adv" if real_bad else "hz_unsafe",
                    "cert_source": "milp",
                    "witness_checked": bool(args.check_witness and xi_mi is not None),
                    "real_unsafe": bool(real_bad),
                })
                if args.stop_on_unsafe and (not args.check_witness or real_adv_count):
                    print("  EARLY_STOP unsafe", flush=True)
                    stop_after_query = True
            elif status.startswith("EMPTY:") or (status.startswith("OPTIMAL") and mi is not None and mi > 1e-9):
                cert_count += 1
                record.update({"verdict": "cert", "cert_source": "milp"})
            else:
                unknown_count += 1
                record.update({"verdict": "unknown", "cert_source": "milp"})
            query_results.append(record)
            if stop_after_query:
                break
        else:
            record.update({"verdict": "not_run", "cert_source": "lp_only"})
            query_results.append(record)

    if args.milp_all:
        print(
            f"verdict_summary checked={checked_count} cert={cert_count} "
            f"hz_unsafe={hz_unsafe_count} unknown={unknown_count} real_adv={real_adv_count}",
            flush=True,
        )

    instance_status = "unknown"
    if real_adv_count:
        instance_status = "adv"
    elif hz_unsafe_count:
        instance_status = "hz_unsafe"
    elif flat_specs and checked_count == len(flat_specs) and cert_count == len(flat_specs):
        instance_status = "verified"
    if args.summary_json:
        summary = {
            "bench": args.bench,
            "iid": args.iid,
            "onnx": onnx_path.name,
            "vnnlib": vnnlib_path.name,
            "queries": len(queries),
            "flat_queries": len(flat_specs),
            "checked": checked_count,
            "cert": cert_count,
            "hz_unsafe": hz_unsafe_count,
            "real_adv": real_adv_count,
            "unknown": unknown_count,
            "instance_status": instance_status,
            "base_hz_feasible": bool(base_hz_feasible),
            "base_hz_feas_msg": base_hz_feas_msg,
            "n_out": hz.n_out,
            "n_cont": hz.n_cont,
            "n_bin": hz.n_bin,
            "n_eq": hz.n_eq,
            "value_nnz": hz.value_nnz,
            "eq_nnz": hz.eq_nnz,
            "n_ub": hz.n_ub,
            "ub_nnz": hz.ub_nnz,
            "tight_schedule": args.tight_schedule,
            "tight_relu_fix_mode": args.tight_relu_fix_mode,
            "highs_options": args.highs_option,
            "fbbt_passes": int(args.fbbt_passes),
            "relax_precheck_timeout": float(args.relax_precheck_timeout),
            "relu_cuts": bool(args.relu_cuts),
            "compressed_relu": bool(args.compressed_relu),
            "compressed_sigmoid": bool(args.compressed_sigmoid),
            "scurve_domain_cuts": bool(args.scurve_domain_cuts),
            "scurve_graph_cuts": bool(args.scurve_graph_cuts),
            "scurve_grid": args.scurve_grid,
            "sigmoid_k": int(args.sigmoid_k),
            "tanh_k": int(args.tanh_k),
            "sigmoid_prune_degenerate": bool(args.sigmoid_prune_degenerate),
            "query_results": query_results,
            "wall_s": time.time() - t0,
        }
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"total_wall_s={time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
