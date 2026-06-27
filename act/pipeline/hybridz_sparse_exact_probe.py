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
from scipy.optimize import linprog

from act.back_end.solver.sparse_hz import SparseHZono as SparseHZ  # noqa: E402
from act.back_end.solver.solver_hz_verdict import (  # noqa: E402
    hz_base_feasibility as _solver_hz_base_feasibility,
    sparse_highs_relaxation_empty_precheck as _highs_relaxation_empty_precheck,
    sparse_fbbt_tighten_bounds as _fbbt_tighten_bounds,
    sparse_row_bound_infeasible as _row_bound_infeasible,
    sparse_solver_start_from_xi as _solver_start_from_xi,
)
from act.pipeline.hybridz_option_utils import parse_key_value_options  # noqa: E402
from act.pipeline.hybridz_spec_utils import (  # noqa: E402
    check_real_unsafe as _check_real_unsafe,
    flatten_query_specs as _flatten_query_specs,
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


def _lp_min_margin(
    hz: SparseHZ,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
) -> Tuple[Optional[float], str]:
    Cmat = C.reshape(C.shape[0], -1)
    tvec = t.reshape(-1)
    if Cmat.shape[0] != tvec.size:
        raise ValueError(f"C/t row mismatch: {Cmat.shape} vs {tvec.shape}")
    n_base = hz.n_cont + hz.n_bin
    if n_base == 0:
        if hz.n_eq and np.max(np.abs(hz.b)) > 1e-8:
            return None, "fixed_hz_infeasible_eq"
        if hz.n_ub and np.min(hz.ub) < -1e-8:
            return None, "fixed_hz_infeasible_ub"
        vals = np.asarray(Cmat @ hz.c - tvec, dtype=np.float64).reshape(-1)
        margin = float(np.max(vals) if vals.size > 1 else vals[0])
        return margin, "fixed"
    if Cmat.shape[0] > 1:
        # UNSAFE_LINEAR is an AND polytope C y <= t.  The LP relaxation proves
        # safety only if min max_i(C_i y - t_i) > 0.
        Csp = sp.csr_matrix(Cmat)
        CGc = Csp @ hz.Gc
        CGb = Csp @ hz.Gb if hz.n_bin else sp.csr_matrix((Cmat.shape[0], 0), dtype=np.float64)
        epi = sp.hstack([CGc, CGb, -sp.csr_matrix(np.ones((Cmat.shape[0], 1)))], format="csr")
        epi_b = tvec - Cmat @ hz.c
        Aeq = None
        if hz.n_eq:
            Aeq = sp.hstack([hz.Ac, hz.Ab, sp.csr_matrix((hz.n_eq, 1))], format="csr")
        Aub_rows = [epi]
        bub_rows = [epi_b]
        if hz.n_ub:
            Aub_rows.append(sp.hstack([hz.Auc, hz.Aub, sp.csr_matrix((hz.n_ub, 1))], format="csr"))
            bub_rows.append(hz.ub)
        bounds = [(-1.0, 1.0)] * n_base + [(-1e12, 1e12)]
        obj = np.zeros(n_base + 1, dtype=np.float64)
        obj[-1] = 1.0
        opts = {"time_limit": float(time_limit)} if time_limit > 0 else None
        r = linprog(
            obj,
            A_eq=Aeq,
            b_eq=hz.b if hz.n_eq else None,
            A_ub=sp.vstack(Aub_rows, format="csr"),
            b_ub=np.concatenate(bub_rows),
            bounds=bounds,
            method="highs",
            options=opts,
        )
        if not r.success:
            return None, str(r.message)
        return float(r.fun), "ok"

    c_row = Cmat[0]
    obj_c = np.asarray(c_row @ hz.Gc).reshape(-1)
    obj_b = np.asarray(c_row @ hz.Gb).reshape(-1) if hz.n_bin else np.zeros(0)
    const = float(c_row @ hz.c - tvec[0])
    obj = np.concatenate([obj_c, obj_b])
    Aeq = sp.hstack([hz.Ac, hz.Ab], format="csr") if hz.n_eq else None
    Aub = sp.hstack([hz.Auc, hz.Aub], format="csr") if hz.n_ub else None
    bub = hz.ub if hz.n_ub else None
    bounds = [(-1.0, 1.0)] * (hz.n_cont + hz.n_bin)
    opts = {"time_limit": float(time_limit)} if time_limit > 0 else None
    r = linprog(
        obj,
        A_eq=Aeq,
        b_eq=hz.b if hz.n_eq else None,
        A_ub=Aub,
        b_ub=bub,
        bounds=bounds,
        method="highs",
        options=opts,
    )
    if not r.success:
        return None, str(r.message)
    return const + float(r.fun), "ok"


def _input_center_rad(inspec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lb = inspec.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ub = inspec.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    center = (lb + ub) * 0.5
    rad = (ub - lb) * 0.5
    idx = np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
    return center, rad, idx


def _milp_cutoff_highs(
    hz: SparseHZ,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    *,
    cutoff: float = 0.0,
    elim_singletons: bool = False,
    highs_threads: int = 0,
    highs_parallel: str = "",
    highs_heuristic_effort: Optional[float] = None,
    cutoff_as_row: bool = False,
    highs_options: Optional[Dict[str, object]] = None,
    mip_start_xi: Optional[np.ndarray] = None,
    mip_start_binary_only: bool = False,
    connected_presolve: bool = False,
    base_xi: Optional[np.ndarray] = None,
    elim_eq_subst: bool = False,
    fbbt_passes: int = 0,
    relax_precheck_timeout: float = 0.0,
) -> Tuple[str, Optional[float], Optional[np.ndarray]]:
    try:
        import highspy
    except Exception as exc:
        return f"no_highspy:{exc}", None, None

    Cmat = C.reshape(C.shape[0], -1)
    tvec = t.reshape(-1)
    if Cmat.shape[0] != tvec.size:
        raise ValueError(f"C/t row mismatch: {Cmat.shape} vs {tvec.shape}")
    multirow_feas = Cmat.shape[0] > 1
    base_ncols = hz.n_cont + hz.n_bin
    extra_epigraph_cols = 1 if multirow_feas else 0
    if multirow_feas:
        # Exact conjunctive unsafe set C y <= t is equivalent to
        # min_s s subject to C_i y - t_i <= s.  This gives HiGHS a real
        # objective-bound search direction instead of a zero-objective
        # feasibility MILP, while keeping the same exact HZ semantics.
        cost = np.zeros(base_ncols + 1, dtype=np.float64)
        cost[-1] = 1.0
        const_z = 0.0
        obj_thr = float(cutoff)
    else:
        c_row = Cmat[0]
        obj_c = np.asarray(c_row @ hz.Gc).reshape(-1)
        obj_b = np.asarray(c_row @ hz.Gb).reshape(-1) if hz.n_bin else np.zeros(0)
        const = float(c_row @ hz.c - tvec[0])
        cost = np.concatenate([obj_c, 2.0 * obj_b])
        const_z = const - float(obj_b.sum())
        obj_thr = float(cutoff - const_z)

    if hz.n_eq:
        A = sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr")
        rhs = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
    else:
        A = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        rhs = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        A = sp.hstack([A, sp.csr_matrix((A.shape[0], extra_epigraph_cols))], format="csr")
    if hz.n_ub:
        Ale = sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr")
        ble = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
    else:
        Ale = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        ble = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        Ale = sp.hstack([Ale, sp.csr_matrix((Ale.shape[0], extra_epigraph_cols))], format="csr")
    if multirow_feas:
        Csp = sp.csr_matrix(Cmat)
        unsafe_Ac = Csp @ hz.Gc
        unsafe_Ab = Csp @ hz.Gb if hz.n_bin else sp.csr_matrix((Cmat.shape[0], 0), dtype=np.float64)
        unsafe_A = sp.hstack(
            [
                unsafe_Ac,
                2.0 * unsafe_Ab,
                -sp.csr_matrix(np.ones((Cmat.shape[0], 1))),
            ],
            format="csr",
        )
        unsafe_b = tvec - Cmat @ hz.c + np.asarray(unsafe_Ab.sum(axis=1)).reshape(-1)
        Ale = sp.vstack([Ale, unsafe_A], format="csr")
        ble = np.concatenate([ble, unsafe_b.astype(np.float64)])

    lb = np.concatenate([-np.ones(hz.n_cont), np.zeros(hz.n_bin)])
    ub = np.ones(base_ncols)
    if extra_epigraph_cols:
        lb = np.concatenate([lb, np.array([-1e12], dtype=np.float64)])
        ub = np.concatenate([ub, np.array([1e12], dtype=np.float64)])

    mip_start_values: Optional[np.ndarray] = None
    mip_start_indices: Optional[np.ndarray] = None
    mip_start_status = "off"
    if mip_start_xi is not None:
        raw_start = np.asarray(mip_start_xi, dtype=np.float64).reshape(-1)
        if raw_start.size < base_ncols:
            mip_start_status = f"skipped:short_xi:{raw_start.size}<{base_ncols}"
        elif mip_start_binary_only and not hz.n_bin:
            mip_start_status = "skipped:no_binary"
        else:
            start = np.zeros(cost.size, dtype=np.float64)
            start[:hz.n_cont] = np.clip(raw_start[:hz.n_cont], -1.0, 1.0)
            if hz.n_bin:
                relaxed = np.clip(raw_start[hz.n_cont:base_ncols], -1.0, 1.0)
                start[hz.n_cont:base_ncols] = (relaxed >= 0.0).astype(np.float64)
            if extra_epigraph_cols:
                xi_c = start[:hz.n_cont]
                if hz.n_bin:
                    xi_b = 2.0 * start[hz.n_cont:base_ncols] - 1.0
                    y = hz.c + np.asarray(hz.Gc @ xi_c).reshape(-1) + np.asarray(hz.Gb @ xi_b).reshape(-1)
                else:
                    y = hz.c + np.asarray(hz.Gc @ xi_c).reshape(-1)
                start[-1] = float(np.max(Cmat @ y - tvec))
            start = np.clip(start, lb, ub)
            if np.all(np.isfinite(start)):
                mip_start_values = start
                if mip_start_binary_only:
                    mip_start_indices = np.arange(hz.n_cont, hz.n_cont + hz.n_bin, dtype=np.int64)
                mip_start_status = "prepared"
            else:
                mip_start_status = "skipped:nonfinite"

    rl = rhs.astype(np.float64).copy()
    ru = rhs.astype(np.float64).copy()
    keep_cols = None
    original_ncols = int(cost.size)
    original_margin_cost = cost.copy()
    original_const_z = float(const_z)
    reconstruct_base = _solver_start_from_xi(base_xi, original_ncols, hz.n_cont, hz.n_bin)
    solve_obj_offset = 0.0
    elim_count = 0
    eq_subst_count = 0
    fixed_subst_stats = None
    if elim_eq_subst and A.shape[0] and A.shape[1]:
        Aeq = A.tocsr()
        Aeq_csc = Aeq.tocsc()
        Ale_csr = Ale.tocsr()
        Ale_csc = Ale_csr.tocsc() if Ale_csr.shape[0] else sp.csc_matrix((0, Aeq.shape[1]))
        used_rows: set[int] = set()
        pivots: List[Tuple[int, int, float]] = []
        for j in range(hz.n_cont):
            if abs(float(cost[j])) > 1e-12:
                continue
            es, ee = Aeq_csc.indptr[j], Aeq_csc.indptr[j + 1]
            if ee - es != 1:
                continue
            r = int(Aeq_csc.indices[es])
            if r in used_rows:
                continue
            ls, le = Ale_csc.indptr[j], Ale_csc.indptr[j + 1]
            # Keep this presolve conservative. The ReLU xi1 column has one
            # projected inequality; broader columns can densify the root LP.
            if le - ls > 2:
                continue
            a = float(Aeq_csc.data[es])
            if abs(a) < 1e-12:
                continue
            used_rows.add(r)
            pivots.append((j, r, a))
        if pivots:
            pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
            pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
            pivot_col_set = set(int(x) for x in pivot_cols)
            pivot_row_set = set(int(x) for x in pivot_rows)
            corr_rr: List[np.ndarray] = []
            corr_cc: List[np.ndarray] = []
            corr_dd: List[np.ndarray] = []
            bound_rr: List[np.ndarray] = []
            bound_cc: List[np.ndarray] = []
            bound_dd: List[np.ndarray] = []
            bound_rhs: List[float] = []
            ble = ble.astype(np.float64, copy=True)
            skipped = 0

            for out_i, (j, r, a) in enumerate(pivots):
                row = Aeq.getrow(r).tocoo()
                mask = row.col != j
                cols = row.col[mask].astype(np.int64, copy=False)
                data = row.data[mask].astype(np.float64, copy=False)
                if any(int(c) in pivot_col_set for c in cols):
                    skipped += 1
                    pivot_col_set.discard(int(j))
                    pivot_row_set.discard(int(r))
                    continue
                rhs_j = float(rhs[r])
                const = rhs_j / a
                coeff = -data / a
                ls, le = Ale_csc.indptr[j], Ale_csc.indptr[j + 1]
                if le > ls and cols.size:
                    rows_i = Ale_csc.indices[ls:le].astype(np.int64, copy=False)
                    vals_i = Ale_csc.data[ls:le].astype(np.float64, copy=False)
                    for row_i, val_i in zip(rows_i, vals_i):
                        corr_rr.append(np.full(cols.size, int(row_i), dtype=np.int32))
                        corr_cc.append(cols.astype(np.int32, copy=False))
                        corr_dd.append((float(val_i) * coeff).astype(np.float64, copy=False))
                        ble[int(row_i)] -= float(val_i) * const
                if cols.size:
                    br = 2 * out_i
                    # x_j <= ub_j
                    bound_rr.append(np.full(cols.size, br, dtype=np.int32))
                    bound_cc.append(cols.astype(np.int32, copy=False))
                    bound_dd.append(coeff.astype(np.float64, copy=False))
                    bound_rhs.append(float(ub[j]) - const)
                    # -x_j <= -lb_j
                    bound_rr.append(np.full(cols.size, br + 1, dtype=np.int32))
                    bound_cc.append(cols.astype(np.int32, copy=False))
                    bound_dd.append((-coeff).astype(np.float64, copy=False))
                    bound_rhs.append(-float(lb[j]) + const)
                else:
                    # Constant pivot: preserve feasibility through scalar bounds.
                    bound_rhs.extend([float(ub[j]) - const, -float(lb[j]) + const])

            if skipped:
                pivots = [(j, r, a) for (j, r, a) in pivots if int(j) in pivot_col_set]
                pivot_cols = np.asarray([p[0] for p in pivots], dtype=np.int64)
                pivot_rows = np.asarray([p[1] for p in pivots], dtype=np.int64)
            if pivot_cols.size:
                if corr_rr:
                    corr = sp.coo_matrix(
                        (np.concatenate(corr_dd), (np.concatenate(corr_rr), np.concatenate(corr_cc))),
                        shape=Ale_csr.shape,
                    ).tocsr()
                    Ale_csr = (Ale_csr + corr).tocsr()
                    Ale_csr.eliminate_zeros()
                if bound_rhs:
                    n_bound = len(bound_rhs)
                    if bound_rr:
                        bound_A = sp.coo_matrix(
                            (np.concatenate(bound_dd), (np.concatenate(bound_rr), np.concatenate(bound_cc))),
                            shape=(n_bound, Aeq.shape[1]),
                        ).tocsr()
                    else:
                        bound_A = sp.csr_matrix((n_bound, Aeq.shape[1]), dtype=np.float64)
                    Ale_csr = sp.vstack([Ale_csr, bound_A], format="csr")
                    ble = np.concatenate([ble, np.asarray(bound_rhs, dtype=np.float64)])
                keep = np.ones(Aeq.shape[1], dtype=bool)
                keep[pivot_cols] = False
                keep_rows = np.ones(Aeq.shape[0], dtype=bool)
                keep_rows[pivot_rows] = False
                keep_idx = np.nonzero(keep)[0]
                A = Aeq[keep_rows, :][:, keep_idx].tocsr()
                rhs = rhs[keep_rows]
                Ale = Ale_csr[:, keep_idx].tocsr()
                cost = cost[keep_idx]
                lb = lb[keep_idx]
                ub = ub[keep_idx]
                keep_cols = keep_idx.astype(np.int64)
                rl = rhs.astype(np.float64).copy()
                ru = rhs.astype(np.float64).copy()
                if mip_start_values is not None:
                    mip_start_values = mip_start_values[keep_cols]
                    if mip_start_indices is not None:
                        reverse = {int(col): int(pos) for pos, col in enumerate(keep_cols)}
                        mip_start_indices = np.asarray(
                            [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                            dtype=np.int64,
                        )
                        if mip_start_indices.size == 0:
                            mip_start_values = None
                            mip_start_status = "skipped:no_start_cols_after_eq_subst"
                eq_subst_count = int(pivot_cols.size)
                print(
                    "  elim_eq_subst "
                    f"removed_cont={eq_subst_count} removed_eq={eq_subst_count} "
                    f"kept_cols={A.shape[1]} eq_rows={A.shape[0]} "
                    f"ineq_rows={Ale.shape[0]} skipped={skipped}",
                    flush=True,
                )
    if elim_singletons and A.shape[0] and A.shape[1]:
        Acsc = A.tocsc()
        Alecsc = Ale.tocsc() if Ale.shape[0] else None
        removable = []
        current_orig_cols = (
            np.arange(A.shape[1], dtype=np.int64)
            if keep_cols is None
            else np.asarray(keep_cols, dtype=np.int64)
        )
        current_cont_positions = np.nonzero(current_orig_cols < hz.n_cont)[0]
        for j in current_cont_positions:
            if abs(cost[int(j)]) > 1e-12:
                continue
            if Alecsc is not None and Alecsc.indptr[int(j) + 1] > Alecsc.indptr[int(j)]:
                continue
            start, end = Acsc.indptr[int(j)], Acsc.indptr[int(j) + 1]
            if end - start == 1:
                removable.append(int(j))
        if removable:
            elim_min = np.zeros(A.shape[0], dtype=np.float64)
            elim_max = np.zeros(A.shape[0], dtype=np.float64)
            for j in removable:
                start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
                r = int(Acsc.indices[start])
                a = float(Acsc.data[start])
                vals = (a * lb[j], a * ub[j])
                elim_min[r] += min(vals)
                elim_max[r] += max(vals)
            rl = rhs - elim_max
            ru = rhs - elim_min
            keep = np.ones(A.shape[1], dtype=bool)
            keep[np.asarray(removable, dtype=np.int64)] = False
            local_keep_cols = np.nonzero(keep)[0]
            A = A[:, local_keep_cols].tocsr()
            Ale = Ale[:, local_keep_cols].tocsr()
            cost = cost[local_keep_cols]
            lb = lb[local_keep_cols]
            ub = ub[local_keep_cols]
            keep_cols = local_keep_cols if keep_cols is None else keep_cols[local_keep_cols]
            if mip_start_values is not None:
                mip_start_values = mip_start_values[local_keep_cols]
                if mip_start_indices is not None:
                    reverse = {int(col): int(pos) for pos, col in enumerate(local_keep_cols)}
                    mip_start_indices = np.asarray(
                        [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                        dtype=np.int64,
                    )
                    if mip_start_indices.size == 0:
                        mip_start_values = None
                        mip_start_status = "skipped:no_start_cols_after_elim"
            elim_count = len(removable)
            print(f"  elim_singletons removed_cont={elim_count} kept_cols={A.shape[1]} rows={A.shape[0]}", flush=True)
    if Ale.shape[0]:
        A = sp.vstack([A, Ale], format="csr")
        rl = np.concatenate([rl, np.full(Ale.shape[0], -1e30, dtype=np.float64)])
        ru = np.concatenate([ru, ble.astype(np.float64)])
    margin_cost = cost.copy()
    if cutoff_as_row and not multirow_feas:
        A = sp.vstack([A, sp.csr_matrix(margin_cost.reshape(1, -1))], format="csr")
        rl = np.concatenate([rl, np.array([-1e30], dtype=np.float64)])
        ru = np.concatenate([ru, np.array([obj_thr], dtype=np.float64)])
        solve_cost = np.zeros_like(margin_cost)
    else:
        solve_cost = margin_cost
    connected_stats = None
    if connected_presolve and A.shape[0] and A.shape[1]:
        root = np.flatnonzero(
            (np.abs(margin_cost) > 1e-12) | (np.abs(solve_cost) > 1e-12)
        )
        if root.size:
            A_csr = A.tocsr()
            A_csc = A_csr.tocsc()
            keep_col = np.zeros(A_csr.shape[1], dtype=bool)
            keep_row = np.zeros(A_csr.shape[0], dtype=bool)
            stack = [int(x) for x in root]
            for x in stack:
                keep_col[x] = True
            while stack:
                col = stack.pop()
                for p in range(A_csc.indptr[col], A_csc.indptr[col + 1]):
                    row = int(A_csc.indices[p])
                    if keep_row[row]:
                        continue
                    keep_row[row] = True
                    for q in range(A_csr.indptr[row], A_csr.indptr[row + 1]):
                        nxt = int(A_csr.indices[q])
                        if not keep_col[nxt]:
                            keep_col[nxt] = True
                            stack.append(nxt)
            conn_cols = np.flatnonzero(keep_col)
            conn_rows = np.flatnonzero(keep_row)
            connected_stats = {
                "cols_before": int(A.shape[1]),
                "rows_before": int(A.shape[0]),
                "nnz_before": int(A.nnz),
                "cols_after": int(conn_cols.size),
                "rows_after": int(conn_rows.size),
            }
            if conn_cols.size < A.shape[1] or conn_rows.size < A.shape[0]:
                prev_cols, prev_rows, prev_nnz = A.shape[1], A.shape[0], A.nnz
                A = A[conn_rows, :][:, conn_cols].tocsr()
                connected_stats["nnz_after"] = int(A.nnz)
                rl = rl[conn_rows]
                ru = ru[conn_rows]
                cost = cost[conn_cols]
                margin_cost = margin_cost[conn_cols]
                solve_cost = solve_cost[conn_cols]
                lb = lb[conn_cols]
                ub = ub[conn_cols]
                if keep_cols is None:
                    keep_cols = conn_cols.astype(np.int64)
                else:
                    keep_cols = keep_cols[conn_cols]
                if mip_start_values is not None:
                    mip_start_values = mip_start_values[conn_cols]
                    if mip_start_indices is not None:
                        reverse = {int(col): int(pos) for pos, col in enumerate(conn_cols)}
                        mip_start_indices = np.asarray(
                            [reverse[int(col)] for col in mip_start_indices if int(col) in reverse],
                            dtype=np.int64,
                        )
                        if mip_start_indices.size == 0:
                            mip_start_values = None
                            mip_start_status = "skipped:no_start_cols_after_connected"
                print(
                    "  connected_presolve "
                    f"cols={prev_cols}->{A.shape[1]} rows={prev_rows}->{A.shape[0]} "
                    f"nnz={prev_nnz}->{A.nnz}",
                    flush=True,
                )
            else:
                connected_stats["nnz_after"] = int(A.nnz)
                print(
                    "  connected_presolve no_reduction "
                    f"cols={A.shape[1]} rows={A.shape[0]} nnz={A.nnz}",
                    flush=True,
                )
    int_mask = None
    if A.shape[1]:
        if keep_cols is None:
            col_orig = np.arange(A.shape[1], dtype=np.int64)
        else:
            col_orig = np.asarray(keep_cols, dtype=np.int64)
        int_mask = (col_orig >= hz.n_cont) & (col_orig < hz.n_cont + hz.n_bin)
    fbbt_stats = None
    if int(fbbt_passes) > 0 and A.shape[0] and A.shape[1]:
        fbbt_empty, lb, ub, fbbt_stats = _fbbt_tighten_bounds(
            A, rl, ru, lb, ub,
            integer_mask=int_mask,
            max_passes=int(fbbt_passes),
        )
        print(
            "  fbbt_presolve "
            f"status={'infeasible' if fbbt_empty else 'ok'} "
            f"passes={fbbt_stats.get('passes')} "
            f"tightened={fbbt_stats.get('tightened')} "
            f"fixed_int={fbbt_stats.get('fixed_int')} "
            f"max_width_delta={fbbt_stats.get('max_width_delta')}",
            flush=True,
        )
        if fbbt_empty:
            stats = {
                "status": "fbbt_infeasible",
                "nodes": 0,
                "dual_bound": None,
                "obj": None,
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "const_z": const_z,
                "margin_dual_bound": None,
                "elim_singletons": elim_count,
                "elim_eq_subst": eq_subst_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "mip_start": bool(mip_start_values is not None),
                "mip_start_status": mip_start_status,
                "connected_presolve": connected_stats,
                "fbbt": fbbt_stats,
                "fbbt_fixed_subst": fixed_subst_stats,
            }
            _milp_cutoff_highs.last_stats = stats
            return "EMPTY:fbbt_infeasible", None, None
        if mip_start_values is not None:
            mip_start_values = np.clip(mip_start_values, lb, ub)
        fixed_mask = np.isfinite(lb) & np.isfinite(ub) & ((ub - lb) <= 1e-9)
        if np.any(fixed_mask):
            fixed_pos = np.flatnonzero(fixed_mask)
            fixed_vals = 0.5 * (lb[fixed_pos] + ub[fixed_pos])
            current_cols = (
                np.arange(A.shape[1], dtype=np.int64)
                if keep_cols is None
                else np.asarray(keep_cols, dtype=np.int64)
            )
            fixed_orig = current_cols[fixed_pos]
            contrib = np.asarray(A[:, fixed_pos] @ fixed_vals, dtype=np.float64).reshape(-1)
            rl = rl - contrib
            ru = ru - contrib
            solve_obj_offset += float(solve_cost[fixed_pos] @ fixed_vals)
            reconstruct_base[fixed_orig] = fixed_vals

            keep_local = np.ones(A.shape[1], dtype=bool)
            keep_local[fixed_pos] = False
            before_cols, before_rows, before_nnz = int(A.shape[1]), int(A.shape[0]), int(A.nnz)
            A = A[:, keep_local].tocsr()
            A.eliminate_zeros()
            cost = cost[keep_local]
            margin_cost = margin_cost[keep_local]
            solve_cost = solve_cost[keep_local]
            lb = lb[keep_local]
            ub = ub[keep_local]
            keep_cols = current_cols[keep_local]
            if mip_start_values is not None:
                mip_start_values = mip_start_values[keep_local]
                if mip_start_indices is not None:
                    old_to_new = -np.ones(keep_local.size, dtype=np.int64)
                    old_to_new[np.flatnonzero(keep_local)] = np.arange(int(np.count_nonzero(keep_local)))
                    mip_start_indices = np.asarray(
                        [
                            int(old_to_new[int(col)])
                            for col in mip_start_indices
                            if 0 <= int(col) < old_to_new.size and old_to_new[int(col)] >= 0
                        ],
                        dtype=np.int64,
                    )
                    if mip_start_indices.size == 0:
                        mip_start_values = None
                        mip_start_status = "skipped:no_start_cols_after_fbbt_fixed"

            row_nnz = np.diff(A.indptr)
            zero_rows = np.flatnonzero(row_nnz == 0)
            dropped_zero_rows = 0
            if zero_rows.size:
                bad_zero = zero_rows[
                    ((np.isfinite(rl[zero_rows])) & (rl[zero_rows] > 1e-9))
                    | ((np.isfinite(ru[zero_rows])) & (ru[zero_rows] < -1e-9))
                ]
                if bad_zero.size:
                    r0 = int(bad_zero[0])
                    fixed_subst_stats = {
                        "fixed_cols": int(fixed_pos.size),
                        "cols_before": before_cols,
                        "cols_after": int(A.shape[1]),
                        "rows_before": before_rows,
                        "rows_after": int(A.shape[0]),
                        "nnz_before": before_nnz,
                        "nnz_after": int(A.nnz),
                        "zero_row_infeasible": {
                            "bad_row": r0,
                            "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                            "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                        },
                    }
                    stats = {
                        "status": "fbbt_fixed_zero_row_infeasible",
                        "nodes": 0,
                        "dual_bound": None,
                        "obj": None,
                        "gap": None,
                        "max_integrality": None,
                        "obj_thr": obj_thr,
                        "solver_obj_thr": obj_thr - solve_obj_offset,
                        "const_z": const_z,
                        "solve_obj_offset": solve_obj_offset,
                        "margin_dual_bound": None,
                        "elim_singletons": elim_count,
                        "elim_eq_subst": eq_subst_count,
                        "cutoff_as_row": bool(cutoff_as_row),
                        "mip_start": bool(mip_start_values is not None),
                        "mip_start_status": mip_start_status,
                        "connected_presolve": connected_stats,
                        "fbbt": fbbt_stats,
                        "fbbt_fixed_subst": fixed_subst_stats,
                    }
                    _milp_cutoff_highs.last_stats = stats
                    print(
                        "  fbbt_fixed_subst "
                        f"fixed_cols={fixed_pos.size} cols={before_cols}->{A.shape[1]} "
                        f"rows={before_rows}->{A.shape[0]} nnz={before_nnz}->{A.nnz} "
                        f"zero_row_infeasible={r0}",
                        flush=True,
                    )
                    return "EMPTY:fbbt_fixed_zero_row_infeasible", None, None
                keep_rows = np.ones(A.shape[0], dtype=bool)
                keep_rows[zero_rows] = False
                dropped_zero_rows = int(zero_rows.size)
                A = A[keep_rows, :].tocsr()
                rl = rl[keep_rows]
                ru = ru[keep_rows]

            fixed_subst_stats = {
                "fixed_cols": int(fixed_pos.size),
                "cols_before": before_cols,
                "cols_after": int(A.shape[1]),
                "rows_before": before_rows,
                "rows_after": int(A.shape[0]),
                "nnz_before": before_nnz,
                "nnz_after": int(A.nnz),
                "dropped_zero_rows": dropped_zero_rows,
                "solve_obj_offset": float(solve_obj_offset),
            }
            print(
                "  fbbt_fixed_subst "
                f"fixed_cols={fixed_pos.size} cols={before_cols}->{A.shape[1]} "
                f"rows={before_rows}->{A.shape[0]} nnz={before_nnz}->{A.nnz}",
                flush=True,
            )

    rb_empty, rb_stats = _row_bound_infeasible(A, rl, ru, lb, ub)
    if rb_empty:
        stats = {
            "status": "row_bound_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "const_z": const_z,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "elim_eq_subst": eq_subst_count,
            "cutoff_as_row": bool(cutoff_as_row),
            "mip_start": bool(mip_start_values is not None),
            "mip_start_status": mip_start_status,
            "connected_presolve": connected_stats,
            "fbbt": fbbt_stats,
            "fbbt_fixed_subst": fixed_subst_stats,
            "row_bound_infeasible": rb_stats,
        }
        _milp_cutoff_highs.last_stats = stats
        print(
            "  row_bound_infeasible "
            f"bad_row={rb_stats.get('bad_row')} "
            f"bad_count={rb_stats.get('bad_count')} "
            f"row_min={rb_stats.get('row_min')} "
            f"row_max={rb_stats.get('row_max')} "
            f"rl={rb_stats.get('rl')} ru={rb_stats.get('ru')}",
            flush=True,
        )
        return "EMPTY:row_bound_infeasible", None, None

    relax_stats = None
    if float(relax_precheck_timeout) > 0.0 and A.shape[0] and A.shape[1]:
        relax_status, relax_margin, relax_stats = _highs_relaxation_empty_precheck(
            highspy,
            A,
            rl,
            ru,
            lb,
            ub,
            solve_cost,
            cutoff=cutoff,
            const_z=const_z + solve_obj_offset,
            time_limit=float(relax_precheck_timeout),
            cutoff_as_row=bool(cutoff_as_row),
            multirow_feas=bool(multirow_feas),
            highs_threads=highs_threads,
            highs_parallel=highs_parallel,
            highs_options=highs_options,
        )
        print(
            "  relax_precheck "
            f"status={relax_status} margin={relax_margin} "
            f"sec={relax_stats.get('sec')} nodes={relax_stats.get('nodes')}",
            flush=True,
        )
        if relax_status.startswith("EMPTY:"):
            stats = {
                "status": relax_status,
                "nodes": 0,
                "dual_bound": relax_stats.get("dual_bound"),
                "obj": relax_stats.get("obj"),
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "const_z": const_z,
                "margin_dual_bound": relax_stats.get("margin_dual_bound"),
                "elim_singletons": elim_count,
                "elim_eq_subst": eq_subst_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "mip_start": bool(mip_start_values is not None),
                "mip_start_status": mip_start_status,
                "connected_presolve": connected_stats,
                "fbbt": fbbt_stats,
                "fbbt_fixed_subst": fixed_subst_stats,
                "relax_precheck": relax_stats,
            }
            _milp_cutoff_highs.last_stats = stats
            return relax_status, relax_margin, None
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-9)
    solver_obj_thr = float(obj_thr - solve_obj_offset)
    if (not cutoff_as_row) or multirow_feas:
        h.setOptionValue("objective_target", solver_obj_thr)
        h.setOptionValue("objective_bound", solver_obj_thr)
    h.setOptionValue("presolve", "on")
    if highs_threads > 0:
        h.setOptionValue("threads", int(highs_threads))
    if highs_parallel:
        h.setOptionValue("parallel", highs_parallel)
    if highs_heuristic_effort is not None:
        h.setOptionValue("mip_heuristic_effort", float(highs_heuristic_effort))
    if highs_options:
        for key, value in highs_options.items():
            h.setOptionValue(str(key), value)
    ncols = cost.size
    h.addCols(
        ncols,
        solve_cost.astype(float),
        lb.astype(float),
        ub.astype(float),
        0,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=float),
    )
    if keep_cols is None:
        int_idx = np.arange(hz.n_cont, hz.n_cont + hz.n_bin, dtype=np.int32)
    else:
        int_idx = np.nonzero((keep_cols >= hz.n_cont) & (keep_cols < hz.n_cont + hz.n_bin))[0].astype(np.int32)
    if int_idx.size:
        types = np.array([highspy.HighsVarType.kInteger] * int_idx.size)
        h.changeColsIntegrality(int_idx.size, int_idx, types)
    if A.shape[0]:
        h.addRows(
            A.shape[0],
            rl.astype(float),
            ru.astype(float),
            A.nnz,
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(float),
        )
    if mip_start_values is not None:
        start_entry_count = 0
        try:
            if mip_start_indices is None:
                start_indices = np.arange(ncols, dtype=np.int32)
            else:
                start_indices = mip_start_indices.astype(np.int32)
            start_entry_count = int(start_indices.size)
            start_values = np.clip(mip_start_values[start_indices], lb[start_indices], ub[start_indices]).astype(np.float64)
            ret = h.setSolution(start_indices.size, start_indices, start_values)
            mip_start_status = str(ret)
        except Exception as exc:
            mip_start_status = f"error:{type(exc).__name__}:{str(exc)[:80]}"
        print(
            f"  highs_mip_start status={mip_start_status} "
            f"entries={start_entry_count} "
            f"binary_only={bool(mip_start_binary_only)}",
            flush=True,
        )
    run_status = h.run()
    MS = highspy.HighsModelStatus
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    info = h.getInfo()
    def stat_float(name: str) -> Optional[float]:
        val = getattr(info, name, None)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    dual_bound = stat_float("mip_dual_bound")
    obj_value = stat_float("objective_function_value")
    margin_dual_bound = None if dual_bound is None else const_z + dual_bound
    nodes_raw = getattr(info, "mip_node_count", None)
    try:
        nodes = None if nodes_raw is None else int(nodes_raw)
    except (TypeError, ValueError):
        nodes = None
    stats = {
        "run_status": str(run_status),
        "status": status,
        "nodes": nodes,
        "dual_bound": dual_bound,
        "obj": obj_value,
        "gap": stat_float("mip_gap"),
        "max_integrality": stat_float("max_integrality_violation"),
        "obj_thr": obj_thr,
        "solver_obj_thr": solver_obj_thr,
        "const_z": const_z,
        "solve_obj_offset": solve_obj_offset,
        "margin_dual_bound": None if dual_bound is None else const_z + solve_obj_offset + dual_bound,
        "elim_singletons": elim_count,
        "elim_eq_subst": eq_subst_count,
        "cutoff_as_row": bool(cutoff_as_row),
        "mip_start": bool(mip_start_values is not None),
        "mip_start_status": mip_start_status,
        "connected_presolve": connected_stats,
        "fbbt": fbbt_stats,
        "fbbt_fixed_subst": fixed_subst_stats,
        "relax_precheck": relax_stats,
    }
    _milp_cutoff_highs.last_stats = stats
    print(
        "  highs_stats "
        f"status={status} nodes={stats['nodes']} "
        f"dual_bound={stats['dual_bound']} "
        f"obj={stats['obj']} "
        f"gap={stats['gap']} "
        f"max_integrality={stats['max_integrality']} "
        f"obj_thr={obj_thr} const_z={const_z} "
        f"margin_dual_bound={stats['margin_dual_bound']}",
        flush=True,
    )

    def dual_bound_proves_empty() -> bool:
        if cutoff_as_row and not multirow_feas:
            return False
        if run_status != highspy.HighsStatus.kOk or status == "Not Set":
            return False
        mb = stats.get("margin_dual_bound")
        return mb is not None and np.isfinite(float(mb)) and float(mb) > float(cutoff) + 1e-7

    def solution(solver_values: Optional[np.ndarray] = None):
        v = (
            np.asarray(h.getSolution().col_value, dtype=np.float64)
            if solver_values is None
            else np.asarray(solver_values, dtype=np.float64)
        )
        if keep_cols is None:
            full = v.copy()
        else:
            # Exact singleton projection removes objective-free continuous slack
            # columns only.  Re-expand the reduced solver vector so the input
            # generator coordinates are still available for MILP witness replay.
            full = reconstruct_base.copy()
            full[np.asarray(keep_cols, dtype=np.int64)] = v
        xi = full[:base_ncols].copy()
        if hz.n_bin:
            xi[hz.n_cont:] = 2.0 * xi[hz.n_cont:] - 1.0
        if multirow_feas:
            y = hz.c + np.asarray(hz.Gc @ xi[:hz.n_cont]).reshape(-1)
            if hz.n_bin:
                y = y + np.asarray(hz.Gb @ xi[hz.n_cont:]).reshape(-1)
            val = float(np.max(Cmat @ y - tvec))
        else:
            val = original_const_z + float(original_margin_cost @ full)
        return val, xi

    def target_incumbent_from_nonterminal():
        """Accept a HiGHS incumbent only after an independent MILP residual check."""
        validation = {
            "available": False,
            "accepted": False,
            "reason": "not_checked",
        }
        try:
            raw = np.asarray(h.getSolution().col_value, dtype=np.float64)
        except Exception as exc:
            validation.update({"reason": f"get_solution_error:{type(exc).__name__}:{str(exc)[:80]}"})
            stats["incumbent_validation"] = validation
            return None
        validation["available"] = bool(raw.size)
        if raw.size != ncols or not np.all(np.isfinite(raw)):
            validation.update({"reason": f"bad_solution_shape:{raw.size}!={ncols}"})
            stats["incumbent_validation"] = validation
            return None

        v = raw.copy()
        int_vio = 0.0
        if int_idx.size:
            ints = v[int_idx]
            rounded = np.rint(ints)
            int_vio = float(np.max(np.abs(ints - rounded))) if ints.size else 0.0
            validation["max_integrality"] = int_vio
            if int_vio > 1e-5:
                validation.update({"reason": "integrality_violation"})
                stats["incumbent_validation"] = validation
                return None
            v[int_idx] = np.clip(rounded, 0.0, 1.0)
        else:
            validation["max_integrality"] = 0.0

        lb_vio = float(np.max(np.maximum(lb - v, 0.0))) if lb.size else 0.0
        ub_vio = float(np.max(np.maximum(v - ub, 0.0))) if ub.size else 0.0
        bound_vio = max(lb_vio, ub_vio)
        validation["max_bound_vio"] = bound_vio
        if bound_vio > 1e-6:
            validation.update({"reason": "bound_violation"})
            stats["incumbent_validation"] = validation
            return None

        row_vio = 0.0
        row_vio_scaled = 0.0
        if A.shape[0]:
            av = np.asarray(A @ v, dtype=np.float64).reshape(-1)
            finite_lower = rl > -1e20
            finite_upper = ru < 1e20
            lower = np.where(finite_lower, rl - av, -np.inf)
            upper = np.where(finite_upper, av - ru, -np.inf)
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(finite_lower, np.abs(rl), 0.0),
                    np.where(finite_upper, np.abs(ru), 0.0),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
        validation["max_row_vio"] = row_vio
        validation["max_row_vio_scaled"] = row_vio_scaled
        if row_vio > 5e-5:
            validation.update({"reason": "row_violation"})
            stats["incumbent_validation"] = validation
            return None

        val, xi = solution(v)
        validation["margin"] = float(val)
        if not np.isfinite(val) or val > cutoff + 1e-7:
            validation.update({"reason": "cutoff_not_met"})
            stats["incumbent_validation"] = validation
            return None
        validation.update({"accepted": True, "reason": "target_incumbent"})
        stats["incumbent_validation"] = validation
        return val, xi

    if multirow_feas:
        if st == MS.kObjectiveTarget:
            val, xi = solution()
            return f"TARGET:{status}", val, xi
        if st in (MS.kObjectiveBound, MS.kInfeasible):
            return f"EMPTY:{status}", None, None
        if st == MS.kOptimal:
            val, xi = solution()
            if val <= cutoff + 1e-7:
                return f"TARGET:{status}", val, xi
            return f"EMPTY:{status}", val, None
        if dual_bound_proves_empty():
            return f"EMPTY:dual_bound:{status}", stats["margin_dual_bound"], None
        incumbent = target_incumbent_from_nonterminal()
        if incumbent is not None:
            val, xi = incumbent
            return f"TARGET:{status}:incumbent", val, xi
        return status, None, None

    if cutoff_as_row:
        if st == MS.kOptimal:
            val, xi = solution()
            if val <= cutoff + 1e-7:
                return f"TARGET:{status}", val, xi
            return f"EMPTY:{status}", val, None
        if st == MS.kInfeasible:
            return f"EMPTY:{status}", None, None
        incumbent = target_incumbent_from_nonterminal()
        if incumbent is not None:
            val, xi = incumbent
            return f"TARGET:{status}:incumbent", val, xi
        return status, None, None

    if st == MS.kObjectiveTarget:
        val, xi = solution()
        return f"TARGET:{status}", val, xi
    if st in (MS.kObjectiveBound, MS.kInfeasible):
        return f"EMPTY:{status}", None, None
    if st == MS.kOptimal:
        val, xi = solution()
        return f"OPTIMAL:{status}", val, xi
    if dual_bound_proves_empty():
        return f"EMPTY:dual_bound:{status}", stats["margin_dual_bound"], None
    incumbent = target_incumbent_from_nonterminal()
    if incumbent is not None:
        val, xi = incumbent
        return f"TARGET:{status}:incumbent", val, xi
    return status, None, None


def _milp_cutoff_scip(
    hz: SparseHZ,
    C: np.ndarray,
    t: np.ndarray,
    time_limit: float,
    *,
    cutoff: float = 0.0,
    elim_singletons: bool = False,
    cutoff_as_row: bool = False,
    fbbt_passes: int = 0,
    scip_threads: int = 0,
    scip_options: Optional[Dict[str, object]] = None,
) -> Tuple[str, Optional[float], Optional[np.ndarray]]:
    try:
        from pyscipopt import Model, quicksum
    except Exception as exc:
        return f"no_pyscipopt:{exc}", None, None

    Cmat = C.reshape(C.shape[0], -1)
    tvec = t.reshape(-1)
    if Cmat.shape[0] != tvec.size:
        raise ValueError(f"C/t row mismatch: {Cmat.shape} vs {tvec.shape}")
    multirow_feas = Cmat.shape[0] > 1
    base_ncols = hz.n_cont + hz.n_bin
    extra_epigraph_cols = 1 if multirow_feas else 0
    if multirow_feas:
        cost = np.zeros(base_ncols + 1, dtype=np.float64)
        cost[-1] = 1.0
        const_z = 0.0
        obj_thr = float(cutoff)
    else:
        c_row = Cmat[0]
        obj_c = np.asarray(c_row @ hz.Gc).reshape(-1)
        obj_b = np.asarray(c_row @ hz.Gb).reshape(-1) if hz.n_bin else np.zeros(0)
        const = float(c_row @ hz.c - tvec[0])
        cost = np.concatenate([obj_c, 2.0 * obj_b])
        const_z = const - float(obj_b.sum())
        obj_thr = float(cutoff - const_z)

    if hz.n_eq:
        A = sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr")
        rhs = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
    else:
        A = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        rhs = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        A = sp.hstack([A, sp.csr_matrix((A.shape[0], extra_epigraph_cols))], format="csr")
    if hz.n_ub:
        Ale = sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr")
        ble = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
    else:
        Ale = sp.csr_matrix((0, hz.n_cont + hz.n_bin), dtype=np.float64)
        ble = np.zeros(0, dtype=np.float64)
    if extra_epigraph_cols:
        Ale = sp.hstack([Ale, sp.csr_matrix((Ale.shape[0], extra_epigraph_cols))], format="csr")
    if multirow_feas:
        Csp = sp.csr_matrix(Cmat)
        unsafe_Ac = Csp @ hz.Gc
        unsafe_Ab = Csp @ hz.Gb if hz.n_bin else sp.csr_matrix((Cmat.shape[0], 0), dtype=np.float64)
        unsafe_A = sp.hstack(
            [
                unsafe_Ac,
                2.0 * unsafe_Ab,
                -sp.csr_matrix(np.ones((Cmat.shape[0], 1))),
            ],
            format="csr",
        )
        unsafe_b = tvec - Cmat @ hz.c + np.asarray(unsafe_Ab.sum(axis=1)).reshape(-1)
        Ale = sp.vstack([Ale, unsafe_A], format="csr")
        ble = np.concatenate([ble, unsafe_b.astype(np.float64)])

    lb = np.concatenate([-np.ones(hz.n_cont), np.zeros(hz.n_bin)])
    ub = np.ones(base_ncols)
    if extra_epigraph_cols:
        lb = np.concatenate([lb, np.array([-1e12], dtype=np.float64)])
        ub = np.concatenate([ub, np.array([1e12], dtype=np.float64)])

    rl = rhs.astype(np.float64).copy()
    ru = rhs.astype(np.float64).copy()
    keep_cols = None
    original_ncols = int(cost.size)
    reconstruct_base = np.zeros(original_ncols, dtype=np.float64)
    solve_obj_offset = 0.0
    elim_count = 0
    fbbt_stats = None
    fixed_subst_stats = None
    if elim_singletons and A.shape[0] and A.shape[1]:
        Acsc = A.tocsc()
        Alecsc = Ale.tocsc() if Ale.shape[0] else None
        removable = []
        for j in range(hz.n_cont):
            if abs(cost[j]) > 1e-12:
                continue
            if Alecsc is not None and Alecsc.indptr[j + 1] > Alecsc.indptr[j]:
                continue
            start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
            if end - start == 1:
                removable.append(j)
        if removable:
            elim_min = np.zeros(A.shape[0], dtype=np.float64)
            elim_max = np.zeros(A.shape[0], dtype=np.float64)
            for j in removable:
                start, end = Acsc.indptr[j], Acsc.indptr[j + 1]
                r = int(Acsc.indices[start])
                a = float(Acsc.data[start])
                vals = (a * lb[j], a * ub[j])
                elim_min[r] += min(vals)
                elim_max[r] += max(vals)
            rl = rhs - elim_max
            ru = rhs - elim_min
            keep = np.ones(A.shape[1], dtype=bool)
            keep[np.asarray(removable, dtype=np.int64)] = False
            keep_cols = np.nonzero(keep)[0]
            A = A[:, keep_cols].tocsr()
            Ale = Ale[:, keep_cols].tocsr()
            cost = cost[keep_cols]
            lb = lb[keep_cols]
            ub = ub[keep_cols]
            elim_count = len(removable)
            print(f"  scip_elim_singletons removed_cont={elim_count} kept_cols={A.shape[1]} rows={A.shape[0]}", flush=True)
    if Ale.shape[0]:
        A = sp.vstack([A, Ale], format="csr")
        rl = np.concatenate([rl, np.full(Ale.shape[0], -1e30, dtype=np.float64)])
        ru = np.concatenate([ru, ble.astype(np.float64)])
    margin_cost = cost.copy()
    if cutoff_as_row and not multirow_feas:
        A = sp.vstack([A, sp.csr_matrix(margin_cost.reshape(1, -1))], format="csr")
        rl = np.concatenate([rl, np.array([-1e30], dtype=np.float64)])
        ru = np.concatenate([ru, np.array([obj_thr], dtype=np.float64)])
        solve_cost = np.zeros_like(margin_cost)
    else:
        solve_cost = margin_cost

    if int(fbbt_passes) > 0 and A.shape[0] and A.shape[1]:
        if keep_cols is None:
            col_orig = np.arange(A.shape[1], dtype=np.int64)
        else:
            col_orig = np.asarray(keep_cols, dtype=np.int64)
        int_mask = (col_orig >= hz.n_cont) & (col_orig < hz.n_cont + hz.n_bin)
        fbbt_empty, lb, ub, fbbt_stats = _fbbt_tighten_bounds(
            A, rl, ru, lb, ub,
            integer_mask=int_mask,
            max_passes=int(fbbt_passes),
        )
        print(
            "  scip_fbbt_presolve "
            f"status={'infeasible' if fbbt_empty else 'ok'} "
            f"passes={fbbt_stats.get('passes')} "
            f"tightened={fbbt_stats.get('tightened')} "
            f"fixed_int={fbbt_stats.get('fixed_int')} "
            f"max_width_delta={fbbt_stats.get('max_width_delta')}",
            flush=True,
        )
        if fbbt_empty:
            _milp_cutoff_scip.last_stats = {
                "status": "fbbt_infeasible",
                "nodes": 0,
                "dual_bound": None,
                "obj": None,
                "gap": None,
                "max_integrality": None,
                "obj_thr": obj_thr,
                "solver_obj_thr": obj_thr - solve_obj_offset,
                "const_z": const_z,
                "solve_obj_offset": solve_obj_offset,
                "margin_dual_bound": None,
                "elim_singletons": elim_count,
                "cutoff_as_row": bool(cutoff_as_row),
                "fbbt": fbbt_stats,
            }
            return "EMPTY:fbbt_infeasible", None, None
        fixed_mask = np.isfinite(lb) & np.isfinite(ub) & ((ub - lb) <= 1e-9)
        if np.any(fixed_mask):
            fixed_pos = np.flatnonzero(fixed_mask)
            fixed_vals = 0.5 * (lb[fixed_pos] + ub[fixed_pos])
            current_cols = (
                np.arange(A.shape[1], dtype=np.int64)
                if keep_cols is None
                else np.asarray(keep_cols, dtype=np.int64)
            )
            fixed_orig = current_cols[fixed_pos]
            contrib = np.asarray(A[:, fixed_pos] @ fixed_vals, dtype=np.float64).reshape(-1)
            rl = rl - contrib
            ru = ru - contrib
            solve_obj_offset += float(solve_cost[fixed_pos] @ fixed_vals)
            reconstruct_base[fixed_orig] = fixed_vals

            keep_local = np.ones(A.shape[1], dtype=bool)
            keep_local[fixed_pos] = False
            before_cols, before_rows, before_nnz = int(A.shape[1]), int(A.shape[0]), int(A.nnz)
            A = A[:, keep_local].tocsr()
            A.eliminate_zeros()
            cost = cost[keep_local]
            margin_cost = margin_cost[keep_local]
            solve_cost = solve_cost[keep_local]
            lb = lb[keep_local]
            ub = ub[keep_local]
            keep_cols = current_cols[keep_local]

            row_nnz = np.diff(A.indptr)
            zero_rows = np.flatnonzero(row_nnz == 0)
            dropped_zero_rows = 0
            if zero_rows.size:
                bad_zero = zero_rows[
                    ((np.isfinite(rl[zero_rows])) & (rl[zero_rows] > 1e-9))
                    | ((np.isfinite(ru[zero_rows])) & (ru[zero_rows] < -1e-9))
                ]
                if bad_zero.size:
                    r0 = int(bad_zero[0])
                    fixed_subst_stats = {
                        "fixed_cols": int(fixed_pos.size),
                        "cols_before": before_cols,
                        "cols_after": int(A.shape[1]),
                        "rows_before": before_rows,
                        "rows_after": int(A.shape[0]),
                        "nnz_before": before_nnz,
                        "nnz_after": int(A.nnz),
                        "zero_row_infeasible": {
                            "bad_row": r0,
                            "rl": float(rl[r0]) if np.isfinite(rl[r0]) else None,
                            "ru": float(ru[r0]) if np.isfinite(ru[r0]) else None,
                        },
                    }
                    _milp_cutoff_scip.last_stats = {
                        "status": "fbbt_fixed_zero_row_infeasible",
                        "nodes": 0,
                        "dual_bound": None,
                        "obj": None,
                        "gap": None,
                        "max_integrality": None,
                        "obj_thr": obj_thr,
                        "solver_obj_thr": obj_thr - solve_obj_offset,
                        "const_z": const_z,
                        "solve_obj_offset": solve_obj_offset,
                        "margin_dual_bound": None,
                        "elim_singletons": elim_count,
                        "cutoff_as_row": bool(cutoff_as_row),
                        "fbbt": fbbt_stats,
                        "fbbt_fixed_subst": fixed_subst_stats,
                    }
                    return "EMPTY:fbbt_fixed_zero_row_infeasible", None, None
                keep_rows = np.ones(A.shape[0], dtype=bool)
                keep_rows[zero_rows] = False
                dropped_zero_rows = int(zero_rows.size)
                A = A[keep_rows, :].tocsr()
                rl = rl[keep_rows]
                ru = ru[keep_rows]

            fixed_subst_stats = {
                "fixed_cols": int(fixed_pos.size),
                "cols_before": before_cols,
                "cols_after": int(A.shape[1]),
                "rows_before": before_rows,
                "rows_after": int(A.shape[0]),
                "nnz_before": before_nnz,
                "nnz_after": int(A.nnz),
                "dropped_zero_rows": dropped_zero_rows,
                "solve_obj_offset": float(solve_obj_offset),
            }
            print(
                "  scip_fbbt_fixed_subst "
                f"fixed_cols={fixed_pos.size} cols={before_cols}->{A.shape[1]} "
                f"rows={before_rows}->{A.shape[0]} nnz={before_nnz}->{A.nnz}",
                flush=True,
            )

    rb_empty, rb_stats = _row_bound_infeasible(A, rl, ru, lb, ub)
    if rb_empty:
        _milp_cutoff_scip.last_stats = {
            "status": "row_bound_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "solver_obj_thr": obj_thr - solve_obj_offset,
            "const_z": const_z,
            "solve_obj_offset": solve_obj_offset,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "cutoff_as_row": bool(cutoff_as_row),
            "fbbt": fbbt_stats,
            "fbbt_fixed_subst": fixed_subst_stats,
            "row_bound_infeasible": rb_stats,
        }
        print(
            "  scip_row_bound_infeasible "
            f"bad_row={rb_stats.get('bad_row')} "
            f"bad_count={rb_stats.get('bad_count')} "
            f"row_min={rb_stats.get('row_min')} "
            f"row_max={rb_stats.get('row_max')} "
            f"rl={rb_stats.get('rl')} ru={rb_stats.get('ru')}",
            flush=True,
        )
        return "EMPTY:row_bound_infeasible", None, None

    model = Model()
    model.hideOutput(True)
    model.setParam("limits/time", float(time_limit))
    if int(scip_threads) > 0:
        for key in ("parallel/maxnthreads", "lp/threads"):
            try:
                model.setParam(key, int(scip_threads))
            except Exception:
                pass
    if scip_options:
        for key, value in scip_options.items():
            try:
                model.setParam(str(key), value)
            except Exception:
                pass
    solver_obj_thr = float(obj_thr - solve_obj_offset)
    try:
        model.setParam("limits/primal", solver_obj_thr)
    except Exception:
        pass
    vars_ = []
    col_map = keep_cols if keep_cols is not None else np.arange(cost.size, dtype=np.int64)
    for j, orig_j in enumerate(col_map):
        lo = float(lb[j])
        hi = float(ub[j])
        if int(orig_j) >= hz.n_cont and int(orig_j) < hz.n_cont + hz.n_bin:
            vars_.append(model.addVar(vtype="B", lb=max(0.0, lo), ub=min(1.0, hi), name=f"z{int(orig_j - hz.n_cont)}"))
        else:
            vars_.append(model.addVar(vtype="C", lb=lo, ub=hi, name=f"x{int(orig_j)}"))

    def row_expr(row: int):
        start, end = A.indptr[row], A.indptr[row + 1]
        return quicksum(float(A.data[p]) * vars_[int(A.indices[p])] for p in range(start, end))

    infeas_const = False
    for r in range(A.shape[0]):
        lhs = float(rl[r])
        rhs_v = float(ru[r])
        if A.indptr[r] == A.indptr[r + 1]:
            if lhs > 1e-9 or rhs_v < -1e-9:
                infeas_const = True
                break
            continue
        expr = row_expr(r)
        lhs_finite = lhs > -1e20
        rhs_finite = rhs_v < 1e20
        if lhs_finite and rhs_finite and abs(lhs - rhs_v) <= 1e-9:
            model.addCons(expr == rhs_v)
        else:
            if lhs_finite:
                model.addCons(expr >= lhs)
            if rhs_finite:
                model.addCons(expr <= rhs_v)
    if infeas_const:
        _milp_cutoff_scip.last_stats = {
            "status": "constant_infeasible",
            "nodes": 0,
            "dual_bound": None,
            "obj": None,
            "gap": None,
            "max_integrality": None,
            "obj_thr": obj_thr,
            "const_z": const_z,
            "margin_dual_bound": None,
            "elim_singletons": elim_count,
            "cutoff_as_row": bool(cutoff_as_row),
        }
        return "EMPTY:constant_infeasible", None, None

    obj_terms = [(j, float(v)) for j, v in enumerate(solve_cost) if abs(float(v)) > 1e-12]
    if obj_terms:
        model.setObjective(quicksum(v * vars_[j] for j, v in obj_terms), "minimize")
    else:
        model.setObjective(0.0, "minimize")
    model.optimize()
    status = str(model.getStatus())
    sol = model.getBestSol()

    def solution() -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if sol is None:
            return None, None, None
        v = np.asarray([model.getSolVal(sol, var) for var in vars_], dtype=np.float64)
        if keep_cols is None:
            full = v.copy()
        else:
            full = reconstruct_base.copy()
            full[np.asarray(keep_cols, dtype=np.int64)] = v
        xi = full[:base_ncols].copy()
        if hz.n_bin:
            xi[hz.n_cont:] = 2.0 * xi[hz.n_cont:] - 1.0
        y = hz.c + np.asarray(hz.Gc @ xi[:hz.n_cont]).reshape(-1)
        if hz.n_bin:
            y = y + np.asarray(hz.Gb @ xi[hz.n_cont:]).reshape(-1)
        if multirow_feas:
            val = float(np.max(Cmat @ y - tvec))
        else:
            val = float(Cmat[0] @ y - tvec[0])
        return val, xi, v

    val, xi, solver_v = solution()
    try:
        obj_val = float(model.getObjVal()) if sol is not None else None
    except Exception:
        obj_val = None
    try:
        dual_bound = float(model.getDualbound())
    except Exception:
        dual_bound = None
    stats = {
        "status": status,
        "nodes": int(model.getNNodes()),
        "dual_bound": dual_bound,
        "obj": obj_val,
        "gap": float(model.getGap()) if sol is not None else None,
        "max_integrality": None,
        "obj_thr": obj_thr,
        "solver_obj_thr": solver_obj_thr,
        "const_z": const_z,
        "solve_obj_offset": solve_obj_offset,
        "margin_dual_bound": None if dual_bound is None else const_z + solve_obj_offset + dual_bound,
        "elim_singletons": elim_count,
        "cutoff_as_row": bool(cutoff_as_row),
        "fbbt": fbbt_stats,
        "fbbt_fixed_subst": fixed_subst_stats,
    }
    if solver_v is not None:
        int_vio = 0.0
        if vars_:
            int_positions = [
                j for j, orig_j in enumerate(col_map)
                if int(orig_j) >= hz.n_cont and int(orig_j) < hz.n_cont + hz.n_bin
            ]
            if int_positions:
                ints = solver_v[np.asarray(int_positions, dtype=np.int64)]
                int_vio = float(np.max(np.abs(ints - np.rint(ints)))) if ints.size else 0.0
        lb_vio = float(np.max(np.maximum(lb - solver_v, 0.0))) if lb.size else 0.0
        ub_vio = float(np.max(np.maximum(solver_v - ub, 0.0))) if ub.size else 0.0
        row_vio = 0.0
        row_vio_scaled = 0.0
        if A.shape[0]:
            av = np.asarray(A @ solver_v, dtype=np.float64).reshape(-1)
            lower = np.where(rl > -1e20, rl - av, -np.inf)
            upper = np.where(ru < 1e20, av - ru, -np.inf)
            vio = np.maximum(np.maximum(lower, upper), 0.0)
            row_vio = float(np.max(vio)) if vio.size else 0.0
            scale = 1.0 + np.maximum(
                np.abs(av),
                np.maximum(
                    np.where(rl > -1e20, np.abs(rl), 0.0),
                    np.where(ru < 1e20, np.abs(ru), 0.0),
                ),
            )
            row_vio_scaled = float(np.max(vio / scale)) if vio.size else 0.0
            if cutoff_as_row and not multirow_feas:
                stats["cutoff_row_lhs"] = float(av[-1])
                stats["cutoff_row_rhs"] = float(ru[-1])
        stats["max_integrality"] = int_vio
        stats["max_bound_vio"] = max(lb_vio, ub_vio)
        stats["max_row_vio"] = row_vio
        stats["max_row_vio_scaled"] = row_vio_scaled
    _milp_cutoff_scip.last_stats = stats
    print(
        "  scip_stats "
        f"status={status} nodes={stats['nodes']} "
        f"dual_bound={stats['dual_bound']} obj={stats['obj']} gap={stats['gap']} "
        f"obj_thr={obj_thr} const_z={const_z} margin={val} "
        f"row_vio={stats.get('max_row_vio')} "
        f"int_vio={stats.get('max_integrality')}",
        flush=True,
    )

    st = status.lower()
    feasible_incumbent = (
        solver_v is not None
        and stats.get("max_bound_vio", 0.0) <= 1e-6
        and stats.get("max_integrality", 0.0) <= 1e-5
        and stats.get("max_row_vio", 0.0) <= 5e-5
    )
    if feasible_incumbent and val is not None and val <= cutoff + 1e-7:
        return f"TARGET:{status}", val, xi
    if st in {"infeasible", "inforunbd"}:
        return f"EMPTY:{status}", None, None
    if st == "optimal":
        if val is not None and val > cutoff + 1e-7:
            return f"EMPTY:{status}", val, None
        if feasible_incumbent:
            return f"TARGET:{status}", val, xi
        return status, val, xi if val is not None else None
    if not (cutoff_as_row and not multirow_feas):
        mb = stats.get("margin_dual_bound")
        if mb is not None and np.isfinite(float(mb)) and float(mb) > float(cutoff) + 1e-7:
            return f"EMPTY:dual_bound:{status}", float(mb), None
    return status, val, xi if val is not None else None


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
