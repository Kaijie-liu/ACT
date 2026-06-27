# ===- act/back_end/hybridz_tf/sparse_ops.py - Sparse HZ operators -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
"""Sparse exact-HZ affine and structural operators.

These are the production-safe, benchmark-independent primitives extracted from
the local sparse-HZ prototype.  They operate on a single global variable frame:
if two branches share a generator, they must use the same column index.  This is
the sparse counterpart of dense HZ shared-generator carrying and is required
for residual/concat structures without dense materialization.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

from act.back_end.core import Bounds
from act.back_end.solver.sparse_hz import SparseHZono


def _pair(x) -> Tuple[int, int]:
    if isinstance(x, int):
        return int(x), int(x)
    return int(x[0]), int(x[1])


def _shape4(shape) -> Tuple[int, int, int, int]:
    dims = tuple(int(d) for d in shape)
    if len(dims) == 4:
        return dims
    if len(dims) == 3:
        c, h, w = dims
        return 1, c, h, w
    raise ValueError(f"expected 3D or 4D NCHW shape, got {shape!r}")


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def sparse_empty(rows: int, cols: int) -> sp.csr_matrix:
    return sp.csr_matrix((int(rows), int(cols)), dtype=np.float64)


def sparse_pad_cols(mat: sp.csr_matrix, cols: int) -> sp.csr_matrix:
    mat = mat.tocsr()
    cols = int(cols)
    if mat.shape[1] == cols:
        return mat
    if mat.shape[1] > cols:
        raise ValueError(f"cannot shrink sparse matrix from {mat.shape[1]} to {cols}")
    return sp.hstack([mat, sparse_empty(mat.shape[0], cols - mat.shape[1])], format="csr")


def sparse_hz_pad_frame(hz: SparseHZono, n_cont: int, n_bin: int) -> SparseHZono:
    Auc = None if hz.Auc is None else sparse_pad_cols(hz.Auc, n_cont)
    Aub = None if hz.Aub is None else sparse_pad_cols(hz.Aub, n_bin)
    return SparseHZono(
        c=hz.c,
        Gc=sparse_pad_cols(hz.Gc, n_cont),
        Gb=sparse_pad_cols(hz.Gb, n_bin),
        Ac=sparse_pad_cols(hz.Ac, n_cont),
        Ab=sparse_pad_cols(hz.Ab, n_bin),
        b=hz.b,
        Auc=Auc,
        Aub=Aub,
        ub=hz.ub,
    )


def sparse_hz_from_bounds(bounds: Bounds, *, drop_zero_radius: bool = True) -> SparseHZono:
    """Create a sparse HZ box from ACT ``Bounds``.

    By default, zero-radius dimensions do not allocate useless generator
    columns.  The represented set is identical to the dense ``hz_from_bounds``
    set, just with structurally-zero columns removed.
    """

    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    center = (lb + ub) * 0.5
    rad = (ub - lb) * 0.5
    if drop_zero_radius:
        rows = np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
    else:
        rows = np.arange(rad.size, dtype=np.int32)
    cols = np.arange(rows.size, dtype=np.int32)
    Gc = sp.csr_matrix((rad[rows], (rows, cols)), shape=(rad.size, rows.size), dtype=np.float64)
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=sparse_empty(rad.size, 0),
        Ac=sparse_empty(0, rows.size),
        Ab=sparse_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse_empty(0, rows.size),
        Aub=sparse_empty(0, 0),
        ub=np.zeros(0, dtype=np.float64),
    )


def sparse_hz_linear(hz: SparseHZono, W, bias: Optional[Sequence[float]] = None) -> SparseHZono:
    """Apply an exact affine map ``W @ z + bias``."""

    Wsp = W.tocsr().astype(np.float64) if sp.issparse(W) else sp.csr_matrix(np.asarray(W, dtype=np.float64))
    if Wsp.shape[1] != hz.n_out:
        raise ValueError(f"linear shape mismatch: W={Wsp.shape}, hz.n_out={hz.n_out}")
    b = (
        np.zeros(Wsp.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != Wsp.shape[0]:
        raise ValueError(f"bias shape mismatch: bias={b.size}, rows={Wsp.shape[0]}")
    Gc = (Wsp @ hz.Gc).tocsr()
    Gb = (Wsp @ hz.Gb).tocsr() if hz.n_bin else sparse_empty(Wsp.shape[0], 0)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.asarray(Wsp @ hz.c).reshape(-1) + b,
        Gc=Gc,
        Gb=Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def sparse_dense_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return exact sparse matrix/bias for an ACT DENSE layer."""

    W = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    b = (
        np.zeros(W.shape[0], dtype=np.float64)
        if bias is None
        else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    )
    mat = sp.csr_matrix(W)
    mat.eliminate_zeros()
    return mat, b


def sparse_hz_apply_dense_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_dense_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_hz_is_point(hz: SparseHZono) -> bool:
    return hz.n_cont == 0 and hz.n_bin == 0 and hz.n_eq == 0 and hz.n_ub == 0


def sparse_matmul_const_matrix_from_layer(
    layer,
    const,
    *,
    variable_is_left: bool,
) -> sp.csr_matrix:
    """Exact linear matrix for MATMUL when one operand is a point tensor."""

    x_shape = tuple(int(d) for d in layer.params["x_shape"])
    y_shape = tuple(int(d) for d in layer.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim <= 0:
        raise ValueError("MATMUL variable operand has empty shape")
    const_shape = y_shape if variable_is_left else x_shape
    W = np.asarray(const, dtype=np.float64).reshape(const_shape)
    eye = np.eye(in_dim, dtype=np.float64).reshape((in_dim, *in_shape))
    out = np.matmul(eye, W) if variable_is_left else np.matmul(W, eye)
    mat = sp.csr_matrix(out.reshape(in_dim, -1).T)
    mat.eliminate_zeros()
    return mat


def sparse_hz_apply_per_batch_linear(hz: SparseHZono, W: sp.csr_matrix) -> SparseHZono:
    W = W.tocsr()
    in_dim = int(W.shape[1])
    if in_dim <= 0 or hz.n_out % in_dim:
        raise ValueError(f"per-batch linear mismatch: hz.n_out={hz.n_out}, W={W.shape}")
    B = hz.n_out // in_dim
    if B == 1:
        big = W
    else:
        big = sp.kron(sp.eye(B, format="csr", dtype=np.float64), W, format="csr")
    return sparse_hz_linear(hz, big, np.zeros(big.shape[0], dtype=np.float64))


def sparse_hz_apply_matmul_const_layer(
    variable_hz: SparseHZono,
    const_hz: SparseHZono,
    layer,
    *,
    variable_is_left: bool,
) -> SparseHZono:
    if not sparse_hz_is_point(const_hz):
        raise ValueError("constant-side MATMUL requires a point SparseHZono")
    W = sparse_matmul_const_matrix_from_layer(
        layer,
        const_hz.c,
        variable_is_left=variable_is_left,
    )
    return sparse_hz_apply_per_batch_linear(variable_hz, W)


def _prod_interval_bounds(xl: float, xu: float, yl: float, yu: float) -> Tuple[float, float]:
    vals = (xl * yl, xl * yu, xu * yl, xu * yu)
    return float(min(vals)), float(max(vals))


def sparse_hz_apply_matmul_product_interval_layer(
    x: SparseHZono,
    y: SparseHZono,
    layer,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    y_lb: np.ndarray,
    y_ub: np.ndarray,
    xi_c: np.ndarray,
    xi_b: np.ndarray,
) -> Tuple[SparseHZono, np.ndarray]:
    """Sound product-interval HZ lift for var-var MATMUL."""

    x_shape = tuple(int(v) for v in layer.params["x_shape"])
    y_shape = tuple(int(v) for v in layer.params["y_shape"])
    if len(x_shape) == 2:
        bsz, m, k = 1, x_shape[0], x_shape[1]
        y_bsz, k2, n = 1, y_shape[0], y_shape[1]

        def x_index(_b: int, i: int, kk: int) -> int:
            return i * k + kk

        def y_index(_b: int, kk: int, j: int) -> int:
            return kk * n + j

        def out_index(_b: int, i: int, j: int) -> int:
            return i * n + j

    elif len(x_shape) == 3:
        bsz, m, k = x_shape
        y_bsz, k2, n = y_shape
        if y_bsz not in (1, bsz):
            raise NotImplementedError(f"unsupported batched MATMUL shapes {x_shape} @ {y_shape}")

        def x_index(b: int, i: int, kk: int) -> int:
            return (b * m + i) * k + kk

        def y_index(b: int, kk: int, j: int) -> int:
            yy_b = 0 if y_bsz == 1 else b
            return (yy_b * k + kk) * n + j

        def out_index(b: int, i: int, j: int) -> int:
            return (b * m + i) * n + j

    else:
        raise NotImplementedError(f"unsupported MATMUL rank {x_shape} @ {y_shape}")
    if k != k2:
        raise ValueError(f"MATMUL inner mismatch {x_shape} @ {y_shape}")

    n_out = bsz * m * n
    old_c = x.n_cont
    if y.n_cont != old_c or x.n_bin != y.n_bin:
        raise ValueError("MATMUL operands must be in the same variable frame")
    n_bin = x.n_bin
    prod_cols: List[int] = []
    prod_rows: List[int] = []
    prod_data: List[float] = []
    cut_rr_c: List[np.ndarray] = []
    cut_cc_c: List[np.ndarray] = []
    cut_dd_c: List[np.ndarray] = []
    cut_rr_b: List[np.ndarray] = []
    cut_cc_b: List[np.ndarray] = []
    cut_dd_b: List[np.ndarray] = []
    cut_ub: List[float] = []
    carry_x_constraints = False
    carry_y_constraints = False
    c_out = np.zeros(n_out, dtype=np.float64)
    xi_extra: List[float] = []
    x_nnz = np.diff(x.Gc.indptr) + (np.diff(x.Gb.indptr) if n_bin else 0)
    y_nnz = np.diff(y.Gc.indptr) + (np.diff(y.Gb.indptr) if n_bin else 0)

    def add_cut_c(row: int, cols: np.ndarray, data: np.ndarray) -> None:
        if cols.size:
            cut_rr_c.append(np.full(cols.size, int(row), dtype=np.int32))
            cut_cc_c.append(cols.astype(np.int32, copy=False))
            cut_dd_c.append(data.astype(np.float64, copy=False))

    def add_cut_b(row: int, cols: np.ndarray, data: np.ndarray) -> None:
        if cols.size:
            cut_rr_b.append(np.full(cols.size, int(row), dtype=np.int32))
            cut_cc_b.append(cols.astype(np.int32, copy=False))
            cut_dd_b.append(data.astype(np.float64, copy=False))

    def add_nonnegative_product_cuts(
        *,
        prod_col: int,
        rad: float,
        center: float,
        w_hz: SparseHZono,
        w_row: int,
        other_lo: float,
        other_hi: float,
    ) -> None:
        """Add ``other_lo*w <= p <= other_hi*w`` for non-negative ``w``."""

        upper = len(cut_ub)
        lower = upper + 1
        cut_ub.extend([
            float(other_hi * w_hz.c[w_row] - center),
            float(center - other_lo * w_hz.c[w_row]),
        ])

        add_cut_c(upper, np.asarray([prod_col], dtype=np.int32), np.asarray([rad], dtype=np.float64))
        c0, c1 = w_hz.Gc.indptr[w_row], w_hz.Gc.indptr[w_row + 1]
        if c1 > c0:
            add_cut_c(upper, w_hz.Gc.indices[c0:c1], -other_hi * w_hz.Gc.data[c0:c1])
        if n_bin:
            b0, b1 = w_hz.Gb.indptr[w_row], w_hz.Gb.indptr[w_row + 1]
            if b1 > b0:
                add_cut_b(upper, w_hz.Gb.indices[b0:b1], -other_hi * w_hz.Gb.data[b0:b1])

        add_cut_c(lower, np.asarray([prod_col], dtype=np.int32), np.asarray([-rad], dtype=np.float64))
        if c1 > c0:
            add_cut_c(lower, w_hz.Gc.indices[c0:c1], other_lo * w_hz.Gc.data[c0:c1])
        if n_bin:
            b0, b1 = w_hz.Gb.indptr[w_row], w_hz.Gb.indptr[w_row + 1]
            if b1 > b0:
                add_cut_b(lower, w_hz.Gb.indices[b0:b1], other_lo * w_hz.Gb.data[b0:b1])

    x_val = x.c + np.asarray(x.Gc @ xi_c[:old_c]).reshape(-1)
    y_val = y.c + np.asarray(y.Gc @ xi_c[:old_c]).reshape(-1)
    if n_bin:
        x_val = x_val + np.asarray(x.Gb @ xi_b[:n_bin]).reshape(-1)
        y_val = y_val + np.asarray(y.Gb @ xi_b[:n_bin]).reshape(-1)

    for b in range(bsz):
        for i in range(m):
            for j in range(n):
                oi = out_index(b, i, j)
                for kk in range(k):
                    xi = x_index(b, i, kk)
                    yi = y_index(b, kk, j)
                    lo, hi = _prod_interval_bounds(x_lb[xi], x_ub[xi], y_lb[yi], y_ub[yi])
                    center = 0.5 * (lo + hi)
                    rad = 0.5 * (hi - lo)
                    c_out[oi] += center
                    if rad > 1e-12:
                        col = old_c + len(xi_extra)
                        prod_rows.append(oi)
                        prod_cols.append(col)
                        prod_data.append(rad)
                        actual = float(x_val[xi] * y_val[yi])
                        xi_extra.append(float(np.clip((actual - center) / rad, -1.0, 1.0)))
                        x_nonneg = bool(x_lb[xi] >= -1e-12)
                        y_nonneg = bool(y_lb[yi] >= -1e-12)
                        if x_nonneg and (not y_nonneg or x_nnz[xi] <= y_nnz[yi]):
                            add_nonnegative_product_cuts(
                                prod_col=col,
                                rad=rad,
                                center=center,
                                w_hz=x,
                                w_row=xi,
                                other_lo=float(y_lb[yi]),
                                other_hi=float(y_ub[yi]),
                            )
                            carry_x_constraints = True
                        elif y_nonneg:
                            add_nonnegative_product_cuts(
                                prod_col=col,
                                rad=rad,
                                center=center,
                                w_hz=y,
                                w_row=yi,
                                other_lo=float(x_lb[xi]),
                                other_hi=float(x_ub[xi]),
                            )
                            carry_y_constraints = True

    n_total = old_c + len(xi_extra)
    Gc = sp.csr_matrix(
        (
            np.asarray(prod_data, dtype=np.float64),
            (np.asarray(prod_rows, dtype=np.int32), np.asarray(prod_cols, dtype=np.int32)),
        ),
        shape=(n_out, n_total),
    )
    Gc.eliminate_zeros()
    parts = []
    if carry_x_constraints:
        parts.append(sparse_hz_pad_frame(x, n_total, n_bin))
    if carry_y_constraints:
        parts.append(sparse_hz_pad_frame(y, n_total, n_bin))
    if parts:
        Ac, Ab, bvec = _merge_equalities(parts, n_total, n_bin)
        Auc, Aub, ubvec = _merge_uppers(parts, n_total, n_bin)
    else:
        Ac = sparse_empty(0, n_total)
        Ab = sparse_empty(0, n_bin)
        bvec = np.zeros(0, dtype=np.float64)
        Auc = sparse_empty(0, n_total)
        Aub = sparse_empty(0, n_bin)
        ubvec = np.zeros(0, dtype=np.float64)
    if cut_ub:
        n_cuts = len(cut_ub)
        cut_Auc = (
            sp.coo_matrix(
                (np.concatenate(cut_dd_c), (np.concatenate(cut_rr_c), np.concatenate(cut_cc_c))),
                shape=(n_cuts, n_total),
            ).tocsr()
            if cut_rr_c
            else sparse_empty(n_cuts, n_total)
        )
        cut_Aub = (
            sp.coo_matrix(
                (np.concatenate(cut_dd_b), (np.concatenate(cut_rr_b), np.concatenate(cut_cc_b))),
                shape=(n_cuts, n_bin),
            ).tocsr()
            if cut_rr_b
            else sparse_empty(n_cuts, n_bin)
        )
        cut_Auc.eliminate_zeros()
        cut_Aub.eliminate_zeros()
        Auc = sp.vstack([Auc, cut_Auc], format="csr") if Auc.shape[0] else cut_Auc
        Aub = sp.vstack([Aub, cut_Aub], format="csr") if Aub.shape[0] else cut_Aub
        ubvec = np.concatenate([ubvec, np.asarray(cut_ub, dtype=np.float64)])

    hz = SparseHZono(
        c=c_out,
        Gc=Gc,
        Gb=sparse_empty(n_out, n_bin),
        Ac=Ac,
        Ab=Ab,
        b=bvec,
        Auc=Auc,
        Aub=Aub,
        ub=ubvec,
    )
    return hz, np.asarray(xi_extra, dtype=np.float64)


def _exp_sum_stable(diffs: np.ndarray) -> float:
    if diffs.size == 0:
        return 0.0
    if float(np.max(diffs)) > 700.0:
        return float("inf")
    return float(np.exp(diffs).sum())


def sparse_softmax_interval_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    shape: Tuple[int, ...],
    axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound interval bounds for softmax outputs."""

    if not shape:
        return np.ones_like(lb, dtype=np.float64), np.ones_like(lb, dtype=np.float64)
    axis = int(axis)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"softmax axis {axis} incompatible with shape {shape}")
    lb_arr = lb.reshape(shape)
    ub_arr = ub.reshape(shape)
    lb_last = np.moveaxis(lb_arr, axis, -1).reshape(-1, shape[axis])
    ub_last = np.moveaxis(ub_arr, axis, -1).reshape(-1, shape[axis])
    lo = np.zeros_like(lb_last, dtype=np.float64)
    hi = np.zeros_like(ub_last, dtype=np.float64)
    for g in range(lb_last.shape[0]):
        for i in range(lb_last.shape[1]):
            others = np.arange(lb_last.shape[1]) != i
            lo_den = 1.0 + _exp_sum_stable(ub_last[g, others] - lb_last[g, i])
            hi_den = 1.0 + _exp_sum_stable(lb_last[g, others] - ub_last[g, i])
            lo[g, i] = 0.0 if np.isinf(lo_den) else 1.0 / lo_den
            hi[g, i] = 1.0 if np.isinf(hi_den) else 1.0 / hi_den
    lo = np.moveaxis(lo.reshape(np.moveaxis(lb_arr, axis, -1).shape), -1, axis).reshape(-1)
    hi = np.moveaxis(hi.reshape(np.moveaxis(ub_arr, axis, -1).shape), -1, axis).reshape(-1)
    hi = np.maximum(hi, lo)
    return lo, hi


def sparse_hz_apply_softmax_simplex_layer(
    prev: SparseHZono,
    layer,
    input_shape: Tuple[int, ...],
    lb: np.ndarray,
    ub: np.ndarray,
    xi_c: np.ndarray,
    xi_b: np.ndarray,
) -> Tuple[SparseHZono, np.ndarray]:
    """Sound softmax HZ relaxation with simplex and ratio cuts."""

    axis = int(layer.params.get("axis", -1))
    lo, hi = sparse_softmax_interval_bounds(lb, ub, input_shape, axis)
    n = int(lo.size)
    old_c, n_bin = prev.n_cont, prev.n_bin
    center = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    active = np.nonzero(rad > 1e-12)[0].astype(np.int32)
    n_total = old_c + int(active.size)
    cols = old_c + np.arange(active.size, dtype=np.int32)
    Gc = sp.csr_matrix((rad[active], (active, cols)), shape=(n, n_total), dtype=np.float64)
    Gc.eliminate_zeros()

    logits = prev.c + np.asarray(prev.Gc @ xi_c[:old_c]).reshape(-1)
    if n_bin:
        logits = logits + np.asarray(prev.Gb @ xi_b[:n_bin]).reshape(-1)
    norm_axis = axis if axis >= 0 else axis + len(input_shape)
    logits_arr = logits.reshape(input_shape)
    logits_last = np.moveaxis(logits_arr, norm_axis, -1)
    maxes = np.max(logits_last, axis=-1, keepdims=True)
    probs_last = np.exp(logits_last - maxes)
    probs_last = probs_last / np.sum(probs_last, axis=-1, keepdims=True)
    probs = np.moveaxis(probs_last, -1, norm_axis).reshape(-1)
    xi_extra = np.zeros(active.size, dtype=np.float64)
    if active.size:
        xi_extra = np.clip((probs[active] - center[active]) / rad[active], -1.0, 1.0)

    moved_index = np.arange(n, dtype=np.int64).reshape(input_shape)
    groups = np.moveaxis(moved_index, norm_axis, -1).reshape(-1, input_shape[norm_axis])
    eq_rows: List[np.ndarray] = []
    eq_cols: List[np.ndarray] = []
    eq_data: List[np.ndarray] = []
    eq_b = np.zeros(groups.shape[0], dtype=np.float64)
    active_to_col = {int(row): int(col) for row, col in zip(active, cols)}
    for g, rows in enumerate(groups):
        eq_b[g] = 1.0 - float(center[rows].sum())
        active_rows = [int(r) for r in rows if int(r) in active_to_col]
        cc = [active_to_col[r] for r in active_rows]
        if cc:
            eq_rows.append(np.full(len(cc), g, dtype=np.int32))
            eq_cols.append(np.asarray(cc, dtype=np.int32))
            eq_data.append(rad[active_rows].astype(np.float64))
    if eq_rows:
        Ac = sp.csr_matrix(
            (np.concatenate(eq_data), (np.concatenate(eq_rows), np.concatenate(eq_cols))),
            shape=(groups.shape[0], n_total),
        )
    else:
        Ac = sparse_empty(groups.shape[0], n_total)
    Ac.eliminate_zeros()

    cut_rows: List[np.ndarray] = []
    cut_cols: List[np.ndarray] = []
    cut_data: List[np.ndarray] = []
    cut_ub: List[float] = []
    ratio_hi_cap = 1e6
    ratio_lo_floor = 1e-6
    for rows in groups:
        rows = np.asarray(rows, dtype=np.int64)
        for ii in rows:
            i = int(ii)
            for jj in rows:
                j = int(jj)
                if i == j:
                    continue
                diff_hi = float(ub[i] - lb[j])
                if diff_hi < np.log(ratio_hi_cap):
                    ratio_hi = float(np.exp(diff_hi))
                    row = len(cut_ub)
                    cc: List[int] = []
                    dd: List[float] = []
                    if i in active_to_col:
                        cc.append(active_to_col[i])
                        dd.append(float(rad[i]))
                    if j in active_to_col:
                        cc.append(active_to_col[j])
                        dd.append(float(-ratio_hi * rad[j]))
                    if cc:
                        cut_rows.append(np.full(len(cc), row, dtype=np.int32))
                        cut_cols.append(np.asarray(cc, dtype=np.int32))
                        cut_data.append(np.asarray(dd, dtype=np.float64))
                        cut_ub.append(float(ratio_hi * center[j] - center[i]))
                diff_lo = float(lb[i] - ub[j])
                if diff_lo > np.log(ratio_lo_floor):
                    ratio_lo = float(np.exp(diff_lo))
                    row = len(cut_ub)
                    cc = []
                    dd = []
                    if j in active_to_col:
                        cc.append(active_to_col[j])
                        dd.append(float(ratio_lo * rad[j]))
                    if i in active_to_col:
                        cc.append(active_to_col[i])
                        dd.append(float(-rad[i]))
                    if cc:
                        cut_rows.append(np.full(len(cc), row, dtype=np.int32))
                        cut_cols.append(np.asarray(cc, dtype=np.int32))
                        cut_data.append(np.asarray(dd, dtype=np.float64))
                        cut_ub.append(float(center[i] - ratio_lo * center[j]))
    if cut_rows:
        Auc = sp.csr_matrix(
            (np.concatenate(cut_data), (np.concatenate(cut_rows), np.concatenate(cut_cols))),
            shape=(len(cut_ub), n_total),
        )
        Auc.eliminate_zeros()
        ub_vec = np.asarray(cut_ub, dtype=np.float64)
    else:
        Auc = sparse_empty(0, n_total)
        ub_vec = np.zeros(0, dtype=np.float64)

    hz = SparseHZono(
        c=center,
        Gc=Gc,
        Gb=sparse_empty(n, n_bin),
        Ac=Ac,
        Ab=sparse_empty(groups.shape[0], n_bin),
        b=eq_b,
        Auc=Auc,
        Aub=sparse_empty(Auc.shape[0], n_bin),
        ub=ub_vec,
    )
    return hz, xi_extra


def sparse_conv2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for an ACT CONV2D layer."""

    weight = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    bias_np = None if bias is None else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    out_ch, in_ch_per_group, kh, kw = [int(v) for v in weight.shape]
    groups = int(layer.params.get("groups", 1))
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    bsz, in_ch, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch_shape, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or out_ch != out_ch_shape:
        raise ValueError(f"conv2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    if in_ch_per_group * groups != in_ch:
        raise ValueError(f"conv2d group/input mismatch at layer {getattr(layer, 'id', '?')}")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    out_ch_per_group = out_ch // groups
    for n in range(bsz):
        for co in range(out_ch):
            group = co // out_ch_per_group
            ci_base = group * in_ch_per_group
            for oh in range(out_h):
                ih0 = oh * stride[0] - padding[0]
                for ow in range(out_w):
                    iw0 = ow * stride[1] - padding[1]
                    out_idx = ((n * out_ch + co) * out_h + oh) * out_w + ow
                    cur_cols: List[int] = []
                    cur_vals: List[float] = []
                    for ci_local in range(in_ch_per_group):
                        ci = ci_base + ci_local
                        for rr in range(kh):
                            ih = ih0 + rr * dilation[0]
                            if ih < 0 or ih >= in_h:
                                continue
                            for cc in range(kw):
                                iw = iw0 + cc * dilation[1]
                                if iw < 0 or iw >= in_w:
                                    continue
                                val = float(weight[co, ci_local, rr, cc])
                                if val == 0.0:
                                    continue
                                cur_cols.append(((n * in_ch + ci) * in_h + ih) * in_w + iw)
                                cur_vals.append(val)
                    if cur_cols:
                        arr_c = np.asarray(cur_cols, dtype=np.int32)
                        rows.append(np.full(arr_c.size, out_idx, dtype=np.int32))
                        cols.append(arr_c)
                        data.append(np.asarray(cur_vals, dtype=np.float64))

    if rows:
        rr = np.concatenate(rows)
        cc = np.concatenate(cols)
        dd = np.concatenate(data)
    else:
        rr = cc = np.empty(0, dtype=np.int32)
        dd = np.empty(0, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * out_ch * out_h * out_w, bsz * in_ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    if bias_np is None:
        bvec = np.zeros(mat.shape[0], dtype=np.float64)
    else:
        bvec = np.tile(np.repeat(bias_np, out_h * out_w), bsz)
    return mat, bvec


def sparse_hz_apply_conv2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_conv2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_convtranspose2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for ACT CONVTRANSPOSE2D."""

    weight = _to_numpy(layer.params["weight"]).astype(np.float64, copy=False)
    bias = layer.params.get("bias")
    bias_np = None if bias is None else _to_numpy(bias).astype(np.float64, copy=False).reshape(-1)
    in_ch, out_ch_per_group, kh, kw = [int(v) for v in weight.shape]
    groups = int(layer.params.get("groups", 1))
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    bsz, in_ch_shape, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or in_ch != in_ch_shape:
        raise ValueError(f"convtranspose2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    if out_ch != out_ch_per_group * groups:
        raise ValueError(f"convtranspose2d group/output mismatch at layer {getattr(layer, 'id', '?')}")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    in_ch_per_group = in_ch // groups
    for n in range(bsz):
        for ci in range(in_ch):
            group = ci // in_ch_per_group
            co_base = group * out_ch_per_group
            for ih in range(in_h):
                oh0 = ih * stride[0] - padding[0]
                for iw in range(in_w):
                    ow0 = iw * stride[1] - padding[1]
                    in_idx = ((n * in_ch + ci) * in_h + ih) * in_w + iw
                    cur_rows: List[int] = []
                    cur_vals: List[float] = []
                    for co_local in range(out_ch_per_group):
                        co = co_base + co_local
                        for rr in range(kh):
                            oh = oh0 + rr * dilation[0]
                            if oh < 0 or oh >= out_h:
                                continue
                            for cc in range(kw):
                                ow = ow0 + cc * dilation[1]
                                if ow < 0 or ow >= out_w:
                                    continue
                                val = float(weight[ci, co_local, rr, cc])
                                if val == 0.0:
                                    continue
                                cur_rows.append(((n * out_ch + co) * out_h + oh) * out_w + ow)
                                cur_vals.append(val)
                    if cur_rows:
                        arr_r = np.asarray(cur_rows, dtype=np.int32)
                        rows.append(arr_r)
                        cols.append(np.full(arr_r.size, in_idx, dtype=np.int32))
                        data.append(np.asarray(cur_vals, dtype=np.float64))

    if rows:
        rr = np.concatenate(rows)
        cc = np.concatenate(cols)
        dd = np.concatenate(data)
    else:
        rr = cc = np.empty(0, dtype=np.int32)
        dd = np.empty(0, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * out_ch * out_h * out_w, bsz * in_ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    if bias_np is None:
        bvec = np.zeros(mat.shape[0], dtype=np.float64)
    else:
        bvec = np.tile(np.repeat(bias_np, out_h * out_w), bsz)
    return mat, bvec


def sparse_hz_apply_convtranspose2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_convtranspose2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_avgpool2d_matrix_from_layer(layer) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Build the exact NCHW sparse affine matrix for ACT AVGPOOL2D."""

    bsz, ch, in_h, in_w = _shape4(layer.params["input_shape"])
    out_bsz, out_ch, out_h, out_w = _shape4(layer.params["output_shape"])
    if bsz != out_bsz or ch != out_ch:
        raise ValueError(f"avgpool2d shape mismatch at layer {getattr(layer, 'id', '?')}")
    kh, kw = _pair(layer.params["kernel_size"])
    stride = layer.params.get("stride", layer.params["kernel_size"])
    sh, sw = _pair(stride)
    ph, pw = _pair(layer.params.get("padding", 0))
    denom = float(kh * kw)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for n in range(bsz):
        for c in range(ch):
            for oh in range(out_h):
                ih0 = oh * sh - ph
                for ow in range(out_w):
                    iw0 = ow * sw - pw
                    out_idx = ((n * ch + c) * out_h + oh) * out_w + ow
                    cur_cols: List[int] = []
                    for ki in range(kh):
                        ih = ih0 + ki
                        if ih < 0 or ih >= in_h:
                            continue
                        for kj in range(kw):
                            iw = iw0 + kj
                            if iw < 0 or iw >= in_w:
                                continue
                            cur_cols.append(((n * ch + c) * in_h + ih) * in_w + iw)
                    if cur_cols:
                        rows.append(np.full(len(cur_cols), out_idx, dtype=np.int32))
                        cols.append(np.asarray(cur_cols, dtype=np.int32))
    rr = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    cc = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    dd = np.full(rr.size, 1.0 / denom, dtype=np.float64)
    mat = sp.csr_matrix(
        (dd, (rr, cc)),
        shape=(bsz * ch * out_h * out_w, bsz * ch * in_h * in_w),
        dtype=np.float64,
    )
    mat.eliminate_zeros()
    return mat, np.zeros(mat.shape[0], dtype=np.float64)


def sparse_hz_apply_avgpool2d_layer(hz: SparseHZono, layer) -> SparseHZono:
    W, b = sparse_avgpool2d_matrix_from_layer(layer)
    return sparse_hz_linear(hz, W, b)


def sparse_maxpool2d_candidate_rows(layer) -> List[np.ndarray]:
    """Return one output-to-input row map per MaxPool2D window candidate.

    Invalid padding locations are encoded as ``-1``.  The exact MaxPool fold
    replaces those with a constant below the input lower bound, so padding
    locations cannot win a maximum.
    """

    bsz, ch, in_h, in_w = _shape4(layer.params["input_shape"])
    kh, kw = _pair(layer.params["kernel_size"])
    stride = layer.params.get("stride", layer.params["kernel_size"])
    sh, sw = _pair(stride)
    ph, pw = _pair(layer.params.get("padding", 0))
    dh, dw = _pair(layer.params.get("dilation", 1))
    output_shape = layer.params.get("output_shape")
    if output_shape is None:
        out_bsz, out_ch = bsz, ch
        out_h = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    else:
        out_bsz, out_ch, out_h, out_w = _shape4(output_shape)
    if bsz != out_bsz or ch != out_ch:
        raise ValueError(f"maxpool2d shape mismatch at layer {getattr(layer, 'id', '?')}")

    n_out = bsz * ch * out_h * out_w
    candidates: List[np.ndarray] = []
    for ki in range(kh):
        for kj in range(kw):
            rows = np.full(n_out, -1, dtype=np.int64)
            for n in range(bsz):
                for c in range(ch):
                    for oh in range(out_h):
                        ih = oh * sh - ph + ki * dh
                        for ow in range(out_w):
                            iw = ow * sw - pw + kj * dw
                            out_idx = ((n * ch + c) * out_h + oh) * out_w + ow
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                rows[out_idx] = ((n * ch + c) * in_h + ih) * in_w + iw
            candidates.append(rows)
    return candidates


def sparse_upsample_nearest_row_indices(layer, n_in: int, n_out: int) -> Optional[np.ndarray]:
    """Output-to-input row map for nearest-neighbor upsample."""

    mode = str(layer.params.get("mode", "nearest")).lower()
    if mode != "nearest":
        return None
    in_shape = layer.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    if _prod(in_shape) != int(n_in) or len(in_shape) < 3:
        return None
    view_shape = (1, *in_shape) if len(in_shape) == 3 else in_shape
    spatial_rank = len(view_shape) - 2
    size = layer.params.get("size")
    scale_factor = layer.params.get("scale_factor")
    if size is not None and isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
        if len(size) > spatial_rank:
            size = size[-spatial_rank:]
    if scale_factor is not None and isinstance(scale_factor, (list, tuple)):
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) > spatial_rank:
            scale_factor = scale_factor[-spatial_rank:]
    if size is None and scale_factor is None:
        out_shape = layer.params.get("output_shape")
        if out_shape is None:
            return None
        out_shape = tuple(int(d) for d in out_shape)
        out_view_shape = (1, *out_shape) if len(out_shape) == 3 else out_shape
        if len(out_view_shape) != len(view_shape):
            return None
        size = out_view_shape[2:]
    base = torch.arange(int(n_in), dtype=torch.float64).view(*view_shape)
    out = torch.nn.functional.interpolate(
        base,
        size=size,
        scale_factor=scale_factor,
        mode="nearest",
    )
    idx = out.reshape(-1).detach().cpu().numpy().astype(np.int64)
    if idx.size != int(n_out):
        return None
    return idx


def sparse_slice_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Slice, mirroring interval.tf_slice."""

    if "input_shape" not in layer.params:
        return None
    inp_shape = tuple(int(d) for d in layer.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    starts = layer.params.get("starts", [])
    ends = layer.params.get("ends", [])
    axes = layer.params.get("axes", list(range(len(inp_shape))))
    steps = layer.params.get("steps", [1] * len(axes))
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        end = ends[i]
        if end > inp_shape[axis]:
            end = inp_shape[axis]
        slices[axis + 1] = slice(starts[i], end, steps[i])
    return idx[tuple(slices)].reshape(-1).detach().cpu().numpy().astype(np.int64)


def sparse_gather_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Gather, mirroring interval.tf_gather."""

    if "input_shape" not in layer.params:
        return None
    inp_shape = tuple(int(d) for d in layer.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    axis = int(layer.params.get("axis", 0))
    if axis < 0:
        axis += len(inp_shape)
    raw_idx = layer.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    elif hasattr(raw_idx, "detach"):
        indices = raw_idx.detach().cpu().long()
    else:
        indices = torch.as_tensor(raw_idx, dtype=torch.long)
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    return (
        torch.index_select(idx, dim=axis + 1, index=indices.reshape(-1))
        .reshape(-1)
        .numpy()
        .astype(np.int64)
    )


def sparse_expand_row_indices(layer, n: int) -> Optional[np.ndarray]:
    """Output-to-input row map for Expand/broadcast, mirroring interval.tf_expand."""

    in_shape = layer.params.get("input_shape")
    out_shape = layer.params.get("output_shape") or layer.params.get("shape")
    if in_shape is None or out_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    per = _prod(in_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    try:
        rows = torch.arange(int(n)).view(batch, *in_shape).broadcast_to(batch, *out_shape)
    except RuntimeError:
        return None
    return rows.reshape(-1).detach().cpu().numpy().astype(np.int64)


def sparse_reduce_sum_row_indices(layer, n_in: int, n_out: int) -> Optional[np.ndarray]:
    """Input-to-output row map for ReduceSum, mirroring interval.tf_reduce_sum."""

    in_shape = layer.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    per = _prod(in_shape)
    if per == 0 or int(n_in) % per != 0:
        return None
    batch = int(n_in) // per
    axes = layer.params.get("axes")
    axes = list(range(len(in_shape))) if not axes else [int(a) for a in axes]
    axes = [(a + len(in_shape)) if a < 0 else a for a in axes]
    keepdims = bool(layer.params.get("keepdims", 0))
    out_shape: List[int] = []
    for i, dim in enumerate(in_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(dim)
    if _prod(out_shape) * batch != int(n_out):
        return None
    out_idx = np.arange(int(n_out), dtype=np.int64).reshape((batch, *out_shape))
    view_shape = [batch]
    for i, dim in enumerate(in_shape):
        view_shape.append(1 if i in axes else dim)
    return np.broadcast_to(out_idx.reshape(tuple(view_shape)), (batch, *in_shape)).reshape(-1)


def sparse_hz_reduce_sum_rows(
    hz: SparseHZono,
    out_rows: Sequence[int],
    n_out: int,
) -> SparseHZono:
    """Exact ReduceSum row aggregation using a sparse linear map."""

    rows = np.asarray(out_rows, dtype=np.int64).reshape(-1)
    if rows.size != hz.n_out:
        raise ValueError(f"reduce-sum row map length {rows.size} != hz.n_out {hz.n_out}")
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(n_out)):
        raise ValueError("reduce-sum row map contains output index outside n_out")
    mat = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, np.arange(rows.size, dtype=np.int64))),
        shape=(int(n_out), hz.n_out),
        dtype=np.float64,
    )
    return sparse_hz_linear(hz, mat, np.zeros(int(n_out), dtype=np.float64))


def sparse_hz_gather_rows_like(
    base: SparseHZono,
    rows: Sequence[int],
    *,
    fill_value: float,
    template: Optional[SparseHZono] = None,
) -> SparseHZono:
    """Gather rows from ``base`` while using ``template``'s variable frame."""

    row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    tmpl = template if template is not None else base
    n = int(row_idx.size)
    valid = row_idx >= 0
    c = np.full(n, float(fill_value), dtype=np.float64)
    if np.any(valid):
        pos = np.nonzero(valid)[0].astype(np.int32)
        src = row_idx[valid].astype(np.int64)
        c[pos] = base.c[src]
        sub_c = sparse_pad_cols(base.Gc[src].tocsr(), tmpl.n_cont).tocoo()
        Gc = sp.coo_matrix(
            (sub_c.data, (pos[sub_c.row], sub_c.col)),
            shape=(n, tmpl.n_cont),
            dtype=np.float64,
        ).tocsr()
        if tmpl.n_bin:
            sub_b = sparse_pad_cols(base.Gb[src].tocsr(), tmpl.n_bin).tocoo()
            Gb = sp.coo_matrix(
                (sub_b.data, (pos[sub_b.row], sub_b.col)),
                shape=(n, tmpl.n_bin),
                dtype=np.float64,
            ).tocsr()
        else:
            Gb = sparse_empty(n, 0)
    else:
        Gc = sparse_empty(n, tmpl.n_cont)
        Gb = sparse_empty(n, tmpl.n_bin)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=c,
        Gc=Gc,
        Gb=Gb,
        Ac=tmpl.Ac,
        Ab=tmpl.Ab,
        b=tmpl.b,
        Auc=tmpl.Auc,
        Aub=tmpl.Aub,
        ub=tmpl.ub,
    )


def sparse_hz_apply_maxpool2d_layer(
    hz: SparseHZono,
    layer,
    input_bounds: Bounds,
    *,
    compressed_relu: bool = False,
    valid_cuts: bool = False,
) -> SparseHZono:
    """Exact sparse MaxPool2D as a fold of ``max(a,b)=b+ReLU(a-b)``."""

    candidates = sparse_maxpool2d_candidate_rows(layer)
    if not candidates:
        raise ValueError(f"maxpool2d layer {getattr(layer, 'id', '?')} has no candidates")
    in_lb = input_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    in_ub = input_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    fill_value = float(np.min(in_lb) - 1.0) if in_lb.size else -1.0

    def gather_bounds(rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        valid = row_idx >= 0
        lb = np.full(row_idx.size, fill_value, dtype=np.float64)
        ub = np.full(row_idx.size, fill_value, dtype=np.float64)
        if np.any(valid):
            lb[valid] = in_lb[row_idx[valid]]
            ub[valid] = in_ub[row_idx[valid]]
        return lb, ub

    out = sparse_hz_gather_rows_like(hz, candidates[0], fill_value=fill_value)
    cur_lb, cur_ub = gather_bounds(candidates[0])
    for rows in candidates[1:]:
        nxt = sparse_hz_gather_rows_like(hz, rows, fill_value=fill_value, template=out)
        nxt_lb, nxt_ub = gather_bounds(rows)
        diff = sparse_hz_sub_same_frame(out, nxt)
        pre_bounds = Bounds(
            lb=torch.from_numpy(cur_lb - nxt_ub).reshape(1, -1).double(),
            ub=torch.from_numpy(cur_ub - nxt_lb).reshape(1, -1).double(),
        )
        relu = sparse_hz_apply_relu_exact(
            diff,
            pre_bounds=pre_bounds,
            compressed=compressed_relu,
            valid_cuts=valid_cuts,
        )
        out = sparse_hz_add_same_frame(nxt, relu)
        cur_lb = np.maximum(cur_lb, nxt_lb)
        cur_ub = np.maximum(cur_ub, nxt_ub)
    return out


def sparse_hz_fast_bounds(hz: SparseHZono) -> Bounds:
    """Sound unconstrained box bounds for a sparse HZ."""

    rad_c = np.asarray(np.abs(hz.Gc).sum(axis=1)).reshape(-1)
    rad_b = np.asarray(np.abs(hz.Gb).sum(axis=1)).reshape(-1) if hz.n_bin else 0.0
    rad = rad_c + rad_b
    lb = torch.from_numpy(hz.c - rad).reshape(1, -1).double()
    ub = torch.from_numpy(hz.c + rad).reshape(1, -1).double()
    return Bounds(lb=lb, ub=ub)


def _bounds_arrays(bounds: Bounds, n: int) -> Tuple[np.ndarray, np.ndarray]:
    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1)
    if lb.size != n or ub.size != n:
        raise ValueError(f"bounds shape mismatch: bounds={lb.size}/{ub.size}, hz.n_out={n}")
    return lb.astype(np.float64, copy=False), ub.astype(np.float64, copy=False)


def sparse_hz_row_lp_bound(
    hz: SparseHZono,
    row: int,
    *,
    maximize: bool,
    time_limit: float,
) -> Optional[float]:
    """Tighten one sparse-HZ output row with a continuous LP relaxation."""

    gc = hz.Gc.getrow(row)
    gb = hz.Gb.getrow(row) if hz.n_bin else sparse_empty(1, 0)
    if hz.n_eq == 0:
        width = float(np.abs(gc.data).sum() + np.abs(gb.data).sum())
        return float(hz.c[row] + (width if maximize else -width))
    coeff = np.concatenate([
        np.asarray(gc.toarray()).reshape(-1),
        np.asarray(gb.toarray()).reshape(-1),
    ])
    obj = -coeff if maximize else coeff
    Aeq = sp.hstack([hz.Ac, hz.Ab], format="csr")
    Aub = None
    bub = None
    if hz.n_ub:
        Aub = sp.hstack([hz.Auc, hz.Aub], format="csr")
        bub = hz.ub
    opts = {"time_limit": float(time_limit)} if time_limit > 0 else None
    result = linprog(
        obj,
        A_eq=Aeq,
        b_eq=hz.b,
        A_ub=Aub,
        b_ub=bub,
        bounds=[(-1.0, 1.0)] * (hz.n_cont + hz.n_bin),
        method="highs",
        options=opts,
    )
    if not result.success:
        return None
    val = float(coeff @ result.x)
    return float(hz.c[row] + val)


def _sparse_hz_tighten_relu_bounds_highs(
    hz: SparseHZono,
    pre_bounds,
    *,
    time_limit: float,
    rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    import highspy

    lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64).copy()
    ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64).copy()
    nvars = hz.n_cont + hz.n_bin
    A = sp.hstack([hz.Ac, hz.Ab], format="csr")

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    # Reusing the same basis across objective changes is the point here.
    h.setOptionValue("solver", "simplex")
    h.addCols(
        nvars,
        np.zeros(nvars, dtype=np.float64),
        -np.ones(nvars, dtype=np.float64),
        np.ones(nvars, dtype=np.float64),
        0,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
    )
    h.addRows(
        A.shape[0],
        hz.b.astype(np.float64),
        hz.b.astype(np.float64),
        A.nnz,
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
    )
    if hz.n_ub:
        Aub = sp.hstack([hz.Auc, hz.Aub], format="csr")
        h.addRows(
            Aub.shape[0],
            np.full(Aub.shape[0], -1e30, dtype=np.float64),
            hz.ub.astype(np.float64),
            Aub.nnz,
            Aub.indptr.astype(np.int32),
            Aub.indices.astype(np.int32),
            Aub.data.astype(np.float64),
        )

    all_cols = np.arange(nvars, dtype=np.int32)
    model_status = highspy.HighsModelStatus
    improved = fixed_on = fixed_off = solved = 0

    def coeff(row: int) -> np.ndarray:
        out = np.zeros(nvars, dtype=np.float64)
        gc = hz.Gc.getrow(row)
        if gc.nnz:
            out[gc.indices] = gc.data
        if hz.n_bin:
            gb = hz.Gb.getrow(row)
            if gb.nnz:
                out[hz.n_cont + gb.indices] = gb.data
        return out

    def solve(cost: np.ndarray) -> Optional[float]:
        h.changeColsCost(nvars, all_cols, cost.astype(np.float64))
        h.run()
        if h.getModelStatus() != model_status.kOptimal:
            return None
        return float(h.getObjectiveValue())

    for row in rows:
        row = int(row)
        cost = coeff(row)
        lo_obj = solve(cost)
        if lo_obj is None:
            continue
        hi_obj = solve(-cost)
        if hi_obj is None:
            continue
        solved += 1
        lo = float(hz.c[row] + lo_obj)
        hi = float(hz.c[row] - hi_obj)
        old = ub[row] - lb[row]
        lb[row] = max(lb[row], lo)
        ub[row] = min(ub[row], hi)
        if (ub[row] - lb[row]) < old - 1e-9:
            improved += 1
        if lb[row] >= 0.0:
            fixed_on += 1
        elif ub[row] <= 0.0:
            fixed_off += 1
    return lb, ub, (solved, improved, fixed_on, fixed_off)


def sparse_hz_tighten_relu_bounds(
    hz: SparseHZono,
    pre_bounds,
    *,
    limit: int,
    time_limit: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Tighten interval-unstable ReLU preactivation bounds exactly by LP."""

    lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64).copy()
    ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64).copy()
    rows = np.nonzero((lb < 0.0) & (ub > 0.0))[0]
    if limit > 0:
        widths = ub[rows] - lb[rows]
        rows = rows[np.argsort(-widths)[:limit]]
    if hz.n_eq:
        try:
            return _sparse_hz_tighten_relu_bounds_highs(
                hz,
                pre_bounds,
                time_limit=time_limit,
                rows=rows,
            )
        except Exception as exc:
            print(f"  tightLP HiGHS fallback to scipy: {type(exc).__name__}:{exc}", flush=True)

    improved = fixed_on = fixed_off = solved = 0
    for row in rows:
        lo = sparse_hz_row_lp_bound(hz, int(row), maximize=False, time_limit=time_limit)
        hi = sparse_hz_row_lp_bound(hz, int(row), maximize=True, time_limit=time_limit)
        if lo is None or hi is None:
            continue
        solved += 1
        old = ub[row] - lb[row]
        lb[row] = max(lb[row], lo)
        ub[row] = min(ub[row], hi)
        if (ub[row] - lb[row]) < old - 1e-9:
            improved += 1
        if lb[row] >= 0.0:
            fixed_on += 1
        elif ub[row] <= 0.0:
            fixed_off += 1
    return lb, ub, (solved, improved, fixed_on, fixed_off)


def _coo_matrix_from_parts(
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    data: List[np.ndarray],
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    if not rows:
        return sparse_empty(*shape)
    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
        dtype=np.float64,
    ).tocsr()


def _coo_appender(
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    data: List[np.ndarray],
):
    def append(row_idx, col_idx, values) -> None:
        row_arr = np.asarray(row_idx, dtype=np.int32).reshape(-1)
        if row_arr.size:
            rows.append(row_arr)
            cols.append(np.asarray(col_idx, dtype=np.int32).reshape(-1))
            data.append(np.asarray(values, dtype=np.float64).reshape(-1))

    return append


def sparse_hz_apply_relu_exact(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    compressed: bool = False,
    valid_cuts: bool = False,
    return_info: bool = False,
):
    """Exact sparse eq_lagr ReLU.

    Every unstable neuron gets the exact binary HZ graph.  ``compressed`` keeps
    the same exact graph after projecting the two slack equalities into two
    upper-inequality rows.  ``valid_cuts`` appends redundant graph facets for
    solver tightening; it never replaces the exact binary construction.
    """

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    lb, ub = _bounds_arrays(pre_bounds, hz.n_out)
    active = lb >= 0.0
    inactive = ub <= 0.0
    unstable = ~(active | inactive)
    active_idx = np.nonzero(active)[0].astype(np.int32)
    unstable_idx = np.nonzero(unstable)[0].astype(np.int32)
    k = int(unstable_idx.size)
    info = {
        "lb": lb.copy(),
        "ub": ub.copy(),
        "active": active.copy(),
        "inactive": inactive.copy(),
        "unstable_idx": unstable_idx.copy(),
        "compressed": bool(compressed),
    }

    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin
    ng_new = ng + (2 * k if compressed else 4 * k)
    nb_new = nb + k

    out_c = np.zeros(n, dtype=np.float64)
    blocks_c: List[sp.csr_matrix] = []
    blocks_b: List[sp.csr_matrix] = []
    if active_idx.size:
        out_c[active_idx] = hz.c[active_idx]
        act_c = hz.Gc[active_idx].tocoo()
        blocks_c.append(
            sp.coo_matrix(
                (act_c.data, (active_idx[act_c.row], act_c.col)),
                shape=(n, ng_new),
                dtype=np.float64,
            ).tocsr()
        )
        if nb:
            act_b = hz.Gb[active_idx].tocoo()
            blocks_b.append(
                sp.coo_matrix(
                    (act_b.data, (active_idx[act_b.row], act_b.col)),
                    shape=(n, nb_new),
                    dtype=np.float64,
                ).tocsr()
            )
    if k:
        beta = ub[unstable_idx]
        out_c[unstable_idx] = beta / 2.0
        xi2_cols = ng + k + np.arange(k, dtype=np.int32)
        blocks_c.append(
            sp.csr_matrix(
                (-beta / 2.0, (unstable_idx, xi2_cols)),
                shape=(n, ng_new),
                dtype=np.float64,
            )
        )

    out_Gc = sum(blocks_c[1:], blocks_c[0]).tocsr() if blocks_c else sparse_empty(n, ng_new)
    out_Gb = sum(blocks_b[1:], blocks_b[0]).tocsr() if blocks_b else sparse_empty(n, nb_new)
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    old_Ac = sparse_pad_cols(hz.Ac, ng_new)
    old_Ab = sparse_pad_cols(hz.Ab, nb_new)
    if k:
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]
        te = np.arange(k, dtype=np.int32)
        col_xi1 = ng + te
        col_xi2 = ng + k + te
        col_z = nb + te

        rr_c: List[np.ndarray] = []
        cc_c: List[np.ndarray] = []
        dd_c: List[np.ndarray] = []
        rr_b: List[np.ndarray] = []
        cc_b: List[np.ndarray] = []
        dd_b: List[np.ndarray] = []
        add_c = _coo_appender(rr_c, cc_c, dd_c)
        add_b = _coo_appender(rr_b, cc_b, dd_b)

        if compressed:
            r3 = te
            n_relu_eq = k
        else:
            col_xi3 = ng + 2 * k + te
            col_xi4 = ng + 3 * k + te
            r1 = 3 * te
            r2 = r1 + 1
            r3 = r1 + 2
            add_c(r1, col_xi1, np.ones(k))
            add_c(r1, col_xi3, np.ones(k))
            add_c(r2, col_xi2, np.ones(k))
            add_c(r2, col_xi4, np.ones(k))
            add_b(r1, col_z, np.ones(k))
            add_b(r2, col_z, -np.ones(k))
            n_relu_eq = 3 * k

        add_c(r3, col_xi1, alpha / 2.0)
        add_c(r3, col_xi2, -beta / 2.0)
        add_b(r3, col_z, alpha / 2.0)

        pre_gc = hz.Gc[unstable_idx].tocoo()
        if pre_gc.nnz:
            add_c(r3[pre_gc.row], pre_gc.col, -pre_gc.data)
        if nb:
            pre_gb = hz.Gb[unstable_idx].tocoo()
            if pre_gb.nnz:
                add_b(r3[pre_gb.row], pre_gb.col, -pre_gb.data)

        eq_Ac = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_relu_eq, ng_new))
        eq_Ab = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_relu_eq, nb_new))
        eq_b = np.zeros(n_relu_eq, dtype=np.float64)
        if not compressed:
            eq_b[r1] = 1.0
            eq_b[r2] = 1.0
        eq_b[r3] = hz.c[unstable_idx] - beta / 2.0
    else:
        eq_Ac = sparse_empty(0, ng_new)
        eq_Ab = sparse_empty(0, nb_new)
        eq_b = np.zeros(0, dtype=np.float64)

    Ac = sp.vstack([old_Ac, eq_Ac], format="csr")
    Ab = sp.vstack([old_Ab, eq_Ab], format="csr")
    b = np.concatenate([hz.b, eq_b])
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()

    Auc_base = sparse_pad_cols(hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont), ng_new)
    Aub_base = sparse_pad_cols(hz.Aub if hz.Aub is not None else sparse_empty(0, hz.n_bin), nb_new)
    ub_base = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    upper_blocks_c = [Auc_base]
    upper_blocks_b = [Aub_base]
    upper_rhs = [ub_base]

    if compressed and k:
        rows = np.arange(k, dtype=np.int32)
        proj_Ac = sp.coo_matrix(
            (
                np.concatenate([-np.ones(k), -np.ones(k)]),
                (
                    np.concatenate([rows, k + rows]),
                    np.concatenate([ng + rows, ng + k + rows]),
                ),
            ),
            shape=(2 * k, ng_new),
            dtype=np.float64,
        ).tocsr()
        proj_Ab = sp.coo_matrix(
            (
                np.concatenate([-np.ones(k), np.ones(k)]),
                (
                    np.concatenate([rows, k + rows]),
                    np.concatenate([nb + rows, nb + rows]),
                ),
            ),
            shape=(2 * k, nb_new),
            dtype=np.float64,
        ).tocsr()
        upper_blocks_c.append(proj_Ac)
        upper_blocks_b.append(proj_Ab)
        upper_rhs.append(np.zeros(2 * k, dtype=np.float64))

    if valid_cuts and k:
        rr_c = []
        cc_c = []
        dd_c = []
        rr_b = []
        cc_b = []
        dd_b = []
        rhs = np.zeros(2 * k, dtype=np.float64)
        add_cut_c = _coo_appender(rr_c, cc_c, dd_c)
        add_cut_b = _coo_appender(rr_b, cc_b, dd_b)

        row1 = np.arange(k, dtype=np.int32)
        row2 = k + row1
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]
        slope = beta / np.maximum(beta - alpha, 1e-12)
        xi2_cols = ng + k + np.arange(k, dtype=np.int32)
        pre_gc = hz.Gc[unstable_idx].tocoo()
        if pre_gc.nnz:
            add_cut_c(pre_gc.row, pre_gc.col, pre_gc.data)
            add_cut_c(k + pre_gc.row, pre_gc.col, -slope[pre_gc.row] * pre_gc.data)
        if nb:
            pre_gb = hz.Gb[unstable_idx].tocoo()
            if pre_gb.nnz:
                add_cut_b(pre_gb.row, pre_gb.col, pre_gb.data)
                add_cut_b(k + pre_gb.row, pre_gb.col, -slope[pre_gb.row] * pre_gb.data)
        add_cut_c(row1, xi2_cols, beta / 2.0)
        add_cut_c(row2, xi2_cols, -beta / 2.0)
        rhs[row1] = beta / 2.0 - hz.c[unstable_idx]
        rhs[row2] = -slope * alpha - beta / 2.0 + slope * hz.c[unstable_idx]
        upper_blocks_c.append(_coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * k, ng_new)))
        upper_blocks_b.append(_coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * k, nb_new)))
        upper_rhs.append(rhs)

    Auc = sp.vstack(upper_blocks_c, format="csr")
    Aub = sp.vstack(upper_blocks_b, format="csr")
    ub_rhs = np.concatenate(upper_rhs)
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    out = SparseHZono(
        out_c, out_Gc, out_Gb, Ac, Ab, b, Auc, Aub, ub_rhs,
    )
    if return_info:
        return out, (int(active.sum()), int(inactive.sum()), k), info
    return out


def sparse_hz_apply_relu_exact_tightened(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    tight_lp_limit: int = 0,
    tight_lp_timeout: float = 0.0,
    tight_fix_mode: str = "both",
    compressed: bool = False,
    valid_cuts: bool = False,
    return_info: bool = False,
):
    """Apply exact ReLU after optional exact LP bound tightening.

    ``tight_fix_mode="off-only"`` preserves active interval-unstable phases as
    unstable even when the LP proves them active.  That keeps sparse rows from
    expanding in later affine layers while still taking inactive fixes, which
    are sparsity-reducing and exact.
    """

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    base_lb, base_ub = _bounds_arrays(pre_bounds, hz.n_out)
    if tight_lp_limit != 0:
        lb, ub, tight_stats = sparse_hz_tighten_relu_bounds(
            hz,
            pre_bounds,
            limit=tight_lp_limit,
            time_limit=tight_lp_timeout,
        )
        if tight_fix_mode == "off-only":
            was_unstable = (base_lb < 0.0) & (base_ub > 0.0)
            active_by_tight = was_unstable & (lb >= 0.0) & (ub > 0.0)
            if np.any(active_by_tight):
                lb = lb.copy()
                lb[active_by_tight] = base_lb[active_by_tight]
    else:
        lb = base_lb
        ub = base_ub
        tight_stats = (0, 0, 0, 0)

    tightened_bounds = Bounds(
        lb=torch.as_tensor(lb, dtype=torch.float64),
        ub=torch.as_tensor(ub, dtype=torch.float64),
    )
    if return_info:
        out, counts, info = sparse_hz_apply_relu_exact(
            hz,
            tightened_bounds,
            compressed=compressed,
            valid_cuts=valid_cuts,
            return_info=True,
        )
        return out, counts, tight_stats, info
    return sparse_hz_apply_relu_exact(
        hz,
        tightened_bounds,
        compressed=compressed,
        valid_cuts=valid_cuts,
    )


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_deriv_np(x: np.ndarray) -> np.ndarray:
    y = _sigmoid_np(x)
    return y * (1.0 - y)


def _tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0))


def _tanh_deriv_np(x: np.ndarray) -> np.ndarray:
    y = _tanh_np(x)
    return 1.0 - y * y


def _scurve_breakpoints(
    lo: float,
    hi: float,
    K: int,
    *,
    grid: str,
    func,
    dfunc,
) -> np.ndarray:
    """Segment breakpoints for one convex/concave side of an S-curve."""

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
    for i in range(1, cuts.size):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]
    return cuts


def _scurve_domain_cut_matrices(
    hz: SparseHZono,
    wide_idx: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    lb_w: np.ndarray,
    ub_w: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Exact-valid cuts tying selected S-curve segments to their x-domain."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64).reshape(-1)
    seg_b = np.asarray(seg_b, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_a.size == seg_b.size == z_cols.size == r):
        raise ValueError("S-curve domain cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    upper_rows = np.arange(r, dtype=np.int32)
    lower_rows = r + upper_rows
    g_lo = np.asarray(lb_w, dtype=np.float64).reshape(-1)[owner]
    g_hi = np.asarray(ub_w, dtype=np.float64).reshape(-1)[owner]
    x_center = hz.c[np.asarray(wide_idx, dtype=np.int64)[owner]]
    rhs = np.empty(2 * r, dtype=np.float64)
    rhs[upper_rows] = (g_hi + seg_b) / 2.0 - x_center
    rhs[lower_rows] = -(g_lo + seg_a) / 2.0 + x_center

    add_b(upper_rows, z_cols, -(g_hi - seg_b) / 2.0)
    add_b(lower_rows, z_cols, -(seg_a - g_lo) / 2.0)

    pre_gc = hz.Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    pre_gb = hz.Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr() if hz.n_bin else sparse_empty(len(wide_idx), 0)
    for j in range(len(wide_idx)):
        flats = np.nonzero(owner == j)[0].astype(np.int32)
        if flats.size == 0:
            continue
        c0, c1 = pre_gc.indptr[j], pre_gc.indptr[j + 1]
        if c1 > c0:
            cols = pre_gc.indices[c0:c1].astype(np.int32)
            data = pre_gc.data[c0:c1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_c(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_c(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))
        if hz.n_bin:
            b0, b1 = pre_gb.indptr[j], pre_gb.indptr[j + 1]
            if b1 > b0:
                cols = pre_gb.indices[b0:b1].astype(np.int32)
                data = pre_gb.data[b0:b1].astype(np.float64)
                rows = np.repeat(flats, cols.size)
                add_b(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
                add_b(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))

    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * r, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * r, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, rhs


def _scurve_range_cut_matrices(
    out_c: np.ndarray,
    out_Gc: sp.csr_matrix,
    out_Gb: sp.csr_matrix,
    wide_idx: np.ndarray,
    seg_y_lo: np.ndarray,
    seg_y_hi: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    y_lo_w: np.ndarray,
    y_hi_w: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Exact-valid cuts tying selected S-curve segments to their y-range."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_y_lo = np.asarray(seg_y_lo, dtype=np.float64).reshape(-1)
    seg_y_hi = np.asarray(seg_y_hi, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_y_lo.size == seg_y_hi.size == z_cols.size == r):
        raise ValueError("S-curve range cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    upper_rows = np.arange(r, dtype=np.int32)
    lower_rows = r + upper_rows
    g_lo = np.asarray(y_lo_w, dtype=np.float64).reshape(-1)[owner]
    g_hi = np.asarray(y_hi_w, dtype=np.float64).reshape(-1)[owner]
    y_center = out_c[np.asarray(wide_idx, dtype=np.int64)[owner]]
    rhs = np.empty(2 * r, dtype=np.float64)
    rhs[upper_rows] = (g_hi + seg_y_hi) / 2.0 - y_center
    rhs[lower_rows] = -(g_lo + seg_y_lo) / 2.0 + y_center

    add_b(upper_rows, z_cols, -(g_hi - seg_y_hi) / 2.0)
    add_b(lower_rows, z_cols, -(seg_y_lo - g_lo) / 2.0)

    y_gc = out_Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    y_gb = out_Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    for j in range(len(wide_idx)):
        flats = np.nonzero(owner == j)[0].astype(np.int32)
        if flats.size == 0:
            continue
        c0, c1 = y_gc.indptr[j], y_gc.indptr[j + 1]
        if c1 > c0:
            cols = y_gc.indices[c0:c1].astype(np.int32)
            data = y_gc.data[c0:c1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_c(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_c(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))
        b0, b1 = y_gb.indptr[j], y_gb.indptr[j + 1]
        if b1 > b0:
            cols = y_gb.indices[b0:b1].astype(np.int32)
            data = y_gb.data[b0:b1].astype(np.float64)
            rows = np.repeat(flats, cols.size)
            add_b(rows, np.tile(cols, flats.size), np.tile(data, flats.size))
            add_b(r + rows, np.tile(cols, flats.size), -np.tile(data, flats.size))

    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (2 * r, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (2 * r, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, rhs


def _scurve_graph_cut_matrices(
    hz: SparseHZono,
    out_c: np.ndarray,
    out_Gc: sp.csr_matrix,
    out_Gb: sp.csr_matrix,
    wide_idx: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    owner: np.ndarray,
    z_cols: np.ndarray,
    n_cont: int,
    n_bin: int,
    lb_w: np.ndarray,
    ub_w: np.ndarray,
    *,
    func,
    dfunc,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Conditional convex/concave graph cuts for S-curve segments."""

    owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    r = int(owner.size)
    if r == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    seg_a = np.asarray(seg_a, dtype=np.float64).reshape(-1)
    seg_b = np.asarray(seg_b, dtype=np.float64).reshape(-1)
    z_cols = np.asarray(z_cols, dtype=np.int32).reshape(-1)
    if not (seg_a.size == seg_b.size == z_cols.size == r):
        raise ValueError("S-curve graph cut metadata size mismatch")

    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    rhs_vals: List[float] = []

    pre_gc = hz.Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    pre_gb = hz.Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr() if hz.n_bin else sparse_empty(len(wide_idx), 0)
    y_gc = out_Gc[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    y_gb = out_Gb[np.asarray(wide_idx, dtype=np.int64)].tocsr()
    x_center = hz.c[np.asarray(wide_idx, dtype=np.int64)]
    y_center = out_c[np.asarray(wide_idx, dtype=np.int64)]
    x_lo = np.asarray(lb_w, dtype=np.float64).reshape(-1)
    x_hi = np.asarray(ub_w, dtype=np.float64).reshape(-1)
    y_lo = func(x_lo)
    y_hi = func(x_hi)

    def add_c(row: int, cols: np.ndarray, data: np.ndarray, scale: float) -> None:
        if cols.size and abs(scale) > 0.0:
            rr_c.append(np.full(cols.size, int(row), dtype=np.int32))
            cc_c.append(cols.astype(np.int32, copy=False))
            dd_c.append((float(scale) * data).astype(np.float64, copy=False))

    def add_b(row: int, cols: np.ndarray, data: np.ndarray, scale: float) -> None:
        if cols.size and abs(scale) > 0.0:
            rr_b.append(np.full(cols.size, int(row), dtype=np.int32))
            cc_b.append(cols.astype(np.int32, copy=False))
            dd_b.append((float(scale) * data).astype(np.float64, copy=False))

    def append_linear_cut(seg_i: int, ax: float, ay: float, c0: float) -> None:
        seg_owner = int(owner[seg_i])
        max_g = (
            ax * (x_hi[seg_owner] if ax >= 0.0 else x_lo[seg_owner])
            + ay * (y_hi[seg_owner] if ay >= 0.0 else y_lo[seg_owner])
            + c0
        )
        M = max(0.0, float(max_g))
        row = len(rhs_vals)
        rhs_vals.append(
            M / 2.0 - ax * float(x_center[seg_owner]) - ay * float(y_center[seg_owner]) - c0
        )

        c0p, c1p = pre_gc.indptr[seg_owner], pre_gc.indptr[seg_owner + 1]
        add_c(row, pre_gc.indices[c0p:c1p], pre_gc.data[c0p:c1p], ax)
        b0p, b1p = pre_gb.indptr[seg_owner], pre_gb.indptr[seg_owner + 1]
        add_b(row, pre_gb.indices[b0p:b1p], pre_gb.data[b0p:b1p], ax)
        c0y, c1y = y_gc.indptr[seg_owner], y_gc.indptr[seg_owner + 1]
        add_c(row, y_gc.indices[c0y:c1y], y_gc.data[c0y:c1y], ay)
        b0y, b1y = y_gb.indptr[seg_owner], y_gb.indptr[seg_owner + 1]
        add_b(row, y_gb.indices[b0y:b1y], y_gb.data[b0y:b1y], ay)
        rr_b.append(np.array([row], dtype=np.int32))
        cc_b.append(np.array([int(z_cols[seg_i])], dtype=np.int32))
        dd_b.append(np.array([-M / 2.0], dtype=np.float64))

    fa = func(seg_a)
    fb = func(seg_b)
    da = dfunc(seg_a)
    db = dfunc(seg_b)
    for s in range(r):
        a = float(seg_a[s])
        b = float(seg_b[s])
        if b - a <= 1e-12:
            continue
        if a < -1e-12 and b > 1e-12:
            continue
        sec_slope = float((fb[s] - fa[s]) / (b - a))
        sec_inter = float(fa[s] - sec_slope * a)
        tan_x = np.array([a, b], dtype=np.float64)
        tan_y = func(tan_x)
        tan_slope = dfunc(tan_x)
        tan_inter = tan_y - tan_slope * tan_x
        convex = b <= 1e-12
        if convex:
            append_linear_cut(s, -sec_slope, 1.0, -sec_inter)
            for slope_i, inter_i in zip(tan_slope, tan_inter):
                append_linear_cut(s, float(slope_i), -1.0, float(inter_i))
        else:
            append_linear_cut(s, sec_slope, -1.0, sec_inter)
            for slope_i, inter_i in zip(tan_slope, tan_inter):
                append_linear_cut(s, -float(slope_i), 1.0, -float(inter_i))

    n_rows = len(rhs_vals)
    if n_rows == 0:
        return sparse_empty(0, n_cont), sparse_empty(0, n_bin), np.zeros(0, dtype=np.float64)
    Auc = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_rows, n_cont))
    Aub = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_rows, n_bin))
    Auc.eliminate_zeros()
    Aub.eliminate_zeros()
    return Auc, Aub, np.asarray(rhs_vals, dtype=np.float64)


def sparse_hz_apply_scurve_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    K: int = 2,
    func=_sigmoid_np,
    dfunc=_sigmoid_deriv_np,
    inflection: float = 0.0,
    torch_func=None,
    torch_dfunc=None,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
    return_info: bool = False,
):
    """Compressed sparse HZ S-curve encoding with pruned zero-width segments.

    This mirrors the dense ``hz_apply_piecewise(..., compressed=True,
    inflection=0)`` path: each nondegenerate segment gets two continuous local
    generators and one binary selector; the local generator box is represented
    by exact inequality rows rather than four slack equality columns.
    """

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    grid = str(grid or "uniform")
    lb, ub = _bounds_arrays(pre_bounds, hz.n_out)
    wide = (ub - lb) > 1e-12
    wide_idx = np.nonzero(wide)[0].astype(np.int32)
    narrow_idx = np.nonzero(~wide)[0].astype(np.int32)
    m = int(wide_idx.size)
    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin

    out_c = np.zeros(n, dtype=np.float64)
    if narrow_idx.size:
        out_c[narrow_idx] = func(hz.c[narrow_idx])
    if m == 0:
        out = SparseHZono(
            c=out_c,
            Gc=sparse_empty(n, ng),
            Gb=sparse_empty(n, nb),
            Ac=hz.Ac,
            Ab=hz.Ab,
            b=hz.b,
            Auc=hz.Auc,
            Aub=hz.Aub,
            ub=hz.ub,
        )
        if return_info:
            info = {
                "wide_idx": wide_idx,
                "m": 0,
                "compressed": True,
                "pruned": True,
                "r": 0,
                "owner_arr": np.zeros(0, dtype=np.int32),
                "seg_count": np.zeros(0, dtype=np.int32),
                "a": np.zeros(0, dtype=np.float64),
                "b_seg": np.zeros(0, dtype=np.float64),
                "centers_x": np.zeros(0, dtype=np.float64),
                "centers_y": np.zeros(0, dtype=np.float64),
                "g1_x": np.zeros(0, dtype=np.float64),
                "g1_y": np.zeros(0, dtype=np.float64),
                "g2_x": np.zeros(0, dtype=np.float64),
                "g2_y": np.zeros(0, dtype=np.float64),
                "grid": grid,
            }
            return out, (0, int(narrow_idx.size)), info
        return out

    K_side = max(1, int(K))
    lb_w = lb[wide_idx]
    ub_w = ub[wide_idx]
    pivot = np.maximum(np.minimum(float(inflection), ub_w), lb_w)
    if grid == "uniform":
        sid = np.arange(K_side, dtype=np.float64).reshape(-1, 1)
        w_left = (pivot - lb_w).reshape(1, -1) / K_side
        w_right = (ub_w - pivot).reshape(1, -1) / K_side
        a_left = lb_w.reshape(1, -1) + sid * w_left
        a_right = pivot.reshape(1, -1) + sid * w_right
        a_grid = np.vstack([a_left, a_right])
        b_grid = np.vstack([a_left + w_left, a_right + w_right])
        owner_grid = np.broadcast_to(
            np.arange(m, dtype=np.int32).reshape(1, -1),
            (2 * K_side, m),
        )
        nondeg = (b_grid - a_grid) > 1e-12
        a = a_grid[nondeg].astype(np.float64, copy=False)
        b_seg = b_grid[nondeg].astype(np.float64, copy=False)
        owner = owner_grid[nondeg].astype(np.int32, copy=False)
    else:
        seg_a: List[float] = []
        seg_b: List[float] = []
        seg_owner: List[int] = []
        for j in range(m):
            for lo, hi in ((lb_w[j], pivot[j]), (pivot[j], ub_w[j])):
                if float(hi) - float(lo) <= 1e-12:
                    continue
                cuts = _scurve_breakpoints(
                    float(lo),
                    float(hi),
                    K_side,
                    grid=grid,
                    func=func,
                    dfunc=dfunc,
                )
                for s in range(K_side):
                    if float(cuts[s + 1]) - float(cuts[s]) <= 1e-12:
                        continue
                    seg_a.append(float(cuts[s]))
                    seg_b.append(float(cuts[s + 1]))
                    seg_owner.append(j)
        a = np.asarray(seg_a, dtype=np.float64)
        b_seg = np.asarray(seg_b, dtype=np.float64)
        owner = np.asarray(seg_owner, dtype=np.int32)
    r = int(a.size)
    if r == 0:
        raise ValueError("wide S-curve neuron produced no nondegenerate segment")

    if torch_func is not None and torch_dfunc is not None:
        at = torch.from_numpy(a).double()
        bt = torch.from_numpy(b_seg).double()
        fa_t = torch_func(at)
        fb_t = torch_func(bt)
        la_t = torch_dfunc(at)
        lb_slope_t = torch_dfunc(bt)
        centers_x_t = (at + bt) / 2.0
        centers_y_t = (fa_t + fb_t) / 2.0
        nearly_linear_t = (la_t - lb_slope_t).abs() < 1e-10

        denom_t = lb_slope_t - la_t
        safe_denom_t = torch.where(nearly_linear_t, torch.ones_like(denom_t), denom_t)
        p1_t = (fb_t - fa_t + lb_slope_t * at - la_t * bt) / safe_denom_t
        p2_t = at + bt - p1_t
        g1x_tang_t = (p1_t - at) / 2.0
        g1y_tang_t = lb_slope_t * (p1_t - at) / 2.0
        g2x_tang_t = (p2_t - at) / 2.0
        g2y_tang_t = la_t * (p2_t - at) / 2.0

        half_width_t = (bt - at) / 2.0
        slope_t = (fb_t - fa_t) / (bt - at + 1e-30)
        t_pts = torch.linspace(0.0, 1.0, 50, dtype=torch.float64).view(50, 1)
        pts_t = at.unsqueeze(0) + t_pts * (bt - at).unsqueeze(0)
        f_pts_t = torch_func(pts_t)
        resid_t = f_pts_t - (slope_t.unsqueeze(0) * pts_t + (fa_t - slope_t * at).unsqueeze(0))
        max_err_t = resid_t.abs().max(dim=0).values
        g1x_lin_t = half_width_t
        g1y_lin_t = slope_t * half_width_t
        g2x_lin_t = torch.zeros_like(half_width_t)
        g2y_lin_t = max_err_t

        g1_x_t = torch.where(nearly_linear_t, g1x_lin_t, g1x_tang_t)
        g1_y_t = torch.where(nearly_linear_t, g1y_lin_t, g1y_tang_t)
        g2_x_t = torch.where(nearly_linear_t, g2x_lin_t, g2x_tang_t)
        g2_y_t = torch.where(nearly_linear_t, g2y_lin_t, g2y_tang_t)

        dx_t = pts_t - centers_x_t.unsqueeze(0)
        dy_t = f_pts_t - centers_y_t.unsqueeze(0)
        det_t = g1_y_t * g2_x_t - g1_x_t * g2_y_t
        safe_det_t = torch.where(det_t.abs() < 1e-30, torch.ones_like(det_t), det_t)
        xi1_t = (dy_t * g2_x_t.unsqueeze(0) - dx_t * g2_y_t.unsqueeze(0)) / safe_det_t.unsqueeze(0)
        xi2_t = (dy_t * g1_x_t.unsqueeze(0) - dx_t * g1_y_t.unsqueeze(0)) / (-safe_det_t.unsqueeze(0))
        max_xi_t = torch.maximum(xi1_t.abs().amax(dim=0), xi2_t.abs().amax(dim=0))
        scale_factor_t = torch.where(max_xi_t > 1.0, max_xi_t * 1.01, torch.ones_like(max_xi_t))
        scale_factor_t = torch.where(det_t.abs() < 1e-30, torch.ones_like(scale_factor_t), scale_factor_t)
        g1_x = (g1_x_t * scale_factor_t).numpy()
        g1_y = (g1_y_t * scale_factor_t).numpy()
        g2_x = (g2_x_t * scale_factor_t).numpy()
        g2_y = (g2_y_t * scale_factor_t).numpy()
        centers_x = centers_x_t.numpy()
        centers_y = centers_y_t.numpy()
        fa = fa_t.numpy()
        fb = fb_t.numpy()
    else:
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

        half_width = (b_seg - a) / 2.0
        slope = (fb - fa) / (b_seg - a + 1e-30)
        t_pts = np.linspace(0.0, 1.0, 50, dtype=np.float64).reshape(50, 1)
        pts = a.reshape(1, r) + t_pts * (b_seg - a).reshape(1, r)
        f_pts = func(pts)
        resid = f_pts - (slope.reshape(1, r) * pts + (fa - slope * a).reshape(1, r))
        max_err = np.max(np.abs(resid), axis=0)
        g1x_lin = half_width
        g1y_lin = slope * half_width
        g2x_lin = np.zeros_like(half_width)
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

    out_c[wide_idx] = np.bincount(owner, weights=centers_y, minlength=m) / 2.0
    seg_count = np.bincount(owner, minlength=m).astype(np.float64)
    center_x_sum = np.bincount(owner, weights=centers_x, minlength=m)

    ng_total = ng + 2 * r
    nb_total = nb + r
    g1_cols = ng + np.arange(r, dtype=np.int32)
    g2_cols = ng + r + np.arange(r, dtype=np.int32)
    z_cols = nb + np.arange(r, dtype=np.int32)
    wide_rows = wide_idx[owner]
    out_Gc = sp.coo_matrix(
        (
            np.concatenate([g1_y, g2_y]),
            (
                np.concatenate([wide_rows, wide_rows]),
                np.concatenate([g1_cols, g2_cols]),
            ),
        ),
        shape=(n, ng_total),
        dtype=np.float64,
    ).tocsr()
    out_Gb = sp.coo_matrix(
        (-centers_y / 2.0, (wide_rows, z_cols)),
        shape=(n, nb_total),
        dtype=np.float64,
    ).tocsr()
    out_Gc.eliminate_zeros()
    out_Gb.eliminate_zeros()

    n_eq_new = 2 * m
    rr_c: List[np.ndarray] = []
    cc_c: List[np.ndarray] = []
    dd_c: List[np.ndarray] = []
    rr_b: List[np.ndarray] = []
    cc_b: List[np.ndarray] = []
    dd_b: List[np.ndarray] = []
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

    link_rows = np.arange(m, dtype=np.int32)
    sum_rows = m + link_rows
    add_c(link_rows[owner], g1_cols, -g1_x)
    add_c(link_rows[owner], g2_cols, -g2_x)
    add_b(link_rows[owner], z_cols, centers_x / 2.0)
    pre_gc = hz.Gc[wide_idx].tocoo()
    if pre_gc.nnz:
        add_c(link_rows[pre_gc.row], pre_gc.col, pre_gc.data)
    if nb:
        pre_gb = hz.Gb[wide_idx].tocoo()
        if pre_gb.nnz:
            add_b(link_rows[pre_gb.row], pre_gb.col, pre_gb.data)
    add_b(sum_rows[owner], z_cols, np.ones(r, dtype=np.float64))

    eq_Ac = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_eq_new, ng_total))
    eq_Ab = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_eq_new, nb_total))
    eq_b = np.zeros(n_eq_new, dtype=np.float64)
    eq_b[link_rows] = center_x_sum / 2.0 - hz.c[wide_idx]
    eq_b[sum_rows] = seg_count - 2.0

    Ac = sp.vstack([sparse_pad_cols(hz.Ac, ng_total), eq_Ac], format="csr")
    Ab = sp.vstack([sparse_pad_cols(hz.Ab, nb_total), eq_Ab], format="csr")
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()
    b = np.concatenate([hz.b, eq_b])

    base_Auc = sparse_pad_cols(
        hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont),
        ng_total,
    )
    base_Aub = sparse_pad_cols(
        hz.Aub if hz.Aub is not None else sparse_empty(0, hz.n_bin),
        nb_total,
    )
    base_ub = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    box_rows = 4 * np.arange(r, dtype=np.int32)
    box_Auc = sp.coo_matrix(
        (
            np.concatenate([np.ones(r), -np.ones(r), np.ones(r), -np.ones(r)]),
            (
                np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                np.concatenate([g1_cols, g1_cols, g2_cols, g2_cols]),
            ),
        ),
        shape=(4 * r, ng_total),
        dtype=np.float64,
    ).tocsr()
    box_Aub = sp.coo_matrix(
        (
            0.5 * np.ones(4 * r, dtype=np.float64),
            (
                np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                np.concatenate([z_cols, z_cols, z_cols, z_cols]),
            ),
        ),
        shape=(4 * r, nb_total),
        dtype=np.float64,
    ).tocsr()
    Auc = sp.vstack([base_Auc, box_Auc], format="csr")
    Aub = sp.vstack([base_Aub, box_Aub], format="csr")
    ub_rhs = np.concatenate([base_ub, 0.5 * np.ones(4 * r, dtype=np.float64)])

    if domain_cuts:
        dom_Auc, dom_Aub, dom_rhs = _scurve_domain_cut_matrices(
            hz,
            wide_idx,
            a,
            b_seg,
            owner,
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
            fa,
            fb,
            owner,
            z_cols,
            ng_total,
            nb_total,
            func(lb_w),
            func(ub_w),
        )
        Auc = sp.vstack([Auc, dom_Auc, rng_Auc], format="csr")
        Aub = sp.vstack([Aub, dom_Aub, rng_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, dom_rhs, rng_rhs])

    if graph_cuts:
        graph_Auc, graph_Aub, graph_rhs = _scurve_graph_cut_matrices(
            hz,
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            a,
            b_seg,
            owner,
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

    Auc.eliminate_zeros()
    Aub.eliminate_zeros()

    out = SparseHZono(
        out_c, out_Gc, out_Gb, Ac, Ab, b, Auc, Aub, ub_rhs,
    )
    if return_info:
        info = {
            "wide_idx": wide_idx,
            "m": m,
            "compressed": True,
            "pruned": True,
            "r": r,
            "owner_arr": owner,
            "seg_count": seg_count.astype(np.int32, copy=False),
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
        return out, (m, int(narrow_idx.size)), info
    return out


def sparse_hz_apply_scurve_piecewise_full(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    K: int = 2,
    compressed: bool = False,
    func=_sigmoid_np,
    dfunc=_sigmoid_deriv_np,
    inflection: float = 0.0,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
    return_info: bool = False,
):
    """Full sparse HZ S-curve encoding, preserving zero-width side segments."""

    if pre_bounds is None:
        pre_bounds = sparse_hz_fast_bounds(hz)
    grid = str(grid or "uniform")
    lb, ub = _bounds_arrays(pre_bounds, hz.n_out)
    wide = (ub - lb) > 1e-12
    wide_idx = np.nonzero(wide)[0].astype(np.int32)
    narrow_idx = np.nonzero(~wide)[0].astype(np.int32)
    n = hz.n_out
    ng = hz.n_cont
    nb = hz.n_bin
    m = int(wide_idx.size)

    out_c = np.zeros(n, dtype=np.float64)
    if narrow_idx.size:
        out_c[narrow_idx] = func(hz.c[narrow_idx])
    if m == 0:
        out = SparseHZono(
            out_c,
            sparse_empty(n, ng),
            sparse_empty(n, nb),
            hz.Ac,
            hz.Ab,
            hz.b,
            hz.Auc,
            hz.Aub,
            hz.ub,
        )
        if return_info:
            return out, (0, int(narrow_idx.size)), {
                "wide_idx": wide_idx,
                "m": 0,
                "compressed": bool(compressed),
                "grid": grid,
            }
        return out

    K_side = max(1, int(K))
    lb_w = lb[wide_idx]
    ub_w = ub[wide_idx]
    pivot = np.maximum(np.minimum(float(inflection), ub_w), lb_w)
    if grid == "uniform":
        sid = np.arange(K_side, dtype=np.float64).reshape(-1, 1)
        w_left = (pivot - lb_w).reshape(1, -1) / K_side
        w_right = (ub_w - pivot).reshape(1, -1) / K_side
        a_left = lb_w.reshape(1, -1) + sid * w_left
        a_right = pivot.reshape(1, -1) + sid * w_right
        a = np.vstack([a_left, a_right])
        b_seg = np.vstack([a_left + w_left, a_right + w_right])
    else:
        a = np.zeros((2 * K_side, m), dtype=np.float64)
        b_seg = np.zeros((2 * K_side, m), dtype=np.float64)
        for j in range(m):
            br_l = _scurve_breakpoints(
                lb_w[j],
                pivot[j],
                K_side,
                grid=grid,
                func=func,
                dfunc=dfunc,
            )
            br_r = _scurve_breakpoints(
                pivot[j],
                ub_w[j],
                K_side,
                grid=grid,
                func=func,
                dfunc=dfunc,
            )
            a[:K_side, j] = br_l[:-1]
            b_seg[:K_side, j] = br_l[1:]
            a[K_side:, j] = br_r[:-1]
            b_seg[K_side:, j] = br_r[1:]
    S = int(2 * K_side)

    fa = func(a)
    fb = func(b_seg)
    deriv_a = dfunc(a)
    deriv_b = dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = np.abs(deriv_a - deriv_b) < 1e-10

    denom = deriv_b - deriv_a
    safe_denom = np.where(nearly_linear, 1.0, denom)
    p1 = (fb - fa + deriv_b * a - deriv_a * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = deriv_b * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = deriv_a * (p2 - a) / 2.0

    half_width = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = np.linspace(0.0, 1.0, 50, dtype=np.float64).reshape(50, 1, 1)
    pts = a.reshape(1, S, m) + t_pts * (b_seg - a).reshape(1, S, m)
    f_pts = func(pts)
    resid = f_pts - (slope.reshape(1, S, m) * pts + (fa - slope * a).reshape(1, S, m))
    max_err = np.max(np.abs(resid), axis=0)
    g1x_lin = half_width
    g1y_lin = slope * half_width
    g2x_lin = np.zeros_like(half_width)
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
    owner = np.broadcast_to(np.arange(m, dtype=np.int32).reshape(1, -1), (S, m)).reshape(-1)
    out_Gc = sp.coo_matrix(
        (
            np.concatenate([g1_y.reshape(-1), g2_y.reshape(-1)]),
            (
                np.concatenate([wide_rows, wide_rows]),
                np.concatenate([g1_cols, g2_cols]),
            ),
        ),
        shape=(n, ng_total),
        dtype=np.float64,
    ).tocsr()
    out_Gb = sp.coo_matrix(
        (-centers_y.reshape(-1) / 2.0, (wide_rows, z_cols)),
        shape=(n, nb_total),
        dtype=np.float64,
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
    add_c = _coo_appender(rr_c, cc_c, dd_c)
    add_b = _coo_appender(rr_b, cc_b, dd_b)

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

    eq_Ac = _coo_matrix_from_parts(rr_c, cc_c, dd_c, (n_eq_new, ng_total))
    eq_Ab = _coo_matrix_from_parts(rr_b, cc_b, dd_b, (n_eq_new, nb_total))
    Ac = sp.vstack([sparse_pad_cols(hz.Ac, ng_total), eq_Ac], format="csr")
    Ab = sp.vstack([sparse_pad_cols(hz.Ab, nb_total), eq_Ab], format="csr")
    Ac.eliminate_zeros()
    Ab.eliminate_zeros()

    base_Auc = sparse_pad_cols(
        hz.Auc if hz.Auc is not None else sparse_empty(0, hz.n_cont),
        ng_total,
    )
    base_Aub = sparse_pad_cols(
        hz.Aub if hz.Aub is not None else sparse_empty(0, hz.n_bin),
        nb_total,
    )
    base_ub = hz.ub if hz.ub is not None else np.zeros(0, dtype=np.float64)
    if compressed:
        box_rows = 4 * np.arange(S * m, dtype=np.int32)
        box_Auc = sp.coo_matrix(
            (
                np.concatenate([
                    np.ones(S * m),
                    -np.ones(S * m),
                    np.ones(S * m),
                    -np.ones(S * m),
                ]),
                (
                    np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                    np.concatenate([g1_cols, g1_cols, g2_cols, g2_cols]),
                ),
            ),
            shape=(4 * S * m, ng_total),
            dtype=np.float64,
        ).tocsr()
        box_Aub = sp.coo_matrix(
            (
                0.5 * np.ones(4 * S * m, dtype=np.float64),
                (
                    np.concatenate([box_rows, box_rows + 1, box_rows + 2, box_rows + 3]),
                    np.concatenate([z_cols, z_cols, z_cols, z_cols]),
                ),
            ),
            shape=(4 * S * m, nb_total),
            dtype=np.float64,
        ).tocsr()
        Auc = sp.vstack([base_Auc, box_Auc], format="csr")
        Aub = sp.vstack([base_Aub, box_Aub], format="csr")
        ub_rhs = np.concatenate([base_ub, 0.5 * np.ones(4 * S * m, dtype=np.float64)])
    else:
        Auc = base_Auc
        Aub = base_Aub
        ub_rhs = base_ub

    if domain_cuts:
        dom_Auc, dom_Aub, dom_rhs = _scurve_domain_cut_matrices(
            hz,
            wide_idx,
            a.reshape(-1),
            b_seg.reshape(-1),
            owner,
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
            owner,
            z_cols,
            ng_total,
            nb_total,
            func(lb_w),
            func(ub_w),
        )
        Auc = sp.vstack([Auc, dom_Auc, rng_Auc], format="csr")
        Aub = sp.vstack([Aub, dom_Aub, rng_Aub], format="csr")
        ub_rhs = np.concatenate([ub_rhs, dom_rhs, rng_rhs])

    if graph_cuts:
        graph_Auc, graph_Aub, graph_rhs = _scurve_graph_cut_matrices(
            hz,
            out_c,
            out_Gc,
            out_Gb,
            wide_idx,
            a.reshape(-1),
            b_seg.reshape(-1),
            owner,
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

    Auc.eliminate_zeros()
    Aub.eliminate_zeros()

    out = SparseHZono(
        out_c,
        out_Gc,
        out_Gb,
        Ac,
        Ab,
        np.concatenate([hz.b, eq_b]),
        Auc,
        Aub,
        ub_rhs,
    )
    if return_info:
        return out, (m, int(narrow_idx.size)), {
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
            "compressed": bool(compressed),
            "grid": grid,
        }
    return out


def sparse_hz_apply_sigmoid_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    K: int = 2,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
) -> SparseHZono:
    return sparse_hz_apply_scurve_piecewise(
        hz,
        pre_bounds,
        K=K,
        func=_sigmoid_np,
        dfunc=_sigmoid_deriv_np,
        inflection=0.0,
        torch_func=torch.sigmoid,
        torch_dfunc=lambda x: torch.sigmoid(x) * (1.0 - torch.sigmoid(x)),
        domain_cuts=domain_cuts,
        graph_cuts=graph_cuts,
        grid=grid,
    )


def sparse_hz_apply_tanh_piecewise(
    hz: SparseHZono,
    pre_bounds: Optional[Bounds] = None,
    *,
    K: int = 1,
    domain_cuts: bool = False,
    graph_cuts: bool = False,
    grid: str = "uniform",
) -> SparseHZono:
    return sparse_hz_apply_scurve_piecewise(
        hz,
        pre_bounds,
        K=K,
        func=_tanh_np,
        dfunc=_tanh_deriv_np,
        inflection=0.0,
        torch_func=torch.tanh,
        torch_dfunc=lambda x: 1.0 - torch.tanh(x) ** 2,
        domain_cuts=domain_cuts,
        graph_cuts=graph_cuts,
        grid=grid,
    )



def _broadcast_param(value, n: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().double().numpy().reshape(-1)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if n % arr.size == 0:
        return np.tile(arr, n // arr.size)
    raise ValueError(f"cannot broadcast parameter of size {arr.size} to {n}")


def sparse_hz_add_const(hz: SparseHZono, bias) -> SparseHZono:
    return SparseHZono(
        c=hz.c + _broadcast_param(bias, hz.n_out),
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def sparse_hz_scale(hz: SparseHZono, scale) -> SparseHZono:
    a = _broadcast_param(scale, hz.n_out)
    D = sp.diags(a, format="csr")
    return sparse_hz_linear(hz, D, None)


def sparse_hz_gather_rows(hz: SparseHZono, rows: Sequence[int]) -> SparseHZono:
    row_idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    return SparseHZono(
        c=hz.c[row_idx].copy(),
        Gc=hz.Gc[row_idx].tocsr(),
        Gb=hz.Gb[row_idx].tocsr(),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
    )


def _csr_equal(a: sp.csr_matrix, b: sp.csr_matrix, *, atol: float = 0.0) -> bool:
    if a.shape != b.shape:
        return False
    d = (a - b).tocsr()
    d.eliminate_zeros()
    if d.nnz == 0:
        return True
    return bool(np.max(np.abs(d.data)) <= atol)


def _constraints_start_with(
    Ac_big: sp.csr_matrix,
    Ab_big: sp.csr_matrix,
    b_big: np.ndarray,
    Ac_small: sp.csr_matrix,
    Ab_small: sp.csr_matrix,
    b_small: np.ndarray,
) -> bool:
    n = int(Ac_small.shape[0])
    if n > Ac_big.shape[0] or n != Ab_small.shape[0] or n > Ab_big.shape[0]:
        return False
    return (
        _csr_equal(Ac_big[:n], Ac_small)
        and _csr_equal(Ab_big[:n], Ab_small)
        and np.array_equal(b_big[:n], b_small)
    )


def _merge_equalities(parts: List[SparseHZono], n_cont: int, n_bin: int):
    Ac = sparse_empty(0, n_cont)
    Ab = sparse_empty(0, n_bin)
    b = np.zeros(0, dtype=np.float64)
    for part in parts:
        pAc = sparse_pad_cols(part.Ac, n_cont)
        pAb = sparse_pad_cols(part.Ab, n_bin)
        if _constraints_start_with(pAc, pAb, part.b, Ac, Ab, b):
            Ac, Ab, b = pAc, pAb, part.b
        elif _constraints_start_with(Ac, Ab, b, pAc, pAb, part.b):
            continue
        elif pAc.shape[0]:
            Ac = sp.vstack([Ac, pAc], format="csr")
            Ab = sp.vstack([Ab, pAb], format="csr")
            b = np.concatenate([b, part.b])
    return Ac, Ab, b


def _merge_uppers(parts: List[SparseHZono], n_cont: int, n_bin: int):
    Auc = sparse_empty(0, n_cont)
    Aub = sparse_empty(0, n_bin)
    ub = np.zeros(0, dtype=np.float64)
    for part in parts:
        pAuc = sparse_pad_cols(part.Auc if part.Auc is not None else sparse_empty(0, part.n_cont), n_cont)
        pAub = sparse_pad_cols(part.Aub if part.Aub is not None else sparse_empty(0, part.n_bin), n_bin)
        pub = part.ub if part.ub is not None else np.zeros(0, dtype=np.float64)
        if _constraints_start_with(pAuc, pAub, pub, Auc, Aub, ub):
            Auc, Aub, ub = pAuc, pAub, pub
        elif _constraints_start_with(Auc, Aub, ub, pAuc, pAub, pub):
            continue
        elif pAuc.shape[0]:
            Auc = sp.vstack([Auc, pAuc], format="csr")
            Aub = sp.vstack([Aub, pAub], format="csr")
            ub = np.concatenate([ub, pub])
    return Auc, Aub, ub


def sparse_hz_concat(parts: Iterable[SparseHZono]) -> SparseHZono:
    parts = list(parts)
    if not parts:
        raise ValueError("sparse_hz_concat requires at least one part")
    n_cont = max(p.n_cont for p in parts)
    n_bin = max(p.n_bin for p in parts)
    padded = [sparse_hz_pad_frame(p, n_cont, n_bin) for p in parts]
    Ac, Ab, b = _merge_equalities(padded, n_cont, n_bin)
    Auc, Aub, ub = _merge_uppers(padded, n_cont, n_bin)
    Gc = sp.vstack([p.Gc for p in padded], format="csr")
    Gb = sp.vstack([p.Gb for p in padded], format="csr")
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.concatenate([p.c for p in padded]),
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
    )


def sparse_hz_add_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    """Exact sum when both HZs use the same global generator frame."""

    n_cont = max(x.n_cont, y.n_cont)
    n_bin = max(x.n_bin, y.n_bin)
    xp = sparse_hz_pad_frame(x, n_cont, n_bin)
    yp = sparse_hz_pad_frame(y, n_cont, n_bin)
    if xp.n_out != yp.n_out:
        raise ValueError(f"add shape mismatch: {xp.n_out} vs {yp.n_out}")
    Ac, Ab, b = _merge_equalities([xp, yp], n_cont, n_bin)
    Auc, Aub, ub = _merge_uppers([xp, yp], n_cont, n_bin)
    Gc = (xp.Gc + yp.Gc).tocsr()
    Gb = (xp.Gb + yp.Gb).tocsr()
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=xp.c + yp.c,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
    )


def sparse_hz_sub_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    return sparse_hz_add_same_frame(x, sparse_hz_scale(y, -1.0))


__all__ = [
    "sparse_empty",
    "sparse_matmul_const_matrix_from_layer",
    "sparse_avgpool2d_matrix_from_layer",
    "sparse_conv2d_matrix_from_layer",
    "sparse_convtranspose2d_matrix_from_layer",
    "sparse_dense_matrix_from_layer",
    "sparse_hz_add_const",
    "sparse_hz_add_same_frame",
    "sparse_hz_apply_avgpool2d_layer",
    "sparse_hz_apply_conv2d_layer",
    "sparse_hz_apply_convtranspose2d_layer",
    "sparse_hz_apply_dense_layer",
    "sparse_hz_apply_matmul_const_layer",
    "sparse_hz_apply_matmul_product_interval_layer",
    "sparse_hz_apply_maxpool2d_layer",
    "sparse_hz_apply_relu_exact",
    "sparse_hz_apply_relu_exact_tightened",
    "sparse_hz_apply_softmax_simplex_layer",
    "sparse_hz_apply_scurve_piecewise",
    "sparse_hz_apply_scurve_piecewise_full",
    "sparse_hz_apply_sigmoid_piecewise",
    "sparse_hz_apply_tanh_piecewise",
    "sparse_hz_concat",
    "sparse_hz_fast_bounds",
    "sparse_hz_from_bounds",
    "sparse_hz_gather_rows",
    "sparse_hz_gather_rows_like",
    "sparse_hz_is_point",
    "sparse_hz_linear",
    "sparse_hz_apply_per_batch_linear",
    "sparse_hz_pad_frame",
    "sparse_hz_reduce_sum_rows",
    "sparse_hz_row_lp_bound",
    "sparse_hz_scale",
    "sparse_hz_sub_same_frame",
    "sparse_hz_tighten_relu_bounds",
    "sparse_expand_row_indices",
    "sparse_gather_row_indices",
    "sparse_maxpool2d_candidate_rows",
    "sparse_pad_cols",
    "sparse_reduce_sum_row_indices",
    "sparse_slice_row_indices",
    "sparse_upsample_nearest_row_indices",
]
