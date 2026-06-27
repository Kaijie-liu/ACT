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
from act.back_end.hybridz_tf.sparse_ops import (  # noqa: E402
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
    sparse_gather_row_indices as _gather_row_idx,
    sparse_hz_apply_relu_exact as _backend_relu_exact,
    sparse_hz_apply_scurve_piecewise as _backend_scurve_piecewise,
    sparse_hz_apply_scurve_piecewise_full as _backend_scurve_piecewise_full,
    sparse_hz_apply_softmax_simplex_layer as _softmax_simplex_hz,
    sparse_hz_apply_matmul_product_interval_layer as _matmul_product_interval_hz,
    sparse_hz_check_center_witness as _check_center_witness,
    sparse_hz_extend_relu_center_witness as _extend_relu_center_witness,
    sparse_hz_extend_scurve_center_witness as _extend_scurve_center_witness,
    sparse_hz_from_bounds,
    sparse_hz_linear as _linear_apply,
    sparse_hz_pad_frame as _pad_hz,
    sparse_hz_scale as _scale_apply,
    sparse_hz_sub_same_frame as _sub_same_frame,
    sparse_maxpool2d_candidate_rows as _maxpool2d_candidate_rows,
    sparse_hz_tighten_relu_bounds as _tighten_relu_bounds,
    sparse_slice_row_indices as _slice_row_idx,
    sparse_upsample_nearest_row_indices as _upsample_nearest_row_idx,
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
    if drop_degenerate:
        if not compressed:
            raise ValueError("drop_degenerate sigmoid is currently implemented for compressed mode only")
        out, counts, meta = _backend_scurve_piecewise(
            hz,
            pre_bounds,
            K=K,
            func=func,
            dfunc=dfunc,
            inflection=0.0,
            domain_cuts=domain_cuts,
            graph_cuts=graph_cuts,
            grid=grid,
            return_info=True,
        )
    else:
        out, counts, meta = _backend_scurve_piecewise_full(
            hz,
            pre_bounds,
            K=K,
            compressed=compressed,
            func=func,
            dfunc=dfunc,
            inflection=0.0,
            domain_cuts=domain_cuts,
            graph_cuts=graph_cuts,
            grid=grid,
            return_info=True,
        )
    _sigmoid_piecewise.last_meta = meta
    return out, counts


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
        xi_c, xi_b, err = _extend_relu_center_witness(
            prev, out, xi_c, xi_b, meta, layer_id=layer_id
        )
        if err is not None:
            mark_witness_bad(err)

    def extend_scurve_witness(prev: SparseHZ, out: SparseHZ, meta: dict, layer_id: int, func) -> None:
        nonlocal xi_c, xi_b
        if not witness_ok:
            return
        xi_c, xi_b, err = _extend_scurve_center_witness(
            prev, out, xi_c, xi_b, meta, func, layer_id=layer_id
        )
        if err is not None:
            mark_witness_bad(err)

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
            rows = _upsample_nearest_row_idx(L, prev.n_out, len(L.out_vars))
            if rows is None:
                raise NotImplementedError(f"unsupported sparse UPSAMPLE shape at {L.id}")
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
            rows = _slice_row_idx(L, prev.n_out)
            if rows is None or rows.size != len(L.out_vars):
                raise NotImplementedError(f"unsupported sparse SLICE shape at {L.id}")
            hz = _gather_hz_rows(prev, rows)
        elif kind == "GATHER":
            prev = _pad_hz(source_hz(L), global_c, global_b)
            rows = _gather_row_idx(L, prev.n_out)
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
        witness_ok, witness_msg = _check_center_witness(final, xi_c, xi_b)
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
    curv_hz, curv_counts = _sigmoid_piecewise(
        scurve_hz,
        scurve_bounds,
        K=3,
        compressed=True,
        drop_degenerate=True,
        domain_cuts=True,
        graph_cuts=True,
        grid="curvature",
    )
    curv_meta = dict(getattr(_sigmoid_piecewise, "last_meta", {}))
    assert backend_counts == curv_counts == (3, 0)
    assert backend_meta["grid"] == "uniform"
    assert curv_meta["grid"] == "curvature"
    assert np.array_equal(np.asarray(backend_meta["wide_idx"]), np.asarray(curv_meta["wide_idx"]))
    for meta in (backend_meta, curv_meta):
        assert np.asarray(meta["owner_arr"]).size == int(meta["r"])
        assert np.asarray(meta["a"]).size == int(meta["r"])
        assert np.asarray(meta["b_seg"]).size == int(meta["r"])
    from act.back_end.solver.solver_hz_verdict import hz_base_feasibility, hz_row_max

    assert hz_base_feasibility(backend_hz, time_limit=2.0)[0] == "FEASIBLE"
    assert hz_base_feasibility(curv_hz, time_limit=2.0)[0] == "FEASIBLE"
    for row in (
        np.array([1.0, -0.5, 0.25], dtype=np.float64),
        np.array([-0.25, 1.0, 1.5], dtype=np.float64),
        np.array([0.3, 0.7, -1.0], dtype=np.float64),
    ):
        bmx = hz_row_max(backend_hz, row, integer=True, time_limit=2.0)
        cmx = hz_row_max(curv_hz, row, integer=True, time_limit=2.0)
        assert bmx is not None and cmx is not None
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
