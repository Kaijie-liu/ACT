#!/usr/bin/env python
"""Lazy exact-HZ structural census for CIFAR-style VNNLIB instances.

This module also provides the lightweight benchmark-loading helpers used by the
packaged sparse exact-HZ probe.

  y = c + Gc xi_c + Gb xi_b,  Ac xi_c + Ab xi_b = b

The script tracks only sparse support structure for Gc/Gb plus exact ReLU
constraint sizes. Numeric coefficients are deliberately not materialized here;
that belongs in the later solver-facing sparse/lazy implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


BENCH_ROOT = Path(os.environ.get(
    "ACT_VNNCOMP_BENCH_ROOT",
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks",
))


@dataclass
class StructState:
    Sc: sp.csr_matrix
    Sb: sp.csr_matrix
    n_cont: int
    n_bin: int
    eq_rows: int
    eq_nnz: int

    @property
    def n_out(self) -> int:
        return int(self.Sc.shape[0])

    @property
    def value_nnz(self) -> int:
        return int(self.Sc.nnz + self.Sb.nnz)


@dataclass
class LayerRow:
    lid: int
    kind: str
    n_out: int
    active: int = 0
    inactive: int = 0
    unstable: int = 0
    n_cont: int = 0
    n_bin: int = 0
    eq_rows: int = 0
    value_nnz: int = 0
    eq_nnz: int = 0
    dense_value_cells: int = 0
    dense_eq_cells: int = 0
    wall_s: float = 0.0
    note: str = ""


def _shape4(shape: Iterable[int]) -> Tuple[int, int, int, int]:
    vals = tuple(int(x) for x in shape)
    if len(vals) == 4:
        return vals
    if len(vals) == 3:
        c, h, w = vals
        return (1, c, h, w)
    raise ValueError(f"expected NCHW or CHW shape, got {vals}")


def _pair(x) -> Tuple[int, int]:
    if isinstance(x, int):
        return (int(x), int(x))
    vals = tuple(int(v) for v in x)
    if len(vals) == 2:
        return vals
    if len(vals) == 4:
        # ONNX pads are often [top, left, bottom, right]. This census supports
        # symmetric padding; asymmetric padding needs a separate pattern builder.
        if vals[0] != vals[2] or vals[1] != vals[3]:
            raise ValueError(f"asymmetric padding not supported in census: {vals}")
        return (vals[0], vals[1])
    raise ValueError(f"expected scalar/pair/4-pad tuple, got {vals}")


def _empty(rows: int, cols: int) -> sp.csr_matrix:
    return sp.csr_matrix((int(rows), int(cols)), dtype=bool)


def _as_bool_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    out = mat.tocsr().astype(bool)
    out.eliminate_zeros()
    return out


def _pad_cols(mat: sp.csr_matrix, cols: int) -> sp.csr_matrix:
    if mat.shape[1] == cols:
        return mat
    if mat.shape[1] > cols:
        raise ValueError(f"cannot shrink sparse matrix from {mat.shape[1]} to {cols}")
    return sp.hstack([mat, _empty(mat.shape[0], cols - mat.shape[1])], format="csr", dtype=bool)


def _pad_state(st: StructState, n_cont: int, n_bin: int) -> StructState:
    return StructState(
        Sc=_pad_cols(st.Sc, n_cont),
        Sb=_pad_cols(st.Sb, n_bin),
        n_cont=n_cont,
        n_bin=n_bin,
        eq_rows=st.eq_rows,
        eq_nnz=st.eq_nnz,
    )


def _union_rows(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for union: {a.shape} vs {b.shape}")
    return _as_bool_csr(a + b)


def _matmul_support(pattern: sp.csr_matrix, support: sp.csr_matrix) -> sp.csr_matrix:
    if support.shape[1] == 0:
        return _empty(pattern.shape[0], 0)
    return _as_bool_csr(pattern @ support)


_CONV_PATTERN_CACHE: Dict[Tuple, sp.csr_matrix] = {}


def _mask_key(mask: np.ndarray) -> Tuple[str, Tuple[int, ...]]:
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))
    digest = hashlib.sha1(packed.tobytes()).hexdigest()
    return (digest, tuple(int(x) for x in mask.shape))


def _conv2d_pattern(L) -> sp.csr_matrix:
    import torch

    weight = L.params["weight"].detach().cpu()
    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight)
    out_ch, in_ch_per_group, kh, kw = [int(v) for v in weight.shape]
    groups = int(L.params.get("groups", 1))
    stride = _pair(L.params.get("stride", 1))
    padding = _pair(L.params.get("padding", 0))
    dilation = _pair(L.params.get("dilation", 1))
    input_shape = _shape4(L.params["input_shape"])
    output_shape = _shape4(L.params["output_shape"])
    bsz, in_ch, in_h, in_w = input_shape
    out_bsz, out_ch_shape, out_h, out_w = output_shape
    if bsz != out_bsz or out_ch != out_ch_shape:
        raise ValueError(f"conv shape mismatch: weight={tuple(weight.shape)} output={output_shape}")
    if in_ch != in_ch_per_group * groups:
        raise ValueError(f"conv groups mismatch: in_ch={in_ch}, per_group={in_ch_per_group}, groups={groups}")

    mask = (weight.numpy() != 0)
    dense_kernel = bool(mask.all())
    key = (
        input_shape,
        output_shape,
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        groups,
        "dense" if dense_kernel else _mask_key(mask),
    )
    cached = _CONV_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    n_out = bsz * out_ch * out_h * out_w
    n_in = bsz * in_ch * in_h * in_w
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []

    out_ch_per_group = out_ch // groups
    for n in range(bsz):
        for co in range(out_ch):
            g = co // out_ch_per_group
            ci_base = g * in_ch_per_group
            for oh in range(out_h):
                ih0 = oh * stride[0] - padding[0]
                for ow in range(out_w):
                    iw0 = ow * stride[1] - padding[1]
                    out_idx = ((n * out_ch + co) * out_h + oh) * out_w + ow
                    col_acc: List[int] = []
                    for ci_local in range(in_ch_per_group):
                        ci = ci_base + ci_local
                        for r in range(kh):
                            ih = ih0 + r * dilation[0]
                            if ih < 0 or ih >= in_h:
                                continue
                            for c in range(kw):
                                if not dense_kernel and not mask[co, ci_local, r, c]:
                                    continue
                                iw = iw0 + c * dilation[1]
                                if iw < 0 or iw >= in_w:
                                    continue
                                col_acc.append(((n * in_ch + ci) * in_h + ih) * in_w + iw)
                    if col_acc:
                        arr = np.asarray(col_acc, dtype=np.int32)
                        rows.append(np.full(arr.size, out_idx, dtype=np.int32))
                        cols.append(arr)

    if rows:
        rr = np.concatenate(rows)
        cc = np.concatenate(cols)
        data = np.ones(rr.size, dtype=bool)
    else:
        rr = cc = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=bool)
    pat = sp.csr_matrix((data, (rr, cc)), shape=(n_out, n_in), dtype=bool)
    pat.eliminate_zeros()
    _CONV_PATTERN_CACHE[key] = pat
    return pat


def _dense_pattern(L) -> sp.csr_matrix:
    weight = L.params["weight"].detach().cpu().numpy()
    mask = np.abs(weight) > 0.0
    return sp.csr_matrix(mask, dtype=bool)


def _input_spec_state(inspec, n_in: int) -> StructState:
    lb = inspec.lb.detach().cpu().numpy().reshape(-1)
    ub = inspec.ub.detach().cpu().numpy().reshape(-1)
    if lb.size != n_in:
        raise ValueError(f"input spec size {lb.size} != input layer size {n_in}")
    rad = (ub - lb) * 0.5
    idx = np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
    cols = np.arange(idx.size, dtype=np.int32)
    Sc = sp.csr_matrix((np.ones(idx.size, dtype=bool), (idx, cols)), shape=(n_in, idx.size), dtype=bool)
    return StructState(Sc=Sc, Sb=_empty(n_in, 0), n_cont=int(idx.size), n_bin=0, eq_rows=0, eq_nnz=0)


def _relu_exact(L, st: StructState, pre_bounds) -> Tuple[StructState, Tuple[int, int, int]]:
    lb = pre_bounds.lb.detach().cpu().numpy().reshape(-1)
    ub = pre_bounds.ub.detach().cpu().numpy().reshape(-1)
    if lb.size != st.n_out:
        raise ValueError(f"relu {L.id}: bounds size {lb.size} != state rows {st.n_out}")

    active = lb >= 0.0
    inactive = ub <= 0.0
    unstable = ~(active | inactive)
    active_idx = np.nonzero(active)[0].astype(np.int32)
    unstable_idx = np.nonzero(unstable)[0].astype(np.int32)
    k = int(unstable_idx.size)

    old_c = int(st.n_cont)
    old_b = int(st.n_bin)
    new_c = old_c + 4 * k
    new_b = old_b + k

    # Output value rows: active keeps the preactivation support; inactive is zero;
    # unstable exact ReLU output is xi2 for that neuron in the eq_lagr encoding.
    blocks_c: List[sp.csr_matrix] = []
    if active_idx.size:
        act_c = st.Sc[active_idx].tocoo()
        blocks_c.append(
            sp.coo_matrix(
                (np.ones(act_c.nnz, dtype=bool), (active_idx[act_c.row], act_c.col)),
                shape=(st.n_out, new_c),
                dtype=bool,
            ).tocsr()
        )
    if k:
        xi2_cols = old_c + k + np.arange(k, dtype=np.int32)
        blocks_c.append(
            sp.csr_matrix(
                (np.ones(k, dtype=bool), (unstable_idx, xi2_cols)),
                shape=(st.n_out, new_c),
                dtype=bool,
            )
        )
    Sc = _as_bool_csr(sum(blocks_c[1:], blocks_c[0]) if blocks_c else _empty(st.n_out, new_c))

    if active_idx.size:
        act_b = st.Sb[active_idx].tocoo()
        Sb = sp.coo_matrix(
            (np.ones(act_b.nnz, dtype=bool), (active_idx[act_b.row], act_b.col)),
            shape=(st.n_out, new_b),
            dtype=bool,
        ).tocsr()
    else:
        Sb = _empty(st.n_out, new_b)

    pre_row_nnz = np.diff(st.Sc.indptr) + np.diff(st.Sb.indptr)
    # Exact ReLU eq_lagr per unstable neuron:
    #   xi1 + xi3 + z = 1                    -> 3 nnz
    #   xi2 + xi4 - z = 1                    -> 3 nnz
    #   alpha/2 xi1 - beta/2 xi2 - pre + ... -> pre_nnz + 3 nnz
    eq_rows = st.eq_rows + 3 * k
    eq_nnz = st.eq_nnz + int(9 * k + pre_row_nnz[unstable_idx].sum())
    out = StructState(Sc=Sc, Sb=Sb, n_cont=new_c, n_bin=new_b, eq_rows=eq_rows, eq_nnz=eq_nnz)
    return out, (int(active.sum()), int(inactive.sum()), k)


def _identity_like(L, st: StructState) -> StructState:
    n_out = len(L.out_vars)
    if n_out == st.n_out:
        return st
    if n_out == 0:
        return StructState(_empty(0, st.n_cont), _empty(0, st.n_bin), st.n_cont, st.n_bin, st.eq_rows, st.eq_nnz)
    raise ValueError(f"{L.kind} {L.id}: unsupported reshape from {st.n_out} to {n_out}")


def _state_for_pred(states: Dict[int, StructState], net, layer_id: int, pos: int = 0) -> StructState:
    pred = net.preds.get(layer_id, [None])[pos]
    if pred is None:
        raise KeyError(f"layer {layer_id} has no predecessor {pos}")
    return states[pred]


def _record(L, st: StructState, active=0, inactive=0, unstable=0, wall_s=0.0, note="") -> LayerRow:
    n_vars = st.n_cont + st.n_bin
    return LayerRow(
        lid=int(L.id),
        kind=str(L.kind).upper(),
        n_out=st.n_out,
        active=int(active),
        inactive=int(inactive),
        unstable=int(unstable),
        n_cont=int(st.n_cont),
        n_bin=int(st.n_bin),
        eq_rows=int(st.eq_rows),
        value_nnz=st.value_nnz,
        eq_nnz=int(st.eq_nnz),
        dense_value_cells=int(st.n_out * n_vars),
        dense_eq_cells=int(st.eq_rows * n_vars),
        wall_s=float(wall_s),
        note=note,
    )


def format_big(x: int) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}k"
    return str(x)


def _print_table(rows: List[LayerRow], max_rows: Optional[int] = None) -> None:
    shown = rows if max_rows is None else rows[:max_rows]
    header = (
        "lid kind       n_out    relu(a/off/u)   n_cont   n_bin  eq_rows  "
        "val_nnz   eq_nnz dense_val dense_eq   sec note"
    )
    print(header)
    print("-" * len(header))
    for r in shown:
        relu = f"{r.active}/{r.inactive}/{r.unstable}" if r.kind == "RELU" else "-"
        print(
            f"{r.lid:>3} {r.kind:<9} {r.n_out:>7} {relu:>15} "
            f"{format_big(r.n_cont):>8} {format_big(r.n_bin):>7} "
            f"{format_big(r.eq_rows):>8} {format_big(r.value_nnz):>8} "
            f"{format_big(r.eq_nnz):>8} {format_big(r.dense_value_cells):>9} "
            f"{format_big(r.dense_eq_cells):>8} {r.wall_s:>5.2f} {r.note}"
        )
    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more layer rows omitted")


def build_net_and_interval(bench: str, iid: int, device: str):
    import torch

    from act.back_end.core import Bounds as ABounds
    from act.back_end.transfer_functions import get_transfer_function, set_transfer_function_mode
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel
    from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch, get_onnx_input_shape
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.pipeline.verification.torch2act import TorchToACT

    base = BENCH_ROOT / bench
    rows = [line.strip().split(",") for line in open(base / "instances.csv") if line.strip()]
    onnx_path = base / rows[iid][0].replace("./", "")
    vnnlib_path = base / rows[iid][1].replace("./", "")

    input_shape = tuple(get_onnx_input_shape(onnx_path))
    pt = convert_onnx_to_pytorch(onnx_path).float().eval()
    lab = LabeledInputTensor(tensor=torch.zeros(input_shape, dtype=torch.float32), label=torch.tensor([0]))
    queries = parse_vnnlib_queries(vnnlib_path, labeled_tensor=lab)

    wrapped = VerifiableModel(
        input_layer=InputLayer(labeled_input=lab, shape=input_shape, dtype=torch.float32),
        input_spec=InputSpecLayer(queries[0][0]),
        model=pt,
        output_spec=OutputSpecLayer(queries[0][1]),
    )
    net = TorchToACT(wrapped).run()
    inspec = queries[0][0]
    lb = inspec.lb.detach().cpu().reshape(1, -1).to(torch.float32)
    ub = inspec.ub.detach().cpu().reshape(1, -1).to(torch.float32)
    if device == "cuda" and torch.cuda.is_available():
        lb = lb.cuda()
        ub = ub.cuda()
    ib = ABounds(lb=lb, ub=ub)

    before = {}
    after = {}
    set_transfer_function_mode("interval")
    tf = get_transfer_function()
    t0 = time.time()
    for L in net.layers:
        preds = net.preds.get(L.id, [])
        inb = ib if (L.id == 0 or not preds) else after[preds[0]].bounds
        before[L.id] = inb
        after[L.id] = tf.apply(L, inb, net, before, after)
    interval_s = time.time() - t0
    return onnx_path, vnnlib_path, input_shape, queries, net, before, after, interval_s


_format_big = format_big
_build_net_and_interval = build_net_and_interval


def _propagate_struct(net, queries, before, max_layers: Optional[int] = None) -> Tuple[Dict[int, StructState], List[LayerRow]]:
    states: Dict[int, StructState] = {}
    rows: List[LayerRow] = []
    global_c = 0
    global_b = 0
    global_eq_rows = 0
    global_eq_nnz = 0

    for idx, L in enumerate(net.layers):
        if max_layers is not None and idx >= max_layers:
            break
        t0 = time.time()
        kind = str(L.kind).upper()
        note = ""

        if kind == "INPUT":
            st = StructState(_empty(len(L.out_vars), 0), _empty(len(L.out_vars), 0), global_c, global_b, global_eq_rows, global_eq_nnz)
            states[L.id] = st
            rows.append(_record(L, st, wall_s=time.time() - t0, note=note))
            continue

        if kind == "INPUT_SPEC":
            st = _input_spec_state(queries[0][0], len(L.out_vars))
            global_c, global_b = st.n_cont, st.n_bin
            global_eq_rows, global_eq_nnz = st.eq_rows, st.eq_nnz
            states[L.id] = st
            rows.append(_record(L, st, wall_s=time.time() - t0, note="box gens"))
            continue

        if kind == "ASSERT":
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            st = _identity_like(L, prev)
            states[L.id] = st
            rows.append(_record(L, st, wall_s=time.time() - t0, note=note))
            continue

        if kind in {"FLATTEN", "RESHAPE", "SQUEEZE", "UNSQUEEZE", "TRANSPOSE"}:
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            st = _identity_like(L, prev)

        elif kind in {"BIAS", "SCALE", "BN"}:
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            st = _identity_like(L, prev)

        elif kind == "CONV2D":
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            pat = _conv2d_pattern(L)
            if pat.shape[1] != prev.n_out or pat.shape[0] != len(L.out_vars):
                raise ValueError(f"conv {L.id}: pattern {pat.shape} incompatible with state {prev.n_out} and out {len(L.out_vars)}")
            st = StructState(
                Sc=_matmul_support(pat, prev.Sc),
                Sb=_matmul_support(pat, prev.Sb),
                n_cont=global_c,
                n_bin=global_b,
                eq_rows=global_eq_rows,
                eq_nnz=global_eq_nnz,
            )
            note = f"Pnnz={format_big(pat.nnz)}"

        elif kind == "DENSE":
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            pat = _dense_pattern(L)
            if pat.shape[1] != prev.n_out or pat.shape[0] != len(L.out_vars):
                raise ValueError(f"dense {L.id}: pattern {pat.shape} incompatible with state {prev.n_out} and out {len(L.out_vars)}")
            st = StructState(
                Sc=_matmul_support(pat, prev.Sc),
                Sb=_matmul_support(pat, prev.Sb),
                n_cont=global_c,
                n_bin=global_b,
                eq_rows=global_eq_rows,
                eq_nnz=global_eq_nnz,
            )
            note = f"Pnnz={format_big(pat.nnz)}"

        elif kind == "ADD":
            a = _pad_state(_state_for_pred(states, net, L.id, 0), global_c, global_b)
            b = _pad_state(_state_for_pred(states, net, L.id, 1), global_c, global_b)
            st = StructState(
                Sc=_union_rows(a.Sc, b.Sc),
                Sb=_union_rows(a.Sb, b.Sb),
                n_cont=global_c,
                n_bin=global_b,
                eq_rows=global_eq_rows,
                eq_nnz=global_eq_nnz,
            )

        elif kind == "SUB":
            a = _pad_state(_state_for_pred(states, net, L.id, 0), global_c, global_b)
            b = _pad_state(_state_for_pred(states, net, L.id, 1), global_c, global_b)
            st = StructState(
                Sc=_union_rows(a.Sc, b.Sc),
                Sb=_union_rows(a.Sb, b.Sb),
                n_cont=global_c,
                n_bin=global_b,
                eq_rows=global_eq_rows,
                eq_nnz=global_eq_nnz,
            )

        elif kind == "RELU":
            prev = _pad_state(_state_for_pred(states, net, L.id), global_c, global_b)
            st, relu_counts = _relu_exact(L, prev, before[L.id])
            global_c, global_b = st.n_cont, st.n_bin
            global_eq_rows, global_eq_nnz = st.eq_rows, st.eq_nnz
            active, inactive, unstable = relu_counts
            states[L.id] = st
            rows.append(_record(L, st, active, inactive, unstable, time.time() - t0, note=note))
            continue

        else:
            raise NotImplementedError(f"unsupported layer kind for census: {kind} at id={L.id}")

        states[L.id] = st
        rows.append(_record(L, st, wall_s=time.time() - t0, note=note))
        if idx % 8 == 0:
            gc.collect()

    return states, rows


def _query_c_t(query):
    spec = query[1]
    if hasattr(spec, "c") and spec.c is not None:
        C = spec.c.detach().cpu().numpy().astype(np.float64)
        t = spec.d.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return C, t, str(spec.kind)
    raise ValueError("only OutputSpec.c/d style specs are supported by this census")


def _interval_hard_rivals(queries, final_bounds) -> Tuple[int, List[float]]:
    lb = final_bounds.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ub = final_bounds.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)
    hard = 0
    lows: List[float] = []
    for q in queries:
        C, t, kind = _query_c_t(q)
        if "UNSAFE_LINEAR" not in kind:
            continue
        c = C.reshape(C.shape[0], -1)
        lo = c.clip(min=0) @ lb + c.clip(max=0) @ ub - t
        m = float(np.min(lo))
        lows.append(m)
        if m <= 0.0:
            hard += 1
    return hard, lows


def _margin_support_stats(queries, final_state: StructState) -> Dict[str, float]:
    vals: List[int] = []
    for q in queries:
        C, _, _ = _query_c_t(q)
        pat = sp.csr_matrix(np.abs(C.reshape(C.shape[0], -1)) > 0.0, dtype=bool)
        Scm = _matmul_support(pat, final_state.Sc)
        Sbm = _matmul_support(pat, final_state.Sb)
        vals.append(int(Scm.nnz + Sbm.nnz))
    if not vals:
        return {"min": 0, "median": 0, "max": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cifar100_2024")
    ap.add_argument("--iid", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--show-layers", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    t0 = time.time()
    onnx_path, vnnlib_path, input_shape, queries, net, before, after, interval_s = build_net_and_interval(
        args.bench, args.iid, args.device
    )
    max_layers = args.max_layers or None
    states, rows = _propagate_struct(net, queries, before, max_layers=max_layers)

    print(f"bench={args.bench} iid={args.iid}")
    print(f"onnx={onnx_path.name}")
    print(f"vnnlib={vnnlib_path.name}")
    print(f"input_shape={input_shape} queries={len(queries)} layers={len(net.layers)} interval_s={interval_s:.2f}")
    print()
    _print_table(rows, max_rows=(args.show_layers or None))

    final_layer = net.layers[-1]
    final_pred = net.preds.get(final_layer.id, [None])[0]
    final_state = states.get(final_pred) or states.get(final_layer.id)
    final_bounds = after[final_pred].bounds if final_pred in after else after[final_layer.id].bounds
    hard, low_margins = _interval_hard_rivals(queries, final_bounds)
    margin_stats = _margin_support_stats(queries, final_state) if final_state is not None else {"min": 0, "median": 0, "max": 0}

    last = rows[-1]
    relu_rows = [r for r in rows if r.kind == "RELU"]
    unstable_total = sum(r.unstable for r in relu_rows)
    peak_value_nnz = max((r.value_nnz for r in rows), default=0)
    peak_dense_value = max((r.dense_value_cells for r in rows), default=0)
    peak_dense_eq = max((r.dense_eq_cells for r in rows), default=0)
    print()
    print("summary")
    print(f"  relu_layers={len(relu_rows)} unstable_total={format_big(unstable_total)}")
    print(f"  final_n_cont={format_big(last.n_cont)} final_n_bin={format_big(last.n_bin)} final_eq_rows={format_big(last.eq_rows)}")
    print(f"  final_value_nnz={format_big(last.value_nnz)} final_eq_nnz={format_big(last.eq_nnz)}")
    print(f"  peak_value_nnz={format_big(peak_value_nnz)}")
    print(f"  peak_dense_value_cells_if_materialized={format_big(peak_dense_value)}")
    print(f"  peak_dense_eq_cells_if_materialized={format_big(peak_dense_eq)}")
    print(f"  interval_hard_rivals={hard}/{len(queries)}")
    if low_margins:
        arr = np.asarray(low_margins)
        print(f"  interval_margin_lower_min/median/max={arr.min():.6g}/{np.median(arr):.6g}/{arr.max():.6g}")
    print(
        "  row_only_margin_support_nnz_min/median/max="
        f"{format_big(int(margin_stats['min']))}/"
        f"{format_big(int(margin_stats['median']))}/"
        f"{format_big(int(margin_stats['max']))}"
    )
    print(f"  total_wall_s={time.time() - t0:.2f}")

    if last.n_bin > 50_000:
        print("  diagnosis=binary_wall_after_representation_fix")
    elif hard > 0:
        print("  diagnosis=hard_rival_or_phase_fixing_needed")
    else:
        print("  diagnosis=interval_already_filters_all_rivals")


__all__ = [
    "StructState",
    "LayerRow",
    "build_net_and_interval",
    "format_big",
]


if __name__ == "__main__":
    main()
