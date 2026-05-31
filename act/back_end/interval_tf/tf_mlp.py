#===- act/back_end/interval_tf/tf_mlp.py - MLP Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   MLP Interval Transfer Functions. Provides interval-based transfer functions
#   for multi-layer perceptron operations including linear layers and
#   activation functions.
#
#===---------------------------------------------------------------------===#

import torch
import itertools
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta

# -------- MLP Basics --------
def tf_dense(L: Layer, Bin: Bounds) -> Fact:
    # Parameter names aligned with PyTorch: weight, bias, weight_pos, weight_neg
    W = L.params["weight"]
    W_pos = L.params.get("weight_pos", torch.clamp(W, min=0))
    W_neg = L.params.get("weight_neg", torch.clamp(W, max=0))
    b = L.params.get("bias", torch.zeros(W.shape[0]))

    # affine_bounds is documented as 2D ``[B, n_in] -> [B, n_out]``. Most
    # ACT models present DENSE input that way already, but attention-style
    # graphs (vit_2023) apply the linear per token: input_shape is
    # ``(1, T, in_features)`` and Bin arrives flattened as ``(1, T*in_features)``.
    # Reshape into ``[B*T, in_features]`` for the matmul, then fold back.
    # The reshape is sound: the linear is the same map applied independently
    # to every prefix index, so interval bounds compose row-wise.
    in_shape = L.params.get("input_shape")
    in_features = int(W.shape[1])
    n_per_sample = int(Bin.lb.shape[1])
    if (in_shape is not None and len(in_shape) >= 3
            and n_per_sample == in_features * (n_per_sample // in_features)
            and n_per_sample != in_features):
        prefix_numel = n_per_sample // in_features
        B_in = int(Bin.lb.shape[0])
        # Reshape to (B*prefix, in_features). The prefix indices stay
        # interval-bound independently per row, so the unflattened
        # apply-then-flatten is equivalent to per-row affine.
        lb2 = Bin.lb.reshape(B_in * prefix_numel, in_features)
        ub2 = Bin.ub.reshape(B_in * prefix_numel, in_features)
        B2 = affine_bounds(W_pos, W_neg, b, Bounds(lb2, ub2))
        out_lb = B2.lb.reshape(B_in, prefix_numel * int(W.shape[0]))
        out_ub = B2.ub.reshape(B_in, prefix_numel * int(W.shape[0]))
        B = Bounds(out_lb, out_ub)
    else:
        B = affine_bounds(W_pos, W_neg, b, Bin)
    C = ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"dense:{L.id}", "W": W, "b": b}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_bias(L: Layer, Bin: Bounds) -> Fact:
    c=L.params["c"]
    B=Bounds(Bin.lb+c, Bin.ub+c)
    C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"bias:{L.id}", "c": c}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_scale(L: Layer, Bin: Bounds) -> Fact:
    a=L.params["a"]
    lb=torch.where(a>=0, a*Bin.lb, a*Bin.ub); ub=torch.where(a>=0, a*Bin.ub, a*Bin.lb)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"scale:{L.id}", "a": a}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_relu(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.where(off,0.0,torch.where(on,l,0.0)); ub=torch.where(off,0.0,torch.where(on,u,u))
    if torch.any(amb):
        ua,la=u[amb],l[amb]; gap=ua-la
        finite=torch.isfinite(gap) & (gap>1e-12)
        s=torch.where(finite, ua/torch.clamp(gap,min=1e-12), torch.ones_like(gap))
        t=torch.where(finite, -s*la, torch.zeros_like(gap))
    else: s=t=torch.empty(0)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"relu:{L.id}",
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    # The PyTorch-side LeakyReLU converter stores the slope under
    # ``negative_slope`` (the nn.LeakyReLU attribute name); the older
    # back-end ops used ``alpha``. Accept either key; default to 0.01 which
    # matches both ONNX's LeakyReLU default and torch's nn.LeakyReLU default.
    a=float(L.params.get("alpha", L.params.get("negative_slope", 0.01)))
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    z=torch.zeros_like(l)
    lb=torch.minimum(a*torch.minimum(l,z), torch.maximum(l,z))
    ub=torch.maximum(a*torch.maximum(u,z), torch.maximum(u,z))
    if torch.any(amb):
        la,ua=l[amb],u[amb]; gap=ua-la
        # Guard against inf-inf=NaN: when gap is not finite, use slope=max(a,1)
        finite=torch.isfinite(gap) & (gap>1e-12)
        s=torch.where(finite, (ua-a*la)/torch.clamp(gap,min=1e-12), torch.full_like(gap,max(a,1.0)))
        t=torch.where(finite, a*la-s*la, torch.zeros_like(gap))
    else: s=t=torch.empty(0)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"lrelu:{L.id}","alpha":a,
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_abs(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; pos=l>=0; neg=u<=0; amb=~(pos|neg)
    lb=torch.minimum(torch.zeros_like(l), torch.minimum(torch.abs(l), torch.abs(u)))
    ub=torch.maximum(torch.abs(l), torch.abs(u))
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"abs:{L.id}",
        "idx_pos": torch.nonzero(pos,as_tuple=True)[0],
        "idx_neg": torch.nonzero(neg,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_clip(L: Layer, Bin: Bounds) -> Fact:
    a,b=L.params["a"],L.params["b"]; B=Bounds(torch.clamp(Bin.lb,a,b), torch.clamp(Bin.ub,a,b))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"clip:{L.id}","a":a,"b":b}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_floor(L: Layer, Bin: Bounds) -> Fact:
    """Sound interval transfer for y = floor(x).

    floor is monotone non-decreasing, so for x in [lb, ub]:
        y_lb = floor(lb), y_ub = floor(ub).
    This is sound AND tight (every integer in [y_lb, y_ub] is reachable
    by some x in the input interval whenever x's interval crosses the
    integer boundary)."""
    B = Bounds(torch.floor(Bin.lb), torch.floor(Bin.ub))
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"floor:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_ceil(L: Layer, Bin: Bounds) -> Fact:
    """Sound interval transfer for y = ceil(x). Mirror of floor."""
    B = Bounds(torch.ceil(Bin.lb), torch.ceil(Bin.ub))
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"ceil:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_round(L: Layer, Bin: Bounds) -> Fact:
    """Sound interval transfer for y = round(x) (PyTorch banker's /
    half-to-even rounding).

    Round-to-nearest-even is discontinuous at half integers but remains
    monotone non-decreasing. Therefore the tight interval image is
        [round(lb), round(ub)].
    For example, the singleton interval [0.5, 0.5] maps exactly to
    [0, 0], rather than the looser [0, 1]."""
    B = Bounds(torch.round(Bin.lb), torch.round(Bin.ub))
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"round:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_lut_bounds(L: Layer, Bin: Bounds) -> Fact:
    """Zero-indegree source layer emitting precomputed per-element bounds.

    Used by the cctsdb_yolo_2023 dynamic-Slice envelope (Option A in
    ``CCTSDB_DYNAMIC_SLICE_DESIGN.md``): a static initializer ``T`` is
    windowed by integer offsets bounded by VNNLIB input variables; at
    conversion time we precompute ``min/max T[candidate]`` for every
    output position over the candidate-window envelope. The result is a
    sealed (lb, ub) pair stored on the layer.

    ``Bin`` is the analyze-default sentinel (zero predecessors) and is
    ignored. We return the stored bounds as the layer's output. The
    bounds are batched to ``[B, numel]`` so downstream tfs see the
    standard 2-D layout."""
    lb = L.params["lb"]
    ub = L.params["ub"]
    if not isinstance(lb, torch.Tensor) or not isinstance(ub, torch.Tensor):
        raise TypeError(
            f"LUT_BOUNDS layer {L.id}: 'lb' and 'ub' params must be tensors; "
            f"got {type(lb).__name__} / {type(ub).__name__}"
        )
    if lb.shape != ub.shape:
        raise ValueError(
            f"LUT_BOUNDS layer {L.id}: lb shape {tuple(lb.shape)} != ub shape "
            f"{tuple(ub.shape)}"
        )
    if (lb > ub).any():
        raise ValueError(
            f"LUT_BOUNDS layer {L.id}: stored bounds have lb > ub at some "
            f"position — soundness invariant violated upstream of analyze"
        )
    # Flatten + restore batch dim to match the ACT 2-D bounds convention.
    flat_lb = lb.reshape(-1)
    flat_ub = ub.reshape(-1)
    B_size = Bin.lb.shape[0] if (Bin is not None and Bin.lb.dim() >= 1) else 1
    out_lb = flat_lb.unsqueeze(0).expand(B_size, -1).contiguous()
    out_ub = flat_ub.unsqueeze(0).expand(B_size, -1).contiguous()
    B = Bounds(out_lb, out_ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars), {"tag": f"lut_bounds:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def precompute_lut_envelope(
    T: torch.Tensor,
    *,
    window_size: tuple,
    starts_lb: tuple,
    starts_ub: tuple,
    steps: tuple = None,
) -> tuple:
    """Precompute the per-output-element envelope of a static initializer
    ``T`` under a bounded-start dynamic-Slice window.

    Args:
      T: the static initializer tensor (any rank).
      window_size: the FIXED output spatial size (one entry per axis;
        same rank as starts_lb / starts_ub). For cctsdb's slice_23 this
        is e.g. (height, width) of the crop.
      starts_lb / starts_ub: inclusive integer bounds on the per-axis
        start position. Both have one entry per axis. The candidate set
        for each axis is ``{starts_lb[i], starts_lb[i]+1, ...,
        starts_ub[i]}`` (intersected with the valid index range).
      steps: per-axis stride (default 1 each).

    Returns:
      ``(lb_tensor, ub_tensor)`` each shaped ``window_size``. For each
      output position ``j``, lb_tensor[j] = min over candidate windows of
      ``T[window_at(start, j)]``; ub_tensor[j] = max.

    Soundness: the candidate set is exactly the set of integer starts
    the runtime could choose given the input bounds. The min/max over
    this set is a sound (and minimal among element-wise envelopes)
    enclosure of every possible runtime output value.

    Complexity: for a window of total size W and total candidate count
    K (product of per-axis interval sizes), the naive implementation
    is O(W * K * T.dim()). We use a tighter loop-over-candidates with
    in-place per-window updates, which is O(K * W). Acceptable for
    CCTSDB (T ~ a small feature-map, K < 10^4).
    """
    import torch as _t
    T64 = T.to(torch.float64)
    rank = len(window_size)
    assert len(starts_lb) == rank and len(starts_ub) == rank, (
        f"window_size / starts_lb / starts_ub rank mismatch: {window_size} / "
        f"{starts_lb} / {starts_ub}"
    )
    if steps is None:
        steps = (1,) * rank
    assert len(steps) == rank
    for i in range(rank):
        if starts_lb[i] > starts_ub[i]:
            raise ValueError(
                f"LUT envelope: starts_lb[{i}]={starts_lb[i]} > "
                f"starts_ub[{i}]={starts_ub[i]}"
            )
        if window_size[i] < 1:
            raise ValueError(f"LUT envelope: window_size[{i}] must be >= 1")
    # Enumerate per-axis candidate start values (integer lattice).
    cand_per_axis = [list(range(int(starts_lb[i]), int(starts_ub[i]) + 1))
                     for i in range(rank)]

    out_lb = _t.full(window_size, float("inf"), dtype=torch.float64)
    out_ub = _t.full(window_size, float("-inf"), dtype=torch.float64)

    # Iterate over Cartesian product of per-axis starts. For each
    # (start_0, ..., start_{rank-1}), slice T at the corresponding
    # window and update min/max element-wise.
    from itertools import product
    for starts in product(*cand_per_axis):
        slices = []
        valid = True
        for i, s in enumerate(starts):
            end = s + window_size[i] * steps[i]
            if s < 0 or end > T64.shape[i]:
                valid = False
                break
            slices.append(slice(s, end, steps[i]))
        if not valid:
            # An out-of-range start is skipped: at runtime the verifier
            # cannot reach a window that extends past T's edge, because
            # the original ONNX Slice would have raised. Skipping is
            # sound (we don't widen for impossible runtime paths); if
            # ALL starts were skipped, the resulting envelope would be
            # ``[+inf, -inf]`` and the layer flagged as empty below.
            continue
        window = T64[tuple(slices)]
        # Reduce to the right rank (drop any trailing dims of T that
        # extend beyond the windowed axes — same shape contract as
        # ONNX Slice with starts/ends/axes covering the first `rank` dims).
        out_lb = torch.minimum(out_lb, window)
        out_ub = torch.maximum(out_ub, window)

    if torch.isinf(out_lb).any() or torch.isinf(out_ub).any():
        raise ValueError(
            "LUT envelope: every candidate window was out of range; "
            "the dynamic-Slice envelope is empty (likely a bug in "
            "starts_lb / starts_ub / window_size derivation)"
        )
    return out_lb, out_ub


def _tf_periodic_trig(
    L: Layer, Bin: Bounds, *, fn, max_phase: float, min_phase: float, tag: str
) -> Fact:
    """Sound element-wise interval envelope for sine/cosine.

    Extrema occur at ``phase + 2*pi*k``. Endpoint evaluation plus an
    extrema-containment check is exact for a scalar interval. Bounds are
    padded by a small floating-point guard only for the phase membership
    test, so a representational rounding at an extremum can widen rather
    than under-approximate.
    """
    l, u = Bin.lb, Bin.ub
    two_pi = 2.0 * torch.pi
    eps = torch.finfo(l.dtype).eps * 32.0 * (1.0 + torch.maximum(l.abs(), u.abs()))
    lp, up = l - eps, u + eps

    def contains(phase: float) -> torch.Tensor:
        first_k = torch.ceil((lp - phase) / two_pi)
        last_k = torch.floor((up - phase) / two_pi)
        return first_k <= last_k

    fl, fu = fn(l), fn(u)
    lb = torch.minimum(fl, fu)
    ub = torch.maximum(fl, fu)
    lb = torch.where(contains(min_phase), torch.full_like(lb, -1.0), lb)
    ub = torch.where(contains(max_phase), torch.full_like(ub, 1.0), ub)
    B = Bounds(lb, ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"{tag}:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_sin(L: Layer, Bin: Bounds) -> Fact:
    return _tf_periodic_trig(
        L, Bin, fn=torch.sin, max_phase=float(torch.pi / 2),
        min_phase=float(-torch.pi / 2), tag="sin",
    )


def tf_cos(L: Layer, Bin: Bounds) -> Fact:
    return _tf_periodic_trig(
        L, Bin, fn=torch.cos, max_phase=0.0,
        min_phase=float(torch.pi), tag="cos",
    )


def tf_add(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B=Bounds(Bx.lb+By.lb, Bx.ub+By.ub); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag":f"add:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_sub(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B = Bounds(Bx.lb - By.lb, Bx.ub - By.ub)
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag": f"sub:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_mul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    cand=torch.stack([Bx.lb*By.lb, Bx.lb*By.ub, Bx.ub*By.lb, Bx.ub*By.ub], dim=0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]),
        {"tag":f"mcc:{L.id}","lx":Bx.lb,"ux":Bx.ub,"ly":By.lb,"uy":By.ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_div(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    ly, uy = By.lb, By.ub
    crosses_zero = (ly <= 0) & (uy >= 0)
    cand = torch.stack([
        Bx.lb / ly,
        Bx.lb / uy,
        Bx.ub / ly,
        Bx.ub / uy,
    ], dim=0)
    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values
    big = 1e6
    lb = torch.where(crosses_zero, torch.full_like(lb, -big), lb)
    ub = torch.where(crosses_zero, torch.full_like(ub, +big), ub)
    B = Bounds(lb, ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag": f"div:{L.id}", "safe": not torch.any(crosses_zero).item()}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_matmul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    batch_size = Bx.lb.shape[0]
    x_shape = tuple(L.params["x_shape"])
    y_shape = tuple(L.params["y_shape"])
    A_lb = Bx.lb.view(batch_size, *x_shape).unsqueeze(-1)
    A_ub = Bx.ub.view(batch_size, *x_shape).unsqueeze(-1)
    B_lb = By.lb.view(batch_size, *y_shape).unsqueeze(-3)
    B_ub = By.ub.view(batch_size, *y_shape).unsqueeze(-3)
    c1, c2 = A_lb * B_lb, A_lb * B_ub
    c3, c4 = A_ub * B_lb, A_ub * B_ub
    lo = torch.minimum(torch.minimum(c1, c2), torch.minimum(c3, c4))
    hi = torch.maximum(torch.maximum(c1, c2), torch.maximum(c3, c4))
    out_lb = lo.sum(dim=-2).reshape(batch_size, -1)
    out_ub = hi.sum(dim=-2).reshape(batch_size, -1)
    Bres = Bounds(out_lb, out_ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bres)
    return Fact(Bres, C)


def tf_arg_extremum(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    in_shape = L.params.get("input_shape")
    axis = int(L.params.get("axis", 0))
    if in_shape is not None and axis < 0:
        axis += len(in_shape)
    axis_dim = int(in_shape[axis]) if in_shape else 1
    n_out = len(L.out_vars)
    lb = Bin.lb.new_zeros(batch_size, n_out)
    ub = Bin.lb.new_full((batch_size, n_out), float(max(0, axis_dim - 1)))
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_scatter_nd(L: Layer, Bdata: Bounds, Bidx: Bounds, Bupdates: Bounds) -> Fact:
    batch_size = Bdata.lb.shape[0]
    n = len(L.out_vars)
    data_lb = Bdata.lb.view(batch_size, -1)
    data_ub = Bdata.ub.view(batch_size, -1)
    if data_lb.shape[1] != n:
        repeat = (n + data_lb.shape[1] - 1) // data_lb.shape[1]
        data_lb = data_lb[:, :n] if data_lb.shape[1] > n else data_lb.repeat(1, repeat)[:, :n]
        data_ub = data_ub[:, :n] if data_ub.shape[1] > n else data_ub.repeat(1, repeat)[:, :n]
    if Bupdates.lb.numel() > 0:
        updates_lb = Bupdates.lb.view(batch_size, -1)
        updates_ub = Bupdates.ub.view(batch_size, -1)
        u_min = updates_lb.min(dim=1, keepdim=True).values.expand(batch_size, n)
        u_max = updates_ub.max(dim=1, keepdim=True).values.expand(batch_size, n)
        lb = torch.minimum(data_lb, u_min)
        ub = torch.maximum(data_ub, u_max)
    else:
        lb, ub = data_lb, data_ub
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)

def tf_concat(L: Layer, Bs: List[Bounds]) -> Fact:
    """Concatenates tensors on dim=-1 (feature axis)."""
    B=Bounds(torch.cat([b.lb for b in Bs], dim=-1), torch.cat([b.ub for b in Bs], dim=-1))
    C=ConSet(); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_constant(L: Layer, Bin: Bounds) -> Fact:
    # CONSTANT bounds come from L.params["value"], not from Bin. The graph
    # may carry spurious predecessor edges (e.g. ONNX initializers wired to
    # topological ancestors) whose Bounds have unrelated shape; ignoring
    # Bin keeps the output shaped to L.out_vars regardless of routing.
    B_size = Bin.lb.shape[0]
    val = L.params["value"].reshape(-1).to(Bin.lb)
    val_b = val.unsqueeze(0).expand(B_size, -1).contiguous()
    Bout = Bounds(val_b.clone(), val_b.clone())
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)

def tf_sign(L: Layer, Bin: Bounds) -> Fact:
    l, u = Bin.lb, Bin.ub
    lb = torch.sign(l)
    ub = torch.sign(u)
    B = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def _bcast(b: torch.Tensor, B: int, n: int) -> torch.Tensor:
    if b.dim() == 0:
        return b.reshape(1, 1).expand(B, n)
    if b.dim() == 1:
        if b.numel() == 1:
            return b.reshape(1, 1).expand(B, n)
        if b.numel() == n:
            return b.reshape(1, n).expand(B, n)
    if b.dim() == 2:
        if b.shape == (B, n):
            return b
        if b.shape == (1, n):
            return b.expand(B, n)
        if b.shape == (B, 1):
            return b.expand(B, n)
        if b.shape == (1, 1):
            return b.expand(B, n)
    raise ValueError(f"COMPARE/WHERE bcast: cannot align shape {tuple(b.shape)} -> {(B, n)}")


def tf_compare(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    op = L.params["op"]
    batch_size = Bx.lb.shape[0]
    n = len(L.out_vars)
    lb_x, ub_x = _bcast(Bx.lb, batch_size, n), _bcast(Bx.ub, batch_size, n)
    lb_y, ub_y = _bcast(By.lb, batch_size, n), _bcast(By.ub, batch_size, n)
    if op == "lt":
        defin_t, defin_f = ub_x < lb_y, lb_x >= ub_y
    elif op == "le":
        defin_t, defin_f = ub_x <= lb_y, lb_x > ub_y
    elif op == "gt":
        defin_t, defin_f = lb_x > ub_y, ub_x <= lb_y
    elif op == "ge":
        defin_t, defin_f = lb_x >= ub_y, ub_x < lb_y
    elif op == "eq":
        is_pt = (lb_x == ub_x) & (lb_y == ub_y)
        defin_t = is_pt & (lb_x == lb_y)
        defin_f = (ub_x < lb_y) | (lb_x > ub_y)
    elif op == "ne":
        is_pt = (lb_x == ub_x) & (lb_y == ub_y)
        defin_t = (ub_x < lb_y) | (lb_x > ub_y)
        defin_f = is_pt & (lb_x == lb_y)
    else:
        raise ValueError(f"COMPARE: unknown op '{op}'")
    z, o = torch.zeros_like(lb_x), torch.ones_like(lb_x)
    lb = torch.where(defin_t, o, z)
    ub = torch.where(defin_f, z, o)
    Bout = Bounds(lb, ub); C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_where(L: Layer, Bcond: Bounds, Bx: Bounds, By: Bounds) -> Fact:
    batch_size = Bcond.lb.shape[0]
    n = len(L.out_vars)
    cond_lb, cond_ub = _bcast(Bcond.lb, batch_size, n), _bcast(Bcond.ub, batch_size, n)
    lb_x, ub_x = _bcast(Bx.lb, batch_size, n), _bcast(Bx.ub, batch_size, n)
    lb_y, ub_y = _bcast(By.lb, batch_size, n), _bcast(By.ub, batch_size, n)
    cond_true = cond_lb >= 0.5
    cond_false = cond_ub < 0.5
    lb = torch.where(cond_true, lb_x, torch.where(cond_false, lb_y, torch.minimum(lb_x, lb_y)))
    ub = torch.where(cond_true, ub_x, torch.where(cond_false, ub_y, torch.maximum(ub_x, ub_y)))
    Bout = Bounds(lb, ub); C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_reduce_sum(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    axes = L.params.get("axes")
    keepdims = bool(L.params.get("keepdims", 0))
    in_shape = L.params.get("input_shape")
    lb_in, ub_in = Bin.lb, Bin.ub
    if in_shape is not None and len(in_shape) > 0:
        lb_in = lb_in.view(batch_size, *in_shape)
        ub_in = ub_in.view(batch_size, *in_shape)
    dim = tuple(int(a) + 1 for a in axes) if axes else tuple(range(1, lb_in.dim()))
    lb_out = lb_in.sum(dim=dim, keepdim=keepdims)
    ub_out = ub_in.sum(dim=dim, keepdim=keepdims)
    Bout = Bounds(lb_out.reshape(batch_size, -1), ub_out.reshape(batch_size, -1))
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"reduce_sum:{L.id}",
        "axes": list(axes) if axes is not None else None,
        "keepdims": keepdims,
        "input_shape": tuple(in_shape) if in_shape is not None else None,
    }))
    C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)

def tf_bn(L: Layer, Bin: Bounds) -> Fact:
    A,c=L.params["A"],L.params["c"]
    lb=torch.where(A>=0, A*Bin.lb+c, A*Bin.ub+c); ub=torch.where(A>=0, A*Bin.ub+c, A*Bin.lb+c)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"bn:{L.id}","A":A,"c":c}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Less-common MLP-ish --------
def tf_sigmoid(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: 1/(1+torch.exp(-x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"sigmoid:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_tanh(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.tanh(Bin.lb), torch.tanh(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"tanh:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_softplus(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: torch.log1p(torch.exp(x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"softplus:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_silu(L: Layer, Bin: Bounds) -> Fact:
    s_lb, s_ub = 1/(1+torch.exp(-Bin.lb)), 1/(1+torch.exp(-Bin.ub))
    cand=torch.stack([Bin.lb*s_lb, Bin.lb*s_ub, Bin.ub*s_lb, Bin.ub*s_ub],0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values)
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"silu:{L.id}","s_lb":s_lb,"s_ub":s_ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_max(L: Layer, By_list: List[Bounds]) -> Fact:
    lb = torch.stack([b.lb for b in By_list], dim=0).amax(dim=0)
    ub = torch.stack([b.ub for b in By_list], dim=0).amax(dim=0)
    B=Bounds(lb,ub)
    # Flatten list of y_vars_list efficiently
    all_y = list(itertools.chain.from_iterable(L.params.get("y_vars_list", [])))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"max:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_min(L: Layer, By_list: List[Bounds]) -> Fact:
    lb = torch.stack([b.lb for b in By_list], dim=0).amin(dim=0)
    ub = torch.stack([b.ub for b in By_list], dim=0).amin(dim=0)
    B=Bounds(lb,ub)
    # Flatten list of y_vars_list efficiently
    all_y = list(itertools.chain.from_iterable(L.params.get("y_vars_list", [])))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"min:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_square(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub
    lb=torch.where((l<=0)&(u>=0), 0.0, torch.minimum(l*l, u*u)); ub=torch.maximum(l*l, u*u)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"square:{L.id}","segs":pwl_meta(l,u,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_power(L: Layer, Bin: Bounds) -> Fact:
    p=float(L.params["p"]); f=lambda x: torch.pow(torch.clamp(x,min=0.0), p)
    B=Bounds(f(Bin.lb), f(Bin.ub)); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"power:{L.id}","p":p,"segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Additional Activations --------
def tf_relu6(L: Layer, Bin: Bounds) -> Fact:
    """ReLU6: clamp(x, 0, 6)"""
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=0.0, max=6.0)
    ub = torch.clamp(u, min=0.0, max=6.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"relu6:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardtanh(L: Layer, Bin: Bounds) -> Fact:
    """HardTanh: clamp(x, min_val, max_val)"""
    min_val = float(L.params.get("min_val", -1.0))
    max_val = float(L.params.get("max_val", 1.0))
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=min_val, max=max_val)
    ub = torch.clamp(u, min=min_val, max=max_val)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardtanh:{L.id}", "min_val": min_val, "max_val": max_val}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardsigmoid(L: Layer, Bin: Bounds) -> Fact:
    """HardSigmoid: clamp(alpha * x + beta, 0, 1)"""
    alpha = float(L.params.get("alpha", 1/6))
    beta = float(L.params.get("beta", 0.5))
    l, u = Bin.lb, Bin.ub
    # Apply linear transformation then clamp
    l_linear = alpha * l + beta
    u_linear = alpha * u + beta
    lb = torch.clamp(l_linear, min=0.0, max=1.0)
    ub = torch.clamp(u_linear, min=0.0, max=1.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardsigmoid:{L.id}", "alpha": alpha, "beta": beta}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardswish(L: Layer, Bin: Bounds) -> Fact:
    """HardSwish: x * clamp(x+3, 0, 6) / 6.

    Has a single global minimum at x = -1.5 with f = -0.375 (the dip
    inside [-3, 0]); endpoint-only evaluation is unsound when the input
    range crosses x = -1.5.
    """
    HS_MIN_X = -1.5
    HS_MIN_Y = -0.375
    f = lambda x: x * torch.clamp(x + 3, min=0.0, max=6.0) / 6.0
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= HS_MIN_X) & (Bin.ub >= HS_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, HS_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardswish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_mish(L: Layer, Bin: Bounds) -> Fact:
    """Mish: x * tanh(softplus(x)).

    Has a single global minimum at x ≈ -1.1924 with f ≈ -0.30884.
    Monotone-decreasing for x < min_x (towards 0 as x → -∞) and
    monotone-increasing for x > min_x.
    """
    MISH_MIN_X = -1.1924
    MISH_MIN_Y = -0.30884
    f = lambda x: x * torch.tanh(torch.nn.functional.softplus(x))
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= MISH_MIN_X) & (Bin.ub >= MISH_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, MISH_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"mish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_softsign(L: Layer, Bin: Bounds) -> Fact:
    """SoftSign: x / (1 + |x|)"""
    l, u = Bin.lb, Bin.ub
    # SoftSign is bounded between -1 and 1
    lb = l / (1 + torch.abs(l))
    ub = u / (1 + torch.abs(u))
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"softsign:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

# -------- Tensor Operations --------
def tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """Reshape: identity operation for bounds propagation"""
    # Reshape doesn't change the values, only the tensor shape
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"reshape:{L.id}", "target_shape": L.params.get("target_shape")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_transpose(L: Layer, Bin: Bounds) -> Fact:
    """Transpose: permute dimensions (identity for bounds)"""
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape")
    perm = L.params.get("perm")
    if in_shape is not None and perm is not None:
        in_shape = tuple(int(x) for x in in_shape)
        perm = tuple(int(x) for x in perm)
        lb = Bin.lb.reshape(Bin.lb.shape[0], *in_shape).permute(
            0, *(p + 1 for p in perm)
        ).reshape(Bin.lb.shape[0], -1)
        ub = Bin.ub.reshape(Bin.ub.shape[0], *in_shape).permute(
            0, *(p + 1 for p in perm)
        ).reshape(Bin.ub.shape[0], -1)
        B = Bounds(lb, ub)
    else:
        # Legacy fallback for older converter paths that did not record shape.
        B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"transpose:{L.id}",
        "perm": L.params.get("perm"),
        "input_shape": L.params.get("input_shape"),
        "output_shape": L.params.get("output_shape"),
    }))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_squeeze(L: Layer, Bin: Bounds) -> Fact:
    """Squeeze: remove singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"squeeze:{L.id}", "dims": L.params.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_unsqueeze(L: Layer, Bin: Bounds) -> Fact:
    """Unsqueeze: add singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"unsqueeze:{L.id}", "dims": L.params.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_expand(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape") or L.params.get("shape")
    n_out = len(L.out_vars)
    if in_shape is not None and out_shape is not None:
        in_shape = tuple(int(d) for d in in_shape)
        out_shape = tuple(int(d) for d in out_shape)
        lb = Bin.lb.view(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
        ub = Bin.ub.view(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
    elif Bin.lb.shape[1] == n_out:
        lb, ub = Bin.lb.clone(), Bin.ub.clone()
    else:
        width = Bin.lb.shape[1]
        repeat = (n_out + width - 1) // width
        lb = Bin.lb.repeat(1, repeat)[:, :n_out].clone()
        ub = Bin.ub.repeat(1, repeat)[:, :n_out].clone()
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)
    
def tf_slice(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    inp_shape = tuple(L.params["input_shape"])  # e.g. (1, 3, 32, 32)
    # ROUND 9 (advisor 2026-05-25): ACT's sequential analyzer feeds
    # ``Bin`` from the topologically-preceding layer's output, but
    # ONNX Slice can reference the original graph input by name
    # (e.g. linearizenn AllInOne_10_10 slices ``input`` while the
    # preceding layer is the controller MatMul of a different dim).
    # When the actual feed size mismatches input_shape, we degrade
    # gracefully to ±inf bounds so analyze() completes and
    # ``_try_small_dense_lp`` can still take the small-dense verdict
    # without the interval-pass crashing.
    expected_elems = int(__import__("torch").Size(inp_shape).numel())
    if Bin.lb.numel() // batch_size != expected_elems:
        import torch as _torch
        n_out = len(L.out_vars)
        sentinel_lb = _torch.full(
            (batch_size, n_out), float("-inf"),
            dtype=Bin.lb.dtype, device=Bin.lb.device,
        )
        sentinel_ub = _torch.full(
            (batch_size, n_out), float("inf"),
            dtype=Bin.ub.dtype, device=Bin.ub.device,
        )
        Bout = Bounds(sentinel_lb, sentinel_ub)
        C = ConSet()
        C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
            "tag": f"slice:{L.id}",
            "starts": L.params.get("starts", []),
            "ends":   L.params.get("ends", []),
            "axes":   L.params.get("axes", []),
            "steps":  L.params.get("steps", []),
            "input_shape": inp_shape,
            "ROUND9_shape_mismatch_sentinel": True,
        }))
        C.add_box(L.id, L.out_vars, Bout)
        return Fact(Bout, C)
    x_lb = Bin.lb.view(batch_size, *inp_shape)
    x_ub = Bin.ub.view(batch_size, *inp_shape)

    starts = L.params.get("starts", [])
    ends   = L.params.get("ends", [])
    axes   = L.params.get("axes", list(range(len(inp_shape))))
    steps  = L.params.get("steps", [1] * len(axes))

    # Build slice objects for each dimension
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        s = starts[i]
        e = ends[i]
        st = steps[i]
        if e > inp_shape[axis]:
            e = inp_shape[axis]
        slices[axis + 1] = slice(s, e, st)

    out_lb = x_lb[tuple(slices)]
    out_ub = x_ub[tuple(slices)]
    assert out_lb.shape[0] == batch_size, f"slice batch mismatch {out_lb.shape[0]} != {batch_size}"
    assert out_lb[0].numel() == len(L.out_vars), f"slice out_vars length {len(L.out_vars)} != output elements {out_lb[0].numel()}"
    assert torch.all(out_lb <= out_ub), "slice produced invalid bounds (lb > ub)"

    Bout = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))

    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"slice:{L.id}",
        "starts": starts,
        "ends": ends,
        "axes": axes,
        "steps": steps,
        "input_shape": inp_shape,
    }))
    C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_gather(L: Layer, Bin: Bounds) -> Fact:

    batch_size = Bin.lb.shape[0]
    inp_shape = tuple(L.params["input_shape"])
    axis = int(L.params.get("axis", 0))
    # ONNX Gather allows negative axis (axis-from-end). The torch2act
    # handler stores the raw axis; normalize here for torch.index_select.
    norm_axis = axis if axis >= 0 else axis + len(inp_shape)
    x_lb = Bin.lb.view(batch_size, *inp_shape)
    x_ub = Bin.ub.view(batch_size, *inp_shape)

    raw_idx = L.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    else:
        indices = raw_idx.to(x_lb.device).long()

    # ONNX Gather semantics:
    #   * indices may be 0-d (scalar) → torch.index_select needs 1-d, so
    #     reshape and squeeze the gathered axis afterwards
    #   * indices may be NEGATIVE (axis-from-end) → torch.index_select
    #     rejects, so wrap to non-negative against the gathered axis size
    # nn4sys pensieve hits both: scalar -1 to take the last element.
    scalar_index = (indices.dim() == 0)
    if scalar_index:
        indices = indices.reshape(1)
    axis_dim = int(inp_shape[norm_axis])
    if (indices < 0).any():
        indices = torch.where(indices < 0, indices + axis_dim, indices)
    if (indices < 0).any() or (indices >= axis_dim).any():
        raise ValueError(
            f"GATHER layer id={L.id}: index out of range after wrap "
            f"(axis={axis}, axis_dim={axis_dim}, indices={indices.tolist()[:10]})"
        )

    out_lb = torch.index_select(x_lb, dim=norm_axis + 1, index=indices)
    out_ub = torch.index_select(x_ub, dim=norm_axis + 1, index=indices)

    if scalar_index:
        # Drop the size-1 gathered axis so the output rank matches ONNX semantics.
        out_lb = out_lb.squeeze(norm_axis + 1)
        out_ub = out_ub.squeeze(norm_axis + 1)

    Bout = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))

    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"gather:{L.id}",
        "axis": axis,
        "indices": indices.detach().cpu().tolist(),
        "input_shape": inp_shape,
        "output_shape": list(out_lb.shape),
    }))
    C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)
