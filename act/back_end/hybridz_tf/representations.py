#===- act/back_end/hybridz_tf/representations.py - HZ Phase-1-3 Representations -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Sound HZ flavour surrogates that bound ng on large networks where
#   materialising a dense HZono exceeds GPU memory: BoxHZ (IBP),
#   LazyChainHZ (deferred linear chain), SparseGcZ (sparse Gc storage).
#
#===---------------------------------------------------------------------===#

"""Phase-1-3 HZ flavour representations: BoxHZ, LazyChainHZ, SparseGcZ.

These are sound HZ surrogates that the forward TF pipeline uses to keep
``ng`` bounded on big networks where materialising a full dense
``HZono`` (with ``Gc: (n, ng)`` for ``n = 65536`` and ``ng = n_root``)
exceeds GPU memory. Each surrogate offers a subset of the HZono facade
sufficient for the ACT batch-native TF chain to make progress:

  * **BoxHZ**  -- sound IBP wrapper over ``(lb, ub)``. ``ng = dim``
    conceptually (one per-pixel diagonal generator). Promotes to full
    ``HZono`` via ``to_hzono()`` when ``dim`` fits the dense cap.

  * **LazyChainHZ** -- linear-op chain rooted at a BoxHZ. Defers Gc
    materialisation across a sequence of affine ops (conv / dense /
    scale / add-const). At a non-linear op (ReLU / Sigmoid / Tanh) the
    chain ``freeze()``s into either a full HZono, a SparseGcZ, or
    falls back to a BoxHZ snapshot.

  * **SparseGcZ** -- HZ with sparse-COO Gc and optional sparse linear
    constraint rows; nb = 0. Useful for a single-conv chain where Gc
    has well-defined sparsity but dense storage doesn't fit. Supports
    triangle ReLU and sparse selective convex-hull facet cuts; encodings
    requiring binary slots still promote to dense HZono.

This module is a direct port of the corresponding HyZor classes in
``HyZor/__init__.py`` (which return ``HybridZonotope``); here we return
``HZono`` and otherwise preserve the same dispatch behaviour.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple

import torch

from act.back_end.solver.solver_hz import HZono, hz_from_bounds
from act.back_end.core import Bounds


# ─── Env knobs (parity with HyZor/__init__.py) ─────────────────────────


def _hz_cap() -> int:
    """Dim above which `_make_or_box` returns BoxHZ instead of dense HZono."""
    return int(os.environ.get("HYZOR_LARGE_HZ_DIM_CAP", "8192"))


def _materialize_budget_bytes() -> int:
    """LazyChainHZ -> full HZono materialisation memory budget (default 4 GB)."""
    return int(float(os.environ.get("HYZOR_LAZY_MAT_BUDGET_GB", "4.0")) * (1024 ** 3))


def _materialize_dim_cap() -> int:
    """LazyChainHZ output-dim cap above which materialisation is skipped."""
    return int(os.environ.get("HYZOR_LAZY_MAT_DIM_CAP", "512"))


def _enable_lazy_chain() -> bool:
    return os.environ.get("HYZOR_LAZY_CHAIN", "1") == "1"


def _enable_sparse_gc() -> bool:
    return os.environ.get("HYZOR_SPARSE_GC", "1") == "1"


def _relu_selection_score(lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Forward-local score for selecting sparse ReLU hull facets.

    ``width`` is the legacy behavior. ``mu`` and ``area`` target the
    DeepZ/triangle slack directly: for unstable ReLU bounds ``l < 0 < u``,
    the added slack amplitude is ``-l*u/(2*(u-l))`` and the hull gap area is
    proportional to ``-l*u``. These scores use only current forward bounds,
    not gradients, backward bounds, splitting, or sampling.
    """
    mode = (
        os.environ.get("ACT_HZ_EARLY_SELECTIVE_SCORE")
        or os.environ.get("ACT_HZ_SELECTIVE_SCORE")
        or "width"
    ).strip().lower()
    width = ub - lb
    if mode in ("mu", "slack", "height"):
        return (-lb * ub) / torch.clamp(2.0 * width, min=1e-30)
    if mode in ("area", "product", "gap"):
        return -lb * ub
    if mode in ("balanced", "minside", "min_side"):
        return torch.minimum(-lb, ub)
    return width


def _sparse_gc_budget_bytes() -> int:
    return int(float(os.environ.get("HYZOR_SPARSE_GC_BUDGET_GB", "8.0")) * (1024 ** 3))


# ─── Sparse Conv helpers (Y2 Stage 2 port of HyZor's __init__.py) ─────


def _ceil_div(a: int, b: int) -> int:
    return -((-int(a)) // int(b))


def _conv2d_sparse_nnz(weight, in_shape, stride, pad) -> int:
    """Exact nnz count for the sparse Conv2d linear operator.

    Used by ``SparseGcZ.apply_conv`` to decide whether to take the
    sparse-mm fast path (cheaper) or fall back to per-column chunked
    densify (more memory tolerant). Faithful port of HyZor
    ``_conv2d_sparse_nnz`` (__init__.py:774).
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    Cin, Hin, Win = map(int, in_shape)
    Cout, Cin_w, kH, kW = map(int, weight.shape)
    assert Cin == Cin_w, f"in_shape Cin={Cin} != weight Cin={Cin_w}"
    Hout = (Hin + 2 * int(pad) - kH) // int(stride[0]) + 1
    Wout = (Win + 2 * int(pad) - kW) // int(stride[1]) + 1

    h_counts = []
    for kh in range(kH):
        lo = _ceil_div(int(pad) - kh, int(stride[0]))
        hi = (Hin + int(pad) - kh - 1) // int(stride[0])
        lo = max(0, lo)
        hi = min(Hout - 1, hi)
        h_counts.append(max(0, hi - lo + 1))

    w_counts = []
    for kw in range(kW):
        lo = _ceil_div(int(pad) - kw, int(stride[1]))
        hi = (Win + int(pad) - kw - 1) // int(stride[1])
        lo = max(0, lo)
        hi = min(Wout - 1, hi)
        w_counts.append(max(0, hi - lo + 1))

    valid_spatial = sum(hc * wc for hc in h_counts for wc in w_counts)
    return int(Cout) * int(Cin) * int(valid_spatial)


def _sparsify_with_row_slack(Gc_sparse, threshold: float, *, dtype, device):
    """Drop small sparse entries soundly via per-row diagonal slack.

    Faithful port of HyZor ``_sparsify_with_row_slack`` (__init__.py:804).
    """
    threshold = float(threshold or 0.0)
    Gc_sparse = Gc_sparse.coalesce()
    if threshold <= 0.0 or Gc_sparse._nnz() == 0:
        return Gc_sparse

    n, ng = map(int, Gc_sparse.shape)
    ind = Gc_sparse.indices()
    val = Gc_sparse.values()
    keep = val.abs() > threshold
    if bool(keep.all()):
        return Gc_sparse

    kept_ind = ind[:, keep]
    kept_val = val[keep]

    dropped = ~keep
    slack = torch.zeros(n, dtype=dtype, device=device)
    slack.scatter_add_(0, ind[0, dropped], val[dropped].abs())
    slack_rows = torch.nonzero(slack > 0, as_tuple=False).view(-1)
    if slack_rows.numel() > 0:
        slack_cols = ng + torch.arange(
            int(slack_rows.numel()), dtype=torch.long, device=device,
        )
        slack_ind = torch.stack([slack_rows, slack_cols])
        all_ind = torch.cat([kept_ind, slack_ind], dim=1)
        all_val = torch.cat([kept_val, slack[slack_rows]])
    else:
        all_ind = kept_ind
        all_val = kept_val
    return torch.sparse_coo_tensor(
        all_ind, all_val,
        (n, ng + int(slack_rows.numel())),
        dtype=dtype, device=device,
    ).coalesce()


def build_sparse_conv_matrix(weight, in_shape, stride, pad,
                              *, dtype=None, device=None):
    """Build the sparse matrix for a Conv2d linear operator.

    For each output pixel ``(cout, hout, wout)`` and each kernel offset
    ``(kh, kw)`` with valid ``(hin, win)``, contributes one sparse entry.
    Returns ``torch.sparse_coo_tensor`` of shape
    ``(Cout*Hout*Wout, Cin*Hin*Win)``. Faithful port of HyZor
    ``build_sparse_conv_matrix`` (__init__.py:839).
    """
    if dtype is None:
        dtype = weight.dtype
    if device is None:
        device = weight.device

    if isinstance(stride, int):
        stride = (stride, stride)
    Cout, Cin_w, kH, kW = weight.shape
    Cin, Hin, Win = in_shape
    assert Cin == Cin_w, f"in_shape Cin={Cin} != weight Cin={Cin_w}"
    Hout = (Hin + 2 * pad - kH) // stride[0] + 1
    Wout = (Win + 2 * pad - kW) // stride[1] + 1
    n_out = Cout * Hout * Wout
    n_in = Cin * Hin * Win

    hout = torch.arange(Hout, device=device).view(Hout, 1, 1, 1)
    wout = torch.arange(Wout, device=device).view(1, Wout, 1, 1)
    kh = torch.arange(kH, device=device).view(1, 1, kH, 1)
    kw = torch.arange(kW, device=device).view(1, 1, 1, kW)
    hin = hout * stride[0] + kh - pad
    win = wout * stride[1] + kw - pad
    hin_b = hin.expand(Hout, Wout, kH, kW)
    win_b = win.expand(Hout, Wout, kH, kW)
    valid = (hin_b >= 0) & (hin_b < Hin) & (win_b >= 0) & (win_b < Win)

    valid_idx = torch.nonzero(valid, as_tuple=False)
    if valid_idx.numel() == 0:
        ind = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros((0,), dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            ind, val, (n_out, n_in),
            dtype=dtype, device=device,
        ).coalesce()

    M = int(valid_idx.shape[0])
    cout_idx = torch.arange(Cout, device=device).view(Cout, 1, 1).expand(Cout, Cin, M)
    cin_idx = torch.arange(Cin, device=device).view(1, Cin, 1).expand(Cout, Cin, M)
    spatial_idx = valid_idx.t().view(4, 1, 1, M).expand(4, Cout, Cin, M)
    hout_e, wout_e, kh_e, kw_e = spatial_idx[0], spatial_idx[1], spatial_idx[2], spatial_idx[3]
    hin_e = hout_e * stride[0] + kh_e - pad
    win_e = wout_e * stride[1] + kw_e - pad

    row_idx = (cout_idx * Hout * Wout + hout_e * Wout + wout_e).reshape(-1)
    col_idx = (cin_idx * Hin * Win + hin_e * Win + win_e).reshape(-1)
    vals = weight[cout_idx, cin_idx, kh_e, kw_e].reshape(-1).to(dtype=dtype)

    ind = torch.stack([row_idx, col_idx])
    return torch.sparse_coo_tensor(
        ind, vals, (n_out, n_in),
        dtype=dtype, device=device,
    ).coalesce()


# ─── BoxHZ ─────────────────────────────────────────────────────────────


class BoxHZ:
    """Sound IBP wrapper: ``Z = {y : lb ≤ y ≤ ub}``.

    Conceptually a per-pixel diagonal HZ with ``ng = dim`` continuous
    generators and no binary or constraint structure. Used as a
    fallback whenever the dense HZono representation would exceed
    memory.
    """

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor, *, dtype, device):
        self.lb = lb.to(dtype=dtype, device=device).flatten()
        self.ub = ub.to(dtype=dtype, device=device).flatten()
        self.dtype = dtype
        self.device = device

    @property
    def dim(self) -> int:
        return int(self.lb.numel())

    @property
    def n(self) -> int:
        return self.dim

    @property
    def ng(self) -> int:
        rad = ((self.ub - self.lb) / 2.0).clamp_min(0)
        return int((rad > 0).sum().item())

    @property
    def nb(self) -> int:
        return 0

    @property
    def nc(self) -> int:
        return 0

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lb, self.ub

    def _bounds_unconstrained(self):
        """HyZor-API alias: ``(lb, ub)`` as ``(n,1)`` tensors."""
        return self.lb.view(-1, 1), self.ub.view(-1, 1)

    def to_hzono(self) -> HZono:
        """Materialise as a dense diagonal HZono. Caller ensures dim is
        small enough to fit (use ``_make_or_box`` for budgeted picks)."""
        return hz_from_bounds(
            Bounds(lb=self.lb, ub=self.ub),
            dtype=self.dtype, device=self.device,
        )

    # HyZor-API alias: HyZor's BoxHZ.to_hz returns HybridZonotope; we
    # return HZono with same 6-tuple. Naming-only difference.
    to_hz = to_hzono


def _make_or_box(lb: torch.Tensor, ub: torch.Tensor, *, dtype, device):
    """Build a BoxHZ if ``dim > _hz_cap()`` else a full diagonal HZono."""
    n = int(lb.numel())
    if n > _hz_cap():
        return BoxHZ(lb, ub, dtype=dtype, device=device)
    box = BoxHZ(lb, ub, dtype=dtype, device=device)
    return box.to_hzono()


def _bounds_of(z) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (lb, ub) flat tensors for BoxHZ / LazyChainHZ / SparseGcZ / HZono."""
    if isinstance(z, BoxHZ):
        return z.lb, z.ub
    if isinstance(z, LazyChainHZ):
        return z.bounds()
    if isinstance(z, SparseGcZ):
        return z.bounds()
    # HZono: |Gc|+|Gb| over the diagonal.
    if z.Gc.numel() > 0:
        absGc = z.Gc.abs().sum(dim=1)
    else:
        absGc = torch.zeros(z.c.shape[0], dtype=z.c.dtype, device=z.c.device)
    if z.Gb.numel() > 0:
        absGb = z.Gb.abs().sum(dim=1)
    else:
        absGb = torch.zeros_like(absGc)
    rad = absGc + absGb
    c_flat = z.c.reshape(-1)
    return (c_flat - rad).flatten(), (c_flat + rad).flatten()


def _to_box(z) -> "BoxHZ":
    """Coerce any HZ flavour to BoxHZ via interval bounds."""
    if isinstance(z, BoxHZ):
        return z
    lb, ub = _bounds_of(z)
    dtype = getattr(z, "dtype", None) or z.c.dtype
    device = getattr(z, "device", None) or z.c.device
    return BoxHZ(lb, ub, dtype=dtype, device=device)


# ─── LazyChainHZ ───────────────────────────────────────────────────────


class LazyChainHZ:
    """Linear-op chain rooted at a BoxHZ.

    Represents ``y = c_chain + L_chain(rad_root ⊙ ξ)``, ``ξ ∈ [-1, 1]^n_root``,
    where ``L_chain`` is a composition of affine ops applied via the
    underlying torch primitives. Materialisation to a full HZono is
    deferred until a non-linear op is encountered.
    """

    def __init__(self, *, root_lb: torch.Tensor, root_ub: torch.Tensor,
                 ops: List[Dict[str, Any]], c_chain: torch.Tensor,
                 dim: int, dtype, device):
        self.root_lb = root_lb.flatten().to(dtype=dtype, device=device)
        self.root_ub = root_ub.flatten().to(dtype=dtype, device=device)
        self.ops = ops
        self.c_chain = c_chain.flatten().to(dtype=dtype, device=device)
        self.dim = int(dim)
        self.dtype = dtype
        self.device = device

    @property
    def n(self) -> int:
        return self.dim

    @property
    def n_root(self) -> int:
        return int(self.root_lb.numel())

    @property
    def n_active_root(self) -> int:
        rad = ((self.root_ub - self.root_lb) / 2.0).clamp_min(0)
        return int((rad > 0).sum().item())

    @property
    def ng(self) -> int:
        return self.n_active_root

    @property
    def nb(self) -> int:
        return 0

    @property
    def nc(self) -> int:
        return 0

    @classmethod
    def from_box(cls, box: "BoxHZ") -> "LazyChainHZ":
        c_chain = ((box.lb + box.ub) / 2.0)
        return cls(root_lb=box.lb, root_ub=box.ub, ops=[],
                   c_chain=c_chain, dim=box.dim,
                   dtype=box.dtype, device=box.device)

    def _push_through_ops(self, x_in: torch.Tensor) -> torch.Tensor:
        """Apply the op chain to a single vector x_in. (Used by
        materialisation; for batched form see ``_push_batch_through_ops``.)"""
        x = x_in
        for op in self.ops:
            kind = op["kind"]
            if kind == "conv":
                C, H, W = op["in_shape"]
                x4 = x.view(1, C, H, W)
                y4 = torch.nn.functional.conv2d(
                    x4, op["weight"], None,
                    stride=op["stride"], padding=op["pad"],
                )
                x = y4.flatten()
            elif kind == "dense":
                x = op["W"] @ x.flatten()
            elif kind == "scale":
                x = x.flatten() * op["a"].flatten()
            else:
                raise ValueError(f"Unknown op kind: {kind}")
        return x.flatten()

    def _radius_through_ops(self) -> torch.Tensor:
        """IBP-style: rad_out_i = sum_j |Op_ij| rad_root_j."""
        x_rad = ((self.root_ub - self.root_lb) / 2.0).abs()
        for op in self.ops:
            kind = op["kind"]
            if kind == "conv":
                C, H, W = op["in_shape"]
                xr4 = x_rad.view(1, C, H, W)
                x_rad = torch.nn.functional.conv2d(
                    xr4, op["weight"].abs(), None,
                    stride=op["stride"], padding=op["pad"],
                ).flatten()
            elif kind == "dense":
                x_rad = op["W"].abs() @ x_rad
            elif kind == "scale":
                x_rad = op["a"].flatten().abs() * x_rad
            else:
                raise ValueError(f"Unknown op kind: {kind}")
        return x_rad

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rad_out = self._radius_through_ops()
        return self.c_chain - rad_out, self.c_chain + rad_out

    def materialization_bytes(self) -> int:
        return int(self.dim) * int(self.n_active_root) * torch.empty(
            (), dtype=self.dtype, device=self.device
        ).element_size()

    def can_materialize(self) -> bool:
        if self.dim > _materialize_dim_cap():
            return False
        return self.materialization_bytes() <= _materialize_budget_bytes()

    def _push_batch_through_ops(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, n_root) -> (B, dim) batched application of the chain."""
        x = batch
        for op in self.ops:
            kind = op["kind"]
            if kind == "conv":
                C, H, W = op["in_shape"]
                B = x.shape[0]
                x4 = x.view(B, C, H, W)
                y4 = torch.nn.functional.conv2d(
                    x4, op["weight"], None,
                    stride=op["stride"], padding=op["pad"],
                )
                x = y4.view(B, -1)
            elif kind == "dense":
                x = x @ op["W"].T
            elif kind == "scale":
                x = x * op["a"].flatten().unsqueeze(0)
            else:
                raise ValueError(f"Unknown op kind: {kind}")
        return x

    def to_full_hzono(self) -> HZono:
        """Materialise Gc by pushing diag(rad_root) through the chain."""
        rad = ((self.root_ub - self.root_lb) / 2.0).clamp_min(0)
        active_idx = torch.nonzero(rad > 0, as_tuple=False).view(-1)
        n_root = int(rad.numel())
        n_active = int(active_idx.numel())
        Gc_cols = []
        if n_active > 0:
            chunk = max(1, min(n_active, 2048))
        for i0 in range(0, n_active, chunk if n_active > 0 else 1):
            i1 = min(i0 + chunk, n_active)
            cols = active_idx[i0:i1]
            batch = torch.zeros((i1 - i0, n_root),
                                dtype=self.dtype, device=self.device)
            rows = torch.arange(i1 - i0, dtype=torch.long, device=self.device)
            batch[rows, cols] = rad[cols]
            cols_out = self._push_batch_through_ops(batch)
            Gc_cols.append(cols_out)
        if Gc_cols:
            Gc = torch.cat(Gc_cols, dim=0).T.contiguous()  # (dim, n_active)
        else:
            Gc = torch.zeros((self.dim, 0), dtype=self.dtype, device=self.device)
        n = self.dim
        c = self.c_chain.view(-1, 1)
        Gb = torch.zeros((n, 0), dtype=self.dtype, device=self.device)
        Ac = torch.zeros((0, n_active), dtype=self.dtype, device=self.device)
        Ab = torch.zeros((0, 0), dtype=self.dtype, device=self.device)
        b = torch.zeros((0, 1), dtype=self.dtype, device=self.device)
        return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)

    def snapshot_to_box(self) -> "BoxHZ":
        lb, ub = self.bounds()
        return BoxHZ(lb, ub, dtype=self.dtype, device=self.device)

    def with_conv(self, weight, bias, stride, pad,
                  in_shape, out_shape, out_dim) -> "LazyChainHZ":
        new_ops = self.ops + [{"kind": "conv", "weight": weight,
                                "stride": stride, "pad": pad,
                                "in_shape": in_shape}]
        c4 = self.c_chain.view(1, *in_shape)
        c_new4 = torch.nn.functional.conv2d(
            c4, weight,
            bias if bias is not None and bias.numel() > 0 else None,
            stride=stride, padding=pad,
        )
        c_new = c_new4.flatten()
        return LazyChainHZ(root_lb=self.root_lb, root_ub=self.root_ub,
                           ops=new_ops, c_chain=c_new,
                           dim=int(c_new.numel()),
                           dtype=self.dtype, device=self.device)

    def with_dense(self, W: torch.Tensor, b: Optional[torch.Tensor]) -> "LazyChainHZ":
        new_ops = self.ops + [{"kind": "dense", "W": W}]
        c_new = W @ self.c_chain
        if b is not None:
            c_new = c_new + b.flatten()
        return LazyChainHZ(root_lb=self.root_lb, root_ub=self.root_ub,
                           ops=new_ops, c_chain=c_new,
                           dim=int(c_new.numel()),
                           dtype=self.dtype, device=self.device)

    def with_scale(self, a: torch.Tensor) -> "LazyChainHZ":
        new_ops = self.ops + [{"kind": "scale", "a": a}]
        c_new = self.c_chain * a.flatten()
        return LazyChainHZ(root_lb=self.root_lb, root_ub=self.root_ub,
                           ops=new_ops, c_chain=c_new,
                           dim=int(c_new.numel()),
                           dtype=self.dtype, device=self.device)

    def with_add_const(self, c_add: torch.Tensor) -> "LazyChainHZ":
        c_new = self.c_chain + c_add.flatten()
        return LazyChainHZ(root_lb=self.root_lb, root_ub=self.root_ub,
                           ops=list(self.ops), c_chain=c_new,
                           dim=int(c_new.numel()),
                           dtype=self.dtype, device=self.device)

    def freeze(self):
        """Pick a sound representation by descending preference:
            1. Full dense HZono (correlated, ReLU-eq-encoding ready)
            2. SparseGcZ (sparse-Gc, triangle-only ReLU)
            3. BoxHZ snapshot (IBP, lossy)
        """
        if self.can_materialize():
            return self.to_full_hzono()
        sparse = self.to_sparse_gc_z()
        if sparse is not None:
            return sparse
        return self.snapshot_to_box()

    def to_sparse_gc_z(self) -> "Optional[SparseGcZ]":
        """If the chain is exactly one conv and the sparse Gc fits the
        budget, materialise as a SparseGcZ; else return None.

        Faithful port of HyZor ``LazyChainHZ.to_sparse_gc_hz``
        (__init__.py:306). Builds sparse Gc by pushing ``diag(rad_root)``
        through the single conv op using ``build_sparse_conv_matrix`` +
        per-column scaling. Preserves input-pixel correlation through
        the conv, vs the BoxHZ snapshot fallback which collapses it.
        """
        if not _enable_sparse_gc():
            return None
        if len(self.ops) != 1 or self.ops[0]["kind"] != "conv":
            return None
        op = self.ops[0]
        rad = ((self.root_ub - self.root_lb) / 2.0).clamp_min(0)
        active_idx = torch.nonzero(rad > 0, as_tuple=False).view(-1)
        n_active = int(active_idx.numel())
        if n_active == 0:
            empty = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=self.dtype, device=self.device),
                (self.dim, 0), dtype=self.dtype, device=self.device,
            ).coalesce()
            return SparseGcZ(c=self.c_chain, Gc_sparse=empty,
                             dtype=self.dtype, device=self.device)

        # For VNNLIBs that perturb only a handful of input pixels, building
        # the full sparse convolution matrix is wasteful: it creates columns
        # for every zero-radius input. Push only the active root basis columns
        # through the lazy chain, then sparsify the compact result.
        elem = torch.empty((), dtype=self.dtype, device=self.device).element_size()
        dense_bytes = n_active * (self.n_root + self.dim) * elem
        direct_active_cap = int(os.environ.get("ACT_HZ_DIRECT_ACTIVE_MAX", "2048"))
        if n_active <= direct_active_cap and dense_bytes <= _sparse_gc_budget_bytes():
            all_ind = []
            all_val = []
            max_chunk = max(1, int(_sparse_gc_budget_bytes() // max((self.n_root + self.dim) * elem, 1)))
            chunk = max(1, min(n_active, direct_active_cap, max_chunk))
            for i0 in range(0, n_active, chunk):
                i1 = min(i0 + chunk, n_active)
                cols = active_idx[i0:i1]
                batch = torch.zeros((i1 - i0, self.n_root),
                                    dtype=self.dtype, device=self.device)
                rows = torch.arange(i1 - i0, dtype=torch.long, device=self.device)
                batch[rows, cols] = rad[cols]
                out = self._push_batch_through_ops(batch)
                nz = torch.nonzero(out != 0, as_tuple=False)
                if nz.numel() == 0:
                    continue
                # out is (local_active_col, output_row); sparse Gc wants
                # (output_row, compact_active_col).
                ind = torch.stack([nz[:, 1], nz[:, 0] + i0], dim=0)
                all_ind.append(ind)
                all_val.append(out[nz[:, 0], nz[:, 1]])
            if all_ind:
                ind = torch.cat(all_ind, dim=1)
                vals = torch.cat(all_val, dim=0)
            else:
                ind = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                vals = torch.zeros(0, dtype=self.dtype, device=self.device)
            Gc_sparse = torch.sparse_coo_tensor(
                ind, vals, (self.dim, n_active),
                dtype=self.dtype, device=self.device,
            ).coalesce()
            if Gc_sparse._nnz() * (elem + 8) <= _sparse_gc_budget_bytes():
                return SparseGcZ(
                    c=self.c_chain, Gc_sparse=Gc_sparse,
                    dtype=self.dtype, device=self.device,
                )

        # Pre-flight memory check via exact nnz count.
        try:
            est_nnz = _conv2d_sparse_nnz(
                op["weight"], op["in_shape"], op["stride"], op["pad"],
            )
        except Exception:
            return None
        if est_nnz * 16 > _sparse_gc_budget_bytes():
            return None
        try:
            W_sparse = build_sparse_conv_matrix(
                op["weight"], op["in_shape"], op["stride"], op["pad"],
                dtype=self.dtype, device=self.device,
            )
        except Exception:
            return None
        if W_sparse._nnz() == 0:
            return None
        rad_flat = rad.flatten().to(dtype=self.dtype, device=self.device)
        ind = W_sparse.indices()
        active_map = torch.full((self.n_root,), -1, dtype=torch.long, device=self.device)
        active_map[active_idx] = torch.arange(n_active, dtype=torch.long, device=self.device)
        keep = active_map[ind[1]] >= 0
        if not bool(keep.any().item()):
            return None
        # Scale each kept conv-matrix entry by the radius of its input column,
        # and remap original input columns to compact active columns.
        kept_in = ind[1, keep]
        vals = W_sparse.values()[keep] * rad_flat[kept_in]
        nz = vals != 0
        if not bool(nz.any().item()):
            return None
        vals = vals[nz]
        row_idx = ind[0, keep][nz]
        col_idx = active_map[kept_in[nz]]
        ind = torch.stack([row_idx, col_idx], dim=0)
        Gc_sparse = torch.sparse_coo_tensor(
            ind, vals, (W_sparse.shape[0], n_active),
            dtype=self.dtype, device=self.device,
        ).coalesce()
        if Gc_sparse._nnz() * (elem + 8) > _sparse_gc_budget_bytes():
            return None
        return SparseGcZ(
            c=self.c_chain, Gc_sparse=Gc_sparse,
            dtype=self.dtype, device=self.device,
        )

    # HyZor-API aliases (Y2 Stage 3a):
    #   HyZor: LazyChainHZ.to_full_hz / to_sparse_gc_hz
    #   ACT:   to_full_hzono / to_sparse_gc_z
    # Provide the HyZor names too so dispatch code can call either.
    to_full_hz = to_full_hzono
    to_sparse_gc_hz = to_sparse_gc_z

    def _bounds_unconstrained(self):
        """HyZor-API alias: ``(lb, ub)`` as ``(n,1)`` tensors."""
        lb, ub = self.bounds()
        return lb.view(-1, 1), ub.view(-1, 1)


# ─── SparseGcZ ─────────────────────────────────────────────────────────


class SparseGcZ:
    """HZ with sparse-COO storage and optional binary generators.

    Until 2026-05-28 this stored only ``Gc_sparse`` with ``nb = 0``.
    Extended with optional ``Gb_sparse`` (binary generators) and
    ``Ab_sparse`` (binary-side constraint coefficients) so that the
    sparse-eq_lagr ReLU encoding (introduced for B3, see
    ``act/back_end/hybridz_tf/algorithms/sparse_eq_lagr.py``) can run
    on the sparse path without densifying. ``Gb_sparse = None``
    preserves the original nb=0 behaviour and all existing callers
    (T2/T2b sparse conv chain + sparse triangle ReLU) work unchanged.
    """

    def __init__(self, *, c: torch.Tensor, Gc_sparse: torch.Tensor,
                 dtype, device, Ac_sparse: Optional[torch.Tensor] = None,
                 Gb_sparse: Optional[torch.Tensor] = None,
                 Ab_sparse: Optional[torch.Tensor] = None,
                 b: Optional[torch.Tensor] = None,
                 eq_mask: Optional[torch.Tensor] = None):
        self.c = c.flatten().to(dtype=dtype, device=device)
        self.Gc_sparse = Gc_sparse.coalesce()
        self.dtype = dtype
        self.device = device
        n = int(self.c.numel())
        ng = int(self.Gc_sparse.shape[1])

        # Binary generators (optional).
        if Gb_sparse is None:
            Gb_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=dtype, device=device),
                (n, 0), dtype=dtype, device=device,
            )
        self.Gb_sparse = Gb_sparse.coalesce().to(dtype=dtype, device=device)
        nb = int(self.Gb_sparse.shape[1])

        if Ac_sparse is None:
            Ac_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=dtype, device=device),
                (0, ng), dtype=dtype, device=device,
            )
        self.Ac_sparse = Ac_sparse.coalesce().to(dtype=dtype, device=device)
        nc = int(self.Ac_sparse.shape[0])

        if Ab_sparse is None:
            Ab_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=dtype, device=device),
                (nc, nb), dtype=dtype, device=device,
            )
        self.Ab_sparse = Ab_sparse.coalesce().to(dtype=dtype, device=device)
        if int(self.Ab_sparse.shape[0]) != nc:
            raise ValueError(
                f"SparseGcZ: Ab_sparse row count {self.Ab_sparse.shape[0]} != Ac_sparse row count {nc}"
            )
        if int(self.Ab_sparse.shape[1]) != nb:
            raise ValueError(
                f"SparseGcZ: Ab_sparse col count {self.Ab_sparse.shape[1]} != Gb_sparse col count {nb}"
            )

        self.b = (
            torch.zeros((nc, 1), dtype=dtype, device=device)
            if b is None else b.to(dtype=dtype, device=device).view(-1, 1)
        )
        if int(self.b.shape[0]) != nc:
            raise ValueError("SparseGcZ: Ac_sparse/b row mismatch")
        self.eq_mask = (
            torch.zeros(nc, dtype=torch.bool, device=device)
            if eq_mask is None
            else eq_mask.to(dtype=torch.bool, device=device).view(-1)
        )
        if int(self.eq_mask.numel()) != nc:
            raise ValueError("SparseGcZ: Ac_sparse/eq_mask row mismatch")
        object.__setattr__(self, "_base_ng", int(ng))
        object.__setattr__(self, "_base_nb", int(nb))
        object.__setattr__(self, "_base_nc", int(nc))
        object.__setattr__(self, "_base_root_id", None)

    def _inherit_base(self, out: "SparseGcZ") -> "SparseGcZ":
        """Preserve shared-factor prefix metadata across exact sparse ops."""
        object.__setattr__(
            out, "_base_ng",
            int(min(getattr(self, "_base_ng", self.ng), out.ng)),
        )
        object.__setattr__(
            out, "_base_nb",
            int(min(getattr(self, "_base_nb", self.nb), out.nb)),
        )
        object.__setattr__(
            out, "_base_nc",
            int(min(getattr(self, "_base_nc", self.nc), out.nc)),
        )
        object.__setattr__(
            out, "_base_root_id", getattr(self, "_base_root_id", None)
        )
        return out

    @property
    def dim(self) -> int:
        return int(self.c.numel())

    @property
    def n(self) -> int:
        return self.dim

    @property
    def ng(self) -> int:
        return int(self.Gc_sparse.shape[1])

    @property
    def nb(self) -> int:
        return int(self.Gb_sparse.shape[1])

    @property
    def nc(self) -> int:
        return int(self.b.shape[0])

    def _constraints_with_ng(self, ng: int) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.Ac_sparse.indices(), self.Ac_sparse.values(),
            (self.nc, int(ng)), dtype=self.dtype, device=self.device,
        ).coalesce()

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rad = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        if self.Gc_sparse._nnz() > 0:
            ind = self.Gc_sparse.indices()
            val = self.Gc_sparse.values().abs()
            rad.scatter_add_(0, ind[0], val)
        # Binary generators contribute |Gb_sparse| with xi_b in {-1,+1}.
        if self.Gb_sparse._nnz() > 0:
            ind_b = self.Gb_sparse.indices()
            val_b = self.Gb_sparse.values().abs()
            rad.scatter_add_(0, ind_b[0], val_b)
        return self.c - rad, self.c + rad

    def to_hzono(self) -> HZono:
        """Densify Gc + Gb → return HZono."""
        Gc_dense = self.Gc_sparse.to_dense()
        Gb_dense = (
            self.Gb_sparse.to_dense()
            if self.nb > 0
            else torch.zeros((self.dim, 0), dtype=self.dtype, device=self.device)
        )
        Ab_dense = (
            self.Ab_sparse.to_dense()
            if self.nb > 0
            else torch.zeros((self.nc, 0), dtype=self.dtype, device=self.device)
        )
        out = HZono(
            c=self.c.view(-1, 1),
            Gc=Gc_dense,
            Gb=Gb_dense,
            Ac=self.Ac_sparse.to_dense(),
            Ab=Ab_dense,
            b=self.b.clone(),
            eq_mask=self.eq_mask.clone(),
        )
        object.__setattr__(
            out, "_base_ng",
            int(min(getattr(self, "_base_ng", self.ng), out.ng)),
        )
        object.__setattr__(
            out, "_base_nb",
            int(min(getattr(self, "_base_nb", self.nb), out.nb)),
        )
        object.__setattr__(
            out, "_base_nc",
            int(min(getattr(self, "_base_nc", self.nc), out.nc)),
        )
        root_id = getattr(self, "_base_root_id", None)
        object.__setattr__(
            out, "_base_root_id",
            None if root_id is None else int(root_id),
        )
        return out

    # Alias for parity with HyZor's HybridZonotope-style API. HyZor's
    # `to_dense_hz` returns HybridZonotope; here we return HZono with
    # the same 6-tuple contents.
    to_dense_hz = to_hzono

    # ─── Y2 Stage 1: methods needed for Phase 1-3 routing ──────────────

    def _bounds_unconstrained(self):
        """HyZor-style API: returns (n,1) tensors. Alias for ``bounds()``
        with column-vector shape, matching HybridZonotope's convention."""
        lb, ub = self.bounds()
        return lb.view(-1, 1), ub.view(-1, 1)

    def density_bytes(self) -> int:
        """Estimated sparse storage cost: 16 bytes per nnz (idx + val)."""
        return self.Gc_sparse._nnz() * 16

    def apply_scale(self, a) -> "SparseGcZ":
        """Exact row-wise scaling ``y_i = a_i x_i``, preserves sparse Gc.

        Faithful port of HyZor ``SparseGcZ.apply_scale`` (__init__.py:482).
        """
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device).flatten()
        if a.numel() == 1:
            a = a.expand(self.dim)
        if a.numel() != self.dim:
            raise ValueError(f"apply_scale: size mismatch {a.numel()} vs {self.dim}")
        c_new = self.c * a
        ind = self.Gc_sparse.indices()
        val = self.Gc_sparse.values()
        if val.numel() > 0:
            scaled_val = val * a[ind[0]]
            nz = scaled_val != 0
            ind = ind[:, nz]
            scaled_val = scaled_val[nz]
        else:
            scaled_val = val
        Gc_new = torch.sparse_coo_tensor(
            ind, scaled_val, self.Gc_sparse.shape,
            dtype=self.dtype, device=self.device,
        ).coalesce()
        return self._inherit_base(SparseGcZ(
            c=c_new, Gc_sparse=Gc_new, dtype=self.dtype, device=self.device,
            Ac_sparse=self.Ac_sparse, b=self.b, eq_mask=self.eq_mask,
        ))

    def apply_dense(self, W, b=None) -> HZono:
        """``y = W x + b``. Sparse Gc → dense Gc' = W @ Gc; return HZono.

        Faithful port of HyZor ``SparseGcZ.apply_dense`` (__init__.py:464).
        Note: the result is dense (HZono), not sparse — densification of
        ``W @ Gc_sparse`` typically loses sparsity since W is dense.
        """
        W_t = torch.as_tensor(W, dtype=self.dtype, device=self.device)
        # (W @ Gc_sparse) computed via (Gc_sparse^T @ W^T)^T to use
        # torch.sparse.mm efficiently.
        Gc_dense = torch.sparse.mm(self.Gc_sparse.t(), W_t.t()).t()  # (n_out, ng)
        c_new = W_t @ self.c
        if b is not None:
            b_t = torch.as_tensor(b, dtype=self.dtype, device=self.device).flatten()
            c_new = c_new + b_t
        n_out = c_new.numel()
        ng = Gc_dense.shape[1]
        out = HZono(
            c=c_new.view(-1, 1),
            Gc=Gc_dense,
            Gb=torch.zeros((n_out, 0), dtype=self.dtype, device=self.device),
            Ac=self.Ac_sparse.to_dense(),
            Ab=torch.zeros((self.nc, 0), dtype=self.dtype, device=self.device),
            b=self.b.clone(),
            eq_mask=self.eq_mask.clone(),
        )
        object.__setattr__(
            out, "_base_ng",
            int(min(getattr(self, "_base_ng", self.ng), out.ng)),
        )
        object.__setattr__(
            out, "_base_nb",
            int(min(getattr(self, "_base_nb", self.nb), out.nb)),
        )
        object.__setattr__(
            out, "_base_nc",
            int(min(getattr(self, "_base_nc", self.nc), out.nc)),
        )
        root_id = getattr(self, "_base_root_id", None)
        object.__setattr__(
            out, "_base_root_id",
            None if root_id is None else int(root_id),
        )
        return out

    def apply_relu_triangle(
        self, *,
        external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> "SparseGcZ":
        """DeepZ triangle ReLU on sparse Gc.

        Per unstable neuron i with bounds ``[lb_i, ub_i]`` (l<0<u)::

            lam_i = ub_i / (ub_i - lb_i)
            mu_i  = -lb_i * ub_i / (2 (ub_i - lb_i))
            y_i   = lam_i x_i + mu_i + mu_i eps_new_i

        Sound. Adds k new generators (one per unstable neuron). Faithful
        port of HyZor ``SparseGcZ.apply_relu_triangle`` (__init__.py:506).
        """
        n = self.dim
        ng0 = self.ng
        if external_bounds is None:
            lb, ub = self.bounds()
        else:
            lb, ub = external_bounds
            lb = lb.to(dtype=self.dtype, device=self.device).view(-1)
            ub = ub.to(dtype=self.dtype, device=self.device).view(-1)
            if int(lb.numel()) != self.dim or int(ub.numel()) != self.dim:
                raise ValueError("SparseGcZ triangle external bounds dimension mismatch")
        is_active = (lb >= 0)
        is_inactive = (ub <= 0)
        is_unstable = ~(is_active | is_inactive)
        unstable_idx = torch.nonzero(is_unstable, as_tuple=False).view(-1)
        k = int(unstable_idx.numel())

        c_out = torch.zeros(n, dtype=self.dtype, device=self.device)
        c_out[is_active] = self.c[is_active]
        if k > 0:
            l_uns = lb[unstable_idx]
            u_uns = ub[unstable_idx]
            lam = u_uns / (u_uns - l_uns)
            mu = -l_uns * u_uns / (2.0 * (u_uns - l_uns))
            c_out[unstable_idx] = lam * self.c[unstable_idx] + mu

        # Row scaling on old sparse Gc.
        row_scale = torch.zeros(n, dtype=self.dtype, device=self.device)
        row_scale[is_active] = 1.0
        if k > 0:
            row_scale[unstable_idx] = lam

        old_ind = self.Gc_sparse.indices()
        old_val = self.Gc_sparse.values()
        if old_val.numel() > 0:
            scaled_val = old_val * row_scale[old_ind[0]]
            nz = scaled_val != 0
            kept_ind = old_ind[:, nz]
            kept_val = scaled_val[nz]
        else:
            kept_ind = old_ind
            kept_val = old_val

        # New unstable column block: (unstable_idx[j], ng0 + j) = mu[j].
        if k > 0:
            new_rows = unstable_idx
            new_cols = ng0 + torch.arange(k, dtype=torch.long, device=self.device)
            new_ind = torch.stack([new_rows, new_cols])
            new_val = mu
            all_ind = torch.cat([kept_ind, new_ind], dim=1)
            all_val = torch.cat([kept_val, new_val])
        else:
            all_ind = kept_ind
            all_val = kept_val

        ng_out = ng0 + k
        Gc_out = torch.sparse_coo_tensor(
            all_ind, all_val, (n, ng_out),
            dtype=self.dtype, device=self.device,
        )
        return self._inherit_base(SparseGcZ(
            c=c_out, Gc_sparse=Gc_out, dtype=self.dtype, device=self.device,
            Ac_sparse=self._constraints_with_ng(ng_out),
            b=self.b, eq_mask=self.eq_mask,
        ))

    def apply_relu_selective_chull(
        self, *, chull_mask: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        external_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> "SparseGcZ":
        """Triangle ReLU plus sparse convex-hull facets for selected neurons.

        Unlike the dense HZono implementation, this keeps both generator
        coordinates and the added inequality rows sparse. It therefore
        allows property-directed cuts in the wide early ReLUs without
        materialising ``dim x ng`` dense matrices.
        """
        if external_bounds is None:
            lb, ub = self.bounds()
        else:
            lb, ub = external_bounds
            lb = lb.to(dtype=self.dtype, device=self.device).view(-1)
            ub = ub.to(dtype=self.dtype, device=self.device).view(-1)
            if int(lb.numel()) != self.dim or int(ub.numel()) != self.dim:
                raise ValueError("SparseGcZ selective external bounds dimension mismatch")
        unstable = (lb < 0) & (ub > 0)
        unstable_idx = torch.nonzero(unstable, as_tuple=False).view(-1)
        k = int(unstable_idx.numel())
        out = self.apply_relu_triangle(external_bounds=(lb, ub))
        if k == 0:
            return out
        if chull_mask is not None:
            mask = chull_mask.to(device=self.device, dtype=torch.bool).view(-1)
            if int(mask.numel()) != self.dim:
                raise ValueError("SparseGcZ selective mask dimension mismatch")
            selected_local = torch.nonzero(
                mask[unstable_idx], as_tuple=False
            ).view(-1)
        elif top_k is not None and int(top_k) > 0:
            kk = min(int(top_k), k)
            selected_local = torch.topk(
                _relu_selection_score(lb[unstable_idx], ub[unstable_idx]), kk
            ).indices
        else:
            return out
        n_sel = int(selected_local.numel())
        if n_sel == 0:
            return out

        selected_rows = unstable_idx[selected_local]
        l = lb[selected_rows]
        u = ub[selected_rows]
        lam = u / (u - l)
        mu = -l * u / (2.0 * (u - l))
        sel_c = self.c[selected_rows]

        row_lookup = -torch.ones(
            self.dim, dtype=torch.long, device=self.device
        )
        row_lookup[selected_rows] = torch.arange(
            n_sel, dtype=torch.long, device=self.device
        )
        old_idx = self.Gc_sparse.indices()
        old_val = self.Gc_sparse.values()
        local = row_lookup[old_idx[0]]
        useful = local >= 0
        local = local[useful]
        cols = old_idx[1, useful]
        vals = old_val[useful]
        cut_rows1 = 2 * local
        cut_rows2 = cut_rows1 + 1
        cut_idx_old = torch.cat([
            torch.stack([cut_rows1, cols]),
            torch.stack([cut_rows2, cols]),
        ], dim=1)
        cut_val_old = torch.cat([
            (1.0 - lam[local]) * vals,
            -lam[local] * vals,
        ])
        eps_cols = self.ng + selected_local
        cut_idx_eps = torch.stack([
            torch.cat([2 * torch.arange(n_sel, device=self.device),
                       2 * torch.arange(n_sel, device=self.device) + 1]),
            torch.cat([eps_cols, eps_cols]),
        ])
        cut_val_eps = torch.cat([-mu, -mu])
        cut_idx = torch.cat([cut_idx_old, cut_idx_eps], dim=1)
        cut_val = torch.cat([cut_val_old, cut_val_eps])
        cut_Ac = torch.sparse_coo_tensor(
            cut_idx, cut_val, (2 * n_sel, out.ng),
            dtype=self.dtype, device=self.device,
        ).coalesce()

        old_idx_out = out.Ac_sparse.indices()
        all_idx = torch.cat([
            old_idx_out,
            torch.stack([
                cut_Ac.indices()[0] + out.nc,
                cut_Ac.indices()[1],
            ]),
        ], dim=1)
        all_val = torch.cat([out.Ac_sparse.values(), cut_Ac.values()])
        Ac_out = torch.sparse_coo_tensor(
            all_idx, all_val, (out.nc + 2 * n_sel, out.ng),
            dtype=self.dtype, device=self.device,
        ).coalesce()
        b_cut = torch.empty((2 * n_sel, 1), dtype=self.dtype, device=self.device)
        b_cut[0::2, 0] = mu - (1.0 - lam) * sel_c
        b_cut[1::2, 0] = lam * sel_c + mu
        return out._inherit_base(SparseGcZ(
            c=out.c, Gc_sparse=out.Gc_sparse,
            dtype=self.dtype, device=self.device,
            Ac_sparse=Ac_out,
            b=torch.cat([out.b, b_cut], dim=0),
            eq_mask=torch.cat([
                out.eq_mask,
                torch.zeros(2 * n_sel, dtype=torch.bool, device=self.device),
            ]),
        ))

    def apply_conv(self, weight, bias, in_shape, stride, pad,
                   *, chunk=None, sparsify_thresh=None) -> "SparseGcZ":
        """Apply Conv2d to each generator column, preserve sparse Gc.

        Fast path: build conv as sparse linear operator and sparse-mm
        against the sparse Gc. Falls back to per-column chunked densify
        + F.conv2d + resparsify when the operator's nnz exceeds budget.

        Faithful port of HyZor ``SparseGcZ.apply_conv`` (__init__.py:577).
        """
        import torch.nn.functional as F

        if isinstance(stride, int):
            stride = (stride, stride)
        Cin, Hin, Win = in_shape
        Cout, Cin_w, kH, kW = weight.shape
        assert Cin == Cin_w, f"in_shape Cin={Cin} != weight Cin={Cin_w}"
        Hout = (Hin + 2 * pad - kH) // stride[0] + 1
        Wout = (Win + 2 * pad - kW) // stride[1] + 1
        n_in = self.dim
        n_out = Cout * Hout * Wout

        if chunk is None:
            budget = _sparse_gc_budget_bytes()
            per_col = (n_in + n_out) * 8
            chunk = max(8, min(self.ng, budget // max(per_col, 1)))
        if sparsify_thresh is None:
            sparsify_thresh = float(os.environ.get("HYZOR_SPARSE_GC_THRESH", "0.0"))

        weight_t = weight.to(dtype=self.dtype, device=self.device)
        bias_t = None
        if bias is not None and bias.numel() > 0:
            bias_t = bias.to(dtype=self.dtype, device=self.device)

        # Convolve the center (with bias).
        c4 = self.c.view(1, Cin, Hin, Win)
        c_new = F.conv2d(c4, weight_t, bias_t,
                          stride=stride, padding=pad).view(-1)

        # Exact sparse scatter path. When the generator matrix is genuinely
        # sparse (the common case after zero-width input pruning), avoid both
        # building the full convolution matrix and densifying each generator
        # column into a full feature map. Each sparse input entry contributes
        # to at most Cout*kH*kW sparse output entries.
        if os.environ.get("HYZOR_SPARSE_CONV_SCATTER", "1") == "1":
            ind0 = self.Gc_sparse.indices()
            val0 = self.Gc_sparse.values()
            nnz0 = int(val0.numel())
            est_entries = nnz0 * int(Cout) * int(kH) * int(kW)
            max_scatter_entries = int(os.environ.get(
                "HYZOR_SPARSE_CONV_SCATTER_MAX_ENTRIES", "5000000"
            ))
            elem = torch.empty((), dtype=self.dtype, device=self.device).element_size()
            if (0 < est_entries <= max_scatter_entries
                    and est_entries * (elem + 16) <= _sparse_gc_budget_bytes()):
                rows_in = ind0[0]
                cols_in = ind0[1]
                cin = rows_in // (Hin * Win)
                rem = rows_in % (Hin * Win)
                ih = rem // Win
                iw = rem % Win
                all_ind = []
                all_val = []
                cout_idx = torch.arange(Cout, dtype=torch.long, device=self.device).view(-1, 1)
                for kh in range(int(kH)):
                    oh_num = ih + int(pad) - kh
                    ok_h = (oh_num >= 0) & (oh_num % int(stride[0]) == 0)
                    oh = oh_num // int(stride[0])
                    ok_h = ok_h & (oh >= 0) & (oh < Hout)
                    if not bool(ok_h.any().item()):
                        continue
                    for kw in range(int(kW)):
                        ow_num = iw + int(pad) - kw
                        ok = ok_h & (ow_num >= 0) & (ow_num % int(stride[1]) == 0)
                        ow = ow_num // int(stride[1])
                        ok = ok & (ow >= 0) & (ow < Wout)
                        if not bool(ok.any().item()):
                            continue
                        sel_cin = cin[ok]
                        sel_oh = oh[ok]
                        sel_ow = ow[ok]
                        sel_cols = cols_in[ok]
                        sel_vals = val0[ok]
                        w = weight_t[:, sel_cin, kh, kw]
                        contrib = w * sel_vals.view(1, -1)
                        nz = contrib != 0
                        if not bool(nz.any().item()):
                            continue
                        out_rows = (
                            cout_idx * (Hout * Wout)
                            + sel_oh.view(1, -1) * Wout
                            + sel_ow.view(1, -1)
                        )
                        out_cols = sel_cols.view(1, -1).expand(Cout, -1)
                        all_ind.append(torch.stack([out_rows[nz], out_cols[nz]], dim=0))
                        all_val.append(contrib[nz])
                if all_ind:
                    Gc_new = torch.sparse_coo_tensor(
                        torch.cat(all_ind, dim=1),
                        torch.cat(all_val, dim=0),
                        (n_out, self.ng),
                        dtype=self.dtype, device=self.device,
                    ).coalesce()
                else:
                    Gc_new = torch.sparse_coo_tensor(
                        torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        torch.zeros(0, dtype=self.dtype, device=self.device),
                        (n_out, self.ng),
                        dtype=self.dtype, device=self.device,
                    ).coalesce()
                Gc_new = _sparsify_with_row_slack(
                    Gc_new, sparsify_thresh,
                    dtype=self.dtype, device=self.device,
                )
                return self._inherit_base(SparseGcZ(
                    c=c_new, Gc_sparse=Gc_new,
                    dtype=self.dtype, device=self.device,
                    Ac_sparse=self.Ac_sparse, b=self.b, eq_mask=self.eq_mask,
                ))

        # Fast path: sparse conv operator × sparse Gc.
        if os.environ.get("HYZOR_SPARSE_CONV_MM", "1") == "1":
            try:
                op_nnz = _conv2d_sparse_nnz(weight_t, in_shape, stride, pad)
                if op_nnz * 16 <= _sparse_gc_budget_bytes():
                    W_sparse = build_sparse_conv_matrix(
                        weight_t, in_shape, stride, pad,
                        dtype=self.dtype, device=self.device,
                    )
                    Gc_raw = torch.sparse.mm(W_sparse, self.Gc_sparse).coalesce()
                    Gc_new = _sparsify_with_row_slack(
                        Gc_raw, sparsify_thresh,
                        dtype=self.dtype, device=self.device,
                    )
                    return self._inherit_base(SparseGcZ(
                        c=c_new, Gc_sparse=Gc_new,
                        dtype=self.dtype, device=self.device,
                        Ac_sparse=self.Ac_sparse, b=self.b, eq_mask=self.eq_mask,
                    ))
            except (MemoryError, RuntimeError) as e:
                msg = str(e).lower()
                if "memory" not in msg and "alloc" not in msg:
                    raise

        # Fallback: per-column chunked densify + F.conv2d.
        ind = self.Gc_sparse.indices()
        val = self.Gc_sparse.values()
        ng = self.ng
        if sparsify_thresh > 0:
            # Slack-based pruning incompatible with chunk path: keep exact.
            sparsify_thresh = 0.0

        out_ind_chunks = []
        out_val_chunks = []
        for j0 in range(0, ng, chunk):
            j1 = min(j0 + chunk, ng)
            col_mask = (ind[1] >= j0) & (ind[1] < j1)
            if not col_mask.any():
                continue
            sub_ind = ind[:, col_mask]
            sub_val = val[col_mask]
            batch = torch.zeros(
                (j1 - j0, n_in), dtype=self.dtype, device=self.device,
            )
            batch[sub_ind[1] - j0, sub_ind[0]] = sub_val
            batch4 = batch.view(j1 - j0, Cin, Hin, Win)
            out4 = F.conv2d(batch4, weight_t, None,
                             stride=stride, padding=pad)
            out_dense = out4.view(j1 - j0, n_out)
            if sparsify_thresh > 0:
                nz = out_dense.abs() > sparsify_thresh
            else:
                nz = out_dense != 0
            if nz.any():
                rows_l, cols_l = torch.nonzero(nz, as_tuple=True)
                out_ind_chunks.append(torch.stack([cols_l, j0 + rows_l]))
                out_val_chunks.append(out_dense[rows_l, cols_l])

        if out_ind_chunks:
            all_ind = torch.cat(out_ind_chunks, dim=1)
            all_val = torch.cat(out_val_chunks)
        else:
            all_ind = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            all_val = torch.zeros((0,), dtype=self.dtype, device=self.device)

        Gc_new = torch.sparse_coo_tensor(
            all_ind, all_val, (n_out, ng),
            dtype=self.dtype, device=self.device,
        ).coalesce()
        return self._inherit_base(SparseGcZ(
            c=c_new, Gc_sparse=Gc_new, dtype=self.dtype, device=self.device,
            Ac_sparse=self.Ac_sparse, b=self.b, eq_mask=self.eq_mask,
        ))

    def reduce_generators(self, target_ng: int) -> "SparseGcZ":
        """Girard-style box overapproximation of low-magnitude generators.

        Sound zonotope reduction: drop the smallest-norm generators and
        replace their effect with diagonal box generators (one per output
        dim). Caps ``ng`` at ``target_ng + n``. Faithful port of HyZor
        ``SparseGcZ.reduce_generators`` (__init__.py:688).
        """
        ng0 = self.ng
        n = self.dim
        if ng0 <= target_ng:
            return self
        if self.nc > 0:
            # Remapping constrained factors through sparse Girard widening
            # requires widening each retained inequality row. Preserve the
            # tighter constrained representation until that path is needed.
            return self

        ind = self.Gc_sparse.indices()
        val = self.Gc_sparse.values()
        col_sq = torch.zeros(ng0, dtype=self.dtype, device=self.device)
        col_sq.scatter_add_(0, ind[1], val * val)
        _, top_idx = torch.topk(col_sq, k=target_ng, largest=True, sorted=False)
        keep_mask = torch.zeros(ng0, dtype=torch.bool, device=self.device)
        keep_mask[top_idx] = True
        new_col_map = -torch.ones(ng0, dtype=torch.long, device=self.device)
        new_col_map[top_idx] = torch.arange(
            target_ng, dtype=torch.long, device=self.device,
        )

        col_kept_mask = keep_mask[ind[1]]
        kept_rows = ind[0, col_kept_mask]
        kept_new_cols = new_col_map[ind[1, col_kept_mask]]
        kept_vals = val[col_kept_mask]

        dropped_mask = ~col_kept_mask
        slack = torch.zeros(n, dtype=self.dtype, device=self.device)
        slack.scatter_add_(0, ind[0, dropped_mask], val[dropped_mask].abs())
        nz_slack_idx = torch.nonzero(slack, as_tuple=False).view(-1)

        slack_cols = target_ng + torch.arange(
            int(nz_slack_idx.numel()),
            dtype=torch.long, device=self.device,
        )
        slack_ind = torch.stack([nz_slack_idx, slack_cols])
        slack_vals = slack[nz_slack_idx]

        all_ind = torch.cat(
            [torch.stack([kept_rows, kept_new_cols]), slack_ind],
            dim=1,
        )
        all_val = torch.cat([kept_vals, slack_vals])
        ng_new = target_ng + int(nz_slack_idx.numel())
        Gc_new = torch.sparse_coo_tensor(
            all_ind, all_val, (n, ng_new),
            dtype=self.dtype, device=self.device,
        ).coalesce()
        return SparseGcZ(
            c=self.c, Gc_sparse=Gc_new, dtype=self.dtype, device=self.device,
            Ac_sparse=self._constraints_with_ng(ng_new),
            b=self.b, eq_mask=self.eq_mask,
        )
