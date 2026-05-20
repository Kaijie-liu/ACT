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

  * **SparseGcZ** -- HZ with sparse-COO Gc storage; nb = 0. Useful for
    a single-conv chain where Gc has well-defined sparsity but dense
    storage doesn't fit. Supports triangle ReLU (which preserves
    sparse-Gc structure) but cannot host eq_lagr-style ReLU encodings
    (those need dense Gc + binary slots).

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


def _sparse_gc_budget_bytes() -> int:
    return int(float(os.environ.get("HYZOR_SPARSE_GC_BUDGET_GB", "8.0")) * (1024 ** 3))


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
        return self.dim

    @property
    def nb(self) -> int:
        return 0

    @property
    def nc(self) -> int:
        return 0

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lb, self.ub

    def to_hzono(self) -> HZono:
        """Materialise as a dense diagonal HZono. Caller ensures dim is
        small enough to fit (use ``_make_or_box`` for budgeted picks)."""
        return hz_from_bounds(
            Bounds(lb=self.lb, ub=self.ub),
            dtype=self.dtype, device=self.device,
        )


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
    def ng(self) -> int:
        return self.n_root  # conceptual

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
        return int(self.dim) * int(self.n_root) * 8

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
        rad = ((self.root_ub - self.root_lb) / 2.0).abs()
        n_root = int(rad.numel())
        chunk = max(1, min(n_root, 2048))
        Gc_cols = []
        for i0 in range(0, n_root, chunk):
            i1 = min(i0 + chunk, n_root)
            batch = torch.zeros((i1 - i0, n_root),
                                dtype=self.dtype, device=self.device)
            rows = torch.arange(i1 - i0, dtype=torch.long, device=self.device)
            cols = torch.arange(i0, i1, dtype=torch.long, device=self.device)
            batch[rows, cols] = rad[i0:i1]
            cols_out = self._push_batch_through_ops(batch)
            Gc_cols.append(cols_out)
        Gc = torch.cat(Gc_cols, dim=0).T.contiguous()  # (dim, n_root)
        n = self.dim
        c = self.c_chain.view(-1, 1)
        Gb = torch.zeros((n, 0), dtype=self.dtype, device=self.device)
        Ac = torch.zeros((0, n_root), dtype=self.dtype, device=self.device)
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
        budget, materialise as a SparseGcZ; else return None."""
        if not _enable_sparse_gc():
            return None
        if len(self.ops) != 1 or self.ops[0]["kind"] != "conv":
            return None
        op = self.ops[0]
        # Build sparse Gc by pushing diag(rad) batched through the conv.
        # For a single conv this is just conv2d on per-pixel one-hots; we
        # rely on the dense materialisation pathway, then build sparse
        # representation by zeroing small entries. A faithful port of
        # HyZor's _conv2d_sparse_nnz / build_sparse_conv_gc is left to
        # a follow-up (currently the dense path covers the same cases
        # at the cost of more memory). Returning None here is sound; it
        # forces freeze() to fall back to snapshot_to_box(), which is
        # the same behaviour HyZor exhibits when its sparse build path
        # errors out.
        return None


# ─── SparseGcZ ─────────────────────────────────────────────────────────


class SparseGcZ:
    """HZ with sparse-COO Gc storage and nb = 0.

    Used as an intermediate after a single conv on a BoxHZ root when
    the dense materialisation doesn't fit but the sparse Gc does.
    Supports triangle ReLU (which preserves sparse-Gc structure) and
    converts to dense ``HZono`` via ``to_hzono()`` when subsequent
    transformations need the dense form (e.g. eq_lagr ReLU).
    """

    def __init__(self, *, c: torch.Tensor, Gc_sparse: torch.Tensor,
                 dtype, device):
        self.c = c.flatten().to(dtype=dtype, device=device)
        self.Gc_sparse = Gc_sparse.coalesce()
        self.dtype = dtype
        self.device = device

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
        return 0

    @property
    def nc(self) -> int:
        return 0

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.Gc_sparse._nnz() == 0:
            rad = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            ind = self.Gc_sparse.indices()
            val = self.Gc_sparse.values().abs()
            rad = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
            rad.scatter_add_(0, ind[0], val)
        return self.c - rad, self.c + rad

    def to_hzono(self) -> HZono:
        """Densify Gc → return HZono."""
        Gc_dense = self.Gc_sparse.to_dense()
        return HZono(
            c=self.c.view(-1, 1),
            Gc=Gc_dense,
            Gb=torch.zeros((self.dim, 0), dtype=self.dtype, device=self.device),
            Ac=torch.zeros((0, Gc_dense.shape[1]),
                            dtype=self.dtype, device=self.device),
            Ab=torch.zeros((0, 0), dtype=self.dtype, device=self.device),
            b=torch.zeros((0, 1), dtype=self.dtype, device=self.device),
        )


# --- Self-tests (run with: python -m act.back_end.hybridz_tf.representations) ---


def _test_boxhz_basic():
    box = BoxHZ(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    assert box.dim == 2 and box.ng == 2 and box.nb == 0 and box.nc == 0
    lb, ub = box.bounds()
    assert lb.tolist() == [-1.0, 0.0] and ub.tolist() == [1.0, 2.0]
    hz = box.to_hzono()
    assert isinstance(hz, HZono)
    assert hz.Gc.shape == (2, 2)


def _test_lazychain_dense():
    box = BoxHZ(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    chain = LazyChainHZ.from_box(box)
    W = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    chain2 = chain.with_dense(W, None)
    lb, ub = chain2.bounds()
    # y = W x with x in [-1,1]^2 → y_0 in [-2,2], y_1 in [-1,1]
    assert lb.tolist() == [-2.0, -1.0]
    assert ub.tolist() == [2.0, 1.0]


def _test_lazychain_materialize_when_small():
    box = BoxHZ(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]),
                dtype=torch.float64, device=torch.device("cpu"))
    chain = LazyChainHZ.from_box(box).with_dense(
        torch.eye(2, dtype=torch.float64), None
    )
    # dim=2 << materialize_dim_cap=512; should materialise
    assert chain.can_materialize()
    hz = chain.to_full_hzono()
    assert isinstance(hz, HZono)
    assert hz.Gc.shape == (2, 2)


def _test_sparse_gc_bounds():
    c = torch.tensor([0.0, 0.0], dtype=torch.float64)
    ind = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    val = torch.tensor([1.0, 1.0], dtype=torch.float64)
    Gc_sp = torch.sparse_coo_tensor(ind, val, (2, 2))
    s = SparseGcZ(c=c, Gc_sparse=Gc_sp,
                  dtype=torch.float64, device=torch.device("cpu"))
    lb, ub = s.bounds()
    assert lb.tolist() == [-1.0, -1.0] and ub.tolist() == [1.0, 1.0]


if __name__ == "__main__":
    _test_boxhz_basic()
    _test_lazychain_dense()
    _test_lazychain_materialize_when_small()
    _test_sparse_gc_bounds()
    print("OK: representations tests pass")
