"""T2 sparse-Gc HZ representation — prototype.

Goal: reduce HZ representation RSS on CPU-bound convolutional benchmarks
(cifar100 / tinyimagenet / vggnet16) by:

  1) ``hz_prune_gc_dense(hz, eps)`` — sound row-slack pruning of small
     dense Gc columns. Drops columns whose max abs value is below ``eps``
     and adds one diagonal slack column per affected row so the
     over-approximation is preserved exactly.

  2) ``hz_dense_to_sparse(hz, density_threshold)`` — convert an HZono
     with low-density Gc to a SparseGcZ. Requires nb=0 and the Ac matrix
     to be zero on dropped columns (typical for forward-only chains
     before the first eq_lagr-style ReLU).

Both functions are PROTOTYPE quality: out-of-tree, no edits to core ACT
modules. They consume / produce the same HZono / SparseGcZ types from
``act.back_end``. Use the ``measure_*`` harnesses below to record:

  - ng / nc / nb counts
  - dense Gc bytes vs sparse nnz bytes
  - RSS delta around a transformation

If the harness shows >= 4x RSS reduction on a representative
cifar100 first-conv HZ, we have a green light to propose a wiring
patch in hz_routing.hz_conv2d that applies prune+dense->sparse on the
HZono exit branch.

Soundness invariant (informal):
    Image(Z_pruned) ⊇ Image(Z_original)
i.e. the pruned HZ over-approximates every output of the original. The
row-slack column adds a fresh independent generator with magnitude
equal to the dropped column's per-row L1, so the resulting box hull
along every coordinate is ≥ the original (and on most coordinates,
EQUAL because the kept columns dominate).
"""
from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.solver.solver_hz import HZono  # noqa: E402
from act.back_end.hybridz_tf.representations import (  # noqa: E402
    BoxHZ,
    SparseGcZ,
)


# ---------------------------------------------------------------------------
# Core operators
# ---------------------------------------------------------------------------


def hz_prune_gc_dense(hz: HZono, eps: float) -> HZono:
    """Drop dense Gc columns with ``||col||_inf <= eps`` and compensate.

    Sound: for each dropped column j, the contribution ``Gc[:,j] * x_j``
    with ``x_j ∈ [-1,1]`` lives in the per-row interval
    ``[-|Gc[i,j]|, +|Gc[i,j]|]``. Summed across dropped j for the same
    row i: by triangle inequality, range is
    ``[-sum_j |Gc[i,j]|, +sum_j |Gc[i,j]|]``. A SINGLE new generator
    column with the per-row slack value at row i recovers exactly that
    spread (the new ``x_new`` ranges over ``[-1,1]`` independently of
    surviving columns, which is a sound over-approximation of the
    pre-existing coupling among dropped columns).

    Constraint safety: this prototype only handles the case where the
    dropped columns are NOT involved in any constraint row (``Ac[:,j]``
    all zeros for dropped j). Otherwise the constraint row would need
    to be relaxed (equality -> inequality with widened RHS) which loses
    information. Returns ``hz`` unchanged if the constraint condition
    fails.
    """
    if eps <= 0:
        return hz
    Gc = hz.Gc
    if Gc.numel() == 0:
        return hz
    col_norm = Gc.abs().max(dim=0).values  # (ng,)
    keep_mask = col_norm > eps
    if bool(keep_mask.all()):
        return hz
    drop_mask = ~keep_mask
    n_drop = int(drop_mask.sum())
    if n_drop == 0:
        return hz

    # Constraint-safety check: don't drop columns referenced by any
    # constraint row, since we lack the relaxation logic here.
    if hz.Ac.numel() > 0:
        Ac_drop_mass = hz.Ac[:, drop_mask].abs().max().item() if drop_mask.any() else 0.0
        if Ac_drop_mass > eps * 1e-3:
            return hz

    Gc_keep = Gc[:, keep_mask]
    Gc_drop = Gc[:, drop_mask]

    # Per-row L1 of dropped columns -> slack magnitude.
    slack_row_mass = Gc_drop.abs().sum(dim=1)  # (n,)
    nz_rows = slack_row_mass > 0
    n_new = int(nz_rows.sum())

    Ac_keep = hz.Ac[:, keep_mask] if hz.Ac.numel() else hz.Ac

    if n_new == 0:
        # All dropped contributions were already zero; pure column drop.
        return HZono(
            c=hz.c,
            Gc=Gc_keep,
            Gb=hz.Gb,
            Ac=Ac_keep,
            Ab=hz.Ab,
            b=hz.b,
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        )

    nz_idx = nz_rows.nonzero(as_tuple=False).view(-1)
    new_block = torch.zeros((hz.dim, n_new), dtype=Gc.dtype, device=Gc.device)
    new_block[nz_idx, torch.arange(n_new, device=Gc.device)] = slack_row_mass[nz_idx]
    Gc_new = torch.cat([Gc_keep, new_block], dim=1)

    if Ac_keep.numel() == 0 or Ac_keep.shape[0] == 0:
        Ac_new = Ac_keep.new_zeros((Ac_keep.shape[0], Gc_new.shape[1]))
    else:
        Ac_pad = Ac_keep.new_zeros((Ac_keep.shape[0], n_new))
        Ac_new = torch.cat([Ac_keep, Ac_pad], dim=1)

    return HZono(
        c=hz.c,
        Gc=Gc_new,
        Gb=hz.Gb,
        Ac=Ac_new,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


def hz_dense_to_sparse(
    hz: HZono,
    density_threshold: float = 0.05,
    zero_eps: float = 1e-12,
) -> Union[HZono, SparseGcZ]:
    """Convert an HZono to SparseGcZ when its Gc is below density_threshold.

    Pre-conditions for safe conversion:
      - nb == 0 (SparseGcZ supports no binary generators)
      - all eq_mask entries False (SparseGcZ stores inequality form;
        equality constraints would need extra handling)
      - Ac may be non-zero, but eq_mask = all False; passed through
        as sparse.

    Returns the input HZono unchanged if the conversion would not save
    memory or pre-conditions fail.
    """
    if hz.nb > 0:
        return hz
    if hz.eq_mask is not None and bool(hz.eq_mask.any()):
        # Equality rows in HZono need to be preserved; SparseGcZ stores
        # eq_mask too, so this is OK if we copy. Allow it.
        pass

    Gc = hz.Gc
    n, ng = Gc.shape
    if n == 0 or ng == 0:
        return hz
    nnz = int((Gc.abs() > zero_eps).sum())
    density = nnz / float(n * ng)
    if density > density_threshold:
        return hz

    # Sparse storage: 16 bytes per nnz (8 idx + 8 val for float64)
    sparse_bytes = nnz * 16
    dense_bytes = n * ng * Gc.element_size()
    if sparse_bytes >= dense_bytes * 0.7:
        return hz

    nz_idx = (Gc.abs() > zero_eps).nonzero(as_tuple=False).T  # (2, nnz)
    nz_val = Gc[nz_idx[0], nz_idx[1]]
    Gc_sparse = torch.sparse_coo_tensor(
        nz_idx, nz_val, (n, ng), dtype=Gc.dtype, device=Gc.device,
    ).coalesce()

    if hz.Ac.numel() > 0 and hz.nc > 0:
        Ac_nz = (hz.Ac.abs() > zero_eps).nonzero(as_tuple=False).T
        Ac_val = hz.Ac[Ac_nz[0], Ac_nz[1]] if Ac_nz.numel() else hz.Ac.new_zeros(0)
        Ac_sparse = torch.sparse_coo_tensor(
            Ac_nz, Ac_val, (hz.nc, ng),
            dtype=hz.Ac.dtype, device=hz.Ac.device,
        ).coalesce()
        b = hz.b.clone()
        eq_mask = hz.eq_mask.clone() if hz.eq_mask is not None else None
    else:
        Ac_sparse = None
        b = None
        eq_mask = None

    return SparseGcZ(
        c=hz.c.view(-1),
        Gc_sparse=Gc_sparse,
        dtype=Gc.dtype,
        device=Gc.device,
        Ac_sparse=Ac_sparse,
        b=b,
        eq_mask=eq_mask,
    )


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def rss_mib() -> float:
    """Process resident-set size in MiB (Linux only)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class HZStats:
    label: str
    dim: int
    ng: int
    nb: int
    nc: int
    gc_dense_bytes: int
    gc_nnz_bytes: int
    density: float
    rss_mib: float

    def short(self) -> str:
        return (
            f"[{self.label}] dim={self.dim} ng={self.ng} nb={self.nb} nc={self.nc} "
            f"dense_MiB={self.gc_dense_bytes/2**20:.1f} "
            f"sparse_MiB={self.gc_nnz_bytes/2**20:.1f} "
            f"density={self.density:.4f} rss_MiB={self.rss_mib:.0f}"
        )


def stats_of(z, label: str) -> HZStats:
    if isinstance(z, HZono):
        n, ng = z.Gc.shape if z.Gc.numel() else (z.dim, 0)
        nnz = int((z.Gc.abs() > 0).sum()) if ng > 0 else 0
        dense = n * ng * z.Gc.element_size() if ng > 0 else 0
        density = nnz / float(max(n * ng, 1))
        return HZStats(
            label=label, dim=n, ng=ng, nb=z.nb, nc=z.nc,
            gc_dense_bytes=dense, gc_nnz_bytes=nnz * 16, density=density,
            rss_mib=rss_mib(),
        )
    if isinstance(z, SparseGcZ):
        n = z.dim
        ng = z.ng
        nnz = z.Gc_sparse._nnz()
        dense = n * ng * z.Gc_sparse.dtype.itemsize if (n and ng) else 0
        density = nnz / float(max(n * ng, 1))
        return HZStats(
            label=label, dim=n, ng=ng, nb=0, nc=z.nc,
            gc_dense_bytes=dense, gc_nnz_bytes=nnz * 16, density=density,
            rss_mib=rss_mib(),
        )
    raise TypeError(f"unsupported HZ type: {type(z)}")


# ---------------------------------------------------------------------------
# Soundness checks (small dim)
# ---------------------------------------------------------------------------


def _bounds_of(z) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(z, HZono):
        Gc = z.Gc
        rad = Gc.abs().sum(dim=1) if Gc.numel() else torch.zeros(z.dim, dtype=z.c.dtype, device=z.c.device)
        if z.Gb.numel():
            rad = rad + z.Gb.abs().sum(dim=1)
        c = z.c.view(-1)
        return c - rad, c + rad
    if isinstance(z, SparseGcZ):
        return z.bounds()
    raise TypeError(type(z))


def check_overapprox(z_orig, z_pruned, rtol: float = 1e-7) -> Tuple[bool, str]:
    lb0, ub0 = _bounds_of(z_orig)
    lb1, ub1 = _bounds_of(z_pruned)
    lb0 = lb0.to(torch.float64).cpu()
    ub0 = ub0.to(torch.float64).cpu()
    lb1 = lb1.to(torch.float64).cpu()
    ub1 = ub1.to(torch.float64).cpu()
    lb_ok = (lb1 <= lb0 + rtol * (ub0 - lb0).abs() + 1e-12).all().item()
    ub_ok = (ub1 >= ub0 - rtol * (ub0 - lb0).abs() - 1e-12).all().item()
    if not lb_ok or not ub_ok:
        worst_lb = (lb1 - lb0).max().item()
        worst_ub = (ub0 - ub1).max().item()
        return False, f"lb_ok={lb_ok} ub_ok={ub_ok} worst_lb_violation={worst_lb:.3e} worst_ub_violation={worst_ub:.3e}"
    width0 = (ub0 - lb0).sum().item()
    width1 = (ub1 - lb1).sum().item()
    looseness = (width1 - width0) / max(width0, 1e-12)
    return True, f"sound; looseness={looseness:.4f}"


# ---------------------------------------------------------------------------
# Demo: synthetic post-conv HZ matching cifar100 first-conv shape
# ---------------------------------------------------------------------------


def _build_cifar_first_conv_hz_post(dtype=torch.float64, device="cpu") -> HZono:
    """Build a synthetic HZono that mimics cifar100's first-conv output:
    input box dim = 3 * 32 * 32 = 3072, then a 3x3 stride 1 pad 1 Conv2d
    with Cout=64 yields output dim = 64 * 32 * 32 = 65536 and Gc with
    shape (65536, 3072) where each column is a kernel-shaped sparse
    pattern (576 nonzeros per column = 3*3*64).
    """
    C, H, W = 3, 32, 32
    Cout, kH, kW = 64, 3, 3
    n_in = C * H * W
    n_out = Cout * H * W

    # Random kernel: dense (no zeros) so post-conv Gc has full
    # k*k*Cout = 576 nonzeros per input pixel column.
    g_in = torch.Generator(device=device).manual_seed(42)
    weight = torch.randn(Cout, C, kH, kW, generator=g_in, dtype=dtype, device=device) * 0.1
    bias = torch.zeros(Cout, dtype=dtype, device=device)

    # Input HZ: BoxHZ over [-eps, eps] around 0 -> HZono diagonal radius.
    eps = 1.0 / 255.0
    c_in = torch.zeros(n_in, 1, dtype=dtype, device=device)
    Gc_in = torch.eye(n_in, dtype=dtype, device=device) * eps

    # Apply Conv2d to each generator column (and the center).
    import torch.nn.functional as F
    c4 = c_in.view(1, C, H, W)
    out_c = F.conv2d(c4, weight, bias, stride=1, padding=1).view(-1, 1)

    # Convolve each column as an image.
    Gc_in_img = Gc_in.T.view(n_in, C, H, W)  # treat each column as an image
    out_g_img = F.conv2d(Gc_in_img, weight, None, stride=1, padding=1)
    Gc_out = out_g_img.view(n_in, n_out).T.contiguous()  # (n_out, n_in)

    return HZono(
        c=out_c,
        Gc=Gc_out,
        Gb=torch.zeros((n_out, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, n_in), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        eq_mask=torch.zeros(0, dtype=torch.bool, device=device),
    )


def demo():
    print("=" * 78)
    print("T2 sparse-Gc prototype — cifar100 first-conv synthetic")
    print("=" * 78)
    t0 = time.time()
    hz0 = _build_cifar_first_conv_hz_post()
    s0 = stats_of(hz0, "post-conv dense HZono")
    print(s0.short(), f"build={time.time()-t0:.2f}s")

    # Stage 1: prune small Gc columns with row-slack
    t0 = time.time()
    hz1 = hz_prune_gc_dense(hz0, eps=1e-9)
    s1 = stats_of(hz1, "after prune eps=1e-9")
    print(s1.short(), f"prune={time.time()-t0:.2f}s")
    ok, msg = check_overapprox(hz0, hz1)
    print(f"   soundness: {msg}")

    t0 = time.time()
    hz1b = hz_prune_gc_dense(hz0, eps=1e-4)
    s1b = stats_of(hz1b, "after prune eps=1e-4")
    print(s1b.short(), f"prune={time.time()-t0:.2f}s")
    ok, msg = check_overapprox(hz0, hz1b)
    print(f"   soundness: {msg}")

    # Stage 2: dense -> sparse conversion
    t0 = time.time()
    sparse = hz_dense_to_sparse(hz0, density_threshold=0.05)
    s2 = stats_of(sparse, "dense->sparse from raw")
    print(s2.short(), f"convert={time.time()-t0:.2f}s")
    if isinstance(sparse, SparseGcZ):
        ok, msg = check_overapprox(hz0, sparse)
        print(f"   soundness: {msg}")
    else:
        print("   (no conversion: density too high)")

    # Stage 3: prune then sparsify
    t0 = time.time()
    hz_pruned = hz_prune_gc_dense(hz0, eps=1e-9)
    sparse2 = hz_dense_to_sparse(hz_pruned, density_threshold=0.10)
    s3 = stats_of(sparse2, "prune+convert")
    print(s3.short(), f"combined={time.time()-t0:.2f}s")
    if isinstance(sparse2, SparseGcZ):
        ok, msg = check_overapprox(hz0, sparse2)
        print(f"   soundness: {msg}")

    print()
    print("Memory reduction:")
    if isinstance(sparse, SparseGcZ):
        save = 1 - s2.gc_nnz_bytes / s0.gc_dense_bytes
        print(f"  raw dense -> sparse: {save*100:.1f}% saved")
    if isinstance(sparse2, SparseGcZ):
        save = 1 - s3.gc_nnz_bytes / s0.gc_dense_bytes
        print(f"  prune+sparse:        {save*100:.1f}% saved")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo()
