"""Week 1.1 diagnostic — measure B3 cumulative Ac rank vs row count
to decide whether cross-layer redundancy elimination is worth pursuing.

Approach: instrument apply_relu_eq_lagr_sparse to record Ac after each
layer, then run a representative cifar100 forward chain (input → 7
ReLUs) and measure:

  - row_count(ℓ)  = number of inequality rows after layer ℓ
  - rank(Ac, ℓ)   = numerical rank of Ac after layer ℓ
  - rows_per_layer added
  - nnz(Ac)       = sparse storage

If rank << row_count: cross-layer redundancy is real, pursue PEE-style
elimination. If rank ≈ row_count: rows are essentially independent,
no elimination possible; pivot to row-count-reduction (sparse bigM).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.hybridz_tf.algorithms.sparse_eq_lagr import (
    apply_relu_eq_lagr_sparse,
)


def _build_input_hz(C=3, H=32, W=32, eps=1.0/255.0,
                     dtype=torch.float64, device="cpu") -> SparseGcZ:
    n = C * H * W
    c = torch.zeros(n, dtype=dtype, device=device)
    # Diagonal sparse Gc with eps perturbation per pixel
    rows = torch.arange(n, dtype=torch.long, device=device)
    cols = torch.arange(n, dtype=torch.long, device=device)
    vals = torch.full((n,), eps, dtype=dtype, device=device)
    ind = torch.stack([rows, cols])
    Gc_sp = torch.sparse_coo_tensor(ind, vals, (n, n), dtype=dtype, device=device).coalesce()
    return SparseGcZ(c=c, Gc_sparse=Gc_sp, dtype=dtype, device=device)


def _apply_sparse_conv(hz: SparseGcZ, weight: torch.Tensor, bias: torch.Tensor,
                       in_shape, stride=1, pad=1) -> SparseGcZ:
    """Apply conv2d to SparseGcZ. Reuses internal apply_conv."""
    return hz.apply_conv(weight, bias, in_shape, stride, pad)


def _measure_rank(Ac_sparse: torch.Tensor, tol: float = 1e-9) -> int:
    """Numerical rank of sparse Ac via dense SVD on small matrices,
    or sparse iterative singular-value computation on large ones.

    For diagnostic purposes, densify if rows*cols < 1e8, else estimate
    via column-sample SVD.
    """
    nc, ng = Ac_sparse.shape
    if nc == 0 or ng == 0:
        return 0
    if nc * ng < 1_000_000:
        # Dense rank via SVD on small Ac
        Ac_d = Ac_sparse.to_dense().double().cpu().numpy()
        sv = np.linalg.svd(Ac_d, compute_uv=False)
        return int((sv > tol * sv[0]).sum()) if sv.size > 0 else 0
    elif nc * ng < 100_000_000:
        # Densify but use CPU only
        Ac_d = Ac_sparse.to_dense().double().cpu().numpy()
        sv = np.linalg.svd(Ac_d, compute_uv=False)
        return int((sv > tol * sv[0]).sum()) if sv.size > 0 else 0
    else:
        # Too big to densify; return -1 (unknown)
        return -1


def diagnose_layers(L=7, k_target=2000, C=3, H=32, W=32, eps=1.0/255.0,
                     dtype=torch.float64, device="cpu"):
    """Build a synthetic chain of L conv → ReLU layers mimicking cifar100
    resnet_large early structure. Record Ac stats after each ReLU."""
    print(f"=== B3 Ac rank scan ===")
    print(f"L={L} layers, k_target={k_target} per layer, init C={C} H={H} W={W}, eps={eps}")
    print()

    # Build input HZ
    hz = _build_input_hz(C=C, H=H, W=W, eps=eps, dtype=dtype, device=device)
    print(f"input: dim={hz.dim} ng={hz.ng}")

    cur_shape = (C, H, W)

    for layer_idx in range(1, L + 1):
        # Generate a random conv kernel (deterministic seed)
        g = torch.Generator(device=device).manual_seed(layer_idx)
        Cin = cur_shape[0]
        Cout = min(64, Cin * 2)  # Modest growth, matches resnet_large early
        weight = (torch.randn(Cout, Cin, 3, 3, generator=g, dtype=dtype, device=device)
                  * 0.1)
        bias = torch.zeros(Cout, dtype=dtype, device=device)

        t0 = time.time()
        hz_post_conv = _apply_sparse_conv(hz, weight, bias, cur_shape, stride=1, pad=1)
        conv_time = time.time() - t0
        new_shape = (Cout, cur_shape[1], cur_shape[2])

        # Apply B3 ReLU
        t0 = time.time()
        hz = apply_relu_eq_lagr_sparse(hz_post_conv)
        relu_time = time.time() - t0

        # Stats
        nc = hz.nc
        ng = hz.ng
        nb = hz.nb
        Ac_nnz = hz.Ac_sparse._nnz()
        Ab_nnz = hz.Ab_sparse._nnz()
        Gc_nnz = hz.Gc_sparse._nnz()
        Gb_nnz = hz.Gb_sparse._nnz()

        # Measure rank if affordable
        t0 = time.time()
        rank = _measure_rank(hz.Ac_sparse) if nc < 50000 else -1
        rank_time = time.time() - t0

        print(f"L{layer_idx}: shape={new_shape} dim={hz.dim} ng={ng} nb={nb} nc={nc} "
              f"Ac_nnz={Ac_nnz} ({Ac_nnz/max(nc*ng,1)*100:.3f}%) "
              f"Ab_nnz={Ab_nnz} Gc_nnz={Gc_nnz}")
        if rank >= 0:
            print(f"     rank(Ac) = {rank} / row_count = {nc} → ratio={rank/max(nc,1):.3f}")
        else:
            print(f"     rank(Ac) not measured (too large)")
        print(f"     conv={conv_time:.2f}s relu={relu_time:.2f}s rank={rank_time:.2f}s")

        # Memory check
        Ac_bytes = Ac_nnz * 16  # 8 idx + 8 val for float64
        Gc_bytes = Gc_nnz * 16
        print(f"     est memory: Ac={Ac_bytes/2**20:.1f} MiB Gc={Gc_bytes/2**20:.1f} MiB")

        cur_shape = new_shape


if __name__ == "__main__":
    # Mimic cifar100 resnet_large early-layer dimensions roughly
    diagnose_layers(L=4, C=3, H=32, W=32, eps=1.0/255.0)
