"""Smaller-scale rank scan — use tiny dimensions where dense SVD fits.
Goal: measure rank/row ratio to decide if cross-layer redundancy is real.

Sweep: input dim 8 × 8 × 1 = 64, 3 layers, k ~ small.
Should allow dense SVD up to ~10k × 10k.
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


def _build_input_hz(C=1, H=8, W=8, eps=1.0/255.0,
                     dtype=torch.float64, device="cpu") -> SparseGcZ:
    n = C * H * W
    c = torch.zeros(n, dtype=dtype, device=device)
    rows = torch.arange(n, dtype=torch.long, device=device)
    cols = torch.arange(n, dtype=torch.long, device=device)
    vals = torch.full((n,), eps, dtype=dtype, device=device)
    Gc_sp = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n),
                                     dtype=dtype, device=device).coalesce()
    return SparseGcZ(c=c, Gc_sparse=Gc_sp, dtype=dtype, device=device)


def _apply_sparse_conv(hz, weight, bias, in_shape, stride=1, pad=1):
    return hz.apply_conv(weight, bias, in_shape, stride, pad)


def diagnose(L=3, init_shape=(1, 8, 8), Cout_init=4, eps=1.0/255.0):
    dtype = torch.float64
    device = "cpu"
    hz = _build_input_hz(C=init_shape[0], H=init_shape[1], W=init_shape[2],
                          eps=eps, dtype=dtype, device=device)
    cur_shape = init_shape

    print(f"=== Small-scale B3 Ac rank scan ===")
    print(f"L={L} layers, init shape={init_shape}")
    print(f"input: dim={hz.dim} ng={hz.ng}")
    print()

    for layer_idx in range(1, L + 1):
        g = torch.Generator(device=device).manual_seed(layer_idx)
        Cin = cur_shape[0]
        Cout = Cout_init * layer_idx if layer_idx <= 2 else Cout_init * 2
        weight = (torch.randn(Cout, Cin, 3, 3, generator=g, dtype=dtype, device=device)
                  * 0.3)
        bias = torch.zeros(Cout, dtype=dtype, device=device)

        hz = _apply_sparse_conv(hz, weight, bias, cur_shape, stride=1, pad=1)
        cur_shape = (Cout, cur_shape[1], cur_shape[2])
        hz = apply_relu_eq_lagr_sparse(hz)

        nc = hz.nc
        ng = hz.ng
        Ac_nnz = hz.Ac_sparse._nnz()

        # Dense SVD on Ac
        print(f"L{layer_idx}: shape={cur_shape} dim={hz.dim} ng={ng} nc={nc} Ac_nnz={Ac_nnz}")
        if nc * ng < 10_000_000:
            Ac_d = hz.Ac_sparse.to_dense().double().cpu().numpy()
            t0 = time.time()
            sv = np.linalg.svd(Ac_d, compute_uv=False)
            svd_t = time.time() - t0
            tol = 1e-9 * sv[0] if sv.size else 1e-9
            rank = int((sv > tol).sum())
            n_near_zero = int((sv < tol * 100).sum())
            print(f"     rank(Ac) = {rank} / row_count = {nc} → ratio = {rank/max(nc,1):.4f}")
            print(f"     near-zero singular values (≤100×tol): {n_near_zero}")
            print(f"     largest sv: {sv[0]:.4f}, smallest: {sv[-1]:.6e}")
            print(f"     SVD wall: {svd_t:.2f}s")
            print()
        else:
            print(f"     Ac too large for dense SVD ({nc}×{ng})")


if __name__ == "__main__":
    diagnose(L=3, init_shape=(1, 8, 8), Cout_init=4)
