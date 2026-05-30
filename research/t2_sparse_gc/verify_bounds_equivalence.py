"""Verify that compact_rows=True gives IDENTICAL LP bounds as full,
not just containment. Since the dropped rows are LP-redundant (proven
in verify_redundancy_multi.py), their removal cannot change LP optimum.
"""
from __future__ import annotations

import sys
import numpy as np
import scipy.optimize as opt

sys.path.insert(0, "/data1/Kane/ACT")

import torch
from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.hybridz_tf.algorithms.sparse_eq_lagr import (
    apply_relu_eq_lagr_sparse,
)


def lp_bounds(hz: SparseGcZ):
    """Compute LP-optimal per-coord bounds for hz."""
    n = hz.dim
    if hz.nc == 0:
        # Abstract = LP = c ± rad
        lb, ub = hz.bounds()
        return lb.numpy(), ub.numpy()

    Ac = hz.Ac_sparse.to_dense().numpy()
    Ab = hz.Ab_sparse.to_dense().numpy() if hz.nb > 0 else np.zeros((hz.nc, 0))
    A_ub = np.concatenate([Ac, Ab], axis=1)
    b_ub = hz.b.numpy().reshape(-1)
    Gc = hz.Gc_sparse.to_dense().numpy()
    Gb = hz.Gb_sparse.to_dense().numpy() if hz.nb > 0 else np.zeros((n, 0))
    c_arr = hz.c.numpy()
    bounds_lp = [(-1, 1)] * (hz.ng + hz.nb)

    lb_out = np.zeros(n)
    ub_out = np.zeros(n)
    for i in range(n):
        obj = np.concatenate([Gc[i], Gb[i]])
        if np.linalg.norm(obj) < 1e-14:
            lb_out[i] = c_arr[i]
            ub_out[i] = c_arr[i]
            continue
        lo = opt.linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
        hi = opt.linprog(-obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
        lb_out[i] = c_arr[i] + (lo.fun if lo.success else 0)
        ub_out[i] = c_arr[i] - (hi.fun if hi.success else 0)
    return lb_out, ub_out


def make_hz(n, ng, c_vals, Gc_dense):
    dtype = torch.float64
    c = torch.tensor(c_vals, dtype=dtype)
    Gc_d = torch.tensor(Gc_dense, dtype=dtype)
    nz = (Gc_d.abs() > 0).nonzero(as_tuple=False).T
    val = Gc_d[nz[0], nz[1]]
    Gc_sp = torch.sparse_coo_tensor(nz, val, (n, ng), dtype=dtype).coalesce()
    return SparseGcZ(c=c, Gc_sparse=Gc_sp, dtype=dtype, device=torch.device("cpu"))


def main():
    print("=== Bounds-equivalence verification: compact vs full ===\n")

    max_diff_all = 0.0
    for seed in range(8):
        torch.manual_seed(seed)
        c_vals = (torch.rand(2) * 0.4 - 0.2).tolist()  # small bias for unstable
        Gc = (torch.rand(2, 3) * 1.0 - 0.5).tolist()
        sp = make_hz(2, 3, c_vals, Gc)

        full = apply_relu_eq_lagr_sparse(sp, compact_rows=False)
        compact = apply_relu_eq_lagr_sparse(sp, compact_rows=True)

        lb_full, ub_full = lp_bounds(full)
        lb_comp, ub_comp = lp_bounds(compact)

        diff_lb = np.abs(lb_full - lb_comp).max()
        diff_ub = np.abs(ub_full - ub_comp).max()
        max_diff = max(diff_lb, diff_ub)
        max_diff_all = max(max_diff_all, max_diff)

        status = "✓ IDENTICAL" if max_diff < 1e-7 else f"✗ DIFFERS by {max_diff:.4e}"
        print(f"  seed={seed}: full nc={full.nc} compact nc={compact.nc}  "
              f"max LP-bound diff: {max_diff:.4e}  {status}")

    print(f"\nMax LP-bound difference across all seeds: {max_diff_all:.4e}")
    if max_diff_all < 1e-7:
        print("✓ CONCLUSION: compact mode gives IDENTICAL LP bounds as full (zero precision loss)")
    else:
        print("✗ CONCLUSION: compact differs from full beyond numerical tolerance")


if __name__ == "__main__":
    main()
