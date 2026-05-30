"""Verify B3 row redundancy holds across diverse alpha/beta settings
and multi-neuron / multi-layer chains."""
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


def test_redundancy_on_hz(sp_after_relu, label):
    out = sp_after_relu
    nvars = out.ng + out.nb
    if nvars == 0:
        print(f"  {label}: empty HZ")
        return

    Ac = out.Ac_sparse.to_dense().numpy()
    Ab = out.Ab_sparse.to_dense().numpy() if out.nb > 0 else np.zeros((out.nc, 0))
    A = np.concatenate([Ac, Ab], axis=1)
    b = out.b.numpy().reshape(-1)
    bounds = [(-1, 1)] * nvars

    redundant = []
    needed = []
    failed = []
    for test_row in range(out.nc):
        others = list(range(out.nc))
        others.remove(test_row)
        A_ub = A[others] if others else None
        b_ub = b[others] if others else None

        c_obj = -A[test_row]
        rhs_test = b[test_row]
        try:
            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                              method="highs")
            if res.success:
                max_lhs = -res.fun
                if max_lhs <= rhs_test + 1e-7:
                    redundant.append(test_row)
                else:
                    needed.append(test_row)
            else:
                failed.append(test_row)
        except Exception:
            failed.append(test_row)
    n_red = len(redundant)
    n_need = len(needed)
    print(f"  {label}: {n_need} needed + {n_red} redundant of {out.nc} (= {n_red/max(out.nc,1)*100:.0f}%)")
    if n_red > 0:
        print(f"    redundant rows: {redundant}")
    return redundant


def make_hz(n, ng, c_vals, Gc_dense, dtype=torch.float64):
    """Build SparseGcZ from c (n,) and dense Gc (n, ng)."""
    c = torch.tensor(c_vals, dtype=dtype)
    Gc_d = torch.tensor(Gc_dense, dtype=dtype)
    nz = (Gc_d.abs() > 0).nonzero(as_tuple=False).T
    val = Gc_d[nz[0], nz[1]]
    Gc_sp = torch.sparse_coo_tensor(nz, val, (n, ng), dtype=dtype).coalesce()
    return SparseGcZ(c=c, Gc_sparse=Gc_sp, dtype=dtype, device=torch.device("cpu"))


def main():
    print("=== B3 row redundancy across various unstable neuron settings ===\n")

    # Test 1: single neuron, varied alpha/beta
    print("Single-neuron tests (n=1, ng=2):")
    for trial in range(5):
        torch.manual_seed(trial)
        c0 = float(torch.rand(1).item()) - 0.5  # [-0.5, 0.5]
        gc = (torch.rand(2) * 2 - 1).tolist()
        # Ensure unstable: |c| < sum(|gc|)
        s = sum(abs(g) for g in gc)
        if abs(c0) >= s * 0.9:
            c0 = c0 * 0.3  # shrink c
        sp = make_hz(1, 2, [c0], [gc])
        lb, ub = sp.bounds()
        out = apply_relu_eq_lagr_sparse(sp)
        test_redundancy_on_hz(out, f"trial{trial}: c={c0:.2f} α={lb.item():.2f} β={ub.item():.2f}")

    # Test 2: 2-neuron HZ
    print("\n2-neuron tests (n=2, ng=3):")
    for trial in range(3):
        torch.manual_seed(100 + trial)
        c_vals = [0.0, 0.05]
        Gc = (torch.rand(2, 3) * 1.0 - 0.5).tolist()
        sp = make_hz(2, 3, c_vals, Gc)
        out = apply_relu_eq_lagr_sparse(sp)
        test_redundancy_on_hz(out, f"trial{trial}: n=2 neurons")

    # Test 3: 2-layer chain
    print("\n2-layer chain (n=2, ng=3, then apply 2 ReLUs):")
    torch.manual_seed(42)
    c_vals = [0.0, 0.0]
    Gc = (torch.rand(2, 3) * 1.0 - 0.5).tolist()
    sp = make_hz(2, 3, c_vals, Gc)
    out1 = apply_relu_eq_lagr_sparse(sp)
    test_redundancy_on_hz(out1, "after layer 1")
    out2 = apply_relu_eq_lagr_sparse(out1)
    test_redundancy_on_hz(out2, "after layer 2")


if __name__ == "__main__":
    main()
