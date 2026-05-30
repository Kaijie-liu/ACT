"""Verify which of B3's 6 rows per unstable neuron are LP-redundant
given variable box constraints xi_c ∈ [-1, 1] and z ∈ [-1, 1].

For each candidate row, we test:
    max_{others + boxes} LHS_of_row  vs  RHS_of_row

If max_LHS ≤ RHS for all (xi_old, xi_b, xi2, z) satisfying other rows
+ boxes, then this row is redundant (its constraint is already
implied).

Approach: for each row r, build the LP:
    maximize LHS_r
    subject to: all other rows + box constraints
If optimum ≤ RHS_r, row r is redundant.
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


def test_redundancy_on_single_neuron():
    """Build a tiny SparseGcZ with 1 unstable neuron, apply B3, then
    check which of the 6 resulting rows are LP-redundant."""
    dtype = torch.float64
    device = torch.device("cpu")
    n = 1
    ng = 2  # 2 input generators
    c = torch.tensor([0.1], dtype=dtype)
    Gc_d = torch.tensor([[0.5, -0.3]], dtype=dtype)
    nz = (Gc_d.abs() > 0).nonzero(as_tuple=False).T
    val = Gc_d[nz[0], nz[1]]
    Gc_sp = torch.sparse_coo_tensor(nz, val, (n, ng), dtype=dtype).coalesce()
    sp = SparseGcZ(c=c, Gc_sparse=Gc_sp, dtype=dtype, device=device)
    # Pre-act bounds: c +/- sum|Gc[0,:]| = 0.1 +/- 0.8 = [-0.7, 0.9] → unstable
    lb, ub = sp.bounds()
    print(f"Pre-act bounds: lb={lb.item()}, ub={ub.item()}")

    out = apply_relu_eq_lagr_sparse(sp)
    print(f"After B3: ng={out.ng} nb={out.nb} nc={out.nc}")
    print(f"output Ac shape={out.Ac_sparse.shape}, Ab shape={out.Ab_sparse.shape}")

    # Materialize the 6 rows as dense
    Ac = out.Ac_sparse.to_dense().numpy()
    Ab = out.Ab_sparse.to_dense().numpy() if out.nb > 0 else np.zeros((out.nc, 0))
    b = out.b.numpy().reshape(-1)
    print(f"\nFull A matrix (Ac | Ab):")
    A = np.concatenate([Ac, Ab], axis=1)
    print(f"shape: {A.shape}, vars: {[f'xi{i}' for i in range(out.ng)] + [f'z{i}' for i in range(out.nb)]}")
    for row_i in range(out.nc):
        coeffs = A[row_i]
        nz_coeffs = [(i, c) for i, c in enumerate(coeffs) if abs(c) > 1e-12]
        nz_str = ", ".join([f"{c:.4f}*v{i}" for i, c in nz_coeffs])
        print(f"  row {row_i}: {nz_str} ≤ {b[row_i]:.4f}")

    # For each row, test if it's redundant
    nvars = out.ng + out.nb
    # Variable bounds: all in [-1, 1]
    bounds = [(-1, 1)] * nvars

    print(f"\n=== LP-redundancy test for each row ===")
    for test_row in range(out.nc):
        # Maximize LHS_test_row subject to all OTHER rows + boxes
        other_rows = list(range(out.nc))
        other_rows.remove(test_row)
        if other_rows:
            A_ub = A[other_rows]
            b_ub = b[other_rows]
        else:
            A_ub = None
            b_ub = None

        c_obj = -A[test_row]  # minimize -LHS = maximize LHS
        rhs_test = b[test_row]

        try:
            res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                              method="highs")
            if res.success:
                max_lhs = -res.fun
                if max_lhs <= rhs_test + 1e-9:
                    status = "REDUNDANT"
                else:
                    status = "NEEDED"
                print(f"  row {test_row}: max(LHS)={max_lhs:.6f} vs RHS={rhs_test:.6f}  → {status}")
            else:
                print(f"  row {test_row}: LP solver failed ({res.message})")
        except Exception as e:
            print(f"  row {test_row}: error {e}")


if __name__ == "__main__":
    test_redundancy_on_single_neuron()
