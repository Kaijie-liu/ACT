"""Bounded-binary MILP exact ReLU encoding.

P3 relaxed (per user 2026-06-06): MILP with binary indicators allowed,
constrained to OPEN-SOURCE solvers only (HiGHS via scipy.optimize.milp).
NO Gurobi.

Mechanism: Tjeng et al. 2017 / Tjeng-Tedrake exact ReLU encoding:
  For each unstable neuron i with z_i ∈ [l_i, u_i] (l_i < 0 < u_i):
    Variables: z_i (continuous), y_i (continuous), b_i ∈ {0, 1}
    Constraints:
      y_i ≥ 0
      y_i ≥ z_i
      y_i ≤ u_i · b_i           (if b_i=0, y_i ≤ 0 → y_i = 0)
      y_i ≤ z_i - l_i·(1 - b_i)  (if b_i=1, y_i ≤ z_i → y_i = z_i)
    Interpretation: b_i = 1 means ReLU "active" (y = z), b_i = 0 means "inactive" (y = 0)

For top-K MOST CRITICAL unstable neurons by |d_eff|, use binary indicators.
For remaining unstable, use standard triangle relaxation.

This still respects:
  P1 forward only: yes, the MILP is built on forward walker output
  P2 no gradient: yes, MILP is combinatorial not gradient-based
  P3 RELAXED: MILP with K_max binary budget (was: continuous LP only)
  P4 no input split / no BaB on input: MILP B&B is on ACTIVATION, not input box
  P5 no random / no PGD: yes, MILP is deterministic
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/data1/Kane/ACT')
import numpy as np
import scipy.optimize as sopt
from scipy.optimize import milp, LinearConstraint, Bounds


def milp_last_relu_ub(last_rec, W_rem, b_rem, d_out, K_max=20):
    """Bounded-binary MILP UB on d_out · (W_rem @ y_relu + b_rem).

    For top-K unstable neurons (by |W_rem.T @ d_out|), use exact ReLU
    via binary indicators. Remaining unstable use triangle relaxation.

    Returns: (ub, info) where info = {'n_binaries', 'top_k_neuron_indices', 'milp_status'}
    """
    n_pre = last_rec.n_pre
    K = last_rec.G_z.shape[1]
    has_tail = last_rec.tail_z is not None
    n_xitail = n_pre if has_tail else 0

    is_active = last_rec.stable_active_mask()
    is_inactive = last_rec.stable_inactive_mask()
    is_unstable = last_rec.unstable_mask()
    unstable_idx = np.where(is_unstable)[0]
    n_unstable = len(unstable_idx)

    # Effective per-neuron cost for picking top-K
    d_eff = W_rem.T @ d_out  # (n_pre,)
    if n_unstable > K_max:
        # Pick top K_max by |d_eff|
        order = np.argsort(-np.abs(d_eff[unstable_idx]))[:K_max]
        binary_idx = unstable_idx[order]
    else:
        binary_idx = unstable_idx
    binary_set = set(int(i) for i in binary_idx)
    n_b = len(binary_idx)

    # Variable layout:
    #   [0:K]                              xi      (continuous [-1, 1])
    #   [K:K+n_xitail]                      xi_tail (continuous [-1, 1])
    #   [K+n_xitail:K+n_xitail+n_pre]        y      (continuous, per-neuron bounded)
    #   [K+n_xitail+n_pre:K+n_xitail+n_pre+n_b]  b   (binary {0, 1})
    n_vars = K + n_xitail + n_pre + n_b
    y_offset = K + n_xitail
    b_offset = K + n_xitail + n_pre

    # Bounds per variable
    lb_var = np.zeros(n_vars)
    ub_var = np.zeros(n_vars)
    lb_var[:K] = -1.0; ub_var[:K] = 1.0
    if has_tail:
        lb_var[K:K + n_xitail] = -1.0; ub_var[K:K + n_xitail] = 1.0
    for i in range(n_pre):
        if is_inactive[i]:
            lb_var[y_offset + i] = 0.0; ub_var[y_offset + i] = 0.0
        elif is_active[i]:
            lo = max(0.0, float(last_rec.l[i]))
            hi = max(0.0, float(last_rec.u[i]))
            lb_var[y_offset + i] = lo; ub_var[y_offset + i] = hi
        else:  # unstable
            lb_var[y_offset + i] = 0.0
            ub_var[y_offset + i] = max(0.0, float(last_rec.u[i]))
    # Binaries
    lb_var[b_offset:] = 0.0; ub_var[b_offset:] = 1.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[b_offset:] = 1  # binaries

    # Objective: max d_out · (W_rem @ y + b_rem)
    #          = (W_rem.T @ d_out) @ y + d_out·b_rem
    # MILP minimizes, so c = -(d_eff_on_y) extended
    c_obj = np.zeros(n_vars)
    c_obj[y_offset:y_offset + n_pre] = -d_eff
    obj_const = float(d_out @ b_rem)

    # Constraints
    A_rows = []
    lb_con = []
    ub_con = []

    # Helper: z_i = c_z[i] + G_z[i,:] @ xi + tail_z[i] * xi_tail[i] (if tail)
    for i in range(n_pre):
        if is_inactive[i]:
            continue  # y_i = 0 via bounds, z_i not constrained
        if is_active[i]:
            # y_i = z_i = c_z + G_z·xi + tail_z·xi_tail
            row = np.zeros(n_vars)
            row[:K] = -last_rec.G_z[i, :]
            if has_tail:
                row[K + i] = -last_rec.tail_z[i]
            row[y_offset + i] = 1.0
            A_rows.append(row)
            lb_con.append(float(last_rec.c_z[i]))
            ub_con.append(float(last_rec.c_z[i]))
            continue
        # Unstable case
        if i in binary_set:
            k_b = int(np.where(binary_idx == i)[0][0])
            l_i = float(last_rec.l[i])
            u_i = float(last_rec.u[i])
            # Constraint 1: y_i ≥ z_i  →  z_i - y_i ≤ 0
            #               c_z[i] + G_z·xi + tail·xitail - y_i ≤ 0
            row = np.zeros(n_vars)
            row[:K] = last_rec.G_z[i, :]
            if has_tail:
                row[K + i] = last_rec.tail_z[i]
            row[y_offset + i] = -1.0
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(-float(last_rec.c_z[i]))
            # Constraint 2: y_i ≤ u_i · b_i  →  y_i - u_i·b_i ≤ 0
            row = np.zeros(n_vars)
            row[y_offset + i] = 1.0
            row[b_offset + k_b] = -u_i
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(0.0)
            # Constraint 3: y_i ≤ z_i - l_i·(1 - b_i)
            #             → y_i - z_i - l_i·b_i ≤ -l_i
            #             → -G_z·xi - tail·xitail + y_i - l_i·b_i ≤ -l_i + c_z[i]·(-1) ... wait
            # y_i ≤ z_i - l_i + l_i·b_i
            # y_i - z_i - l_i·b_i ≤ -l_i
            # z_i = c_z[i] + G_z·xi + tail·xitail
            # y_i - c_z[i] - G_z·xi - tail·xitail - l_i·b_i ≤ -l_i
            # - G_z·xi - tail·xitail + y_i - l_i·b_i ≤ -l_i + c_z[i]
            row = np.zeros(n_vars)
            row[:K] = -last_rec.G_z[i, :]
            if has_tail:
                row[K + i] = -last_rec.tail_z[i]
            row[y_offset + i] = 1.0
            row[b_offset + k_b] = -l_i
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(-l_i + float(last_rec.c_z[i]))
        else:
            # Triangle relaxation (no binary)
            l_i = float(last_rec.l[i])
            u_i = float(last_rec.u[i])
            # y_i ≥ z_i  (same as above)
            row = np.zeros(n_vars)
            row[:K] = last_rec.G_z[i, :]
            if has_tail:
                row[K + i] = last_rec.tail_z[i]
            row[y_offset + i] = -1.0
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(-float(last_rec.c_z[i]))
            # y_i ≤ lam_i·(z_i - l_i) = lam·z - lam·l = lam·c_z + lam·G_z·xi + lam·tail·xitail - lam·l
            if u_i - l_i > 1e-300:
                lam = u_i / (u_i - l_i)
                row = np.zeros(n_vars)
                row[:K] = -lam * last_rec.G_z[i, :]
                if has_tail:
                    row[K + i] = -lam * last_rec.tail_z[i]
                row[y_offset + i] = 1.0
                A_rows.append(row)
                lb_con.append(-np.inf)
                ub_con.append(lam * (float(last_rec.c_z[i]) - l_i))

    A_arr = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    lb_arr = np.array(lb_con) if lb_con else np.zeros(0)
    ub_arr = np.array(ub_con) if ub_con else np.zeros(0)

    constraints = LinearConstraint(A_arr, lb=lb_arr, ub=ub_arr)
    res = milp(
        c=c_obj,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=lb_var, ub=ub_var),
        options={'time_limit': 60, 'mip_rel_gap': 1e-9},
    )
    info = {
        'n_binaries': n_b,
        'n_unstable_total': n_unstable,
        'K_max': K_max,
        'top_k_neuron_indices': binary_idx.tolist(),
        'milp_status': res.status,
        'milp_message': res.message,
    }
    if res.status != 0:
        return None, info
    ub_val = obj_const - float(res.fun)
    return ub_val, info


def main():
    """Self-test: validate MILP matches SETPH on synthetic LastReluRecord."""
    print("=== milp_relu self-test ===\n")
    # Use the existing constrained_lp_ub for triangle baseline
    from research.sc_hz.constrained_lp import constrained_lp_ub
    from research.canonical_provenance import load_instance
    from research.sc_hz.vnnlib_parse import parse_vnnlib
    from research.sc_hz.constrained_lp_integration import forward_resnet_capture
    import onnx
    import signal

    def _to(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _to)

    # Test on cgan iid 3 (known SETPH CERT)
    iid = 3
    print(f"Test: cgan_2023 iid {iid} (known SETPH CERT @ top_k=12)")
    signal.alarm(15)
    onnx_p, vnn_p = load_instance('cgan_2023', iid)
    m = onnx.load(str(onnx_p))
    init_names = {x.name for x in m.graph.initializer}
    din = [x for x in m.graph.input if x.name not in init_names][0]
    dims = [d.dim_value if d.dim_value > 0 else 1
            for d in din.type.tensor_type.shape.dim]
    n_in = int(np.prod(dims[1:])) if dims[0] in (0, 1) else int(np.prod(dims))
    od = [d.dim_value if d.dim_value > 0 else 1
          for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_cls = int(np.prod(od[1:])) if len(od) > 1 else od[0]
    lb, ub, unsafe = parse_vnnlib(str(vnn_p), n_in, n_cls)
    r = forward_resnet_capture(str(onnx_p), lb, ub, K_per_layer=100000)
    signal.alarm(0)

    print(f"  n_unstable = {int(r.last_relu_record.unstable_mask().sum())}")
    for d, t, _ in unsafe:
        f1_val, _ = constrained_lp_ub(r.last_relu_record, r.W_remaining, r.b_remaining, d)
        f1_excess = float(f1_val) - float(t)
        milp_ub, info = milp_last_relu_ub(r.last_relu_record, r.W_remaining,
                                              r.b_remaining, d, K_max=20)
        if milp_ub is None:
            print(f"  rival: F1={f1_excess:+.4e}, MILP FAILED ({info['milp_message']})")
        else:
            milp_excess = milp_ub - float(t)
            improv = (f1_excess - milp_excess) / max(abs(f1_excess), 1e-9) * 100
            print(f"  rival: F1={f1_excess:+.4e}, MILP={milp_excess:+.4e} ({improv:+.1f}% over F1), "
                  f"n_b={info['n_binaries']}")
    print(f"\nMILP self-test complete. If MILP excess strictly < F1 excess, the encoding works.")


if __name__ == "__main__":
    main()
