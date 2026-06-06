"""Phase F1: Constrained HZ-LP for forward-only tighter LP UB.

Per advisor 2026-06-05 Phase F directive: the current HZ closed-form LP UB
treats every generator's xi as fully independent in [-1, 1]. For ReLU triangle
slack generators (one per unstable neuron), this is LOOSER than the truth
because the slack xi_aux is IMPLICITLY constrained by the underlying z_i.

The Gate 2 v2 diagnostic on cifar iid 113 showed:
  PHANTOM rival contribution: +1.526 (all from slack), tail=0
  17% slack reduction would flip to CERT

This module implements a CONTINUOUS LP that puts the ReLU triangle
constraints back in EXPLICITLY for the FINAL N layers, where the precision
matters most. The LP is:

  Variables:
    xi_root[k]  ∈ [-1, +1]   (n_root input-coord generators)
    xi_aux[i]   ∈ [-1, +1]   (per "previously committed" slack — earlier layers)
    z_i         ∈ [l_i, u_i] (pre-activation for tracked unstable neuron i)
    y_i         ∈ [0, u_i]   (post-activation, with triangle bounds)

  Constraints:
    z_i = c_z_i + G_z_i @ xi_root + (G_z_i_aux @ xi_aux)   (linearity)
    y_i >= 0
    y_i >= z_i
    y_i <= lambda_i * (z_i - l_i)        (triangle upper chord)
    where lambda_i = u_i / (u_i - l_i)

  Objective:
    max sum_j d_out_eff[j] * y_j + d_out · b_remaining

  d_out_eff = W_remaining^T @ d_out      (effective coeffs on final ReLU's y)

PRINCIPLE compliance:
  - Forward-only: no backward bound refinement
  - Continuous LP: HiGHS/scipy.optimize.linprog; no MILP, no integers
  - No backward gradient: the LP coefficients come from forward propagation
  - No BaB, no random sampling

For prototype, this module provides:
  - `LastReluRecord`: per-neuron pre-activation affine form + bounds
  - `constrained_lp_ub(state_pre_relu, relu_records, W_remaining, b_remaining, d_out)`:
    solve the LP and return UB
  - `closed_form_hz_lp_ub`: reference (current closed-form, looser)
  - Verification utilities: brute force sample to verify sound LP UB
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize as sopt

from research.sc_hz.prune import PrunedState


@dataclass
class LastReluRecord:
    """Per-neuron data captured at the LAST ReLU layer.

    For each unstable neuron i in the last ReLU layer:
      z_i = c_z[i] + G_z[i, :] @ xi_root + tail_z[i] * xi_tail_z[i]
      l[i] <= z_i <= u[i]   (pre-activation bounds)
      y_i = ReLU(z_i)
    """
    c_z: np.ndarray            # (n_pre,)   pre-act center
    G_z: np.ndarray            # (n_pre, K) pre-act generator matrix (from xi_root + earlier aux)
    tail_z: Optional[np.ndarray]  # (n_pre,)  per-coord interval tail
    l: np.ndarray              # (n_pre,)   pre-act lower bound
    u: np.ndarray              # (n_pre,)   pre-act upper bound

    @property
    def n_pre(self) -> int:
        return self.c_z.shape[0]

    @property
    def n_gen(self) -> int:
        return self.G_z.shape[1]

    def unstable_mask(self) -> np.ndarray:
        return (self.l < 0) & (self.u > 0)

    def stable_active_mask(self) -> np.ndarray:
        return self.l >= 0

    def stable_inactive_mask(self) -> np.ndarray:
        return self.u <= 0


def closed_form_hz_lp_ub(
    relu_rec: LastReluRecord, W_remaining: np.ndarray,
    b_remaining: np.ndarray, d_out: np.ndarray,
) -> float:
    """Reference: the CURRENT closed-form LP UB (loose).

    Treats the post-ReLU y as an HZ where:
      y_active = z (= c_z + G_z xi + tail*xi_tail)
      y_inactive = 0
      y_unstable = lambda*z + mu*(1 + xi_aux)/2  with xi_aux ∈ [-1, 1]

    The closed form sums independent ξ contributions.
    """
    is_active = relu_rec.stable_active_mask()
    is_inactive = relu_rec.stable_inactive_mask()
    is_unstable = relu_rec.unstable_mask()

    # Per-neuron: y_i = lam_i * z_i + mu_i for unstable, with aux ∈ [-mu, +mu]
    den = np.where(is_unstable, relu_rec.u - relu_rec.l, 1.0)
    lam = np.where(is_unstable, relu_rec.u / np.maximum(den, 1e-300), 0.0)
    lam = np.where(is_active, 1.0, lam)
    mu = np.where(is_unstable, -relu_rec.l * relu_rec.u / (2.0 * np.maximum(den, 1e-300)), 0.0)

    # Center
    new_c = lam * relu_rec.c_z + mu
    new_c = np.where(is_inactive, 0.0, new_c)

    # Generator and tail (scaled by lam, plus aux contributes to inactive
    # rows as 0)
    new_G = relu_rec.G_z * lam[:, None]
    new_G = np.where(is_inactive[:, None], 0.0, new_G)
    new_tail = (lam * relu_rec.tail_z if relu_rec.tail_z is not None else None)
    if new_tail is not None:
        new_tail = np.where(is_inactive, 0.0, new_tail)
    # Aux contributions (per unstable neuron, magnitude mu)
    aux_mu = np.where(is_unstable, mu, 0.0)

    # Now apply W_remaining and b_remaining to compute d_out·y_out
    # y_out = W_remaining @ y + b_remaining
    # d_out·y_out = d_out·W_remaining @ y + d_out·b_remaining = d_eff @ y + const
    d_eff = W_remaining.T @ d_out  # (n_pre,)
    const = float(d_out @ b_remaining)
    ub = const + float(d_eff @ new_c) + float(np.abs(d_eff @ new_G).sum())
    if new_tail is not None:
        ub += float(np.abs(d_eff) @ new_tail)
    # Aux: each unstable neuron's aux ∈ [-mu_i, +mu_i] contributes
    # independently to d_eff·y_unstable[i] = ... + aux_i (in y-space).
    # So d_eff·y picks up sum |d_eff[i]| * |aux_mu[i]|
    ub += float(np.abs(d_eff) @ aux_mu)
    return ub


def constrained_lp_ub(
    relu_rec: LastReluRecord, W_remaining: np.ndarray,
    b_remaining: np.ndarray, d_out: np.ndarray,
    return_solution: bool = False,
) -> Tuple[float, Optional[dict]]:
    """Solve the continuous LP with ReLU triangle constraints explicitly.

    Variables (in order):
      xi[0..K-1]      ∈ [-1, +1]  (root + earlier-layer aux generators)
      xi_tail[0..n-1] ∈ [-1, +1]  (tail per coord, n=n_pre)
      y[0..n-1]                    (post-activation; only unstable have non-trivial bounds)

    For inactive: y_i = 0 (eliminate variable)
    For active:   y_i = z_i = c_z[i] + G_z[i,:]@xi + tail_z[i]*xi_tail[i]
    For unstable: y_i >= 0, y_i >= z_i, y_i <= lam_i*(z_i - l_i)

    Objective: max d_eff @ y + const
    """
    K = relu_rec.n_gen
    n = relu_rec.n_pre
    has_tail = relu_rec.tail_z is not None

    is_active = relu_rec.stable_active_mask()
    is_inactive = relu_rec.stable_inactive_mask()
    is_unstable = relu_rec.unstable_mask()

    den = np.where(is_unstable, relu_rec.u - relu_rec.l, 1.0)
    lam = np.where(is_unstable, relu_rec.u / np.maximum(den, 1e-300), 0.0)

    d_eff = W_remaining.T @ d_out  # (n,)
    const = float(d_out @ b_remaining)

    # Variable layout
    # x = [xi (K), xi_tail (n if has_tail), y (n)]
    n_xitail = n if has_tail else 0
    n_y = n
    n_vars = K + n_xitail + n_y

    # We will use scipy.optimize.linprog with c = -d_eff @ y to MAXIMIZE
    c = np.zeros(n_vars, dtype=np.float64)
    c[K + n_xitail:K + n_xitail + n_y] = -d_eff   # minimize -d_eff·y

    # Bounds
    bounds = [(-1.0, 1.0)] * K
    if has_tail:
        bounds += [(-1.0, 1.0)] * n_xitail
    # y bounds: inactive y = 0; active free (will be tied via eq); unstable in [0, u_i]
    for i in range(n_y):
        if is_inactive[i]:
            bounds.append((0.0, 0.0))
        elif is_active[i]:
            bounds.append((max(0.0, relu_rec.l[i]), max(0.0, relu_rec.u[i])))
        else:  # unstable
            bounds.append((0.0, max(0.0, relu_rec.u[i])))

    # Equality constraints for active rows: y_i = z_i = c_z[i] + G_z[i,:]@xi + tail_z[i]*xi_tail[i]
    A_eq_rows = []
    b_eq_rows = []
    for i in np.where(is_active)[0]:
        row = np.zeros(n_vars)
        row[0:K] = -relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = -relu_rec.tail_z[i]
        row[K + n_xitail + i] = 1.0
        # y_i - G[i]·xi - tail_i·xi_tail_i = c_z[i]
        A_eq_rows.append(row)
        b_eq_rows.append(relu_rec.c_z[i])
    # Also for inactive: y_i = 0 already by bounds; no eq needed.

    # Inequality constraints (in linprog form A_ub x <= b_ub)
    A_ub_rows = []
    b_ub_rows = []
    # For unstable: y_i >= 0  → -y_i <= 0 (already bound y_i >= 0)
    # For unstable: y_i >= z_i  → -y_i + z_i <= 0
    #   -y_i + c_z[i] + G[i]·xi + tail_i*xi_tail_i <= 0
    #   G[i]·xi + tail_i*xi_tail_i - y_i <= -c_z[i]
    for i in np.where(is_unstable)[0]:
        row = np.zeros(n_vars)
        row[0:K] = relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = relu_rec.tail_z[i]
        row[K + n_xitail + i] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(-relu_rec.c_z[i])

        # y_i <= lam_i * (z_i - l_i)  →  y_i - lam_i z_i <= -lam_i l_i
        #   y_i - lam_i(c_z[i] + G[i]·xi + tail_i*xi_tail_i) <= -lam_i*l_i
        #   -lam_i G[i]·xi - lam_i tail_i*xi_tail_i + y_i <= -lam_i*l_i + lam_i*c_z[i]
        row = np.zeros(n_vars)
        row[0:K] = -lam[i] * relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = -lam[i] * relu_rec.tail_z[i]
        row[K + n_xitail + i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(lam[i] * (relu_rec.c_z[i] - relu_rec.l[i]))

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None

    res = sopt.linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if res.status != 0:
        # Infeasible/unbounded — return +inf as conservative UB
        return float("inf"), {"status": res.status, "message": res.message}

    ub = const - float(res.fun)
    sol = None
    if return_solution:
        x = res.x
        sol = {
            "xi_root": x[0:K], "xi_tail": x[K:K+n_xitail] if has_tail else None,
            "y": x[K+n_xitail:K+n_xitail+n_y],
            "status": res.status, "message": res.message,
        }
    return float(ub), sol


def brute_force_max_d_out_y(
    relu_rec: LastReluRecord, W_remaining: np.ndarray,
    b_remaining: np.ndarray, d_out: np.ndarray,
    n_samples: int = 1000, seed: int = 20260605,
) -> float:
    """Random sample xi values; compute max d_out·y_out. For soundness test."""
    rng = np.random.default_rng(seed)
    is_active = relu_rec.stable_active_mask()
    is_inactive = relu_rec.stable_inactive_mask()
    is_unstable = relu_rec.unstable_mask()
    K = relu_rec.n_gen
    n = relu_rec.n_pre
    has_tail = relu_rec.tail_z is not None
    max_val = -np.inf
    for _ in range(n_samples):
        xi = rng.uniform(-1, 1, size=K)
        xi_tail = rng.uniform(-1, 1, size=n) if has_tail else np.zeros(n)
        z = (relu_rec.c_z + relu_rec.G_z @ xi
             + (relu_rec.tail_z * xi_tail if has_tail else 0))
        y = np.where(z >= 0, z, 0.0)
        y_out = W_remaining @ y + b_remaining
        val = float(d_out @ y_out)
        if val > max_val:
            max_val = val
    return max_val
