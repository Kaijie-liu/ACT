#===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and basic arithmetic operations.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds, Fact, Layer, ConSet

if TYPE_CHECKING:
    from act.back_end.hybridz_tf.hybridz_tf import HZono

# Lazy import to avoid circular dependency (hybridz_tf.py imports tf_mlp.py)
_HZono = None

def _get_HZono():
    global _HZono
    if _HZono is None:
        from act.back_end.hybridz_tf.hybridz_tf import HZono as _cls
        _HZono = _cls
    return _HZono

# Optional solver imports
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

try:
    import numpy as np
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# Hybrid Zonotope helper functions
# ============================================================================

def _hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    """Linear map: c'=R@c, Gc'=R@Gc, Gb'=R@Gb, constraints unchanged."""
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return _get_HZono()(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )


def _hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    """Translate center: c'=c+v, generators and constraints unchanged."""
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return _get_HZono()(
        c=hz.c + v,
        Gc=hz.Gc.clone(),
        Gb=hz.Gb.clone(),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )


def _hz_scale_elem(hz: HZono, s: torch.Tensor) -> HZono:
    """Element-wise scale: equivalent to _hz_multiply(hz, diag(s))."""
    s = s.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    S = torch.diag(s)
    return _hz_multiply(hz, S)


# ---- Bounds computation ----------------------------------------------------

def _hz_is_unconstrained(hz: HZono) -> bool:
    """Check if Ac, Ab, b are all near-zero (no active constraints)."""
    tol = 1e-12
    return (
        torch.all(torch.abs(hz.Ac) < tol).item()
        and torch.all(torch.abs(hz.Ab) < tol).item()
        and torch.all(torch.abs(hz.b) < tol).item()
    )


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
    """Fast path: lb = c - |Gc|_rowsum - |Gb|_rowsum."""
    n = hz.c.shape[0]
    dtype, device = hz.c.dtype, hz.c.device
    absGc = hz.Gc.abs().sum(dim=1, keepdim=True) if hz.Gc.numel() else torch.zeros((n, 1), dtype=dtype, device=device)
    absGb = hz.Gb.abs().sum(dim=1, keepdim=True) if hz.Gb.numel() else torch.zeros((n, 1), dtype=dtype, device=device)
    rad = absGc + absGb
    lb = (hz.c - rad).flatten()
    ub = (hz.c + rad).flatten()
    return Bounds(lb=lb, ub=ub)


def _hz_compute_bounds(hz: HZono) -> Bounds:
    """Compute bounds: unconstrained fast path -> Gurobi -> SciPy -> unconstrained fallback."""
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    if _HAS_GUROBI:
        try:
            return _hz_compute_bounds_gurobi(hz)
        except Exception:
            pass
    if _HAS_SCIPY:
        try:
            return _hz_compute_bounds_scipy(hz)
        except Exception:
            pass
    # Final fallback: unconstrained over-approximation
    return _hz_bounds_unconstrained(hz)


def _hz_compute_bounds_gurobi(hz: HZono) -> Bounds:
    """Gurobi MILP bounds computation."""
    c_np = hz.c.detach().cpu().numpy().astype('float64').reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype('float64')
    Gb_np = hz.Gb.detach().cpu().numpy().astype('float64')
    Ac_np = hz.Ac.detach().cpu().numpy().astype('float64')
    Ab_np = hz.Ab.detach().cpu().numpy().astype('float64')
    b_np = hz.b.detach().cpu().numpy().astype('float64').reshape(-1)

    n, m_c = Gc_np.shape
    _, m_b = Gb_np.shape
    p = b_np.shape[0]

    model = gp.Model("hz_bounds")
    model.Params.OutputFlag = 0

    xi_c = model.addMVar(shape=m_c, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="xi_c") if m_c > 0 else None
    zeta = model.addMVar(shape=m_b, vtype=GRB.BINARY, name="zeta") if m_b > 0 else None
    xi_b = (2 * zeta - 1) if zeta is not None else None

    if p > 0:
        lhs = 0
        if xi_c is not None:
            lhs = lhs + Ac_np @ xi_c
        if xi_b is not None:
            lhs = lhs + Ab_np @ xi_b
        model.addConstr(lhs <= b_np)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)

    for i in range(n):
        expr = float(c_np[i])
        if xi_c is not None and m_c > 0:
            expr = expr + gp.quicksum(float(Gc_np[i, j]) * xi_c[j] for j in range(m_c))
        if xi_b is not None and m_b > 0:
            expr = expr + gp.quicksum(float(Gb_np[i, k]) * xi_b[k] for k in range(m_b))

        model.setObjective(expr, GRB.MAXIMIZE)
        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi MAX infeasible at dim {i}, status={model.status}")
        UB[i] = model.ObjVal

        model.setObjective(expr, GRB.MINIMIZE)
        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi MIN infeasible at dim {i}, status={model.status}")
        LB[i] = model.ObjVal

    dtype, device = hz.c.dtype, hz.c.device
    lb = torch.from_numpy(LB).to(device=device, dtype=dtype).flatten()
    ub = torch.from_numpy(UB).to(device=device, dtype=dtype).flatten()
    return Bounds(lb=lb, ub=ub)


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    """SciPy LP relaxation (fallback, treats binary generators as continuous)."""
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])

    c_np = hz.c.detach().cpu().numpy().astype('float64').reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype('float64')
    Gb_np = hz.Gb.detach().cpu().numpy().astype('float64')
    Ac_np = hz.Ac.detach().cpu().numpy().astype('float64')
    Ab_np = hz.Ab.detach().cpu().numpy().astype('float64')
    b_np = hz.b.detach().cpu().numpy().astype('float64').reshape(-1)

    A_ub = np.concatenate([Ac_np, Ab_np], axis=1) if (Ac_np.size or Ab_np.size) else None
    b_ub = b_np if (A_ub is not None) else None
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)

    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)

        res_min = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method="highs")
        if not res_min.success:
            raise RuntimeError(f"[linprog] MIN infeasible at dim {i}: {res_min.message}")
        LB[i] = c_np[i] + res_min.fun

        res_max = linprog(c=-obj, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method="highs")
        if not res_max.success:
            raise RuntimeError(f"[linprog] MAX infeasible at dim {i}: {res_max.message}")
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    lb = torch.from_numpy(LB).to(device=device, dtype=dtype).flatten()
    ub = torch.from_numpy(UB).to(device=device, dtype=dtype).flatten()
    return Bounds(lb=lb, ub=ub)


# ---- Nonlinear activations --------------------------------------------------

def _hz_apply_relu(hz: HZono) -> HZono:
    """Graph-exact ReLU preserving input generators (hz1/hz2 paper method).

    For each neuron i with bounds [lb_i, ub_i]:
      - active  (lb >= 0): y_i = x_i  (identity, no new generators)
      - inactive (ub <= 0): y_i = 0   (zero row, no new generators)
      - unstable (lb < 0 < ub): encode ReLU graph with 2 new continuous
        generators (u_i, v_i) and 1 new binary generator (z_i), plus
        6 constraint rows (4 graph + 2 linking).

    The output HZ has:
      Gc: (n, ng_in + 2k),  Gb: (n, nb_in + k),
      Ac: (nc_in + 6k, ng_in + 2k),  Ab: (nc_in + 6k, nb_in + k)
    where k = number of unstable neurons.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = _hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active   = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    # Output center: active keeps c, inactive zeroed
    new_c = hz.c.clone()
    new_c[inactive] = 0.0

    # Output Gc rows: active keeps row, inactive zeroed
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[inactive] = 0.0

    # Output Gb rows: active keeps row, inactive zeroed
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[inactive] = 0.0

    if k == 0:
        return _get_HZono()(c=new_c, Gc=new_Gc_base, Gb=new_Gb_base,
                            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # Unstable neurons: graph-exact encoding
    # a_i = max(|lb_i|, ub_i) for each unstable neuron
    unstable_idx = torch.where(unstable)[0]
    a = torch.maximum(torch.abs(lb[unstable_idx]), ub[unstable_idx])

    # New center for unstable: y_i = a_i/4
    new_c[unstable_idx] = (a / 4.0).unsqueeze(1)

    # Zero out input generator rows for unstable (will be linked via constraints)
    new_Gc_base[unstable_idx] = 0.0
    new_Gb_base[unstable_idx] = 0.0

    # New continuous generators: 2k columns for (u, v) per unstable neuron
    # y_i = a_i/4 + (a_i/2)*v_j  (j-th unstable neuron)
    Gc_uv = torch.zeros((n, 2 * k), dtype=dtype, device=device)
    for j, idx in enumerate(unstable_idx):
        Gc_uv[idx, k + j] = a[j] / 2.0  # v_j column contributes to output

    # New binary generators: k columns for z per unstable neuron
    # y_i += (a_i/4)*z_j
    Gb_z = torch.zeros((n, k), dtype=dtype, device=device)
    for j, idx in enumerate(unstable_idx):
        Gb_z[idx, j] = a[j] / 4.0

    # Assemble output generators: [Gc_base | Gc_uv], [Gb_base | Gb_z]
    out_Gc = torch.cat([new_Gc_base, Gc_uv], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_z], dim=1)

    # ---- New constraints (6k rows) ----
    ng_new = ng + 2 * k
    nb_new = nb + k

    # Graph constraints (4k rows): |u_j| ≤ (1-z_j)/2,  |v_j| ≤ (z_j+1)/2
    #   u_j + 0.5*z_j ≤ 0.5
    #  -u_j + 0.5*z_j ≤ 0.5
    #   v_j - 0.5*z_j ≤ 0.5
    #  -v_j - 0.5*z_j ≤ 0.5
    graph_Ac = torch.zeros((4 * k, ng_new), dtype=dtype, device=device)
    graph_Ab = torch.zeros((4 * k, nb_new), dtype=dtype, device=device)
    graph_b  = torch.full((4 * k, 1), 0.5, dtype=dtype, device=device)

    for j in range(k):
        u_col = ng + j        # u_j in new Gc columns
        v_col = ng + k + j    # v_j in new Gc columns
        z_col = nb + j        # z_j in new Gb columns
        r = 4 * j

        # u_j + 0.5*z_j ≤ 0.5
        graph_Ac[r, u_col] = 1.0
        graph_Ab[r, z_col] = 0.5
        # -u_j + 0.5*z_j ≤ 0.5
        graph_Ac[r + 1, u_col] = -1.0
        graph_Ab[r + 1, z_col] = 0.5
        # v_j - 0.5*z_j ≤ 0.5
        graph_Ac[r + 2, v_col] = 1.0
        graph_Ab[r + 2, z_col] = -0.5
        # -v_j - 0.5*z_j ≤ 0.5
        graph_Ac[r + 3, v_col] = -1.0
        graph_Ab[r + 3, z_col] = -0.5

    # Linking constraints (2k rows): encode x_i = (a_i/2)*(u_j + v_j) + (a_i/2)*z_j
    #   Gc[i]*ξ_c + Gb[i]*ξ_b - (a_i/2)*(u_j+v_j) - (a_i/2)*z_j ≤ -c_i
    #  -Gc[i]*ξ_c - Gb[i]*ξ_b + (a_i/2)*(u_j+v_j) + (a_i/2)*z_j ≤  c_i
    link_Ac = torch.zeros((2 * k, ng_new), dtype=dtype, device=device)
    link_Ab = torch.zeros((2 * k, nb_new), dtype=dtype, device=device)
    link_b  = torch.zeros((2 * k, 1), dtype=dtype, device=device)

    for j, idx in enumerate(unstable_idx):
        idx_i = int(idx.item())
        u_col = ng + j
        v_col = ng + k + j
        z_col = nb + j

        # Row 2j: Gc[i]*ξ_c + Gb[i]*ξ_b - (a_i/2)*(u+v) - (a_i/2)*z ≤ -c_i
        link_Ac[2 * j, :ng] = hz.Gc[idx_i]
        link_Ac[2 * j, u_col] = -a[j] / 2.0
        link_Ac[2 * j, v_col] = -a[j] / 2.0
        link_Ab[2 * j, :nb] = hz.Gb[idx_i]
        link_Ab[2 * j, z_col] = -a[j] / 2.0
        link_b[2 * j, 0] = -hz.c[idx_i, 0]

        # Row 2j+1: -Gc[i]*ξ_c - Gb[i]*ξ_b + (a_i/2)*(u+v) + (a_i/2)*z ≤ c_i
        link_Ac[2 * j + 1, :ng] = -hz.Gc[idx_i]
        link_Ac[2 * j + 1, u_col] = a[j] / 2.0
        link_Ac[2 * j + 1, v_col] = a[j] / 2.0
        link_Ab[2 * j + 1, :nb] = -hz.Gb[idx_i]
        link_Ab[2 * j + 1, z_col] = a[j] / 2.0
        link_b[2 * j + 1, 0] = hz.c[idx_i, 0]

    # Extend existing constraints to new column dimensions
    old_Ac_ext = torch.cat([hz.Ac, torch.zeros((nc, 2 * k), dtype=dtype, device=device)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)

    out_Ac = torch.cat([old_Ac_ext, graph_Ac, link_Ac], dim=0)
    out_Ab = torch.cat([old_Ab_ext, graph_Ab, link_Ab], dim=0)
    out_b  = torch.cat([hz.b, graph_b, link_b], dim=0)

    return _get_HZono()(c=new_c, Gc=out_Gc, Gb=out_Gb,
                        Ac=out_Ac, Ab=out_Ab, b=out_b)


def _hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Graph-exact LeakyReLU preserving input generators.

    Same structure as ReLU but with y = alpha*x for x < 0.
    Active/inactive cases handled directly; unstable neurons use
    graph encoding with slope alpha on the negative branch.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = _hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active   = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    # Active: y = x (identity). Inactive: y = alpha*x (scale).
    new_c = hz.c.clone()
    new_Gc_base = hz.Gc.clone()
    new_Gb_base = hz.Gb.clone()
    for idx in torch.where(inactive)[0]:
        i = int(idx.item())
        new_c[i] *= alpha_arg
        new_Gc_base[i] *= alpha_arg
        new_Gb_base[i] *= alpha_arg

    if k == 0:
        return _get_HZono()(c=new_c, Gc=new_Gc_base, Gb=new_Gb_base,
                            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # Unstable: triangle relaxation (sound over-approximation)
    # Upper bound line through (lb, alpha*lb) and (ub, ub): slope_u, shift_u
    # Lower bound: lambda*x where lambda = min(1, alpha) for lower facet
    unstable_idx = torch.where(unstable)[0]
    lb_u = lb[unstable_idx]
    ub_u = ub[unstable_idx]
    slope_u = (ub_u - alpha_arg * lb_u) / (ub_u - lb_u + 1e-30)
    shift_u = (1.0 - alpha_arg) * lb_u * ub_u / (ub_u - lb_u + 1e-30)

    # Output for unstable: midpoint of [alpha*x, slope_u*x + shift_u]
    # Use interval approach: compute the center and radius of the output range
    mid_slope = (slope_u + alpha_arg) / 2.0
    half_slope = (slope_u - alpha_arg) / 2.0

    new_c_unstable = mid_slope * hz.c[unstable_idx, 0] + shift_u / 2.0
    new_c[unstable_idx] = new_c_unstable.unsqueeze(1)

    # Scale input generators by mid_slope
    new_Gc_base[unstable_idx] = mid_slope.unsqueeze(1) * hz.Gc[unstable_idx]
    new_Gb_base[unstable_idx] = mid_slope.unsqueeze(1) * hz.Gb[unstable_idx]

    # Add 1 new continuous generator per unstable for the approximation gap
    Gc_gap = torch.zeros((n, k), dtype=dtype, device=device)
    for j, idx in enumerate(unstable_idx):
        Gc_gap[idx, j] = half_slope[j] * (ub_u[j] - lb_u[j]) / 2.0 + shift_u[j] / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_gap], dim=1)
    out_Gb = new_Gb_base

    # Extend constraints for new columns
    old_Ac_ext = torch.cat([hz.Ac, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)
    out_Ac = old_Ac_ext
    out_Ab = hz.Ab.clone()
    out_b  = hz.b.clone()

    return _get_HZono()(c=new_c, Gc=out_Gc, Gb=out_Gb,
                        Ac=out_Ac, Ab=out_Ab, b=out_b)


def _hz_apply_monotone(hz: HZono, func, dfunc) -> HZono:
    """DeepZ-style abstraction for monotone activations (sigmoid, tanh).

    Preserves input generators by computing a linear approximation:
      y ≈ lambda * x + mu,  with added error generator for the gap.
    lambda = min slope on [lb, ub] (for soundness with monotone concave/convex).
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nc = hz.Ac.shape[0]

    bounds = _hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    f_lb = func(lb)
    f_ub = func(ub)

    # Optimal linear slope: minimum derivative on [lb, ub] for convex-concave
    lam = torch.zeros(n, dtype=dtype, device=device)
    mu_lb = torch.zeros(n, dtype=dtype, device=device)
    mu_ub = torch.zeros(n, dtype=dtype, device=device)

    wide = (ub - lb) > 1e-12
    # Slope of secant line
    lam[wide] = (f_ub[wide] - f_lb[wide]) / (ub[wide] - lb[wide])
    lam[~wide] = dfunc(lb[~wide])

    # Compute range of (f(x) - lam*x) to find optimal shift
    resid_lb = f_lb - lam * lb
    resid_ub = f_ub - lam * ub
    mu_lb = torch.minimum(resid_lb, resid_ub)
    mu_ub = torch.maximum(resid_lb, resid_ub)

    # Also check at inflection points (x=0 for tanh/sigmoid)
    zero_in_range = (lb < 0) & (ub > 0)
    if zero_in_range.any():
        f_zero = func(torch.zeros(1, dtype=dtype, device=device))
        resid_zero = f_zero - lam[zero_in_range] * 0.0
        mu_lb[zero_in_range] = torch.minimum(mu_lb[zero_in_range], resid_zero)
        mu_ub[zero_in_range] = torch.maximum(mu_ub[zero_in_range], resid_zero)

    # Output: y = lam*x + (mu_lb+mu_ub)/2 ± (mu_ub-mu_lb)/2
    mu_mid = (mu_lb + mu_ub) / 2.0
    mu_rad = (mu_ub - mu_lb) / 2.0

    # Scale existing generators by lambda
    new_c = lam.view(-1, 1) * hz.c + mu_mid.view(-1, 1)
    new_Gc_base = lam.view(-1, 1) * hz.Gc
    new_Gb = lam.view(-1, 1) * hz.Gb

    # Add 1 error generator per neuron
    Gc_err = torch.diag(mu_rad)

    out_Gc = torch.cat([new_Gc_base, Gc_err], dim=1)

    # Extend constraints for new Gc columns
    old_Ac_ext = torch.cat([hz.Ac, torch.zeros((nc, n), dtype=dtype, device=device)], dim=1)

    return _get_HZono()(c=new_c, Gc=out_Gc, Gb=new_Gb,
                        Ac=old_Ac_ext, Ab=hz.Ab.clone(), b=hz.b.clone())


def _hz_apply_sigmoid(hz: HZono) -> HZono:
    """DeepZ-style sigmoid preserving input generators."""
    def _sig(x):
        return torch.sigmoid(x)
    def _dsig(x):
        s = torch.sigmoid(x)
        return s * (1 - s)
    return _hz_apply_monotone(hz, _sig, _dsig)


def _hz_apply_tanh(hz: HZono) -> HZono:
    """DeepZ-style tanh preserving input generators."""
    def _tanh(x):
        return torch.tanh(x)
    def _dtanh(x):
        return 1 - torch.tanh(x) ** 2
    return _hz_apply_monotone(hz, _tanh, _dtanh)


# ---- Utilities --------------------------------------------------------------

def _hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
    """Minkowski sum: c1+c2, block-diag generators, block constraints."""
    dtype, device = hz1.c.dtype, hz1.c.device

    new_c = hz1.c + hz2.c.to(dtype=dtype, device=device)

    # Minkowski sum: horizontal concat of generators (same rows, more columns)
    new_Gc = torch.cat([hz1.Gc, hz2.Gc.to(dtype=dtype, device=device)], dim=1)
    new_Gb = torch.cat([hz1.Gb, hz2.Gb.to(dtype=dtype, device=device)], dim=1)

    # Block-diagonal constraints
    nc1 = hz1.Ac.shape[0]
    nc2 = hz2.Ac.shape[0]
    ng1 = hz1.Gc.shape[1]
    ng2 = hz2.Gc.shape[1]
    nb1 = hz1.Gb.shape[1]
    nb2 = hz2.Gb.shape[1]

    Ac_top = torch.cat([hz1.Ac, torch.zeros((nc1, ng2), dtype=dtype, device=device)], dim=1)
    Ac_bot = torch.cat([torch.zeros((nc2, ng1), dtype=dtype, device=device),
                         hz2.Ac.to(dtype=dtype, device=device)], dim=1)
    new_Ac = torch.cat([Ac_top, Ac_bot], dim=0)

    Ab_top = torch.cat([hz1.Ab, torch.zeros((nc1, nb2), dtype=dtype, device=device)], dim=1)
    Ab_bot = torch.cat([torch.zeros((nc2, nb1), dtype=dtype, device=device),
                         hz2.Ab.to(dtype=dtype, device=device)], dim=1)
    new_Ab = torch.cat([Ab_top, Ab_bot], dim=0)

    new_b = torch.cat([hz1.b, hz2.b.to(dtype=dtype, device=device)], dim=0)

    return _get_HZono()(c=new_c, Gc=new_Gc, Gb=new_Gb, Ac=new_Ac, Ab=new_Ab, b=new_b)


def _hz_from_bounds_fresh(bounds: Bounds, dtype, device) -> HZono:
    """Create fresh HZ from Bounds (for ops that lose correlation)."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    Gc = torch.diag(rad)
    Gb = torch.zeros((n, 0), dtype=dtype, device=device)
    Ac = torch.zeros((0, n), dtype=dtype, device=device)
    Ab = torch.zeros((0, 0), dtype=dtype, device=device)
    b = torch.zeros((0, 1), dtype=dtype, device=device)
    return _get_HZono()(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)


# ---- Complexity reduction (PhD thesis Chapter 6) ---------------------------

def _hz_reduce(hz: HZono, max_order: float = 10.0) -> HZono:
    """Reduce HZ complexity by relaxing binary generators and removing
    low-impact continuous generators (sound over-approximation).

    Args:
        max_order: maximum ratio ng/n. When exceeded, reduction is applied.
                   Default 10 balances precision vs memory for typical networks.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if n == 0:
        return hz

    # Target: at most max_order * n continuous generators, at most 2*n binary
    max_ng = max(int(max_order * n), n + 1)
    max_nb = max(2 * n, 1)

    # Step 1: Relax excess binary generators to continuous (Prop 6.2.4)
    # Move smallest-norm Gb columns to Gc (ξ_b ∈ {-1,1} → ξ_b ∈ [-1,1])
    if nb > max_nb:
        col_norms = hz.Gb.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort()
        n_relax = nb - max_nb
        relax_idx = sorted_idx[:n_relax]
        keep_idx  = sorted_idx[n_relax:]

        # Move relaxed Gb columns to Gc, and their Ab columns to Ac
        extra_Gc = hz.Gb[:, relax_idx]
        extra_Ac = hz.Ab[:, relax_idx] if nc > 0 else torch.zeros((0, n_relax), dtype=dtype, device=device)

        hz = _get_HZono()(
            c=hz.c,
            Gc=torch.cat([hz.Gc, extra_Gc], dim=1),
            Gb=hz.Gb[:, keep_idx],
            Ac=torch.cat([hz.Ac, extra_Ac], dim=1) if nc > 0 else torch.zeros((0, ng + n_relax), dtype=dtype, device=device),
            Ab=hz.Ab[:, keep_idx] if nc > 0 else torch.zeros((0, max_nb), dtype=dtype, device=device),
            b=hz.b.clone(),
        )
        ng = hz.Gc.shape[1]
        nb = hz.Gb.shape[1]

    # Step 2: Reduce continuous generators via zonotope order reduction
    # (lift-then-reduce, Prop 6.2.3 simplified: Girard's method on Gc)
    if ng > max_ng:
        # Girard's heuristic: keep n largest-norm columns, box the rest
        col_norms = hz.Gc.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort(descending=True)
        keep_idx = sorted_idx[:max_ng - n]  # keep best columns, leave room for n box columns
        drop_idx = sorted_idx[max_ng - n:]

        Gc_keep = hz.Gc[:, keep_idx]
        Gc_drop = hz.Gc[:, drop_idx]

        # Over-approximate dropped generators with an axis-aligned box
        box_rad = Gc_drop.abs().sum(dim=1)  # (n,)
        Gc_box = torch.diag(box_rad)

        new_Gc = torch.cat([Gc_keep, Gc_box], dim=1)

        # Constraints referencing dropped columns become invalid → remove all
        # constraints that reference dropped generators (conservative: drop all)
        if nc > 0:
            # Check if any constraint row references a dropped column
            drop_set = set(drop_idx.tolist())
            keep_rows = []
            for r in range(nc):
                refs_dropped = any(
                    abs(hz.Ac[r, c].item()) > 1e-15 for c in drop_set
                )
                if not refs_dropped:
                    keep_rows.append(r)

            if keep_rows:
                keep_rows_t = torch.tensor(keep_rows, dtype=torch.long, device=device)
                new_Ac_partial = hz.Ac[keep_rows_t][:, keep_idx]
                new_Ab = hz.Ab[keep_rows_t]
                new_b = hz.b[keep_rows_t]
                # Extend Ac for box columns (zeros)
                nc_kept = len(keep_rows)
                new_Ac = torch.cat([new_Ac_partial,
                                    torch.zeros((nc_kept, n), dtype=dtype, device=device)], dim=1)
            else:
                nc_kept = 0
                new_Ac = torch.zeros((0, new_Gc.shape[1]), dtype=dtype, device=device)
                new_Ab = torch.zeros((0, nb), dtype=dtype, device=device)
                new_b = torch.zeros((0, 1), dtype=dtype, device=device)
        else:
            new_Ac = torch.zeros((0, new_Gc.shape[1]), dtype=dtype, device=device)
            new_Ab = torch.zeros((0, nb), dtype=dtype, device=device)
            new_b = torch.zeros((0, 1), dtype=dtype, device=device)

        hz = _get_HZono()(c=hz.c, Gc=new_Gc, Gb=hz.Gb,
                          Ac=new_Ac, Ab=new_Ab, b=new_b)

    return hz


# ============================================================================
# Transfer functions — pure: (L, Bounds, hz_in) -> (Fact, hz_out)
# ============================================================================

@torch.no_grad()
def hybridz_tf_dense(L: Layer, Bin: Bounds, hz_in=None):
    """Dense layer. Returns (Fact, hz_out)."""
    W = L.params["weight"]
    b = L.params.get("bias", None)

    hz_out = None
    if hz_in is not None:
        hz_out = _hz_multiply(hz_in, W)
        if b is not None:
            b_col = b.to(dtype=hz_out.c.dtype, device=hz_out.c.device)
            if b_col.ndim == 1:
                b_col = b_col.view(-1, 1)
            hz_out = _hz_add_const(hz_out, b_col)
        Bout = _hz_compute_bounds(hz_out)
    else:
        if W.shape[1] != Bin.lb.shape[0]:
            raise ValueError(f"Dense layer input mismatch: W expects {W.shape[1]}, got {Bin.lb.shape[0]}")
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        lb = W_pos @ Bin.lb + W_neg @ Bin.ub
        ub = W_pos @ Bin.ub + W_neg @ Bin.lb
        if b is not None:
            lb = lb + b
            ub = ub + b
        Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"dense:{L.id}", list(L.out_vars + L.in_vars), W=W, b=b)
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_bias(L: Layer, Bin: Bounds, hz_in=None):
    """Bias addition. Returns (Fact, hz_out)."""
    c = L.params["c"]

    hz_out = None
    if hz_in is not None:
        c_col = c.to(dtype=hz_in.c.dtype, device=hz_in.c.device)
        if c_col.ndim == 1:
            c_col = c_col.view(-1, 1)
        hz_out = _hz_add_const(hz_in, c_col)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = Bin.lb + c
        ub = Bin.ub + c
        Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"bias:{L.id}", list(L.out_vars + L.in_vars), c=c)
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_scale(L: Layer, Bin: Bounds, hz_in=None):
    """Element-wise scaling. Returns (Fact, hz_out)."""
    a = L.params["a"]

    hz_out = None
    if hz_in is not None:
        hz_out = _hz_scale_elem(hz_in, a)
        Bout = _hz_compute_bounds(hz_out)
    else:
        a_pos = torch.clamp(a, min=0)
        a_neg = torch.clamp(a, max=0)
        lb = a_pos * Bin.lb + a_neg * Bin.ub
        ub = a_pos * Bin.ub + a_neg * Bin.lb
        Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"scale:{L.id}", list(L.out_vars + L.in_vars), a=a)
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_relu(L: Layer, Bin: Bounds, hz_in=None):
    """ReLU with graph-exact HZ encoding. Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_relu(hz_in)
        hz_out = _hz_reduce(hz_out)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.clamp(Bin.lb, min=0)
        ub = torch.clamp(Bin.ub, min=0)
        Bout = Bounds(lb=lb, ub=ub)

    # Constraint generation (always from interval Bin)
    cons = ConSet()
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    if len(idx_amb) > 0:
        slope = Bin.lb[idx_amb] / torch.clamp(Bin.ub[idx_amb] - Bin.lb[idx_amb], min=1e-12)
        shift = -slope * Bin.lb[idx_amb]
    cons.add_op(f"relu:{L.id}", list(L.out_vars + L.in_vars),
                idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb,
                slope=slope, shift=shift)
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_lrelu(L: Layer, Bin: Bounds, hz_in=None):
    """LeakyReLU. Returns (Fact, hz_out)."""
    alpha = float(L.params.get("negative_slope", 0.01))

    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_leaky_relu(hz_in, alpha)
        hz_out = _hz_reduce(hz_out)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
        ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
        Bout = Bounds(lb=lb, ub=ub)

    # Constraint generation
    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    if len(idx_amb) > 0:
        y_at_ub = Bin.ub[idx_amb]
        y_at_lb = alpha * Bin.lb[idx_amb]
        denom = Bin.ub[idx_amb] - Bin.lb[idx_amb]
        slope[idx_amb] = torch.where(denom > 1e-8, (y_at_ub - y_at_lb) / denom, torch.ones_like(denom))
        shift[idx_amb] = y_at_lb - slope[idx_amb] * Bin.lb[idx_amb]
    cons = ConSet()
    cons.add_op(f"lrelu:{L.id}", list(L.out_vars + L.in_vars), alpha=alpha,
                idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb,
                slope=slope[idx_amb], shift=shift[idx_amb])
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_tanh(L: Layer, Bin: Bounds, hz_in=None):
    """Tanh. Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_tanh(hz_in)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.tanh(Bin.lb)
        ub = torch.tanh(Bin.ub)
        Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    cons = ConSet()
    cons.add_op(f"tanh:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_sigmoid(L: Layer, Bin: Bounds, hz_in=None):
    """Sigmoid. Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_sigmoid(hz_in)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.sigmoid(Bin.lb)
        ub = torch.sigmoid(Bin.ub)
        Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    cons = ConSet()
    cons.add_op(f"sigmoid:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_abs(L: Layer, Bin: Bounds, hz_in=None):
    """Absolute value. Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in is not None:
        bounds = _hz_compute_bounds(hz_in)
        lb_out = torch.where(bounds.lb >= 0, bounds.lb,
                             torch.where(bounds.ub <= 0, -bounds.ub,
                                         torch.zeros_like(bounds.lb)))
        ub_out = torch.maximum(torch.abs(bounds.lb), torch.abs(bounds.ub))
        Bout = Bounds(lb=lb_out, ub=ub_out)
        hz_out = _hz_from_bounds_fresh(Bout, hz_in.c.dtype, hz_in.c.device)
    else:
        idx_pos = torch.where(Bin.lb >= 0)[0]
        idx_neg = torch.where(Bin.ub <= 0)[0]
        idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
        lb = torch.where(idx_amb[:, None] == torch.arange(len(Bin.lb))[None, :],
                         torch.zeros_like(Bin.lb),
                         torch.where(Bin.lb >= 0, Bin.lb, -Bin.ub))
        ub = torch.maximum(torch.abs(Bin.lb), torch.abs(Bin.ub))
        Bout = Bounds(lb=lb, ub=ub)

    # ConSet indices always from interval Bin
    idx_pos = torch.where(Bin.lb >= 0)[0]
    idx_neg = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    cons = ConSet()
    cons.add_op(f"abs:{L.id}", list(L.out_vars + L.in_vars),
                idx_pos=idx_pos, idx_neg=idx_neg, idx_amb=idx_amb)
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_add(L: Layer, Bin1: Bounds, Bin2: Bounds, hz_in1=None, hz_in2=None):
    """Element-wise addition. Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in1 is not None and hz_in2 is not None:
        hz_out = _hz_minkowski_sum(hz_in1, hz_in2)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = Bin1.lb + Bin2.lb
        ub = Bin1.ub + Bin2.ub
        Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"add:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons), hz_out


@torch.no_grad()
def hybridz_tf_mul(L: Layer, Bin1: Bounds, Bin2: Bounds, hz_in1=None, hz_in2=None):
    """Element-wise multiplication (McCormick). Returns (Fact, hz_out)."""
    hz_out = None
    if hz_in1 is not None:
        bounds1 = _hz_compute_bounds(hz_in1)
        bounds2 = _hz_compute_bounds(hz_in2) if hz_in2 is not None else Bin2
        lx, ux = bounds1.lb, bounds1.ub
        ly, uy = bounds2.lb, bounds2.ub
    else:
        lx, ux = Bin1.lb, Bin1.ub
        ly, uy = Bin2.lb, Bin2.ub

    corners = torch.stack([lx * ly, lx * uy, ux * ly, ux * uy])
    lb = torch.min(corners, dim=0)[0]
    ub = torch.max(corners, dim=0)[0]
    Bout = Bounds(lb=lb, ub=ub)

    if hz_in1 is not None:
        hz_out = _hz_from_bounds_fresh(Bout, hz_in1.c.dtype, hz_in1.c.device)

    cons = ConSet()
    cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars),
                lx=Bin1.lb, ux=Bin1.ub, ly=Bin2.lb, uy=Bin2.ub)
    return Fact(bounds=Bout, cons=cons), hz_out
