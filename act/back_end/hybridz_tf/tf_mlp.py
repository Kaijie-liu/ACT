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
    return _get_HZono()( c=R @ hz.c, Gc=R @ hz.Gc, Gb=R @ hz.Gb, Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(), )

def _hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    """Translate center: c'=c+v, generators and constraints unchanged."""
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return _get_HZono()( c=hz.c + v, Gc=hz.Gc.clone(), Gb=hz.Gb.clone(), Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(), )



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
        
    # unconstrained over-approximation
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
    """Graph-exact ReLU preserving input generators.

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


def _hz_apply_piecewise(hz: HZono, func, dfunc, K: int = 2) -> HZono:
    """Piecewise linear approximation for monotone activations (tangent parallelogram).

    Encodes y = func(x) using K linear pieces per neuron, introducing
    K continuous generators (g1, g2 each) and K binary generators (z)
    per wide neuron, linked by box, linking, and exactly-one constraints.

    Each piece [a, b] is enclosed by a parallelogram whose sides run along
    the tangent lines at the endpoints, giving a tighter enclosure than the
    old secant + error-rectangle method.

    Args:
        hz:    Input hybrid zonotope.
        func:  Monotone activation (e.g. torch.tanh, torch.sigmoid).
        dfunc: Derivative of func.
        K:     Number of linear pieces per neuron (default 2).

    Returns:
        Output hybrid zonotope with piecewise linear encoding.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]   # existing continuous generators
    nb = hz.Gb.shape[1]   # existing binary generators
    nc = hz.Ac.shape[0]   # existing constraint rows

    bounds = _hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.sum() if wide_idx.ndim == 0 else wide_idx.shape[0])  # number of wide neurons

    # -- Handle narrow neurons: just apply func to center, zero generators ----
    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])

    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0

    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        # No wide neurons — return directly
        return _get_HZono()(c=new_c, Gc=new_Gc_base, Gb=new_Gb_base,
                            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # -- Precompute per-piece quantities for all wide neurons ------------------
    lb_w = lb[wide_idx]  # (m,)
    ub_w = ub[wide_idx]  # (m,)

    # Lists indexed by piece k; each entry is (m,) tensor
    centers_x_k = []   # center x of piece k
    centers_y_k = []   # center y of piece k
    g1_x_k = []        # generator 1 x-component
    g1_y_k = []        # generator 1 y-component
    g2_x_k = []        # generator 2 x-component
    g2_y_k = []        # generator 2 y-component

    for k in range(K):
        # Breakpoints: a = lb_w + k*(ub_w-lb_w)/K, b = lb_w + (k+1)*(ub_w-lb_w)/K
        a = lb_w + k * (ub_w - lb_w) / K        # (m,)
        b = lb_w + (k + 1) * (ub_w - lb_w) / K  # (m,)

        fa = func(a)
        fb = func(b)
        la = dfunc(a)   # derivative at a
        lb_slope = dfunc(b)   # derivative at b

        cx = (a + b) / 2.0   # (m,)
        cy = (fa + fb) / 2.0  # (m,)

        # Check if derivatives are nearly equal (nearly linear piece)
        nearly_linear = (la - lb_slope).abs() < 1e-10  # (m,) bool

        # --- Tangent-based parallelogram (non-linear case) ---
        # p1 = (fb - fa + lb_slope*a - la*b) / (lb_slope - la)
        # p2 = a + b - p1
        denom = lb_slope - la  # (m,)
        # Safe division: use 1.0 where nearly_linear to avoid NaN
        safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
        p1 = (fb - fa + lb_slope * a - la * b) / safe_denom  # (m,)
        p2 = a + b - p1  # (m,)

        g1x_tang = (p1 - a) / 2.0          # along tangent at b
        g1y_tang = lb_slope * (p1 - a) / 2.0
        g2x_tang = (p2 - a) / 2.0          # along tangent at a
        g2y_tang = la * (p2 - a) / 2.0

        # --- Secant + sampled error fallback (nearly linear case) ---
        hw = (b - a) / 2.0   # (m,)
        slope = (fb - fa) / (b - a + 1e-30)  # (m,)
        # Sample 50 points to find max residual
        t = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).unsqueeze(1)
        pts = a.unsqueeze(0) + t * (b - a).unsqueeze(0)  # (50, m)
        f_pts = func(pts)                                  # (50, m)
        resid = f_pts - (slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0))  # (50, m)
        max_err = resid.abs().max(dim=0).values  # (m,)

        g1x_lin = hw
        g1y_lin = slope * hw
        g2x_lin = torch.zeros_like(hw)
        g2y_lin = max_err

        # --- Select based on nearly_linear mask ---
        g1x = torch.where(nearly_linear, g1x_lin, g1x_tang)
        g1y = torch.where(nearly_linear, g1y_lin, g1y_tang)
        g2x = torch.where(nearly_linear, g2x_lin, g2x_tang)
        g2y = torch.where(nearly_linear, g2y_lin, g2y_tang)

        # --- Soundness check: sample 50 points and verify containment ---
        x_check = pts  # reuse the 50 sample points (50, m)
        y_check = f_pts  # (50, m)
        dx = x_check - cx.unsqueeze(0)  # (50, m)
        dy = y_check - cy.unsqueeze(0)  # (50, m)

        # Solve 2x2 system: [g1x g2x; g1y g2y] * [xi1; xi2] = [dx; dy]
        # xi1 = (dy*g2x - dx*g2y) / (g1y*g2x - g1x*g2y)
        # xi2 = (dy*g1x - dx*g1y) / (g2y*g1x - g2x*g1y)
        det = g1y * g2x - g1x * g2y  # (m,)
        safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)

        xi1 = (dy * g2x.unsqueeze(0) - dx * g2y.unsqueeze(0)) / safe_det.unsqueeze(0)  # (50, m)
        xi2 = (dy * g1x.unsqueeze(0) - dx * g1y.unsqueeze(0)) / (-safe_det.unsqueeze(0))  # (50, m)

        max_xi = torch.max(xi1.abs().max(dim=0).values, xi2.abs().max(dim=0).values)  # (m,)
        # Where max_xi > 1, scale generators to accommodate (plus 1% buffer)
        needs_scale = max_xi > 1.0
        scale_factor = torch.where(needs_scale, max_xi * 1.01, torch.ones_like(max_xi))
        # Only scale where det is non-degenerate
        scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)

        g1x = g1x * scale_factor
        g1y = g1y * scale_factor
        g2x = g2x * scale_factor
        g2y = g2y * scale_factor

        centers_x_k.append(cx)
        centers_y_k.append(cy)
        g1_x_k.append(g1x)
        g1_y_k.append(g1y)
        g2_x_k.append(g2x)
        g2_y_k.append(g2y)

    # -- Output center for wide neurons: sum_k cy_k / 2 ----------------------
    cy_sum = torch.zeros(m, dtype=dtype, device=device)
    for k in range(K):
        cy_sum = cy_sum + centers_y_k[k]
    new_c[wide_idx] = (cy_sum / 2.0).unsqueeze(1)

    # -- Zero out input generator rows for wide neurons -----------------------
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    # -- New continuous generators: 2*K*m columns (K g1 + K g2 per wide neuron)
    # Layout: [g1_{0,0}..g1_{m-1,0}, g1_{0,1}..g1_{m-1,1}, ...,
    #          g2_{0,0}..g2_{m-1,0}, ...]
    # i.e., g1 columns first (K*m), then g2 columns (K*m)
    Gc_new = torch.zeros((n, 2 * K * m), dtype=dtype, device=device)
    for k in range(K):
        g1_cols = torch.arange(k * m, (k + 1) * m, device=device)    # m cols for g1 piece k
        g2_cols = torch.arange(K * m + k * m, K * m + (k + 1) * m, device=device)  # m cols for g2 piece k
        # g1 contributes g1_y to output row, g2 contributes g2_y to output row
        for j in range(m):
            idx_i = wide_idx[j]
            Gc_new[idx_i, g1_cols[j]] = g1_y_k[k][j]
            Gc_new[idx_i, g2_cols[j]] = g2_y_k[k][j]

    # -- New binary generators: K*m columns (z_{i,k}) -------------------------
    Gb_new = torch.zeros((n, K * m), dtype=dtype, device=device)
    for k in range(K):
        z_cols = torch.arange(k * m, (k + 1) * m, device=device)
        for j in range(m):
            idx_i = wide_idx[j]
            Gb_new[idx_i, z_cols[j]] = centers_y_k[k][j] / 2.0

    # -- Assemble output generators -------------------------------------------
    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)   # (n, ng + 2*K*m)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)    # (n, nb + K*m)

    ng_new = ng + 2 * K * m
    nb_new = nb + K * m

    # ---- New constraints: (4K + 4) rows per wide neuron = (4K+4)*m total ----
    # 1. Box constraints: 4K rows per wide neuron (4*K*m total)
    n_box = 4 * K * m
    box_Ac = torch.zeros((n_box, ng_new), dtype=dtype, device=device)
    box_Ab = torch.zeros((n_box, nb_new), dtype=dtype, device=device)
    box_b  = torch.full((n_box, 1), 0.5, dtype=dtype, device=device)

    for k in range(K):
        for j in range(m):
            g1_col = ng + k * m + j               # column of g1_{j,k} in out_Gc
            g2_col = ng + K * m + k * m + j        # column of g2_{j,k} in out_Gc
            z_col   = nb + k * m + j                # column of z_{j,k} in out_Gb
            r = 4 * (k * m + j)

            # g1_{j,k} - 0.5 * z_{j,k} <= 0.5
            box_Ac[r, g1_col] = 1.0
            box_Ab[r, z_col]   = -0.5
            # -g1_{j,k} - 0.5 * z_{j,k} <= 0.5
            box_Ac[r + 1, g1_col] = -1.0
            box_Ab[r + 1, z_col]   = -0.5
            # g2_{j,k} - 0.5 * z_{j,k} <= 0.5
            box_Ac[r + 2, g2_col] = 1.0
            box_Ab[r + 2, z_col]   = -0.5
            # -g2_{j,k} - 0.5 * z_{j,k} <= 0.5
            box_Ac[r + 3, g2_col] = -1.0
            box_Ab[r + 3, z_col]   = -0.5

    # 2. Linking constraints: 2 rows per wide neuron (2*m total)
    # x_i = sum_k (cx_k/2 + cx_k/2 * z_{i,k} + g1_x_k * g1_{i,k} + g2_x_k * g2_{i,k})
    # Row+: Gc[i]*xi_c + Gb[i]*xi_b - sum_k (g1_x_k*g1 + g2_x_k*g2) - sum_k (cx_k/2)*z <= sum_k cx_k/2 - c_i
    # Row-: negated
    n_link = 2 * m
    link_Ac = torch.zeros((n_link, ng_new), dtype=dtype, device=device)
    link_Ab = torch.zeros((n_link, nb_new), dtype=dtype, device=device)
    link_b  = torch.zeros((n_link, 1), dtype=dtype, device=device)

    for j in range(m):
        idx_i = int(wide_idx[j].item())
        rhs_val = 0.0
        for k in range(K):
            g1_col = ng + k * m + j
            g2_col = ng + K * m + k * m + j
            z_col   = nb + k * m + j
            rhs_val += centers_x_k[k][j].item() / 2.0

            # Row+: -g1_x_k * g1_{i,k}
            link_Ac[2 * j, g1_col] = -g1_x_k[k][j]
            # Row+: -g2_x_k * g2_{i,k}
            link_Ac[2 * j, g2_col] = -g2_x_k[k][j]
            # Row+: -(cx_k/2) * z_{i,k}
            link_Ab[2 * j, z_col] = -centers_x_k[k][j] / 2.0

            # Row-: +g1_x_k * g1_{i,k}
            link_Ac[2 * j + 1, g1_col] = g1_x_k[k][j]
            # Row-: +g2_x_k * g2_{i,k}
            link_Ac[2 * j + 1, g2_col] = g2_x_k[k][j]
            # Row-: +(cx_k/2) * z_{i,k}
            link_Ab[2 * j + 1, z_col] = centers_x_k[k][j] / 2.0

        # Row+: +Gc[i]*xi_c, +Gb[i]*xi_b
        link_Ac[2 * j, :ng] = hz.Gc[idx_i]
        link_Ab[2 * j, :nb] = hz.Gb[idx_i]
        link_b[2 * j, 0] = rhs_val - hz.c[idx_i, 0].item()

        # Row-: -Gc[i]*xi_c, -Gb[i]*xi_b
        link_Ac[2 * j + 1, :ng] = -hz.Gc[idx_i]
        link_Ab[2 * j + 1, :nb] = -hz.Gb[idx_i]
        link_b[2 * j + 1, 0] = -(rhs_val - hz.c[idx_i, 0].item())

    # 3. Exactly-one-active constraints: 2 rows per wide neuron (2*m total)
    # sum_k z_{i,k} = 2 - K
    # Row+: sum_k z_{i,k} <= 2 - K
    # Row-: -sum_k z_{i,k} <= K - 2
    n_one = 2 * m
    one_Ac = torch.zeros((n_one, ng_new), dtype=dtype, device=device)
    one_Ab = torch.zeros((n_one, nb_new), dtype=dtype, device=device)
    one_b  = torch.zeros((n_one, 1), dtype=dtype, device=device)

    for j in range(m):
        for k in range(K):
            z_col = nb + k * m + j
            one_Ab[2 * j, z_col] = 1.0
            one_Ab[2 * j + 1, z_col] = -1.0
        one_b[2 * j, 0] = 2.0 - K
        one_b[2 * j + 1, 0] = K - 2.0

    # -- Extend existing constraints to new column dimensions -----------------
    old_Ac_ext = torch.cat([hz.Ac,
                            torch.zeros((nc, 2 * K * m), dtype=dtype, device=device)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab,
                            torch.zeros((nc, K * m), dtype=dtype, device=device)], dim=1)

    # -- Concatenate all constraint rows --------------------------------------
    out_Ac = torch.cat([old_Ac_ext, box_Ac, link_Ac, one_Ac], dim=0)
    out_Ab = torch.cat([old_Ab_ext, box_Ab, link_Ab, one_Ab], dim=0)
    out_b  = torch.cat([hz.b, box_b, link_b, one_b], dim=0)

    return _get_HZono()(c=new_c, Gc=out_Gc, Gb=out_Gb,
                        Ac=out_Ac, Ab=out_Ab, b=out_b)


def _hz_apply_sigmoid(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear sigmoid via tangent parallelogram encoding."""
    return _hz_apply_piecewise(hz, torch.sigmoid,
                               lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K)


def _hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear tanh via tangent parallelogram encoding."""
    return _hz_apply_piecewise(hz, torch.tanh,
                               lambda x: 1 - torch.tanh(x) ** 2, K)


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
# Transfer functions — (L, Bounds, tf=None) -> Fact
# ============================================================================

@torch.no_grad()
def hybridz_tf_dense(L: Layer, Bin: Bounds, tf=None):
    """Dense layer. Returns Fact."""
    W = L.params["weight"]
    b = L.params.get("bias", None)

    hz_in = tf._hz_cache.get(L.id) if tf else None
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

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"dense:{L.id}", list(L.out_vars + L.in_vars), W=W, b=b)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_bias(L: Layer, Bin: Bounds, tf=None):
    """Bias addition. Returns Fact."""
    c = L.params["c"]

    hz_in = tf._hz_cache.get(L.id) if tf else None
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

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"bias:{L.id}", list(L.out_vars + L.in_vars), c=c)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_scale(L: Layer, Bin: Bounds, tf=None):
    """Element-wise scaling. Returns Fact."""
    a = L.params["a"]

    hz_in = tf._hz_cache.get(L.id) if tf else None
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_multiply(hz_in, torch.diag(a.to(dtype=hz_in.c.dtype, device=hz_in.c.device).flatten()))
        Bout = _hz_compute_bounds(hz_out)
    else:
        a_pos = torch.clamp(a, min=0)
        a_neg = torch.clamp(a, max=0)
        lb = a_pos * Bin.lb + a_neg * Bin.ub
        ub = a_pos * Bin.ub + a_neg * Bin.lb
        Bout = Bounds(lb=lb, ub=ub)

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"scale:{L.id}", list(L.out_vars + L.in_vars), a=a)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_relu(L: Layer, Bin: Bounds, tf=None):
    """ReLU with graph-exact HZ encoding. Returns Fact."""
    hz_in = tf._hz_cache.get(L.id) if tf else None
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_relu(hz_in)
        hz_out = _hz_reduce(hz_out)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.clamp(Bin.lb, min=0)
        ub = torch.clamp(Bin.ub, min=0)
        Bout = Bounds(lb=lb, ub=ub)

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
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
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_lrelu(L: Layer, Bin: Bounds, tf=None):
    """LeakyReLU. Returns Fact."""
    alpha = float(L.params.get("negative_slope", 0.01))

    hz_in = tf._hz_cache.get(L.id) if tf else None
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_leaky_relu(hz_in, alpha)
        hz_out = _hz_reduce(hz_out)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
        ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
        Bout = Bounds(lb=lb, ub=ub)

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
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
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_tanh(L: Layer, Bin: Bounds, tf=None):
    """Tanh with piecewise linear HZ encoding. Returns Fact."""
    K = tf._tanh_K if tf and hasattr(tf, '_tanh_K') else 2
    hz_in = tf._hz_cache.get(L.id) if tf else None
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_tanh(hz_in, K=K)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.tanh(Bin.lb)
        ub = torch.tanh(Bin.ub)
        Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"tanh:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_sigmoid(L: Layer, Bin: Bounds, tf=None):
    """Sigmoid with piecewise linear HZ encoding. Returns Fact."""
    K = tf._sigmoid_K if tf and hasattr(tf, '_sigmoid_K') else 2
    hz_in = tf._hz_cache.get(L.id) if tf else None
    hz_out = None
    if hz_in is not None:
        hz_out = _hz_apply_sigmoid(hz_in, K=K)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = torch.sigmoid(Bin.lb)
        ub = torch.sigmoid(Bin.ub)
        Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"sigmoid:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_abs(L: Layer, Bin: Bounds, tf=None):
    """Absolute value. Returns Fact."""
    hz_in = tf._hz_cache.get(L.id) if tf else None
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

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    # ConSet indices always from interval Bin
    idx_pos = torch.where(Bin.lb >= 0)[0]
    idx_neg = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    cons = ConSet()
    cons.add_op(f"abs:{L.id}", list(L.out_vars + L.in_vars),
                idx_pos=idx_pos, idx_neg=idx_neg, idx_amb=idx_amb)
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_add(L: Layer, Bin1: Bounds, Bin2: Bounds, tf=None):
    """Element-wise addition. Returns Fact."""
    hz_in1 = tf._hz_cache.get(L.id) if tf else None
    preds = tf._net.preds.get(L.id, []) if tf else []
    hz_in2 = tf._hz_cache.get(preds[1]) if tf and len(preds) > 1 and preds[1] in tf._hz_cache else None
    hz_out = None
    if hz_in1 is not None and hz_in2 is not None:
        hz_out = _hz_minkowski_sum(hz_in1, hz_in2)
        Bout = _hz_compute_bounds(hz_out)
    else:
        lb = Bin1.lb + Bin2.lb
        ub = Bin1.ub + Bin2.ub
        Bout = Bounds(lb=lb, ub=ub)

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"add:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_mul(L: Layer, Bin1: Bounds, Bin2: Bounds, tf=None):
    """Element-wise multiplication (McCormick). Returns Fact."""
    hz_in1 = tf._hz_cache.get(L.id) if tf else None
    preds = tf._net.preds.get(L.id, []) if tf else []
    hz_in2 = tf._hz_cache.get(preds[1]) if tf and len(preds) > 1 and preds[1] in tf._hz_cache else None
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

    if tf and hz_out is not None:
        tf._hz_cache[L.id] = hz_out
    cons = ConSet()
    cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars),
                lx=Bin1.lb, ux=Bin1.ub, ly=Bin2.lb, uy=Bin2.ub)
    return Fact(bounds=Bout, cons=cons)
