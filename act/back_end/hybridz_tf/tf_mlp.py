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
        model.addConstr(lhs == b_np)

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

    A_eq = np.concatenate([Ac_np, Ab_np], axis=1) if (Ac_np.size or Ab_np.size) else None
    b_eq = b_np if (A_eq is not None) else None
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)

    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)

        res_min = linprog(c=obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs")
        if not res_min.success:
            raise RuntimeError(f"[linprog] MIN infeasible at dim {i}: {res_min.message}")
        LB[i] = c_np[i] + res_min.fun

        res_max = linprog(c=-obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs")
        if not res_max.success:
            raise RuntimeError(f"[linprog] MAX infeasible at dim {i}: {res_max.message}")
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    lb = torch.from_numpy(LB).to(device=device, dtype=dtype).flatten()
    ub = torch.from_numpy(UB).to(device=device, dtype=dtype).flatten()
    return Bounds(lb=lb, ub=ub)


# ---- Nonlinear activations --------------------------------------------------

def _hz_apply_relu(hz: HZono) -> HZono:
    """Exact ReLU via equality constraints + linking equality.

    Per unstable neuron i with bounds [α, β] (α < 0 < β):
      ng += 4 (ξ1, ξ2, ξ3, ξ4)
      nb += 1 (z)
      nc += 3 equalities:
        (1) ξ1 + ξ3 + z = 1
        (2) ξ2 + ξ4 - z = 1
        (3) α/2·ξ1 - β/2·ξ2 + α/2·z - Gc[i]·ξ_old - Gb[i]·ξ_b_old = c_i - β/2

    When z=1 (inactive): ξ2=1 forced → y=0, x∈[α,0]
    When z=-1 (active): ξ1=1 forced, linking gives y=x, x∈[0,β]
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

    # ---- Stable neurons: identity (active) or zero (inactive) ----
    out_Gc = torch.zeros((n, ng + 4 * k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c = torch.zeros((n, 1), dtype=dtype, device=device)

    # Active: y = x (copy original generators)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]
    # Inactive: y = 0 (already zeros)

    if k == 0:
        return _get_HZono()(c=out_c, Gc=out_Gc[:, :ng], Gb=out_Gb[:, :nb],
                            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # ---- Unstable neurons ----
    unstable_idx = torch.where(unstable)[0]
    alpha = lb[unstable_idx]  # (k,) negative
    beta  = ub[unstable_idx]  # (k,) positive
    t = torch.arange(k, device=device)

    # Column indices for new generators
    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z   = nb + t

    # Output: y_i = β/2 + (-β/2)·ξ2
    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0

    # ---- 3 equality constraints per unstable neuron (3k total) ----
    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = torch.zeros((3 * k, ng_new), dtype=dtype, device=device)
    eq_Ab = torch.zeros((3 * k, nb_new), dtype=dtype, device=device)
    eq_b  = torch.zeros((3 * k, 1), dtype=dtype, device=device)

    r1 = 3 * t       # graph eq 1
    r2 = 3 * t + 1   # graph eq 2
    r3 = 3 * t + 2   # linking eq

    # Eq 1: ξ1 + ξ3 + z = 1
    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z]   = 1.0
    eq_b[r1, 0]        = 1.0

    # Eq 2: ξ2 + ξ4 - z = 1
    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z]   = -1.0
    eq_b[r2, 0]        = 1.0

    # Eq 3: α/2·ξ1 - β/2·ξ2 + α/2·z - Gc[i]·ξ_old - Gb[i]·ξ_b_old = c_i - β/2
    for j in range(k):
        idx_i = int(unstable_idx[j].item())
        eq_Ac[3 * j + 2, col_xi1[j]] = alpha[j] / 2.0
        eq_Ac[3 * j + 2, col_xi2[j]] = -beta[j] / 2.0
        eq_Ac[3 * j + 2, :ng] -= hz.Gc[idx_i]  # -Gc[i]·ξ_old (use -= since row was zeros)
        eq_Ab[3 * j + 2, :nb] -= hz.Gb[idx_i]   # -Gb[i]·ξ_b_old
        eq_Ab[3 * j + 2, col_z[j]] = alpha[j] / 2.0
        eq_b[3 * j + 2, 0] = hz.c[idx_i, 0] - beta[j] / 2.0

    # Extend old constraints to new column dimensions
    old_Ac_ext = torch.cat([hz.Ac, torch.zeros((nc, 4 * k), dtype=dtype, device=device)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)

    out_Ac = torch.cat([old_Ac_ext, eq_Ac], dim=0)
    out_Ab = torch.cat([old_Ab_ext, eq_Ab], dim=0)
    out_b  = torch.cat([hz.b, eq_b], dim=0)

    return _get_HZono()(c=out_c, Gc=out_Gc, Gb=out_Gb,
                        Ac=out_Ac, Ab=out_Ab, b=out_b)


def _hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Exact LeakyReLU via equality constraints + box equalities with slack.

    Per unstable neuron i with bounds [l, u] (l < 0 < u):
      ng += 6 (g1, g2 real + s1+, s1-, s2+, s2- slack)
      nb += 1 (z)
      nc += 5 equalities:
        (1-2) g1 + s1+ + 0.5z = 0.5,  -g1 + s1- + 0.5z = 0.5  (g1 active when z=+1)
        (3-4) g2 + s2+ - 0.5z = 0.5,  -g2 + s2- - 0.5z = 0.5  (g2 active when z=-1)
        (5)   linking equality

    When z=-1 (active):  g1=0, g2 free → y = u/2·(1-g2) = x   ∈ [0, u]
    When z=+1 (inactive): g2=0, g1 free → y = αl/2·(1+g1) = αx ∈ [αl, 0]
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    a = alpha_arg  # negative slope

    bounds = _hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active   = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    # ---- Stable neurons ----
    out_Gc = torch.zeros((n, ng + 6 * k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c  = torch.zeros((n, 1), dtype=dtype, device=device)

    # Active: y = x
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    # Inactive: y = alpha*x
    if inactive.any():
        out_c[inactive] = a * hz.c[inactive]
        out_Gc[inactive, :ng] = a * hz.Gc[inactive]
        out_Gb[inactive, :nb] = a * hz.Gb[inactive]

    if k == 0:
        return _get_HZono()(c=out_c, Gc=out_Gc[:, :ng], Gb=out_Gb[:, :nb],
                            Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # ---- Unstable neurons ----
    unstable_idx = torch.where(unstable)[0]
    l = lb[unstable_idx]  # (k,) negative
    u = ub[unstable_idx]  # (k,) positive
    t = torch.arange(k, device=device)

    # Column indices
    col_g1 = ng + t                 # real generator g1
    col_g2 = ng + k + t             # real generator g2
    col_s1p = ng + 2 * k + t        # slack s1+
    col_s1m = ng + 3 * k + t        # slack s1-
    col_s2p = ng + 4 * k + t        # slack s2+
    col_s2m = ng + 5 * k + t        # slack s2-
    col_z  = nb + t                  # binary z

    # Output center and generators
    # c_y = (u + a*l) / 4
    out_c[unstable_idx, 0] = (u + a * l) / 4.0

    # g1 y-contribution: a*l/2 (inactive branch)
    out_Gc[unstable_idx, col_g1] = a * l / 2.0
    # g2 y-contribution: -u/2 (active branch)
    out_Gc[unstable_idx, col_g2] = -u / 2.0
    # slack: zero output (already zeros)

    # Binary y-contribution: (a*l - u) / 4
    out_Gb[unstable_idx, col_z] = (a * l - u) / 4.0

    # ---- 5 equality constraints per unstable neuron ----
    ng_total = ng + 6 * k
    nb_total = nb + k

    eq_Ac = torch.zeros((5 * k, ng_total), dtype=dtype, device=device)
    eq_Ab = torch.zeros((5 * k, nb_total), dtype=dtype, device=device)
    eq_b  = torch.zeros((5 * k, 1), dtype=dtype, device=device)

    r0 = 5 * t       # box eq for g1 (+)
    r1 = 5 * t + 1   # box eq for g1 (-)
    r2 = 5 * t + 2   # box eq for g2 (+)
    r3 = 5 * t + 3   # box eq for g2 (-)
    r4 = 5 * t + 4   # linking eq

    # Box eq 0: g1 + s1+ + 0.5*z = 0.5  (g1 active when z=+1)
    eq_Ac[r0, col_g1] = 1.0
    eq_Ac[r0, col_s1p] = 1.0
    eq_Ab[r0, col_z] = 0.5
    eq_b[r0, 0] = 0.5

    # Box eq 1: -g1 + s1- + 0.5*z = 0.5
    eq_Ac[r1, col_g1] = -1.0
    eq_Ac[r1, col_s1m] = 1.0
    eq_Ab[r1, col_z] = 0.5
    eq_b[r1, 0] = 0.5

    # Box eq 2: g2 + s2+ - 0.5*z = 0.5  (g2 active when z=-1)
    eq_Ac[r2, col_g2] = 1.0
    eq_Ac[r2, col_s2p] = 1.0
    eq_Ab[r2, col_z] = -0.5
    eq_b[r2, 0] = 0.5

    # Box eq 3: -g2 + s2- - 0.5*z = 0.5
    eq_Ac[r3, col_g2] = -1.0
    eq_Ac[r3, col_s2m] = 1.0
    eq_Ab[r3, col_z] = -0.5
    eq_b[r3, 0] = 0.5

    # Linking eq 4:
    # Gc[i]*ξ_old + Gb[i]*ζ_old - (l/2)*g1 + (u/2)*g2 - ((l-u)/4)*z = (u+l)/4 - c_i
    for j in range(k):
        idx_i = int(unstable_idx[j].item())
        eq_Ac[5 * j + 4, :ng] = hz.Gc[idx_i]
        eq_Ac[5 * j + 4, col_g1[j]] = -l[j] / 2.0
        eq_Ac[5 * j + 4, col_g2[j]] = u[j] / 2.0
        eq_Ab[5 * j + 4, :nb] = hz.Gb[idx_i]
        eq_Ab[5 * j + 4, col_z[j]] = -(l[j] - u[j]) / 4.0  # = (u-l)/4
        eq_b[5 * j + 4, 0] = (u[j] + l[j]) / 4.0 - hz.c[idx_i, 0]

    # ---- Extend old constraints ----
    old_Ac_ext = torch.cat([hz.Ac, torch.zeros((nc, 6 * k), dtype=dtype, device=device)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)

    out_Ac = torch.cat([old_Ac_ext, eq_Ac], dim=0)
    out_Ab = torch.cat([old_Ab_ext, eq_Ab], dim=0)
    out_b  = torch.cat([hz.b, eq_b], dim=0)

    return _get_HZono()(c=out_c, Gc=out_Gc, Gb=out_Gb,
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

    # -- New continuous generators: 6*K*m columns per wide neuron ---------------
    # Layout per piece k: g1 (m cols), g2 (m cols) = 2K*m "real" columns
    #   then slack: s1+ (m), s1- (m), s2+ (m), s2- (m) = 4K*m "slack" columns
    # Real columns: ng + [0, 2K*m), Slack columns: ng + 2K*m + [0, 4K*m)
    n_real = 2 * K * m
    n_slack = 4 * K * m
    Gc_new = torch.zeros((n, n_real + n_slack), dtype=dtype, device=device)

    for k in range(K):
        g1_cols = torch.arange(k * m, (k + 1) * m, device=device)
        g2_cols = torch.arange(K * m + k * m, K * m + (k + 1) * m, device=device)
        for j in range(m):
            idx_i = wide_idx[j]
            Gc_new[idx_i, g1_cols[j]] = g1_y_k[k][j]
            Gc_new[idx_i, g2_cols[j]] = g2_y_k[k][j]
    # Slack generators: zero in output rows (already zeros)

    # -- New binary generators: K*m columns (z_{i,k}) -------------------------
    Gb_new = torch.zeros((n, K * m), dtype=dtype, device=device)
    for k in range(K):
        z_cols = torch.arange(k * m, (k + 1) * m, device=device)
        for j in range(m):
            idx_i = wide_idx[j]
            Gb_new[idx_i, z_cols[j]] = -centers_y_k[k][j] / 2.0

    # -- Assemble output generators -------------------------------------------
    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)   # (n, ng + 6K*m)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)    # (n, nb + K*m)

    ng_total = ng + n_real + n_slack   # ng + 6K*m
    nb_total = nb + K * m

    # ---- New equality constraints: (4K + 2) rows per wide neuron -------------
    #
    # 1. Box equalities (4K rows/neuron): per real generator, 2 slack + equality
    #    g_{k,j} + s+ - 0.5*z_k = 0.5     (active when z_k=1: g free; z_k=-1: g=0)
    #   -g_{k,j} + s- - 0.5*z_k = 0.5
    #
    # 2. Linking equality (1 row/neuron): x_i from original gens = piecewise x
    #
    # 3. Exactly-one equality (1 row/neuron): sum_k z_k = 2-K
    #
    n_box = 4 * K * m
    n_link = m
    n_one = m
    n_eq_total = n_box + n_link + n_one

    eq_Ac = torch.zeros((n_eq_total, ng_total), dtype=dtype, device=device)
    eq_Ab = torch.zeros((n_eq_total, nb_total), dtype=dtype, device=device)
    eq_b  = torch.zeros((n_eq_total, 1), dtype=dtype, device=device)

    # 1. Box equalities: 4K*m rows
    for k in range(K):
        for j in range(m):
            g1_col = ng + k * m + j
            g2_col = ng + K * m + k * m + j
            z_col  = nb + k * m + j
            # Slack columns for piece k, neuron j:
            # s1+ at ng + n_real + (k*m+j)*4 + 0
            # s1- at ng + n_real + (k*m+j)*4 + 1
            # s2+ at ng + n_real + (k*m+j)*4 + 2
            # s2- at ng + n_real + (k*m+j)*4 + 3
            s_base = ng + n_real + (k * m + j) * 4
            r = 4 * (k * m + j)

            # g1 + s1+ - 0.5*z = 0.5
            eq_Ac[r, g1_col] = 1.0
            eq_Ac[r, s_base] = 1.0
            eq_Ab[r, z_col] = -0.5
            eq_b[r, 0] = 0.5

            # -g1 + s1- - 0.5*z = 0.5
            eq_Ac[r + 1, g1_col] = -1.0
            eq_Ac[r + 1, s_base + 1] = 1.0
            eq_Ab[r + 1, z_col] = -0.5
            eq_b[r + 1, 0] = 0.5

            # g2 + s2+ - 0.5*z = 0.5
            eq_Ac[r + 2, g2_col] = 1.0
            eq_Ac[r + 2, s_base + 2] = 1.0
            eq_Ab[r + 2, z_col] = -0.5
            eq_b[r + 2, 0] = 0.5

            # -g2 + s2- - 0.5*z = 0.5
            eq_Ac[r + 3, g2_col] = -1.0
            eq_Ac[r + 3, s_base + 3] = 1.0
            eq_Ab[r + 3, z_col] = -0.5
            eq_b[r + 3, 0] = 0.5

    # 2. Linking equality: 1 row per neuron (m rows)
    # With Ab=-0.5 in box equalities, z_k=-1 means piece k active.
    # x_i = Σ_k [(1-z_k)/2 * (cx_k + g1x*g1 + g2x*g2)]
    #      = Σ_k cx_k/2 - Σ_k cx_k/2*z_k + Σ_k(g1x*g1 + g2x*g2)
    # So: Gc[i]*ξ + Gb[i]*ζ - Σ_k(g1x*g1+g2x*g2) + Σ_k(cx_k/2)*z_k = Σ_k cx_k/2 - c_i
    for j in range(m):
        idx_i = int(wide_idx[j].item())
        r = n_box + j
        rhs_val = 0.0

        for k in range(K):
            g1_col = ng + k * m + j
            g2_col = ng + K * m + k * m + j
            z_col  = nb + k * m + j
            rhs_val += centers_x_k[k][j].item() / 2.0

            eq_Ac[r, g1_col] = -g1_x_k[k][j]
            eq_Ac[r, g2_col] = -g2_x_k[k][j]
            eq_Ab[r, z_col] = centers_x_k[k][j] / 2.0  # positive (z=-1 active)

        eq_Ac[r, :ng] = hz.Gc[idx_i]
        eq_Ab[r, :nb] = hz.Gb[idx_i]
        eq_b[r, 0] = rhs_val - hz.c[idx_i, 0].item()

    # 3. Exactly-one equality: 1 row per neuron (m rows)
    # Σ_k z_k = 2 - K
    for j in range(m):
        r = n_box + n_link + j
        for k in range(K):
            z_col = nb + k * m + j
            eq_Ab[r, z_col] = 1.0
        eq_b[r, 0] = float(K - 2)

    # -- Extend existing constraints to new column dimensions -----------------
    old_Ac_ext = torch.cat([hz.Ac,
                            torch.zeros((nc, n_real + n_slack), dtype=dtype, device=device)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab,
                            torch.zeros((nc, K * m), dtype=dtype, device=device)], dim=1)

    # -- Concatenate all constraint rows --------------------------------------
    out_Ac = torch.cat([old_Ac_ext, eq_Ac], dim=0)
    out_Ab = torch.cat([old_Ab_ext, eq_Ab], dim=0)
    out_b  = torch.cat([hz.b, eq_b], dim=0)

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


# ---- Complexity reductio ---------------------------

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
