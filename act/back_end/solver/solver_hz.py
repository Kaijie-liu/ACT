from __future__ import annotations
import math, time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from act.back_end.solver.solver_base import Solver, SolveStatus, SolverCaps
from act.back_end.core import Bounds
from act.util.device_manager import get_default_device, get_default_dtype

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# HZono dataclass
# ============================================================================

@dataclass
class HZono:
    """Hybrid Zonotope data container.

    Z = {c + Gc @ xi_c + Gb @ xi_b | Ac @ xi_c + Ab @ xi_b = b,
         xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}
    """
    c:  torch.Tensor   # (n, 1)   center vector
    Gc: torch.Tensor   # (n, ng)  continuous generator matrix
    Gb: torch.Tensor   # (n, nb)  binary generator matrix
    Ac: torch.Tensor   # (nc, ng) continuous constraint matrix
    Ab: torch.Tensor   # (nc, nb) binary constraint matrix
    b:  torch.Tensor   # (nc, 1)  constraint RHS vector


# ============================================================================
# HZSolver — middle-precision solver (Solver subclass)
#
# Part A: Full Solver interface (LP via Torch+Adam, replaces TorchLPSolver)
# Part B: HZ abstract domain operations (bounds tightening for HybridzTF)
# ============================================================================

class HZSolver(Solver):
    """Middle-precision solver: LP solving + HZ abstract domain bounds tightening.

    Part A implements the full Solver interface for the verification pipeline
    (verify_once, export_to_solver). This is the LP solver formerly TorchLPSolver.

    Part B provides HZ domain operations used by HybridzTF for tighter bounds.
    HZ bounds computation uses LP relaxation (SciPy) or unconstrained fast path,
    not Gurobi MILP — that belongs to the most-precise tier (solver_gurobi.py).
    """

    _HZ_MAX_INPUT_DIM = 1024

    def __init__(self, tanh_K: int = 2, sigmoid_K: int = 2):
        # LP solver state
        self._device = get_default_device()
        self._dtype = get_default_dtype()
        self._n = 0
        self._x = None
        self._lb = None
        self._ub = None
        self._eq = []
        self._le = []
        self._ge = []
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None
        self._timelimit = None
        # LP solver parameters
        self.rho_eq = 10.0
        self.rho_ineq = 10.0
        self.max_iter = 2000
        self.tol_feas = 1e-4
        self.lr = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0
        self._large_n_threshold = 20000
        self._large_n_max_iter = 800
        self._large_n_tol = 1e-3
        self._log_every = 200
        self._stagnation_patience = 300
        self._stagnation_tol = 1e-5
        self._feas_check_stride = 5
        # HZ config
        self._tanh_K = tanh_K
        self._sigmoid_K = sigmoid_K

    # ================================================================
    # Part A: Full Solver interface (LP via Torch+Adam)
    # ================================================================

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True)

    @property
    def n(self) -> int:
        return self._n

    def begin(self, name: str = "verify", device: Optional[str] = None):
        if device is not None:
            self._device = torch.device(device)
        self._n = 0
        self._x = None
        self._lb = None
        self._ub = None
        self._eq.clear(); self._le.clear(); self._ge.clear()
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None

    def add_vars(self, n: int) -> None:
        if n <= 0:
            return
        if self._n == 0:
            self._n = n
            self._lb = torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)
            self._ub = torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)
        else:
            self._n += n
            self._lb = torch.cat([self._lb, torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)])
            self._ub = torch.cat([self._ub, torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)])

    def add_binary_vars(self, n: int) -> List[int]:
        start = self._n
        self.add_vars(n)
        idxs = list(range(start, start + n))
        self._lb[idxs] = 0.0
        self._ub[idxs] = 1.0
        return idxs

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:
        lb_t = torch.as_tensor(lb, device=self._device, dtype=self._dtype)
        ub_t = torch.as_tensor(ub, device=self._device, dtype=self._dtype)
        self._lb[idxs] = lb_t
        self._ub[idxs] = ub_t

    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._eq.append((vids, coeffs, rhs))

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, coeffs, rhs))

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, [-float(a) for a in coeffs], -float(rhs)))

    def add_sum_eq(self, vids: List[int], rhs: float) -> None:
        self.add_lin_eq(vids, [1.0] * len(vids), rhs)

    def add_ge_zero(self, vids: List[int]) -> None:
        for i in vids:
            self.add_lin_le([i], [-1.0], 0.0)

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:
        return  # no-op

    def set_objective_linear(self, vids: List[int], coeffs: List[float], const: float = 0.0, sense: str = "min") -> None:
        self._objective = (vids, coeffs, float(const), "min" if sense != "max" else "max")

    def optimize(self, timelimit: Optional[float] = None) -> None:
        self._timelimit = timelimit
        t_end = None if timelimit is None else (time.time() + float(timelimit))

        eff_max_iter = self.max_iter
        eff_tol_feas = self.tol_feas
        if self._n > self._large_n_threshold:
            eff_max_iter = min(eff_max_iter, self._large_n_max_iter)
            eff_tol_feas = max(eff_tol_feas, self._large_n_tol)

        if self._x is None:
            lb = torch.where(torch.isfinite(self._lb), self._lb, torch.zeros_like(self._lb))
            ub = torch.where(torch.isfinite(self._ub), self._ub, torch.zeros_like(self._ub))
            mid = 0.5 * (lb + ub)
            both_inf = (~torch.isfinite(self._lb)) & (~torch.isfinite(self._ub))
            mid = torch.where(both_inf, torch.zeros_like(mid), mid)
            self._x = torch.nn.Parameter(mid.clone().to(device=self._device, dtype=self._dtype), requires_grad=True)
        else:
            self._x = torch.nn.Parameter(self._x.detach().to(device=self._device, dtype=self._dtype), requires_grad=True)

        vids, coeffs, c0, sense = self._objective
        is_max = (sense == "max")
        if vids:
            vids_t = torch.as_tensor(vids, device=self._device, dtype=torch.long)
            coeffs_t = torch.as_tensor(coeffs, device=self._device, dtype=self._dtype)
        else:
            vids_t = None
            coeffs_t = None
        const_term = float(c0)

        lb = self._lb.to(device=self._device, dtype=self._dtype)
        ub = self._ub.to(device=self._device, dtype=self._dtype)

        def build_sparse(rows):
            if not rows:
                return None, torch.zeros((0,), device=self._device, dtype=self._dtype)
            m = len(rows)
            b = torch.empty((m,), device=self._device, dtype=self._dtype)
            ri, ci, vals = [], [], []
            for r, (vids_row, coeffs_row, rhs) in enumerate(rows):
                b[r] = float(rhs)
                ri.extend([r] * len(vids_row))
                ci.extend(vids_row)
                vals.extend(coeffs_row)
            if not vals:
                idx = torch.zeros((2, 0), device=self._device, dtype=torch.long)
                val_t = torch.zeros((0,), device=self._device, dtype=self._dtype)
            else:
                idx = torch.tensor([ri, ci], device=self._device, dtype=torch.long)
                val_t = torch.as_tensor(vals, device=self._device, dtype=self._dtype)
            A = torch.sparse_coo_tensor(idx, val_t, (m, self._n), device=self._device, dtype=self._dtype)
            return A.coalesce(), b

        Aeq, beq = build_sparse(self._eq)
        Ale, ble = build_sparse(self._le)

        opt = torch.optim.Adam([self._x], lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        best_max_viol = math.inf
        stagnation_steps = 0
        viol_eq = None
        viol_le = None

        for it in range(eff_max_iter):
            if t_end is not None and time.time() >= t_end:
                break

            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                obj = 0.001 * (self._x * self._x).sum() + const_term
                if vids_t is not None and coeffs_t is not None and vids_t.numel() > 0:
                    obj = obj + torch.dot(coeffs_t, self._x.index_select(0, vids_t))
                if is_max:
                    obj = -obj

                viol_eq = None if Aeq is None else torch.sparse.mm(Aeq, self._x.unsqueeze(1)).squeeze(1) - beq
                if viol_eq is not None:
                    obj = obj + self.rho_eq * (viol_eq * viol_eq).sum()

                viol_le = None if Ale is None else torch.sparse.mm(Ale, self._x.unsqueeze(1)).squeeze(1) - ble
                if viol_le is not None:
                    obj = obj + self.rho_ineq * (torch.relu(viol_le) ** 2).sum()

                obj.backward()
            opt.step()

            with torch.no_grad():
                self._x.data.clamp_(lb, ub)

            recompute = (it % self._feas_check_stride == 0)
            with torch.no_grad():
                if recompute:
                    v_eq = None if Aeq is None else torch.sparse.mm(Aeq, self._x.unsqueeze(1)).squeeze(1) - beq
                    v_le = None if Ale is None else torch.sparse.mm(Ale, self._x.unsqueeze(1)).squeeze(1) - ble
                else:
                    v_eq = viol_eq
                    v_le = viol_le

                max_viol = 0.0
                if v_eq is not None and v_eq.numel() > 0:
                    max_viol = max(max_viol, float(v_eq.abs().max().item()))
                if v_le is not None and v_le.numel() > 0:
                    max_viol = max(max_viol, float(torch.relu(v_le).max().item()))

            if max_viol < best_max_viol - self._stagnation_tol:
                best_max_viol = max_viol
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            if max_viol <= eff_tol_feas or stagnation_steps >= self._stagnation_patience:
                break

        self._status = SolveStatus.SAT
        self._has_solution = True
        self._sol = self._x.detach().clone()

    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return bool(self._has_solution)

    def get_values(self, vids: List[int]) -> np.ndarray:
        assert self._sol is not None, "No solution available"
        with torch.no_grad():
            return self._sol[vids].detach().cpu().to(self._dtype).numpy()

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        return self.get_values(input_ids)

    # ================================================================
    # Part B: HZ abstract domain operations
    # ================================================================

    # ---- Construction ----

    def hz_from_bounds(self, bounds: Bounds, dtype, device) -> Optional[HZono]:
        """Create HZono from Bounds. Returns None if dim > _HZ_MAX_INPUT_DIM."""
        lb = bounds.lb.flatten().to(dtype=dtype, device=device)
        ub = bounds.ub.flatten().to(dtype=dtype, device=device)
        n = lb.shape[0]
        if n > self._HZ_MAX_INPUT_DIM:
            return None
        c = ((lb + ub) / 2.0).view(-1, 1)
        rad = (ub - lb) / 2.0
        return HZono(
            c=c, Gc=torch.diag(rad),
            Gb=torch.zeros((n, 0), dtype=dtype, device=device),
            Ac=torch.zeros((0, n), dtype=dtype, device=device),
            Ab=torch.zeros((0, 0), dtype=dtype, device=device),
            b=torch.zeros((0, 1), dtype=dtype, device=device),
        )

    # ---- Algebra ----

    def hz_multiply(self, hz: HZono, R: torch.Tensor) -> HZono:
        R = R.to(dtype=hz.c.dtype, device=hz.c.device)
        return HZono(c=R @ hz.c, Gc=R @ hz.Gc, Gb=R @ hz.Gb,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    def hz_add_const(self, hz: HZono, v: torch.Tensor) -> HZono:
        v = v.to(dtype=hz.c.dtype, device=hz.c.device)
        if v.ndim == 1:
            v = v.view(-1, 1)
        return HZono(c=hz.c + v, Gc=hz.Gc.clone(), Gb=hz.Gb.clone(),
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    def hz_minkowski_sum(self, hz1: HZono, hz2: HZono) -> HZono:
        dtype, device = hz1.c.dtype, hz1.c.device
        new_c = hz1.c + hz2.c.to(dtype=dtype, device=device)
        new_Gc = torch.cat([hz1.Gc, hz2.Gc.to(dtype=dtype, device=device)], dim=1)
        new_Gb = torch.cat([hz1.Gb, hz2.Gb.to(dtype=dtype, device=device)], dim=1)
        nc1, nc2 = hz1.Ac.shape[0], hz2.Ac.shape[0]
        ng1, ng2 = hz1.Gc.shape[1], hz2.Gc.shape[1]
        nb1, nb2 = hz1.Gb.shape[1], hz2.Gb.shape[1]
        Ac_top = torch.cat([hz1.Ac, torch.zeros((nc1, ng2), dtype=dtype, device=device)], dim=1)
        Ac_bot = torch.cat([torch.zeros((nc2, ng1), dtype=dtype, device=device),
                             hz2.Ac.to(dtype=dtype, device=device)], dim=1)
        Ab_top = torch.cat([hz1.Ab, torch.zeros((nc1, nb2), dtype=dtype, device=device)], dim=1)
        Ab_bot = torch.cat([torch.zeros((nc2, nb1), dtype=dtype, device=device),
                             hz2.Ab.to(dtype=dtype, device=device)], dim=1)
        return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb,
                     Ac=torch.cat([Ac_top, Ac_bot], dim=0),
                     Ab=torch.cat([Ab_top, Ab_bot], dim=0),
                     b=torch.cat([hz1.b, hz2.b.to(dtype=dtype, device=device)], dim=0))

    # ---- Activations ----

    def hz_apply_relu(self, hz: HZono) -> HZono:
        return _hz_apply_relu(hz, self.hz_compute_bounds)

    def hz_apply_leaky_relu(self, hz: HZono, alpha: float) -> HZono:
        return _hz_apply_leaky_relu(hz, alpha, self.hz_compute_bounds)

    def hz_apply_piecewise(self, hz: HZono, func, dfunc, K: int = 2) -> HZono:
        return _hz_apply_piecewise(hz, func, dfunc, K, self.hz_compute_bounds)

    def hz_apply_tanh(self, hz: HZono, K: int = None) -> HZono:
        K = K or self._tanh_K
        return self.hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K)

    def hz_apply_sigmoid(self, hz: HZono, K: int = None) -> HZono:
        K = K or self._sigmoid_K
        return self.hz_apply_piecewise(hz, torch.sigmoid,
                                       lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K)

    # ---- Conv2d ----

    def hz_conv2d(self, hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape) -> HZono:
        dtype, device = hz.c.dtype, hz.c.device
        C, H, W = _parse_input_shape(input_shape)
        weight = weight.to(dtype=dtype, device=device)
        c_img = hz.c.view(C, H, W).unsqueeze(0)
        out_c = F.conv2d(c_img, weight,
                         bias=bias.to(dtype=dtype, device=device) if bias is not None else None,
                         stride=stride, padding=padding, dilation=dilation, groups=groups)
        new_c = out_c.reshape(-1, 1)
        n_out = new_c.shape[0]
        new_Gc = _conv2d_generators(hz.Gc, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device)
        new_Gb = _conv2d_generators(hz.Gb, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device)
        return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    # ---- Order reduction ----

    def hz_reduce(self, hz: HZono, max_order: float = 10.0) -> HZono:
        return _hz_reduce(hz, max_order)

    # ---- Bounds computation (middle precision: LP relaxation, no Gurobi) ----

    def hz_compute_bounds(self, hz: HZono) -> Bounds:
        if _hz_is_unconstrained(hz):
            return _hz_bounds_unconstrained(hz)
        if _HAS_SCIPY:
            try:
                return _hz_bounds_lp(hz)
            except Exception:
                pass
        return _hz_bounds_unconstrained(hz)


# ============================================================================
# Module-level private helpers
# ============================================================================

def _hz_is_unconstrained(hz: HZono) -> bool:
    tol = 1e-12
    return (torch.all(torch.abs(hz.Ac) < tol).item()
            and torch.all(torch.abs(hz.Ab) < tol).item()
            and torch.all(torch.abs(hz.b) < tol).item())


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
    n = hz.c.shape[0]
    dtype, device = hz.c.dtype, hz.c.device
    absGc = hz.Gc.abs().sum(dim=1, keepdim=True) if hz.Gc.numel() else torch.zeros((n, 1), dtype=dtype, device=device)
    absGb = hz.Gb.abs().sum(dim=1, keepdim=True) if hz.Gb.numel() else torch.zeros((n, 1), dtype=dtype, device=device)
    rad = absGc + absGb
    return Bounds(lb=(hz.c - rad).flatten(), ub=(hz.c + rad).flatten())


def _hz_bounds_lp(hz: HZono) -> Bounds:
    """LP relaxation: treat binary generators as continuous [-1,1]."""
    n = int(hz.c.shape[0]);  p = int(hz.Gc.shape[1]);  q = int(hz.Gb.shape[1])
    c_np = hz.c.detach().cpu().numpy().astype('float64').reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype('float64')
    Gb_np = hz.Gb.detach().cpu().numpy().astype('float64')
    Ac_np = hz.Ac.detach().cpu().numpy().astype('float64')
    Ab_np = hz.Ab.detach().cpu().numpy().astype('float64')
    b_np = hz.b.detach().cpu().numpy().astype('float64').reshape(-1)

    A_eq = np.concatenate([Ac_np, Ab_np], axis=1) if (Ac_np.size or Ab_np.size) else None
    b_eq = b_np if (A_eq is not None) else None
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64);  UB = np.empty((n,), dtype=np.float64)
    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)
        res_min = linprog(c=obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs")
        if not res_min.success:
            raise RuntimeError(f"LP MIN infeasible at dim {i}: {res_min.message}")
        LB[i] = c_np[i] + res_min.fun
        res_max = linprog(c=-obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs")
        if not res_max.success:
            raise RuntimeError(f"LP MAX infeasible at dim {i}: {res_max.message}")
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(lb=torch.from_numpy(LB).to(device=device, dtype=dtype).flatten(),
                  ub=torch.from_numpy(UB).to(device=device, dtype=dtype).flatten())


def _parse_input_shape(input_shape):
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    return C, H, W


def _conv2d_generators(G, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device):
    if G.shape[1] == 0:
        return torch.zeros((n_out, 0), dtype=dtype, device=device)
    ncols = G.shape[1]
    imgs = G.t().contiguous().view(ncols, C, H, W)
    out = F.conv2d(imgs, weight, bias=None,
                   stride=stride, padding=padding, dilation=dilation, groups=groups)
    return out.permute(1, 2, 3, 0).contiguous().reshape(-1, ncols)


# ---- Activation encodings (module-level to keep class lean) ----

def _hz_apply_relu(hz, compute_bounds_fn):
    """Exact ReLU: +4ng +1nb +3nc per unstable neuron."""
    dtype, device = hz.c.dtype, hz.c.device
    n, ng, nb, nc = hz.c.shape[0], hz.Gc.shape[1], hz.Gb.shape[1], hz.Ac.shape[0]

    bounds = compute_bounds_fn(hz)
    lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
    active = lb >= 0;  inactive = ub <= 0;  unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    out_Gc = torch.zeros((n, ng + 4 * k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c = torch.zeros((n, 1), dtype=dtype, device=device)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]
    if k == 0:
        return HZono(c=out_c, Gc=out_Gc[:, :ng], Gb=out_Gb[:, :nb],
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    uidx = torch.where(unstable)[0]
    alpha, beta = lb[uidx], ub[uidx]
    t = torch.arange(k, device=device)
    c1, c2, c3, c4, cz = ng + t, ng + k + t, ng + 2*k + t, ng + 3*k + t, nb + t

    out_c[uidx, 0] = beta / 2.0
    out_Gc[uidx, c2] = -beta / 2.0

    eq_Ac = torch.zeros((3 * k, ng + 4*k), dtype=dtype, device=device)
    eq_Ab = torch.zeros((3 * k, nb + k), dtype=dtype, device=device)
    eq_b = torch.zeros((3 * k, 1), dtype=dtype, device=device)

    r1, r2 = 3 * t, 3 * t + 1
    eq_Ac[r1, c1] = 1.0;  eq_Ac[r1, c3] = 1.0;  eq_Ab[r1, cz] = 1.0;   eq_b[r1, 0] = 1.0
    eq_Ac[r2, c2] = 1.0;  eq_Ac[r2, c4] = 1.0;  eq_Ab[r2, cz] = -1.0;  eq_b[r2, 0] = 1.0

    for j in range(k):
        i = int(uidx[j].item())
        eq_Ac[3*j+2, c1[j]] = alpha[j] / 2.0
        eq_Ac[3*j+2, c2[j]] = -beta[j] / 2.0
        eq_Ac[3*j+2, :ng] -= hz.Gc[i]
        eq_Ab[3*j+2, :nb] -= hz.Gb[i]
        eq_Ab[3*j+2, cz[j]] = alpha[j] / 2.0
        eq_b[3*j+2, 0] = hz.c[i, 0] - beta[j] / 2.0

    old_Ac = torch.cat([hz.Ac, torch.zeros((nc, 4*k), dtype=dtype, device=device)], dim=1)
    old_Ab = torch.cat([hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)
    return HZono(c=out_c, Gc=out_Gc, Gb=out_Gb,
                 Ac=torch.cat([old_Ac, eq_Ac], dim=0),
                 Ab=torch.cat([old_Ab, eq_Ab], dim=0),
                 b=torch.cat([hz.b, eq_b], dim=0))


def _hz_apply_leaky_relu(hz, alpha_arg, compute_bounds_fn):
    """Exact LeakyReLU: +6ng +1nb +5nc per unstable neuron."""
    dtype, device = hz.c.dtype, hz.c.device
    n, ng, nb, nc = hz.c.shape[0], hz.Gc.shape[1], hz.Gb.shape[1], hz.Ac.shape[0]
    a = alpha_arg

    bounds = compute_bounds_fn(hz)
    lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
    active = lb >= 0;  inactive = ub <= 0;  unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    out_Gc = torch.zeros((n, ng + 6*k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c = torch.zeros((n, 1), dtype=dtype, device=device)
    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]
    if inactive.any():
        out_c[inactive] = a * hz.c[inactive]
        out_Gc[inactive, :ng] = a * hz.Gc[inactive]
        out_Gb[inactive, :nb] = a * hz.Gb[inactive]
    if k == 0:
        return HZono(c=out_c, Gc=out_Gc[:, :ng], Gb=out_Gb[:, :nb],
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    uidx = torch.where(unstable)[0]
    l, u = lb[uidx], ub[uidx]
    t = torch.arange(k, device=device)
    cg1, cg2 = ng + t, ng + k + t
    cs1p, cs1m, cs2p, cs2m = ng + 2*k + t, ng + 3*k + t, ng + 4*k + t, ng + 5*k + t
    cz = nb + t

    out_c[uidx, 0] = (u + a * l) / 4.0
    out_Gc[uidx, cg1] = a * l / 2.0
    out_Gc[uidx, cg2] = -u / 2.0
    out_Gb[uidx, cz] = (a * l - u) / 4.0

    ng_t, nb_t = ng + 6*k, nb + k
    eq_Ac = torch.zeros((5*k, ng_t), dtype=dtype, device=device)
    eq_Ab = torch.zeros((5*k, nb_t), dtype=dtype, device=device)
    eq_b = torch.zeros((5*k, 1), dtype=dtype, device=device)

    r0, r1, r2, r3 = 5*t, 5*t+1, 5*t+2, 5*t+3
    eq_Ac[r0, cg1] = 1.0;  eq_Ac[r0, cs1p] = 1.0;  eq_Ab[r0, cz] = 0.5;   eq_b[r0, 0] = 0.5
    eq_Ac[r1, cg1] = -1.0; eq_Ac[r1, cs1m] = 1.0;  eq_Ab[r1, cz] = 0.5;   eq_b[r1, 0] = 0.5
    eq_Ac[r2, cg2] = 1.0;  eq_Ac[r2, cs2p] = 1.0;  eq_Ab[r2, cz] = -0.5;  eq_b[r2, 0] = 0.5
    eq_Ac[r3, cg2] = -1.0; eq_Ac[r3, cs2m] = 1.0;  eq_Ab[r3, cz] = -0.5;  eq_b[r3, 0] = 0.5

    for j in range(k):
        i = int(uidx[j].item())
        eq_Ac[5*j+4, :ng] = hz.Gc[i]
        eq_Ac[5*j+4, cg1[j]] = -l[j] / 2.0
        eq_Ac[5*j+4, cg2[j]] = u[j] / 2.0
        eq_Ab[5*j+4, :nb] = hz.Gb[i]
        eq_Ab[5*j+4, cz[j]] = -(l[j] - u[j]) / 4.0
        eq_b[5*j+4, 0] = (u[j] + l[j]) / 4.0 - hz.c[i, 0]

    old_Ac = torch.cat([hz.Ac, torch.zeros((nc, 6*k), dtype=dtype, device=device)], dim=1)
    old_Ab = torch.cat([hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1)
    return HZono(c=out_c, Gc=out_Gc, Gb=out_Gb,
                 Ac=torch.cat([old_Ac, eq_Ac], dim=0),
                 Ab=torch.cat([old_Ab, eq_Ab], dim=0),
                 b=torch.cat([hz.b, eq_b], dim=0))


def _hz_apply_piecewise(hz, func, dfunc, K, compute_bounds_fn):
    """Piecewise tangent parallelogram: +6K ng, +K nb, +(4K+2) nc per wide neuron."""
    dtype, device = hz.c.dtype, hz.c.device
    n, ng, nb, nc = hz.c.shape[0], hz.Gc.shape[1], hz.Gb.shape[1], hz.Ac.shape[0]

    bounds = compute_bounds_fn(hz)
    lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
    wide = (ub - lb) > 1e-12;  narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.shape[0])

    new_c = hz.c.clone();  new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone();  new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone();  new_Gb_base[narrow] = 0.0
    if m == 0:
        return HZono(c=new_c, Gc=new_Gc_base, Gb=new_Gb_base,
                     Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    cx_k, cy_k, g1x_k, g1y_k, g2x_k, g2y_k = [], [], [], [], [], []

    for ki in range(K):
        a = lb_w + ki * (ub_w - lb_w) / K
        b = lb_w + (ki + 1) * (ub_w - lb_w) / K
        fa, fb = func(a), func(b)
        la, lb_s = dfunc(a), dfunc(b)
        cx, cy = (a + b) / 2.0, (fa + fb) / 2.0
        nl = (la - lb_s).abs() < 1e-10

        denom = lb_s - la
        sd = torch.where(nl, torch.ones_like(denom), denom)
        p1 = (fb - fa + lb_s * a - la * b) / sd;  p2 = a + b - p1
        g1xt = (p1 - a) / 2.0;  g1yt = lb_s * (p1 - a) / 2.0
        g2xt = (p2 - a) / 2.0;  g2yt = la * (p2 - a) / 2.0

        hw = (b - a) / 2.0;  slope = (fb - fa) / (b - a + 1e-30)
        tp = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).unsqueeze(1)
        pts = a.unsqueeze(0) + tp * (b - a).unsqueeze(0)
        fp = func(pts)
        me = (fp - (slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0))).abs().max(dim=0).values
        g1xl, g1yl = hw, slope * hw
        g2xl, g2yl = torch.zeros_like(hw), me

        g1x = torch.where(nl, g1xl, g1xt);  g1y = torch.where(nl, g1yl, g1yt)
        g2x = torch.where(nl, g2xl, g2xt);  g2y = torch.where(nl, g2yl, g2yt)

        dx = pts - cx.unsqueeze(0);  dy = fp - cy.unsqueeze(0)
        det = g1y * g2x - g1x * g2y
        sdet = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
        xi1 = (dy * g2x.unsqueeze(0) - dx * g2y.unsqueeze(0)) / sdet.unsqueeze(0)
        xi2 = (dy * g1x.unsqueeze(0) - dx * g1y.unsqueeze(0)) / (-sdet.unsqueeze(0))
        mxi = torch.max(xi1.abs().max(dim=0).values, xi2.abs().max(dim=0).values)
        sf = torch.where(mxi > 1.0, mxi * 1.01, torch.ones_like(mxi))
        sf = torch.where(det.abs() < 1e-30, torch.ones_like(sf), sf)
        g1x *= sf;  g1y *= sf;  g2x *= sf;  g2y *= sf

        cx_k.append(cx);  cy_k.append(cy)
        g1x_k.append(g1x);  g1y_k.append(g1y);  g2x_k.append(g2x);  g2y_k.append(g2y)

    cys = torch.zeros(m, dtype=dtype, device=device)
    for ki in range(K): cys += cy_k[ki]
    new_c[wide_idx] = (cys / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0;  new_Gb_base[wide_idx] = 0.0

    nr = 2 * K * m;  ns = 4 * K * m
    Gc_new = torch.zeros((n, nr + ns), dtype=dtype, device=device)
    for ki in range(K):
        gc1 = torch.arange(ki * m, (ki + 1) * m, device=device)
        gc2 = torch.arange(K * m + ki * m, K * m + (ki + 1) * m, device=device)
        for j in range(m):
            Gc_new[wide_idx[j], gc1[j]] = g1y_k[ki][j]
            Gc_new[wide_idx[j], gc2[j]] = g2y_k[ki][j]

    Gb_new = torch.zeros((n, K * m), dtype=dtype, device=device)
    for ki in range(K):
        zc = torch.arange(ki * m, (ki + 1) * m, device=device)
        for j in range(m): Gb_new[wide_idx[j], zc[j]] = -cy_k[ki][j] / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)
    ng_t = ng + nr + ns;  nb_t = nb + K * m

    nb_eq = 4 * K * m;  neq = nb_eq + m + m
    eq_Ac = torch.zeros((neq, ng_t), dtype=dtype, device=device)
    eq_Ab = torch.zeros((neq, nb_t), dtype=dtype, device=device)
    eq_b = torch.zeros((neq, 1), dtype=dtype, device=device)

    for ki in range(K):
        for j in range(m):
            g1c = ng + ki * m + j;  g2c = ng + K * m + ki * m + j
            zc = nb + ki * m + j;   sb = ng + nr + (ki * m + j) * 4
            r = 4 * (ki * m + j)
            eq_Ac[r, g1c] = 1.0;     eq_Ac[r, sb] = 1.0;      eq_Ab[r, zc] = -0.5;    eq_b[r, 0] = 0.5
            eq_Ac[r+1, g1c] = -1.0;  eq_Ac[r+1, sb+1] = 1.0;  eq_Ab[r+1, zc] = -0.5;  eq_b[r+1, 0] = 0.5
            eq_Ac[r+2, g2c] = 1.0;   eq_Ac[r+2, sb+2] = 1.0;  eq_Ab[r+2, zc] = -0.5;  eq_b[r+2, 0] = 0.5
            eq_Ac[r+3, g2c] = -1.0;  eq_Ac[r+3, sb+3] = 1.0;  eq_Ab[r+3, zc] = -0.5;  eq_b[r+3, 0] = 0.5

    for j in range(m):
        i = int(wide_idx[j].item());  r = nb_eq + j;  rv = 0.0
        for ki in range(K):
            g1c = ng + ki * m + j;  g2c = ng + K * m + ki * m + j;  zc = nb + ki * m + j
            rv += cx_k[ki][j].item() / 2.0
            eq_Ac[r, g1c] = -g1x_k[ki][j];  eq_Ac[r, g2c] = -g2x_k[ki][j]
            eq_Ab[r, zc] = cx_k[ki][j] / 2.0
        eq_Ac[r, :ng] = hz.Gc[i];  eq_Ab[r, :nb] = hz.Gb[i]
        eq_b[r, 0] = rv - hz.c[i, 0].item()

    for j in range(m):
        r = nb_eq + m + j
        for ki in range(K): eq_Ab[r, nb + ki * m + j] = 1.0
        eq_b[r, 0] = float(K - 2)

    old_Ac = torch.cat([hz.Ac, torch.zeros((nc, nr + ns), dtype=dtype, device=device)], dim=1)
    old_Ab = torch.cat([hz.Ab, torch.zeros((nc, K * m), dtype=dtype, device=device)], dim=1)
    return HZono(c=new_c, Gc=out_Gc, Gb=out_Gb,
                 Ac=torch.cat([old_Ac, eq_Ac], dim=0),
                 Ab=torch.cat([old_Ab, eq_Ab], dim=0),
                 b=torch.cat([hz.b, eq_b], dim=0))


def _hz_reduce(hz, max_order=10.0):
    dtype, device = hz.c.dtype, hz.c.device
    n, ng, nb, nc = hz.c.shape[0], hz.Gc.shape[1], hz.Gb.shape[1], hz.Ac.shape[0]
    if n == 0: return hz
    max_ng = max(int(max_order * n), n + 1);  max_nb = max(2 * n, 1)

    if nb > max_nb:
        cn = hz.Gb.abs().sum(dim=0)
        _, si = cn.sort()
        nr = nb - max_nb;  ri, ki = si[:nr], si[nr:]
        eGc = hz.Gb[:, ri]
        eAc = hz.Ab[:, ri] if nc > 0 else torch.zeros((0, nr), dtype=dtype, device=device)
        hz = HZono(c=hz.c, Gc=torch.cat([hz.Gc, eGc], dim=1), Gb=hz.Gb[:, ki],
                   Ac=torch.cat([hz.Ac, eAc], dim=1) if nc > 0 else torch.zeros((0, ng + nr), dtype=dtype, device=device),
                   Ab=hz.Ab[:, ki] if nc > 0 else torch.zeros((0, max_nb), dtype=dtype, device=device),
                   b=hz.b.clone())
        ng, nb = hz.Gc.shape[1], hz.Gb.shape[1]

    if ng > max_ng:
        cn = hz.Gc.abs().sum(dim=0)
        _, si = cn.sort(descending=True)
        ki, di = si[:max_ng - n], si[max_ng - n:]
        Gk = hz.Gc[:, ki]
        nGc = torch.cat([Gk, torch.diag(hz.Gc[:, di].abs().sum(dim=1))], dim=1)
        if nc > 0:
            ds = set(di.tolist())
            kr = [r for r in range(nc) if not any(abs(hz.Ac[r, c].item()) > 1e-15 for c in ds)]
            if kr:
                krt = torch.tensor(kr, dtype=torch.long, device=device)
                nAc = torch.cat([hz.Ac[krt][:, ki], torch.zeros((len(kr), n), dtype=dtype, device=device)], dim=1)
                nAb, nb_ = hz.Ab[krt], hz.b[krt]
            else:
                nAc = torch.zeros((0, nGc.shape[1]), dtype=dtype, device=device)
                nAb = torch.zeros((0, nb), dtype=dtype, device=device)
                nb_ = torch.zeros((0, 1), dtype=dtype, device=device)
        else:
            nAc = torch.zeros((0, nGc.shape[1]), dtype=dtype, device=device)
            nAb = torch.zeros((0, nb), dtype=dtype, device=device)
            nb_ = torch.zeros((0, 1), dtype=dtype, device=device)
        hz = HZono(c=hz.c, Gc=nGc, Gb=hz.Gb, Ac=nAc, Ab=nAb, b=nb_)
    return hz
