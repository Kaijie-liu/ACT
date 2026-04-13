from __future__ import annotations
from typing import List, Optional
import numpy as np

from act.back_end.solver.solver_base import Solver, SolverCaps, SolveStatus


class BoxSolver(Solver):
    """Most imprecise solver: pure box/interval arithmetic.

    Conservative one-sided baseline for the three-tier solver architecture:
      solver_gurobi.py (GurobiSolver)  — most precise: MILP exact
      solver_hz.py     (HZSolver)      — middle: LP relaxation + HZ domain
      solver_box.py    (BoxSolver)     — least precise: box + interval propagation

    Semantics:
      - Can prove UNSAT via box infeasibility or interval constraint checking.
      - Returns UNKNOWN otherwise (never claims SAT from box alone).
      - has_solution() is always False — no concrete witness produced.
    """

    def __init__(self):
        self._n = 0
        self._lb = None   # np.ndarray
        self._ub = None   # np.ndarray
        self._eq = []     # (vids, coeffs, rhs)
        self._le = []     # (vids, coeffs, rhs)  — all inequalities stored as <=
        self._status = SolveStatus.UNKNOWN

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False)

    @property
    def n(self) -> int:
        return self._n

    def begin(self, name: str = "box", device: Optional[str] = None):
        self._n = 0
        self._lb = None
        self._ub = None
        self._eq.clear()
        self._le.clear()
        self._status = SolveStatus.UNKNOWN

    def add_vars(self, n: int) -> None:
        if n <= 0:
            return
        if self._n == 0:
            self._n = n
            self._lb = np.full(n, -np.inf)
            self._ub = np.full(n, +np.inf)
        else:
            self._n += n
            self._lb = np.concatenate([self._lb, np.full(n, -np.inf)])
            self._ub = np.concatenate([self._ub, np.full(n, +np.inf)])

    def add_binary_vars(self, n: int) -> List[int]:
        start = self._n
        self.add_vars(n)
        idxs = list(range(start, start + n))
        self._lb[idxs] = 0.0
        self._ub[idxs] = 1.0
        return idxs

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:
        for i, lo, hi in zip(idxs, lb, ub):
            self._lb[i] = max(self._lb[i], float(lo))
            self._ub[i] = min(self._ub[i], float(hi))

    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._eq.append((vids, coeffs, rhs))

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, coeffs, rhs))

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, [-float(c) for c in coeffs], -float(rhs)))

    def add_sum_eq(self, vids: List[int], rhs: float) -> None:
        self.add_lin_eq(vids, [1.0] * len(vids), rhs)

    def add_ge_zero(self, vids: List[int]) -> None:
        for i in vids:
            self._lb[i] = max(self._lb[i], 0.0)

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:
        pass

    def set_objective_linear(self, vids: List[int], coeffs: List[float],
                             const: float = 0.0, sense: str = "min") -> None:
        pass  # Box solver does not optimize an objective

    def optimize(self, timelimit: Optional[float] = None) -> None:
        """Box feasibility check: bounds consistency + interval constraint propagation."""
        # 1. Check box bounds consistency
        if self._lb is not None and np.any(self._lb > self._ub + 1e-12):
            self._status = SolveStatus.UNSAT
            return

        # 2. Equality constraint interval propagation
        changed = True
        max_rounds = 10
        for _ in range(max_rounds):
            if not changed:
                break
            changed = False
            for vids, coeffs, rhs in self._eq:
                if self._propagate_eq(vids, coeffs, rhs):
                    changed = True
                if self._status == SolveStatus.UNSAT:
                    return

        # 3. Inequality feasibility check: for each a·x <= rhs,
        #    compute min(a·x) over box. If min > rhs, infeasible.
        for vids, coeffs, rhs in self._le:
            lhs_min, _ = self._interval_range(vids, coeffs)
            if lhs_min is not None and lhs_min > rhs + 1e-12:
                self._status = SolveStatus.UNSAT
                return

        # Cannot determine feasibility from box alone
        self._status = SolveStatus.UNKNOWN

    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return False  # Box never claims a concrete solution

    def get_values(self, vids: List[int]) -> np.ndarray:
        if self._lb is None:
            return np.zeros(len(vids))
        mid = (self._lb[vids] + self._ub[vids]) / 2.0
        # Clamp inf to 0 for stability
        mid = np.where(np.isfinite(mid), mid, 0.0)
        return mid

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        return self.get_values(input_ids)

    # ---- Internal helpers ----

    def _interval_range(self, vids, coeffs):
        """Compute (min, max) of a·x over current box. Returns (None, None) if any bound is inf."""
        lo_sum, hi_sum = 0.0, 0.0
        for v, c in zip(vids, coeffs):
            lo, hi = self._lb[v], self._ub[v]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return None, None
            if c >= 0:
                lo_sum += c * lo;  hi_sum += c * hi
            else:
                lo_sum += c * hi;  hi_sum += c * lo
        return lo_sum, hi_sum

    def _propagate_eq(self, vids, coeffs, rhs) -> bool:
        """Tighten bounds via one equality constraint. Returns True if any bound changed."""
        lhs_min, lhs_max = self._interval_range(vids, coeffs)
        if lhs_min is not None and lhs_min > rhs + 1e-12:
            self._status = SolveStatus.UNSAT;  return False
        if lhs_max is not None and lhs_max < rhs - 1e-12:
            self._status = SolveStatus.UNSAT;  return False

        changed = False
        for v, c in zip(vids, coeffs):
            if abs(c) < 1e-15:
                continue
            if not (np.isfinite(self._lb[v]) and np.isfinite(self._ub[v])):
                continue
            # Residual interval excluding v
            others = [(v2, c2) for v2, c2 in zip(vids, coeffs) if v2 != v]
            res_min, res_max = self._interval_range([v2 for v2, _ in others],
                                                     [c2 for _, c2 in others])
            if res_min is None:
                continue
            # c*x_v = rhs - residual
            if c > 0:
                new_lo = (rhs - res_max) / c;  new_hi = (rhs - res_min) / c
            else:
                new_lo = (rhs - res_min) / c;  new_hi = (rhs - res_max) / c

            old_lo, old_hi = self._lb[v], self._ub[v]
            self._lb[v] = max(self._lb[v], new_lo)
            self._ub[v] = min(self._ub[v], new_hi)

            if self._lb[v] > self._ub[v] + 1e-12:
                self._status = SolveStatus.UNSAT
                return False
            if abs(self._lb[v] - old_lo) > 1e-12 or abs(self._ub[v] - old_hi) > 1e-12:
                changed = True

        return changed
