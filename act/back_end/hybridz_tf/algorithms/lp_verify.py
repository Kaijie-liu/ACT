"""Unsafe-feasibility LP and witness recovery for HZono.

Given an output HZono ``out_hz`` and an ASSERT layer carrying an
``OutKind``-typed spec, decide whether the unsafe set ``Y_unsafe`` is
reachable from ``out_hz`` (i.e. spec is violated for some input).

The check is by linear programming over the factor space of
``out_hz`` (LP relaxation: ``xi_c in [-1, 1]^ng``, ``xi_b in [-1, 1]^nb``,
plus ``Ac xi_c + Ab xi_b [op] b`` per ``eq_mask``). LP-infeasible per
disjunct on every disjunct ⇒ spec verified (sound). LP-feasible on any
disjunct ⇒ candidate witness in factor space; caller may replay it
through the concrete network to confirm or reject.

This is a simplified port of HyZor's ``check_unsafe_for_act`` /
``lp_witness_to_input`` (``__init__.py`` ~1494, 1572) covering the five
OutKinds used in our paper:

  * ``LINEAR_LE``    : spec ``c·y <= d``. Unsafe iff ``c·y > d``.
  * ``UNSAFE_LINEAR``: spec ``C y > d`` (any row holds). Unsafe iff
                       ALL rows ``C[r]·y <= d[r]`` → OR per disjunct.
  * ``TOP1_ROBUST``  : spec ``y[t] > y[j]`` ∀ j≠t. Unsafe iff some
                       ``y[j] >= y[t]`` → OR per rival.
  * ``MARGIN_ROBUST``: spec ``y[t] - y[j] > m`` ∀ j≠t. Unsafe iff some
                       ``y[j] >= y[t] - m``.
  * ``RANGE``        : spec ``lb_i <= y_i <= ub_i``. Unsafe iff some
                       ``y_i < lb_i`` or ``y_i > ub_i``.

LP backend: scipy.optimize.linprog (HiGHS). Per-disjunct timeout +
overall budget.
"""

from __future__ import annotations
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import torch

from act.back_end.solver.solver_hz import HZono, _eq_mask_of


# OutKind values from act.back_end.layer_schema (don't import the enum at
# module load time; just compare against the string form).


def _to_np_f64(t):
    if torch.is_tensor(t):
        return t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=np.float64)


def _kind_str(kind) -> str:
    """Coerce any kind representation to its trailing identifier string."""
    if kind is None:
        return ""
    s = str(kind)
    return s.split(".")[-1]


def _build_factor_lp(hz: HZono):
    """Build the constant parts of every LP we'll solve over hz.

    Returns dict with numpy arrays:
        Gc_n (n, p), Gb_n (n, q), c_n (n,)
        A_eq (m_eq, p+q), b_eq (m_eq,)
        A_le (m_le, p+q), b_le (m_le,)
        p, q
    """
    em_t = _eq_mask_of(hz)
    em = em_t.detach().cpu().numpy().astype(bool)
    le = ~em
    Ac_np = _to_np_f64(hz.Ac)
    Ab_np = _to_np_f64(hz.Ab)
    b_np = _to_np_f64(hz.b).reshape(-1)
    A_eq = np.concatenate([Ac_np[em], Ab_np[em]], axis=1) \
        if em.any() else np.zeros((0, hz.Ac.shape[1] + hz.Ab.shape[1]))
    b_eq = b_np[em] if em.any() else np.zeros(0)
    A_le = np.concatenate([Ac_np[le], Ab_np[le]], axis=1) \
        if le.any() else np.zeros((0, hz.Ac.shape[1] + hz.Ab.shape[1]))
    b_le = b_np[le] if le.any() else np.zeros(0)
    return {
        "Gc": _to_np_f64(hz.Gc),
        "Gb": _to_np_f64(hz.Gb),
        "c": _to_np_f64(hz.c).reshape(-1),
        "A_eq": A_eq, "b_eq": b_eq,
        "A_le": A_le, "b_le": b_le,
        "p": int(hz.Gc.shape[1]),
        "q": int(hz.Gb.shape[1]),
    }


def _lp_feas_or_minimize(prob, obj_row: np.ndarray, rhs_threshold: Optional[float],
                          sense: str = "maximize",
                          timeout_s: Optional[float] = None
                          ) -> Tuple[str, Optional[np.ndarray]]:
    """Solve one LP: ``sense (obj_row . xi + obj_const)`` s.t. hz constraints.

    Returns ``("feasible", xi_star)`` / ``("infeasible", None)`` /
    ``("timeout", None)``.

    If ``rhs_threshold`` is given, we EARLY-EXIT: once we have a feasible
    point whose objective beats the threshold we return feasible (the
    LP also gives back xi_star). If the LP's optimum is bounded by
    ``rhs_threshold`` the disjunct is infeasible (spec holds on that
    branch).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return "feasible", None  # conservative

    p, q = prob["p"], prob["q"]
    nvars = p + q
    bounds = [(-1.0, 1.0)] * nvars
    # linprog minimizes; for maximize we negate.
    c = -obj_row.copy() if sense == "maximize" else obj_row.copy()
    A_eq = prob["A_eq"] if prob["A_eq"].shape[0] > 0 else None
    b_eq = prob["b_eq"] if prob["A_eq"].shape[0] > 0 else None
    A_ub = prob["A_le"] if prob["A_le"].shape[0] > 0 else None
    b_ub = prob["b_le"] if prob["A_le"].shape[0] > 0 else None
    options = {}
    if timeout_s is not None:
        options["time_limit"] = float(timeout_s)
    try:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method="highs", options=options)
    except Exception:
        return "feasible", None
    if res.status == 0 and res.success:
        # Optimal found.
        obj_val = -res.fun if sense == "maximize" else res.fun
        if rhs_threshold is not None:
            # spec says obj_row . y <= threshold (LINEAR_LE) or analogous.
            # Disjunct infeasible iff max obj_row . y <= threshold.
            if (sense == "maximize" and obj_val <= rhs_threshold) or \
               (sense == "minimize" and obj_val >= rhs_threshold):
                return "infeasible", None
        return "feasible", res.x
    if res.status == 2:
        # Primal infeasible.
        return "infeasible", None
    if res.status == 1:
        return "timeout", None
    return "feasible", None  # conservative on unknown status


def check_unsafe_for_act(out_hz: HZono, assert_layer, *,
                         output_ids=None,
                         timeout_s: float = 30.0
                         ) -> Tuple[str, Optional[np.ndarray]]:
    """Decide whether the unsafe set is reachable from ``out_hz`` for the
    spec stored in ``assert_layer.params``.

    Returns:
        ``("feasible", xi_star)`` -- unsafe set has a candidate witness
                                     in factor space; spec may be
                                     violated. Caller must replay.
        ``("infeasible", None)``  -- spec verified on every disjunct.
        ``("timeout", None)``     -- gave up; report UNKNOWN.

    ``out_hz``'s LP relaxation is used (binary xi_b relaxed to [-1, 1]).
    This is sound for ``infeasible`` (no integer point can satisfy what
    the relaxation rules out) but ``feasible`` may have spurious
    witnesses; caller must concretely replay and confirm.
    """
    prob = _build_factor_lp(out_hz)
    p, q = prob["p"], prob["q"]
    Gc, Gb, c_vec = prob["Gc"], prob["Gb"], prob["c"]
    nvars = p + q
    kind = _kind_str(assert_layer.params.get("kind"))

    def _row_to_obj_y(coef: np.ndarray) -> Tuple[np.ndarray, float]:
        """y = c + Gc xi_c + Gb xi_b ⇒ coef·y = (coef·c) + obj_row · xi
        where obj_row = [coef·Gc, coef·Gb] of length nvars."""
        obj_row = np.concatenate([coef @ Gc, coef @ Gb], axis=0)
        obj_const = float(coef @ c_vec)
        return obj_row, obj_const

    t0 = time.perf_counter()
    def _remaining():
        return max(0.05, timeout_s - (time.perf_counter() - t0))

    if kind == "LINEAR_LE":
        coef = _to_np_f64(assert_layer.params["c"]).reshape(-1)
        d = float(_to_np_f64(assert_layer.params["d"]).reshape(-1)[0])
        obj_row, obj_const = _row_to_obj_y(coef)
        st, x = _lp_feas_or_minimize(
            prob, obj_row, rhs_threshold=d - obj_const,
            sense="maximize", timeout_s=_remaining(),
        )
        return st, x

    if kind == "UNSAFE_LINEAR":
        C = _to_np_f64(assert_layer.params["c"])
        d_vec = _to_np_f64(assert_layer.params["d"]).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        # Unsafe set = {y : Cy <= d}. Polytope is unsafe-reachable iff
        # exists xi with C(c + Gc xi_c + Gb xi_b) <= d AND hz constraints.
        # Build one combined LP where we add C·y <= d as extra <=
        # inequalities and check feasibility (any feasible point works).
        # Equivalent to: find any xi satisfying hz + C·y <= d.
        N = C.shape[0]
        # Augment A_le with N rows: C @ Gc | C @ Gb, rhs = d - C @ c
        A_le_aug = np.concatenate(
            [prob["A_le"],
             np.concatenate([C @ Gc, C @ Gb], axis=1)],
            axis=0,
        )
        b_le_aug = np.concatenate([prob["b_le"], d_vec - C @ c_vec])
        prob2 = dict(prob)
        prob2["A_le"] = A_le_aug
        prob2["b_le"] = b_le_aug
        obj_row = np.zeros(nvars)  # pure feasibility
        st, x = _lp_feas_or_minimize(
            prob2, obj_row, rhs_threshold=None,
            sense="minimize", timeout_s=_remaining(),
        )
        return st, x

    if kind == "TOP1_ROBUST":
        t = int(_to_np_f64(assert_layer.params["y_true"]).reshape(-1)[0])
        n_out = c_vec.size
        # Disjunct j (≠ t): unsafe iff y[j] >= y[t]
        for j in range(n_out):
            if j == t:
                continue
            coef = np.zeros(n_out)
            coef[j] = 1.0
            coef[t] = -1.0
            obj_row, obj_const = _row_to_obj_y(coef)
            # Want max (y[j] - y[t]) >= 0
            st, x = _lp_feas_or_minimize(
                prob, obj_row, rhs_threshold=-obj_const,
                sense="maximize", timeout_s=_remaining(),
            )
            if st == "feasible":
                return "feasible", x
            if st == "timeout":
                return "timeout", None
        return "infeasible", None

    if kind == "MARGIN_ROBUST":
        t = int(_to_np_f64(assert_layer.params["y_true"]).reshape(-1)[0])
        m = float(_to_np_f64(assert_layer.params["margin"]).reshape(-1)[0])
        n_out = c_vec.size
        # Disjunct j: unsafe iff y[j] >= y[t] - m  →  (y[j] - y[t]) >= -m
        for j in range(n_out):
            if j == t:
                continue
            coef = np.zeros(n_out)
            coef[j] = 1.0
            coef[t] = -1.0
            obj_row, obj_const = _row_to_obj_y(coef)
            st, x = _lp_feas_or_minimize(
                prob, obj_row, rhs_threshold=-m - obj_const,
                sense="maximize", timeout_s=_remaining(),
            )
            if st == "feasible":
                return "feasible", x
            if st == "timeout":
                return "timeout", None
        return "infeasible", None

    if kind == "RANGE":
        n_out = c_vec.size
        lb_spec = assert_layer.params.get("lb")
        ub_spec = assert_layer.params.get("ub")
        # Disjunct i_low: y[i] < lb_spec[i]  →  -y[i] > -lb_spec[i]
        if lb_spec is not None:
            lb_v = _to_np_f64(lb_spec).reshape(-1)
            for i in range(n_out):
                coef = np.zeros(n_out); coef[i] = -1.0
                obj_row, obj_const = _row_to_obj_y(coef)
                st, x = _lp_feas_or_minimize(
                    prob, obj_row, rhs_threshold=-lb_v[i] - obj_const,
                    sense="maximize", timeout_s=_remaining(),
                )
                if st == "feasible":
                    return "feasible", x
                if st == "timeout":
                    return "timeout", None
        if ub_spec is not None:
            ub_v = _to_np_f64(ub_spec).reshape(-1)
            for i in range(n_out):
                coef = np.zeros(n_out); coef[i] = 1.0
                obj_row, obj_const = _row_to_obj_y(coef)
                st, x = _lp_feas_or_minimize(
                    prob, obj_row, rhs_threshold=ub_v[i] - obj_const,
                    sense="maximize", timeout_s=_remaining(),
                )
                if st == "feasible":
                    return "feasible", x
                if st == "timeout":
                    return "timeout", None
        return "infeasible", None

    # Unknown kind ⇒ conservative report.
    return "feasible", None


def lp_witness_to_input(xi_star: np.ndarray, input_hz: HZono) -> np.ndarray:
    """Map a factor-space witness xi_star back to input space.

    For a network whose first HZ corresponds to the input box, the
    "input" coordinates are ``c_in + Gc_in @ xi_c + Gb_in @ xi_b`` for
    the relevant factor slots. If ``input_hz`` is a BoxHZ-style HZono
    (diagonal Gc, no Gb), the first p_in entries of ``xi_star`` are the
    factor coordinates of the input pixels.
    """
    p_in = int(input_hz.Gc.shape[1])
    q_in = int(input_hz.Gb.shape[1])
    xi_c = xi_star[:p_in]
    xi_b = xi_star[p_in: p_in + q_in] if q_in > 0 else np.zeros(0)
    c = _to_np_f64(input_hz.c).reshape(-1)
    Gc = _to_np_f64(input_hz.Gc)
    Gb = _to_np_f64(input_hz.Gb)
    x = c + (Gc @ xi_c if p_in > 0 else 0.0)
    if q_in > 0:
        x = x + Gb @ xi_b
    return x


# --- Self-tests (run with: python -m act.back_end.hybridz_tf.algorithms.lp_verify) ---


def _test_infeasible_top1_robust_trivial():
    """y in [0, 1]^2, spec TOP1_ROBUST with y_true=0.
    Then y[0] in [0,1], y[1] in [0,1]; y[1] >= y[0] is feasible (e.g. y[1]=1, y[0]=0).
    So check_unsafe should report 'feasible'."""
    from act.back_end.solver.solver_hz import hz_from_bounds
    from act.back_end.core import Bounds
    hz = hz_from_bounds(
        Bounds(lb=torch.tensor([0.0, 0.0]), ub=torch.tensor([1.0, 1.0])),
        torch.float64, "cpu",
    )

    class _Mock:
        params = {"kind": "TOP1_ROBUST", "y_true": np.array([0])}
    st, x = check_unsafe_for_act(hz, _Mock(), timeout_s=5.0)
    assert st == "feasible"


def _test_infeasible_top1_robust_dominated():
    """y = [10, 0] +/- 1 each, spec TOP1_ROBUST y_true=0.
    y[0] in [9, 11], y[1] in [-1, 1]; y[1] >= y[0] requires y[1] - y[0] >= 0
    which max is 1 - 9 = -8 < 0. Infeasible."""
    from act.back_end.solver.solver_hz import hz_from_bounds
    from act.back_end.core import Bounds
    hz = hz_from_bounds(
        Bounds(lb=torch.tensor([9.0, -1.0]), ub=torch.tensor([11.0, 1.0])),
        torch.float64, "cpu",
    )

    class _Mock:
        params = {"kind": "TOP1_ROBUST", "y_true": np.array([0])}
    st, x = check_unsafe_for_act(hz, _Mock(), timeout_s=5.0)
    assert st == "infeasible"


def _test_lp_witness_recovery():
    from act.back_end.solver.solver_hz import hz_from_bounds
    from act.back_end.core import Bounds
    hz = hz_from_bounds(
        Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0])),
        torch.float64, "cpu",
    )
    xi = np.array([0.5, -0.5])
    x = lp_witness_to_input(xi, hz)
    # x = c + Gc @ xi = [0,0] + diag(1,1) @ [0.5, -0.5] = [0.5, -0.5]
    assert np.allclose(x, [0.5, -0.5])


if __name__ == "__main__":
    _test_infeasible_top1_robust_trivial()
    _test_infeasible_top1_robust_dominated()
    _test_lp_witness_recovery()
    print("OK: lp_verify tests pass")
