"""Binary probing for HZono: fix binary generators by LP feasibility.

Background
----------
After an HZ has been propagated through layers using HyZor's
``eq_lagr_v8`` ReLU encoding (which adds one fresh binary generator
``z_i in {-1,+1}`` per unstable neuron), some of these binaries can be
proven to take a determined value by a residual LP/feasibility check.
Each such fixed binary collapses the corresponding ``Gb`` column into a
known additive contribution to ``c`` and shrinks the unsafe-set search
space.

Two probing strategies are combined here:

1. **Row-Interval Implication Mining (RIIM)** — a pure-interval pass
   that, for each equality row ``a_c^T xi_c + a_b^T xi_b = b``,
   considers fixing ``z_i = +1`` and tests whether the residual range
   of ``a_c^T xi_c + sum_{j != i} a_{b,j} xi_{b,j}`` contains
   ``b - a_{b,i}``. If not, ``z_i = -1`` is forced (analogous for
   ``z_i = -1``). No LP call required. Implements
   ``Proposition`` in the eq_lagr write-up.

2. **Singleton LP probing** — for each surviving binary not yet fixed,
   solve two small LPs (one with ``z_i = +1`` plugged in, one with
   ``z_i = -1`` plugged in) and look for an infeasible side. Uses
   HiGHS via scipy.optimize.linprog.

The ACT port is a **deliberately simplified** version of HyZor's
``binary_probe_v8`` (``HybridZonotope.py`` ~1710+, ~1361 lines): it
keeps the two strategies above and the budget-aware loop but drops the
priority scoring, pairwise RIIM v2, and elaborate profiling. For
verification this captures the essential mechanism; the dropped
features are performance tweaks that can be added back per-need.

Output
------
``binary_probe(hz, ...)`` returns a new HZono with the same set
semantics but ``nb`` reduced by the count of fixed binaries; each
fixed binary contributes ``+sign * Gb[:, i]`` to ``c`` and is removed
from ``Gb``. ``Ac/Ab/b/eq_mask`` are updated accordingly.
"""

from __future__ import annotations
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from act.back_end.solver.solver_hz import HZono, _eq_mask_of


def _to_np_f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def _riim_pass(Ac_np: np.ndarray, Ab_np: np.ndarray, b_np: np.ndarray,
               em_np: np.ndarray, fixed_signs: np.ndarray,
               tol: float = 1e-9) -> Tuple[np.ndarray, int]:
    """One RIIM sweep: for each equality row, test fixing each free
    binary to +1 vs -1 and check interval feasibility.

    Args:
        Ac_np, Ab_np, b_np: equality constraint block.
        em_np: bool array, True where the row is eq. We only mine eq rows.
        fixed_signs: int8 array of length nb. 0 = free, ±1 = already fixed.
        tol: feasibility tolerance.

    Returns:
        (new fixed_signs, n_newly_fixed)
    """
    nb = int(fixed_signs.size)
    fixed_signs = fixed_signs.copy()
    n_eq = int(em_np.sum())
    if n_eq == 0 or nb == 0:
        return fixed_signs, 0
    Ac_eq = Ac_np[em_np]
    Ab_eq = Ab_np[em_np]
    b_eq = b_np[em_np]
    # Reduce rhs by already-fixed binaries.
    if np.any(fixed_signs != 0):
        # add Ab_eq[:, i] * (-fixed_signs[i]) to b_eq for each fixed i,
        # then drop that column from active set
        b_eq = b_eq - Ab_eq @ fixed_signs.astype(np.float64)
    nb_now = nb
    new_fixed = 0
    for i in range(nb_now):
        if fixed_signs[i] != 0:
            continue
        a_b_i = Ab_eq[:, i]
        # Residual: a_c xi_c + sum_{j != i, j free} a_{b,j} xi_{b,j}
        free_mask = (fixed_signs == 0).copy()
        free_mask[i] = False
        Ab_eq_rest = Ab_eq[:, free_mask]
        # Continuous range: sum |a_c[r, k]| over k (xi_c in [-1,1])
        cont_radius = np.abs(Ac_eq).sum(axis=1)
        # Binary range (over free): sum |a_{b,j}|
        bin_radius = np.abs(Ab_eq_rest).sum(axis=1)
        total_radius = cont_radius + bin_radius
        # Test z_i = +1 ⇒ residual must contain b_eq[r] - a_b_i[r] for all r
        target_plus = b_eq - a_b_i
        # Residual is centered at 0 (continuous symmetric, binaries antisymmetric),
        # so feasibility: |target_r| <= total_radius_r + tol for all r
        plus_feasible = np.all(np.abs(target_plus) <= total_radius + tol)
        target_minus = b_eq + a_b_i
        minus_feasible = np.all(np.abs(target_minus) <= total_radius + tol)
        if plus_feasible and not minus_feasible:
            fixed_signs[i] = 1
            new_fixed += 1
        elif minus_feasible and not plus_feasible:
            fixed_signs[i] = -1
            new_fixed += 1
        # else: both feasible (no info) or both infeasible (system already empty)
    return fixed_signs, new_fixed


def _pairwise_row_mining_pass(
    Ac_np: np.ndarray, Ab_np: np.ndarray, b_np: np.ndarray,
    em_np: np.ndarray, fixed_signs: np.ndarray,
    tol: float = 1e-9, coef_eps: float = 1e-10,
) -> Tuple[np.ndarray, int, list]:
    """Port of HyZor binary_probe_v8 Stage 2 row mining.

    For each equality row with ≤2 nonzero binary coefficients, test the
    feasibility of each sign combination (z_i, z_j) ∈ {(+,+), (+,-),
    (-,+), (-,-)} against the row's continuous range. Three outcomes:

      n_feas == 1: both binaries forced (unary vote on each).
      n_feas == 2 in {(++,--), (+-,-+)}: yields a relation
                  z_i = +z_j  or  z_i = -z_j  via union-find.
      n_feas == 3 or 4: no implication, skip.

    Unanimous unary votes get promoted; relations propagate fixings
    through the union-find graph. Returns ``rel_triplets`` for
    surviving relation chains -- caller may inject these as eq rows
    on the HZ to strengthen downstream passes.

    All interval-based (no LP). Faithful to HZ:1858-2050.
    """
    nb = int(fixed_signs.size)
    fixed_signs = fixed_signs.copy()
    if nb == 0 or em_np is None or not em_np.any():
        return fixed_signs, 0, []
    Ac_eq = Ac_np[em_np]
    Ab_eq = Ab_np[em_np]
    b_eq = b_np[em_np].astype(np.float64)

    # Reduce rhs by already-fixed binaries.
    if np.any(fixed_signs != 0):
        b_eq = b_eq - Ab_eq @ fixed_signs.astype(np.float64)

    parent = list(range(nb))
    parity = [1] * nb

    def _find(x: int):
        if parent[x] != x:
            r, p = _find(parent[x])
            parity[x] *= p
            parent[x] = r
        return parent[x], parity[x]

    def _union(x: int, y: int, sgn: int) -> bool:
        rx, px = _find(x)
        ry, py = _find(y)
        if rx == ry:
            return (px == sgn * py)
        parent[rx] = ry
        parity[rx] = int(sgn * py * px)
        return True

    vote_pos = np.zeros(nb, dtype=np.int32)
    vote_neg = np.zeros(nb, dtype=np.int32)
    rel_seen = set()

    for r in range(Ab_eq.shape[0]):
        nz = np.where(np.abs(Ab_eq[r]) > coef_eps)[0]
        # Skip rows where free binaries don't participate, and rows with
        # >2 binary cols (HyZor only mines sparse-binary rows).
        if nz.size == 0 or nz.size > 2:
            continue
        # Skip already-fixed positions for this row.
        nz = [int(j) for j in nz if fixed_signs[j] == 0]
        if len(nz) == 0 or len(nz) > 2:
            continue
        rad = float(np.abs(Ac_eq[r]).sum())
        rhs = float(b_eq[r])
        if len(nz) == 1:
            j = nz[0]
            a = float(Ab_eq[r, j])
            if abs(a) <= coef_eps:
                continue
            feas_p = abs(rhs - a) <= (rad + tol)
            feas_n = abs(rhs + a) <= (rad + tol)
            if feas_p and (not feas_n):
                vote_pos[j] += 1
            elif feas_n and (not feas_p):
                vote_neg[j] += 1
            continue

        i, j = nz[0], nz[1]
        ai = float(Ab_eq[r, i]); aj = float(Ab_eq[r, j])
        feas_pp = abs(rhs - ai - aj) <= (rad + tol)
        feas_pn = abs(rhs - ai + aj) <= (rad + tol)
        feas_np = abs(rhs + ai - aj) <= (rad + tol)
        feas_nn = abs(rhs + ai + aj) <= (rad + tol)
        n_feas = int(feas_pp) + int(feas_pn) + int(feas_np) + int(feas_nn)

        if n_feas == 1:
            if feas_pp:
                vote_pos[i] += 1; vote_pos[j] += 1
            elif feas_pn:
                vote_pos[i] += 1; vote_neg[j] += 1
            elif feas_np:
                vote_neg[i] += 1; vote_pos[j] += 1
            else:
                vote_neg[i] += 1; vote_neg[j] += 1
            continue

        if n_feas == 2:
            if feas_pp and feas_nn and (not feas_pn) and (not feas_np):
                s = 1
            elif feas_pn and feas_np and (not feas_pp) and (not feas_nn):
                s = -1
            else:
                continue
            key = (i, j, s) if i < j else (j, i, s)
            if key in rel_seen:
                continue
            rel_seen.add(key)
            _union(i, j, s)

    # Promote unanimous unary votes.
    seed_fixed = {}
    for j in range(nb):
        if fixed_signs[j] != 0:
            continue
        pv = int(vote_pos[j]); nv = int(vote_neg[j])
        if pv > 0 and nv == 0:
            seed_fixed[j] = 1
        elif nv > 0 and pv == 0:
            seed_fixed[j] = -1

    # Propagate fixings through relation graph.
    root_fix = {}
    bad_root = set()
    for j, v in seed_fixed.items():
        r, p = _find(int(j))
        rv = float(p) * float(v)
        if r in root_fix and abs(root_fix[r] - rv) > 1e-9:
            bad_root.add(r)
        else:
            root_fix[r] = rv

    new_fixed = 0
    for j in range(nb):
        if fixed_signs[j] != 0:
            continue
        r, p = _find(j)
        if r in bad_root:
            continue
        if r in root_fix:
            vv = float(p) * float(root_fix[r])
            fixed_signs[j] = 1 if vv >= 0 else -1
            new_fixed += 1

    # Residual relations: build minimal tree per unfixed component for
    # downstream injection.
    comp = {}
    for j in range(nb):
        if fixed_signs[j] != 0:
            continue
        r, p = _find(j)
        comp.setdefault(r, []).append((j, p))

    rel_triplets = []
    rel_cap = min(96, max(16, nb // 2))
    for r, members in comp.items():
        if len(members) <= 1:
            continue
        if r in root_fix and r not in bad_root:
            continue
        base_j, base_p = min(members, key=lambda x: x[0])
        for j, pj in members:
            if j == base_j:
                continue
            s = int(pj * base_p)  # z_j = s * z_base
            rel_triplets.append((j, base_j, s))
            if len(rel_triplets) >= rel_cap:
                break
        if len(rel_triplets) >= rel_cap:
            break

    return fixed_signs, new_fixed, rel_triplets


def _inject_pairwise_relations(
    hz: HZono, rel_triplets: list,
) -> HZono:
    """Append eq rows ``z_j - s * z_base = 0`` to the HZ.

    Each triplet ``(j, base, s)`` becomes an equality row with
    coefficient 1.0 on column j, -s on column base, all other cols 0.
    Marked as eq in eq_mask. Continuous coefficients are 0.
    """
    if not rel_triplets:
        return hz
    nb = int(hz.Gb.shape[1])
    ng = int(hz.Gc.shape[1])
    nc = int(hz.b.shape[0])
    device = hz.b.device
    dtype = hz.b.dtype

    n_rel = len(rel_triplets)
    Ac_new = torch.zeros(n_rel, ng, dtype=dtype, device=device)
    Ab_new = torch.zeros(n_rel, nb, dtype=dtype, device=device)
    b_new = torch.zeros(n_rel, 1, dtype=dtype, device=device)
    for r, (j, base, s) in enumerate(rel_triplets):
        Ab_new[r, j] = 1.0
        Ab_new[r, base] = -float(s)
    Ac_full = torch.cat([hz.Ac, Ac_new], dim=0)
    Ab_full = torch.cat([hz.Ab, Ab_new], dim=0)
    b_full = torch.cat([hz.b, b_new], dim=0)
    em_old = _eq_mask_of(hz)
    em_new = torch.cat([em_old, torch.ones(n_rel, dtype=torch.bool, device=device)])
    return HZono(c=hz.c, Gc=hz.Gc, Gb=hz.Gb,
                 Ac=Ac_full, Ab=Ab_full, b=b_full,
                 eq_mask=em_new)


def _lp_singleton_pass(Ac_np: np.ndarray, Ab_np: np.ndarray, b_np: np.ndarray,
                       em_np: np.ndarray, fixed_signs: np.ndarray,
                       p: int, q: int, time_budget: float) -> Tuple[np.ndarray, int]:
    """Probe each unfixed binary by solving an LP with that binary
    plugged at +1 and another at -1; if exactly one side is infeasible,
    fix the binary."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return fixed_signs.copy(), 0

    fixed_signs = fixed_signs.copy()
    nb = int(fixed_signs.size)
    new_fixed = 0
    t0 = time.perf_counter()

    # We solve feasibility LPs of the form
    #   find xi_c in [-1,1]^p, xi_b in [-1,1]^q with z_i = ±1
    #   s.t. eq rows: Ac_eq xi_c + Ab_eq xi_b = b_eq
    #        le rows: Ac_le xi_c + Ab_le xi_b <= b_le
    # Note: relaxing xi_b ∈ [-1,1] (instead of {-1,+1}) is LP — safe
    # for feasibility tests since any integer feasible point is also LP
    # feasible. INFEASIBLE under relaxation ⇒ also infeasible at integer.
    le_np = ~em_np
    Ac_eq = Ac_np[em_np]
    Ab_eq = Ab_np[em_np]
    b_eq = b_np[em_np]
    Ac_le = Ac_np[le_np]
    Ab_le = Ab_np[le_np]
    b_le = b_np[le_np]

    nvars = p + q
    bounds = [(-1.0, 1.0)] * nvars

    def _solve_with_z(i: int, z_val: float) -> bool:
        """Return True if LP is feasible with z_i = z_val."""
        # Fold z_val into rhs.
        eq_rhs = b_eq.copy() - Ab_eq[:, i] * z_val
        le_rhs = b_le.copy() - Ab_le[:, i] * z_val
        A_eq_red = np.delete(np.concatenate([Ac_eq, Ab_eq], axis=1), p + i, axis=1)
        A_le_red = np.delete(np.concatenate([Ac_le, Ab_le], axis=1), p + i, axis=1)
        # Set bounds on remaining vars
        bnds = [(-1.0, 1.0)] * (nvars - 1)
        c_obj = np.zeros(nvars - 1)
        try:
            res = linprog(
                c=c_obj,
                A_eq=A_eq_red if A_eq_red.shape[0] > 0 else None,
                b_eq=eq_rhs if A_eq_red.shape[0] > 0 else None,
                A_ub=A_le_red if A_le_red.shape[0] > 0 else None,
                b_ub=le_rhs if A_le_red.shape[0] > 0 else None,
                bounds=bnds, method="highs",
            )
            # success=False AND status==2 (infeasible) is what we want
            return res.success or res.status not in (2,)  # 2 = infeasible
        except Exception:
            return True  # conservative: treat as feasible

    for i in range(nb):
        if fixed_signs[i] != 0:
            continue
        if time.perf_counter() - t0 > time_budget:
            break
        plus_feas = _solve_with_z(i, +1.0)
        minus_feas = _solve_with_z(i, -1.0)
        if plus_feas and not minus_feas:
            fixed_signs[i] = 1
            new_fixed += 1
        elif minus_feas and not plus_feas:
            fixed_signs[i] = -1
            new_fixed += 1

    return fixed_signs, new_fixed


def _apply_fixings(hz: HZono, fixed_signs: np.ndarray) -> HZono:
    """Remove fixed binary columns from Gb/Ab, fold their contribution
    into c and b."""
    if not np.any(fixed_signs != 0):
        return hz
    device = hz.c.device
    dtype = hz.c.dtype
    signs_t = torch.from_numpy(fixed_signs).to(dtype=dtype, device=device)
    keep_mask_t = (signs_t == 0)
    fix_mask_t = ~keep_mask_t
    # New c: c + Gb[:, fix] @ signs[fix]
    Gb_fix = hz.Gb[:, fix_mask_t]
    signs_fix = signs_t[fix_mask_t].view(-1, 1)
    new_c = hz.c + Gb_fix @ signs_fix
    new_Gb = hz.Gb[:, keep_mask_t]
    # New b: b - Ab[:, fix] @ signs[fix]
    Ab_fix = hz.Ab[:, fix_mask_t]
    new_b = hz.b - (Ab_fix @ signs_fix)
    new_Ab = hz.Ab[:, keep_mask_t]
    return HZono(
        c=new_c, Gc=hz.Gc.clone(), Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=new_Ab, b=new_b,
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )


def binary_probe(
    hz: HZono,
    *,
    timeout: float = 10.0,
    enable_riim: bool = True,
    enable_lp: bool = True,
    lp_time_fraction: float = 0.7,
) -> HZono:
    """Probe binaries via RIIM then optional LP, fix proven values.

    Args:
        hz: input HZono with nb > 0.
        timeout: total wall-clock budget (seconds).
        enable_riim: run interval-only pass (cheap, no LP calls).
            Disable for clean RIIM-ON-OFF ablations.
        enable_lp: run the singleton LP pass after RIIM.
        lp_time_fraction: fraction of remaining budget the LP pass may use.

    Returns:
        New HZono with fixed binaries folded into c/b and removed from
        Gb/Ab.
    """
    if int(hz.Gb.shape[1]) == 0:
        return hz
    nb = int(hz.Gb.shape[1])
    nc = int(hz.b.shape[0])
    p = int(hz.Gc.shape[1])
    q = nb
    Ac_np = _to_np_f64(hz.Ac)
    Ab_np = _to_np_f64(hz.Ab)
    b_np = _to_np_f64(hz.b).reshape(-1)
    em_t = _eq_mask_of(hz)
    em_np = em_t.detach().cpu().numpy().astype(bool)
    fixed_signs = np.zeros(nb, dtype=np.int8)
    t0 = time.perf_counter()

    if enable_riim and os.environ.get("HYZOR_DISABLE_RIIM", "0") != "1":
        # Iterate RIIM (singleton) until quiescent.
        while True:
            fixed_signs, n_new = _riim_pass(
                Ac_np, Ab_np, b_np, em_np, fixed_signs
            )
            if n_new == 0 or time.perf_counter() - t0 > 0.2 * timeout:
                break

        # Pairwise row mining (HyZor v8 Stage 2): mine sparse-binary eq
        # rows for both unary votes (chains with above) and pairwise
        # relations z_j = s * z_base. Pure interval (no LP).
        fixed_signs, n_pair, rel_triplets = _pairwise_row_mining_pass(
            Ac_np, Ab_np, b_np, em_np, fixed_signs,
        )

        # If relations were mined and there's still budget, inject as eq
        # rows and re-run singleton RIIM (the new rows may unlock more
        # singleton fixings via chain effect on the post-injection HZ).
        if rel_triplets:
            hz_inj = _inject_pairwise_relations(hz, rel_triplets)
            Ac_np = _to_np_f64(hz_inj.Ac)
            Ab_np = _to_np_f64(hz_inj.Ab)
            b_np = _to_np_f64(hz_inj.b).reshape(-1)
            em_np = _eq_mask_of(hz_inj).detach().cpu().numpy().astype(bool)
            # Update fixed_signs length if Gb shape changed (unlikely
            # since _inject_pairwise_relations doesn't drop cols).
            while True:
                fixed_signs, n_new = _riim_pass(
                    Ac_np, Ab_np, b_np, em_np, fixed_signs
                )
                if n_new == 0 or time.perf_counter() - t0 > 0.3 * timeout:
                    break
            hz = hz_inj  # carry injected rows into LP / final HZ

    if enable_lp:
        remaining = timeout - (time.perf_counter() - t0)
        lp_budget = max(0.5, remaining * lp_time_fraction)
        fixed_signs, _ = _lp_singleton_pass(
            Ac_np, Ab_np, b_np, em_np, fixed_signs, p, q, lp_budget
        )

    return _apply_fixings(hz, fixed_signs)


# --- Self-tests (run with: python -m act.back_end.hybridz_tf.algorithms.binary_probe) ---


def _test_no_binaries_passthrough():
    """nb == 0 → unchanged."""
    n = 2
    hz = HZono(
        c=torch.zeros(n, 1), Gc=torch.eye(n),
        Gb=torch.zeros(n, 0), Ac=torch.zeros(0, n),
        Ab=torch.zeros(0, 0), b=torch.zeros(0, 1),
    )
    out = binary_probe(hz)
    assert out is hz


def _test_riim_fixes_forced_binary():
    """A single eq row that forces z=+1 should fix it via RIIM alone."""
    # x = 0.0 + 0 * xi_c + (-2.0) * z = 0   → z = 0/(-2) = 0, but z in {-1,+1}
    # Build a row that says: 1.0 * z = -1 (so z must be -1)
    n = 1
    Gc = torch.zeros(n, 0)
    Gb = torch.tensor([[1.0]])  # so y = c + 1*z
    c = torch.zeros(n, 1)
    Ac = torch.zeros(1, 0)
    Ab = torch.tensor([[1.0]])  # eq: 1*z = -1 → z = -1
    b = torch.tensor([[-1.0]])
    eq_mask = torch.tensor([True])
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=eq_mask)
    out = binary_probe(hz, enable_lp=False)
    # After fixing z=-1, c becomes c + Gb @ (-1) = 0 + (-1) = -1, nb=0
    assert int(out.Gb.shape[1]) == 0, f"expected nb=0 after fix, got {out.Gb.shape[1]}"
    assert float(out.c.item()) == -1.0


if __name__ == "__main__":
    _test_no_binaries_passthrough()
    _test_riim_fixes_forced_binary()
    print("OK: binary_probe tests pass")
