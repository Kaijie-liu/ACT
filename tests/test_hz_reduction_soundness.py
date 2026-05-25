#===- tests/test_hz_reduction_soundness.py - reduction soundness tests -===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Pin three independent soundness bugs found in HZ reductions:
#
#   1) _qr_pivoted_cpu had overwrite_a=True; caller read Ac_eq[:, free_idx]
#      AFTER the QR, getting garbage and a wrong substitution matrix.
#   2) project_eq_elim merged removed Gc cols into a single shared (n,1)
#      box col → artificial cross-output correlation, set shrinks.
#   3) _hz_reduce_constraints (Girard cap) had the same shared-col pattern.
#   Plus a precision regression: _hz_reduce_constraints must not apply
#   row-rank elimination to inequalities because parallel rows with
#   different RHS are not redundant (dropping one is sound but looser).
#
#   Plus a shape invariant for the new convex-hull ReLU encoding (Ab0 must
#   be retained when nb0==0).
#
#   Also verifies the chull routing: with a method-specific peak estimate,
#   convex_hull_cont must execute (output gets +2k constraint rows) rather
#   than silently downgrading to DeepZ triangle (which adds 0 rows).
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import math
import numpy as np
import torch
from scipy.optimize import linprog

from act.back_end.solver.solver_hz import HZono, _hz_reduce_constraints
from act.back_end.hybridz_tf.algorithms.eq_elim import (
    project_eq_elim, _qr_pivoted_cpu,
)
from act.back_end.hybridz_tf.algorithms.relu_methods import (
    hz_apply_relu_convex_hull,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _support(hz: HZono, d: np.ndarray) -> float:
    """LP support of HZ in direction d, with binaries relaxed to [-1, +1]."""
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    nc = int(hz.b.shape[0])
    p = ng + nb
    c_obj = np.zeros(p)
    if ng > 0:
        c_obj[:ng] = -(d @ hz.Gc.cpu().numpy())
    if nb > 0:
        c_obj[ng:] = -(d @ hz.Gb.cpu().numpy())
    em = (hz.eq_mask.cpu().numpy()
          if hz.eq_mask is not None and hz.eq_mask.numel() == nc
          else np.ones(nc, dtype=bool))
    A_full = np.zeros((nc, p))
    if ng > 0:
        A_full[:, :ng] = hz.Ac.cpu().numpy()
    if nb > 0:
        A_full[:, ng:] = hz.Ab.cpu().numpy()
    b_full = hz.b.cpu().numpy().reshape(-1)
    A_ub = A_full[~em] if (~em).any() else None
    b_ub = b_full[~em] if (~em).any() else None
    A_eq = A_full[em] if em.any() else None
    b_eq = b_full[em] if em.any() else None
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(-1.0, 1.0)] * p, method="highs")
    if not res.success:
        return float("nan")
    return -res.fun + float(d @ hz.c.cpu().numpy().reshape(-1))


# ---------------------------------------------------------------------------
# Bug 1: _qr_pivoted_cpu overwrite_a soundness
# ---------------------------------------------------------------------------


def test_qr_pivoted_cpu_does_not_overwrite_input():
    """The caller (project_eq_elim) reads A[:, free_idx] AFTER the QR.
    overwrite_a=True corrupted A and silently produced a wrong substitution
    matrix; this test pins the input-preservation invariant.
    """
    A = np.array([[-0.298, 0.091, -0.428, 0.550]], dtype=np.float64)
    A_orig = A.copy()
    Q, R, piv, rank = _qr_pivoted_cpu(A, rank_tol=1e-10)
    assert np.allclose(A, A_orig), \
        "QR helper must not mutate its input array"
    assert rank == 1


def test_pee_substitution_matrix_correct_2x4():
    """End-to-end algebra check: with the overwrite_a fix, the M_Ac_free
    third entry (free_idx[2]=xi_0, Ac_eq[0,0]=-0.298) must be
    1/0.550 * -0.298 = -0.542, NOT 1.817 (= 1/R[0,0]).
    """
    Ac_eq = np.array([[-0.298, 0.091, -0.428, 0.550]], dtype=np.float64)
    Q, R, piv, rank = _qr_pivoted_cpu(Ac_eq, rank_tol=1e-10)
    free_idx = piv[rank:]
    R_dep = R[:rank, :rank]
    from scipy.linalg import solve_triangular
    M = solve_triangular(R_dep, Q[:, :rank].T)
    M_Ac_free = M @ Ac_eq[:, free_idx]
    # Third entry corresponds to xi_0 (Ac_eq[0,0]=-0.298): M @ -0.298 ≈ -0.542.
    assert M_Ac_free.shape == (1, 3)
    assert abs(M_Ac_free[0, 2] - (-0.542)) < 0.001, \
        f"M_Ac_free[0,2] should be ~-0.542 (i.e. 1/0.550 * -0.298); got {M_Ac_free[0, 2]}"


# ---------------------------------------------------------------------------
# Bug 2: project_eq_elim merge soundness (diagonal slack, not shared col)
# ---------------------------------------------------------------------------


def test_pee_merge_uses_diagonal_slack_not_shared_col():
    """Pre-fix: PEE merged removed Gc cols into shared (n,1) col, which
    introduces artificial cross-output correlation. Post-fix: must NOT
    shrink the set in any sampled direction (containment).

    Setup chosen to FORCE the diagonal-slack path: small n, large ng_free,
    small ng_base so n_to_merge_target > n_out.
    """
    # n=2, ng=12, nc_eq=1, ng_base=1 → free=11, keep=1, merge=10 cols → diag path.
    g = torch.Generator().manual_seed(10)
    Gc = torch.randn(2, 12, generator=g, dtype=torch.float64) * 0.3
    c = torch.randn(2, 1, generator=g, dtype=torch.float64) * 0.2
    Ac = torch.randn(1, 12, generator=g, dtype=torch.float64) * 0.4
    Ab = torch.zeros(1, 0, dtype=torch.float64)
    b = torch.zeros(1, 1, dtype=torch.float64)
    Gb = torch.zeros(2, 0, dtype=torch.float64)
    em = torch.ones(1, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    hz_red = project_eq_elim(hz, ng_base=1)

    # Confirm the diagonal-slack path actually fired (output ng > ng_base).
    assert int(hz_red.Gc.shape[1]) >= 2, \
        "Expected the diag-slack reduce path to be taken"

    rng = np.random.default_rng(seed=99)
    for _ in range(40):
        d = rng.standard_normal(2)
        s_o = _support(hz, d)
        s_r = _support(hz_red, d)
        if math.isnan(s_o) or math.isnan(s_r):
            continue
        assert s_r >= s_o - 1e-8, (
            f"PEE under-approximated: d={d.tolist()} "
            f"orig_support={s_o:.4f} reduced_support={s_r:.4f}"
        )


# ---------------------------------------------------------------------------
# Bug 3: _hz_reduce_constraints (Girard cap) merge soundness
# ---------------------------------------------------------------------------


def test_girard_reduce_uses_diagonal_slack_not_shared_col():
    """2D HZ with Gc=I, ng_budget=1: pre-fix support([1,-1]) collapsed
    2.0 → 0.0; post-fix the cap is unachievable for n=2,budget=1 so the
    skip-guard fires and support is preserved exactly.

    Plus: diag-slack containment on configs where the cap IS achievable
    (n=2, ng=12, ng_budget=6 → keep_k=4, slack=2 → output ng=6).
    """
    # Skip-guard case: cap unachievable, must pass through unchanged.
    hz_id = HZono(
        c=torch.zeros(2, 1, dtype=torch.float64),
        Gc=torch.eye(2, dtype=torch.float64),
        Gb=torch.zeros(2, 0, dtype=torch.float64),
        Ac=torch.zeros(0, 2, dtype=torch.float64),
        Ab=torch.zeros(0, 0, dtype=torch.float64),
        b=torch.zeros(0, 1, dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    hz_red = _hz_reduce_constraints(hz_id, ng_budget=1)
    for d in [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
              np.array([1.0, -1.0]), np.array([1.0, 1.0])]:
        s_o = _support(hz_id, d)
        s_r = _support(hz_red, d)
        assert s_r >= s_o - 1e-9, (
            f"Girard skip-guard violated containment: d={d.tolist()} "
            f"{s_o} -> {s_r}"
        )

    # Diag-slack-path case: cap achievable, diag widening must be sound.
    g = torch.Generator().manual_seed(20)
    Gc = torch.randn(2, 12, generator=g, dtype=torch.float64) * 0.3
    c = torch.zeros(2, 1, dtype=torch.float64)
    Gb = torch.zeros(2, 0, dtype=torch.float64)
    Ac = torch.zeros(0, 12, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    em = torch.zeros(0, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    hz_red = _hz_reduce_constraints(hz, ng_budget=6)
    assert int(hz_red.Gc.shape[1]) <= 6
    rng = np.random.default_rng(seed=101)
    for _ in range(40):
        d = rng.standard_normal(2)
        s_o = _support(hz, d)
        s_r = _support(hz_red, d)
        if math.isnan(s_o) or math.isnan(s_r):
            continue
        assert s_r >= s_o - 1e-8, (
            f"Girard diag-slack violated containment: d={d.tolist()} "
            f"{s_o:.4f} -> {s_r:.4f}"
        )


def test_reduce_rank_phase_preserves_parallel_inequality_tightness():
    """Rank-one inequalities with different RHS must retain the tight bound.

    Pre-fix Phase 1.3 ran QR over all constraint rows, so the pair
    ``xi <= 0.8`` and ``xi <= 0.2`` could be replaced by the first row,
    changing support from 0.2 to 0.8. This is a sound widening, but it is
    avoidable precision loss; inequality row rank alone is not redundancy.
    """
    hz = HZono(
        c=torch.zeros(1, 1, dtype=torch.float64),
        Gc=torch.ones(1, 1, dtype=torch.float64),
        Gb=torch.zeros(1, 0, dtype=torch.float64),
        Ac=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
        Ab=torch.zeros(2, 0, dtype=torch.float64),
        b=torch.tensor([[0.8], [0.2]], dtype=torch.float64),
        eq_mask=torch.zeros(2, dtype=torch.bool),
    )
    reduced = _hz_reduce_constraints(hz)
    original_support = _support(hz, np.array([1.0]))
    reduced_support = _support(reduced, np.array([1.0]))
    assert abs(original_support - 0.2) < 1e-9
    assert abs(reduced_support - original_support) < 1e-9, (
        f"inequality rank elimination dropped the tight RHS: "
        f"{original_support} -> {reduced_support}"
    )


def test_constraint_aware_girard_retains_hull_carrying_factor():
    """Constraint-aware ranking can avoid widening away a useful cut.

    Both reductions remain sound. With output-only ranking the small
    constrained factor is removed and its inequality is widened; when its
    constraint contribution is ranked too, the original support is retained.
    """
    hz = HZono(
        c=torch.zeros(2, 1, dtype=torch.float64),
        Gc=torch.tensor(
            [[10.0, 0.0, 0.5, 0.2],
             [0.0, 1.0, 0.1, -0.3]],
            dtype=torch.float64,
        ),
        Gb=torch.zeros(2, 0, dtype=torch.float64),
        Ac=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
        Ab=torch.zeros(1, 0, dtype=torch.float64),
        b=torch.tensor([[0.0]], dtype=torch.float64),
        eq_mask=torch.zeros(1, dtype=torch.bool),
    )
    output_only = _hz_reduce_constraints(hz, ng_budget=3)
    cut_aware = _hz_reduce_constraints(
        hz, ng_budget=3, constraint_keep_weight=20.0
    )
    direction = np.array([0.0, 1.0])
    original = _support(hz, direction)
    loose = _support(output_only, direction)
    retained = _support(cut_aware, direction)
    assert abs(original - 0.4) < 1e-9
    assert loose > original + 0.9, (
        f"test setup did not widen the removed constrained factor: {loose}"
    )
    assert abs(retained - original) < 1e-9, (
        f"constraint-aware retention failed to preserve useful cut: "
        f"{original} -> {retained}"
    )


# ---------------------------------------------------------------------------
# Memory blocker guard: dense diag is O(n^2), must not blow up at n=25088
# ---------------------------------------------------------------------------


def test_pee_skip_merge_when_diag_slack_would_expand():
    """Skip-merge guard: at large n the dense (n,n) diag slack is O(n^2)
    (~5 GiB at n=25088). When n_out >= ng_free - keep_count, the diag
    slack would EXPAND the generator count instead of reducing it; PEE
    must skip the merge.

    Test: n=512, ng_free=4 after one rank-1 equality → skip-merge fires.
    """
    n = 512
    g = torch.Generator().manual_seed(30)
    Gc = torch.randn(n, 4, generator=g, dtype=torch.float64) * 0.1
    c = torch.zeros(n, 1, dtype=torch.float64)
    Gb = torch.zeros(n, 0, dtype=torch.float64)
    Ac = torch.randn(1, 4, generator=g, dtype=torch.float64) * 0.4
    Ab = torch.zeros(1, 0, dtype=torch.float64)
    b = torch.zeros(1, 1, dtype=torch.float64)
    em = torch.ones(1, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    hz_red = project_eq_elim(hz, ng_base=1)
    # If the guard didn't fire, output ng would be 1 + 512 = 513 (and we'd
    # have allocated a 512×512 = ~2 MiB Gc, plus larger constraint blocks).
    # With the guard, output ng = ng_free = 3.
    assert int(hz_red.Gc.shape[1]) <= 4, \
        f"Skip-merge guard failed: got ng={int(hz_red.Gc.shape[1])}"


def test_girard_skip_when_cap_unachievable():
    """n=512, ng_budget=4: cap < n_dim, can't add 512 diag-slack cols and
    still fit budget. Reduce must skip — output ng unchanged.
    """
    n = 512
    g = torch.Generator().manual_seed(31)
    Gc = torch.randn(n, 100, generator=g, dtype=torch.float64) * 0.05
    c = torch.zeros(n, 1, dtype=torch.float64)
    Gb = torch.zeros(n, 0, dtype=torch.float64)
    Ac = torch.zeros(0, 100, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    em = torch.zeros(0, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    hz_red = _hz_reduce_constraints(hz, ng_budget=4)
    assert int(hz_red.Gc.shape[1]) == 100, \
        f"Girard skip-guard failed: got ng={int(hz_red.Gc.shape[1])}"


# ---------------------------------------------------------------------------
# Bug 4: chull Ab0 row count invariant
# ---------------------------------------------------------------------------


def test_chull_preserves_ab0_when_nb0_zero():
    """Pre-fix: when nb0==0 and nc0>0, Ab0 was silently dropped, leaving
    Ab with 2k rows while Ac/b/eq_mask have nc0+2k. Post-fix: row count
    invariants across Ac/Ab/b/eq_mask must hold.
    """
    Gc = torch.tensor([[1.0]], dtype=torch.float64)
    Gb = torch.zeros(1, 0, dtype=torch.float64)
    c = torch.tensor([[0.0]], dtype=torch.float64)
    Ac = torch.tensor([[1.0]], dtype=torch.float64)
    Ab = torch.zeros(1, 0, dtype=torch.float64)
    b = torch.tensor([[0.5]], dtype=torch.float64)
    em = torch.zeros(1, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    out = hz_apply_relu_convex_hull(hz)
    assert int(out.Ab.shape[0]) == int(out.Ac.shape[0]), \
        f"Ab/Ac row mismatch: {out.Ab.shape} vs {out.Ac.shape}"
    assert int(out.b.shape[0]) == int(out.Ac.shape[0])
    assert int(out.eq_mask.shape[0]) == int(out.Ac.shape[0])


# ---------------------------------------------------------------------------
# Routing: chull must execute when requested (not silently downgrade)
# ---------------------------------------------------------------------------


def test_chull_routing_actually_executes_not_downgraded():
    """The eq_native peak budget would force a downgrade to triangle if it
    were applied unconditionally. Method-specific peak estimate means
    convex_hull_cont's cheaper budget is used, so chull actually runs.

    Distinguishing signature: chull adds 2*k inequality rows per unstable;
    triangle (DeepZ parallelogram) adds 0 rows. We verify the output nc
    grew by 2*k_unstable.
    """
    import os
    from act.back_end.hybridz_tf.hz_routing import hz_apply_relu_v8

    # Build a moderately-sized HZ where ALL neurons are unstable.
    n = 64
    g = torch.Generator().manual_seed(40)
    # c straddles 0, generators within [-0.5, 0.5] → unstable bounds.
    c = torch.zeros(n, 1, dtype=torch.float64)
    Gc = torch.eye(n, dtype=torch.float64) * 0.5
    Gb = torch.zeros(n, 0, dtype=torch.float64)
    Ac = torch.zeros(0, n, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    em = torch.zeros(0, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)

    # Force a tiny budget that would force eq_native → triangle, but
    # chull's smaller estimate should still fit.
    old_budget = os.environ.get("HYZOR_RELU_MEM_BUDGET_GB")
    os.environ["HYZOR_RELU_MEM_BUDGET_GB"] = "0.001"
    try:
        out_chull = hz_apply_relu_v8(hz, method="convex_hull_cont")
        out_tri = hz_apply_relu_v8(hz, method="triangle")
    finally:
        if old_budget is None:
            del os.environ["HYZOR_RELU_MEM_BUDGET_GB"]
        else:
            os.environ["HYZOR_RELU_MEM_BUDGET_GB"] = old_budget

    # Triangle adds 0 constraint rows; chull adds 2*k_unstable = 2*n.
    # If routing silently downgraded chull → triangle, out_chull.b.shape
    # would equal 0.
    assert int(out_chull.b.shape[0]) == 2 * n, (
        f"chull was silently downgraded: expected nc=2n={2*n} rows, "
        f"got {int(out_chull.b.shape[0])} (triangle has "
        f"{int(out_tri.b.shape[0])})"
    )


# ---------------------------------------------------------------------------
# Algebra equivalence (reductions disabled): chull == eq_lagr_v8 LP-relax
# ---------------------------------------------------------------------------


def test_chull_vs_eq_lagr_v8_lp_relax_equivalent_raw():
    """Raw operator equivalence: LP support of hz_apply_relu_convex_hull
    matches LP support of hz_apply_relu (eq_lagr_v8 raw) under binary
    relaxation, with NO PEE / Girard reduction on either side.

    This is the algebraic claim. The full-pipeline comparison is a
    different (engineering) question because the production eq_lagr_v8
    path runs PEE downstream.
    """
    from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu

    g = torch.Generator().manual_seed(7)
    n, ng = 8, 6
    c = torch.randn(n, 1, generator=g, dtype=torch.float64) * 0.3
    Gc = torch.randn(n, ng, generator=g, dtype=torch.float64) * 0.4
    Gb = torch.zeros(n, 0, dtype=torch.float64)
    Ac = torch.zeros(0, ng, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    em = torch.zeros(0, dtype=torch.bool)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)

    hz_eq = hz_apply_relu(hz)             # eq_lagr_v8 raw (no PEE)
    hz_ch = hz_apply_relu_convex_hull(hz)

    rng = np.random.default_rng(seed=42)
    for _ in range(20):
        d = rng.standard_normal(n)
        s_eq = _support(hz_eq, d)
        s_ch = _support(hz_ch, d)
        if math.isnan(s_eq) or math.isnan(s_ch):
            continue
        # Should match to LP tolerance — both encode the same per-neuron
        # triangle convex hull joined with input correlation.
        assert abs(s_eq - s_ch) < 1e-8, (
            f"Raw operator equivalence violated in direction d: "
            f"eq_lagr_v8={s_eq:.6e} chull={s_ch:.6e} diff={s_eq - s_ch:.2e}"
        )


def test_tail_preserve_dim_skips_reduce_below_threshold():
    """Property-facing tail preservation: when ``tail_preserve_dim`` is set
    and ``hz.dim < tail_preserve_dim``, _maybe_reduce must return the HZ
    unchanged (skipping the reduce is always sound — it's the identity).

    Soundness rationale: reduce_constraints replaces removed Gc cols with
    diagonal slack which is a *widening*. Skipping it preserves the
    original (tighter) representation; never adds slack. The skip is
    safe in all directions.

    This test pins soundness/mechanics only. Whether skipping a reduction
    improves verification strength is configuration-dependent and is
    measured separately.
    """
    from act.back_end.solver.solver_hz import HZVerifier
    # Build a small HZ that would normally trigger reduce (ng > cap).
    n = 50  # below threshold
    ng = 100  # above cap
    g = torch.Generator().manual_seed(50)
    hz = HZono(
        c=torch.zeros(n, 1, dtype=torch.float64),
        Gc=torch.randn(n, ng, generator=g, dtype=torch.float64) * 0.1,
        Gb=torch.zeros(n, 0, dtype=torch.float64),
        Ac=torch.zeros(0, ng, dtype=torch.float64),
        Ab=torch.zeros(0, 0, dtype=torch.float64),
        b=torch.zeros(0, 1, dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    # Verifier with tail_preserve_dim=256, girard_cap=60.
    # With preservation disabled this cap is achievable: n=50 < cap=60.
    solver = HZVerifier(girard_cap=60, tail_preserve_dim=256)
    out = solver._maybe_reduce(hz)
    # dim=50 < threshold 256 → reduce skipped → identity.
    assert int(out.Gc.shape[1]) == ng, (
        f"tail_preserve should skip reduce when dim < threshold; "
        f"expected ng={ng}, got {int(out.Gc.shape[1])}"
    )
    # And containment: every direction preserves support exactly.
    rng = np.random.default_rng(seed=77)
    for _ in range(20):
        d = rng.standard_normal(n)
        s_o = _support(hz, d)
        s_r = _support(out, d)
        assert abs(s_o - s_r) < 1e-12, \
            f"tail_preserve skip should be identity; got {s_o} vs {s_r}"

    # Now verify the opposite: with tail_preserve_dim=0 (disabled), the
    # same HZ goes through reduce_constraints (the test isn't asserting
    # the reduced support, just that reduce was actually invoked).
    solver2 = HZVerifier(girard_cap=60, tail_preserve_dim=0)
    out2 = solver2._maybe_reduce(hz)
    # The sound reduction keeps cap-n independent columns plus n diagonal
    # slack columns, exactly filling the achievable cap.
    assert int(out2.Gc.shape[0]) == n
    assert int(out2.Gc.shape[1]) == 60, (
        f"with tail preservation disabled, reduction should execute; "
        f"expected ng=60, got {int(out2.Gc.shape[1])}"
    )


def test_selective_chull_endpoints_and_soundness():
    """Selective chull endpoint properties:
    - top_k=0 (no mask): result LP support equals DeepZ triangle.
    - top_k=n (full mask on unstable): result LP support equals full chull.
    - 0 < top_k < n: support is between (inclusive) full triangle and chull
      in every direction — tighter than triangle, looser than chull.
    """
    from act.back_end.hybridz_tf.algorithms.relu_methods import (
        hz_apply_relu_triangle, hz_apply_relu_convex_hull,
        hz_apply_relu_selective_chull,
    )
    g = torch.Generator().manual_seed(42)
    n, ng = 6, 8
    hz = HZono(
        c=torch.randn(n, 1, generator=g, dtype=torch.float64) * 0.2,
        Gc=torch.randn(n, ng, generator=g, dtype=torch.float64) * 0.4,
        Gb=torch.zeros(n, 0, dtype=torch.float64),
        Ac=torch.zeros(0, ng, dtype=torch.float64),
        Ab=torch.zeros(0, 0, dtype=torch.float64),
        b=torch.zeros(0, 1, dtype=torch.float64),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    hz_tri = hz_apply_relu_triangle(hz)
    hz_chull = hz_apply_relu_convex_hull(hz)
    hz_sel_empty = hz_apply_relu_selective_chull(hz)
    hz_sel_full = hz_apply_relu_selective_chull(hz, top_k=n)
    hz_sel_partial = hz_apply_relu_selective_chull(hz, top_k=3)
    rng = np.random.default_rng(seed=43)
    for _ in range(30):
        d = rng.standard_normal(n)
        s_tri = _support(hz_tri, d)
        s_chull = _support(hz_chull, d)
        s_empty = _support(hz_sel_empty, d)
        s_full = _support(hz_sel_full, d)
        s_partial = _support(hz_sel_partial, d)
        if any(math.isnan(s) for s in (s_tri, s_chull, s_empty, s_full, s_partial)):
            continue
        # Endpoint: empty = triangle, full = chull.
        assert abs(s_empty - s_tri) < 1e-9, \
            f"sel_empty != triangle: {s_empty} vs {s_tri}"
        assert abs(s_full - s_chull) < 1e-8, \
            f"sel_full != chull: {s_full} vs {s_chull}"
        # Soundness sandwich.
        assert s_partial >= s_chull - 1e-9, \
            f"partial selective TIGHTER than chull: {s_partial} < {s_chull}"
        assert s_partial <= s_tri + 1e-9, \
            f"partial selective LOOSER than triangle: {s_partial} > {s_tri}"


def test_sparse_selective_chull_carries_facets_without_densifying_gc():
    """Sparse early ReLU cuts retain the full-chull endpoint in factor LP.

    The sparse representation must hold the new inequality rows without
    materialising its wide Gc matrix. Once promoted explicitly to HZono,
    full selection has the same support as dense convex_hull_cont.
    """
    from act.back_end.hybridz_tf.representations import SparseGcZ

    c = torch.zeros(2, dtype=torch.float64)
    Gc_sp = torch.eye(2, dtype=torch.float64).to_sparse()
    sparse = SparseGcZ(
        c=c, Gc_sparse=Gc_sp, dtype=torch.float64, device=torch.device("cpu")
    )
    selected = sparse.apply_relu_selective_chull(top_k=2)
    assert isinstance(selected, SparseGcZ)
    assert selected.Gc_sparse.is_sparse
    assert selected.nc == 4
    assert int(selected.Ac_sparse.shape[1]) == selected.ng

    dense_selected = selected.to_hzono()
    dense_full = hz_apply_relu_convex_hull(sparse.to_hzono())
    for direction in (np.array([1.0, -1.0]), np.array([1.0, 1.0]),
                      np.array([-1.0, 1.0])):
        assert abs(_support(dense_selected, direction) -
                   _support(dense_full, direction)) < 1e-9

    # Exact affine operations preserve the sparse factor constraints.
    scaled = selected.apply_scale(torch.tensor([2.0, 0.5], dtype=torch.float64))
    assert scaled.nc == selected.nc
    assert torch.equal(scaled.eq_mask, selected.eq_mask)

    # A caller may provide independently certified tighter bounds so sparse
    # constraints can affect the next ReLU without densifying Gc.
    constrained = SparseGcZ(
        c=torch.zeros(1, dtype=torch.float64),
        Gc_sparse=torch.ones(1, 1, dtype=torch.float64).to_sparse(),
        Ac_sparse=torch.ones(1, 1, dtype=torch.float64).to_sparse(),
        b=torch.zeros(1, 1, dtype=torch.float64),
        eq_mask=torch.zeros(1, dtype=torch.bool),
        dtype=torch.float64, device=torch.device("cpu"),
    )
    bounded = constrained.apply_relu_triangle(
        external_bounds=(
            torch.tensor([-1.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        )
    )
    assert abs(_support(bounded.to_hzono(), np.array([1.0]))) < 1e-12


def test_small_dense_witness_mode_routes_to_sat_with_witness():
    """``small_dense_lp='witness'`` dispatches via WitnessExtract; ORT-confirmed
    falsified verdict must route to SolveStatus.SAT with a valid witness.

    Soundness rationale: WitnessExtract.verify_with_falsification only returns
    'falsified' when ORT replay confirms NN(witness) violates the spec. So:
      - verdict='falsified' ⇒ real counterexample ⇒ SAT (sound)
      - verdict='verified' ⇒ SpecAware proved infeasible ⇒ UNSAT (sound)
      - verdict='unknown' ⇒ no decision ⇒ UNKNOWN
    """
    import sys
    import types
    import numpy as np
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier

    old_base = sys.modules.get("GlobalTriangleLP")
    old_we = sys.modules.get("WitnessExtract")
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.extract_layers = lambda _: (
        None,
        [(np.zeros((3, 128)), np.zeros(128))],
        (np.zeros((128, 2)), np.zeros(2)),
    )
    base.verify = lambda *_, **__: ("verified", 0.01)
    sys.modules["GlobalTriangleLP"] = base

    # Stub WitnessExtract returning 'falsified' with a synthetic witness.
    fake_witness = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    fake_y = np.array([1.0, 2.0], dtype=np.float64)
    we = types.ModuleType("WitnessExtract")
    called = {}
    def _falsify(*_args, **kwargs):
        called.update(kwargs)
        return "falsified", fake_witness, fake_y, 0.05
    we.verify_with_falsification = _falsify
    sys.modules["WitnessExtract"] = we
    try:
        # CONTRACT CHANGE (advisor 2026-05-24): _try_small_dense_lp called
        # without net+assert_layer MUST fail-closed (return UNKNOWN), not
        # promote external falsified to SAT. The old SAT-bypass path was
        # the soundness leak the strict-replay gate was introduced to plug.
        s = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        assert s._try_small_dense_lp(conv_count=0) == SolveStatus.UNKNOWN
        assert s._stats["small_dense_lp_backend"] == "witness"
        assert s._stats["small_dense_lp_verdict"] == "falsified"
        assert s._stats["small_dense_lp_refinement_passes"] == 20
        assert s._stats["small_dense_lp_refinement_policy"] == "shallow_20"
        assert s._stats.get("small_dense_lp_strict_replay_unavailable") is True
        assert called["max_refinement_passes"] == 20
        assert not s.has_solution()

        # Explicit audit configuration overrides the adaptive shallow policy;
        # contract still fail-closed without context.
        s_explicit = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
            small_dense_lp_refinement_passes=3,
        )
        assert s_explicit._try_small_dense_lp(conv_count=0) == SolveStatus.UNKNOWN
        assert s_explicit._stats["small_dense_lp_refinement_passes"] == 3
        assert s_explicit._stats["small_dense_lp_refinement_policy"] == "explicit"
        assert called["max_refinement_passes"] == 3

        # Verdict='verified' through witness mode → UNSAT
        we.verify_with_falsification = lambda *_a, **kw: (
            "verified", None, None, 0.05
        )
        s2 = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        assert s2._try_small_dense_lp(conv_count=0) == SolveStatus.UNSAT

        # Verdict='unknown' → UNKNOWN (default fallback_on_unknown=False)
        we.verify_with_falsification = lambda *_a, **kw: (
            "unknown", None, None, 0.05
        )
        s3 = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        assert s3._try_small_dense_lp(conv_count=0) == SolveStatus.UNKNOWN
    finally:
        if old_base is None:
            sys.modules.pop("GlobalTriangleLP", None)
        else:
            sys.modules["GlobalTriangleLP"] = old_base
        if old_we is None:
            sys.modules.pop("WitnessExtract", None)
        else:
            sys.modules["WitnessExtract"] = old_we


def test_small_dense_strict_replay_blocks_phantom_falsified():
    """SOUNDNESS GATE (advisor 2026-05-24):

    The external WitnessExtract backend's `_ort_replay` uses +1e-6
    slack (small_tol acceptance). A 'falsified' verdict from it can be
    a boundary witness rather than a hard FAL. ``_try_small_dense_lp``
    must therefore re-run ACT's strict zero-tol replay on x* before
    promoting to SAT, and downgrade to UNKNOWN when the strict replay
    rejects.

    This test stubs WitnessExtract to return 'falsified' with a synthetic
    witness, but monkey-patches ACT's strict_replay_for_act to return
    False (= the model's output at x* does NOT actually violate the
    spec under zero tolerance). The result must be SolveStatus.UNKNOWN
    with the ``small_dense_lp_phantom_rejected`` stat set.
    """
    import sys
    import types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier

    old_base = sys.modules.get("GlobalTriangleLP")
    old_we = sys.modules.get("WitnessExtract")
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.extract_layers = lambda _: (
        None,
        [(np.zeros((3, 128)), np.zeros(128))],
        (np.zeros((128, 2)), np.zeros(2)),
    )
    base.verify = lambda *_, **__: ("verified", 0.01)
    sys.modules["GlobalTriangleLP"] = base

    fake_witness = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    we = types.ModuleType("WitnessExtract")
    we.verify_with_falsification = lambda *_a, **kw: (
        "falsified", fake_witness, np.array([1.0, 2.0]), 0.05,
    )
    sys.modules["WitnessExtract"] = we

    # Synthetic net + assert_layer carriers (not used by stubbed replay).
    fake_net = types.SimpleNamespace(onnx_path="/tmp/fake.onnx")
    fake_assert = types.SimpleNamespace(params={})

    # Monkey-patch strict_replay_for_act to return False (= safe under
    # zero-tol; the external 'falsified' was a +1e-6 boundary artefact).
    original_strict = solver_hz_mod.strict_replay_for_act
    solver_hz_mod.strict_replay_for_act = lambda **kwargs: False
    try:
        s = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        result = s._try_small_dense_lp(
            conv_count=0, net=fake_net, assert_layer=fake_assert,
        )
        assert result == SolveStatus.UNKNOWN, (
            f"phantom (+1e-6 boundary) FAL must downgrade to UNKNOWN; "
            f"got {result}. This is the soundness gate that prevents "
            f"WitnessExtract's small_tol slack from leaking into paper-"
            f"grade SAT claims."
        )
        assert s._stats.get("small_dense_lp_phantom_rejected") is True, (
            "phantom-rejected stat flag must be set for audit visibility"
        )
        assert not s.has_solution(), "rejected witness must NOT remain on solver"
    finally:
        solver_hz_mod.strict_replay_for_act = original_strict
        if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
        else: sys.modules["GlobalTriangleLP"] = old_base
        if old_we is None: sys.modules.pop("WitnessExtract", None)
        else: sys.modules["WitnessExtract"] = old_we


def test_small_dense_strict_replay_confirms_real_falsified():
    """The complement of the phantom-rejection test: when ACT's strict
    zero-tol replay CONFIRMS the witness genuinely violates the spec,
    promote to SAT and stash the witness for caller extraction. The
    ``small_dense_lp_strict_replay_passed`` stat flag must be set.
    """
    import sys
    import types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier

    old_base = sys.modules.get("GlobalTriangleLP")
    old_we = sys.modules.get("WitnessExtract")
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.extract_layers = lambda _: (
        None,
        [(np.zeros((3, 128)), np.zeros(128))],
        (np.zeros((128, 2)), np.zeros(2)),
    )
    base.verify = lambda *_, **__: ("verified", 0.01)
    sys.modules["GlobalTriangleLP"] = base

    fake_witness = np.array([0.7, -0.4, 0.1], dtype=np.float64)
    we = types.ModuleType("WitnessExtract")
    we.verify_with_falsification = lambda *_a, **kw: (
        "falsified", fake_witness, np.array([5.0, 0.0]), 0.05,
    )
    sys.modules["WitnessExtract"] = we

    fake_net = types.SimpleNamespace(onnx_path="/tmp/fake.onnx")
    fake_assert = types.SimpleNamespace(params={})

    original_strict = solver_hz_mod.strict_replay_for_act
    solver_hz_mod.strict_replay_for_act = lambda **kwargs: True
    try:
        s = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        result = s._try_small_dense_lp(
            conv_count=0, net=fake_net, assert_layer=fake_assert,
        )
        assert result == SolveStatus.SAT, (
            f"strict-replay-confirmed FAL must remain SAT; got {result}"
        )
        assert s._stats.get("small_dense_lp_strict_replay_passed") is True
        assert s.has_solution()
        assert np.allclose(s._witness, fake_witness)
    finally:
        solver_hz_mod.strict_replay_for_act = original_strict
        if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
        else: sys.modules["GlobalTriangleLP"] = old_base
        if old_we is None: sys.modules.pop("WitnessExtract", None)
        else: sys.modules["WitnessExtract"] = old_we


def test_fal_receipt_writes_zero_and_small_tol_columns():
    """Receipt layer must record BOTH zero_tol and small_tol verdicts as
    independent fields (per advisor 2026-05-24: two-tier reporting must
    survive into provenance). The receipt write must be idempotent on
    artifact-disk side (atomic JSON + .npy)."""
    import os, json, tempfile, types
    from pathlib import Path
    import numpy as np
    import act.back_end.solver.fal_receipt as fr

    # Build a tiny ONNX model: y = identity(x), 2-dim
    import onnx
    from onnx import helper, TensorProto
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        W = np.eye(2, dtype=np.float32)
        b = np.zeros((2,), dtype=np.float32)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        W_init = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], W.flatten().tolist())
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [2], b.tolist())
        node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"], alpha=1.0, beta=1.0, transB=1)
        graph = helper.make_graph([node], "id2", [X], [Y], initializer=[W_init, b_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx_p = td_p / "id2.onnx"
        onnx.save(model, str(onnx_p))
        spec_p = td_p / "spec.vnnlib"
        spec_p.write_text(
            "(declare-const X_0 Real) (declare-const X_1 Real)\n"
            "(declare-const Y_0 Real) (declare-const Y_1 Real)\n"
            "(assert (>= X_0 0.0)) (assert (<= X_0 1.0))\n"
            "(assert (>= X_1 0.0)) (assert (<= X_1 1.0))\n"
            "(assert (<= Y_0 0.0))\n"
        )

        # Synthesize an UNSAFE_LINEAR assert_layer: spec is "Y_0 <= 0" (one row).
        assert_layer = types.SimpleNamespace(params={
            "kind": "UNSAFE_LINEAR",
            "c": np.array([[1.0, 0.0]]),
            "d": np.array([0.0]),
        })

        receipt_dir = td_p / "receipts"
        # Witness 1: x = [0.5e-7, 0.5] → y_0 = 5e-8, satisfies ≤ 0 + 1e-6 (small_tol)
        # but NOT ≤ 0 strictly. small_tol_holds=True, zero_tol_holds=False.
        x_boundary = np.array([5e-8, 0.5])
        rp = fr.write_receipt(
            x_star=x_boundary,
            model_path=onnx_p, spec_path=spec_p, assert_layer=assert_layer,
            benchmark="testbench", instance_id=42, source="unit",
            receipt_dir=receipt_dir,
        )
        assert rp is not None and rp.exists(), "receipt JSON must be written"
        rec = json.loads(rp.read_text())
        # Strictly y_0 = 5e-8 > 0 → unsafe (LE) fails strictly; with 1e-6 tol passes
        assert rec["spec_zero_tol_holds"] is False, (
            f"x=5e-8 strictly does NOT satisfy Y_0 ≤ 0; got zero={rec['spec_zero_tol_holds']}"
        )
        assert rec["spec_small_tol_holds"] is True, (
            f"x=5e-8 DOES satisfy Y_0 ≤ 0 + 1e-6; got small={rec['spec_small_tol_holds']}"
        )
        # SHA fields populated
        assert len(rec["model_sha256"]) == 64
        assert len(rec["spec_sha256"]) == 64
        assert len(rec["x_star_sha256"]) == 64

        # Witness 2: x = [-0.0001, 0.5] but clipped → outside box; sample x = [0, 0.5]
        # → y_0 = 0, strictly satisfies Y_0 ≤ 0. Both tols True.
        x_strict = np.array([0.0, 0.5])
        rp2 = fr.write_receipt(
            x_star=x_strict,
            model_path=onnx_p, spec_path=spec_p, assert_layer=assert_layer,
            benchmark="testbench", instance_id=43, source="unit",
            receipt_dir=receipt_dir,
        )
        rec2 = json.loads(rp2.read_text())
        assert rec2["spec_zero_tol_holds"] is True
        assert rec2["spec_small_tol_holds"] is True

        # Witness 3: x = [0.5, 0.5] → y_0 = 0.5 > 0 → neither tol holds.
        x_safe = np.array([0.5, 0.5])
        rp3 = fr.write_receipt(
            x_star=x_safe,
            model_path=onnx_p, spec_path=spec_p, assert_layer=assert_layer,
            benchmark="testbench", instance_id=44, source="unit",
            receipt_dir=receipt_dir,
        )
        rec3 = json.loads(rp3.read_text())
        assert rec3["spec_zero_tol_holds"] is False
        assert rec3["spec_small_tol_holds"] is False

        # MANIFEST.csv must list all three (query_index column added per
        # advisor 2026-05-24; default query_index=0 when not multi-OR)
        manifest_csv = (receipt_dir / "MANIFEST.csv").read_text()
        assert manifest_csv.count("\n") == 4  # header + 3
        assert "testbench,42,0,unit,0,0,1," in manifest_csv  # boundary
        assert "testbench,43,0,unit,0,1,1," in manifest_csv  # strict
        assert "testbench,44,0,unit,0,0,0," in manifest_csv  # safe


def test_fal_receipt_disabled_when_env_unset():
    """Receipt write is a no-op when ACT_FAL_RECEIPT_DIR is unset AND no
    explicit receipt_dir is passed. Verdict must never depend on receipt."""
    import os, types
    import numpy as np
    import act.back_end.solver.fal_receipt as fr

    old = os.environ.pop(fr.ENV_RECEIPT_DIR, None)
    try:
        assert_layer = types.SimpleNamespace(params={
            "kind": "UNSAFE_LINEAR",
            "c": np.array([[1.0]]),
            "d": np.array([0.0]),
        })
        rp = fr.write_receipt(
            x_star=np.array([0.5]),
            model_path="/no/such/file.onnx", spec_path="/no/such/file.vnnlib",
            assert_layer=assert_layer,
            benchmark="x", instance_id=0, source="unit",
        )
        assert rp is None, "receipt must return None when no dir is configured"
    finally:
        if old is not None:
            os.environ[fr.ENV_RECEIPT_DIR] = old


def test_fal_receipt_formal_mode_rejects_sentinel_instance_id():
    """Formal-mode receipt MUST refuse a sentinel instance_id (-1).

    Background: the old defaults used instance_id=-1 as fallback, which
    caused filename collisions on benchmarks with model reuse (acasxu).
    Formal-mode (paper-grade audit ledger) closes this loophole."""
    import tempfile, types
    from pathlib import Path
    import numpy as np
    import act.back_end.solver.fal_receipt as fr

    assert_layer = types.SimpleNamespace(params={
        "kind": "UNSAFE_LINEAR",
        "c": np.array([[1.0]]),
        "d": np.array([0.0]),
    })
    with tempfile.TemporaryDirectory() as td:
        try:
            fr.write_receipt(
                x_star=np.array([0.0]),
                model_path="/no/such/file.onnx",
                spec_path="/no/such/file.vnnlib",
                assert_layer=assert_layer,
                benchmark="x", instance_id=-1, source="unit",
                receipt_dir=td, formal_mode=True,
            )
            assert False, "formal-mode must raise on instance_id=-1"
        except fr.ReceiptCollisionError as e:
            assert "sentinel" in str(e).lower() or "-1" in str(e)


def test_fal_receipt_formal_mode_rejects_collision():
    """Two calls with the same (benchmark, instance_id, query_index, source,
    attempt) in formal-mode must raise on the second; the audit ledger
    cannot silently overwrite a prior receipt."""
    import tempfile, types
    from pathlib import Path
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import act.back_end.solver.fal_receipt as fr

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Tiny identity ONNX so compute_ort_y doesn't fail
        W = np.eye(1, dtype=np.float32)
        b = np.zeros((1,), dtype=np.float32)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
        Wi = helper.make_tensor("W", TensorProto.FLOAT, [1, 1], W.flatten().tolist())
        bi = helper.make_tensor("B", TensorProto.FLOAT, [1], b.tolist())
        node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"], alpha=1.0, beta=1.0, transB=1)
        g = helper.make_graph([node], "id1", [X], [Y], initializer=[Wi, bi])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        onnx_p = td_p / "id1.onnx"
        onnx.save(m, str(onnx_p))
        spec_p = td_p / "s.vnnlib"
        spec_p.write_text(
            "(declare-const X_0 Real) (declare-const Y_0 Real)\n"
            "(assert (<= Y_0 0.0))\n"
        )
        receipt_dir = td_p / "rec"
        assert_layer = types.SimpleNamespace(params={
            "kind": "UNSAFE_LINEAR",
            "c": np.array([[1.0]]),
            "d": np.array([0.0]),
        })
        # First call OK
        r1 = fr.write_receipt(
            x_star=np.array([-0.5]), model_path=onnx_p, spec_path=spec_p,
            assert_layer=assert_layer,
            benchmark="b", instance_id=7, query_index=0, source="u", attempt=0,
            receipt_dir=receipt_dir, formal_mode=True,
        )
        assert r1 is not None and r1.exists()
        # Second call with SAME triple → must raise
        try:
            fr.write_receipt(
                x_star=np.array([-0.5]), model_path=onnx_p, spec_path=spec_p,
                assert_layer=assert_layer,
                benchmark="b", instance_id=7, query_index=0, source="u", attempt=0,
                receipt_dir=receipt_dir, formal_mode=True,
            )
            assert False, "formal-mode must raise on collision"
        except fr.ReceiptCollisionError as e:
            assert "collision" in str(e).lower()
        # Same instance_id but different query_index → must succeed (multi-OR)
        r2 = fr.write_receipt(
            x_star=np.array([-0.5]), model_path=onnx_p, spec_path=spec_p,
            assert_layer=assert_layer,
            benchmark="b", instance_id=7, query_index=1, source="u", attempt=0,
            receipt_dir=receipt_dir, formal_mode=True,
        )
        assert r2 is not None and r2.exists() and r1 != r2


def test_formal_mode_no_receipt_dir_keeps_sat_marks_error_receipt():
    """ROUND 4 CORRECTION (advisor 2026-05-24):

    Round 3 over-downgraded to UNKNOWN when receipt was missing,
    making the strict-replay-confirmed adversary disappear from
    solver state. This was DISHONEST: the witness genuinely exists.

    Round 4 contract: internal _status STAYS SAT (math truth), but
    formal_result is set to ERROR_RECEIPT_MISSING so the CLI counts
    it in the ERROR_RECEIPT bucket, NOT in FAL.

    This preserves both soundness (no fake UNKNOWN) and audit honesty
    (no receipt → no reportable FAL).
    """
    import os, sys, types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier, reportable_verdict_for_cli

    old_we = sys.modules.get("WitnessExtract")
    old_base = sys.modules.get("GlobalTriangleLP")
    old_formal = os.environ.get("ACT_FAL_RECEIPT_FORMAL")
    old_dir = os.environ.get("ACT_FAL_RECEIPT_DIR")
    os.environ["ACT_FAL_RECEIPT_FORMAL"] = "1"
    os.environ.pop("ACT_FAL_RECEIPT_DIR", None)

    we = types.ModuleType("WitnessExtract")
    fake = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    we.verify_with_falsification = lambda *_a, **kw: (
        "falsified", fake, np.array([1.0, 2.0]), 0.05,
    )
    sys.modules["WitnessExtract"] = we
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.extract_layers = lambda _: (
        None,
        [(np.zeros((3, 128)), np.zeros(128))],
        (np.zeros((128, 2)), np.zeros(2)),
    )
    base.verify = lambda *_, **__: ("verified", 0.01)
    sys.modules["GlobalTriangleLP"] = base
    original_strict = solver_hz_mod.strict_replay_for_act
    solver_hz_mod.strict_replay_for_act = lambda **kw: True

    fake_net = types.SimpleNamespace(onnx_path="/tmp/fake.onnx")
    fake_assert = types.SimpleNamespace(params={
        "kind": "UNSAFE_LINEAR",
        "c": np.array([[1.0, 0.0]]),
        "d": np.array([0.0]),
    })
    try:
        s = HZVerifier(
            onnx_path="/tmp/net.onnx", vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
            benchmark="testbench", instance_id=42,
        )
        result = s._try_small_dense_lp(
            conv_count=0, net=fake_net, assert_layer=fake_assert,
        )
        # MATH TRUTH: internal status is SAT (witness verified by strict_replay)
        assert result == SolveStatus.SAT, (
            f"strict-replay-confirmed witness must keep internal SAT (math truth); "
            f"got {result}. Round-3-style downgrade to UNKNOWN was DISHONEST."
        )
        assert s.has_solution()
        # AUDIT HONESTY: formal_result is ERROR_RECEIPT_MISSING in formal mode
        assert s._stats.get("formal_result") == "ERROR_RECEIPT_MISSING", (
            f"formal-mode + no receipt_dir must yield ERROR_RECEIPT_MISSING; "
            f"got formal_result={s._stats.get('formal_result')}"
        )
        # CLI mapping: reportable verdict is ERROR_RECEIPT_MISSING (NOT FALSIFIED)
        reportable = reportable_verdict_for_cli(s, "SAT")
        assert reportable == "ERROR_RECEIPT_MISSING", (
            f"CLI reportable must be ERROR_RECEIPT_MISSING in formal mode; got {reportable}"
        )
    finally:
        solver_hz_mod.strict_replay_for_act = original_strict
        if old_we is None: sys.modules.pop("WitnessExtract", None)
        else: sys.modules["WitnessExtract"] = old_we
        if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
        else: sys.modules["GlobalTriangleLP"] = old_base
        if old_formal is None: os.environ.pop("ACT_FAL_RECEIPT_FORMAL", None)
        else: os.environ["ACT_FAL_RECEIPT_FORMAL"] = old_formal
        if old_dir is not None: os.environ["ACT_FAL_RECEIPT_DIR"] = old_dir


def test_formal_mode_receipt_written_emits_sat():
    """Symmetric: with ACT_FAL_RECEIPT_FORMAL=1 AND a real receipt_dir,
    the SAT verdict IS emitted and the receipt path stashed in stats."""
    import os, sys, tempfile, types
    from pathlib import Path
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier

    old_we = sys.modules.get("WitnessExtract")
    old_base = sys.modules.get("GlobalTriangleLP")
    old_formal = os.environ.get("ACT_FAL_RECEIPT_FORMAL")
    old_dir = os.environ.get("ACT_FAL_RECEIPT_DIR")
    os.environ["ACT_FAL_RECEIPT_FORMAL"] = "1"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # tiny ONNX so fal_receipt.compute_ort_y can run
        W = np.eye(2, dtype=np.float32)
        b = np.zeros((2,), dtype=np.float32)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        Wi = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], W.flatten().tolist())
        bi = helper.make_tensor("B", TensorProto.FLOAT, [2], b.tolist())
        node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"], alpha=1.0, beta=1.0, transB=1)
        g = helper.make_graph([node], "id2", [X], [Y], initializer=[Wi, bi])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        onnx_p = td_p / "id2.onnx"
        onnx.save(m, str(onnx_p))
        spec_p = td_p / "s.vnnlib"
        spec_p.write_text(
            "(declare-const X_0 Real) (declare-const X_1 Real)\n"
            "(declare-const Y_0 Real) (declare-const Y_1 Real)\n"
            "(assert (<= Y_0 0.0))\n"
        )
        receipt_dir = td_p / "rec"
        os.environ["ACT_FAL_RECEIPT_DIR"] = str(receipt_dir)

        we = types.ModuleType("WitnessExtract")
        # x = [-0.5, 0.5] → y0 = -0.5 ≤ 0 strictly → zero_tol holds
        fake_x = np.array([-0.5, 0.5], dtype=np.float64)
        we.verify_with_falsification = lambda *_a, **kw: (
            "falsified", fake_x, np.array([-0.5, 0.5]), 0.05,
        )
        sys.modules["WitnessExtract"] = we
        base = types.ModuleType("GlobalTriangleLP")
        base.is_small_dense = lambda _: True
        base.extract_layers = lambda _: (
            None,
            [(np.zeros((2, 128)), np.zeros(128))],
            (np.zeros((128, 2)), np.zeros(2)),
        )
        base.verify = lambda *_, **__: ("verified", 0.01)
        sys.modules["GlobalTriangleLP"] = base
        original_strict = solver_hz_mod.strict_replay_for_act
        solver_hz_mod.strict_replay_for_act = lambda **kw: True

        fake_net = types.SimpleNamespace(onnx_path=str(onnx_p))
        fake_assert = types.SimpleNamespace(params={
            "kind": "UNSAFE_LINEAR",
            "c": np.array([[1.0, 0.0]]),
            "d": np.array([0.0]),
        })
        try:
            s = HZVerifier(
                onnx_path=str(onnx_p), vnnlib_path=str(spec_p),
                small_dense_lp="witness",
                benchmark="testbench", instance_id=77, query_index=2,
            )
            result = s._try_small_dense_lp(
                conv_count=0, net=fake_net, assert_layer=fake_assert,
            )
            assert result == SolveStatus.SAT, (
                f"formal-mode + receipt_dir + zero_tol passing must emit SAT, got {result}"
            )
            assert s._stats.get("fal_receipt_path") is not None
            # query_index must be in the artifact filename
            assert "q2" in s._stats["fal_receipt_path"]
            # Round 4: formal_result must be REPORTABLE_FALSIFIED
            assert s._stats.get("formal_result") == "REPORTABLE_FALSIFIED"
        finally:
            solver_hz_mod.strict_replay_for_act = original_strict
            if old_we is None: sys.modules.pop("WitnessExtract", None)
            else: sys.modules["WitnessExtract"] = old_we
            if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
            else: sys.modules["GlobalTriangleLP"] = old_base
            if old_formal is None: os.environ.pop("ACT_FAL_RECEIPT_FORMAL", None)
            else: os.environ["ACT_FAL_RECEIPT_FORMAL"] = old_formal
            if old_dir is None: os.environ.pop("ACT_FAL_RECEIPT_DIR", None)
            else: os.environ["ACT_FAL_RECEIPT_DIR"] = old_dir


def test_formal_mode_receipt_collision_keeps_sat_marks_collision():
    """ROUND 4: with ACT_FAL_RECEIPT_FORMAL=1 and a colliding receipt
    (same benchmark/iid/query_index already on disk), internal SAT is
    preserved but formal_result is ERROR_RECEIPT_COLLISION."""
    import os, sys, tempfile, types
    from pathlib import Path
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier
    import act.back_end.solver.fal_receipt as fr

    old_we = sys.modules.get("WitnessExtract")
    old_base = sys.modules.get("GlobalTriangleLP")
    old_formal = os.environ.get("ACT_FAL_RECEIPT_FORMAL")
    old_dir = os.environ.get("ACT_FAL_RECEIPT_DIR")
    os.environ["ACT_FAL_RECEIPT_FORMAL"] = "1"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Tiny ONNX
        W = np.eye(2, dtype=np.float32); b = np.zeros((2,), dtype=np.float32)
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        Wi = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], W.flatten().tolist())
        bi = helper.make_tensor("B", TensorProto.FLOAT, [2], b.tolist())
        node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"], alpha=1.0, beta=1.0, transB=1)
        g = helper.make_graph([node], "id2", [X], [Y], initializer=[Wi, bi])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        onnx_p = td_p / "id2.onnx"; onnx.save(m, str(onnx_p))
        spec_p = td_p / "s.vnnlib"
        spec_p.write_text("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(assert (<= Y_0 0.0))\n")
        receipt_dir = td_p / "rec"
        os.environ["ACT_FAL_RECEIPT_DIR"] = str(receipt_dir)

        # Pre-create a colliding receipt by writing one first
        fake_assert = types.SimpleNamespace(params={
            "kind": "UNSAFE_LINEAR",
            "c": np.array([[1.0, 0.0]]), "d": np.array([0.0]),
        })
        first = fr.write_receipt(
            x_star=np.array([-0.5, 0.5]), model_path=onnx_p, spec_path=spec_p,
            assert_layer=fake_assert,
            benchmark="testbench", instance_id=99, query_index=0,
            source="small_dense_lp_witness",
            receipt_dir=receipt_dir, formal_mode=True,
        )
        assert first is not None

        # Now simulate the solver path: it will try to write the SAME
        # (benchmark, instance_id, query_index, source) → collision
        we = types.ModuleType("WitnessExtract")
        we.verify_with_falsification = lambda *_a, **kw: (
            "falsified", np.array([-0.5, 0.5]), np.array([-0.5, 0.5]), 0.05,
        )
        sys.modules["WitnessExtract"] = we
        base = types.ModuleType("GlobalTriangleLP")
        base.is_small_dense = lambda _: True
        base.extract_layers = lambda _: (None, [(np.zeros((2,128)), np.zeros(128))], (np.zeros((128,2)), np.zeros(2)))
        base.verify = lambda *_, **__: ("verified", 0.01)
        sys.modules["GlobalTriangleLP"] = base
        orig_strict = solver_hz_mod.strict_replay_for_act
        solver_hz_mod.strict_replay_for_act = lambda **kw: True

        fake_net = types.SimpleNamespace(onnx_path=str(onnx_p))
        try:
            s = HZVerifier(
                onnx_path=str(onnx_p), vnnlib_path=str(spec_p),
                small_dense_lp="witness",
                benchmark="testbench", instance_id=99, query_index=0,
            )
            result = s._try_small_dense_lp(
                conv_count=0, net=fake_net, assert_layer=fake_assert,
            )
            assert result == SolveStatus.SAT, (
                f"witness genuinely exists; internal SAT must persist; got {result}"
            )
            assert s._stats.get("formal_result") == "ERROR_RECEIPT_COLLISION", (
                f"colliding receipt must yield ERROR_RECEIPT_COLLISION; "
                f"got formal_result={s._stats.get('formal_result')}"
            )
            assert s.has_solution()
        finally:
            solver_hz_mod.strict_replay_for_act = orig_strict
            if old_we is None: sys.modules.pop("WitnessExtract", None)
            else: sys.modules["WitnessExtract"] = old_we
            if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
            else: sys.modules["GlobalTriangleLP"] = old_base
            if old_formal is None: os.environ.pop("ACT_FAL_RECEIPT_FORMAL", None)
            else: os.environ["ACT_FAL_RECEIPT_FORMAL"] = old_formal
            if old_dir is None: os.environ.pop("ACT_FAL_RECEIPT_DIR", None)
            else: os.environ["ACT_FAL_RECEIPT_DIR"] = old_dir


def test_non_formal_mode_keeps_sat_regardless_of_receipt():
    """Without ACT_FAL_RECEIPT_FORMAL set, SAT is emitted regardless of
    receipt outcome (receipt is best-effort logging). formal_result
    must NOT be set, so reportable_verdict_for_cli falls back to
    normalize {SAT→FALSIFIED}."""
    import os, sys, types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier, reportable_verdict_for_cli

    old_we = sys.modules.get("WitnessExtract")
    old_base = sys.modules.get("GlobalTriangleLP")
    old_formal = os.environ.get("ACT_FAL_RECEIPT_FORMAL")
    os.environ.pop("ACT_FAL_RECEIPT_FORMAL", None)

    we = types.ModuleType("WitnessExtract")
    we.verify_with_falsification = lambda *_a, **kw: (
        "falsified", np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]), 0.05,
    )
    sys.modules["WitnessExtract"] = we
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.extract_layers = lambda _: (None, [(np.zeros((3,128)), np.zeros(128))], (np.zeros((128,2)), np.zeros(2)))
    base.verify = lambda *_, **__: ("verified", 0.01)
    sys.modules["GlobalTriangleLP"] = base
    orig_strict = solver_hz_mod.strict_replay_for_act
    solver_hz_mod.strict_replay_for_act = lambda **kw: True

    fake_net = types.SimpleNamespace(onnx_path="/tmp/fake.onnx")
    fake_assert = types.SimpleNamespace(params={
        "kind": "UNSAFE_LINEAR",
        "c": np.array([[1.0, 0.0]]), "d": np.array([0.0]),
    })
    try:
        s = HZVerifier(
            onnx_path="/tmp/net.onnx", vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="witness",
        )
        result = s._try_small_dense_lp(
            conv_count=0, net=fake_net, assert_layer=fake_assert,
        )
        assert result == SolveStatus.SAT
        # Non-formal mode: formal_result must NOT be set
        assert "formal_result" not in s._stats
        # CLI reportable falls back to normalize → FALSIFIED
        assert reportable_verdict_for_cli(s, "SAT") == "FALSIFIED"
    finally:
        solver_hz_mod.strict_replay_for_act = orig_strict
        if old_we is None: sys.modules.pop("WitnessExtract", None)
        else: sys.modules["WitnessExtract"] = old_we
        if old_base is None: sys.modules.pop("GlobalTriangleLP", None)
        else: sys.modules["GlobalTriangleLP"] = old_base
        if old_formal is not None: os.environ["ACT_FAL_RECEIPT_FORMAL"] = old_formal


def test_round9_2_input_box_gate_rejects_out_of_box_witness():
    """ROUND 9.2 (advisor 2026-05-25 P1 audit finding):

    sat_relu small-dense witness path produced 31/49 receipts where
    x* was outside the declared input box [0,1]^100 (2 values < 0,
    5 values > 1). The witness then satisfied the unsafe predicate on
    the unconstrained NN output, but does NOT prove the spec is
    falsifiable. Pre-R9.2 _eval_unsafe_strict didn't check input
    domain, so these were emitted as SAT.

    R9.2 fix: ``strict_replay_for_act`` now gates on
    ``_x_star_in_input_box(net, x)`` BEFORE evaluating the unsafe
    predicate. Out-of-box witnesses return False (verdict downgrades
    to UNKNOWN per the small-dense phantom_rejected path).
    """
    import sys, types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_hz import _x_star_in_input_box

    # Synthesize a minimal net with one INPUT_SPEC carrying lb/ub.
    fake_input_spec = types.SimpleNamespace(
        kind="input_spec",  # actual LayerKind value
        params={"lb": np.array([0.0, 0.0]), "ub": np.array([1.0, 1.0])},
    )
    fake_net = types.SimpleNamespace(layers=[fake_input_spec])
    # Patch LayerKind to match the synthesized kind string.
    from act.back_end import layer_schema as ls
    try:
        actual_input_spec_kind = ls.LayerKind.INPUT_SPEC.value
    except Exception:
        actual_input_spec_kind = "input_spec"
    fake_input_spec.kind = actual_input_spec_kind

    # in-box x  (R9.3: returns (True, "ok"))
    holds, reason = _x_star_in_input_box(fake_net, np.array([0.5, 0.5]))
    assert holds is True and reason == "ok"
    # boundary x
    holds, reason = _x_star_in_input_box(fake_net, np.array([0.0, 1.0]))
    assert holds is True and reason == "ok"
    # out-of-box x (one dim too high)
    holds, reason = _x_star_in_input_box(fake_net, np.array([1.5, 0.5]))
    assert holds is False and reason == "out_of_box"
    # out-of-box x (one dim too low)
    holds, reason = _x_star_in_input_box(fake_net, np.array([0.5, -0.1]))
    assert holds is False and reason == "out_of_box"


def test_round9_3_fail_closed_missing_input_spec():
    """ROUND 9.3 (advisor 2026-05-25 P0): R9.2 returned True for nets
    with no InputSpec layer — that was FAIL-OPEN. R9.3 closes it:
    no spec → no positive validation → reject."""
    import types
    import numpy as np
    from act.back_end.solver.solver_hz import _x_star_in_input_box
    bare_net = types.SimpleNamespace(layers=[])
    holds, reason = _x_star_in_input_box(bare_net, np.array([99.0, 99.0]))
    assert holds is False
    assert reason == "no_input_spec"


def test_round9_3_fail_closed_shape_mismatch():
    """R9.3: if x and the recorded lb/ub differ in shape, we cannot
    do an element-wise box check → reject (was fail-open in R9.2)."""
    import types
    import numpy as np
    import act.back_end.solver.solver_hz as solver_hz_mod
    from act.back_end.solver.solver_hz import _x_star_in_input_box
    from act.back_end import layer_schema as ls
    try:
        kind = ls.LayerKind.INPUT_SPEC.value
    except Exception:
        kind = "input_spec"
    spec = types.SimpleNamespace(
        kind=kind,
        params={"lb": np.zeros(4), "ub": np.ones(4)},
    )
    net = types.SimpleNamespace(layers=[spec])
    holds, reason = _x_star_in_input_box(net, np.array([0.5, 0.5]))
    assert holds is False
    assert reason == "shape_mismatch"


def test_round9_3_fail_closed_missing_bounds_or_nan():
    """R9.3: if the InputSpec is missing lb/ub, or the bounds contain
    NaN, the witness cannot be validated → reject (fail-closed)."""
    import types
    import numpy as np
    from act.back_end.solver.solver_hz import _x_star_in_input_box
    from act.back_end import layer_schema as ls
    try:
        kind = ls.LayerKind.INPUT_SPEC.value
    except Exception:
        kind = "input_spec"
    # missing lb
    spec1 = types.SimpleNamespace(kind=kind, params={"ub": np.ones(2)})
    net1 = types.SimpleNamespace(layers=[spec1])
    holds, reason = _x_star_in_input_box(net1, np.array([0.5, 0.5]))
    assert holds is False
    assert reason == "missing_bounds"
    # NaN in bounds
    spec2 = types.SimpleNamespace(
        kind=kind,
        params={"lb": np.array([0.0, float("nan")]), "ub": np.ones(2)},
    )
    net2 = types.SimpleNamespace(layers=[spec2])
    holds, reason = _x_star_in_input_box(net2, np.array([0.5, 0.5]))
    assert holds is False
    assert reason == "nan_bounds"


def test_round9_3_input_box_gate_accepts_cuda_bounds():
    """A valid GPU witness must not fail input-box replay at NumPy conversion."""
    import types
    import numpy as np
    import torch
    from act.back_end.solver.solver_hz import _x_star_in_input_box
    from act.back_end.layer_schema import LayerKind

    if not torch.cuda.is_available():
        return

    spec = types.SimpleNamespace(
        kind=LayerKind.INPUT_SPEC.value,
        params={
            "lb": torch.tensor([0.0, 0.0], device="cuda"),
            "ub": torch.tensor([1.0, 1.0], device="cuda"),
        },
    )
    holds, reason = _x_star_in_input_box(
        types.SimpleNamespace(layers=[spec]),
        np.array([0.25, 0.75]),
    )
    assert holds is True
    assert reason == "ok"


def test_data_loader_headered_csv_first_iid_is_zero():
    """ROUND 5 (advisor 2026-05-24): the Round-3 header heuristic
    substring-matched 'onnx' AND 'vnnlib' against column 0/1, which
    incorrectly classified a literal header like 'onnx,vnnlib,timeout'
    as a data row. The synthetic 'data' row had non-existent files
    and got silently dropped, shifting all subsequent iids by +1.

    Verify: a CSV with header 'onnx,vnnlib,timeout' yields first
    DATA row at official_instance_id=0 (not 1).
    """
    import tempfile
    from pathlib import Path
    from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        cat = td_p / "hdrbench"
        (cat / "onnx").mkdir(parents=True); (cat / "vnnlib").mkdir()
        for n in ("a.onnx", "b.onnx"):
            (cat / "onnx" / n).write_bytes(b"x")
        for n in ("a.vnnlib", "b.vnnlib"):
            (cat / "vnnlib" / n).write_text("(declare-const X_0 Real)")
        (cat / "instances.csv").write_text(
            "onnx,vnnlib,timeout\n"
            "onnx/a.onnx,vnnlib/a.vnnlib,100\n"
            "onnx/b.onnx,vnnlib/b.vnnlib,100\n"
        )
        pairs = sorted(
            [p for p in list_downloaded_pairs(root_dir=str(td_p))
             if p["category"] == "hdrbench"],
            key=lambda p: p["official_instance_id"],
        )
        assert len(pairs) == 2, (
            f"headered CSV with 2 data rows must yield 2 pairs; got {len(pairs)}"
        )
        assert pairs[0]["official_instance_id"] == 0, (
            f"first DATA row must have iid=0 (Round-3 bug gave iid=1); "
            f"got {pairs[0]['official_instance_id']}"
        )
        assert pairs[0]["onnx_model"] == "onnx/a.onnx"
        assert pairs[1]["official_instance_id"] == 1


def test_data_loader_nested_paths_preserved():
    """ROUND 5: safenlp_2024 has nested specs like
    ``onnx/medical/perturbations_0.onnx``. The pre-fix loader used
    ``Path(...).name`` which dropped the 'medical/' segment, producing
    a non-existent ``onnx/perturbations_0.onnx`` path and silently
    skipping the instance — safenlp enumerated as 0.

    Verify: nested subdirs survive resolution and the file path exists.
    """
    import tempfile
    from pathlib import Path
    from act.front_end.vnnlib_loader.data_model_loader import (
        list_downloaded_pairs, load_vnnlib_pair, _resolve_relative_path,
    )

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        cat = td_p / "nestbench"
        (cat / "onnx" / "medical").mkdir(parents=True)
        (cat / "vnnlib" / "medical").mkdir(parents=True)
        (cat / "onnx" / "medical" / "p0.onnx").write_bytes(b"x")
        (cat / "vnnlib" / "medical" / "h0.vnnlib").write_text(
            "(declare-const X_0 Real)"
        )
        (cat / "instances.csv").write_text(
            "onnx/medical/p0.onnx,vnnlib/medical/h0.vnnlib,100\n"
        )
        pairs = [p for p in list_downloaded_pairs(root_dir=str(td_p))
                 if p["category"] == "nestbench"]
        assert len(pairs) == 1, (
            f"nested-path instance must enumerate; got {len(pairs)}. "
            "Pre-fix Path(...).name dropped subdirs → file missing → skipped."
        )
        p = pairs[0]
        assert p["onnx_model"] == "onnx/medical/p0.onnx"
        assert "medical/p0.onnx" in p["paths"]["onnx"], (
            f"resolved path must preserve nested subdir; got {p['paths']['onnx']}"
        )
        assert Path(p["paths"]["onnx"]).exists()
        # _resolve_relative_path direct unit test
        resolved = _resolve_relative_path(
            "onnx/medical/p0.onnx", "onnx", cat,
        )
        assert resolved == cat / "onnx" / "medical" / "p0.onnx"


def test_data_loader_canonical_root_cli_propagation():
    """ROUND 5 (most important): ACT_VNNLIB_ROOT must propagate from
    enumeration to load_vnnlib_pair so the CLI doesn't silently fall
    back to a different root when loading. Pre-fix: sat_relu enumerable
    via canonical root, but load FAILED with FileNotFoundError because
    the loader defaulted to ACT data root.
    """
    from pathlib import Path
    from act.front_end.vnnlib_loader.data_model_loader import (
        list_downloaded_pairs, load_vnnlib_pair,
    )
    canonical = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
    if not canonical.is_dir():
        import unittest as _u
        raise _u.SkipTest("canonical benchmark root not present")
    pairs = [p for p in list_downloaded_pairs(root_dir=str(canonical))
             if p["category"] == "sat_relu"]
    if not pairs:
        import unittest as _u
        raise _u.SkipTest("sat_relu not in canonical root")
    p = pairs[0]
    # Critical: pass SAME root_dir to loader
    result = load_vnnlib_pair(
        category="sat_relu",
        onnx_model=p["onnx_model"],
        vnnlib_spec=p["vnnlib_spec"],
        auto_download=False,
        root_dir=str(canonical),
    )
    assert "model" in result
    assert "labeled_tensor" in result
    # Same call WITHOUT root_dir would fail unless sat_relu is also in
    # ACT default root (it isn't on this machine — confirmed by the
    # advisor's empirical reproduction). The fix is: CLI must pass
    # root_dir; loader signature already accepts it.


def test_data_loader_real_acasxu_prop6_iid_still_181():
    """REGRESSION: Round 5 header refactor must NOT break Round 3's
    fix. acasxu local CSV is headerless; total still 186, prop_6
    still iid=181."""
    from pathlib import Path
    acasxu_dir = Path("/data1/Kane/ACT/data/vnnlib/acasxu_2023")
    if not acasxu_dir.exists():
        import unittest as _u
        raise _u.SkipTest("ACAS xu local data not present")
    from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs
    pairs = [p for p in list_downloaded_pairs() if p["category"] == "acasxu_2023"]
    assert len(pairs) == 186, (
        f"acasxu total must remain 186 after Round 5; got {len(pairs)}"
    )
    prop6_1_1 = [
        p for p in pairs
        if "prop_6" in p["vnnlib_spec"] and "1_1" in p["onnx_model"]
    ]
    assert prop6_1_1 and prop6_1_1[0]["official_instance_id"] == 181


def test_download_path_header_detection_matches_enumeration():
    """ROUND 7 (advisor 2026-05-24): the auto-download path at
    ``data_model_loader.download_vnnlib_category`` had its own
    ``next(reader, None)  # Skip header`` that R5 did NOT update,
    creating an internal inconsistency: the SAME headerless CSV gave
    different counts from the downloader (N-1) vs ``list_downloaded_pairs``
    (N).

    This test exercises the already-downloaded short-circuit (which
    runs the count) and confirms it matches the enumeration count for
    both headerless and headered CSVs.
    """
    import tempfile
    from pathlib import Path
    from act.front_end.vnnlib_loader.data_model_loader import (
        download_vnnlib_category, list_downloaded_pairs,
    )

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)

        # CASE A: headerless CSV (ACAS xu shape)
        cat_a = td_p / "headerless_bench"
        (cat_a / "onnx").mkdir(parents=True)
        (cat_a / "vnnlib").mkdir()
        for n in ("a.onnx", "b.onnx", "c.onnx"):
            (cat_a / "onnx" / n).write_bytes(b"x")
        for n in ("a.vnnlib", "b.vnnlib", "c.vnnlib"):
            (cat_a / "vnnlib" / n).write_text("(declare-const X_0 Real)")
        (cat_a / "instances.csv").write_text(
            "onnx/a.onnx,vnnlib/a.vnnlib,100\n"
            "onnx/b.onnx,vnnlib/b.vnnlib,100\n"
            "onnx/c.onnx,vnnlib/c.vnnlib,100\n"
        )
        result_a = download_vnnlib_category("headerless_bench", root_dir=str(td_p))
        enum_a = [p for p in list_downloaded_pairs(root_dir=str(td_p))
                  if p["category"] == "headerless_bench"]
        assert result_a["num_instances"] == 3, (
            f"headerless CSV: download count must equal 3 data rows; "
            f"got {result_a['num_instances']} (pre-R7 was 2 via skip-header)"
        )
        assert len(enum_a) == 3
        assert result_a["num_instances"] == len(enum_a), (
            "download count and enumeration count must agree on the SAME csv"
        )

        # CASE B: headered CSV
        cat_b = td_p / "headered_bench"
        (cat_b / "onnx").mkdir(parents=True)
        (cat_b / "vnnlib").mkdir()
        for n in ("x.onnx", "y.onnx"):
            (cat_b / "onnx" / n).write_bytes(b"x")
        for n in ("x.vnnlib", "y.vnnlib"):
            (cat_b / "vnnlib" / n).write_text("(declare-const X_0 Real)")
        (cat_b / "instances.csv").write_text(
            "onnx,vnnlib,timeout\n"
            "onnx/x.onnx,vnnlib/x.vnnlib,100\n"
            "onnx/y.onnx,vnnlib/y.vnnlib,100\n"
        )
        result_b = download_vnnlib_category("headered_bench", root_dir=str(td_p))
        enum_b = [p for p in list_downloaded_pairs(root_dir=str(td_p))
                  if p["category"] == "headered_bench"]
        assert result_b["num_instances"] == 2, (
            f"headered CSV: download count must equal 2 data rows; got {result_b['num_instances']}"
        )
        assert len(enum_b) == 2
        assert result_b["num_instances"] == len(enum_b)


def test_data_loader_headerless_csv_keeps_first_row():
    """ROUND 3 FIX: ACAS xu local instances.csv has no header. The
    pre-fix loader unconditionally next(reader) consumed the first
    DATA row, silently losing one instance and shifting all subsequent
    official_instance_ids by -1. Verify the header sniff keeps row 0."""
    import csv as _csv, tempfile, sys
    from pathlib import Path
    import shutil
    # Patch the function to operate on a fake category root
    from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs
    import act.front_end.vnnlib_loader.data_model_loader as dml

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        cat = td_p / "fakebench"
        (cat / "onnx").mkdir(parents=True)
        (cat / "vnnlib").mkdir()
        # Two onnx + two vnnlib files (so both rows are file-exists-valid)
        for name in ("a.onnx", "b.onnx"):
            (cat / "onnx" / name).write_bytes(b"x")
        for name in ("a.vnnlib", "b.vnnlib"):
            (cat / "vnnlib" / name).write_text("(declare-const X_0 Real)")
        # Headerless CSV
        (cat / "instances.csv").write_text(
            "onnx/a.onnx,vnnlib/a.vnnlib,100\n"
            "onnx/b.onnx,vnnlib/b.vnnlib,100\n"
        )
        # Pass root_dir directly (function supports it as kwarg)
        from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs
        pairs = [p for p in list_downloaded_pairs(root_dir=str(td_p))
                 if p["category"] == "fakebench"]
        # Both rows must survive
        assert len(pairs) == 2, (
            f"headerless CSV must keep both rows; got {len(pairs)}. "
            f"Pre-fix this returned 1 (first row dropped)."
        )
        # official_instance_id must be 0-indexed in CSV order
        pairs_sorted = sorted(pairs, key=lambda p: p["official_instance_id"])
        assert pairs_sorted[0]["official_instance_id"] == 0
        assert pairs_sorted[0]["onnx_model"] == "onnx/a.onnx"
        assert pairs_sorted[1]["official_instance_id"] == 1


def test_data_loader_real_acasxu_prop6_iid_is_181():
    """Smoke: with the headerless ACAS local CSV, prop_6 1_1 must
    receive official_instance_id=181 (matches the VNN-COMP manifest),
    not 180 (the pre-fix off-by-one)."""
    from pathlib import Path
    acasxu_dir = Path("/data1/Kane/ACT/data/vnnlib/acasxu_2023")
    if not acasxu_dir.exists():
        import unittest as _u
        raise _u.SkipTest("ACAS xu local data not present")
    from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs
    pairs = [p for p in list_downloaded_pairs() if p["category"] == "acasxu_2023"]
    prop6_1_1 = [
        p for p in pairs
        if "prop_6" in p["vnnlib_spec"] and "1_1" in p["onnx_model"]
    ]
    assert len(prop6_1_1) == 1
    assert prop6_1_1[0]["official_instance_id"] == 181, (
        f"prop_6 1_1 must have official_iid=181; got "
        f"{prop6_1_1[0]['official_instance_id']} (off-by-one regression)"
    )
    # Also: total count must be 186 (the off-by-one bug returned 185)
    assert len(pairs) == 186, (
        f"acasxu instance count must be 186; got {len(pairs)} "
        f"(pre-fix loader dropped the first row as fake header)"
    )


def test_small_dense_lp_portfolio_only_promotes_verified():
    """Optional small-dense LP can certify, but cannot turn unknown into SAT.

    Its detector must also prevent use on convolutional networks.  Fake
    modules exercise routing without coupling the regression test to the
    externally audited research backend.
    """
    import sys
    import types
    from act.back_end.solver.solver_base import SolveStatus
    from act.back_end.solver.solver_hz import HZVerifier

    old_base = sys.modules.get("GlobalTriangleLP")
    old_spec = sys.modules.get("SpecAwareLP")
    base = types.ModuleType("GlobalTriangleLP")
    base.is_small_dense = lambda _: True
    base.verify = lambda *_, **__: ("verified", 0.01)
    spec = types.ModuleType("SpecAwareLP")
    spec.verify = lambda *_, **__: ("unknown", 0.02)
    sys.modules["GlobalTriangleLP"] = base
    sys.modules["SpecAwareLP"] = spec
    try:
        certified = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="base",
        )
        assert certified._try_small_dense_lp(conv_count=0) == SolveStatus.UNSAT
        assert certified._stats["small_dense_lp_verdict"] == "verified"

        unresolved = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="specaware",
        )
        assert unresolved._try_small_dense_lp(conv_count=0) == SolveStatus.UNKNOWN
        assert unresolved._status == SolveStatus.UNKNOWN
        assert unresolved._stats["small_dense_lp_verdict"] == "unknown"

        fallback = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="specaware",
            small_dense_lp_fallback_on_unknown=True,
        )
        assert fallback._try_small_dense_lp(conv_count=0) is None

        conv_guard = HZVerifier(
            onnx_path="/tmp/net.onnx",
            vnnlib_path="/tmp/spec.vnnlib",
            small_dense_lp="base",
        )
        assert conv_guard._try_small_dense_lp(conv_count=1) is None
        assert "small_dense_lp_dispatch" not in conv_guard._stats
    finally:
        if old_base is None:
            sys.modules.pop("GlobalTriangleLP", None)
        else:
            sys.modules["GlobalTriangleLP"] = old_base
        if old_spec is None:
            sys.modules.pop("SpecAwareLP", None)
        else:
            sys.modules["SpecAwareLP"] = old_spec


if __name__ == "__main__":
    test_qr_pivoted_cpu_does_not_overwrite_input()
    test_pee_substitution_matrix_correct_2x4()
    test_pee_merge_uses_diagonal_slack_not_shared_col()
    test_girard_reduce_uses_diagonal_slack_not_shared_col()
    test_reduce_rank_phase_preserves_parallel_inequality_tightness()
    test_constraint_aware_girard_retains_hull_carrying_factor()
    test_pee_skip_merge_when_diag_slack_would_expand()
    test_girard_skip_when_cap_unachievable()
    test_chull_preserves_ab0_when_nb0_zero()
    test_chull_routing_actually_executes_not_downgraded()
    test_chull_vs_eq_lagr_v8_lp_relax_equivalent_raw()
    test_tail_preserve_dim_skips_reduce_below_threshold()
    test_selective_chull_endpoints_and_soundness()
    test_sparse_selective_chull_carries_facets_without_densifying_gc()
    test_small_dense_lp_portfolio_only_promotes_verified()
    test_small_dense_witness_mode_routes_to_sat_with_witness()
    test_small_dense_strict_replay_blocks_phantom_falsified()
    test_small_dense_strict_replay_confirms_real_falsified()
    test_fal_receipt_writes_zero_and_small_tol_columns()
    test_fal_receipt_disabled_when_env_unset()
    test_fal_receipt_formal_mode_rejects_sentinel_instance_id()
    test_fal_receipt_formal_mode_rejects_collision()
    test_formal_mode_no_receipt_dir_keeps_sat_marks_error_receipt()
    test_formal_mode_receipt_written_emits_sat()
    test_formal_mode_receipt_collision_keeps_sat_marks_collision()
    test_non_formal_mode_keeps_sat_regardless_of_receipt()
    test_data_loader_headerless_csv_keeps_first_row()
    test_data_loader_real_acasxu_prop6_iid_is_181()
    test_data_loader_headered_csv_first_iid_is_zero()
    test_data_loader_nested_paths_preserved()
    test_data_loader_canonical_root_cli_propagation()
    test_data_loader_real_acasxu_prop6_iid_still_181()
    test_download_path_header_detection_matches_enumeration()
    test_round9_2_input_box_gate_rejects_out_of_box_witness()
    test_round9_3_fail_closed_missing_input_spec()
    test_round9_3_fail_closed_shape_mismatch()
    test_round9_3_fail_closed_missing_bounds_or_nan()
    test_round9_3_input_box_gate_accepts_cuda_bounds()
    print("OK: 38 reduction-soundness regression tests pass")
