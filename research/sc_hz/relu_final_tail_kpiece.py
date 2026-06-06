"""S2 Gate 3: tighter ReLU relaxation on the FINAL 1-2 ReLU layers.

Per advisor 2026-06-05 Phase E spec: recover precision lost to streaming-prune
tail compression by tightening the ReLU relaxation only on the network's last
1-2 ReLU layers. Selective application keeps memory cost bounded.

DeepZ triangle baseline:
  unstable neuron with bounds [l, u], l < 0 < u
  triangle bound: y >= 0, y >= z, y <= slope * (z - l) where slope = u/(u-l)
  In HZ form: 1 new generator column per unstable neuron with value mu_i =
  -l*u / (2*(u-l)) at row i, zero elsewhere
  This is the tightest 3-segment convex hull.

k=2 piecewise-tighter (the "lambda-slope refinement"):
  Same triangle bound but cap the upper-line at z = midpoint:
    z < l_mid:  y <= slope_lo * (z - l)
    z >= l_mid: y <= slope_hi * (z - l_mid) + lambda_mid
  This adds 0 new generators (linear refinement) but introduces a per-neuron
  per-piece constraint.

For HZ, since we have continuous LP only, the simplest tightening within
DeepZ-triangle is to APPLY THE TRIANGLE STRICTLY (not over-relaxed) via
reducing the upper-slope generator's magnitude when the neuron's bounds
permit. In our case this means recomputing `mu` per neuron with tighter
slopes after the streaming-prune tail has reduced `[l, u]` to a wider
[l_compressed, u_compressed].

This module provides:
  apply_relu_kpiece_final_layer(state, k=2, ...) — k=2 or k=3 piecewise
    refinement applied to the FINAL hidden layer only.
  apply_relu_anderson_forward_facets(state, k_group=2, ...) — Anderson-2020
    multi-neuron facets using forward pre-activation bounds (no backward).

Monotonicity gate (G9): all of these MUST satisfy:
  LP UB after cut <= LP UB before cut (triangle baseline)
  per-coord box range no widen
  nb (binary count) no increase
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from research.sc_hz.prune import PrunedState
from research.sc_hz.ops import bounds


def apply_relu_kpiece_final_layer(
    state: PrunedState, k: int = 2,
) -> Tuple[PrunedState, np.ndarray]:
    """k-piece DeepZ-tighter ReLU on the LAST hidden layer.

    Goal: tighten the upper-bound segment of the triangle by replacing the
    single-slope chord [l, u] with a k-piece concave envelope of relu(z)
    on [l, u].

    For k=2:
      Split [l, 0] and [0, u] into pieces. Since relu(z) = 0 on [l, 0] and
      relu(z) = z on [0, u], the EXACT convex hull on the FULL interval is:
        y >= 0, y >= z, y <= (z - l) * u / (u - l)  ← triangle
      This IS the tightest convex upper envelope. To go tighter than triangle,
      we either need to add a binary, OR we need to split [l, u] into
      sub-intervals where the slope is variable.

      The valid CONTINUOUS sharper move: replace the single chord with a
      piecewise-linear UPPER envelope at 2 break-points:
        - At z = l: y <= 0 (exact)
        - At z = 0: y <= 0 (exact, where relu(0) = 0)
        - At z = u: y <= u (exact)
      Two linear pieces: l <= z <= 0 → y <= 0 ; 0 <= z <= u → y <= z
      So the 2-piece envelope is EXACT (= relu(z) itself!).

      This is achievable in HZ ONLY if we have a way to encode the
      conditional "either z < 0 or z > 0". In continuous LP that requires
      a constraint like `(z >= 0) OR (y = 0)` which is non-convex / disjunctive.

      Workaround: we use a "softened" k=2 envelope:
        y >= 0, y >= z, y <= max(0, alpha * (z - l))
      with alpha < (u/(u-l)) = standard slope. This is a SLOPE TIGHTENING.
      It's tighter than the triangle ONLY if the relaxation slope can be
      reduced — but the triangle slope is ALREADY tight on the chord.

    Conclusion: a pure-continuous "k-piece ReLU" is NOT meaningfully
    tighter than triangle on a single neuron. The actual precision lever
    is MULTI-NEURON joint hulls (Anderson-style) where the polytope of
    (z, y) over a small group of neurons has more vertices than the product
    of per-neuron triangles.

    This function therefore is a NO-OP placeholder. It returns state
    unchanged. Use `apply_relu_anderson_forward_facets` for real tightening.
    """
    # Per-neuron triangle is already the tightest convex; no further single-
    # neuron continuous tightening is possible.
    return state, np.zeros(state.c.shape[0], dtype=bool)


def apply_relu_anderson_forward_facets_pair(
    state: PrunedState, n_pairs: int = 4,
    seed: int = 20260605,
) -> Tuple[PrunedState, dict]:
    """Add Anderson-2020 multi-neuron facets for top-`n_pairs` pre-act bound
    pairs using FORWARD pre-activation bounds (no backward).

    Mechanism (continuous, forward-only):
      1. Identify unstable neuron pairs (i, j) where both have l < 0 < u.
      2. For each pair, compute the EXACT convex hull of
         (z_i, z_j, relu(z_i), relu(z_j)) over the box [l_i, u_i] x [l_j, u_j].
      3. The hull has 4 facets in (z_i, z_j, y_i, y_j) — 2 are the per-neuron
         triangles (already in state), 2 are NEW joint facets:
            (a) y_i + y_j >= z_i + z_j  (jointly above-zero)
            (b) y_i + y_j <= mu_i + mu_j + slope_i * (z_i - l_i) + slope_j * (z_j - l_j)
                — the joint upper facet, parameter slope_i/j.
      4. Add facet (b) as a new linear constraint via 1 new generator pair.

    For our HZ representation, joint facet (b) is the actual precision lever.
    It is added by:
      - introducing 1 new constraint row in the affine constraint matrix
        Ac · xi_c + Ab · xi_b <= b (but we are continuous-only, so Ab is empty)
      - Or equivalently, by reducing the upper-bound chord on z_i + z_j

    The HZ we currently use has NO explicit Ac matrix (we only track
    box-domain xi_c ∈ [-1,1]^K). Adding a polyhedral constraint here would
    require extending PrunedState to carry Ac, b. Phase E v2 needs this
    extension.

    For Phase E first cut: rather than extending PrunedState's constraint
    set, we apply a HEURISTIC slope-tightening on the existing triangle
    generator: when neuron i has joint-facet implication with j, reduce
    the magnitude of the triangle slack generator at i by a calibrated
    factor based on the joint geometry. This is SOUND only if we can
    prove the reduction is safe.

    Returns (new_state, info) with info containing:
      - n_pair_candidates: total unstable pairs found
      - n_pair_applied: how many got facet cuts
      - mean_slope_reduction: average tightening factor
    """
    # Identify unstable pre-activation rows (this function expects state to
    # be pre-ReLU; in practice we apply it AFTER apply_relu_triangle which
    # has already added slack generators, but we can identify which slack
    # cols belong to which unstable neuron via metadata).

    # For Phase E first-cut: NO-OP. Document the design and return state
    # unchanged. The real implementation requires PrunedState extension to
    # carry constraint matrix Ac.
    info = {
        "method": "anderson_forward_facets_pair_NOP",
        "reason": "PrunedState does not yet carry constraint matrix Ac/b; "
                  "real Anderson facets require this extension. Phase E v3.",
        "n_pair_candidates": 0,
        "n_pair_applied": 0,
    }
    return state, info


def apply_relu_pre_activation_intersect(
    state: PrunedState, K_per_layer: int = 100000,
) -> Tuple[PrunedState, dict]:
    """Tighten state by intersecting per-row [l_pre, u_pre] obtained from
    the forward HZ with the existing G_kept + tail bounds.

    This is the "intersect_box" move that ACT's production code uses but
    we have NOT applied in our forward HZ Phase A path. It is:
      1. Compute per-coord lb_i, ub_i from current PrunedState.
      2. For each coord, if abs(lb_i) << abs(state.c[i] - r_i) OR
         abs(ub_i) << abs(state.c[i] + r_i), the box is loose and we can
         tighten by clamping the state's effective range.
      3. The tightening is a new constraint row: state.c[i] + Σ G[i, k] xi_k
         + tail[i] xi_tail_i ∈ [lb_i, ub_i].
      4. Since this is a linear constraint, it goes in Ac matrix.

    Same Ac-extension dependency as Anderson — Phase E v3 work.

    For Phase E v2 first cut: return state unchanged + diagnostic info.
    """
    info = {
        "method": "intersect_box_NOP_diagnostic",
        "reason": "Requires PrunedState extension for Ac constraint matrix",
    }
    return state, info


def diagnose_relu_tightening_potential(
    state: PrunedState, n_top: int = 16,
) -> dict:
    """Diagnostic: how much room is there for ReLU tightening on this state?

    For each unstable coord:
      - tail dominance ratio = tail[i] / (|G_kept[i,:]|.sum() + tail[i])
      - if dominance > 0.5, that coord's value is mostly determined by tail
         (independent intervals), and a multi-neuron joint cut would help.

    Returns a summary dict with worst-offender coords (highest tail dominance).
    """
    G = state.G_kept
    tail = state.tail_radius
    if tail is None or G.shape[1] == 0:
        return {"n_coords": int(state.c.shape[0]), "tail_dominance_max": 0.0,
                  "n_high_dominance": 0}
    abs_G_row_sum = np.abs(G).sum(axis=1)
    denom = abs_G_row_sum + tail
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    tail_dom = tail / denom_safe
    # Top n_top coords by tail dominance
    top_idx = np.argsort(-tail_dom)[:n_top]
    return {
        "n_coords": int(state.c.shape[0]),
        "tail_dominance_max": float(tail_dom.max()),
        "tail_dominance_median": float(np.median(tail_dom)),
        "n_coords_dominance_gt_0p5": int((tail_dom > 0.5).sum()),
        "n_coords_dominance_gt_0p9": int((tail_dom > 0.9).sum()),
        "top_coords_by_dominance": top_idx.tolist(),
        "top_tail_values": tail[top_idx].tolist(),
        "top_G_l1_values": abs_G_row_sum[top_idx].tolist(),
    }
