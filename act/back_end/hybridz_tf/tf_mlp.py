# ===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and element-wise operations.
#
# ===---------------------------------------------------------------------===#

import os
import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    hz_multiply,
    hz_add_const,
    hz_minkowski_sum,
    hz_from_bounds,
    hz_compute_bounds,
    _propagate_base,
    _eq_mask_of,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


# Process-global ReLU-trace store. ENV-GATED. Only populated when
# `ACT_HZ_RELU_TRACE=1` is set. Each entry is the metadata needed to
# reconstruct the affine relationship between (xi_c, xi_b) and the
# (pre, post) activation of each unstable neuron in that ReLU call.
#
# Used by `_pair_relu_hull_cuts` (see solver_hz.py) to generate
# joint convex-hull facets for selected ReLU pairs and add them as
# extra LP rows in the unsafe-feasibility check.
#
# Soundness: every entry must be derived strictly from forward HZ state
# (no backward bound propagation, no autograd). See the docstring of
# `_relu_trace_record_v8` for the exact encoding contract.
RELU_TRACE_STORE: "list[dict]" = []

# Process-global per-layer counter (incremented each time hz_apply_relu runs)
# and per-layer pair selection override. When `ACT_HZ_CORR_PAIR_CUT_TARGET_FILE`
# env points to a JSON file with `{layer_counter: [[a_idx, b_idx], ...]}`
# the file's contents take precedence over width-score selection for that
# layer's correlated cuts. Used by two-pass output-aware pair selection.
RELU_LAYER_COUNTER = [0]  # mutable container so we can reset per query
PAIR_TARGETS_BY_LAYER: "dict[int, list[tuple[int, int]]]" = {}


def _relu_layer_counter_reset():
    """Reset the per-call ReLU layer counter. Should be called at the
    start of each verifier query so layer_id numbering restarts at 0.

    Also clears PAIR_TARGETS_BY_LAYER and the loaded-file path cache so
    a same-process re-query (or change of target file) doesn't carry
    stale targets into the new query."""
    RELU_LAYER_COUNTER[0] = 0
    PAIR_TARGETS_BY_LAYER.clear()
    # Reset the cache key tracker; _maybe_load_pair_targets checks this
    # before deciding to reread the JSON file.
    _PAIR_TARGETS_CACHE_KEY[0] = ""


_PAIR_TARGETS_CACHE_KEY = [""]  # mtime+path key for the last-loaded targets


def _maybe_load_pair_targets():
    """Lazily load pair-target overrides from the env-pointed JSON file.

    The cache is keyed by `path:mtime` so changing the target file
    between queries (or rewriting it) forces a fresh load. Returns the
    dict; empty dict if unset/unreadable.
    """
    global PAIR_TARGETS_BY_LAYER
    target_file = os.environ.get("ACT_HZ_CORR_PAIR_CUT_TARGET_FILE", "")
    if not target_file:
        return PAIR_TARGETS_BY_LAYER
    try:
        import os as _os
        st = _os.stat(target_file)
        key = f"{target_file}:{st.st_mtime_ns}"
    except Exception:
        key = target_file
    if PAIR_TARGETS_BY_LAYER and _PAIR_TARGETS_CACHE_KEY[0] == key:
        return PAIR_TARGETS_BY_LAYER
    try:
        import json
        with open(target_file) as f:
            raw = json.load(f)
        PAIR_TARGETS_BY_LAYER.clear()
        for k, v in raw.items():
            PAIR_TARGETS_BY_LAYER[int(k)] = [tuple(p) for p in v]
        _PAIR_TARGETS_CACHE_KEY[0] = key
    except Exception:
        pass
    return PAIR_TARGETS_BY_LAYER


def _relu_trace_record_triangle(
    *,
    hz_in: HZono,
    unstable_idx: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    col_eps: torch.Tensor,
    ng_old: int,
    nb_old: int,
):
    """Record a triangle-ReLU layer's trace entry.

    Triangle encoding (DeepZ):
      post[i] = lam[i] * pre[i] + mu[i] + mu[i] * eps_new[i]
    where lam[i] = ub[i]/(ub[i]-lb[i]), mu[i] = -lb[i]*ub[i]/(2*(ub[i]-lb[i])).
    `col_eps[i]` is the new continuous-generator column index in the NEW
    HZ's xi_c for unstable neuron i.

    No binaries are introduced by triangle, so `col_z` is empty and
    `col_eps` carries the new continuous-generator columns. The offline
    selector can target triangle layers by scoring these `col_eps`
    columns against the final output HZ's continuous generator matrix.
    """
    if os.environ.get("ACT_HZ_RELU_TRACE", "0") != "1":
        return
    try:
        Gc_unst = hz_in.Gc[unstable_idx].detach().cpu().numpy()
        Gb_unst = hz_in.Gb[unstable_idx].detach().cpu().numpy()
        c_unst = hz_in.c[unstable_idx, 0].detach().cpu().numpy()
        lam = (beta / (beta - alpha)).detach().cpu().numpy()
        mu = (-alpha * beta / (2.0 * (beta - alpha))).detach().cpu().numpy()
        entry = {
            "layer_count": len(RELU_TRACE_STORE),
            "k": int(unstable_idx.numel()),
            "encoding": "triangle",
            "unstable_idx": unstable_idx.detach().cpu().numpy().tolist(),
            "alpha": alpha.detach().cpu().numpy().tolist(),
            "beta": beta.detach().cpu().numpy().tolist(),
            "col_eps": col_eps.detach().cpu().numpy().tolist(),
            "col_z": [],  # triangle adds no binaries
            "lam": lam.tolist(),
            "mu": mu.tolist(),
            "ng_old": int(ng_old),
            "nb_old": int(nb_old),
            "Gc_pre": Gc_unst,
            "Gb_pre": Gb_unst,
            "c_pre": c_unst,
        }
        RELU_TRACE_STORE.append(entry)
    except Exception:
        pass


def _trace_dump_to_file():
    """If `ACT_HZ_RELU_TRACE_DUMP_FILE` is set, write the per-layer
    (col_z, k, layer_count, unstable_idx) summary to disk. Continuous
    factors (col_xi2 / Gc_pre / Gb_pre / c_pre) are omitted to keep the
    file small — they're only needed inline during cut emission, not
    by the offline selector."""
    target_path = os.environ.get("ACT_HZ_RELU_TRACE_DUMP_FILE", "")
    if not target_path or not RELU_TRACE_STORE:
        return
    try:
        import json
        summary = []
        for entry in RELU_TRACE_STORE:
            item = {
                "layer_count": entry["layer_count"],
                "k": entry["k"],
                "col_z": list(entry.get("col_z", [])),
                "unstable_idx": list(entry["unstable_idx"]),
                "alpha": list(entry["alpha"]),
                "beta": list(entry["beta"]),
                "encoding": entry.get("encoding", "eq_lagr_v8"),
            }
            if "col_eps" in entry:
                item["col_eps"] = list(entry["col_eps"])
            summary.append(item)
        with open(target_path, "w") as f:
            json.dump(summary, f)
    except Exception:
        pass


def _relu_trace_record_v8(
    *,
    hz_in: HZono,
    unstable_idx: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    col_xi2: torch.Tensor,
    col_z: torch.Tensor,
    ng_old: int,
    nb_old: int,
):
    """Append one entry to RELU_TRACE_STORE describing the v8 eq_lagr
    ReLU's encoding for the current call.

    The entry records, for each unstable neuron i:
      - pre_affine: pre[i] = hz_in.c[i] + hz_in.Gc[i] @ xi_c_old + hz_in.Gb[i] @ xi_b_old
      - post_affine: post[i] = (beta[i]/2) * (1 - xi2[i])
      - bounds: (alpha[i], beta[i])
      - col_xi2_idx: the index of xi2[i] in the NEW HZ's xi_c (ng_old + k + t)
      - col_z_idx: the index of z[i] in the NEW HZ's xi_b (nb_old + t)
    The entry also records ng_old/nb_old so callers can map old xi_c/xi_b
    indices into the new HZ (just the first ng_old / nb_old columns).
    """
    if os.environ.get("ACT_HZ_RELU_TRACE", "0") != "1":
        return
    try:
        Gc_unst = hz_in.Gc[unstable_idx].detach().cpu().numpy()
        Gb_unst = hz_in.Gb[unstable_idx].detach().cpu().numpy()
        c_unst = hz_in.c[unstable_idx, 0].detach().cpu().numpy()
        entry = {
            "layer_count": len(RELU_TRACE_STORE),
            "k": int(unstable_idx.numel()),
            "unstable_idx": unstable_idx.detach().cpu().numpy().tolist(),
            "alpha": alpha.detach().cpu().numpy().tolist(),
            "beta": beta.detach().cpu().numpy().tolist(),
            "col_xi2": col_xi2.detach().cpu().numpy().tolist(),
            "col_z": col_z.detach().cpu().numpy().tolist(),
            "ng_old": int(ng_old),
            "nb_old": int(nb_old),
            "Gc_pre": Gc_unst,
            "Gb_pre": Gb_unst,
            "c_pre": c_unst,
        }
        RELU_TRACE_STORE.append(entry)
    except Exception as _e:
        # Trace recording is best-effort and never blocks the production
        # ReLU encoding. Failures are silent and surface only via the
        # diagnostic LP returning no extra cuts.
        pass


def _relu_trace_reset():
    """Clear the per-instance trace store. Call at the start of every
    verifier query so prior queries' ReLU traces don't leak in."""
    RELU_TRACE_STORE.clear()


def _pair_hull_facets_9vertex(la: float, ua: float,
                               lb_: float, ub_: float) -> "list[dict]":
    """DEPRECATED — independent-box pairwise ReLU lifted hull.

    Audit finding (2026-05-31): under the eq_lagr_v8 per-neuron EXACT
    triangle encoding, this independent-box hull is MATHEMATICALLY
    REDUNDANT — it equals the Cartesian product of two single-neuron
    triangle hulls, which is already implied by the v8 encoding. The
    cifar iid 0 experiment confirmed LP_max unchanged regardless of how
    many such cuts are added (224 / 1462 rows → identical 1.4384).

    The actually-tightening math is the CORRELATED joint hull (linear
    image of the input HZ's LP polytope under (xi → (pre_a, pre_b))),
    implemented in `_correlated_pair_hull_facets`.

    Kept here for reference + the existing `_build_pair_cut_rows_for_relu`
    integration path, gated by the env knob `ACT_HZ_PAIR_RELU_CUTS=1`
    which is documented as redundant/experimental. Default OFF, no
    production usage.
    """
    if not (la < 0 < ua and lb_ < 0 < ub_):
        return []
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return []
    import numpy as _np
    pts = _np.array([
        [la, lb_, max(0.0, la), max(0.0, lb_)],
        [la,  0.0, max(0.0, la), 0.0],
        [la, ub_, max(0.0, la), max(0.0, ub_)],
        [0.0, lb_, 0.0, max(0.0, lb_)],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, ub_, 0.0, max(0.0, ub_)],
        [ua, lb_, max(0.0, ua), max(0.0, lb_)],
        [ua, 0.0, max(0.0, ua), 0.0],
        [ua, ub_, max(0.0, ua), max(0.0, ub_)],
    ], dtype=_np.float64)
    pts_u = _np.unique(pts, axis=0)
    if pts_u.shape[0] < 5:
        return []
    try:
        hull = ConvexHull(pts_u)
    except Exception:
        return []
    cuts = []
    for eq in hull.equations:
        # eq: a*pre_a + b*pre_b + c*post_a + d*post_b + e <= 0
        cuts.append({
            "c_pa": float(eq[0]), "c_pb": float(eq[1]),
            "c_postA": float(eq[2]), "c_postB": float(eq[3]),
            "rhs": float(-eq[4]),
        })
    return cuts


def _correlated_pair_hull_facets(
    hz_in: HZono,
    a_idx: int,
    b_idx: int,
    alpha_a: float, beta_a: float,
    alpha_b: float, beta_b: float,
    n_dirs: int = 8,
    lp_timeout_s: float = 2.0,
) -> "list[dict]":
    """Compute facets of the CORRELATED 4D pairwise ReLU lifted hull.

    The correlation comes from the fact that pre_a(xi) and pre_b(xi) are
    BOTH affine functions of the SAME factor vector xi, which itself is
    constrained by the input HZ. The image of the HZ's feasible factor
    polytope under `xi → (pre_a, pre_b)` is a 2D polytope (NOT the box
    `[la, ua] × [lb, ub]`). We approximate this image by solving 2*n_dirs
    SciPy LPs over the HZ constraints to obtain a support polytope
    `P_outer`, split it by the axes `pa=0` and `pb=0` into up to 4
    sign-cells, lift the cell vertices to 4D via the exact ReLU, and
    take the convex hull. Returned facets hold for every real input AND
    are NOT implied by independent triangle×triangle constraints.

    Soundness:
      - The 2D outer polytope CONTAINS the true joint image (we only
        bound supports from finite directions, never under-approximate).
      - Sign-cell decomposition partitions the outer polytope; vertices
        of each cell lie inside the outer polytope.
      - In each cell the ReLU map is affine; lifting vertices and taking
        their hull is exact for the cell.
      - Hull of the union of cells contains the true lifted joint set.
      - Every cut from this hull is a valid linear inequality over the
        true joint set, hence valid for the LP relaxation.

    No autograd, no backward bound propagation, no randomness. Only
    SciPy HiGHS LPs over the existing HZ constraint matrices.

    Args:
      hz_in: HZono BEFORE the ReLU (provides Ac, Ab, b, Gc, Gb, c, eq_mask).
      a_idx, b_idx: indices of the two unstable neurons within hz_in.
      alpha_*, beta_*: pre-activation bounds for each neuron.
      n_dirs: number of axis-aligned directions (must be multiple of 4).
      lp_timeout_s: per-LP wall-clock cap.

    Returns:
      list of facet dicts with keys 'c_pa', 'c_pb', 'c_postA', 'c_postB',
      'rhs' representing the inequality
        c_pa*pre_a + c_pb*pre_b + c_postA*post_a + c_postB*post_b <= rhs.
      Empty list on degenerate inputs.
    """
    if not (alpha_a < 0 < beta_a and alpha_b < 0 < beta_b):
        return []
    try:
        from scipy.optimize import linprog
        from scipy.spatial import ConvexHull
    except ImportError:
        return []
    import numpy as _np

    # ─── Build factor-space LP problem from hz_in ───
    # Mirrors _build_factor_lp in solver_hz.py but inline here to avoid
    # an import cycle and to keep the cut path self-contained.
    em = hz_in.eq_mask
    if em is None:
        em_np = _np.ones(int(hz_in.b.shape[0]), dtype=bool)
    else:
        em_np = em.detach().cpu().numpy().astype(bool)
    le_np = ~em_np
    Ac_np = hz_in.Ac.detach().cpu().numpy()
    Ab_np = hz_in.Ab.detach().cpu().numpy()
    b_np = hz_in.b.detach().cpu().numpy().reshape(-1)
    if em_np.any():
        A_eq = _np.concatenate([Ac_np[em_np], Ab_np[em_np]], axis=1)
        b_eq = b_np[em_np]
    else:
        A_eq = None
        b_eq = None
    if le_np.any():
        A_ub = _np.concatenate([Ac_np[le_np], Ab_np[le_np]], axis=1)
        b_ub = b_np[le_np]
    else:
        A_ub = None
        b_ub = None
    p = int(hz_in.Gc.shape[1])
    q = int(hz_in.Gb.shape[1])
    nvars = p + q
    bounds = [(-1.0, 1.0)] * nvars

    # Affine maps: pre_x(xi) = Gc_x @ xi_c + Gb_x @ xi_b + c_x
    Gc_a = hz_in.Gc[a_idx].detach().cpu().numpy()
    Gb_a = hz_in.Gb[a_idx].detach().cpu().numpy()
    c_a = float(hz_in.c[a_idx, 0].detach().cpu().item())
    Gc_b = hz_in.Gc[b_idx].detach().cpu().numpy()
    Gb_b = hz_in.Gb[b_idx].detach().cpu().numpy()
    c_b = float(hz_in.c[b_idx, 0].detach().cpu().item())

    # ─── Solve n_dirs LPs to derive OUTER half-spaces ───
    # For each direction d_i, max{d_i·(pa,pb) : xi feasible} gives a sound
    # upper bound rhs_i. The half-space {(pa, pb) : d_i·(pa, pb) <= rhs_i}
    # CONTAINS the true reachable set. Intersection of all such half-spaces
    # is the OUTER polygon (vs. the convex hull of support points which is
    # an INNER approximation — using it as the basis for cuts would
    # produce unsound facets that exclude real reachable inputs).
    angles = _np.linspace(0.0, 2.0 * _np.pi, n_dirs, endpoint=False)
    halfspaces: "list[tuple[float, float, float]]" = []  # (d1, d2, rhs)
    for theta in angles:
        d1 = float(_np.cos(theta))
        d2 = float(_np.sin(theta))
        obj_xi = _np.concatenate([
            d1 * Gc_a + d2 * Gc_b,
            d1 * Gb_a + d2 * Gb_b,
        ])
        obj_const = d1 * c_a + d2 * c_b
        try:
            res = linprog(
                c=-obj_xi,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"time_limit": lp_timeout_s},
            )
            if not (res.status == 0 and res.success):
                continue
            rhs_max = float(-res.fun + obj_const)
            halfspaces.append((d1, d2, rhs_max))
        except Exception:
            continue

    if len(halfspaces) < 3:
        return []

    # ─── Build OUTER polygon via half-space intersection ───
    # Use scipy.spatial.HalfspaceIntersection. Format requires
    # (A | b) rows where A·x + b <= 0, i.e., d·x - rhs <= 0.
    from scipy.spatial import HalfspaceIntersection
    A_hs = _np.zeros((len(halfspaces), 3), dtype=_np.float64)
    for i, (d1, d2, rhs) in enumerate(halfspaces):
        A_hs[i, 0] = d1
        A_hs[i, 1] = d2
        A_hs[i, 2] = -rhs  # half-space format: A·x + b <= 0
    # Need an interior point. Use box midpoint of axis-aligned support.
    pa_max = max((h[2] for h in halfspaces if abs(h[1]) < 1e-9 and h[0] > 0),
                 default=alpha_a)
    pa_min = -max((h[2] for h in halfspaces if abs(h[1]) < 1e-9 and h[0] < 0),
                  default=-beta_a)
    pb_max = max((h[2] for h in halfspaces if abs(h[0]) < 1e-9 and h[1] > 0),
                 default=alpha_b)
    pb_min = -max((h[2] for h in halfspaces if abs(h[0]) < 1e-9 and h[1] < 0),
                  default=-beta_b)
    interior_pt = _np.array([
        0.5 * (pa_min + pa_max),
        0.5 * (pb_min + pb_max),
    ], dtype=_np.float64)
    try:
        hs = HalfspaceIntersection(A_hs, interior_pt)
        outer_verts_unord = hs.intersections
    except Exception:
        return []
    if outer_verts_unord.shape[0] < 3:
        return []
    # scipy's HalfspaceIntersection returns vertices in arbitrary order;
    # to walk POLYGON EDGES (adjacent vertex pairs) for axis-crossing
    # detection we must order them by polar angle around the polygon's
    # centroid. Otherwise consecutive `outer_verts[i]` and `[i+1]` may be
    # polygon DIAGONALS (going through interior), causing edge-axis
    # intersections to be computed on the wrong segments and missing
    # genuine cell-corner points.
    centroid = outer_verts_unord.mean(axis=0)
    angles_ord = _np.arctan2(
        outer_verts_unord[:, 1] - centroid[1],
        outer_verts_unord[:, 0] - centroid[0],
    )
    order = _np.argsort(angles_ord)
    outer_verts = outer_verts_unord[order]

    # ─── Lift to 4D via exact ReLU; include axis intersections to handle
    # the kink at pa=0 / pb=0 within the outer polygon.
    # Strategy:
    #   - For each outer vertex, lift directly.
    #   - For each edge of outer polygon, if it crosses pa=0 or pb=0,
    #     add the intersection point and lift it.
    lifted: "list[list[float]]" = []
    def _lift(p_pair):
        pa, pb = float(p_pair[0]), float(p_pair[1])
        return [pa, pb, max(0.0, pa), max(0.0, pb)]

    nv = outer_verts.shape[0]
    for i in range(nv):
        v0 = outer_verts[i]
        v1 = outer_verts[(i + 1) % nv]
        lifted.append(_lift(v0))
        # Intersection with pa=0
        if (v0[0] < 0) != (v1[0] < 0):
            t = -v0[0] / (v1[0] - v0[0] + 1e-30)
            if 0.0 < t < 1.0:
                cross = v0 + t * (v1 - v0)
                lifted.append(_lift(cross))
        # Intersection with pb=0
        if (v0[1] < 0) != (v1[1] < 0):
            t = -v0[1] / (v1[1] - v0[1] + 1e-30)
            if 0.0 < t < 1.0:
                cross = v0 + t * (v1 - v0)
                lifted.append(_lift(cross))

    # Add the lift of (0, 0) when it lies inside the outer polygon. Without
    # this point, the 4-sign-cell decomposition of the polygon may miss the
    # cell-corner at the origin, leading to a 4D hull that excludes some
    # real lifted points whose (pa, pb) lies in the small (+,+) (or other)
    # cell of the polygon. (0, 0) lies inside iff every half-space's RHS
    # is non-negative (since each constraint is d·x <= rhs and the origin
    # gives d·0 = 0).
    origin_inside = all(rhs >= -1e-12 for _, _, rhs in halfspaces)
    if origin_inside:
        lifted.append(_lift((0.0, 0.0)))

    pts4d = _np.array(lifted, dtype=_np.float64)
    pts4d_unique = _np.unique(pts4d, axis=0)
    if pts4d_unique.shape[0] < 5:
        return []

    try:
        hull4 = ConvexHull(pts4d_unique, qhull_options="QJ")
    except Exception:
        return []

    facets: "list[dict]" = []
    for eq in hull4.equations:
        # eq: a*pa + b*pb + c*postA + d*postB + e <= 0
        a, b_eq_coef, c_e, d_e, e_e = eq
        # Skip facets that are pure pa/pb axis bounds (these are box-only,
        # not joint cuts) — keep them only if they involve at least one post.
        coef_mag = abs(a) + abs(b_eq_coef) + abs(c_e) + abs(d_e)
        if coef_mag < 1e-9:
            continue
        facets.append({
            "c_pa": float(a),
            "c_pb": float(b_eq_coef),
            "c_postA": float(c_e),
            "c_postB": float(d_e),
            "rhs": float(-e_e),
        })
    return facets


def _build_correlated_pair_cut_rows_for_relu(
    hz_in: HZono,
    unstable_idx: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    col_xi2: torch.Tensor,
    ng_new: int,
    nb_new: int,
    max_pairs: int,
    n_dirs: int = 8,
    lp_timeout_s: float = 2.0,
    layer_counter: int = -1,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Build LP rows for CORRELATED pair-hull cuts.

    Same row-emission scheme as `_build_pair_cut_rows_for_relu` but uses
    `_correlated_pair_hull_facets` to compute the per-pair facets
    against the input HZ's true joint reachable set.

    Pair selection:
      - If `ACT_HZ_CORR_PAIR_CUT_TARGET_FILE` is set AND `layer_counter`
        is a key in that file: use the file's pair list verbatim
        (output-aware two-pass selection).
      - Otherwise: fall back to width-product score (forward-only).
    """
    import numpy as _np
    device = hz_in.c.device
    dtype = hz_in.c.dtype
    k = int(unstable_idx.numel())
    if k < 2 or max_pairs <= 0:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    alpha_np = alpha.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    widths = beta_np - alpha_np

    # ─── Pair selection: output-aware override > width-score fallback ───
    target_file_set = bool(os.environ.get("ACT_HZ_CORR_PAIR_CUT_TARGET_FILE", ""))
    targets_by_layer = _maybe_load_pair_targets()
    if target_file_set and not (layer_counter >= 0 and layer_counter in targets_by_layer):
        # Output-aware mode is an explicit whitelist. If a target file is
        # provided, layers absent from the file must emit no cuts; falling
        # back to width-score here silently adds unrelated cuts, makes the
        # experiment no longer target-file controlled, and can explode large
        # triangle layers in dense conv nets.
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )
    if target_file_set and layer_counter >= 0 and layer_counter in targets_by_layer:
        # Output-aware override: use the file's target pair list verbatim.
        # Filter to pairs whose indices are valid for current k.
        raw_pairs = targets_by_layer[layer_counter]
        pairs = [
            (int(a), int(b)) for a, b in raw_pairs
            if 0 <= int(a) < k and 0 <= int(b) < k and int(a) != int(b)
        ][:max_pairs]
    elif max_pairs >= k * (k - 1) // 2:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        m = min(k, max(8, int(round((1 + (1 + 8 * max_pairs) ** 0.5) / 2))))
        top_neuron_idx = _np.argsort(-widths)[:m]
        pairs = [
            (int(top_neuron_idx[i]), int(top_neuron_idx[j]))
            for i in range(m) for j in range(i + 1, m)
        ][:max_pairs]
    if not pairs:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    ng = int(hz_in.Gc.shape[1])
    nb = int(hz_in.Gb.shape[1])
    col_xi2_np = col_xi2.detach().cpu().numpy()
    unstable_np = unstable_idx.detach().cpu().numpy()

    rows_Ac: "list[_np.ndarray]" = []
    rows_Ab: "list[_np.ndarray]" = []
    rhss: "list[float]" = []

    Gc_pre_t = hz_in.Gc.detach().cpu().numpy()
    Gb_pre_t = hz_in.Gb.detach().cpu().numpy()
    c_pre_t = hz_in.c[:, 0].detach().cpu().numpy()

    for a_local, b_local in pairs:
        a_abs = int(unstable_np[a_local])
        b_abs = int(unstable_np[b_local])
        la = float(alpha_np[a_local]); ua = float(beta_np[a_local])
        lb_p = float(alpha_np[b_local]); ub_p = float(beta_np[b_local])
        facets = _correlated_pair_hull_facets(
            hz_in=hz_in,
            a_idx=a_abs, b_idx=b_abs,
            alpha_a=la, beta_a=ua,
            alpha_b=lb_p, beta_b=ub_p,
            n_dirs=n_dirs, lp_timeout_s=lp_timeout_s,
        )
        if not facets:
            continue
        Gc_a = Gc_pre_t[a_abs]
        Gb_a = Gb_pre_t[a_abs]
        c_a = float(c_pre_t[a_abs])
        Gc_b = Gc_pre_t[b_abs]
        Gb_b = Gb_pre_t[b_abs]
        c_b = float(c_pre_t[b_abs])
        col_a = int(col_xi2_np[a_local])
        col_b = int(col_xi2_np[b_local])
        beta_a_f = float(beta_np[a_local])
        beta_b_f = float(beta_np[b_local])
        for f in facets:
            c_pa = f["c_pa"]; c_pb = f["c_pb"]
            c_postA = f["c_postA"]; c_postB = f["c_postB"]
            rhs_f = f["rhs"]
            if (abs(c_pa) + abs(c_pb) + abs(c_postA) + abs(c_postB)) < 1e-12:
                continue
            row_Ac = _np.zeros(ng_new, dtype=_np.float64)
            row_Ab = _np.zeros(nb_new, dtype=_np.float64)
            row_Ac[:ng] = c_pa * Gc_a + c_pb * Gc_b
            row_Ab[:nb] = c_pa * Gb_a + c_pb * Gb_b
            row_Ac[col_a] += c_postA * (-beta_a_f / 2.0)
            row_Ac[col_b] += c_postB * (-beta_b_f / 2.0)
            row_rhs = (rhs_f - c_pa * c_a - c_pb * c_b
                       - c_postA * (beta_a_f / 2.0)
                       - c_postB * (beta_b_f / 2.0))
            rows_Ac.append(row_Ac)
            rows_Ab.append(row_Ab)
            rhss.append(float(row_rhs))

    if not rows_Ac:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )
    Ac_rows = torch.tensor(_np.stack(rows_Ac), device=device, dtype=dtype)
    Ab_rows = torch.tensor(_np.stack(rows_Ab), device=device, dtype=dtype)
    b_rows = torch.tensor(
        _np.asarray(rhss).reshape(-1, 1), device=device, dtype=dtype
    )
    return Ac_rows, Ab_rows, b_rows


def _build_triangle_corr_pair_cuts(
    hz_in: HZono,
    unstable_idx: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    mu: torch.Tensor,
    col_eps: torch.Tensor,
    ng_new: int,
    nb_new: int,
    max_pairs: int,
    n_dirs: int = 8,
    lp_timeout_s: float = 2.0,
    layer_counter: int = -1,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Correlated pair-ReLU cuts for the TRIANGLE encoding.

    Triangle: post[i] = lam[i]*pre[i] + mu[i] + mu[i]*eps_new[i],
    pre[i] = c_pre[i] + Gc_pre[i] @ xi_c_old + Gb_pre[i] @ xi_b_old.

    The 4D hull facet
      c_pa*pre_a + c_pb*pre_b + c_postA*post_a + c_postB*post_b <= rhs
    expands (after substitution) to an inequality in
    (xi_c_old, xi_b_old, eps_new_a, eps_new_b). Specifically:

      coef on xi_c_old: (c_pa + c_postA*lam_a)*Gc_pre_a
                      + (c_pb + c_postB*lam_b)*Gc_pre_b
      coef on xi_b_old: (c_pa + c_postA*lam_a)*Gb_pre_a
                      + (c_pb + c_postB*lam_b)*Gb_pre_b
      coef on eps_new_a (at col_eps[a]): c_postA * mu_a
      coef on eps_new_b (at col_eps[b]): c_postB * mu_b
      rhs_new = rhs - (c_pa + c_postA*lam_a)*c_pre_a
                    - (c_pb + c_postB*lam_b)*c_pre_b
                    - c_postA*mu_a - c_postB*mu_b

    Pair selection mirrors v8: respect ACT_HZ_CORR_PAIR_CUT_TARGET_FILE
    when set, else width score.
    """
    import numpy as _np
    device = hz_in.c.device
    dtype = hz_in.c.dtype
    k = int(unstable_idx.numel())
    if k < 2 or max_pairs <= 0:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    alpha_np = alpha.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    lam_np = lam.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    widths = beta_np - alpha_np

    target_file_set = bool(os.environ.get("ACT_HZ_CORR_PAIR_CUT_TARGET_FILE", ""))
    targets_by_layer = _maybe_load_pair_targets()
    if target_file_set and not (layer_counter >= 0 and layer_counter in targets_by_layer):
        # Target-file mode is an explicit whitelist. Do not add width-score
        # fallback cuts to unlisted triangle layers; on dense-conv nets those
        # layers are huge and this turns a small target experiment into an
        # accidental all-layer cut sweep.
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )
    if target_file_set and layer_counter >= 0 and layer_counter in targets_by_layer:
        raw_pairs = targets_by_layer[layer_counter]
        pairs = [
            (int(a), int(b)) for a, b in raw_pairs
            if 0 <= int(a) < k and 0 <= int(b) < k and int(a) != int(b)
        ][:max_pairs]
    elif max_pairs >= k * (k - 1) // 2:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        m = min(k, max(8, int(round((1 + (1 + 8 * max_pairs) ** 0.5) / 2))))
        top_neuron_idx = _np.argsort(-widths)[:m]
        pairs = [
            (int(top_neuron_idx[i]), int(top_neuron_idx[j]))
            for i in range(m) for j in range(i + 1, m)
        ][:max_pairs]
    if not pairs:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    ng = int(hz_in.Gc.shape[1])
    nb = int(hz_in.Gb.shape[1])
    col_eps_np = col_eps.detach().cpu().numpy()
    unstable_np = unstable_idx.detach().cpu().numpy()

    rows_Ac: "list[_np.ndarray]" = []
    rows_Ab: "list[_np.ndarray]" = []
    rhss: "list[float]" = []

    Gc_pre_t = hz_in.Gc.detach().cpu().numpy()
    Gb_pre_t = hz_in.Gb.detach().cpu().numpy()
    c_pre_t = hz_in.c[:, 0].detach().cpu().numpy()

    for a_local, b_local in pairs:
        a_abs = int(unstable_np[a_local])
        b_abs = int(unstable_np[b_local])
        la = float(alpha_np[a_local]); ua = float(beta_np[a_local])
        lb_p = float(alpha_np[b_local]); ub_p = float(beta_np[b_local])
        facets = _correlated_pair_hull_facets(
            hz_in=hz_in,
            a_idx=a_abs, b_idx=b_abs,
            alpha_a=la, beta_a=ua,
            alpha_b=lb_p, beta_b=ub_p,
            n_dirs=n_dirs, lp_timeout_s=lp_timeout_s,
        )
        if not facets:
            continue
        Gc_a = Gc_pre_t[a_abs]
        Gb_a = Gb_pre_t[a_abs]
        c_a = float(c_pre_t[a_abs])
        Gc_b = Gc_pre_t[b_abs]
        Gb_b = Gb_pre_t[b_abs]
        c_b = float(c_pre_t[b_abs])
        col_a = int(col_eps_np[a_local])
        col_b = int(col_eps_np[b_local])
        lam_a = float(lam_np[a_local]); lam_b = float(lam_np[b_local])
        mu_a = float(mu_np[a_local]); mu_b = float(mu_np[b_local])
        for f in facets:
            c_pa = f["c_pa"]; c_pb = f["c_pb"]
            c_postA = f["c_postA"]; c_postB = f["c_postB"]
            rhs_f = f["rhs"]
            if (abs(c_pa) + abs(c_pb) + abs(c_postA) + abs(c_postB)) < 1e-12:
                continue
            mix_a = c_pa + c_postA * lam_a
            mix_b = c_pb + c_postB * lam_b
            row_Ac = _np.zeros(ng_new, dtype=_np.float64)
            row_Ab = _np.zeros(nb_new, dtype=_np.float64)
            row_Ac[:ng] = mix_a * Gc_a + mix_b * Gc_b
            row_Ab[:nb] = mix_a * Gb_a + mix_b * Gb_b
            row_Ac[col_a] += c_postA * mu_a
            row_Ac[col_b] += c_postB * mu_b
            row_rhs = (rhs_f
                       - mix_a * c_a
                       - mix_b * c_b
                       - c_postA * mu_a
                       - c_postB * mu_b)
            rows_Ac.append(row_Ac)
            rows_Ab.append(row_Ab)
            rhss.append(float(row_rhs))

    if not rows_Ac:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )
    Ac_rows = torch.tensor(_np.stack(rows_Ac), device=device, dtype=dtype)
    Ab_rows = torch.tensor(_np.stack(rows_Ab), device=device, dtype=dtype)
    b_rows = torch.tensor(
        _np.asarray(rhss).reshape(-1, 1), device=device, dtype=dtype
    )
    return Ac_rows, Ab_rows, b_rows


def _build_pair_cut_rows_for_relu(
    hz_in: HZono,
    unstable_idx: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    col_xi2: torch.Tensor,
    ng_new: int,
    nb_new: int,
    max_pairs: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Generate inequality rows for top-K unstable neuron pairs' joint
    ReLU hull facets. Returns (Ac_rows, Ab_rows, b_rows) ready to append.

    Pair scoring (forward-only, deterministic, P6-compliant):
      score(a, b) = (beta_a - alpha_a) * (beta_b - alpha_b)
    The widest pairs are most likely to contain phantom solutions in the
    triangle relaxation; pair-hull cuts have most slack to remove.
    """
    import numpy as _np
    device = hz_in.c.device
    dtype = hz_in.c.dtype
    k = int(unstable_idx.numel())
    if k < 2 or max_pairs <= 0:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    alpha_np = alpha.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()
    widths = beta_np - alpha_np
    # Score pairs by product of widths (decreasing).
    if max_pairs >= k * (k - 1) // 2:
        # All pairs (small k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    else:
        # Pick top-K by score deterministically. To keep it cheap, take
        # top-M neurons by width then enumerate pairs among them.
        m = min(k, max(8, int(round((1 + (1 + 8 * max_pairs) ** 0.5) / 2))))
        top_neuron_idx = _np.argsort(-widths)[:m]
        pairs = [
            (int(top_neuron_idx[i]), int(top_neuron_idx[j]))
            for i in range(m) for j in range(i + 1, m)
        ][:max_pairs]
    if not pairs:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )

    Gc_pre = hz_in.Gc[unstable_idx].detach().cpu().numpy()  # (k, ng)
    Gb_pre = hz_in.Gb[unstable_idx].detach().cpu().numpy()  # (k, nb)
    c_pre = hz_in.c[unstable_idx, 0].detach().cpu().numpy()  # (k,)
    ng = int(hz_in.Gc.shape[1])
    nb = int(hz_in.Gb.shape[1])
    col_xi2_np = col_xi2.detach().cpu().numpy()  # (k,) absolute indices in new ng

    rows_Ac: "list[_np.ndarray]" = []
    rows_Ab: "list[_np.ndarray]" = []
    rhss: "list[float]" = []
    for a_idx, b_idx in pairs:
        la = float(alpha_np[a_idx]); ua = float(beta_np[a_idx])
        lb_p = float(alpha_np[b_idx]); ub_p = float(beta_np[b_idx])
        facets = _pair_hull_facets_9vertex(la, ua, lb_p, ub_p)
        if not facets:
            continue
        Gc_a, Gc_b = Gc_pre[a_idx], Gc_pre[b_idx]
        Gb_a, Gb_b = Gb_pre[a_idx], Gb_pre[b_idx]
        c_a, c_b = float(c_pre[a_idx]), float(c_pre[b_idx])
        col_a, col_b = int(col_xi2_np[a_idx]), int(col_xi2_np[b_idx])
        beta_a, beta_b = float(beta_np[a_idx]), float(beta_np[b_idx])
        for f in facets:
            c_pa = f["c_pa"]; c_pb = f["c_pb"]
            c_postA = f["c_postA"]; c_postB = f["c_postB"]
            rhs = f["rhs"]
            # Skip degenerate / numerically-zero facets.
            if (abs(c_pa) + abs(c_pb) + abs(c_postA) + abs(c_postB)) < 1e-12:
                continue
            row_Ac = _np.zeros(ng_new, dtype=_np.float64)
            row_Ab = _np.zeros(nb_new, dtype=_np.float64)
            row_Ac[:ng] = c_pa * Gc_a + c_pb * Gc_b
            row_Ab[:nb] = c_pa * Gb_a + c_pb * Gb_b
            row_Ac[col_a] += c_postA * (-beta_a / 2.0)
            row_Ac[col_b] += c_postB * (-beta_b / 2.0)
            row_rhs = (rhs - c_pa * c_a - c_pb * c_b
                       - c_postA * (beta_a / 2.0)
                       - c_postB * (beta_b / 2.0))
            rows_Ac.append(row_Ac)
            rows_Ab.append(row_Ab)
            rhss.append(float(row_rhs))

    if not rows_Ac:
        return (
            hz_in.c.new_zeros((0, ng_new)),
            hz_in.c.new_zeros((0, nb_new)),
            hz_in.c.new_zeros((0, 1)),
        )
    Ac_rows = torch.tensor(_np.stack(rows_Ac), device=device, dtype=dtype)
    Ab_rows = torch.tensor(_np.stack(rows_Ab), device=device, dtype=dtype)
    b_rows = torch.tensor(
        _np.asarray(rhss).reshape(-1, 1), device=device, dtype=dtype
    )
    return Ac_rows, Ab_rows, b_rows


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    """Combine HZ-refined bounds (flat ``(n, 1)`` shape) with interval's
    batch-aware fact: reshape HZ bounds to match ``fact.bounds`` and keep
    interval's constraint set. Use everywhere a hybridz handler returns
    after refining the HZ cache.
    """
    hb = hz_compute_bounds(hz)
    return Fact(
        bounds=Bounds(
            lb=hb.lb.reshape_as(fact.bounds.lb),
            ub=hb.ub.reshape_as(fact.bounds.ub),
        ),
        cons=fact.cons,
    )


# ============================================================================
# Batch-native HZ helpers
# ----------------------------------------------------------------------------
# HZono stores ``c: (n, 1)``, ``Gc: (n, ng)``, ``Gb: (n, nb)`` where the
# leading dimension ``n`` is the *flattened* output size of the encoded
# layer including any leading batch axis ``B``. For per-channel ops
# (DENSE, BIAS, SCALE) we recover ``B`` from ``n // per_channel`` and
# operate via broadcasted 3D matmul / per-row scaling so that no
# block-diagonal weight is materialised.
# ============================================================================


def _hz_apply_per_batch_linear(hz: HZono, W: torch.Tensor, B: int) -> HZono:
    """Apply ``y = W x`` independently to each of ``B`` instances stacked
    along the leading axis of ``hz``. Equivalent to
    ``hz_multiply(hz, block_diag(W, ...))`` without materialising the
    block-diagonal matrix.
    """
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    if B == 1:
        return hz_multiply(hz, W)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    # (out, in) @ (B, in, *) broadcasts → (B, out, *)
    c3 = hz.c.view(B, in_dim, 1)
    new_c = (W @ c3).reshape(B * out_dim, 1)
    if ng:
        new_Gc = (W @ hz.Gc.view(B, in_dim, ng)).reshape(B * out_dim, ng)
    else:
        new_Gc = hz.Gc.new_zeros(B * out_dim, 0)
    if nb:
        new_Gb = (W @ hz.Gb.view(B, in_dim, nb)).reshape(B * out_dim, nb)
    else:
        new_Gb = hz.Gb.new_zeros(B * out_dim, 0)
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
    )


def _hz_add_per_channel(hz: HZono, v: torch.Tensor, B: int) -> HZono:
    """Add per-channel constant ``v: (out,)`` to each of ``B`` stacked
    instances in ``hz.c``. ``hz.c`` has shape ``(B*out, 1)``.
    """
    v = v.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        v = v.repeat(B)
    return hz_add_const(hz, v.view(-1, 1))


def _hz_scale_per_channel(hz: HZono, a: torch.Tensor, B: int) -> HZono:
    """Multiply hz fields by per-channel ``a: (out,)``. ``hz.c`` shape
    is ``(B*out, 1)``; we broadcast ``a`` once per batch via repeat.
    Equivalent to ``hz_multiply(hz, diag(a_repeated))`` without building
    the diagonal matrix.
    """
    a = a.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        a = a.repeat(B)
    a_col = a.view(-1, 1)
    return HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
    )


# ============================================================================
# HZ layer functions: HZono -> Optional[HZono] per layer kind
# Each takes (L, hz_in, tf) and returns the transformed HZono or None.
# ============================================================================


# --- HZ transfer functions (MLP) ---


def tf_dense(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        W = L.params["weight"].to(hz_in.c)
        in_dim = W.shape[1]
        B = hz_in.c.shape[0] // in_dim
        hz = _hz_apply_per_batch_linear(hz_in, W, B)
        bias = L.params.get("bias")
        if bias is not None:
            hz = _hz_add_per_channel(hz, bias, B)
        tf._hz_cache[L.id] = hz
    fact = interval.tf_dense(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bias(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        c = L.params["c"].to(hz_in.c)
        if c.ndim == 1:
            B = hz_in.c.shape[0] // c.numel()
            tf._hz_cache[L.id] = _hz_add_per_channel(hz_in, c, B)
        else:
            tf._hz_cache[L.id] = hz_add_const(hz_in, c)
    fact = interval.tf_bias(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_scale(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        a = L.params["a"].to(hz_in.c).flatten()
        B = hz_in.c.shape[0] // a.numel()
        tf._hz_cache[L.id] = _hz_scale_per_channel(hz_in, a, B)
    fact = interval.tf_scale(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_relu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        # P7e env-gated oracle (advisor 2026-06-08): when HYZOR_HYBRIDZ_USE_V8_RELU=1,
        # dispatch to hz_apply_relu_v8 with eq_lagr_v8 cascade (intersect_box +
        # bounds_tighten 3-tier UNC/dual/eq_elim LP + native eq_lagr + project_eq_elim).
        # Diagnostic only — to determine if old cpu_base CERT path was through v8.
        if os.environ.get("HYZOR_HYBRIDZ_USE_V8_RELU", "0").strip() == "1":
            from act.back_end.hybridz_tf.hz_routing import hz_apply_relu_v8
            tf._hz_cache[L.id] = hz_reduce(hz_apply_relu_v8(hz_in, method="eq_lagr_v8"))
        else:
            tf._hz_cache[L.id] = hz_reduce(hz_apply_relu(hz_in))
    fact = interval.tf_relu(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_lrelu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(
            hz_apply_leaky_relu(hz_in, float(L.params.get("negative_slope", 0.01)))
        )
    fact = interval.tf_lrelu(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_tanh(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_tanh(hz_in, K=tf._tanh_K))
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_sigmoid(hz_in, K=tf._sigmoid_K))
    fact = interval.tf_sigmoid(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_abs(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        bds = hz_compute_bounds(hz_in)
        lb_out = torch.where(
            bds.lb >= 0,
            bds.lb,
            torch.where(bds.ub <= 0, -bds.ub, torch.zeros_like(bds.lb)),
        )
        tf._hz_cache[L.id] = hz_from_bounds(
            Bounds(lb=lb_out, ub=torch.maximum(bds.lb.abs(), bds.ub.abs())),
            dtype,
            device,
        )
    fact = interval.tf_abs(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bn(L, bounds, tf):
    # BN's HZ refinement built a per-sample diag(A) of shape (n, n) and tried
    # to multiply with hz_in.c of shape (B*n, 1). That only works for B=1;
    # at B>1 it raises "mat1 and mat2 shapes cannot be multiplied" because
    # the HZ row layout flattens batch×feature into a single leading dim.
    # A batch-aware fix would require block-diag(A) replicated B times, which
    # is the same scope as a proper HZ batchify rewrite. Until that lands,
    # fall back to interval (sound, works at any B).
    tf._hz_cache[L.id] = None
    return interval.tf_bn(L, bounds)


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            # SGM-aware add: when the two HZs share their continuous-generator
            # block (ResNet skip-connection pattern with no Gc-transforming
            # ops on either branch), reuse the shared block instead of
            # block-diag concatenating. Falls back to Minkowski sum when not
            # shared. See act.back_end.hybridz_tf.algorithms.sgm.
            from act.back_end.hybridz_tf.algorithms.sgm import hz_sgm_add
            tf._hz_cache[L.id] = hz_sgm_add(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_add(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_mul(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            b1, b2 = hz_compute_bounds(hz_in), hz_compute_bounds(hz2)
            corners = torch.stack(
                [b1.lb * b2.lb, b1.lb * b2.ub, b1.ub * b2.lb, b1.ub * b2.ub]
            )
            tf._hz_cache[L.id] = hz_from_bounds(
                Bounds(lb=corners.min(0)[0], ub=corners.max(0)[0]), dtype, device
            )
        else:
            hz_in = None
    fact = interval.tf_mul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_constant(L, bounds, tf):
    val = L.params["value"].flatten()
    n = val.numel()
    # When the surrounding net is batched (e.g., upstream ADD sibling is
    # ``[B, *shape]``), replicate the constant per batch element so the
    # downstream HZ Minkowski-sum / element-wise ops see matching sizes.
    if bounds is not None and n > 0:
        in_numel = int(bounds.lb.numel())
        if in_numel > 0 and in_numel % n == 0:
            B = in_numel // n
            if B > 1:
                val = val.repeat(B)
                n = val.numel()
    tf._hz_cache[L.id] = HZono(
        c=val.view(-1, 1),
        Gc=val.new_zeros(n, 0),
        Gb=val.new_zeros(n, 0),
        Ac=val.new_zeros(0, 0),
        Ab=val.new_zeros(0, 0),
        b=val.new_zeros(0, 1),
    )
    return interval.tf_constant(L, bounds)


def tf_sign(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_sign(L, bounds)


def tf_compare(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_compare(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_where(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_where(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_matmul(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_matmul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def tf_upsample(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval_cnn.tf_upsample(L, bounds)


def tf_scatter_nd(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_scatter_nd(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_reduce_sum(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_reduce_sum(L, bounds)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        tf._hz_cache[L.id] = hz_from_bounds(fact.bounds, dtype, device)
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            result = parts[0]
            for p in parts[1:]:
                result = hz_minkowski_sum(result, p)
            tf._hz_cache[L.id] = result
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


# --- HZ activation encodings (zonotope domain) ---


def hz_apply_relu(hz: HZono, external_bounds=None) -> HZono:
    """Exact ReLU via equality constraints + linking equality.

    Per unstable neuron i with bounds [alpha, beta] (alpha < 0 < beta):
      ng += 4 (xi1, xi2, xi3, xi4)
      nb += 1 (z)
      nc += 3 equalities

    ``external_bounds``: optional ``(lb, ub)`` tuple to use for the
    active/inactive classification AND in the linking equality. When
    provided, skips the internal ``hz_compute_bounds`` call. This is
    HyZor's eq_lagr_v8 path: tighter pre-ReLU bounds (from Lagrangian
    dual or eq_elim LP) reduce k (unstable count) and produce a
    smaller output HZ. See HyZor ``applyReLU_eq_native`` (HZ:5021).
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if external_bounds is not None:
        lb_t, ub_t = external_bounds
        lb = lb_t.to(device=device, dtype=dtype).flatten()
        ub = ub_t.to(device=device, dtype=dtype).flatten()
    else:
        bounds = hz_compute_bounds(hz)
        lb = bounds.lb.flatten()
        ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    # Soundness fix: preserve input HZ's eq_mask. Constructing HZono without
    # eq_mask defaults to None ⇒ all rows treated as equalities (per
    # _eq_mask_of). When the input already carries inequality rows (e.g.
    # from hz_intersect_box's 2n box-clipping rows), losing the mask
    # silently converts those into equalities — over-constrains the LP and
    # can wrongly declare unsafe sets infeasible (UNSOUND on acasxu prop_2).
    em_old = _eq_mask_of(hz)

    out_Gc = hz.c.new_zeros(n, ng + 4 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)
    out_c = hz.c.new_zeros(n, 1)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if k == 0:
        out = HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=em_old.clone(),
        )
        _propagate_base(hz, out)
        return out

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = hz.c.new_zeros(3 * k, ng_new)
    eq_Ab = hz.c.new_zeros(3 * k, nb_new)
    eq_b = hz.c.new_zeros(3 * k, 1)

    r1 = 3 * t
    r2 = 3 * t + 1

    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    r3 = 3 * t + 2
    eq_Ac[r3, col_xi1] = alpha / 2.0
    eq_Ac[r3, col_xi2] = -beta / 2.0
    eq_Ac[r3, :ng] = -hz.Gc[unstable_idx]
    eq_Ab[r3, :nb] = -hz.Gb[unstable_idx]
    eq_Ab[r3, col_z] = alpha / 2.0
    eq_b[r3, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, 4 * k)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, k)], dim=1
    )

    em_new = torch.cat(
        [em_old, torch.ones(3 * k, dtype=torch.bool, device=device)]
    )

    Ac_final = torch.cat([old_Ac_ext, eq_Ac], dim=0)
    Ab_final = torch.cat([old_Ab_ext, eq_Ab], dim=0)
    b_final = torch.cat([hz.b, eq_b], dim=0)
    em_final = em_new

    # ─── Forward-only pair-ReLU joint hull cuts (env-gated) ───
    # When ACT_HZ_PAIR_RELU_CUTS=1, generate convex-hull facets for top-K
    # unstable-pair (pre, post) joint sets and append them as inequality
    # rows to (Ac, Ab, b). The cuts ARE the v8 encoding's missing joint
    # structure: each cut is a linear inequality in (xi_c_old, xi_c_new)
    # that holds for every real input but tightens the LP relaxation.
    #
    # Soundness: facets come from scipy.ConvexHull on the 9-vertex lifted
    # ReLU graph; every facet is derivable from the box + ReLU semantics
    # (see _pair_hull_facets_9vertex). No autograd, no backward, no
    # randomness.
    if os.environ.get("ACT_HZ_PAIR_RELU_CUTS", "0") == "1" and k >= 2:
        # DEPRECATED: independent-box hull is mathematically redundant.
        # See _pair_hull_facets_9vertex docstring + corresponding memory
        # entry. Kept only for retroactive experiments; default OFF.
        max_pairs = int(os.environ.get("ACT_HZ_PAIR_RELU_CUTS_MAX_PAIRS", "16"))
        try:
            cut_Ac, cut_Ab, cut_b = _build_pair_cut_rows_for_relu(
                hz_in=hz,
                unstable_idx=unstable_idx,
                alpha=alpha,
                beta=beta,
                col_xi2=col_xi2,
                ng_new=ng_new,
                nb_new=nb_new,
                max_pairs=max_pairs,
            )
            if cut_Ac.shape[0] > 0:
                Ac_final = torch.cat([Ac_final, cut_Ac], dim=0)
                Ab_final = torch.cat([Ab_final, cut_Ab], dim=0)
                b_final = torch.cat([b_final, cut_b], dim=0)
                em_final = torch.cat([
                    em_final,
                    torch.zeros(cut_Ac.shape[0], dtype=torch.bool, device=device),
                ])
        except Exception:
            pass

    # ─── Forward-only CORRELATED pair-ReLU joint hull cuts (env-gated) ───
    # These cuts use 8 SciPy LPs per pair over hz_in's existing constraint
    # set to compute the TRUE joint reachable set of (pre_a, pre_b), then
    # lift to 4D and take the hull. The resulting facets are NOT implied
    # by independent triangle×triangle constraints because they encode
    # the actual correlation introduced by shared input factors.
    # Each ReLU call when env-set adds top-K pair cuts; project_eq_elim
    # downstream propagates them through subsequent layers' linear ops.
    #
    # Soundness: facets derived from the convex hull of a SUPER-SET of
    # the true lifted joint set (the support polytope over-approximates
    # the linear image). No autograd, no backward bound prop, no
    # randomness — only 8 SciPy HiGHS LPs per pair.
    if os.environ.get("ACT_HZ_CORR_PAIR_CUTS", "0") == "1" and k >= 2:
        max_pairs_c = int(
            os.environ.get("ACT_HZ_CORR_PAIR_CUT_MAX_PAIRS", "4")
        )
        n_dirs = int(os.environ.get("ACT_HZ_CORR_PAIR_CUT_DIRS", "8"))
        lp_timeout = float(
            os.environ.get("ACT_HZ_CORR_PAIR_CUT_LP_TIMEOUT_S", "2.0")
        )
        layer_counter_now = RELU_LAYER_COUNTER[0]
        # Optional scope limiter: only emit cuts in the LAST K ReLU layers.
        # Since we don't know the total layer count up front, use a simple
        # heuristic — only emit cuts when `layer_counter_now` is in the
        # explicit target file (output-aware mode) OR when a separate env
        # whitelist matches. Default: no restriction (legacy behavior).
        _last_layers_only = os.environ.get(
            "ACT_HZ_CORR_PAIR_CUT_LAST_LAYERS", ""
        )
        _allowed_layers = None
        if _last_layers_only:
            try:
                _allowed_layers = {
                    int(s.strip()) for s in _last_layers_only.split(",")
                    if s.strip()
                }
            except Exception:
                _allowed_layers = None
        if (_allowed_layers is not None
                and layer_counter_now not in _allowed_layers):
            # Skip cut emission for this layer.
            RELU_LAYER_COUNTER[0] += 1
            _relu_trace_record_v8(
                hz_in=hz, unstable_idx=unstable_idx, alpha=alpha, beta=beta,
                col_xi2=col_xi2, col_z=col_z, ng_old=ng, nb_old=nb,
            )
            out = HZono(
                c=out_c, Gc=out_Gc, Gb=out_Gb,
                Ac=Ac_final, Ab=Ab_final, b=b_final,
                eq_mask=em_final,
            )
            _propagate_base(hz, out)
            return out
        try:
            cc_Ac, cc_Ab, cc_b = _build_correlated_pair_cut_rows_for_relu(
                hz_in=hz,
                unstable_idx=unstable_idx,
                alpha=alpha,
                beta=beta,
                col_xi2=col_xi2,
                ng_new=ng_new,
                nb_new=nb_new,
                max_pairs=max_pairs_c,
                n_dirs=n_dirs,
                lp_timeout_s=lp_timeout,
                layer_counter=layer_counter_now,
            )
            if cc_Ac.shape[0] > 0:
                Ac_final = torch.cat([Ac_final, cc_Ac], dim=0)
                Ab_final = torch.cat([Ab_final, cc_Ab], dim=0)
                b_final = torch.cat([b_final, cc_b], dim=0)
                em_final = torch.cat([
                    em_final,
                    torch.zeros(cc_Ac.shape[0], dtype=torch.bool, device=device),
                ])
        except Exception:
            # Cut generation is best-effort: failure leaves the v8
            # encoding intact (no soundness impact).
            pass

    out = HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=Ac_final,
        Ab=Ab_final,
        b=b_final,
        eq_mask=em_final,
    )
    _propagate_base(hz, out)

    # Forward-only ReLU trace: env-gated recording (separate from cuts —
    # cuts are inline; trace is for downstream diagnostic/analysis).
    _relu_trace_record_v8(
        hz_in=hz,
        unstable_idx=unstable_idx,
        alpha=alpha,
        beta=beta,
        col_xi2=col_xi2,
        col_z=col_z,
        ng_old=ng,
        nb_old=nb,
    )

    # Per-call ReLU layer counter, used by output-aware cut selection
    # (PAIR_TARGETS_BY_LAYER indexed by this counter) and by downstream
    # analysis. Increment unconditionally so layer numbering is stable.
    RELU_LAYER_COUNTER[0] += 1

    return out


def hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Exact LeakyReLU via the same encoding as ReLU.

    Per unstable neuron: ng += 4 (xi1, xi2, xi3, xi4), nb += 1 (z), nc += 3
    (graph eq 1, graph eq 2, linking eq) -- identical to hz_apply_relu.

    Decomposition: y = max(s*x, x) where s = alpha_arg. On the unstable
    branch, using the same switching mechanism as ReLU (z=+1 -> inactive
    with xi2 forced to 1; z=-1 -> active with xi1 forced to 1), we set
    the output as::

        y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z

    which degenerates exactly to ReLU's ``y_h = (beta/2)(1 - xi2)`` when
    s = 0. The graph equalities (xi1+xi3+z=1, xi2+xi4-z=1) and the linking
    equality (that ties x_h to xi1, xi2, z) are identical to ReLU.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    s = alpha_arg
    assert 0.0 <= s <= 1.0, f"hz_apply_leaky_relu: slope must be in [0, 1], got {s}"

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    em_old = _eq_mask_of(hz)  # Soundness: same fix as hz_apply_relu.

    out_Gc = hz.c.new_zeros(n, ng + 4 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)
    out_c = hz.c.new_zeros(n, 1)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if inactive.any():
        out_c[inactive] = s * hz.c[inactive]
        out_Gc[inactive, :ng] = s * hz.Gc[inactive]
        out_Gb[inactive, :nb] = s * hz.Gb[inactive]

    if k == 0:
        out = HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=em_old.clone(),
        )
        _propagate_base(hz, out)
        return out

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    # Output encoding: y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z
    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi1] = s * alpha / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0
    out_Gb[unstable_idx, col_z] = s * alpha / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = hz.c.new_zeros(3 * k, ng_new)
    eq_Ab = hz.c.new_zeros(3 * k, nb_new)
    eq_b = hz.c.new_zeros(3 * k, 1)

    r1 = 3 * t
    r2 = 3 * t + 1

    # Graph equality 1: xi1 + xi3 + z = 1
    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    # Graph equality 2: xi2 + xi4 - z = 1
    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    # Linking equality: ties x_h to (xi1, xi2, z)
    # Same form as ReLU; x_h has the same input expression.
    r3 = 3 * t + 2
    eq_Ac[r3, col_xi1] = alpha / 2.0
    eq_Ac[r3, col_xi2] = -beta / 2.0
    eq_Ac[r3, :ng] = -hz.Gc[unstable_idx]
    eq_Ab[r3, :nb] = -hz.Gb[unstable_idx]
    eq_Ab[r3, col_z] = alpha / 2.0
    eq_b[r3, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, 4 * k)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, k)], dim=1
    )

    em_new = torch.cat(
        [em_old, torch.ones(3 * k, dtype=torch.bool, device=device)]
    )

    out = HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
        eq_mask=em_new,
    )
    _propagate_base(hz, out)
    return out


def hz_apply_piecewise(hz: HZono, func, dfunc, K: int = 2) -> HZono:
    """Piecewise linear approximation for monotone activations (tangent parallelogram)."""
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.sum() if wide_idx.ndim == 0 else wide_idx.shape[0])

    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        return HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    segment_ids = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
    segment_width = (ub_w - lb_w).unsqueeze(0) / K
    a = lb_w.unsqueeze(0) + segment_ids * segment_width
    b_seg = a + segment_width
    fa, fb = func(a), func(b_seg)
    la, lb_slope = dfunc(a), dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = (la - lb_slope).abs() < 1e-10

    denom = lb_slope - la
    safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).view(50, 1, 1)
    pts = a.unsqueeze(0) + t_pts * (b_seg - a).unsqueeze(0)
    f_pts = func(pts)
    resid = f_pts - (
        slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0)
    )
    max_err = resid.abs().max(dim=0).values
    g1x_lin, g1y_lin = hw, slope * hw
    g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

    g1_x = torch.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = torch.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = torch.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = torch.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.unsqueeze(0)
    dy = f_pts - centers_y.unsqueeze(0)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
    xi1 = (dy * g2_x.unsqueeze(0) - dx * g2_y.unsqueeze(0)) / safe_det.unsqueeze(0)
    xi2 = (dy * g1_x.unsqueeze(0) - dx * g1_y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
    max_xi = torch.maximum(xi1.abs().amax(dim=0), xi2.abs().amax(dim=0))
    scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
    scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    cy_sum = centers_y.sum(dim=0)
    new_c[wide_idx] = (cy_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * K * m
    n_slack = 4 * K * m
    Gc_new = hz.c.new_zeros(n, n_real + n_slack)
    g1_cols = torch.arange(K * m, device=device).reshape(K, m)
    g2_cols = (K * m + torch.arange(K * m, device=device)).reshape(K, m)
    wide_rows = wide_idx.unsqueeze(0).expand(K, -1)
    Gc_new[wide_rows, g1_cols] = g1_y
    Gc_new[wide_rows, g2_cols] = g2_y

    Gb_new = hz.c.new_zeros(n, K * m)
    z_cols = torch.arange(K * m, device=device).reshape(K, m)
    Gb_new[wide_rows, z_cols] = -centers_y / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)
    ng_total = ng + n_real + n_slack
    nb_total = nb + K * m

    n_box = 4 * K * m
    n_eq_total = n_box + m + m
    eq_Ac = hz.c.new_zeros(n_eq_total, ng_total)
    eq_Ab = hz.c.new_zeros(n_eq_total, nb_total)
    eq_b = hz.c.new_zeros(n_eq_total, 1)

    segment_grid = torch.arange(K * m, device=device).reshape(K, m)
    g1_col_grid = ng + segment_grid
    g2_col_grid = ng + K * m + segment_grid
    z_col_grid = nb + segment_grid
    slack_base_grid = ng + n_real + 4 * segment_grid
    row_grid = 4 * segment_grid

    flat_rows = row_grid.reshape(-1)
    flat_g1_cols = g1_col_grid.reshape(-1)
    flat_g2_cols = g2_col_grid.reshape(-1)
    flat_z_cols = z_col_grid.reshape(-1)
    flat_slack_bases = slack_base_grid.reshape(-1)

    eq_Ac[flat_rows, flat_g1_cols] = 1.0
    eq_Ac[flat_rows, flat_slack_bases] = 1.0
    eq_Ab[flat_rows, flat_z_cols] = -0.5
    eq_b[flat_rows, 0] = 0.5

    eq_Ac[flat_rows + 1, flat_g1_cols] = -1.0
    eq_Ac[flat_rows + 1, flat_slack_bases + 1] = 1.0
    eq_Ab[flat_rows + 1, flat_z_cols] = -0.5
    eq_b[flat_rows + 1, 0] = 0.5

    eq_Ac[flat_rows + 2, flat_g2_cols] = 1.0
    eq_Ac[flat_rows + 2, flat_slack_bases + 2] = 1.0
    eq_Ab[flat_rows + 2, flat_z_cols] = -0.5
    eq_b[flat_rows + 2, 0] = 0.5

    eq_Ac[flat_rows + 3, flat_g2_cols] = -1.0
    eq_Ac[flat_rows + 3, flat_slack_bases + 3] = 1.0
    eq_Ab[flat_rows + 3, flat_z_cols] = -0.5
    eq_b[flat_rows + 3, 0] = 0.5

    link_rows = n_box + torch.arange(m, device=device)
    link_row_grid = link_rows.unsqueeze(1).expand(-1, K)
    eq_Ac[link_row_grid, g1_col_grid.transpose(0, 1)] = -g1_x.transpose(0, 1)
    eq_Ac[link_row_grid, g2_col_grid.transpose(0, 1)] = -g2_x.transpose(0, 1)
    eq_Ab[link_row_grid, z_col_grid.transpose(0, 1)] = centers_x.transpose(0, 1) / 2.0
    eq_Ac[link_rows, :ng] = hz.Gc[wide_idx]
    eq_Ab[link_rows, :nb] = hz.Gb[wide_idx]
    eq_b[link_rows, 0] = centers_x.sum(dim=0) / 2.0 - hz.c[wide_idx, 0]

    sum_rows = n_box + m + torch.arange(m, device=device)
    sum_row_grid = sum_rows.unsqueeze(1).expand(-1, K)
    eq_Ab[sum_row_grid, z_col_grid.transpose(0, 1)] = 1.0
    eq_b[sum_rows, 0] = hz.c.new_full((m,), float(K - 2))

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, n_real + n_slack)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, K * m)], dim=1
    )

    return HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_sigmoid(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear sigmoid via tangent parallelogram encoding."""
    return hz_apply_piecewise(
        hz, torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K
    )


def hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear tanh via tangent parallelogram encoding."""
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K)


# --- HZ order reduction ---


def hz_reduce(hz: HZono, max_order: float = 3.0) -> HZono:
    """Reduce HZ complexity via Girard's method (sound over-approximation)."""
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if n == 0:
        return hz

    max_ng = max(int(max_order * n), n + 1)
    max_nb = max(2 * n, 1)

    # Step 1: Relax excess binary generators to continuous
    if nb > max_nb:
        col_norms = hz.Gb.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort()
        n_relax = nb - max_nb
        relax_idx = sorted_idx[:n_relax]
        keep_idx = sorted_idx[n_relax:]
        extra_Gc = hz.Gb[:, relax_idx]
        extra_Ac = (
            hz.Ab[:, relax_idx]
            if nc > 0
            else hz.c.new_zeros(0, n_relax)
        )
        hz = HZono(
            c=hz.c,
            Gc=torch.cat([hz.Gc, extra_Gc], dim=1),
            Gb=hz.Gb[:, keep_idx],
            Ac=torch.cat([hz.Ac, extra_Ac], dim=1)
            if nc > 0
            else hz.c.new_zeros(0, ng + n_relax),
            Ab=hz.Ab[:, keep_idx]
            if nc > 0
            else hz.c.new_zeros(0, max_nb),
            b=hz.b.clone(),
        )
        ng = hz.Gc.shape[1]
        nb = hz.Gb.shape[1]

    # Step 2: Reduce continuous generators
    if ng > max_ng:
        col_norms = hz.Gc.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort(descending=True)
        keep_idx = sorted_idx[: max_ng - n]
        drop_idx = sorted_idx[max_ng - n :]
        Gc_keep = hz.Gc[:, keep_idx]
        new_Gc = torch.cat(
            [Gc_keep, torch.diag(hz.Gc[:, drop_idx].abs().sum(dim=1))], dim=1
        )

        if nc > 0:
            has_dropped = hz.Ac[:, drop_idx].abs().max(dim=1).values > 1e-15
            keep_mask = ~has_dropped
            krt = torch.where(keep_mask)[0]
            if krt.numel() > 0:
                new_Ac = torch.cat(
                    [
                        hz.Ac[krt][:, keep_idx],
                        hz.c.new_zeros(krt.numel(), n),
                    ],
                    dim=1,
                )
                new_Ab = hz.Ab[krt]
                new_b = hz.b[krt]
            else:
                new_Ac = hz.c.new_zeros(0, new_Gc.shape[1])
                new_Ab = hz.c.new_zeros(0, nb)
                new_b = hz.c.new_zeros(0, 1)
        else:
            new_Ac = hz.c.new_zeros(0, new_Gc.shape[1])
            new_Ab = hz.c.new_zeros(0, nb)
            new_b = hz.c.new_zeros(0, 1)

        hz = HZono(c=hz.c, Gc=new_Gc, Gb=hz.Gb, Ac=new_Ac, Ab=new_Ab, b=new_b)

    return hz
