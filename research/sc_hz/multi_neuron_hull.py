"""F2b: multi-neuron joint hull cuts for forward HZ-LP.

Per advisor 2026-06-05 directive: F1 single-neuron triangle in LP gave 17%
median tightening on cifar — sound but insufficient. Math projection shows
single-neuron mechanism cannot break 1472. The actual missing constraint is
the CORRELATION between multiple ReLU slacks at the SAME layer when their
pre-activations z_i, z_j share input dependencies (G[i,:] and G[j,:] are
not independent).

The key observation:
  In our HZ, z_i = c_i + G[i,:] @ xi where xi ∈ [-1,1]^K.
  The (z_i, z_j) joint set is a 2D zonotope (= projection of K-dim hypercube
  through (G[i,:], G[j,:])), NOT the full box [l_i, u_i] × [l_j, u_j].
  Convex hull of ReLU over this 2D zonotope can be strictly TIGHTER than
  the product of per-neuron triangles.

This module implements:
  - `pairwise_joint_hull_cuts`: for each pair (i, j) of unstable neurons at
    the last ReLU, derive valid cuts on (y_i + α y_j, z_i, z_j)
  - `multi_neuron_lp_ub`: extends `constrained_lp_ub` with these cuts

PRINCIPLE compliance:
  - Forward-only: cuts use ONLY forward HZ pre-activation data
  - Continuous LP: no MILP, no binary, no integer reasoning
  - No gradient, no random, no BaB
  - All cuts must satisfy MONOTONICITY: UB_new ≤ UB_old + 1e-8

Per advisor F2b hard gates:
  1. UB_new ≤ UB_old + 1e-8 (no widening) — tested on synthetic
  2. brute-force samples satisfy LP UB (soundness)
  3. ≥40% median drop OR ≥1 NEW CERT on 8-iid sentinel
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize as sopt

from research.sc_hz.constrained_lp import LastReluRecord, constrained_lp_ub


def _zonotope_2d_extremes(
    G_i: np.ndarray, G_j: np.ndarray,
    tail_i: float = 0.0, tail_j: float = 0.0,
) -> List[Tuple[float, float]]:
    """For a 2D zonotope (sum over xi_k ∈ [-1,1] of (G_i[k] xi_k, G_j[k] xi_k))
    return its EXTREME vertices.

    The 2D zonotope on K generators has at most 2K vertices, which are the
    images of the K-cube's "leading directions". We compute them by sorting
    columns by angle and taking the partial sums (standard 2D zonotope
    construction).
    """
    K = len(G_i)
    if K == 0:
        return [(0.0, 0.0)]
    # Add tail as two extra generators (per-coord independent)
    if tail_i != 0 or tail_j != 0:
        G_i = np.concatenate([G_i, [tail_i, 0.0]])
        G_j = np.concatenate([G_j, [0.0, tail_j]])
    # Each generator: vector (G_i[k], G_j[k]). For zonotope vertices,
    # sort by angle of vector, then for each k take +1 if vector points
    # in positive half-plane wrt a sweep direction.
    vecs = np.array([G_i, G_j]).T  # (K, 2)
    # Magnitudes
    mags = np.linalg.norm(vecs, axis=1)
    nonzero = mags > 1e-12
    vecs = vecs[nonzero]
    if vecs.shape[0] == 0:
        return [(0.0, 0.0)]
    K2 = vecs.shape[0]
    # Sort by angle (use only upper half-plane representatives)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    # Make all vectors point "upward" (angle in [0, pi))
    flip = angles < 0
    vecs_up = vecs.copy()
    vecs_up[flip] = -vecs_up[flip]
    angles_up = np.arctan2(vecs_up[:, 1], vecs_up[:, 0])  # in [0, pi)
    order = np.argsort(angles_up)
    vecs_sorted = vecs_up[order]
    # First vertex: -sum_k vec_k
    start = -vecs_sorted.sum(axis=0)
    vertices = [tuple(start)]
    cur = np.array(start)
    for v in vecs_sorted:
        cur = cur + 2 * v
        vertices.append(tuple(cur))
    return vertices


def _exact_joint_relu_max_on_box(
    l_i: float, u_i: float, l_j: float, u_j: float,
    alpha_i: float, alpha_j: float,
) -> float:
    """Compute max alpha_i * relu(z_i) + alpha_j * relu(z_j) on [l_i, u_i] × [l_j, u_j].

    relu is piecewise. Max is at one of the 4 corner-region maxima.
    Each region:
      Region (z_i > 0, z_j > 0): obj = alpha_i z_i + alpha_j z_j, max at corner where signs align
      Region (z_i > 0, z_j ≤ 0): obj = alpha_i z_i, max at (max(u_i, 0), [l_j, u_j]∩(-inf,0])
      etc.
    """
    candidates = []
    # 4 box corners
    for zi in [l_i, u_i]:
        for zj in [l_j, u_j]:
            v = alpha_i * max(0, zi) + alpha_j * max(0, zj)
            candidates.append(v)
    # Boundary points at z=0
    for zi in [l_i, u_i]:
        v = alpha_i * max(0, zi) + alpha_j * max(0, 0.0)
        candidates.append(v)
    for zj in [l_j, u_j]:
        v = alpha_i * max(0, 0.0) + alpha_j * max(0, zj)
        candidates.append(v)
    return float(max(candidates))


@dataclass
class JointCut:
    """One linear cut: alpha_i * y_i + alpha_j * y_j ≤ rhs (constant)."""
    i: int            # neuron index 1
    j: int            # neuron index 2
    alpha_i: float
    alpha_j: float
    rhs: float


def _compute_zonotope_2d_polygon(G_i, G_j, tail_i=0.0, tail_j=0.0):
    """Build 2D zonotope as ordered polygon vertices via angular sort.

    The zonotope is { sum_k (G_i[k], G_j[k]) * xi_k : xi_k ∈ [-1, 1] }
    plus tail contributions (tail_i, 0) and (0, tail_j) as extra generators.
    Returns ORDERED vertices (counter-clockwise).
    """
    vecs = np.stack([G_i, G_j], axis=1).astype(np.float64)
    if tail_i != 0.0:
        vecs = np.concatenate([vecs, [[tail_i, 0.0]]], axis=0)
    if tail_j != 0.0:
        vecs = np.concatenate([vecs, [[0.0, tail_j]]], axis=0)
    mags = np.linalg.norm(vecs, axis=1)
    keep = mags > 1e-12
    vecs = vecs[keep]
    if vecs.shape[0] == 0:
        return np.array([[0.0, 0.0]])
    # Make each vector point "upward" (angle in [-pi/2, pi/2))
    flip = vecs[:, 0] < 0
    flip |= (vecs[:, 0] == 0) & (vecs[:, 1] < 0)
    vecs[flip] = -vecs[flip]
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    order = np.argsort(angles)
    vecs_sorted = vecs[order]
    # Start at -sum_k vec_k (most negative corner)
    start = -vecs_sorted.sum(axis=0)
    n = vecs_sorted.shape[0]
    verts = np.empty((2 * n + 1, 2), dtype=np.float64)
    verts[0] = start
    cur = start.copy()
    for k in range(n):
        cur = cur + 2 * vecs_sorted[k]
        verts[k + 1] = cur
    for k in range(n):
        cur = cur - 2 * vecs_sorted[k]
        verts[n + 1 + k] = cur
    # Last vertex should equal start (close the loop); drop the duplicate
    return verts[:-1]


def derive_pairwise_zonotope_cuts(
    relu_rec: LastReluRecord, d_eff: np.ndarray,
    top_k: int = 4,
) -> List[JointCut]:
    """Derive pairwise joint cuts via 2D zonotope vertex + relu-kink enumeration.

    For each pair (i, j):
      1. Project the K-dim hypercube via (G_z[i,:], G_z[j,:]) to 2D polygon
         (2D zonotope) — O(K log K).
      2. Walk each polygon edge; find intersections with the ReLU kink lines
         (z_i = 0 in (vx, vy) space at vx = -c_i; same for z_j) — O(K).
      3. Evaluate alpha_i * relu(c_i + vx) + alpha_j * relu(c_j + vy) at every
         polygon vertex AND every edge-kink intersection. Take max.
    Soundness: max of piecewise-linear function on convex polygon is at one
    of these critical points (polygon vertex OR slope change line crossing).
    """
    unstable = relu_rec.unstable_mask()
    unstable_idx = np.where(unstable)[0]
    if len(unstable_idx) < 2:
        return []
    d_eff_unstable = np.abs(d_eff[unstable_idx])
    top = np.argsort(-d_eff_unstable)[:top_k]
    top_neuron_idx = unstable_idx[top]
    has_tail = relu_rec.tail_z is not None
    cuts: List[JointCut] = []

    for ii_pos in range(len(top_neuron_idx)):
        for jj_pos in range(ii_pos + 1, len(top_neuron_idx)):
            i = int(top_neuron_idx[ii_pos])
            j = int(top_neuron_idx[jj_pos])
            G_i = relu_rec.G_z[i, :]
            G_j = relu_rec.G_z[j, :]
            c_i = float(relu_rec.c_z[i])
            c_j = float(relu_rec.c_z[j])
            tail_i = float(relu_rec.tail_z[i]) if has_tail else 0.0
            tail_j = float(relu_rec.tail_z[j]) if has_tail else 0.0
            alpha_i = float(d_eff[i])
            alpha_j = float(d_eff[j])

            poly = _compute_zonotope_2d_polygon(G_i, G_j, tail_i, tail_j)
            n_v = poly.shape[0]

            # Collect critical points: all polygon vertices + edge-kink intersections
            critical = [poly[k] for k in range(n_v)]
            for k in range(n_v):
                p1 = poly[k]; p2 = poly[(k + 1) % n_v]
                # Intersection with vx = -c_i (z_i = 0 line in zonotope frame)
                if (p1[0] + c_i) * (p2[0] + c_i) < 0:
                    t = (-c_i - p1[0]) / (p2[0] - p1[0])
                    critical.append(p1 + t * (p2 - p1))
                if (p1[1] + c_j) * (p2[1] + c_j) < 0:
                    t = (-c_j - p1[1]) / (p2[1] - p1[1])
                    critical.append(p1 + t * (p2 - p1))

            crit = np.array(critical)
            z_i_vals = crit[:, 0] + c_i
            z_j_vals = crit[:, 1] + c_j
            y_i_vals = np.maximum(0.0, z_i_vals)
            y_j_vals = np.maximum(0.0, z_j_vals)
            obj_vals = alpha_i * y_i_vals + alpha_j * y_j_vals
            best = float(obj_vals.max())

            cuts.append(JointCut(
                i=i, j=j, alpha_i=alpha_i, alpha_j=alpha_j, rhs=float(best),
            ))
    return cuts


def multi_neuron_lp_ub(
    relu_rec: LastReluRecord, W_remaining: np.ndarray,
    b_remaining: np.ndarray, d_out: np.ndarray,
    top_k_neurons: int = 4,
    return_solution: bool = False,
) -> Tuple[float, Optional[dict]]:
    """Constrained LP UB with F1 (per-neuron triangle) + F2b (pairwise joint cuts).

    Adds pairwise zonotope-derived cuts to the F1 LP.
    """
    K = relu_rec.n_gen
    n = relu_rec.n_pre
    has_tail = relu_rec.tail_z is not None

    is_active = relu_rec.stable_active_mask()
    is_inactive = relu_rec.stable_inactive_mask()
    is_unstable = relu_rec.unstable_mask()

    den = np.where(is_unstable, relu_rec.u - relu_rec.l, 1.0)
    lam = np.where(is_unstable, relu_rec.u / np.maximum(den, 1e-300), 0.0)

    d_eff = W_remaining.T @ d_out  # (n,)
    const = float(d_out @ b_remaining)

    n_xitail = n if has_tail else 0
    n_y = n
    n_vars = K + n_xitail + n_y

    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[K + n_xitail:K + n_xitail + n_y] = -d_eff

    bounds = [(-1.0, 1.0)] * K
    if has_tail:
        bounds += [(-1.0, 1.0)] * n_xitail
    for i in range(n_y):
        if is_inactive[i]:
            bounds.append((0.0, 0.0))
        elif is_active[i]:
            bounds.append((max(0.0, relu_rec.l[i]), max(0.0, relu_rec.u[i])))
        else:
            bounds.append((0.0, max(0.0, relu_rec.u[i])))

    # Equality for active
    A_eq_rows = []; b_eq_rows = []
    for i in np.where(is_active)[0]:
        row = np.zeros(n_vars)
        row[0:K] = -relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = -relu_rec.tail_z[i]
        row[K + n_xitail + i] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(relu_rec.c_z[i])

    # Triangle inequalities for unstable
    A_ub_rows = []; b_ub_rows = []
    for i in np.where(is_unstable)[0]:
        # y_i >= z_i
        row = np.zeros(n_vars)
        row[0:K] = relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = relu_rec.tail_z[i]
        row[K + n_xitail + i] = -1.0
        A_ub_rows.append(row); b_ub_rows.append(-relu_rec.c_z[i])
        # y_i ≤ lam_i (z_i - l_i)
        row = np.zeros(n_vars)
        row[0:K] = -lam[i] * relu_rec.G_z[i, :]
        if has_tail:
            row[K + i] = -lam[i] * relu_rec.tail_z[i]
        row[K + n_xitail + i] = 1.0
        A_ub_rows.append(row); b_ub_rows.append(lam[i] * (relu_rec.c_z[i] - relu_rec.l[i]))

    # F2b: pairwise joint cuts on top-K unstable neurons by |d_eff|
    pairs_cuts = derive_pairwise_zonotope_cuts(relu_rec, d_eff, top_k=top_k_neurons)
    n_cuts = 0
    for cut in pairs_cuts:
        # cut: alpha_i * y_i + alpha_j * y_j ≤ rhs
        # If both alpha_i and alpha_j are NEGATIVE, the LP wants to minimize the LHS
        # which makes the cut potentially trivial. Apply only if max-direction makes sense:
        # We want to bound max alpha_i*y_i + alpha_j*y_j; the cut is valid for ALL feasible
        # (y_i, y_j) so it's always a valid upper bound on the LHS.
        # But it's only HELPFUL if it tightens (not redundant). Check:
        # per-neuron triangle gives: alpha_i*y_i ≤ alpha_i * (lam_i * (c_z_i + |G_i| - l_i))
        # which is approximately u_i for unstable. So per-neuron is ≤ alpha_i*u_i + alpha_j*u_j.
        # If cut's rhs is < this, it tightens.
        # Just add and let LP find best.
        row = np.zeros(n_vars)
        row[K + n_xitail + cut.i] = cut.alpha_i
        row[K + n_xitail + cut.j] = cut.alpha_j
        A_ub_rows.append(row); b_ub_rows.append(cut.rhs)
        n_cuts += 1

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None

    res = sopt.linprog(
        c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if res.status != 0:
        return float("inf"), {"status": res.status, "n_cuts": n_cuts}

    ub = const - float(res.fun)
    sol = {"status": res.status, "n_cuts": n_cuts} if return_solution else None
    return float(ub), sol
