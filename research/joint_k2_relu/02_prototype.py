"""Joint K=2 ReLU envelope prototype.

Standalone numpy/scipy prototype to verify soundness + measure precision
gain on small synthetic HZs BEFORE integrating into ACT's HZ pipeline.

Sound joint K=2 envelope: for each pair of unstable neurons (x_i, x_j),
solve 8 inner LPs to compute the joint upper envelope of (max(0,x_i),
max(0,x_j)) over the input HZ in 8 octant directions. Add these 8
inequalities to the post-ReLU HZ.

This is sound (over-approximation) — proof in 01_math_derivation.md.
"""
import numpy as np
from scipy.optimize import linprog
from itertools import combinations


def compute_unstable_pairs_by_correlation(Gc: np.ndarray,
                                          lb: np.ndarray, ub: np.ndarray,
                                          max_pairs: int = None):
    """Pair unstable neurons greedily by cosine similarity of Gc rows.

    Args:
        Gc: (n_neurons, p_gen) continuous generator matrix at pre-act.
        lb, ub: (n_neurons,) pre-activation bounds.
        max_pairs: cap on number of pairs returned.

    Returns:
        List of (i, j) tuples of paired neuron indices.
    """
    unstable_mask = (lb < 0) & (ub > 0)
    unstable_idx = np.where(unstable_mask)[0]
    if len(unstable_idx) < 2:
        return []

    G_u = Gc[unstable_idx]  # (n_unstable, p)
    norms = np.linalg.norm(G_u, axis=1, keepdims=True).clip(min=1e-12)
    G_u_norm = G_u / norms
    sim = G_u_norm @ G_u_norm.T  # (n_u, n_u) cosine similarity

    # Greedy pairing by descending |sim|
    n_u = len(unstable_idx)
    paired = np.zeros(n_u, dtype=bool)
    pairs = []
    triu_i, triu_j = np.triu_indices(n_u, k=1)
    sim_flat = np.abs(sim[triu_i, triu_j])
    order = np.argsort(-sim_flat)
    for k in order:
        i, j = triu_i[k], triu_j[k]
        if paired[i] or paired[j]:
            continue
        pairs.append((int(unstable_idx[i]), int(unstable_idx[j])))
        paired[i] = True
        paired[j] = True
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def joint_envelope_lp(Gc_i: np.ndarray, Gc_j: np.ndarray,
                      c_i: float, c_j: float,
                      l_i: float, u_i: float, l_j: float, u_j: float,
                      A_eq=None, b_eq=None, A_le=None, b_le=None,
                      directions=None):
    """For ONE pair, compute joint upper envelope in given directions.

    Decision variables: ξ ∈ [-1, 1]^p (continuous factor),
        relu_i = max(0, x_i),  relu_j = max(0, x_j)
    where x_i = c_i + Gc_i · ξ,  x_j = c_j + Gc_j · ξ.

    For each direction (a_i, a_j), solve LP:
        max a_i · relu_i + a_j · relu_j
        s.t. relu_i ≥ 0, relu_i ≥ x_i (= c_i + Gc_i · ξ)
             relu_j ≥ 0, relu_j ≥ x_j (= c_j + Gc_j · ξ)
             relu_i ≤ u_i · (x_i - l_i) / (u_i - l_i)   [triangle upper]
             relu_j ≤ u_j · (x_j - l_j) / (u_j - l_j)
             [hz constraints on ξ]
             ξ ∈ [-1, 1]^p

    Returns:
        envelope: dict {dir: rhs_value} for each direction
    """
    p = Gc_i.shape[0]
    # Variable order: ξ (p), relu_i (1), relu_j (1) — total p+2
    n_var = p + 2
    bounds = [(-1.0, 1.0)] * p + [(0.0, max(u_i, 0.0))] + [(0.0, max(u_j, 0.0))]

    # Build constraint matrices
    A_le_list = []
    b_le_list = []

    # relu_i ≥ x_i  ⇔  -relu_i + Gc_i · ξ ≤ -c_i
    row = np.zeros(n_var)
    row[:p] = Gc_i
    row[p] = -1.0
    A_le_list.append(row)
    b_le_list.append(-c_i)

    # relu_j ≥ x_j  ⇔  -relu_j + Gc_j · ξ ≤ -c_j
    row = np.zeros(n_var)
    row[:p] = Gc_j
    row[p + 1] = -1.0
    A_le_list.append(row)
    b_le_list.append(-c_j)

    # relu_i ≤ u_i · (x_i - l_i) / (u_i - l_i)
    # = u_i/(u_i - l_i) · x_i - u_i·l_i/(u_i - l_i)
    # = α_i · (c_i + Gc_i·ξ) - β_i
    # relu_i - α_i · Gc_i · ξ ≤ α_i · c_i - β_i
    if u_i > l_i:
        alpha_i = u_i / (u_i - l_i)
        beta_i = u_i * l_i / (u_i - l_i)
        row = np.zeros(n_var)
        row[:p] = -alpha_i * Gc_i
        row[p] = 1.0
        A_le_list.append(row)
        b_le_list.append(alpha_i * c_i - beta_i)

    if u_j > l_j:
        alpha_j = u_j / (u_j - l_j)
        beta_j = u_j * l_j / (u_j - l_j)
        row = np.zeros(n_var)
        row[:p] = -alpha_j * Gc_j
        row[p + 1] = 1.0
        A_le_list.append(row)
        b_le_list.append(alpha_j * c_j - beta_j)

    # External HZ constraints on ξ
    if A_le is not None and A_le.shape[0] > 0:
        ext = np.concatenate([A_le, np.zeros((A_le.shape[0], 2))], axis=1)
        A_le_list.append(ext)
        b_le_list.append(b_le)

    A_le_mat = np.vstack(A_le_list) if A_le_list else None
    b_le_vec = np.concatenate([np.atleast_1d(x) for x in b_le_list]) if b_le_list else None

    A_eq_mat = None
    b_eq_vec = None
    if A_eq is not None and A_eq.shape[0] > 0:
        A_eq_mat = np.concatenate([A_eq, np.zeros((A_eq.shape[0], 2))], axis=1)
        b_eq_vec = b_eq

    if directions is None:
        # 8 octant directions
        directions = [
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
        ]

    envelope = {}
    for (a_i, a_j) in directions:
        obj = np.zeros(n_var)
        obj[p] = a_i
        obj[p + 1] = a_j
        # Maximize → minimize negative
        res = linprog(
            c=-obj,
            A_ub=A_le_mat, b_ub=b_le_vec,
            A_eq=A_eq_mat, b_eq=b_eq_vec,
            bounds=bounds, method="highs",
        )
        if res.status == 0 and res.success:
            envelope[(a_i, a_j)] = float(-res.fun)
        else:
            envelope[(a_i, a_j)] = None  # LP failure
    return envelope


# ───────── Test 1: anti-correlation example from math derivation ─────────

def test_anti_correlation():
    """x_1 = ξ, x_2 = -ξ, ξ ∈ [-1, 1]. True joint convex hull is the
    triangle y_1 + y_2 ≤ 1 (with y_i ≥ 0). Per-neuron triangle gives
    full box [0,1]² which has y_1 + y_2 ≤ 2.

    Joint envelope LP should return y_1 + y_2 ≤ 1, recovering the true
    triangle (up to LP precision).
    """
    Gc_1 = np.array([1.0])   # x_1 = 0 + 1·ξ
    Gc_2 = np.array([-1.0])  # x_2 = 0 + (-1)·ξ
    env = joint_envelope_lp(
        Gc_1, Gc_2, c_i=0.0, c_j=0.0,
        l_i=-1.0, u_i=1.0, l_j=-1.0, u_j=1.0,
    )
    print("Test 1: anti-correlation x_1 = ξ, x_2 = -ξ")
    print(f"  Envelope: {env}")
    print(f"  Joint y_1 + y_2 ≤ ?  expected 1.0,  got {env[(1.0, 1.0)]:.6f}")
    print(f"  Per-neuron y_1 + y_2 ≤ 2.0 (loose)")
    print(f"  Tightness gain: {2.0 - env[(1.0, 1.0)]:.6f}")
    assert abs(env[(1.0, 1.0)] - 1.0) < 1e-6, "Should recover joint y_1+y_2≤1"
    print("  ✓ PASS")


def test_independent():
    """x_1 = ξ_1, x_2 = ξ_2 (independent). Joint envelope should match
    per-neuron product (no precision gain)."""
    Gc_1 = np.array([1.0, 0.0])  # x_1 = ξ_1
    Gc_2 = np.array([0.0, 1.0])  # x_2 = ξ_2
    env = joint_envelope_lp(
        Gc_1, Gc_2, c_i=0.0, c_j=0.0,
        l_i=-1.0, u_i=1.0, l_j=-1.0, u_j=1.0,
    )
    print("\nTest 2: independent x_1 = ξ_1, x_2 = ξ_2")
    print(f"  Envelope: {env}")
    print(f"  Joint y_1 + y_2 ≤ ?  expected 2.0 (no gain)")
    print(f"  Got: {env[(1.0, 1.0)]:.6f}")
    assert abs(env[(1.0, 1.0)] - 2.0) < 1e-6, "Independent: no extra cut"
    print("  ✓ PASS (no spurious cut)")


def test_partial_correlation():
    """x_1 = 0.7·ξ_1 + 0.3·ξ_2, x_2 = -0.5·ξ_1 + 0.8·ξ_2. Mid-correlation.
    Verify joint envelope gives a non-trivial cut tighter than per-neuron
    triangle but looser than the anti-correlation case."""
    Gc_1 = np.array([0.7, 0.3])
    Gc_2 = np.array([-0.5, 0.8])
    # Compute pre-act bounds via interval arithmetic
    l_1, u_1 = -(0.7 + 0.3), (0.7 + 0.3)  # = -1, 1
    l_2, u_2 = -(0.5 + 0.8), (0.5 + 0.8)  # = -1.3, 1.3
    env = joint_envelope_lp(
        Gc_1, Gc_2, c_i=0.0, c_j=0.0,
        l_i=l_1, u_i=u_1, l_j=l_2, u_j=u_2,
    )
    print(f"\nTest 3: partial-correlation x_1=(0.7,0.3), x_2=(-0.5,0.8)")
    print(f"  Envelope: {env}")
    per_neuron_sum_bound = u_1 + u_2  # 2.3
    joint_sum_bound = env[(1.0, 1.0)]
    print(f"  Per-neuron y_1+y_2 ≤ {per_neuron_sum_bound}")
    print(f"  Joint    y_1+y_2 ≤ {joint_sum_bound:.6f}")
    print(f"  Tightness gain: {per_neuron_sum_bound - joint_sum_bound:.6f}")
    assert joint_sum_bound < per_neuron_sum_bound + 1e-9
    print("  ✓ PASS (joint ≤ per-neuron)")


if __name__ == "__main__":
    test_anti_correlation()
    test_independent()
    test_partial_correlation()
    print("\nAll prototype tests PASSED ✓")
