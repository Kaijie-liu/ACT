"""Ideal single-unit cuts over the latent box (THEORY.md Section 12, Lemma 5).

In the projection-aligned HZ every pre-activation is affine in the latent vector,
x_k = cx_k + a_k^T w with w in [-1,1]^K (input factors and earlier eta factors).  The
ideal MILP formulation of y = ReLU(a^T w + c) over the box (Anderson et al. 2020)
consists of, for every subset I of latent coordinates,
    y <= sum_{i in I} (a_i w_i + |a_i| (1 - d)) + (c + sum_{i not in I} |a_i|) d .
Each cut is valid for every exact point (w, ReLU(x), d=[x>=0]) with w in the box,
hence for gamma (whose w also satisfy the LP rows).  Rounding: x and y carry radii
e_x, e_y; adding 2 e_x + e_y to the right-hand side keeps the cut valid for the
real-arithmetic values (the perturbation of x is an extra input with box [-e_x, e_x]).
Separation at (w*, d*): I = {i : a_i w*_i < |a_i| (2 d* - 1)}.
Cuts are sound by construction; they only remove LP points that no exact point
attains.  They do not delete, fix or relax any binary.
"""
import numpy as np
import scipy.sparse as sp


def separate(units, wstar, dstar, ystar, tol=1e-6):
    """units: list of dicts with a (K,), cx, gy (K,), cy, ex, ey, col (d column).
    Returns list of (row over [w | d-columns dict], rhs) for violated cuts."""
    cuts = []
    for un, ds, ys in zip(units, dstar, ystar):
        a = un["a"]; aa = np.abs(a)
        inI = a * wstar < aa * (2 * ds - 1)
        rhs_val = (a[inI] @ wstar[inI] + aa[inI].sum() * (1 - ds) + (un["cx"] + aa[~inI].sum()) * ds)
        if ys > rhs_val + tol:
            coef_w = un["gy"].copy(); coef_w[inI] -= a[inI]
            coef_d = aa[inI].sum() - un["cx"] - aa[~inI].sum()
            rhs = aa[inI].sum() - un["cy"] + 2 * un["ex"] + un["ey"]
            cuts.append((coef_w, un["col"], coef_d, rhs, ys - rhs_val))
    return cuts
