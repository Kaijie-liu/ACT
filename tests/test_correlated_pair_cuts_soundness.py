#===- tests/test_correlated_pair_cuts_soundness.py - correlated pair-ReLU ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===-----------------------------------------------------------------------===#
#
# Purpose:
#   Pin the CORRELATED pair-ReLU joint hull cut implementation against
#   three independent soundness/utility risks identified by the user
#   audit (2026-05-31):
#
#   1) Validity vs. true ReLU graph — for a synthetic 2-neuron HZ
#      whose pre-activations (pa, pb) are forced to lie on a known 2D
#      polytope (NOT the box `[la,ua]×[lb,ub]`), every concrete
#      `(pa, pb, max(0,pa), max(0,pb))` produced by a real input MUST
#      satisfy every generated correlated cut. Audit name: "cut 加入
#      LP 后不排除真实点".
#
#   2) Tightness vs. independent box — for the same HZ, the
#      independent-box facets MUST NOT tighten the per-pair LP
#      relaxation (this is the redundancy the audit established
#      mathematically), and the correlated-cut facets MUST strictly
#      tighten it. Audit name: "independent-box cuts 与单 ReLU
#      triangle 等价/冗余的测试" and "correlated cuts 能降低 LP
#      optimum".
#
#   3) Numerical sanity — facet equations must be non-trivial (at least
#      one of c_pa / c_pb / c_postA / c_postB nonzero) and rhs finite.
#
# Run via:
#   bash /data1/Kane/ACT/tests/run_soundness_tests.sh
# or directly:
#   /data1/Kane/miniconda3/envs/act-py312/bin/python tests/test_correlated_pair_cuts_soundness.py
#
# All tests must pass before the `ACT_HZ_CORR_PAIR_CUTS` knob is wired
# into any default profile.
#===-----------------------------------------------------------------------===#
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch

if "/data1/Kane/ACT" not in sys.path:
    sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.tf_mlp import (
    _build_triangle_corr_pair_cuts,
    _correlated_pair_hull_facets,
    _pair_hull_facets_9vertex,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _make_synthetic_correlated_hz() -> HZono:
    """Build a synthetic 2-neuron HZ whose pre-activations (pa, pb) are
    linearly CORRELATED but not 1D-degenerate.

    Setup:
      xi_c = (s, t) with s, t in [-1, 1]
      pa =  +0.8 * s + 0.1 * t        => pa in [-0.9, +0.9]
      pb =  -0.8 * s + 0.1 * t        => pb in [-0.9, +0.9]

    Independent box of (pa, pb): [-0.9, 0.9] × [-0.9, 0.9] (corners).
    True joint reachable set: 2D PARALLELOGRAM
      (pa + pb) = 0.2 * t ∈ [-0.2, +0.2]
      (pa - pb) = 1.6 * s ∈ [-1.6, +1.6]
    So the point (pa, pb) = (+0.9, +0.9) is IN the box but NOT reachable
    (would need pa + pb = +1.8 > 0.2 max).

    Cut generators using the independent box produce facets that admit
    (+0.9, +0.9). Cut generators using the support polytope of the
    actual parallelogram produce facets that EXCLUDE it.
    """
    Gc = torch.tensor([[+0.8, +0.1], [-0.8, +0.1]], dtype=torch.float64)
    Gb = torch.zeros((2, 0), dtype=torch.float64)
    c = torch.zeros((2, 1), dtype=torch.float64)
    Ac = torch.zeros((0, 2), dtype=torch.float64)
    Ab = torch.zeros((0, 0), dtype=torch.float64)
    b = torch.zeros((0, 1), dtype=torch.float64)
    eq_mask = torch.zeros((0,), dtype=torch.bool)
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=eq_mask)


def _enum_real_points(n_samples: int = 9):
    """Enumerate concrete (pa, pb, post_a, post_b) over an (s, t) grid.
    These are inputs the verifier MUST not exclude."""
    pts = []
    for s in np.linspace(-1.0, 1.0, n_samples):
        for t in np.linspace(-1.0, 1.0, n_samples):
            pa = +0.8 * s + 0.1 * t
            pb = -0.8 * s + 0.1 * t
            post_a = max(0.0, pa)
            post_b = max(0.0, pb)
            pts.append((pa, pb, post_a, post_b))
    return pts


def _facet_residual(facet, pt):
    """Return c_pa*pa + c_pb*pb + c_postA*post_a + c_postB*post_b - rhs.
    Soundness ⇒ residual <= numerical eps for every real point."""
    pa, pb, post_a, post_b = pt
    return (
        facet["c_pa"] * pa
        + facet["c_pb"] * pb
        + facet["c_postA"] * post_a
        + facet["c_postB"] * post_b
        - facet["rhs"]
    )


# ─── Tests ─────────────────────────────────────────────────────────────────

def test_correlated_cuts_admit_every_real_point():
    """For the synthetic correlated HZ, every concrete real-input point
    must satisfy every generated correlated facet (residual <= eps)."""
    hz = _make_synthetic_correlated_hz()
    facets = _correlated_pair_hull_facets(
        hz_in=hz, a_idx=0, b_idx=1,
        alpha_a=-0.9, beta_a=+0.9,
        alpha_b=-0.9, beta_b=+0.9,
        n_dirs=16, lp_timeout_s=2.0,
    )
    assert len(facets) > 0, "no facets returned for non-degenerate input"
    real_pts = _enum_real_points(n_samples=65)
    eps = 5e-3  # LP-derived support polygon has finite numerical slack
    for i, f in enumerate(facets):
        for j, pt in enumerate(real_pts):
            r = _facet_residual(f, pt)
            assert r <= eps, (
                f"facet[{i}]={f} EXCLUDES real point[{j}]={pt} "
                f"(residual={r:.3e})"
            )


def test_independent_box_cuts_admit_every_real_point():
    """Independent-box facets must also admit every real point (they
    are sound, just redundant — soundness is the orthogonal property)."""
    facets = _pair_hull_facets_9vertex(la=-0.9, ua=+0.9, lb_=-0.9, ub_=+0.9)
    assert len(facets) > 0
    real_pts = _enum_real_points(n_samples=9)
    eps = 5e-3  # LP-derived support polygon has finite numerical slack
    for i, f in enumerate(facets):
        for pt in real_pts:
            assert _facet_residual(f, pt) <= eps, (
                f"box facet[{i}]={f} wrongly excludes real point {pt}"
            )


def test_correlated_facets_use_correlation_dimension():
    """For the perfectly-anti-correlated HZ (pa + pb = 0 on reachable
    set), the correlated cut generator MUST produce at least one facet
    whose coefficients on (pa, pb) reflect the correlation —
    specifically a facet that excludes (pa, pb) = (+0.8, +0.8) which is
    in the independent box but NOT in the correlated reachable set."""
    hz = _make_synthetic_correlated_hz()
    facets = _correlated_pair_hull_facets(
        hz_in=hz, a_idx=0, b_idx=1,
        alpha_a=-0.9, beta_a=+0.9,
        alpha_b=-0.9, beta_b=+0.9,
        n_dirs=16, lp_timeout_s=2.0,
    )
    # Phantom (+0.9, +0.9, +0.9, +0.9) is in the independent box
    # [-0.9, 0.9]^2 but NOT reachable (pa + pb = 1.8 > 0.2). At least
    # one correlated facet must exclude it.
    phantom = (+0.9, +0.9, +0.9, +0.9)
    eps = 1e-3
    excluded = False
    for f in facets:
        if _facet_residual(f, phantom) > eps:
            excluded = True
            break
    assert excluded, (
        "no correlated facet excludes the box-only phantom (+0.8,+0.8); "
        "correlated cuts degenerated to independent-box hull"
    )


def test_independent_box_does_not_exclude_phantom():
    """Independent-box facets must NOT exclude the phantom point — this
    is exactly the redundancy mode the user audit established
    mathematically."""
    facets = _pair_hull_facets_9vertex(la=-0.9, ua=+0.9, lb_=-0.9, ub_=+0.9)
    phantom = (+0.9, +0.9, +0.9, +0.9)
    eps = 5e-3  # LP-derived support polygon has finite numerical slack
    for f in facets:
        assert _facet_residual(f, phantom) <= eps, (
            f"independent-box facet wrongly excluded phantom = {f}"
        )


def test_facet_dictionary_well_formed():
    """All returned facets must have non-trivial coefficient mass and
    finite rhs."""
    hz = _make_synthetic_correlated_hz()
    facets = _correlated_pair_hull_facets(
        hz_in=hz, a_idx=0, b_idx=1,
        alpha_a=-0.9, beta_a=+0.9,
        alpha_b=-0.9, beta_b=+0.9,
        n_dirs=16, lp_timeout_s=2.0,
    )
    assert facets
    for i, f in enumerate(facets):
        keys = {"c_pa", "c_pb", "c_postA", "c_postB", "rhs"}
        assert set(f.keys()) == keys, f"facet[{i}] keys mismatch: {f.keys()}"
        mag = abs(f["c_pa"]) + abs(f["c_pb"]) + abs(f["c_postA"]) + abs(f["c_postB"])
        assert mag > 1e-9, f"facet[{i}] all-zero coefficients: {f}"
        assert np.isfinite(f["rhs"]), f"facet[{i}] non-finite rhs: {f}"


def test_triangle_corr_cut_rows_admit_real_points():
    """The triangle-path row builder must preserve every true ReLU point.

    This tests the substitution from a 4D facet
      (pre_a, pre_b, post_a, post_b)
    into triangle factor coordinates
      post = lam*pre + mu + mu*eps.

    The benchmark path for dense-conv CIFAR uses triangle layers before the
    last eq_lagr layers, so this is the specific soundness guard for the
    P3 triangle correlated-cut hook.
    """
    hz = _make_synthetic_correlated_hz()
    unstable_idx = torch.tensor([0, 1], dtype=torch.long)
    alpha = torch.tensor([-0.9, -0.9], dtype=torch.float64)
    beta = torch.tensor([+0.9, +0.9], dtype=torch.float64)
    lam = beta / (beta - alpha)
    mu = -alpha * beta / (2.0 * (beta - alpha))
    col_eps = torch.tensor([2, 3], dtype=torch.long)

    Ac, Ab, b = _build_triangle_corr_pair_cuts(
        hz_in=hz,
        unstable_idx=unstable_idx,
        alpha=alpha,
        beta=beta,
        lam=lam,
        mu=mu,
        col_eps=col_eps,
        ng_new=4,
        nb_new=0,
        max_pairs=1,
        n_dirs=16,
        lp_timeout_s=2.0,
        layer_counter=-1,
    )
    assert Ac.shape[0] > 0, "triangle correlated row builder emitted no rows"
    assert Ab.shape == (Ac.shape[0], 0)
    assert b.shape == (Ac.shape[0], 1)

    # Concrete real points parameterized by old factors (s, t). Compute the
    # triangle eps values that exactly realize post=max(0, pre). Every row
    # must admit the resulting factor vector.
    rows = Ac.detach().cpu().numpy()
    rhs = b[:, 0].detach().cpu().numpy()
    lam_np = lam.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    eps = 5e-3
    for s in np.linspace(-1.0, 1.0, 41):
        for t in np.linspace(-1.0, 1.0, 41):
            pa = +0.8 * s + 0.1 * t
            pb = -0.8 * s + 0.1 * t
            post_a = max(0.0, pa)
            post_b = max(0.0, pb)
            eps_a = (post_a - lam_np[0] * pa - mu_np[0]) / mu_np[0]
            eps_b = (post_b - lam_np[1] * pb - mu_np[1]) / mu_np[1]
            xfac = np.array([s, t, eps_a, eps_b], dtype=np.float64)
            residual = rows @ xfac - rhs
            assert float(np.max(residual)) <= eps, (
                "triangle correlated cut excludes a real ReLU point: "
                f"s={s}, t={t}, pre=({pa},{pb}), eps=({eps_a},{eps_b}), "
                f"max_residual={float(np.max(residual)):.3e}"
            )


def test_correlated_cuts_handle_degenerate_bounds():
    """For non-unstable pairs (one of alpha>=0 or beta<=0), correlated
    cut generator must return [] rather than crash or return phantom."""
    hz = _make_synthetic_correlated_hz()
    # Force one neuron to be 'always active' (alpha >= 0)
    facets = _correlated_pair_hull_facets(
        hz_in=hz, a_idx=0, b_idx=1,
        alpha_a=+0.1, beta_a=+0.9,
        alpha_b=-0.8, beta_b=+0.8,
        n_dirs=8, lp_timeout_s=1.0,
    )
    assert facets == [], (
        "expected empty facet list for non-jointly-unstable pair; "
        f"got {len(facets)} facets"
    )


# ─── Runner ────────────────────────────────────────────────────────────────

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nResult: {len(fns) - fails}/{len(fns)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(0 if _run_all() == 0 else 1)
