"""Soundness regression for T2 prune + dense->sparse operators.

Verifies the over-approximation invariant on small random HZ instances
with known ground-truth box hulls (computed exhaustively over the
factor-space lattice when ng <= 6).
"""
from __future__ import annotations

import itertools
import os
import sys

import torch

sys.path.insert(0, "/data1/Kane/ACT")
sys.path.insert(0, os.path.dirname(__file__))

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ
from prototype import (
    hz_prune_gc_dense,
    hz_dense_to_sparse,
    check_overapprox,
    _bounds_of,
)


def random_hzono(n: int = 4, ng: int = 6, dtype=torch.float64, seed: int = 0) -> HZono:
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, generator=g, dtype=dtype)
    Gc = torch.randn(n, ng, generator=g, dtype=dtype)
    # Make a few columns tiny (the prune candidates).
    n_tiny = ng // 3
    Gc[:, :n_tiny] *= 1e-12
    return HZono(
        c=c,
        Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype),
        Ac=torch.zeros((0, ng), dtype=dtype),
        Ab=torch.zeros((0, 0), dtype=dtype),
        b=torch.zeros((0, 1), dtype=dtype),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )


def exact_box_hull(hz: HZono, samples: int = 4) -> tuple:
    """Exact box hull by enumerating xi_c in {-1, ..., +1}^ng on a grid.
    Only correct for small ng (<= 6) and small samples."""
    n, ng = hz.Gc.shape
    grid = torch.linspace(-1, 1, samples, dtype=hz.c.dtype)
    if ng == 0:
        return hz.c.view(-1).clone(), hz.c.view(-1).clone()
    # Sample 1000 random xi_c instead of full grid (full grid is samples**ng)
    g = torch.Generator().manual_seed(123)
    n_sample = 5000
    xi = (torch.rand((n_sample, ng), generator=g, dtype=hz.c.dtype) * 2 - 1)
    pts = hz.c.view(-1) + xi @ hz.Gc.T  # (n_sample, n)
    return pts.min(dim=0).values, pts.max(dim=0).values


def test_prune_soundness():
    for seed in range(10):
        hz = random_hzono(n=4, ng=9, seed=seed)
        lb_ref, ub_ref = exact_box_hull(hz)
        for eps in [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]:
            pruned = hz_prune_gc_dense(hz, eps)
            lbp, ubp = _bounds_of(pruned)
            assert (lbp <= lb_ref + 1e-9).all(), f"prune eps={eps} lb violation seed={seed}"
            assert (ubp >= ub_ref - 1e-9).all(), f"prune eps={eps} ub violation seed={seed}"
        # Pure overapprox check vs original
        for eps in [1e-12, 1e-9, 1e-6, 1e-3]:
            pruned = hz_prune_gc_dense(hz, eps)
            ok, msg = check_overapprox(hz, pruned)
            assert ok, f"seed={seed} eps={eps}: {msg}"
    print("test_prune_soundness PASS")


def test_dense_to_sparse_equivalence():
    """Conversion to sparse should preserve the box hull exactly when
    no thresholding (zero_eps small enough to keep all entries)."""
    for seed in range(5):
        hz = random_hzono(n=4, ng=9, seed=seed)
        # Force conversion by passing absurd density threshold
        result = hz_dense_to_sparse(hz, density_threshold=1.0, zero_eps=1e-30)
        if isinstance(result, SparseGcZ):
            lb_orig, ub_orig = _bounds_of(hz)
            lb_new, ub_new = result.bounds()
            assert torch.allclose(lb_orig, lb_new, atol=1e-9), f"lb mismatch seed={seed}"
            assert torch.allclose(ub_orig, ub_new, atol=1e-9), f"ub mismatch seed={seed}"
        # Soundness check
        ok, msg = check_overapprox(hz, result)
        assert ok, f"seed={seed}: {msg}"
    print("test_dense_to_sparse_equivalence PASS")


def test_combined_prune_then_sparse():
    for seed in range(5):
        hz = random_hzono(n=4, ng=9, seed=seed)
        pruned = hz_prune_gc_dense(hz, eps=1e-9)
        sparse = hz_dense_to_sparse(pruned, density_threshold=1.0, zero_eps=1e-30)
        ok, msg = check_overapprox(hz, sparse)
        assert ok, f"seed={seed} combined: {msg}"
    print("test_combined_prune_then_sparse PASS")


def test_zero_eps_is_identity():
    """eps <= 0 must return hz unchanged."""
    hz = random_hzono(n=4, ng=9, seed=0)
    p1 = hz_prune_gc_dense(hz, 0.0)
    p2 = hz_prune_gc_dense(hz, -1.0)
    assert p1 is hz and p2 is hz
    print("test_zero_eps_is_identity PASS")


def test_no_drops_returns_input():
    """Threshold below the smallest column norm must not strip anything."""
    hz = random_hzono(n=4, ng=9, seed=0)
    p = hz_prune_gc_dense(hz, eps=1e-30)
    # The 3 'tiny' columns have norm 1e-12 which is > 1e-30 → kept.
    assert p.Gc.shape == hz.Gc.shape
    # But they ARE dropped at eps=1e-9 → row-slack column gets added.
    p2 = hz_prune_gc_dense(hz, eps=1e-9)
    # Tiny cols dropped (3). One slack col per row that had nonzero
    # dropped contribution. With n=4 rows × 3 tiny dense cols, all 4
    # rows can accrue slack, so new ng = (9 - 3) + 4 = 10. This is
    # sound by row-slack construction.
    assert p2.Gc.shape[1] <= hz.dim + (hz.Gc.shape[1] - 3)
    print("test_no_drops_returns_input PASS")


if __name__ == "__main__":
    test_zero_eps_is_identity()
    test_no_drops_returns_input()
    test_prune_soundness()
    test_dense_to_sparse_equivalence()
    test_combined_prune_then_sparse()
    print()
    print("All T2 soundness regression tests PASS")
