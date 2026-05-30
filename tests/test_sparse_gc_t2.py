#===- tests/test_sparse_gc_t2.py - T2 prune + dense->sparse soundness ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Soundness regression for act_hz_prune_gc_dense and
#   act_hz_dense_to_sparse in act.back_end.hybridz_tf.sparse_gc_t2.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import os

import torch

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ
from act.back_end.hybridz_tf.sparse_gc_t2 import (
    act_hz_prune_gc_dense,
    act_hz_dense_to_sparse,
    act_maybe_compact_hz,
)


def _random_hzono(n=4, ng=9, dtype=torch.float64, seed=0, tiny_frac=1/3) -> HZono:
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, generator=g, dtype=dtype)
    Gc = torch.randn(n, ng, generator=g, dtype=dtype)
    n_tiny = int(ng * tiny_frac)
    if n_tiny > 0:
        Gc[:, :n_tiny] *= 1e-12
    return HZono(
        c=c, Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype),
        Ac=torch.zeros((0, ng), dtype=dtype),
        Ab=torch.zeros((0, 0), dtype=dtype),
        b=torch.zeros((0, 1), dtype=dtype),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )


def _bounds_of(z):
    if isinstance(z, HZono):
        Gc = z.Gc
        rad = Gc.abs().sum(dim=1) if Gc.numel() else torch.zeros(z.dim, dtype=z.c.dtype, device=z.c.device)
        if z.Gb.numel():
            rad = rad + z.Gb.abs().sum(dim=1)
        c = z.c.view(-1)
        return c - rad, c + rad
    if isinstance(z, SparseGcZ):
        return z.bounds()
    raise TypeError(type(z))


def _check_overapprox(orig, transformed):
    lb0, ub0 = _bounds_of(orig)
    lb1, ub1 = _bounds_of(transformed)
    assert (lb1 <= lb0 + 1e-9).all(), f"prune lb violation: max={(lb1 - lb0).max().item():.3e}"
    assert (ub1 >= ub0 - 1e-9).all(), f"prune ub violation: max={(ub0 - ub1).max().item():.3e}"


def test_prune_eps_zero_is_identity():
    hz = _random_hzono()
    assert act_hz_prune_gc_dense(hz, 0.0) is hz
    assert act_hz_prune_gc_dense(hz, -1.0) is hz


def test_prune_soundness_random():
    for seed in range(8):
        hz = _random_hzono(n=4, ng=9, seed=seed)
        for eps in [1e-12, 1e-9, 1e-6, 1e-3]:
            p = act_hz_prune_gc_dense(hz, eps)
            _check_overapprox(hz, p)


def test_dense_to_sparse_preserves_bounds():
    for seed in range(5):
        hz = _random_hzono(n=4, ng=9, seed=seed)
        result = act_hz_dense_to_sparse(hz, density_threshold=1.0, zero_eps=1e-30)
        if isinstance(result, SparseGcZ):
            lb0, ub0 = _bounds_of(hz)
            lb1, ub1 = _bounds_of(result)
            assert torch.allclose(lb0, lb1, atol=1e-9)
            assert torch.allclose(ub0, ub1, atol=1e-9)


def test_dense_to_sparse_nb_gt_zero_skipped():
    dtype = torch.float64
    n = 3; ng = 4; nb = 2
    hz = HZono(
        c=torch.zeros(n, 1, dtype=dtype),
        Gc=torch.zeros((n, ng), dtype=dtype),
        Gb=torch.ones((n, nb), dtype=dtype),  # nb > 0
        Ac=torch.zeros((0, ng), dtype=dtype),
        Ab=torch.zeros((0, nb), dtype=dtype),
        b=torch.zeros((0, 1), dtype=dtype),
        eq_mask=torch.zeros(0, dtype=torch.bool),
    )
    out = act_hz_dense_to_sparse(hz, density_threshold=1.0, zero_eps=1e-30)
    assert out is hz, "nb>0 must skip conversion"


def test_compact_default_off():
    """With knobs unset, act_maybe_compact_hz must return input unchanged."""
    # Ensure knobs are off in this test process.
    for k in ["ACT_HZ_PRUNE_GC", "ACT_HZ_DENSE_TO_SPARSE"]:
        os.environ.pop(k, None)
    hz = _random_hzono()
    assert act_maybe_compact_hz(hz) is hz


def test_compact_knobs_on():
    os.environ["ACT_HZ_PRUNE_GC"] = "1"
    os.environ["ACT_HZ_DENSE_TO_SPARSE"] = "1"
    os.environ["ACT_HZ_PRUNE_GC_THRESH"] = "1e-9"
    try:
        hz = _random_hzono()
        out = act_maybe_compact_hz(hz)
        _check_overapprox(hz, out)
    finally:
        for k in ["ACT_HZ_PRUNE_GC", "ACT_HZ_DENSE_TO_SPARSE", "ACT_HZ_PRUNE_GC_THRESH"]:
            os.environ.pop(k, None)


if __name__ == "__main__":
    test_prune_eps_zero_is_identity()
    test_prune_soundness_random()
    test_dense_to_sparse_preserves_bounds()
    test_dense_to_sparse_nb_gt_zero_skipped()
    test_compact_default_off()
    test_compact_knobs_on()
    print("All T2 sparse_gc_t2 tests PASS")
