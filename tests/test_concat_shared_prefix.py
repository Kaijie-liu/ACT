"""Soundness + tightness tests for the env-gated shared-prefix concat
fast path in `act/back_end/hybridz_tf/hz_routing.py`.

The shared-prefix branch is opt-in via `ACT_HZ_CONCAT_SHARED_PREFIX=1`.
It MUST satisfy:

  1. Sound vs block-diag: for every concrete point admitted by the
     shared-prefix output, block-diag would also admit the same
     concrete output.

  2. Strictly-no-looser than block-diag: the LP min/max box of every
     output coordinate over the shared-prefix HZ should be ≤ the LP
     min/max box over the block-diag HZ.

  3. Same-HZ pattern: when two slices of the SAME HZ are concat'd
     back, the shared continuous generators must NOT be duplicated;
     the output's `ng` must equal the source HZ's `ng` (not 2× as
     block-diag would give).

  4. Fallback safety: if `_base_nc > 0` but the shared constraint
     rows don't match across inputs, the fast path must return None
     and the caller falls back to block-diag (not produce an
     incorrect result).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.hz_routing import (
    hz_concat, _try_shared_prefix_concat,
)


def _box_bounds(hz: HZono):
    """Cheap UNC box bounds (no constraint solve)."""
    c = hz.c.view(-1)
    half = hz.Gc.abs().sum(dim=1) + hz.Gb.abs().sum(dim=1)
    return (c - half).cpu().numpy(), (c + half).cpu().numpy()


def _make_parent_hz(n=4, ng=3, nb=0, nc=0, seed=0):
    """Create a small parent HZ to slice from. Synthetically assigns
    a fresh `_base_root_id` so slices share it (mirroring what
    `hz_from_bounds` does in production)."""
    from act.back_end.solver.solver_hz import _assign_fresh_root_id
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 1, generator=g, dtype=torch.float64)
    Gc = torch.randn(n, ng, generator=g, dtype=torch.float64)
    Gb = torch.zeros((n, nb), dtype=torch.float64)
    Ac = torch.zeros((nc, ng), dtype=torch.float64)
    Ab = torch.zeros((nc, nb), dtype=torch.float64)
    b = torch.zeros((nc, 1), dtype=torch.float64)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)
    _assign_fresh_root_id(hz)
    return hz


def _slice_hz(hz: HZono, idx):
    """Slice a HZ along the output (row) axis. Preserves
    `_base_ng/_nb/_nc` AND inherits the parent's `_base_root_id`
    (mirrors what `_propagate_base` would do in production)."""
    sub = HZono(
        c=hz.c[idx], Gc=hz.Gc[idx], Gb=hz.Gb[idx],
        Ac=hz.Ac, Ab=hz.Ab, b=hz.b,
        eq_mask=hz.eq_mask,
    )
    root = getattr(hz, "_base_root_id", None)
    if root is not None:
        object.__setattr__(sub, "_base_root_id", int(root))
    return sub


# ─── Test 1: same-HZ slice/concat does NOT duplicate gens ─────────────────


def test_same_hz_concat_preserves_shared_generators():
    parent = _make_parent_hz(n=6, ng=4)
    a = _slice_hz(parent, slice(0, 3))
    b = _slice_hz(parent, slice(3, 6))
    assert a._base_ng == parent._base_ng == 4
    assert b._base_ng == parent._base_ng == 4

    merged = _try_shared_prefix_concat([a, b])
    assert merged is not None, "shared-prefix path should succeed for sliced same-HZ"
    # Output's ng must equal source's ng (NOT 2× as block-diag gives).
    assert merged.ng == 4, (
        f"shared-prefix concat should keep ng={4}, got {merged.ng}"
    )
    # Output's dim must equal original n.
    assert merged.dim == 6
    # _base_ng should reflect the shared prefix.
    assert merged._base_ng == 4


# ─── Test 2: shared-prefix concat is sound (subset of block-diag) ─────────


def test_shared_prefix_is_sound_vs_block_diag():
    """For every concrete xi, the shared-prefix output value must equal
    what we'd get from concat'ing the inputs' concrete outputs."""
    parent = _make_parent_hz(n=4, ng=3)
    a = _slice_hz(parent, slice(0, 2))
    b = _slice_hz(parent, slice(2, 4))

    merged = _try_shared_prefix_concat([a, b])
    assert merged is not None

    # Sample concrete xi (uniform in [-1, 1]).
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi = torch.tensor(rng.uniform(-1, 1, size=merged.ng).reshape(-1, 1),
                          dtype=torch.float64)
        # Output value under the merged HZ at this xi.
        y_merged = (merged.c + merged.Gc @ xi).view(-1).numpy()

        # Reference: parent value at the same xi (shared columns)
        y_parent = (parent.c + parent.Gc @ xi).view(-1).numpy()

        # The merged output's first 2 rows should match parent rows 0..1,
        # next 2 rows should match parent rows 2..3 (under shared-prefix
        # semantics, since both inputs reference the SAME xi factors).
        assert np.allclose(y_merged, y_parent, atol=1e-12), (
            f"shared-prefix concat doesn't match parent at xi: "
            f"merged={y_merged} parent={y_parent}"
        )


# ─── Test 3: shared-prefix concat strictly tightens block-diag box ────────


def test_shared_prefix_reduces_generator_count():
    """Shared-prefix concat must yield strictly fewer generators than
    block-diag concat when inputs share generators.

    (The UNC box bound stays the same because UNC ignores constraints
    AND treats variables as independent — the row magnitudes sum the
    same way regardless of column layout. The real tightening is in
    (a) the generator COUNT and (b) the LP-constrained polytope shape,
    not in the cheap interval bound. This test asserts (a).)"""
    parent = _make_parent_hz(n=4, ng=3)
    a = _slice_hz(parent, slice(0, 2))
    b = _slice_hz(parent, slice(2, 4))

    os.environ.pop("ACT_HZ_CONCAT_SHARED_PREFIX", None)
    blkdiag = hz_concat([a, b])

    os.environ["ACT_HZ_CONCAT_SHARED_PREFIX"] = "1"
    try:
        shared = hz_concat([a, b])
    finally:
        os.environ.pop("ACT_HZ_CONCAT_SHARED_PREFIX", None)

    assert blkdiag.ng == a.ng + b.ng, (
        f"block-diag ng should be sum of input ngs: "
        f"got {blkdiag.ng}, expected {a.ng + b.ng}"
    )
    assert shared.ng < blkdiag.ng, (
        f"shared-prefix should yield fewer gens than block-diag: "
        f"shared.ng={shared.ng}, blkdiag.ng={blkdiag.ng}"
    )
    # When all of the parent's gens are shared, shared-prefix ng equals
    # the parent's ng.
    assert shared.ng == parent.ng, (
        f"shared-prefix ng should equal parent ng={parent.ng}, "
        f"got {shared.ng}"
    )


# ─── Test 4: fallback when shared constraint rows differ ──────────────────


def test_shared_prefix_falls_back_when_shared_rows_differ():
    """If _base_nc reports shared rows that aren't actually identical,
    the fast path must return None (fallback to block-diag).

    This simulates a bug in upstream constraint propagation where two
    HZ from the same root drift apart in their shared constraint
    rows. The numerical sanity check inside `_try_shared_prefix_concat`
    must catch this even when root_id matches."""
    from act.back_end.solver.solver_hz import _assign_fresh_root_id
    parent = _make_parent_hz(n=4, ng=3, nc=1)
    # Construct a parent with a real constraint row.
    Ac = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float64)
    Ab = torch.zeros((1, 0), dtype=torch.float64)
    b = torch.tensor([[0.3]], dtype=torch.float64)
    parent2 = HZono(c=parent.c, Gc=parent.Gc, Gb=parent.Gb,
                    Ac=Ac, Ab=Ab, b=b)
    object.__setattr__(parent2, "_base_root_id",
                       int(getattr(parent, "_base_root_id")))

    a = _slice_hz(parent2, slice(0, 2))
    # Manually corrupt b's shared constraint to differ from a's, but
    # keep root_id matching to exercise the numerical guard.
    b_corrupt = HZono(c=parent2.c[2:4], Gc=parent2.Gc[2:4], Gb=parent2.Gb[2:4],
                      Ac=Ac, Ab=Ab,
                      b=torch.tensor([[0.99]], dtype=torch.float64))
    object.__setattr__(b_corrupt, "_base_root_id",
                       int(getattr(parent2, "_base_root_id")))

    merged = _try_shared_prefix_concat([a, b_corrupt])
    assert merged is None, (
        "shared-prefix should fall back when shared rows mismatch"
    )


# ─── Adversarial test: independent roots must NOT be merged ─────────────
#
# This is the soundness boundary case raised in the 2026-06-01 review:
# the current shared-prefix detection only checks `_base_ng` counts, not
# root identity. Two HZ from genuinely independent roots that happen to
# both have `_base_ng = K` would be incorrectly merged — the merged HZ
# would couple `xi_c[:K]` between blocks, EXCLUDING valid concrete
# points (an under-approximation, i.e., UNSOUND).
#
# Concrete construction (n=1 dim per input, ng=1, no constraints):
#   hz_a: y_a = xi_a, xi_a in [-1, +1]
#   hz_b: y_b = xi_b, xi_b in [-1, +1]   ← independent xi
# True concat reachable set: (y_a, y_b) in [-1, 1] × [-1, 1] (the unit
# square).
#
# If we manually set `_base_ng = 1` on BOTH (simulating either a bug in
# upstream `_propagate_base` or a coincidence across roots), the current
# `_try_shared_prefix_concat` would build a single shared column block
# `[[1]; [1]]` with the same shared `xi_shared` for both rows. The
# resulting reachable set is `(xi_shared, xi_shared) in {(t, t): t in
# [-1, 1]}` — the diagonal of the square, missing 99% of valid points.
#
# This test:
#   1. Verifies the adversarial input would in fact produce an
#      under-approximation (i.e., block-diag's reachable set is strictly
#      larger).
#   2. Asserts that the implementation REJECTS the merge (returns None)
#      whenever a root-identity guard cannot confirm shared lineage.


def test_independent_roots_with_same_base_ng_must_not_merge():
    """Adversarial: two independent HZ with coincidentally equal
    `_base_ng` must NOT be merged under shared-prefix layout, because
    the implementation cannot distinguish 'same root, same factors'
    from 'different roots, same _base_ng count'. Merging would be
    unsound."""
    # Two genuinely independent HZ, each with ng=1, dim=1, no constraints
    hz_a = HZono(
        c=torch.zeros((1, 1), dtype=torch.float64),
        Gc=torch.ones((1, 1), dtype=torch.float64),
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 1), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
    )
    hz_b = HZono(
        c=torch.zeros((1, 1), dtype=torch.float64),
        Gc=torch.ones((1, 1), dtype=torch.float64),
        Gb=torch.zeros((1, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 1), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
    )
    # SIMULATE COINCIDENTAL _base_ng=1 on independent roots
    object.__setattr__(hz_a, "_base_ng", 1)
    object.__setattr__(hz_b, "_base_ng", 1)
    object.__setattr__(hz_a, "_base_nb", 0)
    object.__setattr__(hz_b, "_base_nb", 0)
    object.__setattr__(hz_a, "_base_nc", 0)
    object.__setattr__(hz_b, "_base_nc", 0)

    # First: confirm the adversarial case really IS adversarial
    # (block-diag would admit the full square, shared-prefix-merged
    # would only admit the diagonal).
    block_diag = hz_concat([hz_a, hz_b])
    assert block_diag.ng == 2, (
        f"block-diag of two ng=1 inputs should give ng=2 (independent "
        f"factor per input), got {block_diag.ng}"
    )
    # Sample (y_a=1, y_b=-1): valid for block-diag, invalid for any
    # shared-prefix merge (would require xi_shared = 1 AND xi_shared
    # = -1 simultaneously).
    # We can't easily concretize the LP membership without solving an
    # LP, but the ng count is a sufficient witness: a 2-ng box and a
    # 1-ng segment are clearly different sets.

    # The TEST: implementation MUST refuse to merge these inputs
    # because no root-identity evidence exists.
    merged = _try_shared_prefix_concat([hz_a, hz_b])
    if merged is not None and merged.ng < block_diag.ng:
        # The merge happened AND it reduced ng count → UNSOUND
        # under-approximation. This is the bug we are guarding against.
        raise AssertionError(
            f"UNSOUND: shared-prefix concat merged two independent-root "
            f"HZ inputs with coincidentally equal _base_ng=1. "
            f"block_diag ng={block_diag.ng} but merged ng={merged.ng}. "
            f"Implementation needs a root-identity guard (e.g. "
            f"`_base_root_id` metadata) before this default flips to ON."
        )
    assert merged is None, (
        f"shared-prefix concat must return None for independent-root "
        f"inputs, got merged ng={merged.ng if merged else None}"
    )


# ─── Test 5: empty shared prefix returns None (no-op signal) ──────────────


def test_empty_shared_prefix_returns_none():
    """If _base_ng = _base_nb = _base_nc = 0 across all inputs (no
    shared ancestor), fast path returns None and block-diag is used."""
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    hz_a = HZono(
        c=torch.randn(3, 1, generator=g1, dtype=torch.float64),
        Gc=torch.randn(3, 2, generator=g1, dtype=torch.float64),
        Gb=torch.zeros((3, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
    )
    object.__setattr__(hz_a, "_base_ng", 0)
    object.__setattr__(hz_a, "_base_nb", 0)
    object.__setattr__(hz_a, "_base_nc", 0)
    hz_b = HZono(
        c=torch.randn(3, 1, generator=g2, dtype=torch.float64),
        Gc=torch.randn(3, 2, generator=g2, dtype=torch.float64),
        Gb=torch.zeros((3, 0), dtype=torch.float64),
        Ac=torch.zeros((0, 2), dtype=torch.float64),
        Ab=torch.zeros((0, 0), dtype=torch.float64),
        b=torch.zeros((0, 1), dtype=torch.float64),
    )
    object.__setattr__(hz_b, "_base_ng", 0)
    object.__setattr__(hz_b, "_base_nb", 0)
    object.__setattr__(hz_b, "_base_nc", 0)

    merged = _try_shared_prefix_concat([hz_a, hz_b])
    assert merged is None, "empty shared prefix should signal no-op"


if __name__ == "__main__":
    tests = [
        test_same_hz_concat_preserves_shared_generators,
        test_shared_prefix_is_sound_vs_block_diag,
        test_shared_prefix_reduces_generator_count,
        test_shared_prefix_falls_back_when_shared_rows_differ,
        test_independent_roots_with_same_base_ng_must_not_merge,
        test_empty_shared_prefix_returns_none,
    ]
    n_pass = 0
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\nResult: {n_pass}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
