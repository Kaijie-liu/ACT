"""PRUNE soundness regression: metadata-direct containment.

Per dc_hz_phase_a_plan.md §1.4 and EXECUTION §1.1: PRUNE must over-approximate
the original HZ for ANY value of d_L. Single-column r_tail[:, None] is unsound
and MUST be rejected by these tests (it would fail to contain points where
dropped generators move along independent coordinates).

Containment is checked by directly constructing a feasible coefficient in the
pruned representation from the original xi, rather than running an LP. This is
faster and catches the single-column bug deterministically.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.sc_hz.prune import prune  # noqa: E402


# ─── Helpers ──────────────────────────────────────────────────────


def make_orthogonal_to_max_norm_col(G: np.ndarray,
                                    seed: int = 20260607) -> np.ndarray:
    """Build a direction orthogonal to the largest-norm column of G.

    Useful as an adversarial d_L: the relevance score is 0 on the
    largest-norm column, so PRUNE will discard high-norm directions — a
    stress test for the tail merge.
    """
    rng = np.random.default_rng(seed)
    col_norms = np.linalg.norm(G, axis=0)
    m = int(np.argmax(col_norms))
    g = G[:, m]
    v = rng.normal(size=G.shape[0])
    return v - (v @ g) / max(g @ g, 1e-12) * g


def is_contained_by_metadata(p_orig: np.ndarray, state, xi_orig: np.ndarray,
                              atol: float = 1e-9) -> bool:
    """Witness-based containment check.

    Given the ORIGINAL point `p_orig = c + G @ xi_orig` and the pruned
    state, directly construct (xi_keep, xi_tail) that should reproduce
    p_orig in the pruned representation. Return True iff this witness is
    feasible (i.e. xi_keep ∈ [-1, 1]^K and xi_tail ∈ [-1, 1]^n) AND it
    reconstructs p_orig within atol.

    This is the "metadata-direct" containment witness:
      - xi_keep_i = xi_orig[keep_indices[i]]      (use the original ξ on kept cols)
      - xi_tail_i = clip( dropped_i / tail_radius_i , -1, 1 )

    where dropped_i = Σ_{j ∈ drop} G[i, j] · xi_orig[j] is the leftover
    contribution from the dropped generators in coordinate i.

    If single-column r_tail[:, None] were used (unsound), this witness
    construction would fail because a single xi_tail scalar cannot
    independently produce the right value in every row simultaneously.
    """
    c = state.c
    G_kept = state.G_kept
    tail_radius = state.tail_radius
    keep = np.asarray(state.metadata["keep"], dtype=int)
    drop = np.asarray(state.metadata["drop"], dtype=int)
    n = c.shape[0]

    # xi_keep comes directly from xi_orig on the kept indices
    xi_keep = xi_orig[keep]
    if np.any(np.abs(xi_keep) > 1.0 + 1e-12):
        return False  # original ξ already out of box

    # If no tail (K >= ng case): pruned set is just (c, G) — no tail to check
    if drop.size == 0:
        recon = c + G_kept @ xi_keep
        return bool(np.allclose(recon, p_orig, atol=atol))

    if tail_radius is None:
        return False  # tail required but missing

    # dropped contribution in each row
    # NOTE: the test caller must pass the ORIGINAL G via state.metadata
    G_orig = state.metadata.get("G_orig")
    assert G_orig is not None, "test requires state.metadata['G_orig'] for containment witness"
    dropped_per_row = G_orig[:, drop] @ xi_orig[drop]    # (n,)

    # The pruned tail row i is tail_radius[i] · xi_tail[i] (one independent
    # xi_tail variable per row). To absorb dropped_per_row[i] exactly we
    # need xi_tail[i] = dropped_per_row[i] / tail_radius[i].
    # If tail_radius[i] == 0 then dropped_per_row[i] must also be 0 for
    # the point to be representable.
    xi_tail = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if tail_radius[i] > 0:
            xi_tail[i] = dropped_per_row[i] / tail_radius[i]
        else:
            if abs(dropped_per_row[i]) > atol:
                return False  # tail too tight; point not representable

    # Box check
    if np.any(np.abs(xi_tail) > 1.0 + 1e-9):
        return False

    # Reconstruction check
    recon = c + G_kept @ xi_keep + tail_radius * xi_tail
    return bool(np.allclose(recon, p_orig, atol=atol))


# ─── Test cases ───────────────────────────────────────────────────


class TestPruneSoundness(unittest.TestCase):
    """The original HZ must be a subset of the pruned HZ, for every d.

    Soundness must hold for: zero, random, sign-flipped, and orthogonal d.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(20260604)
        self.n, self.ng, self.K = 8, 20, 6
        self.c = rng.normal(size=self.n)
        self.G = rng.normal(size=(self.n, self.ng))
        self.xi_samples = rng.uniform(-1.0, 1.0,
                                       size=(1000, self.ng)).astype(np.float64)

    def _check_all_samples(self, d: np.ndarray, label: str) -> None:
        state = prune(self.c, self.G, d, self.K, return_metadata=True)
        # Attach the original G so the witness construction can compute
        # dropped contributions per row.
        state.metadata["G_orig"] = self.G
        n_fail = 0
        first_fail = None
        for xi in self.xi_samples:
            p = self.c + self.G @ xi
            if not is_contained_by_metadata(p, state, xi):
                n_fail += 1
                if first_fail is None:
                    first_fail = xi
        self.assertEqual(
            n_fail, 0,
            msg=f"d={label}: {n_fail}/1000 original points not in pruned set; "
                f"first failing xi={first_fail}",
        )

    def test_pruned_contains_original_zero_d(self) -> None:
        self._check_all_samples(np.zeros(self.n), "zero")

    def test_pruned_contains_original_random_d(self) -> None:
        rng = np.random.default_rng(20260605)
        self._check_all_samples(rng.normal(size=self.n), "random")

    def test_pruned_contains_original_sign_flipped_d(self) -> None:
        rng = np.random.default_rng(20260606)
        d_true = rng.normal(size=self.n)
        self._check_all_samples(-d_true, "sign_flipped")

    def test_pruned_contains_original_orth_d(self) -> None:
        d = make_orthogonal_to_max_norm_col(self.G)
        self._check_all_samples(d, "orth_to_max_col")


if __name__ == "__main__":
    unittest.main()
