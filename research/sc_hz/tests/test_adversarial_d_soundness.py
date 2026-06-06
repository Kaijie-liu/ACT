"""I10: PRUNE soundness is independent of d_L.

Per dc_hz_phase_a_plan.md §1.4 / Invariant I10: PRUNE must over-approximate
the original HZ for ANY value of d_L — including pathological choices.

This is the operationalization of the principle that d_L is a
representation-choice heuristic, not bound information; soundness MUST
hold even when the heuristic is meaningless or actively wrong.

This test runs PRUNE four times on the same (c, G) with four different
d_L choices, and checks containment by metadata witness on 1000 samples
of xi.
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
from research.sc_hz.tests.test_prune_soundness import (  # noqa: E402
    is_contained_by_metadata,
    make_orthogonal_to_max_norm_col,
)


class TestAdversarialDSoundness(unittest.TestCase):
    """All four adversarial d_L choices must preserve soundness."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20260604)
        self.n, self.ng, self.K = 8, 20, 6
        self.c = rng.normal(size=self.n)
        self.G = rng.normal(size=(self.n, self.ng))
        self.xi_samples = rng.uniform(-1.0, 1.0,
                                       size=(1000, self.ng)).astype(np.float64)

    def _check_containment(self, d: np.ndarray, label: str) -> None:
        state = prune(self.c, self.G, d, self.K, return_metadata=True)
        state.metadata["G_orig"] = self.G
        n_fail = 0
        for xi in self.xi_samples:
            p = self.c + self.G @ xi
            if not is_contained_by_metadata(p, state, xi):
                n_fail += 1
        self.assertEqual(
            n_fail, 0,
            msg=f"d={label}: {n_fail}/1000 samples failed containment",
        )

    def test_d_zero(self) -> None:
        """All-zero d_L: relevance score is uniformly zero. PRUNE will
        select the first K columns (or in some tie-breaking order); the
        soundness still must hold."""
        self._check_containment(np.zeros(self.n), "zero")

    def test_d_random(self) -> None:
        """Random d_L: any non-pathological direction."""
        rng = np.random.default_rng(20260605)
        self._check_containment(rng.normal(size=self.n), "random")

    def test_d_sign_flipped(self) -> None:
        """Sign-flipped from a hypothetical 'true' direction."""
        rng = np.random.default_rng(20260606)
        d_true = rng.normal(size=self.n)
        self._check_containment(-d_true, "sign_flipped")

    def test_d_orthogonal_to_max_col(self) -> None:
        """d orthogonal to the largest-norm column. PRUNE will discard
        the high-norm direction (since its relevance score is 0), which
        stress-tests the tail merge for high-norm absorbed generators."""
        d = make_orthogonal_to_max_norm_col(self.G)
        self._check_containment(d, "orth_to_max_col")


if __name__ == "__main__":
    unittest.main()
