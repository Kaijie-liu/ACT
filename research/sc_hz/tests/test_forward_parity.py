"""When K=ng, PRUNE is the identity; the pruned HZ equals the original.

Per dc_hz_phase_a_plan.md §1.3 and EXECUTION §1.3: the prune algorithm
returns the original (c, G) unchanged when the budget K is >= ng. This
test pins that contract — any pruning at K=ng would be a bug, since
all generators fit in budget.
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


class TestForwardParity(unittest.TestCase):
    """K >= ng must be a no-op."""

    def test_prune_at_K_equals_ng_is_identity(self) -> None:
        rng = np.random.default_rng(20260604)
        n, ng = 8, 12
        c = rng.normal(size=n)
        G = rng.normal(size=(n, ng))
        d = rng.normal(size=n)

        state_p = prune(c, G, d, K=ng, return_metadata=True)

        np.testing.assert_allclose(state_p.c, c,
            err_msg="K=ng: center must be unchanged")
        np.testing.assert_allclose(state_p.G_kept, G,
            err_msg="K=ng: G_kept must equal original G")
        self.assertEqual(state_p.metadata["drop"].size, 0,
            msg="K=ng: drop indices must be empty")
        # tail must be absent (None) or zero
        self.assertTrue(
            state_p.tail_radius is None or
            np.all(np.abs(state_p.tail_radius) < 1e-15),
            msg="K=ng: tail_radius must be None or all-zero",
        )

    def test_prune_at_K_larger_than_ng_is_identity(self) -> None:
        rng = np.random.default_rng(20260609)
        n, ng = 6, 8
        c = rng.normal(size=n)
        G = rng.normal(size=(n, ng))
        d = rng.normal(size=n)

        state_p = prune(c, G, d, K=ng + 5, return_metadata=True)

        np.testing.assert_allclose(state_p.c, c)
        np.testing.assert_allclose(state_p.G_kept, G)
        self.assertEqual(state_p.metadata["drop"].size, 0)


if __name__ == "__main__":
    unittest.main()
