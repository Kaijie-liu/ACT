"""Multi-layer PRUNE soundness regression — pins the 2026-06-04 bug.

The bug: `prune(c, G, d, K)` in research/sc_hz/prune.py did NOT preserve
the incoming `tail_radius` from prior prune steps. In a multi-layer
forward propagation where tail accumulates across layers, calling prune
at intermediate layers ERASED the accumulated tail and replaced it only
with the row-L1 of the newly dropped columns.

Effect: LP UB became under-approximated (too small), leading to false
CERT verdicts. The 71 relusplitter CERTs from the horizontal extension
were ALL bug artifacts (0 survived the fix).

This test pins the soundness invariant:
    For any K_small < K_∞, UB(K=K_small) >= UB(K=∞) on a multi-layer
    forward propagation.

A correct PRUNE can only LOOSEN the UB (over-approximation); under-
approximation indicates the bug. The test runs a small dense+ReLU
forward chain and verifies UB(K=small) >= UB(K=large) within numerical
tolerance.

Originally caught by:
  - Manually comparing K=256 LP UB to K=∞ LP UB on relusplitter iid 2
    when implementing the forward-coefficient witness extractor; the
    K=256 LP UB was -1.00 but K=∞ LP UB was +0.93, violating the
    over-approximation property.

Fix:
  - prune.py: add `incoming_tail_radius` parameter; new tail = incoming
    + row-L1 of dropped columns.
  - onnx_walker.py: pass `state.tail_radius` to every prune call.
"""
from __future__ import annotations

import unittest

import numpy as np

from research.sc_hz.onnx_walker import forward_propagate, LayerOp
from research.sc_hz.precompute_direction import precompute_d_per_layer_chain
from research.sc_hz.prune import PrunedState
from research.sc_hz.ops import lp_ub_rival_margin


def _build_multilayer_test_case(seed: int = 20260604):
    """Build a synthetic 4-layer Dense+ReLU network with random weights.

    Input dim 32, hidden 64-64-64, output 8. Initial G_0 = diag(r_in)
    has 32 columns, so prune at K=8 fires at every layer with substantial
    drop count.
    """
    rng = np.random.default_rng(seed)
    layers = []
    dims = [32, 64, 64, 64, 8]
    for i in range(len(dims) - 1):
        W = rng.normal(scale=1.0 / np.sqrt(dims[i]), size=(dims[i+1], dims[i]))
        b = rng.normal(scale=0.1, size=(dims[i+1],))
        layers.append(LayerOp("dense", {"W": W, "b": b}))
        if i < len(dims) - 2:  # ReLU between hidden layers, not after final
            layers.append(LayerOp("relu", {}))

    c_in = rng.uniform(-1, 1, size=(dims[0],))
    r_in = np.abs(rng.uniform(0.05, 0.5, size=(dims[0],)))
    # Output direction (8-D)
    d_out = rng.normal(size=(dims[-1],))

    return layers, c_in, r_in, d_out


class TestPruneMultiLayerSoundness(unittest.TestCase):
    """Pin: UB(K=K_small) >= UB(K=∞) for any K_small < ∞."""

    def setUp(self):
        self.layers, self.c_in, self.r_in, self.d_out = (
            _build_multilayer_test_case(seed=20260604)
        )
        from research.sc_hz.run_sentinels import _layer_output_shapes
        self.out_shapes = _layer_output_shapes(self.layers, (32,))
        self.d_chain = precompute_d_per_layer_chain(
            self.layers, self.d_out, self.out_shapes,
        )
        self.init_state = PrunedState(
            c=self.c_in.copy(), G_kept=np.diag(self.r_in),
            tail_radius=None, metadata={},
        )

    def _ub_at_K(self, K: int) -> float:
        state, _ = forward_propagate(
            self.init_state, self.layers, self.d_chain,
            K_per_layer=K, initial_shape=(32,),
        )
        return lp_ub_rival_margin(state, self.d_out)

    def test_K_inf_is_tightest(self):
        """UB(K=∞) must be the TIGHTEST (smallest) UB."""
        ub_inf = self._ub_at_K(100000)
        for K in [4, 8, 16, 32, 64, 128]:
            ub_K = self._ub_at_K(K)
            self.assertGreaterEqual(
                ub_K + 1e-9, ub_inf,
                msg=f"UB(K={K})={ub_K:.6e} < UB(K=∞)={ub_inf:.6e}: PRUNE UNSOUND",
            )

    def test_monotonic_K_increase_tightens_ub(self):
        """UB(K=K_small) >= UB(K=K_large) for K_small < K_large."""
        K_list = [4, 8, 16, 32, 64, 128, 100000]
        ubs = [self._ub_at_K(K) for K in K_list]
        for i in range(len(K_list) - 1):
            self.assertGreaterEqual(
                ubs[i] + 1e-9, ubs[i+1],
                msg=(
                    f"non-monotone: UB(K={K_list[i]})={ubs[i]:.6e} < "
                    f"UB(K={K_list[i+1]})={ubs[i+1]:.6e}: PRUNE UNSOUND"
                ),
            )

    def test_pruned_set_contains_unpruned_via_brute_force(self):
        """Soundness via direct sampling.

        Sample 200 points from the K=∞ (raw) set; each should produce
        d.y <= UB(K=K_small) for any smaller K.
        """
        from research.sc_hz import ops as scops
        # K=∞ forward; collect final state
        state_inf, _ = forward_propagate(
            self.init_state, self.layers, self.d_chain,
            K_per_layer=100000, initial_shape=(32,),
        )
        rng = np.random.default_rng(20260604)
        N = 200
        K_test = 8
        ub_K_test = self._ub_at_K(K_test)
        max_violation = 0.0
        for _ in range(N):
            xi = rng.uniform(-1, 1, size=(state_inf.G_kept.shape[1],))
            y = state_inf.c + state_inf.G_kept @ xi
            d_y = float(self.d_out @ y)
            if d_y > ub_K_test:
                max_violation = max(max_violation, d_y - ub_K_test)
        self.assertEqual(
            max_violation, 0.0,
            msg=f"K={K_test} UB violated by raw-set sample: max excess {max_violation}"
        )


class TestPrunePreservesIncomingTail(unittest.TestCase):
    """Direct unit test for the prune.py fix.

    The bug-fixed prune must add the incoming tail to the new dropped-cols
    tail. Verify with an explicit incoming_tail_radius argument.
    """

    def test_incoming_tail_added_to_dropped_cols(self):
        from research.sc_hz.prune import prune
        n, ng, K = 4, 10, 4
        c = np.zeros(n)
        G = np.eye(n).repeat(3, axis=1)[:, :ng]  # mostly zeros with structure
        G = G + 0.1
        d = np.ones(n)
        incoming = np.array([1.0, 2.0, 3.0, 4.0])
        result = prune(c, G, d, K, return_metadata=True,
                        incoming_tail_radius=incoming)
        dropped_cols_row_L1 = np.abs(
            G[:, result.metadata["drop"]]
        ).sum(axis=1)
        expected_tail = incoming + dropped_cols_row_L1
        np.testing.assert_allclose(result.tail_radius, expected_tail)

    def test_incoming_tail_preserved_when_K_geq_ng(self):
        """When K >= ng (no prune), incoming tail must still be preserved."""
        from research.sc_hz.prune import prune
        n, ng = 4, 5
        c = np.zeros(n)
        G = np.random.default_rng(0).normal(size=(n, ng))
        d = np.ones(n)
        incoming = np.array([10.0, 20.0, 30.0, 40.0])
        result = prune(c, G, d, K=100, return_metadata=True,
                        incoming_tail_radius=incoming)
        np.testing.assert_allclose(result.tail_radius, incoming)


if __name__ == "__main__":
    unittest.main()
