"""Analytic inclusion and false-witness controls for the relationship ablation."""
import unittest
from dataclasses import replace

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import sparse_hz_from_bounds, sparse_hz_linear, hz_add_output_inequalities
from act.back_end.moe.weighted_top2 import (
    shared_input_pair_hz, independent_input_pair_hz,
    build_weighted_top2_f0, solve_weighted_top2_f0,
)
from act.back_end.moe.test_weighted_top2 import _expert_with_private_factors


class IndependentPairTests(unittest.TestCase):
    def setUp(self):
        self.entry = sparse_hz_from_bounds(Bounds(
            torch.tensor([[-1.]], dtype=torch.float64),
            torch.tensor([[1.]], dtype=torch.float64)), frame_id=901)

    def solve(self, constructor, offset=.2):
        a = sparse_hz_linear(self.entry, [[1.]], [offset])
        b = sparse_hz_linear(self.entry, [[-1.]], [offset])
        joint = constructor(self.entry, a, b)
        router = sparse_hz_linear(self.entry, [[0.], [0.]])
        enc = build_weighted_top2_f0(joint, router, (0, 1), [1.], 0.,
                                     margin_time_limit=2., difference_time_limit=2.)
        return solve_weighted_top2_f0(enc, input_shape=(1, 1), time_limit=2.), enc

    def test_cancellation_is_lost_but_not_reported_unsafe(self):
        shared, a = self.solve(shared_input_pair_hz)
        independent, b = self.solve(independent_input_pair_hz)
        self.assertEqual(shared.status, "SAFE")
        self.assertAlmostEqual(shared.minimum, .2, places=7)
        self.assertEqual(independent.status, "UNKNOWN")
        self.assertAlmostEqual(independent.minimum, -.8, places=7)
        self.assertEqual(a.margin_bounds, b.margin_bounds)
        self.assertIsNotNone(independent.candidate_input)
        # Every real same-input mixture equals .2, including this false witness.
        x = float(independent.candidate_input.reshape(-1)[0])
        self.assertAlmostEqual(.5*(x+.2)+.5*(-x+.2), .2)

    def test_genuinely_unsafe_control_never_becomes_safe(self):
        for constructor in (shared_input_pair_hz, independent_input_pair_hz):
            decision, _ = self.solve(constructor, offset=-.2)
            self.assertEqual(decision.status, "UNKNOWN")
            self.assertLess(decision.minimum, 0)

    def test_private_binary_and_all_guard_constraints_remain_disjoint(self):
        entry = hz_add_output_inequalities(self.entry, [[1.]], [.5])
        a = _expert_with_private_factors(entry, 2., 3.)
        b = _expert_with_private_factors(entry, -4., 5.)
        pair = independent_input_pair_hz(entry, a, b)
        hz = pair.output_hz
        self.assertEqual((hz.n_cont, hz.n_bin), (a.n_cont+b.n_cont, a.n_bin+b.n_bin))
        self.assertEqual((pair.shared_continuous, pair.shared_binary), (0, 0))
        self.assertNotEqual(hz.frame_id, entry.frame_id)
        self.assertEqual(pair.source_frame_id, entry.frame_id)
        for name, width in (("Gc", a.n_cont), ("Gb", a.n_bin),
                            ("Ac", a.n_cont), ("Ab", a.n_bin),
                            ("Auc", a.n_cont), ("Aub", a.n_bin)):
            original = getattr(a, name); other = getattr(b, name); merged = getattr(hz, name)
            rows = original.shape[0]
            self.assertEqual((merged[:rows, :width]-original).nnz, 0)
            self.assertEqual((merged[rows:, width:]-other).nnz, 0)
            self.assertEqual(merged[:rows, width:].nnz + merged[rows:, :width].nnz, 0)
        # Diagonal assignment (one input, separate expert-private factors).
        for x in (-1., 0., .5):
            ca, cb, ba, bb = np.array([x, 0.]), np.array([x, 0.]), np.array([-1.]), np.array([-1.])
            c, binary = np.r_[ca, cb], np.r_[ba, bb]
            self.assertTrue(np.all(hz.Auc@c + hz.Aub@binary <= hz.ub))
            np.testing.assert_allclose(hz.c+hz.Gc@c+hz.Gb@binary,
                np.r_[a.c+a.Gc@ca+a.Gb@ba, b.c+b.Gc@cb+b.Gb@bb])
            np.testing.assert_allclose(pair.input_hz.c+pair.input_hz.Gc@c+pair.input_hz.Gb@binary, [x])

    def test_marginal_guard_is_not_dropped(self):
        entry = hz_add_output_inequalities(self.entry, [[-1.]], [0.])  # x >= 0
        a = sparse_hz_linear(entry, [[1.]], [.2])
        pair = independent_input_pair_hz(entry, a, a)
        router = sparse_hz_linear(entry, [[0.], [0.]])
        enc = build_weighted_top2_f0(pair, router, (0, 1), [1.], 0., margin_time_limit=2., difference_time_limit=2.)
        result = solve_weighted_top2_f0(enc, input_shape=(1, 1), time_limit=2.)
        self.assertEqual(result.status, "SAFE")
        self.assertAlmostEqual(result.minimum, .2, places=7)
        with self.assertRaises(ValueError):
            build_weighted_top2_f0(replace(pair, source_frame_id=902), router, (0, 1), [1.], 0.,
                                  margin_time_limit=2., difference_time_limit=2.)

    def test_entry_binary_is_duplicated_not_reidentified(self):
        entry = _expert_with_private_factors(self.entry, 2., 3.)
        a = sparse_hz_linear(entry, [[1.]])
        b = sparse_hz_linear(entry, [[-1.]])
        pair = independent_input_pair_hz(entry, a, b)
        self.assertEqual(pair.output_hz.n_bin, 2*entry.n_bin)
        self.assertEqual(pair.shared_binary, 0)
        self.assertEqual(pair.output_hz.Aub.shape[0], 2*entry.n_ineq)


if __name__ == "__main__":
    unittest.main()
