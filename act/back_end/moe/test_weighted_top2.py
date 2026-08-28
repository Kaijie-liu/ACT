# ===- act/back_end/moe/test_weighted_top2.py - Weighted Fallback Tests -====#

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.moe.hz_routing import (
    analyze_topk_sets,
    condition_topk_set,
)
from act.back_end.moe.weighted_top2 import (
    UNKNOWN_WEIGHTED_RELAXATION,
    build_weighted_top2_f0,
    mccormick_contains,
    mccormick_inequalities,
    shared_input_pair_hz,
    solve_weighted_top2_f0,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_add_output_inequalities,
    sparse_empty,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


def _expert_with_private_factors(entry, private_cont, private_binary):
    n_cont, n_bin = entry.n_cont + 1, entry.n_bin + 1
    Gc = sp.csr_matrix([[1.0, float(private_cont)]])
    Gb = sp.csr_matrix([[float(private_binary)]])
    prefix_Auc = sp.hstack(
        [entry.Auc, sparse_empty(entry.n_ineq, 1)], format="csr"
    )
    prefix_Aub = sp.hstack(
        [entry.Aub, sparse_empty(entry.n_ineq, 1)], format="csr"
    )
    return SparseHZono(
        c=np.zeros(1),
        Gc=Gc,
        Gb=Gb,
        Ac=sp.hstack([entry.Ac, sparse_empty(entry.n_eq, 1)], format="csr"),
        Ab=sp.hstack([entry.Ab, sparse_empty(entry.n_eq, 1)], format="csr"),
        b=entry.b.copy(),
        Auc=sp.vstack(
            [prefix_Auc, sp.csr_matrix([[0.0, 1.0]])], format="csr"
        ),
        Aub=sp.vstack(
            [prefix_Aub, sp.csr_matrix([[1.0]])], format="csr"
        ),
        ub=np.concatenate([entry.ub, np.asarray([1.0])]),
        frame_id=entry.frame_id,
        exact=True,
    )


def _equal_expert_encoding(offset=0.0):
    entry = sparse_hz_from_bounds(
        Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
        frame_id=31,
    )
    expert_a = sparse_hz_linear(entry, [[1.0]], [offset])
    expert_b = sparse_hz_linear(entry, [[1.0]], [offset])
    pair = shared_input_pair_hz(entry, expert_a, expert_b)
    router = SparseHZono(
        c=np.zeros(2),
        Gc=sparse_empty(2, entry.n_cont),
        Gb=sparse_empty(2, entry.n_bin),
        Ac=entry.Ac.copy(),
        Ab=entry.Ab.copy(),
        b=entry.b.copy(),
        Auc=entry.Auc.copy(),
        Aub=entry.Aub.copy(),
        ub=entry.ub.copy(),
        frame_id=entry.frame_id,
        exact=True,
    )
    return build_weighted_top2_f0(
        pair,
        router,
        (0, 1),
        [1.0],
        0.0,
        margin_time_limit=2.0,
        difference_time_limit=2.0,
    )


class WeightedTop2FallbackTests(unittest.TestCase):
    def test_zero_router_margin_has_half_gate(self):
        encoding = _equal_expert_encoding()
        self.assertEqual(encoding.margin_bounds, (0.0, 0.0))
        self.assertEqual(encoding.bounds.lambda_lower, 0.5)
        self.assertEqual(encoding.bounds.lambda_upper, 0.5)

    def test_equal_experts_have_zero_product_difference(self):
        encoding = _equal_expert_encoding()
        self.assertAlmostEqual(encoding.bounds.difference_lower, 0.0)
        self.assertAlmostEqual(encoding.bounds.difference_upper, 0.0)
        self.assertAlmostEqual(encoding.bounds.product_lower, 0.0)
        self.assertAlmostEqual(encoding.bounds.product_upper, 0.0)

    def test_mccormick_contains_fixed_positive_difference(self):
        A, b = mccormick_inequalities(0.2, 0.8, 1.0, 3.0)
        for gate in np.linspace(0.2, 0.8, 7):
            for difference in np.linspace(1.0, 3.0, 7):
                self.assertTrue(
                    mccormick_contains(
                        gate,
                        difference,
                        gate * difference,
                        A,
                        b,
                    )
                )

    def test_mccormick_contains_difference_crossing_zero(self):
        A, b = mccormick_inequalities(0.1, 0.9, -2.0, 3.0)
        for gate in np.linspace(0.1, 0.9, 9):
            for difference in np.linspace(-2.0, 3.0, 11):
                self.assertTrue(
                    mccormick_contains(
                        gate,
                        difference,
                        gate * difference,
                        A,
                        b,
                    )
                )

    def test_outside_expert_tie_is_legal_without_selection_binary(self):
        router = sparse_hz_from_bounds(
            Bounds(torch.ones(1, 3), torch.ones(1, 3)),
            frame_id=41,
        )
        guarded = condition_topk_set(router, (0, 1)).hz
        self.assertEqual(guarded.n_bin, router.n_bin)
        report = analyze_topk_sets(router, 2, time_limit_per_set=2.0)
        self.assertIn((0, 1), report.feasible)

    def test_all_tied_router_has_multiple_legal_pairs(self):
        router = sparse_hz_from_bounds(
            Bounds(torch.ones(1, 3), torch.ones(1, 3)),
            frame_id=43,
        )
        report = analyze_topk_sets(router, 2, time_limit_per_set=2.0)
        self.assertTrue(report.exact)
        self.assertEqual(report.feasible, ((0, 1), (0, 2), (1, 2)))

    def test_shared_input_identity_and_private_binary_separation(self):
        entry = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=47,
        )
        entry = hz_add_output_inequalities(entry, [[1.0]], [0.75])
        expert_a = _expert_with_private_factors(entry, 2.0, 3.0)
        expert_b = _expert_with_private_factors(entry, -4.0, -5.0)
        pair = shared_input_pair_hz(entry, expert_a, expert_b)
        Gc = pair.output_hz.Gc.toarray()
        Gb = pair.output_hz.Gb.toarray()
        self.assertEqual(pair.shared_continuous, 1)
        self.assertEqual(pair.a_private_continuous, 1)
        self.assertEqual(pair.b_private_continuous, 1)
        self.assertTrue(np.array_equal(Gc[:, 0], [1.0, 1.0]))
        self.assertTrue(np.array_equal(Gc[:, 1], [2.0, 0.0]))
        self.assertTrue(np.array_equal(Gc[:, 2], [0.0, -4.0]))
        self.assertTrue(np.array_equal(Gb[:, 0], [3.0, 0.0]))
        self.assertTrue(np.array_equal(Gb[:, 1], [0.0, -5.0]))
        self.assertEqual(pair.output_hz.n_ineq, 3)
        self.assertEqual(pair.input_hz.n_cont, 3)
        self.assertEqual(pair.input_hz.n_bin, 2)

    def test_random_concrete_products_satisfy_all_constraints(self):
        rng = np.random.default_rng(0)
        A, b = mccormick_inequalities(0.17, 0.83, -4.0, 2.5)
        for _ in range(200):
            gate = rng.uniform(0.17, 0.83)
            difference = rng.uniform(-4.0, 2.5)
            self.assertTrue(
                mccormick_contains(
                    gate,
                    difference,
                    gate * difference,
                    A,
                    b,
                )
            )

    def test_random_shared_inputs_satisfy_generated_encoding_constraints(self):
        entry = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=53,
        )
        expert_a = sparse_hz_linear(entry, [[2.0]], [1.0])
        expert_b = sparse_hz_linear(entry, [[-1.0]], [0.5])
        pair = shared_input_pair_hz(entry, expert_a, expert_b)
        router = sparse_hz_linear(entry, [[1.0], [-0.5]])
        encoding = build_weighted_top2_f0(
            pair,
            router,
            (0, 1),
            [1.0],
            0.0,
            margin_time_limit=2.0,
            difference_time_limit=2.0,
        )
        rng = np.random.default_rng(7)
        for value in rng.uniform(-1.0, 1.0, 200):
            gate = float(torch.sigmoid(torch.tensor(1.5 * value)))
            difference = 3.0 * value + 0.5
            self.assertTrue(
                mccormick_contains(
                    gate,
                    difference,
                    gate * difference,
                    encoding.mccormick_A,
                    encoding.mccormick_b,
                )
            )

    def test_relaxation_candidate_is_not_directly_unsafe(self):
        encoding = _equal_expert_encoding(offset=0.0)
        decision = solve_weighted_top2_f0(
            encoding,
            input_shape=(1, 1),
            time_limit=2.0,
        )
        self.assertEqual(decision.status, "UNKNOWN")
        self.assertEqual(decision.reason, UNKNOWN_WEIGHTED_RELAXATION)
        self.assertIsNotNone(decision.candidate_input)

    def test_positive_relaxed_lower_bound_is_safe(self):
        encoding = _equal_expert_encoding(offset=2.0)
        decision = solve_weighted_top2_f0(
            encoding,
            input_shape=(1, 1),
            time_limit=2.0,
        )
        self.assertEqual(decision.status, "SAFE")

    def test_reversed_mccormick_row_fails_mutation_control(self):
        A, b = mccormick_inequalities(
            0.2,
            0.8,
            -2.0,
            3.0,
            reverse_row=0,
        )
        gate, difference = 0.5, 0.5
        self.assertFalse(
            mccormick_contains(
                gate,
                difference,
                gate * difference,
                A,
                b,
            )
        )


if __name__ == "__main__":
    unittest.main()
