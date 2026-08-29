# ===- act/back_end/moe/test_weighted_topk.py - Top-k Fallback Tests -===#

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.moe.hz_routing import analyze_topk_sets, condition_topk_set
from act.back_end.moe.schema import GateKind
from act.back_end.moe.weighted_top2 import (
    UNKNOWN_WEIGHTED_RELAXATION,
    build_weighted_top2_f0,
    compute_weighted_top2_gate_range,
    mccormick_contains,
    shared_input_pair_hz,
)
from act.back_end.moe.weighted_topk import (
    NormalizedTopKGateBox,
    UnsupportedNormalizedGateError,
    build_weighted_topk_range,
    compute_normalized_topk_gate_box,
    decompose_normalized_topk_scalar,
    normalized_gate_box_from_score_bounds,
    normalized_gate_support,
    shared_input_experts_hz,
    simplex_anchor_contains,
    simplex_box_contains,
    solve_weighted_topk_range,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_add_output_inequalities,
    sparse_empty,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


def _expert_with_private_factors(entry, slope, private_cont, private_binary):
    n_cont, n_bin = entry.n_cont + 1, entry.n_bin + 1
    Gc = sp.csr_matrix([[float(slope), float(private_cont)]])
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


def _linear_components(count=3, *, offset=0.0, frame_id=71):
    entry = sparse_hz_from_bounds(
        Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
        frame_id=frame_id,
    )
    experts = {
        index: sparse_hz_linear(
            entry,
            [[float(index + 1)]],
            [float(offset + index * 0.25)],
        )
        for index in range(count)
    }
    router = sparse_hz_linear(
        entry,
        [[0.25 * (index + 1)] for index in range(count)],
        [0.1 * index for index in range(count)],
    )
    return entry, experts, router


class WeightedTopKFallbackTests(unittest.TestCase):
    def test_gate_support_matrix_is_explicit(self):
        for kind in (
            GateKind.HARD_TOP1,
            GateKind.SELECTED_SOFTMAX,
            GateKind.NORMALIZED_SIGMOID,
        ):
            self.assertTrue(normalized_gate_support(kind)[0])
        supported, reason = normalized_gate_support(GateKind.SWITCH_PROB)
        self.assertFalse(supported)
        self.assertIn("scale", reason)

    def test_switch_probability_is_not_silently_normalized(self):
        _, _, router = _linear_components(1)
        with self.assertRaisesRegex(
            UnsupportedNormalizedGateError, "independent scale"
        ):
            normalized_gate_box_from_score_bounds(
                router,
                (0,),
                GateKind.SWITCH_PROB,
                [-1.0],
                [1.0],
            )

    def test_score_boxes_contain_concrete_normalized_gates(self):
        _, _, router = _linear_components(4)
        lower = np.asarray([-2.0, -0.5, 0.25, 1.0])
        upper = np.asarray([-0.25, 0.75, 1.5, 2.0])
        rng = np.random.default_rng(3)
        for kind in (
            GateKind.SELECTED_SOFTMAX,
            GateKind.NORMALIZED_SIGMOID,
        ):
            box = normalized_gate_box_from_score_bounds(
                router,
                (0, 1, 2, 3),
                kind,
                lower,
                upper,
            )
            for _ in range(200):
                scores = torch.from_numpy(rng.uniform(lower, upper)).double()
                if kind == GateKind.SELECTED_SOFTMAX:
                    weights = torch.softmax(scores, dim=0)
                else:
                    raw = torch.sigmoid(scores)
                    weights = raw / raw.sum()
                self.assertTrue(simplex_box_contains(weights.numpy(), box))

    def test_normalized_decomposition_uses_exactly_k_minus_one_products(self):
        rng = np.random.default_rng(5)
        for count in range(1, 7):
            for _ in range(50):
                values = rng.normal(size=count)
                weights = rng.dirichlet(np.ones(count))
                direct, anchor, differences, products = (
                    decompose_normalized_topk_scalar(
                        values, weights, constant=0.37
                    )
                )
                self.assertEqual(len(products), count - 1)
                self.assertEqual(len(differences), count - 1)
                self.assertAlmostEqual(direct, anchor + sum(products), places=12)

    def test_shared_input_identity_and_all_private_factors_are_separate(self):
        entry = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=73,
        )
        entry = hz_add_output_inequalities(entry, [[1.0]], [0.75])
        experts = {
            2: _expert_with_private_factors(entry, 1.0, 2.0, 3.0),
            5: _expert_with_private_factors(entry, 1.0, -4.0, -5.0),
            7: _expert_with_private_factors(entry, 1.0, 6.0, 7.0),
        }
        merged = shared_input_experts_hz(entry, experts)
        Gc = merged.output_hz.Gc.toarray()
        Gb = merged.output_hz.Gb.toarray()
        self.assertEqual(merged.route_set, (2, 5, 7))
        self.assertEqual(merged.private_continuous, (1, 1, 1))
        self.assertEqual(merged.private_binary, (1, 1, 1))
        self.assertTrue(np.array_equal(Gc[:, 0], [1.0, 1.0, 1.0]))
        self.assertTrue(np.array_equal(Gc[:, 1:], np.diag([2.0, -4.0, 6.0])))
        self.assertTrue(np.array_equal(Gb, np.diag([3.0, -5.0, 7.0])))
        self.assertEqual(merged.output_hz.n_ineq, 4)
        self.assertEqual(merged.input_hz.n_cont, 4)
        self.assertEqual(merged.input_hz.n_bin, 3)

    def test_all_tied_router_requires_every_legal_topk_set(self):
        router = sparse_hz_from_bounds(
            Bounds(torch.ones(1, 4), torch.ones(1, 4)), frame_id=79
        )
        report = analyze_topk_sets(router, 3, time_limit_per_set=2.0)
        self.assertTrue(report.exact)
        self.assertEqual(
            report.feasible,
            ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        )
        for route_set in report.feasible:
            guarded = condition_topk_set(router, route_set).hz
            box = compute_normalized_topk_gate_box(
                guarded,
                route_set,
                GateKind.SELECTED_SOFTMAX,
                time_limit=2.0,
            )
            self.assertTrue(simplex_box_contains([1.0 / 3.0] * 3, box))

    def test_random_concrete_products_satisfy_every_term_hull(self):
        entry, experts, router = _linear_components(4, frame_id=83)
        merged = shared_input_experts_hz(entry, experts)
        box = compute_normalized_topk_gate_box(
            router,
            (0, 1, 2, 3),
            GateKind.SELECTED_SOFTMAX,
            time_limit=2.0,
        )
        encoding = build_weighted_topk_range(
            merged,
            router,
            box,
            [1.0],
            0.2,
            difference_time_limit=2.0,
        )
        self.assertEqual(len(encoding.term_bounds), 3)
        self.assertTrue(
            np.array_equal(
                encoding.simplex_A,
                np.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            )
        )
        rng = np.random.default_rng(11)
        for value in rng.uniform(-1.0, 1.0, 200):
            scores = np.asarray(
                [0.25 * (index + 1) * value + 0.1 * index for index in range(4)]
            )
            weights = torch.softmax(torch.from_numpy(scores), dim=0).numpy()
            expert_values = np.asarray(
                [(index + 1) * value + index * 0.25 for index in range(4)]
            )
            direct, anchor, differences, products = decompose_normalized_topk_scalar(
                expert_values, weights, constant=0.2
            )
            self.assertAlmostEqual(direct, anchor + sum(products), places=12)
            self.assertTrue(simplex_box_contains(weights, box))
            self.assertTrue(simplex_anchor_contains(weights[:-1], box))
            self.assertTrue(
                np.all(
                    encoding.simplex_A @ weights[:-1]
                    <= encoding.simplex_b + 1e-12
                )
            )
            lambda_coordinates = []
            product_coordinates = []
            for position in range(3):
                bounds = encoding.term_bounds[position]
                self.assertTrue(
                    mccormick_contains(
                        weights[position],
                        differences[position],
                        products[position],
                        encoding.mccormick_A[position],
                        encoding.mccormick_b[position],
                    )
                )
                lambda_center = (
                    bounds.lambda_lower + bounds.lambda_upper
                ) * 0.5
                lambda_radius = (
                    bounds.lambda_upper - bounds.lambda_lower
                ) * 0.5
                product_center = (
                    bounds.product_lower + bounds.product_upper
                ) * 0.5
                product_radius = (
                    bounds.product_upper - bounds.product_lower
                ) * 0.5
                lambda_coordinates.append(
                    0.0
                    if lambda_radius == 0.0
                    else (weights[position] - lambda_center) / lambda_radius
                )
                product_coordinates.append(
                    0.0
                    if product_radius == 0.0
                    else (products[position] - product_center) / product_radius
                )
            generators = np.asarray(
                [value, *lambda_coordinates, *product_coordinates]
            )
            self.assertTrue(np.all(np.abs(generators) <= 1.0 + 1e-10))
            self.assertTrue(
                np.all(
                    encoding.output_hz.Auc @ generators
                    <= encoding.output_hz.ub + 1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    encoding.output_hz.Ac @ generators,
                    encoding.output_hz.b,
                    atol=1e-10,
                )
            )
            relaxed_value = float(
                encoding.output_hz.c[0]
                + (encoding.output_hz.Gc.getrow(0) @ generators)[0]
            )
            self.assertAlmostEqual(relaxed_value, direct, places=10)

    def test_anchor_constraints_reject_box_feasible_non_simplex_weights(self):
        _, _, router = _linear_components(3, frame_id=87)
        box = NormalizedTopKGateBox(
            route_set=(0, 1, 2),
            gate_kind=GateKind.SELECTED_SOFTMAX,
            conditioned_router=router,
            router_frame_id=router.frame_id,
            router_output_width=router.n_out,
            lower=(0.0, 0.0, 0.2),
            upper=(0.8, 0.8, 0.8),
        )
        self.assertTrue(simplex_anchor_contains([0.3, 0.3], box))
        self.assertFalse(simplex_anchor_contains([0.7, 0.7], box))
        self.assertFalse(simplex_anchor_contains([0.05, 0.05], box))

    def test_top2_generic_encoding_preserves_f0_bounds_and_hulls(self):
        entry, experts, router = _linear_components(2, offset=0.5, frame_id=89)
        pair = shared_input_pair_hz(entry, experts[0], experts[1])
        top2_gate = compute_weighted_top2_gate_range(router, (0, 1), time_limit=2.0)
        legacy = build_weighted_top2_f0(
            pair,
            router,
            (0, 1),
            [1.0],
            0.3,
            difference_time_limit=2.0,
            gate_range=top2_gate,
        )
        merged = shared_input_experts_hz(entry, experts)
        generic_box = NormalizedTopKGateBox(
            route_set=(0, 1),
            gate_kind=GateKind.SELECTED_SOFTMAX,
            conditioned_router=router,
            router_frame_id=router.frame_id,
            router_output_width=router.n_out,
            lower=(top2_gate.lambda_bounds[0], 1.0 - top2_gate.lambda_bounds[1]),
            upper=(top2_gate.lambda_bounds[1], 1.0 - top2_gate.lambda_bounds[0]),
            score_support=top2_gate.margin_support,
        )
        generic = build_weighted_topk_range(
            merged,
            router,
            generic_box,
            [1.0],
            0.3,
            difference_time_limit=2.0,
        )
        self.assertEqual(len(generic.term_bounds), 1)
        self.assertEqual(generic.term_bounds[0], legacy.bounds)
        self.assertTrue(np.array_equal(generic.mccormick_A[0], legacy.mccormick_A))
        self.assertTrue(np.array_equal(generic.mccormick_b[0], legacy.mccormick_b))

    def test_gate_box_rejects_another_domain_in_same_frame(self):
        entry, experts, router = _linear_components(3, frame_id=97)
        merged = shared_input_experts_hz(entry, experts)
        box = compute_normalized_topk_gate_box(
            router,
            (0, 1, 2),
            GateKind.NORMALIZED_SIGMOID,
            time_limit=2.0,
        )
        copied_router = sparse_hz_linear(router, np.eye(3))
        with self.assertRaisesRegex(ValueError, "different conditioned router"):
            build_weighted_topk_range(
                merged,
                copied_router,
                box,
                [1.0],
                0.0,
                difference_time_limit=2.0,
            )

    def test_relaxation_candidate_is_never_directly_unsafe(self):
        entry, experts, router = _linear_components(3, frame_id=101)
        zero_experts = {
            index: sparse_hz_linear(entry, [[0.0]], [0.0]) for index in experts
        }
        merged = shared_input_experts_hz(entry, zero_experts)
        box = compute_normalized_topk_gate_box(
            router,
            (0, 1, 2),
            GateKind.SELECTED_SOFTMAX,
            time_limit=2.0,
        )
        encoding = build_weighted_topk_range(
            merged,
            router,
            box,
            [1.0],
            0.0,
            difference_time_limit=2.0,
        )
        decision = solve_weighted_topk_range(
            encoding,
            input_shape=(1, 1),
            time_limit=2.0,
        )
        self.assertEqual(decision.status, "UNKNOWN")
        self.assertEqual(decision.reason, UNKNOWN_WEIGHTED_RELAXATION)
        self.assertIsNotNone(decision.candidate_input)

    def test_positive_topk_relaxed_lower_bound_is_safe(self):
        entry, _, router = _linear_components(3, frame_id=102)
        positive_experts = {
            index: sparse_hz_linear(entry, [[0.0]], [2.0 + index])
            for index in range(3)
        }
        merged = shared_input_experts_hz(entry, positive_experts)
        box = compute_normalized_topk_gate_box(
            router,
            (0, 1, 2),
            GateKind.NORMALIZED_SIGMOID,
            time_limit=2.0,
        )
        encoding = build_weighted_topk_range(
            merged,
            router,
            box,
            [1.0],
            0.25,
            difference_time_limit=2.0,
        )
        decision = solve_weighted_topk_range(
            encoding,
            input_shape=(1, 1),
            time_limit=2.0,
        )
        self.assertEqual(decision.status, "SAFE")

    def test_hard_top1_has_no_product_and_can_certify(self):
        entry, experts, router = _linear_components(1, offset=2.0, frame_id=103)
        merged = shared_input_experts_hz(entry, experts)
        box = compute_normalized_topk_gate_box(
            router,
            (0,),
            GateKind.HARD_TOP1,
            time_limit=2.0,
        )
        encoding = build_weighted_topk_range(
            merged,
            router,
            box,
            [1.0],
            0.1,
            difference_time_limit=2.0,
        )
        self.assertEqual(encoding.term_bounds, ())
        decision = solve_weighted_topk_range(
            encoding,
            input_shape=(1, 1),
            time_limit=2.0,
        )
        self.assertEqual(decision.status, "SAFE")


if __name__ == "__main__":
    unittest.main()
