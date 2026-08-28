# ===- act/back_end/moe/test_moe.py - MoE Route-A Tests ----------------====#

import unittest

import torch
import torch.nn as nn

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.tf_mlp import _sparse_apply_relu
from act.back_end.moe.factory import (
    OutputMoEFactoryConfig,
    build_act_moe_program,
    build_output_moe,
)
from act.back_end.moe.hz_routing import (
    analyze_candidates,
    analyze_topk_sets,
    condition_topk_membership,
    guarded_input_domain,
)
from act.back_end.moe.route_a import RouteAEngine
from act.back_end.moe.routing import (
    interval_candidate_mask,
    pairwise_margin_bounds,
    top2_sigmoid_gate_range,
)
from act.back_end.moe.schema import GateKind, OutputLevelMoESpec
from act.back_end.moe.verifier import verify_output_gate_elimination
from act.back_end.solver.solver_hz import (
    hz_add_const,
    hz_add_output_inequalities,
    hz_compute_bounds,
    hz_from_bounds,
    hz_multiply,
    hz_support_bounds,
    sparse_hz_from_bounds,
    sparse_hz_linear,
    sparse_hz_to_dense,
)
from act.config.config import HybridZConfig
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyResult, VerifyStatus


class _ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value, dtype=torch.float64))

    def forward(self, x):
        return self.value.unsqueeze(0).expand(x.shape[0], -1)


class MoEConcreteTests(unittest.TestCase):
    def test_selected_softmax_is_convex_combination(self):
        router = nn.Linear(2, 3, bias=True, dtype=torch.float64)
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64))
        spec = OutputLevelMoESpec(
            num_experts=3,
            top_k=2,
            gate=GateKind.SELECTED_SOFTMAX,
        )
        from act.back_end.moe.model import OutputLevelMoE

        model = OutputLevelMoE(
            router,
            [_ConstantExpert([1.0, 0.0]), _ConstantExpert([0.0, 1.0]), _ConstantExpert([9.0, 9.0])],
            spec,
        )
        output, decision = model.forward_with_routing(torch.zeros(4, 2, dtype=torch.float64))
        self.assertTrue(torch.equal(decision.indices, torch.tensor([[0, 1]]).expand(4, -1)))
        self.assertTrue(torch.allclose(decision.weights.sum(-1), torch.ones(4, dtype=torch.float64)))
        self.assertTrue(torch.allclose(output, decision.weights))

    def test_controlled_factory_shape(self):
        model = build_output_moe(
            OutputMoEFactoryConfig(
                input_shape=(1, 4, 4),
                num_classes=5,
                num_experts=4,
                top_k=2,
                gate=GateKind.SELECTED_SOFTMAX,
                seed=3,
            )
        ).double()
        output = model(torch.randn(2, 1, 4, 4, dtype=torch.float64))
        self.assertEqual(tuple(output.shape), (2, 5))


class RouterMathTests(unittest.TestCase):
    def test_interval_candidates_keep_ties(self):
        bounds = Bounds(
            lb=torch.tensor([[0.0, 0.0, -3.0]]),
            ub=torch.tensor([[1.0, 1.0, -2.0]]),
        )
        mask = interval_candidate_mask(bounds, top_k=1)
        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_pairwise_and_monotone_gate_ranges(self):
        scores = Bounds(
            lb=torch.tensor([[0.0, 1.0]]),
            ub=torch.tensor([[2.0, 3.0]]),
        )
        margins = pairwise_margin_bounds(scores)
        self.assertEqual(float(margins.lb[0, 0, 1]), -1.0)
        self.assertEqual(float(margins.ub[0, 0, 1]), 3.0)
        gate = top2_sigmoid_gate_range(Bounds(torch.tensor([-1.0]), torch.tensor([3.0])))
        self.assertTrue(torch.allclose(gate.lb, torch.sigmoid(torch.tensor([-1.0]))))
        self.assertTrue(torch.allclose(gate.ub, torch.sigmoid(torch.tensor([3.0]))))


class HZRoutingTests(unittest.TestCase):
    @staticmethod
    def _correlated_dense_router():
        input_hz = hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            torch.float64,
            torch.device("cpu"),
            track_ids=True,
        )
        router = hz_multiply(
            input_hz,
            torch.tensor([[1.0], [-1.0], [0.0]], dtype=torch.float64),
        )
        router = hz_add_const(router, torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64))
        return input_hz, router

    def test_exact_candidate_set_uses_correlation(self):
        input_hz, router = self._correlated_dense_router()
        report = analyze_candidates(router, 1, input_hz=input_hz, time_limit_per_expert=5.0)
        self.assertEqual(report.candidates, (0, 1))
        self.assertEqual(report.infeasible, (2,))
        self.assertTrue(report.minimal)

    def test_exact_unordered_top2_sets_use_inclusive_legality(self):
        _input_hz, router = self._correlated_dense_router()
        report = analyze_topk_sets(router, 2, time_limit_per_set=5.0)
        self.assertEqual(report.feasible, ((0, 1),))
        self.assertTrue(report.exact)

    def test_guard_is_transferred_to_input(self):
        input_hz, router = self._correlated_dense_router()
        guarded = guarded_input_domain(input_hz, router, expert=0, top_k=1)
        bounds = hz_compute_bounds(guarded.hz, exact=True)
        self.assertGreaterEqual(float(bounds.lb.item()), -1e-7)
        self.assertLessEqual(float(bounds.ub.item()), 1.0 + 1e-7)

    def test_top2_member_encoding_adds_at_most_e_minus_one_binaries(self):
        scores = hz_from_bounds(
            Bounds(torch.full((1, 3), -1.0), torch.full((1, 3), 1.0)),
            torch.float64,
            torch.device("cpu"),
            track_ids=True,
        )
        membership = condition_topk_membership(scores, expert=0, top_k=2)
        self.assertEqual(membership.selection_binaries, 2)
        self.assertLessEqual(membership.selection_binaries, 2)

    def test_sparse_guard_preserves_exact_frame(self):
        input_hz = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=11,
        )
        router = sparse_hz_linear(
            input_hz,
            torch.tensor([[1.0], [-1.0], [0.0]], dtype=torch.float64).numpy(),
            torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64).numpy(),
        )
        guarded = guarded_input_domain(input_hz, router, expert=0, top_k=1)
        self.assertEqual(guarded.hz.frame_id, 11)
        self.assertTrue(guarded.hz.exact)
        exact_bounds = hz_compute_bounds(sparse_hz_to_dense(guarded.hz), exact=True)
        self.assertGreaterEqual(float(exact_bounds.lb.item()), -1e-7)

    def test_constraint_aware_support_uses_sparse_guard(self):
        domain = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=19,
        )
        guarded = hz_add_output_inequalities(
            domain,
            torch.tensor([[-1.0]], dtype=torch.float64),
            torch.tensor([-0.5], dtype=torch.float64),
        )
        fast = hz_compute_bounds(sparse_hz_to_dense(guarded), exact=False)
        self.assertLess(float(fast.lb.item()), 0.0)
        support = hz_support_bounds(
            guarded,
            [0],
            time_limit=5.0,
            relax_binaries=True,
        )
        self.assertGreater(float(support.bounds.lb.item()), 0.49)
        self.assertLessEqual(float(support.bounds.ub.item()), 1.00000001)

    def test_guarded_support_precedes_relu_binary_allocation(self):
        domain = sparse_hz_from_bounds(
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            frame_id=23,
        )
        guarded = hz_add_output_inequalities(
            domain,
            torch.tensor([[-1.0]], dtype=torch.float64),
            torch.tensor([-0.5], dtype=torch.float64),
        )
        tf = HybridzTF(
            config=HybridZConfig(
                guarded_support_enabled=True,
                guarded_support_lp_neurons=1,
                guarded_support_lp_time_limit=5.0,
            )
        )
        tf.set_entry_hz(guarded)
        tf._sparse_frame_widths[23] = (guarded.n_cont, guarded.n_bin)
        layer = type("Layer", (), {"id": 101})()
        output, reason = _sparse_apply_relu(
            layer,
            guarded,
            Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]])),
            tf,
        )
        self.assertIsNone(reason)
        self.assertEqual(output.n_bin, guarded.n_bin)
        stats = tf.guarded_support_stats()[0]
        self.assertEqual(stats["fast_unstable"], 1)
        self.assertEqual(stats["after_lp_unstable"], 0)
        self.assertEqual(stats["after_milp_unstable"], 0)


class SchedulerTests(unittest.TestCase):
    def test_weighted_expert_violation_is_unknown(self):
        input_hz, router = HZRoutingTests._correlated_dense_router()
        candidates = analyze_candidates(router, 1, input_hz=input_hz, time_limit_per_expert=5.0)
        spec = OutputLevelMoESpec(
            num_experts=3,
            top_k=1,
            gate=GateKind.SWITCH_PROB,
            normalized=False,
        )

        def verify(expert, _branch):
            status = VerifyStatus.FALSIFIED if expert == 0 else VerifyStatus.CERTIFIED
            return VerifyResult(status)

        result = verify_output_gate_elimination(
            spec,
            candidates,
            verify,
            property_is_convex_cone=True,
        )
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)


class RouteAIntegrationTests(unittest.TestCase):
    def test_act_hyzor_route_a_end_to_end(self):
        model = build_output_moe(
            OutputMoEFactoryConfig(
                input_shape=(2,),
                num_classes=2,
                num_experts=3,
                top_k=1,
                gate=GateKind.HARD_TOP1,
                router_hidden=(),
                expert_hidden=(),
                seed=1,
            )
        ).double()
        with torch.no_grad():
            router_linear = model.router[1]
            router_linear.weight.zero_()
            router_linear.bias.copy_(torch.tensor([2.0, 0.0, -2.0], dtype=torch.float64))
            for expert in model.experts:
                linear = expert[1]
                linear.weight.zero_()
                linear.bias.copy_(torch.tensor([2.0, 0.0], dtype=torch.float64))
        center = torch.zeros(1, 2, dtype=torch.float64)
        program = build_act_moe_program(
            model,
            center=center,
            lower=center - 0.1,
            upper=center + 0.1,
            output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0])),
        )
        report = RouteAEngine(
            program,
            concrete_model=model,
            expert_models=tuple(model.experts),
            time_limit_per_route=5.0,
        ).run()
        self.assertEqual(report.router.candidates.candidates, (0,))
        self.assertEqual(report.result.status, VerifyStatus.CERTIFIED)


if __name__ == "__main__":
    unittest.main()
