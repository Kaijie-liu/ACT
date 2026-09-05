"""Tests for the explicitly labeled AdvMoE numerical compatibility bridge."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from act.pipeline.moe.advmoe_softmax_underflow_bridge import (
    SoftmaxUnderflowGradientBridge,
)


def _native_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(q_logits, dim=1),
        F.softmax(p_logits, dim=1),
        reduction="sum",
    )


def _stable_logit_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(p_logits, dim=1)
    return (log_p.exp() * (log_p - F.log_softmax(q_logits, dim=1))).sum()


class AdvMoeSoftmaxUnderflowBridgeTests(unittest.TestCase):
    def test_finite_regime_value_and_gradients_match_native(self) -> None:
        p_native = torch.tensor([[1.5, -0.5], [-0.2, 0.3]], requires_grad=True)
        q_native = torch.tensor([[0.1, 0.2], [0.7, -0.1]], requires_grad=True)
        native_value = _native_kl(p_native, q_native)
        native_gradients = torch.autograd.grad(native_value, (p_native, q_native))

        p_bridge = p_native.detach().clone().requires_grad_()
        q_bridge = q_native.detach().clone().requires_grad_()
        with SoftmaxUnderflowGradientBridge() as bridge:
            bridge_value = _native_kl(p_bridge, q_bridge)
            bridge_gradients = torch.autograd.grad(
                bridge_value, (p_bridge, q_bridge)
            )
        self.assertTrue(torch.equal(native_value, bridge_value))
        self.assertTrue(torch.equal(native_gradients[0], bridge_gradients[0]))
        self.assertTrue(torch.equal(native_gradients[1], bridge_gradients[1]))
        self.assertEqual(bridge.replaced_elements, 0)

    def test_extreme_regime_matches_stable_logit_expression(self) -> None:
        p_bridge = torch.tensor([[200.0, -200.0]], requires_grad=True)
        q_bridge = torch.tensor([[0.0, 0.0]], requires_grad=True)
        with SoftmaxUnderflowGradientBridge() as bridge:
            bridge_value = _native_kl(p_bridge, q_bridge)
            bridge_gradients = torch.autograd.grad(
                bridge_value, (p_bridge, q_bridge)
            )

        p_stable = p_bridge.detach().clone().requires_grad_()
        q_stable = q_bridge.detach().clone().requires_grad_()
        stable_value = _stable_logit_kl(p_stable, q_stable)
        stable_gradients = torch.autograd.grad(stable_value, (p_stable, q_stable))
        self.assertTrue(torch.isfinite(bridge_gradients[0]).all())
        self.assertTrue(torch.isfinite(bridge_gradients[1]).all())
        self.assertTrue(torch.allclose(bridge_value, stable_value, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(bridge_gradients[0], stable_gradients[0]))
        self.assertTrue(torch.allclose(bridge_gradients[1], stable_gradients[1]))
        self.assertEqual(bridge.replaced_elements, 1)

    def test_unbridged_mutation_control_produces_nan(self) -> None:
        p_logits = torch.tensor([[200.0, -200.0]], requires_grad=True)
        q_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        gradients = torch.autograd.grad(_native_kl(p_logits, q_logits), p_logits)
        self.assertTrue(torch.isnan(gradients[0]).all())

    def test_nonfinite_gradient_at_positive_probability_fails_closed(self) -> None:
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        with SoftmaxUnderflowGradientBridge():
            probabilities = F.softmax(logits, dim=1)
            with self.assertRaisesRegex(RuntimeError, "positive probability"):
                probabilities.backward(torch.tensor([[float("nan"), 0.0]]))


if __name__ == "__main__":
    unittest.main()
