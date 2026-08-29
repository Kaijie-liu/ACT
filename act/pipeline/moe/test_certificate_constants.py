import math
import unittest

import torch
import torch.nn as nn

from act.pipeline.moe.certificate_constants import (
    ConstantProvider,
    ConstantStatus,
    OutputReading,
    RouterReading,
    ScalarConstant,
    Theorem54Constants,
    author_unspecified_constants,
    batchnorm_linf_operator_norm,
    conv2d_linf_operator_upper,
    empirical_gradient_linf_estimates,
    evaluate_theorem54_paper_formula,
    hard_argmax_router_constants,
    linear_linf_operator_norm,
    module_linf_lipschitz_upper,
    official_cifar_resnet18_logit_bounds,
    residual_linf_lipschitz_upper,
    sound_probability_expert_constant,
    sound_softmax_router_constants,
)


class CertificateConstantTests(unittest.TestCase):
    def test_linear_linf_norm_is_exact_row_sum(self):
        layer = nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, -2.0], [4.0, 0.5]]))
        self.assertEqual(linear_linf_operator_norm(layer), 4.5)

    def test_conv_bound_is_kernel_row_sum(self):
        layer = nn.Conv2d(2, 2, 2, bias=False).double()
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[0].fill_(0.25)
            layer.weight[1].fill_(-0.5)
        self.assertEqual(conv2d_linf_operator_upper(layer), 4.0)

    def test_batchnorm_requires_eval_and_uses_fixed_scale(self):
        layer = nn.BatchNorm1d(2).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([2.0, 1.0]))
            layer.running_var.copy_(torch.tensor([4.0, 1.0]))
        with self.assertRaises(ValueError):
            batchnorm_linf_operator_norm(layer)
        layer.eval()
        expected = max(2.0 / math.sqrt(4.0 + layer.eps), 1.0 / math.sqrt(1.0 + layer.eps))
        self.assertAlmostEqual(batchnorm_linf_operator_norm(layer), expected)

    def test_sequential_and_residual_composition(self):
        first = nn.Linear(2, 2, bias=False).double()
        second = nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            first.weight.copy_(torch.eye(2) * 2.0)
            second.weight.copy_(torch.tensor([[1.0, -1.0]]))
        path = nn.Sequential(first, nn.ReLU(), second)
        self.assertEqual(module_linf_lipschitz_upper(path), 4.0)
        self.assertEqual(residual_linf_lipschitz_upper(path), 5.0)

    def test_unknown_module_is_rejected(self):
        class AddsSkip(nn.Module):
            def forward(self, x):
                return x + x

        with self.assertRaises(TypeError):
            module_linf_lipschitz_upper(AddsSkip())

    def test_official_resnet_structural_adapter_composes_residuals(self):
        class BasicBlockShape(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(1)
                self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(1)
                self.shortcut = nn.Sequential()

        class ResNetShape(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(1)
                self.layer1 = nn.Sequential(BasicBlockShape())
                self.layer2 = nn.Sequential(BasicBlockShape())
                self.layer3 = nn.Sequential(BasicBlockShape())
                self.layer4 = nn.Sequential(BasicBlockShape())
                self.linear = nn.Linear(1, 2, bias=False)

        model = ResNetShape().double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.linear.weight.copy_(torch.tensor([[2.0], [-3.0]]))
        model.eval()
        vector, rows = official_cifar_resnet18_logit_bounds(model)
        self.assertGreater(vector, 0.0)
        self.assertAlmostEqual(rows[1] / rows[0], 1.5)
        self.assertEqual(vector, rows[1])

    def test_probability_constants_are_formal(self):
        lipschitz, upper = sound_probability_expert_constant(8.0, expert_index=2)
        self.assertEqual(lipschitz.value, 4.0)
        self.assertEqual(upper.value, 1.0)
        self.assertTrue(lipschitz.formal)
        self.assertTrue(upper.formal)

    def test_hard_router_is_not_assigned_zero(self):
        constants = hard_argmax_router_constants(num_experts=2)
        self.assertTrue(all(item.value is None for item in constants))
        self.assertTrue(all(item.status == ConstantStatus.NOT_APPLICABLE for item in constants))

    def test_empirical_gradient_is_never_formal(self):
        layer = nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, -2.0], [4.0, 0.5]]))
        estimates = empirical_gradient_linf_estimates(
            layer, torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
        )
        self.assertEqual([item.value for item in estimates], [3.0, 4.5])
        self.assertTrue(all(item.status == ConstantStatus.DIAGNOSTIC_ONLY for item in estimates))

    def test_paper_formula_with_sound_constants(self):
        router = sound_softmax_router_constants(2.0, num_experts=2)
        expert = tuple(
            sound_probability_expert_constant(value, expert_index=index)[0]
            for index, value in enumerate((4.0, 6.0))
        )
        upper = tuple(
            sound_probability_expert_constant(1.0, expert_index=index)[1]
            for index in range(2)
        )
        constants = Theorem54Constants(
            router,
            expert,
            upper,
            OutputReading.PROBABILITY,
            RouterReading.CONTINUOUS_SOFTMAX,
        )
        result = evaluate_theorem54_paper_formula(
            [0.7, 0.2, 0.1], 0, [0.25, 0.75], constants
        )
        self.assertEqual(result.status, ConstantStatus.FORMAL_BOUND)
        self.assertAlmostEqual(result.clean_margin, 0.5)
        self.assertAlmostEqual(result.denominator, 4.75)
        self.assertAlmostEqual(result.radius, 0.5 / 4.75)

    def test_unspecified_provider_cannot_produce_radius(self):
        missing = author_unspecified_constants(num_experts=2, quantity_prefix="r_R")
        formal = tuple(
            ScalarConstant(
                1.0,
                ConstantProvider.SOUND_GLOBAL_SPECTRAL,
                ConstantStatus.FORMAL_BOUND,
                f"value[{index}]",
                "test",
            )
            for index in range(2)
        )
        constants = Theorem54Constants(
            missing,
            formal,
            formal,
            OutputReading.PROBABILITY,
            RouterReading.CONTINUOUS_SOFTMAX,
        )
        result = evaluate_theorem54_paper_formula([0.6, 0.4], 0, [0.5, 0.5], constants)
        self.assertIsNone(result.radius)
        self.assertEqual(result.status, ConstantStatus.NOT_FORMALLY_INSTANTIATED)

    def test_unnormalized_gate_is_not_applicable(self):
        formal = tuple(
            ScalarConstant(
                1.0,
                ConstantProvider.SOUND_GLOBAL_SPECTRAL,
                ConstantStatus.FORMAL_BOUND,
                f"value[{index}]",
                "test",
            )
            for index in range(2)
        )
        constants = Theorem54Constants(
            formal,
            formal,
            formal,
            OutputReading.PROBABILITY,
            RouterReading.CONTINUOUS_SOFTMAX,
        )
        result = evaluate_theorem54_paper_formula([0.6, 0.4], 0, [0.2, 0.2], constants)
        self.assertEqual(result.status, ConstantStatus.NOT_APPLICABLE)
        self.assertIsNone(result.radius)


if __name__ == "__main__":
    unittest.main()
