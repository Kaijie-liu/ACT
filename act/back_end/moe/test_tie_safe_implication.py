import unittest

import torch
import torch.nn as nn

from act.back_end.moe.tie_safe_implication import (
    TieSafeTop1Implication,
    audit_eta_overcheck_band,
    relu_pairwise_max,
    relu_pairwise_min,
    top1_branch_guard_values,
)


class TieSafeImplicationTests(unittest.TestCase):
    def _module(self, *, eta=1e-7):
        router = nn.Linear(1, 2, bias=False).double()
        expert = nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            router.weight.copy_(torch.tensor([[1.0], [1.0]], dtype=torch.double))
            expert.weight.fill_(-1.0)
        return TieSafeTop1Implication(
            router, expert, 0, [[1.0]], [0.0], eta=eta
        )

    def test_relu_max_and_min_match_direct_reductions(self):
        torch.manual_seed(12)
        values = torch.randn(20, 7, dtype=torch.double)
        torch.testing.assert_close(relu_pairwise_max(values), values.max(dim=1).values)
        torch.testing.assert_close(relu_pairwise_min(values), values.min(dim=1).values)

    def test_zero_margin_compiler_is_unsound_at_tie_but_eta_is_not(self):
        module = self._module(eta=1e-7)
        guard, safety, compiled = module.forward_components(
            torch.tensor([[1.0]], dtype=torch.double)
        )
        naive = torch.maximum(guard, safety)
        self.assertEqual(float(guard.item()), 0.0)
        self.assertLess(float(safety.item()), 0.0)
        self.assertEqual(float(naive.item()), 0.0)
        self.assertLess(float(compiled.item()), 0.0)

    def test_every_accepted_legal_branch_has_safe_expert(self):
        generator = torch.Generator().manual_seed(17)
        guard = -torch.rand(
            1000, generator=generator, dtype=torch.double, device="cpu"
        )
        safety = torch.randn(
            1000, generator=generator, dtype=torch.double, device="cpu"
        )
        eta = 1e-7
        compiled = torch.maximum(guard - eta, safety)
        accepted = compiled >= 0.0
        self.assertTrue(bool(torch.all(safety[accepted] >= 0.0)))

    def test_only_nonmember_eta_band_is_overchecked(self):
        eta = 0.1
        safety = torch.tensor(-1.0)
        self.assertLess(float(torch.maximum(torch.tensor(0.05) - eta, safety)), 0.0)
        self.assertGreaterEqual(
            float(torch.maximum(torch.tensor(eta) - eta, safety)), 0.0
        )

    def test_branch_guard_supports_varying_experts(self):
        scores = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 2.5]])
        guards = top1_branch_guard_values(scores, torch.tensor([0, 1]))
        torch.testing.assert_close(guards, torch.tensor([-1.0, -0.5]))

    def test_eta_band_audit_counts_branch_and_sample_units(self):
        scores = torch.tensor(
            [[1.0, 1.0 - 0.5e-7, 0.0], [2.0, 0.0, -1.0]],
            dtype=torch.double,
        )
        audit = audit_eta_overcheck_band(
            scores, eta=1e-7, boundary_tolerance=1e-9
        )
        self.assertEqual(audit.samples, 2)
        self.assertEqual(audit.branch_obligations, 6)
        self.assertEqual(audit.legal_branches, 2)
        self.assertEqual(audit.overcheck_branches, 1)
        self.assertEqual(audit.overcheck_samples, 1)
        self.assertEqual(audit.numerical_boundary_branches, 0)
        self.assertEqual(audit.as_dict()["overcheck_definition"], "0 < g_i < eta")

    def test_forward_matches_direct_guard_and_property_semantics(self):
        router = nn.Linear(2, 3, bias=True).double()
        expert = nn.Linear(2, 2, bias=True).double()
        torch.manual_seed(23)
        with torch.no_grad():
            router.weight.normal_()
            router.bias.normal_()
            expert.weight.normal_()
            expert.bias.normal_()
        matrix = torch.tensor([[1.0, -1.0], [-0.5, 2.0]], dtype=torch.double)
        offset = torch.tensor([0.2, -0.1], dtype=torch.double)
        module = TieSafeTop1Implication(
            router, expert, 1, matrix, offset, eta=1e-7
        )
        x = torch.randn(16, 2, dtype=torch.double)
        guard, safety, compiled = module.forward_components(x)
        scores = router(x)
        expected_guard = torch.maximum(scores[:, 0], scores[:, 2]) - scores[:, 1]
        expected_safety = (expert(x) @ matrix.T + offset).min(dim=1).values
        expected = torch.maximum(expected_guard - module.eta, expected_safety)
        torch.testing.assert_close(guard, expected_guard)
        torch.testing.assert_close(safety, expected_safety)
        torch.testing.assert_close(compiled, expected)
        torch.testing.assert_close(module(x).squeeze(1), expected)


if __name__ == "__main__":
    unittest.main()
