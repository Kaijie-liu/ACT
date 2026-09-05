"""Unit tests for numerical checkpoint checks in the AdvMoE audit."""

from __future__ import annotations

import unittest

import torch

from act.pipeline.moe.audit_advmoe_training import floating_tensor_summary


class AuditAdvMoeTrainingTests(unittest.TestCase):
    def test_nested_finite_state_is_counted(self) -> None:
        summary = floating_tensor_summary(
            {"state": [{"momentum": torch.tensor([1.0, -2.0])}]}
        )
        self.assertTrue(summary["all_finite"])
        self.assertEqual(summary["elements"], 2)
        self.assertEqual(summary["nonfinite_tensors"], 0)

    def test_nan_and_infinity_fail_the_gate(self) -> None:
        summary = floating_tensor_summary(
            {"weight": torch.tensor([float("nan"), float("inf"), 1.0])}
        )
        self.assertFalse(summary["all_finite"])
        self.assertEqual(summary["finite_elements"], 1)
        self.assertEqual(summary["nan_elements"], 1)
        self.assertEqual(summary["inf_elements"], 1)


if __name__ == "__main__":
    unittest.main()
