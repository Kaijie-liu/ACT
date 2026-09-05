"""Regression tests for accepted AdvMoE endpoint telemetry helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.advmoe_training_endpoint_telemetry import (
    _clean_accuracy,
    _distribution,
)
from act.pipeline.moe.audit_advmoe_training_endpoint_telemetry import (
    json_nonfinite_paths,
)


class _ThresholdClassifier(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scalar = inputs.flatten(1).mean(dim=1)
        return torch.stack((1.0 - scalar, scalar), dim=1)


class AdvMoeTrainingEndpointTelemetryTests(unittest.TestCase):
    def test_distribution_uses_population_statistics(self) -> None:
        result = _distribution(np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(result["minimum"], 1.0)
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["maximum"], 3.0)
        self.assertAlmostEqual(
            result["standard_deviation"], np.sqrt(2.0 / 3.0)
        )

    def test_distribution_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            _distribution(np.asarray([1.0, np.nan]))

    def test_json_nonfinite_paths_are_explicit(self) -> None:
        paths = json_nonfinite_paths({"row": [1.0, float("nan")]})
        self.assertEqual(paths, ["$.row[1]"])

    def test_clean_accuracy_batches_without_changing_denominator(self) -> None:
        inputs = np.asarray(
            [0.0, 0.8, 0.2, 1.0], dtype=np.float32
        ).reshape(4, 1, 1, 1)
        labels = np.asarray([0, 1, 1, 1], dtype=np.int64)
        accuracy = _clean_accuracy(
            _ThresholdClassifier(),
            inputs,
            labels,
            device=torch.device("cpu"),
            batch_size=3,
        )
        self.assertEqual(accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
