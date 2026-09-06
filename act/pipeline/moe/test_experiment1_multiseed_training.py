from __future__ import annotations

import json
import unittest
from pathlib import Path

from act.pipeline.moe.experiment1_multiseed_training import (
    DEFAULT_CONFIG,
    _training_command,
    _validate_config,
)
from act.pipeline.moe.audit_experiment1_multiseed_training import _check_metrics


class Experiment1MultiseedTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_seeds_and_training_config(self) -> None:
        _validate_config(self.config)
        self.assertEqual([row["seed"] for row in self.config["seeds"]], [1, 2])

    def test_command_preserves_registered_semantics(self) -> None:
        training = self.config["training"]
        command = _training_command(
            Path(self.config["python"]),
            training,
            2,
            Path(self.config["seeds"][1]["checkpoint"]),
        )
        rendered = " ".join(command)
        self.assertIn("--top-k 2", rendered)
        self.assertIn("--gate selected_softmax", rendered)
        self.assertIn("--router-hidden 128", rendered)
        self.assertIn("--expert-hidden 256 128", rendered)
        self.assertIn("--balance-loss switch", rendered)
        self.assertIn("--balance-coefficient 0.1", rendered)
        self.assertIn("--seed 2", rendered)
        self.assertIn("--no-download", rendered)

    def test_rejects_seed_substitution(self) -> None:
        self.config["seeds"][1]["seed"] = 3
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            _validate_config(self.config)

    def test_rejects_hyperparameter_drift(self) -> None:
        self.config["training"]["balance_coefficient"] = 0.05
        with self.assertRaisesRegex(RuntimeError, "balance_coefficient"):
            _validate_config(self.config)

    def test_metric_replay_detects_changed_route_count(self) -> None:
        metrics = {
            "accuracy": 0.5,
            "route_counts": [3, 1],
            "route_frequencies": [0.75, 0.25],
            "load_entropy": 0.5623351446188083,
            "effective_experts": 1.7547653506033232,
            "max_expert_load": 0.75,
            "min_expert_load": 0.25,
            "samples": 2,
        }
        changed = dict(metrics, route_counts=[2, 2])
        issues: list[str] = []
        _check_metrics(issues, "test", metrics, changed)
        self.assertEqual(issues, ["test route_counts differs"])


if __name__ == "__main__":
    unittest.main()
