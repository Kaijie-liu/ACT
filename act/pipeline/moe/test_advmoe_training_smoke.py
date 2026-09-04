from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from .advmoe_training_smoke import (
    _changed_parameter_count,
    _inside_workspace,
    _official_one_batch_loader_item,
    _state_hash,
    build_official_args,
)
from .audit_advmoe_training_smoke import audit


class AdvMoETrainingSmokeTest(unittest.TestCase):
    def test_paths_must_resolve_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            workspace = Path(directory)
            self.assertTrue(_inside_workspace(workspace / "a", workspace))
            self.assertFalse(_inside_workspace(Path("/tmp/outside"), workspace))

    def test_state_hash_is_key_order_independent_and_detects_change(self) -> None:
        first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
        self.assertEqual(_state_hash(first), _state_hash(second))
        self.assertNotEqual(_state_hash(first), _state_hash(changed))
        self.assertEqual(_changed_parameter_count(first, second), 0)
        self.assertEqual(_changed_parameter_count(first, changed), 1)

    def test_frozen_config_maps_to_official_arguments(self) -> None:
        config_path = Path(__file__).parent / "configs" / "advmoe_training_seed0_r1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        arguments = build_official_args(config)
        self.assertEqual(arguments.arch, "resnet18_cifar_moe")
        self.assertEqual(arguments.n_expert, 2)
        self.assertEqual(arguments.ratio, 0.5)
        self.assertEqual(arguments.epochs, 100)
        self.assertEqual(arguments.seed, 0)
        self.assertEqual(arguments.num_steps, 2)
        self.assertEqual(arguments.num_steps_test, 10)

    def test_one_batch_smoke_matches_released_list_only_helper(self) -> None:
        images = torch.zeros(2, 3, 32, 32)
        targets = torch.zeros(2, dtype=torch.long)
        item = _official_one_batch_loader_item(images, targets)
        self.assertIsInstance(item, list)
        self.assertIs(item[0], images)
        self.assertIs(item[1], targets)

    def test_audit_fails_closed_on_false_executable_check(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "official_source": {"commit": "c", "tree": "t"},
                "preflight_gates": {"required_smoke_checks": ["x"]},
                "run": {"batch_size": 128},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = {
                "status": "PASS",
                "config": {},
                "official_source": {"commit": "c", "tree": "t", "clean": True},
                "checks": {"x": False},
                "batch": {"size": 128},
                "device": {"capability": [12, 0]},
                "resume": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": "wrong",
                    "maximum_logit_error": 0.0,
                    "maximum_router_score_error": 0.0,
                },
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            report = audit(config_path, result_path)
            self.assertEqual(report["status"], "FAIL")
            self.assertGreaterEqual(report["issue_count"], 3)


if __name__ == "__main__":
    unittest.main()
