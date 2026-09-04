from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from .advmoe_training_supervisor import (
    _checkpoint_epoch,
    build_training_command,
    snapshot_checkpoint,
)


class AdvMoETrainingSupervisorTest(unittest.TestCase):
    def test_command_preserves_frozen_official_configuration(self) -> None:
        config_path = Path(__file__).parent / "configs" / "advmoe_training_seed0_r1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        command = build_training_command(config)
        joined = " ".join(command)
        for fragment in (
            "--arch resnet18_cifar_moe",
            "--n-expert 2",
            "--ratio 0.5",
            "--seed 0",
            "--epochs 100",
            "--num-steps 2",
            "--num-steps-test 10",
        ):
            self.assertIn(fragment, joined)
        self.assertNotIn("--normalize", command)

    def test_snapshot_is_immutable_and_epoch_addressed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            live = root / "checkpoint.pth.tar"
            snapshots = root / "snapshots"
            payload = {
                "epoch": 3,
                "state_dict": {"x": torch.tensor([1.0])},
                "router": {"x": torch.tensor([2.0])},
                "optimizer": {},
                "router_optimizer": {},
            }
            torch.save(payload, live)
            first = snapshot_checkpoint(live, snapshots)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first["epoch"], 3)
            self.assertFalse(first["existing"])
            self.assertEqual(_checkpoint_epoch(Path(first["path"])), 3)
            payload["state_dict"]["x"] = torch.tensor([9.0])
            torch.save(payload, live)
            with self.assertRaisesRegex(RuntimeError, "rewritten with different"):
                snapshot_checkpoint(live, snapshots)

    def test_partial_checkpoint_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            live = root / "checkpoint.pth.tar"
            live.write_bytes(b"partial")
            self.assertIsNone(snapshot_checkpoint(live, root / "snapshots"))


if __name__ == "__main__":
    unittest.main()
