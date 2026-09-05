from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from .advmoe_training_supervisor import (
    NonfiniteCheckpointError,
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

    def test_successor_command_changes_only_run_identity(self) -> None:
        configs = Path(__file__).parent / "configs"
        r1 = json.loads((configs / "advmoe_training_seed0_r1.json").read_text(encoding="utf-8"))
        successors = [
            json.loads((configs / name).read_text(encoding="utf-8"))
            for name in ("advmoe_training_seed0_r2.json", "advmoe_training_seed0_r3.json")
        ]
        commands = [build_training_command(config) for config in (r1, *successors)]
        for command in commands:
            command[command.index("--data-dir") + 1] = "<RUN_DATA_ROOT>"
            command[command.index("--exp-identifier") + 1] = "<RUN_IDENTITY>"
        for command in commands[1:]:
            self.assertEqual(commands[0], command)

    def test_compatibility_command_wraps_unchanged_official_arguments(self) -> None:
        configs = Path(__file__).parent / "configs"
        official = json.loads(
            (configs / "advmoe_training_seed0_r3.json").read_text(encoding="utf-8")
        )
        compatibility = json.loads(
            (configs / "advmoe_training_seed0_compat_r1.json").read_text(
                encoding="utf-8"
            )
        )
        official_command = build_training_command(official)
        compatibility_command = build_training_command(compatibility)
        self.assertIn("act.pipeline.moe.advmoe_compatibility_train", compatibility_command)
        official_arguments = official_command[2:]
        compatibility_arguments = compatibility_command[
            compatibility_command.index("--") + 1 :
        ]
        for arguments in (official_arguments, compatibility_arguments):
            arguments[arguments.index("--data-dir") + 1] = "<RUN_DATA_ROOT>"
            arguments[arguments.index("--exp-identifier") + 1] = "<RUN_IDENTITY>"
        self.assertEqual(official_arguments, compatibility_arguments)

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
            existing = snapshot_checkpoint(live, snapshots)
            self.assertIsNotNone(existing)
            assert existing is not None
            self.assertTrue(existing["existing"])
            self.assertEqual(existing["size_bytes"], Path(existing["path"]).stat().st_size)
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

    def test_nonfinite_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            live = root / "checkpoint.pth.tar"
            torch.save(
                {
                    "epoch": 1,
                    "state_dict": {"x": torch.tensor([1.0])},
                    "router": {"x": torch.tensor([float("nan")])},
                    "optimizer": {},
                    "router_optimizer": {},
                },
                live,
            )
            with self.assertRaisesRegex(NonfiniteCheckpointError, "router state"):
                snapshot_checkpoint(live, root / "snapshots")

    def test_source_change_during_snapshot_is_retryable(self) -> None:
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
            original_copy2 = __import__("shutil").copy2

            def copy_then_rewrite(source: Path, destination: Path) -> None:
                original_copy2(source, destination)
                payload["epoch"] = 4
                torch.save(payload, source)

            with mock.patch(
                "act.pipeline.moe.advmoe_training_supervisor.shutil.copy2",
                side_effect=copy_then_rewrite,
            ):
                self.assertIsNone(snapshot_checkpoint(live, snapshots))
            self.assertFalse((snapshots / "epoch_003.pth.tar").exists())

if __name__ == "__main__":
    unittest.main()
