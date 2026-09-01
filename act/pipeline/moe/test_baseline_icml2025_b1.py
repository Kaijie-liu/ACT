"""Unit tests for the isolated official RT-ER B1 execution wrappers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from act.pipeline.moe.baseline_icml2025_b1_smoke import AUTHOR_BATCH_SIZE
from act.pipeline.moe.baseline_icml2025_b1_supervisor import (
    CHECKPOINT_EPOCHS,
    REPRODUCTION_LABEL,
    _checkpoint_epoch,
    epoch_checkpoint_name,
    epoch_directory_name,
    metrics_for_epoch,
    official_launcher_command,
    telemetry_command,
)
from act.pipeline.moe.baseline_icml2025_official_launcher import author_arguments
from act.pipeline.moe.baseline_icml2025_b1_seed1_orchestrator import resource_snapshot


class OfficialLauncherTest(unittest.TestCase):
    def test_author_arguments_preserve_paper_code_configuration(self) -> None:
        self.assertEqual(
            author_arguments(130, 0),
            [
                "--net",
                "res18_moe",
                "--n_epochs",
                "130",
                "--beta",
                "6",
                "--bs",
                "512",
                "--lr",
                "0.0001",
                "--opt",
                "adam",
                "--nowandb",
            ],
        )
        self.assertEqual(AUTHOR_BATCH_SIZE, 512)

    def test_supervisor_launcher_command_uses_unbuffered_python(self) -> None:
        command = official_launcher_command(
            Path("/tmp/python"),
            seed=0,
            epochs=130,
            wandb_log=Path("/tmp/wandb.jsonl"),
            launcher_manifest=Path("/tmp/launcher.json"),
        )
        self.assertEqual(command[:2], ["/tmp/python", "-u"])
        self.assertEqual(command[-8:], [
            "--seed", "0", "--epochs", "130", "--wandb-log", "/tmp/wandb.jsonl",
            "--launcher-manifest", "/tmp/launcher.json",
        ])


class SupervisorIdentityTest(unittest.TestCase):
    def test_frozen_epoch_names(self) -> None:
        self.assertEqual(CHECKPOINT_EPOCHS, tuple(range(10, 131, 10)))
        self.assertEqual(epoch_checkpoint_name(10), "epoch_010.t7")
        self.assertEqual(epoch_directory_name(130), "epoch_130")
        with self.assertRaises(ValueError):
            epoch_checkpoint_name(11)

    def test_checkpoint_payload_uses_one_based_public_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.t7"
            torch.save({"epoch": 9, "net": {}}, path)
            self.assertEqual(_checkpoint_epoch(path), 10)

    def test_metrics_pair_validation_and_training_log(self) -> None:
        records = [
            {"event": "log", "payload": {"epoch": 8, "train_loss": 3.0}},
            {
                "event": "log",
                "payload": {"val_loss": 1.2, "val_sa": 41.0, "val_ra": 20.0},
            },
            {
                "event": "log",
                "payload": {
                    "epoch": 9,
                    "train_loss": 2.0,
                    "sa": 42.0,
                    "ra": 21.0,
                    "lr": 0.0001,
                },
            },
        ]
        metrics = metrics_for_epoch(records, 10)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["official_zero_based_epoch"], 9)
        self.assertEqual(metrics["validation"]["val_sa"], 41.0)
        self.assertEqual(metrics["training"]["sa"], 42.0)
        self.assertIsNone(metrics_for_epoch(records, 20))

    def test_telemetry_command_carries_all_artifact_identities(self) -> None:
        command = telemetry_command(
            Path("/tmp/act-python"),
            config=Path("/tmp/config.json"),
            checkpoint=Path("/tmp/epoch_010.t7"),
            output_dir=Path("/tmp/telemetry/epoch_010"),
            metrics=Path("/tmp/metrics/epoch_010.json"),
            seed=0,
            epoch=10,
            device="cuda",
        )
        self.assertIn("act.pipeline.moe.icml2025_route_telemetry", command)
        self.assertEqual(command[command.index("--epoch") + 1], "10")
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(command[command.index("--device") + 1], "cuda")

    def test_blackwell_telemetry_configs_are_single_seed_registered(self) -> None:
        root = Path("/data1/Kane/MOE/ACT") / "act/pipeline/moe/configs"
        for seed in (0, 1):
            config_path = root / f"icml2025_route_telemetry_blackwell_seed{seed}.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["label"], REPRODUCTION_LABEL)
            self.assertEqual(config["status"], "PREREGISTERED_NOT_RUN")
            self.assertEqual(config["training"]["seeds"], [seed])
            self.assertEqual(
                config["training"]["checkpoint_epochs"], list(CHECKPOINT_EPOCHS)
            )
            self.assertFalse(config["execution_gate"]["substituted_data_pipeline"])

    def test_seed1_launch_gate_requires_resources_clean_branch_and_sync(self) -> None:
        protocol = {
            "branch": "feat/moe-route-verification",
            "remote": "origin",
            "launch_resource_gate": {
                "device_index": 0,
                "minimum_free_memory_bytes": 30,
                "minimum_free_disk_bytes": 60,
            },
        }
        with (
            mock.patch(
                "act.pipeline.moe.baseline_icml2025_b1_seed1_orchestrator.gpu_memory_bytes",
                return_value=(31, 100),
            ),
            mock.patch(
                "act.pipeline.moe.baseline_icml2025_b1_seed1_orchestrator.os.statvfs",
                return_value=SimpleNamespace(f_bavail=70, f_frsize=1),
            ),
            mock.patch(
                "act.pipeline.moe.baseline_icml2025_b1_seed1_orchestrator._git",
                side_effect=["feat/moe-route-verification", "", "abc", "abc"],
            ),
        ):
            observed = resource_snapshot(protocol)
        self.assertTrue(observed["ready"])
        self.assertTrue(observed["worktree_clean"])
        self.assertTrue(observed["local_remote_synchronized"])

    def test_seed1_attempt_logs_are_namespaced_by_run_root(self) -> None:
        source = Path(
            "/data1/Kane/MOE/ACT/act/pipeline/moe/"
            "baseline_icml2025_b1_seed1_orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertIn('f"{run_root.name}_supervisor.log"', source)
        self.assertIn('f"{run_root.name}_landing_watch.log"', source)


if __name__ == "__main__":
    unittest.main()
