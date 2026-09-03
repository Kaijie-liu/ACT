"""Tests for the frozen B1 landing sequence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from act.pipeline.moe.baseline_icml2025_b1_endpoint import (
    _normalized_batch,
    _pixel_linf,
    _pixel_range,
)
from act.pipeline.moe.baseline_icml2025_b1_landing import (
    RetryableGpuLandingError,
    _artifact_subprocess_environment,
    endpoint_decisions,
    require_gpu_resource,
    validate_completed_epoch,
)
from act.pipeline.moe.baseline_icml2025_b1_landing_watch import (
    _attempt_rehearsal,
    _read_json_with_retries,
    _record_hook_failure,
    _staleness_record,
)
from act.pipeline.moe.baseline_icml2025_b1_smoke import _sha256


class B1LandingTests(unittest.TestCase):
    def test_endpoint_and_audit_subprocesses_disable_bytecode_writes(self) -> None:
        with mock.patch.dict(os.environ, {"PYTHONDONTWRITEBYTECODE": "0"}):
            environment = _artifact_subprocess_environment()
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")

    def test_rt_er_python_can_import_current_act_schema(self) -> None:
        completed = subprocess.run(
            [
                "/data1/Kane/MOE/envs/rt-er-blackwell/bin/python",
                "-c",
                "import act.back_end.layer_schema; print('IMPORT_OK')",
            ],
            cwd=Path("/data1/Kane/MOE/ACT"),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("IMPORT_OK", completed.stdout)

    def test_rt_er_python_imports_every_blackwell_b1_entrypoint(self) -> None:
        modules = [
            "act.pipeline.moe.baseline_icml2025_official_launcher",
            "act.pipeline.moe.baseline_icml2025_b1_endpoint",
            "act.pipeline.moe.audit_baseline_icml2025_b1_endpoint",
        ]
        statement = "; ".join(f"import {module}" for module in modules)
        completed = subprocess.run(
            [
                "/data1/Kane/MOE/envs/rt-er-blackwell/bin/python",
                "-c",
                statement + "; print('B1_ENTRYPOINTS_IMPORT_OK')",
            ],
            cwd=Path("/data1/Kane/MOE/ACT"),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("B1_ENTRYPOINTS_IMPORT_OK", completed.stdout)

    def test_act_python_imports_control_and_telemetry_entrypoints(self) -> None:
        modules = [
            "act.pipeline.moe.baseline_icml2025_b1_supervisor",
            "act.pipeline.moe.icml2025_route_telemetry",
            "act.pipeline.moe.baseline_icml2025_b1_landing",
            "act.pipeline.moe.baseline_icml2025_b1_landing_watch",
            "act.pipeline.moe.baseline_icml2025_b1_seed1_orchestrator",
        ]
        statement = "; ".join(f"import {module}" for module in modules)
        completed = subprocess.run(
            [
                "/data1/Kane/miniconda3/envs/act-py312/bin/python",
                "-c",
                statement + "; print('B1_CONTROL_ENTRYPOINTS_IMPORT_OK')",
            ],
            cwd=Path("/data1/Kane/MOE/ACT"),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("B1_CONTROL_ENTRYPOINTS_IMPORT_OK", completed.stdout)

    def test_endpoint_normalization_and_pixel_norm_use_official_255_domain(self) -> None:
        raw = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        raw[..., 0] = 125
        raw[..., 1] = 123
        raw[..., 2] = 114
        normalized = _normalized_batch(raw, torch.device("cpu"))
        self.assertEqual(normalized.dtype, torch.float16)
        expected = (
            torch.tensor([125.0, 123.0, 114.0], dtype=torch.float16)
            - torch.tensor([125.307, 122.961, 113.8575], dtype=torch.float16)
        ) / torch.tensor([51.5865, 50.847, 51.255], dtype=torch.float16)
        self.assertTrue(torch.equal(normalized[0, :, 0, 0], expected))
        adversarial = normalized.clone()
        adversarial[:, 0] += torch.tensor(8.0 / 51.5865, dtype=torch.float16)
        observed = _pixel_linf(normalized, adversarial)
        self.assertAlmostEqual(float(observed.item()), 8.0 / 255.0, places=4)
        minimum, maximum = _pixel_range(adversarial)
        self.assertGreaterEqual(float(minimum.item()), 0.0)
        self.assertLessEqual(float(maximum.item()), 255.0)

    def test_endpoint_decisions_use_inclusive_frozen_intervals(self) -> None:
        interpretation = {
            "primary_standard_accuracy_rule": {
                "inclusive_interval_percent": [72.81, 82.81],
                "inside": "SA_IN",
                "outside": "SA_OUT",
            },
            "secondary_pgd50_rule": {
                "inclusive_interval_percent": [64.09, 74.09],
                "inside": "RA_IN",
                "outside": "RA_OUT",
            },
        }
        passed = endpoint_decisions(72.81, 74.09, interpretation)
        self.assertEqual(passed["standard_accuracy_branch"], "SA_IN")
        self.assertEqual(passed["pgd50_accuracy_branch"], "RA_IN")
        self.assertFalse(passed["seed1_followup_required"])
        self.assertEqual(
            passed["pipeline_claim_status"],
            "EXISTENCE_SUPPORTED_BY_SEED0_NO_SEED1_REQUIRED",
        )
        failed = endpoint_decisions(50.0, 20.0, interpretation)
        self.assertEqual(failed["standard_accuracy_branch"], "SA_OUT")
        self.assertTrue(failed["seed1_followup_required"])
        self.assertEqual(
            failed["pipeline_claim_status"],
            "SEED1_REQUIRED_BEFORE_PIPELINE_LEVEL_FAILURE_WORDING",
        )
        second_failure = endpoint_decisions(
            50.0, 20.0, interpretation, seed=1
        )
        self.assertFalse(second_failure["seed1_followup_required"])
        self.assertEqual(
            second_failure["pipeline_claim_status"],
            "TWO_REGISTERED_RUNS_MISS_SA_INTERVAL_PIPELINE_LEVEL_WORDING_ALLOWED",
        )
        second_pass = endpoint_decisions(
            75.0, 70.0, interpretation, seed=1
        )
        self.assertEqual(
            second_pass["pipeline_claim_status"],
            "EXISTENCE_SUPPORTED_BY_SEED1_AFTER_SEED0_MISS",
        )

    def test_gpu_resource_gate_waits_below_frozen_minimum(self) -> None:
        protocol = {
            "gpu_resource_gate": {
                "device_index": 0,
                "minimum_free_memory_bytes": 30,
            }
        }
        with mock.patch(
            "act.pipeline.moe.baseline_icml2025_b1_landing.gpu_memory_bytes",
            return_value=(29, 100),
        ):
            with self.assertRaises(RetryableGpuLandingError):
                require_gpu_resource(protocol)
        with mock.patch(
            "act.pipeline.moe.baseline_icml2025_b1_landing.gpu_memory_bytes",
            return_value=(30, 100),
        ):
            observed = require_gpu_resource(protocol)
        self.assertEqual(observed["free_memory_bytes"], 30)

    def test_progress_json_read_tolerates_two_transient_partial_writes(self) -> None:
        path = Path("/data1/Kane/MOE/transient-progress.json")
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=['{"status":', '{"status":', '{"status":"RUNNING"}'],
        ) as read_text:
            observed = _read_json_with_retries(
                path, attempts=3, delay_seconds=0
            )
        self.assertEqual(observed["status"], "RUNNING")
        self.assertEqual(read_text.call_count, 3)

    def test_staleness_requires_progress_and_live_heartbeat_to_be_old(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            progress_path = root / "progress.json"
            heartbeat_path = root / "wandb.jsonl"
            metrics_path = root / "epoch.json"
            progress_path.write_text("{}\n", encoding="utf-8")
            heartbeat_path.write_text("{}\n", encoding="utf-8")
            metrics_path.write_text(
                json.dumps({"training": {"epoch_time": 100.0}}) + "\n",
                encoding="utf-8",
            )
            now = time.time()
            os.utime(progress_path, (now - 1000, now - 1000))
            protocol = {
                "staleness_detection": {
                    "epoch_duration_multiplier": 3,
                    "fallback_epoch_seconds": 100,
                    "minimum_staleness_seconds": 300,
                    "heartbeat_paths": [str(heartbeat_path)],
                }
            }
            progress = {"completed": [{"metrics": str(metrics_path)}]}
            active = _staleness_record(
                progress_path, progress, protocol, now=now
            )
            self.assertFalse(active["suspected"])
            os.utime(heartbeat_path, (now - 1000, now - 1000))
            stalled = _staleness_record(
                progress_path, progress, protocol, now=now
            )
            self.assertTrue(stalled["suspected"])
            self.assertEqual(stalled["threshold_seconds"], 300.0)

    def test_rehearsal_failure_is_recorded_and_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            failure_path = root / "REHEARSAL_FAILED.json"
            protocol_path.write_text("{}\n", encoding="utf-8")
            with mock.patch(
                "act.pipeline.moe.baseline_icml2025_b1_landing_watch.run_rehearsal",
                side_effect=RuntimeError("transient rehearsal failure"),
            ):
                completed = _attempt_rehearsal(
                    protocol_path, failure_path, attempt_count=2
                )
            self.assertFalse(completed)
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "REHEARSAL_FAILED")
            self.assertTrue(failure["nonfatal_to_final_landing"])
            self.assertEqual(failure["attempt_count"], 2)

    def test_restarted_watcher_retains_a_new_failure_history(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            legacy = root / "landing_hook_failure.json"
            state = root / "hook_state.json"
            legacy.write_text('{"status":"OLD"}\n', encoding="utf-8")
            observed = _record_hook_failure(
                legacy,
                state,
                {
                    "schema_version": 1,
                    "status": "FAILED",
                    "error_type": "RuntimeError",
                    "error": "new failure",
                    "protocol_sha256": "abc",
                },
            )
            self.assertTrue(observed.is_file())
            self.assertEqual(
                json.loads(legacy.read_text(encoding="utf-8"))["status"], "OLD"
            )
            current = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(current["status"], "FAILED")
            self.assertEqual(current["failure_record"], str(observed))

    def test_completed_epoch_validates_all_hashes_and_telemetry_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE") as directory:
            root = Path(directory)
            checkpoint = root / "epoch_050.t7"
            metrics = root / "epoch_050.json"
            telemetry = root / "telemetry"
            telemetry.mkdir()
            torch.save({"epoch": 49}, checkpoint)
            metrics.write_text('{"epoch": 50}\n', encoding="utf-8")
            telemetry_summary = telemetry / "summary.json"
            telemetry_summary.write_text(
                json.dumps(
                    {
                        "epoch": 50,
                        "checkpoint": {"sha256": _sha256(checkpoint)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            progress = {
                "completed": [
                    {
                        "epoch": 50,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "metrics": str(metrics),
                        "metrics_sha256": _sha256(metrics),
                        "telemetry": str(telemetry),
                        "telemetry_summary_sha256": _sha256(telemetry_summary),
                    }
                ]
            }
            row = validate_completed_epoch(progress, 50)
            self.assertTrue(row["telemetry_checkpoint_identity_passed"])


if __name__ == "__main__":
    unittest.main()
