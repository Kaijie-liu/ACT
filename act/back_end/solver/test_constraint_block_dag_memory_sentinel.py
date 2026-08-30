#!/usr/bin/env python3
# ===- test_constraint_block_dag_memory_sentinel.py ---------------------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Independent bounded tests for the RANGE/DAG fresh-process RSS sentinel."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from act.back_end.solver import constraint_block_dag_memory_sentinel as sentinel


def _config(
    mode: str = "dual_le",
    *,
    profile: str = sentinel._TOY_PROFILE_NAME,
    measurement_stage: str = "source_build_seal",
    deadline: float | None = None,
) -> dict:
    return sentinel._make_worker_config(
        profile_name=profile,
        mode=mode,
        measurement_stage=measurement_stage,
        repeat_index=0,
        order_index=0 if mode == "dual_le" else 1,
        nonce="a" * 32,
        absolute_deadline_monotonic=(
            time.monotonic() + 5.0 if deadline is None else deadline
        ),
    )


class _CompletedProcess:
    def __init__(self, raw: bytes, stdout_handle, stderr_handle):
        stdout_handle.write(raw)
        stdout_handle.flush()
        stderr_handle.flush()
        self.pid = os.getpid()
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class _DeadlineProcess:
    def __init__(self):
        self.pid = os.getpid()
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9


class _UnreapableProcess(_DeadlineProcess):
    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True


class _InterruptingProcess(_DeadlineProcess):
    def __init__(self, exception):
        super().__init__()
        self._exception = exception
        self._raised = False

    def poll(self):
        if not self._raised:
            self._raised = True
            raise self._exception
        return self.returncode


class ConstraintBlockDAGMemorySentinelTests(unittest.TestCase):
    def test_checksum_tamper_and_nonfinite_fail_closed(self):
        diagnostic = sentinel._closed("test")
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))
        tampered = copy.deepcopy(diagnostic)
        tampered["reason"] = "changed"
        self.assertFalse(sentinel.verify_diagnostic_checksum(tampered))
        nonfinite = copy.deepcopy(diagnostic)
        nonfinite["bad"] = math.nan
        self.assertFalse(sentinel.verify_diagnostic_checksum(nonfinite))

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        invalid = (
            '{"a":1,"a":2}',
            '{"outer":{"x":1,"x":2}}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"x":-Infinity}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                sentinel._strict_json_loads(raw)
        self.assertEqual(sentinel._strict_json_loads('{"a":1}'), {"a": 1})

    def test_hidden_cli_uses_strict_json_loader(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = sentinel.main(
                ["--_worker-config-json", '{"schema":NaN,"schema":"x"}']
            )
        self.assertEqual(code, 2)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "closed")
        self.assertIn("nonfinite_json_constant_rejected", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_candidate_hash_is_checked_before_executing_same_frozen_bytes(self):
        source = Path(sentinel.__file__).read_text(encoding="utf-8")
        self.assertLess(
            source.index("_CANDIDATE_SOURCE_SHA256 != EXPECTED_"),
            source.index("compile(\n            _CANDIDATE_SOURCE_BYTES"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied_sentinel = root / "constraint_block_dag_memory_sentinel.py"
            copied_candidate = root / "constraint_block_dag_candidate.py"
            marker = root / "candidate_executed"
            copied_sentinel.write_text(source, encoding="utf-8")
            copied_candidate.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('bad', encoding='ascii')\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(copied_sentinel), "--help"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5.0,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertIn(b"candidate_sha256_mismatch", completed.stderr)

    def test_fixed_profiles_are_bounded_and_public_api_has_no_cap_knobs(self):
        fixed = sentinel._profile_geometry(sentinel.PROFILE_NAME)
        toy = sentinel._profile_geometry(sentinel._TOY_PROFILE_NAME)
        self.assertEqual(
            (fixed["pair_count"], fixed["columns"], fixed["high_width"]),
            (2_048, 2_618, 74),
        )
        self.assertLess(toy["pair_count"], fixed["pair_count"])
        self.assertEqual(sentinel.COLD_REPEATS, 3)
        self.assertEqual(sentinel.STREAM_MAX_ROWS, 128)
        self.assertEqual(sentinel.HARD_RSS_CAP_BYTES, 512 * (1 << 20))
        self.assertEqual(
            sentinel.HARD_RETAINED_PAYLOAD_CAP_BYTES, 32 * (1 << 20)
        )
        self.assertEqual(sentinel.HARD_WALL_SECONDS, 20.0)
        with self.assertRaises(TypeError):
            sentinel.run_constraint_block_dag_memory_sentinel(
                absolute_deadline_monotonic=time.monotonic() + 1.0,
                hard_rss_cap_bytes=1 << 40,
            )

    def test_toy_real_candidate_dual_and_range_structure_replay(self):
        cgroup_unavailable = {
            "available": False,
            "version": None,
            "leaf": None,
            "current_bytes": None,
            "error": "test_unavailable",
        }
        products = {}
        configs = {}
        with mock.patch.object(
            sentinel,
            "_read_status_bytes",
            side_effect=lambda _pid, key: (
                64 * (1 << 20) if key == "VmRSS" else 65 * (1 << 20)
            ),
        ), mock.patch.object(
            sentinel,
            "_read_cgroup_v2_current",
            return_value=cgroup_unavailable,
        ):
            for stage in sentinel._STAGES:
                for mode in ("dual_le", "range"):
                    config = _config(mode, measurement_stage=stage)
                    configs[(stage, mode)] = config
                    result = sentinel._execute_worker(config)
                    self.assertEqual(
                        result["status"], "ok", result.get("reason")
                    )
                    self.assertIsNone(
                        sentinel._validate_child_success(result, config)
                    )
                    self.assertTrue(result["structure_complete"])
                    self.assertTrue(result["replay_complete"])
                    self.assertTrue(result["candidate_receipt_safe"])
                    self.assertTrue(sentinel.verify_diagnostic_checksum(result))
                    products[(stage, mode)] = result
        for stage in sentinel._STAGES:
            self.assertEqual(
                products[(stage, "dual_le")]["replay_sha256"],
                products[(stage, "range")]["replay_sha256"],
            )
        self.assertTrue(
            products[("full_build_stream_replay", "dual_le")][
                "fraction_membership_complete"
            ]
        )
        stream_config = configs[("full_build_stream_replay", "range")]
        damaged_stream = copy.deepcopy(
            products[("full_build_stream_replay", "range")]
        )
        damaged_stream["fraction_membership_complete"] = False
        damaged_stream = sentinel._seal(damaged_stream)
        self.assertEqual(
            sentinel._validate_child_success(damaged_stream, stream_config),
            "worker_stream_replay_contract_invalid",
        )
        source_products = {
            mode: products[("source_build_seal", mode)]
            for mode in ("dual_le", "range")
        }
        self.assertEqual(source_products["dual_le"]["source_rows"], 16)
        self.assertEqual(source_products["range"]["source_rows"], 8)
        self.assertEqual(source_products["dual_le"]["virtual_facet_rows"], 16)
        self.assertEqual(source_products["range"]["virtual_facet_rows"], 16)
        self.assertLessEqual(
            source_products["range"]["retained_payload_bytes"],
            0.60 * source_products["dual_le"]["retained_payload_bytes"],
        )

    def test_cgroup_v2_entry_terminal_fields_and_delta_are_bound(self):
        samples = [
            {
                "available": True,
                "version": 2,
                "leaf": "/sys/fs/cgroup/test-leaf",
                "current_bytes": 100_000_000,
                "error": None,
            },
            {
                "available": True,
                "version": 2,
                "leaf": "/sys/fs/cgroup/test-leaf",
                "current_bytes": 103_000_000,
                "error": None,
            },
        ]
        with mock.patch.object(
            sentinel,
            "_read_status_bytes",
            side_effect=lambda _pid, key: (
                64 * (1 << 20) if key == "VmRSS" else 65 * (1 << 20)
            ),
        ), mock.patch.object(
            sentinel, "_read_cgroup_v2_current", side_effect=samples
        ):
            config = _config("range")
            result = sentinel._execute_worker(config)
        self.assertEqual(result["status"], "ok", result.get("reason"))
        self.assertEqual(result["cgroup_current_delta_bytes"], 3_000_000)
        self.assertTrue(result["entry"]["cgroup_v2"]["available"])
        self.assertIsNone(sentinel._validate_child_success(result, config))
        damaged = copy.deepcopy(result)
        damaged["terminal"]["cgroup_v2"]["current_bytes"] += 1
        damaged = sentinel._seal(damaged)
        self.assertEqual(
            sentinel._validate_child_success(damaged, config),
            "worker_cgroup_delta_invalid",
        )

    def test_stream_checker_never_calls_full_replay_and_always_closes(self):
        program, _metrics, context = sentinel._construct_and_seal(
            sentinel._TOY_PROFILE_NAME, "range"
        )
        context.clear()
        with mock.patch.object(
            sentinel,
            "_read_status_bytes",
            return_value=64 * (1 << 20),
        ), mock.patch.object(
            sentinel._dag,
            "replay_virtual_facets",
            side_effect=AssertionError("expanded replay forbidden"),
        ), mock.patch.object(
            sentinel.sp,
            "vstack",
            side_effect=AssertionError("full expected CSR forbidden"),
        ):
            result = sentinel._validate_stream_replay(
                program,
                sentinel._TOY_PROFILE_NAME,
                absolute_deadline_monotonic=time.monotonic() + 2.0,
            )
        self.assertTrue(result["replay_complete"])
        self.assertTrue(result["fraction_membership_complete"])
        self.assertEqual(result["replay_kind"], "bounded_exact_stream")

        for failure in (KeyboardInterrupt(), SystemExit(5)):
            inner = sentinel._dag.iter_virtual_facet_batches(
                program, max_rows=sentinel.STREAM_MAX_ROWS
            )

            class InterruptingIterator:
                def __init__(self, source, exception):
                    self.source = source
                    self.exception = exception
                    self.closed = False

                def __iter__(self):
                    return self

                def __next__(self):
                    raise self.exception

                def close(self):
                    self.closed = True
                    self.source.close()

            interrupting = InterruptingIterator(inner, failure)
            with mock.patch.object(
                sentinel._dag,
                "iter_virtual_facet_batches",
                return_value=interrupting,
            ):
                with self.assertRaises(type(failure)):
                    sentinel._validate_stream_replay(
                        program,
                        sentinel._TOY_PROFILE_NAME,
                        absolute_deadline_monotonic=time.monotonic() + 2.0,
                    )
            self.assertTrue(interrupting.closed)
            self.assertTrue(inner.closed)

    def test_real_fixed_synthetic_uses_three_six_worker_collections(self):
        result = sentinel.run_constraint_block_dag_memory_sentinel(
            absolute_deadline_monotonic=time.monotonic()
            + sentinel.HARD_WALL_SECONDS
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        self.assertIn(result["status"], {"closed", "rss_gate_passed"})
        expected_modes = [
            "dual_le", "range", "range", "dual_le", "dual_le", "range"
        ]
        self.assertEqual(
            [item["mode"] for item in result["execution_order"]],
            expected_modes * 3,
        )
        self.assertEqual(
            [item["measurement_stage"] for item in result["execution_order"]],
            ["source_build_seal"] * 6
            + ["full_build_replay"] * 6
            + ["full_build_stream_replay"] * 6,
        )
        self.assertEqual(len(result["runs"]), 18)
        self.assertEqual(
            [run["repeat_index"] for run in result["runs"]],
            [0, 0, 1, 1, 2, 2] * 3,
        )
        for key in (
            "source_build_seal_receipt",
            "full_build_replay_receipt",
            "full_build_stream_replay_receipt",
        ):
            receipt = result[key]
            self.assertTrue(
                receipt["gate_checks"]["all_structure_and_replay_complete"]
            )
            self.assertTrue(receipt["gate_checks"]["all_workers_reaped"])
            self.assertLessEqual(
                receipt["range_to_dual_le_retained_payload_ratio"], 0.60
            )
        self.assertIs(result["full_expanded_gate_closed"], True)
        self.assertIs(result["full_rss_gate_passed"], False)
        self.assertIs(
            result["full_build_replay_receipt"]["stage_gate_passed"], False
        )
        self.assertIs(
            result["full_build_replay_receipt"]["promotion_gate_closed"], True
        )
        self.assertEqual(
            result["rss_gate_passed"],
            bool(
                result["source_stage_rss_gate_passed"]
                and result["stream_full_rss_gate_passed"]
            ),
        )
        self.assertIs(result["production_promotion_claim"], False)
        self.assertTrue(
            all(run["worker_cleanup"]["reaped"] for run in result["runs"])
        )

    def test_malformed_checksum_schema_and_extra_key_child_are_rejected(self):
        config = _config()
        wrong_schema = sentinel._seal({"schema": "wrong", "status": "ok"})
        valid_shape = None
        cgroup_unavailable = {
            "available": False,
            "version": None,
            "leaf": None,
            "current_bytes": None,
            "error": "test_unavailable",
        }
        with mock.patch.object(
            sentinel,
            "_read_status_bytes",
            side_effect=lambda _pid, key: 64 * (1 << 20),
        ), mock.patch.object(
            sentinel,
            "_read_cgroup_v2_current",
            return_value=cgroup_unavailable,
        ):
            valid_shape = sentinel._execute_worker(config)
        extra = copy.deepcopy(valid_shape)
        extra["unexpected"] = 1
        extra = sentinel._seal(extra)
        canonical = json.dumps(
            valid_shape,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        documents = (
            b"{}",
            json.dumps(wrong_schema).encode("ascii"),
            json.dumps(extra).encode("ascii"),
            b'{"schema":"substituted",' + canonical[1:],
            b'{"schema":NaN,' + canonical[1:],
        )
        for raw in documents:
            with self.subTest(raw=raw[:32]):
                def factory(*_args, **kwargs):
                    return _CompletedProcess(raw, kwargs["stdout"], kwargs["stderr"])

                with mock.patch.object(
                    sentinel.subprocess, "Popen", side_effect=factory
                ):
                    result = sentinel._run_one_child(config)
                self.assertEqual(result["status"], "closed")
                self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_overflowing_hex_fields_are_checksummed_protocol_rejections(self):
        config = _config()
        cgroup_unavailable = {
            "available": False,
            "version": None,
            "leaf": None,
            "current_bytes": None,
            "error": "test_unavailable",
        }
        with mock.patch.object(
            sentinel, "_read_status_bytes", return_value=64 * (1 << 20)
        ), mock.patch.object(
            sentinel,
            "_read_cgroup_v2_current",
            return_value=cgroup_unavailable,
        ):
            child = sentinel._execute_worker(config)
        self.assertEqual(child["status"], "ok", child.get("reason"))
        for path in ("entry_time", "worker_wall"):
            with self.subTest(path=path):
                damaged = copy.deepcopy(child)
                if path == "entry_time":
                    damaged["entry"]["sampled_monotonic_hex"] = (
                        "0x1p+999999999"
                    )
                else:
                    damaged["worker_stage_wall_seconds_hex"] = (
                        "0x1p+999999999"
                    )
                damaged = sentinel._seal(damaged)
                raw = json.dumps(damaged).encode("ascii")

                def factory(*_args, **kwargs):
                    return _CompletedProcess(
                        raw, kwargs["stdout"], kwargs["stderr"]
                    )

                with mock.patch.object(
                    sentinel.subprocess, "Popen", side_effect=factory
                ):
                    result = sentinel._run_one_child(config)
                self.assertEqual(result["status"], "closed")
                self.assertTrue(sentinel.verify_diagnostic_checksum(result))

        damaged_config = copy.deepcopy(config)
        damaged_config["absolute_deadline_monotonic_hex"] = "0x1p+999999999"
        body = dict(damaged_config)
        body.pop("config_sha256")
        damaged_config["config_sha256"] = sentinel._sha256(body)
        result = sentinel._execute_worker(damaged_config)
        self.assertEqual(result["status"], "closed")
        self.assertIn("worker_deadline_hex_invalid", result["reason"])

    def test_contract_validator_exception_is_closed_not_raised(self):
        config = _config()
        raw = b'{}'

        def factory(*_args, **kwargs):
            return _CompletedProcess(raw, kwargs["stdout"], kwargs["stderr"])

        with mock.patch.object(
            sentinel.subprocess, "Popen", side_effect=factory
        ), mock.patch.object(
            sentinel,
            "_strict_json_loads",
            return_value={"diagnostic_sha256": "0" * 64},
        ), mock.patch.object(
            sentinel,
            "_validate_child_success",
            side_effect=RuntimeError("validator exploded"),
        ):
            result = sentinel._run_one_child(config)
        self.assertEqual(result["status"], "closed")
        self.assertIn("validator_exception", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_absolute_deadline_terminates_and_reaps_worker(self):
        worker = _DeadlineProcess()
        config = _config(deadline=time.monotonic() + 0.02)
        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(
            sentinel, "_read_status_bytes", return_value=64 * (1 << 20)
        ):
            result = sentinel._run_one_child(config)
        self.assertEqual(result["status"], "closed")
        self.assertIn("absolute_deadline", result["reason"])
        self.assertTrue(worker.terminate_called)
        self.assertTrue(result["worker_cleanup"]["reaped"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_unreaped_worker_is_an_explicit_closed_cleanup_failure(self):
        worker = _UnreapableProcess()
        config = _config(deadline=time.monotonic() + 0.01)
        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(
            sentinel, "_read_status_bytes", return_value=64 * (1 << 20)
        ):
            result = sentinel._run_one_child(config)
        self.assertEqual(result["status"], "closed")
        self.assertIn("not_reaped", result["reason"])
        self.assertTrue(worker.terminate_called)
        self.assertTrue(worker.kill_called)
        self.assertFalse(result["worker_cleanup"]["reaped"])
        self.assertEqual(
            result["worker_cleanup"]["cleanup_error"],
            "worker_not_reaped_after_term_and_kill",
        )

    def test_keyboard_interrupt_and_system_exit_cleanup_then_propagate(self):
        for exception in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(exception=type(exception).__name__):
                worker = _InterruptingProcess(exception)
                with mock.patch.object(
                    sentinel.subprocess, "Popen", return_value=worker
                ):
                    with self.assertRaises(type(exception)):
                        sentinel._run_one_child(_config())
                self.assertTrue(worker.terminate_called)
                self.assertEqual(worker.returncode, -15)

    def test_nan_config_is_a_checksummed_closed_diagnostic(self):
        for bad in (math.nan, math.inf, -math.inf, 1, True, "soon"):
            with self.subTest(bad=bad):
                result = sentinel.run_constraint_block_dag_memory_sentinel(
                    absolute_deadline_monotonic=bad
                )
                self.assertEqual(result["status"], "closed")
                self.assertFalse(result["rss_gate_passed"])
                self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_hidden_worker_cannot_widen_twenty_second_hard_wall(self):
        config = _config(deadline=time.monotonic() + 60.0)
        result = sentinel._execute_worker(config)
        self.assertEqual(result["status"], "closed")
        self.assertIn("exceed_hard_wall", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_stdout_cap_forces_cleanup(self):
        oversized = b"x" * (sentinel.MAX_WORKER_STDOUT_BYTES + 1)
        worker = _DeadlineProcess()

        def factory(*_args, **kwargs):
            kwargs["stdout"].write(oversized)
            kwargs["stdout"].flush()
            return worker

        with mock.patch.object(
            sentinel.subprocess, "Popen", side_effect=factory
        ), mock.patch.object(
            sentinel, "_read_status_bytes", return_value=64 * (1 << 20)
        ):
            result = sentinel._run_one_child(_config())
        self.assertEqual(result["status"], "closed")
        self.assertIn("stdout", result["reason"])
        self.assertTrue(worker.terminate_called)
        self.assertTrue(result["worker_cleanup"]["reaped"])


if __name__ == "__main__":
    unittest.main()
