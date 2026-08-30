#!/usr/bin/env python3
"""Toy-only tests for the isolated split-CG synthetic RSS sentinel.

No test in this file consents to or constructs the 10.5M-nnz profile.  Large
profile tests stop at pure estimation/preflight and assert that no worker is
started.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import split_cg_memory_sentinel as sentinel


def _toy_kwargs(**overrides):
    values = {
        "n_cont": 5,
        "n_bin": 2,
        "n_upper": 12,
        "n_eq": 2,
        "constraint_nnz": 50,
        "selected_upper_rows": 3,
        "deadline": time.monotonic() + 20.0,
    }
    values.update(overrides)
    return values


def _fake_ok_child(config, estimates, preflight):
    bounded = bool(preflight.get("kernel_hard_limit_enforced", False))
    leaf = (preflight.get("cgroup") or {}).get("leaf")
    baseline, current, trimmed, peak = 16, 17, 16, 32
    diagnostic = sentinel._base_diagnostic(config, estimates)
    diagnostic.update(
        status="ok",
        reason=None,
        kernel_hard_limit_enforced=bounded,
        cgroup_leaf_path=leaf,
        candidate_return_status="returned_candidate_only",
        native_model_clear_status="cleared_before_return",
        candidate_status="full_scan_candidate_feasible",
        candidate_full_scan_count=1,
        full_scan_rows=estimates["full_scan_rows"],
        source_frame_sha256="0" * 64,
        candidate_receipt_sha256="1" * 64,
        candidate_frame_sha256="2" * 64,
        process_peak_rss_bytes=peak,
        child_process_baseline_current_rss_bytes=baseline,
        child_process_current_after_cg_bytes=current,
        child_process_current_after_trim_bytes=trimmed,
        child_process_peak_rss_bytes=peak,
        child_process_current_delta_from_baseline_bytes=current - baseline,
        child_process_post_trim_delta_from_baseline_bytes=trimmed - baseline,
        child_process_peak_delta_bytes=peak - baseline,
        source_frame_build_elapsed_seconds_hex=(0.01).hex(),
        cg_elapsed_seconds_hex=(0.01).hex(),
        total_elapsed_seconds_hex=(0.03).hex(),
        worker_terminal_monotonic_hex=time.monotonic().hex(),
        worker_terminal_deadline_respected=True,
    )
    if bounded:
        diagnostic.update(
            cgroup_aggregate_memory_current_after_frame_bytes=100,
            cgroup_aggregate_memory_current_after_cg_bytes=101,
            cgroup_aggregate_memory_current_terminal_bytes=99,
            cgroup_aggregate_memory_peak_terminal_bytes=110,
        )
    else:
        diagnostic.update(
            allowed_rss_increment_bytes=1000,
            baseline_current_rss_bytes=baseline,
            current_rss_bytes=current,
            post_trim_current_rss_bytes=trimmed,
            peak_rss_bytes=peak,
            current_delta_from_baseline_bytes=current - baseline,
            post_trim_delta_from_baseline_bytes=trimmed - baseline,
            peak_delta_from_baseline_bytes=peak - baseline,
        )
    return sentinel._seal_diagnostic(diagnostic)


class SplitCGMemorySentinelStaticTests(unittest.TestCase):
    def test_fixed_profile_values_and_staged_selection_policy(self):
        profile = sentinel.get_fixed_profile()
        self.assertEqual(profile["name"], "cifar100_medium_iid2_v1")
        self.assertEqual(
            profile["parent"],
            {
                "n_cont": 52657,
                "n_bin": 4,
                "n_upper": 98974,
                "n_eq": 0,
                "total_constraint_nnz": 10498232,
            },
        )
        self.assertEqual(
            profile["fresh"],
            {"n_cont": 52661, "n_bin": 4, "n_upper": 98975, "n_eq": 3},
        )
        self.assertEqual(profile["synthetic"]["constraint_nnz"], 10498232)
        self.assertEqual(profile["synthetic"]["selected_upper_rows"], 8192)
        self.assertEqual(
            profile["execution_policy"]["first_selected_upper_rows"], 8192
        )
        self.assertEqual(
            profile["execution_policy"][
                "separate_followup_selected_upper_rows"
            ],
            24576,
        )
        self.assertFalse(
            profile["execution_policy"]["automatic_retry_or_scale_up"]
        )

    def test_4k_8k_selected_estimates_scale_without_large_allocation(self):
        synthetic = sentinel.get_fixed_profile()["synthetic"]
        common = {
            key: synthetic[key]
            for key in (
                "n_cont",
                "n_bin",
                "n_upper",
                "n_eq",
                "constraint_nnz",
            )
        }
        estimate_4k = sentinel.estimate_topology_resources(
            **common, selected_upper_rows=4096
        )
        estimate_8k = sentinel.estimate_topology_resources(
            **common, selected_upper_rows=8192
        )
        self.assertEqual(
            estimate_4k["source_csr_payload_bytes"],
            estimate_8k["source_csr_payload_bytes"],
        )
        self.assertEqual(estimate_4k["selected_upper_nnz"], 4096 * 107)
        self.assertEqual(
            estimate_8k["selected_upper_nnz"], 6564 * 107 + 1628 * 106
        )
        self.assertGreater(
            estimate_8k["estimated_candidate_increment_bytes"],
            estimate_4k["estimated_candidate_increment_bytes"],
        )
        self.assertLess(
            estimate_8k["selected_model_nnz"],
            2 * estimate_4k["selected_model_nnz"],
        )

    def test_small_topology_is_exact_canonical_and_deterministic(self):
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        forbidden = AssertionError("stacked or merged source frame is forbidden")
        with (
            mock.patch.object(sp, "hstack", side_effect=forbidden),
            mock.patch.object(sp, "vstack", side_effect=forbidden),
            mock.patch.object(np, "hstack", side_effect=forbidden),
            mock.patch.object(np, "vstack", side_effect=forbidden),
        ):
            first = sentinel._build_synthetic_frame(config, estimates)
            second = sentinel._build_synthetic_frame(config, estimates)
        self.assertEqual(first.source_frame_sha256, second.source_frame_sha256)
        self.assertEqual(first.source_csr_payload_bytes, 728)
        self.assertEqual(
            sum(matrix.nnz for matrix in (first.Auc, first.Aub, first.Ac, first.Ab)),
            50,
        )
        for matrix in (first.Auc, first.Aub, first.Ac, first.Ab):
            self.assertTrue(sp.isspmatrix_csr(matrix))
            self.assertEqual(matrix.dtype, np.dtype(np.float64))
            self.assertEqual(matrix.indptr.dtype, np.dtype(np.int32))
            self.assertEqual(matrix.indices.dtype, np.dtype(np.int32))
            self.assertTrue(matrix.has_canonical_format)
            self.assertFalse(np.any(matrix.data == 0.0))
        selected_widths = np.diff(first.Auc.indptr[:4]) + np.diff(
            first.Aub.indptr[:4]
        )
        self.assertTrue(np.all(selected_widths > 0))
        self.assertTrue(
            np.all(np.diff(first.Ac.indptr) + np.diff(first.Ab.indptr) > 0)
        )

    def test_checksum_detects_tampering_and_nonfinite_payload(self):
        diagnostic = sentinel._seal_diagnostic(
            {"schema": sentinel.SCHEMA, "status": "toy", "proof_authority": False}
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))
        tampered = copy.deepcopy(diagnostic)
        tampered["status"] = "ok"
        self.assertFalse(sentinel.verify_diagnostic_checksum(tampered))
        nonfinite = copy.deepcopy(diagnostic)
        nonfinite["x"] = float("nan")
        self.assertFalse(sentinel.verify_diagnostic_checksum(nonfinite))

    def test_candidate_contract_rejects_dense_stack_and_authority_tampering(self):
        deadline = time.monotonic() + 10.0
        receipt = {
            "status": "full_scan_candidate_feasible",
            "candidate_only": True,
            "full_split_scan_count": 1,
            "full_split_rows_scanned": 14,
            "native_model_closed_before_return": True,
            "proof_authority": False,
            "verdict_authority": False,
            "primal_feasibility_authority": False,
            "parent_binding": False,
            "parent_binding_authority": False,
            "caps": {"max_rounds": 1},
            "absolute_deadline_hex": deadline.hex(),
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "uses_dense_hstack": False,
            "uses_dense_vstack": False,
            "used_merged_sparse_frame": False,
            "materialized_full_candidate_csr": False,
        }
        self.assertTrue(
            sentinel._candidate_receipt_contract_ok(
                receipt, expected_full_scan_rows=14, deadline=deadline
            )
        )
        for field, value in (
            ("uses_dense_hstack", True),
            ("uses_dense_vstack", True),
            ("candidate_only", False),
            ("primal_feasibility_authority", True),
            ("parent_binding", True),
        ):
            tampered = {**receipt, field: value}
            self.assertFalse(
                sentinel._candidate_receipt_contract_ok(
                    tampered,
                    expected_full_scan_rows=14,
                    deadline=deadline,
                ),
                field,
            )

    def test_expired_deadline_fails_before_worker(self):
        with mock.patch.object(sentinel, "_run_worker_process") as worker:
            diagnostic = sentinel.run_split_cg_memory_sentinel(
                **_toy_kwargs(deadline=time.monotonic() - 1.0)
            )
        worker.assert_not_called()
        self.assertEqual(diagnostic["status"], "preflight_rejected")
        self.assertEqual(
            diagnostic["reason"], "deadline_expired_before_child_start"
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_absolute_cap_and_64mib_reserve_fail_closed(self):
        with mock.patch.object(sentinel, "_run_worker_process") as worker:
            low_cap = sentinel.run_split_cg_memory_sentinel(
                **_toy_kwargs(),
                absolute_rss_cap_bytes=256 * (1 << 20),
            )
            low_reserve = sentinel.run_split_cg_memory_sentinel(
                **_toy_kwargs(), rss_reserve_bytes=(64 * (1 << 20)) - 1
            )
            raised_cap = sentinel.run_split_cg_memory_sentinel(
                **_toy_kwargs(),
                absolute_rss_cap_bytes=(5 * (1 << 29)) + 1,
            )
        worker.assert_not_called()
        self.assertEqual(low_cap["status"], "preflight_rejected")
        self.assertEqual(low_cap["reason"], "absolute_rss_cap_preflight_failed")
        self.assertEqual(low_reserve["status"], "config_rejected")
        self.assertIn("64_mib", low_reserve["reason"])
        self.assertEqual(raised_cap["status"], "config_rejected")
        self.assertIn("2_5_gib", raised_cap["reason"])

    def test_arbitrary_invalid_api_objects_return_safe_checksum_json(self):
        cases = (
            {"n_cont": object()},
            {"profile_name": object()},
            {"absolute_rss_cap_bytes": object()},
        )
        for replacement in cases:
            arguments = _toy_kwargs()
            arguments.update(replacement)
            diagnostic = sentinel.run_split_cg_memory_sentinel(**arguments)
            self.assertEqual(diagnostic["status"], "config_rejected")
            self.assertFalse(diagnostic["proof_authority"])
            self.assertFalse(diagnostic["verdict_authority"])
            self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))
            json.dumps(diagnostic, allow_nan=False)

    def test_child_abnormal_exit_is_a_checksummed_non_authoritative_result(self):
        command = [sys.executable, "-c", "import os; os._exit(7)"]
        with mock.patch.object(sentinel, "_worker_command", return_value=command):
            diagnostic = sentinel.run_split_cg_memory_sentinel(**_toy_kwargs())
        self.assertEqual(diagnostic["status"], "child_abnormal_exit")
        self.assertEqual(diagnostic["worker_exit_code"], 7)
        self.assertFalse(diagnostic["proof_authority"])
        self.assertFalse(diagnostic["verdict_authority"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_popen_oserror_returns_checksummed_launch_failure(self):
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        preflight = {
            "kernel_hard_limit_enforced": False,
            "cgroup": {},
            "cgroup_aggregate_metrics": {},
            "effective_rss_limit_bytes": (
                config["absolute_rss_cap_bytes"]
                - config["rss_reserve_bytes"]
            ),
        }
        with mock.patch.object(
            sentinel.subprocess,
            "Popen",
            side_effect=OSError("synthetic launch failure"),
        ):
            diagnostic = sentinel._run_worker_process(
                config, estimates, preflight
            )
        self.assertEqual(diagnostic["status"], "child_launch_failed")
        self.assertFalse(diagnostic["proof_authority"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_kill_races_and_communicate_error_are_fail_closed(self):
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        effective = (
            config["absolute_rss_cap_bytes"] - config["rss_reserve_bytes"]
        )
        preflight = {
            "kernel_hard_limit_enforced": False,
            "cgroup": {},
            "cgroup_aggregate_metrics": {},
            "effective_rss_limit_bytes": effective,
        }

        class FakeProcess:
            pid = 12345

            def __init__(self, *, polls, kill_error=None, communicate_error=None):
                self._polls = list(polls)
                self._kill_error = kill_error
                self._communicate_error = communicate_error
                self.returncode = None

            def poll(self):
                value = self._polls.pop(0) if self._polls else self.returncode
                if value is not None:
                    self.returncode = value
                return value

            def kill(self):
                if self._kill_error is not None:
                    raise self._kill_error
                self.returncode = -9

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = -15
                return self.returncode

            def communicate(self):
                if self._communicate_error is not None:
                    raise self._communicate_error
                return "", ""

        cases = (
            (
                FakeProcess(
                    polls=[None], kill_error=ProcessLookupError("race")
                ),
                "rss_cap_exceeded",
            ),
            (
                FakeProcess(
                    polls=[None, None], kill_error=OSError("kill failed")
                ),
                "child_control_failed",
            ),
            (
                FakeProcess(
                    polls=[0], communicate_error=OSError("pipe failed")
                ),
                "child_communicate_failed",
            ),
        )
        for process, expected_status in cases:
            with (
                mock.patch.object(
                    sentinel.subprocess, "Popen", return_value=process
                ),
                mock.patch.object(
                    sentinel,
                    "_current_rss_bytes",
                    return_value=effective + 1,
                ),
            ):
                diagnostic = sentinel._run_worker_process(
                    config, estimates, preflight
                )
            self.assertEqual(diagnostic["status"], expected_status)
            self.assertFalse(diagnostic["proof_authority"])
            self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_checksummed_malformed_child_is_rejected_before_arithmetic(self):
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        preflight = {
            "kernel_hard_limit_enforced": False,
            "cgroup": {},
            "cgroup_aggregate_metrics": {},
            "effective_rss_limit_bytes": (
                config["absolute_rss_cap_bytes"]
                - config["rss_reserve_bytes"]
            ),
        }
        for field, value in (
            ("process_peak_rss_bytes", "not-an-integer"),
            ("proof_authority", True),
            ("status", "self_consistent_but_unknown"),
        ):
            child = _fake_ok_child(config, estimates, preflight)
            child[field] = value
            child = sentinel._seal_diagnostic(child)
            encoded = json.dumps(child, sort_keys=True, separators=(",", ":"))
            command = [sys.executable, "-c", f"print({encoded!r})"]
            with (
                mock.patch.object(
                    sentinel, "_worker_command", return_value=command
                ),
                mock.patch.object(
                    sentinel, "_current_rss_bytes", return_value=32
                ),
            ):
                diagnostic = sentinel._run_worker_process(
                    config, estimates, preflight
                )
            self.assertEqual(
                diagnostic["status"], "child_diagnostic_schema_invalid"
            )
            self.assertFalse(diagnostic["proof_authority"])
            self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_bounded_aggregate_peak_overrides_low_child_rss_and_rejects(self):
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        cap = config["absolute_rss_cap_bytes"]
        effective = cap - config["rss_reserve_bytes"]
        leaf = "/sys/fs/cgroup/run-test.scope"
        cgroup = {
            "detected": True,
            "version": 2,
            "leaf": leaf,
            "current_bytes": 64 * (1 << 20),
            "max_bytes": cap,
            "unlimited": False,
            "headroom_bytes": cap - 64 * (1 << 20),
            "ancestor_limits": [
                {
                    "path": leaf,
                    "current_bytes": 64 * (1 << 20),
                    "max_bytes": cap,
                    "unlimited": False,
                    "headroom_bytes": cap - 64 * (1 << 20),
                }
            ],
            "delegation_boundary": "/sys/fs/cgroup",
            "boundary_complete": True,
            "error": None,
        }
        aggregate = {
            "readable": True,
            "leaf": leaf,
            "current_bytes": 128 * (1 << 20),
            "peak_bytes": effective + 1,
            "sampled_monotonic_hex": time.monotonic().hex(),
            "error": None,
        }
        preflight = {
            "kernel_hard_limit_enforced": True,
            "cgroup": cgroup,
            "cgroup_aggregate_metrics": aggregate,
            "effective_rss_limit_bytes": effective,
        }
        child = _fake_ok_child(config, estimates, preflight)
        encoded = json.dumps(child, sort_keys=True, separators=(",", ":"))
        command = [sys.executable, "-c", f"print({encoded!r})"]
        with (
            mock.patch.object(sentinel, "_worker_command", return_value=command),
            mock.patch.object(
                sentinel,
                "_read_v2_leaf_aggregate_memory",
                return_value=aggregate,
            ),
            mock.patch.object(
                sentinel, "_current_rss_bytes", return_value=32 * (1 << 20)
            ),
        ):
            diagnostic = sentinel._run_worker_process(
                config, estimates, preflight
            )
        self.assertEqual(diagnostic["status"], "rss_cap_exceeded")
        self.assertIsNone(diagnostic["peak_rss_bytes"])
        self.assertEqual(diagnostic["stoploss_peak_bytes"], effective + 1)
        self.assertEqual(
            diagnostic["stoploss_peak_source"],
            "cgroup_v2_leaf_aggregate_memory_peak",
        )
        self.assertFalse(diagnostic["rss_cap_respected"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_parent_rejects_child_terminal_receipt_at_deadline(self):
        raw = {
            **_toy_kwargs(deadline=time.monotonic() + 5.0),
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        preflight = {
            "kernel_hard_limit_enforced": False,
            "cgroup": {},
            "cgroup_aggregate_metrics": {},
            "effective_rss_limit_bytes": (
                config["absolute_rss_cap_bytes"]
                - config["rss_reserve_bytes"]
            ),
        }
        child = _fake_ok_child(config, estimates, preflight)
        child["worker_terminal_monotonic_hex"] = config["deadline"].hex()
        child = sentinel._seal_diagnostic(child)
        encoded = json.dumps(child, sort_keys=True, separators=(",", ":"))
        command = [sys.executable, "-c", f"print({encoded!r})"]
        with (
            mock.patch.object(sentinel, "_worker_command", return_value=command),
            mock.patch.object(
                sentinel, "_current_rss_bytes", return_value=32 * (1 << 20)
            ),
        ):
            diagnostic = sentinel._run_worker_process(
                config, estimates, preflight
            )
        self.assertEqual(
            diagnostic["status"], "child_diagnostic_schema_invalid"
        )
        self.assertEqual(
            diagnostic["reason"], "worker_terminal_after_deadline"
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_default_cli_refuses_large_profile_without_starting_worker(self):
        output = io.StringIO()
        with (
            mock.patch.object(sentinel, "_run_worker_process") as worker,
            contextlib.redirect_stdout(output),
        ):
            return_code = sentinel.main([])
        worker.assert_not_called()
        diagnostic = json.loads(output.getvalue())
        self.assertEqual(return_code, 2)
        self.assertEqual(diagnostic["status"], "preflight_rejected")
        self.assertEqual(
            diagnostic["reason"],
            "large_profile_requires_explicit_execute_flag",
        )
        self.assertEqual(
            diagnostic["topology"]["selected_upper_rows"], 8192
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))

    def test_explicit_large_is_still_blocked_under_unlimited_cgroup(self):
        unlimited = {
            "detected": True,
            "version": 2,
            "leaf": "/sys/fs/cgroup/test.scope",
            "current_bytes": 1 << 20,
            "max_bytes": None,
            "unlimited": True,
            "headroom_bytes": None,
            "ancestor_limits": [
                {
                    "path": "/sys/fs/cgroup/test.scope",
                    "current_bytes": 1 << 20,
                    "max_bytes": None,
                    "unlimited": True,
                    "headroom_bytes": None,
                }
            ],
            "delegation_boundary": "/sys/fs/cgroup",
            "boundary_complete": True,
            "error": None,
        }
        with (
            mock.patch.object(sentinel, "_read_cgroup_memory", return_value=unlimited),
            mock.patch.object(
                sentinel,
                "_read_mem_available_bytes",
                return_value=32 * (1 << 30),
            ),
            mock.patch.object(
                sentinel,
                "_current_rss_bytes",
                return_value=64 * (1 << 20),
            ),
            mock.patch.object(sentinel, "_run_worker_process") as worker,
        ):
            diagnostic = sentinel.run_fixed_profile(
                deadline=time.monotonic() + 180.0,
                execute_large_profile=True,
            )
        worker.assert_not_called()
        self.assertEqual(diagnostic["status"], "preflight_rejected")
        self.assertEqual(
            diagnostic["reason"], "large_profile_cgroup_has_no_finite_hard_limit"
        )
        self.assertFalse(diagnostic["kernel_hard_limit_enforced"])

    def test_bounded_leaf_at_requested_cap_is_kernel_enforced_preflight(self):
        cap = sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
        bounded = {
            "detected": True,
            "version": 2,
            "leaf": "/sys/fs/cgroup/user.slice/run-test.scope",
            "current_bytes": 64 * (1 << 20),
            "max_bytes": cap,
            "unlimited": False,
            "headroom_bytes": cap - 64 * (1 << 20),
            "ancestor_limits": [
                {
                    "path": "/sys/fs/cgroup/user.slice/run-test.scope",
                    "current_bytes": 64 * (1 << 20),
                    "max_bytes": cap,
                    "unlimited": False,
                    "headroom_bytes": cap - 64 * (1 << 20),
                }
            ],
            "delegation_boundary": "/sys/fs/cgroup",
            "boundary_complete": True,
            "error": None,
        }
        raw = {
            **_toy_kwargs(),
            "absolute_rss_cap_bytes": cap,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 4,
            "execute_large_profile": False,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        with (
            mock.patch.object(sentinel, "_read_cgroup_memory", return_value=bounded),
            mock.patch.object(
                sentinel,
                "_read_mem_available_bytes",
                return_value=32 * (1 << 30),
            ),
            mock.patch.object(
                sentinel,
                "_current_rss_bytes",
                return_value=64 * (1 << 20),
            ),
        ):
            checks, reason = sentinel._preflight(config, estimates)
        self.assertIsNone(reason)
        self.assertTrue(checks["kernel_hard_limit_enforced"])
        self.assertEqual(
            checks["kernel_hard_limit_leaf_max_bytes"], cap
        )

    def test_large_preflight_rejects_unreadable_aggregate_current_or_peak(self):
        cap = sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
        bounded = {
            "detected": True,
            "version": 2,
            "leaf": "/sys/fs/cgroup/run-test.scope",
            "current_bytes": 64 * (1 << 20),
            "max_bytes": cap,
            "unlimited": False,
            "headroom_bytes": cap - 64 * (1 << 20),
            "ancestor_limits": [
                {
                    "path": "/sys/fs/cgroup/run-test.scope",
                    "current_bytes": 64 * (1 << 20),
                    "max_bytes": cap,
                    "unlimited": False,
                    "headroom_bytes": cap - 64 * (1 << 20),
                }
            ],
            "delegation_boundary": "/sys/fs/cgroup",
            "boundary_complete": True,
            "error": None,
        }
        synthetic = sentinel.get_fixed_profile()["synthetic"]
        raw = {
            **synthetic,
            "deadline": time.monotonic() + 180.0,
            "absolute_rss_cap_bytes": cap,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "scan_chunk_rows": 8192,
            "execute_large_profile": True,
            "profile_name": sentinel.PROFILE_NAME,
        }
        config, estimates = sentinel._validate_config(raw)
        unreadable = {
            "readable": False,
            "leaf": bounded["leaf"],
            "current_bytes": None,
            "peak_bytes": None,
            "sampled_monotonic_hex": time.monotonic().hex(),
            "error": "aggregate_metric_missing:memory.peak",
        }
        with (
            mock.patch.object(sentinel, "_read_cgroup_memory", return_value=bounded),
            mock.patch.object(
                sentinel,
                "_read_v2_leaf_aggregate_memory",
                return_value=unreadable,
            ),
            mock.patch.object(
                sentinel,
                "_read_mem_available_bytes",
                return_value=32 * (1 << 30),
            ),
            mock.patch.object(
                sentinel,
                "_current_rss_bytes",
                return_value=64 * (1 << 20),
            ),
        ):
            checks, reason = sentinel._preflight(config, estimates)
        self.assertTrue(checks["kernel_hard_limit_enforced"])
        self.assertEqual(
            reason, "large_profile_cgroup_aggregate_metrics_unreadable"
        )

    def test_v2_leaf_and_ancestors_are_resolved_from_proc_membership(self):
        cap = sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
        contents = {
            "/proc/self/cgroup": "0::/user.slice/test.scope\n",
            "/sys/fs/cgroup/user.slice/test.scope/memory.max": str(cap),
            "/sys/fs/cgroup/user.slice/test.scope/memory.current": str(100),
            "/sys/fs/cgroup/user.slice/memory.max": "max",
            "/sys/fs/cgroup/user.slice/memory.current": str(200),
        }

        def exists(path):
            return str(path) in contents

        def read_text(path, *args, **kwargs):
            key = str(path)
            if key not in contents:
                raise FileNotFoundError(key)
            return contents[key]

        with (
            mock.patch.object(Path, "exists", exists),
            mock.patch.object(Path, "read_text", read_text),
        ):
            observed = sentinel._read_cgroup_memory()
        self.assertTrue(observed["detected"])
        self.assertEqual(
            observed["leaf"], "/sys/fs/cgroup/user.slice/test.scope"
        )
        self.assertEqual(observed["max_bytes"], cap)
        self.assertEqual(observed["headroom_bytes"], cap - 100)
        self.assertEqual(observed["delegation_boundary"], "/sys/fs/cgroup")
        self.assertTrue(observed["boundary_complete"])
        self.assertIsNone(observed["error"])

    def test_single_missing_cgroup_controller_file_fails_closed(self):
        contents = {
            "/proc/self/cgroup": "0::/user.slice/test.scope\n",
            "/sys/fs/cgroup/user.slice/test.scope/memory.max": str(
                sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
            ),
        }

        def exists(path):
            return str(path) in contents

        def read_text(path, *args, **kwargs):
            key = str(path)
            if key not in contents:
                raise FileNotFoundError(key)
            return contents[key]

        with (
            mock.patch.object(Path, "exists", exists),
            mock.patch.object(Path, "read_text", read_text),
        ):
            observed = sentinel._read_cgroup_memory()
        self.assertFalse(observed["boundary_complete"])
        self.assertIn("partial_memory_controller_files", observed["error"])

    def test_interior_double_missing_controller_pair_is_not_a_boundary(self):
        cap = sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
        contents = {
            "/proc/self/cgroup": "0::/delegated/leaf.scope\n",
            "/sys/fs/cgroup/delegated/leaf.scope/memory.max": str(cap),
            "/sys/fs/cgroup/delegated/leaf.scope/memory.current": "100",
        }

        def exists(path):
            return str(path) in contents

        def read_text(path, *args, **kwargs):
            key = str(path)
            if key not in contents:
                raise FileNotFoundError(key)
            return contents[key]

        with (
            mock.patch.object(Path, "exists", exists),
            mock.patch.object(Path, "read_text", read_text),
        ):
            observed = sentinel._read_cgroup_memory()
        self.assertEqual(observed["delegation_boundary"], "/sys/fs/cgroup/delegated")
        self.assertFalse(observed["boundary_complete"])
        self.assertIn("interior_memory_controller_gap", observed["error"])


@unittest.skipIf(sentinel._scg._highspy is None, "highspy is optional")
class SplitCGMemorySentinelChildToyTests(unittest.TestCase):
    def test_public_child_toy_records_rss_scan_close_and_no_merge(self):
        diagnostic = sentinel.run_split_cg_memory_sentinel(**_toy_kwargs())
        self.assertEqual(diagnostic["status"], "ok", diagnostic.get("reason"))
        self.assertTrue(sentinel.verify_diagnostic_checksum(diagnostic))
        self.assertFalse(diagnostic["proof_authority"])
        self.assertFalse(diagnostic["verdict_authority"])
        child_baseline = diagnostic[
            "child_process_baseline_current_rss_bytes"
        ]
        self.assertIsInstance(child_baseline, int)
        self.assertEqual(
            diagnostic["child_process_allowed_rss_increment_bytes"],
            sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES
            - sentinel.MINIMUM_RSS_RESERVE_BYTES
            - child_baseline,
        )
        child_current = diagnostic["child_process_current_after_cg_bytes"]
        child_trimmed = diagnostic[
            "child_process_current_after_trim_bytes"
        ]
        child_peak = diagnostic["child_process_peak_rss_bytes"]
        self.assertIsInstance(child_current, int)
        self.assertIsInstance(child_trimmed, int)
        self.assertIsInstance(child_peak, int)
        self.assertEqual(
            diagnostic["child_process_current_delta_from_baseline_bytes"],
            child_current - child_baseline,
        )
        self.assertEqual(
            diagnostic[
                "child_process_post_trim_delta_from_baseline_bytes"
            ],
            child_trimmed - child_baseline,
        )
        self.assertEqual(
            diagnostic["child_process_peak_delta_bytes"],
            max(0, child_peak - child_baseline),
        )
        sampled_peak = diagnostic["cg_sampled_peak_rss_bytes"]
        sampled_delta = diagnostic[
            "cg_sampled_peak_delta_from_baseline_bytes"
        ]
        if sampled_delta is not None:
            self.assertEqual(
                sampled_delta, max(0, sampled_peak - child_baseline)
            )
        self.assertEqual(diagnostic["source_csr_payload_bytes"], 728)
        self.assertEqual(diagnostic["selected_constraint_nnz"], 18)
        self.assertEqual(diagnostic["selected_model_rows"], 5)
        self.assertEqual(diagnostic["full_scan_rows"], 14)
        self.assertEqual(diagnostic["candidate_max_rounds"], 1)
        self.assertEqual(diagnostic["candidate_full_scan_count"], 1)
        self.assertEqual(
            diagnostic["candidate_return_status"], "returned_candidate_only"
        )
        self.assertEqual(
            diagnostic["native_model_clear_status"], "cleared_before_return"
        )
        self.assertIn(
            diagnostic["allocator_trim_status"],
            {"released", "no_release", "unavailable"},
        )
        self.assertFalse(diagnostic["uses_sparse_hstack"])
        self.assertFalse(diagnostic["uses_sparse_vstack"])
        self.assertFalse(diagnostic["uses_dense_hstack"])
        self.assertFalse(diagnostic["uses_dense_vstack"])
        self.assertFalse(diagnostic["used_merged_sparse_frame"])
        self.assertFalse(diagnostic["materialized_full_candidate_csr"])
        self.assertTrue(diagnostic["rss_cap_respected"])
        for field in (
            "cgroup_aggregate_memory_current_start_bytes",
            "cgroup_aggregate_memory_current_after_frame_bytes",
            "cgroup_aggregate_memory_current_after_cg_bytes",
            "cgroup_aggregate_memory_current_terminal_bytes",
            "cgroup_aggregate_memory_peak_terminal_bytes",
        ):
            self.assertIsInstance(diagnostic[field], int, field)
        self.assertFalse(diagnostic["cgroup_aggregate_peak_reset_attempted"])
        for field in (
            "source_frame_build_elapsed_seconds_hex",
            "cg_elapsed_seconds_hex",
            "total_elapsed_seconds_hex",
        ):
            self.assertGreaterEqual(float.fromhex(diagnostic[field]), 0.0)
        self.assertTrue(diagnostic["worker_terminal_deadline_respected"])
        if diagnostic["kernel_hard_limit_enforced"]:
            self.assertIn("aggregate", diagnostic["rss_cap_enforcement"])
            self.assertEqual(
                diagnostic["stoploss_peak_source"],
                "cgroup_v2_leaf_aggregate_memory_peak",
            )
            for field in (
                "allowed_rss_increment_bytes",
                "baseline_current_rss_bytes",
                "current_rss_bytes",
                "post_trim_current_rss_bytes",
                "peak_rss_bytes",
                "current_delta_from_baseline_bytes",
                "post_trim_delta_from_baseline_bytes",
                "peak_delta_from_baseline_bytes",
                "cgroup_aggregate_peak_delta_from_after_frame_bytes",
            ):
                self.assertIsNone(diagnostic[field], field)
            self.assertEqual(
                diagnostic["stoploss_peak_bytes"],
                diagnostic["cgroup_aggregate_memory_peak_terminal_bytes"],
            )
        else:
            self.assertIn(
                "diagnostic_only", diagnostic["rss_cap_enforcement"]
            )
            baseline = diagnostic["baseline_current_rss_bytes"]
            self.assertEqual(baseline, child_baseline)
            self.assertEqual(diagnostic["current_rss_bytes"], child_current)
            self.assertEqual(diagnostic["peak_rss_bytes"], child_peak)
            self.assertEqual(
                diagnostic["current_delta_from_baseline_bytes"],
                diagnostic["current_rss_bytes"] - baseline,
            )
            self.assertEqual(
                diagnostic["peak_delta_from_baseline_bytes"],
                max(0, diagnostic["peak_rss_bytes"] - baseline),
            )


if __name__ == "__main__":
    unittest.main()
