#!/usr/bin/env python3
"""Toy-only gates for the isolated K3 pair/scheduled memory sentinel."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import subprocess
import time
import unittest
from unittest import mock

from act.back_end.hybridz_tf import (
    operator_phase_conditioned_k3_memory_sentinel as sentinel,
)


_TOY = {
    "n_cont": 32,
    "n_bin": 4,
    "n_upper": 64,
    "n_eq": 0,
    "constraint_nnz": 256,
}


def _bounded_cgroup(leaf="/sys/fs/cgroup/k3-sentinel-test"):
    mib = 1 << 20
    leaf_current = 128 * mib
    leaf_max = 2048 * mib
    root_current = 192 * mib
    root_max = 2560 * mib
    return {
        "detected": True,
        "version": 2,
        "leaf": leaf,
        "leaf_current_bytes": leaf_current,
        "leaf_peak_bytes": 256 * mib,
        "leaf_max_bytes": leaf_max,
        "effective_max_bytes": leaf_max,
        "effective_headroom_bytes": leaf_max - leaf_current,
        "ancestor_limits": [
            {
                "path": leaf,
                "current_bytes": leaf_current,
                "max_bytes": leaf_max,
                "headroom_bytes": leaf_max - leaf_current,
            },
            {
                "path": "/sys/fs/cgroup",
                "current_bytes": root_current,
                "max_bytes": root_max,
                "headroom_bytes": root_max - root_current,
            },
        ],
        "boundary_complete": True,
        "error": None,
    }


class _NeverStartedWorker:
    def __init__(self):
        self.pid = 424242
        self.returncode = None
        self.stop_signal = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.stop_signal is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = -int(self.stop_signal)
        return self.returncode

    def terminate(self):
        self.stop_signal = 15

    def kill(self):
        self.stop_signal = 9


class _UnreapableWorker(_NeverStartedWorker):
    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("fake-unreapable", timeout)


class _CompletedPayloadWorker:
    def __init__(self, payload, stdout):
        self.pid = 434343
        self.returncode = 0
        stdout.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        stdout.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class _CompletedRawWorker:
    def __init__(self, raw, stdout):
        self.pid = 444444
        self.returncode = 0
        stdout.write(raw)
        stdout.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class K3PairScheduledMemorySentinelToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = sentinel.run_k3_pair_scheduled_memory_sentinel(
            **_TOY,
            mode="complete",
            deadline=time.monotonic() + 30.0,
        )
        cls.early = sentinel.run_k3_pair_scheduled_memory_sentinel(
            **_TOY,
            mode="early_stop",
            deadline=time.monotonic() + 30.0,
        )

    def _assert_authority_firewall(self, result):
        self.assertTrue(result["diagnostic_only"])
        self.assertTrue(result["candidate_only"])
        for field in (
            "proof_authority",
            "verdict_authority",
            "primal_feasibility_authority",
            "parent_binding_authority",
            "ground_truth_loaded",
            "full_parent_lp_called",
            "fresh_issue_called",
            "fresh_materialization_called",
            "partial_certificates_returned",
        ):
            self.assertIs(result[field], False, field)

    def test_complete_runs_real_pair_and_all_eight_scheduled_patterns(self):
        result = self.complete
        self.assertEqual(result["status"], "ok_complete")
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        self._assert_authority_firewall(result)
        self.assertEqual(result["pair_query_count"], 12)
        self.assertTrue(result["pair_models_closed"])
        self.assertEqual(result["scheduled_patterns_completed"], 8)
        self.assertIsInstance(result["conditional_checker_actual_calls"], int)
        self.assertEqual(
            result["conditional_checker_actual_calls"],
            1
            + result["scheduled_patterns_completed"]
            + result["scheduled_candidate_dual_accepted"],
        )
        self.assertLessEqual(
            result["scheduled_local_lp_actual_calls"],
            result["scheduled_patterns_completed"],
        )
        self.assertIsNotNone(result["scheduled_bundle_sha256"])
        self.assertIsNone(result["scheduled_stop_record_sha256"])
        self.assertEqual(
            [item["stage"] for item in result["memory_checkpoints"]],
            ["baseline", "after_pair", "pre_s"]
            + [f"pattern_{index}" for index in range(8)]
            + ["terminal"],
        )
        self.assertTrue(
            all(
                item["process_current_rss_bytes"] is not None
                and item["process_peak_rss_bytes"] is not None
                for item in result["memory_checkpoints"]
            )
        )

    def test_early_stop_samples_first_pattern_and_returns_no_partial_cover(self):
        result = self.early
        self.assertEqual(result["status"], "ok_early_stop")
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        self._assert_authority_firewall(result)
        self.assertEqual(result["pair_query_count"], 12)
        self.assertEqual(result["scheduled_patterns_completed"], 1)
        self.assertIsInstance(result["conditional_checker_actual_calls"], int)
        self.assertEqual(
            result["conditional_checker_actual_calls"],
            1
            + result["scheduled_patterns_completed"]
            + result["scheduled_candidate_dual_accepted"],
        )
        self.assertIsNone(result["scheduled_bundle_sha256"])
        self.assertIsNotNone(result["scheduled_stop_record_sha256"])
        self.assertEqual(
            [item["stage"] for item in result["memory_checkpoints"]],
            ["baseline", "after_pair", "pre_s", "pattern_0", "terminal"],
        )

    def test_fixed_large_profile_is_inert_without_explicit_consent(self):
        with mock.patch.object(
            sentinel.subprocess,
            "Popen",
            side_effect=AssertionError("large worker must not start"),
        ) as popen:
            result = sentinel.run_fixed_profile(
                mode="complete",
                deadline=time.monotonic() + 300.0,
            )
        self.assertEqual(result["status"], "preflight_rejected")
        self.assertEqual(
            result["reason"],
            "large_profile_requires_explicit_execute_flag",
        )
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        popen.assert_not_called()
        self.assertEqual(
            result["topology"], sentinel.get_fixed_profile()["topology"]
        )
        self.assertFalse(
            result["resource_estimates"][
                "allocator_increment_bound_available"
            ]
        )

    def test_large_consent_still_fails_without_complete_bounded_v2_leaf(self):
        unavailable = {
            "detected": False,
            "version": None,
            "leaf": None,
            "leaf_current_bytes": None,
            "leaf_peak_bytes": None,
            "leaf_max_bytes": None,
            "effective_max_bytes": None,
            "effective_headroom_bytes": None,
            "ancestor_limits": [],
            "boundary_complete": False,
            "error": "test_unavailable",
        }
        with mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=unavailable
        ), mock.patch.object(
            sentinel.subprocess,
            "Popen",
            side_effect=AssertionError("unbounded worker must not start"),
        ) as popen:
            result = sentinel.run_fixed_profile(
                mode="complete",
                deadline=time.monotonic() + 300.0,
                execute_large_profile=True,
            )
        self.assertEqual(result["status"], "preflight_rejected")
        self.assertEqual(
            result["reason"],
            "large_profile_requires_complete_cgroup_v2_ancestors",
        )
        popen.assert_not_called()

    def test_strict_types_and_checksum_tamper_fail_closed(self):
        result = sentinel.run_k3_pair_scheduled_memory_sentinel(
            **{**_TOY, "n_bin": True},
            mode="complete",
            deadline=time.monotonic() + 30.0,
        )
        self.assertEqual(result["status"], "config_rejected")
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        tampered = copy.deepcopy(self.complete)
        tampered["pair_query_count"] = 11
        self.assertFalse(sentinel.verify_diagnostic_checksum(tampered))

    def test_nonfinite_and_arbitrary_invalid_config_are_json_safe(self):
        nonfinite = sentinel.run_k3_pair_scheduled_memory_sentinel(
            **_TOY,
            mode="complete",
            deadline=float("nan"),
        )
        self.assertEqual(nonfinite["status"], "config_rejected")
        self.assertTrue(sentinel.verify_diagnostic_checksum(nonfinite))
        json.dumps(nonfinite, allow_nan=False)
        self.assertEqual(
            nonfinite["rejected_config"]["deadline"]["reason"],
            "nonfinite_float",
        )

        arbitrary = sentinel.run_k3_pair_scheduled_memory_sentinel(
            **_TOY,
            mode=object(),
            deadline=time.monotonic() + 30.0,
        )
        self.assertEqual(arbitrary["status"], "config_rejected")
        self.assertTrue(sentinel.verify_diagnostic_checksum(arbitrary))
        json.dumps(arbitrary, allow_nan=False)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = sentinel.main(["--deadline-seconds", "nan"])
        cli = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(cli["status"], "config_rejected")
        self.assertTrue(sentinel.verify_diagnostic_checksum(cli))

    def test_preflight_replays_exact_ancestor_records_not_summary_flags(self):
        topology = sentinel.get_fixed_profile()["topology"]
        raw = {
            **topology,
            "mode": "complete",
            "deadline": time.monotonic() + 300.0,
            "execute_large_profile": True,
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "profile_name": sentinel.PROFILE_NAME,
        }
        config, estimates = sentinel._validate_config(raw)
        with mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=_bounded_cgroup()
        ):
            checks, reason = sentinel._preflight(config, estimates)
        self.assertIsNone(reason)
        self.assertTrue(checks["complete_cgroup_v2_ancestor_walk"])

        forged = _bounded_cgroup()
        forged["ancestor_limits"] = []
        with mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=forged
        ):
            checks, reason = sentinel._preflight(config, estimates)
        self.assertEqual(
            reason, "large_profile_requires_complete_cgroup_v2_ancestors"
        )
        self.assertFalse(checks["complete_cgroup_v2_ancestor_walk"])

        forged = _bounded_cgroup()
        forged["ancestor_limits"][0]["headroom_bytes"] += 1
        with mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=forged
        ):
            checks, reason = sentinel._preflight(config, estimates)
        self.assertEqual(
            reason, "large_profile_requires_complete_cgroup_v2_ancestors"
        )
        self.assertIn("headroom_invalid", checks["cgroup_contract_error"])

    def _large_contract_fixture(self):
        config = {
            "n_cont": _TOY["n_cont"],
            "n_bin": _TOY["n_bin"],
            "n_upper": _TOY["n_upper"],
            "n_eq": _TOY["n_eq"],
            "constraint_nnz": _TOY["constraint_nnz"],
            "mode": "complete",
            "large_topology": True,
        }
        child = copy.deepcopy(self.complete)
        child["large_topology"] = True
        cgroup = _bounded_cgroup()
        parent_preflight = {"cgroup": copy.deepcopy(cgroup)}
        child["preflight"] = {"cgroup": copy.deepcopy(cgroup)}
        for item in child["memory_checkpoints"]:
            item.update(
                process_current_rss_bytes=32 * (1 << 20),
                process_peak_rss_bytes=64 * (1 << 20),
                cgroup_v2_leaf=cgroup["leaf"],
                cgroup_current_bytes=cgroup["leaf_current_bytes"],
                cgroup_peak_bytes=cgroup["leaf_peak_bytes"],
                cgroup_leaf_max_bytes=cgroup["leaf_max_bytes"],
                cgroup_effective_max_bytes=cgroup["effective_max_bytes"],
                cgroup_effective_headroom_bytes=cgroup[
                    "effective_headroom_bytes"
                ],
                cgroup_boundary_complete=True,
                cgroup_error=None,
            )
        child["memory_trace_sha256"] = sentinel._canonical_sha256(
            {"memory_checkpoints": child["memory_checkpoints"]}
        )
        return config, child, parent_preflight

    def test_success_contract_rejects_coherent_cgroup_move_and_unbounding(self):
        config, child, parent_preflight = self._large_contract_fixture()
        self.assertIsNone(
            sentinel._valid_success_contract(
                child, config, parent_preflight
            )
        )

        moved = copy.deepcopy(child)
        moved_cgroup = _bounded_cgroup("/sys/fs/cgroup/moved")
        moved["preflight"] = {"cgroup": moved_cgroup}
        for item in moved["memory_checkpoints"]:
            item["cgroup_v2_leaf"] = moved_cgroup["leaf"]
        moved["memory_trace_sha256"] = sentinel._canonical_sha256(
            {"memory_checkpoints": moved["memory_checkpoints"]}
        )
        error = sentinel._valid_success_contract(
            moved, config, parent_preflight
        )
        self.assertIn("entry_cgroup_binding_mismatch", error)

        unbounded = copy.deepcopy(child)
        unbounded_cgroup = _bounded_cgroup()
        unbounded_cgroup["ancestor_limits"][0]["max_bytes"] = None
        unbounded_cgroup["ancestor_limits"][0]["headroom_bytes"] = None
        unbounded_cgroup["leaf_max_bytes"] = None
        unbounded["preflight"] = {"cgroup": unbounded_cgroup}
        error = sentinel._valid_success_contract(
            unbounded, config, parent_preflight
        )
        self.assertEqual(error, "child_entry_cgroup_contract_invalid")

    def _assert_monitor_failure_cleans_worker(self, patch_name):
        raw = {
            **_TOY,
            "mode": "complete",
            "deadline": time.monotonic() + 30.0,
            "execute_large_profile": False,
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        worker = _NeverStartedWorker()

        def killpg(_pid, sig):
            worker.stop_signal = sig

        patches = {
            "current_rss": mock.patch.object(
                sentinel,
                "_current_rss_bytes",
                side_effect=RuntimeError("rss sample failed"),
            ),
            "cgroup": mock.patch.object(
                sentinel,
                "_read_cgroup_v2",
                side_effect=RuntimeError("cgroup sample failed"),
            ),
            "clock": mock.patch.object(
                sentinel.time,
                "monotonic",
                side_effect=RuntimeError("clock sample failed"),
            ),
            "sleep": mock.patch.object(
                sentinel.time,
                "sleep",
                side_effect=RuntimeError("sleep failed"),
            ),
        }
        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(sentinel.os, "killpg", side_effect=killpg), mock.patch.object(
            sentinel, "_current_rss_bytes", return_value=1
        ), mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value={}
        ), patches[patch_name]:
            result = sentinel._run_worker_process(
                config, estimates, {"cgroup": {}}
            )
        self.assertEqual(result["status"], "worker_monitor_error")
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))
        self.assertIsNotNone(worker.stop_signal)
        self.assertIsNotNone(worker.returncode)

    def test_every_monitor_sampling_exception_cleans_child_and_seals_error(self):
        for patch_name in ("current_rss", "cgroup", "clock", "sleep"):
            with self.subTest(patch_name=patch_name):
                self._assert_monitor_failure_cleans_worker(patch_name)

    def test_keyboard_interrupt_propagates_only_after_child_cleanup(self):
        raw = {
            **_TOY,
            "mode": "complete",
            "deadline": time.monotonic() + 30.0,
            "execute_large_profile": False,
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        worker = _NeverStartedWorker()

        def killpg(_pid, sig):
            worker.stop_signal = sig

        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(sentinel.os, "killpg", side_effect=killpg), mock.patch.object(
            sentinel, "_current_rss_bytes", side_effect=KeyboardInterrupt()
        ):
            with self.assertRaises(KeyboardInterrupt):
                sentinel._run_worker_process(
                    config, estimates, {"cgroup": {}}
                )
        self.assertIsNotNone(worker.stop_signal)
        self.assertIsNotNone(worker.returncode)

    def test_term_and_kill_timeout_records_unreaped_cleanup_failure(self):
        raw = {
            **_TOY,
            "mode": "complete",
            "deadline": time.monotonic() + 30.0,
            "execute_large_profile": False,
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        worker = _UnreapableWorker()
        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(
            sentinel.os, "killpg", return_value=None
        ), mock.patch.object(
            sentinel,
            "_current_rss_bytes",
            side_effect=RuntimeError("force cleanup"),
        ):
            result = sentinel._run_worker_process(
                config, estimates, {"cgroup": {}}
            )
        self.assertEqual(result["status"], "worker_cleanup_failed")
        self.assertFalse(result["worker_cleanup"]["reaped"])
        self.assertIsNone(result["worker_cleanup"]["exit_code"])
        self.assertEqual(
            result["worker_cleanup"]["cleanup_error"],
            "worker_not_reaped_after_term_and_kill",
        )
        self.assertTrue(result["worker_cleanup"]["sigterm_attempted"])
        self.assertTrue(result["worker_cleanup"]["sigkill_attempted"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def _toy_worker_inputs(self):
        raw = {
            **_TOY,
            "mode": "complete",
            "deadline": time.monotonic() + 30.0,
            "execute_large_profile": False,
            "absolute_rss_cap_bytes": sentinel.HARD_ABSOLUTE_RSS_CAP_BYTES,
            "rss_reserve_bytes": sentinel.MINIMUM_RSS_RESERVE_BYTES,
            "profile_name": None,
        }
        config, estimates = sentinel._validate_config(raw)
        return config, estimates, {"cgroup": {}}

    def _run_completed_payload(self, payload, *, validator_error=False):
        config, estimates, preflight = self._toy_worker_inputs()

        def popen(*_args, **kwargs):
            return _CompletedPayloadWorker(payload, kwargs["stdout"])

        validator_patch = (
            mock.patch.object(
                sentinel,
                "_valid_success_contract",
                side_effect=RuntimeError("validator exploded"),
            )
            if validator_error
            else contextlib.nullcontext()
        )
        with mock.patch.object(
            sentinel.subprocess, "Popen", side_effect=popen
        ), validator_patch:
            return sentinel._run_worker_process(
                config, estimates, preflight
            )

    def _run_completed_raw(self, raw, *, checksum_error=False, json_error=False):
        config, estimates, preflight = self._toy_worker_inputs()

        def popen(*_args, **kwargs):
            return _CompletedRawWorker(raw, kwargs["stdout"])

        checksum_patch = (
            mock.patch.object(
                sentinel,
                "verify_diagnostic_checksum",
                side_effect=RuntimeError("checksum verifier exploded"),
            )
            if checksum_error
            else contextlib.nullcontext()
        )
        json_patch = (
            mock.patch.object(
                sentinel.json,
                "loads",
                side_effect=MemoryError("json allocation failed"),
            )
            if json_error
            else contextlib.nullcontext()
        )
        with mock.patch.object(
            sentinel.subprocess, "Popen", side_effect=popen
        ), checksum_patch, json_patch:
            return sentinel._run_worker_process(
                config, estimates, preflight
            )

    def test_malformed_checksummed_checkpoint_is_sealed_protocol_error(self):
        for malformed in ([], {"stage": "baseline"}):
            with self.subTest(kind=type(malformed).__name__):
                child = copy.deepcopy(self.complete)
                child["memory_checkpoints"][0] = malformed
                child["memory_trace_sha256"] = sentinel._canonical_sha256(
                    {"memory_checkpoints": child["memory_checkpoints"]}
                )
                child = sentinel._seal(child)
                result = self._run_completed_payload(child)
                self.assertEqual(result["status"], "worker_protocol_error")
                self.assertIn("checkpoint", result["reason"])
                self.assertTrue(sentinel.verify_diagnostic_checksum(result))

        result = self._run_completed_payload(
            self.complete, validator_error=True
        )
        self.assertEqual(result["status"], "worker_protocol_error")
        self.assertIn("validator_exception", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_recursive_json_and_checksum_structures_fail_closed(self):
        recursive_json = (
            ("[" * 10_000) + "0" + ("]" * 10_000)
        ).encode("ascii")
        result = self._run_completed_raw(recursive_json)
        self.assertEqual(result["status"], "worker_protocol_error")
        self.assertIn("worker_json_invalid:RecursionError", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

        nested = {"diagnostic_sha256": "0" * 64}
        cursor = nested
        for _ in range(800):
            child = {}
            cursor["nested"] = child
            cursor = child
        self.assertFalse(sentinel.verify_diagnostic_checksum(nested))

        result = self._run_completed_raw(
            b"{}", checksum_error=True
        )
        self.assertEqual(result["status"], "worker_protocol_error")
        self.assertIn("checksum_verifier_exception", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

        result = self._run_completed_raw(b"{}", json_error=True)
        self.assertEqual(result["status"], "worker_protocol_error")
        self.assertIn("worker_json_invalid:MemoryError", result["reason"])
        self.assertTrue(sentinel.verify_diagnostic_checksum(result))

    def test_parent_monitor_reads_worker_pid_cgroup_and_stops_migration(self):
        config, estimates, _ = self._toy_worker_inputs()
        config["large_topology"] = True
        parent_cgroup = _bounded_cgroup()
        moved_cgroup = _bounded_cgroup("/sys/fs/cgroup/moved")
        preflight = {"cgroup": parent_cgroup}
        worker = _NeverStartedWorker()

        def killpg(_pid, sig):
            worker.stop_signal = sig

        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(
            sentinel.os, "killpg", side_effect=killpg
        ), mock.patch.object(
            sentinel, "_current_rss_bytes", return_value=1
        ), mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=moved_cgroup
        ) as read_cgroup:
            result = sentinel._run_worker_process(
                config, estimates, preflight
            )
        self.assertEqual(result["status"], "worker_stopped")
        self.assertEqual(
            result["reason"], "cgroup_binding_changed_during_worker"
        )
        self.assertTrue(result["worker_cleanup"]["reaped"])
        self.assertTrue(read_cgroup.call_args_list)
        self.assertTrue(
            all(call.args == (worker.pid,) for call in read_cgroup.call_args_list)
        )

    def test_parent_monitor_stops_live_worker_when_pid_membership_disappears(self):
        config, estimates, _ = self._toy_worker_inputs()
        config["large_topology"] = True
        preflight = {"cgroup": _bounded_cgroup()}
        worker = _NeverStartedWorker()

        def killpg(_pid, sig):
            worker.stop_signal = sig

        unavailable = sentinel._empty_cgroup("membership_read_failed:test")
        with mock.patch.object(
            sentinel.subprocess, "Popen", return_value=worker
        ), mock.patch.object(
            sentinel.os, "killpg", side_effect=killpg
        ), mock.patch.object(
            sentinel, "_current_rss_bytes", return_value=1
        ), mock.patch.object(
            sentinel, "_read_cgroup_v2", return_value=unavailable
        ) as read_cgroup:
            result = sentinel._run_worker_process(
                config, estimates, preflight
            )
        self.assertEqual(result["status"], "worker_stopped")
        self.assertEqual(
            result["reason"],
            "worker_pid_cgroup_contract_unavailable_during_worker",
        )
        self.assertTrue(result["worker_cleanup"]["reaped"])
        read_cgroup.assert_called_with(worker.pid)


if __name__ == "__main__":
    unittest.main()
