#!/usr/bin/env python3
"""Small, non-timing tests for the V5.1 controlled audit harness."""

from __future__ import annotations

import hashlib
import inspect
import io
import tempfile
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest import mock
from contextlib import redirect_stderr

import numpy as np

from act.back_end.hybridz_tf.query_dual_blas_contract import (
    QueryDualBlasContract,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
)
from act.pipeline.verification import (
    query_dual_v51_controlled_audit as audit,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _tiny_parameters() -> audit.AuditParameters:
    return audit.AuditParameters(
        stages=(
            audit.StageSpec(0, 3, 2, 2),
            audit.StageSpec(1, 6, 5, 1),
            audit.StageSpec(2, 10, 9, 2),
            audit.StageSpec(3, 15, 14, 1),
            audit.StageSpec(4, None, None, 3),
        ),
        pairs=5,
        blas_threads=4,
        chunk_size=2,
        workspace_bytes=8 * audit._MIB,
        rss_limit_bytes=256 * audit._MIB,
        bootstrap_samples=100,
        session_timeout_s=60.0,
        root_timeout_s=60.0,
    )


class QueryDualV51ControlledAuditTests(unittest.TestCase):
    def test_official_configuration_is_fixed_and_cli_has_no_knobs(self):
        audit._assert_official_configuration(
            audit.OFFICIAL_PARAMETERS, audit.OFFICIAL_TOPOLOGY
        )
        self.assertEqual(
            tuple(
                stage.objective_count
                for stage in audit.OFFICIAL_PARAMETERS.stages
            ),
            (32, 16, 48, 32, 99),
        )
        self.assertEqual(
            tuple(
                stage.cone_start_lid
                for stage in audit.OFFICIAL_PARAMETERS.stages
            ),
            (2, 5, 9, 14, None),
        )
        self.assertEqual(audit.OFFICIAL_PARAMETERS.pairs, 5)
        self.assertEqual(audit.OFFICIAL_PARAMETERS.blas_threads, 4)
        self.assertEqual(
            audit.OFFICIAL_PARAMETERS.workspace_bytes, 512 * audit._MIB
        )
        self.assertEqual(audit._OFFICIAL_CPU_AFFINITY, (4, 5, 6, 7))
        run_source = inspect.getsource(audit.run_audit)
        self.assertIn("warm_v3_cpu_start", run_source)
        self.assertIn("warm_v51_cpu_start", run_source)
        self.assertIn("implementation_cpu_start", run_source)
        self.assertIn(
            "per_timed_implementation_external_cpu_passed", run_source
        )
        changed = audit.AuditParameters(
            **{
                **audit.asdict(audit.OFFICIAL_PARAMETERS),
                "pairs": 4,
            }
        )
        with self.assertRaisesRegex(
            RuntimeError, "configuration was substituted"
        ):
            audit._assert_official_configuration(
                changed, audit.OFFICIAL_TOPOLOGY
            )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                audit._parse_arguments(
                    ["--output", "unused.json", "--pairs", "3"]
                )

    def test_fixed_environment_requires_four_and_dynamic_off(self):
        valid = {
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "MKL_DYNAMIC": "FALSE",
            "OMP_DYNAMIC": "0",
        }
        self.assertTrue(audit._fixed_environment(valid)["passed"])
        invalid = dict(valid)
        invalid["MKL_NUM_THREADS"] = "8"
        invalid["OMP_DYNAMIC"] = "TRUE"
        record = audit._fixed_environment(invalid)
        self.assertFalse(record["passed"])
        self.assertEqual(record["thread_mismatches"]["MKL_NUM_THREADS"], "8")
        self.assertEqual(record["dynamic_mismatches"]["OMP_DYNAMIC"], "TRUE")

    def test_fraction_manifest_binds_both_5000_row_tests(self):
        root = Path(__file__).resolve().parents[3]
        manifest = audit._fraction_gate_manifest(root)
        self.assertEqual(set(manifest), {"dense", "conv"})
        self.assertEqual(manifest["dense"]["minimum_rows"], 5_000)
        self.assertEqual(manifest["conv"]["minimum_rows"], 5_000)
        self.assertTrue(
            manifest["dense"]["static_minimum_rows_verified"]
        )
        self.assertTrue(
            manifest["conv"]["static_minimum_rows_verified"]
        )

    def test_frozen_numeric_source_manifest_is_current_and_fail_closed(self):
        root = Path(__file__).resolve().parents[3]
        observed = audit._source_hashes(
            root, audit.NUMERIC_SOURCE_PATHS
        )
        record = audit._validate_expected_numeric_sources(observed)
        self.assertTrue(record["passed"])
        changed = dict(observed)
        first = audit.NUMERIC_SOURCE_PATHS[0]
        changed[first] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source closure changed"):
            audit._validate_expected_numeric_sources(changed)

    def test_numeric_ast_closure_passes_and_cuda_injection_fails(self):
        root = Path(__file__).resolve().parents[3]
        record = audit._numeric_device_audit(root)
        self.assertTrue(record["passed"], record)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "bad.py").write_text(
                "import cupy\nvalue.cuda()\n", encoding="utf-8"
            )
            bad = audit._numeric_device_audit(
                path, relative_paths=("bad.py",)
            )
        self.assertFalse(bad["passed"])
        self.assertGreaterEqual(
            len(bad["forbidden_gpu_compute_findings"]), 2
        )

    def test_solver_bootstrap_is_transparently_recorded_not_called(self):
        root = Path(__file__).resolve().parents[3]
        record = audit._operator_solver_usage_audit(root)
        self.assertTrue(record["passed"], record)
        self.assertTrue(
            record["direct_operator_or_solver_imports_absent"]
        )
        self.assertFalse(record["operator_or_solver_called"])
        self.assertFalse(record["solver_verdict_created"])
        self.assertIsInstance(record["ambient_solver_modules_loaded"], list)
        self.assertIn("package bootstrap", record["ambient_import_explanation"])

    def test_host_preflight_excludes_full_process_ancestry(self):
        sample = {
            "requested_sample_seconds": 0.25,
            "observed_sample_seconds": 0.25,
            "clock_ticks_per_second": 100,
            "threshold_cpu_core_equivalents": 0.5,
            "readable_process_count_first": 1,
            "readable_process_count_second": 1,
            "high_cpu_competitors": (),
            "aggregate_external_cpu": {
                "external_cpu_core_equivalents": 0.0,
                "passed": True,
            },
            "passed": True,
        }
        with mock.patch.object(
            audit, "_sample_cpu_competitors", return_value=sample
        ) as sampler:
            record = audit._host_preflight(load_limit=None)
        self.assertFalse(record["load_gate_enforced"])
        self.assertTrue(record["competing_cpu_process_gate_enforced"])
        self.assertEqual(record["high_cpu_competitors"], ())
        self.assertEqual(record["required_cpu_affinity"], [4, 5, 6, 7])
        self.assertIn(
            audit.os.getppid(), record["excluded_process_ancestry"]
        )
        self.assertTrue(
            all(
                item["pid"] not in record["excluded_process_ancestry"]
                for item in record["other_query_dual_audit_workers"]
            )
        )
        excluded = sampler.call_args.kwargs["excluded_pids"]
        self.assertIn(audit.os.getpid(), excluded)
        self.assertTrue(
            set(record["excluded_process_ancestry"]).issubset(excluded)
        )

    def test_cpu_competitor_gate_uses_pid_seal_and_core_rate(self):
        first = {
            10: {
                "pid": 10,
                "uid": 1000,
                "cpu_ticks": 100,
                "starttime_ticks": 20,
                "command": "busy",
            },
            11: {
                "pid": 11,
                "uid": 1001,
                "cpu_ticks": 100,
                "starttime_ticks": 30,
                "command": "slow",
            },
            12: {
                "pid": 12,
                "uid": 1000,
                "cpu_ticks": 100,
                "starttime_ticks": 40,
                "command": "reused-pid",
            },
            13: {
                "pid": 13,
                "uid": 1000,
                "cpu_ticks": 100,
                "starttime_ticks": 50,
                "command": "ancestor",
            },
        }
        second = {
            10: {**first[10], "cpu_ticks": 125},
            11: {**first[11], "cpu_ticks": 112},
            12: {
                **first[12],
                "cpu_ticks": 130,
                "starttime_ticks": 41,
            },
            13: {**first[13], "cpu_ticks": 200},
        }
        competitors = audit._cpu_competitors_from_snapshots(
            first,
            second,
            elapsed_s=0.25,
            ticks_per_second=100,
            excluded_pids={13},
            threshold_cores=0.5,
        )
        self.assertEqual([item["pid"] for item in competitors], [10])
        self.assertEqual(competitors[0]["uid"], 1000)
        self.assertAlmostEqual(
            competitors[0]["cpu_core_equivalents"], 1.0
        )
        self.assertEqual(competitors[0]["cpu_ticks_delta"], 25)

    def test_aggregate_cpu_gate_catches_subthreshold_and_process_churn(self):
        first = {
            pid: {
                "pid": pid,
                "uid": 1000,
                "cpu_ticks": 0,
                "starttime_ticks": pid * 10,
                "command": f"worker-{pid}",
            }
            for pid in range(20, 24)
        }
        second = {
            pid: {**record, "cpu_ticks": 48}
            for pid, record in first.items()
        }
        individual = audit._cpu_competitors_from_snapshots(
            first,
            second,
            elapsed_s=1.0,
            ticks_per_second=100,
            excluded_pids=(),
            threshold_cores=0.5,
        )
        self.assertEqual(individual, ())
        aggregate = audit._external_cpu_window_from_counters(
            global_busy_ticks_start=1_000,
            global_busy_ticks_end=1_217,
            self_cpu_ticks_start=100,
            self_cpu_ticks_end=125,
            elapsed_ns=1_000_000_000,
            ticks_per_second=100,
            limit_cores=0.5,
        )
        self.assertAlmostEqual(
            aggregate["external_cpu_core_equivalents"], 1.92
        )
        self.assertFalse(aggregate["passed"])

        exited = {30: first[20]}
        started = {31: second[20]}
        self.assertEqual(
            audit._cpu_competitors_from_snapshots(
                exited,
                started,
                elapsed_s=1.0,
                ticks_per_second=100,
                excluded_pids=(),
                threshold_cores=0.5,
            ),
            (),
        )
        churn = audit._external_cpu_window_from_counters(
            global_busy_ticks_start=2_000,
            global_busy_ticks_end=2_105,
            self_cpu_ticks_start=200,
            self_cpu_ticks_end=225,
            elapsed_ns=1_000_000_000,
            ticks_per_second=100,
            limit_cores=0.5,
        )
        self.assertAlmostEqual(
            churn["external_cpu_core_equivalents"], 0.8
        )
        self.assertFalse(churn["passed"])

        exact_boundary = audit._external_cpu_window_from_counters(
            global_busy_ticks_start=3_000,
            global_busy_ticks_end=3_075,
            self_cpu_ticks_start=300,
            self_cpu_ticks_end=325,
            elapsed_ns=1_000_000_000,
            ticks_per_second=100,
            limit_cores=0.5,
        )
        self.assertAlmostEqual(
            exact_boundary["external_cpu_core_equivalents"], 0.5
        )
        self.assertFalse(exact_boundary["passed"])
        self.assertEqual(
            exact_boundary["strict_integer_comparison"]["operator"], "<"
        )
        biased_run = audit._external_cpu_window_from_counters(
            global_busy_ticks_start=4_000,
            global_busy_ticks_end=4_125,
            self_cpu_ticks_start=400,
            self_cpu_ticks_end=425,
            elapsed_ns=1_000_000_000,
            ticks_per_second=100,
            limit_cores=0.5,
        )
        diluted_total = audit._external_cpu_window_from_counters(
            global_busy_ticks_start=5_000,
            global_busy_ticks_end=5_270,
            self_cpu_ticks_start=500,
            self_cpu_ticks_end=750,
            elapsed_ns=10_000_000_000,
            ticks_per_second=100,
            limit_cores=0.5,
        )
        self.assertFalse(biased_run["passed"])
        self.assertTrue(diluted_total["passed"])

    def test_affinity_busy_fields_do_not_double_count_guest_or_iowait(self):
        record = audit._read_affinity_cpu_counters(
            (0,),
            proc_stat_text=(
                "cpu 0 0 0 0 0 0 0 0 0 0\n"
                "cpu0 10 2 3 4 5 6 7 8 9 10\n"
            ),
        )
        self.assertEqual(record["busy_ticks"], 36)
        self.assertEqual(record["iowait_ticks"], 5)
        self.assertEqual(
            record["busy_components"],
            {
                "user": 10,
                "nice": 2,
                "system": 3,
                "irq": 6,
                "softirq": 7,
                "steal": 8,
            },
        )
        self.assertIn("guest already included", record["field_policy"])

    def test_process_snapshot_retains_denied_cmdline_processes(self):
        with mock.patch.object(
            Path, "read_bytes", side_effect=PermissionError
        ):
            snapshot = audit._read_process_cpu_snapshot()
        record = snapshot[audit.os.getpid()]
        self.assertEqual(record["uid"], audit.os.getuid())
        self.assertGreater(record["starttime_ticks"], 0)
        self.assertGreaterEqual(record["cpu_ticks"], 0)
        self.assertRegex(record["command"], r"^\[.*\]$")

    def test_bootstrap_and_tightness_comparison(self):
        arrays = tuple(
            np.asarray([float(index), -0.0], dtype=np.float64)
            for index in range(5)
        )
        old = {"arrays": arrays}
        same = {"arrays": tuple(value.copy() for value in arrays)}
        comparison = audit._compare_runs(old, same)
        self.assertEqual(comparison["objective_count"], 10)
        self.assertEqual(comparison["tightness_regression_count"], 0)
        changed = [value.copy() for value in arrays]
        changed[2][0] = np.nextafter(changed[2][0], -np.inf)
        regression = audit._compare_runs(old, {"arrays": tuple(changed)})
        self.assertEqual(regression["tightness_regression_count"], 1)
        lower = audit._bootstrap_lower(
            [4.0, 4.1, 3.9, 4.2, 3.8],
            [2.0, 2.05, 1.95, 2.1, 1.9],
            samples=1_000,
        )
        self.assertGreaterEqual(lower, 1.99)

    def test_receipt_integrity_is_candidate_only(self):
        gates = {
            name: False for name in audit._REQUIRED_PROMOTION_GATE_KEYS
        }
        body = {
            "schema": audit.SCHEMA,
            "status": "rejected",
            "proof_authority": False,
            "controlled_synthetic_only": True,
            "real_onnx_or_vnnlib_accessed": False,
            "direct_operator_or_solver_imports": False,
            "operator_or_solver_called": False,
            "solver_verdict_created": False,
            "ambient_solver_modules_loaded": ["gurobipy"],
            "official_configuration_sha256": (
                audit.OFFICIAL_CONFIGURATION_SHA256
            ),
            "configuration": audit._configuration_record(
                audit.OFFICIAL_PARAMETERS, audit.OFFICIAL_TOPOLOGY
            ),
            "gates": gates,
            "operator_solver_usage_audit": {
                "direct_operator_or_solver_imports_absent": True,
                "operator_or_solver_called": False,
                "solver_verdict_created": False,
                "ambient_solver_modules_loaded": ["gurobipy"],
            },
        }
        body["receipt_sha256"] = audit._json_sha256(body)
        self.assertTrue(
            audit.verify_query_dual_v51_controlled_audit_receipt(body)
        )
        changed = dict(body)
        changed["proof_authority"] = True
        self.assertFalse(
            audit.verify_query_dual_v51_controlled_audit_receipt(changed)
        )
        substituted = dict(body)
        substituted["configuration"] = {
            **body["configuration"],
            "topology": {
                **body["configuration"]["topology"],
                "classes": 99,
            },
        }
        substituted["receipt_sha256"] = audit._json_sha256(
            {
                key: value
                for key, value in substituted.items()
                if key != "receipt_sha256"
            }
        )
        self.assertFalse(
            audit.verify_query_dual_v51_controlled_audit_receipt(
                substituted
            )
        )
        inconsistent = dict(body)
        inconsistent["status"] = "passed"
        inconsistent["receipt_sha256"] = audit._json_sha256(
            {
                key: value
                for key, value in inconsistent.items()
                if key != "receipt_sha256"
            }
        )
        self.assertFalse(
            audit.verify_query_dual_v51_controlled_audit_receipt(
                inconsistent
            )
        )

    def test_receipt_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            value = {
                "schema": "test",
                "proof_authority": False,
            }
            audit._write_atomic(output, value)
            first = output.read_bytes()
            with self.assertRaises(FileExistsError):
                audit._write_atomic(output, {"changed": True})
            self.assertEqual(output.read_bytes(), first)

    def test_fatal_precondition_leaves_no_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fatal.json"
            with mock.patch.object(
                audit, "_fixed_environment", return_value={"passed": False}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "fixed four-thread"
                ):
                    audit.run_audit(
                        project_root=Path(__file__).resolve().parents[3],
                        output=output,
                    )
            self.assertFalse(output.exists())

            output.write_text("preserve", encoding="ascii")
            with self.assertRaises(FileExistsError):
                audit.run_audit(
                    project_root=Path(__file__).resolve().parents[3],
                    output=output,
                )
            self.assertEqual(
                output.read_text(encoding="ascii"), "preserve"
            )

    def test_tiny_real_root_and_full_sessions_smoke(self):
        """Exercise the real root/frame/stage/commit path without timing."""

        profile = audit.TopologyProfile(
            channels=2,
            high_hw=2,
            low_hw=1,
            classes=3,
            weight_scale=0.025,
        )
        parameters = _tiny_parameters()
        net = audit._build_topology(profile, seed=1234)
        spec = net.by_id[1]
        self.assertIn("lb", spec.params)
        self.assertIn("ub", spec.params)
        root = certify_query_dual_boxes(
            net, timeout_s=30.0, conv_channel_chunk=1
        )
        self.assertTrue(verify_query_dual_box_certificate(root, net=net))
        stages = audit._query_schedule(net, parameters, seed=5678)
        material_before = audit._stage_material_seals(stages)
        self.assertEqual(
            tuple(stage["cone_start_lid"] for stage in stages),
            (2, 5, 9, 14, None),
        )
        for stage in stages:
            self.assertFalse(stage["query_rows"].flags.writeable)
            with self.assertRaises(ValueError):
                stage["query_rows"].setflags(write=True)
            self.assertEqual(
                stage["query_rows_sha256"],
                audit._array_sha256(stage["query_rows"]),
            )
            self.assertEqual(
                stage["alpha_sha256"],
                audit._alpha_sha256(stage["alpha"]),
            )
            self.assertTrue(
                all(
                    not value.flags.writeable
                    for value in stage["alpha"].values()
                )
            )
            for value in stage["alpha"].values():
                with self.assertRaises(ValueError):
                    value.setflags(write=True)
        contract = QueryDualBlasContract(
            required_threads=4,
            content_sha256=_sha("tiny-controlled-blas"),
            receipt=MappingProxyType({}),
        )
        old = audit._run_v3_schedule(net, root, stages, parameters)
        with mock.patch.object(
            audit.v51_session,
            "validate_query_dual_blas_contract",
            return_value=True,
        ):
            new = audit._run_v51_schedule(
                net, root, stages, parameters, contract
            )
        self.assertTrue(old["single_root_certificate"])
        self.assertTrue(old["single_bounds_frame"])
        self.assertTrue(new["single_root_certificate"])
        self.assertTrue(new["single_bounds_frame"])
        self.assertTrue(new["stage_uses_committed_once_in_order"])
        self.assertTrue(new["commit_live_blas_recheck_bound"])
        self.assertEqual(
            old["same_certified_bounds_sha256"],
            new["same_certified_bounds_sha256"],
        )
        comparison = audit._compare_runs(old, new)
        self.assertEqual(comparison["objective_count"], 9)
        self.assertEqual(comparison["tightness_regression_count"], 0)
        self.assertEqual(
            audit._stage_material_seals(stages), material_before
        )


if __name__ == "__main__":
    unittest.main()
