"""Toy-only tests for the bounded phase-clique build probe."""

from __future__ import annotations

from argparse import Namespace
import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from types import MappingProxyType, ModuleType, SimpleNamespace
import weakref

import numpy as np
import scipy.sparse as sp

from act.pipeline.verification import hybridz_phase_clique_build_probe as probe


class _HZ:
    def __init__(self, *, upper_rows: int) -> None:
        self.c = np.array([0.0, 0.0], dtype=np.float64)
        self.Gc = sp.csr_matrix(np.eye(2, dtype=np.float64))
        self.Gb = sp.csr_matrix((2, 0), dtype=np.float64)
        self.Auc = sp.csr_matrix((upper_rows, 2), dtype=np.float64)
        self.Aub = sp.csr_matrix((upper_rows, 0), dtype=np.float64)
        self.Ac = sp.csr_matrix((0, 2), dtype=np.float64)
        self.Ab = sp.csr_matrix((0, 0), dtype=np.float64)
        self.ub = np.ones(upper_rows, dtype=np.float64)
        self.b = np.empty(0, dtype=np.float64)
        self.n_cont = 2
        self.n_bin = 0
        self.n_ub = upper_rows
        self.n_eq = 0


def _native_hz(*, upper_rows: int):
    from act.back_end.solver.solver_hz import SparseHZono

    return SparseHZono(
        c=np.array([0.0, 0.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.eye(2, dtype=np.float64)),
        Gb=sp.csr_matrix((2, 0), dtype=np.float64),
        Auc=sp.csr_matrix((upper_rows, 2), dtype=np.float64),
        Aub=sp.csr_matrix((upper_rows, 0), dtype=np.float64),
        Ac=sp.csr_matrix((0, 2), dtype=np.float64),
        Ab=sp.csr_matrix((0, 0), dtype=np.float64),
        ub=np.ones(upper_rows, dtype=np.float64),
        b=np.empty(0, dtype=np.float64),
        col_ids=np.array([0, 1], dtype=np.int64),
        bcol_ids=np.empty(0, dtype=np.int64),
    )


class _Build:
    def __init__(self, hz: _HZ) -> None:
        self.hz = hz


class _Result:
    def __init__(self, public: _Build) -> None:
        self.build = public
        self.status = "fresh_verified_k4_clique_materialized"
        self.materialized = True
        self.identity_preserved = False
        self.receipt = {
            "receipt_sha256": "a" * 64,
            "focused_encoded_row": 1,
            "clique_count": 1,
            "cut_row_count": 1,
            "timings": {"total_seconds": 0.01},
        }


class PhaseCliqueBuildProbeTests(unittest.TestCase):
    def tearDown(self) -> None:
        with probe._PCOH_K2_TRUSTED_TRANSACTION_LOCK:
            probe._PCOH_K2_TRUSTED_TRANSACTIONS.clear()
        with probe._PCOH_K3_TRUSTED_TRANSACTION_LOCK:
            probe._PCOH_K3_TRUSTED_TRANSACTIONS.clear()

    def test_resource_capture_reports_current_rss_without_gate_authority(
        self,
    ) -> None:
        receipt = probe._capture_resource_peaks()
        self.assertIsInstance(receipt["peak_rss_bytes"], int)
        current = receipt["current_rss_bytes"]
        self.assertTrue(current is None or type(current) is int)
        if current is not None:
            self.assertGreater(current, 0)

    def test_pcoh_k3_fixed_baseline_artifact_and_source_preflight_are_strict(
        self,
    ) -> None:
        anchor = probe._pcoh_k3_fixed_baseline_artifact_anchor(
            deadline=time.monotonic() + 5.0
        )
        self.assertTrue(probe._pcoh_k3_baseline_anchor_receipt_valid(anchor))
        self.assertEqual(
            anchor["artifact_sha256"],
            "01625add9f435eefef20e3eaa6dcaf72f2ce0f50137f19a611c576c1829846b0",
        )
        artifact_path = (
            Path(probe.__file__).resolve().parents[3]
            / probe._PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        real_summary = copy.deepcopy(
            artifact["pcoh_k2_build_only"][
                "materialized_tightness_summary"
            ]
        )
        self.assertEqual(
            Fraction(*real_summary["global_cube_upper_exact"]),
            probe._PCOH_K3_GLOBAL_CUBE_UPPER_EXACT,
        )
        stored_global = Fraction.from_float(
            float.fromhex(real_summary["global_cube_upper_hex"])
        )
        self.assertGreater(
            stored_global, probe._PCOH_K3_GLOBAL_CUBE_UPPER_EXACT
        )
        real_summary["stable_bit_ids"] = [52557, 52558, 52559]
        real_gate = probe._pcoh_k3_strong_tightness_gate(
            real_summary,
            source_semantic_digest=(
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            selection_digest=probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
            focused_encoded_row=probe._PCOH_K3_FOCUSED_ENCODED_ROW,
            focused_rival_id=probe._PCOH_K3_FOCUSED_RIVAL_ID,
            retained_k2_stable_bit_ids=(52557, 52558),
            stable_bit_ids=(52557, 52558, 52559),
        )
        self.assertEqual(real_gate["status"], "built_but_not_strong")
        collapsed = copy.deepcopy(real_summary)
        collapsed["global_cube_upper_exact"] = [
            stored_global.numerator,
            stored_global.denominator,
        ]
        with self.assertRaisesRegex(
            probe.PhaseCliqueBuildProbeError,
            "exact fixed-anchor mismatch",
        ):
            probe._pcoh_k3_strong_tightness_gate(
                collapsed,
                source_semantic_digest=(
                    probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
                ),
                selection_digest=probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
                focused_encoded_row=probe._PCOH_K3_FOCUSED_ENCODED_ROW,
                focused_rival_id=probe._PCOH_K3_FOCUSED_RIVAL_ID,
                retained_k2_stable_bit_ids=(52557, 52558),
                stable_bit_ids=(52557, 52558, 52559),
            )
        shape = {
            "output_dimension": 100,
            "continuous_columns": 52_657,
            "binary_columns": 4,
            "upper_rows": 98_974,
            "equality_rows": 0,
            "constraint_nonzeros": 9_267_556,
            "generator_nonzeros": 10_100,
        }
        preflight = probe._pcoh_k3_source_build_preflight(
            shape,
            build_seconds=1.0,
            input_sha256=self._pcoh_inputs(),
            implementation_sha256=self._pcoh_k3_implementation(),
            baseline_anchor_receipt=anchor,
        )
        self.assertEqual(preflight["status"], "passed")
        self.assertTrue(probe._pcoh_k3_source_build_preflight_valid(preflight))
        for name, mutate in (
            (
                "artifact_digest",
                lambda value: value["baseline_anchor_receipt"].__setitem__(
                    "artifact_sha256", "f" * 64
                ),
            ),
            (
                "implementation_keyset",
                lambda value: value["implementation_sha256"].pop(
                    next(iter(value["implementation_sha256"]))
                ),
            ),
        ):
            with self.subTest(name=name):
                forged = copy.deepcopy(preflight)
                mutate(forged)
                forged.pop("receipt_sha256")
                forged["receipt_sha256"] = hashlib.sha256(
                    probe._canonical_json(forged)
                ).hexdigest()
                self.assertFalse(
                    probe._pcoh_k3_source_build_preflight_valid(forged)
                )

        with tempfile.TemporaryDirectory() as raw:
            temporary_root = Path(raw)
            target = (
                Path(probe.__file__).resolve().parents[3]
                / probe._PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
            )
            link = temporary_root / probe._PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            with mock.patch.object(probe, "_REPO_ROOT", temporary_root):
                with self.assertRaisesRegex(
                    probe.PhaseCliqueBuildProbeError, "symlinked"
                ):
                    probe._pcoh_k3_fixed_baseline_artifact_anchor(
                        deadline=time.monotonic() + 5.0
                    )

    def test_glibc_malloc_trim_contract_and_active_buffer_are_exact(self) -> None:
        class FakeFunction:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error
                self.argtypes = None
                self.restype = None
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                if self.error is not None:
                    raise self.error
                return self.result

        version = FakeFunction(b"2.36")
        trim = FakeFunction(1)
        libc = SimpleNamespace(
            gnu_get_libc_version=version,
            malloc_trim=trim,
        )
        rss = mock.Mock(side_effect=(1_000_000, 700_000))
        active = np.arange(4096, dtype=np.float64)
        active_sha256 = hashlib.sha256(active.tobytes()).hexdigest()
        with mock.patch.object(probe.sys, "platform", "linux"):
            receipt = probe._glibc_malloc_trim_diagnostic(
                rss_reader=rss,
                library_loader=lambda *_args, **_kwargs: libc,
            )
        self.assertEqual(
            receipt["schema"],
            "act.hybridz_glibc_malloc_trim_diagnostic.v1",
        )
        self.assertEqual(receipt["status"], "called_memory_released")
        self.assertEqual(receipt["return_code"], 1)
        self.assertEqual(receipt["current_rss_before_bytes"], 1_000_000)
        self.assertEqual(receipt["current_rss_after_bytes"], 700_000)
        self.assertEqual(receipt["released_bytes"], 300_000)
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["verdict_authority"])
        self.assertFalse(receipt["gate_authority"])
        self.assertEqual(trim.calls, [(0,)])
        self.assertEqual(
            hashlib.sha256(active.tobytes()).hexdigest(),
            active_sha256,
        )

    def test_malloc_trim_missing_or_exception_is_diagnostic_only(self) -> None:
        class FakeFunction:
            def __init__(self, result=None, error=None):
                self.result = result
                self.error = error
                self.argtypes = None
                self.restype = None

            def __call__(self, *_args):
                if self.error is not None:
                    raise self.error
                return self.result

        active = np.linspace(-1.0, 1.0, 8192, dtype=np.float64)
        expected = active.copy()
        missing = SimpleNamespace()
        failing = SimpleNamespace(
            gnu_get_libc_version=FakeFunction(b"2.36"),
            malloc_trim=FakeFunction(error=RuntimeError("toy failure")),
        )
        for name, libc, expected_status in (
            (
                "missing",
                missing,
                "unsupported_non_glibc_or_symbol_missing",
            ),
            ("exception", failing, "call_error"),
        ):
            with self.subTest(name=name), mock.patch.object(
                probe.sys, "platform", "linux"
            ):
                receipt = probe._glibc_malloc_trim_diagnostic(
                    rss_reader=mock.Mock(side_effect=(2048, 2048)),
                    library_loader=lambda *_args, **_kwargs: libc,
                )
                self.assertEqual(receipt["status"], expected_status)
                self.assertIsNone(receipt["return_code"])
                self.assertEqual(receipt["released_bytes"], 0)
                self.assertTrue(np.array_equal(active, expected))

    def test_glibc_malloc_trim_controlled_subprocess_contract(self) -> None:
        source = r'''
import ctypes
import hashlib
import json
import numpy as np
from act.pipeline.verification.hybridz_phase_clique_build_probe import _glibc_malloc_trim_diagnostic
libc = ctypes.CDLL("libc.so.6")
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
active = np.arange(4096, dtype=np.float64)
before = hashlib.sha256(active.tobytes()).hexdigest()
size = 64 * 1024
pointers = []
for _ in range(2048):
    pointer = libc.malloc(size)
    if not pointer:
        raise MemoryError("controlled allocation failed")
    libc.memset(pointer, 0x5A, size)
    pointers.append(pointer)
for pointer in pointers:
    libc.free(pointer)
receipt = _glibc_malloc_trim_diagnostic()
print(json.dumps({
    "receipt": receipt,
    "active_unchanged": before == hashlib.sha256(active.tobytes()).hexdigest(),
}, sort_keys=True))
'''
        environment = dict(os.environ)
        environment.update({
            "MALLOC_ARENA_MAX": "1",
            "MALLOC_MMAP_THRESHOLD_": str(1024 * 1024),
            "MALLOC_TRIM_THRESHOLD_": str(1024 * 1024 * 1024),
        })
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout.splitlines()[-1])
        receipt = payload["receipt"]
        self.assertTrue(payload["active_unchanged"])
        self.assertEqual(
            receipt["schema"],
            "act.hybridz_glibc_malloc_trim_diagnostic.v1",
        )
        self.assertIn(
            receipt["status"],
            {"called_memory_released", "called_no_memory_released"},
        )
        self.assertIn(receipt["return_code"], {0, 1})
        self.assertIsInstance(receipt["current_rss_before_bytes"], int)
        self.assertIsInstance(receipt["current_rss_after_bytes"], int)
        self.assertIsInstance(receipt["released_bytes"], int)
        self.assertGreaterEqual(receipt["released_bytes"], 0)

    def test_candidate_kept_nnz_replay_is_exact_chunked_and_deadline_bound(
        self,
    ) -> None:
        columns = (1 << 18) + 17
        data = np.ones(columns, dtype=np.float64)
        data[0] = np.nextafter(1.0e-12, 0.0)
        indices = np.arange(columns, dtype=np.int32)
        indptr = np.array([0, columns], dtype=np.int32)
        auc = sp.csr_matrix(
            (data, indices, indptr),
            shape=(1, columns),
        )
        hz = SimpleNamespace(
            Auc=auc,
            Aub=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, columns), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
        )
        chunk_sizes = []

        def bounded_abs(value):
            chunk_sizes.append(int(np.asarray(value).size))
            return np.absolute(value)

        with mock.patch.object(probe.np, "abs", side_effect=bounded_abs):
            kept = probe._exact_candidate_kept_nonzeros(
                hz,
                deadline=time.monotonic() + 5.0,
            )
        self.assertEqual(kept, columns - 1)
        self.assertGreaterEqual(len(chunk_sizes), 2)
        self.assertLessEqual(max(chunk_sizes), 1 << 18)
        with self.assertRaisesRegex(
            probe.PhaseCliqueBuildProbeError,
            "exceeded deadline",
        ):
            probe._exact_candidate_kept_nonzeros(
                hz,
                deadline=time.monotonic() - 1.0,
            )

    def _args(self, **changes):
        values = {
            "iid": 2,
            "candidate_mode": "k4",
            "wall_timeout": 60.0,
            "phase_time_limit": 20.0,
            "residual_time_limit": 4.0,
            "operator_exact_budget": 4,
            "residual_budget": 4,
            "cpu_threads": 20,
            "family": "cifar100_medium",
            "benchmark_root": Path("/tmp"),
            "parent_hard_deadline_monotonic": time.monotonic() + 60.0,
        }
        values.update(changes)
        return Namespace(**values)

    def _pcoh_inputs(self):
        return {
            "onnx": probe._RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
            "vnnlib": probe._RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
            "instances_csv": probe._RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
        }

    def _pcoh_implementation(self):
        return {
            path: hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in probe._IMPLEMENTATION_RELATIVE_PATHS
        }

    def _pcoh_k3_implementation(self):
        return {
            path: hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in probe._PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS
        }

    def _pcoh_k3_baseline_anchor(self):
        semantic_anchor = probe._pcoh_k3_focused_semantic_anchor(
            source_semantic_digest=(
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            full_batch_sha256=probe._PCOH_K3_EXPECTED_FULL_BATCH_SHA256,
            focused_encoded_row=probe._PCOH_K3_FOCUSED_ENCODED_ROW,
            focused_rival_id=probe._PCOH_K3_FOCUSED_RIVAL_ID,
            selection_digest=probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
            selection_property_digest=(
                probe._PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            ),
            selection_parent_semantic_digest=(
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            selection_operator_row_tag_digest=(
                probe._PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            ),
        )
        return probe._checksummed({
            "schema": probe._PCOH_K3_BASELINE_ANCHOR_SCHEMA,
            "status": "fixed_baseline_verified",
            "diagnostic_only": True,
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "artifact_relative_path": (
                probe._PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
            ),
            "artifact_sha256": probe._PCOH_K3_BASELINE_ARTIFACT_SHA256,
            "artifact_bytes": 4096,
            "artifact_receipt_sha256": "1" * 64,
            "baseline_summary_sha256": (
                probe._PCOH_K3_BASELINE_SUMMARY_SHA256
            ),
            "source_semantic_digest": (
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            "selection_digest": probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
            "selection_property_digest": (
                probe._PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            ),
            "selection_parent_semantic_digest": (
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            "selection_operator_row_tag_digest": (
                probe._PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            ),
            "full_batch_sha256": probe._PCOH_K3_EXPECTED_FULL_BATCH_SHA256,
            "focused_subset_digest": "d" * 64,
            "focused_semantic_anchor": semantic_anchor,
            "focused_semantic_anchor_sha256": semantic_anchor[
                "semantic_sha256"
            ],
            "focused_encoded_row": probe._PCOH_K3_FOCUSED_ENCODED_ROW,
            "focused_rival_id": probe._PCOH_K3_FOCUSED_RIVAL_ID,
            "retained_k2_stable_bit_ids": list(
                probe._PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
            ),
            "global_cube_upper_hex": probe._PCOH_K3_GLOBAL_CUBE_UPPER_HEX,
            "materialized_payload_detached_verified": True,
            "tightness_gate_detached_verified": True,
        })

    def _pcoh_k3_summary(self, *, final_hex=None, rounding_tax=None):
        from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
            _K3_STRONG_TARGET,
        )

        global_value = float.fromhex(probe._PCOH_K3_GLOBAL_CUBE_UPPER_HEX)
        final_value = (
            float(_K3_STRONG_TARGET).hex()
            if final_hex is None
            else final_hex
        )
        return {
            "summary_sha256": "2" * 64,
            "parent_semantic_digest": (
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            "stable_bit_ids": [52557, 52558, 52559],
            "global_cube_upper_hex": global_value.hex(),
            "global_cube_upper_exact": [
                probe._PCOH_K3_GLOBAL_CUBE_UPPER_EXACT.numerator,
                probe._PCOH_K3_GLOBAL_CUBE_UPPER_EXACT.denominator,
            ],
            "final_structural_upper_hex": final_value,
            "ideal_union_upper_hex": final_value,
            "rounding_tax_exact": list(
                rounding_tax if rounding_tax is not None else (0, 1)
            ),
        }

    def _pcoh_k3_detached_success(self, *, final_hex=None, rounding_tax=None):
        summary = self._pcoh_k3_summary(
            final_hex=final_hex, rounding_tax=rounding_tax
        )
        canonical = [list(item) for item in itertools.product((-1, 1), repeat=3)]
        return {
            "schema": "act.hybridz_pcoh_k3_build_only_diagnostic.v1",
            "status": "k3_build_only_materialized_validated_consumed_and_released",
            "source_semantic_digest": probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST,
            "focused_rival_id": probe._PCOH_K3_FOCUSED_RIVAL_ID,
            "retained_k2_stable_bit_ids": [52557, 52558],
            "stable_bit_ids": [52557, 52558, 52559],
            "third_stable_bit_id": 52559,
            "pair_bundle_sha256": "3" * 64,
            "active_pattern_mask": [True] * 8,
            "evaluation_schedule": canonical,
            "threshold_pattern_indices": list(range(8)),
            "source_dimensions": [100, 52657, 4, 0, 98974],
            "fresh_dimensions": [100, 52665, 4, 4, 98975],
            "fresh_semantic_digest": "4" * 64,
            "materialized_tightness_summary": summary,
            "execution_telemetry": {
                "pair_local_lp_actual_calls": 12,
                "scheduled_local_lp_actual_calls": 8,
                "local_lp_actual_calls": 20,
                "conditional_checker_actual_calls": 26,
                "local_lp_actual_call_cap": 20,
                "conditional_checker_actual_call_cap": 34,
            },
            "receipt": {
                "receipt_sha256": "5" * 64,
                "proof_authority": False,
                "verdict_authority": False,
                "full_parent_lp_called": False,
            },
        }

    def _pcoh_k3_detached_stop(self):
        detached = self._pcoh_k3_detached_success()
        detached.update({
            "schema": "act.hybridz_pcoh_k3_build_only_stop.v1",
            "status": "stopped_by_strong_target_no_partial_output",
            "fresh_issue_called": False,
            "partial_certificates_returned": False,
        })
        detached.pop("fresh_dimensions")
        detached.pop("fresh_semantic_digest")
        detached.pop("materialized_tightness_summary")
        detached["execution_telemetry"].update({
            "scheduled_local_lp_actual_calls": 1,
            "local_lp_actual_calls": 13,
            "conditional_checker_actual_calls": 3,
        })
        detached["receipt"]["source_dimensions"] = [
            100, 52657, 4, 0, 98974
        ]
        return detached

    def _pcoh_k3_detached_resource_stop(
        self, *, stage="pre_scheduled", scheduled_lp=0, accepted=0
    ):
        detached = self._pcoh_k3_detached_success()
        reason = "resource_preflight_stop_loss:toy_resource_gate"
        if stage == "pre_scheduled":
            scheduled_bundle_sha256 = None
            completed = 0
            checker_calls = 0
        else:
            self.assertEqual(stage, "pre_fresh_materialization")
            scheduled_bundle_sha256 = "a" * 64
            completed = 8
            checker_calls = 9 + accepted
        detached.update({
            "schema": "act.hybridz_pcoh_k3_build_only_resource_stop.v1",
            "status": "stopped_by_resource_gate_no_partial_output",
            "stage": stage,
            "reason": reason,
            "scheduled_bundle_sha256": scheduled_bundle_sha256,
            "completed_conditional_certificate_count": completed,
            "partial_certificates_returned": False,
            "conditional_certificate_payload_returned": False,
            "fresh_issue_called": False,
            "fresh_build_returned": False,
            "fresh_descriptor_returned": False,
            "provenance_authority": False,
            "authenticity_authority": False,
        })
        detached.pop("source_dimensions")
        detached.pop("fresh_dimensions")
        detached.pop("fresh_semantic_digest")
        detached.pop("materialized_tightness_summary")
        detached["execution_telemetry"].update({
            "scheduled_local_lp_actual_calls": scheduled_lp,
            "local_lp_actual_calls": 12 + scheduled_lp,
            "conditional_checker_actual_calls": checker_calls,
            "scheduled_patterns_completed": completed,
            "scheduled_candidate_dual_accepted": accepted,
        })
        rejection = {
            "schema": "act.hybridz_pcoh_k3_resource_gate_rejection.v1",
            "stage": stage,
            "reason": reason,
            "rejection_sha256": "7" * 64,
        }
        detached["receipt"].update({
            "stage": stage,
            "reason": reason,
            "source_dimensions": [100, 52657, 4, 0, 98974],
            "resource_gate_rejection": rejection,
        })
        return detached

    def _run_pcoh_k3_direct_mock(
        self,
        detached,
        *,
        raised: BaseException | None = None,
        focused_subset_digest: str = "d" * 64,
        residual_selector_receipt_sha256: str = "e" * 64,
    ):
        from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
            PCOHK3BuildOnlyDiagnostic,
            PCOHK3BuildOnlyResourceStopDiagnostic,
            PCOHK3BuildOnlyStopDiagnostic,
        )

        rival = SimpleNamespace(rival_id=probe._PCOH_K3_FOCUSED_RIVAL_ID)
        batch = SimpleNamespace(
            rivals=(rival,),
            batch_sha256=probe._PCOH_K3_EXPECTED_FULL_BATCH_SHA256,
            live_assert_sha256="6" * 64,
        )
        focused = SimpleNamespace(
            rivals=(rival,),
            method=probe._PCOH_K3_FOCUS_METHOD,
            focus_count=probe._PCOH_K3_FOCUS_COUNT,
            focused_subset_digest=focused_subset_digest,
            residual_selector_receipt_sha256=(
                residual_selector_receipt_sha256
            ),
        )
        selection = SimpleNamespace(
            mappings=tuple(
                SimpleNamespace(stable_bcol_id=value)
                for value in (52557, 52558, 52559, 52560)
            ),
            selection_digest=probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
            property_digest=(
                probe._PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            ),
            parent_semantic_digest=(
                probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            ),
            operator_row_tag_digest=(
                probe._PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            ),
        )
        if detached["schema"].endswith("diagnostic.v1"):
            outcome = object.__new__(PCOHK3BuildOnlyDiagnostic)
            object.__setattr__(outcome, "diagnostic_sha256", "8" * 64)
            outcome_sha = outcome.diagnostic_sha256
        elif detached["schema"].endswith("resource_stop.v1"):
            outcome = object.__new__(PCOHK3BuildOnlyResourceStopDiagnostic)
            object.__setattr__(outcome, "resource_stop_sha256", "7" * 64)
            outcome_sha = outcome.resource_stop_sha256
        else:
            outcome = object.__new__(PCOHK3BuildOnlyStopDiagnostic)
            object.__setattr__(outcome, "stop_sha256", "9" * 64)
            outcome_sha = outcome.stop_sha256
        run_k3 = mock.Mock(
            side_effect=raised if raised is not None else None,
            return_value=outcome,
        )
        raw_module = "act.back_end.hybridz_tf.raw_vnnlib_rival_adapter"
        focus_module = (
            "act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge"
        )
        literal_module = (
            "act.back_end.hybridz_tf.operator_exact_relu_phase_literals"
        )
        clique_module = (
            "act.back_end.hybridz_tf.operator_phase_clique_pipeline"
        )
        k3_module = (
            "act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only"
        )
        with (
            mock.patch(
                raw_module + ".issue_raw_vnnlib_top1_candidate",
                return_value=object(),
            ),
            mock.patch(
                raw_module + ".consume_raw_vnnlib_top1_candidate",
                return_value=batch,
            ),
            mock.patch(
                raw_module + ".validate_consumed_raw_vnnlib_rival_batch",
                return_value=True,
            ),
            mock.patch(
                focus_module + ".issue_raw_rival_exact_hardness_receipt",
                return_value=object(),
            ),
            mock.patch(
                focus_module + ".select_raw_focused_rivals",
                return_value=focused,
            ),
            mock.patch(
                focus_module + ".verify_raw_rival_exact_hardness_receipt",
                return_value=True,
            ),
            mock.patch(
                focus_module + ".verify_raw_focused_rival_selection",
                return_value=True,
            ),
            mock.patch(
                literal_module
                + ".derive_operator_exact_relu_property_phase_literals",
                return_value=selection,
            ),
            mock.patch(
                literal_module
                + ".verify_operator_exact_relu_property_phase_selection",
                return_value=True,
            ),
            mock.patch(
                clique_module + "._snapshot_b1_bounds",
                return_value=(np.zeros((1, 100)), np.ones((1, 100))),
            ),
            mock.patch(
                clique_module + "._exact_interval_upper_violations",
                return_value=(1.0,),
            ),
            mock.patch(
                clique_module + "._interval_frame_digest",
                return_value="a" * 64,
            ),
            mock.patch(
                "act.back_end.hybridz_tf.adaptive_phase_forest."
                "sparse_hz_semantic_digest",
                return_value=probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST,
            ),
            mock.patch(
                k3_module
                + ".run_phase_conditioned_objective_hull_k3_build_only",
                run_k3,
            ),
            mock.patch(
                k3_module
                + ".verify_phase_conditioned_objective_hull_k3_build_only_outcome",
                return_value=True,
            ),
            mock.patch(
                k3_module
                + ".export_phase_conditioned_objective_hull_k3_build_only_detached",
                return_value=detached,
            ),
            mock.patch(
                k3_module
                + ".verify_detached_phase_conditioned_objective_hull_k3_build_only",
                return_value=True,
            ),
            mock.patch.object(
                probe, "_capture_resource_peaks",
                return_value=self._pcoh_resources(),
            ),
        ):
            result = probe._run_pcoh_k3_build_only_pipeline(
                SimpleNamespace(hz=SimpleNamespace(n_out=100)),
                input_sha256=self._pcoh_inputs(),
                implementation_sha256=self._pcoh_k3_implementation(),
                vnnlib_path="/toy/property.vnnlib",
                expected_vnnlib_sha256=self._pcoh_inputs()["vnnlib"],
                live_assert_params={},
                output_lower=np.zeros((1, 100), dtype=np.float64),
                output_upper=np.ones((1, 100), dtype=np.float64),
                residual_selector_receipt={
                    "joint_focus_rival_id": probe._PCOH_K3_FOCUSED_ENCODED_ROW
                },
                residual_selector_property_sha256="b" * 64,
                deadline=time.monotonic() + 30.0,
                phase_time_limit=probe._PCOH_K3_INTERNAL_PHASE_SECONDS,
                torch_module=SimpleNamespace(),
                baseline_anchor_receipt=self._pcoh_k3_baseline_anchor(),
            )
        return result, run_k3, outcome_sha

    def test_pcoh_k3_focused_semantic_anchor_is_stable_and_strict(self):
        expected_fields = {
            "schema", "candidate_only", "proof_authority", "focus_method",
            "focus_count", "source_semantic_digest", "full_batch_sha256",
            "focused_encoded_row", "focused_rival_id", "selection_digest",
            "selection_property_digest", "selection_parent_semantic_digest",
            "selection_operator_row_tag_digest", "semantic_sha256",
        }
        baseline = self._pcoh_k3_baseline_anchor()
        semantic = baseline["focused_semantic_anchor"]
        self.assertEqual(set(semantic), expected_fields)
        self.assertTrue(
            probe._pcoh_k3_focused_semantic_anchor_valid(semantic)
        )
        self.assertTrue(
            probe._pcoh_k3_fixed_focused_semantic_anchor_valid(semantic)
        )
        self.assertEqual(
            semantic["semantic_sha256"],
            "bb938b9f23f4e0909a77f8f547d30402121ca3592267e817e3ff6083fb62c862",
        )
        self.assertNotIn("focused_subset_digest", semantic)
        self.assertNotIn("residual_selector_receipt_sha256", semantic)
        class EqualString(str):
            pass

        class DictSubclass(dict):
            pass

        self.assertFalse(
            probe._pcoh_k3_focused_semantic_anchor_valid(
                DictSubclass(semantic)
            )
        )
        for name in ("schema", "focus_method"):
            with self.subTest(non_builtin_string=name):
                forged_type = copy.deepcopy(semantic)
                forged_type[name] = EqualString(forged_type[name])
                self.assertFalse(
                    probe._pcoh_k3_focused_semantic_anchor_valid(
                        forged_type
                    )
                )
        extra_field = copy.deepcopy(semantic)
        extra_field["focused_subset_digest"] = "0" * 64
        self.assertFalse(
            probe._pcoh_k3_focused_semantic_anchor_valid(extra_field)
        )
        float_row_payload = copy.deepcopy(semantic)
        float_row_payload.pop("semantic_sha256")
        float_row_payload["focused_encoded_row"] = 50.0
        float_row_semantic = {
            **float_row_payload,
            "semantic_sha256": hashlib.sha256(
                probe._canonical_json(float_row_payload)
            ).hexdigest(),
        }
        self.assertFalse(
            probe._pcoh_k3_focused_semantic_anchor_valid(
                float_row_semantic
            )
        )
        float_row_baseline = copy.deepcopy(baseline)
        float_row_baseline["focused_encoded_row"] = 50.0
        float_row_baseline["focused_semantic_anchor"] = (
            float_row_semantic
        )
        float_row_baseline["focused_semantic_anchor_sha256"] = (
            float_row_semantic["semantic_sha256"]
        )
        float_row_baseline.pop("receipt_sha256")
        float_row_baseline["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(float_row_baseline)
        ).hexdigest()
        self.assertFalse(
            probe._pcoh_k3_baseline_anchor_receipt_valid(
                float_row_baseline
            )
        )

        elapsed_receipts = (
            {"schema": "toy.residual.v1", "elapsed_seconds": 0.125},
            {"schema": "toy.residual.v1", "elapsed_seconds": 0.875},
        )
        provenance_sha256 = tuple(
            hashlib.sha256(probe._canonical_json(receipt)).hexdigest()
            for receipt in elapsed_receipts
        )
        self.assertNotEqual(*provenance_sha256)
        projected_after_elapsed_drift = probe._pcoh_k3_focused_semantic_anchor(
            source_semantic_digest=semantic["source_semantic_digest"],
            full_batch_sha256=semantic["full_batch_sha256"],
            focused_encoded_row=semantic["focused_encoded_row"],
            focused_rival_id=semantic["focused_rival_id"],
            selection_digest=semantic["selection_digest"],
            selection_property_digest=semantic[
                "selection_property_digest"
            ],
            selection_parent_semantic_digest=semantic[
                "selection_parent_semantic_digest"
            ],
            selection_operator_row_tag_digest=semantic[
                "selection_operator_row_tag_digest"
            ],
        )
        self.assertEqual(projected_after_elapsed_drift, semantic)

        artifact_names = (
            "pcoh_k2_build_only_cifar100_medium_iid2_first.json",
            "pcoh_k2_materialized_tightness_cifar100_medium_iid2_first.json",
        )
        artifact_root = (
            Path(probe.__file__).resolve().parents[3]
            / "artifacts/hybridz_largecls_gates"
        )
        artifact_semantic_sha256 = []
        artifact_focused_subset_digest = []
        for name in artifact_names:
            artifact = json.loads(
                (artifact_root / name).read_text(encoding="utf-8")
            )
            transaction = artifact["pcoh_k2_build_only"]
            self.assertTrue(
                set(transaction).issubset(
                    probe._PCOH_K2_TRANSACTION_FIELDS
                )
            )
            self.assertTrue(
                probe._local_receipt_checksum_valid(
                    transaction, schema=transaction["schema"]
                )
            )
            projected = probe._pcoh_k3_focused_semantic_anchor(
                source_semantic_digest=transaction[
                    "source_semantic_digest"
                ],
                full_batch_sha256=transaction["full_batch_sha256"],
                focused_encoded_row=transaction["focused_encoded_row"],
                focused_rival_id=transaction["focused_rival_id"],
                selection_digest=transaction["selection_digest"],
                selection_property_digest=transaction[
                    "selection_property_digest"
                ],
                selection_parent_semantic_digest=transaction[
                    "selection_parent_semantic_digest"
                ],
                selection_operator_row_tag_digest=transaction[
                    "selection_operator_row_tag_digest"
                ],
            )
            artifact_semantic_sha256.append(projected["semantic_sha256"])
            artifact_focused_subset_digest.append(
                transaction["focused_subset_digest"]
            )
        self.assertEqual(
            artifact_semantic_sha256,
            [probe._PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256] * 2,
        )
        self.assertNotEqual(*artifact_focused_subset_digest)

        coherently_rehashed = copy.deepcopy(baseline)
        forged_semantic = probe._pcoh_k3_focused_semantic_anchor(
            source_semantic_digest=semantic["source_semantic_digest"],
            full_batch_sha256=semantic["full_batch_sha256"],
            focused_encoded_row=semantic["focused_encoded_row"],
            focused_rival_id=semantic["focused_rival_id"],
            selection_digest="f" * 64,
            selection_property_digest=semantic[
                "selection_property_digest"
            ],
            selection_parent_semantic_digest=semantic[
                "selection_parent_semantic_digest"
            ],
            selection_operator_row_tag_digest=semantic[
                "selection_operator_row_tag_digest"
            ],
        )
        self.assertTrue(
            probe._pcoh_k3_focused_semantic_anchor_valid(forged_semantic)
        )
        self.assertFalse(
            probe._pcoh_k3_fixed_focused_semantic_anchor_valid(
                forged_semantic
            )
        )
        coherently_rehashed["selection_digest"] = "f" * 64
        coherently_rehashed["focused_semantic_anchor"] = forged_semantic
        coherently_rehashed["focused_semantic_anchor_sha256"] = (
            forged_semantic["semantic_sha256"]
        )
        coherently_rehashed.pop("receipt_sha256")
        coherently_rehashed["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(coherently_rehashed)
        ).hexdigest()
        self.assertFalse(
            probe._pcoh_k3_baseline_anchor_receipt_valid(
                coherently_rehashed
            )
        )

    def test_pcoh_k3_live_provenance_varies_but_semantic_anchor_enters(self):
        first, _, _ = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success(),
            focused_subset_digest="a" * 64,
            residual_selector_receipt_sha256="b" * 64,
        )
        second, _, _ = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success(),
            focused_subset_digest="c" * 64,
            residual_selector_receipt_sha256="d" * 64,
        )
        try:
            self.assertEqual(first["status"], "strong_promotion")
            self.assertEqual(second["status"], "strong_promotion")
            verifier = (
                "act.back_end.hybridz_tf."
                "operator_phase_conditioned_k3_build_only."
                "verify_detached_phase_conditioned_objective_hull_k3_build_only"
            )
            with mock.patch(verifier, return_value=True):
                self.assertTrue(
                    probe._pcoh_k3_transaction_receipt_valid(first)
                )
                self.assertTrue(
                    probe._pcoh_k3_transaction_receipt_valid(second)
                )
            self.assertEqual(first["focused_subset_digest"], "a" * 64)
            self.assertEqual(second["focused_subset_digest"], "c" * 64)
            self.assertEqual(
                first["residual_selector_receipt_sha256"], "b" * 64
            )
            self.assertEqual(
                second["residual_selector_receipt_sha256"], "d" * 64
            )
            self.assertEqual(
                first["focused_semantic_anchor"],
                second["focused_semantic_anchor"],
            )
            self.assertEqual(
                first["focused_semantic_anchor_sha256"],
                probe._PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256,
            )
        finally:
            probe._release_pcoh_k3_trusted_transaction(first)
            probe._release_pcoh_k3_trusted_transaction(second)

    def _pcoh_k3_terminal_body(self, detached):
        transaction, _, _ = self._run_pcoh_k3_direct_mock(detached)
        implementation = self._pcoh_k3_implementation()
        anchor = self._pcoh_k3_baseline_anchor()
        shape = {
            "output_dimension": 100,
            "continuous_columns": 52_657,
            "binary_columns": 4,
            "upper_rows": 98_974,
            "equality_rows": 0,
            "constraint_nonzeros": 9_267_556,
            "generator_nonzeros": 10_100,
        }
        source_preflight = probe._pcoh_k3_source_build_preflight(
            shape,
            build_seconds=1.0,
            input_sha256=self._pcoh_inputs(),
            implementation_sha256=implementation,
            baseline_anchor_receipt=anchor,
        )
        now = time.monotonic()
        return {
            "candidate_mode": probe._PCOH_K3_BUILD_ONLY_MODE,
            "diagnostic_only": True,
            "candidate_only": True,
            "build_only": True,
            "instance_count": 1,
            "proof_authority": False,
            "verdict_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "solver_handoff_called": False,
            "hz_base_feasibility_called": False,
            "hz_objbound_decide_called": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "k3_transaction_called": transaction["k3_transaction_called"],
            "k2_build_only_called": False,
            "phase_transaction_called": False,
            "family": probe._PCOH_K3_FAMILY,
            "iid": 2,
            "wall_timeout_seconds": 60.0,
            "phase_time_limit_seconds": 25.0,
            "operator_exact_budget": 4,
            "residual_budget": 4,
            "residual_time_limit_seconds": 4.0,
            "cpu_threads": 20,
            "parent_hard_deadline_monotonic": now + 10.0,
            "shared_worker_deadline_monotonic": now + 6.0,
            "shared_worker_deadline_met": True,
            "input_sha256": self._pcoh_inputs(),
            "input_sha256_after": self._pcoh_inputs(),
            "inputs_unchanged": True,
            "implementation_sha256": implementation,
            "implementation_sha256_after": implementation,
            "implementation_unchanged": True,
            "resource_usage": self._pcoh_resources(),
            "timings": {"total_seconds": 1.0},
            "pcoh_k3_source_build_preflight": source_preflight,
            "pcoh_k3_build_only": transaction,
            "pair_local_lp_actual_calls": transaction[
                "pair_local_lp_actual_calls"
            ],
            "conditional_local_lp_actual_calls": transaction[
                "conditional_local_lp_actual_calls"
            ],
            "total_local_lp_actual_calls": transaction[
                "total_local_lp_actual_calls"
            ],
            "conditional_checker_actual_calls": transaction[
                "conditional_checker_actual_calls"
            ],
            "local_lp_actual_call_cap": 20,
            "conditional_checker_actual_call_cap": 34,
            "certified_edge_count": 0,
            "phase_status": transaction["status"],
            "failed_stage": None,
            "fallback_reason": None,
            "error_type": None,
            "error": None,
        }, anchor

    def _pcoh_k3_stop_terminal_body(
        self,
        *,
        stage="verified_literal_selection_max4",
        reason="KeyboardInterrupt:toy helper stop",
        k3_transaction_called=False,
    ):
        body, anchor = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        probe._release_pcoh_k3_trusted_transaction(
            body["pcoh_k3_build_only"]
        )
        stop = probe._pcoh_k3_stop_loss_receipt(
            stage=stage,
            reason=reason,
            started=time.monotonic(),
            input_sha256=self._pcoh_inputs(),
            implementation_sha256=self._pcoh_k3_implementation(),
            stage_resources={
                "entry": self._pcoh_resources(),
                "unique_helper_stop": self._pcoh_resources(),
            },
            timings={
                "raw_batch_seconds": 0.125,
                "focused_rival_seconds": 0.25,
            },
            k3_transaction_called=k3_transaction_called,
            baseline_anchor_receipt=anchor,
        )
        body.update({
            "pcoh_k3_build_only": stop,
            "k3_transaction_called": stop["k3_transaction_called"],
            "pair_local_lp_actual_calls": 0,
            "conditional_local_lp_actual_calls": 0,
            "total_local_lp_actual_calls": 0,
            "conditional_checker_actual_calls": 0,
            "phase_status": "stop_loss",
            "failed_stage": stop["failed_stage"],
            "fallback_reason": stop["reason"],
        })
        return body, anchor, stop

    def _pcoh_resources(self):
        return {
            "peak_rss_bytes": 4096,
            "current_rss_bytes": 2048,
            "cuda_initialized": True,
            "cuda_peak_allocated_bytes": 1024,
            "cuda_peak_reserved_bytes": 2048,
        }

    def _pcoh_tightness_summary(
        self,
        *,
        stable_ids=(3, 7),
        certificate_sha256=None,
    ):
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_build_only as build_only,
        )

        certificates = list(
            certificate_sha256
            if certificate_sha256 is not None
            else ("3" * 64, "4" * 64, "5" * 64, "6" * 64)
        )
        summary = {
            "schema": "act.hybridz_pc_materialized_tightness_summary.toy.v1",
            "status": "sound_materialized_structural_upper",
            "parent_semantic_digest": "a" * 64,
            "adapter_candidate_sha256": "b" * 64,
            "descriptor_representation_sha256": "c" * 64,
            "row_frame_sha256": "d" * 64,
            "stable_bit_ids": list(stable_ids),
            "canonical_patterns": [
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1],
            ],
            "active_pattern_mask": [True, True, True, True],
            "empty_evidence_descriptor_sha256": [],
            "conditional_certificate_schema": (
                "act.operator_phase_conditioned_objective_bound.v2"
            ),
            "conditional_certificate_sha256": certificates,
            "conditional_pattern_sha256": [
                "7" * 64,
                "8" * 64,
                "9" * 64,
                "a" * 64,
            ],
            "conditional_selected_source": [
                "global_cube_baseline",
                "global_cube_baseline",
                "global_cube_baseline",
                "global_cube_baseline",
            ],
            "conditional_checker_route": (
                "native_hz_preformed_objective_split_csr_no_generator_read_v1"
            ),
            "objective_binding_sha256": "e" * 64,
            "objective_envelope_sha256": "f" * 64,
            "global_checker_sha256": "0" * 64,
            "global_cube_upper_exact": [4, 1],
            "global_cube_upper_hex": (4.0).hex(),
            "pattern_upper_hex": [(1.0).hex()] * 4,
            "objective_center_exact": [0, 1],
            "row_raw_rhs_exact": [1, 1],
            "row_stored_rhs_hex": (1.0).hex(),
            "row_total_coefficient_guard_exact": [0, 1],
            "free_parent_mismatch_exact": [0, 1],
            "all_parent_mismatch_exact": [0, 1],
            "linked_support_exact": [[0, 1]] * 4,
            "direct_eta_support_exact": [[0, 1]] * 4,
            "ideal_union_upper_hex": (1.0).hex(),
            "materialized_linked_upper_exact": [1, 1],
            "materialized_linked_upper_hex": (1.0).hex(),
            "materialized_direct_upper_exact": [1, 1],
            "materialized_direct_upper_hex": (1.0).hex(),
            "materialized_guard_upper_exact": [1, 1],
            "materialized_guard_upper_hex": (1.0).hex(),
            "rounding_tax_exact": [0, 1],
            "final_structural_upper_hex": (1.0).hex(),
            "diagnostic_only": True,
            "full_parent_lp_called": False,
            "proof_authority": False,
            "verdict_authority": False,
        }
        summary["summary_sha256"] = build_only._canonical_sha256(summary)
        return summary

    def _pcoh_success_transaction(self):
        resources = self._pcoh_resources()
        source_digest = "a" * 64
        summary = self._pcoh_tightness_summary()
        summary_sha256 = summary["summary_sha256"]
        tightness_gate = probe._pcoh_k2_tightness_gate(
            summary,
            expected_summary_sha256=summary_sha256,
        )
        result = probe._checksummed({
            "schema": probe._PCOH_K2_TRANSACTION_SCHEMA,
            "status": "built_and_released",
            "reason": None,
            "failed_stage": None,
            "diagnostic_only": True,
            "candidate_only": True,
            "build_only": True,
            "instance_count": 1,
            "proof_authority": False,
            "verdict_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "build_only_transaction_called": True,
            "transaction_verified_before_serialization": True,
            "solver_handoff_called": False,
            "diagnostic_lp_called": False,
            "hz_base_feasibility_called": False,
            "hz_objbound_decide_called": False,
            "strict_replay_called": False,
            "fresh_build_returned": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "input_sha256": self._pcoh_inputs(),
            "implementation_sha256": self._pcoh_implementation(),
            "full_batch_sha256": "b" * 64,
            "focused_subset_digest": "c" * 64,
            "focused_encoded_row": 9,
            "focused_rival_id": 7,
            "successful_selection_binding_retained": True,
            "selection_digest": "d" * 64,
            "selection_property_digest": "e" * 64,
            "selection_parent_semantic_digest": source_digest,
            "selection_operator_row_tag_digest": "f" * 64,
            "stable_bit_selection_method": (
                "lowest_two_canonical_ids_from_verified_selection"
            ),
            "stable_bit_ids": [3, 7],
            "diagnostic_schema": (
                "act.hybridz_pcoh_build_only_diagnostic.v2"
            ),
            "diagnostic_sha256": "0" * 64,
            "transaction_receipt_sha256": "1" * 64,
            "source_semantic_digest": source_digest,
            "fresh_semantic_digest": "2" * 64,
            "source_dimensions": [2, 10, 4, 0, 20],
            "fresh_dimensions": [2, 14, 4, 3, 21],
            "conditional_certificate_sha256": [
                "3" * 64,
                "4" * 64,
                "5" * 64,
                "6" * 64,
            ],
            "pair_bundle_sha256": "7" * 64,
            "fresh_issuance_sha256": "8" * 64,
            "materialized_tightness_summary_sha256": summary_sha256,
            "materialized_tightness_summary": summary,
            "tightness_gate": tightness_gate,
            "resource_preflight": {
                "passed": True,
                "caller_supplied": False,
            },
            "resource_postflight": {
                "passed": True,
                "caller_supplied": False,
            },
            "stage_resources": {
                name: dict(resources)
                for name in (
                    "entry",
                    "raw_batch",
                    "focused_rival",
                    "literal_selection",
                    "build_only_transaction",
                )
            },
            "timings": {"total_seconds": 1.0},
        })
        probe._register_pcoh_k2_trusted_transaction(
            result, trusted_summary_sha256=summary_sha256
        )
        return result

    def _pcoh_terminal_body(self):
        inputs = self._pcoh_inputs()
        implementation = self._pcoh_implementation()
        source_shape = {
            "output_dimension": 100,
            "continuous_columns": 60_000,
            "binary_columns": 4,
            "upper_rows": 104_900,
            "equality_rows": 0,
            "constraint_nonzeros": 11_000_000,
            "generator_nonzeros": 20_000,
        }
        body = {
            "candidate_mode": probe._PCOH_K2_BUILD_ONLY_MODE,
            "family": probe._PCOH_K2_FAMILY,
            "iid": 2,
            "wall_timeout_seconds": 60.0,
            "phase_time_limit_seconds": 25.0,
            "operator_exact_budget": 4,
            "residual_budget": 4,
            "residual_time_limit_seconds": 4.0,
            "cpu_threads": 20,
            "diagnostic_only": True,
            "candidate_only": True,
            "build_only": True,
            "instance_count": 1,
            "proof_authority": False,
            "verdict_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "solver_handoff_called": False,
            "diagnostic_lp_called": False,
            "hz_base_feasibility_called": False,
            "hz_objbound_decide_called": False,
            "strict_replay_called": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "certified_edge_count": 0,
            "input_sha256": inputs,
            "input_sha256_after": dict(inputs),
            "inputs_unchanged": True,
            "implementation_sha256": implementation,
            "implementation_sha256_after": dict(implementation),
            "implementation_unchanged": True,
            "resource_usage": self._pcoh_resources(),
            "timings": {"total_seconds": 1.0},
            "shared_worker_deadline_met": True,
            "completed_before_deadline": True,
            "phase_status": "built_and_released",
            "failed_stage": None,
            "fallback_reason": None,
            "error_type": None,
            "error": None,
            "pcoh_k2_build_only": self._pcoh_success_transaction(),
        }
        body["pcoh_source_build_preflight"] = (
            probe._pcoh_k2_source_build_preflight(
                source_shape,
                build_seconds=27.0,
                input_sha256=inputs,
                implementation_sha256=implementation,
            )
        )
        return body

    def _adaptive_selector_plan(
        self,
        coordinates,
        *,
        property_sha256: str = "1" * 64,
    ):
        schedule = [
            {
                "layer_id": int(layer_id),
                "row": int(row),
                "guard": "both",
                "score": float(index + 1),
                "facility_gain": float(index + 1) / 10.0,
                "dominant_rival": 0,
            }
            for index, (layer_id, row) in enumerate(coordinates)
        ]
        digest_payload = {
            "property_sha256": property_sha256,
            "targets": [
                {
                    "layer_id": item["layer_id"],
                    "row": item["row"],
                    "guard": item["guard"],
                }
                for item in schedule
            ],
        }
        targets_sha256 = hashlib.sha256(
            probe._canonical_json(digest_payload)
        ).hexdigest()
        targets = tuple(SimpleNamespace(**item) for item in schedule)
        receipt = {
            "schema": "property_residual_selector_v1",
            "status": "selected",
            "candidate_only": True,
            "proof_authority": False,
            "selection_policy": "facility_first_then_same_rival_joint",
            "property_sha256": property_sha256,
            "rival_ids": [0],
            "joint_focus_rival_id": 0,
            "rivals_processed": 1,
            "targets_selected": len(targets),
            "all_interval_survivors_processed": True,
            "schedule": copy.deepcopy(schedule),
        }
        return SimpleNamespace(
            targets=targets,
            builder_targets=tuple(
                (item["layer_id"], item["row"], item["guard"])
                for item in schedule
            ),
            property_sha256=property_sha256,
            targets_sha256=targets_sha256,
            receipt=receipt,
        )

    def _adaptive_property_cube_receipt(
        self,
        *,
        rows: int = 99,
        outputs: int = 100,
        maximum: float = 75.0,
    ):
        hz = SimpleNamespace(
            c=np.zeros(outputs, dtype=np.float64),
            Gc=sp.csr_matrix((outputs, 0), dtype=np.float64),
            Gb=sp.csr_matrix((outputs, 4), dtype=np.float64),
        )
        C = np.zeros((rows, outputs), dtype=np.float64)
        thresholds = np.zeros(rows, dtype=np.float64)
        return probe._rbs_adaptive_property_cube_receipt(
            hz,
            C,
            thresholds,
            cube_upper=lambda *_args: (
                np.full(rows, maximum, dtype=np.float64),
                np.zeros(rows, dtype=np.float64),
            ),
        )

    def _adaptive_build_and_schedule(self):
        selector = self._adaptive_selector_plan(
            [(7, row) for row in range(16)]
        )
        primary, reservoir, schedule = probe._split_rbs_adaptive_schedule(
            selector,
            primary_budget=4,
            expected_selector_budget=16,
            expected_property_sha256=selector.property_sha256,
            require_all_interval_survivors_processed=True,
        )
        self.assertEqual(primary, tuple((7, row, "both") for row in range(4)))
        self.assertEqual(reservoir, tuple((7, row) for row in range(4, 16)))
        hz = SimpleNamespace(
            c=np.zeros(100, dtype=np.float64),
            Gc=sp.csr_matrix((100, 0), dtype=np.float64),
            Gb=sp.csr_matrix((100, 4), dtype=np.float64),
            Auc=sp.csr_matrix((10, 0), dtype=np.float64),
            Aub=sp.csr_matrix((10, 4), dtype=np.float64),
            Ac=sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=sp.csr_matrix((0, 4), dtype=np.float64),
            b=np.empty(0, dtype=np.float64),
            ub=np.ones(10, dtype=np.float64),
            col_ids=np.empty(0, dtype=np.int64),
            bcol_ids=np.arange(4, dtype=np.int64),
            n_cont=0,
            n_bin=4,
            n_ub=10,
            n_eq=0,
        )
        prepared = [
            {
                "schema": "operator_hz_residual_phase_screen_v1",
                "status": "prepared",
                "mode": "strict_bound_improvement",
                "proof_authority": True,
                "retained_count": 308,
            }
            for _ in range(4)
        ]
        applied_layers = [
            {
                "kind": "RELU",
                "residual_phase_screen": {
                    "status": "applied",
                    "rows_applied": 308,
                },
            }
            for _ in range(4)
        ]
        conv_layers = [
            {
                "layer_id": index,
                "kind": "CONV2D",
                "operator_nnz": 1,
                "operator_csr_builder": "vectorized_exact_csr_v1",
            }
            for index in range(
                probe._RBS_ADAPTIVE_K4_EXPECTED_CONV_LAYERS
            )
        ]
        layers = conv_layers + applied_layers
        reservoir_receipt = {
            "schema": "operator_hz_exact_target_reservoir_v1",
            "relu_layer_id": 7,
            "enabled": True,
            "candidate_only": True,
            "proof_authority": False,
            "same_layer_only": True,
            "status": "filled",
            "shortfall": 0,
            "all_primary_rows_rbs_tightened": True,
            "all_selected_rows_rbs_tightened": True,
            "unselected_reserves_use_ordinary_triangle": True,
            "selected_rows_use_existing_exact_big_m": True,
            "primary_rows": list(range(4)),
            "reserve_rows": list(range(4, 16)),
            "selected_rows": list(range(4)),
            "pre_screen_cube_unstable_primary": list(range(4)),
            "primary_rows_rbs_tightened": list(range(4)),
            "rbs_newly_stabilized_primary": [],
            "post_screen_stabilized_active_primary": [],
            "post_screen_stabilized_inactive_primary": [],
            "selected_primary_rows": list(range(4)),
            "selected_reserve_rows": [],
            "selected_rows_rbs_tightened": list(range(4)),
            "non_rbs_stable_primary_not_replaced": [],
            "replacement_slots": [],
            "replacement_count": 0,
        }
        metadata = {
            "exact_budget_requested": 4,
            "exact_budget_used": 4,
            "materialize_add": True,
            "sparse_hz_core_assembly": (
                "owned_canonical_no_recopy_v1"
            ),
            "residual_bound_screen_requested": True,
            "residual_phase_screen_requested": False,
            "residual_phase_screen_rows_scanned": 1_232,
            "residual_bound_screen_rows_tightened": 1_232,
            "residual_phase_screen_receipts": prepared,
            "residual_phase_screen_layers_prepared": 4,
            "residual_phase_screen_elapsed_seconds": 1.0,
            "residual_phase_screen_stabilized_active": 26,
            "residual_phase_screen_stabilized_inactive": 296,
            "n_layers": len(layers),
            "layers": layers,
            "traversal_cache_release": {
                "schema": "operator_hz_traversal_cache_release_v1",
                "status": "released_before_final_sparse_assembly",
                "candidate_only": True,
                "proof_authority": False,
                "numeric_semantics_changed": False,
                "expr_count": 1,
                "constraint_blocks_released_before_constructor": True,
            },
            "exact_target_reservoir_requested": True,
            "exact_target_reservoir_primary_count": 4,
            "exact_target_reservoir_backup_count": 12,
            "exact_target_reservoir_shortfall": 0,
            "exact_target_reservoir_receipts": [reservoir_receipt],
            "verified_preactivation_frame_export_requested": False,
            "verified_preactivation_frame_exported": False,
        }
        build = SimpleNamespace(
            hz=hz,
            metadata=metadata,
            performance_diagnostic={
                "schema": "operator_hz_build_performance_diagnostic_v1",
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "layers": [
                    {"layer_id": index, "wall_seconds": 0.0}
                    for index in range(len(layers))
                ],
                "stages": {},
                "total_wall_seconds": 0.0,
            },
            verified_preactivation_frame=None,
            input_col_ids=np.empty(0, dtype=np.int64),
        )
        return build, selector, schedule

    def _adaptive_resources(self):
        return {
            "peak_rss_bytes": probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES,
            "current_rss_bytes": 1024,
            "cuda_initialized": True,
            "cuda_peak_allocated_bytes": (
                probe._RBS_ADAPTIVE_K4_MAX_CUDA_ALLOCATED_BYTES
            ),
            "cuda_peak_reserved_bytes": (
                probe._RBS_ADAPTIVE_K4_MAX_CUDA_ALLOCATED_BYTES
            ),
        }

    def _adaptive_pre_gate(self, build, schedule, *, resources=None):
        return probe._rbs_adaptive_k4_pre_gate(
            build,
            schedule_receipt=schedule,
            property_cube_receipt=self._adaptive_property_cube_receipt(),
            build_seconds=20.0,
            input_sha256={
                "onnx": probe._RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": probe._RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": probe._RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            },
            resources=(
                self._adaptive_resources() if resources is None else resources
            ),
            remaining_seconds=30.0,
        )

    def _execute_adaptive_pre_gate_rejection(self, failure: str):
        C = np.zeros((99, 100), dtype=np.float64)
        thresholds = np.zeros(99, dtype=np.float64)
        kind = "or"
        property_sha256 = probe._binary_property_sha256(
            C, thresholds, kind=kind
        )
        selector = self._adaptive_selector_plan(
            [(7, row) for row in range(16)],
            property_sha256=property_sha256,
        )
        build, _unused_selector, _unused_schedule = (
            self._adaptive_build_and_schedule()
        )
        resources = self._adaptive_resources()
        if failure == "underfill":
            build.metadata["exact_budget_used"] = 3
        elif failure == "rbs_receipt":
            build.metadata["exact_target_reservoir_receipts"][0][
                "selected_rows_rbs_tightened"
            ] = [0, 1, 2]
        elif failure == "resource":
            resources["peak_rss_bytes"] += 1
        elif failure == "elapsed":
            build.metadata["residual_phase_screen_elapsed_seconds"] = (
                np.nextafter(1.0, np.inf)
            )
        elif failure == "conv_builder":
            build.metadata["layers"][0]["operator_csr_builder"] = (
                "legacy_exact_csr_v1"
            )
        elif failure == "traversal_release":
            build.metadata["traversal_cache_release"]["status"] = (
                "not_released"
            )
        elif failure == "telemetry":
            build.performance_diagnostic.pop("total_wall_seconds")
        else:
            raise AssertionError(f"unsupported toy failure: {failure}")

        class ArrayTensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.device = "cpu"
                self.dtype = np.float64

            @property
            def shape(self):
                return self.value.shape

            def detach(self):
                return self

            def cpu(self):
                return self

            def double(self):
                return self

            def numpy(self):
                return self.value

        class Model:
            def to(self, **_kwargs):
                return self

        class TorchToACT:
            def __init__(self, _model):
                pass

            def run(self):
                return SimpleNamespace()

        def module(name, **attributes):
            value = ModuleType(name)
            for attribute, item in attributes.items():
                setattr(value, attribute, item)
            return value

        torch_module = module(
            "torch",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                is_initialized=lambda: True,
                max_memory_allocated=lambda: resources[
                    "cuda_peak_allocated_bytes"
                ],
                max_memory_reserved=lambda: resources[
                    "cuda_peak_reserved_bytes"
                ],
            ),
            float64=object(),
            set_num_threads=lambda _value: None,
            set_num_interop_threads=lambda _value: None,
        )
        seed = SimpleNamespace(
            lb=ArrayTensor(np.zeros((1, 1), dtype=np.float64)),
            ub=ArrayTensor(np.ones((1, 1), dtype=np.float64)),
        )
        output_bounds = SimpleNamespace(
            lb=ArrayTensor(np.zeros((1, 100), dtype=np.float64)),
            ub=ArrayTensor(np.ones((1, 100), dtype=np.float64)),
        )
        assert_layer = SimpleNamespace(params={
            "M": 99,
            "C": ArrayTensor(C),
            "thresholds": ArrayTensor(thresholds),
            "kind": kind,
            "y_true": 0,
        })
        pipeline = mock.Mock(name="forbidden_phase_clique_pipeline")
        consume = mock.Mock(name="forbidden_private_handoff")
        validate = mock.Mock(name="forbidden_handoff_validation")
        builder = mock.Mock(return_value=build)
        modules = {
            "torch": torch_module,
            "act.back_end.analyze": module(
                "act.back_end.analyze",
                analyze=lambda *_args: (object(), object(), object()),
            ),
            "act.back_end.core": module(
                "act.back_end.core",
                ConSet=lambda: SimpleNamespace(),
                Fact=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            "act.back_end.hybridz_tf.operator_hz": module(
                "act.back_end.hybridz_tf.operator_hz",
                build_operator_hz=builder,
            ),
            "act.back_end.hybridz_tf.operator_phase_clique_pipeline": module(
                "act.back_end.hybridz_tf.operator_phase_clique_pipeline",
                maybe_run_operator_phase_clique_pipeline=pipeline,
                consume_operator_phase_clique_pipeline_solver_handoff=consume,
                validate_consumed_operator_phase_clique_solver_build=validate,
            ),
            "act.back_end.transfer_functions": module(
                "act.back_end.transfer_functions",
                set_solver_mode=lambda _value: None,
                set_transfer_function_mode=lambda _value: None,
            ),
            "act.back_end.verifier": module(
                "act.back_end.verifier",
                _ensure_assert_linear_encoding=lambda *_args, **_kwargs: None,
                _get_output_layer_bounds=lambda *_args: output_bounds,
                _get_output_layer_id=lambda _net: 12,
                add_all_input_specs=lambda *_args: None,
                find_entry_layer_id=lambda _net: 0,
                gather_input_spec_layers=lambda _net: (),
                get_assert_layer=lambda _net: assert_layer,
                get_input_ids=lambda _net: (),
                seed_from_input_specs=lambda _layers: seed,
            ),
            "act.front_end.model_synthesis": module(
                "act.front_end.model_synthesis",
                synthesize_models_from_specs=lambda _specs: {0: Model()},
            ),
            "act.front_end.vnnlib_loader.create_specs": module(
                "act.front_end.vnnlib_loader.create_specs",
                create_specs_from_paths=lambda *_args, **_kwargs: object(),
            ),
            "act.pipeline.verification.torch2act": module(
                "act.pipeline.verification.torch2act",
                TorchToACT=TorchToACT,
            ),
            "act.util.device_manager": module(
                "act.util.device_manager",
                initialize_device=lambda *_args: None,
            ),
            "act.back_end.hybridz_tf.property_residual_targets": module(
                "act.back_end.hybridz_tf.property_residual_targets",
                select_property_residual_targets=lambda **_kwargs: selector,
            ),
        }
        instance = SimpleNamespace(
            onnx_path=Path("/toy/model.onnx"),
            vnnlib_path=Path("/toy/property.vnnlib"),
            csv_path=Path("/toy/instances.csv"),
        )
        input_sha256 = {
            instance.onnx_path: probe._RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
            instance.vnnlib_path: probe._RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
            instance.csv_path: probe._RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
        }
        fixed_environment = {
            "HZ_QUERY_WORKERS": "20",
            "HZ_MILP_THREADS": "20",
            "HZ_LP_PREFILTER_THREADS": "20",
            "HZ_LP_PREFILTER_FRACTION": "1.0",
            "HZ_LP_PREFILTER_MAX_SECONDS": "1.0",
        }
        args = self._args(
            candidate_mode="rbs_adaptive_k4",
            phase_time_limit=30.0,
            residual_budget=16,
            run_nonce="a" * 64,
            fixed_environment=fixed_environment,
            fixed_environment_sha256="b" * 64,
        )
        property_cube = self._adaptive_property_cube_receipt()
        transaction = mock.Mock(
            name="forbidden_run_phase_transaction",
            side_effect=AssertionError("pre-gate rejection reached transaction"),
        )
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(probe, "_select_instance", return_value=instance),
            mock.patch.object(
                probe,
                "_sha256_file",
                side_effect=lambda path: input_sha256[Path(path)],
            ),
            mock.patch.object(
                probe, "_implementation_sha256", return_value="c" * 64
            ),
            mock.patch.object(
                probe,
                "_capture_resource_peaks",
                return_value=resources,
            ),
            mock.patch.object(
                probe,
                "_rbs_adaptive_property_cube_receipt",
                return_value=property_cube,
            ),
            mock.patch.object(probe, "_run_phase_transaction", transaction),
        ):
            receipt = probe._execute_probe(args)
        return receipt, transaction, pipeline, builder

    def test_configuration_is_single_iid_and_hard_bounded(self) -> None:
        probe._validate_args(self._args())
        for changes in (
            {"iid": 3},
            {"candidate_mode": "forged"},
            {"wall_timeout": 60.01},
            {"wall_timeout": True},
            {"phase_time_limit": 40.01},
            {"operator_exact_budget": 0},
            {"residual_budget": 17},
            {"cpu_threads": 21},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                probe.PhaseCliqueBuildProbeError
            ):
                probe._validate_args(self._args(**changes))

    def test_pcoh_mode_is_default_off_and_has_fixed_k2_contract(self) -> None:
        parser = probe._build_parser()
        parsed = parser.parse_args(["--output", "/tmp/toy-pcoh.json"])
        self.assertEqual(parsed.candidate_mode, "k4")
        fixed = {
            "candidate_mode": probe._PCOH_K2_BUILD_ONLY_MODE,
            "family": "cifar100_medium",
            "iid": 2,
            "wall_timeout": 60.0,
            "phase_time_limit": 25.0,
            "operator_exact_budget": 4,
            "residual_budget": 4,
            "cpu_threads": 20,
        }
        probe._validate_args(self._args(**fixed))
        for field, forged in (
            ("family", "cifar100_large"),
            ("iid", 3),
            ("wall_timeout", np.nextafter(60.0, 0.0)),
            ("phase_time_limit", np.nextafter(25.0, np.inf)),
            ("phase_time_limit", np.nextafter(25.0, 0.0)),
            ("operator_exact_budget", 5),
            ("residual_budget", 5),
            ("residual_time_limit", np.nextafter(4.0, 0.0)),
            ("cpu_threads", 19),
        ):
            with self.subTest(field=field), self.assertRaises(
                probe.PhaseCliqueBuildProbeError
            ):
                probe._validate_args(self._args(**{**fixed, field: forged}))

        shared = probe._shared_worker_deadline(
            self._args(
                **fixed,
                parent_hard_deadline_monotonic=160.0,
            ),
            now=100.0,
        )
        self.assertEqual(shared, 158.0)

    def test_pcoh_source_build_preflight_accepts_caps_and_rejects_each_excess(
        self,
    ) -> None:
        shape = {
            "output_dimension": 100,
            "continuous_columns": 60_000,
            "binary_columns": 4,
            "upper_rows": 104_900,
            "equality_rows": 0,
            "constraint_nonzeros": 11_000_000,
            "generator_nonzeros": 20_000,
        }

        def issue(candidate, seconds=27.0):
            return probe._pcoh_k2_source_build_preflight(
                candidate,
                build_seconds=seconds,
                input_sha256=self._pcoh_inputs(),
                implementation_sha256=self._pcoh_implementation(),
            )

        accepted = issue(shape)
        self.assertTrue(probe._pcoh_k2_source_build_preflight_valid(accepted))
        self.assertEqual(accepted["status"], "passed")
        cases = (
            ("build_seconds", shape, np.nextafter(27.0, np.inf)),
            ("outputs", {**shape, "output_dimension": 101}, 27.0),
            ("binaries", {**shape, "binary_columns": 5}, 27.0),
            ("continuous", {**shape, "continuous_columns": 60_001}, 27.0),
            ("rows", {**shape, "upper_rows": 104_901}, 27.0),
            (
                "constraint_nnz",
                {**shape, "constraint_nonzeros": 11_000_001},
                27.0,
            ),
            (
                "generator_nnz",
                {**shape, "generator_nonzeros": 20_001},
                27.0,
            ),
        )
        for name, candidate, seconds in cases:
            with self.subTest(name=name):
                rejected = issue(candidate, seconds)
                self.assertTrue(
                    probe._pcoh_k2_source_build_preflight_valid(rejected)
                )
                self.assertEqual(rejected["status"], "stop_loss")
                self.assertTrue(rejected["failed_conditions"])

    def test_pcoh_transaction_receipt_is_exact_and_tamper_closed(self) -> None:
        success = self._pcoh_success_transaction()
        self.assertEqual(set(success), probe._PCOH_K2_TRANSACTION_FIELDS)
        self.assertTrue(probe._pcoh_k2_transaction_receipt_valid(success))
        stopped = probe._pcoh_k2_stop_loss_receipt(
            stage="toy_stop",
            reason="toy",
            started=time.monotonic(),
            input_sha256=self._pcoh_inputs(),
            implementation_sha256=self._pcoh_implementation(),
            stage_resources={"toy": self._pcoh_resources()},
            build_only_transaction_called=True,
        )
        self.assertEqual(set(stopped), probe._PCOH_K2_TRANSACTION_FIELDS)
        self.assertTrue(probe._pcoh_k2_transaction_receipt_valid(stopped))
        self.assertIsNone(stopped["materialized_tightness_summary"])
        self.assertIsNone(stopped["materialized_tightness_summary_sha256"])
        self.assertIsNone(stopped["tightness_gate"])
        self.assertFalse(stopped["full_parent_lp_called"])
        self.assertFalse(stopped["full_parent_lp_solver_called"])
        mutations = (
            ("missing", lambda value: value.pop("pair_bundle_sha256")),
            ("extra", lambda value: value.__setitem__("forged", False)),
            (
                "authority",
                lambda value: value.__setitem__("verdict_authority", True),
            ),
            (
                "dimension",
                lambda value: value.__setitem__(
                    "fresh_dimensions", [2, 15, 4, 3, 21]
                ),
            ),
            (
                "nan",
                lambda value: value["timings"].__setitem__(
                    "total_seconds", float("nan")
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                forged = copy.deepcopy(success)
                forged.pop("receipt_sha256")
                mutate(forged)
                if name != "nan":
                    forged = probe._checksummed(forged)
                else:
                    forged["receipt_sha256"] = "0" * 64
                self.assertFalse(
                    probe._pcoh_k2_transaction_receipt_valid(forged)
                )

    def test_pcoh_caller_handoff_failure_always_releases_trusted_anchor(
        self,
    ) -> None:
        cases = (
            ("false", False, probe.PhaseCliqueBuildProbeError),
            ("exception", RuntimeError("toy failure"), RuntimeError),
            (
                "base_exception",
                KeyboardInterrupt("toy interrupt"),
                KeyboardInterrupt,
            ),
        )
        for name, outcome, expected_error in cases:
            with self.subTest(name=name):
                transaction = self._pcoh_success_transaction()
                body = {}
                self.assertIsNotNone(
                    probe._pcoh_k2_trusted_transaction_anchor(transaction)
                )
                try:
                    patch_kwargs = (
                        {"return_value": outcome}
                        if outcome is False
                        else {"side_effect": outcome}
                    )
                    with mock.patch.object(
                        probe,
                        "_pcoh_k2_transaction_receipt_valid",
                        **patch_kwargs,
                    ), self.assertRaises(expected_error):
                        probe._adopt_pcoh_k2_trusted_transaction(
                            body, transaction
                        )
                    self.assertNotIn("pcoh_k2_build_only", body)
                    self.assertIsNone(
                        probe._pcoh_k2_trusted_transaction_anchor(
                            transaction
                        )
                    )
                    with probe._PCOH_K2_TRUSTED_TRANSACTION_LOCK:
                        self.assertNotIn(
                            id(transaction),
                            probe._PCOH_K2_TRUSTED_TRANSACTIONS,
                        )
                finally:
                    probe._release_pcoh_k2_trusted_transaction(transaction)

    def test_pcoh_tightness_gate_exact_thresholds_and_nan_fail_closed(
        self,
    ) -> None:
        anchor = "a" * 64

        def summary(global_upper, final_upper, tax):
            return {
                "summary_sha256": anchor,
                "global_cube_upper_hex": float(global_upper).hex(),
                "final_structural_upper_hex": float(final_upper).hex(),
                "ideal_union_upper_hex": (1.0).hex(),
                "rounding_tax_exact": list(tax),
            }

        continuation = summary(100.0, 99.5, (1, 16))
        gate = probe._pcoh_k2_tightness_gate(
            continuation, expected_summary_sha256=anchor
        )
        self.assertTrue(
            probe._pcoh_k2_tightness_gate_valid(
                gate, continuation, expected_summary_sha256=anchor
            )
        )
        self.assertEqual(gate["delta_fraction"], [1, 2])
        self.assertEqual(
            gate["continuation_scale_threshold_fraction"], [1, 2]
        )
        self.assertEqual(gate["rounding_tax_threshold_fraction"], [1, 2])
        self.assertTrue(gate["continuation_candidate"])
        self.assertFalse(gate["strong_candidate"])

        below_continuation = summary(
            100.0,
            np.nextafter(99.5, 100.0),
            (1, 16),
        )
        self.assertFalse(
            probe._pcoh_k2_tightness_gate(
                below_continuation,
                expected_summary_sha256=anchor,
            )["continuation_candidate"]
        )

        strong = summary(100.0, 98.0, (1, 4))
        strong_gate = probe._pcoh_k2_tightness_gate(
            strong, expected_summary_sha256=anchor
        )
        self.assertEqual(strong_gate["delta_fraction"], [2, 1])
        self.assertEqual(
            strong_gate["strong_scale_threshold_fraction"], [2, 1]
        )
        self.assertEqual(
            strong_gate["rounding_tax_threshold_fraction"], [2, 1]
        )
        self.assertTrue(strong_gate["strong_candidate"])
        below_strong = summary(
            100.0,
            np.nextafter(98.0, 100.0),
            (1, 4),
        )
        self.assertFalse(
            probe._pcoh_k2_tightness_gate(
                below_strong, expected_summary_sha256=anchor
            )["strong_candidate"]
        )

        sufficient = summary(0.0, -1.0, (0, 1))
        sufficient_gate = probe._pcoh_k2_tightness_gate(
            sufficient, expected_summary_sha256=anchor
        )
        self.assertTrue(sufficient_gate["cube_already_sufficient"])
        self.assertTrue(sufficient_gate["zero_crossing"])
        self.assertFalse(sufficient_gate["continuation_candidate"])
        self.assertFalse(sufficient_gate["strong_candidate"])
        malformed = summary(1.0, 0.0, (0, 1))
        malformed["global_cube_upper_hex"] = "nan"
        with self.assertRaises(probe.PhaseCliqueBuildProbeError):
            probe._pcoh_k2_tightness_gate(
                malformed, expected_summary_sha256=anchor
            )

    def test_pcoh_json_verifier_and_trusted_anchor_reject_coherent_rehash(
        self,
    ) -> None:
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_build_only as build_only,
        )
        from act.back_end.solver import solver_hz
        import scipy.optimize as spo

        transaction = self._pcoh_success_transaction()
        original_anchor = transaction[
            "materialized_tightness_summary_sha256"
        ]
        try:
            json_summary = json.loads(json.dumps(
                transaction["materialized_tightness_summary"]
            ))
            self.assertTrue(
                build_only.verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                    json_summary,
                    expected_source_semantic_digest=transaction[
                        "source_semantic_digest"
                    ],
                    expected_stable_bit_ids=transaction["stable_bit_ids"],
                    expected_conditional_certificate_sha256=transaction[
                        "conditional_certificate_sha256"
                    ],
                    expected_summary_sha256=original_anchor,
                )
            )
            self.assertFalse(
                probe._pcoh_k2_transaction_receipt_valid(
                    json.loads(json.dumps(transaction))
                )
            )

            forbidden_lp = mock.Mock(
                side_effect=AssertionError("detached verifier LP forbidden")
            )
            forbidden_full_lp = mock.Mock(
                side_effect=AssertionError("full parent LP forbidden")
            )
            forbidden_base = mock.Mock(
                side_effect=AssertionError("base solver forbidden")
            )
            forbidden_objbound = mock.Mock(
                side_effect=AssertionError("objective solver forbidden")
            )
            with (
                mock.patch.object(spo, "linprog", forbidden_lp),
                mock.patch.object(
                    solver_hz, "hz_compute_lp_bounds", forbidden_full_lp
                ),
                mock.patch.object(
                    solver_hz, "hz_base_feasibility", forbidden_base
                ),
                mock.patch.object(
                    solver_hz, "hz_objbound_decide", forbidden_objbound
                ),
            ):
                self.assertTrue(
                    probe._pcoh_k2_transaction_receipt_valid(transaction)
                )
            for forbidden in (
                forbidden_lp,
                forbidden_full_lp,
                forbidden_base,
                forbidden_objbound,
            ):
                forbidden.assert_not_called()

            tampered_gate = copy.deepcopy(transaction["tightness_gate"])
            tampered_gate.pop("receipt_sha256")
            tampered_gate["strong_candidate"] = not tampered_gate[
                "strong_candidate"
            ]
            tampered_gate = probe._checksummed(tampered_gate)
            self.assertFalse(
                probe._pcoh_k2_tightness_gate_valid(
                    tampered_gate,
                    transaction["materialized_tightness_summary"],
                    expected_summary_sha256=original_anchor,
                )
            )

            coherent_summary = copy.deepcopy(
                transaction["materialized_tightness_summary"]
            )
            coherent_summary.pop("summary_sha256")
            coherent_summary["objective_binding_sha256"] = "1" * 64
            coherent_anchor = build_only._canonical_sha256(coherent_summary)
            coherent_summary["summary_sha256"] = coherent_anchor
            self.assertNotEqual(coherent_anchor, original_anchor)
            self.assertTrue(
                build_only.verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                    coherent_summary,
                    expected_source_semantic_digest=transaction[
                        "source_semantic_digest"
                    ],
                    expected_stable_bit_ids=transaction["stable_bit_ids"],
                    expected_conditional_certificate_sha256=transaction[
                        "conditional_certificate_sha256"
                    ],
                    expected_summary_sha256=coherent_anchor,
                )
            )
            coherent_gate = probe._pcoh_k2_tightness_gate(
                coherent_summary,
                expected_summary_sha256=coherent_anchor,
            )
            coherent_transaction = dict(transaction)
            coherent_transaction.pop("receipt_sha256")
            coherent_transaction["materialized_tightness_summary"] = (
                coherent_summary
            )
            coherent_transaction[
                "materialized_tightness_summary_sha256"
            ] = coherent_anchor
            coherent_transaction["tightness_gate"] = coherent_gate
            coherent_transaction = probe._checksummed(coherent_transaction)
            transaction.clear()
            transaction.update(coherent_transaction)
            self.assertFalse(
                probe._pcoh_k2_transaction_receipt_valid(transaction)
            )
        finally:
            probe._release_pcoh_k2_trusted_transaction(transaction)

    def test_pcoh_finalizer_rejects_coherent_rehash_and_always_releases_anchor(
        self,
    ) -> None:
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_build_only as build_only,
        )

        body = self._pcoh_terminal_body()
        transaction = body["pcoh_k2_build_only"]
        original_anchor = transaction[
            "materialized_tightness_summary_sha256"
        ]
        coherent_summary = copy.deepcopy(
            transaction["materialized_tightness_summary"]
        )
        coherent_summary.pop("summary_sha256")
        coherent_summary["objective_binding_sha256"] = "1" * 64
        coherent_anchor = build_only._canonical_sha256(coherent_summary)
        coherent_summary["summary_sha256"] = coherent_anchor
        self.assertNotEqual(coherent_anchor, original_anchor)
        self.assertTrue(
            build_only.verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                coherent_summary,
                expected_source_semantic_digest=transaction[
                    "source_semantic_digest"
                ],
                expected_stable_bit_ids=transaction["stable_bit_ids"],
                expected_conditional_certificate_sha256=transaction[
                    "conditional_certificate_sha256"
                ],
                expected_summary_sha256=coherent_anchor,
            )
        )
        transaction_body = dict(transaction)
        transaction_body.pop("receipt_sha256")
        transaction_body["materialized_tightness_summary"] = coherent_summary
        transaction_body[
            "materialized_tightness_summary_sha256"
        ] = coherent_anchor
        transaction_body["tightness_gate"] = probe._pcoh_k2_tightness_gate(
            coherent_summary,
            expected_summary_sha256=coherent_anchor,
        )
        resealed = probe._checksummed(transaction_body)
        transaction.clear()
        transaction.update(resealed)
        probe._finalize_pcoh_k2_integrity(body)
        self.assertEqual(body["phase_status"], "stop_loss")
        self.assertIn(
            "original_transaction_receipt_valid",
            body["pcoh_terminal_integrity"]["failed_conditions"],
        )
        self.assertIsNone(
            probe._pcoh_k2_trusted_transaction_anchor(transaction)
        )

        interrupted = self._pcoh_terminal_body()
        interrupted_transaction = interrupted["pcoh_k2_build_only"]
        self.assertIsNotNone(
            probe._pcoh_k2_trusted_transaction_anchor(
                interrupted_transaction
            )
        )
        with mock.patch.object(
            probe,
            "_finalize_pcoh_k2_integrity_registered",
            side_effect=KeyboardInterrupt("toy interrupt"),
        ), self.assertRaises(KeyboardInterrupt):
            probe._finalize_pcoh_k2_integrity(interrupted)
        self.assertIsNone(
            probe._pcoh_k2_trusted_transaction_anchor(
                interrupted_transaction
            )
        )

        mode_tampered = self._pcoh_terminal_body()
        mode_tampered_transaction = mode_tampered["pcoh_k2_build_only"]
        mode_tampered["candidate_mode"] = "k4"
        self.assertIsNotNone(
            probe._pcoh_k2_trusted_transaction_anchor(
                mode_tampered_transaction
            )
        )
        probe._finalize_pcoh_k2_integrity(mode_tampered)
        self.assertIsNone(
            probe._pcoh_k2_trusted_transaction_anchor(
                mode_tampered_transaction
            )
        )

    def test_pcoh_direct_pipeline_reuses_front_half_and_only_calls_build_only(
        self,
    ) -> None:
        source_digest = "a" * 64
        rival = SimpleNamespace(rival_id=7)
        batch = SimpleNamespace(
            rivals=(rival,),
            batch_sha256="b" * 64,
            live_assert_sha256="c" * 64,
        )
        hardness = object()
        focused = SimpleNamespace(
            rivals=(rival,),
            focused_subset_digest="d" * 64,
        )
        selection = SimpleNamespace(
            mappings=tuple(
                SimpleNamespace(stable_bcol_id=value)
                for value in (11, 5, 3, 7)
            ),
            selection_digest="e" * 64,
            property_digest="f" * 64,
            parent_semantic_digest=source_digest,
            operator_row_tag_digest="0" * 64,
        )
        diagnostic_certificates = (
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
        )
        materialized_tightness_summary = MappingProxyType(
            self._pcoh_tightness_summary(
                stable_ids=(3, 5),
                certificate_sha256=diagnostic_certificates,
            )
        )
        diagnostic = SimpleNamespace(
            schema="act.hybridz_pcoh_build_only_diagnostic.v2",
            status="build_only_materialized_validated_and_released",
            source_semantic_digest=source_digest,
            fresh_semantic_digest="1" * 64,
            source_dimensions=(2, 10, 4, 0, 20),
            fresh_dimensions=(2, 14, 4, 3, 21),
            conditional_certificate_sha256=diagnostic_certificates,
            pair_bundle_sha256="6" * 64,
            fresh_issuance_sha256="7" * 64,
            diagnostic_sha256="8" * 64,
            materialized_tightness_summary=(
                materialized_tightness_summary
            ),
            full_parent_lp_called=False,
            receipt=MappingProxyType({
                "receipt_sha256": "9" * 64,
                "full_parent_lp_called": False,
                "full_parent_lp_solver_called": False,
                "materialized_tightness_summary_sha256": (
                    materialized_tightness_summary["summary_sha256"]
                ),
                "materialized_tightness_summary": (
                    materialized_tightness_summary
                ),
                "resource_preflight": MappingProxyType({
                    "passed": True,
                    "caller_supplied": False,
                }),
                "resource_postflight": MappingProxyType({
                    "passed": True,
                    "caller_supplied": False,
                }),
            }),
        )
        source_build = SimpleNamespace(hz=SimpleNamespace(n_out=2))
        raw_token = object()
        run_build_only = mock.Mock(return_value=diagnostic)
        verify_build_only = mock.Mock(return_value=True)
        forbidden_phase_transaction = mock.Mock(
            side_effect=AssertionError("phase transaction forbidden")
        )
        forbidden_lp = mock.Mock(side_effect=AssertionError("LP forbidden"))
        forbidden_base = mock.Mock(
            side_effect=AssertionError("base feasibility forbidden")
        )
        forbidden_objbound = mock.Mock(
            side_effect=AssertionError("objective verdict forbidden")
        )
        forbidden_replay = mock.Mock(
            side_effect=AssertionError("strict replay forbidden")
        )
        raw_module = "act.back_end.hybridz_tf.raw_vnnlib_rival_adapter"
        focus_module = (
            "act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge"
        )
        literal_module = (
            "act.back_end.hybridz_tf.operator_exact_relu_phase_literals"
        )
        clique_module = (
            "act.back_end.hybridz_tf.operator_phase_clique_pipeline"
        )
        build_only_module = (
            "act.back_end.hybridz_tf.operator_phase_conditioned_build_only"
        )
        with (
            mock.patch(
                raw_module + ".issue_raw_vnnlib_top1_candidate",
                return_value=raw_token,
            ) as issue_raw,
            mock.patch(
                raw_module + ".consume_raw_vnnlib_top1_candidate",
                return_value=batch,
            ) as consume_raw,
            mock.patch(
                raw_module + ".validate_consumed_raw_vnnlib_rival_batch",
                return_value=True,
            ) as validate_raw,
            mock.patch(
                focus_module + ".issue_raw_rival_exact_hardness_receipt",
                return_value=hardness,
            ) as issue_hardness,
            mock.patch(
                focus_module + ".select_raw_focused_rivals",
                return_value=focused,
            ) as select_focus,
            mock.patch(
                focus_module + ".verify_raw_rival_exact_hardness_receipt",
                return_value=True,
            ) as verify_hardness,
            mock.patch(
                focus_module + ".verify_raw_focused_rival_selection",
                return_value=True,
            ) as verify_focus,
            mock.patch(
                literal_module
                + ".derive_operator_exact_relu_property_phase_literals",
                return_value=selection,
            ) as derive_selection,
            mock.patch(
                literal_module
                + ".verify_operator_exact_relu_property_phase_selection",
                return_value=True,
            ) as verify_selection,
            mock.patch(
                clique_module + "._snapshot_b1_bounds",
                return_value=(
                    np.zeros((1, 2), dtype=np.float64),
                    np.ones((1, 2), dtype=np.float64),
                ),
            ),
            mock.patch(
                clique_module + "._exact_interval_upper_violations",
                return_value=(1.0,),
            ),
            mock.patch(
                clique_module + "._interval_frame_digest",
                return_value="a" * 64,
            ),
            mock.patch(
                "act.back_end.hybridz_tf.adaptive_phase_forest."
                "sparse_hz_semantic_digest",
                return_value=source_digest,
            ),
            mock.patch(
                build_only_module
                + ".run_phase_conditioned_objective_hull_build_only",
                run_build_only,
            ),
            mock.patch(
                build_only_module
                + ".verify_phase_conditioned_objective_hull_build_only_diagnostic",
                verify_build_only,
            ),
            mock.patch.object(
                probe, "_capture_resource_peaks",
                return_value=self._pcoh_resources(),
            ),
            mock.patch.object(
                probe, "_run_phase_transaction", forbidden_phase_transaction
            ),
            mock.patch.object(
                probe, "_certified_relaxed_upper", forbidden_lp
            ),
            mock.patch(
                "act.back_end.solver.solver_hz.hz_base_feasibility",
                forbidden_base,
            ),
            mock.patch(
                "act.back_end.solver.solver_hz.hz_objbound_decide",
                forbidden_objbound,
            ),
            mock.patch(
                "act.pipeline.verification.strict_replay.make_strict_replay",
                forbidden_replay,
            ),
        ):
            result = probe._run_pcoh_k2_build_only_pipeline(
                source_build,
                input_sha256=self._pcoh_inputs(),
                implementation_sha256=self._pcoh_implementation(),
                vnnlib_path="/toy/property.vnnlib",
                expected_vnnlib_sha256=self._pcoh_inputs()["vnnlib"],
                live_assert_params={},
                output_lower=np.zeros((1, 2), dtype=np.float64),
                output_upper=np.ones((1, 2), dtype=np.float64),
                residual_selector_receipt={"joint_focus_rival_id": 9},
                residual_selector_property_sha256="b" * 64,
                deadline=time.monotonic() + 30.0,
                phase_time_limit=25.0,
                torch_module=SimpleNamespace(),
            )
        self.assertTrue(probe._pcoh_k2_transaction_receipt_valid(result))
        self.assertEqual(result["status"], "built_and_released")
        self.assertEqual(result["stable_bit_ids"], [3, 5])
        self.assertEqual(result["focused_encoded_row"], 9)
        self.assertEqual(result["focused_rival_id"], 7)
        self.assertFalse(result["full_parent_lp_called"])
        self.assertFalse(result["full_parent_lp_solver_called"])
        self.assertEqual(
            result["materialized_tightness_summary_sha256"],
            materialized_tightness_summary["summary_sha256"],
        )
        self.assertEqual(
            result["materialized_tightness_summary"][
                "final_structural_upper_hex"
            ],
            (1.0).hex(),
        )
        self.assertTrue(result["tightness_gate"]["strong_candidate"])
        self.assertFalse(result["tightness_gate"]["proof_authority"])
        self.assertFalse(result["tightness_gate"]["verdict_authority"])
        run_build_only.assert_called_once()
        build_kwargs = run_build_only.call_args.kwargs
        self.assertEqual(build_kwargs["stable_bit_ids"], (3, 5))
        self.assertEqual(build_kwargs["focused_rival_id"], 7)
        verify_build_only.assert_called_once_with(diagnostic)
        for called in (
            issue_raw,
            consume_raw,
            validate_raw,
            issue_hardness,
            select_focus,
            verify_hardness,
            verify_focus,
            derive_selection,
            verify_selection,
        ):
            self.assertEqual(called.call_count, 1)
        for forbidden in (
            forbidden_phase_transaction,
            forbidden_lp,
            forbidden_base,
            forbidden_objbound,
            forbidden_replay,
        ):
            forbidden.assert_not_called()

    def test_pcoh_k3_direct_success_early_stop_and_notstrong_are_distinct(self):
        strong, run_strong, strong_sha = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success()
        )
        self.assertEqual(strong["status"], "strong_promotion")
        self.assertIsNotNone(
            probe._pcoh_k3_trusted_transaction_anchor(strong)
        )
        self.assertEqual(strong["trusted_outcome_sha256"], strong_sha)
        self.assertEqual(strong["pair_local_lp_actual_calls"], 12)
        self.assertEqual(strong["conditional_local_lp_actual_calls"], 8)
        self.assertEqual(strong["total_local_lp_actual_calls"], 20)
        self.assertEqual(strong["conditional_checker_actual_calls"], 26)
        self.assertTrue(strong["strong_tightness_gate"]["strong_candidate"])
        run_strong.assert_called_once()
        self.assertEqual(
            run_strong.call_args.kwargs["retained_k2_stable_bit_ids"],
            (52557, 52558),
        )
        probe._release_pcoh_k3_trusted_transaction(strong)

        notstrong, _, _ = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success(final_hex=(109.0).hex())
        )
        self.assertEqual(notstrong["status"], "built_but_not_strong")
        self.assertIsNotNone(
            probe._pcoh_k3_trusted_transaction_anchor(notstrong)
        )
        self.assertFalse(
            notstrong["strong_tightness_gate"]["strong_candidate"]
        )
        probe._release_pcoh_k3_trusted_transaction(notstrong)

        stopped, _, stop_sha = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_stop()
        )
        self.assertEqual(stopped["status"], "strong_target_stop")
        self.assertIsNotNone(
            probe._pcoh_k3_trusted_transaction_anchor(stopped)
        )
        self.assertEqual(stopped["trusted_outcome_sha256"], stop_sha)
        self.assertIsNone(stopped["fresh_dimensions"])
        self.assertIsNone(stopped["materialized_tightness_summary"])
        self.assertEqual(stopped["conditional_local_lp_actual_calls"], 1)
        probe._release_pcoh_k3_trusted_transaction(stopped)

    def test_pcoh_k3_resource_stop_preserves_stage_counters_and_no_fresh(self):
        verifier = (
            "act.back_end.hybridz_tf."
            "operator_phase_conditioned_k3_build_only."
            "verify_detached_phase_conditioned_objective_hull_k3_build_only"
        )
        cases = (
            ("pre_scheduled", 0, 0, 12, 0),
            ("pre_fresh_materialization", 3, 2, 15, 11),
        )
        for stage, scheduled_lp, accepted, total_lp, checker in cases:
            with self.subTest(stage=stage):
                detached = self._pcoh_k3_detached_resource_stop(
                    stage=stage,
                    scheduled_lp=scheduled_lp,
                    accepted=accepted,
                )
                transaction, run_k3, outcome_sha = (
                    self._run_pcoh_k3_direct_mock(detached)
                )
                try:
                    self.assertEqual(transaction["status"], "resource_stop")
                    self.assertEqual(
                        transaction["outcome_kind"], "resource_stop"
                    )
                    self.assertEqual(transaction["failed_stage"], stage)
                    self.assertEqual(
                        transaction["reason"], detached["reason"]
                    )
                    self.assertEqual(
                        transaction["trusted_outcome_sha256"], outcome_sha
                    )
                    self.assertEqual(
                        transaction["pair_local_lp_actual_calls"], 12
                    )
                    self.assertEqual(
                        transaction["conditional_local_lp_actual_calls"],
                        scheduled_lp,
                    )
                    self.assertEqual(
                        transaction["total_local_lp_actual_calls"], total_lp
                    )
                    self.assertEqual(
                        transaction["conditional_checker_actual_calls"],
                        checker,
                    )
                    self.assertEqual(
                        transaction["resource_gate_rejection_sha256"],
                        detached["receipt"]["resource_gate_rejection"][
                            "rejection_sha256"
                        ],
                    )
                    self.assertEqual(
                        transaction["resource_gate_rejection"],
                        detached["receipt"]["resource_gate_rejection"],
                    )
                    for name in (
                        "fresh_dimensions",
                        "fresh_semantic_digest",
                        "materialized_tightness_summary_sha256",
                        "materialized_tightness_summary",
                        "strong_tightness_gate",
                    ):
                        self.assertIsNone(transaction[name], name)
                    for name in (
                        "proof_authority",
                        "verdict_authority",
                        "provenance_authority",
                        "authenticity_authority",
                        "fresh_build_returned",
                    ):
                        self.assertIs(transaction[name], False, name)
                    for name in (
                        "fresh_issue_called",
                        "fresh_build_returned",
                        "fresh_descriptor_returned",
                    ):
                        self.assertIs(
                            transaction["detached_outcome"][name], False, name
                        )
                    self.assertIsNotNone(
                        probe._pcoh_k3_trusted_transaction_anchor(transaction)
                    )
                    with mock.patch(verifier, return_value=True):
                        self.assertTrue(
                            probe._pcoh_k3_transaction_receipt_valid(
                                transaction
                            )
                        )
                    run_k3.assert_called_once()
                finally:
                    probe._release_pcoh_k3_trusted_transaction(transaction)

    def test_pcoh_k3_resource_stop_finalizes_and_coherent_tamper_closes(self):
        verifier = (
            "act.back_end.hybridz_tf."
            "operator_phase_conditioned_k3_build_only."
            "verify_detached_phase_conditioned_objective_hull_k3_build_only"
        )
        detached = self._pcoh_k3_detached_resource_stop(
            stage="pre_fresh_materialization",
            scheduled_lp=3,
            accepted=2,
        )
        body, anchor = self._pcoh_k3_terminal_body(detached)
        original = body["pcoh_k3_build_only"]
        with mock.patch(verifier, return_value=True), mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(body)
        self.assertIs(body["pcoh_k3_build_only"], original)
        self.assertEqual(body["phase_status"], "resource_stop")
        self.assertEqual(body["failed_stage"], "pre_fresh_materialization")
        self.assertEqual(body["fallback_reason"], detached["reason"])
        integrity = body["pcoh_k3_terminal_integrity"]
        self.assertTrue(integrity["terminal_integrity_passed"])
        self.assertEqual(integrity["status"], "resource_stop")
        self.assertEqual(integrity["total_local_lp_actual_calls"], 15)
        self.assertEqual(integrity["conditional_checker_actual_calls"], 11)
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

        vetoed, anchor = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_resource_stop(
                stage="pre_fresh_materialization",
                scheduled_lp=3,
                accepted=2,
            )
        )
        original = vetoed["pcoh_k3_build_only"]
        snapshot = copy.deepcopy(original)
        vetoed["resource_usage"]["peak_rss_bytes"] = (
            probe._PCOH_K2_MAX_RSS_BYTES + 1
        )
        with mock.patch(verifier, return_value=True), mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(vetoed)
        integrity = vetoed["pcoh_k3_terminal_integrity"]
        self.assertIs(vetoed["pcoh_k3_build_only"], original)
        self.assertEqual(original, snapshot)
        self.assertEqual(vetoed["phase_status"], "stop_loss")
        self.assertEqual(
            vetoed["failed_stage"], "pcoh_k3_terminal_integrity"
        )
        self.assertIn(
            "terminal_resources_recorded", integrity["failed_conditions"]
        )
        self.assertFalse(integrity["terminal_integrity_passed"])
        self.assertEqual(
            integrity["original_transaction_sha256"],
            snapshot["receipt_sha256"],
        )
        self.assertEqual(
            integrity["trusted_outcome_sha256"],
            snapshot["trusted_outcome_sha256"],
        )
        for name in (
            "pair_local_lp_actual_calls",
            "conditional_local_lp_actual_calls",
            "total_local_lp_actual_calls",
            "conditional_checker_actual_calls",
        ):
            self.assertEqual(integrity[name], snapshot[name], name)
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

        tampered, anchor = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_resource_stop()
        )
        original = tampered["pcoh_k3_build_only"]
        original["resource_gate_rejection"]["reason"] = "coherent_tamper"
        original.pop("receipt_sha256")
        original["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(original)
        ).hexdigest()
        with mock.patch(verifier, return_value=True), mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(tampered)
        self.assertIs(tampered["pcoh_k3_build_only"], original)
        self.assertEqual(tampered["phase_status"], "stop_loss")
        self.assertIn(
            "original_transaction_receipt_valid",
            tampered["pcoh_k3_terminal_integrity"]["failed_conditions"],
        )
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

    def test_pcoh_k3_resource_stop_baseexception_releases_registry(self):
        original_register = probe._register_pcoh_k3_trusted_transaction
        captured = []

        def register_then_interrupt(transaction, **kwargs):
            original_register(transaction, **kwargs)
            captured.append(transaction)
            raise KeyboardInterrupt("resource registration interrupted")

        with mock.patch.object(
            probe,
            "_register_pcoh_k3_trusted_transaction",
            side_effect=register_then_interrupt,
        ):
            stopped, run_k3, _ = self._run_pcoh_k3_direct_mock(
                self._pcoh_k3_detached_resource_stop()
            )
        self.assertEqual(stopped["status"], "stop_loss")
        self.assertEqual(stopped["pair_local_lp_actual_calls"], 0)
        self.assertEqual(stopped["conditional_local_lp_actual_calls"], 0)
        self.assertTrue(captured)
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(captured[0])
        )
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(stopped)
        )
        run_k3.assert_called_once()

    def test_pcoh_k3_adoption_failure_preserves_safe_original_for_terminal(self):
        for name, validator in (
            ("false", mock.Mock(return_value=False)),
            (
                "keyboard_interrupt",
                mock.Mock(side_effect=KeyboardInterrupt("toy validator")),
            ),
        ):
            with self.subTest(validator=name):
                transaction, _, _ = self._run_pcoh_k3_direct_mock(
                    self._pcoh_k3_detached_success()
                )
                body = {}
                self.assertTrue(
                    probe._pcoh_k3_transaction_basic_receipt_valid(
                        transaction
                    )
                )
                expected = (
                    probe.PhaseCliqueBuildProbeError
                    if name == "false"
                    else KeyboardInterrupt
                )
                with mock.patch.object(
                    probe,
                    "_pcoh_k3_transaction_receipt_valid",
                    validator,
                ), self.assertRaises(expected):
                    probe._adopt_pcoh_k3_trusted_transaction(
                        body, transaction
                    )
                self.assertIs(body["pcoh_k3_build_only"], transaction)
                self.assertNotEqual(transaction["status"], "stop_loss")
                self.assertIsNone(
                    probe._pcoh_k3_trusted_transaction_anchor(transaction)
                )

    def test_pcoh_k3_direct_baseexception_and_forbidden_routes_fail_closed(self):
        forbidden_k2 = mock.Mock(
            side_effect=AssertionError("K2 build-only route forbidden")
        )
        forbidden_phase = mock.Mock(
            side_effect=AssertionError("phase transaction forbidden")
        )
        with mock.patch.object(
            probe, "_run_pcoh_k2_build_only_pipeline", forbidden_k2
        ), mock.patch.object(
            probe, "_run_phase_transaction", forbidden_phase
        ):
            result, run_k3, _ = self._run_pcoh_k3_direct_mock(
                self._pcoh_k3_detached_success(),
                raised=KeyboardInterrupt("toy interruption"),
            )
        self.assertEqual(result["status"], "stop_loss")
        self.assertTrue(result["k3_transaction_called"])
        self.assertTrue(probe._pcoh_k3_transaction_receipt_valid(result))
        self.assertIn("KeyboardInterrupt", result["reason"])
        run_k3.assert_called_once()
        forbidden_k2.assert_not_called()
        forbidden_phase.assert_not_called()
        with probe._PCOH_K3_TRUSTED_TRANSACTION_LOCK:
            self.assertFalse(probe._PCOH_K3_TRUSTED_TRANSACTIONS)

        terminal_body, anchor, placeholder = (
            self._pcoh_k3_stop_terminal_body()
        )
        self.assertIsNot(placeholder, result)
        terminal_body.update({
            "pcoh_k3_build_only": result,
            "k3_transaction_called": result["k3_transaction_called"],
            "phase_status": result["status"],
            "failed_stage": result["failed_stage"],
            "fallback_reason": result["reason"],
        })
        result_snapshot = copy.deepcopy(result)
        with mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(terminal_body)
        self.assertIs(terminal_body["pcoh_k3_build_only"], result)
        self.assertEqual(result, result_snapshot)
        self.assertEqual(
            terminal_body["failed_stage"], result_snapshot["failed_stage"]
        )
        self.assertEqual(
            terminal_body["fallback_reason"], result_snapshot["reason"]
        )
        self.assertTrue(
            terminal_body["pcoh_k3_terminal_integrity"][
                "terminal_integrity_passed"
            ]
        )

        original_register = probe._register_pcoh_k3_trusted_transaction

        def insert_then_interrupt(transaction, **kwargs):
            original_register(transaction, **kwargs)
            raise KeyboardInterrupt("interrupt after registry insertion")

        with mock.patch.object(
            probe,
            "_register_pcoh_k3_trusted_transaction",
            side_effect=insert_then_interrupt,
        ):
            interrupted, run_k3, _ = self._run_pcoh_k3_direct_mock(
                self._pcoh_k3_detached_success()
            )
        self.assertEqual(interrupted["status"], "stop_loss")
        self.assertIn("KeyboardInterrupt", interrupted["reason"])
        run_k3.assert_called_once()
        with probe._PCOH_K3_TRUSTED_TRANSACTION_LOCK:
            self.assertFalse(probe._PCOH_K3_TRUSTED_TRANSACTIONS)

    def test_pcoh_k3_contract_deadline_and_parent_error_are_independent(self):
        k3 = self._args(
            candidate_mode=probe._PCOH_K3_BUILD_ONLY_MODE,
            phase_time_limit=25.0,
            parent_hard_deadline_monotonic=160.0,
        )
        self.assertEqual(probe._shared_worker_deadline(k3, now=100.0), 156.0)
        probe._validate_args(k3)
        bad = copy.copy(k3)
        bad.phase_time_limit = 22.0
        with self.assertRaisesRegex(
            probe.PhaseCliqueBuildProbeError,
            "pcoh_k3_build_only fixed contract mismatch: phase_time_limit",
        ):
            probe._validate_args(bad)
        k2 = copy.copy(k3)
        k2.candidate_mode = probe._PCOH_K2_BUILD_ONLY_MODE
        self.assertEqual(probe._shared_worker_deadline(k2, now=100.0), 158.0)

        parent = probe._parent_error_receipt(
            k3,
            run_nonce="a" * 64,
            failed_stage="outer_hard_stop",
            error_type="TimeoutError",
            error="toy timeout",
            elapsed_seconds=60.0,
        )
        self.assertEqual(parent["phase_status"], "stop_loss")
        self.assertTrue(parent["build_only"])
        self.assertNotIn("diagnostic_lp_called", parent)
        self.assertNotIn("strict_replay_called", parent)
        self.assertEqual(parent["pair_local_lp_actual_calls"], 0)
        self.assertEqual(parent["conditional_checker_actual_calls"], 0)
        self.assertFalse(parent["proof_authority"])
        self.assertFalse(parent["verdict_authority"])
        nan_summary = self._pcoh_k3_summary(final_hex="nan")
        with self.assertRaises(probe.PhaseCliqueBuildProbeError):
            probe._pcoh_k3_strong_tightness_gate(
                nan_summary,
                source_semantic_digest=(
                    probe._PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
                ),
                selection_digest=probe._PCOH_K3_EXPECTED_SELECTION_DIGEST,
                focused_encoded_row=probe._PCOH_K3_FOCUSED_ENCODED_ROW,
                focused_rival_id=probe._PCOH_K3_FOCUSED_RIVAL_ID,
                retained_k2_stable_bit_ids=(52557, 52558),
                stable_bit_ids=(52557, 52558, 52559),
            )

    def test_pcoh_k3_finalizer_accepts_three_terminals_and_tamper_closes(self):
        verifier = (
            "act.back_end.hybridz_tf."
            "operator_phase_conditioned_k3_build_only."
            "verify_detached_phase_conditioned_objective_hull_k3_build_only"
        )
        for name, detached, expected in (
            (
                "strong",
                self._pcoh_k3_detached_success(),
                "strong_promotion",
            ),
            (
                "not_strong",
                self._pcoh_k3_detached_success(
                    final_hex=(109.0).hex()
                ),
                "built_but_not_strong",
            ),
            (
                "early_stop",
                self._pcoh_k3_detached_stop(),
                "strong_target_stop",
            ),
        ):
            with self.subTest(name=name):
                body, anchor = self._pcoh_k3_terminal_body(detached)
                original = body["pcoh_k3_build_only"]
                with mock.patch(verifier, return_value=True), mock.patch.object(
                    probe,
                    "_pcoh_k3_fixed_baseline_artifact_anchor",
                    return_value=anchor,
                ):
                    probe._finalize_pcoh_k3_integrity(body)
                self.assertEqual(body["phase_status"], expected)
                self.assertEqual(
                    body["pcoh_k3_terminal_integrity"]["status"], expected
                )
                self.assertFalse(
                    body["pcoh_k3_terminal_integrity"]["failed_conditions"]
                )
                self.assertIsNone(
                    probe._pcoh_k3_trusted_transaction_anchor(original)
                )
                if expected == "strong_target_stop":
                    self.assertIsNone(
                        original["materialized_tightness_summary"]
                    )
                    self.assertFalse(
                        original["detached_outcome"]["fresh_issue_called"]
                    )

        tampered, anchor = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        original = tampered["pcoh_k3_build_only"]
        forged = copy.deepcopy(original)
        forged["total_local_lp_actual_calls"] = 19
        forged.pop("receipt_sha256")
        forged["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(forged)
        ).hexdigest()
        original.clear()
        original.update(forged)
        malformed_snapshot = copy.deepcopy(original)
        with mock.patch(verifier, return_value=True), mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(tampered)
        self.assertEqual(tampered["phase_status"], "stop_loss")
        self.assertIn(
            "original_transaction_receipt_valid",
            tampered["pcoh_k3_terminal_integrity"]["failed_conditions"],
        )
        self.assertIs(tampered["pcoh_k3_build_only"], original)
        self.assertEqual(original, malformed_snapshot)
        self.assertFalse(
            tampered["pcoh_k3_terminal_integrity"][
                "terminal_integrity_passed"
            ]
        )
        self.assertTrue(
            tampered["pcoh_k3_terminal_integrity"][
                "original_transaction_preserved"
            ]
        )
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

        mode_tampered, anchor = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        original = mode_tampered["pcoh_k3_build_only"]
        original_snapshot = copy.deepcopy(original)
        mode_tampered["candidate_mode"] = "k4"
        with mock.patch(verifier, return_value=True), mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(mode_tampered)
        self.assertEqual(mode_tampered["phase_status"], "stop_loss")
        self.assertIn(
            "fixed_candidate_mode",
            mode_tampered["pcoh_k3_terminal_integrity"][
                "failed_conditions"
            ],
        )
        self.assertIs(mode_tampered["pcoh_k3_build_only"], original)
        self.assertEqual(original, original_snapshot)
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

        interrupted, _ = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        original = interrupted["pcoh_k3_build_only"]
        with mock.patch.object(
            probe,
            "_finalize_pcoh_k3_integrity_registered",
            side_effect=KeyboardInterrupt("toy finalizer interruption"),
        ):
            probe._finalize_pcoh_k3_integrity(interrupted)
        self.assertEqual(interrupted["phase_status"], "stop_loss")
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

    def test_pcoh_k3_finalizer_preserves_helper_stop_and_separates_veto(self):
        body, anchor, original = self._pcoh_k3_stop_terminal_body()
        snapshot = copy.deepcopy(original)
        with mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(body)

        self.assertIs(body["pcoh_k3_build_only"], original)
        self.assertEqual(original, snapshot)
        self.assertEqual(body["phase_status"], "stop_loss")
        self.assertEqual(body["failed_stage"], snapshot["failed_stage"])
        self.assertEqual(body["fallback_reason"], snapshot["reason"])
        self.assertEqual(
            body["pcoh_k3_transaction_sha256"],
            snapshot["receipt_sha256"],
        )
        integrity = body["pcoh_k3_terminal_integrity"]
        self.assertTrue(integrity["terminal_integrity_passed"])
        self.assertFalse(integrity["transaction_terminal_candidate"])
        self.assertTrue(integrity["transaction_stop_loss"])
        self.assertTrue(integrity["original_transaction_preserved"])
        self.assertEqual(
            integrity["original_transaction_sha256"],
            snapshot["receipt_sha256"],
        )
        self.assertNotIn("original_transaction", integrity)
        self.assertFalse(integrity["failed_conditions"])
        self.assertEqual(integrity["reason"], "upstream_stop_loss_preserved")
        self.assertFalse(integrity["proof_authority"])
        self.assertFalse(integrity["verdict_authority"])
        self.assertTrue(
            probe._local_receipt_checksum_valid(
                original, schema=probe._PCOH_K3_TRANSACTION_SCHEMA
            )
        )

        vetoed, anchor, original = self._pcoh_k3_stop_terminal_body()
        snapshot = copy.deepcopy(original)
        vetoed["inputs_unchanged"] = False
        with mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(vetoed)
        self.assertIs(vetoed["pcoh_k3_build_only"], original)
        self.assertEqual(original, snapshot)
        self.assertEqual(vetoed["phase_status"], "stop_loss")
        self.assertEqual(
            vetoed["failed_stage"], "pcoh_k3_terminal_integrity"
        )
        self.assertIn(
            "inputs_unchanged",
            vetoed["pcoh_k3_terminal_integrity"]["failed_conditions"],
        )
        self.assertFalse(
            vetoed["pcoh_k3_terminal_integrity"][
                "terminal_integrity_passed"
            ]
        )

    def test_pcoh_k3_stop_rejects_resigned_partial_payloads_and_mismatches(
        self,
    ):
        _, _, clean = self._pcoh_k3_stop_terminal_body()

        def resigned(name, value):
            forged = copy.deepcopy(clean)
            forged[name] = value
            forged.pop("receipt_sha256")
            forged["receipt_sha256"] = hashlib.sha256(
                probe._canonical_json(forged)
            ).hexdigest()
            return forged

        for name, value in (
            ("fresh_dimensions", [100, 52665, 4, 4, 98975]),
            ("trusted_outcome_sha256", "a" * 64),
            ("full_batch_sha256", "b" * 64),
            ("stable_bit_ids", [52557, 52558, 52559]),
            ("materialized_tightness_summary_sha256", "c" * 64),
            ("active_pattern_mask", [True] * 8),
            ("local_lp_actual_call_cap", 20.0),
            ("conditional_checker_actual_call_cap", 34.0),
            ("instance_count", True),
        ):
            with self.subTest(partial_output_field=name):
                forged = resigned(name, value)
                self.assertTrue(
                    probe._local_receipt_checksum_valid(
                        forged, schema=probe._PCOH_K3_TRANSACTION_SCHEMA
                    )
                )
                self.assertFalse(
                    probe._pcoh_k3_transaction_receipt_valid(forged)
                )

        input_mismatch, anchor, original = (
            self._pcoh_k3_stop_terminal_body()
        )
        forged = copy.deepcopy(original)
        forged["input_sha256"]["onnx"] = "f" * 64
        forged.pop("receipt_sha256")
        forged["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(forged)
        ).hexdigest()
        self.assertTrue(probe._pcoh_k3_transaction_receipt_valid(forged))
        input_mismatch["pcoh_k3_build_only"] = forged
        with mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(input_mismatch)
        self.assertIn(
            "fixed_input_sha256",
            input_mismatch["pcoh_k3_terminal_integrity"][
                "failed_conditions"
            ],
        )
        self.assertIs(input_mismatch["pcoh_k3_build_only"], forged)

        baseline_mismatch, anchor, original = (
            self._pcoh_k3_stop_terminal_body()
        )
        forged = copy.deepcopy(original)
        forged["baseline_anchor_verified"] = False
        forged["baseline_anchor_receipt_sha256"] = None
        forged.pop("receipt_sha256")
        forged["receipt_sha256"] = hashlib.sha256(
            probe._canonical_json(forged)
        ).hexdigest()
        self.assertTrue(probe._pcoh_k3_transaction_receipt_valid(forged))
        baseline_mismatch["pcoh_k3_build_only"] = forged
        with mock.patch.object(
            probe,
            "_pcoh_k3_fixed_baseline_artifact_anchor",
            return_value=anchor,
        ):
            probe._finalize_pcoh_k3_integrity(baseline_mismatch)
        self.assertIn(
            "baseline_anchor_bound_end_to_end",
            baseline_mismatch["pcoh_k3_terminal_integrity"][
                "failed_conditions"
            ],
        )
        self.assertIs(baseline_mismatch["pcoh_k3_build_only"], forged)

        for field, value, expected_failure in (
            ("k3_transaction_called", True, "transaction_call_flags_bound"),
            ("k3_transaction_called", 0, "transaction_call_flags_bound"),
            ("k2_build_only_called", True, "forbidden_routes_not_called"),
            ("phase_transaction_called", True, "forbidden_routes_not_called"),
            ("local_lp_actual_call_cap", 19, "fixed_call_caps_and_zero_edges"),
            (
                "local_lp_actual_call_cap",
                20.0,
                "fixed_call_caps_and_zero_edges",
            ),
            (
                "conditional_checker_actual_call_cap",
                33,
                "fixed_call_caps_and_zero_edges",
            ),
            ("certified_edge_count", 1, "fixed_call_caps_and_zero_edges"),
            ("certified_edge_count", False, "fixed_call_caps_and_zero_edges"),
            (
                "operator_exact_budget",
                4.0,
                "fixed_operator_and_residual_budgets",
            ),
            ("iid", 2.0, "fixed_iid"),
            (
                "cpu_threads",
                20.0,
                "fixed_operator_and_residual_budgets",
            ),
        ):
            with self.subTest(top_level_mismatch=field, forged_value=value):
                mismatched, anchor, original = (
                    self._pcoh_k3_stop_terminal_body()
                )
                mismatched[field] = value
                snapshot = copy.deepcopy(original)
                with mock.patch.object(
                    probe,
                    "_pcoh_k3_fixed_baseline_artifact_anchor",
                    return_value=anchor,
                ):
                    probe._finalize_pcoh_k3_integrity(mismatched)
                self.assertIn(
                    expected_failure,
                    mismatched["pcoh_k3_terminal_integrity"][
                        "failed_conditions"
                    ],
                )
                self.assertIs(mismatched["pcoh_k3_build_only"], original)
                self.assertEqual(original, snapshot)

    def test_pcoh_k3_finalizer_baseexception_never_overwrites_transaction(self):
        body, _ = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        original = body["pcoh_k3_build_only"]
        snapshot = copy.deepcopy(original)
        with mock.patch.object(
            probe,
            "_finalize_pcoh_k3_integrity_registered",
            side_effect=KeyboardInterrupt("toy finalizer interruption"),
        ):
            probe._finalize_pcoh_k3_integrity(body)
        self.assertIs(body["pcoh_k3_build_only"], original)
        self.assertEqual(original, snapshot)
        self.assertEqual(
            body["pcoh_k3_terminal_integrity"][
                "original_transaction_sha256"
            ],
            snapshot["receipt_sha256"],
        )
        self.assertFalse(
            body["pcoh_k3_terminal_integrity"][
                "terminal_integrity_passed"
            ]
        )
        self.assertEqual(
            body["failed_stage"], "pcoh_k3_terminal_finalizer"
        )
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

        nested_failure, _ = self._pcoh_k3_terminal_body(
            self._pcoh_k3_detached_success()
        )
        original = nested_failure["pcoh_k3_build_only"]
        snapshot = copy.deepcopy(original)
        with mock.patch.object(
            probe,
            "_finalize_pcoh_k3_integrity_registered",
            side_effect=KeyboardInterrupt("toy finalizer interruption"),
        ), mock.patch.object(
            probe,
            "_checksummed",
            side_effect=KeyboardInterrupt("toy integrity receipt interruption"),
        ):
            probe._finalize_pcoh_k3_integrity(nested_failure)
        self.assertIs(nested_failure["pcoh_k3_build_only"], original)
        self.assertEqual(original, snapshot)
        emergency = nested_failure["pcoh_k3_terminal_integrity"]
        self.assertFalse(emergency["terminal_integrity_passed"])
        self.assertEqual(emergency["status"], "stop_loss")
        self.assertEqual(
            emergency["original_transaction_sha256"],
            snapshot["receipt_sha256"],
        )
        self.assertTrue(
            probe._local_receipt_checksum_valid(
                emergency, schema=probe._PCOH_K3_INTEGRITY_SCHEMA
            )
        )
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(original)
        )

    def test_execute_probe_routes_only_explicit_pcoh_mode_to_build_only(
        self,
    ) -> None:
        class ArrayTensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.device = "cpu"
                self.dtype = np.float64

            @property
            def shape(self):
                return self.value.shape

            def detach(self):
                return self

            def cpu(self):
                return self

            def double(self):
                return self

            def contiguous(self):
                return self

            def clone(self):
                return self

            def numpy(self):
                return self.value

        class Model:
            def to(self, **_kwargs):
                return self

        class TorchToACT:
            def __init__(self, _model):
                pass

            def run(self):
                return SimpleNamespace()

        def module(name, **attributes):
            value = ModuleType(name)
            for attribute, item in attributes.items():
                setattr(value, attribute, item)
            return value

        resources = self._pcoh_resources()
        torch_module = module(
            "torch",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                is_initialized=lambda: True,
                max_memory_allocated=lambda: resources[
                    "cuda_peak_allocated_bytes"
                ],
                max_memory_reserved=lambda: resources[
                    "cuda_peak_reserved_bytes"
                ],
                empty_cache=lambda: None,
            ),
            float64=object(),
            set_num_threads=lambda _value: None,
            set_num_interop_threads=lambda _value: None,
        )
        seed = SimpleNamespace(
            lb=ArrayTensor(np.zeros((1, 1), dtype=np.float64)),
            ub=ArrayTensor(np.ones((1, 1), dtype=np.float64)),
        )
        output_bounds = SimpleNamespace(
            lb=ArrayTensor(np.zeros((1, 100), dtype=np.float64)),
            ub=ArrayTensor(np.ones((1, 100), dtype=np.float64)),
        )
        C = np.zeros((1, 100), dtype=np.float64)
        C[0, :2] = (1.0, -1.0)
        thresholds = np.zeros(1, dtype=np.float64)
        assert_layer = SimpleNamespace(params={
            "M": 1,
            "C": ArrayTensor(C),
            "thresholds": ArrayTensor(thresholds),
            "kind": "or",
            "y_true": 0,
        })
        hz = SimpleNamespace(
            c=np.zeros(100, dtype=np.float64),
            Gc=sp.csr_matrix((100, 2), dtype=np.float64),
            Gb=sp.csr_matrix((100, 4), dtype=np.float64),
            Auc=sp.csr_matrix((1, 2), dtype=np.float64),
            Aub=sp.csr_matrix((1, 4), dtype=np.float64),
            Ac=sp.csr_matrix((0, 2), dtype=np.float64),
            Ab=sp.csr_matrix((0, 4), dtype=np.float64),
            ub=np.ones(1, dtype=np.float64),
            b=np.empty(0, dtype=np.float64),
            n_out=100,
            n_cont=2,
            n_bin=4,
            n_ub=1,
            n_eq=0,
        )
        source_build = SimpleNamespace(
            hz=hz,
            input_col_ids=np.empty(0, dtype=np.int64),
            metadata={},
        )
        residual_plan = SimpleNamespace(
            builder_targets=((7, 0, "both"),),
            receipt={"joint_focus_rival_id": 9},
            property_sha256="a" * 64,
        )
        residual_selector = mock.Mock(return_value=residual_plan)
        builder = mock.Mock(return_value=source_build)
        forbidden_pipeline = mock.Mock(
            side_effect=AssertionError("phase-clique pipeline forbidden")
        )
        forbidden_consume = mock.Mock(
            side_effect=AssertionError("solver handoff forbidden")
        )
        forbidden_validate = mock.Mock(
            side_effect=AssertionError("handoff validation forbidden")
        )
        modules = {
            "torch": torch_module,
            "act.back_end.analyze": module(
                "act.back_end.analyze",
                analyze=lambda *_args: (object(), object(), object()),
            ),
            "act.back_end.core": module(
                "act.back_end.core",
                ConSet=lambda: SimpleNamespace(),
                Fact=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            "act.back_end.hybridz_tf.operator_hz": module(
                "act.back_end.hybridz_tf.operator_hz",
                build_operator_hz=builder,
            ),
            "act.back_end.hybridz_tf.operator_phase_clique_pipeline": module(
                "act.back_end.hybridz_tf.operator_phase_clique_pipeline",
                maybe_run_operator_phase_clique_pipeline=forbidden_pipeline,
                consume_operator_phase_clique_pipeline_solver_handoff=(
                    forbidden_consume
                ),
                validate_consumed_operator_phase_clique_solver_build=(
                    forbidden_validate
                ),
            ),
            "act.back_end.transfer_functions": module(
                "act.back_end.transfer_functions",
                set_solver_mode=lambda _value: None,
                set_transfer_function_mode=lambda _value: None,
            ),
            "act.back_end.verifier": module(
                "act.back_end.verifier",
                _ensure_assert_linear_encoding=lambda *_args, **_kwargs: None,
                _get_output_layer_bounds=lambda *_args: output_bounds,
                _get_output_layer_id=lambda _net: 12,
                add_all_input_specs=lambda *_args: None,
                find_entry_layer_id=lambda _net: 0,
                gather_input_spec_layers=lambda _net: (),
                get_assert_layer=lambda _net: assert_layer,
                get_input_ids=lambda _net: (),
                seed_from_input_specs=lambda _layers: seed,
            ),
            "act.front_end.model_synthesis": module(
                "act.front_end.model_synthesis",
                synthesize_models_from_specs=lambda _specs: {0: Model()},
            ),
            "act.front_end.vnnlib_loader.create_specs": module(
                "act.front_end.vnnlib_loader.create_specs",
                create_specs_from_paths=lambda *_args, **_kwargs: object(),
            ),
            "act.pipeline.verification.torch2act": module(
                "act.pipeline.verification.torch2act",
                TorchToACT=TorchToACT,
            ),
            "act.util.device_manager": module(
                "act.util.device_manager",
                initialize_device=lambda *_args: None,
            ),
            "act.back_end.hybridz_tf.property_residual_targets": module(
                "act.back_end.hybridz_tf.property_residual_targets",
                select_property_residual_targets=residual_selector,
            ),
        }
        instance = SimpleNamespace(
            onnx_path=Path("/toy/model.onnx"),
            vnnlib_path=Path("/toy/property.vnnlib"),
            csv_path=Path("/toy/instances.csv"),
        )
        inputs_by_path = {
            instance.onnx_path: self._pcoh_inputs()["onnx"],
            instance.vnnlib_path: self._pcoh_inputs()["vnnlib"],
            instance.csv_path: self._pcoh_inputs()["instances_csv"],
        }
        implementation = self._pcoh_implementation()
        run_pcoh = mock.Mock(return_value=self._pcoh_success_transaction())
        forbidden_transaction = mock.Mock(
            side_effect=AssertionError("diagnostic LP transaction forbidden")
        )
        forbidden_localized = mock.Mock(
            side_effect=AssertionError("localized path forbidden")
        )
        fixed_environment = {
            "HZ_QUERY_WORKERS": "20",
            "HZ_MILP_THREADS": "20",
            "HZ_LP_PREFILTER_THREADS": "20",
            "HZ_LP_PREFILTER_FRACTION": "1.0",
            "HZ_LP_PREFILTER_MAX_SECONDS": "1.0",
        }
        args = self._args(
            candidate_mode=probe._PCOH_K2_BUILD_ONLY_MODE,
            phase_time_limit=25.0,
            residual_time_limit=4.0,
            run_nonce="a" * 64,
            fixed_environment=fixed_environment,
            fixed_environment_sha256="b" * 64,
        )
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(probe, "_select_instance", return_value=instance),
            mock.patch.object(
                probe,
                "_sha256_file",
                side_effect=lambda path: inputs_by_path[Path(path)],
            ),
            mock.patch.object(
                probe, "_implementation_sha256", return_value=implementation
            ),
            mock.patch.object(
                probe, "_capture_resource_peaks", return_value=resources
            ),
            mock.patch.object(
                probe,
                "_glibc_malloc_trim_diagnostic",
                return_value={"status": "toy"},
            ),
            mock.patch.object(probe.gc, "collect", return_value=0),
            mock.patch.object(
                probe, "_run_pcoh_k2_build_only_pipeline", run_pcoh
            ),
            mock.patch.object(
                probe, "_run_phase_transaction", forbidden_transaction
            ),
            mock.patch.object(
                probe, "_run_localized_e2_pipeline", forbidden_localized
            ),
        ):
            receipt = probe._execute_probe(args)
        self.assertEqual(receipt["phase_status"], "built_and_released")
        self.assertTrue(receipt["diagnostic_only"])
        self.assertTrue(receipt["candidate_only"])
        self.assertTrue(receipt["build_only"])
        self.assertEqual(receipt["instance_count"], 1)
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["verdict_authority"])
        self.assertFalse(receipt["ground_truth_loaded"])
        self.assertFalse(receipt["reference_label_used"])
        self.assertFalse(receipt["full_parent_lp_called"])
        self.assertFalse(receipt["full_parent_lp_solver_called"])
        self.assertEqual(
            receipt["pcoh_transaction_sha256"],
            receipt["pcoh_k2_build_only"]["receipt_sha256"],
        )
        nested = receipt["pcoh_k2_build_only"]
        self.assertIsInstance(nested["materialized_tightness_summary"], dict)
        self.assertIsInstance(nested["tightness_gate"], dict)
        self.assertEqual(
            receipt["pcoh_materialized_tightness_summary_sha256"],
            nested["materialized_tightness_summary_sha256"],
        )
        self.assertEqual(
            receipt["pcoh_tightness_gate_sha256"],
            nested["tightness_gate"]["receipt_sha256"],
        )
        self.assertTrue(
            probe._local_receipt_checksum_valid(receipt, schema=probe._SCHEMA)
        )
        run_pcoh.assert_called_once()
        self.assertEqual(
            run_pcoh.call_args.kwargs["phase_time_limit"], 25.0
        )
        builder.assert_called_once()
        self.assertEqual(builder.call_args.kwargs["exact_budget"], 4)
        residual_selector.assert_called_once()
        self.assertEqual(residual_selector.call_args.kwargs["budget"], 4)
        for forbidden in (
            forbidden_pipeline,
            forbidden_consume,
            forbidden_validate,
            forbidden_transaction,
            forbidden_localized,
        ):
            forbidden.assert_not_called()

        k3_transaction, _, _ = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success()
        )
        k3_anchor = self._pcoh_k3_baseline_anchor()
        k3_implementation = self._pcoh_k3_implementation()
        run_k3 = mock.Mock(return_value=k3_transaction)
        forbidden_k2 = mock.Mock(
            side_effect=AssertionError("K2 build-only route forbidden")
        )
        forbidden_phase_k3 = mock.Mock(
            side_effect=AssertionError("phase transaction forbidden")
        )
        forbidden_localized_k3 = mock.Mock(
            side_effect=AssertionError("localized route forbidden")
        )
        builder_k3 = mock.Mock(return_value=source_build)
        residual_selector_k3 = mock.Mock(return_value=residual_plan)
        modules_k3 = dict(modules)
        modules_k3["act.back_end.hybridz_tf.operator_hz"] = module(
            "act.back_end.hybridz_tf.operator_hz",
            build_operator_hz=builder_k3,
        )
        modules_k3[
            "act.back_end.hybridz_tf.property_residual_targets"
        ] = module(
            "act.back_end.hybridz_tf.property_residual_targets",
            select_property_residual_targets=residual_selector_k3,
        )
        k3_args = self._args(
            candidate_mode=probe._PCOH_K3_BUILD_ONLY_MODE,
            phase_time_limit=25.0,
            residual_time_limit=4.0,
            run_nonce="c" * 64,
            fixed_environment=fixed_environment,
            fixed_environment_sha256="d" * 64,
        )
        detached_verifier = (
            "act.back_end.hybridz_tf."
            "operator_phase_conditioned_k3_build_only."
            "verify_detached_phase_conditioned_objective_hull_k3_build_only"
        )
        with (
            mock.patch.dict(sys.modules, modules_k3),
            mock.patch.object(probe, "_select_instance", return_value=instance),
            mock.patch.object(
                probe,
                "_sha256_file",
                side_effect=lambda path: inputs_by_path[Path(path)],
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_implementation_sha256",
                return_value=k3_implementation,
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_fixed_baseline_artifact_anchor",
                return_value=k3_anchor,
            ) as read_baseline,
            mock.patch.object(
                probe, "_capture_resource_peaks", return_value=resources
            ),
            mock.patch.object(
                probe,
                "_glibc_malloc_trim_diagnostic",
                return_value={"status": "toy"},
            ),
            mock.patch.object(probe.gc, "collect", return_value=0),
            mock.patch.object(
                probe, "_run_pcoh_k3_build_only_pipeline", run_k3
            ),
            mock.patch.object(
                probe, "_run_pcoh_k2_build_only_pipeline", forbidden_k2
            ),
            mock.patch.object(
                probe, "_run_phase_transaction", forbidden_phase_k3
            ),
            mock.patch.object(
                probe, "_run_localized_e2_pipeline", forbidden_localized_k3
            ),
            mock.patch(detached_verifier, return_value=True),
        ):
            k3_receipt = probe._execute_probe(k3_args)
        self.assertEqual(k3_receipt["phase_status"], "strong_promotion")
        self.assertTrue(k3_receipt["build_only"])
        self.assertNotIn("diagnostic_lp_called", k3_receipt)
        self.assertNotIn("strict_replay_called", k3_receipt)
        self.assertEqual(k3_receipt["pair_local_lp_actual_calls"], 12)
        self.assertEqual(k3_receipt["conditional_local_lp_actual_calls"], 8)
        self.assertEqual(k3_receipt["total_local_lp_actual_calls"], 20)
        self.assertEqual(k3_receipt["conditional_checker_actual_calls"], 26)
        self.assertEqual(
            k3_receipt["pcoh_k3_transaction_sha256"],
            k3_transaction["receipt_sha256"],
        )
        self.assertEqual(read_baseline.call_count, 2)
        run_k3.assert_called_once()
        self.assertEqual(
            run_k3.call_args.kwargs["phase_time_limit"],
            probe._PCOH_K3_INTERNAL_PHASE_SECONDS,
        )
        self.assertEqual(
            run_k3.call_args.kwargs["baseline_anchor_receipt"],
            k3_anchor,
        )
        forbidden_k2.assert_not_called()
        forbidden_phase_k3.assert_not_called()
        forbidden_localized_k3.assert_not_called()
        self.assertIsNone(
            probe._pcoh_k3_trusted_transaction_anchor(k3_transaction)
        )

        helper_stop = probe._pcoh_k3_stop_loss_receipt(
            stage="verified_literal_selection_max4",
            reason="KeyboardInterrupt:full execute helper stop",
            started=time.monotonic(),
            input_sha256=self._pcoh_inputs(),
            implementation_sha256=k3_implementation,
            stage_resources={
                "entry": resources,
                "unique_full_execute_stop": resources,
            },
            timings={"raw_batch_seconds": 0.125},
            k3_transaction_called=False,
            baseline_anchor_receipt=k3_anchor,
        )
        helper_stop_snapshot = copy.deepcopy(helper_stop)
        run_helper_stop = mock.Mock(return_value=helper_stop)
        helper_stop_args = copy.copy(k3_args)
        helper_stop_args.parent_hard_deadline_monotonic = (
            time.monotonic() + 60.0
        )
        with (
            mock.patch.dict(sys.modules, modules_k3),
            mock.patch.object(probe, "_select_instance", return_value=instance),
            mock.patch.object(
                probe,
                "_sha256_file",
                side_effect=lambda path: inputs_by_path[Path(path)],
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_implementation_sha256",
                return_value=k3_implementation,
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_fixed_baseline_artifact_anchor",
                return_value=k3_anchor,
            ),
            mock.patch.object(
                probe, "_capture_resource_peaks", return_value=resources
            ),
            mock.patch.object(
                probe,
                "_glibc_malloc_trim_diagnostic",
                return_value={"status": "toy"},
            ),
            mock.patch.object(probe.gc, "collect", return_value=0),
            mock.patch.object(
                probe,
                "_run_pcoh_k3_build_only_pipeline",
                run_helper_stop,
            ),
            mock.patch.object(
                probe, "_run_pcoh_k2_build_only_pipeline", forbidden_k2
            ),
            mock.patch.object(
                probe, "_run_phase_transaction", forbidden_phase_k3
            ),
            mock.patch.object(
                probe, "_run_localized_e2_pipeline", forbidden_localized_k3
            ),
            mock.patch(detached_verifier, return_value=True),
        ):
            helper_stop_receipt = probe._execute_probe(helper_stop_args)
        self.assertEqual(helper_stop_receipt["phase_status"], "stop_loss")
        self.assertEqual(
            helper_stop_receipt["failed_stage"],
            helper_stop_snapshot["failed_stage"],
        )
        self.assertEqual(
            helper_stop_receipt["fallback_reason"],
            helper_stop_snapshot["reason"],
        )
        self.assertIs(
            helper_stop_receipt["pcoh_k3_build_only"], helper_stop
        )
        self.assertEqual(helper_stop, helper_stop_snapshot)
        self.assertEqual(
            helper_stop_receipt["pcoh_k3_transaction_sha256"],
            helper_stop_snapshot["receipt_sha256"],
        )
        helper_integrity = helper_stop_receipt[
            "pcoh_k3_terminal_integrity"
        ]
        self.assertTrue(helper_integrity["terminal_integrity_passed"])
        self.assertFalse(helper_integrity["transaction_terminal_candidate"])
        self.assertTrue(helper_integrity["original_transaction_preserved"])
        run_helper_stop.assert_called_once()

        interrupted_transaction, _, _ = self._run_pcoh_k3_direct_mock(
            self._pcoh_k3_detached_success()
        )
        run_interrupted = mock.Mock(return_value=interrupted_transaction)
        interrupted_args = copy.copy(k3_args)
        interrupted_args.parent_hard_deadline_monotonic = (
            time.monotonic() + 60.0
        )
        with (
            mock.patch.dict(sys.modules, modules_k3),
            mock.patch.object(probe, "_select_instance", return_value=instance),
            mock.patch.object(
                probe,
                "_sha256_file",
                side_effect=lambda path: inputs_by_path[Path(path)],
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_implementation_sha256",
                return_value=k3_implementation,
            ),
            mock.patch.object(
                probe,
                "_pcoh_k3_fixed_baseline_artifact_anchor",
                return_value=k3_anchor,
            ),
            mock.patch.object(
                probe, "_capture_resource_peaks", return_value=resources
            ),
            mock.patch.object(
                probe,
                "_glibc_malloc_trim_diagnostic",
                return_value={"status": "toy"},
            ),
            mock.patch.object(probe.gc, "collect", return_value=0),
            mock.patch.object(
                probe,
                "_run_pcoh_k3_build_only_pipeline",
                run_interrupted,
            ),
            mock.patch.object(
                probe, "_run_pcoh_k2_build_only_pipeline", forbidden_k2
            ),
            mock.patch.object(
                probe, "_run_phase_transaction", forbidden_phase_k3
            ),
            mock.patch.object(
                probe, "_run_localized_e2_pipeline", forbidden_localized_k3
            ),
            mock.patch.object(
                probe,
                "_finalize_localized_e2_integrity",
                side_effect=KeyboardInterrupt(
                    "localized finalizer interruption"
                ),
            ),
            mock.patch(detached_verifier, return_value=True),
        ):
            interrupted_receipt = probe._execute_probe(interrupted_args)
        run_interrupted.assert_called_once()
        self.assertIs(
            interrupted_receipt["pcoh_k3_build_only"],
            interrupted_transaction,
        )
        self.assertEqual(interrupted_receipt["phase_status"], "stop_loss")
        self.assertFalse(
            interrupted_receipt["pcoh_k3_terminal_integrity"][
                "terminal_integrity_passed"
            ]
        )
        self.assertIn(
            "shared_worker_deadline_met",
            interrupted_receipt["pcoh_k3_terminal_integrity"][
                "failed_conditions"
            ],
        )
        self.assertTrue(
            probe._local_receipt_checksum_valid(
                interrupted_receipt, schema=probe._SCHEMA
            )
        )
        with probe._PCOH_K3_TRUSTED_TRANSACTION_LOCK:
            self.assertFalse(probe._PCOH_K3_TRUSTED_TRANSACTIONS)

    def test_pcoh_terminal_integrity_binds_digests_and_vetoes_each_boundary(
        self,
    ) -> None:
        accepted = self._pcoh_terminal_body()
        probe._finalize_pcoh_k2_integrity(accepted)
        self.assertEqual(accepted["phase_status"], "built_and_released")
        self.assertEqual(
            accepted["pcoh_transaction_sha256"],
            accepted["pcoh_k2_build_only"]["receipt_sha256"],
        )
        self.assertEqual(
            accepted["pcoh_source_build_preflight_sha256"],
            accepted["pcoh_source_build_preflight"]["receipt_sha256"],
        )
        self.assertEqual(
            accepted["pcoh_materialized_tightness_summary_sha256"],
            accepted["pcoh_k2_build_only"][
                "materialized_tightness_summary_sha256"
            ],
        )
        self.assertEqual(
            accepted["pcoh_tightness_gate_sha256"],
            accepted["pcoh_k2_build_only"]["tightness_gate"][
                "receipt_sha256"
            ],
        )
        self.assertTrue(
            accepted["pcoh_terminal_integrity"]["conditions"][
                "materialized_tightness_strictly_verified"
            ]
        )
        self.assertTrue(
            accepted["pcoh_terminal_integrity"]["conditions"][
                "tightness_gate_strictly_replayed"
            ]
        )
        self.assertTrue(
            probe._local_receipt_checksum_valid(
                accepted["pcoh_terminal_integrity"],
                schema=probe._PCOH_K2_INTEGRITY_SCHEMA,
            )
        )
        cases = (
            ("input", "inputs_unchanged", False),
            ("implementation", "implementation_unchanged", False),
            ("deadline", "shared_worker_deadline_met", False),
            ("rss", "resource_usage.peak_rss_bytes", probe._PCOH_K2_MAX_RSS_BYTES + 1),
            ("solver", "solver_handoff_called", True),
            ("full_lp", "full_parent_lp_called", True),
            (
                "full_lp_solver",
                "full_parent_lp_solver_called",
                True,
            ),
        )
        for name, path, value in cases:
            with self.subTest(name=name):
                rejected = self._pcoh_terminal_body()
                if "." in path:
                    parent, child = path.split(".")
                    rejected[parent][child] = value
                else:
                    rejected[path] = value
                probe._finalize_pcoh_k2_integrity(rejected)
                self.assertEqual(rejected["phase_status"], "stop_loss")
                self.assertTrue(
                    rejected["pcoh_terminal_integrity"]["failed_conditions"]
                )

        for name, mapping_name, key in (
            ("input_sha_changed", "input_sha256_after", "onnx"),
            (
                "implementation_sha_changed",
                "implementation_sha256_after",
                probe._IMPLEMENTATION_RELATIVE_PATHS[0],
            ),
        ):
            with self.subTest(name=name):
                rejected = self._pcoh_terminal_body()
                rejected[mapping_name][key] = "f" * 64
                probe._finalize_pcoh_k2_integrity(rejected)
                self.assertEqual(rejected["phase_status"], "stop_loss")

        malformed = self._pcoh_terminal_body()
        malformed["pcoh_k2_build_only"]["extra"] = "tamper"
        probe._finalize_pcoh_k2_integrity(malformed)
        self.assertEqual(malformed["phase_status"], "stop_loss")
        self.assertTrue(
            probe._pcoh_k2_transaction_receipt_valid(
                malformed["pcoh_k2_build_only"]
            )
        )
        self.assertTrue(
            malformed["pcoh_k2_build_only"][
                "build_only_transaction_called"
            ]
        )
        self.assertIn(
            "original_transaction_receipt_valid",
            malformed["pcoh_terminal_integrity"]["failed_conditions"],
        )

    def test_adaptive_configuration_is_an_exact_fixed_contract(self) -> None:
        fixed = {
            "candidate_mode": "rbs_adaptive_k4",
            "family": "cifar100_medium",
            "iid": 2,
            "wall_timeout": 60.0,
            "phase_time_limit": 30.0,
            "operator_exact_budget": 4,
            "residual_budget": 16,
            "residual_time_limit": 4.0,
            "cpu_threads": 20,
        }
        probe._validate_args(self._args(**fixed))
        for field, forged in (
            ("family", "cifar100_large"),
            ("wall_timeout", np.nextafter(60.0, 0.0)),
            ("phase_time_limit", np.nextafter(30.0, 0.0)),
            ("operator_exact_budget", 5),
            ("residual_budget", 15),
            ("residual_time_limit", np.nextafter(4.0, 0.0)),
            ("cpu_threads", 19),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                probe.PhaseCliqueBuildProbeError,
                "fixed contract mismatch",
            ):
                candidate = dict(fixed)
                candidate[field] = forged
                probe._validate_args(self._args(**candidate))

    def test_adaptive_shared_worker_deadline_reserves_term_and_finalization(self) -> None:
        args = self._args(
            candidate_mode="rbs_adaptive_k4",
            phase_time_limit=30.0,
            residual_budget=16,
            parent_hard_deadline_monotonic=1_060.0,
        )
        self.assertEqual(probe._parent_term_reserve_seconds(60.0), 1.0)
        self.assertEqual(
            probe._RBS_ADAPTIVE_K4_FINALIZATION_RESERVE_SECONDS, 1.0
        )
        self.assertEqual(
            probe._shared_worker_deadline(args, now=1_000.0),
            1_058.0,
        )

    def test_adaptive_spawn_delay_consumes_one_absolute_worker_budget(self) -> None:
        _child, fixed, digest = probe._probe_worker_environment({})
        args = self._args(
            candidate_mode="rbs_adaptive_k4",
            phase_time_limit=30.0,
            residual_budget=16,
        )
        payload = probe._worker_payload(
            args,
            run_nonce="d" * 64,
            parent_hard_deadline_monotonic=1_060.0,
            fixed_environment=fixed,
            fixed_environment_sha256=digest,
        )
        with (
            mock.patch.dict(os.environ, fixed, clear=True),
            mock.patch.object(probe.time, "monotonic", return_value=1_007.5),
        ):
            namespace = probe._namespace_from_worker_payload(payload)
        worker_deadline = probe._shared_worker_deadline(
            namespace, now=1_007.5
        )
        self.assertEqual(namespace.parent_hard_deadline_monotonic, 1_060.0)
        self.assertEqual(worker_deadline, 1_058.0)
        self.assertEqual(worker_deadline - 1_007.5, 50.5)
        self.assertNotEqual(worker_deadline, 1_007.5 + 58.0)

    def test_adaptive_worker_deadline_payload_fails_closed(self) -> None:
        _child, fixed, digest = probe._probe_worker_environment({})
        args = self._args(
            candidate_mode="rbs_adaptive_k4",
            phase_time_limit=30.0,
            residual_budget=16,
        )
        valid = probe._worker_payload(
            args,
            run_nonce="e" * 64,
            parent_hard_deadline_monotonic=1_060.0,
            fixed_environment=fixed,
            fixed_environment_sha256=digest,
        )

        def rechecksum(candidate):
            candidate = dict(candidate)
            candidate.pop("worker_args_sha256", None)
            candidate["worker_args_sha256"] = hashlib.sha256(
                probe._canonical_json(candidate)
            ).hexdigest()
            return candidate

        missing = dict(valid)
        missing.pop("parent_hard_deadline_monotonic")
        tampered = dict(valid)
        tampered["parent_hard_deadline_monotonic"] = 1_059.0
        expired = dict(valid)
        expired["parent_hard_deadline_monotonic"] = 999.0
        expired = rechecksum(expired)
        too_far = dict(valid)
        too_far["parent_hard_deadline_monotonic"] = 1_061.0
        too_far = rechecksum(too_far)

        cases = (
            ("missing", missing, "fields mismatch"),
            ("tampered", tampered, "checksum mismatch"),
            ("expired", expired, "expired or exceeds"),
            ("too_far", too_far, "expired or exceeds"),
        )
        for name, candidate, message in cases:
            with (
                self.subTest(name=name),
                mock.patch.dict(os.environ, fixed, clear=True),
                mock.patch.object(
                    probe.time, "monotonic", return_value=1_000.0
                ),
                self.assertRaisesRegex(
                    probe.PhaseCliqueBuildProbeError, message
                ),
            ):
                probe._namespace_from_worker_payload(candidate)

    def test_adaptive_schedule_is_one_nested_4_of_16_selector_prefix(self) -> None:
        plan = self._adaptive_selector_plan([(3, row) for row in range(16)])
        primary, reservoir, receipt = probe._split_rbs_adaptive_schedule(
            plan,
            primary_budget=4,
            expected_selector_budget=16,
            expected_property_sha256=plan.property_sha256,
            require_all_interval_survivors_processed=True,
        )
        self.assertEqual(primary, tuple((3, row, "both") for row in range(4)))
        self.assertEqual(reservoir, tuple((3, row) for row in range(4, 16)))
        self.assertEqual(receipt["status"], "ready")
        self.assertFalse(receipt["selector_rerun"])
        self.assertEqual(receipt["selector_target_count"], 16)
        self.assertEqual(receipt["primary_builder_targets"], [
            [3, row, "both"] for row in range(4)
        ])
        self.assertEqual(receipt["exact_target_reservoir"], [
            [3, row] for row in range(4, 16)
        ])
        self.assertTrue(receipt["same_layer_only"])
        self.assertTrue(receipt["per_layer_three_per_primary_cap_enforced"])
        self.assertTrue(
            probe._local_receipt_checksum_valid(
                receipt, schema="act.rbs_adaptive_k4_schedule.v1"
            )
        )

        toy = self._adaptive_selector_plan(
            [(5, 0), (6, 0), (5, 1), (6, 1), (6, 2)]
        )
        toy_primary, toy_reserve, toy_receipt = (
            probe._split_rbs_adaptive_schedule(
                toy,
                primary_budget=4,
                expected_selector_budget=5,
                expected_property_sha256=toy.property_sha256,
            )
        )
        self.assertEqual(toy_primary, (
            (5, 0, "both"),
            (6, 0, "both"),
            (5, 1, "both"),
            (6, 1, "both"),
        ))
        self.assertEqual(toy_reserve, ((6, 2),))
        self.assertEqual(toy_receipt["status"], "ready")

    def test_adaptive_schedule_rejects_every_selector_binding_tamper(self) -> None:
        coordinates = [(4, row) for row in range(5)]
        cases = []

        policy = self._adaptive_selector_plan(coordinates)
        policy.receipt["selection_policy"] = "score_only"
        cases.append(("policy", policy, policy.property_sha256))

        property_plan = self._adaptive_selector_plan(coordinates)
        cases.append(("property", property_plan, "2" * 64))

        target_hash = self._adaptive_selector_plan(coordinates)
        target_hash.targets_sha256 = "0" * 64
        cases.append(("target_hash", target_hash, target_hash.property_sha256))

        order = self._adaptive_selector_plan(coordinates)
        order.receipt["schedule"][0], order.receipt["schedule"][1] = (
            order.receipt["schedule"][1],
            order.receipt["schedule"][0],
        )
        cases.append(("order", order, order.property_sha256))

        duplicate = self._adaptive_selector_plan(
            [(4, 0), (4, 1), (4, 1), (4, 2), (4, 3)]
        )
        cases.append(("duplicate", duplicate, duplicate.property_sha256))

        for name, plan, expected_property in cases:
            with self.subTest(name=name), self.assertRaises(
                probe.PhaseCliqueBuildProbeError
            ):
                probe._split_rbs_adaptive_schedule(
                    plan,
                    primary_budget=4,
                    expected_selector_budget=5,
                    expected_property_sha256=expected_property,
                )

    def test_adaptive_schedule_drops_cross_layer_and_enforces_layer_cap(self) -> None:
        cross_layer = self._adaptive_selector_plan(
            [(1, row) for row in range(4)]
            + [(2, row) for row in range(12)]
        )
        _primary, reservoir, receipt = probe._split_rbs_adaptive_schedule(
            cross_layer,
            primary_budget=4,
            expected_selector_budget=16,
        )
        self.assertEqual(reservoir, ())
        self.assertEqual(receipt["status"], "no_same_layer_reserve")
        self.assertEqual(receipt["dropped_cross_layer_count"], 12)

        capped = self._adaptive_selector_plan(
            [(1, 0), (2, 0), (2, 1), (2, 2)]
            + [(1, row) for row in range(1, 13)]
        )
        _primary, reservoir, receipt = probe._split_rbs_adaptive_schedule(
            capped,
            primary_budget=4,
            expected_selector_budget=16,
        )
        self.assertEqual(reservoir, ((1, 1), (1, 2), (1, 3)))
        self.assertEqual(receipt["dropped_per_layer_cap_count"], 9)
        self.assertEqual(
            receipt["dropped_per_layer_cap_schedule"],
            receipt["full_schedule"][7:],
        )
        forged = dict(receipt)
        forged["dropped_per_layer_cap_count"] = 0
        self.assertFalse(
            probe._local_receipt_checksum_valid(
                forged, schema="act.rbs_adaptive_k4_schedule.v1"
            )
        )

    def test_adaptive_property_cube_gates_exact_99_100_75_boundaries(self) -> None:
        passed = self._adaptive_property_cube_receipt()
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["maximum_upper"], 75.0)
        for name, kwargs, condition in (
            ("rows_98", {"rows": 98}, "fixed_property_row_count_99"),
            ("rows_100", {"rows": 100}, "fixed_property_row_count_99"),
            ("outputs_99", {"outputs": 99}, "fixed_output_dimension_100"),
            ("outputs_101", {"outputs": 101}, "fixed_output_dimension_100"),
            (
                "upper_above_75",
                {"maximum": np.nextafter(75.0, np.inf)},
                "worst_cube_upper_at_most_75",
            ),
        ):
            with self.subTest(name=name):
                receipt = self._adaptive_property_cube_receipt(**kwargs)
                self.assertEqual(receipt["status"], "rejected")
                self.assertFalse(receipt["conditions"][condition])

    def test_adaptive_memory_forecast_v2_recomputes_each_stage(self) -> None:
        build, _selector, _schedule = self._adaptive_build_and_schedule()
        receipt = probe._rbs_adaptive_k4_memory_forecast(build.hz)
        self.assertTrue(
            probe._rbs_adaptive_k4_memory_forecast_valid(receipt),
            receipt,
        )
        core = receipt["hz_core_bytes"]
        candidate = receipt["candidate_csr_bytes"]
        self.assertEqual(
            receipt["stage_increment_bytes"],
            {
                "compact_k4_native_model": core + candidate,
                "verified_cut_reconstruction": 2 * core,
                "materializer_private_handoff": 2 * core,
                "native_objective_dual": core + candidate,
            },
        )
        self.assertEqual(
            receipt["static_peak_increment_lower_bound_bytes"],
            max(2 * core, core + candidate),
        )
        for name, path, value in (
            ("v1", ("schema",), "act.rbs_adaptive_k4_memory_forecast.v1"),
            (
                "core_count",
                (
                    "stage_additional_full_hz_core_counts",
                    "materializer_private_handoff",
                ),
                1,
            ),
            ("formula", ("peak_formula",), "2C+S"),
            (
                "static_understatement",
                ("static_peak_increment_lower_bound_bytes",),
                receipt["static_peak_increment_lower_bound_bytes"] - 1,
            ),
        ):
            with self.subTest(name=name):
                forged = copy.deepcopy(receipt)
                target = forged
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(
                    probe._rbs_adaptive_k4_memory_forecast_valid(forged)
                )

    def test_adaptive_pre_gate_is_fail_closed_on_underfill_rbs_resource_or_elapsed(self) -> None:
        build, _selector, schedule = self._adaptive_build_and_schedule()
        passed = self._adaptive_pre_gate(build, schedule)
        self.assertEqual(passed["status"], "passed", passed)

        mutations = []
        underfill = copy.deepcopy(build)
        underfill.metadata["exact_budget_used"] = 3
        mutations.append((
            "underfill",
            underfill,
            self._adaptive_resources(),
            "exact_budget_requested_and_used_4",
        ))
        rbs_receipt = copy.deepcopy(build)
        rbs_receipt.metadata["exact_target_reservoir_receipts"][0][
            "status"
        ] = "shortfall"
        mutations.append((
            "rbs_receipt",
            rbs_receipt,
            self._adaptive_resources(),
            "reservoir_receipts_revalidated",
        ))
        resource = self._adaptive_resources()
        resource["peak_rss_bytes"] += 1
        mutations.append((
            "resource",
            copy.deepcopy(build),
            resource,
            "peak_rss_within_2_5_gib",
        ))
        forecast_resource = self._adaptive_resources()
        forecast = probe._rbs_adaptive_k4_memory_forecast(build.hz)
        forecast_resource["current_rss_bytes"] = (
            probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            - probe._RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
            - forecast["static_peak_increment_lower_bound_bytes"]
            + 1
        )
        mutations.append((
            "static_memory_forecast",
            copy.deepcopy(build),
            forecast_resource,
            "static_k4_memory_lower_bound_has_64_mib_headroom",
        ))
        elapsed = copy.deepcopy(build)
        elapsed.metadata["residual_phase_screen_elapsed_seconds"] = (
            np.nextafter(1.0, np.inf)
        )
        mutations.append((
            "elapsed",
            elapsed,
            self._adaptive_resources(),
            "rbs_elapsed_at_most_1_second",
        ))
        conv_builder = copy.deepcopy(build)
        conv_builder.metadata["layers"][0]["operator_csr_builder"] = (
            "legacy_exact_csr_v1"
        )
        mutations.append((
            "conv_builder",
            conv_builder,
            self._adaptive_resources(),
            "all_19_convs_use_vectorized_exact_csr",
        ))
        traversal_release = copy.deepcopy(build)
        traversal_release.metadata["traversal_cache_release"]["status"] = (
            "not_released"
        )
        mutations.append((
            "traversal_release",
            traversal_release,
            self._adaptive_resources(),
            "traversal_caches_released_before_final_assembly",
        ))
        telemetry = copy.deepcopy(build)
        telemetry.performance_diagnostic.pop("total_wall_seconds")
        mutations.append((
            "telemetry",
            telemetry,
            self._adaptive_resources(),
            "non_authoritative_build_telemetry_complete",
        ))
        for name, candidate, resources, failed_condition in mutations:
            with self.subTest(name=name):
                gate = self._adaptive_pre_gate(
                    candidate, schedule, resources=resources
                )
                self.assertEqual(gate["status"], "rejected")
                self.assertIn(failed_condition, gate["failed_conditions"])

    def test_c88_trim_claim_cannot_override_actual_rss_forecast_gate(
        self,
    ) -> None:
        build, _selector, schedule = self._adaptive_build_and_schedule()
        static_increment = 336_534_824
        native_static_increment = max(
            2 * 113_326_448,
            113_326_448 + 111_604_188,
        )
        self.assertEqual(native_static_increment, 226_652_896)
        current_rss = 2_405_068_800
        peak_rss = 2_592_681_984
        safe_entry = (
            probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            - probe._RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
            - static_increment
        )
        self.assertEqual(safe_entry, 2_280_710_872)
        self.assertEqual(current_rss - safe_entry, 124_357_928)
        native_safe_entry = (
            probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            - probe._RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
            - native_static_increment
        )
        self.assertEqual(
            current_rss - native_safe_entry,
            14_476_000,
        )
        forecast = {
            "schema": "act.rbs_adaptive_k4_memory_forecast.v1",
            "status": "computed",
            "proof_authority": False,
            "verdict_authority": False,
            "hz_core_bytes": 1,
            "candidate_csr_bytes": 1,
            "static_peak_increment_lower_bound_bytes": static_increment,
            "highs_internal_overhead_included": False,
        }
        resources = self._adaptive_resources()
        resources.update({
            "peak_rss_bytes": peak_rss,
            "current_rss_bytes": current_rss,
        })
        # Even a diagnostic receipt claiming more than the observed deficit
        # is irrelevant: the pre-gate accepts only the fresh RSS sample.
        claimed_trim_release = 200_000_000
        self.assertGreater(claimed_trim_release, current_rss - safe_entry)
        with mock.patch.object(
            probe,
            "_rbs_adaptive_k4_memory_forecast",
            return_value=forecast,
        ):
            gate = self._adaptive_pre_gate(
                build, schedule, resources=resources
            )
        self.assertEqual(gate["status"], "rejected")
        self.assertFalse(
            gate["conditions"][
                "static_k4_memory_lower_bound_has_64_mib_headroom"
            ]
        )
        self.assertIn(
            "static_k4_memory_lower_bound_has_64_mib_headroom",
            gate["failed_conditions"],
        )

    def test_adaptive_pre_gate_rejections_never_attempt_the_pipeline(self) -> None:
        for failure in (
            "underfill",
            "rbs_receipt",
            "resource",
            "elapsed",
            "conv_builder",
            "traversal_release",
            "telemetry",
        ):
            with self.subTest(failure=failure):
                receipt, transaction, pipeline, builder = (
                    self._execute_adaptive_pre_gate_rejection(failure)
                )
                experiment = receipt["rbs_adaptive_k4"]
                self.assertEqual(
                    experiment["status"],
                    "rbs_adaptive_k4_build_stop_loss",
                    receipt,
                )
                self.assertFalse(experiment["phase_clique_attempted"])
                self.assertEqual(experiment["pre_gate"]["status"], "rejected")
                release = receipt["phase_input_release"]
                self.assertEqual(
                    release["status"],
                    "released_before_phase_pipeline",
                )
                self.assertFalse(release["proof_authority"])
                self.assertIn("resource_usage_after", release)
                trim = release["allocator_trim"]
                self.assertEqual(
                    trim["schema"],
                    "act.hybridz_glibc_malloc_trim_diagnostic.v1",
                )
                self.assertFalse(trim["gate_authority"])
                transaction.assert_not_called()
                pipeline.assert_not_called()
                builder.assert_called_once()

    def test_adaptive_post_gate_accepts_exact_boundaries_and_rejects_each_tamper(self) -> None:
        def certified_bound(
            value: float,
            *,
            upper_rows: int,
            constraint_nonzeros: int,
        ):
            input_nonzeros = constraint_nonzeros + 2
            block_shapes = {
                "Gc": [2, 2],
                "Gb": [2, 0],
                "Auc": [upper_rows, 2],
                "Aub": [upper_rows, 0],
                "Ac": [0, 2],
                "Ab": [0, 0],
            }
            certificate = {
                "schema": (
                    probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA
                ),
                "status": "verified_upper",
                "route": (
                    probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE
                ),
                "uses_sparse_hstack": False,
                "uses_sparse_vstack": False,
                "assembled_sparse_nnz": 0,
                "input_sparse_nnz": input_nonzeros,
                "block_shapes": copy.deepcopy(block_shapes),
                "upper": value,
                "upper_float64_rounding": (
                    "toward_positive_infinity_from_longdouble_v1"
                ),
            }
            upper_dual_sha256 = "3" * 64
            equality_dual_sha256 = "4" * 64
            proposal_receipt = probe._checksummed({
                "schema": (
                    probe._RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_PROPOSAL_SCHEMA
                ),
                "status": "optimal_dual_candidate",
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "backend": probe._RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_BACKEND,
                "highs_version": "1.14.0",
                "presolve": "on",
                "row_order": "upper_then_equality",
                "candidate_load_mode": (
                    probe._RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
                ),
                "binary_change_coefficient_cap": (
                    probe._RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
                ),
                "candidate_rows": upper_rows,
                "candidate_columns": 2,
                "candidate_nonzeros": constraint_nonzeros,
                "n_continuous": 2,
                "n_binary": 0,
                "n_upper": upper_rows,
                "n_equality": 0,
                "objective_convention": (
                    "highs_minimize_cost_equals_negative_"
                    "max_factor_objective"
                ),
                "maximization_factor_objective_size": 2,
                "maximization_factor_objective_sha256": "1" * 64,
                "solver_cost_sha256": "2" * 64,
                "upper_row_dual_size": upper_rows,
                "equality_row_dual_size": 0,
                "upper_row_dual_sha256": upper_dual_sha256,
                "equality_row_dual_sha256": equality_dual_sha256,
                "solver_minimization_objective_hex": (-value).hex(),
                "pair_solve_calls": 0,
                "objective_solve_calls": 1,
                "native_model_closed_before_return": True,
                "uses_sparse_hstack": False,
                "uses_sparse_vstack": False,
                "used_merged_sparse_frame": False,
            })
            return {
                "status": "certified_diagnostic_upper",
                "proof_authority": False,
                "verdict_authority": False,
                "independently_certified_upper": value,
                "certificate": certificate,
                "certificate_route": {
                    "schema": certificate["schema"],
                    "route": certificate["route"],
                    "uses_sparse_hstack": False,
                    "uses_sparse_vstack": False,
                    "assembled_sparse_nnz": 0,
                    "input_sparse_nnz": input_nonzeros,
                    "recomputed_input_sparse_nnz": input_nonzeros,
                    "block_shapes": copy.deepcopy(block_shapes),
                    "upper_float64_rounding": (
                        "toward_positive_infinity_from_longdouble_v1"
                    ),
                    "upper_outward_float64": value,
                    "candidate_upper_row_dual_sha256": (
                        upper_dual_sha256
                    ),
                    "candidate_equality_row_dual_sha256": (
                        equality_dual_sha256
                    ),
                },
                "objective_dual_proposal_receipt": proposal_receipt,
                "objective_dual_proposal_route": copy.deepcopy(
                    proposal_receipt
                ),
            }

        transaction = {
            "status": "fresh_verified_k4_clique_materialized",
            "materialized": True,
            "identity_preserved": False,
            "certified_edge_count": 6,
            "clique_count": 1,
            "cut_row_count": 1,
            "source_upper_rows": 10,
            "fresh_upper_rows": 11,
            "private_handoff_consumed": True,
            "terminal_handoff_validated": True,
            "public_build_is_solver_build": False,
            "initial_budget_seconds": 30.0,
            "pipeline_seconds": 30.0,
            "transaction_elapsed_seconds": 30.0,
            "candidate_budget_seconds": 12.0,
            "candidate_elapsed_seconds": 12.0,
            "minimum_materializer_reserve_seconds": 18.0,
            "pipeline_receipt_sha256": "a" * 64,
            "candidate_result_status": (
                probe._RBS_ADAPTIVE_K4_COMPACT_STATUS
            ),
            "candidate_telemetry_schema": (
                probe._RBS_ADAPTIVE_K4_COMPACT_TELEMETRY_SCHEMA
            ),
            "candidate_representation": (
                probe._RBS_ADAPTIVE_K4_COMPACT_REPRESENTATION
            ),
            "candidate_cut_hz_emitted": False,
            "candidate_descriptor_sha256": "b" * 64,
            "candidate_progress_available": True,
            "candidate_progress": {
                "schema": probe._RBS_ADAPTIVE_K4_PROGRESS_SCHEMA,
                "status": "complete",
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "model_load_started": True,
                "model_loaded": True,
                "oracle_backend": probe._RBS_ADAPTIVE_K4_ORACLE_BACKEND,
                "oracle_presolve": "on",
                "candidate_load_mode": (
                    probe._RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
                ),
                "binary_change_coefficient_cap": (
                    probe._RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
                ),
                "candidate_rows": 10,
                "candidate_columns": 2,
                "candidate_nonzeros": 20,
                "pair_target_count": 6,
                "pair_attempted_count": 6,
                "pair_completed_count": 6,
                "certified_conflict_count": 6,
                "last_pair_index": 5,
                "terminal_complete": True,
                "candidate_cut_hz_emitted": False,
                "partial_never_authorizes_edge": True,
                "materializer_reached": False,
            },
            "source_shape": {
                "output_dimension": 2,
                "continuous_columns": 2,
                "binary_columns": 0,
                "upper_rows": 10,
                "equality_rows": 0,
                "constraint_nonzeros": 20,
                "generator_nonzeros": 2,
            },
            "source_exact_kept_candidate_nonzeros": 20,
            "fresh_private_shape": {
                "output_dimension": 2,
                "continuous_columns": 2,
                "binary_columns": 0,
                "upper_rows": 11,
                "equality_rows": 0,
                "constraint_nonzeros": 22,
                "generator_nonzeros": 2,
            },
            "candidate_route_summary": {
                "schema": "act.operator_phase_clique_compact_route.v1",
                "result_mode": "compact_exact_descriptor_v1",
                "result_status": probe._RBS_ADAPTIVE_K4_COMPACT_STATUS,
                "telemetry_schema": (
                    probe._RBS_ADAPTIVE_K4_COMPACT_TELEMETRY_SCHEMA
                ),
                "hz_absent": True,
                "oracle_backend": probe._RBS_ADAPTIVE_K4_ORACLE_BACKEND,
                "oracle_presolve": "on",
                "candidate_load_mode": (
                    probe._RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
                ),
                "binary_change_coefficient_cap": (
                    probe._RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
                ),
                "candidate_rows": 10,
                "candidate_columns": 2,
                "candidate_nonzeros": 20,
                "model_builds": 1,
                "solve_calls": 6,
                "base_solve_calls": 0,
                "pair_count": 6,
                "pair_status_counts": {
                    "certified_conflict": 6,
                    "feasible_or_unknown": 0,
                    "infeasible_without_ray": 0,
                    "exact_replay_rejected": 0,
                },
                "completed_pair_count": 6,
                "proof_authority": False,
            },
            "materialization_receipt_sha256": "c" * 64,
            "materializer_route_summary": {
                "schema": (
                    "act.operator_exact_relu_phase_clique_materialization.v2"
                ),
                "receipt_sha256": "c" * 64,
                "public_core_source": "consumed_verified_cut_zero_copy",
                "parent_prefix_core": "strict_readonly_zero_copy_view",
                "parent_prefix_readonly": True,
                "parent_prefix_aliases_public_cut": True,
                "public_core_readonly": True,
                "materializer_full_core_copy_count": 1,
                "private_solver_core": "single_independent_snapshot",
                "public_private_core_no_alias": True,
                "producer_nonempty_seal_verified": True,
                "one_use_snapshot_consumed": True,
                "solver_handoff_one_use": True,
                "solver_handoff_owner_bound": True,
                "solver_handoff_pid_bound": True,
                "solver_handoff_private_core_readonly": True,
            },
            "phase_rss_before_pipeline": {
                "current_rss_bytes": 1024,
                "peak_rss_bytes": 1024,
            },
            "phase_rss_after_handoff": {
                "current_rss_bytes": 2048,
                "peak_rss_bytes": 2048,
            },
            "phase_rss_after_public_release": {
                "current_rss_bytes": 1536,
                "peak_rss_bytes": 2048,
            },
            "phase_rss_after_transaction": {
                "current_rss_bytes": 1536,
                "peak_rss_bytes": 2048,
            },
            "lp_tightness": {
                "status": "compared",
                "independent_lp_call_count": 2,
                "before": certified_bound(
                    10.0,
                    upper_rows=10,
                    constraint_nonzeros=20,
                ),
                "after": certified_bound(
                    9.0,
                    upper_rows=11,
                    constraint_nonzeros=22,
                ),
                "certified_upper_improvement": 1.0,
                "relative_drop": 0.10,
            },
        }
        gate = probe._rbs_adaptive_k4_post_gate(transaction)
        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(gate["promoted"])

        mutations = (
            (
                "zero_baseline",
                ("lp_tightness", "before", "independently_certified_upper"),
                0.0,
                "positive_baseline_upper",
            ),
            ("five_edges", ("certified_edge_count",), 5, "six_certified_edges"),
            ("seven_edges", ("certified_edge_count",), 7, "six_certified_edges"),
            ("no_cut", ("cut_row_count",), 0, "one_fresh_cut"),
            ("two_cuts", ("cut_row_count",), 2, "one_fresh_cut"),
            (
                "candidate_over_12",
                ("candidate_elapsed_seconds",),
                np.nextafter(12.0, np.inf),
                "candidate_elapsed_at_most_12_seconds",
            ),
            (
                "transaction_over_30",
                ("transaction_elapsed_seconds",),
                np.nextafter(30.0, np.inf),
                "whole_transaction_at_most_30_seconds",
            ),
            (
                "pipeline_over_30",
                ("pipeline_seconds",),
                np.nextafter(30.0, np.inf),
                "phase_window_at_most_30_seconds",
            ),
            (
                "stored_absolute_tamper",
                ("lp_tightness", "certified_upper_improvement"),
                np.nextafter(1.0, 0.0),
                "lp_drop_fields_recomputed",
            ),
            (
                "stored_relative_tamper",
                ("lp_tightness", "relative_drop"),
                np.nextafter(0.10, 0.0),
                "lp_drop_fields_recomputed",
            ),
            (
                "presolve_backend_tamper",
                ("candidate_route_summary", "oracle_backend"),
                "highspy_persistent_simplex_dual_ray_v1",
                "presolve_v2_candidate_route",
            ),
            (
                "kept_nnz_tamper",
                ("candidate_route_summary", "candidate_nonzeros"),
                19,
                "split_loader_exact_shape_route",
            ),
            (
                "partial_pair_tamper",
                ("candidate_route_summary", "completed_pair_count"),
                5,
                "six_pair_route_complete",
            ),
            (
                "partial_progress_tamper",
                ("candidate_progress", "pair_completed_count"),
                5,
                "terminal_six_pair_progress_receipt",
            ),
            (
                "progress_authority_tamper",
                ("candidate_progress", "proof_authority"),
                True,
                "terminal_six_pair_progress_receipt",
            ),
            (
                "materializer_copy_tamper",
                (
                    "materializer_route_summary",
                    "materializer_full_core_copy_count",
                ),
                2,
                "unique_copy_materializer_route",
            ),
            (
                "producer_seal_route_tamper",
                (
                    "materializer_route_summary",
                    "producer_nonempty_seal_verified",
                ),
                False,
                "unique_copy_materializer_route",
            ),
            (
                "phase_entry_headroom_tamper",
                ("phase_rss_before_pipeline", "current_rss_bytes"),
                (
                    probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES
                    - probe._RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
                    + 1
                ),
                "phase_entry_has_64_mib_rss_headroom",
            ),
            (
                "phase_peak_tamper",
                ("phase_rss_after_transaction", "peak_rss_bytes"),
                probe._RBS_ADAPTIVE_K4_MAX_RSS_BYTES + 1,
                "phase_rss_samples_within_2_5_gib",
            ),
            (
                "lp_certificate_schema_tamper",
                (
                    "lp_tightness",
                    "before",
                    "certificate_route",
                    "schema",
                ),
                "hz_lp_lagrangian_longdouble_v1",
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "lp_certificate_hstack_tamper",
                (
                    "lp_tightness",
                    "after",
                    "certificate_route",
                    "uses_sparse_hstack",
                ),
                True,
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "lp_certificate_assembled_nnz_tamper",
                (
                    "lp_tightness",
                    "before",
                    "certificate_route",
                    "assembled_sparse_nnz",
                ),
                1,
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "lp_certificate_input_nnz_tamper",
                (
                    "lp_tightness",
                    "after",
                    "certificate_route",
                    "input_sparse_nnz",
                ),
                23,
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "lp_certificate_block_shape_tamper",
                (
                    "lp_tightness",
                    "before",
                    "certificate_route",
                    "block_shapes",
                    "Auc",
                ),
                [9, 2],
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "lp_certificate_nested_receipt_tamper",
                (
                    "lp_tightness",
                    "after",
                    "certificate",
                    "route",
                ),
                "forged_no_stack_route",
                "split_block_lp_certificate_no_stack_route",
            ),
            (
                "objective_proposal_backend_tamper",
                (
                    "lp_tightness",
                    "before",
                    "objective_dual_proposal_receipt",
                    "backend",
                ),
                "legacy_scipy_hstack",
                "native_split_objective_dual_candidate_route",
            ),
            (
                "objective_proposal_hstack_tamper",
                (
                    "lp_tightness",
                    "after",
                    "objective_dual_proposal_route",
                    "uses_sparse_hstack",
                ),
                True,
                "native_split_objective_dual_candidate_route",
            ),
            (
                "objective_proposal_kept_nnz_tamper",
                (
                    "lp_tightness",
                    "before",
                    "objective_dual_proposal_receipt",
                    "candidate_nonzeros",
                ),
                19,
                "native_split_objective_dual_candidate_route",
            ),
            (
                "producer_checker_dual_hash_tamper",
                (
                    "lp_tightness",
                    "after",
                    "certificate_route",
                    "candidate_upper_row_dual_sha256",
                ),
                "5" * 64,
                "native_split_objective_dual_candidate_route",
            ),
        )
        for name, path, forged_value, failed_condition in mutations:
            with self.subTest(name=name):
                forged = copy.deepcopy(transaction)
                target = forged
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = forged_value
                rejected = probe._rbs_adaptive_k4_post_gate(forged)
                self.assertEqual(rejected["status"], "rejected")
                self.assertFalse(rejected["promoted"])
                self.assertIn(
                    failed_condition, rejected["failed_conditions"]
                )

        coherent_objective_tamper = copy.deepcopy(transaction)
        proposal = dict(
            coherent_objective_tamper["lp_tightness"]["after"][
                "objective_dual_proposal_receipt"
            ]
        )
        proposal.pop("receipt_sha256")
        proposal["maximization_factor_objective_sha256"] = "6" * 64
        proposal = probe._checksummed(proposal)
        coherent_objective_tamper["lp_tightness"]["after"][
            "objective_dual_proposal_receipt"
        ] = proposal
        coherent_objective_tamper["lp_tightness"]["after"][
            "objective_dual_proposal_route"
        ] = copy.deepcopy(proposal)
        coherent_gate = probe._rbs_adaptive_k4_post_gate(
            coherent_objective_tamper
        )
        self.assertIn(
            "native_split_objective_dual_candidate_route",
            coherent_gate["failed_conditions"],
        )

        absolute = copy.deepcopy(transaction)
        absolute_after = float(np.nextafter(9.0, np.inf))
        absolute_improvement = 10.0 - absolute_after
        absolute["lp_tightness"]["after"][
            "independently_certified_upper"
        ] = absolute_after
        absolute["lp_tightness"][
            "certified_upper_improvement"
        ] = absolute_improvement
        absolute["lp_tightness"]["relative_drop"] = (
            absolute_improvement / 10.0
        )
        absolute_gate = probe._rbs_adaptive_k4_post_gate(absolute)
        self.assertFalse(
            absolute_gate["conditions"]["absolute_drop_at_least_1"]
        )
        self.assertTrue(
            absolute_gate["conditions"]["lp_drop_fields_recomputed"]
        )

        relative = copy.deepcopy(transaction)
        relative_before = 20.0
        relative_after = float(np.nextafter(18.0, np.inf))
        relative_improvement = relative_before - relative_after
        relative["lp_tightness"]["before"][
            "independently_certified_upper"
        ] = relative_before
        relative["lp_tightness"]["after"][
            "independently_certified_upper"
        ] = relative_after
        relative["lp_tightness"][
            "certified_upper_improvement"
        ] = relative_improvement
        relative["lp_tightness"]["relative_drop"] = (
            relative_improvement / relative_before
        )
        relative_gate = probe._rbs_adaptive_k4_post_gate(relative)
        self.assertTrue(
            relative_gate["conditions"]["absolute_drop_at_least_1"]
        )
        self.assertFalse(
            relative_gate["conditions"][
                "relative_drop_at_least_10_percent"
            ]
        )
        self.assertTrue(
            relative_gate["conditions"]["lp_drop_fields_recomputed"]
        )

    def test_checksum_and_private_json_are_canonical(self) -> None:
        receipt = probe._checksummed({"schema": "toy", "value": [2, 1]})
        expected = dict(receipt)
        digest = expected.pop("receipt_sha256")
        self.assertEqual(
            digest, hashlib.sha256(probe._canonical_json(expected)).hexdigest()
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            fd, identity = probe._new_worker_inode(directory_fd)
            try:
                probe._write_private_worker_json_fd(
                    fd, receipt, expected_identity=identity
                )
                observed = json.loads(os.pread(fd, 10000, 0).decode())
                self.assertEqual(observed, receipt)
            finally:
                os.close(fd)
                os.close(directory_fd)

    def test_transaction_consumes_private_build_and_compares_lp(self) -> None:
        source = _Build(_HZ(upper_rows=2))
        public = _Build(_HZ(upper_rows=3))
        private = _Build(_HZ(upper_rows=3))
        calls = []

        def run_pipeline(build, **kwargs):
            calls.append(("pipeline", build, kwargs["enabled"]))
            return _Result(public)

        def consume(build, result, *, deadline):
            calls.append(("consume", build, result, deadline))
            return private

        lp_builds = []

        def lp_upper(hz, objective, threshold, *, deadline):
            lp_builds.append((hz, tuple(objective), threshold, deadline))
            return {
                "status": "certified_diagnostic_upper",
                "proof_authority": False,
                "independently_certified_upper": (
                    0.75 if hz is source.hz else 0.25
                ),
            }

        diagnostic, consumed = probe._run_phase_transaction(
            source,
            pipeline_kwargs={"enabled": True},
            objective_rows=np.array([[1.0, 0.0], [0.0, 1.0]]),
            thresholds=np.array([0.0, 0.5]),
            deadline=time.monotonic() + 10.0,
            run_pipeline=run_pipeline,
            consume_handoff=consume,
            validate_consumed=lambda result, build: result is not None and build is private,
            lp_upper=lp_upper,
        )
        self.assertIs(consumed, private)
        self.assertEqual([item[0] for item in calls], ["pipeline", "consume"])
        self.assertEqual([item[0] for item in lp_builds], [source.hz, private.hz])
        self.assertTrue(diagnostic["private_handoff_consumed"])
        self.assertFalse(diagnostic["public_build_is_solver_build"])
        self.assertEqual(diagnostic["source_upper_rows"], 2)
        self.assertEqual(diagnostic["fresh_upper_rows"], 3)
        self.assertEqual(diagnostic["clique_count"], 1)
        self.assertEqual(diagnostic["cut_row_count"], 1)
        self.assertEqual(
            diagnostic["lp_tightness"]["certified_upper_improvement"], 0.5
        )
        self.assertFalse(diagnostic["lp_tightness"]["proof_authority"])

    def test_transaction_releases_public_cut_before_diagnostic_lp(self) -> None:
        source = _Build(_HZ(upper_rows=2))
        private = _Build(_HZ(upper_rows=3))
        public_ref = None

        def run_pipeline(_build, **_kwargs):
            nonlocal public_ref
            public = _Build(_HZ(upper_rows=3))
            public_ref = weakref.ref(public)
            return _Result(public)

        def lp_upper(hz, _objective, _threshold, *, deadline):
            self.assertGreater(deadline, time.monotonic())
            self.assertIsNotNone(public_ref)
            self.assertIsNone(public_ref())
            return {
                "status": "certified_diagnostic_upper",
                "proof_authority": False,
                "independently_certified_upper": (
                    1.0 if hz is source.hz else 0.5
                ),
            }

        diagnostic, consumed = probe._run_phase_transaction(
            source,
            pipeline_kwargs={"enabled": True},
            objective_rows=np.eye(2),
            thresholds=np.zeros(2),
            deadline=time.monotonic() + 10.0,
            run_pipeline=run_pipeline,
            consume_handoff=lambda *_args, **_kwargs: private,
            validate_consumed=lambda _result, build: build is private,
            lp_upper=lp_upper,
        )
        self.assertIs(consumed, private)
        self.assertIsNone(public_ref())
        self.assertEqual(
            diagnostic["lp_tightness"]["certified_upper_improvement"],
            0.5,
        )
        self.assertIn("phase_rss_after_public_release", diagnostic)

    def test_real_diagnostic_lp_is_independently_certified(self) -> None:
        receipt = probe._certified_relaxed_upper(
            _native_hz(upper_rows=0),
            np.array([1.0, 0.0]),
            0.0,
            deadline=time.monotonic() + 5.0,
        )
        self.assertEqual(receipt["status"], "certified_diagnostic_upper")
        self.assertAlmostEqual(receipt["solver_relaxation_value"], 1.0)
        self.assertGreaterEqual(receipt["independently_certified_upper"], 1.0)
        self.assertFalse(receipt["proof_authority"])
        route = receipt["certificate_route"]
        self.assertEqual(
            route["schema"],
            probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA,
        )
        self.assertEqual(
            route["route"],
            probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE,
        )
        self.assertFalse(route["uses_sparse_hstack"])
        self.assertFalse(route["uses_sparse_vstack"])
        self.assertEqual(route["assembled_sparse_nnz"], 0)
        self.assertEqual(route["input_sparse_nnz"], 2)
        self.assertEqual(route["recomputed_input_sparse_nnz"], 2)
        self.assertEqual(
            route["block_shapes"],
            {
                "Gc": [2, 2],
                "Gb": [2, 0],
                "Auc": [0, 2],
                "Aub": [0, 0],
                "Ac": [0, 2],
                "Ab": [0, 0],
            },
        )
        proposal = receipt["objective_dual_proposal_receipt"]
        self.assertTrue(
            probe._rbs_adaptive_k4_objective_dual_proposal_valid(
                receipt,
                probe._hz_shape(_native_hz(upper_rows=0)),
                expected_kept_nonzeros=0,
            ),
            receipt,
        )
        self.assertEqual(
            route["candidate_upper_row_dual_sha256"],
            proposal["upper_row_dual_sha256"],
        )
        self.assertEqual(
            route["candidate_equality_row_dual_sha256"],
            proposal["equality_row_dual_sha256"],
        )

    def test_diagnostic_upper_never_downcasts_longdouble_below_exact(self) -> None:
        from fractions import Fraction
        from act.back_end.solver.solver_hz import (
            _hz_longdouble_to_outward_float64_upper,
        )

        exact = Fraction(
            2_438_547_398_293_295_086_106,
            1_000_000_000_000_000_000_000,
        )
        longdouble_upper = np.longdouble(
            "2.438547398293295242208"
        )
        self.assertLess(
            Fraction.from_float(float(longdouble_upper)), exact
        )
        outward = _hz_longdouble_to_outward_float64_upper(
            longdouble_upper
        )
        certificate = {
            "schema": probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA,
            "status": "verified_upper",
            "route": probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE,
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "assembled_sparse_nnz": 0,
            "input_sparse_nnz": 2,
            "upper": outward,
            "upper_float64_rounding": (
                "toward_positive_infinity_from_longdouble_v1"
            ),
            "block_shapes": {
                "Gc": [2, 2],
                "Gb": [2, 0],
                "Auc": [0, 2],
                "Aub": [0, 0],
                "Ac": [0, 2],
                "Ab": [0, 0],
            },
        }
        with mock.patch(
            "act.back_end.solver.solver_hz."
            "_hz_independent_split_block_lp_lagrangian_upper",
            return_value=(longdouble_upper, certificate),
        ):
            receipt = probe._certified_relaxed_upper(
                _native_hz(upper_rows=0),
                np.asarray([1.0, 0.0], dtype=np.float64),
                0.0,
                deadline=time.monotonic() + 5.0,
            )
        self.assertEqual(receipt["status"], "certified_diagnostic_upper")
        materialized = receipt["independently_certified_upper"]
        self.assertEqual(materialized, outward)
        self.assertGreaterEqual(Fraction.from_float(materialized), exact)

    def test_diagnostic_certificate_receives_native_blocks_without_stack(self) -> None:
        hz = _native_hz(upper_rows=1)
        certificate = {
            "schema": probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA,
            "status": "verified_upper",
            "route": probe._RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE,
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "assembled_sparse_nnz": 0,
            "input_sparse_nnz": 2,
            "upper": 1.0,
            "upper_float64_rounding": (
                "toward_positive_infinity_from_longdouble_v1"
            ),
            "block_shapes": {
                "Gc": [2, 2],
                "Gb": [2, 0],
                "Auc": [1, 2],
                "Aub": [1, 0],
                "Ac": [0, 2],
                "Ab": [0, 0],
            },
        }
        with (
            mock.patch.object(
                probe.sp,
                "hstack",
                side_effect=AssertionError("native hstack forbidden"),
            ) as candidate_hstack,
            mock.patch.object(
                probe.sp,
                "vstack",
                side_effect=AssertionError("native vstack forbidden"),
            ) as candidate_vstack,
            mock.patch(
                "act.back_end.solver.solver_hz."
                "_hz_independent_split_block_lp_lagrangian_upper",
                return_value=(np.longdouble(1.0), certificate),
            ) as checker,
        ):
            receipt = probe._certified_relaxed_upper(
                hz,
                np.asarray([1.0, 0.0], dtype=np.float64),
                0.0,
                deadline=time.monotonic() + 5.0,
            )
        candidate_hstack.assert_not_called()
        candidate_vstack.assert_not_called()
        checker.assert_called_once()
        arguments = checker.call_args.kwargs
        for name in ("Gc", "Gb", "Auc", "Aub", "Ac", "Ab"):
            self.assertIs(arguments[name], getattr(hz, name))
        self.assertEqual(receipt["status"], "certified_diagnostic_upper")
        self.assertEqual(receipt["certificate_route"]["assembled_sparse_nnz"], 0)
        self.assertTrue(
            receipt["objective_dual_proposal_receipt"][
                "native_model_closed_before_return"
            ]
        )

    def _localized_transaction(
        self,
        *,
        edge: bool,
        after_upper: float = 9.4,
        exact_valid: bool = True,
        checksum_valid: bool = True,
        literal_count: int = 2,
        peak_rss_bytes: int = 1024,
        cuda_allocated_bytes: int = 2048,
        before_status: str = "certified_diagnostic_upper",
        before_upper: float = 10.0,
    ):
        source = _Build(_HZ(upper_rows=2))
        selection = SimpleNamespace(
            selection_digest="1" * 64,
            property_digest="2" * 64,
        )
        certificate = SimpleNamespace(certificate_sha256="3" * 64)
        candidate = SimpleNamespace(
            status=(
                "certified_localized_phase_edge"
                if edge
                else "no_certified_localized_phase_edge"
            ),
            reason="toy",
            enabled=True,
            edge_accepted=edge,
            parent_unchanged=True,
            proof_authority=False,
            producer_nonempty_seal_verified=True,
            literals=("left", "right")[:literal_count],
            certificate=certificate if edge else None,
            localized_result=None,
            localized_result_sha256="4" * 64 if edge else None,
            result_sha256="5" * 64,
            build_binding_sha256="6" * 64,
            parent_semantic_digest="7" * 64,
            terminal_parent_semantic_digest="7" * 64,
            operator_row_tag_digest="8" * 64,
            terminal_operator_row_tag_digest="8" * 64,
            selection_digest=selection.selection_digest,
            subset_binding_digest="9" * 64,
            focused_property_digest=selection.property_digest,
            ordered_source_frame_sha256="a" * 64,
            source_modes_sha256="b" * 64,
        )
        calls = []

        def cut(parent, literals):
            calls.append(("cut", parent, literals))
            return _HZ(upper_rows=parent.n_ub + 1)

        def lp(hz, _row, _threshold, *, deadline):
            calls.append(("lp", hz, deadline))
            return {
                "status": (
                    before_status if hz is source.hz else "certified_diagnostic_upper"
                ),
                "proof_authority": False,
                "independently_certified_upper": (
                    before_upper if hz is source.hz else after_upper
                ),
            }

        receipt = probe._run_localized_e2_transaction(
            source,
            (SimpleNamespace(),),
            selection,
            focused_encoded_row=0,
            objective_rows=np.array([[1.0, 0.0]], dtype=np.float64),
            thresholds=np.array([0.0], dtype=np.float64),
            deadline=time.monotonic() + 20.0,
            run_candidate=lambda *_args, **_kwargs: candidate,
            copy_pair_cut=cut,
            lp_upper=lp,
            live_seals=lambda _hz: ("7" * 64, "8" * 64),
            validate_adapter_checksum=lambda _candidate: checksum_valid,
            validate_exact_candidate=lambda *_args, **_kwargs: exact_valid,
            validate_private_cut=lambda *_args: True,
            resource_peaks=lambda: {
                "peak_rss_bytes": peak_rss_bytes,
                "cuda_initialized": True,
                "cuda_peak_allocated_bytes": cuda_allocated_bytes,
                "cuda_peak_reserved_bytes": 4096,
            },
        )
        return receipt, source, calls

    def test_localized_no_edge_skips_cut_and_lp(self) -> None:
        receipt, _source, calls = self._localized_transaction(edge=False)
        self.assertEqual(receipt["status"], "stop_loss_no_exact_edge")
        self.assertFalse(receipt["diagnostic_cut"]["attempted"])
        self.assertFalse(receipt["lp_attempted"])
        self.assertEqual(calls, [])
        self.assertEqual(receipt["adapter_call_count"], 1)

    def test_localized_zero_or_one_literal_skips_cut_and_lp(self) -> None:
        for literal_count in (0, 1):
            with self.subTest(literal_count=literal_count):
                receipt, _source, calls = self._localized_transaction(
                    edge=False, literal_count=literal_count
                )
                self.assertEqual(
                    receipt["status"], "stop_loss_insufficient_literals"
                )
                self.assertFalse(receipt["lp_attempted"])
                self.assertEqual(calls, [])

    def test_localized_fixed_relative_drop_gate_rejects_four_percent(self) -> None:
        receipt, _source, calls = self._localized_transaction(
            edge=True, after_upper=9.6
        )
        self.assertAlmostEqual(receipt["lp_tightness"]["relative_drop"], 0.04)
        self.assertFalse(receipt["promotion_gate"]["promoted"])
        self.assertEqual([item[0] for item in calls], ["cut", "lp", "lp"])

    def test_localized_inconclusive_before_upper_skips_after_lp(self) -> None:
        receipt, _source, calls = self._localized_transaction(
            edge=True, before_status="lp_inconclusive"
        )
        self.assertEqual(receipt["status"], "localized_e2_promotion_rejected")
        self.assertEqual([item[0] for item in calls], ["cut", "lp"])
        self.assertEqual(receipt["lp_tightness"]["after"]["status"], "not_run")

    def test_localized_fixed_relative_drop_gate_accepts_six_percent_privately(self) -> None:
        receipt, source, calls = self._localized_transaction(
            edge=True, after_upper=9.4
        )
        self.assertAlmostEqual(receipt["lp_tightness"]["relative_drop"], 0.06)
        self.assertTrue(receipt["promotion_gate"]["promoted"])
        self.assertTrue(receipt["live_parent_unchanged"])
        self.assertIs(calls[0][1], source.hz)
        self.assertEqual(source.hz.n_ub, 2)
        self.assertEqual(receipt["resource_usage"]["peak_rss_bytes"], 1024)
        self.assertTrue(
            receipt["promotion_gate"]["conditions"]["peak_rss_within_cap"]
        )

    def test_localized_resource_peak_is_numeric_and_blocks_promotion(self) -> None:
        receipt, _source, calls = self._localized_transaction(
            edge=True,
            peak_rss_bytes=probe._LOCALIZED_E2_MAX_RSS_BYTES + 1,
        )
        self.assertEqual(receipt["status"], "stop_loss_resource")
        self.assertIsInstance(receipt["resource_usage"]["peak_rss_bytes"], int)
        self.assertFalse(receipt["promotion_gate"]["promoted"])
        self.assertEqual(calls, [])

    def test_private_pair_cut_validator_rejects_row_prefix_and_alias_before_lp(self) -> None:
        from act.back_end.hybridz_tf.property_phase_conflict_clique import (
            PhaseLiteral,
            _copy_parent_with_clique_cut,
        )
        from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
            _k4_corner_build,
        )
        from act.back_end.solver.solver_hz import SparseHZono

        build = _k4_corner_build()
        parent_digest, tag_digest = probe._localized_e2_live_seals(build.hz)
        literals = tuple(
            PhaseLiteral(int(stable_id), phase, "d" * 64)
            for stable_id, phase in zip(build.hz.bcol_ids[:2], (1, -1))
        )

        def clone(value, **changes):
            fields = {
                "c": value.c.copy(),
                "Gc": value.Gc.copy(),
                "Gb": value.Gb.copy(),
                "Ac": value.Ac.copy(),
                "Ab": value.Ab.copy(),
                "b": value.b.copy(),
                "Auc": value.Auc.copy(),
                "Aub": value.Aub.copy(),
                "ub": value.ub.copy(),
                "col_ids": value.col_ids.copy(),
                "bcol_ids": value.bcol_ids.copy(),
            }
            fields.update(changes)
            return SparseHZono(**fields)

        valid = _copy_parent_with_clique_cut(build.hz, literals)
        self.assertTrue(
            probe._private_localized_pair_cut_valid(build.hz, valid, literals)
        )

        wrong_row_matrix = valid.Aub.copy()
        wrong_row_matrix.data[-1] = 0.5
        wrong_row = clone(valid, Aub=wrong_row_matrix)
        wrong_prefix_matrix = valid.Aub.copy()
        wrong_prefix_matrix.data[0] = np.nextafter(
            wrong_prefix_matrix.data[0], np.inf
        )
        wrong_prefix = clone(valid, Aub=wrong_prefix_matrix)
        aliased = clone(valid, c=build.hz.c)
        selection = SimpleNamespace(
            selection_digest="1" * 64,
            property_digest="2" * 64,
        )
        candidate = SimpleNamespace(
            status="certified_localized_phase_edge",
            reason="toy",
            enabled=True,
            edge_accepted=True,
            parent_unchanged=True,
            proof_authority=False,
            producer_nonempty_seal_verified=True,
            literals=literals,
            certificate=SimpleNamespace(certificate_sha256="3" * 64),
            localized_result=None,
            localized_result_sha256="4" * 64,
            result_sha256="5" * 64,
            build_binding_sha256="6" * 64,
            parent_semantic_digest=parent_digest,
            terminal_parent_semantic_digest=parent_digest,
            operator_row_tag_digest=tag_digest,
            terminal_operator_row_tag_digest=tag_digest,
            selection_digest=selection.selection_digest,
            subset_binding_digest="9" * 64,
            focused_property_digest=selection.property_digest,
            ordered_source_frame_sha256="a" * 64,
            source_modes_sha256="b" * 64,
        )

        for name, forged in (
            ("row", wrong_row),
            ("prefix", wrong_prefix),
            ("alias", aliased),
        ):
            with self.subTest(name=name):
                lp_calls = []
                receipt = probe._run_localized_e2_transaction(
                    build,
                    (SimpleNamespace(),),
                    selection,
                    focused_encoded_row=0,
                    objective_rows=np.zeros((1, build.hz.n_out)),
                    thresholds=np.zeros(1),
                    deadline=time.monotonic() + 20.0,
                    run_candidate=lambda *_args, **_kwargs: candidate,
                    copy_pair_cut=lambda *_args, forged=forged: forged,
                    lp_upper=lambda *_args, **_kwargs: lp_calls.append(True),
                    validate_adapter_checksum=lambda _candidate: True,
                    validate_exact_candidate=lambda *_args, **_kwargs: True,
                    resource_peaks=lambda: {
                        "peak_rss_bytes": 1024,
                        "cuda_initialized": True,
                        "cuda_peak_allocated_bytes": 2048,
                        "cuda_peak_reserved_bytes": 4096,
                    },
                )
                self.assertEqual(
                    receipt["status"], "stop_loss_private_cut_rejected"
                )
                self.assertFalse(
                    receipt["diagnostic_cut"]["structurally_validated"]
                )
                self.assertFalse(receipt["lp_attempted"])
                self.assertEqual(lp_calls, [])

    def test_exact_candidate_verifier_rederives_bindings_caps_and_pair(self) -> None:
        from act.back_end.hybridz_tf import (
            localized_phase_conflict_oracle as localized_module,
        )
        from act.back_end.hybridz_tf import (
            operator_localized_phase_edge_candidate as adapter_module,
        )
        from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
            run_operator_exact_relu_phase_cliques_candidate,
        )
        from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
            derive_operator_exact_relu_property_phase_literals,
        )
        from act.back_end.hybridz_tf.test_operator_exact_relu_phase_cliques import (
            _corner_build,
        )
        from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
            _rivals,
        )

        build = _corner_build(
            bias=-1.5, issue_constructive_nonempty_seal=True
        )
        rivals = _rivals()
        selection = derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
        candidate = adapter_module.run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
            enabled=True,
        )
        self.assertTrue(
            probe._verify_localized_e2_exact_candidate(
                build,
                candidate,
                selection,
                deadline=time.monotonic() + 20.0,
                candidate_kwargs={},
            )
        )

        def reseal(value):
            blank = replace(value, result_sha256="")
            return replace(
                blank,
                result_sha256=adapter_module._sha256(
                    adapter_module._result_payload(
                        blank, include_digest=False
                    )
                ),
            )

        reordered = reseal(
            replace(candidate, literals=tuple(reversed(candidate.literals)))
        )
        forged_caps = reseal(
            replace(
                candidate,
                caps=replace(
                    candidate.caps,
                    localized_max_selected_nnz=(
                        candidate.caps.localized_max_selected_nnz - 1
                    ),
                ),
            )
        )
        forged_build = reseal(
            replace(candidate, build_binding_sha256="0" * 64)
        )
        k4 = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
        )
        other_certificate = k4.certificates[1]
        other_localized_blank = replace(
            candidate.localized_result,
            certificate=other_certificate,
            result_sha256="",
        )
        other_localized = replace(
            other_localized_blank,
            result_sha256=localized_module._sha256(
                localized_module._result_payload(
                    other_localized_blank, include_digest=False
                )
            ),
        )
        other_valid_same_parent_certificate = reseal(
            replace(
                candidate,
                certificate=other_certificate,
                localized_result=other_localized,
                localized_result_sha256=other_localized.result_sha256,
            )
        )
        for name, forged in (
            ("reordered_pair", reordered),
            ("forged_caps", forged_caps),
            ("forged_build_binding", forged_build),
            ("other_same_parent_certificate", other_valid_same_parent_certificate),
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    probe._verify_localized_e2_exact_candidate(
                        build,
                        forged,
                        selection,
                        deadline=time.monotonic() + 20.0,
                        candidate_kwargs={},
                    )
                )

    def test_localized_pipeline_reuses_raw_focus_literal_chain_and_fails_closed(self) -> None:
        from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
            _k4_corner_build,
            _rivals,
        )

        build = _k4_corner_build()
        rivals = _rivals()
        batch = SimpleNamespace(
            rivals=rivals,
            batch_sha256="a" * 64,
            live_assert_sha256="b" * 64,
        )
        hardness = SimpleNamespace(
            vector_digest="c" * 64,
            full_property_digest="d" * 64,
        )
        focused = SimpleNamespace(
            rivals=(rivals[0],),
            focused_subset_digest="e" * 64,
        )
        raw_token = object()
        transaction_receipt = {
            "status": "stop_loss_no_exact_edge",
            "reason": "toy",
            "timings": {},
        }
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(
                is_initialized=lambda: True,
                max_memory_allocated=lambda: 2048,
                max_memory_reserved=lambda: 4096,
            )
        )
        raw_module = "act.back_end.hybridz_tf.raw_vnnlib_rival_adapter"
        focus_module = "act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge"
        with (
            mock.patch(
                raw_module + ".issue_raw_vnnlib_top1_candidate",
                return_value=raw_token,
            ) as issue,
            mock.patch(
                raw_module + ".consume_raw_vnnlib_top1_candidate",
                return_value=batch,
            ) as consume,
            mock.patch(
                raw_module + ".validate_consumed_raw_vnnlib_rival_batch",
                return_value=True,
            ) as validate_batch,
            mock.patch(
                focus_module + ".issue_raw_rival_exact_hardness_receipt",
                return_value=hardness,
            ) as issue_hardness,
            mock.patch(
                focus_module + ".select_raw_focused_rivals",
                return_value=focused,
            ) as select_focus,
            mock.patch(
                focus_module + ".verify_raw_rival_exact_hardness_receipt",
                return_value=True,
            ) as verify_hardness,
            mock.patch(
                focus_module + ".verify_raw_focused_rival_selection",
                return_value=True,
            ) as verify_focus,
            mock.patch.object(
                probe,
                "_run_localized_e2_transaction",
                return_value=transaction_receipt,
            ) as transaction,
        ):
            deadline = time.monotonic() + 30.0
            result = probe._run_localized_e2_pipeline(
                build,
                vnnlib_path="/toy/property.vnnlib",
                expected_vnnlib_sha256="f" * 64,
                live_assert_params={},
                output_lower=np.zeros((1, build.hz.n_out)),
                output_upper=np.ones((1, build.hz.n_out)),
                residual_selector_receipt={"joint_focus_rival_id": 0},
                residual_selector_property_sha256="1" * 64,
                objective_rows=np.zeros((1, build.hz.n_out)),
                thresholds=np.zeros(1),
                deadline=deadline,
                phase_time_limit=5.0,
                overall_started=time.monotonic(),
                torch_module=torch_module,
            )
            self.assertIsNotNone(result["raw_audit"], result)
            self.assertEqual(result["raw_audit"]["full_batch_sha256"], "a" * 64)
            issue.assert_called_once()
            consume.assert_called_once()
            self.assertIs(consume.call_args.args[0], raw_token)
            validate_batch.assert_called_once_with(batch)
            issue_hardness.assert_called_once()
            select_focus.assert_called_once()
            verify_hardness.assert_called_once()
            verify_focus.assert_called_once()
            transaction.assert_called_once()
            candidate_deadline = issue.call_args.kwargs["deadline"]
            self.assertEqual(
                consume.call_args.kwargs["deadline"], candidate_deadline
            )
            self.assertEqual(
                issue_hardness.call_args.kwargs["deadline"], candidate_deadline
            )
            self.assertEqual(
                select_focus.call_args.kwargs["deadline"], candidate_deadline
            )
            self.assertLessEqual(candidate_deadline, deadline - 10.0)
            candidate_options = transaction.call_args.kwargs["candidate_kwargs"]
            self.assertEqual(candidate_options["selection_max_rivals"], 1)
            self.assertEqual(
                candidate_options["localized_max_source_terms"],
                candidate_options["max_source_terms"],
            )

            transaction.reset_mock()
            verify_focus.return_value = False
            rejected = probe._run_localized_e2_pipeline(
                build,
                vnnlib_path="/toy/property.vnnlib",
                expected_vnnlib_sha256="f" * 64,
                live_assert_params={},
                output_lower=np.zeros((1, build.hz.n_out)),
                output_upper=np.ones((1, build.hz.n_out)),
                residual_selector_receipt={"joint_focus_rival_id": 0},
                residual_selector_property_sha256="1" * 64,
                objective_rows=np.zeros((1, build.hz.n_out)),
                thresholds=np.zeros(1),
                deadline=time.monotonic() + 30.0,
                phase_time_limit=5.0,
                overall_started=time.monotonic(),
                torch_module=torch_module,
            )
            self.assertEqual(rejected["status"], "stop_loss_preflight_rejected")
            transaction.assert_not_called()

    def test_localized_tamper_or_noncertified_candidate_stops_before_cut(self) -> None:
        for changes in (
            {"checksum_valid": False},
            {"exact_valid": False},
        ):
            with self.subTest(changes=changes):
                receipt, _source, calls = self._localized_transaction(
                    edge=True, **changes
                )
                self.assertEqual(receipt["status"], "stop_loss_receipt_rejected")
                self.assertFalse(receipt["diagnostic_cut"]["attempted"])
                self.assertFalse(receipt["lp_attempted"])
                self.assertEqual(calls, [])

    def test_nonmaterialized_transaction_skips_lp(self) -> None:
        source = _Build(_HZ(upper_rows=2))

        class Fallback(_Result):
            def __init__(self):
                self.build = source
                self.status = "fallback_no_verified_k4"
                self.materialized = False
                self.identity_preserved = True
                self.receipt = {
                    "receipt_sha256": "b" * 64,
                    "clique_count": 0,
                    "cut_row_count": 0,
                    "timings": {},
                }

        def forbidden_lp(*_args, **_kwargs):
            raise AssertionError("LP must be skipped on fallback")

        diagnostic, private = probe._run_phase_transaction(
            source,
            pipeline_kwargs={},
            objective_rows=np.eye(2),
            thresholds=np.zeros(2),
            deadline=time.monotonic() + 10.0,
            run_pipeline=lambda *_args, **_kwargs: Fallback(),
            consume_handoff=lambda *_args, **_kwargs: _Build(_HZ(upper_rows=2)),
            validate_consumed=lambda *_args: True,
            lp_upper=forbidden_lp,
        )
        self.assertIsNotNone(private)
        self.assertEqual(diagnostic["lp_tightness"]["status"], "not_materialized")

    def test_terminal_private_handoff_validation_is_mandatory(self) -> None:
        source = _Build(_HZ(upper_rows=2))
        public = _Build(_HZ(upper_rows=3))
        with self.assertRaisesRegex(
            probe.PhaseCliqueBuildProbeError, "terminal handoff"
        ):
            probe._run_phase_transaction(
                source,
                pipeline_kwargs={"enabled": True},
                objective_rows=np.eye(2),
                thresholds=np.zeros(2),
                deadline=time.monotonic() + 10.0,
                run_pipeline=lambda *_args, **_kwargs: _Result(public),
                consume_handoff=lambda *_args, **_kwargs: _Build(_HZ(upper_rows=3)),
                validate_consumed=lambda *_args: False,
            )

    def test_source_has_no_verdict_solver_import_or_call(self) -> None:
        source = Path(probe.__file__).read_text(encoding="utf-8")
        forbidden = "from act.back_end.solver.solver_hz import hz_" + "objbound_decide"
        self.assertNotIn(forbidden, source)
        self.assertNotIn("hz_" + "objbound_decide(", source)
        self.assertNotIn("verify_once(", source)
        self.assertNotIn("load_" + "manifest", source)
        self.assertNotIn("reference_" + "diagnostic_label", source)

    def test_old_self_forged_pipe_cli_cannot_enter_worker(self) -> None:
        artifacts = probe._ARTIFACT_ROOT
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            output = Path(directory) / "forged.json"
            auth_read, auth_write = os.pipe()
            payload_read, payload_write = os.pipe()
            os.write(auth_write, b"1" * 64)
            os.write(payload_write, b'{"forged":true}')
            os.close(auth_write)
            os.close(payload_write)
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "act.pipeline.verification.hybridz_phase_clique_build_probe",
                        "--_worker-auth-fd",
                        str(auth_read),
                        "--_worker-payload-fd",
                        str(payload_read),
                        "--output",
                        str(output),
                    ],
                    cwd=probe._REPO_ROOT,
                    pass_fds=(auth_read, payload_read),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10.0,
                )
            finally:
                os.close(auth_read)
                os.close(payload_read)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--_worker-" + "auth-fd", source)
        self.assertNotIn("--_worker-" + "payload-fd", source)
        self.assertNotIn("_internal_worker_" + "fds", source)

    def test_hard_stop_issues_kill_before_wall_limit(self) -> None:
        class Child:
            pid = 41
            exitcode = None

            def __init__(self):
                self.now = 0.0
                self.calls = 0
                self.killed = False

            def join(self, timeout=None):
                self.calls += 1
                self.now += float(timeout or 0.0)
                if self.killed:
                    self.exitcode = -9

            def terminate(self):
                pass

            def kill(self):
                self.killed = True

        child = Child()
        signals = []
        result = probe._bounded_wait(
            child,
            started=0.0,
            wall_timeout=2.0,
            clock=lambda: child.now,
            signal_group=lambda pid, sig: (
                signals.append((pid, sig)),
                setattr(child, "killed", sig == 9 or child.killed),
            ),
        )
        self.assertTrue(result["timed_out"])
        self.assertLessEqual(result["kill_issued_seconds"], 2.0)
        self.assertEqual([signal for _, signal in signals], [15, 9])

    def test_instance_resolution_never_loads_reference_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "cifar100_2024"
            (category / "onnx").mkdir(parents=True)
            (category / "vnnlib").mkdir()
            onnx = category / "onnx" / "CIFAR100_resnet_medium.onnx"
            vnnlib = category / "vnnlib" / "CIFAR100_resnet_medium_prop_toy.vnnlib"
            onnx.write_bytes(b"onnx")
            vnnlib.write_text("; toy\n")
            (category / "instances.csv").write_text(
                "unused\nunused\n"
                "onnx/CIFAR100_resnet_medium.onnx,"
                "vnnlib/CIFAR100_resnet_medium_prop_toy.vnnlib,60\n"
            )
            with mock.patch(
                "act.pipeline.verification.hybridz_largecls_gate.load_manifest",
                side_effect=AssertionError("reference manifest accessed"),
            ):
                selected = probe._select_instance(root, "cifar100_medium", 2)
            self.assertEqual(selected.iid, 2)
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertNotIn("reference_diagnostic_label", source)
        self.assertNotIn("load_manifest", source)

    def test_output_is_new_artifacts_json_and_receipt_binds_nonce(self) -> None:
        artifacts = probe._ARTIFACT_ROOT
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            parent = Path(directory)
            protected = parent / "input.onnx"
            protected.write_bytes(b"x")
            target = parent / "new.json"
            slot = probe._validate_new_output_path(
                target, protected_paths=(protected,)
            )
            self.assertEqual(slot.display_path, target.resolve())
            probe._close_output_slot(slot)
            target.write_text("old")
            with self.assertRaises(probe.PhaseCliqueBuildProbeError):
                probe._validate_new_output_path(target, protected_paths=(protected,))
            outside = Path(directory).parent.parent / "outside.json"
            with self.assertRaises(probe.PhaseCliqueBuildProbeError):
                probe._validate_new_output_path(outside, protected_paths=(protected,))

            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            receipt_fd, receipt_identity = probe._new_worker_inode(parent_fd)
            receipt = probe._checksummed(
                {"schema": probe._SCHEMA, "run_nonce": "a" * 64, "x": 1.0}
            )
            try:
                probe._write_private_worker_json_fd(
                    receipt_fd, receipt, expected_identity=receipt_identity
                )
                probe._validate_worker_receipt_fd(
                    receipt_fd,
                    run_nonce="a" * 64,
                    expected_identity=receipt_identity,
                )
                with self.assertRaises(probe.PhaseCliqueBuildProbeError):
                    probe._validate_worker_receipt_fd(
                        receipt_fd,
                        run_nonce="b" * 64,
                        expected_identity=receipt_identity,
                    )
            finally:
                os.close(receipt_fd)
                os.close(parent_fd)

    def test_fixed_environment_removes_ambient_hz_and_uses_gate_threads(self) -> None:
        child, fixed, digest = probe._probe_worker_environment(
            {"PATH": "/bin", "HZ_MILP_THREADS": "999", "HZ_EVIL": "1"}
        )
        self.assertEqual(child["PATH"], "/bin")
        self.assertNotIn("HZ_EVIL", child)
        self.assertEqual(fixed["HZ_QUERY_WORKERS"], "4")
        self.assertEqual(fixed["HZ_MILP_THREADS"], "5")
        self.assertEqual(fixed["HZ_LP_PREFILTER_THREADS"], "5")
        self.assertEqual(
            digest, hashlib.sha256(probe._canonical_json(fixed)).hexdigest()
        )

    def test_worker_recomputes_canonical_environment_not_payload_hash(self) -> None:
        _child, canonical, _digest = probe._probe_worker_environment({})
        forged = dict(canonical)
        forged["HZ_QUERY_WORKERS"] = "999"
        forged["HZ_MILP_THREADS"] = "888"
        payload = {
            "benchmark_root": "/tmp",
            "family": "cifar100_medium",
            "iid": 2,
            "candidate_mode": "k4",
            "wall_timeout": 60.0,
            "phase_time_limit": 20.0,
            "operator_exact_budget": 4,
            "residual_budget": 4,
            "residual_time_limit": 4.0,
            "cpu_threads": 20,
            "run_nonce": "a" * 64,
            "parent_hard_deadline_monotonic": time.monotonic() + 60.0,
            "fixed_environment": forged,
            "fixed_environment_sha256": hashlib.sha256(
                probe._canonical_json(forged)
            ).hexdigest(),
        }
        payload["worker_args_sha256"] = hashlib.sha256(
            probe._canonical_json(payload)
        ).hexdigest()
        with mock.patch.dict(os.environ, forged, clear=True), self.assertRaises(
            probe.PhaseCliqueBuildProbeError
        ):
            probe._namespace_from_worker_payload(payload)

    def test_worker_payload_rejects_missing_or_forged_candidate_mode(self) -> None:
        _child, fixed, digest = probe._probe_worker_environment({})
        payload = probe._worker_payload(
            self._args(),
            run_nonce="c" * 64,
            parent_hard_deadline_monotonic=time.monotonic() + 60.0,
            fixed_environment=fixed,
            fixed_environment_sha256=digest,
        )
        missing = dict(payload)
        missing.pop("candidate_mode")
        with self.assertRaises(probe.PhaseCliqueBuildProbeError):
            probe._namespace_from_worker_payload(missing)
        forged = dict(payload)
        forged["candidate_mode"] = "localized_e3"
        checksum_body = dict(forged)
        checksum_body.pop("worker_args_sha256")
        forged["worker_args_sha256"] = hashlib.sha256(
            probe._canonical_json(checksum_body)
        ).hexdigest()
        with mock.patch.dict(os.environ, fixed, clear=True), self.assertRaises(
            probe.PhaseCliqueBuildProbeError
        ):
            probe._namespace_from_worker_payload(forged)

    def test_final_input_integrity_vetoes_localized_promotion(self) -> None:
        body = {
            "candidate_mode": "localized_e2",
            "inputs_unchanged": False,
            "resource_usage": {
                "peak_rss_bytes": 1024,
                "cuda_initialized": True,
                "cuda_peak_allocated_bytes": 2048,
                "cuda_peak_reserved_bytes": 4096,
            },
            "timings": {"total_seconds": 1.0},
            "wall_timeout_seconds": 60.0,
            "phase_status": "localized_e2_promoted_diagnostic",
            "fallback_reason": None,
            "localized_e2": {
                "status": "localized_e2_promoted_diagnostic",
                "reason": "fixed_promotion_gate_evaluated",
                "promotion_gate": {
                    "promoted": True,
                    "conditions": {"edge_exact": True},
                },
                "controlled_build_only_gate": {"passed": True},
            },
        }
        probe._finalize_localized_e2_integrity(body)
        transaction = body["localized_e2"]
        self.assertFalse(transaction["promotion_gate"]["promoted"])
        self.assertFalse(
            transaction["promotion_gate"]["conditions"]["inputs_unchanged"]
        )
        self.assertEqual(
            transaction["status"], "localized_e2_promotion_rejected"
        )
        self.assertEqual(body["phase_status"], transaction["status"])
        self.assertEqual(body["fallback_reason"], transaction["reason"])
        self.assertFalse(transaction["controlled_build_only_gate"]["passed"])
        self.assertTrue(body["completed_before_deadline"])

    def test_adaptive_final_promotion_requires_shared_worker_deadline(self) -> None:
        seed = {
            "candidate_mode": "rbs_adaptive_k4",
            "inputs_unchanged": True,
            "implementation_sha256": "a" * 64,
            "implementation_sha256_after": "a" * 64,
            "implementation_integrity_error_type": None,
            "shared_worker_deadline_met": True,
            "resource_usage": self._adaptive_resources(),
            "timings": {"total_seconds": 57.999},
            "wall_timeout_seconds": 60.0,
            "phase_status": "rbs_adaptive_k4_post_gate_pending_terminal",
            "fallback_reason": None,
            "rbs_adaptive_k4": {
                "status": "rbs_adaptive_k4_post_gate_pending_terminal",
                "reason": None,
                "phase_clique_attempted": True,
                "post_gate": {
                    "schema": "act.rbs_adaptive_k4_post_gate.v1",
                    "status": "passed",
                    "promoted": True,
                    "conditions": {"algorithmic_post_gate_passed": True},
                    "failed_conditions": [],
                },
            },
        }
        accepted = copy.deepcopy(seed)
        probe._finalize_rbs_adaptive_k4_integrity(accepted)
        self.assertTrue(
            accepted["rbs_adaptive_k4"]["post_gate"]["promoted"]
        )
        self.assertEqual(
            accepted["phase_status"],
            "rbs_adaptive_k4_promoted_diagnostic",
        )

        for name, value in (("false", False), ("non_bool", 1), ("missing", None)):
            with self.subTest(name=name):
                rejected = copy.deepcopy(seed)
                if name == "missing":
                    rejected.pop("shared_worker_deadline_met")
                else:
                    rejected["shared_worker_deadline_met"] = value
                probe._finalize_rbs_adaptive_k4_integrity(rejected)
                gate = rejected["rbs_adaptive_k4"]["post_gate"]
                self.assertFalse(gate["promoted"])
                self.assertEqual(gate["status"], "rejected")
                self.assertFalse(
                    gate["conditions"]["final_shared_worker_deadline_met"]
                )
                self.assertIn(
                    "final_shared_worker_deadline_met",
                    gate["failed_conditions"],
                )
                self.assertEqual(
                    rejected["phase_status"],
                    "rbs_adaptive_k4_post_gate_rejected",
                )
                self.assertTrue(rejected["completed_before_deadline"])

    def test_finalizer_preserves_pretransaction_error_receipt(self) -> None:
        body = {
            "candidate_mode": "localized_e2",
            "phase_status": "error",
            "failed_stage": "operator_hz_build",
            "error_type": "ToyBuildError",
            "error": "failed before transaction",
            "resource_usage": {
                "peak_rss_bytes": 1024,
                "cuda_initialized": True,
                "cuda_peak_allocated_bytes": 2048,
                "cuda_peak_reserved_bytes": 4096,
            },
        }
        probe._finalize_localized_e2_integrity(body)
        self.assertEqual(body["phase_status"], "error")
        self.assertEqual(body["failed_stage"], "operator_hz_build")
        self.assertEqual(body["error_type"], "ToyBuildError")

    def test_anonymous_inode_publish_has_no_validate_path_to_replace(self) -> None:
        artifacts = probe._ARTIFACT_ROOT
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            parent = Path(directory)
            slot = probe._validate_new_output_path(
                parent / "published.json", protected_paths=()
            )
            fd, identity = probe._new_worker_inode(slot.parent_fd)
            receipt = probe._checksummed(
                {"schema": probe._SCHEMA, "run_nonce": "c" * 64}
            )
            try:
                probe._write_private_worker_json_fd(
                    fd, receipt, expected_identity=identity
                )
                probe._validate_worker_receipt_fd(
                    fd,
                    run_nonce="c" * 64,
                    expected_identity=identity,
                )
                probe._publish_new_json_fd(
                    fd, slot, expected_identity=identity
                )
                target = slot.display_path
                self.assertEqual(target.stat().st_ino, identity[1])
                self.assertEqual(json.loads(target.read_text()), receipt)
            finally:
                os.close(fd)
                probe._close_output_slot(slot)

            named = parent / "existing-0600.json"
            named.write_text("do-not-truncate")
            os.chmod(named, 0o600)
            named_fd = os.open(named, os.O_RDWR)
            named_info = os.fstat(named_fd)
            try:
                with self.assertRaises(probe.PhaseCliqueBuildProbeError):
                    probe._write_private_worker_json_fd(
                        named_fd,
                        receipt,
                        expected_identity=(named_info.st_dev, named_info.st_ino),
                    )
            finally:
                os.close(named_fd)
            self.assertEqual(named.read_text(), "do-not-truncate")

    def test_parent_dirfd_rejects_rename_then_external_symlink_attack(self) -> None:
        artifacts = probe._ARTIFACT_ROOT
        artifacts.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=artifacts) as container_raw,
            tempfile.TemporaryDirectory() as external_raw,
        ):
            container = Path(container_raw)
            slot_dir = container / "slot"
            slot_dir.mkdir()
            slot = probe._validate_new_output_path(
                slot_dir / "receipt.json", protected_paths=()
            )
            fd, identity = probe._new_worker_inode(slot.parent_fd)
            moved = container / "moved-slot"
            os.rename(slot_dir, moved)
            slot_dir.symlink_to(Path(external_raw), target_is_directory=True)
            try:
                with self.assertRaisesRegex(
                    probe.PhaseCliqueBuildProbeError,
                    "canonical location changed",
                ):
                    probe._publish_new_json_fd(
                        fd, slot, expected_identity=identity
                    )
                self.assertFalse((Path(external_raw) / "receipt.json").exists())
                self.assertFalse((moved / "receipt.json").exists())
            finally:
                os.close(fd)
                probe._close_output_slot(slot)

    def test_post_link_check_removes_worker_inode_after_directory_swap(self) -> None:
        artifacts = probe._ARTIFACT_ROOT
        artifacts.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=artifacts) as container_raw,
            tempfile.TemporaryDirectory() as external_raw,
        ):
            container = Path(container_raw)
            slot_dir = container / "slot"
            slot_dir.mkdir()
            slot = probe._validate_new_output_path(
                slot_dir / "receipt.json", protected_paths=()
            )
            fd, identity = probe._new_worker_inode(slot.parent_fd)
            receipt = probe._checksummed(
                {"schema": probe._SCHEMA, "run_nonce": "d" * 64}
            )
            probe._write_private_worker_json_fd(
                fd, receipt, expected_identity=identity
            )
            moved = container / "moved-slot"
            original_validator = probe._validate_output_slot_live
            calls = 0

            def swap_after_precheck(candidate, **kwargs):
                nonlocal calls
                calls += 1
                result = original_validator(candidate, **kwargs)
                if calls == 1:
                    os.rename(slot_dir, moved)
                    slot_dir.symlink_to(
                        Path(external_raw), target_is_directory=True
                    )
                return result

            try:
                with (
                    mock.patch.object(
                        probe,
                        "_validate_output_slot_live",
                        side_effect=swap_after_precheck,
                    ),
                    self.assertRaisesRegex(
                        probe.PhaseCliqueBuildProbeError,
                        "canonical location changed",
                    ),
                ):
                    probe._publish_new_json_fd(
                        fd, slot, expected_identity=identity
                    )
                self.assertEqual(calls, 2)
                self.assertFalse((moved / "receipt.json").exists())
                self.assertFalse((Path(external_raw) / "receipt.json").exists())
            finally:
                os.close(fd)
                probe._close_output_slot(slot)

    def test_unreaped_sigkilled_child_is_removed_from_exit_registry(self) -> None:
        import multiprocessing.process as process_module

        class Child:
            pid = 81234
            exitcode = None
            _parent_pid = os.getpid()

            def __init__(self):
                self.now = 0.0

            def join(self, timeout=None):
                self.now += float(timeout or 0.0)

            def terminate(self):
                pass

            def kill(self):
                pass

        child = Child()
        process_module._children.add(child)
        try:
            result = probe._bounded_wait(
                child,
                started=0.0,
                wall_timeout=2.0,
                clock=lambda: child.now,
                signal_group=lambda *_args: None,
            )
            self.assertFalse(result["reaped_before_deadline"])
            self.assertTrue(result["detached_unreaped_child"])
            self.assertNotIn(child, process_module._children)
        finally:
            process_module._children.discard(child)

    def test_child_error_receipt_exit_mapping_preserves_diagnostics(self) -> None:
        error_receipt = {
            "phase_status": "error",
            "failed_stage": "operator_hz_build",
            "error_type": "ToyBuildError",
        }
        self.assertTrue(
            probe._child_receipt_exit_consistent(
                error_receipt, {"returncode": 2, "timed_out": False}
            )
        )
        self.assertEqual(error_receipt["failed_stage"], "operator_hz_build")
        self.assertEqual(error_receipt["error_type"], "ToyBuildError")
        self.assertFalse(
            probe._child_receipt_exit_consistent(
                error_receipt, {"returncode": 0, "timed_out": False}
            )
        )
        self.assertTrue(
            probe._child_receipt_exit_consistent(
                {"phase_status": "fallback_no_verified_k4"},
                {"returncode": 0, "timed_out": False},
            )
        )
        self.assertFalse(
            probe._child_receipt_exit_consistent(
                error_receipt, {"returncode": 2, "timed_out": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
