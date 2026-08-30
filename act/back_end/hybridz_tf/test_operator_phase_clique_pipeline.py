#!/usr/bin/env python3
"""End-to-end gates for the default-off Operator-HZ K4 pipeline."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from itertools import product
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
import unittest
from unittest import mock
import weakref

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_cliques as clique_module,
    operator_hz as operator_hz_module,
    operator_phase_clique_pipeline as pipeline_module,
)
from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
    _exact_interval_upper_violations,
    maybe_run_operator_phase_clique_pipeline,
    verify_operator_phase_clique_pipeline_result,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
    _relaxed_margin_upper,
)


def _raw_top1_source() -> str:
    return """
    (set-logic QF_LRA)
    (declare-const X_0 Real)
    (declare-const X_1 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (declare-const Y_2 Real)
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (>= X_1 -1))
    (assert (<= X_1 1))
    (assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))
    """


def _live_assert() -> dict:
    return {
        "kind": "TOP1_ROBUST",
        "C": torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        "thresholds": torch.zeros(
            (1, 2), dtype=torch.float64
        ),
        "M": 2,
        "y_true": torch.tensor([0], dtype=torch.int64),
    }


def _output_interval(build) -> tuple[np.ndarray, np.ndarray]:
    hz = build.hz
    radius = (
        np.asarray(abs(hz.Gc).sum(axis=1)).reshape(-1)
        + np.asarray(abs(hz.Gb).sum(axis=1)).reshape(-1)
    )
    lower = np.ascontiguousarray(
        (hz.c - radius).reshape(1, -1), dtype=np.float64
    )
    upper = np.ascontiguousarray(
        (hz.c + radius).reshape(1, -1), dtype=np.float64
    )
    return lower, upper


def _property_sha256(live: dict) -> str:
    C = np.ascontiguousarray(
        live["C"].detach().cpu().double().numpy(),
        dtype=np.float64,
    )
    thresholds = np.ascontiguousarray(
        live["thresholds"]
        .detach()
        .cpu()
        .double()
        .numpy()
        .reshape(-1),
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    for value in (C, thresholds):
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        digest.update(value.tobytes(order="C"))
    digest.update(b"TOP1_ROBUST")
    return digest.hexdigest()


def _residual_receipt(
    property_sha256: str,
    *,
    encoded_row: int = 0,
) -> dict:
    return {
        "schema": "property_residual_selector_v1",
        "status": "selected",
        "candidate_only": True,
        "proof_authority": False,
        "property_sha256": property_sha256,
        "selection_policy": (
            "facility_first_then_same_rival_joint"
        ),
        "joint_focus_rival_id": encoded_row,
        "rival_ids": [0, 1],
        "targets_selected": 1,
    }


def _write_raw(directory: str) -> tuple[Path, str]:
    path = Path(directory) / "top1.vnnlib"
    path.write_text(
        _raw_top1_source().strip() + "\n", encoding="utf-8"
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    build,
    path,
    source_sha256,
    *,
    residual_receipt=None,
    residual_property_sha256=None,
    deadline_seconds: float = 40.0,
):
    _ensure_constructive_nonempty_seal(build)
    live = _live_assert()
    lower, upper = _output_interval(build)
    property_sha256 = (
        _property_sha256(live)
        if residual_property_sha256 is None
        else residual_property_sha256
    )
    receipt = (
        _residual_receipt(property_sha256)
        if residual_receipt is None
        else residual_receipt
    )
    return maybe_run_operator_phase_clique_pipeline(
        build,
        enabled=True,
        vnnlib_path=path,
        expected_vnnlib_sha256=source_sha256,
        live_assert_params=live,
        output_lower=lower,
        output_upper=upper,
        residual_selector_receipt=receipt,
        residual_selector_property_sha256=property_sha256,
        deadline=time.monotonic() + deadline_seconds,
    )


def _ensure_constructive_nonempty_seal(build) -> None:
    if build.constructive_nonempty_seal is not None:
        return
    seal = operator_hz_module._make_operator_hz_constructive_nonempty_seal(
        semantic_digest=sparse_hz_semantic_digest(build.hz),
        reason="operator_phase_clique_test_exact_builder_induction",
    )
    object.__setattr__(build, "constructive_nonempty_seal", seal)
    operator_hz_module._register_operator_hz_constructive_nonempty_seal(
        seal, build
    )


def _enumerated_integer_upper(hz, objective) -> float:
    objective = np.asarray(objective, dtype=np.float64).reshape(-1)
    continuous_objective = np.asarray(
        objective @ hz.Gc
    ).reshape(-1)
    binary_objective = np.asarray(objective @ hz.Gb).reshape(-1)
    constant = float(np.dot(objective, hz.c))
    best = -np.inf
    for assignment in product((-1.0, 1.0), repeat=hz.n_bin):
        binary = np.asarray(assignment, dtype=np.float64)
        upper_rhs = hz.ub - np.asarray(
            hz.Aub @ binary
        ).reshape(-1)
        equality_rhs = hz.b - np.asarray(
            hz.Ab @ binary
        ).reshape(-1)
        solved = linprog(
            -continuous_objective,
            A_ub=hz.Auc if hz.n_ub else None,
            b_ub=upper_rhs if hz.n_ub else None,
            A_eq=hz.Ac if hz.n_eq else None,
            b_eq=equality_rhs if hz.n_eq else None,
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        if solved.success:
            best = max(
                best,
                constant
                + float(np.dot(binary_objective, binary))
                - float(solved.fun),
            )
    if not np.isfinite(best):
        raise AssertionError("integer parent unexpectedly empty")
    return float(best)


class OperatorPhaseCliquePipelineTests(unittest.TestCase):
    def test_default_off_is_zero_touch_identity_with_canonical_receipt(
        self,
    ) -> None:
        build = _k4_corner_build()

        class ExplodesOnRead:
            def __getattribute__(self, name):
                raise AssertionError(f"unexpected read: {name}")

        exploding = ExplodesOnRead()
        with (
            mock.patch.object(
                pipeline_module,
                "issue_raw_vnnlib_top1_candidate",
                side_effect=AssertionError("raw path was touched"),
            ),
            mock.patch.object(
                pipeline_module,
                "run_operator_exact_relu_phase_cliques_candidate",
                side_effect=AssertionError("candidate was touched"),
            ),
            mock.patch.object(
                pipeline_module,
                "materialize_verified_operator_phase_clique_cuts",
                side_effect=AssertionError("materializer was touched"),
            ),
        ):
            result = maybe_run_operator_phase_clique_pipeline(
                build,
                enabled=False,
                vnnlib_path=exploding,
                expected_vnnlib_sha256=exploding,
                live_assert_params=exploding,
                output_lower=exploding,
                output_upper=exploding,
                residual_selector_receipt=exploding,
                residual_selector_property_sha256=exploding,
                deadline=exploding,
                caps=exploding,
            )
        self.assertIs(result.build, build)
        self.assertTrue(result.identity_preserved)
        self.assertFalse(result.materialized)
        self.assertEqual(result.status, "no_op_disabled")
        self.assertIsInstance(result.receipt, MappingProxyType)
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                build, result
            )
        )

    def test_raw_full_chain_k4_tightens_lp_preserves_integer_optimum(
        self,
    ) -> None:
        build = _k4_corner_build()
        live = _live_assert()
        objective = tuple(
            float(value)
            for value in live["C"][0].tolist()
        )
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            result = _run(build, path, source_sha256)

        self.assertEqual(
            result.status,
            "fresh_verified_k4_clique_materialized",
        )
        self.assertIsNot(result.build, build)
        self.assertIsNot(result.build.hz, build.hz)
        self.assertTrue(result.materialized)
        self.assertFalse(result.identity_preserved)
        self.assertFalse(result.proof_authority)
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                build, result, deadline=time.monotonic() + 10.0
            )
        )
        self.assertEqual(result.receipt["full_rival_count"], 2)
        self.assertEqual(result.receipt["ranked_literal_count"], 4)
        self.assertEqual(result.receipt["pair_count"], 6)
        self.assertEqual(result.receipt["certified_edge_count"], 6)
        self.assertEqual(result.receipt["clique_count"], 1)
        self.assertEqual(result.receipt["cut_row_count"], 1)
        self.assertEqual(
            result.receipt["candidate_result_status"],
            "focused_rival_clique_compact_candidate",
        )
        self.assertEqual(
            result.receipt["candidate_telemetry_schema"],
            "act.operator_exact_relu_phase_clique_compact_candidate.v1",
        )
        self.assertFalse(result.receipt["candidate_cut_hz_emitted"])
        route = result.receipt["candidate_route_summary"]
        self.assertTrue(route["hz_absent"])
        self.assertEqual(route["pair_count"], 6)
        self.assertEqual(route["completed_pair_count"], 6)
        self.assertEqual(route["pair_status_counts"]["certified_conflict"], 6)
        self.assertEqual(
            result.receipt["verdict_path"],
            "hz_objbound_decide_only",
        )
        self.assertFalse(
            result.receipt["materialization_receipt"][
                "proof_authority"
            ]
        )
        self.assertTrue(
            result.receipt["producer_nonempty_seal_verified"]
        )
        self.assertTrue(
            result.receipt["materialization_receipt"][
                "producer_nonempty_seal_verified"
            ]
        )
        self.assertEqual(
            result.receipt["candidate_budget_fraction"], 0.4
        )
        self.assertEqual(
            result.receipt["materializer_reserve_fraction"], 0.6
        )
        self.assertAlmostEqual(
            result.receipt["candidate_budget_seconds"],
            0.4 * result.receipt["initial_budget_seconds"],
            places=10,
        )
        self.assertAlmostEqual(
            result.receipt["minimum_materializer_reserve_seconds"],
            0.6 * result.receipt["initial_budget_seconds"],
            places=10,
        )
        self.assertLessEqual(
            result.receipt["candidate_elapsed_seconds"],
            result.receipt["candidate_budget_seconds"],
        )
        timings = result.receipt["timings"]
        self.assertIn("terminal_seal_seconds", timings)
        self.assertGreaterEqual(timings["terminal_seal_seconds"], 0.0)
        self.assertLessEqual(
            result.receipt["candidate_elapsed_seconds"]
            + timings["materializer_and_recheck_seconds"]
            + timings["terminal_seal_seconds"],
            timings["total_seconds"] + 1.0e-9,
        )

        before_lp = _relaxed_margin_upper(build.hz, objective)
        after_lp = _relaxed_margin_upper(result.build.hz, objective)
        self.assertAlmostEqual(before_lp, 0.25, places=10)
        self.assertLess(after_lp, 0.0)
        before_integer = _enumerated_integer_upper(
            build.hz, objective
        )
        after_integer = _enumerated_integer_upper(
            result.build.hz, objective
        )
        self.assertAlmostEqual(before_integer, -0.25, places=10)
        self.assertAlmostEqual(
            before_integer, after_integer, places=10
        )

        forged = dict(result.receipt)
        forged["fresh_semantic_digest"] = "0" * 64
        forged_result = replace(result, receipt=forged)
        self.assertFalse(
            verify_operator_phase_clique_pipeline_result(
                build, forged_result
            )
        )

        tampered_body = dict(result.receipt)
        tampered_body.pop("receipt_sha256")
        tampered_route = dict(tampered_body["candidate_route_summary"])
        tampered_route["candidate_nonzeros"] -= 1
        tampered_body["candidate_route_summary"] = tampered_route
        tampered_result = replace(
            result,
            receipt=pipeline_module._checksummed_receipt(
                tampered_body
            ),
        )
        self.assertFalse(
            verify_operator_phase_clique_pipeline_result(
                build,
                tampered_result,
                deadline=time.monotonic() + 10.0,
            )
        )

        false_seal_body = dict(result.receipt)
        false_seal_body.pop("receipt_sha256")
        false_seal_body[
            "producer_nonempty_seal_verified"
        ] = False
        false_seal_result = replace(
            result,
            receipt=pipeline_module._checksummed_receipt(
                false_seal_body
            ),
        )
        self.assertFalse(
            verify_operator_phase_clique_pipeline_result(
                build,
                false_seal_result,
                deadline=time.monotonic() + 10.0,
            )
        )

    def test_compact_candidate_allocates_no_cut_before_materializer(
        self,
    ) -> None:
        build = _k4_corner_build()
        created_cut_refs = []
        captured = {}
        real_copy = clique_module._copy_parent_with_clique_cut
        real_candidate = (
            pipeline_module.run_operator_exact_relu_phase_cliques_candidate
        )
        real_materializer = (
            pipeline_module.materialize_verified_operator_phase_clique_cuts
        )

        def track_cut(*args, **kwargs):
            cut = real_copy(*args, **kwargs)
            created_cut_refs.append(weakref.ref(cut))
            return cut

        def capture_candidate(*args, **kwargs):
            self.assertIs(kwargs.get("emit_cut_hz"), False)
            candidate = real_candidate(*args, **kwargs)
            captured["candidate"] = candidate
            return candidate

        def check_materializer_entry(*args, **kwargs):
            self.assertEqual(created_cut_refs, [])
            candidate = captured["candidate"]
            self.assertIsNone(candidate.hz)
            self.assertFalse(
                any(
                    type(value) is pipeline_module.SparseHZono
                    for value in vars(candidate).values()
                )
            )
            return real_materializer(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    clique_module,
                    "_copy_parent_with_clique_cut",
                    side_effect=track_cut,
                ),
                mock.patch.object(
                    pipeline_module,
                    "run_operator_exact_relu_phase_cliques_candidate",
                    side_effect=capture_candidate,
                ),
                mock.patch.object(
                    pipeline_module,
                    "materialize_verified_operator_phase_clique_cuts",
                    side_effect=check_materializer_entry,
                ),
            ):
                result = _run(build, path, source_sha256)

        self.assertTrue(result.materialized)
        self.assertGreaterEqual(len(created_cut_refs), 1)
        self.assertTrue(any(ref() is not None for ref in created_cut_refs))

    def test_error_timeout_and_no_clique_preserve_baseline_identity(
        self,
    ) -> None:
        build = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with mock.patch.object(
                pipeline_module,
                "issue_raw_vnnlib_top1_candidate",
                side_effect=RuntimeError("synthetic candidate error"),
            ):
                error = _run(build, path, source_sha256)
            self.assertIs(error.build, build)
            self.assertEqual(
                error.status, "baseline_fallback_error"
            )
            self.assertTrue(
                verify_operator_phase_clique_pipeline_result(
                    build, error
                )
            )

            live = _live_assert()
            lower, upper = _output_interval(build)
            property_sha256 = _property_sha256(live)
            timeout = maybe_run_operator_phase_clique_pipeline(
                build,
                enabled=True,
                vnnlib_path=path,
                expected_vnnlib_sha256=source_sha256,
                live_assert_params=live,
                output_lower=lower,
                output_upper=upper,
                residual_selector_receipt=_residual_receipt(
                    property_sha256
                ),
                residual_selector_property_sha256=property_sha256,
                deadline=time.monotonic() - 1.0,
            )
            self.assertIs(timeout.build, build)
            self.assertEqual(
                timeout.status, "baseline_fallback_timeout"
            )
            self.assertIsNone(
                timeout.receipt["initial_budget_seconds"]
            )
            self.assertIsNone(
                timeout.receipt["candidate_budget_seconds"]
            )
            self.assertIsNone(
                timeout.receipt[
                    "minimum_materializer_reserve_seconds"
                ]
            )
            self.assertIs(
                type(timeout.receipt["candidate_elapsed_seconds"]),
                float,
            )
            self.assertGreaterEqual(
                timeout.receipt["candidate_elapsed_seconds"], 0.0
            )

            with mock.patch.object(
                pipeline_module,
                "run_operator_exact_relu_phase_cliques_candidate",
                return_value=object(),
            ):
                no_clique = _run(build, path, source_sha256)
            self.assertIs(no_clique.build, build)
            self.assertEqual(
                no_clique.status,
                "baseline_fallback_no_k4_clique",
            )
            self.assertEqual(
                no_clique.receipt["fallback_reason"],
                "no_complete_k4_clique",
            )

    def test_fallback_releases_exception_traceback_before_allocation(
        self,
    ) -> None:
        build = _k4_corner_build()
        observed: dict[str, object] = {}
        real_fallback = pipeline_module._fallback_result

        class LargeCandidateFrame:
            pass

        def fail_with_large_frame(*_args, **_kwargs):
            payload = LargeCandidateFrame()
            payload.buffer = np.empty(1 << 20, dtype=np.float64)
            observed["payload_ref"] = weakref.ref(payload)
            raise TimeoutError("synthetic candidate deadline")

        def fallback_after_traceback_release(*args, **kwargs):
            payload_ref = observed.get("payload_ref")
            self.assertIsInstance(payload_ref, weakref.ReferenceType)
            self.assertIsNone(payload_ref())
            observed["released_before_fallback"] = True
            return real_fallback(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    pipeline_module,
                    "run_operator_exact_relu_phase_cliques_candidate",
                    side_effect=fail_with_large_frame,
                ),
                mock.patch.object(
                    pipeline_module,
                    "_fallback_result",
                    side_effect=fallback_after_traceback_release,
                ),
            ):
                result = _run(build, path, source_sha256)

        self.assertTrue(observed["released_before_fallback"])
        self.assertEqual(result.status, "baseline_fallback_timeout")
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                build, result
            )
        )
        initial = result.receipt["initial_budget_seconds"]
        candidate = result.receipt["candidate_budget_seconds"]
        reserve = result.receipt[
            "minimum_materializer_reserve_seconds"
        ]
        elapsed = result.receipt["candidate_elapsed_seconds"]
        for value in (initial, candidate, reserve, elapsed):
            self.assertIs(type(value), float)
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertAlmostEqual(candidate, 0.40 * initial, places=12)
        self.assertAlmostEqual(reserve, 0.60 * initial, places=12)

        receipt_body = dict(result.receipt)
        receipt_body.pop("receipt_sha256")
        malformed = (
            ("initial_budget_seconds", -1.0),
            ("candidate_budget_seconds", -1.0),
            ("minimum_materializer_reserve_seconds", -1.0),
            ("candidate_elapsed_seconds", -1.0),
            ("candidate_budget_seconds", 0.41 * initial),
            ("minimum_materializer_reserve_seconds", 0.59 * initial),
            ("candidate_elapsed_seconds", 0),
            ("initial_budget_seconds", None),
            ("candidate_elapsed_semantics", "clamped"),
        )
        with mock.patch.object(
            pipeline_module,
            "_validate_pipeline_solver_handoff_registration",
            return_value=True,
        ):
            for field, value in malformed:
                with self.subTest(field=field, value=value):
                    forged_body = dict(receipt_body)
                    forged_body[field] = value
                    forged = replace(
                        result,
                        receipt=pipeline_module._checksummed_receipt(
                            forged_body
                        ),
                    )
                    self.assertFalse(
                        verify_operator_phase_clique_pipeline_result(
                            build, forged
                        )
                    )

            # Elapsed time is an honest raw observation at failure entry.
            # It may exceed the 40-percent candidate allocation when a
            # downstream call returns late; the verifier must not require a
            # misleading clamp to the budget boundary.
            over_budget_body = dict(receipt_body)
            over_budget_elapsed = candidate + 1.0
            over_budget_body["candidate_elapsed_seconds"] = (
                over_budget_elapsed
            )
            over_budget_timings = dict(
                over_budget_body["timings"]
            )
            over_budget_timings["total_seconds"] = (
                over_budget_elapsed + 0.25
            )
            over_budget_body["timings"] = over_budget_timings
            over_budget = replace(
                result,
                receipt=pipeline_module._checksummed_receipt(
                    over_budget_body
                ),
            )
            self.assertTrue(
                verify_operator_phase_clique_pipeline_result(
                    build, over_budget
                )
            )

            elapsed_after_total_body = dict(receipt_body)
            elapsed_after_total_body["candidate_elapsed_seconds"] = (
                elapsed_after_total_body["timings"]["total_seconds"]
                + 1.0
            )
            elapsed_after_total = replace(
                result,
                receipt=pipeline_module._checksummed_receipt(
                    elapsed_after_total_body
                ),
            )
            self.assertFalse(
                verify_operator_phase_clique_pipeline_result(
                    build, elapsed_after_total
                )
            )

        for field, value in (
            ("initial_budget_seconds", float("inf")),
            ("candidate_elapsed_seconds", float("nan")),
        ):
            with self.subTest(nonfinite_field=field):
                nonfinite_body = dict(receipt_body)
                nonfinite_body[field] = value
                with self.assertRaisesRegex(
                    pipeline_module.OperatorPhaseCliquePipelineError,
                    "receipt_contains_nonfinite_float",
                ):
                    pipeline_module._checksummed_receipt(
                        nonfinite_body
                    )

    def test_pair_timeout_receipt_preserves_exact_partial_progress(
        self,
    ) -> None:
        build = _k4_corner_build()

        def fail_after_one_pair(*_args, **kwargs):
            sink = kwargs.get("diagnostic_progress")
            self.assertIs(type(sink), dict)
            progress = clique_module._new_candidate_progress()
            progress.update(
                {
                    "status": "pair_probe",
                    "model_load_started": True,
                    "model_loaded": True,
                    "oracle_backend": (
                        "highspy_persistent_simplex_presolve_lazy_dual_ray_v2"
                    ),
                    "oracle_presolve": "on",
                    "candidate_load_mode": (
                        "split_continuous_rows_binary_change_coeff_v1"
                    ),
                    "binary_change_coefficient_cap": 65536,
                    "candidate_rows": build.hz.n_ub + build.hz.n_eq,
                    "candidate_columns": (
                        build.hz.n_cont + build.hz.n_bin
                    ),
                    "candidate_nonzeros": sum(
                        int(getattr(build.hz, name).nnz)
                        for name in ("Auc", "Aub", "Ac", "Ab")
                    ),
                    "pair_target_count": 6,
                    "pair_attempted_count": 2,
                    "pair_completed_count": 1,
                    "certified_conflict_count": 1,
                    "last_pair_index": 1,
                }
            )
            sink.update(progress)
            raise TimeoutError("synthetic_pair_two_timeout")

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with mock.patch.object(
                pipeline_module,
                "run_operator_exact_relu_phase_cliques_candidate",
                side_effect=fail_after_one_pair,
            ):
                result = _run(build, path, source_sha256)

        self.assertEqual(result.status, "baseline_fallback_timeout")
        self.assertIs(
            result.receipt["candidate_progress_available"], True
        )
        progress = result.receipt["candidate_progress"]
        self.assertEqual(progress["status"], "pair_probe")
        self.assertEqual(progress["pair_target_count"], 6)
        self.assertEqual(progress["pair_attempted_count"], 2)
        self.assertEqual(progress["pair_completed_count"], 1)
        self.assertEqual(progress["certified_conflict_count"], 1)
        self.assertIs(progress["partial_never_authorizes_edge"], True)
        self.assertIs(progress["materializer_reached"], False)
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(build, result)
        )

        body = pipeline_module._builtin_copy(result.receipt)
        body.pop("receipt_sha256")
        body["candidate_progress"]["pair_completed_count"] = 3
        tampered = replace(
            result,
            receipt=pipeline_module._checksummed_receipt(body),
        )
        self.assertFalse(
            verify_operator_phase_clique_pipeline_result(build, tampered)
        )

    def test_partial_pair_record_never_becomes_an_edge(self) -> None:
        build = _k4_corner_build()
        real_candidate = (
            pipeline_module
            .run_operator_exact_relu_phase_cliques_candidate
        )

        def return_partial_candidate(*args, **kwargs):
            honest = real_candidate(*args, **kwargs)
            partial = replace(
                honest.pair_records[0],
                status="feasible_or_unknown",
                ray_nonzero_rows=0,
                certificate_sha256=None,
                rationalization=None,
            )
            return replace(
                honest,
                pair_records=(partial, *honest.pair_records[1:]),
            )

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    pipeline_module,
                    "run_operator_exact_relu_phase_cliques_candidate",
                    side_effect=return_partial_candidate,
                ),
                mock.patch.object(
                    pipeline_module,
                    "materialize_verified_operator_phase_clique_cuts",
                    side_effect=AssertionError(
                        "partial pair reached the materializer"
                    ),
                ),
            ):
                result = _run(build, path, source_sha256)

        self.assertIs(result.build, build)
        self.assertFalse(result.materialized)
        self.assertEqual(
            result.status, "baseline_fallback_no_k4_clique"
        )
        self.assertEqual(
            result.receipt["fallback_reason"],
            "no_complete_k4_clique",
        )
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(build, result)
        )

    def test_raw_hash_selector_and_materializer_tamper_reject(
        self,
    ) -> None:
        build = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            wrong_hash = _run(
                build, path, "0" * 64
            )
            self.assertIs(wrong_hash.build, build)
            self.assertEqual(
                wrong_hash.receipt["failed_stage"],
                "raw_top1_issue_consume",
            )

            live = _live_assert()
            property_sha256 = _property_sha256(live)
            bad_selector = _residual_receipt(property_sha256)
            bad_selector["property_sha256"] = "f" * 64
            rejected_selector = _run(
                build,
                path,
                source_sha256,
                residual_receipt=bad_selector,
                residual_property_sha256=property_sha256,
            )
            self.assertIs(rejected_selector.build, build)
            self.assertEqual(
                rejected_selector.receipt["failed_stage"],
                "residual_joint_focus",
            )

            real_materializer = (
                pipeline_module
                .materialize_verified_operator_phase_clique_cuts
            )

            def forged_materializer(*args, **kwargs):
                honest = real_materializer(*args, **kwargs)
                forged = dict(honest.receipt)
                forged["proof_authority"] = True
                return replace(honest, receipt=forged)

            with mock.patch.object(
                pipeline_module,
                "materialize_verified_operator_phase_clique_cuts",
                side_effect=forged_materializer,
            ):
                rejected_materializer = _run(
                    build, path, source_sha256
                )
            self.assertIs(rejected_materializer.build, build)
            self.assertFalse(rejected_materializer.materialized)
            self.assertEqual(
                rejected_materializer.receipt["failed_stage"],
                "fresh_materializer",
            )

            def false_seal_materializer(*args, **kwargs):
                honest = real_materializer(*args, **kwargs)
                forged = dict(honest.receipt)
                forged.pop("receipt_sha256")
                forged["producer_nonempty_seal_verified"] = False
                return replace(
                    honest,
                    receipt=pipeline_module._checksummed_receipt(
                        forged
                    ),
                )

            with mock.patch.object(
                pipeline_module,
                "materialize_verified_operator_phase_clique_cuts",
                side_effect=false_seal_materializer,
            ):
                rejected_false_seal = _run(
                    build, path, source_sha256
                )
            self.assertIs(rejected_false_seal.build, build)
            self.assertFalse(rejected_false_seal.materialized)
            self.assertEqual(
                rejected_false_seal.receipt["failed_stage"],
                "fresh_materializer",
            )

    def test_post_validation_candidate_mutation_uses_frozen_bindings(
        self,
    ) -> None:
        build = _k4_corner_build()
        captured = {}
        real_derive = (
            pipeline_module
            .derive_operator_exact_relu_property_phase_literals
        )
        real_candidate = (
            pipeline_module
            .run_operator_exact_relu_phase_cliques_candidate
        )
        real_stage_seconds = pipeline_module._stage_seconds

        def capture_selection(*args, **kwargs):
            selection = real_derive(*args, **kwargs)
            captured["selection"] = selection
            return selection

        def capture_candidate(*args, **kwargs):
            candidate = real_candidate(*args, **kwargs)
            captured["candidate"] = candidate
            return candidate

        def mutate_after_validation(timings, name, started):
            real_stage_seconds(timings, name, started)
            if (
                name == "materializer_and_recheck_seconds"
                and not captured.get("mutated", False)
            ):
                captured["mutated"] = True
                object.__setattr__(
                    captured["selection"],
                    "selection_digest",
                    "0" * 64,
                )
                object.__setattr__(
                    captured["candidate"],
                    "subset_binding_digest",
                    "1" * 64,
                )
                object.__setattr__(
                    captured["candidate"], "cliques", ()
                )

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    pipeline_module,
                    "derive_operator_exact_relu_property_phase_literals",
                    side_effect=capture_selection,
                ),
                mock.patch.object(
                    pipeline_module,
                    "run_operator_exact_relu_phase_cliques_candidate",
                    side_effect=capture_candidate,
                ),
                mock.patch.object(
                    pipeline_module,
                    "_stage_seconds",
                    side_effect=mutate_after_validation,
                ),
            ):
                result = _run(build, path, source_sha256)

        self.assertTrue(captured["mutated"])
        self.assertTrue(result.materialized)
        self.assertEqual(
            result.receipt["selection_digest"],
            result.receipt["materialization_receipt"][
                "selection_digest"
            ],
        )
        self.assertEqual(
            result.receipt["subset_binding_digest"],
            result.receipt["materialization_receipt"][
                "subset_binding_digest"
            ],
        )
        self.assertNotEqual(
            result.receipt["selection_digest"], "0" * 64
        )
        self.assertNotEqual(
            result.receipt["subset_binding_digest"], "1" * 64
        )
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                build, result
            )
        )

    def test_terminal_seals_reject_post_digest_core_mutation(
        self,
    ) -> None:
        build = _k4_corner_build()
        captured = {}
        real_materializer = (
            pipeline_module
            .materialize_verified_operator_phase_clique_cuts
        )
        real_stage_seconds = pipeline_module._stage_seconds

        def capture_materializer(*args, **kwargs):
            materialized = real_materializer(*args, **kwargs)
            captured["materialized"] = materialized
            return materialized

        def empty_after_validation(timings, name, started):
            real_stage_seconds(timings, name, started)
            if (
                name == "materializer_and_recheck_seconds"
                and not captured.get("emptied", False)
            ):
                captured["emptied"] = True
                captured["materialized"].build.hz.ub[0] = -1.0e100

        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with (
                mock.patch.object(
                    pipeline_module,
                    "materialize_verified_operator_phase_clique_cuts",
                    side_effect=capture_materializer,
                ),
                mock.patch.object(
                    pipeline_module,
                    "_stage_seconds",
                    side_effect=empty_after_validation,
                ),
            ):
                rejected = _run(build, path, source_sha256)
            self.assertTrue(captured["emptied"])
            self.assertIs(rejected.build, build)
            self.assertFalse(rejected.materialized)

            honest = _run(build, path, source_sha256)
        fresh = honest.build.hz
        real_digest = pipeline_module.sparse_hz_semantic_digest
        state = {"mutated": False}

        def mutate_after_honest_digest(hz):
            digest = real_digest(hz)
            if hz is fresh and not state["mutated"]:
                state["mutated"] = True
                fresh.ub[0] = -1.0e100
            return digest

        with mock.patch.object(
            pipeline_module,
            "sparse_hz_semantic_digest",
            side_effect=mutate_after_honest_digest,
        ):
            self.assertFalse(
                verify_operator_phase_clique_pipeline_result(
                    build, honest
                )
            )
        self.assertTrue(state["mutated"])

    def test_terminal_identity_rejects_equal_core_replacement(
        self,
    ) -> None:
        build = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            honest = _run(build, path, source_sha256)
        fresh = honest.build.hz
        real_digest = pipeline_module.sparse_hz_semantic_digest
        state = {"replaced": False}

        def replace_after_honest_digest(hz):
            digest = real_digest(hz)
            if hz is fresh and not state["replaced"]:
                state["replaced"] = True
                fresh.c = fresh.c.copy()
            return digest

        with mock.patch.object(
            pipeline_module,
            "sparse_hz_semantic_digest",
            side_effect=replace_after_honest_digest,
        ):
            self.assertFalse(
                verify_operator_phase_clique_pipeline_result(
                    build, honest
                )
            )
        self.assertTrue(state["replaced"])

    def test_exception_str_hook_cannot_escape_identity_fallback(
        self,
    ) -> None:
        class ExplosiveStringError(Exception):
            def __str__(self):
                raise AssertionError("exception formatting hook ran")

        build = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with mock.patch.object(
                pipeline_module,
                "issue_raw_vnnlib_top1_candidate",
                side_effect=ExplosiveStringError("synthetic"),
            ):
                result = _run(build, path, source_sha256)
        self.assertIs(result.build, build)
        self.assertEqual(result.status, "baseline_fallback_error")
        self.assertEqual(
            result.receipt["error_type"],
            "ExplosiveStringError",
        )
        self.assertTrue(
            verify_operator_phase_clique_pipeline_result(
                build, result
            )
        )

    def test_199_rival_exact_scheduling_vector_is_complete_and_light(
        self,
    ) -> None:
        width = 200
        lower = np.zeros(width, dtype=np.float64)
        upper = np.ones(width, dtype=np.float64)
        rivals = []
        for competitor in range(1, width):
            objective = [0.0] * width
            objective[0] = -1.0
            objective[competitor] = 1.0
            rivals.append(
                RivalSpec(
                    rival_id=competitor,
                    objective=tuple(objective),
                    threshold=0.0,
                    assert_digest=hashlib.sha256(
                        str(competitor).encode("ascii")
                    ).hexdigest(),
                )
            )
        exact = _exact_interval_upper_violations(
            tuple(rivals),
            lower,
            upper,
            deadline=time.monotonic() + 5.0,
        )
        self.assertEqual(len(exact), 199)
        self.assertEqual(set(exact), {(1, 1)})


if __name__ == "__main__":
    unittest.main()
