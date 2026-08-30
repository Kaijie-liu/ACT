#!/usr/bin/env python3
"""Verdict-firewall and stop-loss tests for the PCOH build-only transaction."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import itertools
import json
import time
from types import MappingProxyType
import unittest
from unittest import mock

from act.back_end.hybridz_tf import operator_phase_conditioned_build_only as build_only
from act.back_end.hybridz_tf.adaptive_phase_forest import RivalSpec, sparse_hz_semantic_digest
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_build_only import (
    PCOHBuildOnlyCaps,
    PhaseConditionedBuildOnlyError,
    run_phase_conditioned_objective_hull_build_only,
    verify_phase_conditioned_objective_hull_build_only_diagnostic,
    verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
    _rivals,
)


def _inputs():
    build = _k4_corner_build()
    rivals = _rivals()
    selection = derive_operator_exact_relu_property_phase_literals(build, rivals)
    stable_ids = tuple(
        mapping.stable_bcol_id for mapping in selection.mappings[:2]
    )
    return build, rivals, selection, stable_ids


def _run(*, caps=None):
    build, rivals, selection, stable_ids = _inputs()
    kwargs = {} if caps is None else {"caps": caps}
    result = run_phase_conditioned_objective_hull_build_only(
        build,
        rivals,
        selection,
        focused_rival_id=10,
        stable_bit_ids=stable_ids,
        deadline=time.monotonic() + 60.0,
        **kwargs,
    )
    return build, result


class PCOHBuildOnlyTransactionTests(unittest.TestCase):
    def test_k2_materializes_validates_releases_and_returns_only_receipt(self):
        source, result = _run()
        self.assertTrue(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )
        self.assertFalse(result.proof_authority)
        self.assertFalse(result.verdict_authority)
        self.assertFalse(result.full_parent_lp_called)
        self.assertEqual(len(result.stable_bit_ids), 2)
        self.assertEqual(len(result.conditional_certificate_sha256), 4)
        self.assertEqual(result.receipt["pair_query_count"], 4)
        self.assertTrue(result.receipt["diagnostic_only"])
        self.assertTrue(result.receipt["build_only_sentinel_ready"])
        self.assertFalse(result.receipt["solver_handoff_ready"])
        self.assertFalse(result.receipt["production_ready"])
        self.assertFalse(result.receipt["fresh_build_returned"])
        self.assertTrue(result.receipt["fresh_build_released_before_return"])
        self.assertFalse(result.receipt["final_verifier_solver_called"])
        self.assertFalse(result.receipt["hz_base_feasibility_called"])
        self.assertFalse(result.receipt["hz_objbound_decide_called"])
        self.assertFalse(result.receipt["full_parent_lp_called"])
        self.assertFalse(result.receipt["full_parent_lp_solver_called"])
        self.assertEqual(
            result.schema, "act.hybridz_pcoh_build_only_diagnostic.v2"
        )
        self.assertEqual(
            result.receipt["schema"],
            "act.hybridz_pcoh_build_only_receipt.v2",
        )
        self.assertIs(
            result.materialized_tightness_summary,
            result.receipt["materialized_tightness_summary"],
        )
        summary = result.materialized_tightness_summary
        self.assertEqual(
            summary["schema"],
            "act.hybridz_pc_materialized_tightness_summary.toy.v1",
        )
        self.assertTrue(summary["diagnostic_only"])
        self.assertFalse(summary["full_parent_lp_called"])
        self.assertFalse(summary["proof_authority"])
        self.assertFalse(summary["verdict_authority"])
        self.assertEqual(
            summary["summary_sha256"],
            result.receipt["materialized_tightness_summary_sha256"],
        )
        self.assertEqual(
            summary["conditional_certificate_sha256"],
            result.conditional_certificate_sha256,
        )
        self.assertEqual(
            result.source_semantic_digest,
            sparse_hz_semantic_digest(source.hz),
        )
        self.assertEqual(
            result.fresh_dimensions[1],
            result.source_dimensions[1] + 4,
        )
        self.assertEqual(
            result.fresh_dimensions[4],
            result.source_dimensions[4] + 1,
        )
        self.assertNotIn("hz", vars(result))
        self.assertNotIn("build", vars(result))

    def test_final_solver_and_base_feasibility_are_never_called(self):
        from act.back_end.solver import solver_hz

        with mock.patch.object(
            solver_hz,
            "hz_objbound_decide",
            side_effect=AssertionError("final objective solver forbidden"),
        ) as decide, mock.patch.object(
            solver_hz,
            "hz_base_feasibility",
            side_effect=AssertionError("base feasibility forbidden"),
        ) as feasible:
            _source, result = _run()
        decide.assert_not_called()
        feasible.assert_not_called()
        self.assertTrue(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )

    def test_no_full_parent_lp_helper_is_called(self):
        from act.back_end.solver import solver_hz

        build, rivals, selection, stable_ids = _inputs()
        with mock.patch.object(
            solver_hz,
            "hz_compute_lp_bounds",
            side_effect=AssertionError("full parent LP helper forbidden"),
        ) as full_lp, mock.patch.object(
            solver_hz,
            "hz_objbound_decide",
            side_effect=AssertionError("verdict solver forbidden"),
        ) as decide, mock.patch.object(
            solver_hz,
            "hz_base_feasibility",
            side_effect=AssertionError("base solver forbidden"),
        ) as feasible:
            result = run_phase_conditioned_objective_hull_build_only(
                build,
                rivals,
                selection,
                focused_rival_id=10,
                stable_bit_ids=stable_ids,
                deadline=time.monotonic() + 60.0,
            )
        full_lp.assert_not_called()
        decide.assert_not_called()
        feasible.assert_not_called()
        self.assertTrue(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )

    def test_fresh_strict_tightness_verifier_precedes_consume(self):
        events = []
        real_verify = (
            build_only.verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness
        )
        real_consume = (
            build_only.consume_live_phase_conditioned_objective_hull_fresh_build
        )

        def verify_first(issuance):
            events.append("strict_verify")
            return real_verify(issuance)

        def consume_second(issuance, capability, *, deadline):
            events.append("consume")
            return real_consume(issuance, capability, deadline=deadline)

        with mock.patch.object(
            build_only,
            "verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness",
            side_effect=verify_first,
        ) as verifier, mock.patch.object(
            build_only,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=consume_second,
        ) as consume:
            _source, result = _run()
        self.assertEqual(events, ["strict_verify", "consume"])
        verifier.assert_called_once()
        consume.assert_called_once()
        self.assertTrue(
            result.receipt[
                "materialized_tightness_strict_verified_before_consume"
            ]
        )
        self.assertFalse(
            result.receipt[
                "materialized_tightness_live_verifier_valid_after_consume"
            ]
        )

    def test_k2_empty_highest_is_excluded_in_propagated_summary(self):
        _source, result = _run()
        summary = result.materialized_tightness_summary
        uppers = tuple(
            Fraction.from_float(float.fromhex(value))
            for value in summary["pattern_upper_hex"]
        )
        active = tuple(
            value
            for value, keep in zip(uppers, summary["active_pattern_mask"])
            if keep
        )
        self.assertEqual(
            summary["canonical_patterns"],
            tuple(itertools.product((-1, 1), repeat=2)),
        )
        self.assertEqual(
            summary["active_pattern_mask"], (True, True, True, False)
        )
        self.assertEqual(max(uppers), uppers[-1])
        self.assertGreater(max(uppers), max(active))
        self.assertEqual(
            Fraction.from_float(
                float.fromhex(summary["ideal_union_upper_hex"])
            ),
            max(active),
        )
        self.assertLess(
            float.fromhex(summary["final_structural_upper_hex"]),
            float.fromhex(summary["global_cube_upper_hex"]),
        )
        self.assertTrue(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )

    def test_json_tightness_payload_roundtrip_is_registry_and_solver_free(self):
        from act.back_end.solver import solver_hz

        _source, result = _run()
        thawed = json.loads(
            json.dumps(
                dict(result.materialized_tightness_summary),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        with build_only._DIAGNOSTIC_REGISTRY_LOCK:
            registered = build_only._DIAGNOSTIC_REGISTRY.pop(result)
        try:
            with mock.patch.object(
                solver_hz,
                "hz_compute_lp_bounds",
                side_effect=AssertionError("full parent LP forbidden"),
            ) as full_lp, mock.patch.object(
                solver_hz,
                "hz_objbound_decide",
                side_effect=AssertionError("verdict solver forbidden"),
            ) as decide, mock.patch.object(
                solver_hz,
                "hz_base_feasibility",
                side_effect=AssertionError("base solver forbidden"),
            ) as feasible:
                self.assertTrue(
                    verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                        thawed,
                        expected_source_semantic_digest=(
                            result.source_semantic_digest
                        ),
                        expected_stable_bit_ids=list(result.stable_bit_ids),
                        expected_conditional_certificate_sha256=list(
                            result.conditional_certificate_sha256
                        ),
                        expected_summary_sha256=(
                            result.materialized_tightness_summary[
                                "summary_sha256"
                            ]
                        ),
                    )
                )
            full_lp.assert_not_called()
            decide.assert_not_called()
            feasible.assert_not_called()
        finally:
            with build_only._DIAGNOSTIC_REGISTRY_LOCK:
                build_only._DIAGNOSTIC_REGISTRY[result] = registered
        self.assertTrue(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                thawed,
                expected_source_semantic_digest=result.source_semantic_digest,
                expected_stable_bit_ids=result.stable_bit_ids,
                expected_conditional_certificate_sha256=(
                    result.conditional_certificate_sha256
                ),
            )
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                thawed,
                expected_source_semantic_digest=result.source_semantic_digest,
                expected_stable_bit_ids=result.stable_bit_ids,
                expected_conditional_certificate_sha256=(
                    result.conditional_certificate_sha256
                ),
                expected_summary_sha256=None,
            )
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                thawed,
                expected_source_semantic_digest=result.source_semantic_digest,
                expected_stable_bit_ids=result.stable_bit_ids,
                expected_conditional_certificate_sha256=(
                    result.conditional_certificate_sha256
                ),
                expected_summary_sha256="0" * 64,
            )
        )

    def test_json_tightness_payload_rehashed_tamper_fails_closed(self):
        _source, result = _run()
        original = json.loads(
            json.dumps(
                dict(result.materialized_tightness_summary),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

        def rehashed(**changes):
            candidate = json.loads(json.dumps(original))
            candidate.update(changes)
            candidate.pop("summary_sha256", None)
            candidate["summary_sha256"] = build_only._canonical_sha256(
                candidate
            )
            return candidate

        cases = (
            rehashed(active_pattern_mask=[True, True, True, True]),
            rehashed(parent_semantic_digest="0" * 64),
            rehashed(
                conditional_certificate_sha256=[
                    "f" * 64,
                    *original["conditional_certificate_sha256"][1:],
                ]
            ),
            rehashed(proof_authority=True),
            rehashed(
                final_structural_upper_hex=original["global_cube_upper_hex"]
            ),
            rehashed(
                adapter_candidate_sha256="0" * 64,
                descriptor_representation_sha256="1" * 64,
                row_frame_sha256="2" * 64,
                conditional_pattern_sha256=[
                    "3" * 64,
                    "4" * 64,
                    "5" * 64,
                    "6" * 64,
                ],
                objective_binding_sha256="7" * 64,
                objective_envelope_sha256="8" * 64,
                global_checker_sha256="9" * 64,
            ),
        )
        trusted_summary_sha256 = result.materialized_tightness_summary[
            "summary_sha256"
        ]
        for candidate in cases:
            with self.subTest(change=candidate):
                self.assertFalse(
                    verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                        candidate,
                        expected_source_semantic_digest=(
                            result.source_semantic_digest
                        ),
                        expected_stable_bit_ids=result.stable_bit_ids,
                        expected_conditional_certificate_sha256=(
                            result.conditional_certificate_sha256
                        ),
                        expected_summary_sha256=trusted_summary_sha256,
                    )
                )

    def test_three_output_objective_stops_before_fraction_or_fresh(self):
        build, rivals, selection, stable_ids = _inputs()
        malformed = (
            RivalSpec(
                rival_id=10,
                objective=(1.0, 1.0, 1.0),
                threshold=0.0,
                assert_digest="a" * 64,
            ),
            rivals[1],
        )
        with mock.patch.object(
            build_only,
            "_build_complete_operator_phase_conditioned_objective_bounds_until",
            side_effect=AssertionError("Fraction producer must not start"),
        ) as producer, mock.patch.object(
            build_only,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=AssertionError("fresh must not start"),
        ) as issue:
            with self.assertRaisesRegex(
                PhaseConditionedBuildOnlyError,
                "not_exact_two_output_margin",
            ):
                run_phase_conditioned_objective_hull_build_only(
                    build,
                    malformed,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    deadline=time.monotonic() + 60.0,
                )
        producer.assert_not_called()
        issue.assert_not_called()

        wrong_sign = (
            RivalSpec(
                rival_id=10,
                objective=(1.0, 1.0, 0.0),
                threshold=0.0,
                assert_digest="a" * 64,
            ),
            rivals[1],
        )
        with self.assertRaisesRegex(
            PhaseConditionedBuildOnlyError,
            "not_unit_signed_class_margin",
        ):
            run_phase_conditioned_objective_hull_build_only(
                build,
                wrong_sign,
                selection,
                focused_rival_id=10,
                stable_bit_ids=stable_ids,
                deadline=time.monotonic() + 60.0,
            )

    def test_more_permissive_profiles_are_rejected_before_fraction(self):
        build, rivals, selection, stable_ids = _inputs()
        cases = (
            replace(
                PCOHBuildOnlyCaps(),
                max_selected_generator_nonzeros=1_000_000_000,
            ),
            replace(
                PCOHBuildOnlyCaps(),
                max_source_payload_bytes=10**12,
            ),
            replace(
                PCOHBuildOnlyCaps(),
                static_additional_rss_budget_bytes=1,
            ),
        )
        for caps in cases:
            with self.subTest(caps=caps):
                with mock.patch.object(
                    build_only,
                    "_build_complete_operator_phase_conditioned_objective_bounds_until",
                    side_effect=AssertionError("Fraction producer must not start"),
                ) as producer:
                    with self.assertRaises(PhaseConditionedBuildOnlyError):
                        run_phase_conditioned_objective_hull_build_only(
                            build,
                            rivals,
                            selection,
                            focused_rival_id=10,
                            stable_bit_ids=stable_ids,
                            deadline=time.monotonic() + 60.0,
                            caps=caps,
                        )
                producer.assert_not_called()

    def test_live_resource_stop_loss_runs_before_fraction(self):
        build, rivals, selection, stable_ids = _inputs()
        failed_snapshot = MappingProxyType(
            {
                "current_rss_bytes": 5 * 1024**3 // 2,
                "peak_rss_bytes": 5 * 1024**3 // 2,
                "mem_available_bytes": 10 * 1024**3,
                "cgroup_limit_status": "unbounded",
                "cgroup_headroom_bytes": None,
                "measurement_source": "mock_live_kernel",
                "caller_supplied": False,
            }
        )
        with mock.patch.object(
            build_only, "_live_resource_snapshot", return_value=failed_snapshot
        ), mock.patch.object(
            build_only,
            "_build_complete_operator_phase_conditioned_objective_bounds_until",
            side_effect=AssertionError("Fraction producer must not start"),
        ) as producer:
            with self.assertRaisesRegex(
                PhaseConditionedBuildOnlyError,
                "resource_preflight_stop_loss",
            ):
                run_phase_conditioned_objective_hull_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    deadline=time.monotonic() + 60.0,
                )
        producer.assert_not_called()

    def test_diagnostic_receipt_and_top_level_tamper_fail_closed(self):
        _source, result = _run()
        with mock.patch.object(
            build_only.os, "getpid", return_value=build_only.os.getpid() + 1
        ):
            self.assertFalse(
                verify_phase_conditioned_objective_hull_build_only_diagnostic(
                    result
                )
            )
        tampered_receipt = dict(result.receipt)
        tampered_receipt["verdict_authority"] = True
        tampered = replace(
            result,
            receipt=MappingProxyType(tampered_receipt),
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(tampered)
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(
                replace(result, fresh_semantic_digest="0" * 64)
            )
        )
        summary_body = dict(result.materialized_tightness_summary)
        summary_body["active_pattern_mask"] = (True, True, True, True)
        summary_body.pop("summary_sha256")
        summary_body["summary_sha256"] = build_only._canonical_sha256(
            summary_body
        )
        tampered_summary = MappingProxyType(summary_body)
        with self.assertRaisesRegex(
            PhaseConditionedBuildOnlyError, "pattern_.*invalid"
        ):
            build_only._strict_materialized_tightness_payload(
                tampered_summary,
                source_semantic_digest=result.source_semantic_digest,
                stable_bit_ids=result.stable_bit_ids,
                conditional_certificate_sha256=(
                    result.conditional_certificate_sha256
                ),
                adapter_candidate_sha256=result.receipt[
                    "fresh_adapter_candidate_sha256"
                ],
                descriptor_representation_sha256=result.receipt[
                    "fresh_descriptor_representation_sha256"
                ],
                row_frame_sha256=result.receipt["fresh_row_frame_sha256"],
            )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(
                replace(
                    result,
                    materialized_tightness_summary=tampered_summary,
                )
            )
        )
        object.__setattr__(result, "_hidden_fresh_hz", object())
        self.assertFalse(
            verify_phase_conditioned_objective_hull_build_only_diagnostic(result)
        )

    def test_strict_tightness_failure_discards_registry_and_sanitizes_traceback(self):
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_hull_fresh_materializer as fresh,
        )

        for failure in (False, KeyboardInterrupt("strict verifier interrupted")):
            with self.subTest(failure=type(failure).__name__):
                build, rivals, selection, stable_ids = _inputs()
                caught = None
                kwargs = (
                    {"return_value": False}
                    if failure is False
                    else {"side_effect": failure}
                )
                with mock.patch.object(
                    build_only,
                    "verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness",
                    **kwargs,
                ) as verifier, mock.patch.object(
                    build_only,
                    "consume_live_phase_conditioned_objective_hull_fresh_build",
                    side_effect=AssertionError("consume must not start"),
                ) as consume:
                    try:
                        run_phase_conditioned_objective_hull_build_only(
                            build,
                            rivals,
                            selection,
                            focused_rival_id=10,
                            stable_bit_ids=stable_ids,
                            deadline=time.monotonic() + 60.0,
                        )
                    except PhaseConditionedBuildOnlyError as exc:
                        caught = exc
                verifier.assert_called_once()
                consume.assert_not_called()
                self.assertIsNotNone(caught)
                self.assertFalse(fresh._REGISTRY)
                traceback_cursor = caught.__traceback__
                while traceback_cursor is not None:
                    frame = traceback_cursor.tb_frame
                    if frame.f_code.co_filename == build_only.__file__:
                        self.assertNotIn("fresh_build", frame.f_locals)
                        self.assertNotIn("record", frame.f_locals)
                        self.assertNotIn("issuance", frame.f_locals)
                    traceback_cursor = traceback_cursor.tb_next

    def test_interrupted_public_discard_uses_captured_cleanup_authority(self):
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_hull_fresh_materializer as fresh,
        )

        build, rivals, selection, stable_ids = _inputs()
        caught = None
        real_cleanup_authority = build_only._FRESH_DISCARD_CLEANUP_AUTHORITY
        with mock.patch.object(
            build_only,
            "verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness",
            return_value=False,
        ) as verifier, mock.patch.object(
            build_only,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=AssertionError("consume must not start"),
        ) as consume, mock.patch.object(
            build_only,
            "discard_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=KeyboardInterrupt("public cleanup interrupted"),
        ) as public_discard, mock.patch.object(
            build_only,
            "_FRESH_DISCARD_CLEANUP_AUTHORITY",
            wraps=real_cleanup_authority,
        ) as fallback_discard:
            try:
                run_phase_conditioned_objective_hull_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    deadline=time.monotonic() + 60.0,
                )
            except PhaseConditionedBuildOnlyError as exc:
                caught = exc
        verifier.assert_called_once()
        consume.assert_not_called()
        public_discard.assert_called_once()
        fallback_discard.assert_called_once()
        self.assertIsNotNone(caught)
        self.assertIn("cleanup_recovered", str(caught))
        self.assertFalse(fresh._REGISTRY)
        traceback_cursor = caught.__traceback__
        while traceback_cursor is not None:
            frame = traceback_cursor.tb_frame
            if frame.f_code.co_filename == build_only.__file__:
                self.assertNotIn("fresh_build", frame.f_locals)
                self.assertNotIn("record", frame.f_locals)
                self.assertNotIn("issuance", frame.f_locals)
            traceback_cursor = traceback_cursor.tb_next

    def test_post_issue_failure_consumes_registry_record(self):
        build, rivals, selection, stable_ids = _inputs()
        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_hull_fresh_materializer as fresh,
        )

        real_issue = build_only.issue_live_phase_conditioned_objective_hull_fresh_build

        def stale_issuance(*args, **kwargs):
            issuance = real_issue(*args, **kwargs)
            bad_receipt = dict(issuance.receipt)
            bad_receipt["source_payload_bytes"] += 1
            return replace(issuance, receipt=MappingProxyType(bad_receipt))

        with mock.patch.object(
            build_only,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=stale_issuance,
        ):
            with self.assertRaises(PhaseConditionedBuildOnlyError):
                run_phase_conditioned_objective_hull_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    deadline=time.monotonic() + 60.0,
                )
        self.assertFalse(fresh._REGISTRY)

    def test_expired_consume_cleans_registry_and_traceback_has_no_private_hz(self):
        build, rivals, selection, stable_ids = _inputs()
        real_consume = (
            build_only.consume_live_phase_conditioned_objective_hull_fresh_build
        )

        def expired_consume(issuance, capability, *, deadline):
            return real_consume(
                issuance,
                capability,
                deadline=time.monotonic() - 1.0,
            )

        from act.back_end.hybridz_tf import (
            operator_phase_conditioned_objective_hull_fresh_materializer as fresh,
        )

        caught = None
        with mock.patch.object(
            build_only,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=expired_consume,
        ):
            try:
                run_phase_conditioned_objective_hull_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=stable_ids,
                    deadline=time.monotonic() + 60.0,
                )
            except PhaseConditionedBuildOnlyError as exc:
                caught = exc
        self.assertIsNotNone(caught)
        self.assertFalse(fresh._REGISTRY)
        traceback_cursor = caught.__traceback__
        while traceback_cursor is not None:
            frame = traceback_cursor.tb_frame
            if frame.f_code.co_filename == build_only.__file__:
                self.assertNotIn("fresh_build", frame.f_locals)
                self.assertNotIn("record", frame.f_locals)
            traceback_cursor = traceback_cursor.tb_next

    def test_expired_deadline_fails_without_partial_result(self):
        build, rivals, selection, stable_ids = _inputs()
        with self.assertRaisesRegex(
            PhaseConditionedBuildOnlyError, "absolute_deadline_invalid"
        ):
            run_phase_conditioned_objective_hull_build_only(
                build,
                rivals,
                selection,
                focused_rival_id=10,
                stable_bit_ids=stable_ids,
                deadline=time.monotonic() - 1.0,
            )


if __name__ == "__main__":
    unittest.main()
