#!/usr/bin/env python3
"""Toy and tamper gates for the pair-first K3 build-only wrapper."""

from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import itertools
import time
import unittest
from unittest import mock

import act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only as k3_module
import act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_fresh_materializer as fresh_module

from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    ExactDyadicRivalCoefficient,
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
    _K3_STRONG_TARGET,
    PCOHK3BuildOnlyCaps,
    PCOHK3BuildOnlyDiagnostic,
    PCOHK3BuildOnlyResourceStopDiagnostic,
    PCOHK3BuildOnlyStopDiagnostic,
    PhaseConditionedK3BuildOnlyError,
    build_k3_pair_first_schedule,
    export_phase_conditioned_objective_hull_k3_build_only_detached,
    run_phase_conditioned_objective_hull_k3_build_only,
    select_k3_third_bit_plan,
    verify_detached_phase_conditioned_objective_hull_k3_build_only,
    verify_phase_conditioned_objective_hull_k3_build_only_outcome,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
    PairInfeasibilityBundle,
    PairLocalCaps,
    PatternCoverage,
    run_phase_conditioned_pair_infeasibility_candidate,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    OperatorPhaseConditionedScheduledStop,
    OperatorPhaseConditionedScheduledStopPolicy,
    build_scheduled_complete_operator_phase_conditioned_objective_bounds,
)
from act.back_end.hybridz_tf.test_operator_phase_conditioned_pair_infeasibility import (
    _exact_relu_build,
    _triple_only_build,
)
from act.back_end.hybridz_tf.test_operator_phase_conditioned_objective_bounds import (
    _exact_true_network_conditional_upper,
    _k4_corner_build,
    _rivals,
)


def _coverage(pattern, *, active):
    status = (
        "not_certified_empty"
        if active
        else "certified_empty_by_pair"
    )
    return PatternCoverage(
        pattern=tuple(pattern),
        status=status,
        witness_pair=(None if active else ((1, 1), (2, 1))),
        certificate_sha256=(None if active else "c" * 64),
        eta_fixed_value=(None if active else -1),
        coverage_sha256=("a" if active else "b") * 64,
        proof_authority=False,
    )


def _pair_bundle(active_patterns):
    canonical = tuple(itertools.product((-1, 1), repeat=3))
    active = set(active_patterns)
    return PairInfeasibilityBundle(
        status="complete",
        stable_bit_ids=(1, 2, 3),
        parent_semantic_digest="1" * 64,
        terminal_parent_semantic_digest="1" * 64,
        property_digest="2" * 64,
        selection_digest="3" * 64,
        operator_row_tag_digest="4" * 64,
        ordered_source_frame_sha256="5" * 64,
        caps=PairLocalCaps(max_stable_bits=3, max_signed_pair_queries=12),
        records=(),
        certificates=(),
        coverage=tuple(
            _coverage(pattern, active=pattern in active)
            for pattern in canonical
        ),
        receipt={},
        bundle_sha256="6" * 64,
        proof_authority=False,
    )


def _replace_focused_coefficient(mapping, value: Fraction):
    coefficients = tuple(
        ExactDyadicRivalCoefficient(
            rival_id=item.rival_id,
            numerator=(value.numerator if item.rival_id == 10 else item.numerator),
            denominator=(
                value.denominator if item.rival_id == 10 else item.denominator
            ),
        )
        for item in mapping.rival_coefficients
    )
    return replace(mapping, rival_coefficients=coefficients)


class K3ThirdBitRankingToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()
        cls.rivals = _rivals()
        cls.selection = derive_operator_exact_relu_property_phase_literals(
            cls.build, cls.rivals
        )
        cls.ids = tuple(
            mapping.stable_bcol_id for mapping in cls.selection.mappings
        )

    def _selection_with_remaining(self, left: Fraction, right: Fraction):
        mappings = list(self.selection.mappings)
        mappings[2] = _replace_focused_coefficient(mappings[2], left)
        mappings[3] = _replace_focused_coefficient(mappings[3], right)
        return replace(self.selection, mappings=tuple(mappings))

    def test_exact_magnitude_descending_and_stable_id_tie(self):
        selection = self._selection_with_remaining(
            Fraction(7, 16), Fraction(-7, 16)
        )
        plan = select_k3_third_bit_plan(
            selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=self.ids[:2],
        )
        self.assertEqual(plan.third_stable_bit_id, self.ids[2])
        self.assertEqual(plan.third_coefficient, Fraction(7, 16))
        self.assertEqual(plan.preferred_third_phase, 1)
        self.assertEqual(
            tuple(stable_id for stable_id, _ in plan.ranking),
            self.ids[2:],
        )
        self.assertEqual(plan.stable_bit_ids[:2], self.ids[:2])

    def test_negative_winner_and_zero_positive_tie(self):
        negative = self._selection_with_remaining(
            Fraction(1, 3), Fraction(-17, 32)
        )
        plan = select_k3_third_bit_plan(
            negative,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=self.ids[:2],
        )
        self.assertEqual(plan.third_stable_bit_id, self.ids[3])
        self.assertEqual(plan.preferred_third_phase, -1)
        self.assertEqual(
            plan.preferred_phase_source,
            "negative_exact_focused_coefficient",
        )

        zero = self._selection_with_remaining(Fraction(0), Fraction(0))
        plan = select_k3_third_bit_plan(
            zero,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=self.ids[:2],
        )
        self.assertEqual(plan.third_stable_bit_id, self.ids[2])
        self.assertEqual(plan.preferred_third_phase, 1)
        self.assertEqual(
            plan.preferred_phase_source,
            "zero_coefficient_positive_tie",
        )

    def test_noncanonical_retained_pair_fails_closed(self):
        with self.assertRaisesRegex(
            PhaseConditionedK3BuildOnlyError,
            "lowest_verified_pair",
        ):
            select_k3_third_bit_plan(
                self.selection,
                focused_rival_id=10,
                retained_k2_stable_bit_ids=self.ids[1:3],
            )

    def test_fixed_caps_are_snapshotted_against_caller_aba(self):
        caller_caps = PCOHK3BuildOnlyCaps()
        internal = k3_module._normalize_caps(caller_caps)
        self.assertIsNot(internal, caller_caps)
        object.__setattr__(caller_caps, "max_fresh_payload_bytes", 1)
        self.assertEqual(internal, PCOHK3BuildOnlyCaps())

    def test_fixed_caps_reject_recursively_equal_numeric_type_aliases(self):
        fixed = PCOHK3BuildOnlyCaps()
        attacks = (
            replace(fixed, max_stable_bits=3.0),
            replace(fixed, candidate_timeout_seconds=1),
            replace(
                fixed,
                pair_caps=replace(fixed.pair_caps, max_local_rows=6.0),
            ),
            replace(
                fixed,
                fresh_caps=replace(
                    fixed.fresh_caps, max_registry_entries=True
                ),
            ),
            replace(
                fixed,
                fresh_caps=replace(
                    fixed.fresh_caps, capability_ttl_seconds=15
                ),
            ),
            replace(
                fixed,
                fresh_caps=replace(
                    fixed.fresh_caps,
                    row_caps=replace(
                        fixed.fresh_caps.row_caps, max_eta_columns=8.0
                    ),
                ),
            ),
        )
        for index, caps in enumerate(attacks):
            with self.subTest(index=index), self.assertRaisesRegex(
                PhaseConditionedK3BuildOnlyError,
                "field_type_or_shape_not_exact",
            ):
                k3_module._normalize_caps(caps)


class K3PairFirstScheduleToyTests(unittest.TestCase):
    canonical = tuple(itertools.product((-1, 1), repeat=3))

    def test_both_worst_children_active_are_first_preferred_then_opposite(self):
        active = set(self.canonical)
        plan = build_k3_pair_first_schedule(
            _pair_bundle(active), preferred_third_phase=-1
        )
        self.assertEqual(plan.evaluation_schedule[:2], ((1, 1, -1), (1, 1, 1)))
        self.assertEqual(plan.threshold_pattern_indices, tuple(range(8)))
        self.assertTrue(all(plan.active_pattern_mask))

    def test_one_active_child_is_first_and_empty_sibling_is_second(self):
        active = set(self.canonical)
        active.remove((1, 1, 1))
        plan = build_k3_pair_first_schedule(
            _pair_bundle(active), preferred_third_phase=1
        )
        self.assertEqual(plan.evaluation_schedule[0], (1, 1, -1))
        self.assertEqual(plan.evaluation_schedule[1], (1, 1, 1))
        self.assertIn(0, plan.threshold_pattern_indices)
        self.assertNotIn(1, plan.threshold_pattern_indices)

    def test_both_empty_children_still_precede_remaining_active_patterns(self):
        active = set(self.canonical)
        active.remove((1, 1, -1))
        active.remove((1, 1, 1))
        plan = build_k3_pair_first_schedule(
            _pair_bundle(active), preferred_third_phase=1
        )
        self.assertEqual(plan.evaluation_schedule[:2], ((1, 1, 1), (1, 1, -1)))
        self.assertNotIn(0, plan.threshold_pattern_indices)
        self.assertNotIn(1, plan.threshold_pattern_indices)
        self.assertEqual(set(plan.evaluation_schedule), set(self.canonical))

    def test_threshold_is_exact_fixed_target_and_all_active_indices(self):
        active = {(-1, -1, -1), (1, 1, 1), (-1, 1, -1)}
        plan = build_k3_pair_first_schedule(
            _pair_bundle(active), preferred_third_phase=1
        )
        self.assertEqual(
            plan.stop_policy.strict_upper_threshold,
            Fraction(191135223185129307, 1759218604441600),
        )
        self.assertEqual(plan.stop_policy.strict_upper_threshold, _K3_STRONG_TARGET)
        self.assertEqual(
            set(plan.threshold_pattern_indices),
            {
                index
                for index, pattern in enumerate(plan.evaluation_schedule)
                if pattern in active
            },
        )

    def test_all_empty_fails_without_claiming_verdict(self):
        with self.assertRaisesRegex(
            PhaseConditionedK3BuildOnlyError,
            "no_verdict",
        ):
            build_k3_pair_first_schedule(
                _pair_bundle(set()), preferred_third_phase=1
            )


class K3BuildOnlyEndToEndToyTests(unittest.TestCase):
    @staticmethod
    def _inputs(*, high_margin=False):
        build = _k4_corner_build()
        if high_margin:
            build.hz.c[1] += 200.0
        rivals = _rivals()
        selection = derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
        ids = tuple(
            mapping.stable_bcol_id for mapping in selection.mappings
        )
        return build, rivals, selection, ids

    @staticmethod
    def _passing_resource_snapshot():
        return {
            "current_rss_bytes": 0,
            "peak_rss_bytes": 0,
            "mem_available_bytes": 2 * 1024 * 1024 * 1024,
            "cgroup_limit_status": "unbounded",
            "cgroup_headroom_bytes": None,
            "measurement_source": (
                "live_proc_status_meminfo_and_cgroup_v2"
            ),
            "caller_supplied": False,
        }

    @staticmethod
    def _fresh_registry_state():
        with fresh_module._REGISTRY_LOCK:
            return (
                len(fresh_module._REGISTRY),
                len(fresh_module._REGISTRY_RESERVATIONS),
            )

    @classmethod
    def setUpClass(cls):
        build, rivals, selection, ids = cls._inputs()
        cls.success = run_phase_conditioned_objective_hull_k3_build_only(
            build,
            rivals,
            selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=ids[:2],
            deadline=time.monotonic() + 30.0,
        )

    def test_complete_toy_has_exact_dynamic_shape_counts_and_no_authority(self):
        result = self.success
        self.assertIs(type(result), PCOHK3BuildOnlyDiagnostic)
        self.assertTrue(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                result
            )
        )
        self.assertFalse(hasattr(result, "hz"))
        self.assertFalse(hasattr(result, "build"))
        self.assertFalse(result.ground_truth_loaded)
        self.assertFalse(result.full_parent_lp_called)
        self.assertFalse(result.proof_authority)
        self.assertFalse(result.verdict_authority)
        empty_count = result.active_pattern_mask.count(False)
        self.assertEqual(result.fresh_dimensions[1], result.source_dimensions[1] + 8)
        self.assertEqual(
            result.fresh_dimensions[3],
            result.source_dimensions[3] + 4 + empty_count,
        )
        self.assertEqual(result.fresh_dimensions[4], result.source_dimensions[4] + 1)
        telemetry = result.execution_telemetry
        self.assertEqual(telemetry["pair_local_lp_actual_calls"], 12)
        self.assertEqual(telemetry["local_lp_actual_calls"], 20)
        self.assertEqual(telemetry["local_lp_actual_call_cap"], 20)
        self.assertEqual(
            telemetry["conditional_checker_actual_call_cap"], 34
        )
        self.assertLessEqual(telemetry["conditional_checker_actual_calls"], 34)
        self.assertEqual(
            telemetry["conditional_checker_actual_calls"],
            telemetry["scheduled_producer_checker_actual_calls"]
            + telemetry["fresh_live_replay_checker_actual_calls"],
        )
        self.assertFalse(
            result.receipt["fresh_live_verifier_valid_after_consume"]
        )

        summary = result.materialized_tightness_summary
        for index, pattern in enumerate(summary["canonical_patterns"]):
            oracle = _exact_true_network_conditional_upper(3, tuple(pattern))
            if result.active_pattern_mask[index]:
                self.assertIsNotNone(oracle)
                stored = Fraction.from_float(
                    float.fromhex(summary["pattern_upper_hex"][index])
                )
                self.assertGreaterEqual(stored, oracle)

    def test_stop_ignores_high_empty_children_then_stops_on_first_active(self):
        build, rivals, selection, ids = self._inputs(high_margin=True)
        result = run_phase_conditioned_objective_hull_k3_build_only(
            build,
            rivals,
            selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=ids[:2],
            deadline=time.monotonic() + 30.0,
        )
        self.assertIs(type(result), PCOHK3BuildOnlyStopDiagnostic)
        self.assertTrue(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                result
            )
        )
        self.assertEqual(result.triggering_schedule_index, 2)
        self.assertFalse(result.active_pattern_mask[
            tuple(itertools.product((-1, 1), repeat=3)).index(
                result.evaluation_schedule[0]
            )
        ])
        self.assertFalse(result.active_pattern_mask[
            tuple(itertools.product((-1, 1), repeat=3)).index(
                result.evaluation_schedule[1]
            )
        ])
        observed = result.receipt["scheduled_stop_record"]["telemetry"][
            "observed_upper_exact_in_execution_order"
        ]
        self.assertGreater(Fraction(*observed[0]), _K3_STRONG_TARGET)
        self.assertGreater(Fraction(*observed[1]), _K3_STRONG_TARGET)
        self.assertGreater(Fraction(*observed[2]), _K3_STRONG_TARGET)
        self.assertFalse(result.partial_certificates_returned)
        self.assertFalse(result.fresh_issue_called)
        self.assertFalse(result.fresh_build_returned)
        self.assertEqual(
            result.execution_telemetry["fresh_live_replay_checker_actual_calls"],
            0,
        )

    def test_first_and_second_active_child_stops_and_exact_equality_continues(self):
        first_build = _triple_only_build()
        first_build.hz.c[1] += 200.0
        rivals = _rivals()
        first_selection = derive_operator_exact_relu_property_phase_literals(
            first_build, rivals
        )
        first_ids = tuple(
            mapping.stable_bcol_id
            for mapping in first_selection.mappings
        )
        first = run_phase_conditioned_objective_hull_k3_build_only(
            first_build,
            rivals,
            first_selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=first_ids[:2],
            deadline=time.monotonic() + 30.0,
        )
        self.assertIs(type(first), PCOHK3BuildOnlyStopDiagnostic)
        self.assertEqual(first.triggering_schedule_index, 0)
        self.assertIn(0, first.threshold_pattern_indices)

        second_build = _exact_relu_build(
            ((1.0,), (1.0,), (-1.0,)),
            (0.0, 0.0, 0.9),
        )
        second_build.hz.c[1] += 107.5
        second_selection = derive_operator_exact_relu_property_phase_literals(
            second_build, rivals
        )
        second_ids = tuple(
            mapping.stable_bcol_id
            for mapping in second_selection.mappings
        )
        second = run_phase_conditioned_objective_hull_k3_build_only(
            second_build,
            rivals,
            second_selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=second_ids[:2],
            deadline=time.monotonic() + 30.0,
        )
        self.assertIs(type(second), PCOHK3BuildOnlyStopDiagnostic)
        self.assertEqual(second.triggering_schedule_index, 1)
        observed = second.receipt["scheduled_stop_record"]["telemetry"][
            "observed_upper_exact_in_execution_order"
        ]
        self.assertLessEqual(Fraction(*observed[0]), _K3_STRONG_TARGET)
        self.assertGreater(Fraction(*observed[1]), _K3_STRONG_TARGET)

        # Replay the same pair-first schedule with its first exact dyadic
        # observation as the threshold: equality must continue to child two.
        plan = select_k3_third_bit_plan(
            second_selection,
            focused_rival_id=10,
            retained_k2_stable_bit_ids=second_ids[:2],
        )
        deadline = time.monotonic() + 30.0
        pair = run_phase_conditioned_pair_infeasibility_candidate(
            second_build,
            rivals,
            second_selection,
            stable_bit_ids=plan.stable_bit_ids,
            deadline=deadline,
            caps=PairLocalCaps(
                max_stable_bits=3,
                max_signed_pair_queries=12,
                max_local_rows=6,
                max_local_nonzeros=200_000,
                max_source_terms=6,
                max_multiplier_bits=256,
                max_exact_bits=4096,
                max_exact_nonzeros=200_000,
            ),
        )
        pair_first = build_k3_pair_first_schedule(
            pair, preferred_third_phase=plan.preferred_third_phase
        )
        equality_policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction(*observed[0]),
            threshold_pattern_indices=pair_first.threshold_pattern_indices,
        )
        with self.assertRaises(OperatorPhaseConditionedScheduledStop) as caught:
            build_scheduled_complete_operator_phase_conditioned_objective_bounds(
                second_build,
                rivals,
                second_selection,
                focused_rival_id=10,
                stable_bit_ids=plan.stable_bit_ids,
                evaluation_schedule=pair_first.evaluation_schedule,
                deadline=deadline,
                stop_policy=equality_policy,
                candidate_timeout_seconds=1.0,
            )
        self.assertEqual(caught.exception.record.triggering_schedule_index, 1)
        equality_trace = caught.exception.record.telemetry[
            "observed_upper_exact_in_execution_order"
        ]
        self.assertEqual(equality_trace[0], equality_policy.strict_upper_threshold)
        self.assertGreater(equality_trace[1], equality_policy.strict_upper_threshold)

    def test_pre_scheduled_resource_stop_is_sealed_without_partial_output(self):
        build, rivals, selection, ids = self._inputs()
        passing = self._passing_resource_snapshot()
        rejected = {**passing, "mem_available_bytes": 0}
        registry_before = self._fresh_registry_state()
        with mock.patch.object(
            k3_module,
            "_live_resource_snapshot",
            side_effect=[passing, rejected],
        ), mock.patch.object(
            k3_module,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
        ) as fresh_issue:
            result = run_phase_conditioned_objective_hull_k3_build_only(
                build,
                rivals,
                selection,
                focused_rival_id=10,
                retained_k2_stable_bit_ids=ids[:2],
                deadline=time.monotonic() + 30.0,
            )
        fresh_issue.assert_not_called()
        self.assertIs(type(result), PCOHK3BuildOnlyResourceStopDiagnostic)
        self.assertEqual(result.stage, "pre_scheduled")
        self.assertEqual(
            result.reason,
            "resource_preflight_stop_loss:"
            "mem_available_at_least_fixed_reserve",
        )
        self.assertTrue(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                result
            )
        )
        self.assertIsNone(result.scheduled_bundle_sha256)
        self.assertEqual(result.completed_conditional_certificate_count, 0)
        self.assertEqual(
            result.execution_telemetry["pair_local_lp_actual_calls"], 12
        )
        self.assertEqual(
            result.execution_telemetry["scheduled_local_lp_actual_calls"], 0
        )
        self.assertEqual(
            result.execution_telemetry["local_lp_actual_calls"], 12
        )
        self.assertEqual(
            result.execution_telemetry["conditional_checker_actual_calls"], 0
        )
        for name in (
            "conditional_certificate_sha256",
            "fresh_issuance_sha256",
            "fresh_semantic_digest",
            "fresh_dimensions",
            "materialized_tightness_summary",
        ):
            self.assertFalse(hasattr(result, name))
            self.assertNotIn(name, result.receipt)
        self.assertFalse(result.partial_certificates_returned)
        self.assertFalse(result.conditional_certificate_payload_returned)
        self.assertFalse(result.fresh_issue_called)
        self.assertFalse(result.fresh_build_returned)
        self.assertFalse(result.fresh_descriptor_returned)
        self.assertFalse(result.provenance_authority)
        self.assertFalse(result.authenticity_authority)
        self.assertEqual(
            tuple(result.receipt["fresh_registry_state_before"]),
            registry_before,
        )
        self.assertEqual(
            tuple(result.receipt["fresh_registry_state_terminal"]),
            registry_before,
        )
        self.assertEqual(self._fresh_registry_state(), registry_before)
        detached = export_phase_conditioned_objective_hull_k3_build_only_detached(
            result
        )
        self.assertTrue(
            verify_detached_phase_conditioned_objective_hull_k3_build_only(
                detached, expected_sha256=result.resource_stop_sha256
            )
        )

    def test_pre_fresh_resource_stop_preserves_counts_not_certificate_descriptors(self):
        build, rivals, selection, ids = self._inputs()
        passing = self._passing_resource_snapshot()
        rejected = {**passing, "cgroup_limit_status": "bounded", "cgroup_headroom_bytes": 0}
        registry_before = self._fresh_registry_state()
        with mock.patch.object(
            k3_module,
            "_live_resource_snapshot",
            side_effect=[passing, passing, rejected],
        ), mock.patch.object(
            k3_module,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
        ) as fresh_issue:
            result = run_phase_conditioned_objective_hull_k3_build_only(
                build,
                rivals,
                selection,
                focused_rival_id=10,
                retained_k2_stable_bit_ids=ids[:2],
                deadline=time.monotonic() + 30.0,
            )
        fresh_issue.assert_not_called()
        self.assertIs(type(result), PCOHK3BuildOnlyResourceStopDiagnostic)
        self.assertEqual(result.stage, "pre_fresh_materialization")
        self.assertTrue(k3_module._valid_sha256(result.scheduled_bundle_sha256))
        self.assertEqual(result.completed_conditional_certificate_count, 8)
        telemetry = result.execution_telemetry
        self.assertEqual(telemetry["pair_local_lp_actual_calls"], 12)
        self.assertEqual(telemetry["scheduled_local_lp_actual_calls"], 8)
        self.assertEqual(telemetry["local_lp_actual_calls"], 20)
        self.assertEqual(telemetry["scheduled_patterns_completed"], 8)
        self.assertEqual(
            telemetry["conditional_checker_actual_calls"],
            telemetry["scheduled_producer_checker_actual_calls"],
        )
        self.assertEqual(telemetry["fresh_live_replay_checker_actual_calls"], 0)
        self.assertTrue(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                result
            )
        )
        for name in (
            "conditional_certificate_sha256",
            "conditional_certificate_payload",
            "fresh_issuance_sha256",
            "fresh_semantic_digest",
            "fresh_dimensions",
            "materialized_tightness_summary",
            "descriptor_representation_sha256",
        ):
            self.assertNotIn(name, result.receipt)
        self.assertEqual(
            tuple(result.receipt["fresh_registry_state_before"]),
            registry_before,
        )
        self.assertEqual(
            tuple(result.receipt["fresh_registry_state_terminal"]),
            registry_before,
        )
        self.assertEqual(self._fresh_registry_state(), registry_before)

        coherent_clone = replace(result)
        self.assertFalse(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                coherent_clone
            )
        )
        detached = export_phase_conditioned_objective_hull_k3_build_only_detached(
            result
        )
        detached["completed_conditional_certificate_count"] = 7
        detached["receipt"]["completed_conditional_certificate_count"] = 7
        receipt_body = dict(detached["receipt"])
        receipt_body.pop("receipt_sha256")
        detached["receipt"]["receipt_sha256"] = k3_module._canonical_sha256(
            receipt_body
        )
        outcome_body = dict(detached)
        outcome_body.pop("resource_stop_sha256")
        coherent_digest = k3_module._canonical_sha256(outcome_body)
        detached["resource_stop_sha256"] = coherent_digest
        self.assertFalse(
            verify_detached_phase_conditioned_objective_hull_k3_build_only(
                detached, expected_sha256=coherent_digest
            )
        )

    def test_resource_stop_source_toctou_and_baseexception_fail_closed(self):
        passing = self._passing_resource_snapshot()
        rejected = {**passing, "mem_available_bytes": 0}

        build, rivals, selection, ids = self._inputs()
        original_gate = k3_module._observable_resource_gate

        def mutate_after_rejection(snapshot, *, stage, caps):
            outcome = original_gate(snapshot, stage=stage, caps=caps)
            if stage == "pre_scheduled" and outcome[1] is not None:
                build.hz.c[0] += 1.0
            return outcome

        registry_before = self._fresh_registry_state()
        with mock.patch.object(
            k3_module,
            "_live_resource_snapshot",
            side_effect=[passing, rejected],
        ), mock.patch.object(
            k3_module,
            "_observable_resource_gate",
            side_effect=mutate_after_rejection,
        ), mock.patch.object(
            k3_module,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
        ) as fresh_issue:
            with self.assertRaisesRegex(
                PhaseConditionedK3BuildOnlyError,
                "terminal_source_digest_changed_on_resource_stop",
            ):
                run_phase_conditioned_objective_hull_k3_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    retained_k2_stable_bit_ids=ids[:2],
                    deadline=time.monotonic() + 30.0,
                )
        fresh_issue.assert_not_called()
        self.assertEqual(self._fresh_registry_state(), registry_before)

        build, rivals, selection, ids = self._inputs()
        original_preflight = k3_module._resource_preflight
        preflight_calls = 0

        def interrupt_second_gate(**kwargs):
            nonlocal preflight_calls
            preflight_calls += 1
            if preflight_calls == 2:
                raise KeyboardInterrupt("resource gate interrupted")
            return original_preflight(**kwargs)

        registry_before = self._fresh_registry_state()
        with mock.patch.object(
            k3_module,
            "_live_resource_snapshot",
            side_effect=[passing, passing],
        ), mock.patch.object(
            k3_module,
            "_resource_preflight",
            side_effect=interrupt_second_gate,
        ), mock.patch.object(
            k3_module,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
        ) as fresh_issue:
            with self.assertRaisesRegex(
                PhaseConditionedK3BuildOnlyError,
                "k3_transaction_failed_closed:KeyboardInterrupt",
            ):
                run_phase_conditioned_objective_hull_k3_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    retained_k2_stable_bit_ids=ids[:2],
                    deadline=time.monotonic() + 30.0,
                )
        fresh_issue.assert_not_called()
        self.assertEqual(self._fresh_registry_state(), registry_before)

    def test_resource_stop_coherent_rehash_rejects_all_numeric_type_aliases(self):
        def detached_resource_stop(*, pre_fresh):
            build, rivals, selection, ids = self._inputs()
            passing = self._passing_resource_snapshot()
            rejected = {**passing, "mem_available_bytes": 0}
            snapshots = (
                [passing, passing, rejected]
                if pre_fresh
                else [passing, rejected]
            )
            with mock.patch.object(
                k3_module,
                "_live_resource_snapshot",
                side_effect=snapshots,
            ):
                result = run_phase_conditioned_objective_hull_k3_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    retained_k2_stable_bit_ids=ids[:2],
                    deadline=time.monotonic() + 30.0,
                )
            return export_phase_conditioned_objective_hull_k3_build_only_detached(
                result
            )

        def coherently_rehash(payload):
            receipt = payload["receipt"]
            receipt_body = dict(receipt)
            receipt_body.pop("receipt_sha256")
            receipt["receipt_sha256"] = k3_module._canonical_sha256(
                receipt_body
            )
            outcome_body = dict(payload)
            outcome_body.pop("resource_stop_sha256")
            digest = k3_module._canonical_sha256(outcome_body)
            payload["resource_stop_sha256"] = digest
            return digest

        def telemetry_alias(payload, name, value):
            payload["execution_telemetry"][name] = value
            payload["receipt"]["execution_telemetry"][name] = value

        def reverse_schedule(payload):
            canonical = tuple(itertools.product((-1, 1), repeat=3))
            schedule = list(reversed(payload["evaluation_schedule"]))
            active = payload["active_pattern_mask"]
            threshold = [
                index
                for index, pattern in enumerate(schedule)
                if active[canonical.index(tuple(pattern))]
            ]
            payload["evaluation_schedule"] = schedule
            payload["threshold_pattern_indices"] = threshold
            payload["receipt"]["evaluation_schedule"] = copy.deepcopy(
                schedule
            )
            payload["receipt"]["threshold_pattern_indices"] = list(
                threshold
            )

        def make_all_patterns_empty(payload):
            payload["active_pattern_mask"] = [False] * 8
            payload["threshold_pattern_indices"] = []
            payload["receipt"]["active_pattern_mask"] = [False] * 8
            payload["receipt"]["threshold_pattern_indices"] = []
            payload["receipt"]["certified_empty_pattern_count"] = 8

        def make_impossible_diagonal_single_pair_cover(payload):
            canonical = tuple(itertools.product((-1, 1), repeat=3))
            active = [True] * 8
            active[0] = False
            active[-1] = False
            preferred = payload["preferred_third_phase"]
            children = ((1, 1, preferred), (1, 1, -preferred))
            active_by_pattern = dict(zip(canonical, active))
            schedule = tuple(
                pattern
                for group in (
                    tuple(
                        pattern
                        for pattern in children
                        if active_by_pattern[pattern]
                    ),
                    tuple(
                        pattern
                        for pattern in children
                        if not active_by_pattern[pattern]
                    ),
                    tuple(
                        pattern
                        for pattern in canonical
                        if pattern not in children
                        and active_by_pattern[pattern]
                    ),
                    tuple(
                        pattern
                        for pattern in canonical
                        if pattern not in children
                        and not active_by_pattern[pattern]
                    ),
                )
                for pattern in group
            )
            threshold = tuple(
                index
                for index, pattern in enumerate(schedule)
                if active_by_pattern[pattern]
            )
            schedule_body = {
                "schema": k3_module._SCHEDULE_SCHEMA,
                "canonical_patterns": canonical,
                "active_pattern_mask": tuple(active),
                "active_means": (
                    "not_certified_empty_by_exact_signed_pair"
                ),
                "evaluation_schedule": schedule,
                "threshold_pattern_indices": threshold,
                "worst_k2_children": children,
                "strict_upper_threshold_exact": k3_module._fraction_pair(
                    _K3_STRONG_TARGET
                ),
                "strict_comparison": "observed_upper_exact_gt_target",
            }
            payload["active_pattern_mask"] = list(active)
            payload["evaluation_schedule"] = [list(item) for item in schedule]
            payload["threshold_pattern_indices"] = list(threshold)
            payload["receipt"]["active_pattern_mask"] = list(active)
            payload["receipt"]["evaluation_schedule"] = [
                list(item) for item in schedule
            ]
            payload["receipt"]["threshold_pattern_indices"] = list(
                threshold
            )
            payload["receipt"]["certified_empty_pattern_count"] = 2
            payload["receipt"]["schedule_sha256"] = (
                k3_module._canonical_sha256(schedule_body)
            )
            telemetry_alias(
                payload, "pair_exact_conflict_certificates", 1
            )
            telemetry_alias(
                payload,
                "pair_exact_conflict_certificates_strictly_replayed",
                1,
            )

        def make_pass_peak_below_current(payload):
            resource = payload["receipt"]["resource_entry_preflight"]
            resource["current_rss_bytes"] = 1
            resource["peak_rss_bytes"] = 0
            resource["forecast_rss_bytes"] = (
                resource["static_additional_rss_budget_bytes"] + 1
            )

        def make_rejection_peak_below_current(payload):
            for name in (
                "resource_pre_s_preflight",
                "resource_gate_rejection",
            ):
                resource = payload["receipt"][name]
                resource["current_rss_bytes"] = 1
                resource["peak_rss_bytes"] = 0
                resource["forecast_rss_bytes"] = (
                    resource["static_additional_rss_budget_bytes"] + 1
                )
                body = dict(resource)
                body.pop("rejection_sha256")
                resource["rejection_sha256"] = (
                    k3_module._canonical_sha256(body)
                )

        pre_scheduled = detached_resource_stop(pre_fresh=False)
        pre_fresh = detached_resource_stop(pre_fresh=True)
        attacks = (
            (
                pre_scheduled,
                lambda value: (
                    value.__setitem__(
                        "completed_conditional_certificate_count", False
                    ),
                    value["receipt"].__setitem__(
                        "completed_conditional_certificate_count", False
                    ),
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"].__setitem__(
                    "fresh_registry_entries_created", False
                ),
            ),
            (
                pre_scheduled,
                lambda value: telemetry_alias(
                    value, "scheduled_local_lp_actual_calls", False
                ),
            ),
            (
                pre_scheduled,
                lambda value: (
                    telemetry_alias(
                        value, "pair_exact_conflict_certificates", 0
                    ),
                    telemetry_alias(
                        value,
                        "pair_exact_conflict_certificates_strictly_replayed",
                        0,
                    ),
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"].__setitem__(
                    "pair_query_count", 12.0
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"]["timings"].__setitem__(
                    "scheduled_producer_seconds", 0
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"]["timings"].__setitem__(
                    "scheduled_producer_seconds", -0.0
                ),
            ),
            (
                pre_scheduled,
                lambda value: (
                    value.__setitem__("preferred_third_phase", True),
                    value["receipt"].__setitem__(
                        "preferred_third_phase", True
                    ),
                ),
            ),
            (
                pre_scheduled,
                lambda value: (
                    value.__setitem__(
                        "third_coefficient_exact",
                        [
                            2 * item
                            for item in value["third_coefficient_exact"]
                        ],
                    ),
                    value["receipt"].__setitem__(
                        "third_coefficient_exact",
                        [
                            2 * item
                            for item in value["receipt"][
                                "third_coefficient_exact"
                            ]
                        ],
                    ),
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"][
                    "resource_entry_preflight"
                ].__setitem__("current_rss_bytes", False),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"]["caps"].__setitem__(
                    "candidate_timeout_seconds", 1
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"].__setitem__(
                    "selected_output_positions",
                    list(reversed(value["receipt"]["selected_output_positions"])),
                ),
            ),
            (
                pre_scheduled,
                lambda value: (
                    value.__setitem__(
                        "active_pattern_mask",
                        [int(item) for item in value["active_pattern_mask"]],
                    ),
                    value["receipt"].__setitem__(
                        "active_pattern_mask",
                        [
                            int(item)
                            for item in value["receipt"]["active_pattern_mask"]
                        ],
                    ),
                ),
            ),
            (
                pre_scheduled,
                lambda value: (
                    value["receipt"].__setitem__(
                        "fresh_registry_state_before", [False, False]
                    ),
                    value["receipt"].__setitem__(
                        "fresh_registry_state_terminal", [False, False]
                    ),
                ),
            ),
            (
                pre_fresh,
                lambda value: (
                    value.__setitem__(
                        "completed_conditional_certificate_count", 8.0
                    ),
                    value["receipt"].__setitem__(
                        "completed_conditional_certificate_count", 8.0
                    ),
                ),
            ),
            (
                pre_fresh,
                lambda value: telemetry_alias(
                    value, "fresh_live_replay_checker_actual_calls", False
                ),
            ),
            (
                pre_fresh,
                lambda value: telemetry_alias(
                    value, "scheduled_patterns_completed", 8.0
                ),
            ),
            (
                pre_fresh,
                lambda value: telemetry_alias(
                    value, "scheduled_actual_call_site_counters", 1
                ),
            ),
            (
                pre_fresh,
                lambda value: value["receipt"].__setitem__(
                    "scheduled_bundle_completed", 1
                ),
            ),
            (pre_fresh, reverse_schedule),
            (pre_scheduled, make_all_patterns_empty),
            (pre_fresh, make_impossible_diagonal_single_pair_cover),
            (pre_scheduled, make_pass_peak_below_current),
            (pre_scheduled, make_rejection_peak_below_current),
            (
                pre_scheduled,
                lambda value: value["receipt"]["source_dimensions"].__setitem__(
                    0, 0
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"]["source_dimensions"].__setitem__(
                    2, 0
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"]["source_dimensions"].__setitem__(
                    4, 8
                ),
            ),
            (
                pre_scheduled,
                lambda value: value["receipt"].__setitem__(
                    "source_payload_bytes", 0
                ),
            ),
        )
        for index, (base, mutate) in enumerate(attacks):
            with self.subTest(index=index):
                attacked = copy.deepcopy(base)
                mutate(attacked)
                coherent_digest = coherently_rehash(attacked)
                self.assertFalse(
                    verify_detached_phase_conditioned_objective_hull_k3_build_only(
                        attacked, expected_sha256=coherent_digest
                    )
                )

    def test_process_anchor_detached_anchor_and_baseexception_cleanup(self):
        result = self.success
        coherent_clone = replace(result)
        self.assertFalse(
            verify_phase_conditioned_objective_hull_k3_build_only_outcome(
                coherent_clone
            )
        )
        detached = export_phase_conditioned_objective_hull_k3_build_only_detached(
            result
        )
        self.assertTrue(
            verify_detached_phase_conditioned_objective_hull_k3_build_only(
                detached, expected_sha256=result.diagnostic_sha256
            )
        )
        detached["status"] = "coherently_rehashed_but_not_independently_anchored"
        body = dict(detached)
        body.pop("diagnostic_sha256")
        detached["diagnostic_sha256"] = k3_module._canonical_sha256(body)
        self.assertFalse(
            verify_detached_phase_conditioned_objective_hull_k3_build_only(
                detached, expected_sha256=result.diagnostic_sha256
            )
        )

        build, rivals, selection, ids = self._inputs()
        with fresh_module._REGISTRY_LOCK:
            before = (len(fresh_module._REGISTRY), len(fresh_module._REGISTRY_RESERVATIONS))
        with mock.patch.object(
            k3_module,
            "verify_live_phase_conditioned_objective_hull_fresh_materialized_tightness",
            side_effect=KeyboardInterrupt(),
        ), mock.patch.object(
            k3_module,
            "discard_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=RuntimeError("observable discard interrupted"),
        ):
            with self.assertRaisesRegex(
                PhaseConditionedK3BuildOnlyError,
                "fresh_issue_consume_inspect_release_failed",
            ):
                run_phase_conditioned_objective_hull_k3_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    retained_k2_stable_bit_ids=ids[:2],
                    deadline=time.monotonic() + 30.0,
                )
        with fresh_module._REGISTRY_LOCK:
            after = (len(fresh_module._REGISTRY), len(fresh_module._REGISTRY_RESERVATIONS))
        self.assertEqual(after, before)

    def test_pair_to_strict_replay_source_toctou_fails_before_fresh(self):
        build, rivals, selection, ids = self._inputs()
        original = k3_module.run_phase_conditioned_pair_infeasibility_candidate

        def mutate_after_pair(*args, **kwargs):
            pair = original(*args, **kwargs)
            args[0].hz.c[0] += 1.0
            return pair

        with fresh_module._REGISTRY_LOCK:
            before = (len(fresh_module._REGISTRY), len(fresh_module._REGISTRY_RESERVATIONS))
        with mock.patch.object(
            k3_module,
            "run_phase_conditioned_pair_infeasibility_candidate",
            side_effect=mutate_after_pair,
        ), mock.patch.object(
            k3_module,
            "issue_live_phase_conditioned_objective_hull_fresh_build",
        ) as fresh_issue:
            with self.assertRaisesRegex(
                PhaseConditionedK3BuildOnlyError,
                "strict_replay_failed",
            ):
                run_phase_conditioned_objective_hull_k3_build_only(
                    build,
                    rivals,
                    selection,
                    focused_rival_id=10,
                    retained_k2_stable_bit_ids=ids[:2],
                    deadline=time.monotonic() + 30.0,
                )
        fresh_issue.assert_not_called()
        with fresh_module._REGISTRY_LOCK:
            after = (len(fresh_module._REGISTRY), len(fresh_module._REGISTRY_RESERVATIONS))
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
