#!/usr/bin/env python3
"""One-shot soundness gates for the toy-only live PCOH v2 adapter."""

from __future__ import annotations

from dataclasses import replace
import math
import time
from types import MappingProxyType
import unittest
from unittest import mock

from act.back_end.hybridz_tf import (
    operator_phase_conditioned_live_adapter as adapter,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    _build_complete_operator_phase_conditioned_objective_bounds_until,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_pair_infeasibility import (
    run_phase_conditioned_pair_infeasibility_candidate,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
    _rivals,
)
from act.back_end.hybridz_tf.test_operator_phase_conditioned_pair_infeasibility import (
    _mixed_sign_build,
    _triple_only_build,
)


def _sources(build, *, stable_count=None, seconds=60.0):
    rivals = _rivals()
    selection = derive_operator_exact_relu_property_phase_literals(
        build, rivals
    )
    all_ids = tuple(
        mapping.stable_bcol_id for mapping in selection.mappings
    )
    stable_ids = (
        all_ids
        if stable_count is None
        else all_ids[:stable_count]
    )
    build_deadline = time.monotonic() + seconds
    certificates = (
        _build_complete_operator_phase_conditioned_objective_bounds_until(
            build,
            rivals,
            selection,
            focused_rival_id=10,
            stable_bit_ids=stable_ids,
            deadline=build_deadline,
        )
    )
    pair_bundle = run_phase_conditioned_pair_infeasibility_candidate(
        build,
        rivals,
        selection,
        stable_bit_ids=stable_ids,
        deadline=time.monotonic() + seconds,
    )
    return rivals, selection, stable_ids, certificates, pair_bundle


def _combine(
    build,
    rivals,
    selection,
    stable_ids,
    certificates,
    pair_bundle,
    *,
    deadline=None,
):
    return adapter.build_live_phase_conditioned_objective_hull_candidate(
        build,
        rivals,
        selection,
        focused_rival_id=10,
        stable_bit_ids=stable_ids,
        conditional_certificates=certificates,
        pair_bundle=pair_bundle,
        deadline=(
            time.monotonic() + 60.0
            if deadline is None
            else deadline
        ),
    )


class CombinedLiveAdapterK4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _k4_corner_build()
        (
            cls.rivals,
            cls.selection,
            cls.stable_ids,
            cls.certificates,
            cls.pair_bundle,
        ) = _sources(cls.build)
        cls.candidate = _combine(
            cls.build,
            cls.rivals,
            cls.selection,
            cls.stable_ids,
            cls.certificates,
            cls.pair_bundle,
        )

    def test_complete_bounds_six_edges_and_eleven_rederived_empty_patterns(self):
        candidate = self.candidate
        descriptor = candidate.descriptor
        evidence = candidate.empty_pattern_evidence
        self.assertEqual(len(descriptor.patterns), 16)
        self.assertEqual(len(descriptor.pattern_bounds), 16)
        self.assertEqual(len(candidate.pair_certificate_sha256), 6)
        self.assertEqual(len(evidence), 11)
        self.assertEqual(
            len({item.certificate_sha256 for item in evidence}),
            6,
        )
        self.assertEqual(
            len({item.coverage_sha256 for item in evidence}),
            11,
        )
        self.assertEqual(
            len({item.descriptor_sha256 for item in evidence}),
            11,
        )
        empty_patterns = {
            tuple(phase for _, phase in item.assignments)
            for item in evidence
        }
        self.assertEqual(len(empty_patterns), 11)
        for column in descriptor.eta_columns:
            if column.pattern in empty_patterns:
                self.assertEqual(column.lower, -1)
                self.assertEqual(column.upper, -1)
            else:
                self.assertEqual(column.lower, -1)
                self.assertEqual(column.upper, 1)
        self.assertEqual(len(descriptor.equality_rows), 4 + 1 + 11)
        self.assertEqual(descriptor.receipt["empty_eta_fix_rows"], 11)

    def test_proof_firewall_receipt_and_live_structure(self):
        candidate = self.candidate
        self.assertFalse(candidate.proof_authority)
        self.assertFalse(candidate.verdict_authority)
        self.assertFalse(candidate.descriptor.proof_authority)
        self.assertFalse(candidate.descriptor.verdict_authority)
        receipt = candidate.receipt
        self.assertIs(type(receipt), MappingProxyType)
        self.assertFalse(receipt["external_proof_booleans_used_as_authority"])
        self.assertFalse(receipt["external_coverage_status_used_as_authority"])
        self.assertFalse(receipt["materialized_hz"])
        self.assertFalse(receipt["solver_handoff_capability_issued"])
        self.assertFalse(receipt["one_use_registry_used"])
        self.assertFalse(receipt["one_use_registry_required_for_pure_descriptor"])
        self.assertTrue(
            receipt["future_mutable_solver_handoff_requires_one_use_registry"]
        )
        self.assertTrue(
            adapter.verify_live_phase_conditioned_objective_hull_candidate_structure(
                self.build,
                self.selection,
                candidate,
            )
        )

    def test_same_exact_certificate_may_cover_multiple_unique_pattern_slots(self):
        by_certificate = {}
        for evidence in self.candidate.empty_pattern_evidence:
            by_certificate.setdefault(evidence.certificate_sha256, []).append(
                evidence
            )
        self.assertEqual(len(by_certificate), 6)
        self.assertTrue(any(len(items) > 1 for items in by_certificate.values()))
        for items in by_certificate.values():
            self.assertEqual(
                len({item.source_record_sha256 for item in items}),
                1,
            )
            self.assertEqual(
                len({item.local_row_map_sha256 for item in items}),
                1,
            )
            self.assertEqual(
                len({item.coverage_sha256 for item in items}),
                len(items),
            )

    def test_pair_strict_verifier_and_conditional_replay_share_exact_deadline(self):
        requested_deadline = time.monotonic() + 60.0
        seen = []
        original_pair = (
            adapter.verify_phase_conditioned_pair_infeasibility_bundle
        )
        original_replay = (
            adapter._replay_complete_operator_phase_conditioned_objective_bounds_until
        )

        def pair_wrapper(*args, **kwargs):
            seen.append(("pair", kwargs["deadline"]))
            return original_pair(*args, **kwargs)

        def replay_wrapper(*args, **kwargs):
            seen.append(("conditional", kwargs["deadline"]))
            return original_replay(*args, **kwargs)

        with mock.patch.object(
            adapter,
            "verify_phase_conditioned_pair_infeasibility_bundle",
            side_effect=pair_wrapper,
        ), mock.patch.object(
            adapter,
            "_replay_complete_operator_phase_conditioned_objective_bounds_until",
            side_effect=replay_wrapper,
        ):
            result = _combine(
                self.build,
                self.rivals,
                self.selection,
                self.stable_ids,
                self.certificates,
                self.pair_bundle,
                deadline=requested_deadline,
            )
        self.assertFalse(result.proof_authority)
        self.assertEqual(
            seen,
            [
                ("pair", requested_deadline),
                ("conditional", requested_deadline),
            ],
        )

    def test_adapter_performs_only_one_outer_full_parent_digest_scan(self):
        original = adapter.sparse_hz_semantic_digest
        with mock.patch.object(
            adapter,
            "sparse_hz_semantic_digest",
            wraps=original,
        ) as semantic_digest:
            result = _combine(
                self.build,
                self.rivals,
                self.selection,
                self.stable_ids,
                self.certificates,
                self.pair_bundle,
            )
        self.assertEqual(semantic_digest.call_count, 1)
        self.assertEqual(
            result.receipt[
                "adapter_full_parent_semantic_digest_computations"
            ],
            1,
        )
        self.assertFalse(
            result.receipt["redundant_intermediate_parent_scans"]
        )

    def test_pure_descriptor_is_deliberately_reusable_without_registry(self):
        second = _combine(
            self.build,
            self.rivals,
            self.selection,
            self.stable_ids,
            self.certificates,
            self.pair_bundle,
        )
        self.assertEqual(second.candidate_sha256, self.candidate.candidate_sha256)
        self.assertEqual(
            second.descriptor.representation_sha256,
            self.candidate.descriptor.representation_sha256,
        )
        self.assertTrue(
            adapter.verify_live_phase_conditioned_objective_hull_candidate_structure(
                self.build,
                self.selection,
                second,
            )
        )

    def test_raw_boolean_and_cross_binding_tamper_fail_closed(self):
        cases = (
            {
                "pair_bundle": replace(
                    self.pair_bundle, proof_authority=True
                )
            },
            {
                "conditional_certificates": (
                    replace(self.certificates[0], proof_authority=True),
                    *self.certificates[1:],
                )
            },
            {"conditional_certificates": self.certificates[:-1]},
            {"conditional_certificates": tuple(reversed(self.certificates))},
            {"focused_rival_id": 20},
            {"stable_bit_ids": tuple(reversed(self.stable_ids))},
        )
        base = {
            "focused_rival_id": 10,
            "stable_bit_ids": self.stable_ids,
            "conditional_certificates": self.certificates,
            "pair_bundle": self.pair_bundle,
        }
        for index, changes in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(
                adapter.PhaseConditionedLiveAdapterError
            ):
                adapter.build_live_phase_conditioned_objective_hull_candidate(
                    self.build,
                    self.rivals,
                    self.selection,
                    **{**base, **changes},
                    deadline=time.monotonic() + 60.0,
                )

    def test_coverage_witness_tamper_is_rejected_before_evidence_binding(self):
        first = self.pair_bundle.coverage[0]
        changed = replace(
            self.pair_bundle,
            coverage=(
                replace(first, coverage_sha256="0" * 64),
                *self.pair_bundle.coverage[1:],
            ),
        )
        with mock.patch.object(
            adapter,
            "bind_external_certified_empty_pattern",
            side_effect=AssertionError("unverified coverage reached binder"),
        ), self.assertRaisesRegex(
            adapter.PhaseConditionedLiveAdapterError,
            "pair_bundle_live_verification_failed",
        ):
            _combine(
                self.build,
                self.rivals,
                self.selection,
                self.stable_ids,
                self.certificates,
                changed,
            )

    def test_terminal_parent_mutation_cannot_issue_candidate(self):
        original = adapter.build_phase_conditioned_objective_hull
        saved = self.build.hz.ub.copy()

        def mutate_after_build(*args, **kwargs):
            descriptor = original(*args, **kwargs)
            self.build.hz.ub[0] = math.nextafter(
                float(self.build.hz.ub[0]), math.inf
            )
            return descriptor

        try:
            with mock.patch.object(
                adapter,
                "build_phase_conditioned_objective_hull",
                side_effect=mutate_after_build,
            ), self.assertRaisesRegex(
                adapter.PhaseConditionedLiveAdapterError,
                "terminal_parent",
            ):
                _combine(
                    self.build,
                    self.rivals,
                    self.selection,
                    self.stable_ids,
                    self.certificates,
                    self.pair_bundle,
                )
        finally:
            self.build.hz.ub[:] = saved

    def test_expired_deadline_stops_before_any_replay(self):
        with mock.patch.object(
            adapter,
            "verify_phase_conditioned_pair_infeasibility_bundle",
            side_effect=AssertionError("pair replay started"),
        ), mock.patch.object(
            adapter,
            "_replay_complete_operator_phase_conditioned_objective_bounds_until",
            side_effect=AssertionError("conditional replay started"),
        ), self.assertRaisesRegex(
            adapter.PhaseConditionedLiveAdapterError,
            "deadline",
        ):
            _combine(
                self.build,
                self.rivals,
                self.selection,
                self.stable_ids,
                self.certificates,
                self.pair_bundle,
                deadline=time.monotonic() - 1.0,
            )

    def test_wrapper_descriptor_and_receipt_tamper_fail_structure(self):
        receipt = dict(self.candidate.receipt)
        receipt["one_use_registry_used"] = True
        candidates = (
            replace(
                self.candidate,
                receipt=MappingProxyType(receipt),
            ),
            replace(
                self.candidate,
                descriptor=replace(
                    self.candidate.descriptor,
                    representation_sha256="0" * 64,
                ),
            ),
            replace(self.candidate, candidate_sha256="0" * 64),
        )
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index):
                self.assertFalse(
                    adapter.verify_live_phase_conditioned_objective_hull_candidate_structure(
                        self.build,
                        self.selection,
                        candidate,
                    )
                )
        with self.assertRaisesRegex(ValueError, "proof authority"):
            replace(self.candidate, proof_authority=True)


class CombinedLiveAdapterControlsTests(unittest.TestCase):
    def test_k1_has_complete_bounds_and_no_empty_evidence(self):
        build = _k4_corner_build()
        rivals, selection, ids, certificates, pair_bundle = _sources(
            build, stable_count=1
        )
        candidate = _combine(
            build,
            rivals,
            selection,
            ids,
            certificates,
            pair_bundle,
        )
        self.assertEqual(len(candidate.descriptor.pattern_bounds), 2)
        self.assertFalse(candidate.pair_certificate_sha256)
        self.assertFalse(candidate.empty_pattern_evidence)
        self.assertEqual(len(candidate.descriptor.equality_rows), 2)
        self.assertTrue(
            adapter.verify_live_phase_conditioned_objective_hull_candidate_structure(
                build,
                selection,
                candidate,
            )
        )

    def test_mixed_sign_exact_edge_fixes_only_its_matching_pattern(self):
        build = _mixed_sign_build()
        rivals, selection, ids, certificates, pair_bundle = _sources(build)
        candidate = _combine(
            build,
            rivals,
            selection,
            ids,
            certificates,
            pair_bundle,
        )
        self.assertEqual(len(candidate.descriptor.pattern_bounds), 4)
        self.assertEqual(len(candidate.pair_certificate_sha256), 1)
        self.assertEqual(len(candidate.empty_pattern_evidence), 1)
        evidence = candidate.empty_pattern_evidence[0]
        self.assertEqual(
            tuple(phase for _, phase in evidence.assignments),
            (-1, 1),
        )
        self.assertEqual(evidence.witness_literals, evidence.assignments)
        self.assertEqual(
            [column.upper for column in candidate.descriptor.eta_columns],
            [1, -1, 1, 1],
        )

    def test_triple_only_infeasibility_is_not_invented_by_pair_adapter(self):
        build = _triple_only_build()
        rivals, selection, ids, certificates, pair_bundle = _sources(build)
        candidate = _combine(
            build,
            rivals,
            selection,
            ids,
            certificates,
            pair_bundle,
        )
        self.assertEqual(len(candidate.descriptor.pattern_bounds), 8)
        self.assertFalse(candidate.pair_certificate_sha256)
        self.assertFalse(candidate.empty_pattern_evidence)
        self.assertEqual(
            candidate.receipt["not_certified_empty_patterns"],
            8,
        )


if __name__ == "__main__":
    unittest.main()
