from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import gc
import inspect
import math
import time
from types import MappingProxyType
import unittest
import weakref
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import (
    operator_phase_conditioned_relaxed_lp_live_owner as owner,
)
from act.back_end.hybridz_tf import (
    split_constraint_generation_candidate as split_cg,
)
from act.back_end.hybridz_tf.operator_hz import OperatorHZBuild
from act.back_end.hybridz_tf.operator_phase_conditioned_relaxed_lp_live_owner import (
    PCOHRelaxedLPLiveOwnerError,
    PCOHRelaxedLPToyOwnerCaps,
    PCOHRelaxedLPToyOwnerResult,
    run_private_k4_pcoh_relaxed_lp_live_owner,
    verify_private_k4_pcoh_relaxed_lp_live_owner_result,
)
from act.back_end.hybridz_tf.split_constraint_generation_candidate import (
    SplitConstraintGenerationCandidate,
)
from act.back_end.solver.solver_hz import SparseHZono


def _run(**updates):
    arguments = {"deadline": time.monotonic() + 60.0}
    arguments.update(updates)
    return run_private_k4_pcoh_relaxed_lp_live_owner(**arguments)


def _direct_forbidden(value):
    return isinstance(
        value,
        (
            OperatorHZBuild,
            SparseHZono,
            SplitConstraintGenerationCandidate,
            np.ndarray,
            sp.spmatrix,
        ),
    )


def _assert_pure(testcase, value):
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        testcase.assertTrue(math.isfinite(value))
        return
    if type(value) in {tuple, list}:
        for item in value:
            _assert_pure(testcase, item)
        return
    if isinstance(value, MappingProxyType) or isinstance(value, dict):
        for key, item in value.items():
            testcase.assertIs(type(key), str)
            _assert_pure(testcase, item)
        return
    testcase.fail(f"non-pure receipt object: {type(value).__name__}")


class TestPrivateK4PCOHRelaxedLPLiveOwner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _run()

    def test_positive_exact_strict_ordering_and_authority_firewall(self):
        result = self.result
        self.assertIs(type(result), PCOHRelaxedLPToyOwnerResult)
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(result)
        )
        self.assertEqual(
            result.status, "strict_private_toy_numeric_ordering_certified"
        )
        self.assertEqual(result.parent_lower, 0.25000000000002776)
        self.assertEqual(result.fresh_upper, -0.24999999999997427)
        expected_gap = Fraction.from_float(
            result.parent_lower
        ) - Fraction.from_float(result.fresh_upper)
        self.assertEqual(result.exact_gap, expected_gap)
        self.assertGreater(expected_gap, 0)
        self.assertFalse(result.production_ready)
        self.assertFalse(result.proof_authority)
        self.assertFalse(result.verdict_authority)
        self.assertFalse(result.real_parent_binding_authority)

        receipt = result.receipt
        self.assertTrue(receipt["internal_toy_live_binding"])
        self.assertTrue(receipt["supplied_numeric_strict_ordering_theorem"])
        self.assertTrue(receipt["toy_numeric_frame_authority"])
        self.assertFalse(receipt["toy_sound_extension_claimed"])
        self.assertFalse(receipt["real_sound_extension_authority"])
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["verdict_authority"])
        self.assertFalse(receipt["receipt_sha256_keyed_authenticator"])
        self.assertFalse(
            receipt["pure_checker_authenticates_producer_provenance"]
        )
        self.assertFalse(
            receipt["pure_checker_authenticates_opaque_binding_hashes"]
        )
        self.assertEqual(receipt["source_dimensions"], (3, 9, 4, 0, 12))
        self.assertEqual(receipt["fresh_dimensions"], (3, 25, 4, 16, 13))
        self.assertNotEqual(
            receipt["source_semantic_digest"],
            receipt["fresh_semantic_digest"],
        )
        self.assertEqual(
            receipt["source_semantic_digest"],
            receipt["source_terminal_semantic_digest"],
        )
        self.assertEqual(
            receipt["fresh_semantic_digest"],
            receipt["fresh_terminal_semantic_digest"],
        )
        self.assertTrue(
            receipt["parent_cg_native_model_closed_before_primal_replay"]
        )
        self.assertTrue(
            receipt["fresh_cg_native_model_closed_before_dual_replay"]
        )
        self.assertFalse(receipt["parent_solver_primal_used_as_authority"])
        self.assertTrue(
            receipt["fresh_solver_dual_used_only_after_independent_replay"]
        )
        self.assertEqual(
            receipt["maximum_simultaneous_highs_models_by_construction"], 1
        )
        self.assertLessEqual(
            receipt["post_source_release_rss_bytes"],
            receipt["post_source_release_rss_cap_bytes"],
        )
        self.assertEqual(
            receipt["post_source_release_rss_hard_ceiling_bytes"],
            5 * 1024 * 1024 * 1024 // 2,
        )
        self.assertTrue(
            receipt["post_source_release_rss_cap_caller_tightenable_only"]
        )

    def test_public_contract_has_no_build_array_csr_or_callback_ingress(self):
        signature = inspect.signature(
            run_private_k4_pcoh_relaxed_lp_live_owner
        )
        self.assertEqual(tuple(signature.parameters), ("deadline", "caps"))
        for parameter in signature.parameters.values():
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(TypeError):
            run_private_k4_pcoh_relaxed_lp_live_owner(
                deadline=time.monotonic() + 60.0,
                source_factory=lambda: None,
            )
        caps = PCOHRelaxedLPToyOwnerCaps()
        for item in vars(caps).values():
            self.assertFalse(callable(item))
            self.assertFalse(_direct_forbidden(item))
        normalized = owner._normalize_caps(caps)
        self.assertIsNot(normalized, caps)
        self.assertIsNot(normalized.fresh_caps, caps.fresh_caps)
        self.assertIsNot(
            normalized.parent_primal_caps, caps.parent_primal_caps
        )
        object.__setattr__(
            caps,
            "max_post_source_release_rss_bytes",
            100 * 1024 * 1024 * 1024,
        )
        self.assertEqual(
            normalized.max_post_source_release_rss_bytes,
            5 * 1024 * 1024 * 1024 // 2,
        )
        clean_caps = PCOHRelaxedLPToyOwnerCaps()
        bad_fresh_caps = replace(
            clean_caps.fresh_caps,
            max_parent_variables=np.asarray([64], dtype=np.int64),
        )
        with self.assertRaises(PCOHRelaxedLPLiveOwnerError):
            _run(caps=replace(clean_caps, fresh_caps=bad_fresh_caps))
        self.assertEqual(owner._active_transaction_count(), 0)
        receipt = self.result.receipt
        self.assertFalse(receipt["public_accepts_build_or_numeric_frame"])
        self.assertFalse(receipt["public_accepts_source_callback"])
        self.assertFalse(
            receipt["caller_can_retain_source_alias_through_public_api"]
        )
        self.assertIn("upstream private producer", receipt["real_owner_blocker"])
        self.assertIn(
            "one-use consume", receipt["real_owner_minimum_upstream_api"]
        )

    def test_result_is_pure_immutable_and_contains_no_private_owner(self):
        result = self.result
        self.assertIs(type(result.receipt), MappingProxyType)
        _assert_pure(self, result.receipt)
        self.assertFalse(_direct_forbidden(result))
        with self.assertRaises(TypeError):
            result.receipt["proof_authority"] = True
        with self.assertRaises(TypeError):
            result.receipt["fresh_row_scaling"]["scale_exponent"] = 1
        with self.assertRaises(Exception):
            result.parent_lower = -1.0
        self.assertTrue(result.receipt["owner_registry_empty_before_return"])
        self.assertTrue(result.receipt["owner_registry_one_use"])
        self.assertFalse(result.receipt["owner_registry_contains_hz"])
        self.assertEqual(owner._active_transaction_count(), 0)

        nested_backing = dict(result.receipt)
        nested_backing["fresh_row_scaling"] = dict(
            result.receipt["fresh_row_scaling"]
        )
        nested_backing["timings"] = dict(result.receipt["timings"])
        detached_nested = replace(result, receipt=nested_backing)
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                detached_nested
            )
        )
        self.assertIs(
            type(detached_nested.receipt["fresh_row_scaling"]),
            MappingProxyType,
        )
        self.assertIs(
            type(detached_nested.receipt["timings"]), MappingProxyType
        )
        with self.assertRaises(TypeError):
            detached_nested.receipt["fresh_row_scaling"][
                "proof_authority"
            ] = True
        with self.assertRaises(TypeError):
            detached_nested.receipt["timings"]["total_seconds"] = 0.0

        top_backing = dict(result.receipt)
        detached_top = replace(
            result, receipt=MappingProxyType(top_backing)
        )
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(detached_top)
        )
        top_backing["proof_authority"] = True
        self.assertFalse(detached_top.receipt["proof_authority"])
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(detached_top)
        )

    def test_receipt_and_nested_scaling_tamper_fail_strict_replay(self):
        result = self.result
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(result)
        )

        def rehashed_receipt(changes, **result_changes):
            raw = dict(result.receipt)
            raw.update(changes)
            raw.pop("receipt_sha256", None)
            raw["receipt_sha256"] = owner._canonical_sha256(raw)
            return replace(
                result,
                receipt=owner._deep_freeze(raw),
                receipt_sha256=raw["receipt_sha256"],
                **result_changes,
            )

        tampered = rehashed_receipt(
            {"real_parent_binding_authority": True}
        )
        self.assertFalse(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(tampered)
        )

        forged_parent = 123.0
        forged_fresh = -456.0
        forged_gap = Fraction.from_float(
            forged_parent
        ) - Fraction.from_float(forged_fresh)
        numeric_forgery = rehashed_receipt(
            {
                "parent_lower": forged_parent,
                "parent_lower_hex": forged_parent.hex(),
                "fresh_upper": forged_fresh,
                "fresh_upper_hex": forged_fresh.hex(),
                "exact_gap": (
                    forged_gap.numerator,
                    forged_gap.denominator,
                ),
            },
            parent_lower=forged_parent,
            fresh_upper=forged_fresh,
            exact_gap=forged_gap,
        )
        self.assertFalse(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                numeric_forgery
            )
        )
        for changes in (
            {"toy_only": False},
            {"toy_numeric_frame_authority": "forged"},
            {"strict_comparison": "none"},
            {"extra_authority": True},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                        rehashed_receipt(changes)
                    )
                )

        # These are deliberately opaque, unkeyed diagnostics.  A pure
        # receipt checker cannot authenticate their producer provenance; its
        # documented scope must not be mistaken for a signature.
        for field in (
            "parent_frame_sha256",
            "parent_primal_anchor_sha256",
            "fresh_issuance_sha256",
        ):
            with self.subTest(opaque_field=field):
                self.assertTrue(
                    verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                        rehashed_receipt({field: "0" * 64})
                    )
                )

        def rehashed_scaling(changes):
            raw = dict(result.receipt)
            scaling = dict(raw["fresh_row_scaling"])
            scaling.update(changes)
            scaling.pop("scaling_receipt_sha256", None)
            scaling["scaling_receipt_sha256"] = owner._canonical_sha256(
                scaling
            )
            raw["fresh_row_scaling"] = scaling
            raw.pop("receipt_sha256", None)
            raw["receipt_sha256"] = owner._canonical_sha256(raw)
            return replace(
                result,
                receipt=owner._deep_freeze(raw),
                receipt_sha256=raw["receipt_sha256"],
            )

        scaling = result.receipt["fresh_row_scaling"]
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                rehashed_scaling(
                    {
                        "original_Aub_sha256": scaling[
                            "scaled_Aub_sha256"
                        ]
                    }
                )
            )
        )
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(
                rehashed_scaling({"original_Auc_sha256": "1" * 64})
            )
        )

        raw = dict(result.receipt)
        scaling = dict(raw["fresh_row_scaling"])
        scaling.pop("scaling_receipt_sha256")
        scaling["scale_exponent"] = 25
        scaling["scaling_receipt_sha256"] = owner._canonical_sha256(scaling)
        raw["fresh_row_scaling"] = scaling
        raw.pop("receipt_sha256")
        raw["receipt_sha256"] = owner._canonical_sha256(raw)
        nested = replace(
            result,
            receipt=owner._deep_freeze(raw),
            receipt_sha256=raw["receipt_sha256"],
        )
        self.assertFalse(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(nested)
        )

    def test_exact_positive_row_scaling_is_independently_replayed(self):
        real_scale = owner._scale_new_upper_rows_exactly
        observed = {"items": 0, "rows": None}

        def replaying_scale(fresh, **kwargs):
            Auc, Aub, ub, receipt = real_scale(fresh, **kwargs)
            start = kwargs["source_upper_rows"]
            scale = Fraction(1 << 24, 1)
            rows = tuple(range(start, fresh.hz.n_ub))
            checked = 0
            for row in rows:
                for old_matrix, new_matrix in (
                    (fresh.hz.Auc, Auc),
                    (fresh.hz.Aub, Aub),
                ):
                    old_start = int(old_matrix.indptr[row])
                    old_stop = int(old_matrix.indptr[row + 1])
                    new_start = int(new_matrix.indptr[row])
                    new_stop = int(new_matrix.indptr[row + 1])
                    self.assertEqual(old_stop - old_start, new_stop - new_start)
                    for old, new in zip(
                        old_matrix.data[old_start:old_stop],
                        new_matrix.data[new_start:new_stop],
                    ):
                        self.assertEqual(
                            Fraction.from_float(float(new)),
                            Fraction.from_float(float(old)) * scale,
                        )
                        checked += 1
                self.assertEqual(
                    Fraction.from_float(float(ub[row])),
                    Fraction.from_float(float(fresh.hz.ub[row])) * scale,
                )
                checked += 1
            observed["items"] = checked
            observed["rows"] = rows
            self.assertEqual(
                receipt["exact_fraction_items_replayed"], checked
            )
            return Auc, Aub, ub, receipt

        # Install a plain function, not a MagicMock: call recording would own
        # the private fresh argument until context exit and correctly trip the
        # live owner's terminal weakref-release sentinel.
        with mock.patch.object(
            owner,
            "_scale_new_upper_rows_exactly",
            new=replaying_scale,
        ):
            result = _run()
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(result)
        )
        self.assertEqual(observed["rows"], (12,))
        self.assertEqual(observed["items"], 12)
        scaling = result.receipt["fresh_row_scaling"]
        self.assertEqual(scaling["scaled_upper_rows"], (12,))
        self.assertEqual(scaling["scale_exponent"], 24)
        self.assertTrue(
            scaling["all_changed_coefficients_and_rhs_exactly_scaled"]
        )

    def test_source_and_fresh_are_released_beyond_internal_weakrefs(self):
        real_builder = owner._build_private_corner_toy_source
        real_consume = owner.consume_live_phase_conditioned_objective_hull_fresh_build
        source_refs = []
        fresh_refs = []

        def building():
            build = real_builder()
            source_refs.extend(owner._private_weakrefs(build))
            return build

        def consuming(*args, **kwargs):
            fresh = real_consume(*args, **kwargs)
            fresh_refs.extend(owner._private_weakrefs(fresh))
            return fresh

        with mock.patch.object(
            owner, "_build_private_corner_toy_source", side_effect=building
        ), mock.patch.object(
            owner,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=consuming,
        ):
            result = _run()
        gc.collect()
        self.assertGreater(len(source_refs), 2)
        self.assertGreater(len(fresh_refs), 2)
        self.assertTrue(all(reference() is None for reference in source_refs))
        self.assertTrue(all(reference() is None for reference in fresh_refs))
        self.assertTrue(
            result.receipt["source_weakrefs_released_before_fresh_lp"]
        )
        self.assertTrue(
            result.receipt["fresh_weakrefs_released_before_receipt"]
        )
        self.assertEqual(
            result.receipt["source_and_fresh_overlap_scope"],
            "fresh_materialization_and_detachment_check_only",
        )

    def test_all_tracked_native_highs_models_close_without_overlap(self):
        real_highs = split_cg._highspy.Highs
        state = {"active": 0, "peak": 0, "created": 0, "closed": 0}

        def factory(*args, **kwargs):
            self.assertEqual(state["active"], 0)
            model = real_highs(*args, **kwargs)
            state["active"] += 1
            state["created"] += 1
            state["peak"] = max(state["peak"], state["active"])
            original_clear = model.clear
            closed = False

            def clear():
                nonlocal closed
                status = original_clear()
                if not closed:
                    closed = True
                    state["active"] -= 1
                    state["closed"] += 1
                return status

            model.clear = clear
            return model

        with mock.patch.object(split_cg._highspy, "Highs", new=factory):
            result = _run()
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(result)
        )
        self.assertGreaterEqual(state["created"], 2)
        self.assertEqual(state["created"], state["closed"])
        self.assertEqual(state["active"], 0)
        self.assertEqual(state["peak"], 1)

    def test_closed_candidate_receipt_tamper_fails_before_authorization(self):
        real_candidate = owner.propose_split_constraint_generation_candidate
        calls = 0

        def tampering_candidate(**kwargs):
            nonlocal calls
            candidate = real_candidate(**kwargs)
            calls += 1
            if calls != 1:
                return candidate
            receipt = dict(candidate.receipt)
            receipt.pop("receipt_sha256")
            receipt["native_model_closed_before_return"] = False
            receipt["receipt_sha256"] = split_cg._canonical_json_sha256(receipt)
            return replace(candidate, receipt=receipt)

        with mock.patch.object(
            owner,
            "propose_split_constraint_generation_candidate",
            side_effect=tampering_candidate,
        ):
            with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as caught:
                _run()
        self.assertIn(
            "split_cg_terminal_close_or_scan_contract_failed",
            str(caught.exception),
        )
        self.assertEqual(owner._active_transaction_count(), 0)

    def test_baseexception_discards_fresh_registry_and_scrubs_private_frames(self):
        cleanup = owner._FRESH_DISCARD_CLEANUP_AUTHORITY
        cleanup_results = []

        def cleaning(*args, **kwargs):
            result = cleanup(*args, **kwargs)
            cleanup_results.append(result)
            return result

        with mock.patch.object(
            owner,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=KeyboardInterrupt("toy interruption"),
        ), mock.patch.object(
            owner,
            "discard_live_phase_conditioned_objective_hull_fresh_build",
            new=cleaning,
        ):
            with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as caught:
                _run()
        error = caught.exception
        self.assertIn("interrupted:KeyboardInterrupt", str(error))
        self.assertEqual(cleanup_results, [True])
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        cursor = error.__traceback__
        while cursor is not None:
            for value in cursor.tb_frame.f_locals.values():
                self.assertFalse(isinstance(value, (OperatorHZBuild, SparseHZono)))
            cursor = cursor.tb_next
        self.assertEqual(owner._active_transaction_count(), 0)

        def cleanup_interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt("cleanup interruption")

        # Even two cleanup-path BaseExceptions must not leave the upstream
        # private fresh build waiting for a later TTL sweep.
        with mock.patch.object(
            owner,
            "consume_live_phase_conditioned_objective_hull_fresh_build",
            side_effect=KeyboardInterrupt("second toy interruption"),
        ), mock.patch.object(
            owner,
            "discard_live_phase_conditioned_objective_hull_fresh_build",
            new=cleanup_interrupt,
        ), mock.patch.object(
            owner,
            "_FRESH_DISCARD_CLEANUP_AUTHORITY",
            new=cleanup_interrupt,
        ):
            with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as second:
                _run()
        self.assertIn("interrupted:KeyboardInterrupt", str(second.exception))
        with owner._fresh_module._REGISTRY_LOCK:
            self.assertEqual(len(owner._fresh_module._REGISTRY), 0)
        self.assertEqual(owner._active_transaction_count(), 0)
        # A complete second run demonstrates that the materializer's one-use
        # registry/reservation was not stranded by the interruption.
        self.assertTrue(
            verify_private_k4_pcoh_relaxed_lp_live_owner_result(_run())
        )

    def test_deadline_rss_and_concurrent_owner_gates_fail_closed(self):
        with self.assertRaises(PCOHRelaxedLPLiveOwnerError):
            run_private_k4_pcoh_relaxed_lp_live_owner(
                deadline=time.monotonic() - 1.0
            )
        self.assertEqual(owner._active_transaction_count(), 0)

        loose = replace(
            PCOHRelaxedLPToyOwnerCaps(),
            max_post_source_release_rss_bytes=100 * 1024 * 1024 * 1024,
        )
        with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as caught:
            _run(caps=loose)
        self.assertIn("cannot_relax_2_5_gib_stoploss", str(caught.exception))
        self.assertEqual(owner._active_transaction_count(), 0)

        tight = replace(
            PCOHRelaxedLPToyOwnerCaps(),
            max_post_source_release_rss_bytes=1,
        )
        with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as caught:
            _run(caps=tight)
        self.assertIn("post_source_release_rss_gate_exceeded", str(caught.exception))
        self.assertEqual(owner._active_transaction_count(), 0)

        token = owner._reserve_transaction()
        try:
            with self.assertRaises(PCOHRelaxedLPLiveOwnerError) as caught:
                _run()
            self.assertIn("already_active", str(caught.exception))
        finally:
            self.assertTrue(owner._release_transaction(token))
        self.assertEqual(owner._active_transaction_count(), 0)


if __name__ == "__main__":
    unittest.main()
