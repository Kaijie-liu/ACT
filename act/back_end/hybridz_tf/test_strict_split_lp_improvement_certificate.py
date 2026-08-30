from __future__ import annotations

from dataclasses import replace
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.split_constraint_generation_candidate import (
    _canonical_json_sha256,
    propose_split_constraint_generation_candidate,
)
from act.back_end.hybridz_tf.strict_split_lp_improvement_certificate import (
    PreformedSplitLPProblem,
    SplitRelaxedLPFrame,
    StrictSuppliedSplitLPOrderingCertificate,
    certify_strict_preformed_split_lp_improvement,
)
from act.back_end.solver.solver_hz import (
    _hz_form_exact_factor_objective_envelope_from_live_split_blocks,
)


_OBJECTIVE_ID = "toy:strict:split:ordering"
_PARENT_DIGEST = "31" * 32
_FRESH_DIGEST = "72" * 32


def _ro(values, *, dtype=np.float64):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _csr_ro(values):
    result = sp.csr_matrix(values, dtype=np.float64)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    for array in (result.data, result.indices, result.indptr):
        array.setflags(write=False)
    return result


def _form_envelope(
    *,
    continuous_ids,
    binary_ids,
    continuous_coefficients,
    binary_coefficients,
    parent_digest,
):
    envelope, receipt = (
        _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix(
                np.asarray(continuous_coefficients, dtype=np.float64).reshape(
                    1, -1
                )
            ),
            Gb=sp.csr_matrix(
                np.asarray(binary_coefficients, dtype=np.float64).reshape(
                    1, -1
                )
            ),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=0.0,
            continuous_col_ids=continuous_ids,
            binary_col_ids=binary_ids,
            objective_id=_OBJECTIVE_ID,
            parent_semantic_digest=parent_digest,
        )
    )
    if receipt["status"] != "formed":
        raise AssertionError(receipt)
    return envelope


def _problem(
    *,
    envelope,
    parent_digest,
    continuous_ids,
    binary_ids,
    continuous_lb,
    continuous_ub,
    binary_lb,
    binary_ub,
    frame,
):
    return PreformedSplitLPProblem(
        objective_envelope=envelope,
        expected_parent_semantic_digest=parent_digest,
        expected_exact_objective_sha256=envelope.exact_objective_sha256,
        expected_objective_binding_sha256=envelope.objective_binding_sha256,
        continuous_col_ids=continuous_ids,
        binary_col_ids=binary_ids,
        continuous_lb=continuous_lb,
        continuous_ub=continuous_ub,
        binary_lb=binary_lb,
        binary_ub=binary_ub,
        frame=frame,
    )


def _candidate(problem, *, seed_upper=False):
    q = _ro(
        np.concatenate(
            [
                problem.objective_envelope.q_continuous_hat,
                problem.objective_envelope.q_binary_hat,
            ]
        )
    )
    lower = _ro(
        np.concatenate([problem.continuous_lb, problem.binary_lb])
    )
    upper = _ro(
        np.concatenate([problem.continuous_ub, problem.binary_ub])
    )
    seed_rows = (0,) if seed_upper else ()
    seed_duals = (0.0,) if seed_upper else ()
    return propose_split_constraint_generation_candidate(
        Auc=problem.frame.Auc,
        Aub=problem.frame.Aub,
        Ac=problem.frame.Ac,
        Ab=problem.frame.Ab,
        ub=problem.frame.ub,
        b=problem.frame.b,
        q=q,
        lower_bounds=lower,
        upper_bounds=upper,
        seed_upper_rows=seed_rows,
        seed_upper_duals=seed_duals,
        deadline=time.monotonic() + 30.0,
        max_equality_rows=int(problem.frame.b.size),
        threads=1,
    )


def _continuous_pair(*, fresh_upper=0.0, fresh_coefficient=1.0):
    parent_binary_ids = _ro([], dtype=np.int64)
    fresh_binary_ids = _ro([], dtype=np.int64)
    parent_ids = _ro([10], dtype=np.int64)
    fresh_ids = _ro([10, 20], dtype=np.int64)
    parent_envelope = _form_envelope(
        continuous_ids=parent_ids,
        binary_ids=parent_binary_ids,
        continuous_coefficients=[1.0],
        binary_coefficients=[],
        parent_digest=_PARENT_DIGEST,
    )
    fresh_envelope = _form_envelope(
        continuous_ids=fresh_ids,
        binary_ids=fresh_binary_ids,
        continuous_coefficients=[fresh_coefficient, 0.0],
        binary_coefficients=[],
        parent_digest=_FRESH_DIGEST,
    )
    parent_frame = SplitRelaxedLPFrame(
        Auc=_csr_ro((0, 1)),
        Aub=_csr_ro((0, 0)),
        Ac=_csr_ro((0, 1)),
        Ab=_csr_ro((0, 0)),
        ub=_ro([]),
        b=_ro([]),
    )
    fresh_frame = SplitRelaxedLPFrame(
        Auc=_csr_ro([[1.0, 0.0]]),
        Aub=_csr_ro((1, 0)),
        Ac=_csr_ro((0, 2)),
        Ab=_csr_ro((0, 0)),
        ub=_ro([fresh_upper]),
        b=_ro([]),
    )
    parent = _problem(
        envelope=parent_envelope,
        parent_digest=_PARENT_DIGEST,
        continuous_ids=parent_ids,
        binary_ids=parent_binary_ids,
        continuous_lb=_ro([-1.0]),
        continuous_ub=_ro([1.0]),
        binary_lb=_ro([]),
        binary_ub=_ro([]),
        frame=parent_frame,
    )
    fresh = _problem(
        envelope=fresh_envelope,
        parent_digest=_FRESH_DIGEST,
        continuous_ids=fresh_ids,
        binary_ids=fresh_binary_ids,
        continuous_lb=_ro([-1.0, -1.0]),
        continuous_ub=_ro([1.0, 1.0]),
        binary_lb=_ro([]),
        binary_ub=_ro([]),
        frame=fresh_frame,
    )
    return parent, fresh, _candidate(parent), _candidate(fresh, seed_upper=True)


def _call(parent, fresh, parent_candidate, fresh_candidate, **updates):
    arguments = {
        "expected_objective_id": _OBJECTIVE_ID,
        "parent_problem": parent,
        "fresh_problem": fresh,
        "parent_candidate": parent_candidate,
        "fresh_candidate": fresh_candidate,
        "deadline": time.monotonic() + 30.0,
    }
    arguments.update(updates)
    return certify_strict_preformed_split_lp_improvement(**arguments)


class TestStrictSplitLPImprovementCertificate(unittest.TestCase):
    def test_positive_parent_max_x_fresh_x_le_zero_with_zero_eta_extension(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        descriptor, receipt = _call(
            parent, fresh, parent_candidate, fresh_candidate
        )
        self.assertIsInstance(
            descriptor, StrictSuppliedSplitLPOrderingCertificate
        )
        self.assertTrue(receipt["strict_relaxed_lp_improvement_certified"])
        self.assertEqual(
            receipt["strict_relaxed_lp_improvement_scope"],
            "two_supplied_numeric_frames_only",
        )
        self.assertTrue(
            receipt["strict_supplied_frame_optimum_ordering_certified"]
        )
        self.assertTrue(receipt["objective_extension_equivalent"])
        self.assertTrue(receipt["distinct_objective_envelope_identity"])
        self.assertTrue(receipt["distinct_stable_id_storage"])
        self.assertEqual(
            receipt["fresh_only_zero_objective_continuous_columns"], 1
        )
        self.assertEqual(descriptor.parent_lower, 1.0)
        self.assertLess(descriptor.fresh_upper, descriptor.parent_lower)
        self.assertGreater(descriptor.exact_gap, 0)
        self.assertTrue(descriptor.numeric_frame_authority)
        self.assertFalse(descriptor.proof_authority)
        self.assertFalse(descriptor.verdict_authority)
        self.assertFalse(descriptor.parent_binding_authority)
        self.assertFalse(descriptor.sound_tightening_improvement_authority)
        self.assertFalse(receipt["constraint_frame_sound_extension_authority"])
        self.assertFalse(receipt["hostile_concurrent_aba_resistance"])
        self.assertTrue(receipt["trusted_no_concurrent_mutation_required"])
        self.assertTrue(
            receipt["one_use_live_owner_required_for_sound_tightening"]
        )
        self.assertFalse(receipt["solver_status_numeric_authority"])
        self.assertFalse(receipt["solver_objective_numeric_authority"])
        self.assertFalse(receipt["upper_vs_upper_comparison_used"])
        self.assertTrue(
            receipt["candidate_native_models_closed_before_orchestration"]
        )
        with self.assertRaises(TypeError):
            descriptor.receipt["status"] = "tampered"

    def test_different_upper_row_but_same_true_optimum_is_not_strict(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair(
            fresh_upper=1.0
        )
        descriptor, receipt = _call(
            parent, fresh, parent_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertEqual(
            receipt["status"], "valid_bounds_without_strict_ordering"
        )
        self.assertFalse(receipt["strict_relaxed_lp_improvement_certified"])
        self.assertFalse(
            receipt["strict_supplied_frame_optimum_ordering_certified"]
        )
        self.assertTrue(receipt["numeric_frame_authority"])
        self.assertGreaterEqual(receipt["fresh_upper"], receipt["parent_lower"])

    def test_fractional_binary_relaxation_is_certified_without_integrality(self):
        parent_continuous_ids = _ro([], dtype=np.int64)
        fresh_continuous_ids = _ro([20], dtype=np.int64)
        parent_binary_ids = _ro([100], dtype=np.int64)
        fresh_binary_ids = _ro([100], dtype=np.int64)
        parent_envelope = _form_envelope(
            continuous_ids=parent_continuous_ids,
            binary_ids=parent_binary_ids,
            continuous_coefficients=[],
            binary_coefficients=[1.0],
            parent_digest=_PARENT_DIGEST,
        )
        fresh_envelope = _form_envelope(
            continuous_ids=fresh_continuous_ids,
            binary_ids=fresh_binary_ids,
            continuous_coefficients=[0.0],
            binary_coefficients=[1.0],
            parent_digest=_FRESH_DIGEST,
        )
        parent_frame = SplitRelaxedLPFrame(
            _csr_ro((0, 0)),
            _csr_ro((0, 1)),
            _csr_ro((0, 0)),
            _csr_ro((0, 1)),
            _ro([]),
            _ro([]),
        )
        fresh_frame = SplitRelaxedLPFrame(
            _csr_ro((1, 1)),
            _csr_ro([[1.0]]),
            _csr_ro((0, 1)),
            _csr_ro((0, 1)),
            _ro([0.0]),
            _ro([]),
        )
        parent = _problem(
            envelope=parent_envelope,
            parent_digest=_PARENT_DIGEST,
            continuous_ids=parent_continuous_ids,
            binary_ids=parent_binary_ids,
            continuous_lb=_ro([]),
            continuous_ub=_ro([]),
            binary_lb=_ro([-1.0]),
            binary_ub=_ro([0.5]),
            frame=parent_frame,
        )
        fresh = _problem(
            envelope=fresh_envelope,
            parent_digest=_FRESH_DIGEST,
            continuous_ids=fresh_continuous_ids,
            binary_ids=fresh_binary_ids,
            continuous_lb=_ro([-1.0]),
            continuous_ub=_ro([1.0]),
            binary_lb=_ro([-1.0]),
            binary_ub=_ro([0.5]),
            frame=fresh_frame,
        )
        parent_candidate = _candidate(parent)
        fresh_candidate = _candidate(fresh, seed_upper=True)
        self.assertEqual(parent_candidate.factor_primal[0], 0.5)
        descriptor, receipt = _call(
            parent, fresh, parent_candidate, fresh_candidate
        )
        self.assertIsNotNone(descriptor, receipt)
        self.assertEqual(descriptor.parent_lower, 0.5)
        self.assertTrue(receipt["strict_relaxed_lp_improvement_certified"])

    def test_solver_objective_is_ignored_and_only_fresh_upper_checker_runs(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        parent_candidate = replace(
            parent_candidate, solver_minimization_objective=1.0e300
        )
        fresh_candidate = replace(
            fresh_candidate, solver_minimization_objective=-1.0e300
        )
        from act.back_end.hybridz_tf import (
            strict_split_lp_improvement_certificate as certificate_module,
        )

        real_upper = (
            certificate_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
        )
        with mock.patch.object(
            certificate_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=real_upper,
        ) as checked_upper:
            descriptor, receipt = _call(
                parent, fresh, parent_candidate, fresh_candidate
            )
        self.assertIsNotNone(descriptor, receipt)
        self.assertEqual(checked_upper.call_count, 1)
        self.assertIs(
            checked_upper.call_args.kwargs["objective_envelope"],
            fresh.objective_envelope,
        )
        self.assertFalse(receipt["upper_vs_upper_comparison_used"])

    def test_objective_extension_mismatch_and_stale_anchors_fail_closed(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair(
            fresh_coefficient=2.0
        )
        descriptor, receipt = _call(
            parent, fresh, parent_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("objective semantics differ", receipt["status"])
        self.assertFalse(receipt["strict_relaxed_lp_improvement_certified"])

        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        stale = replace(
            fresh, expected_exact_objective_sha256="00" * 32
        )
        descriptor, receipt = _call(
            parent, stale, parent_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertTrue(receipt["status"].startswith("invalid:"), receipt)

    def test_candidate_nonfinite_receipt_tamper_close_and_cross_frame_rejected(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        nonfinite = np.array(parent_candidate.factor_primal, copy=True)
        nonfinite[0] = np.nan
        nonfinite.setflags(write=False)
        bad_candidate = replace(parent_candidate, factor_primal=nonfinite)
        descriptor, receipt = _call(
            parent, fresh, bad_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("non-finite", receipt["status"])

        tampered_receipt = dict(parent_candidate.receipt)
        tampered_receipt["factor_primal_sha256"] = "00" * 32
        tampered_candidate = replace(
            parent_candidate, receipt=tampered_receipt
        )
        descriptor, receipt = _call(
            parent, fresh, tampered_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("receipt hash mismatch", receipt["status"])

        unclosed_receipt = dict(parent_candidate.receipt)
        unclosed_receipt.pop("receipt_sha256")
        unclosed_receipt["native_model_closed_before_return"] = False
        unclosed_receipt["receipt_sha256"] = _canonical_json_sha256(
            unclosed_receipt
        )
        unclosed = replace(parent_candidate, receipt=unclosed_receipt)
        descriptor, receipt = _call(
            parent, fresh, unclosed, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("receipt contract mismatch", receipt["status"])

        descriptor, receipt = _call(
            parent, fresh, fresh_candidate, parent_candidate
        )
        self.assertIsNone(descriptor)
        self.assertTrue(receipt["status"].startswith("invalid:"), receipt)

    def test_frame_tamper_nonfinite_bounds_and_deadline_fail_closed(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        changed_frame = replace(
            fresh.frame, Auc=_csr_ro([[0.5, 0.0]])
        )
        changed = replace(fresh, frame=changed_frame)
        descriptor, receipt = _call(
            parent, changed, parent_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("binding mismatch", receipt["status"])

        bad_bound = _ro([1.0, np.inf])
        nonfinite = replace(fresh, continuous_ub=bad_bound)
        descriptor, receipt = _call(
            parent, nonfinite, parent_candidate, fresh_candidate
        )
        self.assertIsNone(descriptor)
        self.assertIn("bounds are invalid", receipt["status"])

        descriptor, receipt = _call(
            parent,
            fresh,
            parent_candidate,
            fresh_candidate,
            deadline=time.monotonic() - 1.0,
        )
        self.assertIsNone(descriptor)
        self.assertTrue(receipt["status"].startswith("deadline_exhausted:"))

    def test_no_merged_stack_and_candidate_receipts_are_cross_bound(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        with mock.patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), mock.patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            descriptor, receipt = _call(
                parent, fresh, parent_candidate, fresh_candidate
            )
        self.assertIsNotNone(descriptor, receipt)
        self.assertFalse(receipt["uses_sparse_hstack"])
        self.assertFalse(receipt["uses_sparse_vstack"])
        self.assertFalse(receipt["used_merged_sparse_frame"])
        for name in (
            "parent_candidate_receipt_sha256",
            "fresh_candidate_receipt_sha256",
            "parent_primal_selected_receipt_sha256",
            "fresh_dual_selected_receipt_sha256",
        ):
            self.assertRegex(receipt[name], r"^[0-9a-f]{64}$")

    def test_nonfinite_checker_diagnostics_are_not_hashed_or_copied(self):
        parent, fresh, parent_candidate, fresh_candidate = _continuous_pair()
        from act.back_end.hybridz_tf import (
            strict_split_lp_improvement_certificate as certificate_module,
        )

        real_primal = certificate_module.certify_preformed_split_primal_lower

        def primal_with_infinite_diagnostic(**kwargs):
            lower, inner = real_primal(**kwargs)
            inner["maximum_row_roundoff_guard"] = float("inf")
            return lower, inner

        with mock.patch.object(
            certificate_module,
            "certify_preformed_split_primal_lower",
            side_effect=primal_with_infinite_diagnostic,
        ):
            descriptor, receipt = _call(
                parent, fresh, parent_candidate, fresh_candidate
            )
        self.assertIsNotNone(descriptor, receipt)
        self.assertFalse(receipt["full_checker_receipts_canonical_hashed"])
        self.assertFalse(
            receipt["nonfinite_diagnostics_copied_into_authority_chain"]
        )


if __name__ == "__main__":
    unittest.main()
