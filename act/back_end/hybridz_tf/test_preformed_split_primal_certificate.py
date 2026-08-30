from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import gc
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.preformed_split_primal_certificate import (
    DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS,
    certify_preformed_split_primal_lower,
)
from act.back_end.solver.solver_hz import (
    _HZPreformedFactorObjectiveEnvelope,
    _hz_form_exact_factor_objective_envelope_from_live_split_blocks,
)


_PARENT = "6d" * 32
_OBJECTIVE_ID = "toy:preformed:split:primal"


def _ro(values, *, dtype=np.float64):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _csr_ro(values):
    matrix = sp.csr_matrix(values, dtype=np.float64)
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.setflags(write=False)
    return matrix


def _form_envelope(
    *,
    center,
    continuous_coefficients,
    binary_coefficients,
    continuous_ids,
    binary_ids,
    objective_id=_OBJECTIVE_ID,
    parent=_PARENT,
):
    continuous_coefficients = np.asarray(
        continuous_coefficients, dtype=np.float64
    ).reshape(1, -1)
    binary_coefficients = np.asarray(
        binary_coefficients, dtype=np.float64
    ).reshape(1, -1)
    envelope, receipt = (
        _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
            c=np.asarray([center], dtype=np.float64),
            Gc=sp.csr_matrix(continuous_coefficients),
            Gb=sp.csr_matrix(binary_coefficients),
            C_row=np.asarray([1.0], dtype=np.float64),
            threshold=0.0,
            continuous_col_ids=np.asarray(continuous_ids, dtype=np.int64),
            binary_col_ids=np.asarray(binary_ids, dtype=np.int64),
            objective_id=objective_id,
            parent_semantic_digest=parent,
        )
    )
    if receipt["status"] != "formed":
        raise AssertionError(receipt)
    return envelope


def _base_case():
    continuous_ids = _ro([901, 3, 44], dtype=np.int64)
    binary_ids = _ro([77, 1001], dtype=np.int64)
    envelope = _form_envelope(
        center=-0.125,
        continuous_coefficients=[1.5, -0.25, 2.0],
        binary_coefficients=[0.5, -1.0],
        continuous_ids=continuous_ids,
        binary_ids=binary_ids,
    )
    return {
        "objective_envelope": envelope,
        "expected_parent_semantic_digest": _PARENT,
        "expected_objective_id": _OBJECTIVE_ID,
        "expected_objective_binding_sha256": (
            envelope.objective_binding_sha256
        ),
        "continuous_col_ids": continuous_ids,
        "binary_col_ids": binary_ids,
        "Auc": _csr_ro(
            np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
                dtype=np.float64,
            )
        ),
        "Aub": _csr_ro((3, 2)),
        "Ac": _csr_ro(
            np.asarray([[1.0, 1.0, 0.0]], dtype=np.float64)
        ),
        "Ab": _csr_ro((1, 2)),
        "ub": _ro([0.5, 0.0, 1.0]),
        "b": _ro([0.25]),
        "continuous_lb": _ro([-1.0, -1.0, -1.0]),
        "continuous_ub": _ro([1.0, 1.0, 1.0]),
        "binary_lb": _ro([-1.0, -1.0]),
        "binary_ub": _ro([1.0, 1.0]),
        "continuous_candidate": _ro([0.5, -0.25, 0.75]),
        # Deliberately fractional: this is a relaxed-LP primal certificate.
        "binary_candidate": _ro([0.25, -0.5]),
        "deadline": time.monotonic() + 30.0,
    }


def _call(case, **updates):
    arguments = dict(case)
    arguments.update(updates)
    if "deadline" not in updates:
        arguments["deadline"] = time.monotonic() + 30.0
    return certify_preformed_split_primal_lower(**arguments)


class TestPreformedSplitPrimalCertificate(unittest.TestCase):
    def test_tight_rows_equalities_fractional_binary_and_exact_lower(self):
        lower, receipt = _call(_base_case())
        expected = Fraction(45, 16)
        self.assertEqual(_fraction(lower), expected)
        self.assertEqual(receipt["status"], "verified_numeric_frame_primal_lower")
        self.assertTrue(receipt["proof_authority"])
        self.assertTrue(receipt["numeric_frame_authority"])
        self.assertTrue(receipt["lower_certificate_authority"])
        self.assertFalse(receipt["parent_binding_authority"])
        self.assertFalse(receipt["verdict_authority"])
        self.assertFalse(receipt["pcoh_authority"])
        self.assertFalse(receipt["solver_status_authority"])
        self.assertTrue(receipt["authority_input_identity_rechecked"])
        self.assertTrue(receipt["authority_input_readonly_rechecked"])
        self.assertFalse(receipt["hostile_concurrent_aba_resistance"])
        self.assertTrue(receipt["trusted_no_concurrent_mutation_required"])
        self.assertTrue(receipt["binary_relaxation_bounds_only"])
        self.assertFalse(receipt["binary_integrality_required"])
        self.assertEqual(receipt["upper_rows_exact_replayed"], 1)
        self.assertEqual(receipt["equality_rows_exact_replayed"], 1)
        self.assertGreaterEqual(receipt["upper_rows_interval_verified"], 2)
        self.assertTrue(receipt["numeric_frame_unchanged"])
        self.assertTrue(receipt["stable_ids_cross_bound"])
        self.assertTrue(receipt["objective_binding_cross_bound"])
        self.assertGreater(receipt["stable_id_sort_copy_bytes"], 0)

    def test_random_dyadic_fraction_objective_oracle(self):
        for seed in range(32):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(9000 + seed)
                n_continuous = 4
                n_binary = 3
                continuous_coefficients = (
                    rng.integers(-16, 17, size=n_continuous).astype(np.float64)
                    / 8.0
                )
                binary_coefficients = (
                    rng.integers(-16, 17, size=n_binary).astype(np.float64)
                    / 8.0
                )
                center = float(rng.integers(-8, 9)) / 8.0
                continuous_point = (
                    rng.integers(-8, 9, size=n_continuous).astype(np.float64)
                    / 8.0
                )
                binary_point = (
                    rng.integers(-8, 9, size=n_binary).astype(np.float64)
                    / 8.0
                )
                continuous_ids = _ro(
                    np.arange(100, 100 + n_continuous), dtype=np.int64
                )
                binary_ids = _ro(
                    np.arange(1000, 1000 + n_binary), dtype=np.int64
                )
                envelope = _form_envelope(
                    center=center,
                    continuous_coefficients=continuous_coefficients,
                    binary_coefficients=binary_coefficients,
                    continuous_ids=continuous_ids,
                    binary_ids=binary_ids,
                )
                case = {
                    "objective_envelope": envelope,
                    "expected_parent_semantic_digest": _PARENT,
                    "expected_objective_id": _OBJECTIVE_ID,
                    "expected_objective_binding_sha256": (
                        envelope.objective_binding_sha256
                    ),
                    "continuous_col_ids": continuous_ids,
                    "binary_col_ids": binary_ids,
                    "Auc": _csr_ro((0, n_continuous)),
                    "Aub": _csr_ro((0, n_binary)),
                    "Ac": _csr_ro((0, n_continuous)),
                    "Ab": _csr_ro((0, n_binary)),
                    "ub": _ro([]),
                    "b": _ro([]),
                    "continuous_lb": _ro(-np.ones(n_continuous)),
                    "continuous_ub": _ro(np.ones(n_continuous)),
                    "binary_lb": _ro(-np.ones(n_binary)),
                    "binary_ub": _ro(np.ones(n_binary)),
                    "continuous_candidate": _ro(continuous_point),
                    "binary_candidate": _ro(binary_point),
                    "deadline": time.monotonic() + 30.0,
                }
                lower, receipt = _call(case)
                exact = _fraction(center) + sum(
                    _fraction(coefficient) * _fraction(point)
                    for coefficient, point in zip(
                        np.concatenate(
                            [continuous_coefficients, binary_coefficients]
                        ),
                        np.concatenate([continuous_point, binary_point]),
                    )
                )
                self.assertIsNotNone(lower, receipt)
                self.assertLessEqual(_fraction(lower), exact)
                self.assertEqual(receipt["objective_exact_fraction"], str(exact))

    def test_random_dyadic_full_feasibility_fraction_oracle(self):
        for seed in range(24):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(12000 + seed)
                nc, nb = 4, 3
                point_c = rng.integers(-4, 5, size=nc).astype(np.float64) / 4.0
                point_b = rng.integers(-4, 5, size=nb).astype(np.float64) / 4.0
                Auc_dense = (
                    rng.integers(-4, 5, size=(4, nc)).astype(np.float64) / 4.0
                )
                Aub_dense = (
                    rng.integers(-4, 5, size=(4, nb)).astype(np.float64) / 4.0
                )
                Ac_dense = (
                    rng.integers(-4, 5, size=(2, nc)).astype(np.float64) / 4.0
                )
                Ab_dense = (
                    rng.integers(-4, 5, size=(2, nb)).astype(np.float64) / 4.0
                )

                def exact_rows(left, right):
                    return [
                        sum(
                            _fraction(coefficient) * _fraction(value)
                            for coefficient, value in zip(
                                np.concatenate([left[row], right[row]]),
                                np.concatenate([point_c, point_b]),
                            )
                        )
                        for row in range(left.shape[0])
                    ]

                upper_exact = exact_rows(Auc_dense, Aub_dense)
                equality_exact = exact_rows(Ac_dense, Ab_dense)
                # Include one tight row and three strictly feasible rows.
                upper_rhs = _ro(
                    [
                        float(value + (Fraction(0) if row == 0 else Fraction(1, 4)))
                        for row, value in enumerate(upper_exact)
                    ]
                )
                equality_rhs = _ro([float(value) for value in equality_exact])
                ids_c = _ro(np.arange(100, 100 + nc), dtype=np.int64)
                ids_b = _ro(np.arange(1000, 1000 + nb), dtype=np.int64)
                envelope = _form_envelope(
                    center=0.0,
                    continuous_coefficients=np.zeros(nc),
                    binary_coefficients=np.zeros(nb),
                    continuous_ids=ids_c,
                    binary_ids=ids_b,
                )
                case = {
                    "objective_envelope": envelope,
                    "expected_parent_semantic_digest": _PARENT,
                    "expected_objective_id": _OBJECTIVE_ID,
                    "expected_objective_binding_sha256": (
                        envelope.objective_binding_sha256
                    ),
                    "continuous_col_ids": ids_c,
                    "binary_col_ids": ids_b,
                    "Auc": _csr_ro(Auc_dense),
                    "Aub": _csr_ro(Aub_dense),
                    "Ac": _csr_ro(Ac_dense),
                    "Ab": _csr_ro(Ab_dense),
                    "ub": upper_rhs,
                    "b": equality_rhs,
                    "continuous_lb": _ro(-np.ones(nc)),
                    "continuous_ub": _ro(np.ones(nc)),
                    "binary_lb": _ro(-np.ones(nb)),
                    "binary_ub": _ro(np.ones(nb)),
                    "continuous_candidate": _ro(point_c),
                    "binary_candidate": _ro(point_b),
                    "deadline": time.monotonic() + 30.0,
                }
                lower, receipt = _call(case)
                self.assertEqual(lower, 0.0, receipt)
                self.assertGreaterEqual(receipt["upper_rows_exact_replayed"], 1)
                self.assertEqual(receipt["equality_rows_exact_replayed"], 2)
                violated = np.array(upper_rhs, copy=True)
                violated[0] = float(upper_exact[0] - Fraction(1, 4))
                violated.setflags(write=False)
                lower, receipt = _call(case, ub=violated)
                self.assertIsNone(lower)
                self.assertTrue(
                    receipt["status"].startswith("infeasible:upper_"), receipt
                )

    def test_clear_upper_and_exact_equality_violations_fail_closed(self):
        case = _base_case()
        bad_upper = _ro([-10.0, 0.0, 1.0])
        lower, receipt = _call(case, ub=bad_upper)
        self.assertIsNone(lower)
        self.assertEqual(receipt["status"], "infeasible:upper_interval_violation")
        self.assertFalse(receipt["proof_authority"])

        lower, receipt = _call(case, b=_ro([0.5]))
        self.assertIsNone(lower)
        self.assertEqual(receipt["status"], "infeasible:equality_violation")
        self.assertFalse(receipt["proof_authority"])

    def test_bounds_are_exact_but_binary_integrality_is_not_required(self):
        case = _base_case()
        lower, receipt = _call(case, binary_candidate=_ro([0.125, 0.75]))
        self.assertIsNotNone(lower, receipt)
        self.assertTrue(receipt["binary_relaxation_bounds_only"])
        lower, receipt = _call(
            case,
            continuous_candidate=_ro([1.25, -0.25, 0.75]),
        )
        self.assertIsNone(lower)
        self.assertIn("violates an exact bound", receipt["status"])

    def test_zero_blocks_and_subnormal_huge_cancellation(self):
        continuous_ids = _ro([], dtype=np.int64)
        binary_ids = _ro([], dtype=np.int64)
        envelope = _form_envelope(
            center=-0.25,
            continuous_coefficients=[],
            binary_coefficients=[],
            continuous_ids=continuous_ids,
            binary_ids=binary_ids,
        )
        zero_case = {
            "objective_envelope": envelope,
            "expected_parent_semantic_digest": _PARENT,
            "expected_objective_id": _OBJECTIVE_ID,
            "expected_objective_binding_sha256": envelope.objective_binding_sha256,
            "continuous_col_ids": continuous_ids,
            "binary_col_ids": binary_ids,
            "Auc": _csr_ro((1, 0)),
            "Aub": _csr_ro((1, 0)),
            "Ac": _csr_ro((1, 0)),
            "Ab": _csr_ro((1, 0)),
            "ub": _ro([0.0]),
            "b": _ro([0.0]),
            "continuous_lb": _ro([]),
            "continuous_ub": _ro([]),
            "binary_lb": _ro([]),
            "binary_ub": _ro([]),
            "continuous_candidate": _ro([]),
            "binary_candidate": _ro([]),
            "deadline": time.monotonic() + 30.0,
        }
        lower, receipt = _call(zero_case)
        self.assertEqual(lower, -0.25)
        self.assertEqual(receipt["upper_rows_exact_replayed"], 1)

        tiny = np.nextafter(0.0, 1.0)
        huge = np.finfo(np.float64).max
        ids = _ro([5, 6, 7], dtype=np.int64)
        envelope = _form_envelope(
            center=0.0,
            continuous_coefficients=[huge, -huge, tiny],
            binary_coefficients=[],
            continuous_ids=ids,
            binary_ids=binary_ids,
        )
        cancellation = dict(zero_case)
        cancellation.update(
            {
                "objective_envelope": envelope,
                "expected_objective_binding_sha256": (
                    envelope.objective_binding_sha256
                ),
                "continuous_col_ids": ids,
                "Auc": _csr_ro(
                    np.asarray([[huge, -huge, tiny]], dtype=np.float64)
                ),
                "Ac": _csr_ro(
                    np.asarray([[huge, -huge, tiny]], dtype=np.float64)
                ),
                "continuous_lb": _ro([-1.0, -1.0, -1.0]),
                "continuous_ub": _ro([1.0, 1.0, 1.0]),
                "continuous_candidate": _ro([1.0, 1.0, 0.0]),
            }
        )
        lower, receipt = _call(cancellation)
        self.assertEqual(lower, 0.0, receipt)
        self.assertEqual(receipt["upper_rows_exact_replayed"], 1)
        self.assertEqual(receipt["equality_rows_exact_replayed"], 1)

    def test_stale_forged_tampered_envelope_ids_and_binding_rejected(self):
        case = _base_case()
        failures = (
            {"expected_parent_semantic_digest": "00" * 32},
            {"expected_objective_id": "wrong"},
            {"expected_objective_binding_sha256": "00" * 32},
            {"objective_envelope": object()},
            {
                "continuous_col_ids": _ro(
                    [901, 3, 45], dtype=np.int64
                )
            },
        )
        for updates in failures:
            with self.subTest(updates=tuple(updates)):
                lower, receipt = _call(case, **updates)
                self.assertIsNone(lower)
                self.assertFalse(receipt["proof_authority"])
                self.assertTrue(
                    receipt["status"].startswith("invalid:"), receipt
                )

        envelope = _form_envelope(
            center=0.0,
            continuous_coefficients=[1.0],
            binary_coefficients=[],
            continuous_ids=[10],
            binary_ids=[],
        )
        object.__setattr__(envelope, "_objective_id", "tampered")
        tampered = dict(case)
        tampered["objective_envelope"] = envelope
        lower, receipt = _call(tampered)
        self.assertIsNone(lower)
        self.assertIn("registry identity seal", receipt["status"])

    def test_ids_are_readonly_unique_and_cross_domain_disjoint(self):
        case = _base_case()
        writable = np.asarray([901, 3, 44], dtype=np.int64)
        lower, receipt = _call(case, continuous_col_ids=writable)
        self.assertIsNone(lower)
        self.assertIn("readonly int64", receipt["status"])

        intersect_id = _ro([10], dtype=np.int64)
        envelope = _form_envelope(
            center=0.0,
            continuous_coefficients=[1.0],
            binary_coefficients=[1.0],
            continuous_ids=intersect_id,
            binary_ids=intersect_id,
        )
        intersect = {
            "objective_envelope": envelope,
            "expected_parent_semantic_digest": _PARENT,
            "expected_objective_id": _OBJECTIVE_ID,
            "expected_objective_binding_sha256": envelope.objective_binding_sha256,
            "continuous_col_ids": intersect_id,
            "binary_col_ids": intersect_id,
            "Auc": _csr_ro((0, 1)),
            "Aub": _csr_ro((0, 1)),
            "Ac": _csr_ro((0, 1)),
            "Ab": _csr_ro((0, 1)),
            "ub": _ro([]),
            "b": _ro([]),
            "continuous_lb": _ro([-1.0]),
            "continuous_ub": _ro([1.0]),
            "binary_lb": _ro([-1.0]),
            "binary_ub": _ro([1.0]),
            "continuous_candidate": _ro([0.0]),
            "binary_candidate": _ro([0.0]),
            "deadline": time.monotonic() + 30.0,
        }
        lower, receipt = _call(intersect)
        self.assertIsNone(lower)
        self.assertIn("must be disjoint", receipt["status"])

    def test_candidate_canonical_readonly_finite_contract(self):
        case = _base_case()
        writable = np.asarray([0.5, -0.25, 0.75], dtype=np.float64)
        lower, receipt = _call(case, continuous_candidate=writable)
        self.assertIsNone(lower)
        self.assertIn("readonly before snapshot", receipt["status"])

        backing = np.asarray(
            [0.5, 0.0, -0.25, 0.0, 0.75, 0.0], dtype=np.float64
        )
        noncanonical = backing[::2]
        noncanonical.setflags(write=False)
        lower, receipt = _call(case, continuous_candidate=noncanonical)
        self.assertIsNone(lower)
        self.assertIn("contiguous", receipt["status"])

        nonfinite = _ro([0.5, np.nan, 0.75])
        lower, receipt = _call(case, continuous_candidate=nonfinite)
        self.assertIsNone(lower)
        self.assertIn("non-finite", receipt["status"])

    def test_writable_authority_frame_rejects_original_aba_counterexample(self):
        ids = _ro([10], dtype=np.int64)
        empty_ids = _ro([], dtype=np.int64)
        envelope = _form_envelope(
            center=0.0,
            continuous_coefficients=[0.0],
            binary_coefficients=[],
            continuous_ids=ids,
            binary_ids=empty_ids,
        )
        # Original frame is 1*x <= 0 at x=1 and is infeasible.  The reported
        # audit race changed data 1 -> -1 after pre-hash, let the scan pass,
        # then restored 1 before post-hash.  A writable data owner is now
        # rejected at entry, before any scan can authorize that ABA schedule.
        writable_auc = sp.csr_matrix(
            np.asarray([[1.0]], dtype=np.float64)
        )
        case = {
            "objective_envelope": envelope,
            "expected_parent_semantic_digest": _PARENT,
            "expected_objective_id": _OBJECTIVE_ID,
            "expected_objective_binding_sha256": envelope.objective_binding_sha256,
            "continuous_col_ids": ids,
            "binary_col_ids": empty_ids,
            "Auc": writable_auc,
            "Aub": _csr_ro((1, 0)),
            "Ac": _csr_ro((0, 1)),
            "Ab": _csr_ro((0, 0)),
            "ub": _ro([0.0]),
            "b": _ro([]),
            "continuous_lb": _ro([-1.0]),
            "continuous_ub": _ro([1.0]),
            "binary_lb": _ro([]),
            "binary_ub": _ro([]),
            "continuous_candidate": _ro([1.0]),
            "binary_candidate": _ro([]),
            "deadline": time.monotonic() + 30.0,
        }
        lower, receipt = _call(case)
        self.assertIsNone(lower)
        self.assertIn("strictly readonly", receipt["status"])
        self.assertFalse(receipt["numeric_frame_authority"])

        readonly_auc = _csr_ro(np.asarray([[1.0]], dtype=np.float64))
        for name, writable in (
            ("ub", np.asarray([0.0], dtype=np.float64)),
            ("continuous_lb", np.asarray([-1.0], dtype=np.float64)),
            ("continuous_ub", np.asarray([1.0], dtype=np.float64)),
        ):
            with self.subTest(name=name):
                local = dict(case)
                local["Auc"] = readonly_auc
                local[name] = writable
                lower, receipt = _call(local)
                self.assertIsNone(lower)
                self.assertIn("strictly readonly", receipt["status"])

    def test_authorization_rechecks_flags_and_backing_object_identity(self):
        from act.back_end.hybridz_tf import (
            preformed_split_primal_certificate as certificate_module,
        )

        flags_case = _base_case()
        real_hash = certificate_module._frame_sha256
        calls = 0

        def enable_writes_after_post_hash(**kwargs):
            nonlocal calls
            calls += 1
            value = real_hash(**kwargs)
            if calls == 2:
                flags_case["Auc"].data.setflags(write=True)
            return value

        try:
            with mock.patch.object(
                certificate_module,
                "_frame_sha256",
                side_effect=enable_writes_after_post_hash,
            ):
                lower, receipt = _call(flags_case)
            self.assertIsNone(lower)
            self.assertIn("identity or readonly contract changed", receipt["status"])
            self.assertFalse(receipt["numeric_frame_authority"])
        finally:
            flags_case["Auc"].data.setflags(write=False)

        identity_case = _base_case()
        calls = 0

        def replace_backing_after_post_hash(**kwargs):
            nonlocal calls
            calls += 1
            value = real_hash(**kwargs)
            if calls == 2:
                replacement = np.array(identity_case["Auc"].data, copy=True)
                replacement.setflags(write=False)
                identity_case["Auc"].data = replacement
            return value

        with mock.patch.object(
            certificate_module,
            "_frame_sha256",
            side_effect=replace_backing_after_post_hash,
        ):
            lower, receipt = _call(identity_case)
        self.assertIsNone(lower)
        self.assertIn("identity or readonly contract changed", receipt["status"])
        self.assertFalse(receipt["numeric_frame_authority"])

    def test_deadline_caps_noncanonical_csr_and_frame_rehash_fail_closed(self):
        case = _base_case()
        lower, receipt = _call(case, deadline=time.monotonic() - 1.0)
        self.assertIsNone(lower)
        self.assertTrue(receipt["status"].startswith("deadline_exhausted:"))

        lower, receipt = _call(
            case,
            caps=replace(DEFAULT_PREFORMED_SPLIT_PRIMAL_CAPS, max_columns=2),
        )
        self.assertIsNone(lower)
        self.assertTrue(receipt["status"].startswith("cap_exceeded:"))

        lower, receipt = _call(case, Auc=case["Auc"].tocsc())
        self.assertIsNone(lower)
        self.assertIn("canonical binary64 CSR", receipt["status"])

        from act.back_end.hybridz_tf import (
            preformed_split_primal_certificate as certificate_module,
        )

        real_hash = certificate_module._frame_sha256
        calls = 0

        def changed_on_second(**kwargs):
            nonlocal calls
            calls += 1
            value = real_hash(**kwargs)
            return value if calls == 1 else "00" * 32

        with mock.patch.object(
            certificate_module,
            "_frame_sha256",
            side_effect=changed_on_second,
        ):
            lower, receipt = _call(case)
        self.assertIsNone(lower)
        self.assertIn("numeric frame changed", receipt["status"])
        self.assertFalse(receipt["proof_authority"])

    def test_sparse_stack_functions_and_solver_status_are_never_used(self):
        case = _base_case()
        with mock.patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), mock.patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            lower, receipt = _call(case)
        self.assertIsNotNone(lower, receipt)
        self.assertFalse(receipt["uses_sparse_hstack"])
        self.assertFalse(receipt["uses_sparse_vstack"])
        self.assertFalse(receipt["solver_status_authority"])

    def test_250k_500k_1m_sorted_id_memory_slope_is_snapshot_linear(self):
        peaks = []
        for columns in (250_000, 500_000, 1_000_000):
            with self.subTest(columns=columns):
                ids = np.arange(columns, dtype=np.int64)
                ids.setflags(write=False)
                empty_ids = _ro([], dtype=np.int64)
                envelope = _form_envelope(
                    center=0.0,
                    continuous_coefficients=np.zeros(columns, dtype=np.float64),
                    binary_coefficients=[],
                    continuous_ids=ids,
                    binary_ids=empty_ids,
                )
                lower_bound = _ro(-np.ones(columns, dtype=np.float64))
                upper_bound = _ro(np.ones(columns, dtype=np.float64))
                point = _ro(np.zeros(columns, dtype=np.float64))
                case = {
                    "objective_envelope": envelope,
                    "expected_parent_semantic_digest": _PARENT,
                    "expected_objective_id": _OBJECTIVE_ID,
                    "expected_objective_binding_sha256": (
                        envelope.objective_binding_sha256
                    ),
                    "continuous_col_ids": ids,
                    "binary_col_ids": empty_ids,
                    "Auc": _csr_ro((0, columns)),
                    "Aub": _csr_ro((0, 0)),
                    "Ac": _csr_ro((0, columns)),
                    "Ab": _csr_ro((0, 0)),
                    "ub": _ro([]),
                    "b": _ro([]),
                    "continuous_lb": lower_bound,
                    "continuous_ub": upper_bound,
                    "binary_lb": _ro([]),
                    "binary_ub": _ro([]),
                    "continuous_candidate": point,
                    "binary_candidate": _ro([]),
                    "deadline": time.monotonic() + 60.0,
                }
                tracemalloc.start()
                lower, receipt = _call(case, deadline=time.monotonic() + 60.0)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.assertEqual(lower, 0.0, receipt)
                self.assertEqual(receipt["candidate_snapshot_bytes"], 8 * columns)
                self.assertEqual(receipt["stable_id_sort_copy_bytes"], 0)
                peaks.append(peak)
                del case, point, lower_bound, upper_bound, envelope, ids
                gc.collect()
        slopes = [
            (peaks[index + 1] - peaks[index])
            / ((500_000, 1_000_000)[index] - (250_000, 500_000)[index])
            for index in range(2)
        ]
        # One immutable f64 snapshot dominates; bounded chunk temporaries may
        # add allocator noise, but there is no Python object per column.
        self.assertLess(max(slopes), 20.0, (peaks, slopes))


if __name__ == "__main__":
    unittest.main()
