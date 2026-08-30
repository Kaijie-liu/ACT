"""Soundness gates for the SAFE-only binary-relaxation LP certificate."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np
from scipy import sparse as sp

from act.back_end.solver import solver_hz
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_binary_relaxed_output_frame,
    _hz_independent_lp_lagrangian_upper,
    _hz_persistent_lp_filter,
    hz_mark_constructively_nonempty,
    hz_objbound_decide,
)


class BinaryRelaxedOutputFrameTests(unittest.TestCase):
    def test_center_shift_and_combined_generator_match_z_coordinates(self):
        center, generator, center_error = (
            _hz_binary_relaxed_output_frame(
                np.array([3.0], dtype=np.float64),
                sp.csr_matrix([[5.0]], dtype=np.float64),
                sp.csr_matrix([[2.0]], dtype=np.float64),
            )
        )
        # 3 + 5*x + 2*s == 1 + 5*x + 4*z for s = 2*z - 1.
        np.testing.assert_array_equal(center, np.array([1.0]))
        np.testing.assert_array_equal(
            generator.toarray(),
            np.array([[5.0, 4.0]]),
        )
        self.assertEqual(generator.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(center_error)))
        self.assertTrue(np.all(center_error >= 0))
        for x in (-1.0, 0.25, 1.0):
            for z in (0.0, 1.0):
                signed = 2.0 * z - 1.0
                original = 3.0 + 5.0 * x + 2.0 * signed
                relaxed_frame = float(
                    center[0] + generator.toarray()[0] @ [x, z]
                )
                self.assertEqual(original, relaxed_frame)

    def test_key_rlt_row_tightens_upper_and_both_bounds_are_sound(self):
        # Let w=x*z.  The source row x<=z, multiplied by (1-z), gives
        # the valid degree-one RLT row x-w<=0 because z is binary.  Standard
        # product-hull rows alone admit (x,w,z)=(1/2,0,1/2); the RLT row
        # removes exactly that fractional gap.
        center, generator, center_error = (
            _hz_binary_relaxed_output_frame(
                np.array([0.0], dtype=np.float64),
                sp.csr_matrix([[1.0, -1.0]], dtype=np.float64),
                sp.csr_matrix((1, 1), dtype=np.float64),
            )
        )
        base_A = np.array(
            [
                [1.0, 0.0, -1.0],   # x <= z
                [-1.0, 1.0, 0.0],   # w <= x
                [0.0, 1.0, -1.0],   # w <= z
                [1.0, -1.0, 1.0],   # w >= x+z-1
                [0.0, -1.0, 0.0],   # w >= 0
                [-1.0, 0.0, 0.0],   # x >= 0
            ],
            dtype=np.float64,
        )
        base_ru = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        rlt_row = np.array([[1.0, -1.0, 0.0]], dtype=np.float64)
        with_rlt_A = sp.csr_matrix(
            np.vstack([base_A, rlt_row]),
            dtype=np.float64,
        )
        with_rlt_ru = np.concatenate([base_ru, np.array([0.0])])
        with_rlt_dual = np.zeros(with_rlt_A.shape[0], dtype=np.float64)
        with_rlt_dual[-1] = -1.0
        with_rlt, receipt = _hz_independent_lp_lagrangian_upper(
            c=center,
            Gc=generator,
            C_row=np.array([1.0], dtype=np.float64),
            threshold=0.1,
            A=with_rlt_A,
            rl=np.full(with_rlt_A.shape[0], -np.inf),
            ru=with_rlt_ru,
            lb=np.array([-1.0, -1.0, 0.0], dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
            # q=(1,-1,0) is exactly the critical RLT row.
            row_dual=with_rlt_dual,
            center_error=center_error,
        )
        without_rlt, loose_receipt = (
            _hz_independent_lp_lagrangian_upper(
                c=center,
                Gc=generator,
                C_row=np.array([1.0], dtype=np.float64),
                threshold=0.1,
                A=sp.csr_matrix(base_A, dtype=np.float64),
                rl=np.full(base_A.shape[0], -np.inf),
                ru=base_ru,
                lb=np.array([-1.0, -1.0, 0.0], dtype=np.float64),
                ub=np.ones(3, dtype=np.float64),
                row_dual=np.zeros(base_A.shape[0], dtype=np.float64),
                center_error=center_error,
            )
        )
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertEqual(loose_receipt["status"], "verified_upper")
        self.assertLess(np.longdouble(with_rlt), 0.0)
        self.assertGreater(np.longdouble(without_rlt), 0.0)
        self.assertLess(np.longdouble(with_rlt), np.longdouble(without_rlt))

        # Every integral product point has margin -0.1.  After deleting the
        # RLT row the fractional point below is feasible with margin +0.4.
        # Both certificate values remain outward upper bounds, so deletion
        # loses tightness but cannot create false SAFE authority.
        fractional = np.array([0.5, 0.0, 0.5], dtype=np.float64)
        self.assertTrue(np.all(base_A @ fractional <= base_ru))
        self.assertGreater(float((rlt_row @ fractional).item()), 0.0)
        fractional_margin = fractional[0] - fractional[1] - 0.1
        self.assertGreaterEqual(
            np.longdouble(with_rlt),
            np.longdouble(-0.1),
        )
        self.assertGreaterEqual(
            np.longdouble(without_rlt),
            np.longdouble(fractional_margin),
        )


@unittest.skipUnless(
    solver_hz._HAS_HIGHSPY and solver_hz._HAS_SCIPY,
    "persistent LP candidate generation requires HiGHS and SciPy",
)
class BinaryRelaxedPersistentCertificateTests(unittest.TestCase):
    @staticmethod
    def _run(*, A, rl, ru, threshold=-0.5):
        return _hz_persistent_lp_filter(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix([[1.0]], dtype=np.float64),
            C=np.array([[1.0]], dtype=np.float64),
            t=np.array([threshold], dtype=np.float64),
            candidate_rows=np.array([0], dtype=np.int64),
            A=sp.csr_matrix(A, dtype=np.float64),
            rl=np.asarray(rl, dtype=np.float64),
            ru=np.asarray(ru, dtype=np.float64),
            lb=np.array([0.0], dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
            deadline=time.monotonic() + 4.0,
            time_budget=2.0,
            tol=1e-9,
        )

    @staticmethod
    def _run_product_rlt(*, include_rlt):
        base_A = np.array(
            [
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
                [0.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        base_ru = np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if include_rlt:
            base_A = np.vstack(
                [base_A, np.array([[1.0, -1.0, 0.0]])]
            )
            base_ru = np.concatenate([base_ru, np.array([0.0])])
        return _hz_persistent_lp_filter(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, -1.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 1), dtype=np.float64),
            C=np.array([[1.0]], dtype=np.float64),
            t=np.array([0.1], dtype=np.float64),
            candidate_rows=np.array([0], dtype=np.int64),
            A=sp.csr_matrix(base_A, dtype=np.float64),
            rl=np.full(base_A.shape[0], -np.inf),
            ru=base_ru,
            lb=np.array([-1.0, -1.0, 0.0], dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
            deadline=time.monotonic() + 4.0,
            time_budget=2.0,
            tol=1e-9,
        )

    def test_negative_binary_relaxation_upper_has_safe_authority_only(self):
        survivors, stats, witness = self._run(
            A=[[1.0]],
            rl=[-np.inf],
            ru=[0.0],
        )
        np.testing.assert_array_equal(
            survivors,
            np.zeros(0, dtype=np.int64),
        )
        self.assertIsNone(witness)
        self.assertTrue(stats["lp_safe_certificate_eligible"])
        self.assertTrue(
            stats["lp_binary_relaxation_certificate_eligible"]
        )
        self.assertFalse(stats["lp_certificate_v1_eligible"])
        self.assertFalse(stats["lp_candidate_witness_eligible"])
        self.assertTrue(stats["lp_proof_authority"])
        self.assertEqual(stats["lp_certified_rows"], 1)
        self.assertEqual(stats["lp_candidate_witness_rows"], 0)
        self.assertEqual(stats["lp_validated_witness_rows"], 0)
        self.assertLess(stats["lp_cert_max_upper"], 0.0)

    def test_parent_binary_hz_is_closed_by_relaxed_safe_certificate(self):
        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ac=sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix((1, 0), dtype=np.float64),
            Aub=sp.csr_matrix([[1.0]], dtype=np.float64),
            ub=np.array([-1.0], dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.array([7001], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(
            hz,
            "binary_relax_lp_certificate_toy",
        )
        verdict, witness = hz_objbound_decide(
            hz,
            np.array([[1.0]], dtype=np.float64),
            np.array([-0.5], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            base_witness_precheck=False,
            lp_prefilter_fraction=0.9,
            lp_prefilter_max_seconds=1.5,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["cube_pruned_rows"], 0)
        self.assertTrue(
            stats["lp_binary_relaxation_certificate_eligible"]
        )
        self.assertFalse(stats["lp_candidate_witness_eligible"])
        self.assertEqual(stats["lp_certified_rows"], 1)
        self.assertTrue(stats["all_rivals_covered"])

    def test_nonintegral_positive_binary_candidate_never_emits_witness(self):
        # Without x-w<=0, the unique positive optimum is the fractional
        # product point (x,w,z)=(1/2,0,1/2), with margin +0.4.  Integer
        # assignments still have exact margin -0.1, so accepting the LP
        # primal as a witness would be unsound.
        survivors, stats, witness = self._run_product_rlt(
            include_rlt=False,
        )
        np.testing.assert_array_equal(
            survivors,
            np.array([0], dtype=np.int64),
        )
        self.assertIsNone(witness)
        self.assertFalse(stats["lp_candidate_witness_eligible"])
        self.assertEqual(stats["lp_candidate_witness_rows"], 0)
        self.assertEqual(stats["lp_validated_witness_rows"], 0)
        self.assertEqual(stats["lp_relaxed_nonwitness_rows"], 1)
        self.assertEqual(stats["lp_certified_rows"], 0)
        self.assertGreaterEqual(stats["lp_cert_max_upper"], 0.4)

        closed, closed_stats, closed_witness = self._run_product_rlt(
            include_rlt=True,
        )
        np.testing.assert_array_equal(
            closed,
            np.zeros(0, dtype=np.int64),
        )
        self.assertIsNone(closed_witness)
        self.assertEqual(closed_stats["lp_certified_rows"], 1)
        self.assertLess(closed_stats["lp_cert_max_upper"], 0.0)

    def test_continuous_v1_path_does_not_construct_binary_frame(self):
        with mock.patch.object(
            solver_hz,
            "_hz_binary_relaxed_output_frame",
            side_effect=AssertionError(
                "continuous path must retain its original output frame"
            ),
        ):
            survivors, stats, witness = _hz_persistent_lp_filter(
                c=np.array([0.0], dtype=np.float64),
                Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
                Gb=sp.csr_matrix((1, 0), dtype=np.float64),
                C=np.array([[1.0]], dtype=np.float64),
                t=np.array([0.5], dtype=np.float64),
                candidate_rows=np.array([0], dtype=np.int64),
                A=sp.csr_matrix([[1.0]], dtype=np.float64),
                rl=np.array([0.0], dtype=np.float64),
                ru=np.array([0.0], dtype=np.float64),
                lb=np.array([-1.0], dtype=np.float64),
                ub=np.array([1.0], dtype=np.float64),
                deadline=time.monotonic() + 4.0,
                time_budget=2.0,
                tol=1e-9,
            )
        np.testing.assert_array_equal(
            survivors,
            np.zeros(0, dtype=np.int64),
        )
        self.assertIsNone(witness)
        self.assertTrue(stats["lp_certificate_v1_eligible"])
        self.assertTrue(stats["lp_candidate_witness_eligible"])
        self.assertEqual(stats["lp_certified_rows"], 1)


if __name__ == "__main__":
    unittest.main()
