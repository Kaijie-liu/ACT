#!/usr/bin/env python3
"""Strict disconnected gates for the native HiGHS ranged-row loader."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import exact_ranged_row_native_load_sentinel as sentinel
from act.back_end.hybridz_tf.exact_ranged_row_compaction import (
    SignedUpperSource,
    fold_exact_signed_upper_pairs,
    validate_exact_ranged_candidate,
)


def _csr(rows, shape):
    matrix = sp.csr_matrix(np.asarray(rows, dtype=np.float64), shape=shape)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _one_dimensional_band():
    # 2*x <= 6 and -2*x <= -2, hence 1 <= x <= 3 exactly.
    source = SignedUpperSource(
        A_cont=_csr([[2.0], [-2.0]], (2, 1)),
        A_bin=sp.csr_matrix((2, 0), dtype=np.float64),
        upper=np.asarray([6.0, -2.0], dtype=np.float64),
        row_tags=("fraction_band:forward", "fraction_band:reverse"),
    )
    return (
        source,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([10.0], dtype=np.float64),
        np.asarray([-1.0], dtype=np.float64),
    )


class ExactRangedRowNativeLoadSentinelTests(unittest.TestCase):
    def test_c89_ratio_generator_is_canonical_capped_and_adjustable(self) -> None:
        frame = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        source = frame.source
        self.assertEqual(source.A_cont.shape, (1024, 655))
        self.assertEqual(source.A_bin.shape, (1024, 0))
        self.assertEqual(source.A_cont.nnz, 75608)
        self.assertTrue(source.A_cont.has_canonical_format)
        self.assertTrue(source.A_cont.has_sorted_indices)
        self.assertEqual(source.A_cont.indices.dtype, np.dtype(np.int32))
        self.assertEqual(source.A_cont.indptr.dtype, np.dtype(np.int32))
        self.assertFalse(source.A_cont.data.flags.writeable)
        self.assertFalse(source.upper.flags.writeable)
        self.assertIsInstance(frame.metadata, MappingProxyType)
        self.assertIs(frame.metadata["synthetic_only"], True)
        self.assertIs(frame.metadata["real_model_allowed"], False)
        self.assertIs(frame.metadata["large_model_allowed"], False)
        self.assertLessEqual(
            frame.metadata["source_constraint_nnz"], frame.metadata["nnz_cap"]
        )
        candidate = fold_exact_signed_upper_pairs(source)
        self.assertTrue(validate_exact_ranged_candidate(source, candidate))
        self.assertEqual(candidate.A_cont.shape[0], 512)
        self.assertEqual(candidate.A_cont.nnz, source.A_cont.nnz // 2)

        smaller = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=160)
        self.assertEqual(smaller.source.A_cont.shape[0], 512)
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError, "source_row_cap_exceeded"
        ):
            sentinel.make_c89_ratio_signed_upper_source(scale_divisor=1)
        for bad in (True, np.int64(80), 0, -1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(
                sentinel.ExactRangedNativeLoadSentinelError
            ):
                sentinel.make_c89_ratio_signed_upper_source(scale_divisor=bad)

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_native_baseline_and_range_have_same_exact_small_oracle(self) -> None:
        source, column_lower, column_upper, objective = _one_dimensional_band()
        original_digest = source.source_sha256
        result = sentinel.run_native_ranged_equivalence_sentinel(
            source=source,
            column_lower=column_lower,
            column_upper=column_upper,
            objective=objective,
        )
        self.assertEqual(source.source_sha256, original_digest)
        self.assertEqual(result.source_sha256, original_digest)
        self.assertEqual(result.baseline.model_status, result.candidate.model_status)
        self.assertEqual(
            result.baseline.model_status, str(sentinel._highspy.HighsModelStatus.kOptimal)
        )
        self.assertAlmostEqual(result.baseline.objective_value, -3.0, places=12)
        self.assertAlmostEqual(result.candidate.objective_value, -3.0, places=12)
        self.assertAlmostEqual(float(result.baseline.primal[0]), 3.0, places=12)
        self.assertAlmostEqual(float(result.candidate.primal[0]), 3.0, places=12)
        self.assertEqual((result.baseline.rows, result.candidate.rows), (2, 1))
        self.assertEqual(
            (result.baseline.constraint_nnz, result.candidate.constraint_nnz),
            (2, 1),
        )

        # Fraction is the independent oracle, not a float reconstruction of
        # the candidate: min -x over 0<=x<=10 and 2<=2x<=6 is exactly -3.
        x = Fraction(3)
        self.assertTrue(Fraction(0) <= x <= Fraction(10))
        self.assertTrue(Fraction(2) <= 2 * x <= Fraction(6))
        self.assertEqual(-x, Fraction(-3))
        for native in (result.baseline, result.candidate):
            observed = Fraction(str(float(native.primal[0]))).limit_denominator()
            self.assertEqual(observed, x)
            self.assertFalse(native.primal.flags.writeable)
            self.assertIs(native.receipt["continuous_audit_columns_only"], True)
            self.assertIs(native.receipt["integrality_loaded"], False)
            self.assertIs(native.receipt["branch_and_bound_called"], False)

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_paired_benchmark_reports_fold_load_total_and_loader_only_gate(self) -> None:
        frame = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        report = sentinel.benchmark_native_ranged_row_loader(
            frame, warmup_pairs=2, measured_pairs=7
        )
        self.assertEqual((report.warmup_pairs, report.measured_pairs), (2, 7))
        self.assertGreater(report.folding_preprocess_median_seconds, 0.0)
        self.assertGreater(report.baseline_native_load_median_seconds, 0.0)
        self.assertGreater(report.candidate_native_load_median_seconds, 0.0)
        self.assertGreater(report.candidate_total_median_seconds, 0.0)
        self.assertGreaterEqual(report.native_loader_paired_median_speedup, 1.5)
        self.assertTrue(report.native_loader_candidate_supported)
        self.assertEqual(report.native_loader_speedup_gate, 1.5)
        self.assertIs(report.receipt["single_thread"], True)
        self.assertIs(report.receipt["paired_alternating_order"], True)
        self.assertIs(report.receipt["native_load_is_addRows_only"], True)
        self.assertIs(report.receipt["folding_preprocess_reported_separately"], True)
        self.assertIs(report.receipt["total_includes_folding_and_candidate_addRows"], True)
        self.assertIs(report.receipt["gate_applies_only_to_native_loader"], True)
        self.assertIs(report.receipt["solver_run_called"], False)
        # Folding dominates at this disconnected stage; total speed is reported
        # faithfully and is deliberately not a promotion gate.
        self.assertTrue(math.isfinite(report.total_paired_median_speedup))

    def test_benchmark_floor_warmup_repeat_and_exact_type_gates(self) -> None:
        too_small = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=160)
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "benchmark_pair_floor_not_met",
        ):
            sentinel.benchmark_native_ranged_row_loader(too_small)
        frame = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        for kwargs in (
            {"warmup_pairs": 1, "measured_pairs": 7},
            {"warmup_pairs": 2, "measured_pairs": 6},
            {"warmup_pairs": 2, "measured_pairs": 32},
            {"warmup_pairs": 32, "measured_pairs": 7},
            {"warmup_pairs": 10**12, "measured_pairs": 7},
            {"warmup_pairs": True, "measured_pairs": 7},
            {"warmup_pairs": 2, "measured_pairs": np.int64(7)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                sentinel.ExactRangedNativeLoadSentinelError
            ):
                sentinel.benchmark_native_ranged_row_loader(frame, **kwargs)
        with self.assertRaises(sentinel.ExactRangedNativeLoadSentinelError):
            sentinel.benchmark_native_ranged_row_loader(object())

    def test_frame_factory_seal_snapshot_and_metadata_fail_closed(self) -> None:
        frame = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        self.assertFalse(frame.column_lower.flags.writeable)
        self.assertFalse(frame.column_upper.flags.writeable)
        self.assertFalse(frame.objective.flags.writeable)
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "c89_ratio_frame_must_come_from_factory",
        ):
            sentinel.C89RatioSyntheticFrame(
                source=frame.source,
                column_lower=frame.column_lower,
                column_upper=frame.column_upper,
                objective=frame.objective,
                metadata=frame.metadata,
            )

        mutations = []
        extra = dict(frame.metadata)
        extra["caller_claimed_safe"] = True
        mutations.append(extra)
        missing = dict(frame.metadata)
        del missing["proof_authority"]
        mutations.append(missing)
        real = dict(frame.metadata)
        real["real_model_allowed"] = True
        mutations.append(real)
        authority = dict(frame.metadata)
        authority["proof_authority"] = True
        mutations.append(authority)
        bool_collision = dict(frame.metadata)
        bool_collision["scale_divisor"] = True
        mutations.append(bool_collision)
        float_collision = dict(frame.metadata)
        float_collision["source_rows"] = float(float_collision["source_rows"])
        mutations.append(float_collision)
        stale_formula = dict(frame.metadata)
        stale_formula["source_payload_bytes"] += 1
        mutations.append(stale_formula)
        stale_digest = dict(frame.metadata)
        stale_digest["source_sha256"] = "0" * 64
        mutations.append(stale_digest)
        for metadata in mutations:
            with self.subTest(metadata=metadata), self.assertRaises(
                sentinel.ExactRangedNativeLoadSentinelError
            ):
                replace(frame, metadata=metadata)
        with self.assertRaises(sentinel.ExactRangedNativeLoadSentinelError):
            replace(frame, metadata=list(frame.metadata.items()))

        altered_matrix = frame.source.A_cont.copy()
        pair_count = frame.metadata["pair_count"]
        altered_matrix.data[int(altered_matrix.indptr[0])] = 0.25
        altered_matrix.data[int(altered_matrix.indptr[pair_count])] = -0.25
        altered_source = SignedUpperSource(
            altered_matrix,
            frame.source.A_bin,
            frame.source.upper,
            frame.source.row_tags,
        )
        rebound_metadata = dict(frame.metadata)
        rebound_metadata["source_sha256"] = altered_source.source_sha256
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "c89_ratio_factory_binding_is_stale",
        ):
            replace(
                frame,
                source=altered_source,
                metadata=rebound_metadata,
            )

        # Reproduce the stronger attack: import the nominally private token,
        # forge a matching seal, and update an exact forward/reverse coefficient
        # pair together with source digest and metadata.  The seal-based
        # constructor alone cannot establish provenance; the benchmark's
        # deterministic factory replay must reject the changed coefficient bits.
        forged_seal = sentinel._C89FrameFactorySeal(
            token=sentinel._C89_FRAME_FACTORY_TOKEN,
            source_sha256=altered_source.source_sha256,
            metadata_sha256=sentinel._metadata_sha256(rebound_metadata),
        )
        forged = sentinel.C89RatioSyntheticFrame(
            source=altered_source,
            column_lower=frame.column_lower,
            column_upper=frame.column_upper,
            objective=frame.objective,
            metadata=rebound_metadata,
            _factory_seal=forged_seal,
        )
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "c89_ratio_payload_does_not_match_deterministic_factory",
        ):
            sentinel.benchmark_native_ranged_row_loader(forged)

        synchronized_poison = sentinel.make_c89_ratio_signed_upper_source(
            scale_divisor=80
        )
        object.__setattr__(synchronized_poison, "source", altered_source)
        object.__setattr__(
            synchronized_poison,
            "metadata",
            MappingProxyType(rebound_metadata),
        )
        object.__setattr__(synchronized_poison, "_factory_seal", forged_seal)
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "c89_ratio_payload_does_not_match_deterministic_factory",
        ):
            sentinel.benchmark_native_ranged_row_loader(synchronized_poison)

        # A frozen dataclass can still be poisoned with object.__setattr__ by a
        # hostile caller.  The benchmark reconstructs and validates a private
        # snapshot, so it must reject rather than replacing the fake metadata
        # with factory values and thereby laundering it.
        poisoned = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        fake = dict(poisoned.metadata)
        fake["verdict_authority"] = True
        object.__setattr__(poisoned, "metadata", MappingProxyType(fake))
        with self.assertRaises(sentinel.ExactRangedNativeLoadSentinelError):
            sentinel.benchmark_native_ranged_row_loader(poisoned)

        poisoned = sentinel.make_c89_ratio_signed_upper_source(scale_divisor=80)
        live_lower = poisoned.column_lower.copy()
        live_lower[0] = -2.0
        object.__setattr__(poisoned, "column_lower", live_lower)
        with self.assertRaisesRegex(
            sentinel.ExactRangedNativeLoadSentinelError,
            "c89_ratio_payload_does_not_match_deterministic_factory",
        ):
            sentinel.benchmark_native_ranged_row_loader(poisoned)

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_vector_type_finite_and_bound_checks_fail_closed(self) -> None:
        source, lower, upper, objective = _one_dimensional_band()
        cases = []
        cases.append((lower.astype(np.float32), upper, objective))
        cases.append((lower, upper, objective.astype(np.float32)))
        poisoned = upper.copy()
        poisoned[0] = np.inf
        cases.append((lower, poisoned, objective))
        poisoned = objective.copy()
        poisoned[0] = np.nan
        cases.append((lower, upper, poisoned))
        cases.append((np.asarray([4.0]), np.asarray([3.0]), objective))
        for bad_lower, bad_upper, bad_objective in cases:
            with self.subTest(case=(bad_lower, bad_upper, bad_objective)), self.assertRaises(
                sentinel.ExactRangedNativeLoadSentinelError
            ):
                sentinel.run_native_ranged_equivalence_sentinel(
                    source=source,
                    column_lower=bad_lower,
                    column_upper=bad_upper,
                    objective=bad_objective,
                )

    def test_nan_inf_index_dtype_and_canonical_sources_are_rejected(self) -> None:
        source, _, _, _ = _one_dimensional_band()
        with self.assertRaises(ValueError):
            SignedUpperSource(
                source.A_cont.astype(np.float32),
                source.A_bin,
                source.upper,
                source.row_tags,
            )
        for bad_value in (np.nan, np.inf, -np.inf):
            poisoned = source.upper.copy()
            poisoned[0] = bad_value
            with self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                SignedUpperSource(
                    source.A_cont, source.A_bin, poisoned, source.row_tags
                )
        duplicate = sp.csr_matrix(
            (
                np.asarray([1.0, 1.0, -2.0], dtype=np.float64),
                np.asarray([0, 0, 0], dtype=np.int32),
                np.asarray([0, 2, 3], dtype=np.int32),
            ),
            shape=(2, 1),
        )
        duplicate.has_sorted_indices = True
        duplicate.has_canonical_format = True
        with self.assertRaises(ValueError):
            SignedUpperSource(
                duplicate, source.A_bin, source.upper, source.row_tags
            )
        out_of_range = source.A_cont.copy()
        out_of_range.indices[0] = 1
        with self.assertRaises(ValueError):
            SignedUpperSource(
                out_of_range, source.A_bin, source.upper, source.row_tags
            )

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_baseexception_during_load_still_closes_model(self) -> None:
        real_ok = sentinel._highspy.HighsStatus.kOk

        class ExplodingModel:
            def __init__(self):
                self.closed = False

            def setOptionValue(self, *_args):
                return real_ok

            def addCols(self, *_args):
                return real_ok

            def addRows(self, *_args):
                raise KeyboardInterrupt("audit interrupt")

            def clear(self):
                self.closed = True
                return real_ok

        model = ExplodingModel()
        matrix = _csr([[1.0]], (1, 1))
        with mock.patch.object(sentinel, "_new_highs", return_value=model):
            with self.assertRaisesRegex(
                sentinel.ExactRangedNativeLoadSentinelError,
                "native_model_operation_failed:KeyboardInterrupt",
            ):
                sentinel._load_native_frame(
                    matrix=matrix,
                    row_lower=np.asarray([-1.0], dtype=np.float64),
                    row_upper=np.asarray([1.0], dtype=np.float64),
                    column_lower=np.asarray([-1.0], dtype=np.float64),
                    column_upper=np.asarray([1.0], dtype=np.float64),
                    objective=np.asarray([0.0], dtype=np.float64),
                    solve=False,
                    route="baseline_upper",
                )
        self.assertTrue(model.closed)

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_baseexception_during_model_close_fails_closed(self) -> None:
        real_ok = sentinel._highspy.HighsStatus.kOk

        class CloseExplodingModel:
            def setOptionValue(self, *_args):
                return real_ok

            def addCols(self, *_args):
                return real_ok

            def addRows(self, *_args):
                return real_ok

            def clear(self):
                raise SystemExit("close failed")

        matrix = _csr([[1.0]], (1, 1))
        with mock.patch.object(
            sentinel, "_new_highs", return_value=CloseExplodingModel()
        ):
            with self.assertRaisesRegex(
                sentinel.ExactRangedNativeLoadSentinelError,
                "native_model_close_failed:SystemExit",
            ):
                sentinel._load_native_frame(
                    matrix=matrix,
                    row_lower=np.asarray([-1.0], dtype=np.float64),
                    row_upper=np.asarray([1.0], dtype=np.float64),
                    column_lower=np.asarray([-1.0], dtype=np.float64),
                    column_upper=np.asarray([1.0], dtype=np.float64),
                    objective=np.asarray([0.0], dtype=np.float64),
                    solve=False,
                    route="candidate_range",
                )

    @unittest.skipIf(sentinel._highspy is None, "highspy is optional")
    def test_authority_and_forbidden_route_firewall(self) -> None:
        source, lower, upper, objective = _one_dimensional_band()
        result = sentinel.run_native_ranged_equivalence_sentinel(
            source=source,
            column_lower=lower,
            column_upper=upper,
            objective=objective,
        )
        self.assertIsInstance(result.receipt, MappingProxyType)
        for key in (
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
            "proof_authority",
            "verdict_authority",
            "production_integration",
            "real_model_run",
            "large_model_run",
        ):
            self.assertIs(result.receipt[key], False)
        self.assertIs(result.receipt["source_frame_required_for_replay"], True)
        self.assertIs(result.receipt["source_frame_retained_outside_candidate"], True)
        with self.assertRaises(TypeError):
            result.receipt["proof_authority"] = True


if __name__ == "__main__":
    unittest.main()
