#!/usr/bin/env python3
"""Exact toys for sound PCOH Fraction-to-binary64 row formation."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect
import itertools
import math
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    bind_external_certified_empty_pattern,
    bind_external_pattern_upper_bound,
    build_objective_binding,
    build_phase_conditioned_objective_hull,
    outward_float64,
)
from act.back_end.hybridz_tf import (
    operator_phase_conditioned_objective_hull_row_materializer as row_module,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull_row_materializer import (
    PCOHRowMaterializationCaps,
    PhaseConditionedObjectiveHullRowMaterializationError,
    materialize_phase_conditioned_objective_hull_row_frame,
    verify_phase_conditioned_objective_hull_row_frame,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _complete_bounds(binding, stable_ids, upper_by_pattern):
    result = []
    for pattern in itertools.product((-1, 1), repeat=len(stable_ids)):
        exact = upper_by_pattern[pattern]
        result.append(
            bind_external_pattern_upper_bound(
                assignments=tuple(zip(stable_ids, pattern)),
                upper_exact=exact,
                upper_stored=outward_float64(exact),
                parent_semantic_digest=binding.parent_semantic_digest,
                objective_binding_sha256=(
                    binding.objective_binding_sha256
                ),
                certificate_schema="act.test.row_formation_bound.v1",
                certificate_sha256=_sha(
                    f"row-formation-bound:{binding.objective_id}:{pattern}"
                ),
                upstream_proof_authority=True,
                independently_certified=True,
            )
        )
    return tuple(result)


def _descriptor(
    label,
    *,
    center=Fraction(1, 3),
    continuous_terms=(
        (101, Fraction(1, 3)),
        (103, Fraction(-2, 7)),
    ),
    binary_terms=((201, Fraction(1, 10)),),
    stable_ids=(201,),
    upper_by_pattern=None,
    empty_evidence=(),
):
    parent = _sha(f"row-frame-parent:{label}")
    binding = build_objective_binding(
        objective_id=f"row-frame-objective:{label}",
        parent_semantic_digest=parent,
        center=center,
        continuous_terms=continuous_terms,
        binary_terms=binary_terms,
    )
    if upper_by_pattern is None:
        upper_by_pattern = {
            pattern: Fraction(index + 2)
            for index, pattern in enumerate(
                itertools.product((-1, 1), repeat=len(stable_ids))
            )
        }
    bounds = _complete_bounds(binding, stable_ids, upper_by_pattern)
    baseline = max(bound.upper_stored for bound in bounds)
    descriptor = build_phase_conditioned_objective_hull(
        stable_bit_ids=stable_ids,
        pattern_bounds=bounds,
        objective_binding=binding,
        parent_semantic_digest=parent,
        baseline_upper_stored=baseline,
        empty_pattern_evidence=empty_evidence,
    )
    return descriptor


def _materialize(descriptor, *, col_ids=None, bcol_ids=None, caps=None):
    if col_ids is None:
        col_ids = np.asarray((999, 101, 103), dtype=np.int64)
    if bcol_ids is None:
        bcol_ids = np.asarray((777, 201), dtype=np.int64)
    kwargs = {}
    if caps is not None:
        kwargs["caps"] = caps
    return materialize_phase_conditioned_objective_hull_row_frame(
        descriptor,
        live_parent_semantic_digest=descriptor.parent_semantic_digest,
        parent_col_ids=col_ids,
        parent_bcol_ids=bcol_ids,
        deadline=time.monotonic() + 10.0,
        **kwargs,
    )


def _exact_row_lhs(row, descriptor, col_ids, bcol_ids, ce_values, b_values):
    continuous_position = {
        int(stable_id): index
        for index, stable_id in enumerate(col_ids.tolist())
    }
    binary_position = {
        int(stable_id): index
        for index, stable_id in enumerate(bcol_ids.tolist())
    }
    parent_count = col_ids.size
    value = Fraction(0)
    value += sum(
        coefficient * ce_values[continuous_position[stable_id]]
        for stable_id, coefficient in row.parent_continuous_terms
    )
    value += sum(
        coefficient * b_values[binary_position[stable_id]]
        for stable_id, coefficient in row.parent_binary_terms
    )
    value += sum(
        coefficient * ce_values[parent_count + local_index]
        for local_index, coefficient in row.eta_terms
    )
    return value


def _stored_row_lhs(matrix_ce, matrix_b, ce_values, b_values, row=0):
    value = Fraction(0)
    ce = matrix_ce.getrow(row)
    value += sum(
        Fraction.from_float(float(coefficient)) * ce_values[int(column)]
        for column, coefficient in zip(ce.indices, ce.data)
    )
    binary = matrix_b.getrow(row)
    value += sum(
        Fraction.from_float(float(coefficient)) * b_values[int(column)]
        for column, coefficient in zip(binary.indices, binary.data)
    )
    return value


class PCOHRowMaterializationTests(unittest.TestCase):
    def test_array_backed_sorted_and_unsorted_stable_id_lookup(self):
        descriptor = _descriptor("array-backed-lookup")
        sorted_col_ids = np.asarray((101, 103, 999), dtype=np.int64)
        sorted_bcol_ids = np.asarray((201, 777), dtype=np.int64)
        unsorted_col_ids = np.asarray((999, 101, 103), dtype=np.int64)
        unsorted_bcol_ids = np.asarray((777, 201), dtype=np.int64)

        source = inspect.getsource(row_module._strict_id_array)
        self.assertNotIn(".tolist(", source)
        self.assertNotIn("set(", source)

        with mock.patch.object(
            row_module.np,
            "searchsorted",
            wraps=np.searchsorted,
        ) as searchsorted:
            sorted_frame = _materialize(
                descriptor,
                col_ids=sorted_col_ids,
                bcol_ids=sorted_bcol_ids,
            )
        self.assertGreater(searchsorted.call_count, 0)

        queried = []
        original_position = row_module._StableIdLookup.position

        def recording_position(lookup, identifier, *, deadline):
            queried.append((lookup.name, identifier))
            return original_position(
                lookup, identifier, deadline=deadline
            )

        with mock.patch.object(
            row_module._StableIdLookup,
            "position",
            recording_position,
        ), mock.patch.object(
            row_module.np,
            "searchsorted",
            side_effect=AssertionError(
                "unsorted fallback must use the bounded indirect index"
            ),
        ):
            unsorted_frame = _materialize(
                descriptor,
                col_ids=unsorted_col_ids,
                bcol_ids=unsorted_bcol_ids,
            )

        expected_queries = {
            ("parent_col_ids", identifier)
            for row in descriptor.equality_rows + descriptor.upper_rows
            for identifier, _ in row.parent_continuous_terms
        } | {
            ("parent_bcol_ids", identifier)
            for row in descriptor.equality_rows + descriptor.upper_rows
            for identifier, _ in row.parent_binary_terms
        }
        self.assertEqual(set(queried), expected_queries)
        self.assertNotIn(("parent_col_ids", 999), queried)
        self.assertNotIn(("parent_bcol_ids", 777), queried)

        # Column positions move with the caller's stable-id order, while every
        # exact formation result remains unchanged.
        self.assertEqual(
            sorted_frame.upper_row_guards,
            unsorted_frame.upper_row_guards,
        )
        np.testing.assert_array_equal(
            sorted_frame.equality_rhs, unsorted_frame.equality_rhs
        )
        np.testing.assert_array_equal(
            sorted_frame.upper_rhs, unsorted_frame.upper_rhs
        )
        sorted_upper_ce = sorted_frame.upper_continuous_eta.toarray()[0]
        unsorted_upper_ce = unsorted_frame.upper_continuous_eta.toarray()[0]
        self.assertEqual(sorted_upper_ce[0], unsorted_upper_ce[1])
        self.assertEqual(sorted_upper_ce[1], unsorted_upper_ce[2])
        np.testing.assert_array_equal(
            sorted_upper_ce[3:], unsorted_upper_ce[3:]
        )
        self.assertEqual(
            sorted_frame.upper_binary.toarray()[0, 0],
            unsorted_frame.upper_binary.toarray()[0, 1],
        )

        for row_index, exact_row in enumerate(descriptor.equality_rows):
            stored_ce = sorted_frame.equality_continuous_eta.getrow(
                row_index
            ).toarray()[0]
            stored_binary = sorted_frame.equality_binary.getrow(
                row_index
            ).toarray()[0]
            for identifier, coefficient in exact_row.parent_continuous_terms:
                position = int(np.searchsorted(sorted_col_ids, identifier))
                self.assertEqual(
                    Fraction.from_float(stored_ce[position]), coefficient
                )
            for identifier, coefficient in exact_row.parent_binary_terms:
                position = int(np.searchsorted(sorted_bcol_ids, identifier))
                self.assertEqual(
                    Fraction.from_float(stored_binary[position]), coefficient
                )
            for identifier, coefficient in exact_row.eta_terms:
                self.assertEqual(
                    Fraction.from_float(
                        stored_ce[sorted_col_ids.size + identifier]
                    ),
                    coefficient,
                )

    def test_id_lookup_rejects_negative_duplicate_and_missing_fail_closed(self):
        descriptor = _descriptor("lookup-fail-closed")
        for bad_ids in (
            np.asarray((-1, 101, 103), dtype=np.int64),
            np.asarray((101, 101, 103), dtype=np.int64),
            np.asarray((103, 101, 103), dtype=np.int64),
        ):
            with self.subTest(ids=bad_ids):
                with self.assertRaisesRegex(
                    PhaseConditionedObjectiveHullRowMaterializationError,
                    "negative_or_duplicate",
                ):
                    _materialize(descriptor, col_ids=bad_ids)
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "parent_continuous_stable_id_missing:103",
        ):
            _materialize(
                descriptor,
                col_ids=np.asarray((101, 999), dtype=np.int64),
            )

    def test_unsorted_lookup_allocation_slope_is_sixteen_bytes_per_id(self):
        retained = []
        peaks = []
        for size in (250_000, 500_000, 1_000_000):
            values = np.arange(size, dtype=np.int64)[::-1].copy()
            tracemalloc.start()
            snapshot, lookup = row_module._strict_id_array(
                values,
                name="memory_slope_ids",
                maximum_columns=1_000_000,
                deadline=time.monotonic() + 30.0,
            )
            digest = row_module._id_vector_sha256(
                "memory_slope_ids",
                snapshot,
                deadline=time.monotonic() + 30.0,
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            legacy_digest = hashlib.sha256()
            legacy_digest.update(b"memory_slope_ids:int64_le:v1:")
            legacy_digest.update(
                np.asarray(size, dtype="<i8").tobytes()
            )
            legacy_digest.update(
                values.astype("<i8", copy=False).tobytes(order="C")
            )
            self.assertEqual(digest, legacy_digest.hexdigest())
            self.assertFalse(snapshot.flags.writeable)
            self.assertFalse(lookup.indirect_order.flags.writeable)
            self.assertEqual(
                lookup.position(0, deadline=time.monotonic() + 1.0),
                size - 1,
            )
            self.assertLessEqual(current, 16 * size + 32_768)
            retained.append((snapshot, lookup))
            peaks.append(peak)
        # Bounded validation chunks contribute a constant peak; the measured
        # incremental allocation must remain one snapshot plus one permutation.
        self.assertLessEqual(peaks[1] - peaks[0], 17 * 250_000)
        self.assertLessEqual(peaks[2] - peaks[1], 17 * 500_000)

    def test_deadline_is_checked_after_nonpreemptible_numpy_bulk_steps(self):
        values = np.asarray((3, 1, 2), dtype=np.int64)
        clock = [0.0]
        real_argsort = np.argsort

        def expiring_argsort(*args, **kwargs):
            result = real_argsort(*args, **kwargs)
            clock[0] = 11.0
            return result

        with mock.patch.object(
            row_module.time, "monotonic", side_effect=lambda: clock[0]
        ), mock.patch.object(
            row_module.np, "argsort", side_effect=expiring_argsort
        ):
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullRowMaterializationError,
                "deadline_exhausted:parent_col_ids_argsort",
            ):
                row_module._strict_id_array(
                    values,
                    name="parent_col_ids",
                    maximum_columns=3,
                    deadline=10.0,
                )

        snapshot, lookup = row_module._strict_id_array(
            np.asarray((1, 2, 3), dtype=np.int64),
            name="parent_col_ids",
            maximum_columns=3,
            deadline=time.monotonic() + 1.0,
        )
        del snapshot
        clock[0] = 0.0
        real_searchsorted = np.searchsorted

        def expiring_searchsorted(*args, **kwargs):
            result = real_searchsorted(*args, **kwargs)
            clock[0] = 11.0
            return result

        with mock.patch.object(
            row_module.time, "monotonic", side_effect=lambda: clock[0]
        ), mock.patch.object(
            row_module.np,
            "searchsorted",
            side_effect=expiring_searchsorted,
        ):
            with self.assertRaisesRegex(
                PhaseConditionedObjectiveHullRowMaterializationError,
                "deadline_exhausted:parent_col_ids_searchsorted",
            ):
                lookup.position(2, deadline=10.0)

    def test_four_canonical_blocks_exact_equalities_and_strict_replay(self):
        descriptor = _descriptor("canonical")
        col_ids = np.asarray((999, 101, 103), dtype=np.int64)
        bcol_ids = np.asarray((777, 201), dtype=np.int64)
        frame = _materialize(
            descriptor, col_ids=col_ids, bcol_ids=bcol_ids
        )

        self.assertFalse(frame.proof_authority)
        self.assertFalse(frame.verdict_authority)
        self.assertEqual(frame.equality_continuous_eta.shape, (2, 5))
        self.assertEqual(frame.equality_binary.shape, (2, 2))
        self.assertEqual(frame.upper_continuous_eta.shape, (1, 5))
        self.assertEqual(frame.upper_binary.shape, (1, 2))
        for matrix in (
            frame.equality_continuous_eta,
            frame.equality_binary,
            frame.upper_continuous_eta,
            frame.upper_binary,
        ):
            self.assertTrue(matrix.has_canonical_format)
            self.assertEqual(matrix.dtype, np.dtype(np.float64))
            self.assertFalse(matrix.data.flags.writeable)
            self.assertFalse(matrix.indices.flags.writeable)
            self.assertFalse(matrix.indptr.flags.writeable)
        self.assertFalse(frame.equality_rhs.flags.writeable)
        self.assertFalse(frame.upper_rhs.flags.writeable)

        # k=1 normalization: eta_0 + eta_1 = 0.
        self.assertEqual(
            frame.equality_continuous_eta.getrow(0).toarray().tolist(),
            [[0.0, 0.0, 0.0, 1.0, 1.0]],
        )
        # 2*xi - (-1)*eta_0 - (+1)*eta_1 = 0.
        self.assertEqual(
            frame.equality_continuous_eta.getrow(1).toarray().tolist(),
            [[0.0, 0.0, 0.0, 1.0, -1.0]],
        )
        self.assertEqual(
            frame.equality_binary.getrow(1).toarray().tolist(),
            [[0.0, 2.0]],
        )
        self.assertEqual(frame.equality_rhs.tolist(), [0.0, 0.0])
        self.assertTrue(
            verify_phase_conditioned_objective_hull_row_frame(
                frame,
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=col_ids,
                parent_bcol_ids=bcol_ids,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(frame.receipt["uses_sparse_hstack"])
        self.assertFalse(frame.receipt["uses_sparse_vstack"])
        self.assertEqual(
            frame.receipt["equality_binary64_policy"],
            "finite_bit_exact_or_fail_closed",
        )

    def test_exact_fraction_box_implication_and_stored_jacobian(self):
        descriptor = _descriptor("box-soundness")
        col_ids = np.asarray((999, 101, 103), dtype=np.int64)
        bcol_ids = np.asarray((777, 201), dtype=np.int64)
        frame = _materialize(
            descriptor, col_ids=col_ids, bcol_ids=bcol_ids
        )
        exact_row = descriptor.upper_rows[0]
        stored_rhs = Fraction.from_float(float(frame.upper_rhs[0]))
        checked = 0

        for signs in itertools.product((-1, 1), repeat=7):
            ce_values = tuple(Fraction(value) for value in signs[:5])
            b_values = tuple(Fraction(value) for value in signs[5:])
            exact_lhs = _exact_row_lhs(
                exact_row,
                descriptor,
                col_ids,
                bcol_ids,
                ce_values,
                b_values,
            )
            stored_lhs = _stored_row_lhs(
                frame.upper_continuous_eta,
                frame.upper_binary,
                ce_values,
                b_values,
            )
            if exact_lhs <= exact_row.rhs:
                checked += 1
                self.assertLessEqual(stored_lhs, stored_rhs)
        self.assertGreater(checked, 0)

        rng = np.random.default_rng(7)
        for _ in range(200):
            ce_values = tuple(
                Fraction(int(value), 8)
                for value in rng.integers(-8, 9, size=5)
            )
            b_values = tuple(
                Fraction(int(value), 8)
                for value in rng.integers(-8, 9, size=2)
            )
            exact_lhs = _exact_row_lhs(
                exact_row,
                descriptor,
                col_ids,
                bcol_ids,
                ce_values,
                b_values,
            )
            if exact_lhs <= exact_row.rhs:
                self.assertLessEqual(
                    _stored_row_lhs(
                        frame.upper_continuous_eta,
                        frame.upper_binary,
                        ce_values,
                        b_values,
                    ),
                    stored_rhs,
                )

        guard = frame.upper_row_guards[0]
        self.assertEqual(
            guard.total_coefficient_guard,
            sum(
                (item.absolute_error for item in guard.coefficient_errors),
                Fraction(0),
            ),
        )
        self.assertEqual(
            guard.guarded_rhs_exact,
            guard.raw_rhs_exact + guard.total_coefficient_guard,
        )
        self.assertGreaterEqual(
            Fraction.from_float(guard.stored_rhs), guard.guarded_rhs_exact
        )

        # Stored derivatives equal the individually recorded binary64 values.
        ce_dense = frame.upper_continuous_eta.getrow(0).toarray()[0]
        b_dense = frame.upper_binary.getrow(0).toarray()[0]
        for item in guard.coefficient_errors:
            if item.group == "parent_continuous":
                position = {999: 0, 101: 1, 103: 2}[item.identifier]
                self.assertEqual(ce_dense[position], item.stored)
            elif item.group == "parent_binary":
                position = {777: 0, 201: 1}[item.identifier]
                self.assertEqual(b_dense[position], item.stored)
            else:
                self.assertEqual(ce_dense[3 + item.identifier], item.stored)

    def test_subnormal_large_cancellation_and_outward_rhs(self):
        tiny = Fraction(1, 2**1075)
        large = Fraction(10**16) + Fraction(1, 3)
        descriptor = _descriptor(
            "subnormal-large-cancellation",
            center=Fraction(10**16) - Fraction(1, 3),
            continuous_terms=((101, tiny), (103, large)),
            binary_terms=(),
            upper_by_pattern={(-1,): Fraction(10**16), (1,): Fraction(10**16)},
        )
        frame = _materialize(descriptor)
        guard = frame.upper_row_guards[0]

        self.assertEqual(guard.raw_rhs_exact, Fraction(1, 3))
        error_by_id = {
            item.identifier: item.absolute_error
            for item in guard.coefficient_errors
            if item.group == "parent_continuous"
        }
        self.assertEqual(error_by_id[101], tiny)
        self.assertEqual(error_by_id[103], Fraction(1, 3))
        ce_dense = frame.upper_continuous_eta.getrow(0).toarray()[0]
        self.assertEqual(ce_dense[1], 0.0)
        self.assertEqual(ce_dense[2], 1.0e16)
        self.assertGreaterEqual(
            Fraction.from_float(frame.upper_rhs[0]),
            Fraction(1, 3) + guard.total_coefficient_guard,
        )

        smallest = row_module._fraction_to_exact_binary64(
            Fraction(1, 2**1074), name="smallest_subnormal"
        )
        self.assertEqual(smallest, float.fromhex("0x0.0000000000001p-1022"))
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "not_bit_exact_binary64",
        ):
            row_module._fraction_to_exact_binary64(
                tiny, name="below_smallest_subnormal"
            )

    def test_empty_eta_fix_is_bit_exact_and_empty_upper_term_stays_absent(self):
        parent = _sha("row-frame-parent:empty-fix")
        stable_ids = (301, 307)
        binding = build_objective_binding(
            objective_id="row-frame-empty-fix",
            parent_semantic_digest=parent,
            center=Fraction(0),
        )
        upper_by_pattern = {
            (-1, -1): Fraction(0),
            (-1, 1): Fraction(0),
            (1, -1): Fraction(0),
            (1, 1): Fraction(10**16),
        }
        bounds = _complete_bounds(binding, stable_ids, upper_by_pattern)
        evidence = bind_external_certified_empty_pattern(
            assignments=((301, 1), (307, 1)),
            witness_literals=((301, 1), (307, 1)),
            parent_semantic_digest=parent,
            property_digest=_sha("empty-fix-property"),
            selection_digest=_sha("empty-fix-selection"),
            operator_row_tag_digest=_sha("empty-fix-row-tags"),
            ordered_source_frame_sha256=_sha("empty-fix-source-frame"),
            source_bundle_sha256=_sha("empty-fix-bundle"),
            coverage_sha256=_sha("empty-fix-coverage"),
            source_record_sha256=_sha("empty-fix-record"),
            local_row_map_sha256=_sha("empty-fix-row-map"),
            certificate_schema="act.test.empty-fix.v1",
            certificate_sha256=_sha("empty-fix-certificate"),
            eta_fixed_value=-1,
            upstream_exact_replay_authority=True,
            independently_exact_certified=True,
        )
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=stable_ids,
            pattern_bounds=bounds,
            objective_binding=binding,
            parent_semantic_digest=parent,
            baseline_upper_stored=1.0e16,
            empty_pattern_evidence=(evidence,),
        )
        frame = materialize_phase_conditioned_objective_hull_row_frame(
            descriptor,
            live_parent_semantic_digest=parent,
            parent_col_ids=np.empty(0, dtype=np.int64),
            parent_bcol_ids=np.asarray(stable_ids, dtype=np.int64),
            deadline=time.monotonic() + 10.0,
        )
        empty_index = descriptor.patterns.index((1, 1))
        fix_row = len(descriptor.equality_rows) - 1
        fix_dense = frame.equality_continuous_eta.getrow(fix_row).toarray()[0]
        self.assertEqual(fix_dense[empty_index], 1.0)
        self.assertEqual(frame.equality_rhs[fix_row], -1.0)
        self.assertEqual(frame.upper_continuous_eta.nnz, 0)
        self.assertEqual(frame.upper_rhs[0], 0.0)

    def test_overflow_fails_closed_for_coefficient_and_guarded_rhs(self):
        coefficient_overflow = _descriptor(
            "coefficient-overflow",
            center=Fraction(0),
            continuous_terms=((101, Fraction(10**400)),),
            binary_terms=(),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "binary64_(overflow|nonfinite)",
        ):
            _materialize(
                coefficient_overflow,
                col_ids=np.asarray((101,), dtype=np.int64),
                bcol_ids=np.asarray((201,), dtype=np.int64),
            )

        rhs_overflow = _descriptor(
            "rhs-overflow",
            center=-Fraction(10**400),
            continuous_terms=(),
            binary_terms=(),
        )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "has_no_finite_outward_binary64",
        ):
            _materialize(
                rhs_overflow,
                col_ids=np.empty(0, dtype=np.int64),
                bcol_ids=np.asarray((201,), dtype=np.int64),
            )

    def test_ids_descriptor_frame_caps_deadline_and_no_stack_fail_closed(self):
        descriptor = _descriptor("guards")
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "stable_id_missing",
        ):
            _materialize(
                descriptor,
                col_ids=np.asarray((101,), dtype=np.int64),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "negative_or_duplicate",
        ):
            _materialize(
                descriptor,
                col_ids=np.asarray((101, 101, 103), dtype=np.int64),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "not_canonical_int64_vector",
        ):
            _materialize(
                descriptor,
                col_ids=np.asarray((999, 101, 103), dtype=np.int32),
            )

        bad_receipt = dict(descriptor.receipt)
        bad_receipt["proof_authority"] = True
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "descriptor_structural_verification_failed",
        ):
            _materialize(descriptor=replace(descriptor, receipt=bad_receipt))

        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "total_exact_nonzero_cap_exceeded",
        ):
            _materialize(
                descriptor,
                caps=replace(
                    PCOHRowMaterializationCaps(),
                    max_total_exact_nonzeros=1,
                ),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "upper_total_guard_0_exact_bit_cap_exceeded",
        ):
            _materialize(
                descriptor,
                caps=replace(
                    PCOHRowMaterializationCaps(), max_exact_bits=8
                ),
            )
        with self.assertRaisesRegex(
            PhaseConditionedObjectiveHullRowMaterializationError,
            "deadline_exhausted",
        ):
            materialize_phase_conditioned_objective_hull_row_frame(
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=np.asarray((999, 101, 103), dtype=np.int64),
                parent_bcol_ids=np.asarray((777, 201), dtype=np.int64),
                deadline=time.monotonic() - 1.0,
            )

        with mock.patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), mock.patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            frame = _materialize(descriptor)

        tampered_matrix = frame.upper_continuous_eta.copy()
        tampered_matrix.data[0] = np.nextafter(
            tampered_matrix.data[0], math.inf
        )
        tampered = replace(
            frame, upper_continuous_eta=tampered_matrix
        )
        self.assertFalse(
            verify_phase_conditioned_objective_hull_row_frame(
                tampered,
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=np.asarray((999, 101, 103), dtype=np.int64),
                parent_bcol_ids=np.asarray((777, 201), dtype=np.int64),
                deadline=time.monotonic() + 10.0,
            )
        )
        writable_matrix = frame.upper_continuous_eta.copy()
        self.assertFalse(
            verify_phase_conditioned_objective_hull_row_frame(
                replace(frame, upper_continuous_eta=writable_matrix),
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=np.asarray((999, 101, 103), dtype=np.int64),
                parent_bcol_ids=np.asarray((777, 201), dtype=np.int64),
                deadline=time.monotonic() + 10.0,
            )
        )
        writable_rhs = frame.upper_rhs.copy()
        self.assertFalse(
            verify_phase_conditioned_objective_hull_row_frame(
                replace(frame, upper_rhs=writable_rhs),
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=np.asarray((999, 101, 103), dtype=np.int64),
                parent_bcol_ids=np.asarray((777, 201), dtype=np.int64),
                deadline=time.monotonic() + 10.0,
            )
        )
        tampered_receipt = dict(frame.receipt)
        tampered_receipt["upper_rhs_policy"] = "inward"
        self.assertFalse(
            verify_phase_conditioned_objective_hull_row_frame(
                replace(frame, receipt=tampered_receipt),
                descriptor,
                live_parent_semantic_digest=descriptor.parent_semantic_digest,
                parent_col_ids=np.asarray((999, 101, 103), dtype=np.int64),
                parent_bcol_ids=np.asarray((777, 201), dtype=np.int64),
                deadline=time.monotonic() + 10.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
