from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import ast
import gc
import inspect
import math
import time
import unittest
from unittest import mock
import weakref

import numpy as np

import scratch_phase_projection_direct_sparse_initial_target_compiler as target


class _HostileStop(BaseException):
    pass


def _a(value, dtype):
    return np.ascontiguousarray(value, dtype=dtype)


def _source(*, n=3):
    P, k, o = 4, 3, 2
    first_phase = np.arange(1, P * n + 1, dtype=np.float64).reshape(P, n) / 8.0
    first_output = np.vstack((
        np.arange(1, n + 1, dtype=np.float64) / 4.0,
        -np.arange(1, n + 1, dtype=np.float64) / 8.0,
    ))
    delta_phase = np.zeros((P, k), dtype=np.float64)
    delta_phase[1, 0] = 0.5
    delta_phase[2, 0] = -0.25
    delta_phase[2, 1] = 0.5
    delta_phase[3] = [0.25, -0.5, 0.125]
    delta_output = _a([[0.5, -0.25, 0.125], [-0.25, 0.5, 0.25]], np.float64)
    return target.DirectSparseInitialTargetInput(
        first_phase=_a(first_phase, np.float64),
        initial_delta_phase=_a(delta_phase, np.float64),
        first_output=_a(first_output, np.float64),
        initial_delta_output=delta_output,
        first_active=_a([False, True, False, True], np.bool_),
        target_active=_a([True, False, True, True], np.bool_),
        phase_centers=_a([-1.0, -1.0, -1.0, -1.0], np.float64),
        target_output_center=_a([0.5, -0.25], np.float64),
        change_ordinals=_a([0, 1, 2], np.int64),
        input_rows=_a(np.arange(10, 10 + n), np.int64),
        physical_rows=_a([[0, 0, 10], [0, 1, 11], [1, 0, 20], [1, 1, 21]], np.int64),
        full_row_ids=_a(np.arange(P), np.int64),
        factor_lower=_a(np.full(n, -1.0), np.float64),
        factor_upper=_a(np.full(n, 1.0), np.float64),
        assertion_matrix=_a([[1.0, -1.0], [-0.5, 0.25]], np.float64),
        thresholds=_a([0.125, -0.25], np.float64),
        rival=0,
    )


def _compile(source, checkpoint=None):
    return target.compile_direct_sparse_initial_target(
        source, deadline_monotonic=float(time.monotonic() + 30.0),
        checkpoint=checkpoint,
    )


def _fraction_oracle(source):
    P, n = source.first_phase.shape
    k = source.change_ordinals.size
    U = [[Fraction(0) for _ in range(n)] for _ in range(k)]
    for i, rho_raw in enumerate(source.change_ordinals):
        rho = int(rho_raw)
        for column in range(n):
            base = Fraction.from_float(float(source.first_phase[rho, column]))
            if source.target_active[rho]:
                value = base
                for j in range(i):
                    value += Fraction.from_float(
                        float(source.initial_delta_phase[rho, j])
                    ) * U[j][column]
            else:
                value = -base
            U[i][column] = value
    G = [[Fraction.from_float(float(source.first_phase[row, column]))
          for column in range(n)] for row in range(P)]
    for row in range(P):
        for column in range(n):
            for i in range(k):
                G[row][column] += Fraction.from_float(
                    float(source.initial_delta_phase[row, i])
                ) * U[i][column]
            if source.target_active[row]:
                G[row][column] = -G[row][column]
    Go = [[Fraction.from_float(float(source.first_output[row, column]))
           for column in range(n)] for row in range(source.first_output.shape[0])]
    for row in range(source.first_output.shape[0]):
        for column in range(n):
            for i in range(k):
                Go[row][column] += Fraction.from_float(
                    float(source.initial_delta_output[row, i])
                ) * U[i][column]
    q = []
    for column in range(n):
        value = Fraction(0)
        for row in range(source.first_output.shape[0]):
            value += Fraction.from_float(
                float(source.assertion_matrix[source.rival, row])
            ) * Go[row][column]
        q.append(value)
    return G, q


def _screen_source(coefficients, centers):
    coefficients = _a(coefficients, np.float64)
    P, n = coefficients.shape
    return target.DirectSparseInitialTargetInput(
        first_phase=coefficients,
        initial_delta_phase=_a(np.empty((P, 0)), np.float64),
        first_output=_a(np.zeros((1, n)), np.float64),
        initial_delta_output=_a(np.empty((1, 0)), np.float64),
        first_active=_a(np.zeros(P), np.bool_),
        target_active=_a(np.zeros(P), np.bool_),
        phase_centers=_a(centers, np.float64),
        target_output_center=_a([0.0], np.float64),
        change_ordinals=_a([], np.int64),
        input_rows=_a(np.arange(n), np.int64),
        physical_rows=_a([[0, row, row] for row in range(P)], np.int64),
        full_row_ids=_a(np.arange(P), np.int64),
        factor_lower=_a(np.full(n, -1.0), np.float64),
        factor_upper=_a(np.full(n, 1.0), np.float64),
        assertion_matrix=_a([[1.0]], np.float64),
        thresholds=_a([0.0], np.float64),
        rival=0,
    )


class DirectSparseInitialTargetCompilerTest(unittest.TestCase):
    def test_dyadic_k3_both_flip_directions_and_fraction_oracle(self):
        source = _source()
        result = _compile(source)
        expected_rows, expected_q = _fraction_oracle(source)
        dense = result.full_rows.toarray()
        for row in range(dense.shape[0]):
            for column in range(dense.shape[1]):
                self.assertEqual(Fraction.from_float(float(dense[row, column])),
                                 expected_rows[row][column])
        for column, value in enumerate(result.objective_coefficient):
            self.assertEqual(Fraction.from_float(float(value)), expected_q[column])
        self.assertEqual(
            Fraction.from_float(result.objective_center),
            Fraction.from_float(1.0) * Fraction.from_float(0.5)
            + Fraction.from_float(-1.0) * Fraction.from_float(-0.25)
            - Fraction.from_float(0.125),
        )
        result.assert_intact()
        self.assertEqual(result.receipt.fixed_column_tile_width, 64)
        self.assertFalse(result.receipt.chunking_used)
        self.assertFalse(result.receipt.partition_transaction_used)
        self.assertEqual(result.receipt.atomic_publish_count, 1)
        self.assertIsNone(result.receipt.loaded_nnz_after_tiny_projection)
        self.assertFalse(result.receipt.downstream_loading_run)
        self.assertEqual(result.receipt.resource_estimate_kind,
                         "contract_formula_not_observed_peak")

    def test_single_csr_constructor_and_objective_association_lock(self):
        source_text = inspect.getsource(target)
        tree = ast.parse(source_text)
        csr_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sp"
            and node.func.attr == "csr_matrix"
        ]
        self.assertEqual(len(csr_calls), 1)
        self.assertNotIn(".tocsr", source_text)
        self.assertIn(
            "source.assertion_matrix[[source.rival]] @ output_block",
            source_text,
        )
        self.assertNotIn(
            "source.assertion_matrix[[source.rival]] @ source.initial_delta_output",
            source_text,
        )
        forbidden_calls = {
            "vstack", "hstack", "coo_matrix", "lil_matrix", "dok_matrix",
        }
        self.assertFalse([
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_calls
        ])

    def test_resource_formula_is_contract_estimate(self):
        self.assertEqual(
            target._compile_bytes(
                nnz=3_079_720, phase_rows=1_668,
                input_factors=9_408, changes=57, outputs=200,
                segments=147,
            ),
            75_961_604,
        )

    def test_n65_tile_boundary_63_64(self):
        source = _source(n=65)
        result = _compile(source)
        expected_rows, expected_q = _fraction_oracle(source)
        for column in (62, 63, 64):
            for row in range(4):
                self.assertEqual(
                    Fraction.from_float(float(result.full_rows[row, column])),
                    expected_rows[row][column],
                )
            self.assertEqual(
                Fraction.from_float(float(result.objective_coefficient[column])),
                expected_q[column],
            )
        self.assertEqual(result.receipt.segments, 2)

    def test_branched_add_duplicate_cancellation(self):
        source = _source(n=1)
        delta = source.initial_delta_phase.copy()
        delta[3] = [-2.0, 1.0, 0.0]
        source = replace(source, initial_delta_phase=delta)
        result = _compile(source)
        self.assertEqual(result.full_rows.indptr[4] - result.full_rows.indptr[3], 0)
        self.assertNotIn(0.0, result.full_rows.data)

    def test_screen_lt_eq_gt(self):
        source = _screen_source([[0.5], [1.0], [1.5]], [-0.75, -1.0, -1.25])
        result = _compile(source)
        self.assertTrue(np.array_equal(result.keep, [False, False, True]))
        self.assertEqual(result.screened_rows.shape, (1, 1))
        self.assertTrue(np.array_equal(result.screened_row_ids, [2]))

    def test_exact_screen_all_false_fails_closed(self):
        source = _screen_source([[0.0, -0.0]], [-1.0])
        with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                    "retained no downstream rows"):
            _compile(source)

    def test_tiny_and_subnormal_owner_boundary(self):
        tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
        source = _screen_source([[tiny, -tiny, 0.0, -0.0]], [0.0])
        result = _compile(source)
        self.assertEqual(result.full_rows.nnz, 2)
        self.assertEqual(result.full_rows.data[0].view(np.uint64), tiny.view(np.uint64))
        self.assertEqual(result.full_rows.data[1].view(np.uint64),
                         (-tiny).view(np.uint64))
        self.assertTrue(result.keep[0])

    def test_resource_preflight_precedes_owned_allocation(self):
        source = _screen_source(
            np.zeros((1, target._MAX_INPUT_FACTORS + 1), dtype=np.float64),
            [0.0],
        )
        with mock.patch.object(
            target, "_seal", side_effect=AssertionError("snapshot reached")
        ):
            with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                        "dimension"):
                _compile(source)

    def test_future_or_diagonal_dependency(self):
        source = _source()
        bad = source.initial_delta_phase.copy()
        bad[0, 0] = 1.0
        with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                    "future dependency"):
            _compile(replace(source, initial_delta_phase=bad))

    def test_unsorted_or_duplicate_change(self):
        source = _source()
        for changes in ([1, 0, 2], [0, 0, 2]):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                            "change order"):
                    _compile(replace(source,
                                     change_ordinals=_a(changes, np.int64)))

    def test_mapping_shape_dtype_and_contiguity_drift(self):
        source = _source()
        bad_values = (
            replace(source, physical_rows=source.physical_rows[:, :2].copy()),
            replace(source, full_row_ids=_a(np.arange(4), np.int32)),
            replace(source, input_rows=_a([10, 12, 11], np.int64)),
            replace(source, first_phase=np.asfortranarray(source.first_phase)),
        )
        for bad in bad_values:
            with self.subTest(value=bad):
                with self.assertRaises(target.DirectSparseInitialTargetUnknown):
                    _compile(bad)

    def test_signed_zero_canonicalization(self):
        source = _screen_source([[0.0, -0.0]], [1.0])
        result = _compile(source)
        self.assertEqual(result.full_rows.nnz, 0)
        self.assertTrue(result.keep[0])
        result.assert_intact()

    def test_nan_inf_and_float_overflow(self):
        source = _source()
        for value in (math.nan, math.inf, -math.inf):
            first = source.first_phase.copy()
            first[0, 0] = value
            with self.assertRaises(target.DirectSparseInitialTargetUnknown):
                _compile(replace(source, first_phase=first))
        overflow = _source()
        fp = overflow.first_phase.copy()
        dp = overflow.initial_delta_phase.copy()
        fp[0] = np.finfo(np.float64).max
        dp[3, 0] = np.finfo(np.float64).max
        with self.assertRaises(target.DirectSparseInitialTargetUnknown):
            _compile(replace(overflow, first_phase=fp,
                             initial_delta_phase=dp))

    def test_integer_product_nnz_and_byte_cap_overflow(self):
        with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                    "int32"):
            target._checked_product(200_000, 200_000,
                                    name="hostile product")
        with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                    "bytes"):
            target._compile_bytes(
                nnz=200_000_000, phase_rows=200_000,
                input_factors=200_000, changes=200_000,
                outputs=200_000, segments=3125,
            )

    def test_deadline_every_checkpoint(self):
        source = _source()
        stages = []
        _compile(source, checkpoint=stages.append)
        self.assertEqual(len(stages), len(set(stages)))
        for selected in stages:
            state = {"stage": None}

            def mark(stage):
                state["stage"] = stage

            def clock():
                return 101.0 if state["stage"] == selected else 0.0

            with self.subTest(stage=selected), mock.patch.object(
                target.time, "monotonic", side_effect=clock
            ):
                with self.assertRaisesRegex(
                    target.DirectSparseInitialTargetUnknown, "deadline expired"
                ):
                    target.compile_direct_sparse_initial_target(
                        source, deadline_monotonic=100.0, checkpoint=mark
                    )

    def test_non_exception_baseexception_every_checkpoint(self):
        source = _source()
        stages = []
        _compile(source, checkpoint=stages.append)
        for selected in stages:
            primary = _HostileStop(selected)

            def inject(stage, selected=selected, primary=primary):
                if stage == selected:
                    raise primary

            with self.subTest(stage=selected):
                try:
                    _compile(source, checkpoint=inject)
                except BaseException as caught:
                    self.assertIs(caught, primary)
                    del caught
                else:
                    self.fail("hostile BaseException did not escape")

    def test_segment_temporaries_released_on_hostile_exit(self):
        source = _source(n=65)
        references = []
        snapshot_references = []
        original_new_empty = target._new_empty
        original_seal = target._seal

        def observe(shape, dtype, *, deadline, stage, callback):
            value = original_new_empty(
                shape, dtype, deadline=deadline, stage=stage,
                callback=callback,
            )
            if "segment_" in stage:
                references.append(weakref.ref(value))
            return value

        def observe_snapshot(value, *, name, deadline, callback):
            snapshot = original_seal(
                value, name=name, deadline=deadline, callback=callback
            )
            snapshot_references.append(weakref.ref(snapshot))
            return snapshot

        primary = _HostileStop("release")

        def inject(stage):
            if stage == "segment_0_sealed":
                raise primary

        with mock.patch.object(target, "_new_empty", side_effect=observe), \
             mock.patch.object(target, "_seal", side_effect=observe_snapshot):
            try:
                _compile(source, checkpoint=inject)
            except BaseException as caught:
                self.assertIs(caught, primary)
                del caught
            else:
                self.fail("hostile BaseException did not escape")
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(len(snapshot_references), 16)
        self.assertTrue(all(
            reference() is None for reference in snapshot_references
        ))
        traceback_codes = []
        current = primary.__traceback__
        while current is not None:
            traceback_codes.append(current.tb_frame.f_code)
            current = current.tb_next
        self.assertIn(target.compile_direct_sparse_initial_target.__code__,
                      traceback_codes)
        self.assertNotIn(target._compile_entry.__code__, traceback_codes)
        self.assertNotIn(target._compile_owned.__code__, traceback_codes)

    def test_caller_alias_after_owned_snapshot(self):
        source = _source()
        original = source.first_phase.copy()

        def mutate(stage):
            if stage == "after_owned_snapshot":
                source.first_phase.fill(99.0)

        result = _compile(source, checkpoint=mutate)
        expected = _compile(replace(source, first_phase=original))
        self.assertEqual(result.receipt.content_sha256,
                         expected.receipt.content_sha256)

    def test_internal_snapshot_digest_drift(self):
        result = _compile(_source())
        object.__setattr__(result, "rhs", _a(result.rhs.copy(), np.float64))
        with self.assertRaises(target.DirectSparseInitialTargetUnknown):
            result.assert_intact()

        source = _source()
        original_digest = target._input_digest

        def corrupt(stage):
            if stage == "after_snapshot_validation":
                target._input_digest = lambda values, rival: "0" * 64

        try:
            with self.assertRaisesRegex(target.DirectSparseInitialTargetUnknown,
                                        "snapshot digest drifted"):
                _compile(source, checkpoint=corrupt)
        finally:
            target._input_digest = original_digest


if __name__ == "__main__":
    unittest.main()
