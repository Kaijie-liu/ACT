"""Controlled soundness gates for the isolated V5.1a Conv certificate."""

from __future__ import annotations

import math
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as v51


FRACTION_QUERY_ROWS = 5_000


def _deadline(seconds: float = 30.0) -> frozen._Deadline:
    return frozen._Deadline(time.monotonic() + seconds)


def _conv_layer(
    *,
    weight,
    input_shape,
    output_shape,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    weight_array = np.ascontiguousarray(weight, dtype=np.float64)
    return frozen._FrozenLayer(
        id=2,
        kind="CONV2D",
        preds=(1,),
        width=int(np.prod(output_shape)),
        in_vars=(),
        out_vars=(),
        params=MappingProxyType(
            {
                "weight": weight_array,
                "bias_channels": np.zeros(
                    int(output_shape[0]), dtype=np.float64
                ),
                "input_shape": tuple(input_shape),
                "output_shape": tuple(output_shape),
                "stride": tuple(stride),
                "padding": tuple(padding),
                "dilation": tuple(dilation),
                "groups": int(groups),
            }
        ),
    )


def _box(input_shape, *, seed=1):
    rng = np.random.default_rng(seed)
    width = int(np.prod(input_shape))
    lower = -rng.uniform(0.0, 2.0, size=width).astype(np.float64)
    upper = rng.uniform(0.0, 2.0, size=width).astype(np.float64)
    lower[::11] = 0.0
    upper[::13] = 0.0
    return frozen._Box(lb=lower, ub=upper)


def _exact_reverse(layer, query):
    return frozen._fraction_conv_reverse(
        [
            Fraction.from_float(float(value))
            for value in np.asarray(query).reshape(-1)
        ],
        layer,
        frozen._TermBudget(2_000_000),
    )


def _fraction_required(result, layer, box, coefficient, query_index):
    exact = _exact_reverse(layer, coefficient[query_index])
    support = np.maximum(np.abs(box.lb), np.abs(box.ub))
    return sum(
        abs(
            exact[index]
            - Fraction.from_float(
                float(result.coefficient[query_index, index])
            )
        )
        * Fraction.from_float(float(support[index]))
        for index in range(len(exact))
    )


class QueryDualReplayV51ConvTests(unittest.TestCase):
    def _assert_sound(
        self,
        layer,
        box,
        coefficient,
        *,
        compare_v3=True,
        fraction_rows=None,
    ):
        coefficients = np.ascontiguousarray(
            coefficient, dtype=np.float64
        )
        result = v51.dense_conv_v51(
            coefficients, layer, box, deadline=_deadline(120.0)
        )
        reference, radius = frozen._conv_reverse_with_error(
            coefficients,
            layer,
            _deadline(120.0),
            frozen._ReplayStats(),
        )
        self.assertTrue(np.array_equal(result.coefficient, reference))
        self.assertEqual(
            [float(value).hex() for value in result.coefficient.reshape(-1)],
            [float(value).hex() for value in reference.reshape(-1)],
        )
        selected_rows = (
            range(coefficients.shape[0])
            if fraction_rows is None
            else fraction_rows
        )
        for query_index in selected_rows:
            required = _fraction_required(
                result, layer, box, coefficients, query_index
            )
            supplied = Fraction.from_float(
                float(result.scalar_guard[query_index])
            )
            self.assertGreaterEqual(supplied, required)
            components = (
                Fraction.from_float(
                    float(result.channel_dot_guard[query_index])
                )
                + Fraction.from_float(
                    float(result.accumulation_guard[query_index])
                )
            )
            self.assertGreaterEqual(supplied, components)
        if compare_v3:
            max_abs = np.maximum(np.abs(box.lb), np.abs(box.ub))
            _, absorption_error = frozen._row_dots_with_error(
                radius, max_abs
            )
            absorption_nominal = np.asarray(
                radius @ max_abs, dtype=np.float64
            )
            v3_penalty = frozen._upper_nonnegative_sum(
                absorption_nominal, absorption_error
            )
            zero_penalty_rows = ~np.any(
                (radius != 0.0)
                & (max_abs.reshape(1, -1) != 0.0),
                axis=1,
            )
            v3_penalty[zero_penalty_rows] = 0.0
            self.assertTrue(np.all(result.scalar_guard <= v3_penalty))
            v3_lower = frozen._absorb_radius(
                np.zeros(coefficients.shape[0], dtype=np.float64),
                radius,
                box,
                frozen._ReplayStats(),
            )
            v51_lower = v51.absorb_scalar_guard_row_local(
                np.zeros(coefficients.shape[0], dtype=np.float64),
                result,
            )
            self.assertTrue(np.all(v51_lower >= v3_lower))
        self.assertTrue(v51.verify_dense_conv_v51_result(result))
        self.assertFalse(result.proof_authority)
        self.assertFalse(result.coefficient.flags.writeable)
        self.assertFalse(result.scalar_guard.flags.writeable)
        self.assertFalse(result.active_mask.flags.writeable)
        return result

    def test_wide_dot_fraction_enclosure_and_conditional_ceil(self):
        rng = np.random.default_rng(2026072800)
        values = np.asarray(
            [0.0, 2.0**-20, 0.1, 0.25, 1.0, 1.0e16],
            dtype=np.float64,
        )
        left = rng.choice(values, size=(257, 11)).astype(np.float64)
        right = rng.choice(values, size=11).astype(np.float64)
        left[0] = 0.0
        upper = v51.dot_up_l_rows(left, right, deadline=_deadline())
        self.assertEqual(upper[0], 0.0)
        for row, supplied in zip(left, upper):
            exact = sum(
                (
                    Fraction.from_float(float(a))
                    * Fraction.from_float(float(b))
                    for a, b in zip(row, right)
                ),
                Fraction(0),
            )
            self.assertGreaterEqual(Fraction.from_float(float(supplied)), exact)
        exact_left = np.eye(3, dtype=np.float64)
        exact_right = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        exact_upper = v51.dot_up_l_rows(
            exact_left, exact_right, deadline=_deadline()
        )
        self.assertTrue(np.all(exact_upper >= exact_right))
        self.assertTrue(
            np.array_equal(
                v51._ceil_f64(
                    exact_right.astype(np.longdouble),
                    where="conditional-ceil test",
                ),
                exact_right,
            )
        )

    def test_disjoint_support_uses_true_contraction_overlap(self):
        layer = _conv_layer(
            weight=np.asarray(
                [[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]],
                dtype=np.float64,
            ),
            input_shape=(2, 1, 2),
            output_shape=(2, 1, 2),
        )
        magnitudes = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        box = frozen._Box(lb=-magnitudes, ub=magnitudes)
        plan = v51.prepare_dense_conv_v51_plan(
            layer, box, deadline=_deadline()
        )
        self.assertEqual(len(plan.offsets), 1)
        offset = plan.offsets[0]
        self.assertEqual(
            offset.support_activity_flat.tolist(),
            [False, True, True, False],
        )
        self.assertEqual(
            offset.channel_support_flat[
                ~offset.support_activity_flat
            ].tolist(),
            [0.0, 0.0],
        )
        # Channel-major coefficient storage.  Its only nonzeros select the
        # two structurally inactive (co, position) pairs.
        coefficient = np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        result = v51.replay_dense_conv_v51(
            coefficient, plan, deadline=_deadline()
        )
        self.assertFalse(result.active_mask[0])
        self.assertEqual(result.scalar_guard[0], 0.0)
        self.assertEqual(result.channel_dot_guard[0], 0.0)
        self.assertEqual(result.accumulation_guard[0], 0.0)

    def test_cancellation_is_sound_bit_identical_and_not_looser_than_v3(self):
        layer = _conv_layer(
            weight=np.asarray(
                [[[[1.0e16]]], [[[1.0]]], [[[-1.0e16]]]],
                dtype=np.float64,
            ),
            input_shape=(1, 1, 1),
            output_shape=(3, 1, 1),
        )
        box = frozen._Box(
            lb=np.asarray([-1.0], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
        )
        coefficient = np.asarray(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
            dtype=np.float64,
        )
        result = self._assert_sound(layer, box, coefficient)
        self.assertTrue(np.all(result.coefficient == 0.0))
        self.assertTrue(np.all(result.accumulation_guard == 0.0))
        self.assertGreaterEqual(result.scalar_guard[0], 1.0)

    def test_mixed_zero_dense_rows_and_and_addition_activity(self):
        weight = np.asarray(
            [
                [[[1.0, 2.0]]],
                [[[2.0, -1.0]]],
                [[[-3.0, 0.5]]],
                [[[4.0, 1.5]]],
            ],
            dtype=np.float64,
        )
        layer = _conv_layer(
            weight=weight,
            input_shape=(1, 1, 3),
            output_shape=(4, 1, 2),
        )
        box = frozen._Box(
            lb=-np.ones(3, dtype=np.float64),
            ub=np.ones(3, dtype=np.float64),
        )
        coefficient = np.vstack(
            [
                np.zeros(8, dtype=np.float64),
                np.asarray(
                    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                    dtype=np.float64,
                ),
            ]
        )
        result = self._assert_sound(layer, box, coefficient)
        self.assertEqual(result.scalar_guard[0], 0.0)
        self.assertEqual(result.channel_dot_guard[0], 0.0)
        self.assertEqual(result.accumulation_guard[0], 0.0)
        self.assertFalse(result.active_mask[0])
        self.assertTrue(result.channel_dot_active_mask[1])
        self.assertTrue(result.accumulation_active_mask[1])
        scalar = np.asarray([1.0, 1.0], dtype=np.float64)
        absorbed = v51.absorb_scalar_guard_row_local(scalar, result)
        self.assertEqual(absorbed[0].hex(), scalar[0].hex())
        self.assertLess(absorbed[1], scalar[1])

        # The second offset is identically zero.  Its old+0 operations are
        # exact, so AND activity must leave A inactive.
        zero_second = weight.copy()
        zero_second[:, :, :, 1] = 0.0
        exact_add_layer = _conv_layer(
            weight=zero_second,
            input_shape=(1, 1, 3),
            output_shape=(4, 1, 2),
        )
        exact_add = v51.dense_conv_v51(
            coefficient,
            exact_add_layer,
            box,
            deadline=_deadline(),
        )
        self.assertFalse(np.any(exact_add.accumulation_active_mask))
        self.assertTrue(np.all(exact_add.accumulation_guard == 0.0))

    def test_padding_group_stride_dilation_fraction_oracles(self):
        rng = np.random.default_rng(2026072802)
        cases = (
            {
                "input_shape": (2, 3, 4),
                "output_shape": (3, 3, 4),
                "weight_shape": (3, 2, 3, 3),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "groups": 1,
            },
            {
                "input_shape": (4, 5, 6),
                "output_shape": (4, 3, 4),
                "weight_shape": (4, 2, 2, 2),
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": (2, 1),
                "groups": 2,
            },
            {
                "input_shape": (2, 4, 5),
                "output_shape": (2, 2, 3),
                "weight_shape": (2, 2, 3, 2),
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": (1, 2),
                "groups": 1,
            },
        )
        for case_index, case in enumerate(cases):
            weight = (
                rng.integers(
                    -8, 9, size=case["weight_shape"]
                ).astype(np.float64)
                / 16.0
            )
            layer = _conv_layer(
                weight=weight,
                input_shape=case["input_shape"],
                output_shape=case["output_shape"],
                stride=case["stride"],
                padding=case["padding"],
                dilation=case["dilation"],
                groups=case["groups"],
            )
            coefficient = (
                rng.integers(
                    -8,
                    9,
                    size=(4, int(np.prod(case["output_shape"]))),
                ).astype(np.float64)
                / 16.0
            )
            coefficient[coefficient == 0.0] = 1.0 / 16.0
            self._assert_sound(
                layer,
                _box(case["input_shape"], seed=100 + case_index),
                coefficient,
            )

    def test_5000_fraction_query_rows_and_v3_tightness_gate(self):
        weight = np.asarray(
            [
                [[[1.0e16]], [[0.1]]],
                [[[1.0]], [[-0.2]]],
                [[[-1.0e16]], [[0.3]]],
                [[[0.1]], [[-0.4]]],
            ],
            dtype=np.float64,
        )
        layer = _conv_layer(
            weight=weight,
            input_shape=(2, 1, 1),
            output_shape=(4, 1, 1),
        )
        box = frozen._Box(
            lb=np.asarray([-1.0, -0.75], dtype=np.float64),
            ub=np.asarray([1.0, 0.75], dtype=np.float64),
        )
        rng = np.random.default_rng(2026072801)
        values = np.asarray(
            [-1.0, -0.3, -0.125, 0.0, 0.1, 0.25, 1.0],
            dtype=np.float64,
        )
        coefficient = rng.choice(
            values, size=(FRACTION_QUERY_ROWS, 4)
        ).astype(np.float64)
        coefficient[::97] = 0.0
        coefficient[1::113] = np.asarray(
            [1.0, 1.0, 1.0, 0.1], dtype=np.float64
        )
        result = self._assert_sound(
            layer,
            box,
            coefficient,
            fraction_rows=range(FRACTION_QUERY_ROWS),
        )
        self.assertEqual(result.scalar_guard.size, FRACTION_QUERY_ROWS)
        zero_rows = ~np.any(coefficient != 0.0, axis=1)
        self.assertGreaterEqual(int(np.count_nonzero(zero_rows)), 1)
        self.assertTrue(np.all(result.scalar_guard[zero_rows] == 0.0))

    def test_sparse_threshold_neighbors_signed_zero_and_subnormal(self):
        layer = _conv_layer(
            weight=np.ones((1, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 1, 16),
            output_shape=(1, 1, 16),
        )
        box = frozen._Box(
            lb=-np.ones(16, dtype=np.float64),
            ub=np.ones(16, dtype=np.float64),
        )
        plan = v51.prepare_dense_conv_v51_plan(
            layer, box, deadline=_deadline()
        )
        below = np.zeros((1, 16), dtype=np.float64)
        below[0, 0] = 1.0
        at = below.copy()
        at[0, 1] = 1.0
        for sparse in (below, at):
            with self.assertRaisesRegex(
                frozen.QueryDualReplayError, "V51_SPARSE_UNCHANGED"
            ):
                v51.replay_dense_conv_v51(
                    sparse, plan, deadline=_deadline()
                )
        above = at.copy()
        above[0, 2] = np.nextafter(0.0, 1.0)
        dense = v51.replay_dense_conv_v51(
            above, plan, deadline=_deadline()
        )
        self.assertEqual(dense.receipt["nonzero_count"], 3)
        signed_zero = np.copysign(
            np.zeros((1, 16), dtype=np.float64), -1.0
        )
        signed_zero[0, 0] = 1.0
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V51_SPARSE_UNCHANGED"
        ):
            v51.replay_dense_conv_v51(
                signed_zero, plan, deadline=_deadline()
            )

    def test_subnormal_soundness_and_overflow_rejection(self):
        eta = np.nextafter(np.float64(0.0), np.float64(math.inf))
        layer = _conv_layer(
            weight=np.asarray([[[[eta]]]], dtype=np.float64),
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
        )
        box = frozen._Box(
            lb=np.asarray([-1.0], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
        )
        result = self._assert_sound(
            layer,
            box,
            np.asarray([[0.5], [-0.5]], dtype=np.float64),
        )
        self.assertTrue(np.all(result.scalar_guard > 0.0))

        overflow_layer = _conv_layer(
            weight=np.asarray([[[[1.0e308]]]], dtype=np.float64),
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(
                frozen.QueryDualReplayError, "NONFINITE"
            ):
                v51.dense_conv_v51(
                    np.asarray([[1.0e308]], dtype=np.float64),
                    overflow_layer,
                    box,
                    deadline=_deadline(),
                )

    def test_fully_rehashed_plan_and_receipt_substitutions_fail(self):
        layer = _conv_layer(
            weight=np.asarray(
                [[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]],
                dtype=np.float64,
            ),
            input_shape=(2, 1, 2),
            output_shape=(2, 1, 2),
        )
        box = frozen._Box(
            lb=-np.ones(4, dtype=np.float64),
            ub=np.ones(4, dtype=np.float64),
        )
        plan = v51.prepare_dense_conv_v51_plan(
            layer, box, deadline=_deadline()
        )
        coefficient = np.ones((2, 4), dtype=np.float64)
        result = v51.replay_dense_conv_v51(
            coefficient, plan, deadline=_deadline()
        )

        first = plan.offsets[0]
        changed_activity = np.array(
            first.support_activity_flat, copy=True
        )
        changed_activity[0] = ~changed_activity[0]
        changed_offset = replace(
            first,
            support_activity_flat=v51._immutable_bool(
                changed_activity
            ),
        )
        changed_offsets = (changed_offset,) + plan.offsets[1:]
        body = dict(
            v51._manifest_body(
                layer_id=plan.layer_id,
                input_shape=plan.input_shape,
                output_shape=plan.output_shape,
                stride=plan.stride,
                padding=plan.padding,
                dilation=plan.dilation,
                groups=plan.groups,
                weight=plan.weight,
                support=plan.support,
                offsets=changed_offsets,
            )
        )
        body["content_sha256"] = v51._canonical_digest(body)
        forged_plan = replace(
            plan,
            offsets=changed_offsets,
            manifest=MappingProxyType(body),
        )
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V51_INVALID_PLAN"
        ):
            v51.replay_dense_conv_v51(
                coefficient, forged_plan, deadline=_deadline()
            )

        forged_receipt = dict(result.receipt)
        forged_receipt["branch"] = "sparse"
        forged_receipt.pop("content_sha256")
        forged_receipt["content_sha256"] = v51._canonical_digest(
            forged_receipt
        )
        forged_result = replace(
            result, receipt=MappingProxyType(forged_receipt)
        )
        self.assertFalse(v51.verify_dense_conv_v51_result(forged_result))

    def test_expired_deadline_fails_closed_at_prepare_and_replay(self):
        layer = _conv_layer(
            weight=np.ones((2, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 1, 1),
            output_shape=(2, 1, 1),
        )
        box = frozen._Box(
            lb=np.asarray([-1.0], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
        )
        with self.assertRaises(frozen.QueryDualReplayTimeout):
            v51.prepare_dense_conv_v51_plan(
                layer,
                box,
                deadline=frozen._Deadline(time.monotonic() - 1.0),
            )
        plan = v51.prepare_dense_conv_v51_plan(
            layer, box, deadline=_deadline()
        )
        with self.assertRaises(frozen.QueryDualReplayTimeout):
            v51.replay_dense_conv_v51(
                np.ones((1, 2), dtype=np.float64),
                plan,
                deadline=frozen._Deadline(time.monotonic() - 1.0),
            )


if __name__ == "__main__":
    unittest.main()
