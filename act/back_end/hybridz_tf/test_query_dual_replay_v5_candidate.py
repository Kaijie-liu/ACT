"""Soundness gates for the isolated V5 dense-Conv scalar-guard candidate."""

from __future__ import annotations

import math
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v5_candidate as v5
from act.back_end.hybridz_tf.query_dual_replay_v5_candidate import (
    dense_conv_two_stage_scalar_guard,
    prepare_dense_conv_scalar_guard,
    replay_dense_conv_scalar_guard,
)


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
    weight = np.asarray(weight, dtype=np.float64)
    return frozen._FrozenLayer(
        id=2,
        kind="CONV2D",
        preds=(1,),
        width=int(np.prod(output_shape)),
        in_vars=(),
        out_vars=(),
        params=MappingProxyType(
            {
                "weight": weight,
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


def _exact_conv_reverse(layer, coefficient):
    budget = frozen._TermBudget(2_000_000)
    return frozen._fraction_conv_reverse(
        [
            Fraction.from_float(float(value))
            for value in np.asarray(coefficient).reshape(-1)
        ],
        layer,
        budget,
    )


class QueryDualReplayV5CandidateTests(unittest.TestCase):
    def _assert_sound(self, layer, box, coefficient):
        coefficient = np.asarray(coefficient, dtype=np.float64)
        deadline = frozen._Deadline(time.monotonic() + 30.0)
        candidate = dense_conv_two_stage_scalar_guard(
            coefficient, layer, box, deadline=deadline
        )
        reference_nominal, _ = frozen._conv_reverse_with_error(
            coefficient,
            layer,
            frozen._Deadline(time.monotonic() + 30.0),
            frozen._ReplayStats(),
        )
        self.assertTrue(
            np.array_equal(candidate.coefficient, reference_nominal)
        )
        self.assertEqual(
            [
                float(value).hex()
                for value in candidate.coefficient.reshape(-1)
            ],
            [
                float(value).hex()
                for value in reference_nominal.reshape(-1)
            ],
        )
        support = np.maximum(np.abs(box.lb), np.abs(box.ub))
        support_fraction = [
            Fraction.from_float(float(value)) for value in support
        ]
        for query_index, query in enumerate(coefficient):
            exact = _exact_conv_reverse(layer, query)
            required = sum(
                abs(
                    exact[index]
                    - Fraction.from_float(
                        float(candidate.coefficient[query_index, index])
                    )
                )
                * support_fraction[index]
                for index in range(len(exact))
            )
            supplied = Fraction.from_float(
                float(candidate.scalar_guard[query_index])
            )
            self.assertGreaterEqual(supplied, required)
            component_sum = (
                Fraction.from_float(
                    float(candidate.channel_dot_guard[query_index])
                )
                + Fraction.from_float(
                    float(candidate.accumulation_guard[query_index])
                )
            )
            self.assertGreaterEqual(supplied, component_sum)
        self.assertFalse(candidate.proof_authority)
        self.assertFalse(candidate.coefficient.flags.writeable)
        self.assertFalse(candidate.scalar_guard.flags.writeable)
        return candidate

    def test_padding_boundary_offsets_against_fraction_oracle(self):
        rng = np.random.default_rng(11)
        input_shape = (2, 3, 4)
        output_shape = (3, 3, 4)
        weight = (
            rng.integers(-4, 5, size=(3, 2, 3, 3)).astype(np.float64)
            / 8.0
        )
        layer = _conv_layer(
            weight=weight,
            input_shape=input_shape,
            output_shape=output_shape,
            padding=(1, 1),
        )
        coefficient = (
            rng.integers(-7, 8, size=(4, int(np.prod(output_shape))))
            .astype(np.float64)
            / 8.0
        )
        coefficient[coefficient == 0.0] = 0.125
        self._assert_sound(layer, _box(input_shape, seed=12), coefficient)

    def test_grouped_stride2_dilation_against_fraction_oracle(self):
        input_shape = (4, 5, 6)
        output_shape = (4, 3, 4)
        weight = (
            (np.arange(4 * 2 * 2 * 2, dtype=np.float64) % 7.0) - 3.0
        ).reshape(4, 2, 2, 2)
        layer = _conv_layer(
            weight=weight,
            input_shape=input_shape,
            output_shape=output_shape,
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 1),
            groups=2,
        )
        rng = np.random.default_rng(13)
        coefficient = rng.normal(
            size=(3, int(np.prod(output_shape)))
        ).astype(np.float64)
        self._assert_sound(layer, _box(input_shape, seed=14), coefficient)

    def test_channel_cancellation_against_fraction_oracle(self):
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
        candidate = self._assert_sound(
            layer,
            box,
            np.asarray(
                [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
                dtype=np.float64,
            ),
        )
        self.assertGreaterEqual(candidate.scalar_guard[0], 1.0)

    def test_subnormal_product_and_multiplicity_against_fraction_oracle(self):
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
        candidate = self._assert_sound(
            layer,
            box,
            np.asarray([[0.5], [-0.5]], dtype=np.float64),
        )
        self.assertGreater(candidate.scalar_guard[0], 0.0)

    def test_deterministic_fraction_geometry_fuzz(self):
        rng = np.random.default_rng(20260728)
        completed = 0
        attempts = 0
        while completed < 64 and attempts < 512:
            attempts += 1
            groups = int(rng.choice([1, 2]))
            in_per_group = int(rng.integers(1, 4))
            out_per_group = int(rng.integers(1, 4))
            input_shape = (
                groups * in_per_group,
                int(rng.integers(2, 6)),
                int(rng.integers(2, 6)),
            )
            kernel_h = int(rng.integers(1, 4))
            kernel_w = int(rng.integers(1, 4))
            stride = (
                int(rng.integers(1, 3)),
                int(rng.integers(1, 3)),
            )
            padding = (
                int(rng.integers(0, 3)),
                int(rng.integers(0, 3)),
            )
            dilation = (
                int(rng.integers(1, 3)),
                int(rng.integers(1, 3)),
            )
            output_h = (
                input_shape[1]
                + 2 * padding[0]
                - dilation[0] * (kernel_h - 1)
                - 1
            ) // stride[0] + 1
            output_w = (
                input_shape[2]
                + 2 * padding[1]
                - dilation[1] * (kernel_w - 1)
                - 1
            ) // stride[1] + 1
            if output_h <= 0 or output_w <= 0:
                continue
            output_shape = (
                groups * out_per_group,
                output_h,
                output_w,
            )
            weight = (
                rng.integers(
                    -8,
                    9,
                    size=(
                        output_shape[0],
                        in_per_group,
                        kernel_h,
                        kernel_w,
                    ),
                ).astype(np.float64)
                / 16.0
            )
            layer = _conv_layer(
                weight=weight,
                input_shape=input_shape,
                output_shape=output_shape,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            coefficient = (
                rng.integers(
                    -8,
                    9,
                    size=(2, int(np.prod(output_shape))),
                ).astype(np.float64)
                / 16.0
            )
            coefficient[coefficient == 0.0] = 1.0 / 16.0
            self._assert_sound(
                layer,
                _box(input_shape, seed=1000 + completed),
                coefficient,
            )
            completed += 1
        self.assertEqual(completed, 64)

    def test_sparse_branch_is_explicitly_unchanged(self):
        layer = _conv_layer(
            weight=np.ones((2, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 4, 4),
            output_shape=(2, 4, 4),
        )
        coefficient = np.zeros((8, 32), dtype=np.float64)
        coefficient[np.arange(8), np.arange(8)] = 1.0
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V5_SPARSE_UNCHANGED"
        ):
            dense_conv_two_stage_scalar_guard(
                coefficient,
                layer,
                _box((1, 4, 4)),
                deadline=frozen._Deadline(time.monotonic() + 30.0),
            )

    def test_sparse_threshold_boundary_and_subnormal_count(self):
        layer = _conv_layer(
            weight=np.ones((1, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 1, 16),
            output_shape=(1, 1, 16),
        )
        box = _box((1, 1, 16))
        plan = prepare_dense_conv_scalar_guard(
            layer,
            box,
            deadline=frozen._Deadline(time.monotonic() + 30.0),
        )
        at_boundary = np.zeros((1, 16), dtype=np.float64)
        at_boundary[0, :2] = 1.0  # 8*nz == dense: frozen sparse.
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V5_SPARSE_UNCHANGED"
        ):
            replay_dense_conv_scalar_guard(
                at_boundary,
                plan,
                deadline=frozen._Deadline(time.monotonic() + 30.0),
            )
        above_boundary = at_boundary.copy()
        above_boundary[0, 2] = np.nextafter(0.0, 1.0)
        result = replay_dense_conv_scalar_guard(
            above_boundary,
            plan,
            deadline=frozen._Deadline(time.monotonic() + 30.0),
        )
        self.assertEqual(result.coefficient.shape, (1, 16))
        signed_zero = np.copysign(
            np.zeros((1, 16), dtype=np.float64), -1.0
        )
        signed_zero[0, 0] = 1.0
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V5_SPARSE_UNCHANGED"
        ):
            replay_dense_conv_scalar_guard(
                signed_zero,
                plan,
                deadline=frozen._Deadline(time.monotonic() + 30.0),
            )

    def test_expired_deadline_fails_closed(self):
        layer = _conv_layer(
            weight=np.ones((1, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
        )
        with self.assertRaises(frozen.QueryDualReplayTimeout):
            prepare_dense_conv_scalar_guard(
                layer,
                _box((1, 1, 1)),
                deadline=frozen._Deadline(time.monotonic() - 1.0),
            )

    def test_precomputed_plan_reuse_matches_one_shot(self):
        rng = np.random.default_rng(18)
        layer = _conv_layer(
            weight=rng.normal(size=(3, 2, 3, 3)).astype(np.float64),
            input_shape=(2, 3, 3),
            output_shape=(3, 3, 3),
            padding=(1, 1),
        )
        box = _box((2, 3, 3), seed=19)
        coefficient = rng.normal(size=(5, 27)).astype(np.float64)
        plan = prepare_dense_conv_scalar_guard(
            layer,
            box,
            deadline=frozen._Deadline(time.monotonic() + 30.0),
        )
        reused = replay_dense_conv_scalar_guard(
            coefficient,
            plan,
            deadline=frozen._Deadline(time.monotonic() + 30.0),
        )
        one_shot = dense_conv_two_stage_scalar_guard(
            coefficient,
            layer,
            box,
            deadline=frozen._Deadline(time.monotonic() + 30.0),
        )
        self.assertTrue(
            np.array_equal(reused.coefficient, one_shot.coefficient)
        )
        self.assertTrue(
            np.array_equal(reused.scalar_guard, one_shot.scalar_guard)
        )
        self.assertEqual(
            plan.manifest["guard"],
            "two_stage_channel_dot_D_plus_offset_accumulation_A",
        )
        self.assertEqual(
            plan.manifest["tau_definition"], "up(k*eta/(1-k*u))"
        )
        self.assertEqual(len(plan.manifest["content_sha256"]), 64)
        self.assertFalse(plan.weight.flags.writeable)
        self.assertFalse(plan.support.flags.writeable)
        for offset in plan.offsets:
            self.assertFalse(offset.output_h_indices.flags.writeable)
            self.assertFalse(offset.output_w_indices.flags.writeable)
            self.assertFalse(offset.targets.flags.writeable)
            self.assertFalse(offset.support_flat.flags.writeable)
            self.assertFalse(offset.channel_support_flat.flags.writeable)

        forged_manifest = dict(plan.manifest)
        forged_manifest["offset_count"] += 1
        forged = replace(
            plan, manifest=MappingProxyType(forged_manifest)
        )
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "V5_INVALID_PLAN"
        ):
            replay_dense_conv_scalar_guard(
                coefficient,
                forged,
                deadline=frozen._Deadline(time.monotonic() + 30.0),
            )

    def test_gamma_and_tau_are_outward_exact_fraction_bounds(self):
        for operations in (1, 2, 4, 17, 258):
            gamma, tau = v5._outward_roundoff_parameters(operations)
            product = (
                Fraction(operations)
                * Fraction.from_float(float(frozen._U))
            )
            exact_gamma = product / (1 - product)
            exact_tau = (
                Fraction(operations)
                * Fraction.from_float(float(frozen._ETA))
                / (1 - product)
            )
            self.assertGreaterEqual(
                Fraction.from_float(gamma), exact_gamma
            )
            self.assertGreaterEqual(
                Fraction.from_float(tau), exact_tau
            )


if __name__ == "__main__":
    unittest.main()
