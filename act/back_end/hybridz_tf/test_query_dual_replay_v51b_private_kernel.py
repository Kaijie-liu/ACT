"""Soundness and isolation gates for the private V5.1b numeric kernels."""

from __future__ import annotations

import copy
import dis
import gc
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import types
import unittest
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import numpy as np
from numpy._core import _exceptions as np_exceptions
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense_v51
from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)


def _end(seconds: float = 30.0) -> float:
    return float(time.monotonic() + seconds)


def _run_isolated(source: str) -> None:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(private.__file__), "../../..")
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode:
        raise AssertionError(
            "isolated probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _implementation_spec(factory, expected_name: str) -> tuple:
    for cell in factory.__closure__ or ():
        value = cell.cell_contents
        if (
            type(value) is tuple
            and len(value) == 7
            and type(value[0]) is types.CodeType
            and value[1] == expected_name
        ):
            return value
    raise AssertionError("immutable implementation spec was not captured")


def _replay_deadline(seconds: float = 30.0) -> frozen._Deadline:
    return frozen._Deadline(end=_end(seconds))


def _bytes_backed(value: np.ndarray) -> bool:
    if type(value) is not np.ndarray or value.flags.writeable:
        return False
    current = value
    seen = set()
    while type(current) is np.ndarray:
        if id(current) in seen or current.flags.writeable:
            return False
        seen.add(id(current))
        current = current.base
    return type(current) is bytes


_DENSE_RESULT_TAG = b"act.v51b.private.dense-result.v1"
_CONV_RESULT_TAG = b"act.v51b.private.conv-result.v1"
_F64_RESULT_TAG = np.dtype(np.float64).str.encode("ascii")
_BOOL_RESULT_TAG = np.dtype(np.bool_).str.encode("ascii")
_DENSE_RESULT_FIELDS = (
    "nominal",
    "support_mass_upper",
    "wide_guard",
    "streamed_v3_guard",
    "final_guard",
    "active_mask",
    "fallback_mask",
)
_CONV_RESULT_FIELDS = (
    "coefficient",
    "scalar_guard",
    "channel_dot_guard",
    "accumulation_guard",
    "active_mask",
    "channel_dot_active_mask",
    "accumulation_active_mask",
)


def _decoder_zero_sum(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bit-exact model of the closure-local zero-preserving directed sum."""

    left_active = left != 0.0
    right_active = right != 0.0
    both = left_active & right_active
    result = np.zeros(left.shape, dtype=np.float64)
    result[left_active & ~right_active] = left[
        left_active & ~right_active
    ]
    result[right_active & ~left_active] = right[
        right_active & ~left_active
    ]
    if np.any(both):
        wide = np.nextafter(
            np.asarray(left[both], dtype=np.longdouble)
            + np.asarray(right[both], dtype=np.longdouble),
            np.longdouble(math.inf),
            dtype=np.longdouble,
        )
        nearest = np.ascontiguousarray(wide, dtype=np.float64)
        below = np.asarray(nearest, dtype=np.longdouble) < wide
        nearest[below] = np.nextafter(
            nearest[below], np.float64(math.inf)
        )
        result[both] = nearest
    return result


def _decode_private_result(
    result,
    *,
    kind_tag: bytes,
    fields: tuple[str, ...],
):
    """Strict test-only model of the future session-local ABI consumer.

    This helper is not an authentication boundary and is never called by the
    implementation.  A production session must decode only the value returned
    inside its own operation lock; it must never accept a caller-submitted
    tuple.
    """

    if (
        type(result) is not tuple
        or len(result) != 2 + len(fields)
        or type(result[0]) is not bytes
        or result[0] != kind_tag
        or result[1] is not False
    ):
        raise AssertionError("invalid private result envelope")
    decoded = {}
    for index, name in enumerate(fields, start=2):
        frame = result[index]
        expected_rank = 2 if index == 2 else 1
        expected_tag = (
            _BOOL_RESULT_TAG if "mask" in name else _F64_RESULT_TAG
        )
        if (
            type(frame) is not tuple
            or len(frame) != 3
            or type(frame[0]) is not bytes
            or type(frame[1]) is not tuple
            or len(frame[1]) != expected_rank
            or any(type(extent) is not int for extent in frame[1])
            or any(extent <= 0 for extent in frame[1])
            or type(frame[2]) is not bytes
            or frame[2] != expected_tag
        ):
            raise AssertionError(f"invalid private result frame: {name}")
        count = math.prod(frame[1])
        dtype = np.bool_ if expected_tag == _BOOL_RESULT_TAG else np.float64
        if (
            len(frame[0]) != count * np.dtype(dtype).itemsize
            or (
                dtype is np.bool_
                and any(byte not in (0, 1) for byte in frame[0])
            )
        ):
            raise AssertionError(f"invalid private result length: {name}")
        value = np.frombuffer(frame[0], dtype=dtype).reshape(frame[1])
        if not _bytes_backed(value):
            raise AssertionError(f"mutable private result frame: {name}")
        decoded[name] = value
    view = SimpleNamespace(proof_authority=False, **decoded)
    if kind_tag == _DENSE_RESULT_TAG:
        nominal = view.nominal
        expected_shape = (nominal.shape[0],)
        expected_final = view.wide_guard.copy()
        expected_final[view.fallback_mask] = np.minimum(
            view.wide_guard[view.fallback_mask],
            view.streamed_v3_guard[view.fallback_mask],
        )
        expected_final[~view.active_mask] = 0.0
        if (
            nominal.ndim != 2
            or any(
                getattr(view, name).shape != expected_shape
                for name in fields[1:]
            )
            or not np.all(np.isfinite(nominal))
            or any(
                not np.all(np.isfinite(getattr(view, name)))
                or np.any(getattr(view, name) < 0.0)
                for name in fields[1:5]
            )
            or np.any(view.final_guard > view.wide_guard)
            or np.any(view.final_guard > view.streamed_v3_guard)
            or not np.array_equal(
                view.final_guard.view(np.uint64),
                expected_final.view(np.uint64),
            )
            or not np.array_equal(
                view.active_mask,
                view.support_mass_upper != 0.0,
            )
            or np.any(
                view.final_guard[~view.active_mask].view(np.uint64)
                != 0
            )
            or any(
                np.any(
                    getattr(view, name)[
                        ~view.active_mask
                    ].view(np.uint64)
                    != 0
                )
                for name in fields[1:5]
            )
            or np.any(view.fallback_mask & ~view.active_mask)
        ):
            raise AssertionError("invalid Dense private result semantics")
    else:
        coefficient = view.coefficient
        expected_shape = (coefficient.shape[0],)
        expected_scalar = _decoder_zero_sum(
            view.channel_dot_guard,
            view.accumulation_guard,
        )
        expected_scalar[~view.active_mask] = 0.0
        if (
            coefficient.ndim != 2
            or any(
                getattr(view, name).shape != expected_shape
                for name in fields[1:]
            )
            or not np.all(np.isfinite(coefficient))
            or any(
                not np.all(np.isfinite(getattr(view, name)))
                or np.any(getattr(view, name) < 0.0)
                for name in fields[1:4]
            )
            or not np.array_equal(
                view.active_mask,
                (
                    view.channel_dot_active_mask
                    | view.accumulation_active_mask
                ),
            )
            or np.any(view.scalar_guard < view.channel_dot_guard)
            or np.any(view.scalar_guard < view.accumulation_guard)
            or not np.array_equal(
                view.scalar_guard.view(np.uint64),
                expected_scalar.view(np.uint64),
            )
            or not np.array_equal(
                view.channel_dot_active_mask,
                view.channel_dot_guard != 0.0,
            )
            or not np.array_equal(
                view.accumulation_active_mask,
                view.accumulation_guard != 0.0,
            )
            or np.any(
                view.scalar_guard[~view.active_mask].view(np.uint64)
                != 0
            )
            or np.any(
                view.channel_dot_guard[
                    ~view.channel_dot_active_mask
                ].view(np.uint64)
                != 0
            )
            or np.any(
                view.accumulation_guard[
                    ~view.accumulation_active_mask
                ].view(np.uint64)
                != 0
            )
        ):
            raise AssertionError("invalid Conv private result semantics")
    return view


def _decode_dense_result(result):
    return _decode_private_result(
        result,
        kind_tag=_DENSE_RESULT_TAG,
        fields=_DENSE_RESULT_FIELDS,
    )


def _decode_conv_result(result):
    return _decode_private_result(
        result,
        kind_tag=_CONV_RESULT_TAG,
        fields=_CONV_RESULT_FIELDS,
    )


def _conv_layer(
    *,
    weight: np.ndarray,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> frozen._FrozenLayer:
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
                    output_shape[0], dtype=np.float64
                ),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
        ),
    )


def _admit_conv(
    port,
    *,
    weight: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
):
    return port.admit_conv(
        layer_id=2,
        weight=weight,
        predecessor_lb=lb,
        predecessor_ub=ub,
        input_shape=input_shape,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _public_conv(
    *,
    weight: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    coefficient: np.ndarray,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
):
    layer = _conv_layer(
        weight=weight,
        input_shape=input_shape,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    box = frozen._Box(lb=lb, ub=ub)
    plan = conv_v51.prepare_dense_conv_v51_plan(
        layer, box, deadline=_replay_deadline()
    )
    result = conv_v51.replay_dense_conv_v51(
        coefficient, plan, deadline=_replay_deadline()
    )
    return layer, box, result


class _ProbeArray(np.ndarray):
    reads = 0

    @property
    def dtype(self):
        type(self).reads += 1
        return super().dtype

    @property
    def ndim(self):
        type(self).reads += 1
        return super().ndim

    @property
    def flags(self):
        type(self).reads += 1
        return super().flags

    def tobytes(self, *args, **kwargs):
        type(self).reads += 1
        return super().tobytes(*args, **kwargs)


class PrivateKernelDenseTests(unittest.TestCase):
    def test_dense_public_v51a_bit_differential_fixed_set(self):
        values = np.asarray(
            [-2.0, -0.5, -0.0, 0.0, 0.1, 0.25, 1.0, 3.0],
            dtype=np.float64,
        )
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(2026072800 + seed)
                weight = np.ascontiguousarray(
                    rng.choice(values, size=(5, 7))
                )
                max_abs = np.ascontiguousarray(
                    np.abs(rng.choice(values, size=7))
                )
                coefficient = np.ascontiguousarray(
                    rng.choice(values, size=(6, 5))
                )
                coefficient[0] = 0.0
                port = private.create_private_numeric_kernel(
                    deadline=_end()
                )
                locator = port.admit_dense(
                    weight=weight,
                    predecessor_max_abs=max_abs,
                    tile_width=3,
                )
                actual = _decode_dense_result(
                    port.execute_dense(locator, coefficient)
                )
                support = dense_v51.prepare_dense_support_v51(
                    weight, max_abs, deadline=_end()
                )
                expected = (
                    dense_v51.dense_support_compressed_guard_v51(
                        coefficient,
                        weight,
                        max_abs,
                        support,
                        tile_width=3,
                        deadline=_end(),
                    )
                )
                for name in (
                    "nominal",
                    "support_mass_upper",
                    "wide_guard",
                    "streamed_v3_guard",
                    "final_guard",
                    "active_mask",
                    "fallback_mask",
                ):
                    self.assertTrue(
                        np.array_equal(
                            getattr(actual, name),
                            getattr(expected, name),
                        ),
                        name,
                    )
                self.assertEqual(
                    actual.nominal.view(np.uint64).tolist(),
                    expected.nominal.view(np.uint64).tolist(),
                )

    def test_dense_fraction_soundness_and_v3_tightness(self):
        rng = np.random.default_rng(2026072811)
        values = np.asarray(
            [-4.0, -0.75, -0.1, 0.0, 0.2, 0.5, 1.0, 8.0],
            dtype=np.float64,
        )
        weight = np.ascontiguousarray(rng.choice(values, size=(4, 6)))
        max_abs = np.ascontiguousarray(
            np.abs(rng.choice(values, size=6))
        )
        coefficient = np.ascontiguousarray(
            rng.choice(values, size=(24, 4))
        )
        coefficient[0] = 0.0
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = port.admit_dense(
            weight=weight,
            predecessor_max_abs=max_abs,
            tile_width=2,
        )
        result = _decode_dense_result(
            port.execute_dense(locator, coefficient)
        )
        _, component_radius = frozen._matrix_product_with_error(
            coefficient, weight
        )
        for row in range(coefficient.shape[0]):
            required = Fraction(0)
            for column in range(weight.shape[1]):
                exact = sum(
                    (
                        Fraction.from_float(
                            float(coefficient[row, index])
                        )
                        * Fraction.from_float(
                            float(weight[index, column])
                        )
                        for index in range(weight.shape[0])
                    ),
                    Fraction(0),
                )
                nominal = Fraction.from_float(
                    float(result.nominal[row, column])
                )
                required += abs(exact - nominal) * Fraction.from_float(
                    float(max_abs[column])
                )
            self.assertGreaterEqual(
                Fraction.from_float(float(result.final_guard[row])),
                required,
            )
        _, absorption_error = frozen._row_dots_with_error(
            component_radius, max_abs
        )
        absorption_nominal = np.asarray(
            component_radius @ max_abs, dtype=np.float64
        )
        v3_penalty = frozen._upper_nonnegative_sum(
            absorption_nominal, absorption_error
        )
        zero_rows = ~np.any(
            (component_radius != 0.0)
            & (max_abs.reshape(1, -1) != 0.0),
            axis=1,
        )
        v3_penalty[zero_rows] = 0.0
        self.assertTrue(np.all(result.final_guard <= v3_penalty))
        self.assertTrue(np.all(result.final_guard <= result.wide_guard))

    def test_dense_underflow_fallback_and_exact_zero(self):
        eta = np.nextafter(
            np.float64(0.0), np.float64(math.inf)
        )
        weight = np.ascontiguousarray(
            np.asarray([[eta, 0.0], [0.0, 1.0]], dtype=np.float64)
        )
        max_abs = np.ascontiguousarray(
            np.asarray([1.0, 0.0], dtype=np.float64)
        )
        coefficient = np.ascontiguousarray(
            np.asarray(
                [[eta, 0.0], [0.0, 1.0], [-0.0, 0.0]],
                dtype=np.float64,
            )
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = port.admit_dense(
            weight=weight,
            predecessor_max_abs=max_abs,
            tile_width=1,
        )
        result = _decode_dense_result(
            port.execute_dense(locator, coefficient)
        )
        support = dense_v51.prepare_dense_support_v51(
            weight, max_abs, deadline=_end()
        )
        expected = dense_v51.dense_support_compressed_guard_v51(
            coefficient,
            weight,
            max_abs,
            support,
            tile_width=1,
            deadline=_end(),
        )
        self.assertTrue(result.fallback_mask[0])
        self.assertTrue(
            np.array_equal(result.fallback_mask, expected.fallback_mask)
        )
        self.assertEqual(float(result.final_guard[2]).hex(), "0x0.0p+0")
        self.assertFalse(result.active_mask[2])

    def test_dense_source_mutation_after_admission_is_irrelevant(self):
        weight = np.ascontiguousarray(
            np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
        )
        max_abs = np.ascontiguousarray(
            np.asarray([2.0, 5.0], dtype=np.float64)
        )
        coefficient = np.ascontiguousarray(
            np.asarray([[0.25, -1.0]], dtype=np.float64)
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = port.admit_dense(
            weight=weight, predecessor_max_abs=max_abs
        )
        before = _decode_dense_result(
            port.execute_dense(locator, coefficient)
        )
        weight[:] = 1.0e200
        max_abs[:] = 0.0
        after = _decode_dense_result(
            port.execute_dense(locator, coefficient)
        )
        self.assertTrue(np.array_equal(before.nominal, after.nominal))
        self.assertTrue(
            np.array_equal(before.final_guard, after.final_guard)
        )


class PrivateKernelConvTests(unittest.TestCase):
    def _assert_differential(
        self,
        *,
        seed,
        weight_shape,
        input_shape,
        output_shape,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        rng = np.random.default_rng(seed)
        weight = np.ascontiguousarray(
            rng.normal(size=weight_shape).astype(np.float64)
        )
        width = int(np.prod(input_shape))
        lb = np.ascontiguousarray(
            -rng.uniform(0.0, 2.0, size=width).astype(np.float64)
        )
        ub = np.ascontiguousarray(
            rng.uniform(0.0, 2.0, size=width).astype(np.float64)
        )
        rows = 4
        coefficient = np.ascontiguousarray(
            rng.normal(
                size=(rows, int(np.prod(output_shape)))
            ).astype(np.float64)
        )
        coefficient[0, ::7] = 0.0
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        actual = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        _, _, expected = _public_conv(
            weight=weight,
            lb=lb,
            ub=ub,
            coefficient=coefficient,
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        for name in (
            "coefficient",
            "scalar_guard",
            "channel_dot_guard",
            "accumulation_guard",
            "active_mask",
            "channel_dot_active_mask",
            "accumulation_active_mask",
        ):
            self.assertTrue(
                np.array_equal(
                    getattr(actual, name), getattr(expected, name)
                ),
                name,
            )
        self.assertEqual(
            actual.coefficient.view(np.uint64).tolist(),
            expected.coefficient.view(np.uint64).tolist(),
        )

    def test_conv_public_v51a_bit_differential_geometries(self):
        cases = (
            {
                "seed": 2026072820,
                "weight_shape": (2, 1, 2, 2),
                "input_shape": (1, 3, 3),
                "output_shape": (2, 2, 2),
            },
            {
                "seed": 2026072821,
                "weight_shape": (4, 1, 1, 1),
                "input_shape": (2, 2, 3),
                "output_shape": (4, 2, 3),
                "groups": 2,
            },
            {
                "seed": 2026072822,
                "weight_shape": (2, 1, 3, 3),
                "input_shape": (1, 5, 5),
                "output_shape": (2, 3, 3),
                "stride": (2, 2),
                "padding": (1, 1),
            },
            {
                "seed": 2026072823,
                "weight_shape": (2, 1, 2, 2),
                "input_shape": (1, 5, 5),
                "output_shape": (2, 3, 3),
                "dilation": (2, 2),
            },
        )
        for case in cases:
            with self.subTest(case=case):
                self._assert_differential(**case)

    def test_conv_fraction_soundness_fixed_rows(self):
        rng = np.random.default_rng(2026072830)
        weight = np.ascontiguousarray(
            np.asarray(
                [
                    [[[0.1, -0.25], [0.5, 1.0]]],
                    [[[-0.2, 0.75], [0.125, -0.5]]],
                ],
                dtype=np.float64,
            )
        )
        input_shape = (1, 3, 3)
        output_shape = (2, 2, 2)
        lb = np.ascontiguousarray(
            -rng.uniform(0.0, 2.0, size=9).astype(np.float64)
        )
        ub = np.ascontiguousarray(
            rng.uniform(0.0, 2.0, size=9).astype(np.float64)
        )
        coefficient = np.ascontiguousarray(
            rng.choice(
                np.asarray(
                    [-1.0, -0.2, 0.0, 0.1, 0.5, 2.0],
                    dtype=np.float64,
                ),
                size=(32, 8),
            )
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        result = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        layer = _conv_layer(
            weight=weight,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        support = np.maximum(np.abs(lb), np.abs(ub))
        for row in range(coefficient.shape[0]):
            exact = frozen._fraction_conv_reverse(
                [
                    Fraction.from_float(float(value))
                    for value in coefficient[row]
                ],
                layer,
                frozen._TermBudget(50_000),
            )
            required = sum(
                (
                    abs(
                        exact[index]
                        - Fraction.from_float(
                            float(result.coefficient[row, index])
                        )
                    )
                    * Fraction.from_float(float(support[index]))
                    for index in range(len(exact))
                ),
                Fraction(0),
            )
            self.assertGreaterEqual(
                Fraction.from_float(float(result.scalar_guard[row])),
                required,
            )
        reference, radius = frozen._conv_reverse_with_error(
            coefficient,
            layer,
            _replay_deadline(),
            frozen._ReplayStats(),
        )
        self.assertTrue(np.array_equal(result.coefficient, reference))
        _, absorption_error = frozen._row_dots_with_error(
            radius, support
        )
        absorption_nominal = np.asarray(
            radius @ support, dtype=np.float64
        )
        v3_penalty = frozen._upper_nonnegative_sum(
            absorption_nominal, absorption_error
        )
        zero_rows = ~np.any(
            (radius != 0.0) & (support.reshape(1, -1) != 0.0),
            axis=1,
        )
        v3_penalty[zero_rows] = 0.0
        self.assertTrue(np.all(result.scalar_guard <= v3_penalty))

    def test_conv_one_eighth_threshold_and_signed_zero(self):
        weight = np.ones((1, 1, 1, 1), dtype=np.float64)
        lb = -np.ones(16, dtype=np.float64)
        ub = np.ones(16, dtype=np.float64)
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=(1, 1, 16),
            output_shape=(1, 1, 16),
        )
        below = np.zeros((1, 16), dtype=np.float64)
        below[0, 0] = 1.0
        at = below.copy()
        at[0, 1] = 1.0
        signed = np.copysign(
            np.zeros((1, 16), dtype=np.float64), -1.0
        )
        signed[0, 0] = 1.0
        for value in (below, at, signed):
            with self.assertRaisesRegex(
                private.PrivateNumericKernelError,
                "SPARSE_UNCHANGED",
            ):
                port.execute_conv(locator, value)
        above = at.copy()
        above[0, 2] = np.nextafter(0.0, 1.0)
        result = _decode_conv_result(
            port.execute_conv(locator, above)
        )
        self.assertEqual(result.coefficient.shape, (1, 16))

    def test_conv_structural_zero_row_is_exact_zero(self):
        weight = np.ascontiguousarray(
            np.asarray([[[[1.0]]], [[[-0.5]]]], dtype=np.float64)
        )
        lb = -np.ones(4, dtype=np.float64)
        ub = np.ones(4, dtype=np.float64)
        coefficient = np.ascontiguousarray(
            np.vstack(
                (
                    np.copysign(np.zeros(8, dtype=np.float64), -1.0),
                    np.linspace(-1.0, 1.0, 8, dtype=np.float64),
                )
            )
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=(1, 1, 4),
            output_shape=(2, 1, 4),
        )
        result = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        self.assertFalse(result.active_mask[0])
        self.assertEqual(float(result.scalar_guard[0]).hex(), "0x0.0p+0")
        self.assertEqual(
            float(result.channel_dot_guard[0]).hex(), "0x0.0p+0"
        )
        self.assertEqual(
            float(result.accumulation_guard[0]).hex(), "0x0.0p+0"
        )

    def test_conv_subnormal_bit_differential(self):
        eta = np.nextafter(
            np.float64(0.0), np.float64(math.inf)
        )
        weight = np.ascontiguousarray(
            np.asarray([[[[eta]]]], dtype=np.float64)
        )
        lb = np.ascontiguousarray(
            np.asarray([-1.0], dtype=np.float64)
        )
        ub = np.ascontiguousarray(
            np.asarray([1.0], dtype=np.float64)
        )
        coefficient = np.ascontiguousarray(
            np.asarray([[0.5], [-0.5]], dtype=np.float64)
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
        )
        actual = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        _, _, expected = _public_conv(
            weight=weight,
            lb=lb,
            ub=ub,
            coefficient=coefficient,
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
        )
        self.assertTrue(
            np.array_equal(
                actual.coefficient.view(np.uint64),
                expected.coefficient.view(np.uint64),
            )
        )
        self.assertTrue(
            np.array_equal(
                actual.scalar_guard.view(np.uint64),
                expected.scalar_guard.view(np.uint64),
            )
        )
        self.assertTrue(np.all(actual.scalar_guard > 0.0))

    def test_conv_source_mutation_after_admission_is_irrelevant(self):
        weight = np.ascontiguousarray(
            np.asarray(
                [[[[1.0]]], [[[-0.5]]]], dtype=np.float64
            )
        )
        lb = -np.ones(4, dtype=np.float64)
        ub = np.ones(4, dtype=np.float64)
        coefficient = np.ascontiguousarray(
            np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(
                2, 8
            )
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=(1, 1, 4),
            output_shape=(2, 1, 4),
        )
        before = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        weight[:] = 1.0e200
        lb[:] = 0.0
        ub[:] = 0.0
        after = _decode_conv_result(
            port.execute_conv(locator, coefficient)
        )
        self.assertTrue(
            np.array_equal(before.coefficient, after.coefficient)
        )
        self.assertTrue(
            np.array_equal(before.scalar_guard, after.scalar_guard)
        )

    def test_conv_geometry_rejects_invalid_output_padding(self):
        weight = np.ones((1, 1, 3, 3), dtype=np.float64)
        lb = -np.ones(25, dtype=np.float64)
        ub = np.ones(25, dtype=np.float64)
        port = private.create_private_numeric_kernel(deadline=_end())
        with self.assertRaises(private.PrivateNumericKernelError):
            _admit_conv(
                port,
                weight=weight,
                lb=lb,
                ub=ub,
                input_shape=(1, 5, 5),
                output_shape=(1, 1, 1),
                stride=(2, 2),
            )
        valid_layer = _conv_layer(
            weight=weight,
            input_shape=(1, 5, 5),
            output_shape=(1, 2, 2),
            stride=(2, 2),
        )
        valid_plan = conv_v51.prepare_dense_conv_v51_plan(
            valid_layer,
            frozen._Box(lb=lb, ub=ub),
            deadline=_replay_deadline(),
        )
        exact_forged_plan = replace(
            valid_plan, output_shape=(1, 1, 1)
        )
        forged_port = private.create_private_numeric_kernel(
            deadline=_end()
        )
        with mock.patch.object(
            conv_v51,
            "prepare_dense_conv_v51_plan",
            return_value=exact_forged_plan,
        ) as forged_helper:
            with self.assertRaisesRegex(
                private.PrivateNumericKernelError, "INVALID_GEOMETRY"
            ):
                _admit_conv(
                    forged_port,
                    weight=weight,
                    lb=lb,
                    ub=ub,
                    input_shape=(1, 5, 5),
                    output_shape=(1, 1, 1),
                    stride=(2, 2),
                )
        forged_helper.assert_not_called()
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "INVALID_GEOMETRY"
        ):
            _admit_conv(
                port,
                weight=weight,
                lb=lb,
                ub=ub,
                input_shape=(1, 5, 5),
                output_shape=(1, 2, 2),
                groups=2,
            )


class PrivateKernelIsolationTests(unittest.TestCase):
    def _dense_fixture(self, port):
        weight = np.ascontiguousarray(
            np.asarray(
                [[1.0, -2.0, 0.5], [0.25, 4.0, -1.0]],
                dtype=np.float64,
            )
        )
        max_abs = np.ascontiguousarray(
            np.asarray([2.0, 1.0, 3.0], dtype=np.float64)
        )
        coefficient = np.ascontiguousarray(
            np.asarray([[1.0, -0.5], [0.25, 2.0]], dtype=np.float64)
        )
        locator = port.admit_dense(
            weight=weight, predecessor_max_abs=max_abs
        )
        return locator, coefficient

    def _conv_fixture(self, port):
        weight = np.ascontiguousarray(
            np.asarray(
                [[[[1.0]]], [[[-0.5]]]], dtype=np.float64
            )
        )
        lb = -np.ones(4, dtype=np.float64)
        ub = np.ones(4, dtype=np.float64)
        coefficient = np.ascontiguousarray(
            np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(
                2, 8
            )
        )
        locator = _admit_conv(
            port,
            weight=weight,
            lb=lb,
            ub=ub,
            input_shape=(1, 1, 4),
            output_shape=(2, 1, 4),
        )
        return locator, coefficient

    def test_public_api_exposes_no_core_or_material_accessor(self):
        self.assertEqual(
            set(private.__all__),
            {
                "NUMERIC_PROTOCOL",
                "PrivateNumericKernelError",
                "PrivateNumericKernelTimeout",
                "SCHEMA",
                "create_private_numeric_kernel",
            },
        )
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, _ = self._dense_fixture(port)
        for value in (port, locator):
            self.assertFalse(hasattr(value, "__dict__"))
            self.assertFalse(hasattr(value, "_capability"))
            self.assertFalse(hasattr(value, "get_core"))
            self.assertFalse(hasattr(value, "core"))
            self.assertFalse(hasattr(value, "weight"))
            self.assertFalse(hasattr(value, "support"))
            self.assertFalse(hasattr(value, "token"))
            self.assertFalse(value.proof_authority)
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "PORT_CONSTRUCTION"
        ):
            type(port)()

    def test_pre_factory_public_helper_substitution_fails_before_call(self):
        with mock.patch.object(
            dense_v51,
            "prepare_dense_support_v51",
            return_value=object(),
        ) as dense_forged:
            with self.assertRaisesRegex(
                private.PrivateNumericKernelError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                private.create_private_numeric_kernel(deadline=_end())
        dense_forged.assert_not_called()

        with mock.patch.object(
            conv_v51,
            "prepare_dense_conv_v51_plan",
            return_value=object(),
        ) as conv_forged:
            with self.assertRaisesRegex(
                private.PrivateNumericKernelError,
                "DEPENDENCY_SUBSTITUTION",
            ):
                private.create_private_numeric_kernel(deadline=_end())
        conv_forged.assert_not_called()

    def test_module_subclass_is_rejected_before_dynamic_attribute_call(self):
        _run_isolated(
            r"""
import math
import os
import time
import types
import ctypes

import numpy as np
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense_v51

calls = [0]

class ChangedModule(types.ModuleType):
    def __getattribute__(self, name):
        calls[0] += 1
        raise AssertionError(("changed module attribute called", name))

for module in (ctypes, math, os, np, dense_v51, ufunc_config):
    saved_type = type(module)
    module.__class__ = ChangedModule
    error = None
    try:
        try:
            private.create_private_numeric_kernel(
                deadline=float(time.monotonic() + 30.0)
            )
        except private.PrivateNumericKernelError as exc:
            error = exc
    finally:
        module.__class__ = saved_type
    if error is None or error.code != "DEPENDENCY_SUBSTITUTION":
        raise AssertionError(("factory accepted module subclass", module))

if calls[0] != 0:
    raise AssertionError(("changed module call count", calls[0]))
"""
        )

    def test_context_local_numeric_policy_is_restored_and_bit_exact(self):
        _run_isolated(
            r"""
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

deadline = time.monotonic() + 30.0
tiny = np.finfo(np.float64).tiny
weight = np.asarray([[tiny]], dtype=np.float64)
max_abs = np.asarray([1.0], dtype=np.float64)
coefficient = np.asarray([[0.5]], dtype=np.float64)
conv_weight = np.asarray([[[[tiny]]]], dtype=np.float64)
conv_lb = np.asarray([-1.0], dtype=np.float64)
conv_ub = np.asarray([1.0], dtype=np.float64)

baseline_port = private.create_private_numeric_kernel(deadline=deadline)
baseline_dense_locator = baseline_port.admit_dense(
    weight=weight, predecessor_max_abs=max_abs, tile_width=1
)
baseline_conv_locator = baseline_port.admit_conv(
    layer_id=0,
    weight=conv_weight,
    predecessor_lb=conv_lb,
    predecessor_ub=conv_ub,
    input_shape=(1, 1, 1),
    output_shape=(1, 1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
)
baseline_dense = baseline_port.execute_dense(
    baseline_dense_locator, coefficient
)
baseline_conv = baseline_port.execute_conv(
    baseline_conv_locator, coefficient
)

nominal = np.frombuffer(
    baseline_dense[2][0], dtype=np.float64
).reshape(baseline_dense[2][1])
guard = np.frombuffer(
    baseline_dense[6][0], dtype=np.float64
).reshape(baseline_dense[6][1])
required = abs(
    Fraction.from_float(0.5) * Fraction.from_float(float(tiny))
    - Fraction.from_float(float(nominal[0, 0]))
)
if Fraction.from_float(float(guard[0])) < required:
    raise AssertionError(("subnormal Fraction enclosure", guard[0], required))

callback_calls = [0]

def callback(*args):
    del args
    callback_calls[0] += 1
    raise AssertionError("ambient NumPy callback was invoked")

def snapshot():
    return (np.geterr().copy(), np.geterrcall(), np.getbufsize())

original = snapshot()
try:
    for policy in ("ignore", "raise", "warn", "call"):
        if policy == "call":
            np.seterrcall(callback)
            np.seterr(all="call")
        else:
            np.seterr(all=policy)
        before = snapshot()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            port = private.create_private_numeric_kernel(deadline=deadline)
            dense_locator = port.admit_dense(
                weight=weight,
                predecessor_max_abs=max_abs,
                tile_width=1,
            )
            conv_locator = port.admit_conv(
                layer_id=0,
                weight=conv_weight,
                predecessor_lb=conv_lb,
                predecessor_ub=conv_ub,
                input_shape=(1, 1, 1),
                output_shape=(1, 1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            )
            dense_result = port.execute_dense(
                dense_locator, coefficient
            )
            conv_result = port.execute_conv(
                conv_locator, coefficient
            )
        if dense_result != baseline_dense or conv_result != baseline_conv:
            raise AssertionError(("caller policy changed bits", policy))
        if snapshot() != before:
            raise AssertionError(("caller policy was not restored", policy))

    f64_max = np.finfo(np.float64).max
    overflow_port = private.create_private_numeric_kernel(deadline=deadline)
    overflow_locator = overflow_port.admit_dense(
        weight=np.asarray([[f64_max]], dtype=np.float64),
        predecessor_max_abs=np.asarray([0.0], dtype=np.float64),
    )
    for policy in ("ignore", "warn", "raise"):
        np.seterr(all=policy)
        before = snapshot()
        try:
            overflow_port.execute_dense(
                overflow_locator,
                np.asarray([[f64_max]], dtype=np.float64),
            )
        except private.PrivateNumericKernelError as exc:
            if (
                exc.code != "NUMERIC_GUARD"
                or type(exc.__cause__) is not FloatingPointError
            ):
                raise AssertionError(
                    ("overflow normalization", policy, exc.code)
                )
        else:
            raise AssertionError(("overflow returned", policy))
        if snapshot() != before:
            raise AssertionError(("overflow policy restore", policy))
finally:
    np.setbufsize(original[2])
    np.seterrcall(original[1])
    np.seterr(**original[0])

if callback_calls[0] != 0:
    raise AssertionError(("ambient callback calls", callback_calls[0]))

def worker(policy):
    old = snapshot()
    try:
        if policy == "call":
            np.seterrcall(callback)
            np.seterr(all="call")
        else:
            np.seterr(all=policy)
        before = snapshot()
        dense_result = baseline_port.execute_dense(
            baseline_dense_locator, coefficient
        )
        conv_result = baseline_port.execute_conv(
            baseline_conv_locator, coefficient
        )
        after = snapshot()
        return dense_result, conv_result, before, after
    finally:
        np.setbufsize(old[2])
        np.seterrcall(old[1])
        np.seterr(**old[0])

with ThreadPoolExecutor(max_workers=4) as pool:
    observed = tuple(
        pool.map(worker, ("ignore", "raise", "warn", "call"))
    )
for dense_result, conv_result, before, after in observed:
    if (
        dense_result != baseline_dense
        or conv_result != baseline_conv
        or before != after
    ):
        raise AssertionError("thread-local numeric policy mismatch")
if callback_calls[0] != 0:
    raise AssertionError(("thread callback calls", callback_calls[0]))
"""
        )

    def test_mutable_python_instrumentation_is_rejected(self):
        _run_isolated(
            r"""
import sys
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = port.admit_dense(
    weight=np.asarray([[-0.2]], dtype=np.float64),
    predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
)
coefficient = np.asarray([[-0.75]], dtype=np.float64)

def trace(frame, event, argument):
    del frame, event, argument
    return trace

def profile(frame, event, argument):
    del frame, event, argument

def expect_rejected():
    operations = (
        lambda: private.create_private_numeric_kernel(
            deadline=time.monotonic() + 30.0
        ),
        lambda: port.execute_dense(locator, coefficient),
    )
    for operation in operations:
        try:
            operation()
        except private.PrivateNumericKernelError as exc:
            if exc.code != "NUMERIC_PLATFORM":
                raise AssertionError(("instrumentation code", exc.code))
        else:
            raise AssertionError("numeric instrumentation was accepted")

sys.settrace(trace)
try:
    expect_rejected()
finally:
    sys.settrace(None)

sys.setprofile(profile)
try:
    expect_rejected()
finally:
    sys.setprofile(None)

tool_id = 5
sys.monitoring.use_tool_id(tool_id, "act-private-numeric-test")
try:
    expect_rejected()
finally:
    sys.monitoring.free_tool_id(tool_id)

stats = port.stats()
if stats["dense_executions"] != 0:
    raise AssertionError(("instrumented execution published", dict(stats)))
"""
        )

    def test_native_fenv_reader_state_is_gated_and_base_call_is_inert(self):
        _run_isolated(
            r"""
import ctypes
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

functions = {
    value.__name__: value
    for value in private._GatePrimitivesModule
    if isinstance(value, ctypes._CFuncPtr)
}
if set(functions) != {"fegetenv", "fesetenv"}:
    raise AssertionError(("native functions", sorted(functions)))
reader = functions["fegetenv"]
calls = [0]

def changed(*args, **kwargs):
    del args, kwargs
    calls[0] += 1
    raise AssertionError("changed native call path was invoked")

reader.errcheck = changed
try:
    try:
        private.create_private_numeric_kernel(
            deadline=time.monotonic() + 30.0
        )
    except private.PrivateNumericKernelError as exc:
        if exc.code != "DEPENDENCY_SUBSTITUTION":
            raise AssertionError(("pre-factory", exc.code))
    else:
        raise AssertionError("factory accepted changed native state")
finally:
    del reader.errcheck
if calls[0] != 0:
    raise AssertionError(("pre-factory callback calls", calls[0]))

port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = port.admit_dense(
    weight=np.asarray([[-0.2]], dtype=np.float64),
    predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
)
coefficient = np.asarray([[-0.75]], dtype=np.float64)
baseline = port.execute_dense(locator, coefficient)

reader.errcheck = changed
try:
    try:
        port.execute_dense(locator, coefficient)
    except private.PrivateNumericKernelError as exc:
        if exc.code != "DEPENDENCY_SUBSTITUTION":
            raise AssertionError(("post-factory", exc.code))
    else:
        raise AssertionError("operation accepted changed native state")
finally:
    del reader.errcheck
if calls[0] != 0:
    raise AssertionError(("post-factory callback calls", calls[0]))

function_type = type(reader)
if "__call__" in function_type.__dict__:
    raise AssertionError("unexpected native subclass __call__ override")
function_type.__call__ = changed
try:
    observed = port.execute_dense(locator, coefficient)
finally:
    del function_type.__call__
if calls[0] != 0 or observed != baseline:
    raise AssertionError(("native base-call bypass", calls[0]))
"""
        )

    @unittest.skipUnless(
        platform.machine().lower() in {"x86_64", "amd64"}
        and shutil.which("gcc") is not None,
        "MXCSR probe requires x86-64 and gcc",
    )
    def test_actual_thread_mxcsr_modes_reject_before_dense_or_conv(self):
        _run_isolated(
            r"""
import ctypes
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

source = r'''
#include <fenv.h>
#include <xmmintrin.h>
#include <fpu_control.h>
unsigned act_get_mxcsr(void) { return _mm_getcsr(); }
void act_set_mxcsr(unsigned value) { _mm_setcsr(value); }
unsigned short act_get_x87(void) {
    fpu_control_t value;
    _FPU_GETCW(value);
    return value;
}
void act_set_x87(unsigned short value) {
    fpu_control_t current = value;
    _FPU_SETCW(current);
}
void act_clear_flags(void) {
    __asm__ __volatile__("fnclex");
    _mm_setcsr(_mm_getcsr() & ~0x3fu);
}
unsigned short act_get_x87_status(void) {
    unsigned short value;
    __asm__ __volatile__("fnstsw %0" : "=am"(value));
    return value;
}
void act_set_x87_pending_unmasked(unsigned bit) {
    fenv_t environment;
    unsigned char *bytes = (unsigned char *)&environment;
    fpu_control_t control;
    fegetenv(&environment);
    bytes[0] |= 0x3fu;
    bytes[4] = (unsigned char)(bit | 0x80u);
    bytes[5] &= 0x7fu;
    fesetenv(&environment);
    _FPU_GETCW(control);
    control &= (fpu_control_t)~bit;
    _FPU_SETCW(control);
}
void act_set_mxcsr_pending_unmasked(unsigned bit) {
    unsigned value = _mm_getcsr();
    value |= 0x1f80u;
    value &= ~(bit << 7);
    value |= bit;
    _mm_setcsr(value);
}
'''
with tempfile.TemporaryDirectory(prefix="act_mxcsr_") as directory:
    library_path = pathlib.Path(directory) / "libact_mxcsr.so"
    subprocess.run(
        [
            "gcc", "-shared", "-fPIC", "-O2", "-x", "c",
            "-o", str(library_path), "-", "-lm",
        ],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.act_get_mxcsr.restype = ctypes.c_uint
    library.act_set_mxcsr.argtypes = (ctypes.c_uint,)
    library.act_get_x87.restype = ctypes.c_ushort
    library.act_set_x87.argtypes = (ctypes.c_ushort,)
    library.act_clear_flags.argtypes = ()
    library.act_get_x87_status.restype = ctypes.c_ushort
    library.act_set_x87_pending_unmasked.argtypes = (
        ctypes.c_uint,
    )
    library.act_set_mxcsr_pending_unmasked.argtypes = (
        ctypes.c_uint,
    )

    deadline = time.monotonic() + 30.0
    port = private.create_private_numeric_kernel(
        deadline=deadline
    )
    tiny = np.finfo(np.float64).tiny
    dense_locator = port.admit_dense(
        weight=np.asarray([[tiny]], dtype=np.float64),
        predecessor_max_abs=np.asarray([1.0], dtype=np.float64),
        tile_width=1,
    )
    conv_locator = port.admit_conv(
        layer_id=0,
        weight=np.asarray([[[[tiny]]]], dtype=np.float64),
        predecessor_lb=np.asarray([-1.0], dtype=np.float64),
        predecessor_ub=np.asarray([1.0], dtype=np.float64),
        input_shape=(1, 1, 1),
        output_shape=(1, 1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    coefficient = np.asarray([[0.5]], dtype=np.float64)
    baseline_dense = port.execute_dense(dense_locator, coefficient)
    baseline_conv = port.execute_conv(conv_locator, coefficient)
    if baseline_dense[6][0] == bytes(8) or baseline_conv[3][0] == bytes(8):
        raise AssertionError("baseline subnormal guard was zero")

    old_mxcsr = library.act_get_mxcsr()
    old_x87 = library.act_get_x87()
    modes = (
        ("MXCSR_FTZ", "mx_mode", 0x8000),
        ("MXCSR_DAZ", "mx_mode", 0x0040),
        ("MXCSR_FTZ_DAZ", "mx_mode", 0x8040),
        ("MXCSR_ROUND_DOWN", "mx_mode", 0x2000),
        ("MXCSR_ROUND_UP", "mx_mode", 0x4000),
        ("MXCSR_ROUND_ZERO", "mx_mode", 0x6000),
        ("X87_ROUND_DOWN", "x87_mode", 0x0400),
        ("X87_ROUND_UP", "x87_mode", 0x0800),
        ("X87_ROUND_ZERO", "x87_mode", 0x0C00),
        ("X87_PRECISION_24", "x87_precision", 0x0000),
        ("X87_PRECISION_53", "x87_precision", 0x0200),
    )
    modes += tuple(
        (f"MXCSR_UNMASK_{bit:x}", "mx_unmask", bit)
        for bit in (0x80, 0x100, 0x200, 0x400, 0x800, 0x1000)
    )
    modes += tuple(
        (f"X87_UNMASK_{bit:x}", "x87_unmask", bit)
        for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)
    )

    operations = (
        lambda: private.create_private_numeric_kernel(
            deadline=deadline
        ),
        lambda: port.admit_dense(
            weight=np.asarray([[tiny]], dtype=np.float64),
            predecessor_max_abs=np.asarray([1.0], dtype=np.float64),
            tile_width=1,
        ),
        lambda: port.admit_conv(
            layer_id=1,
            weight=np.asarray([[[[tiny]]]], dtype=np.float64),
            predecessor_lb=np.asarray([-1.0], dtype=np.float64),
            predecessor_ub=np.asarray([1.0], dtype=np.float64),
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        ),
        lambda: port.execute_dense(dense_locator, coefficient),
        lambda: port.execute_conv(conv_locator, coefficient),
    )

    def set_mode(kind, bits):
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)
        library.act_clear_flags()
        if kind == "mx_mode":
            library.act_set_mxcsr(
                (library.act_get_mxcsr() & ~0xE040) | bits
            )
        elif kind == "mx_unmask":
            library.act_set_mxcsr(
                library.act_get_mxcsr() & ~bits
            )
        elif kind == "x87_mode":
            library.act_set_x87((old_x87 & ~0x0C00) | bits)
        elif kind == "x87_precision":
            library.act_set_x87((old_x87 & ~0x0300) | bits)
        else:
            library.act_set_x87(old_x87 & ~bits)

    for name, kind, bits in modes:
        codes = []
        for operation in operations:
            try:
                set_mode(kind, bits)
                expected_state = (
                    library.act_get_mxcsr(),
                    library.act_get_x87(),
                )
                for _attempt in range(2):
                    try:
                        operation()
                    except private.PrivateNumericKernelError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                    observed_state = (
                        library.act_get_mxcsr(),
                        library.act_get_x87(),
                    )
                    if observed_state != expected_state:
                        raise AssertionError(
                            (
                                name,
                                "state changed",
                                expected_state,
                                observed_state,
                            )
                        )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
        if codes != ["NUMERIC_PLATFORM"] * (2 * len(operations)):
            raise AssertionError((name, codes))

    # A pending exception whose own x87 mask is removed cannot be returned
    # safely to Python unchanged.  The checker must preserve both control
    # words, clear only the hazardous x87 status flags, reject, and continue
    # rejecting on the next call.
    for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20):
        for operation in operations:
            try:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
                library.act_clear_flags()
                library.act_set_x87_pending_unmasked(bit)
                before_control = library.act_get_x87()
                before_mxcsr = library.act_get_mxcsr()
                before_status = library.act_get_x87_status()
                if (
                    before_control & bit
                    or not before_status & bit
                ):
                    raise AssertionError(
                        ("pending setup failed", bit, before_status)
                    )
                codes = []
                for _attempt in range(2):
                    try:
                        operation()
                    except private.PrivateNumericKernelError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                after_control = library.act_get_x87()
                after_mxcsr = library.act_get_mxcsr()
                after_status = library.act_get_x87_status()
                if (
                    codes != ["NUMERIC_PLATFORM", "NUMERIC_PLATFORM"]
                    or after_control != before_control
                    or after_mxcsr != before_mxcsr
                    or after_status & 0x80FF
                ):
                    raise AssertionError(
                        (
                            "pending rejection",
                            bit,
                            codes,
                            before_control,
                            after_control,
                            before_mxcsr,
                            after_mxcsr,
                            before_status,
                            after_status,
                        )
                    )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)

    # SSE exceptions are precise rather than deferred, but an already-invalid
    # caller state should receive the same targeted sanitation: clear only
    # the unmasked pending flag while preserving every MXCSR control bit.
    for bit in (0x01, 0x02, 0x04, 0x08, 0x10, 0x20):
        for operation in operations:
            try:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)
                library.act_clear_flags()
                library.act_set_mxcsr_pending_unmasked(bit)
                before_mxcsr = library.act_get_mxcsr()
                before_control = before_mxcsr & ~0x003F
                if (
                    not before_mxcsr & bit
                    or before_mxcsr & (bit << 7)
                ):
                    raise AssertionError(
                        ("MXCSR pending setup failed", bit, before_mxcsr)
                    )
                codes = []
                for _attempt in range(2):
                    try:
                        operation()
                    except private.PrivateNumericKernelError as exc:
                        codes.append(exc.code)
                    else:
                        codes.append("RETURNED")
                after_mxcsr = library.act_get_mxcsr()
                if (
                    codes != ["NUMERIC_PLATFORM", "NUMERIC_PLATFORM"]
                    or after_mxcsr & ~0x003F != before_control
                    or after_mxcsr & bit
                ):
                    raise AssertionError(
                        (
                            "MXCSR pending rejection",
                            bit,
                            codes,
                            before_mxcsr,
                            after_mxcsr,
                        )
                    )
            finally:
                library.act_set_mxcsr(old_mxcsr)
                library.act_set_x87(old_x87)

    # Re-enter after the outer fegetenv has filled its local record but before
    # it parses the record.  Both calls must see the original invalid state;
    # a shared record would let the inner call overwrite the outer evidence.
    try:
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)
        library.act_clear_flags()
        library.act_set_mxcsr(
            library.act_get_mxcsr() & ~0x0100
        )
        expected_state = (
            library.act_get_mxcsr(),
            library.act_get_x87(),
        )
        nested_codes = []
        in_callback = [False]

        def trace_reentry(frame, event, argument):
            del argument
            if (
                event == "line"
                and frame.f_code.co_name == "_check_fenv_control"
                and not in_callback[0]
                and not nested_codes
                and frame.f_locals.get("read_status") == 0
                and "control_word" not in frame.f_locals
            ):
                in_callback[0] = True
                try:
                    try:
                        port.stats()
                    except private.PrivateNumericKernelError as exc:
                        nested_codes.append(exc.code)
                    else:
                        nested_codes.append("RETURNED")
                finally:
                    in_callback[0] = False
            return trace_reentry

        outer_code = "RETURNED"
        sys.settrace(trace_reentry)
        try:
            try:
                port.stats()
            except private.PrivateNumericKernelError as exc:
                outer_code = exc.code
        finally:
            sys.settrace(None)
        second_code = "RETURNED"
        try:
            port.stats()
        except private.PrivateNumericKernelError as exc:
            second_code = exc.code
        if (
            nested_codes != ["NUMERIC_PLATFORM"]
            or outer_code != "NUMERIC_PLATFORM"
            or second_code != "NUMERIC_PLATFORM"
            or (
                library.act_get_mxcsr(),
                library.act_get_x87(),
            )
            != expected_state
        ):
            raise AssertionError(
                (
                    "trace re-entry",
                    nested_codes,
                    outer_code,
                    second_code,
                    expected_state,
                    library.act_get_mxcsr(),
                    library.act_get_x87(),
                )
            )
    finally:
        sys.settrace(None)
        library.act_set_mxcsr(old_mxcsr)
        library.act_set_x87(old_x87)

    # Restoring the caller thread permits the original bit-exact results.
    if (
        port.execute_dense(dense_locator, coefficient) != baseline_dense
        or port.execute_conv(conv_locator, coefficient) != baseline_conv
    ):
        raise AssertionError("MXCSR restoration changed normal results")
"""
        )

    def test_conv_int64_geometry_is_checked_before_public_arithmetic(self):
        max_i64 = (1 << 63) - 1

        def attempt(**geometry):
            port = private.create_private_numeric_kernel(deadline=_end())
            input_size = geometry.pop("input_size")
            with self.assertRaises(
                private.PrivateNumericKernelError
            ) as captured:
                _admit_conv(
                    port,
                    weight=np.ones((1, 1, 1, 1), dtype=np.float64),
                    lb=-np.ones(input_size, dtype=np.float64),
                    ub=np.ones(input_size, dtype=np.float64),
                    **geometry,
                )
            self.assertEqual(captured.exception.code, "INVALID_GEOMETRY")
            self.assertIsNone(captured.exception.__cause__)
            self.assertEqual(port.stats()["conv_admissions"], 0)
            self.assertEqual(port.stats()["material_count"], 0)

        attempt(
            input_size=1,
            input_shape=(1, 1, 1),
            output_shape=(1, 1, 2),
            stride=(1, 1 << 63),
            padding=(0, 1 << 62),
        )
        step = (max_i64 + 1) // 2
        attempt(
            input_size=1,
            input_shape=(1, 1, 1),
            output_shape=(1, 3, 1),
            stride=(step, 1),
            padding=(step, 0),
        )
        for field, value in (
            ("padding", (0, max_i64 + 1)),
            ("dilation", (1, max_i64 + 1)),
        ):
            geometry = {
                "input_size": 1,
                "input_shape": (1, 1, 1),
                "output_shape": (1, 1, 1),
                field: value,
            }
            attempt(**geometry)
        attempt(
            input_size=1,
            input_shape=(1, 1 << 32, 1 << 32),
            output_shape=(1, 1, 1),
        )
        attempt(
            input_size=1,
            input_shape=(1, 1, 1),
            output_shape=(1, 1 << 32, 1 << 32),
            padding=((1 << 31), (1 << 31)),
        )

        boundary = private.create_private_numeric_kernel(deadline=_end())
        locator = _admit_conv(
            boundary,
            weight=np.ones((1, 1, 1, 1), dtype=np.float64),
            lb=-np.ones(2, dtype=np.float64),
            ub=np.ones(2, dtype=np.float64),
            input_shape=(1, 1, 2),
            output_shape=(1, 1, 2),
            stride=(1, max_i64),
            padding=(0, (max_i64 - 1) // 2),
        )
        result = boundary.execute_conv(
            locator, np.ones((1, 2), dtype=np.float64)
        )
        self.assertEqual(result[2][0], bytes(16))
        self.assertEqual(result[3][0], bytes(8))

    def test_conv_resource_preflight_rejects_large_valid_int64_geometry(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            _admit_conv(
                port,
                weight=np.ones((1, 1, 1, 1), dtype=np.float64),
                lb=-np.ones(1, dtype=np.float64),
                ub=np.ones(1, dtype=np.float64),
                input_shape=(1, 1, 1),
                output_shape=(1, 1, 5_000_001),
                stride=(1, 1),
                padding=(0, 2_500_000),
                dilation=(1, 1),
                groups=1,
            )
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        stats = port.stats()
        self.assertEqual(stats["conv_admissions"], 0)
        self.assertEqual(stats["material_count"], 0)

    def test_resource_preflight_precedes_large_snapshots_and_outputs(self):
        conv_port = private.create_private_numeric_kernel(deadline=_end())
        # np.empty reserves the caller's exact ndarray without touching its
        # 256 MiB payload.  The private kernel must reject from metadata and
        # geometry before tobytes can copy or inspect that payload.
        large_weight = np.empty(
            (1, 1, 5793, 5793), dtype=np.float64
        )
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            _admit_conv(
                conv_port,
                weight=large_weight,
                lb=-np.ones(1, dtype=np.float64),
                ub=np.ones(1, dtype=np.float64),
                input_shape=(1, 1, 1),
                output_shape=(1, 1, 1),
                stride=(1, 1),
                padding=(2896, 2896),
                dilation=(1, 1),
                groups=1,
            )
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(conv_port.stats()["conv_admissions"], 0)
        del large_weight

        dense_port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator = dense_port.admit_dense(
            weight=np.ones((1, 16384), dtype=np.float64),
            predecessor_max_abs=np.ones(16384, dtype=np.float64),
        )
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            dense_port.execute_dense(
                dense_locator,
                np.ones((8193, 1), dtype=np.float64),
            )
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(dense_port.stats()["dense_executions"], 0)

        # The final Dense frames are only one column and stay far below
        # 1 GiB.  The B-by-I compensated/tiled intermediates alone exceed the
        # 2 GiB workspace budget, so metadata must reject before reading the
        # caller's untouched np.empty payload.
        dense_port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator = dense_port.admit_dense(
            weight=np.ones((4096, 1), dtype=np.float64),
            predecessor_max_abs=np.ones(1, dtype=np.float64),
        )
        large_coefficients = np.empty((8193, 4096), dtype=np.float64)
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            dense_port.execute_dense(
                dense_locator,
                large_coefficients,
            )
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(dense_port.stats()["dense_executions"], 0)
        del large_coefficients

        conv_port = private.create_private_numeric_kernel(deadline=_end())
        input_shape = (1, 256, 256)
        conv_locator = _admit_conv(
            conv_port,
            weight=np.ones((1, 1, 1, 1), dtype=np.float64),
            lb=-np.ones(256 * 256, dtype=np.float64),
            ub=np.ones(256 * 256, dtype=np.float64),
            input_shape=input_shape,
            output_shape=(1, 1, 1),
            stride=(256, 256),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            conv_port.execute_conv(
                conv_locator,
                np.ones((2049, 1), dtype=np.float64),
            )
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(conv_port.stats()["conv_executions"], 0)

        # Final frames remain below 1 GiB here; rejection is specifically from
        # the per-offset selected/GEMM/merge workspace bound.
        conv_port = private.create_private_numeric_kernel(deadline=_end())
        conv_locator, _ = self._conv_fixture(conv_port)
        large_batch = np.empty((2_300_000, 8), dtype=np.float64)
        with self.assertRaises(
            private.PrivateNumericKernelError
        ) as captured:
            conv_port.execute_conv(conv_locator, large_batch)
        self.assertEqual(captured.exception.code, "RESOURCE_LIMIT")
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(conv_port.stats()["conv_executions"], 0)
        del large_batch

    def test_conv_workspace_preflight_checks_deadline_before_snapshot(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._conv_fixture(port)

        def closure_values(function):
            return {
                name: cell
                for name, cell in zip(
                    function.__code__.co_freevars,
                    function.__closure__ or (),
                )
            }

        guarded_cells = closure_values(port.execute_conv.__func__)
        operation = guarded_cells["operation"].cell_contents
        operation_cells = closure_values(operation)
        conv_runtime = operation_cells["_conv_runtime"].cell_contents
        resolve = operation_cells["_resolve"].cell_contents
        runtime_cells = closure_values(conv_runtime)
        check_live = runtime_cells["_check_live"].cell_contents
        live_cells = closure_values(check_live)
        end_cell = live_cells["end"]
        cores = closure_values(resolve)["cores"].cell_contents
        self.assertEqual(len(cores), 1)
        core = next(iter(cores.values()))

        snapshot_calls = []

        def observe(frame, event, argument):
            del argument
            if (
                event == "call"
                and frame.f_code.co_name == "_snapshot_exact_array"
            ):
                snapshot_calls.append(frame.f_code.co_name)
            return observe

        original_end = end_cell.cell_contents
        end_cell.cell_contents = float(time.monotonic() - 1.0)
        sys.settrace(observe)
        try:
            with self.assertRaises(
                private.PrivateNumericKernelTimeout
            ):
                conv_runtime(core, coefficient)
        finally:
            sys.settrace(None)
            end_cell.cell_contents = original_end
        self.assertEqual(snapshot_calls, [])
        del locator

    def test_conv_admission_checks_deadline_before_each_offset_work(self):
        port = private.create_private_numeric_kernel(deadline=_end())

        def closure_values(function):
            return {
                name: cell
                for name, cell in zip(
                    function.__code__.co_freevars,
                    function.__closure__ or (),
                )
            }

        guarded_cells = closure_values(port.admit_conv.__func__)
        operation = guarded_cells["operation"].cell_contents
        operation_cells = closure_values(operation)
        check_port = operation_cells["_check_port"].cell_contents
        check_live = closure_values(check_port)[
            "_check_live"
        ].cell_contents
        live_cells = closure_values(check_live)
        end = live_cells["end"].cell_contents
        monotonic_cell = live_cells["_monotonic"]
        original_monotonic = monotonic_cell.cell_contents
        armed = []
        dot_calls = []

        def scripted_monotonic():
            if armed:
                return float(end + 1.0)
            return original_monotonic()

        def expire_at_first_offset(frame, event, argument):
            del argument
            if (
                event == "line"
                and frame.f_code is operation.__code__
                and frame.f_locals.get("kw") == 0
                and "input_w_indices" not in frame.f_locals
                and not armed
            ):
                armed.append(True)
            if (
                event == "call"
                and frame.f_code.co_name == "_dot_up_matrix"
                and armed
            ):
                dot_calls.append(True)
            return expire_at_first_offset

        monotonic_cell.cell_contents = scripted_monotonic
        sys.settrace(expire_at_first_offset)
        try:
            with self.assertRaises(
                private.PrivateNumericKernelTimeout
            ):
                operation(
                    port,
                    layer_id=0,
                    weight=np.ones(
                        (1, 1, 1, 2), dtype=np.float64
                    ),
                    predecessor_lb=-np.ones(
                        2, dtype=np.float64
                    ),
                    predecessor_ub=np.ones(
                        2, dtype=np.float64
                    ),
                    input_shape=(1, 1, 2),
                    output_shape=(1, 1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    groups=1,
                )
        finally:
            sys.settrace(None)
            monotonic_cell.cell_contents = original_monotonic
        self.assertEqual(armed, [True])
        self.assertEqual(dot_calls, [])

    def test_exact_numpy_array_memory_error_is_normalized(self):
        _run_isolated(
            r"""
import importlib
import sys
import time

import numpy as np
from numpy._core import _exceptions as np_exceptions

from act.back_end.hybridz_tf import (
    query_dual_scalar_guard_v51 as dense,
)

private_name = (
    "act.back_end.hybridz_tf."
    "query_dual_replay_v51b_private_kernel"
)
sys.modules.pop(private_name, None)
original = dense.prepare_dense_support_v51

def raise_array_memory(*args, **kwargs):
    del args, kwargs
    raise np_exceptions._ArrayMemoryError(
        (1024, 1024), np.dtype(np.float64)
    )

dense.prepare_dense_support_v51 = raise_array_memory
try:
    private = importlib.import_module(private_name)
    port = private.create_private_numeric_kernel(
        deadline=time.monotonic() + 30.0
    )
    try:
        port.admit_dense(
            weight=np.ones((1, 1), dtype=np.float64),
            predecessor_max_abs=np.ones(1, dtype=np.float64),
        )
    except private.PrivateNumericKernelError as exc:
        if (
            exc.code != "RESOURCE_LIMIT"
            or type(exc.__cause__)
            is not np_exceptions._ArrayMemoryError
        ):
            raise AssertionError(
                (
                    "normalization",
                    exc.code,
                    type(exc.__cause__),
                )
            )
    else:
        raise AssertionError("NumPy ArrayMemoryError escaped")
finally:
    dense.prepare_dense_support_v51 = original
    sys.modules.pop(private_name, None)
"""
        )

    def test_factory_low_address_space_does_not_enter_parallel_blas(self):
        source = r"""
import resource
import time

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

with open("/proc/self/status", "r", encoding="ascii") as stream:
    vm_kib = next(
        int(line.split()[1])
        for line in stream
        if line.startswith("VmSize:")
    )
deadline = time.monotonic() + 30.0
_, hard = resource.getrlimit(resource.RLIMIT_AS)
wanted = vm_kib * 1024 + DELTA_MIB * 1024 * 1024
if hard != resource.RLIM_INFINITY and wanted > hard:
    raise AssertionError(("unexpected hard address-space limit", hard))
resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))
try:
    port = private.create_private_numeric_kernel(deadline=deadline)
except private.PrivateNumericKernelError as exc:
    if exc.code != "RESOURCE_LIMIT":
        raise AssertionError(("unexpected stable factory code", exc.code))
else:
    port.close()
"""
        for delta_mib in (2, 4, 8, 16, 24):
            _run_isolated(
                source.replace("DELTA_MIB", str(delta_mib))
            )

    def test_reflective_private_build_class_change_is_inert(self):
        implementation = _implementation_spec(
            private.create_private_numeric_kernel,
            "_create_private_numeric_kernel_impl",
        )
        self.assertFalse(
            any(
                type(cell.cell_contents) is types.FunctionType
                for cell in (
                    private.create_private_numeric_kernel.__closure__ or ()
                )
            )
        )
        first = private.create_private_numeric_kernel(deadline=_end())
        implementation_globals = type(first).stats.__globals__
        private_builtins = implementation_globals["__builtins__"]
        self.assertIs(type(private_builtins), dict)
        self.assertEqual(private_builtins, {})
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("changed private class builder was called")

        private_builtins["__build_class__"] = changed
        try:
            locator, coefficient = self._dense_fixture(first)
            result = first.execute_dense(locator, coefficient)
        finally:
            del private_builtins["__build_class__"]
        self.assertIs(type(result), tuple)
        self.assertEqual(calls[0], 0)

        original_entry = implementation_globals["__builtins__"]
        implementation_globals["__builtins__"] = {
            "__build_class__": changed
        }
        try:
            second = private.create_private_numeric_kernel(deadline=_end())
            self.assertEqual(second.stats()["material_count"], 0)
            self.assertIsNot(
                type(second).stats.__globals__, implementation_globals
            )
        finally:
            implementation_globals["__builtins__"] = original_entry
        self.assertEqual(calls[0], 0)
        self.assertIs(type(implementation[0]), types.CodeType)
        self.assertIs(implementation[4], types.FunctionType)
        self.assertIs(implementation[5], dict)

    def test_recursive_kernel_bytecode_has_no_dynamic_global_or_class_load(
        self,
    ):
        implementation = _implementation_spec(
            private.create_private_numeric_kernel,
            "_create_private_numeric_kernel_impl",
        )
        pending = [
            private.create_private_numeric_kernel.__code__,
            implementation[0],
        ]
        seen = set()
        failures = []
        forbidden = {
            "LOAD_GLOBAL",
            "STORE_GLOBAL",
            "DELETE_GLOBAL",
            "IMPORT_NAME",
            "IMPORT_FROM",
            "LOAD_BUILD_CLASS",
        }
        while pending:
            code = pending.pop()
            if id(code) in seen:
                continue
            seen.add(id(code))
            failures.extend(
                (
                    code.co_name,
                    instruction.offset,
                    instruction.opname,
                    instruction.argrepr,
                )
                for instruction in dis.get_instructions(code)
                if instruction.opname in forbidden
            )
            pending.extend(
                value
                for value in code.co_consts
                if type(value) is types.CodeType
            )
        self.assertEqual(failures, [])

    def test_pre_factory_dependency_substitution_isolated(self):
        _run_isolated(
            r"""
import builtins
import ctypes
import _thread
import math
import os
import time
import types
import weakref

import numpy as np
from numpy._core import _exceptions as np_exceptions
from numpy._core import _ufunc_config as ufunc_config

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

calls = [0]

def evil(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("forged dependency was called")

builtin_names = (
    "bytes", "int", "tuple", "float", "bool", "str", "dict",
    "type", "object", "range", "min", "max", "memoryview",
    "sum", "len", "any",
    "property", "Exception", "FloatingPointError", "MemoryError",
    "OverflowError", "TypeError", "ValueError",
)
targets = [(builtins, name) for name in builtin_names]
targets += [
    (np, "__version__"),
    (np, "nextafter"),
    (np, "any"),
    (np, "asarray"),
    (math, "isfinite"),
    (math, "prod"),
    (ctypes, "Array"),
    (ctypes, "CDLL"),
    (ctypes, "_CFuncPtr"),
    (ctypes, "c_int"),
    (ctypes, "c_ubyte"),
    (os, "getpid"),
    (os, "uname"),
    (_thread, "RLock"),
    (time, "monotonic"),
    (types, "ModuleType"),
    (types, "MappingProxyType"),
    (weakref, "ref"),
    (np_exceptions, "_ArrayMemoryError"),
    (ufunc_config, "_extobj_contextvar"),
    (ufunc_config, "_make_extobj"),
]

for owner, name in targets:
    saved = getattr(owner, name)
    deadline = time.monotonic() + 30.0
    setattr(owner, name, evil)
    try:
        try:
            private.create_private_numeric_kernel(deadline=deadline)
        except private.PrivateNumericKernelError as exc:
            if exc.code != "DEPENDENCY_SUBSTITUTION":
                raise AssertionError((name, exc.code))
        else:
            raise AssertionError((name, "factory accepted substitution"))
    finally:
        setattr(owner, name, saved)

if calls[0] != 0:
    raise AssertionError(("forged call count", calls[0]))

port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = port.admit_dense(
    weight=np.asarray([[-0.2]], dtype=np.float64),
    predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
)
result = port.execute_dense(
    locator, np.asarray([[-0.75]], dtype=np.float64)
)
if type(result) is not tuple or result[1] is not False:
    raise AssertionError("normal result envelope is not exact")
for frame in result[2:]:
    if (
        type(frame) is not tuple
        or type(frame[0]) is not bytes
        or type(frame[1]) is not tuple
        or any(type(extent) is not int for extent in frame[1])
        or type(frame[2]) is not bytes
    ):
        raise AssertionError("normal result frame is not exact")
"""
        )

    def test_pre_factory_ufunc_reduce_override_is_rejected_before_call(self):
        _run_isolated(
            r"""
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed ufunc reduce was called")

for name in (
    "add", "logical_and", "logical_or", "maximum", "minimum"
):
    ufunc = getattr(np, name)
    if "reduce" in ufunc.__dict__:
        raise AssertionError((name, "unexpected baseline override"))
    setattr(ufunc, "reduce", changed)
    error = None
    try:
        try:
            private.create_private_numeric_kernel(
                deadline=time.monotonic() + 30.0
            )
        except private.PrivateNumericKernelError as exc:
            error = exc
    finally:
        delattr(ufunc, "reduce")
    if error is None or error.code != "DEPENDENCY_SUBSTITUTION":
        raise AssertionError((name, "factory accepted reduce override"))

# The full instance-state fingerprint also rejects unrelated additions.
np.logical_and._act_state_probe = object()
try:
    try:
        private.create_private_numeric_kernel(
            deadline=time.monotonic() + 30.0
        )
    except private.PrivateNumericKernelError as exc:
        if exc.code != "DEPENDENCY_SUBSTITUTION":
            raise AssertionError(("state probe", exc.code))
    else:
        raise AssertionError("factory accepted changed ufunc state")
finally:
    delattr(np.logical_and, "_act_state_probe")

if calls[0] != 0:
    raise AssertionError(("changed reduce call count", calls[0]))
"""
        )

    def test_bound_ufunc_reduces_and_platform_finfo_are_lexicalized(self):
        _run_isolated(
            r"""
import time

import numpy as np

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = port.admit_dense(
    weight=np.asarray(
        [[-0.2, 0.5], [1.0, -0.25]], dtype=np.float64
    ),
    predecessor_max_abs=np.asarray([5.0, 2.0], dtype=np.float64),
    tile_width=1,
)
coefficient = np.asarray(
    [[-0.75, 0.25], [-0.0, 0.0], [1.0, -2.0]],
    dtype=np.float64,
)
baseline = port.execute_dense(locator, coefficient)

calls = [0]

def changed(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("changed ufunc reduce was called")

ufuncs = tuple(
    getattr(np, name)
    for name in (
        "add", "logical_and", "logical_or", "maximum", "minimum"
    )
)
try:
    for ufunc in ufuncs:
        setattr(ufunc, "reduce", changed)
    observed = port.execute_dense(locator, coefficient)
finally:
    for ufunc in reversed(ufuncs):
        delattr(ufunc, "reduce")
if calls[0] != 0 or observed != baseline:
    raise AssertionError(("post-factory reduce mutation", calls[0]))

# np.finfo returns mutable cached singletons.  The kernel must retain only
# immutable scalar copies, never either singleton.
if (
    hasattr(private, "_F64InfoModule")
    or hasattr(private, "_WideInfoModule")
    or hasattr(private, "_F64InfoCaptureModule")
    or hasattr(private, "_WideInfoCaptureModule")
):
    raise AssertionError("kernel leaked a mutable finfo singleton")
f64 = np.finfo(np.float64)
wide = np.finfo(np.longdouble)
saved_f64_eps = f64.eps
saved_wide_nmant = wide.nmant
try:
    f64.eps = 1.0
    wide.nmant = 1
    second = private.create_private_numeric_kernel(
        deadline=time.monotonic() + 30.0
    )
finally:
    f64.eps = saved_f64_eps
    wide.nmant = saved_wide_nmant
second_locator = second.admit_dense(
    weight=np.asarray([[-0.2]], dtype=np.float64),
    predecessor_max_abs=np.asarray([5.0], dtype=np.float64),
)
result = second.execute_dense(
    second_locator, np.asarray([[-0.75]], dtype=np.float64)
)
if type(result) is not tuple or result[1] is not False:
    raise AssertionError("lexicalized platform scalars changed result ABI")
"""
        )

    def test_post_factory_numeric_substitution_fraction_regression(self):
        _run_isolated(
            r"""
import builtins
import ctypes
import math
import time
from fractions import Fraction

import numpy as np
from numpy._core import _methods as np_methods

from act.back_end.hybridz_tf import (
    query_dual_replay_v51b_private_kernel as private,
)

weight = np.asarray([[-0.2]], dtype=np.float64)
max_abs = np.asarray([5.0], dtype=np.float64)
coefficient = np.asarray([[-0.75]], dtype=np.float64)
port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
locator = port.admit_dense(
    weight=weight, predecessor_max_abs=max_abs
)
baseline = port.execute_dense(locator, coefficient)

calls = [0]

def evil(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("forged post-factory dependency was called")

targets = [
    (ctypes, "Array"),
    (ctypes, "CDLL"),
    (ctypes, "_CFuncPtr"),
    (ctypes, "c_int"),
    (ctypes, "c_ubyte"),
    (np, "__version__"),
    (np, "nextafter"),
    (np, "any"),
    (np, "asarray"),
    (np, "ascontiguousarray"),
    (np, "matmul"),
    (np_methods, "_all"),
    (np_methods, "_any"),
    (np_methods, "_amax"),
    (np_methods, "_amin"),
    (np_methods, "_sum"),
    (math, "isfinite"),
    (builtins, "bytes"),
    (builtins, "int"),
    (builtins, "tuple"),
    (builtins, "float"),
    (builtins, "bool"),
    (builtins, "str"),
    (builtins, "dict"),
    (builtins, "object"),
    (builtins, "len"),
    (builtins, "max"),
    (builtins, "min"),
    (builtins, "range"),
    (builtins, "sum"),
    (builtins, "any"),
    (builtins, "type"),
]
saved = [(owner, name, getattr(owner, name)) for owner, name in targets]
try:
    for owner, name, _ in saved:
        setattr(owner, name, evil)
    observed = port.execute_dense(locator, coefficient)
finally:
    for owner, name, original in reversed(saved):
        setattr(owner, name, original)

if calls[0] != 0:
    raise AssertionError(("persistent forged calls", calls[0]))
if observed != baseline:
    raise AssertionError("persistent substitution changed output bits")

for owner, name, original in saved:
    setattr(owner, name, evil)
    setattr(owner, name, original)
aba = port.execute_dense(locator, coefficient)
if calls[0] != 0 or aba != baseline:
    raise AssertionError("ABA substitution changed sealed execution")

nominal = np.frombuffer(
    observed[2][0], dtype=np.float64
).reshape(observed[2][1])
guard = np.frombuffer(
    observed[6][0], dtype=np.float64
).reshape(observed[6][1])
active = np.frombuffer(
    observed[7][0], dtype=np.bool_
).reshape(observed[7][1])
required = (
    abs(
        Fraction.from_float(-0.75) * Fraction.from_float(-0.2)
        - Fraction.from_float(float(nominal[0, 0]))
    )
    * Fraction.from_float(5.0)
)
if required != Fraction(5, 72057594037927936):
    raise AssertionError(("unexpected exact requirement", required))
if not active[0]:
    raise AssertionError("sound row was forged inactive")
if Fraction.from_float(float(guard[0])) < required:
    raise AssertionError(("unsound guard", guard[0], required))
"""
        )

    def test_post_factory_public_helper_substitution_is_ignored(self):
        dense_weight = np.ones((2, 3), dtype=np.float64)
        wrong_dense_support = dense_v51.prepare_dense_support_v51(
            dense_weight,
            np.zeros(3, dtype=np.float64),
            deadline=_end(),
        )
        dense_port = private.create_private_numeric_kernel(
            deadline=_end()
        )
        with mock.patch.object(
            dense_v51,
            "prepare_dense_support_v51",
            return_value=wrong_dense_support,
        ) as dense_forged:
            dense_locator = dense_port.admit_dense(
                weight=dense_weight,
                predecessor_max_abs=np.ones(3, dtype=np.float64),
            )
        dense_forged.assert_not_called()
        self.assertEqual(
            _decode_dense_result(
                dense_port.execute_dense(
                    dense_locator,
                    np.ones((1, 2), dtype=np.float64),
                )
            ).nominal.shape,
            (1, 3),
        )

        conv_weight = np.ones((1, 1, 1, 1), dtype=np.float64)
        conv_lb = -np.ones(4, dtype=np.float64)
        conv_ub = np.ones(4, dtype=np.float64)
        layer = _conv_layer(
            weight=conv_weight,
            input_shape=(1, 1, 4),
            output_shape=(1, 1, 4),
        )
        valid_plan = conv_v51.prepare_dense_conv_v51_plan(
            layer,
            frozen._Box(lb=conv_lb, ub=conv_ub),
            deadline=_replay_deadline(),
        )
        changed_offset = replace(
            valid_plan.offsets[0],
            channel_support_flat=np.zeros_like(
                valid_plan.offsets[0].channel_support_flat
            ),
        )
        forged_plans = (
            replace(valid_plan, offsets=()),
            replace(
                valid_plan,
                offsets=(changed_offset,) + valid_plan.offsets[1:],
            ),
        )
        for forged_plan in forged_plans:
            with self.subTest(offset_count=len(forged_plan.offsets)):
                conv_port = private.create_private_numeric_kernel(
                    deadline=_end()
                )
                with mock.patch.object(
                    conv_v51,
                    "prepare_dense_conv_v51_plan",
                    return_value=forged_plan,
                ) as conv_forged:
                    locator = _admit_conv(
                        conv_port,
                        weight=conv_weight,
                        lb=conv_lb,
                        ub=conv_ub,
                        input_shape=(1, 1, 4),
                        output_shape=(1, 1, 4),
                    )
                conv_forged.assert_not_called()
                self.assertIsNotNone(locator)

    def test_module_import_time_exact_forged_material_is_raw_bound(self):
        _run_isolated(
            r"""
import importlib
import sys
import time
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv
from act.back_end.hybridz_tf import query_dual_scalar_guard_v51 as dense

private_name = (
    "act.back_end.hybridz_tf."
    "query_dual_replay_v51b_private_kernel"
)
if private_name in sys.modules:
    raise AssertionError("private module imported before forged baseline")

dense_weight = np.ones((2, 3), dtype=np.float64)
wrong_dense = dense.prepare_dense_support_v51(
    dense_weight,
    np.zeros(3, dtype=np.float64),
    deadline=time.monotonic() + 30.0,
)

conv_weight = np.ones((1, 1, 1, 1), dtype=np.float64)
conv_lb = -np.ones(4, dtype=np.float64)
conv_ub = np.ones(4, dtype=np.float64)
layer = frozen._FrozenLayer(
    id=2,
    kind="CONV2D",
    preds=(1,),
    width=4,
    in_vars=(),
    out_vars=(),
    params=MappingProxyType(
        {
            "weight": conv_weight,
            "bias_channels": np.zeros(1, dtype=np.float64),
            "input_shape": (1, 1, 4),
            "output_shape": (1, 1, 4),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        }
    ),
)
valid_plan = conv.prepare_dense_conv_v51_plan(
    layer,
    frozen._Box(lb=conv_lb, ub=conv_ub),
    deadline=frozen._Deadline(end=time.monotonic() + 30.0),
)
changed_offset = replace(
    valid_plan.offsets[0],
    channel_support_flat=np.zeros_like(
        valid_plan.offsets[0].channel_support_flat
    ),
)
forged_conv_plans = (
    replace(valid_plan, offsets=()),
    replace(
        valid_plan,
        offsets=(changed_offset,) + valid_plan.offsets[1:],
    ),
)
calls = {"dense": 0, "conv": 0}
conv_position = [0]

def forged_dense(*args, **kwargs):
    calls["dense"] += 1
    return wrong_dense

def forged_conv(*args, **kwargs):
    calls["conv"] += 1
    result = forged_conv_plans[conv_position[0]]
    conv_position[0] += 1
    return result

dense.prepare_dense_support_v51 = forged_dense
conv.prepare_dense_conv_v51_plan = forged_conv
private = importlib.import_module(private_name)

dense_port = private.create_private_numeric_kernel(
    deadline=time.monotonic() + 30.0
)
try:
    dense_port.admit_dense(
        weight=dense_weight,
        predecessor_max_abs=np.ones(3, dtype=np.float64),
    )
except private.PrivateNumericKernelError as exc:
    if exc.code != "INVALID_ADMISSION_RESULT":
        raise AssertionError(("dense code", exc.code))
else:
    raise AssertionError("same-type Dense support entered private core")

for _ in forged_conv_plans:
    conv_port = private.create_private_numeric_kernel(
        deadline=time.monotonic() + 30.0
    )
    try:
        conv_port.admit_conv(
            layer_id=2,
            weight=conv_weight,
            predecessor_lb=conv_lb,
            predecessor_ub=conv_ub,
            input_shape=(1, 1, 4),
            output_shape=(1, 1, 4),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
    except private.PrivateNumericKernelError as exc:
        if exc.code != "INVALID_ADMISSION_RESULT":
            raise AssertionError(("conv code", exc.code))
    else:
        raise AssertionError("same-type Conv plan entered private core")

if calls != {"dense": 1, "conv": 2}:
    raise AssertionError(("unexpected forged helper calls", calls))
"""
        )

    def test_exact_ndarray_rejection_precedes_dynamic_reads(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        probe = np.ones((2, 2), dtype=np.float64).view(_ProbeArray)
        _ProbeArray.reads = 0
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "INVALID_ARRAY_TYPE"
        ):
            port.admit_dense(
                weight=probe,
                predecessor_max_abs=np.ones(2, dtype=np.float64),
            )
        self.assertEqual(_ProbeArray.reads, 0)

        locator, coefficient = self._dense_fixture(port)
        coefficient_probe = coefficient.view(_ProbeArray)
        _ProbeArray.reads = 0
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "INVALID_ARRAY_TYPE"
        ):
            port.execute_dense(locator, coefficient_probe)
        self.assertEqual(_ProbeArray.reads, 0)

        conv_weight = np.ones(
            (1, 1, 1, 1), dtype=np.float64
        ).view(_ProbeArray)
        _ProbeArray.reads = 0
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "INVALID_ARRAY_TYPE"
        ):
            _admit_conv(
                port,
                weight=conv_weight,
                lb=-np.ones(4, dtype=np.float64),
                ub=np.ones(4, dtype=np.float64),
                input_shape=(1, 1, 4),
                output_shape=(1, 1, 4),
            )
        self.assertEqual(_ProbeArray.reads, 0)

        conv_locator, conv_coefficient = self._conv_fixture(port)
        conv_probe = conv_coefficient.view(_ProbeArray)
        _ProbeArray.reads = 0
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "INVALID_ARRAY_TYPE"
        ):
            port.execute_conv(conv_locator, conv_probe)
        self.assertEqual(_ProbeArray.reads, 0)

    def test_active_trace_is_rejected_before_snapshot(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        weight = np.ones((2, 2), dtype=np.float64)
        attempts = []

        def trace_resize(frame, event, argument):
            del argument
            if (
                event == "line"
                and frame.f_code.co_name == "_snapshot_exact_array"
                and frame.f_locals.get("value") is weight
                and "exported" in frame.f_locals
                and not attempts
            ):
                try:
                    weight.resize((3, 3), refcheck=False)
                except ValueError:
                    attempts.append("BLOCKED")
                else:
                    attempts.append("RESIZED")
            return trace_resize

        sys.settrace(trace_resize)
        try:
            with self.assertRaises(
                private.PrivateNumericKernelError
            ) as captured:
                port.admit_dense(
                    weight=weight,
                    predecessor_max_abs=np.ones(
                        2, dtype=np.float64
                    ),
                )
        finally:
            sys.settrace(None)
        self.assertEqual(attempts, [])
        self.assertEqual(captured.exception.code, "NUMERIC_PLATFORM")
        self.assertEqual(port.stats()["dense_admissions"], 0)
        self.assertEqual(port.stats()["material_count"], 0)

    def test_outputs_are_bytes_backed_and_non_authoritative(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        dense_result = port.execute_dense(
            dense_locator, dense_coefficient
        )
        conv_result = port.execute_conv(conv_locator, conv_coefficient)
        for result, tag, names, decoder in (
            (
                dense_result,
                _DENSE_RESULT_TAG,
                _DENSE_RESULT_FIELDS,
                _decode_dense_result,
            ),
            (
                conv_result,
                _CONV_RESULT_TAG,
                _CONV_RESULT_FIELDS,
                _decode_conv_result,
            ),
        ):
            self.assertIs(type(result), tuple)
            self.assertEqual(result[0], tag)
            self.assertIs(result[1], False)
            self.assertEqual(len(result), 2 + len(names))
            self.assertFalse(hasattr(result, "receipt"))
            self.assertFalse(hasattr(result, "diagnostics"))
            decoded = decoder(result)
            self.assertFalse(decoded.proof_authority)
            for name, frame in zip(names, result[2:]):
                self.assertIs(type(frame), tuple)
                self.assertIs(type(frame[0]), bytes)
                self.assertIs(type(frame[1]), tuple)
                self.assertTrue(
                    all(type(extent) is int for extent in frame[1])
                )
                self.assertIs(type(frame[2]), bytes)
                self.assertNotIsInstance(frame[0], np.ndarray)
                self.assertTrue(
                    _bytes_backed(getattr(decoded, name)), name
                )

    def test_result_fields_and_builtin_type_cannot_be_substituted(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        pairs = (
            (
                port.execute_dense(dense_locator, dense_coefficient),
                _DENSE_RESULT_FIELDS,
            ),
            (
                port.execute_conv(conv_locator, conv_coefficient),
                _CONV_RESULT_FIELDS,
            ),
        )
        forged = np.zeros(1, dtype=np.float64)
        for result, names in pairs:
            for name in (*names, "proof_authority"):
                with self.subTest(tag=result[0], field=name):
                    with self.assertRaises(
                        (AttributeError, TypeError)
                    ):
                        object.__setattr__(result, name, forged)
                    with self.assertRaises(TypeError):
                        type.__setattr__(
                            type(result),
                            name,
                            property(lambda _: forged),
                        )
            with self.assertRaises(TypeError):
                result[2] = (b"", (), b"")
            self.assertIs(result[1], False)

    def test_result_copy_pickle_transplant_and_gc_are_pure_value_only(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        dense_result = port.execute_dense(
            dense_locator, dense_coefficient
        )
        conv_result = port.execute_conv(conv_locator, conv_coefficient)
        for result, decoder in (
            (dense_result, _decode_dense_result),
            (conv_result, _decode_conv_result),
        ):
            for copied in (
                copy.copy(result),
                copy.deepcopy(result),
                pickle.loads(pickle.dumps(result)),
            ):
                self.assertIs(type(copied), tuple)
                self.assertEqual(copied, result)
                self.assertIs(copied[1], False)
                self.assertFalse(decoder(copied).proof_authority)
        with self.assertRaises(AssertionError):
            _decode_dense_result(conv_result)
        with self.assertRaises(AssertionError):
            _decode_conv_result(dense_result)

        raw = dense_result[2][0]
        references_before = sys.getrefcount(raw)
        del dense_result
        gc.collect()
        self.assertLess(sys.getrefcount(raw), references_before)

    def test_strict_test_decoder_rejects_malformed_pure_values(self):
        class TupleSubclass(tuple):
            pass

        class BytesSubclass(bytes):
            pass

        class IntSubclass(int):
            pass

        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        dense_result = port.execute_dense(
            dense_locator, dense_coefficient
        )
        conv_result = port.execute_conv(conv_locator, conv_coefficient)

        def replace_outer(result, index, replacement):
            values = list(result)
            values[index] = replacement
            return tuple(values)

        def replace_frame(
            result,
            index,
            *,
            payload=None,
            shape=None,
            dtype_tag=None,
        ):
            original = result[index]
            return replace_outer(
                result,
                index,
                (
                    original[0] if payload is None else payload,
                    original[1] if shape is None else shape,
                    original[2] if dtype_tag is None else dtype_tag,
                ),
            )

        final = np.frombuffer(
            dense_result[6][0], dtype=np.float64
        ).copy()
        final[0] = np.nan
        inexact_final = np.frombuffer(
            dense_result[6][0], dtype=np.float64
        ).copy()
        positive_final = np.flatnonzero(inexact_final > 0.0)
        self.assertGreater(positive_final.size, 0)
        inexact_final[positive_final[0]] = np.nextafter(
            inexact_final[positive_final[0]], 0.0
        )
        negative = np.frombuffer(
            dense_result[3][0], dtype=np.float64
        ).copy()
        negative[0] = -1.0
        inactive = np.zeros(
            dense_result[7][1], dtype=np.bool_
        )
        fallback = np.ones(
            dense_result[8][1], dtype=np.bool_
        )
        bad_dense = (
            TupleSubclass(dense_result),
            dense_result[:-1],
            replace_outer(dense_result, 0, BytesSubclass(dense_result[0])),
            replace_outer(dense_result, 1, True),
            replace_outer(dense_result, 2, list(dense_result[2])),
            replace_frame(
                dense_result,
                2,
                payload=bytearray(dense_result[2][0]),
            ),
            replace_frame(
                dense_result,
                2,
                shape=(IntSubclass(dense_result[2][1][0]),)
                + dense_result[2][1][1:],
            ),
            replace_frame(
                dense_result,
                2,
                shape=(True,) + dense_result[2][1][1:],
            ),
            replace_frame(dense_result, 2, shape=(-1,)),
            replace_frame(dense_result, 2, shape=(2**1000,)),
            replace_frame(dense_result, 2, dtype_tag=b"not-native-f64"),
            replace_frame(
                dense_result,
                2,
                payload=dense_result[2][0][:-1],
            ),
            replace_frame(
                dense_result,
                6,
                payload=final.tobytes(),
            ),
            replace_frame(
                dense_result,
                6,
                payload=inexact_final.tobytes(),
            ),
            replace_frame(
                dense_result,
                3,
                payload=negative.tobytes(),
            ),
            replace_frame(
                replace_frame(
                    dense_result,
                    7,
                    payload=inactive.tobytes(),
                ),
                8,
                payload=fallback.tobytes(),
            ),
            replace_frame(
                dense_result,
                6,
                shape=(1, dense_result[6][1][0]),
            ),
        )
        for malformed in bad_dense:
            with self.subTest(dense_case=bad_dense.index(malformed)):
                with self.assertRaises(AssertionError):
                    _decode_dense_result(malformed)

        invalid_bool = bytes(
            [2] * len(conv_result[6][0])
        )
        all_false = np.zeros(
            conv_result[6][1], dtype=np.bool_
        ).tobytes()
        all_true = np.ones(
            conv_result[7][1], dtype=np.bool_
        ).tobytes()
        inexact_scalar = np.frombuffer(
            conv_result[3][0], dtype=np.float64
        ).copy()
        positive_scalar = np.flatnonzero(inexact_scalar > 0.0)
        self.assertGreater(positive_scalar.size, 0)
        inexact_scalar[positive_scalar[0]] = np.nextafter(
            inexact_scalar[positive_scalar[0]], math.inf
        )
        bad_conv = (
            replace_frame(
                conv_result,
                6,
                payload=invalid_bool,
            ),
            replace_frame(
                replace_frame(
                    conv_result,
                    6,
                    payload=all_false,
                ),
                7,
                payload=all_true,
            ),
            replace_frame(
                conv_result,
                3,
                payload=inexact_scalar.tobytes(),
            ),
        )
        for malformed in bad_conv:
            with self.assertRaises(AssertionError):
                _decode_conv_result(malformed)

    def test_result_signed_zero_and_subnormal_bits_round_trip(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        result = port.execute_dense(locator, coefficient)
        nominal = np.frombuffer(
            result[2][0], dtype=np.float64
        ).copy()
        nominal.flat[0] = np.copysign(0.0, -1.0)
        nominal.flat[1] = np.nextafter(0.0, 1.0)
        frames = list(result)
        frames[2] = (
            nominal.tobytes(),
            result[2][1],
            result[2][2],
        )
        decoded = _decode_dense_result(tuple(frames)).nominal
        self.assertEqual(
            decoded.view(np.uint64).flat[0],
            nominal.view(np.uint64).flat[0],
        )
        self.assertEqual(
            decoded.view(np.uint64).flat[1],
            nominal.view(np.uint64).flat[1],
        )

    def test_hot_execution_hits_no_public_validator_hash_or_receipt(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        patched = [
            (dense_v51, "check_v51_platform"),
            (dense_v51, "_validate_support"),
            (dense_v51, "_array_sha256"),
            (dense_v51, "_diagnostics"),
            (conv_v51, "_wide_platform"),
            (conv_v51, "_validate_plan"),
            (conv_v51, "_canonical_digest"),
            (conv_v51, "_manifest_body"),
            (frozen, "_array_digest"),
        ]
        mocks = []
        stack = []
        try:
            for module, name in patched:
                current = mock.Mock(
                    name=f"forbidden_{module.__name__}.{name}",
                    side_effect=AssertionError(name),
                )
                patcher = mock.patch.object(module, name, current)
                patcher.start()
                stack.append(patcher)
                mocks.append(current)
            dense_result = port.execute_dense(
                dense_locator, dense_coefficient
            )
            conv_result = port.execute_conv(
                conv_locator, conv_coefficient
            )
        finally:
            for patcher in reversed(stack):
                patcher.stop()
        self.assertEqual(
            _decode_dense_result(dense_result).nominal.shape, (2, 3)
        )
        self.assertEqual(
            _decode_conv_result(conv_result).coefficient.shape, (2, 4)
        )
        for current in mocks:
            current.assert_not_called()

    def test_captured_execute_ignores_port_class_substitution(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        captured_execute = port.execute_dense
        port_type = type(port)
        original_execute = port_type.execute_dense
        forged_calls = 0

        def forged_execute(*args, **kwargs):
            nonlocal forged_calls
            forged_calls += 1
            return {"forged": True}

        type.__setattr__(port_type, "execute_dense", forged_execute)
        try:
            result = captured_execute(locator, coefficient)
        finally:
            type.__setattr__(
                port_type, "execute_dense", original_execute
            )
        self.assertEqual(forged_calls, 0)
        self.assertIs(type(result), tuple)
        self.assertFalse(_decode_dense_result(result).proof_authority)

    def test_public_methods_use_closure_private_port_check(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        port_type = type(port)
        original = port_type._check_self
        calls = [0]

        def changed(*args, **kwargs):
            del args, kwargs
            calls[0] += 1
            raise AssertionError("dynamic port check was called")

        type.__setattr__(port_type, "_check_self", changed)
        try:
            result = port.execute_dense(locator, coefficient)
            stats = port.stats()
        finally:
            type.__setattr__(port_type, "_check_self", original)
        self.assertEqual(calls[0], 0)
        self.assertIs(type(result), tuple)
        self.assertEqual(stats["dense_executions"], 1)

    def test_forged_port_cannot_close_factory_state(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        forged = object.__new__(type(port))
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "PORT_MISMATCH"
        ):
            type(port).close(forged)
        self.assertEqual(port.stats()["material_count"], 0)

    def test_locator_copy_deepcopy_pickle_and_transplant_fail(self):
        first = private.create_private_numeric_kernel(deadline=_end())
        second = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(first)
        for operation in (
            lambda: copy.copy(first),
            lambda: copy.deepcopy(first),
            lambda: first.__reduce__(),
            lambda: copy.copy(locator),
            lambda: copy.deepcopy(locator),
            lambda: locator.__reduce__(),
        ):
            with self.assertRaisesRegex(
                private.PrivateNumericKernelError, "COPY_FORBIDDEN"
            ):
                operation()
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "LOCATOR_MISMATCH"
        ):
            second.execute_dense(locator, coefficient)
        conv_locator, conv_coefficient = self._conv_fixture(first)
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "LOCATOR_MISMATCH"
        ):
            first.execute_dense(conv_locator, conv_coefficient)

    def test_locator_gc_drops_private_material(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, _ = self._dense_fixture(port)
        reference = weakref.ref(locator)
        self.assertEqual(port.stats()["material_count"], 1)
        del locator
        for _ in range(3):
            gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(port.stats()["material_count"], 0)
        self.assertEqual(port.stats()["locator_count"], 0)

    def test_parallel_read_only_execution_is_deterministic(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        expected = port.execute_dense(locator, coefficient)

        def run_once(_):
            return port.execute_dense(locator, coefficient)

        with ThreadPoolExecutor(max_workers=4) as executor:
            observed = tuple(executor.map(run_once, range(16)))
        self.assertEqual(observed, (expected,) * 16)
        self.assertFalse(_decode_dense_result(expected).proof_authority)
        self.assertEqual(port.stats()["dense_executions"], 17)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_parent_port_and_locator_fail_closed_after_fork(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            # CPython 3.12 warns whenever a multithreaded parent forks.  The
            # child deliberately takes the PID rejection before touching any
            # inherited lock or BLAS state; that ordering is the test.
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded.*fork",
                category=DeprecationWarning,
            )
            pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                try:
                    port.execute_dense(locator, coefficient)
                except private.PrivateNumericKernelError as exc:
                    payload = exc.code.encode("ascii")
                else:
                    payload = b"NO_ERROR"
                os.write(write_fd, payload)
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            payload = os.read(read_fd, 128)
        finally:
            os.close(read_fd)
            _, status = os.waitpid(pid, 0)
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(payload, b"FORKED_PROCESS")
        parent = _decode_dense_result(
            port.execute_dense(locator, coefficient)
        )
        self.assertEqual(parent.nominal.shape, (2, 3))

    def test_expired_deadline_is_rejected_by_factory(self):
        for _ in range(2):
            with self.assertRaises(
                private.PrivateNumericKernelTimeout
            ):
                private.create_private_numeric_kernel(
                    deadline=float(time.monotonic() - 1.0)
                )

    def test_post_admission_expiry_rejects_live_locator(self):
        deadline = float(time.monotonic() + 0.25)
        port = private.create_private_numeric_kernel(
            deadline=deadline
        )
        locator, coefficient = self._dense_fixture(port)
        time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
        for operation in (
            lambda: port.execute_dense(locator, coefficient),
            lambda: port.stats(),
        ):
            with self.assertRaises(
                private.PrivateNumericKernelTimeout
            ):
                operation()

    def test_close_invalidates_port_and_material(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        locator, coefficient = self._dense_fixture(port)
        completed = port.execute_dense(locator, coefficient)
        port.close()
        port.close()
        self.assertIs(port.proof_authority, False)
        self.assertEqual(port.schema, private.SCHEMA)
        with self.assertRaisesRegex(
            private.PrivateNumericKernelError, "CLOSED"
        ):
            port.execute_dense(locator, coefficient)
        self.assertEqual(
            _decode_dense_result(completed).nominal.shape, (2, 3)
        )
        self.assertIs(completed[1], False)
        expiring = private.create_private_numeric_kernel(
            deadline=float(time.monotonic() + 0.05)
        )
        time.sleep(0.06)
        expiring.close()
        expiring.close()

    def test_stats_count_only_and_no_get_core(self):
        port = private.create_private_numeric_kernel(deadline=_end())
        dense_locator, dense_coefficient = self._dense_fixture(port)
        conv_locator, conv_coefficient = self._conv_fixture(port)
        port.execute_dense(dense_locator, dense_coefficient)
        port.execute_conv(conv_locator, conv_coefficient)
        stats = port.stats()
        self.assertEqual(
            dict(stats),
            {
                "material_count": 2,
                "locator_count": 2,
                "dense_materials": 1,
                "conv_materials": 1,
                "dense_admissions": 1,
                "conv_admissions": 1,
                "dense_executions": 1,
                "conv_executions": 1,
            },
        )
        self.assertIs(type(stats), MappingProxyType)
        self.assertFalse(any("hash" in key for key in stats))


if __name__ == "__main__":
    unittest.main()
