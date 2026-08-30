#!/usr/bin/env python3
"""Independent gates for the disconnected exact sparse-CONV affine core."""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
import gc
import inspect
import itertools
import json
import os
from statistics import median
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import exact_sparse_conv_affine_core as affine_core
from act.back_end.hybridz_tf import exact_sparse_conv_csr_candidate as csr_oracle
from act.back_end.hybridz_tf import operator_hz as operator_oracle

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional test-only dependency
    threadpool_limits = None


def _extent(
    input_extent: int,
    kernel_extent: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (
        input_extent
        + 2 * padding
        - dilation * (kernel_extent - 1)
        - 1
    ) // stride + 1


def _layer(
    weight: np.ndarray,
    *,
    input_shape: tuple[int, int, int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    bias: np.ndarray | None = None,
    layer_id: int = 17,
) -> SimpleNamespace:
    batch, _in_ch, in_h, in_w = input_shape
    out_ch, _in_per_group, kh, kw = weight.shape
    output_shape = (
        batch,
        out_ch,
        _extent(in_h, kh, stride[0], padding[0], dilation[0]),
        _extent(in_w, kw, stride[1], padding[1], dilation[1]),
    )
    return SimpleNamespace(
        id=layer_id,
        kind="CONV2D",
        params={
            "weight": weight,
            "bias": bias,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "data_format": "NCHW",
            "padding_mode": "zeros",
            "auto_pad": "NOTSET",
        },
    )


def _row_local_source(
    rng: np.random.Generator,
    rows: int,
    *,
    columns: int | None = None,
    density: float = 0.8,
    permuted: bool = True,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray]:
    columns = rows + 3 if columns is None else columns
    center = rng.normal(size=rows)
    error = np.abs(rng.normal(scale=1e-5, size=rows))
    mapped_count = int(round(rows * density))
    mapped_rows = np.sort(rng.choice(rows, size=mapped_count, replace=False))
    if permuted:
        mapped_columns = rng.choice(columns, size=mapped_count, replace=False)
    else:
        mapped_columns = mapped_rows.copy()
    scales = rng.normal(size=mapped_count)
    scales[scales == 0.0] = 1.0
    generators = sp.csr_matrix(
        (scales, (mapped_rows, mapped_columns)),
        shape=(rows, columns),
        dtype=np.float64,
    )
    generators.sort_indices()
    stable_ids = np.arange(101, 101 + columns, dtype=np.int64)
    return center, generators, error, stable_ids


def _assert_f64_bits_equal(
    testcase: unittest.TestCase,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    testcase.assertEqual(actual.shape, expected.shape, name)
    testcase.assertTrue(
        np.array_equal(actual.view(np.uint64), expected.view(np.uint64)),
        f"{name} differs in binary64 bits",
    )


def _assert_csr_bits_equal(
    testcase: unittest.TestCase,
    actual: sp.csr_matrix,
    expected: sp.csr_matrix,
) -> None:
    expected = expected.copy()
    expected.eliminate_zeros()
    expected.sort_indices()
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertTrue(np.array_equal(actual.indptr, expected.indptr))
    testcase.assertTrue(np.array_equal(actual.indices, expected.indices))
    _assert_f64_bits_equal(
        testcase, actual.data, expected.data, name="canonical G.data"
    )


def _operator_affine_oracle(
    layer: SimpleNamespace,
    center: np.ndarray,
    generators: sp.csr_matrix,
    error: np.ndarray,
    *,
    depth: int,
) -> tuple[sp.csr_matrix, np.ndarray, object]:
    matrix, bias = csr_oracle.exact_sparse_conv2d_matrix_from_layer_candidate(
        layer
    )
    expression = operator_oracle._AffineExpr(
        center, generators, error, affine_depth=depth
    )
    result = operator_oracle._OperatorHZBuilder._affine(
        None,
        expression,
        matrix,
        bias,
        layer_id=int(layer.id),
    )
    return matrix, bias, result


class ExactSparseConvAffineCoreTests(unittest.TestCase):
    maxDiff = None

    def _run_and_compare(
        self,
        layer: SimpleNamespace,
        center: np.ndarray,
        generators: sp.csr_matrix,
        error: np.ndarray,
        stable_ids: np.ndarray,
        *,
        depth: int = 0,
    ) -> tuple[
        affine_core.ExactRowLocalAffineSource,
        affine_core.ExactSparseConvLinearCore,
        affine_core.ExactSparseConvAffineResult,
        affine_core.ExactSparseConvAffineCoreReceipt,
    ]:
        source = affine_core.prepare_exact_row_local_affine_source(
            center,
            generators,
            error,
            stable_column_ids=stable_ids,
        )
        core, receipt = affine_core.apply_exact_sparse_conv_affine_core(
            layer,
            source,
            expected_stable_column_ids=stable_ids,
            return_receipt=True,
        )
        result = affine_core.finalize_exact_sparse_conv_affine_core(
            core, source_affine_depth=depth
        )
        matrix, bias, oracle = _operator_affine_oracle(
            layer, center, generators, error, depth=depth
        )

        _assert_f64_bits_equal(self, core.center_linear, matrix @ center,
                               name="center_linear")
        source_mass = operator_oracle._nonnegative_sum_upper(
            np.abs(center),
            operator_oracle._row_l1_upper(
                generators, name="test.source_G_l1"
            ),
            error,
            name="test.source_mass",
        )
        expected_mass = operator_oracle._positive_spmv_upper(
            abs(matrix).tocsr(), source_mass, name="test.transformed_mass"
        )
        expected_propagated = operator_oracle._positive_spmv_upper(
            abs(matrix).tocsr(), error, name="test.propagated_error"
        )
        _assert_f64_bits_equal(
            self, core.transformed_mass, expected_mass,
            name="transformed_mass",
        )
        _assert_f64_bits_equal(
            self, core.propagated_error, expected_propagated,
            name="propagated_error",
        )
        _assert_f64_bits_equal(
            self,
            core.fanin,
            np.diff(matrix.indptr).astype(np.float64),
            name="fanin",
        )
        _assert_f64_bits_equal(self, core.bias, bias, name="bias snapshot")
        _assert_f64_bits_equal(self, result.center, oracle.c, name="final center")
        _assert_csr_bits_equal(self, result.generators, oracle.G)
        _assert_f64_bits_equal(self, result.error, oracle.err, name="final error")
        canonical_oracle_G = oracle.G.copy()
        canonical_oracle_G.eliminate_zeros()
        canonical_oracle_G.sort_indices()
        factor_point = np.linspace(
            -0.875,
            0.625,
            int(result.generators.shape[1]),
            dtype=np.float64,
        )
        actual_value = np.asarray(
            result.center + result.generators @ factor_point,
            dtype=np.float64,
        )
        expected_value = np.asarray(
            oracle.c + canonical_oracle_G @ factor_point,
            dtype=np.float64,
        )
        _assert_f64_bits_equal(
            self,
            actual_value,
            expected_value,
            name="final canonical affine value",
        )
        self.assertEqual(result.affine_depth, depth + 1)
        self.assertTrue(np.array_equal(result.stable_column_ids, stable_ids))
        return source, core, result, receipt

    def test_random_complete_operator_oracle_is_bit_exact(self) -> None:
        rng = np.random.default_rng(0xC0FFEE)
        completed = 0
        coefficient_count = 0
        for case in range(96):
            groups = int(rng.choice((1, 2)))
            in_per_group = int(rng.integers(1, 4))
            out_per_group = int(rng.integers(1, 4))
            in_ch = groups * in_per_group
            out_ch = groups * out_per_group
            in_h = int(rng.integers(2, 8))
            in_w = int(rng.integers(2, 8))
            kh = int(rng.integers(1, min(4, in_h + 1)))
            kw = int(rng.integers(1, min(4, in_w + 1)))
            stride = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
            dilation = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
            padding = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
            out_h = _extent(in_h, kh, stride[0], padding[0], dilation[0])
            out_w = _extent(in_w, kw, stride[1], padding[1], dilation[1])
            if min(out_h, out_w) <= 0:
                continue
            batch = 2 if case >= 88 else 1
            input_shape = (batch, in_ch, in_h, in_w)
            weight = rng.normal(size=(out_ch, in_per_group, kh, kw))
            weight[rng.random(weight.shape) < 0.3] = 0.0
            if case % 8 == 0:
                weight.reshape(-1)[0] = -0.0
            bias = None if case % 3 == 0 else rng.normal(size=out_ch)
            layer = _layer(
                weight,
                input_shape=input_shape,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                layer_id=case,
            )
            rows = int(np.prod(input_shape))
            center, generators, error, stable_ids = _row_local_source(
                rng,
                rows,
                density=float(rng.uniform(0.0, 1.0)),
                permuted=bool(case % 2),
            )
            _source, _core, result, _receipt = self._run_and_compare(
                layer,
                center,
                generators,
                error,
                stable_ids,
                depth=case % 4,
            )
            completed += 1
            coefficient_count += int(result.generators.nnz)
        self.assertGreaterEqual(completed, 80)
        self.assertGreater(coefficient_count, 1_000)

    def test_cancellation_subnormal_negative_zero_and_no_bias(self) -> None:
        least = np.nextafter(np.float64(0.0), np.float64(1.0))
        weight = np.asarray(
            [[[[1.0, -1.0, -0.0, 0.5]]]], dtype=np.float64
        )
        layer = _layer(
            weight,
            input_shape=(1, 1, 1, 4),
            bias=None,
            layer_id=202,
        )
        center = np.asarray([1.0, 1.0, -0.0, least], dtype=np.float64)
        generators = sp.csr_matrix(
            (
                np.asarray([1.0, -1.0, least, least], dtype=np.float64),
                (np.arange(4), np.arange(4)),
            ),
            shape=(4, 4),
        )
        error = np.asarray([0.0, least, 0.0, least], dtype=np.float64)
        stable_ids = np.asarray([9, 7, 5, 3], dtype=np.int64)
        _source, _core, result, _receipt = self._run_and_compare(
            layer, center, generators, error, stable_ids
        )
        self.assertEqual(result.generators.nnz, 2)
        self.assertGreater(result.error[0], 0.0)

    def test_fraction_point_jacobian_and_envelope_gate(self) -> None:
        weight = np.asarray([[[[0.1, -0.7]]]], dtype=np.float64)
        bias = np.asarray([0.2], dtype=np.float64)
        layer = _layer(
            weight,
            input_shape=(1, 1, 1, 2),
            bias=bias,
            layer_id=303,
        )
        center = np.asarray([0.3, -0.4], dtype=np.float64)
        scales = np.asarray([0.6, -0.8], dtype=np.float64)
        generators = sp.csr_matrix(
            (scales, (np.arange(2), np.arange(2))), shape=(2, 2)
        )
        source_error = np.asarray([2.0**-48, 2.0**-49], dtype=np.float64)
        stable_ids = np.asarray([500, 700], dtype=np.int64)
        _source, _core, result, _receipt = self._run_and_compare(
            layer, center, generators, source_error, stable_ids
        )

        wf = [Fraction.from_float(float(value)) for value in weight.reshape(-1)]
        cf = [Fraction.from_float(float(value)) for value in center]
        sf = [Fraction.from_float(float(value)) for value in scales]
        ef = [Fraction.from_float(float(value)) for value in source_error]
        bf = Fraction.from_float(float(bias[0]))
        stored_center = Fraction.from_float(float(result.center[0]))
        stored_g = [
            Fraction.from_float(float(result.generators[0, column]))
            for column in range(2)
        ]
        stored_error = Fraction.from_float(float(result.error[0]))

        # Independent binary64 Jacobian gate: each row-local coefficient has
        # exactly one rounded multiplication contributor.
        expected_products = np.asarray(
            [weight.reshape(-1)[i] * scales[i] for i in range(2)],
            dtype=np.float64,
        )
        _assert_f64_bits_equal(
            self,
            result.generators.toarray().reshape(-1),
            expected_products,
            name="row-local Jacobian products",
        )

        def exact_value(
            xi: tuple[Fraction, Fraction],
            delta: tuple[Fraction, Fraction],
        ) -> Fraction:
            return bf + sum(
                wf[i] * (cf[i] + sf[i] * xi[i] + delta[i])
                for i in range(2)
            )

        def stored_value(xi: tuple[Fraction, Fraction]) -> Fraction:
            return stored_center + sum(stored_g[i] * xi[i] for i in range(2))

        # All cube/error corners establish an independent exact-Fraction
        # envelope gate.  A non-corner rational point checks the same claim.
        maximum_gap = Fraction(0)
        for xi_signs in itertools.product((-1, 1), repeat=2):
            xi = tuple(Fraction(value) for value in xi_signs)
            for error_signs in itertools.product((-1, 1), repeat=2):
                delta = tuple(
                    Fraction(error_signs[i]) * ef[i] for i in range(2)
                )
                maximum_gap = max(
                    maximum_gap,
                    abs(exact_value(xi, delta) - stored_value(xi)),
                )
        rational_xi = (Fraction(1, 3), Fraction(-2, 5))
        rational_delta = (ef[0] / 7, -ef[1] / 11)
        point_gap = abs(
            exact_value(rational_xi, rational_delta)
            - stored_value(rational_xi)
        )
        self.assertGreaterEqual(stored_error, maximum_gap)
        self.assertGreaterEqual(stored_error, point_gap)

    def test_private_snapshots_survive_weight_source_and_bias_aba(self) -> None:
        rng = np.random.default_rng(44)
        weight = rng.normal(size=(2, 2, 3, 3))
        bias = rng.normal(size=2)
        layer = _layer(
            weight,
            input_shape=(1, 2, 4, 4),
            padding=(1, 1),
            bias=bias,
            layer_id=404,
        )
        center, generators, error, stable_ids = _row_local_source(rng, 32)
        source = affine_core.prepare_exact_row_local_affine_source(
            center, generators, error, stable_column_ids=stable_ids
        )
        source_bits = (
            source.center.tobytes(),
            source.generators.data.tobytes(),
            source.error.tobytes(),
            source.stable_column_ids.tobytes(),
        )
        center[:] = 19.0
        generators.data[:] = -23.0
        error[:] = 29.0
        stable_ids[:] = 31
        self.assertEqual(
            source_bits,
            (
                source.center.tobytes(),
                source.generators.data.tobytes(),
                source.error.tobytes(),
                source.stable_column_ids.tobytes(),
            ),
        )
        live_ids = np.frombuffer(source_bits[3], dtype=np.int64).copy()
        core = affine_core.apply_exact_sparse_conv_affine_core(
            layer, source, expected_stable_column_ids=live_ids
        )
        result_before = affine_core.finalize_exact_sparse_conv_affine_core(core)
        core_bits = (
            core.center_linear.tobytes(),
            core.generators.data.tobytes(),
            core.bias.tobytes(),
            core.stable_column_ids.tobytes(),
        )
        weight[:] = -37.0
        bias[:] = 41.0
        result_after = affine_core.finalize_exact_sparse_conv_affine_core(core)
        self.assertEqual(
            core_bits,
            (
                core.center_linear.tobytes(),
                core.generators.data.tobytes(),
                core.bias.tobytes(),
                core.stable_column_ids.tobytes(),
            ),
        )
        _assert_f64_bits_equal(
            self, result_after.center, result_before.center, name="ABA center"
        )
        _assert_csr_bits_equal(
            self, result_after.generators, result_before.generators
        )
        _assert_f64_bits_equal(
            self, result_after.error, result_before.error, name="ABA error"
        )
        for array in (
            source.center,
            source.generators.data,
            source.generators.indices,
            source.generators.indptr,
            core.center_linear,
            core.generators.data,
            core.bias,
            result_after.center,
            result_after.error,
        ):
            self.assertFalse(array.flags.writeable)
            owner = array
            while isinstance(getattr(owner, "base", None), np.ndarray):
                owner = owner.base
            self.assertIsInstance(owner.base, bytes)

    def test_row_local_not_applicable_is_only_structural_eligibility(self) -> None:
        center = np.zeros(2, dtype=np.float64)
        error = np.zeros(2, dtype=np.float64)
        ids = np.asarray([11, 12], dtype=np.int64)
        nonlocal_g = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0]),
                (np.asarray([0, 0]), np.asarray([0, 1])),
            ),
            shape=(2, 2),
        )
        with self.assertRaises(affine_core.RowLocalNotApplicable):
            affine_core.prepare_exact_row_local_affine_source(
                center, nonlocal_g, error, stable_column_ids=ids
            )
        noninjective = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0]),
                (np.asarray([0, 1]), np.asarray([0, 0])),
            ),
            shape=(2, 2),
        )
        with self.assertRaises(affine_core.RowLocalNotApplicable):
            affine_core.prepare_exact_row_local_affine_source(
                center, noninjective, error, stable_column_ids=ids
            )

        malformed = sp.csr_matrix(
            (np.asarray([0.0]), np.asarray([0]), np.asarray([0, 1, 1])),
            shape=(2, 2),
        )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.prepare_exact_row_local_affine_source(
                center, malformed, error, stable_column_ids=ids
            )

    def test_raw_csr_validation_ignores_forged_flags_and_index_dtype(
        self,
    ) -> None:
        center = np.zeros(2, dtype=np.float64)
        error = np.zeros(2, dtype=np.float64)
        ids = np.asarray([4, 8], dtype=np.int64)
        valid_i64 = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0, 1, 2], dtype=np.int64),
            ),
            shape=(2, 2),
        )
        # Force real int64 storage even when SciPy downcasts small domains.
        valid_i64.indices = valid_i64.indices.astype(np.int64)
        valid_i64.indptr = valid_i64.indptr.astype(np.int64)
        source = affine_core.prepare_exact_row_local_affine_source(
            center, valid_i64, error, stable_column_ids=ids
        )
        self.assertEqual(source.generators.indices.dtype, np.dtype(np.int32))

        out_of_range = sp.csr_matrix(np.eye(2, dtype=np.float64))
        self.assertTrue(out_of_range.has_canonical_format)
        out_of_range.indices[0] = 2
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.prepare_exact_row_local_affine_source(
                center, out_of_range, error, stable_column_ids=ids
            )

        duplicate = sp.csr_matrix(np.eye(2, dtype=np.float64))
        self.assertTrue(duplicate.has_canonical_format)
        duplicate.indices[1] = 0
        duplicate.indptr[:] = np.asarray([0, 2, 2], dtype=duplicate.indptr.dtype)
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.prepare_exact_row_local_affine_source(
                center, duplicate, error, stable_column_ids=ids
            )

    def test_identifier_binding_and_integrity_tampering_fail_closed(self) -> None:
        weight = np.ones((1, 1, 1, 1), dtype=np.float64)
        layer = _layer(weight, input_shape=(1, 1, 1, 2), layer_id=505)
        center = np.asarray([0.25, -0.5], dtype=np.float64)
        generators = sp.eye(2, format="csr", dtype=np.float64)
        error = np.zeros(2, dtype=np.float64)
        ids = np.asarray([10, 20], dtype=np.int64)
        source = affine_core.prepare_exact_row_local_affine_source(
            center, generators, error, stable_column_ids=ids
        )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.apply_exact_sparse_conv_affine_core(
                layer,
                source,
                expected_stable_column_ids=np.asarray([10, 21]),
            )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.prepare_exact_row_local_affine_source(
                center,
                generators,
                error,
                stable_column_ids=np.asarray([10, 10]),
            )
        core = affine_core.apply_exact_sparse_conv_affine_core(
            layer, source, expected_stable_column_ids=ids
        )
        object.__setattr__(core, "source_digest", "0" * 64)
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.finalize_exact_sparse_conv_affine_core(core)

    def test_product_underflow_is_exactly_eliminated_and_overflow_fails(self) -> None:
        least = np.nextafter(np.float64(0.0), np.float64(1.0))
        underflow_layer = _layer(
            np.asarray([[[[0.5]]]], dtype=np.float64),
            input_shape=(1, 1, 1, 1),
            layer_id=606,
        )
        underflow_g = sp.csr_matrix(
            (np.asarray([least]), (np.asarray([0]), np.asarray([0]))),
            shape=(1, 1),
        )
        _source, _core, result, _receipt = self._run_and_compare(
            underflow_layer,
            np.asarray([0.0]),
            underflow_g,
            np.asarray([0.0]),
            np.asarray([1], dtype=np.int64),
        )
        self.assertEqual(result.generators.nnz, 0)
        self.assertGreater(result.error[0], 0.0)

        large_scale = np.finfo(np.float64).max / 16.0
        overflow_g = sp.csr_matrix(
            (np.asarray([large_scale]), (np.asarray([0]), np.asarray([0]))),
            shape=(1, 1),
        )
        overflow_source = affine_core.prepare_exact_row_local_affine_source(
            np.asarray([0.0]),
            overflow_g,
            np.asarray([0.0]),
            stable_column_ids=np.asarray([1], dtype=np.int64),
        )
        overflow_layer = _layer(
            np.asarray([[[[32.0]]]], dtype=np.float64),
            input_shape=(1, 1, 1, 1),
            layer_id=607,
        )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.apply_exact_sparse_conv_affine_core(
                overflow_layer,
                overflow_source,
                expected_stable_column_ids=np.asarray([1], dtype=np.int64),
            )

    def test_geometry_checked_products_and_domains_fail_before_allocation(self) -> None:
        center = np.asarray([0.0])
        generators = sp.csr_matrix((1, 1), dtype=np.float64)
        source = affine_core.prepare_exact_row_local_affine_source(
            center,
            generators,
            np.asarray([0.0]),
            stable_column_ids=np.asarray([1], dtype=np.int64),
        )
        huge = SimpleNamespace(
            params={
                "weight": np.ones((1, 1, 1, 1), dtype=np.float64),
                "bias": None,
                "input_shape": (1, 1, 1, 2**31),
                "output_shape": (1, 1, 1, 2**31),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
            }
        )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.apply_exact_sparse_conv_affine_core(
                huge,
                source,
                expected_stable_column_ids=np.asarray([1], dtype=np.int64),
            )
        wrapped = SimpleNamespace(
            params={
                "weight": np.ones((1, 1, 1, 1), dtype=np.float64),
                "bias": None,
                "input_shape": (1, 1, 1, 1),
                "output_shape": (1, 1, 1, 1),
                "stride": 1,
                "padding": np.iinfo(np.int64).max,
                "dilation": 1,
                "groups": 1,
            }
        )
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreError):
            affine_core.apply_exact_sparse_conv_affine_core(
                wrapped,
                source,
                expected_stable_column_ids=np.asarray([1], dtype=np.int64),
            )

    def test_eligible_execution_calls_no_W_builder_or_sparse_matmul(self) -> None:
        rng = np.random.default_rng(77)
        layer = _layer(
            rng.normal(size=(3, 2, 3, 3)),
            input_shape=(1, 2, 5, 5),
            padding=(1, 1),
            bias=rng.normal(size=3),
            layer_id=707,
        )
        center, generators, error, ids = _row_local_source(
            rng, 50, permuted=False
        )
        source = affine_core.prepare_exact_row_local_affine_source(
            center, generators, error, stable_column_ids=ids
        )
        sparse_base = type(sp.csr_matrix((1, 1)))
        with (
            mock.patch.object(
                csr_oracle,
                "exact_sparse_conv2d_matrix_from_layer_candidate",
                side_effect=AssertionError("W builder was called"),
            ),
            mock.patch.object(
                operator_oracle,
                "_sparse_conv2d_matrix_from_layer_strict_with_mode",
                side_effect=AssertionError("production W builder was called"),
            ),
            mock.patch.object(
                sparse_base,
                "__matmul__",
                side_effect=AssertionError("sparse matmul/SpGEMM was called"),
            ),
        ):
            result = affine_core.exact_sparse_conv_affine_from_layer(
                layer, source, expected_stable_column_ids=ids
            )
        self.assertEqual(result.size, 75)
        apply_source = inspect.getsource(
            affine_core.apply_exact_sparse_conv_affine_core
        )
        self.assertNotIn("exact_sparse_conv2d_matrix_from_layer", apply_source)
        self.assertNotIn(" @ ", apply_source)

    def test_receipt_is_non_authoritative_and_forbidden_modes_absent(
        self,
    ) -> None:
        layer = _layer(
            np.ones((1, 1, 1, 1), dtype=np.float64),
            input_shape=(1, 1, 1, 1),
            layer_id=808,
        )
        source = affine_core.prepare_exact_row_local_affine_source(
            np.asarray([1.0]),
            sp.eye(1, format="csr", dtype=np.float64),
            np.asarray([0.0]),
            stable_column_ids=np.asarray([99], dtype=np.int64),
        )
        _core, receipt = affine_core.apply_exact_sparse_conv_affine_core(
            layer,
            source,
            expected_stable_column_ids=np.asarray([99], dtype=np.int64),
            return_receipt=True,
        )
        self.assertFalse(receipt.linear_primitive_authoritative)
        self.assertFalse(receipt.property_proof_authority)
        self.assertFalse(receipt.verdict_authority)
        self.assertFalse(receipt.full_conv_operator_materialized)
        self.assertFalse(receipt.transient_operator_sparse_matrix_materialized)
        self.assertTrue(receipt.uses_compiled_csr_vector_reduction)
        self.assertEqual(receipt.maximum_coefficient_slab_entries, 1_000_000)
        self.assertFalse(receipt.uses_spgemm)
        self.assertFalse(receipt.uses_triangle_relaxation)
        self.assertFalse(receipt.uses_branch_and_bound)
        self.assertFalse(receipt.uses_backward_or_dual)
        self.assertFalse(receipt.uses_solver)
        self.assertEqual(receipt.geometry_weight_traversals, 1)
        self.assertNotIn("proof", receipt.construction_mode)
        with self.assertRaises(affine_core.ExactSparseConvAffineCoreInternalError):
            affine_core.ExactSparseConvAffineCoreReceipt(
                source_digest="0" * 64,
                geometry_digest="1" * 64,
                weight_digest="2" * 64,
                core_digest="3" * 64,
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 1, 1, 1),
                source_generator_nnz=0,
                output_generator_nnz=0,
                operator_nnz=0,
                linear_primitive_authoritative=True,
            )

    def test_paired_complete_core_performance_gate_is_honest(self) -> None:
        rng = np.random.default_rng(909)
        input_shape = (1, 8, 12, 12)
        weight = rng.normal(size=(16, 8, 3, 3))
        weight[rng.random(weight.shape) < 0.15] = 0.0
        layer = _layer(
            weight,
            input_shape=input_shape,
            padding=(1, 1),
            bias=rng.normal(size=16),
            layer_id=909,
        )
        rows = int(np.prod(input_shape))
        center, generators, error, ids = _row_local_source(
            rng,
            rows,
            columns=rows,
            density=1.0,
            permuted=False,
        )
        source = affine_core.prepare_exact_row_local_affine_source(
            center, generators, error, stable_column_ids=ids
        )
        expression = operator_oracle._AffineExpr(
            center, generators, error, affine_depth=0
        )

        def direct_core() -> object:
            linear = affine_core.apply_exact_sparse_conv_affine_core(
                layer, source, expected_stable_column_ids=ids
            )
            return affine_core.finalize_exact_sparse_conv_affine_core(linear)

        def direct_W_generic_affine() -> object:
            matrix, bias = (
                csr_oracle.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
            )
            return operator_oracle._OperatorHZBuilder._affine(
                None,
                expression,
                matrix,
                bias,
                layer_id=int(layer.id),
            )

        context = (
            threadpool_limits(limits=1)
            if threadpool_limits is not None
            else nullcontext()
        )
        with context:
            direct_core()
            direct_W_generic_affine()
            fast_times: list[float] = []
            baseline_times: list[float] = []
            for iteration in range(7):
                gc.collect()
                ordered = (
                    (
                        direct_core,
                        fast_times,
                        direct_W_generic_affine,
                        baseline_times,
                    )
                    if iteration % 2
                    else (
                        direct_W_generic_affine,
                        baseline_times,
                        direct_core,
                        fast_times,
                    )
                )
                first, first_times, second, second_times = ordered
                started = time.perf_counter()
                first()
                first_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                second()
                second_times.append(time.perf_counter() - started)
        fast_median = median(fast_times)
        baseline_median = median(baseline_times)
        speedup = baseline_median / fast_median
        promotable = bool(speedup >= 1.50)
        report = {
            "schema": "exact_sparse_conv_affine_core_paired_gate_v1",
            "topology": {
                "input_shape": input_shape,
                "weight_shape": tuple(int(value) for value in weight.shape),
            },
            "threads": 1,
            "pairs": 7,
            "direct_core_median_seconds": fast_median,
            "direct_W_generic_median_seconds": baseline_median,
            "speedup": speedup,
            "threshold": 1.50,
            "promotable": promotable,
        }
        print(json.dumps(report, sort_keys=True))
        self.assertEqual(promotable, speedup >= 1.50)
        # A run below threshold is a valid, explicit NO-PROMOTION result.  The
        # primitive receipt intentionally carries no benchmark-derived claim.
        self.assertNotIn("performance_promotable", vars(
            affine_core.ExactSparseConvAffineCoreReceipt
        ))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    unittest.main(verbosity=2)
