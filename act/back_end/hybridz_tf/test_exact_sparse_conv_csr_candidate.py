#!/usr/bin/env python3
# ===- test_exact_sparse_conv_csr_candidate.py - exact CONV gates --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------===#
"""Independent gates for the disconnected exact sparse-CONV candidate."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import gc
import json
import os
import statistics
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import exact_sparse_conv_csr_candidate as candidate
from act.back_end.hybridz_tf import operator_hz
from act.back_end.hybridz_tf.tf_cnn import (
    sparse_conv2d_matrix_from_layer as established_conv_builder,
)


Q = Fraction


def _pair(value):
    if isinstance(value, (int, np.integer)):
        return int(value), int(value)
    return int(value[0]), int(value[1])


def _layer(
    *,
    seed: int,
    batch: int,
    in_ch: int,
    out_ch: int,
    height: int,
    width: int,
    kernel: tuple[int, int],
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    weight=None,
    bias_marker="random",
    three_dimensional_shapes: bool = False,
):
    rng = np.random.default_rng(seed)
    kh, kw = kernel
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("test requested empty output geometry")
    if weight is None:
        weight = rng.standard_normal(
            (out_ch, in_ch // groups, kh, kw), dtype=np.float64
        )
        flat = weight.reshape(-1)
        flat[::7] = 0.0
        flat[3::19] = -0.0
    else:
        weight = np.asarray(weight).copy()
    if isinstance(bias_marker, str) and bias_marker == "random":
        bias = rng.standard_normal(out_ch, dtype=np.float64)
    elif bias_marker is None:
        bias = None
    else:
        bias = np.asarray(bias_marker).copy()
    input_shape = (batch, in_ch, height, width)
    output_shape = (batch, out_ch, out_h, out_w)
    if three_dimensional_shapes:
        if batch != 1:
            raise ValueError("three-dimensional test shapes require batch one")
        input_shape = input_shape[1:]
        output_shape = output_shape[1:]
    params = {
        "weight": weight,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    if bias is not None:
        params["bias"] = bias
    return SimpleNamespace(id=seed, params=params)


def _assert_bitwise_array_equal(case, left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    case.assertEqual(left.shape, right.shape)
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


def _assert_bitwise_csr_equal(case, left, right):
    case.assertTrue(sp.isspmatrix_csr(left))
    case.assertTrue(sp.isspmatrix_csr(right))
    case.assertTrue(left.has_canonical_format)
    case.assertTrue(right.has_canonical_format)
    case.assertEqual(left.shape, right.shape)
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    _assert_bitwise_array_equal(case, left.data, right.data)


def _canonical_product(left, right):
    result = (left @ right).tocsr()
    result.eliminate_zeros()
    result.sum_duplicates()
    result.sort_indices()
    return result


def _fraction_conv_oracle(layer):
    """Independent exact map keyed by flattened (output,input) coordinates."""

    weight = np.asarray(layer.params["weight"], dtype=np.float64)
    input_shape = tuple(layer.params["input_shape"])
    output_shape = tuple(layer.params["output_shape"])
    if len(input_shape) == 3:
        input_shape = (1, *input_shape)
        output_shape = (1, *output_shape)
    batch, in_ch, in_h, in_w = input_shape
    _, out_ch, out_h, out_w = output_shape
    groups = int(layer.params.get("groups", 1))
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    _, in_ch_per_group, kh, kw = weight.shape
    out_ch_per_group = out_ch // groups
    entries = {}
    for n in range(batch):
        for co in range(out_ch):
            ci_base = (co // out_ch_per_group) * in_ch_per_group
            for oh in range(out_h):
                for ow in range(out_w):
                    row = ((n * out_ch + co) * out_h + oh) * out_w + ow
                    for ci_local in range(in_ch_per_group):
                        ci = ci_base + ci_local
                        for rr in range(kh):
                            ih = oh * stride[0] - padding[0] + rr * dilation[0]
                            if not 0 <= ih < in_h:
                                continue
                            for cc in range(kw):
                                iw = ow * stride[1] - padding[1] + cc * dilation[1]
                                if not 0 <= iw < in_w:
                                    continue
                                value = float(weight[co, ci_local, rr, cc])
                                if value == 0.0:
                                    continue
                                col = ((n * in_ch + ci) * in_h + ih) * in_w + iw
                                if (row, col) in entries:
                                    raise AssertionError("ordinary CONV incidence duplicated")
                                entries[(row, col)] = Q.from_float(value)
    return entries


def _csr_fraction_entries(matrix):
    entries = {}
    for row in range(matrix.shape[0]):
        for position in range(matrix.indptr[row], matrix.indptr[row + 1]):
            entries[(row, int(matrix.indices[position]))] = Q.from_float(
                float(matrix.data[position])
            )
    return entries


def _row_local_matrix(
    rows: int,
    columns: int,
    *,
    rng: np.random.Generator,
    density: float = 1.0,
    permute: bool = False,
):
    if columns < rows:
        raise ValueError("test row-local mapping requires enough columns")
    mapped_rows = np.flatnonzero(rng.random(rows) < density)
    available = np.arange(columns, dtype=np.int32)
    if permute:
        rng.shuffle(available)
    generator_columns = available[: mapped_rows.size]
    scales = np.ldexp(
        rng.integers(-7, 8, size=mapped_rows.size).astype(np.float64), -3
    )
    scales[scales == 0.0] = 0.5
    matrix = sp.csr_matrix(
        (scales, (mapped_rows, generator_columns)),
        shape=(rows, columns),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix


class ExactSparseConvCSRCorrectnessTests(unittest.TestCase):
    def setUp(self):
        candidate.clear_exact_conv_topology_cache()

    def test_seeded_128_random_geometries_are_bit_exact_to_established(self):
        rng = np.random.default_rng(0xC1FA_0100)
        built = 0
        attempt = 0
        while built < 128:
            attempt += 1
            batch = int(rng.integers(1, 3))
            in_ch = int(rng.choice((1, 2, 3, 4, 6)))
            divisors = [value for value in range(1, in_ch + 1) if in_ch % value == 0]
            groups = int(rng.choice(divisors))
            out_ch = groups * int(rng.integers(1, 4))
            height = int(rng.integers(2, 8))
            width = int(rng.integers(2, 8))
            kernel = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
            stride = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
            padding = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
            dilation = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
            try:
                layer = _layer(
                    seed=attempt,
                    batch=batch,
                    in_ch=in_ch,
                    out_ch=out_ch,
                    height=height,
                    width=width,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias_marker=None if built % 5 == 0 else "random",
                    three_dimensional_shapes=bool(batch == 1 and built % 7 == 0),
                )
            except ValueError:
                continue
            expected_matrix, expected_bias = established_conv_builder(layer)
            actual_matrix, actual_bias = (
                candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
            )
            _assert_bitwise_csr_equal(self, actual_matrix, expected_matrix)
            _assert_bitwise_array_equal(self, actual_bias, expected_bias)
            built += 1

    def test_fraction_point_and_jacobian_oracle_with_groups_and_dilation(self):
        weight = np.asarray(
            [
                [[[1 / 2, -1 / 4], [0, 3 / 8]]],
                [[[-1 / 8, 1 / 2], [-3 / 4, -0.0]]],
            ],
            dtype=np.float64,
        )
        layer = _layer(
            seed=901,
            batch=2,
            in_ch=2,
            out_ch=2,
            height=5,
            width=4,
            kernel=(2, 2),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(2, 1),
            groups=2,
            weight=weight,
            bias_marker=np.asarray([1 / 8, -3 / 8], dtype=np.float64),
        )
        matrix, bias = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        exact_entries = _fraction_conv_oracle(layer)
        self.assertEqual(_csr_fraction_entries(matrix), exact_entries)

        # The affine Jacobian is exactly the coefficient map above.
        for coordinate, coefficient in exact_entries.items():
            self.assertEqual(
                Q.from_float(float(matrix[coordinate[0], coordinate[1]])),
                coefficient,
            )

        inputs = np.ldexp(
            ((np.arange(matrix.shape[1]) % 9) - 4).astype(np.float64), -3
        )
        observed = np.asarray(matrix @ inputs).reshape(-1) + bias
        bias_values = np.asarray(layer.params["bias"], dtype=np.float64)
        _, out_ch, out_h, out_w = tuple(layer.params["output_shape"])
        for row in range(matrix.shape[0]):
            channel = (row // (out_h * out_w)) % out_ch
            exact = Q.from_float(float(bias_values[channel]))
            for (entry_row, col), coefficient in exact_entries.items():
                if entry_row == row:
                    exact += coefficient * Q.from_float(float(inputs[col]))
            self.assertEqual(Q.from_float(float(observed[row])), exact)

    def test_zero_negative_zero_nan_and_stable_output_types(self):
        layer = _layer(
            seed=44,
            batch=2,
            in_ch=4,
            out_ch=6,
            height=4,
            width=5,
            kernel=(3, 2),
            padding=(1, 0),
            groups=2,
        )
        layer.params["weight"][...] = -0.0
        layer.params["bias"][0] = -0.0
        matrix, bias, receipt = (
            candidate.exact_sparse_conv2d_matrix_from_layer_candidate(
                layer, return_receipt=True
            )
        )
        self.assertEqual(matrix.nnz, 0)
        self.assertEqual(matrix.dtype, np.dtype(np.float64))
        self.assertEqual(matrix.indices.dtype, np.dtype(np.int32))
        self.assertEqual(matrix.indptr.dtype, np.dtype(np.int32))
        self.assertEqual(bias.dtype, np.dtype(np.float64))
        self.assertTrue(matrix.has_canonical_format)
        self.assertTrue(np.signbit(bias[0]))
        self.assertEqual(receipt.matrix_shape, matrix.shape)
        self.assertEqual(receipt.matrix_nnz, 0)
        self.assertEqual(receipt.construction_mode, "direct_canonical_csr_v2")
        self.assertFalse(receipt.coo_row_triplets_materialized)
        self.assertTrue(receipt.immutable_topology_cache)
        self.assertFalse(receipt.candidate_authoritative)
        self.assertFalse(receipt.proof_authority)
        self.assertFalse(receipt.verdict_authority)
        self.assertTrue(receipt.exact_affine_map)
        self.assertFalse(receipt.uses_dense_matrix)
        self.assertFalse(receipt.uses_torch_conv)
        self.assertFalse(receipt.uses_triangle_relaxation)
        self.assertFalse(receipt.uses_branch_and_bound)
        self.assertFalse(receipt.uses_backward_or_dual)
        self.assertFalse(receipt.uses_solver)

        layer.params["weight"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(candidate.ExactSparseConvCandidateError, "NaN"):
            candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        layer.params["weight"][0, 0, 0, 0] = 1.0
        layer.params["bias"][0] = np.nan
        with self.assertRaisesRegex(candidate.ExactSparseConvCandidateError, "NaN"):
            candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)

    def test_torch_parameter_snapshot_matches_established_without_torch_conv(self):
        import torch

        layer = _layer(
            seed=53,
            batch=1,
            in_ch=4,
            out_ch=4,
            height=6,
            width=5,
            kernel=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
            groups=2,
        )
        layer.params["weight"] = torch.tensor(
            layer.params["weight"], dtype=torch.float32, requires_grad=True
        )
        layer.params["bias"] = torch.tensor(
            layer.params["bias"], dtype=torch.float32, requires_grad=True
        )
        expected_matrix, expected_bias = established_conv_builder(layer)
        with mock.patch.object(
            torch.nn.functional,
            "conv2d",
            side_effect=AssertionError("torch conv forbidden"),
        ):
            actual_matrix, actual_bias = (
                candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
            )
        _assert_bitwise_csr_equal(self, actual_matrix, expected_matrix)
        _assert_bitwise_array_equal(self, actual_bias, expected_bias)

    def test_geometry_snapshot_is_cached_by_value_and_physically_readonly(self):
        kwargs = dict(
            input_shape=[1, 4, 9, 8],
            output_shape=[1, 6, 5, 7],
            kernel=[3, 2],
            stride=[2, 1],
            padding=[1, 0],
            dilation=[1, 1],
            groups=2,
        )
        first = candidate.get_exact_conv_spatial_topology(**kwargs)
        kwargs["input_shape"][2] = 999  # normalized cache key cannot alias caller state
        second = candidate.get_exact_conv_spatial_topology(
            input_shape=(1, 4, 9, 8),
            output_shape=(1, 6, 5, 7),
            kernel=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 1),
            groups=2,
        )
        self.assertIs(first, second)
        self.assertGreater(candidate.exact_conv_topology_cache_info().hits, 0)
        self.assertTrue(first.frames)
        for frame in first.frames:
            self.assertFalse(frame.output_spatial.flags.writeable)
            self.assertFalse(frame.input_spatial.flags.writeable)
            with self.assertRaises(ValueError):
                frame.output_spatial.flags.writeable = True
            with self.assertRaises(ValueError):
                frame.input_spatial[0] = 7
        for grid in (
            first.kernel_gather_by_output,
            first.input_spatial_by_output,
        ):
            self.assertFalse(grid.flags.writeable)
            with self.assertRaises(ValueError):
                grid.flags.writeable = True
        with self.assertRaises(FrozenInstanceError):
            first.groups = 1

    def test_weight_and_bias_mutation_A_B_A_never_reuses_stale_coefficients(self):
        layer = _layer(
            seed=71,
            batch=1,
            in_ch=2,
            out_ch=3,
            height=5,
            width=5,
            kernel=(3, 3),
            padding=1,
        )
        original_weight = layer.params["weight"].copy()
        original_bias = layer.params["bias"].copy()
        first_matrix, first_bias = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        first_data_snapshot = first_matrix.data.copy()
        first_bias_snapshot = first_bias.copy()
        layer.params["weight"] *= -2.0
        layer.params["bias"] += 3.0
        middle_matrix, middle_bias = (
            candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        )
        self.assertFalse(
            np.array_equal(
                first_matrix.data.view(np.uint64),
                middle_matrix.data.view(np.uint64),
            )
        )
        self.assertFalse(np.array_equal(first_bias.view(np.uint64), middle_bias.view(np.uint64)))
        np.testing.assert_array_equal(first_matrix.data, first_data_snapshot)
        np.testing.assert_array_equal(first_bias, first_bias_snapshot)
        layer.params["weight"][...] = original_weight
        layer.params["bias"][...] = original_bias
        final_matrix, final_bias = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        _assert_bitwise_csr_equal(self, first_matrix, final_matrix)
        _assert_bitwise_array_equal(self, first_bias, final_bias)

    def test_malformed_geometry_and_shapes_fail_closed(self):
        base = _layer(
            seed=81,
            batch=1,
            in_ch=4,
            out_ch=4,
            height=5,
            width=5,
            kernel=(3, 3),
            padding=1,
            groups=2,
        )
        mutations = (
            ("groups", 0),
            ("groups", 3),
            ("stride", 0),
            ("stride", 1 << 70),
            ("dilation", (1, 0)),
            ("padding", (-1, 0)),
            ("output_shape", (1, 4, 99, 5)),
        )
        for key, value in mutations:
            layer = SimpleNamespace(params=dict(base.params))
            layer.params[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(candidate.ExactSparseConvCandidateError):
                    candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)

    def test_signed_int64_spatial_overflow_and_int32_products_fail_closed(self):
        signed_max = int(np.iinfo(np.int64).max)
        hostile_geometries = (
            # This was previously admitted and reached rr*dilation = 2*INT64_MAX
            # inside NumPy topology construction.
            dict(
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 1, 1, 1),
                kernel=(3, 1),
                stride=1,
                padding=(signed_max, 0),
                dilation=(signed_max, 1),
                groups=1,
            ),
            # Exact Python output arithmetic gives height three, but 2*padding
            # is outside the promised signed-int64 construction domain.
            dict(
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 1, 3, 1),
                kernel=(1, 1),
                stride=(signed_max, 1),
                padding=(signed_max, 0),
                dilation=1,
                groups=1,
            ),
            # Geometry has scalar input/output but its cached KxK topology
            # product exceeds signed int32.
            dict(
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 1, 1, 1),
                kernel=(46_341, 46_341),
                stride=1,
                padding=(23_170, 23_170),
                dilation=1,
                groups=1,
            ),
        )
        for geometry in hostile_geometries:
            with self.subTest(geometry=geometry):
                with self.assertRaises(candidate.ExactSparseConvCandidateError):
                    candidate.get_exact_conv_spatial_topology(**geometry)

    def test_candidate_does_not_call_dense_or_torch_conv_primitive(self):
        layer = _layer(
            seed=90,
            batch=1,
            in_ch=2,
            out_ch=2,
            height=5,
            width=5,
            kernel=(3, 3),
            padding=1,
        )
        # Candidate does not import torch.  Patch the public primitives anyway
        # so a future accidental dependency fails this gate immediately.
        import torch
        import torch.nn.functional as torch_functional

        with (
            mock.patch.object(
                torch_functional,
                "conv2d",
                side_effect=AssertionError("torch conv forbidden"),
            ),
            mock.patch.object(
                torch,
                "conv2d",
                side_effect=AssertionError("torch conv forbidden"),
            ),
            mock.patch.object(
                sp.csr_matrix,
                "toarray",
                side_effect=AssertionError("dense matrix forbidden"),
            ),
            mock.patch.object(
                sp.csr_matrix,
                "todense",
                side_effect=AssertionError("dense matrix forbidden"),
            ),
        ):
            matrix, _ = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        self.assertGreater(matrix.nnz, 0)

    def test_accounted_workspace_has_linear_or_better_memory_slope(self):
        receipts = []
        for spatial in (8, 16, 24):
            layer = _layer(
                seed=100 + spatial,
                batch=1,
                in_ch=4,
                out_ch=8,
                height=spatial,
                width=spatial,
                kernel=(3, 3),
                padding=1,
            )
            _, _, receipt = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(
                layer, return_receipt=True
            )
            receipts.append(receipt)
            self.assertLessEqual(receipt.triplet_nbytes, 16 * receipt.matrix_nnz)
            self.assertGreaterEqual(receipt.peak_workspace_upper_bytes, receipt.triplet_nbytes)
        for left, right in zip(receipts, receipts[1:]):
            nnz_ratio = right.matrix_nnz / left.matrix_nnz
            memory_ratio = right.peak_workspace_upper_bytes / left.peak_workspace_upper_bytes
            self.assertLessEqual(memory_ratio, nnz_ratio * 1.08)


class RowLocalExactConvCandidateTests(unittest.TestCase):
    def test_forged_canonical_flags_invalid_indices_and_indptr_fail_closed(self):
        stable_ids = np.arange(3, dtype=np.int64)

        def forged_eye():
            matrix = sp.eye(3, format="csr", dtype=np.float64)
            # Populate both cached flags before hostile raw-array mutation.
            self.assertTrue(matrix.has_canonical_format)
            self.assertTrue(matrix.has_sorted_indices)
            return matrix

        bad_negative = forged_eye()
        bad_negative.indices[0] = -1
        bad_past_end = forged_eye()
        bad_past_end.indices[-1] = bad_past_end.shape[1]
        bad_indptr = forged_eye()
        bad_indptr.indptr[:] = np.asarray([0, 2, 1, 3], dtype=np.int32)
        duplicate = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2, 2, 2], dtype=np.int32),
            ),
            shape=(3, 3),
        )
        duplicate.has_canonical_format = True
        duplicate.has_sorted_indices = True
        unsorted = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([1, 0], dtype=np.int32),
                np.asarray([0, 2, 2, 2], dtype=np.int32),
            ),
            shape=(3, 3),
        )
        unsorted.has_canonical_format = True
        unsorted.has_sorted_indices = True
        for label, matrix in (
            ("negative", bad_negative),
            ("past_end", bad_past_end),
            ("indptr", bad_indptr),
            ("duplicate", duplicate),
            ("unsorted", unsorted),
        ):
            with self.subTest(label=label):
                with self.assertRaises(candidate.RowLocalGeneratorIneligible):
                    candidate.prepare_row_local_generator_plan(
                        matrix, stable_column_ids=stable_ids
                    )

    def test_exact_csr_array_contract_and_private_snapshot_close_TOCTOU(self):
        matrix = sp.eye(3, format="csr", dtype=np.float64)
        snapshot = candidate._require_canonical_finite_csr(
            matrix, name="audit"
        )
        matrix.data[:] = 7.0
        matrix.indices[:] = np.asarray([2, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(snapshot.data, np.ones(3))
        np.testing.assert_array_equal(snapshot.indices, np.arange(3))
        for array in (snapshot.data, snapshot.indices, snapshot.indptr):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flags.writeable = True

        negative_stride = sp.eye(1, format="csr", dtype=np.float64)
        negative_stride.data = np.asarray([1.0], dtype=np.float64)[::-1]
        self.assertTrue(negative_stride.data.flags.c_contiguous)
        self.assertLess(negative_stride.data.strides[0], 0)

        class ArraySubclass(np.ndarray):
            pass

        array_subclass = sp.eye(1, format="csr", dtype=np.float64)
        array_subclass.data = array_subclass.data.view(ArraySubclass)
        mismatched_indices = sp.eye(1, format="csr", dtype=np.float64)
        mismatched_indices.indptr = mismatched_indices.indptr.astype(np.int64)
        for label, hostile in (
            ("negative_stride", negative_stride),
            ("array_subclass", array_subclass),
            ("mismatched_index_dtypes", mismatched_indices),
        ):
            with self.subTest(label=label):
                with self.assertRaises(candidate.RowLocalGeneratorIneligible):
                    candidate._require_canonical_finite_csr(
                        hostile, name=label
                    )

    def test_seeded_standalone_and_fused_outputs_are_bit_exact(self):
        rng = np.random.default_rng(0xADD0_C0DE)
        for case_index in range(48):
            in_ch = int(rng.choice((1, 2, 4)))
            groups = int(rng.choice([g for g in range(1, in_ch + 1) if in_ch % g == 0]))
            out_ch = groups * int(rng.integers(1, 4))
            layer = _layer(
                seed=2000 + case_index,
                batch=int(rng.integers(1, 3)),
                in_ch=in_ch,
                out_ch=out_ch,
                height=int(rng.integers(3, 7)),
                width=int(rng.integers(3, 7)),
                kernel=(3, 3),
                stride=int(rng.integers(1, 3)),
                padding=1,
                dilation=1,
                groups=groups,
            )
            conv, _ = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
            extra_columns = int(rng.integers(0, 5))
            generators = _row_local_matrix(
                conv.shape[1],
                conv.shape[1] + extra_columns,
                rng=rng,
                density=float(rng.uniform(0.55, 1.0)),
                permute=bool(case_index % 3 == 0),
            )
            stable_ids = np.arange(generators.shape[1], dtype=np.int64) + 10_000
            expected = _canonical_product(conv, generators)
            standalone = candidate.apply_conv_to_row_local_generators_candidate(
                conv, generators, stable_column_ids=stable_ids
            )
            fused = candidate.fused_exact_sparse_conv_row_local_generators_candidate(
                layer, generators, stable_column_ids=stable_ids
            )
            _assert_bitwise_csr_equal(self, standalone, expected)
            _assert_bitwise_csr_equal(self, fused, expected)

    def test_row_local_plan_is_readonly_and_A_B_A_safe(self):
        generators = sp.eye(12, format="csr", dtype=np.float64)
        stable_ids = np.arange(12, dtype=np.int64) + 90
        first = candidate.prepare_row_local_generator_plan(
            generators, stable_column_ids=stable_ids
        )
        first_scale = first.row_scale.copy()
        generators.data *= -2.0
        middle = candidate.prepare_row_local_generator_plan(
            generators, stable_column_ids=stable_ids
        )
        self.assertNotEqual(first.digest, middle.digest)
        np.testing.assert_array_equal(first.row_scale, first_scale)
        generators.data *= -0.5
        final = candidate.prepare_row_local_generator_plan(
            generators, stable_column_ids=stable_ids
        )
        self.assertEqual(first.digest, final.digest)
        for array in (
            first.row_to_generator_column,
            first.row_scale,
            first.stable_column_ids,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flags.writeable = True

    def test_non_row_local_noninjective_nonfinite_and_unstable_ids_are_ineligible(self):
        two_in_one_row = sp.csr_matrix(
            ([1.0, 2.0], ([0, 0], [0, 1])), shape=(3, 3), dtype=np.float64
        )
        with self.assertRaisesRegex(candidate.RowLocalGeneratorIneligible, "not row-local"):
            candidate.prepare_row_local_generator_plan(
                two_in_one_row, stable_column_ids=np.arange(3, dtype=np.int64)
            )
        duplicate_column = sp.csr_matrix(
            ([1.0, 2.0], ([0, 1], [0, 0])), shape=(3, 3), dtype=np.float64
        )
        with self.assertRaisesRegex(candidate.RowLocalGeneratorIneligible, "not injective"):
            candidate.prepare_row_local_generator_plan(
                duplicate_column, stable_column_ids=np.arange(3, dtype=np.int64)
            )
        nan_matrix = sp.eye(3, format="csr", dtype=np.float64)
        nan_matrix.data[0] = np.nan
        with self.assertRaises(candidate.RowLocalGeneratorIneligible):
            candidate.prepare_row_local_generator_plan(
                nan_matrix, stable_column_ids=np.arange(3, dtype=np.int64)
            )
        for bad_ids in (
            np.asarray([1, 1, 2], dtype=np.int64),
            np.asarray([-1, 1, 2], dtype=np.int64),
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1, 2], dtype=np.int64),
        ):
            with self.assertRaises(candidate.RowLocalGeneratorIneligible):
                candidate.prepare_row_local_generator_plan(
                    sp.eye(3, format="csr", dtype=np.float64),
                    stable_column_ids=bad_ids,
                )
        oversized = sp.csr_matrix(
            (0, int(np.iinfo(np.int32).max) + 1), dtype=np.float64
        )
        with self.assertRaisesRegex(
            candidate.RowLocalGeneratorIneligible, "int32"
        ):
            candidate.prepare_row_local_generator_plan(
                oversized, stable_column_ids=np.empty(0, dtype=np.int64)
            )

    def test_underflow_to_zero_matches_eliminated_generic_product(self):
        conv = sp.csr_matrix(
            ([np.finfo(np.float64).tiny], ([0], [0])),
            shape=(1, 1),
            dtype=np.float64,
        )
        generators = sp.csr_matrix(
            ([np.finfo(np.float64).tiny], ([0], [0])),
            shape=(1, 1),
            dtype=np.float64,
        )
        expected = _canonical_product(conv, generators)
        actual = candidate.apply_conv_to_row_local_generators_candidate(
            conv,
            generators,
            stable_column_ids=np.asarray([7], dtype=np.int64),
        )
        _assert_bitwise_csr_equal(self, actual, expected)
        self.assertEqual(actual.nnz, 0)

        layer = _layer(
            seed=302,
            batch=1,
            in_ch=1,
            out_ch=1,
            height=1,
            width=1,
            kernel=(1, 1),
            weight=np.asarray([[[[np.finfo(np.float64).tiny]]]]),
            bias_marker=None,
        )
        fused = candidate.fused_exact_sparse_conv_row_local_generators_candidate(
            layer,
            generators,
            stable_column_ids=np.asarray([7], dtype=np.int64),
        )
        _assert_bitwise_csr_equal(self, fused, expected)

    def test_product_overflow_fails_closed_in_standalone_and_fused_paths(self):
        maximum = np.finfo(np.float64).max
        conv = sp.csr_matrix(([maximum], ([0], [0])), shape=(1, 1))
        generators = sp.csr_matrix(([2.0], ([0], [0])), shape=(1, 1))
        stable_ids = np.asarray([19], dtype=np.int64)
        with self.assertRaisesRegex(
            candidate.ExactSparseConvCandidateError, "overflow"
        ):
            candidate.apply_conv_to_row_local_generators_candidate(
                conv, generators, stable_column_ids=stable_ids
            )
        layer = _layer(
            seed=304,
            batch=1,
            in_ch=1,
            out_ch=1,
            height=1,
            width=1,
            kernel=(1, 1),
            weight=np.asarray([[[[maximum]]]]),
            bias_marker=None,
        )
        with self.assertRaisesRegex(
            candidate.ExactSparseConvCandidateError, "overflow"
        ):
            candidate.fused_exact_sparse_conv_row_local_generators_candidate(
                layer, generators, stable_column_ids=stable_ids
            )

    def test_receipts_are_exact_but_permanently_non_authoritative(self):
        layer = _layer(
            seed=303,
            batch=1,
            in_ch=3,
            out_ch=5,
            height=8,
            width=8,
            kernel=(3, 3),
            padding=1,
        )
        conv, _ = candidate.exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        generators = sp.eye(conv.shape[1], format="csr", dtype=np.float64)
        stable_ids = np.arange(generators.shape[1], dtype=np.int64) + 500
        standalone, standalone_receipt = (
            candidate.apply_conv_to_row_local_generators_candidate(
                conv,
                generators,
                stable_column_ids=stable_ids,
                return_receipt=True,
            )
        )
        fused, fused_receipt = (
            candidate.fused_exact_sparse_conv_row_local_generators_candidate(
                layer,
                generators,
                stable_column_ids=stable_ids,
                return_receipt=True,
            )
        )
        _assert_bitwise_csr_equal(self, standalone, fused)
        for receipt in (standalone_receipt, fused_receipt):
            self.assertTrue(receipt.exact_affine_map)
            self.assertTrue(receipt.row_local_eligible)
            self.assertFalse(receipt.candidate_authoritative)
            self.assertFalse(receipt.proof_authority)
            self.assertFalse(receipt.verdict_authority)
            self.assertFalse(receipt.uses_dense_matrix)
            self.assertFalse(receipt.uses_torch_conv)
            self.assertFalse(receipt.uses_triangle_relaxation)
            self.assertFalse(receipt.uses_branch_and_bound)
            self.assertFalse(receipt.uses_backward_or_dual)
            self.assertFalse(receipt.uses_solver)


class ExactSparseConvPerformanceGate(unittest.TestCase):
    def test_paired_single_thread_builder_relabel_and_fused_gates(self):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # pragma: no cover - project dependency in CI image
            self.skipTest("threadpoolctl is required for a controlled timing gate")

        # Synthetic same-topology layers only: no real benchmark and no net.
        legacy_layer = _layer(
            seed=404,
            batch=1,
            in_ch=12,
            out_ch=32,
            height=12,
            width=12,
            kernel=(3, 3),
            padding=1,
            groups=1,
        )
        legacy_layer.params["weight"].reshape(-1)[::11] = 0.0
        representative_layer = _layer(
            seed=405,
            batch=1,
            in_ch=16,
            out_ch=64,
            height=32,
            width=32,
            kernel=(3, 3),
            padding=1,
            groups=1,
        )
        representative_layer.params["weight"].reshape(-1)[::11] = 0.0
        input_size = 16 * 32 * 32
        scale_rng = np.random.default_rng(0xC05E_DA7A)
        scales = np.ldexp(
            scale_rng.integers(1, 9, size=input_size).astype(np.float64), -3
        )
        generators = sp.csr_matrix(
            (
                scales,
                (
                    np.arange(input_size, dtype=np.int32),
                    np.arange(input_size, dtype=np.int32),
                ),
            ),
            shape=(input_size, input_size),
            dtype=np.float64,
        )
        stable_ids = np.arange(input_size, dtype=np.int64) + 1_000_000

        def legacy_builder_call():
            return established_conv_builder(legacy_layer)[0]

        def candidate_legacy_shape_builder_call():
            return candidate.exact_sparse_conv2d_matrix_from_layer_candidate(
                legacy_layer
            )[0]

        def existing_vectorized_builder_call():
            return operator_hz._vectorized_sparse_conv2d_matrix_from_layer(
                representative_layer
            )[0]

        def candidate_direct_builder_call():
            return candidate.exact_sparse_conv2d_matrix_from_layer_candidate(
                representative_layer
            )[0]

        fixed_conv = candidate_direct_builder_call()

        def generic_relabel_call():
            return _canonical_product(fixed_conv, generators)

        def candidate_relabel_call():
            return candidate.apply_conv_to_row_local_generators_candidate(
                fixed_conv, generators, stable_column_ids=stable_ids
            )

        small_stable_ids = np.arange(12 * 12 * 12, dtype=np.int64) + 2_000_000
        small_generators = sp.eye(12 * 12 * 12, format="csr", dtype=np.float64)

        def legacy_generator_transform_call():
            matrix = established_conv_builder(legacy_layer)[0]
            return _canonical_product(matrix, small_generators)

        def fused_legacy_shape_generator_transform_call():
            return candidate.fused_exact_sparse_conv_row_local_generators_candidate(
                legacy_layer,
                small_generators,
                stable_column_ids=small_stable_ids,
            )

        def existing_vectorized_generator_transform_call():
            matrix = operator_hz._vectorized_sparse_conv2d_matrix_from_layer(
                representative_layer
            )[0]
            return _canonical_product(matrix, generators)

        def fused_representative_generator_transform_call():
            return candidate.fused_exact_sparse_conv_row_local_generators_candidate(
                representative_layer,
                generators,
                stable_column_ids=stable_ids,
            )

        def paired_ratios(left, right, repetitions=7):
            # left is established/generic; right is candidate.
            ratios = []
            for repetition in range(repetitions):
                gc.collect()
                if repetition % 2:
                    start = time.perf_counter_ns()
                    right_value = right()
                    right_ns = time.perf_counter_ns() - start
                    start = time.perf_counter_ns()
                    left_value = left()
                    left_ns = time.perf_counter_ns() - start
                else:
                    start = time.perf_counter_ns()
                    left_value = left()
                    left_ns = time.perf_counter_ns() - start
                    start = time.perf_counter_ns()
                    right_value = right()
                    right_ns = time.perf_counter_ns() - start
                self.assertEqual(left_value.shape, right_value.shape)
                self.assertEqual(left_value.nnz, right_value.nnz)
                ratios.append(left_ns / max(1, right_ns))
                del left_value, right_value
            return ratios

        previous_thread_env = {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        try:
            for key in previous_thread_env:
                os.environ[key] = "1"
            with threadpool_limits(limits=1):
                _assert_bitwise_csr_equal(
                    self,
                    existing_vectorized_builder_call(),
                    candidate_direct_builder_call(),
                )
                _assert_bitwise_csr_equal(
                    self,
                    existing_vectorized_generator_transform_call(),
                    fused_representative_generator_transform_call(),
                )
                # Explicit warmup precedes every measured family.
                for _ in range(2):
                    legacy_builder_call()
                    candidate_legacy_shape_builder_call()
                    existing_vectorized_builder_call()
                    candidate_direct_builder_call()
                    generic_relabel_call()
                    candidate_relabel_call()
                    legacy_generator_transform_call()
                    fused_legacy_shape_generator_transform_call()
                    existing_vectorized_generator_transform_call()
                    fused_representative_generator_transform_call()
                legacy_builder_ratios = paired_ratios(
                    legacy_builder_call, candidate_legacy_shape_builder_call
                )
                vectorized_builder_ratios = paired_ratios(
                    existing_vectorized_builder_call, candidate_direct_builder_call
                )
                relabel_ratios = paired_ratios(
                    generic_relabel_call, candidate_relabel_call
                )
                legacy_fused_ratios = paired_ratios(
                    legacy_generator_transform_call,
                    fused_legacy_shape_generator_transform_call,
                )
                vectorized_fused_ratios = paired_ratios(
                    existing_vectorized_generator_transform_call,
                    fused_representative_generator_transform_call,
                )
        finally:
            for key, value in previous_thread_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        legacy_builder_speedup = float(statistics.median(legacy_builder_ratios))
        vectorized_builder_speedup = float(
            statistics.median(vectorized_builder_ratios)
        )
        relabel_speedup = float(statistics.median(relabel_ratios))
        legacy_fused_speedup = float(statistics.median(legacy_fused_ratios))
        vectorized_fused_speedup = float(
            statistics.median(vectorized_fused_ratios)
        )
        report = {
            "schema": "act.exact_sparse_conv.synthetic_benchmark.v2",
            "real_or_large_run": False,
            "network_run": False,
            "center_bias_error_transform_timed": False,
            "threads": 1,
            "warmups": 2,
            "paired_repetitions": 7,
            "legacy_per_pixel_builder_median_paired_speedup": (
                legacy_builder_speedup
            ),
            "direct_csr_vs_existing_operator_vectorized_median_paired_speedup": (
                vectorized_builder_speedup
            ),
            "standalone_relabel_median_paired_speedup": relabel_speedup,
            "fused_generator_vs_legacy_builder_plus_spgemm_median_paired_speedup": (
                legacy_fused_speedup
            ),
            "fused_generator_vs_existing_vectorized_plus_spgemm_median_paired_speedup": (
                vectorized_fused_speedup
            ),
            "legacy_per_pixel_builder_promotable": legacy_builder_speedup >= 1.5,
            "direct_csr_vs_existing_operator_vectorized_promotable": (
                vectorized_builder_speedup >= 1.5
            ),
            "standalone_relabel_promotable": relabel_speedup >= 1.5,
            "fused_generator_vs_legacy_builder_plus_spgemm_promotable": (
                legacy_fused_speedup >= 1.5
            ),
            "fused_generator_vs_existing_vectorized_plus_spgemm_promotable": (
                vectorized_fused_speedup >= 1.5
            ),
            "candidate_authoritative": False,
            "proof_authority": False,
            "verdict_authority": False,
        }
        print("EXACT_SPARSE_CONV_SYNTHETIC_RECEIPT=" + json.dumps(report, sort_keys=True))
        self.assertGreaterEqual(legacy_builder_speedup, 1.5)
        self.assertGreaterEqual(vectorized_builder_speedup, 1.5)
        self.assertGreaterEqual(legacy_fused_speedup, 1.5)
        self.assertGreaterEqual(vectorized_fused_speedup, 1.5)
        self.assertEqual(report["standalone_relabel_promotable"], relabel_speedup >= 1.5)
        self.assertFalse(report["candidate_authoritative"])


if __name__ == "__main__":
    unittest.main()
