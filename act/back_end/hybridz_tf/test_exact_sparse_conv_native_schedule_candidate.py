#!/usr/bin/env python3
"""Offline gates for the disconnected compact native CONV schedule."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import gc
import inspect
import json
from statistics import median
import threading
import time
import tracemalloc
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import exact_sparse_conv_csr_candidate as csr_oracle
from act.back_end.hybridz_tf import exact_sparse_conv_native_schedule_candidate as native
from act.back_end.hybridz_tf import operator_hz as operator_oracle

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None


def _extent(i: int, k: int, s: int, p: int, d: int) -> int:
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def _layer(
    weight: np.ndarray,
    *,
    input_shape: tuple[int, int, int, int],
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    bias: np.ndarray | None = None,
    layer_id: int = 17,
) -> SimpleNamespace:
    batch, _in_ch, in_h, in_w = input_shape
    out_ch, _in_per, kh, kw = weight.shape
    return SimpleNamespace(
        id=layer_id,
        kind="CONV2D",
        params={
            "weight": weight,
            "bias": bias,
            "input_shape": input_shape,
            "output_shape": (
                batch,
                out_ch,
                _extent(in_h, kh, stride[0], padding[0], dilation[0]),
                _extent(in_w, kw, stride[1], padding[1], dilation[1]),
            ),
            "padding": padding,
            "stride": stride,
            "dilation": dilation,
            "groups": groups,
            "data_format": "NCHW",
            "padding_mode": "zeros",
            "auto_pad": "NOTSET",
        },
    )


def _fixture(seed: int = 909):
    rng = np.random.default_rng(seed)
    input_shape = (1, 8, 12, 12)
    weight = rng.normal(size=(16, 8, 3, 3))
    weight[rng.random(weight.shape) < 0.15] = 0.0
    layer = _layer(
        weight,
        input_shape=input_shape,
        padding=(1, 1),
        bias=rng.normal(size=16),
        layer_id=seed,
    )
    rows = int(np.prod(input_shape))
    center = rng.normal(size=rows)
    error = np.abs(rng.normal(scale=1e-5, size=rows))
    scales = rng.normal(size=rows)
    scales[scales == 0.0] = 1.0
    generators = sp.diags(scales, format="csr", dtype=np.float64)
    stable_ids = np.arange(101, 101 + rows, dtype=np.int64)
    return layer, center, generators, error, stable_ids


def _bits_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return bool(
        actual.shape == expected.shape
        and np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    )


class ExactSparseConvCompactNativeScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        environment = native.native_kernel_environment()
        if not environment["available"]:
            raise unittest.SkipTest(str(environment))

    def _compare_oracle(
        self,
        layer: SimpleNamespace,
        center: np.ndarray,
        generators: sp.csr_matrix,
        error: np.ndarray,
        ids: np.ndarray,
        *,
        depth: int = 0,
    ) -> tuple[native.ExactConvNativeResult, native.ExactConvNativeReceipt]:
        owner = object()
        schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        result, receipt = native.apply_exact_conv_native_schedule(
            owner,
            schedule,
            source,
            expected_stable_column_ids=ids,
            source_affine_depth=depth,
            return_receipt=True,
        )
        matrix, bias = csr_oracle.exact_sparse_conv2d_matrix_from_layer_candidate(
            layer
        )
        oracle = operator_oracle._OperatorHZBuilder._affine(
            None,
            operator_oracle._AffineExpr(center, generators, error, depth),
            matrix,
            bias,
            layer_id=int(layer.id),
        )
        oracle.G.sort_indices()
        self.assertTrue(_bits_equal(result.center, oracle.c))
        self.assertTrue(np.array_equal(result.generators.indptr, oracle.G.indptr))
        self.assertTrue(np.array_equal(result.generators.indices, oracle.G.indices))
        self.assertTrue(_bits_equal(result.generators.data, oracle.G.data))
        self.assertTrue(_bits_equal(result.error, oracle.err))
        abs_matrix = abs(matrix).tocsr()
        source_mass = operator_oracle._nonnegative_sum_upper(
            np.abs(center),
            operator_oracle._row_l1_upper(generators, name="test.G_l1"),
            error,
            name="test.source_mass",
        )
        expected_mass = operator_oracle._positive_spmv_upper(
            abs_matrix, source_mass, name="test.mass"
        )
        expected_propagated = operator_oracle._positive_spmv_upper(
            abs_matrix, error, name="test.propagated"
        )
        self.assertTrue(_bits_equal(result.transformed_mass, expected_mass))
        self.assertTrue(_bits_equal(result.propagated_error, expected_propagated))
        self.assertEqual(result.affine_depth, depth + 1)
        return result, receipt

    def test_fixed_909_is_bitwise_exact(self) -> None:
        result, _receipt = self._compare_oracle(*_fixture())
        self.assertEqual(result.generators.indptr.dtype, np.dtype(np.int32))
        self.assertEqual(result.generators.indices.dtype, np.dtype(np.int32))
        for value in (
            result.center,
            result.error,
            result.transformed_mass,
            result.propagated_error,
            result.generators.data,
            result.generators.indices,
            result.generators.indptr,
        ):
            self.assertFalse(value.flags.writeable)

    def test_seeded_random_groups_stride_padding_dilation_are_bitwise_exact(self) -> None:
        rng = np.random.default_rng(20260809)
        for case in range(24):
            batch = 1 + case % 2
            groups = (1, 2, 1)[case % 3]
            in_per = 1 + case % 3
            in_ch = groups * in_per
            out_per = 1 + (case // 2) % 3
            out_ch = groups * out_per
            in_h = 4 + case % 4
            in_w = 4 + (case // 3) % 4
            kh = 1 + case % 3
            kw = 1 + (case // 4) % 3
            dilation = (1 + case % 2, 1 + (case // 5) % 2)
            padding = (case % 2, (case // 2) % 2)
            stride = (1 + (case // 3) % 2, 1 + (case // 7) % 2)
            if _extent(in_h, kh, stride[0], padding[0], dilation[0]) <= 0:
                continue
            if _extent(in_w, kw, stride[1], padding[1], dilation[1]) <= 0:
                continue
            weight = rng.normal(size=(out_ch, in_per, kh, kw))
            weight[rng.random(weight.shape) < 0.3] = 0.0
            shape = (batch, in_ch, in_h, in_w)
            layer = _layer(
                weight,
                input_shape=shape,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=None if case % 4 == 0 else rng.normal(size=out_ch),
                layer_id=500 + case,
            )
            rows = int(np.prod(shape))
            columns = rows + 7
            live_rows = np.sort(
                rng.choice(rows, size=max(1, int(rows * 0.8)), replace=False)
            )
            live_columns = np.sort(
                rng.choice(columns, size=live_rows.size, replace=False)
            )
            scales = rng.normal(size=live_rows.size)
            scales[scales == 0.0] = 1.0
            generators = sp.csr_matrix(
                (scales, (live_rows, live_columns)), shape=(rows, columns)
            )
            center = rng.normal(size=rows)
            error = np.abs(rng.normal(scale=1e-9, size=rows))
            ids = np.arange(9000, 9000 + columns, dtype=np.int64)
            with self.subTest(case=case):
                self._compare_oracle(
                    layer, center, generators, error, ids, depth=case
                )

    def test_subnormal_cancellation_and_zero_activity_match(self) -> None:
        subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
        weight = np.asarray([[[[subnormal, -subnormal]]]], dtype=np.float64)
        layer = _layer(weight, input_shape=(1, 1, 2, 3), bias=None, layer_id=701)
        center = np.asarray([1.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        generators = sp.diags(np.arange(1.0, 7.0), format="csr")
        error = np.asarray([subnormal, 0.0, subnormal, 0.0, 0.0, 0.0])
        ids = np.arange(6, dtype=np.int64)
        self._compare_oracle(layer, center, generators, error, ids)

    def test_no_expanded_W_builder_or_schedule_is_used(self) -> None:
        layer, center, generators, error, ids = _fixture(912)
        owner = object()
        with mock.patch.object(
            csr_oracle,
            "exact_sparse_conv2d_matrix_from_layer_candidate",
            side_effect=AssertionError("expanded W builder forbidden"),
        ):
            schedule = native.prepare_exact_conv_native_schedule(owner, layer)
            source = native.prepare_exact_monotone_row_local_source(
                center, generators, error, stable_column_ids=ids
            )
            native.apply_exact_conv_native_schedule(
                owner,
                schedule,
                source,
                expected_stable_column_ids=ids,
            )
        self.assertEqual(schedule.weight.size, 16 * 8 * 3 * 3)
        self.assertEqual(schedule.geometry.size, 17)
        self.assertFalse(hasattr(schedule, "indptr"))
        self.assertFalse(hasattr(schedule, "input_indices"))
        module_source = inspect.getsource(native)
        self.assertNotIn("exact_sparse_conv2d_matrix_from_layer_candidate", module_source)

    def test_ineligible_mapping_owner_ids_and_bad_geometry_fail_closed(self) -> None:
        layer, center, generators, error, ids = _fixture(913)
        owner = object()
        schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                object(), schedule, source, expected_stable_column_ids=ids
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                owner, schedule, source, expected_stable_column_ids=ids[::-1]
            )
        with self.assertRaises(native.MonotoneRowLocalNotApplicable):
            native.prepare_exact_monotone_row_local_source(
                center,
                generators[:, ::-1].tocsr(),
                error,
                stable_column_ids=ids,
            )
        broken = SimpleNamespace(params=dict(layer.params))
        broken.params["output_shape"] = (1, 16, 11, 12)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.prepare_exact_conv_native_schedule(owner, broken)

    def test_factory_registry_rejects_forged_mass_digest_and_rebinding(self) -> None:
        layer, center, generators, error, ids = _fixture(916)
        owner = object()
        schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )

        manual_schedule = native.ExactConvNativeSchedule(
            geometry=schedule.geometry,
            weight=schedule.weight,
            bias_channels=schedule.bias_channels,
            digest=schedule.digest,
            _owner=owner,
        )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                owner,
                manual_schedule,
                source,
                expected_stable_column_ids=ids,
            )

        forged_mass = native._private_array(
            source.source_mass + 1.0, dtype=np.dtype(np.float64)
        )
        unsigned_source = native.ExactMonotoneRowLocalSource(
            center=source.center,
            source_mass=forged_mass,
            error=source.error,
            row_to_generator_column=source.row_to_generator_column,
            row_scale=source.row_scale,
            stable_column_ids=source.stable_column_ids,
            generator_columns=source.generator_columns,
            digest="",
        )
        self_signed_source = replace(
            unsigned_source, digest=native._source_digest(unsigned_source)
        )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                owner,
                schedule,
                self_signed_source,
                expected_stable_column_ids=ids,
            )

        rebound_schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        object.__setattr__(rebound_schedule, "digest", "0" * 64)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                owner,
                rebound_schedule,
                source,
                expected_stable_column_ids=ids,
            )

        rebound_source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        same_bytes_new_identity = native._private_array(
            rebound_source.source_mass, dtype=np.dtype(np.float64)
        )
        object.__setattr__(
            rebound_source, "source_mass", same_bytes_new_identity
        )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.apply_exact_conv_native_schedule(
                owner,
                schedule,
                rebound_source,
                expected_stable_column_ids=ids,
            )

    def test_checked_integer_shape_allocation_and_projection_hostiles(self) -> None:
        projection = {
            "weight_entries": 1,
            "bias_entries": 1,
            "conv_count": 1,
        }
        for name in tuple(projection):
            hostile = dict(projection)
            hostile[name] = np.int64(hostile[name])
            with self.subTest(numpy_integer=name):
                with self.assertRaises(native.ExactConvNativeScheduleError):
                    native.project_compact_schedule_cache_bytes(**hostile)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.project_compact_schedule_cache_bytes(
                weight_entries=True, bias_entries=1, conv_count=1
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.project_compact_schedule_cache_bytes(
                weight_entries=np.iinfo(np.intp).max // 8 + 1,
                bias_entries=0,
                conv_count=0,
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.project_compact_schedule_cache_bytes(
                weight_entries=2**63,
                bias_entries=0,
                conv_count=0,
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.project_compact_schedule_cache_bytes(
                weight_entries=0,
                bias_entries=0,
                conv_count=0,
                source_cap_bytes=np.int64(1),
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.project_compact_schedule_cache_bytes(
                weight_entries=0,
                bias_entries=0,
                conv_count=0,
                source_cap_bytes=2**63,
            )

        base_params = {
            "weight": np.ones((1, 1, 1, 1), dtype=np.float64),
            "bias": None,
            "padding": (0, 0),
            "stride": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "data_format": "NCHW",
            "padding_mode": "zeros",
            "auto_pad": "NOTSET",
        }

        def hostile_layer(**changes: object) -> SimpleNamespace:
            params = dict(base_params)
            params.update(changes)
            return SimpleNamespace(id=917, kind="CONV2D", params=params)

        owner = object()
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.prepare_exact_conv_native_schedule(
                owner,
                hostile_layer(
                    input_shape=(True, 1, 1, 1),
                    output_shape=(1, 1, 1, 1),
                ),
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.prepare_exact_conv_native_schedule(
                owner,
                hostile_layer(
                    input_shape=(1, 1, 2**63 - 1, 1),
                    output_shape=(1, 1, 2**63 - 1, 1),
                    padding=(1, 0),
                ),
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.prepare_exact_conv_native_schedule(
                owner,
                hostile_layer(
                    input_shape=(1, 1, 46_341, 46_341),
                    output_shape=(1, 1, 46_341, 46_341),
                ),
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            native.prepare_exact_conv_native_schedule(
                owner,
                hostile_layer(
                    weight=np.ones((1, 2, 1, 1), dtype=np.float64),
                    input_shape=(1, 2**63 - 1, 1, 1),
                    output_shape=(1, 1, 1, 1),
                    groups=2**63 - 1,
                ),
            )

    def test_reusable_registry_is_concurrent_and_weakly_reclaimed(self) -> None:
        layer, center, generators, error, ids = _fixture(918)
        owner = object()
        schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        expected = native.apply_exact_conv_native_schedule(
            owner, schedule, source, expected_stable_column_ids=ids
        )
        before_workers = native._registry_sizes_for_tests()

        def apply_shared(_iteration: int) -> native.ExactConvNativeResult:
            return native.apply_exact_conv_native_schedule(
                owner, schedule, source, expected_stable_column_ids=ids
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(apply_shared, range(24)))
        self.assertEqual(native._registry_sizes_for_tests(), before_workers)
        for result in results:
            self.assertTrue(_bits_equal(result.center, expected.center))
            self.assertTrue(_bits_equal(result.error, expected.error))
            self.assertTrue(
                np.array_equal(result.generators.indptr, expected.generators.indptr)
            )
            self.assertTrue(
                np.array_equal(result.generators.indices, expected.generators.indices)
            )
            self.assertTrue(
                _bits_equal(result.generators.data, expected.generators.data)
            )

        before_temporary = native._registry_sizes_for_tests()
        temporary_owner = object()
        temporary_schedule = native.prepare_exact_conv_native_schedule(
            temporary_owner, layer
        )
        temporary_source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        temporary_schedule_identity = id(temporary_schedule)
        temporary_source_identity = id(temporary_source)
        schedule_reference = weakref.ref(temporary_schedule)
        source_reference = weakref.ref(temporary_source)
        self.assertEqual(
            native._registry_sizes_for_tests(),
            (before_temporary[0] + 1, before_temporary[1] + 1),
        )
        del temporary_schedule, temporary_source
        gc.collect()
        self.assertIsNone(schedule_reference())
        self.assertIsNone(source_reference())
        with native._REGISTRY_LOCK:
            self.assertNotIn(
                temporary_schedule_identity, native._SCHEDULE_REGISTRY
            )
            self.assertNotIn(temporary_source_identity, native._SOURCE_REGISTRY)
        after_temporary = native._registry_sizes_for_tests()
        self.assertLessEqual(after_temporary[0], before_temporary[0])
        self.assertLessEqual(after_temporary[1], before_temporary[1])

    def test_lock_issued_capture_survives_thread_rebind_and_aba(self) -> None:
        for restore_before_release in (False, True):
            with self.subTest(aba=restore_before_release):
                layer, center, generators, error, ids = _fixture(
                    920 + int(restore_before_release)
                )
                owner = object()
                schedule = native.prepare_exact_conv_native_schedule(owner, layer)
                source = native.prepare_exact_monotone_row_local_source(
                    center, generators, error, stable_column_ids=ids
                )
                expected, expected_receipt = (
                    native.apply_exact_conv_native_schedule(
                        owner,
                        schedule,
                        source,
                        expected_stable_column_ids=ids,
                        source_affine_depth=3,
                        return_receipt=True,
                    )
                )

                original_schedule = {
                    name: getattr(schedule, name)
                    for name in (
                        "geometry",
                        "weight",
                        "bias_channels",
                        "digest",
                    )
                }
                original_source = {
                    name: getattr(source, name)
                    for name in (
                        "center",
                        "source_mass",
                        "stable_column_ids",
                        "generator_columns",
                        "digest",
                    )
                }
                forged_geometry = np.array(schedule.geometry, copy=True)
                forged_geometry[4] = 1
                forged_schedule = {
                    "geometry": native._private_array(
                        forged_geometry, dtype=np.dtype(np.int64)
                    ),
                    "weight": native._private_array(
                        np.full_like(schedule.weight, 17.0),
                        dtype=np.dtype(np.float64),
                    ),
                    "bias_channels": native._private_array(
                        np.full_like(schedule.bias_channels, -23.0),
                        dtype=np.dtype(np.float64),
                    ),
                    "digest": "f" * 64,
                }
                forged_source = {
                    "center": native._private_array(
                        np.full_like(source.center, 31.0),
                        dtype=np.dtype(np.float64),
                    ),
                    "source_mass": native._private_array(
                        source.source_mass + 41.0,
                        dtype=np.dtype(np.float64),
                    ),
                    "stable_column_ids": native._private_array(
                        source.stable_column_ids[::-1],
                        dtype=np.dtype(np.int64),
                    ),
                    "generator_columns": source.generator_columns - 1,
                    "digest": "e" * 64,
                }

                admitted = threading.Event()
                release = threading.Event()
                original_admit = native._admit_registered_factory_objects

                def admit_then_wait(*args: object):
                    captured = original_admit(*args)
                    admitted.set()
                    if not release.wait(timeout=10.0):
                        raise AssertionError("capture barrier timed out")
                    return captured

                def restore_public_objects() -> None:
                    for name, value in original_schedule.items():
                        object.__setattr__(schedule, name, value)
                    for name, value in original_source.items():
                        object.__setattr__(source, name, value)

                with mock.patch.object(
                    native,
                    "_admit_registered_factory_objects",
                    side_effect=admit_then_wait,
                ):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            native.apply_exact_conv_native_schedule,
                            owner,
                            schedule,
                            source,
                            expected_stable_column_ids=ids,
                            source_affine_depth=3,
                            return_receipt=True,
                        )
                        self.assertTrue(admitted.wait(timeout=10.0))
                        for name, value in forged_schedule.items():
                            object.__setattr__(schedule, name, value)
                        for name, value in forged_source.items():
                            object.__setattr__(source, name, value)
                        if restore_before_release:
                            restore_public_objects()
                        release.set()
                        actual, receipt = future.result(timeout=10.0)
                if not restore_before_release:
                    restore_public_objects()

                self.assertTrue(_bits_equal(actual.center, expected.center))
                self.assertTrue(_bits_equal(actual.error, expected.error))
                self.assertTrue(
                    _bits_equal(actual.transformed_mass, expected.transformed_mass)
                )
                self.assertTrue(
                    _bits_equal(actual.propagated_error, expected.propagated_error)
                )
                self.assertTrue(
                    np.array_equal(
                        actual.generators.indptr, expected.generators.indptr
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        actual.generators.indices, expected.generators.indices
                    )
                )
                self.assertTrue(
                    _bits_equal(actual.generators.data, expected.generators.data)
                )
                self.assertTrue(
                    np.array_equal(
                        actual.stable_column_ids, expected.stable_column_ids
                    )
                )
                self.assertEqual(actual.schedule_digest, expected.schedule_digest)
                self.assertEqual(actual.source_digest, expected.source_digest)
                self.assertEqual(receipt, expected_receipt)

    def test_receipt_is_permanently_non_authoritative(self) -> None:
        _result, receipt = self._compare_oracle(*_fixture(914))
        self.assertIs(type(receipt), native.ExactConvNativeReceipt)
        self.assertEqual(
            set(vars(receipt)),
            set(native.ExactConvNativeReceipt.__dataclass_fields__),
        )
        self.assertFalse(receipt.expanded_conv_operator_materialized)
        self.assertFalse(receipt.expanded_conv_operator_cached)
        self.assertFalse(receipt.expanded_operator_schedule_cached)
        self.assertFalse(receipt.linear_primitive_authoritative)
        self.assertFalse(receipt.property_proof_authority)
        self.assertFalse(receipt.verdict_authority)
        self.assertFalse(receipt.production_promotion_claim)
        self.assertFalse(receipt.uses_spgemm)
        self.assertFalse(receipt.uses_triangle_relaxation)
        self.assertFalse(receipt.uses_branch_and_bound)
        self.assertFalse(receipt.uses_backward_or_dual)
        self.assertFalse(receipt.uses_solver)
        self.assertTrue(receipt.uses_runtime_compiler)
        self.assertFalse(receipt.cffi_is_declared_act_dependency)
        self.assertFalse(
            receipt.runtime_compile_excluded_from_schedule_source_model_cold_gate
        )
        self.assertFalse(receipt.first_response_native_load_included)
        self.assertFalse(receipt.first_response_gate_passes)
        self.assertTrue(receipt.factory_registry_authenticated)
        self.assertTrue(receipt.factory_registry_reusable)
        self.assertTrue(receipt.source_snapshot_reused_without_second_copy)
        self.assertTrue(receipt.output_ownership_transferred_without_freeze_copy)
        self.assertFalse(receipt.public_output_rebind_protected)
        self.assertFalse(receipt.c89_apply_peak_measured)
        self.assertEqual(receipt.c89_persistent_schedule_only_bytes, 18_516_504)
        self.assertEqual(
            receipt.known_large_layer_freeze_lower_bound_bytes, 95_453_188
        )
        self.assertAlmostEqual(
            receipt.known_large_layer_freeze_lower_bound_mib,
            91.03125381469727,
        )
        self.assertFalse(receipt.network_memory_gate_passes)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            replace(
                receipt,
                compact_weight_entries=np.int64(
                    receipt.compact_weight_entries
                ),
            )
        with self.assertRaises(native.ExactConvNativeScheduleError):
            replace(receipt, verdict_authority=True)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            replace(receipt, construction_mode="caller_self_signed")
        extra_key_receipt = replace(receipt)
        object.__setattr__(extra_key_receipt, "caller_claim", False)
        with self.assertRaises(native.ExactConvNativeScheduleError):
            extra_key_receipt.__post_init__()
        environment = native.native_kernel_environment()
        self.assertIn("-fwrapv", environment["strict_flags"])
        self.assertIn("-fno-strict-overflow", environment["strict_flags"])

    def test_c89_projection_is_schedule_only_and_network_gate_stays_closed(self) -> None:
        # Static C89 ONNX initializer census: 19 CONVs and 2,311,872 weight
        # entries.  The compact factory stores one bias channel per output
        # channel even when the ONNX Conv has no bias: 64 + 18*128 = 2,368.
        projection = native.project_compact_schedule_cache_bytes(
            weight_entries=2_311_872,
            bias_entries=2_368,
            conv_count=19,
        )
        self.assertEqual(projection["weight_f64_bytes"], 18_494_976)
        self.assertEqual(projection["bias_f64_bytes"], 18_944)
        self.assertEqual(projection["geometry_i64_bytes"], 2_584)
        self.assertEqual(
            projection["total_persistent_numeric_buffer_bytes"], 18_516_504
        )
        self.assertAlmostEqual(
            projection["total_persistent_numeric_buffer_mib"],
            17.658714294433594,
        )
        self.assertEqual(projection["expanded_operator_bytes"], 0)
        self.assertEqual(projection["expanded_operator_schedule_bytes"], 0)
        self.assertEqual(
            projection["scope"],
            "persistent_compact_schedule_numeric_buffers_only",
        )
        self.assertTrue(projection["persistent_schedule_within_cap"])
        self.assertFalse(projection["network_total_memory_established"])
        self.assertFalse(projection["c89_apply_peak_measured"])
        self.assertEqual(projection["known_large_layer_operator_nnz"], 7_929_856)
        self.assertEqual(projection["known_large_layer_output_rows"], 8_192)
        self.assertEqual(
            projection["known_large_layer_freeze_lower_bound_bytes"],
            95_453_188,
        )
        self.assertAlmostEqual(
            projection["known_large_layer_freeze_lower_bound_mib"],
            91.03125381469727,
        )
        self.assertFalse(projection["network_memory_gate_passes"])
        self.assertFalse(projection["production_promotion_claim"])
        print(json.dumps(projection, sort_keys=True))

    def test_fixed_raw_full_cold_and_peak_gates_report_honestly(self) -> None:
        layer, center, generators, error, ids = _fixture()
        owner = object()
        schedule = native.prepare_exact_conv_native_schedule(owner, layer)
        source = native.prepare_exact_monotone_row_local_source(
            center, generators, error, stable_column_ids=ids
        )
        def complete() -> native.ExactConvNativeResult:
            return native.apply_exact_conv_native_schedule(
                owner, schedule, source, expected_stable_column_ids=ids
            )

        def candidate_schedule_source_cold() -> native.ExactConvNativeResult:
            cold_schedule = native.prepare_exact_conv_native_schedule(owner, layer)
            cold_source = native.prepare_exact_monotone_row_local_source(
                center, generators, error, stable_column_ids=ids
            )
            return native.apply_exact_conv_native_schedule(
                owner,
                cold_schedule,
                cold_source,
                expected_stable_column_ids=ids,
            )

        def baseline_W_build_plus_generic_affine_cold() -> object:
            matrix, bias = csr_oracle.exact_sparse_conv2d_matrix_from_layer_candidate(
                layer
            )
            return operator_oracle._OperatorHZBuilder._affine(
                None,
                operator_oracle._AffineExpr(center, generators, error, 0),
                matrix,
                bias,
                layer_id=int(layer.id),
            )

        geometry = tuple(int(v) for v in schedule.geometry)
        context = (
            threadpool_limits(limits=1)
            if threadpool_limits is not None
            else nullcontext()
        )
        with context:
            for _ in range(10):
                complete()
            raw_times: list[float] = []
            full_times: list[float] = []
            candidate_cold_times: list[float] = []
            baseline_cold_times: list[float] = []
            gc.disable()
            try:
                for _ in range(101):
                    started = time.perf_counter_ns()
                    native._invoke_native(schedule, source, geometry)
                    raw_times.append((time.perf_counter_ns() - started) / 1e6)
                    started = time.perf_counter_ns()
                    complete()
                    full_times.append((time.perf_counter_ns() - started) / 1e6)
                for iteration in range(31):
                    ordered = (
                        (candidate_schedule_source_cold, candidate_cold_times,
                         baseline_W_build_plus_generic_affine_cold,
                         baseline_cold_times)
                        if iteration % 2
                        else (baseline_W_build_plus_generic_affine_cold,
                              baseline_cold_times,
                              candidate_schedule_source_cold,
                              candidate_cold_times)
                    )
                    first, first_times, second, second_times = ordered
                    started = time.perf_counter_ns()
                    first()
                    first_times.append((time.perf_counter_ns() - started) / 1e6)
                    started = time.perf_counter_ns()
                    second()
                    second_times.append((time.perf_counter_ns() - started) / 1e6)
            finally:
                gc.enable()
            gc.collect()
            tracemalloc.start()
            measured = complete()
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        raw_ms = median(raw_times)
        full_ms = median(full_times)
        candidate_cold_ms = median(candidate_cold_times)
        baseline_cold_ms = median(baseline_cold_times)
        cold_speedup = baseline_cold_ms / candidate_cold_ms
        peak_mb = peak / (1024.0 * 1024.0)
        cold_gate_passes = bool(cold_speedup >= 1.5)
        warm_compute_gates_pass = bool(
            raw_ms <= 1.25
            and full_ms <= 1.95
            and cold_gate_passes
            and peak_mb <= 5.32
        )
        report = {
            "schema": "exact_conv_compact_native_fixed_gate_v4",
            "input_shape": (1, 8, 12, 12),
            "output_shape": (1, 16, 12, 12),
            "raw_kernel_median_ms": raw_ms,
            "raw_threshold_ms": 1.25,
            "full_warm_median_ms": full_ms,
            "full_threshold_ms": 1.95,
            "candidate_schedule_source_prepare_plus_apply_median_ms": (
                candidate_cold_ms
            ),
            "baseline_W_build_plus_generic_affine_median_ms": baseline_cold_ms,
            "schedule_source_model_cold_speedup": cold_speedup,
            "schedule_source_model_cold_speedup_threshold": 1.5,
            "schedule_source_model_cold_gate_passes": cold_gate_passes,
            "runtime_compile_excluded_from_schedule_source_model_cold": False,
            "first_response_native_load_included": False,
            "first_response_gate_passes": False,
            "traced_peak_mb": peak_mb,
            "peak_threshold_mb": 5.32,
            "output_generator_nnz": int(measured.generators.nnz),
            "expanded_W_materialized": False,
            "source_snapshot_reused_without_second_copy": True,
            "output_ownership_transferred_without_freeze_copy": True,
            "c89_apply_peak_measured": False,
            "network_memory_gate_passes": False,
            "warm_compute_gates_pass": warm_compute_gates_pass,
            "synthetic_performance_gates_pass": False,
            "promotable": False,
            "production_promotion_claim": False,
        }
        print(json.dumps(report, sort_keys=True))
        self.assertLessEqual(raw_ms, 1.25)
        self.assertLessEqual(full_ms, 1.95)
        self.assertTrue(cold_gate_passes)
        self.assertLessEqual(peak_mb, 5.32)
        self.assertFalse(report["promotable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
