"""Prerequisite diagnostics for the cached direct-CSR CONV2D experiment.

All network tests use exact ReLU graphs.  The timing receipt is synthetic,
single-layer, and builder-only; it is not a CIFAR or end-to-end claim.
"""

from __future__ import annotations

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
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import exact_sparse_conv_csr_candidate as candidate
from act.back_end.hybridz_tf import operator_hz
from act.back_end.hybridz_tf.tf_cnn import (
    sparse_conv2d_matrix_from_layer as legacy_conv_matrix,
)


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


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
    tensor_parameters: bool = False,
    omit_batch_dimension: bool = False,
    no_bias: bool = False,
):
    rng = np.random.default_rng(seed)
    kh, kw = kernel
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    weight = rng.standard_normal(
        (out_ch, in_ch // groups, kh, kw), dtype=np.float64
    )
    weight.reshape(-1)[::7] = 0.0
    weight.reshape(-1)[3::19] = -0.0
    bias = rng.standard_normal(out_ch, dtype=np.float64)
    if tensor_parameters:
        weight = torch.from_numpy(weight.copy())
        bias = torch.from_numpy(bias.copy())
    input_shape = (batch, in_ch, height, width)
    output_shape = (batch, out_ch, out_h, out_w)
    if omit_batch_dimension:
        if batch != 1:
            raise AssertionError("three-dimensional shape requires batch one")
        input_shape = input_shape[1:]
        output_shape = output_shape[1:]
    params = {
        "weight": weight,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }
    if not no_bias:
        params["bias"] = bias
    return SimpleNamespace(id=seed, params=params)


def _assert_bitwise_csr_equal(
    case: unittest.TestCase,
    left: sp.csr_matrix,
    right: sp.csr_matrix,
) -> None:
    case.assertEqual(left.shape, right.shape)
    case.assertTrue(left.has_canonical_format)
    case.assertTrue(right.has_canonical_format)
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    np.testing.assert_array_equal(
        left.data.view(np.uint64), right.data.view(np.uint64)
    )


def _assert_bitwise_array_equal(
    case: unittest.TestCase,
    left,
    right,
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    case.assertEqual(left_array.shape, right_array.shape)
    case.assertEqual(left_array.dtype, right_array.dtype)
    np.testing.assert_array_equal(
        left_array.view(np.uint8), right_array.view(np.uint8)
    )


def _assert_numeric_hz_core_equal(case, left, right) -> None:
    _assert_bitwise_array_equal(case, left.c, right.c)
    _assert_bitwise_array_equal(case, left.b, right.b)
    _assert_bitwise_array_equal(case, left.ub, right.ub)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        _assert_bitwise_csr_equal(
            case, getattr(left, name), getattr(right, name)
        )


def _exact_relu_toy():
    """Return ``x -> Conv([1,-1]) -> exact ReLU`` with two unstable rows."""

    dtype = torch.float64

    def layer(layer_id, kind, params, width):
        return SimpleNamespace(
            id=int(layer_id),
            kind=str(kind),
            params=dict(params),
            out_vars=list(range(int(width))),
            in_vars=[],
        )

    lower = torch.tensor([[-1.0]], dtype=dtype)
    upper = torch.tensor([[1.0]], dtype=dtype)
    layers = [
        layer(0, "INPUT", {"shape": (1, 1, 1, 1)}, 1),
        layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            1,
        ),
        layer(
            2,
            "CONV2D",
            {
                "weight": torch.tensor(
                    [[[[1.0]]], [[[-1.0]]]], dtype=dtype
                ),
                "bias": torch.zeros(2, dtype=dtype),
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 1, 1, 1),
                "output_shape": (1, 2, 1, 1),
                "data_format": "NCHW",
                "padding_mode": "zeros",
            },
            2,
        ),
        layer(3, "RELU", {}, 2),
        layer(4, "ASSERT", {"kind": "UNSAFE_LINEAR"}, 2),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
    succs = {layer.id: [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {
        0: Fact(Bounds(lower.clone(), upper.clone()), ConSet()),
        1: Fact(Bounds(lower.clone(), upper.clone()), ConSet()),
    }
    for layer_id in (2, 3, 4):
        facts[layer_id] = Fact(
            Bounds(
                -100.0 * torch.ones((1, 2), dtype=dtype),
                100.0 * torch.ones((1, 2), dtype=dtype),
            ),
            ConSet(),
        )
    return net, facts, layers[2]


class CachedDirectOperatorConvCSRTest(unittest.TestCase):
    def setUp(self):
        candidate.clear_exact_conv_topology_cache()
        direct = mock.patch.object(
            operator_hz, "_EXPERIMENTAL_CACHED_DIRECT_CONV", True
        )
        direct.start()
        self.addCleanup(direct.stop)

    def test_production_default_does_not_import_or_execute_candidate(self):
        layer = _layer(
            seed=0,
            batch=1,
            in_ch=1,
            out_ch=1,
            height=2,
            width=2,
            kernel=(1, 1),
        )
        with (
            mock.patch.object(
                operator_hz, "_EXPERIMENTAL_CACHED_DIRECT_CONV", False
            ),
            mock.patch.object(
                candidate,
                "exact_sparse_conv2d_matrix_from_layer_candidate",
                side_effect=AssertionError(
                    "default production path executed disconnected candidate"
                ),
            ) as disconnected,
        ):
            _matrix, _bias, mode = (
                operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                    layer
                )
            )
        disconnected.assert_not_called()
        self.assertEqual(mode, "vectorized_exact_csr_v1")

    def test_full_operator_exact_relu_toy_uses_cached_direct_dispatch(self):
        net, facts, conv_layer = _exact_relu_toy()
        real_direct = candidate.exact_sparse_conv2d_matrix_from_layer_candidate
        with (
            mock.patch.object(
                candidate,
                "exact_sparse_conv2d_matrix_from_layer_candidate",
                wraps=real_direct,
            ) as direct,
            mock.patch.object(
                operator_hz,
                "_vectorized_sparse_conv2d_matrix_from_layer",
                side_effect=AssertionError(
                    "production strict CONV used vectorized oracle"
                ),
            ),
            mock.patch.object(
                operator_hz,
                "_legacy_sparse_conv2d_matrix_from_layer",
                side_effect=AssertionError(
                    "production strict CONV used legacy fallback"
                ),
            ),
        ):
            build = operator_hz.build_operator_hz(
                net, facts, facts, exact_budget=-1, materialize_add=True
            )
        direct.assert_called_once_with(conv_layer)
        self.assertEqual(build.hz.n_out, 2)
        self.assertEqual(build.hz.n_bin, 2)
        self.assertEqual(build.metadata["layers"][2]["operator_nnz"], 2)
        self.assertEqual(
            build.metadata["layers"][2]["operator_csr_builder"],
            "cached_direct_exact_csr_v2",
        )
        relu = build.metadata["layers"][3]
        self.assertEqual(relu["relu_unstable"], 2)
        self.assertEqual(relu["relu_exact"], 2)
        self.assertEqual(relu["relu_relaxed"], 0)
        self.assertEqual(relu["relu_triangle_rows"], 0)
        self.assertEqual(relu["new_bin"], 2)

    def test_direct_vectorized_and_legacy_are_bit_exact_for_strict_geometry(self) -> None:
        cases = (
            _layer(
                seed=1,
                batch=1,
                in_ch=3,
                out_ch=5,
                height=7,
                width=6,
                kernel=(3, 3),
                padding=1,
            ),
            _layer(
                seed=2,
                batch=1,
                in_ch=4,
                out_ch=6,
                height=8,
                width=7,
                kernel=(3, 2),
                stride=(2, 1),
                padding=(1, 0),
                groups=2,
            ),
            _layer(
                seed=3,
                batch=1,
                in_ch=4,
                out_ch=4,
                height=9,
                width=8,
                kernel=(2, 3),
                padding=(1, 2),
                dilation=(2, 2),
                groups=4,
                tensor_parameters=True,
            ),
            _layer(
                seed=4,
                batch=1,
                in_ch=2,
                out_ch=3,
                height=5,
                width=6,
                kernel=(1, 1),
                no_bias=True,
            ),
        )
        for layer in cases:
            with self.subTest(layer=layer.id):
                expected_matrix, expected_bias = legacy_conv_matrix(layer)
                vectorized_matrix, vectorized_bias = (
                    operator_hz._vectorized_sparse_conv2d_matrix_from_layer(
                        layer
                    )
                )
                actual_matrix, actual_bias, mode = (
                    operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                        layer
                    )
                )
                _assert_bitwise_csr_equal(
                    self, vectorized_matrix, expected_matrix
                )
                _assert_bitwise_csr_equal(
                    self, actual_matrix, vectorized_matrix
                )
                _assert_bitwise_array_equal(
                    self, vectorized_bias, expected_bias
                )
                _assert_bitwise_array_equal(
                    self, actual_bias, vectorized_bias
                )
                self.assertEqual(mode, "cached_direct_exact_csr_v2")

    def test_all_zero_weight_preserves_empty_canonical_shape(self) -> None:
        layer = _layer(
            seed=9,
            batch=1,
            in_ch=2,
            out_ch=2,
            height=5,
            width=4,
            kernel=(3, 3),
            padding=1,
        )
        layer.params["weight"][...] = -0.0
        expected_matrix, expected_bias = legacy_conv_matrix(layer)
        vectorized_matrix, vectorized_bias = (
            operator_hz._vectorized_sparse_conv2d_matrix_from_layer(layer)
        )
        actual_matrix, actual_bias, mode = (
            operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                layer
            )
        )
        _assert_bitwise_csr_equal(self, vectorized_matrix, expected_matrix)
        _assert_bitwise_csr_equal(self, actual_matrix, expected_matrix)
        _assert_bitwise_array_equal(self, vectorized_bias, expected_bias)
        _assert_bitwise_array_equal(self, actual_bias, expected_bias)
        self.assertEqual(mode, "cached_direct_exact_csr_v2")

    def test_old_vectorized_oracle_retains_explicit_legacy_fallback(self) -> None:
        layer = SimpleNamespace(
            params={
                "weight": np.zeros((1, 1, 1)),
                "input_shape": (1, 1, 1),
                "output_shape": (1, 1, 1),
            }
        )
        sentinel = (
            sp.csr_matrix((1, 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
        with mock.patch.object(
            operator_hz,
            "_legacy_sparse_conv2d_matrix_from_layer",
            return_value=sentinel,
        ) as fallback:
            matrix, bias, mode = (
                operator_hz._vectorized_sparse_conv2d_matrix_from_layer_with_legacy_fallback(
                    layer
                )
            )
            self.assertIs(matrix, sentinel[0])
            self.assertIs(bias, sentinel[1])
            self.assertEqual(mode, "legacy_explicit_unsupported_fallback_v1")
            fallback.assert_called_once_with(layer)

    def test_vectorized_internal_failure_is_not_silently_fallback(self) -> None:
        layer = _layer(
            seed=10,
            batch=1,
            in_ch=1,
            out_ch=1,
            height=2,
            width=2,
            kernel=(1, 1),
        )
        with (
            mock.patch.object(
                operator_hz,
                "_vectorized_sparse_conv2d_matrix_from_layer",
                side_effect=RuntimeError("vectorized defect"),
            ),
            mock.patch.object(
                operator_hz,
                "_legacy_sparse_conv2d_matrix_from_layer",
            ) as fallback,
        ):
            with self.assertRaisesRegex(RuntimeError, "vectorized defect"):
                operator_hz._vectorized_sparse_conv2d_matrix_from_layer_with_legacy_fallback(
                    layer
                )
            fallback.assert_not_called()

    def test_topology_cache_hits_and_parameter_aba_is_fresh(self) -> None:
        layer = _layer(
            seed=71,
            batch=1,
            in_ch=4,
            out_ch=6,
            height=9,
            width=8,
            kernel=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            groups=2,
        )
        original_weight = layer.params["weight"].copy()
        original_bias = layer.params["bias"].copy()

        first_matrix, first_bias, first_mode = (
            operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                layer
            )
        )
        first_matrix_snapshot = first_matrix.copy()
        first_bias_snapshot = first_bias.copy()
        after_first = candidate.exact_conv_topology_cache_info()
        self.assertEqual(after_first.misses, 1)
        self.assertEqual(after_first.hits, 0)

        layer.params["weight"] *= -2.0
        layer.params["bias"] += 3.0
        middle_matrix, middle_bias, middle_mode = (
            operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                layer
            )
        )
        after_middle = candidate.exact_conv_topology_cache_info()
        self.assertEqual(after_middle.misses, 1)
        self.assertEqual(after_middle.hits, 1)
        self.assertFalse(
            np.array_equal(
                first_matrix.data.view(np.uint64),
                middle_matrix.data.view(np.uint64),
            )
        )
        self.assertFalse(
            np.array_equal(
                first_bias.view(np.uint64), middle_bias.view(np.uint64)
            )
        )
        _assert_bitwise_csr_equal(self, first_matrix, first_matrix_snapshot)
        _assert_bitwise_array_equal(self, first_bias, first_bias_snapshot)

        layer.params["weight"][...] = original_weight
        layer.params["bias"][...] = original_bias
        final_matrix, final_bias, final_mode = (
            operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                layer
            )
        )
        after_final = candidate.exact_conv_topology_cache_info()
        self.assertEqual(after_final.misses, 1)
        self.assertEqual(after_final.hits, 2)
        _assert_bitwise_csr_equal(self, first_matrix, final_matrix)
        _assert_bitwise_array_equal(self, first_bias, final_bias)
        self.assertEqual(
            {first_mode, middle_mode, final_mode},
            {"cached_direct_exact_csr_v2"},
        )

    def test_candidate_nan_malformed_and_int64_rejections_fail_closed(self) -> None:
        malformed = _layer(
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
        malformed.params["groups"] = 3

        nan_weight = _layer(
            seed=82,
            batch=1,
            in_ch=1,
            out_ch=1,
            height=3,
            width=3,
            kernel=(1, 1),
        )
        nan_weight.params["weight"][0, 0, 0, 0] = np.nan

        signed_max = int(np.iinfo(np.int64).max)
        hostile_int64 = SimpleNamespace(
            id=83,
            params={
                "weight": np.ones((1, 1, 3, 1), dtype=np.float64),
                "bias": np.zeros(1, dtype=np.float64),
                "input_shape": (1, 1, 1, 1),
                "output_shape": (1, 1, 1, 1),
                "stride": 1,
                "padding": (signed_max, 0),
                "dilation": (signed_max, 1),
                "groups": 1,
            },
        )

        with (
            mock.patch.object(
                operator_hz,
                "_vectorized_sparse_conv2d_matrix_from_layer",
                side_effect=AssertionError("rejection silently used oracle"),
            ),
            mock.patch.object(
                operator_hz,
                "_legacy_sparse_conv2d_matrix_from_layer",
                side_effect=AssertionError("rejection silently used fallback"),
            ),
        ):
            for layer in (malformed, nan_weight, hostile_int64):
                with self.subTest(layer=layer.id):
                    with self.assertRaises(
                        operator_hz.OperatorHZBuildError
                    ) as captured:
                        operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                            layer
                        )
                    self.assertIsInstance(
                        captured.exception.__cause__,
                        candidate.ExactSparseConvCandidateError,
                    )

        net, facts, conv_layer = _exact_relu_toy()
        conv_layer.params["bias"][0] = torch.nan
        with self.assertRaisesRegex(
            operator_hz.OperatorHZBuildError,
            "failed to build sparse CONV2D operator",
        ) as captured:
            operator_hz.build_operator_hz(
                net, facts, facts, exact_budget=-1
            )
        self.assertIsInstance(
            captured.exception.__cause__, operator_hz.OperatorHZBuildError
        )

    def test_k2_exact_relu_numeric_core_matches_old_conv_oracle(self) -> None:
        net, facts, _conv_layer = _exact_relu_toy()
        promoted_k2 = operator_hz.build_operator_hz(
            net, facts, facts, exact_budget=2
        )
        promoted_all = operator_hz.build_operator_hz(
            net, facts, facts, exact_budget=-1
        )

        def old_vectorized_dispatch(layer):
            matrix, bias = (
                operator_hz._vectorized_sparse_conv2d_matrix_from_layer(layer)
            )
            return matrix, bias, "vectorized_exact_csr_v1"

        with mock.patch.object(
            operator_hz,
            "_sparse_conv2d_matrix_from_layer_strict_with_mode",
            side_effect=old_vectorized_dispatch,
        ):
            oracle_k2 = operator_hz.build_operator_hz(
                net, facts, facts, exact_budget=2
            )

        _assert_numeric_hz_core_equal(
            self, promoted_k2.hz, oracle_k2.hz
        )
        _assert_numeric_hz_core_equal(
            self, promoted_k2.hz, promoted_all.hz
        )
        for build in (promoted_k2, promoted_all, oracle_k2):
            relu = build.metadata["layers"][3]
            self.assertEqual(relu["relu_exact"], 2)
            self.assertEqual(relu["relu_relaxed"], 0)
            self.assertEqual(relu["relu_triangle_rows"], 0)
            self.assertEqual(build.hz.n_bin, 2)
        self.assertEqual(
            promoted_k2.metadata["layers"][2]["operator_csr_builder"],
            "cached_direct_exact_csr_v2",
        )
        self.assertEqual(
            oracle_k2.metadata["layers"][2]["operator_csr_builder"],
            "vectorized_exact_csr_v1",
        )


class CachedDirectOperatorConvPrerequisiteDiagnostic(unittest.TestCase):
    def setUp(self):
        direct = mock.patch.object(
            operator_hz, "_EXPERIMENTAL_CACHED_DIRECT_CONV", True
        )
        direct.start()
        self.addCleanup(direct.stop)

    def test_same_topology_builder_only_paired_receipt(self) -> None:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # pragma: no cover - project CI dependency
            self.skipTest("threadpoolctl is required for controlled timing")

        # Synthetic same-topology operator+bias construction only.  This does
        # not time a network, an affine center/generator transform, or errors.
        layer = _layer(
            seed=405,
            batch=1,
            in_ch=16,
            out_ch=64,
            height=32,
            width=32,
            kernel=(3, 3),
            padding=1,
        )
        layer.params["weight"].reshape(-1)[::11] = 0.0
        candidate.clear_exact_conv_topology_cache()

        def baseline_call():
            return operator_hz._vectorized_sparse_conv2d_matrix_from_layer(
                layer
            )

        def promoted_call():
            matrix, bias, mode = (
                operator_hz._sparse_conv2d_matrix_from_layer_strict_with_mode(
                    layer
                )
            )
            if mode != "cached_direct_exact_csr_v2":
                raise AssertionError(f"unexpected production mode {mode!r}")
            return matrix, bias

        def paired_ratios(repetitions=7):
            ratios = []
            for repetition in range(repetitions):
                gc.collect()
                calls = (
                    (promoted_call, baseline_call)
                    if repetition % 2
                    else (baseline_call, promoted_call)
                )
                elapsed = {}
                values = {}
                for call in calls:
                    started = time.perf_counter_ns()
                    values[call] = call()
                    elapsed[call] = time.perf_counter_ns() - started
                baseline_matrix, baseline_bias = values[baseline_call]
                promoted_matrix, promoted_bias = values[promoted_call]
                _assert_bitwise_csr_equal(
                    self, promoted_matrix, baseline_matrix
                )
                _assert_bitwise_array_equal(
                    self, promoted_bias, baseline_bias
                )
                ratios.append(
                    elapsed[baseline_call] / max(1, elapsed[promoted_call])
                )
                del values
            return ratios

        thread_keys = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
        previous_thread_env = {key: os.environ.get(key) for key in thread_keys}
        try:
            for key in thread_keys:
                os.environ[key] = "1"
            with threadpool_limits(limits=1):
                _assert_bitwise_csr_equal(
                    self, promoted_call()[0], baseline_call()[0]
                )
                for _ in range(2):
                    baseline_call()
                    promoted_call()
                ratios = paired_ratios()
        finally:
            for key, value in previous_thread_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        speedup = float(statistics.median(ratios))
        cache_info = candidate.exact_conv_topology_cache_info()
        receipt = {
            "schema": "act.operator_hz.cached_direct_conv_builder_receipt.v1",
            "scope": "synthetic_same_topology_operator_and_bias_builder_only",
            "primary_baseline": "existing_operator_vectorized_exact_csr_v1",
            "experimental_candidate": "cached_direct_exact_csr_v2",
            "threads": 1,
            "warmups": 2,
            "paired_repetitions": 7,
            "median_paired_speedup": speedup,
            "minimum_builder_only_prerequisite_speedup": 1.5,
            "production_promotion_claim": False,
            "bit_exact_to_primary_baseline": True,
            "topology_cache_hits": int(cache_info.hits),
            "topology_cache_misses": int(cache_info.misses),
            "network_run": False,
            "real_or_large_run": False,
            "cifar_end_to_end_claim": False,
            "affine_center_generator_error_transform_timed": False,
            "fused_row_local_path_timed": False,
        }
        print(
            "OPERATOR_CACHED_DIRECT_CONV_BUILDER_RECEIPT="
            + json.dumps(receipt, sort_keys=True)
        )
        self.assertGreater(cache_info.hits, 0)
        self.assertGreaterEqual(speedup, 1.5)


if __name__ == "__main__":
    unittest.main()
