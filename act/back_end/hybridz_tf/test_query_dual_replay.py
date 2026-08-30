"""Soundness and integrity tests for the independent query-dual replay."""

from __future__ import annotations

import copy
import math
import time
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as replay_module
from act.back_end.hybridz_tf.query_dual_replay import (
    QueryDualReplayError,
    QueryDualReplayResult,
    QueryDualReplayTimeout,
    fraction_replay_lower_bounds,
    replay_query_lower_bounds,
    validate_query_dual_replay_result,
    verify_query_dual_replay_receipt,
)


_F64 = np.float64


def _layer(layer_id, kind, width, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[(int(layer_id), index) for index in range(int(width))],
        cache={},
    )


def _net(layers, preds):
    pred_map = {
        int(layer.id): [int(parent) for parent in preds[int(layer.id)]]
        for layer in layers
    }
    succs = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            succs[parent].append(child)
    return SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=succs,
        by_id={int(layer.id): layer for layer in layers},
    )


def _box(lower, upper):
    return {
        "lb": np.asarray(lower, dtype=np.float64),
        "ub": np.asarray(upper, dtype=np.float64),
    }


def _input_pair(width, lower, upper, *, shape=None):
    if shape is None:
        shape = (1, width)
    return (
        _layer(
            0,
            "INPUT",
            width,
            {"shape": tuple(shape), "dtype": "torch.float64"},
        ),
        _layer(1, "INPUT_SPEC", width, {"kind": "BOX"}),
        {1: _box(lower, upper)},
    )


def _assert_layer(layer_id, width):
    return _layer(layer_id, "ASSERT", width, {"kind": "AUDIT"})


def _assert_numeric_below_fraction(test, numeric, exact):
    test.assertEqual(len(numeric), len(exact))
    for got, wanted in zip(numeric, exact):
        test.assertTrue(math.isfinite(float(got)))
        test.assertLessEqual(Fraction.from_float(float(got)), wanted)


def _point_dense_toy():
    inp, spec, bounds = _input_pair(2, [0.25, -0.5], [0.25, -0.5])
    dense = _layer(
        2,
        "DENSE",
        2,
        {
            "weight": np.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=_F64),
            "bias": np.asarray([0.125, -0.25], dtype=_F64),
            "in_features": 2,
            "out_features": 2,
        },
    )
    assertion = _assert_layer(3, 2)
    net = _net(
        [inp, spec, dense, assertion],
        {0: [], 1: [0], 2: [1], 3: [2]},
    )
    value = np.asarray([1.125, -1.625], dtype=_F64)
    bounds[2] = _box(value, value)
    return net, bounds


def _single_relu_toy():
    inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
    dense = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[2.0]], dtype=_F64),
            "bias": np.asarray([-0.25], dtype=_F64),
            "in_features": 1,
            "out_features": 1,
        },
    )
    relu = _layer(3, "RELU", 1)
    assertion = _assert_layer(4, 1)
    net = _net(
        [inp, spec, dense, relu, assertion],
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3]},
    )
    bounds[2] = _box([-2.25], [1.75])
    # ReLU entries store certified pre-activation bounds.
    bounds[3] = _box([-2.25], [1.75])
    return net, bounds


def _double_relu_toy():
    inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
    dense1 = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[2.0]], dtype=_F64),
            "bias": np.asarray([-0.5], dtype=_F64),
            "in_features": 1,
            "out_features": 1,
        },
    )
    relu1 = _layer(3, "RELU", 1)
    dense2 = _layer(
        4,
        "DENSE",
        1,
        {
            "weight": np.asarray([[-3.0]], dtype=_F64),
            "bias": np.asarray([1.0], dtype=_F64),
            "in_features": 1,
            "out_features": 1,
        },
    )
    relu2 = _layer(5, "RELU", 1)
    assertion = _assert_layer(6, 1)
    net = _net(
        [inp, spec, dense1, relu1, dense2, relu2, assertion],
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]},
    )
    bounds.update(
        {
            2: _box([-2.5], [1.5]),
            3: _box([-2.5], [1.5]),
            4: _box([-3.5], [1.0]),
            5: _box([-3.5], [1.0]),
        }
    )
    return net, bounds


def _residual_toy():
    inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
    dense = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[2.0]], dtype=_F64),
            "bias": np.asarray([0.25], dtype=_F64),
            "in_features": 1,
            "out_features": 1,
        },
    )
    relu = _layer(3, "RELU", 1)
    add = _layer(4, "ADD", 1)
    flatten = _layer(5, "FLATTEN", 1, {"start_dim": 1})
    tail = _layer(
        6,
        "DENSE",
        1,
        {
            "weight": np.asarray([[0.5]], dtype=_F64),
            "bias": np.asarray([0.0], dtype=_F64),
            "in_features": 1,
            "out_features": 1,
        },
    )
    assertion = _assert_layer(7, 1)
    net = _net(
        [inp, spec, dense, relu, add, flatten, tail, assertion],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [1, 3],
            5: [4],
            6: [5],
            7: [6],
        },
    )
    bounds.update(
        {
            2: _box([-1.75], [2.25]),
            3: _box([-1.75], [2.25]),
            4: _box([-1.0], [3.25]),
            5: _box([-1.0], [3.25]),
            6: _box([-0.5], [1.625]),
        }
    )
    return net, bounds


def _conv_toy():
    inp, spec, bounds = _input_pair(
        4,
        [-1.0] * 4,
        [1.0] * 4,
        shape=(1, 1, 2, 2),
    )
    weight = np.asarray(
        [
            [[[1.0, -2.0], [3.0, 0.5]]],
            [[[-1.0, 2.0], [0.25, 4.0]]],
        ],
        dtype=_F64,
    )
    conv = _layer(
        2,
        "CONV2D",
        2,
        {
            "weight": weight,
            "bias": np.asarray([0.1, -0.2], dtype=_F64),
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": (2, 2),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": (1, 1, 2, 2),
            "output_shape": (1, 2, 1, 1),
        },
    )
    assertion = _assert_layer(3, 2)
    net = _net(
        [inp, spec, conv, assertion],
        {0: [], 1: [0], 2: [1], 3: [2]},
    )
    bounds[2] = _box([-6.4, -7.45], [6.6, 7.05])
    return net, bounds


class QueryDualReplayTests(unittest.TestCase):
    def test_point_consistency_receipt_and_one_hot(self):
        net, bounds = _point_dense_toy()
        result = replay_query_lower_bounds(
            net,
            bounds,
            one_hot={"indices": [0, 1], "signs": [1.0, -1.0]},
            query_bias=np.asarray([0.25, -0.5], dtype=_F64),
        )
        oracle = fraction_replay_lower_bounds(
            net,
            bounds,
            one_hot={"indices": [0, 1], "signs": [1.0, -1.0]},
            query_bias=np.asarray([0.25, -0.5], dtype=_F64),
        )
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        self.assertLess(float(oracle[0] - Fraction.from_float(result.lower_bounds[0])), 1e-12)
        self.assertTrue(result.proof_authority)
        self.assertTrue(verify_query_dual_replay_receipt(result.receipt))
        self.assertFalse(result.receipt["candidate_inputs_are_authoritative"])
        self.assertEqual(result.receipt["requested_chunk_size"], 1024)

    def test_single_relu_fraction_endpoint_audit_and_alpha_tree(self):
        net, bounds = _single_relu_toy()
        queries = np.asarray([[1.0], [-1.0]], dtype=_F64)
        # Candidate-native [B=1,M=2,n=1] layout.
        alpha = {3: np.asarray([[[0.75], [0.125]]], dtype=_F64)}
        result = replay_query_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        oracle = fraction_replay_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        # Concrete output range is [0, 1.75].
        self.assertLessEqual(result.lower_bounds[0], 0.0)
        self.assertLessEqual(result.lower_bounds[1], -1.75)
        self.assertEqual(result.receipt["stats"]["fraction_endpoint_audits"], 1)

    def test_double_relu_against_exact_concrete_range(self):
        net, bounds = _double_relu_toy()
        queries = np.asarray([[1.0], [-1.0]], dtype=_F64)
        alpha = {
            3: np.asarray([[[0.2], [0.8]]], dtype=_F64),
            5: np.asarray([[[0.6], [0.4]]], dtype=_F64),
        }
        result = replay_query_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        oracle = fraction_replay_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        # relu(1 - 3*relu(2*x-.5)) has exact range [0,1] on [-1,1].
        self.assertLessEqual(result.lower_bounds[0], 0.0)
        self.assertLessEqual(result.lower_bounds[1], -1.0)

    def test_residual_dag_merge_flatten(self):
        net, bounds = _residual_toy()
        queries = np.asarray([[1.0], [-1.0]], dtype=_F64)
        alpha = {3: np.asarray([0.5], dtype=_F64)}
        result = replay_query_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        oracle = fraction_replay_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        # 0.5*(x+relu(2*x+.25)) has exact min/max -0.5/1.625.
        self.assertLessEqual(result.lower_bounds[0], -0.5)
        self.assertLessEqual(result.lower_bounds[1], -1.625)
        self.assertGreaterEqual(result.receipt["stats"]["dag_merges"], 1)

    def test_direct_conv_toy_matches_independent_fraction_index_oracle(self):
        net, bounds = _conv_toy()
        queries = np.asarray(
            [[1.0, -1.0], [0.3, 0.7], [-0.25, 1.5]], dtype=_F64
        )
        result = replay_query_lower_bounds(net, bounds, query_rows=queries)
        oracle = fraction_replay_lower_bounds(net, bounds, query_rows=queries)
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        self.assertEqual(
            result.receipt["numeric_method"]["conv2d_adjoint"],
            "audited-direct-sparse-scatter-or-kernel-offset-channel-GEMM",
        )
        self.assertEqual(result.receipt["stats"]["conv_dense_blocks"], 1)

    def test_sparse_conv_query_block_equals_scalar_and_fraction(self):
        inp, spec, bounds = _input_pair(
            16, [-1.0] * 16, [1.0] * 16, shape=(1, 1, 4, 4)
        )
        conv = _layer(
            2,
            "CONV2D",
            32,
            {
                "weight": np.asarray([[[[2.0]]], [[[-0.5]]]], dtype=_F64),
                "bias": np.asarray([0.125, -0.25], dtype=_F64),
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 1, 4, 4),
                "output_shape": (1, 2, 4, 4),
            },
        )
        assertion = _assert_layer(3, 32)
        net = _net(
            [inp, spec, conv, assertion],
            {0: [], 1: [0], 2: [1], 3: [2]},
        )
        bounds[2] = _box([-2.25] * 32, [2.25] * 32)
        queries = np.zeros((16, 32), dtype=_F64)
        queries[np.arange(16), np.arange(16)] = 1.0
        batched = replay_query_lower_bounds(
            net, bounds, query_rows=queries, chunk_size=1024
        )
        scalar = replay_query_lower_bounds(
            net, bounds, query_rows=queries, chunk_size=1
        )
        np.testing.assert_array_equal(batched.lower_bounds, scalar.lower_bounds)
        oracle = fraction_replay_lower_bounds(net, bounds, query_rows=queries)
        _assert_numeric_below_fraction(self, batched.lower_bounds, oracle)
        self.assertEqual(batched.receipt["stats"]["conv_sparse_blocks"], 1)

    def test_large_cancellation_is_guarded(self):
        inp, spec, bounds = _input_pair(2, [-1.0, -1.0], [1.0, 1.0])
        dense = _layer(
            2,
            "DENSE",
            2,
            {
                "weight": np.asarray(
                    [[1.0e16, 1.0], [1.0e16, 0.0]], dtype=_F64
                ),
                "bias": np.zeros(2, dtype=_F64),
                "in_features": 2,
                "out_features": 2,
            },
        )
        assertion = _assert_layer(3, 2)
        net = _net(
            [inp, spec, dense, assertion],
            {0: [], 1: [0], 2: [1], 3: [2]},
        )
        bounds[2] = _box([-1.0e16, -1.0e16], [1.0e16, 1.0e16])
        query = np.asarray([[1.0, -1.0]], dtype=_F64)
        result = replay_query_lower_bounds(net, bounds, query_rows=query)
        oracle = fraction_replay_lower_bounds(net, bounds, query_rows=query)
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        self.assertEqual(oracle[0], Fraction(-1))
        self.assertGreater(
            float.fromhex(result.receipt["stats"]["guard_max_hex"]), 1.0
        )

    def test_subnormal_product_cannot_be_misclassified_as_exact_zero(self):
        eta = float(np.nextafter(np.float64(0.0), np.float64(math.inf)))
        inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
        dense = _layer(
            2,
            "DENSE",
            1,
            {
                "weight": np.asarray([[eta]], dtype=_F64),
                "bias": np.asarray([0.0], dtype=_F64),
                "in_features": 1,
                "out_features": 1,
            },
        )
        assertion = _assert_layer(3, 1)
        net = _net(
            [inp, spec, dense, assertion],
            {0: [], 1: [0], 2: [1], 3: [2]},
        )
        bounds[2] = _box([-eta], [eta])
        result = replay_query_lower_bounds(net, bounds, one_hot=0)
        oracle = fraction_replay_lower_bounds(net, bounds, one_hot=0)
        _assert_numeric_below_fraction(self, result.lower_bounds, oracle)
        self.assertEqual(oracle[0], -Fraction.from_float(eta))
        self.assertLess(result.lower_bounds[0], 0.0)
        self.assertGreater(
            float.fromhex(result.receipt["stats"]["guard_by_query_hex"][0]), 0.0
        )

    def test_interior_start_two_stage_and_ancestor_only_hashes(self):
        net, bounds = _double_relu_toy()
        stage1_bounds = {1: bounds[1], 2: bounds[2]}
        stage1 = replay_query_lower_bounds(
            net, stage1_bounds, start_lid=2, one_hot=0
        )
        self.assertEqual(stage1.receipt["start_layer_id"], 2)
        self.assertEqual(stage1.receipt["start_mode"], "EXPLICIT_INTERIOR")

        stage2_bounds = {1: bounds[1], 2: bounds[2], 3: bounds[3], 4: bounds[4]}
        stage2 = replay_query_lower_bounds(
            net,
            stage2_bounds,
            start_lid=4,
            one_hot=0,
            alpha_by_relu={3: np.asarray([[[0.4]]], dtype=_F64)},
        )
        self.assertEqual(stage2.receipt["start_layer_id"], 4)
        self.assertNotEqual(
            stage1.receipt["hashes"]["query_sha256"],
            stage2.receipt["hashes"]["query_sha256"],
        )
        # Downstream, non-ancestor boxes are neither required nor consumed.
        noisy = dict(stage1_bounds)
        noisy[5] = _box([-999.0], [999.0])
        repeated = replay_query_lower_bounds(net, noisy, start_lid=2, one_hot=0)
        self.assertEqual(
            stage1.receipt["hashes"]["bounds_sha256"],
            repeated.receipt["hashes"]["bounds_sha256"],
        )
        malformed_downstream = copy.deepcopy(net)
        malformed_downstream.preds[5] = [999_999]
        malformed_downstream.by_id[5].kind = "SIGMOID"
        cone_only = replay_query_lower_bounds(
            malformed_downstream, stage1_bounds, start_lid=2, one_hot=0
        )
        self.assertEqual(
            cone_only.receipt["hashes"]["net_sha256"],
            stage1.receipt["hashes"]["net_sha256"],
        )
        # Target-ReLU alpha is not silently accepted when replay starts at its
        # affine predecessor.
        with self.assertRaisesRegex(QueryDualReplayError, "INVALID_ALPHA"):
            replay_query_lower_bounds(
                net,
                stage1_bounds,
                start_lid=2,
                one_hot=0,
                alpha_by_relu={3: np.asarray([0.5], dtype=_F64)},
            )

    def test_batched_block_equals_scalar_blocks_and_exact_oracle(self):
        net, bounds = _residual_toy()
        queries = np.asarray(
            [[(-1.0) ** index * (index + 1) / 8.0] for index in range(24)],
            dtype=_F64,
        )
        alpha = {3: np.linspace(0.05, 0.95, 24, dtype=_F64).reshape(1, 24, 1)}
        batched = replay_query_lower_bounds(
            net,
            bounds,
            query_rows=queries,
            alpha_by_relu=alpha,
            chunk_size=1024,
        )
        scalar_blocks = replay_query_lower_bounds(
            net,
            bounds,
            query_rows=queries,
            alpha_by_relu=alpha,
            chunk_size=1,
        )
        np.testing.assert_array_equal(batched.lower_bounds, scalar_blocks.lower_bounds)
        oracle = fraction_replay_lower_bounds(
            net, bounds, query_rows=queries, alpha_by_relu=alpha
        )
        _assert_numeric_below_fraction(self, batched.lower_bounds, oracle)
        self.assertGreater(batched.receipt["effective_chunk_size"], 1)
        self.assertEqual(
            len(batched.receipt["stats"]["guard_by_query_hex"]), len(queries)
        )

    def test_bounds_alpha_net_and_receipt_tamper_fail_closed(self):
        net, bounds = _single_relu_toy()
        query = np.asarray([[1.0], [-1.0]], dtype=_F64)
        alpha = {3: np.asarray([[[0.2], [0.8]]], dtype=_F64)}
        baseline = replay_query_lower_bounds(
            net, bounds, query_rows=query, alpha_by_relu=alpha
        )
        hashes = baseline.receipt["hashes"]
        self.assertTrue(validate_query_dual_replay_result(baseline))

        changed_bounds = copy.deepcopy(bounds)
        changed_bounds[3]["ub"][0] = 1.5
        with self.assertRaisesRegex(QueryDualReplayError, "HASH_MISMATCH"):
            replay_query_lower_bounds(
                net,
                changed_bounds,
                query_rows=query,
                alpha_by_relu=alpha,
                expected_bounds_sha256=hashes["bounds_sha256"],
            )

        changed_alpha = {3: alpha[3].copy()}
        changed_alpha[3][0, 0, 0] = 0.3
        with self.assertRaisesRegex(QueryDualReplayError, "HASH_MISMATCH"):
            replay_query_lower_bounds(
                net,
                bounds,
                query_rows=query,
                alpha_by_relu=changed_alpha,
                expected_alpha_sha256=hashes["alpha_sha256"],
            )

        changed_net = copy.deepcopy(net)
        changed_net.by_id[2].params["weight"][0, 0] = 2.25
        with self.assertRaisesRegex(QueryDualReplayError, "HASH_MISMATCH"):
            replay_query_lower_bounds(
                changed_net,
                bounds,
                query_rows=query,
                alpha_by_relu=alpha,
                expected_net_sha256=hashes["net_sha256"],
            )

        tampered = copy.deepcopy(baseline.receipt)
        tampered["lower_bounds_hex"][0] = "0x0.0p+0"
        self.assertFalse(verify_query_dual_replay_receipt(tampered))
        replaced_values = baseline.lower_bounds.copy()
        replaced_values[0] = np.nextafter(
            replaced_values[0], np.float64(math.inf)
        )
        self.assertFalse(
            validate_query_dual_replay_result(
                QueryDualReplayResult(
                    lower_bounds=replaced_values,
                    receipt=baseline.receipt,
                )
            )
        )
        self.assertFalse(
            validate_query_dual_replay_result(
                QueryDualReplayResult(
                    lower_bounds=baseline.lower_bounds.astype(np.float32),
                    receipt=baseline.receipt,
                )
            )
        )
        self.assertFalse(
            verify_query_dual_replay_receipt(
                baseline.receipt, expected_bounds_sha256="0" * 64
            )
        )

    def test_deadline_and_fraction_budget_fail_closed(self):
        net, bounds = _single_relu_toy()
        with self.assertRaises(QueryDualReplayTimeout):
            replay_query_lower_bounds(
                net,
                bounds,
                one_hot=0,
                deadline=time.monotonic() - 1.0,
            )
        with self.assertRaisesRegex(QueryDualReplayError, "ORACLE_BUDGET"):
            fraction_replay_lower_bounds(
                net, bounds, one_hot=0, max_arithmetic_terms=1
            )

    def test_nonfinite_bad_alpha_and_unsupported_operator_fail_closed(self):
        net, bounds = _single_relu_toy()
        bad_bounds = copy.deepcopy(bounds)
        bad_bounds[2]["lb"][0] = -math.inf
        with self.assertRaisesRegex(QueryDualReplayError, "NONFINITE"):
            replay_query_lower_bounds(net, bad_bounds, one_hot=0)
        with self.assertRaisesRegex(QueryDualReplayError, "ALPHA_NOT_F64"):
            replay_query_lower_bounds(
                net,
                bounds,
                one_hot=0,
                alpha_by_relu={3: np.asarray([0.5], dtype=np.float32)},
            )
        unsupported = copy.deepcopy(net)
        unsupported.by_id[2].kind = "SIGMOID"
        with self.assertRaisesRegex(QueryDualReplayError, "UNSUPPORTED_OPERATOR"):
            replay_query_lower_bounds(unsupported, bounds, one_hot=0)
        residual, residual_bounds = _residual_toy()
        residual.by_id[4].params["bias"] = np.asarray([0.25], dtype=_F64)
        with self.assertRaisesRegex(
            QueryDualReplayError, "unbiased sum"
        ):
            replay_query_lower_bounds(
                residual, residual_bounds, query_rows=np.ones((1, 1))
            )
        three_input, three_input_bounds = _residual_toy()
        three_input.preds[4] = [1, 2, 3]
        with self.assertRaisesRegex(
            QueryDualReplayError, "exactly two predecessors"
        ):
            replay_query_lower_bounds(
                three_input,
                three_input_bounds,
                query_rows=np.ones((1, 1)),
            )

    def test_conv_declared_shape_formula_is_checked(self):
        net, bounds = _conv_toy()
        net.by_id[2].params["output_shape"] = (1, 2, 2, 1)
        net.by_id[2].out_vars = [(2, index) for index in range(4)]
        bounds[2] = _box([-1.0] * 4, [1.0] * 4)
        with self.assertRaisesRegex(QueryDualReplayError, "declared output"):
            replay_query_lower_bounds(
                net, bounds, query_rows=np.ones((1, 4), dtype=_F64)
            )

    def test_wide_longdouble_platform_contract_is_fail_closed_and_recorded(self):
        net, bounds = _point_dense_toy()
        result = replay_query_lower_bounds(net, bounds, one_hot=0)
        numeric = result.receipt["numeric_platform"]
        self.assertGreater(numeric["longdouble_nmant"], numeric["binary64_nmant"])
        self.assertTrue(numeric["longdouble_eps"])
        self.assertTrue(numeric["gradual_underflow"])
        self.assertTrue(numeric["blas_subnormal_dot"])
        with mock.patch.object(
            replay_module, "_has_wide_longdouble", return_value=False
        ):
            with self.assertRaisesRegex(QueryDualReplayError, "NUMERIC_PLATFORM"):
                replay_query_lower_bounds(net, bounds, one_hot=0)


if __name__ == "__main__":
    unittest.main()
