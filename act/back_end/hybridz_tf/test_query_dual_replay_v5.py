"""End-to-end controlled gates for the isolated V5 replay candidate."""

from __future__ import annotations

import copy
import math
import unittest
from fractions import Fraction

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf.query_dual_replay_v5 import (
    QueryDualReplayV5CandidateResult,
    replay_query_lower_bounds_v5_candidate,
    verify_query_dual_replay_v5_candidate,
)
from act.back_end.hybridz_tf.test_query_dual_replay import (
    _assert_layer,
    _box,
    _conv_toy,
    _input_pair,
    _layer,
    _net,
    _point_dense_toy,
    _residual_toy,
    _single_relu_toy,
)


_F64 = np.float64


def _assert_below_fraction(
    test: unittest.TestCase,
    numeric: np.ndarray,
    exact: tuple[Fraction, ...],
) -> None:
    test.assertEqual(numeric.size, len(exact))
    for stored, oracle in zip(numeric, exact):
        test.assertTrue(math.isfinite(float(stored)))
        test.assertLessEqual(Fraction.from_float(float(stored)), oracle)


def _compare_v3_v5_fraction(
    test: unittest.TestCase,
    net,
    bounds,
    **kwargs,
):
    v3 = frozen.replay_query_lower_bounds(net, bounds, **kwargs)
    v5 = replay_query_lower_bounds_v5_candidate(net, bounds, **kwargs)
    oracle_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"chunk_size", "max_workspace_bytes"}
    }
    exact = frozen.fraction_replay_lower_bounds(
        net, bounds, **oracle_kwargs
    )
    _assert_below_fraction(test, v3.lower_bounds, exact)
    _assert_below_fraction(test, v5.lower_bounds, exact)
    # This is a preregistered promotion gate, not a mathematical premise:
    # the faster guard may only advance if it does not weaken the frozen toy.
    test.assertTrue(np.all(v5.lower_bounds >= v3.lower_bounds))
    test.assertTrue(verify_query_dual_replay_v5_candidate(v5))
    test.assertFalse(v5.proof_authority)
    test.assertFalse(v5.lower_bounds.flags.writeable)
    return v3, v5, exact


def _sparse_conv_toy():
    inp, spec, bounds = _input_pair(
        16,
        [-1.0] * 16,
        [1.0] * 16,
        shape=(1, 1, 4, 4),
    )
    conv = _layer(
        2,
        "CONV2D",
        32,
        {
            "weight": np.asarray(
                [[[[2.0]]], [[[-0.5]]]], dtype=_F64
            ),
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
    queries = np.zeros((4, 32), dtype=_F64)
    queries[np.arange(4), np.arange(4)] = 1.0
    return net, bounds, queries


class QueryDualReplayV5Tests(unittest.TestCase):
    def test_dense_relu_residual_and_dense_conv_against_fraction(self):
        net, bounds = _point_dense_toy()
        _compare_v3_v5_fraction(
            self,
            net,
            bounds,
            one_hot={"indices": [0, 1], "signs": [1.0, -1.0]},
            query_bias=np.asarray([0.25, -0.5], dtype=_F64),
        )

        net, bounds = _single_relu_toy()
        _compare_v3_v5_fraction(
            self,
            net,
            bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={
                3: np.asarray([[[0.75], [0.125]]], dtype=_F64)
            },
        )

        net, bounds = _residual_toy()
        _compare_v3_v5_fraction(
            self,
            net,
            bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={3: np.asarray([0.5], dtype=_F64)},
        )

        net, bounds = _conv_toy()
        _, result, _ = _compare_v3_v5_fraction(
            self,
            net,
            bounds,
            query_rows=np.asarray(
                [[1.0, -1.0], [0.3, 0.7], [-0.25, 1.5]],
                dtype=_F64,
            ),
        )
        records = result.receipt["affine_executions"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operator"], "CONV2D")
        self.assertEqual(records[0]["conv_branch"], "dense")
        self.assertEqual(records[0]["scalar_guard_applied_count"], 1)
        self.assertEqual(
            records[0]["componentwise_radius_applied_count"], 0
        )

    def test_sparse_branch_chunking_and_exact_guard_coverage(self):
        net, bounds, queries = _sparse_conv_toy()
        batched = replay_query_lower_bounds_v5_candidate(
            net, bounds, query_rows=queries, chunk_size=1024
        )
        scalar = replay_query_lower_bounds_v5_candidate(
            net, bounds, query_rows=queries, chunk_size=1
        )
        exact = frozen.fraction_replay_lower_bounds(
            net, bounds, query_rows=queries
        )
        np.testing.assert_array_equal(
            batched.lower_bounds.view(np.uint64),
            scalar.lower_bounds.view(np.uint64),
        )
        _assert_below_fraction(self, batched.lower_bounds, exact)
        self.assertEqual(batched.receipt["affine_execution_count"], 1)
        record = batched.receipt["affine_executions"][0]
        self.assertEqual(record["conv_branch"], "sparse")
        self.assertLessEqual(
            record["threshold_lhs"], record["threshold_rhs"]
        )
        self.assertEqual(record["scalar_guard_applied_count"], 0)
        self.assertEqual(
            record["componentwise_radius_applied_count"], 1
        )
        self.assertEqual(scalar.receipt["affine_execution_count"], 4)
        self.assertEqual(scalar.receipt["conv_support_count"], 0)
        self.assertTrue(verify_query_dual_replay_v5_candidate(batched))
        self.assertTrue(verify_query_dual_replay_v5_candidate(scalar))

    def test_deterministic_random_dense_fraction_and_tightness_gate(self):
        rng = np.random.default_rng(20260728)
        audited_objectives = 0
        for case in range(100):
            input_width = int(rng.integers(1, 9))
            output_width = int(rng.integers(1, 9))
            query_count = 50
            lower = np.ascontiguousarray(
                rng.uniform(-2.0, 0.0, size=input_width), dtype=_F64
            )
            upper = np.ascontiguousarray(
                rng.uniform(0.0, 2.0, size=input_width), dtype=_F64
            )
            inp, spec, bounds = _input_pair(
                input_width, lower, upper
            )
            weight = np.ascontiguousarray(
                rng.normal(size=(output_width, input_width)), dtype=_F64
            )
            bias = np.ascontiguousarray(
                rng.normal(size=output_width), dtype=_F64
            )
            dense = _layer(
                2,
                "DENSE",
                output_width,
                {
                    "weight": weight,
                    "bias": bias,
                    "in_features": input_width,
                    "out_features": output_width,
                },
            )
            assertion = _assert_layer(3, output_width)
            net = _net(
                [inp, spec, dense, assertion],
                {0: [], 1: [0], 2: [1], 3: [2]},
            )
            positive = np.maximum(weight, 0.0)
            negative = np.minimum(weight, 0.0)
            output_lower = positive @ lower + negative @ upper + bias
            output_upper = positive @ upper + negative @ lower + bias
            bounds[2] = _box(
                np.nextafter(output_lower, -math.inf),
                np.nextafter(output_upper, math.inf),
            )
            queries = np.ascontiguousarray(
                rng.normal(size=(query_count, output_width)),
                dtype=_F64,
            )
            with self.subTest(case=case):
                _compare_v3_v5_fraction(
                    self,
                    net,
                    bounds,
                    query_rows=queries,
                    chunk_size=17,
                )
            audited_objectives += query_count
        self.assertGreaterEqual(audited_objectives, 5_000)

    def test_subnormal_and_exact_zero_do_not_share_a_shortcut(self):
        eta = float(np.nextafter(_F64(0.0), _F64(math.inf)))
        inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
        dense = _layer(
            2,
            "DENSE",
            1,
            {
                "weight": np.asarray([[eta]], dtype=_F64),
                "bias": np.zeros(1, dtype=_F64),
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
        _, nonzero, _ = _compare_v3_v5_fraction(
            self, net, bounds, one_hot=0
        )
        guard = float.fromhex(
            nonzero.receipt["affine_executions"][0][
                "scalar_guard_hex"
            ][0]
        )
        self.assertGreater(guard, 0.0)

        zero_query = np.zeros((1, 1), dtype=_F64)
        _, zero, _ = _compare_v3_v5_fraction(
            self, net, bounds, query_rows=zero_query
        )
        self.assertEqual(
            zero.receipt["affine_executions"][0]["scalar_guard_hex"],
            [0.0.hex()],
        )

    def test_integrity_tamper_deadline_and_v3_schema_isolation(self):
        net, bounds = _point_dense_toy()
        kwargs = {
            "one_hot": {
                "indices": [0, 1],
                "signs": [1.0, -1.0],
            }
        }
        v3 = frozen.replay_query_lower_bounds(net, bounds, **kwargs)
        v5 = replay_query_lower_bounds_v5_candidate(
            net, bounds, **kwargs
        )
        self.assertTrue(v3.proof_authority)
        self.assertNotEqual(
            v3.receipt.get("schema"), v5.receipt.get("schema")
        )
        self.assertFalse(v5.proof_authority)

        tampered_receipt = copy.deepcopy(dict(v5.receipt))
        tampered_receipt["affine_executions"][0][
            "scalar_guard_hex"
        ][0] = 0.0.hex()
        tampered = QueryDualReplayV5CandidateResult(
            lower_bounds=v5.lower_bounds,
            receipt=tampered_receipt,
        )
        self.assertFalse(
            verify_query_dual_replay_v5_candidate(tampered)
        )

        with self.assertRaises(frozen.QueryDualReplayTimeout):
            replay_query_lower_bounds_v5_candidate(
                net, bounds, timeout_s=0.0, **kwargs
            )


if __name__ == "__main__":
    unittest.main()
