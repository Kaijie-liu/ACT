"""End-to-end controlled gates for the non-authoritative V5.1 replay."""

from __future__ import annotations

import copy
from fractions import Fraction
import math
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51 as v51
from act.back_end.hybridz_tf.query_dual_replay_v51 import (
    QueryDualReplayV51CandidateResult,
    replay_query_lower_bounds_v51_candidate,
    verify_query_dual_replay_v51_candidate,
)
from act.back_end.hybridz_tf.query_dual_scalar_guard_v51 import (
    QueryDualScalarGuardV51Error,
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
_ETA = float(np.nextafter(_F64(0.0), _F64(math.inf)))


def _assert_below_fraction(
    test: unittest.TestCase,
    numeric: np.ndarray,
    exact: tuple[Fraction, ...],
) -> None:
    test.assertEqual(numeric.size, len(exact))
    for stored, oracle in zip(numeric, exact):
        test.assertTrue(math.isfinite(float(stored)))
        test.assertLessEqual(
            Fraction.from_float(float(stored)), oracle
        )


def _compare_v3_v51(
    test: unittest.TestCase,
    net,
    bounds,
    **kwargs,
):
    old = frozen.replay_query_lower_bounds(net, bounds, **kwargs)
    new = replay_query_lower_bounds_v51_candidate(
        net, bounds, **kwargs
    )
    oracle_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"chunk_size", "max_workspace_bytes"}
    }
    exact = frozen.fraction_replay_lower_bounds(
        net, bounds, **oracle_kwargs
    )
    _assert_below_fraction(test, old.lower_bounds, exact)
    _assert_below_fraction(test, new.lower_bounds, exact)
    test.assertTrue(np.all(new.lower_bounds >= old.lower_bounds))
    test.assertTrue(verify_query_dual_replay_v51_candidate(new))
    test.assertFalse(new.proof_authority)
    test.assertFalse(new.lower_bounds.flags.writeable)
    return old, new, exact


def _rehash_receipt(body):
    body["guard_ledger_sha256"] = v51._json_sha256(
        body["affine_executions"]
    )
    body.pop("receipt_sha256", None)
    body["receipt_sha256"] = v51._json_sha256(body)


def _threshold_conv_toy():
    inp, spec, bounds = _input_pair(
        16,
        [-1.0] * 16,
        [1.0] * 16,
        shape=(1, 1, 4, 4),
    )
    conv = _layer(
        2,
        "CONV2D",
        16,
        {
            "weight": np.asarray([[[[1.25]]]], dtype=_F64),
            "bias": np.asarray([0.0], dtype=_F64),
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, 1, 4, 4),
            "output_shape": (1, 1, 4, 4),
        },
    )
    assertion = _assert_layer(3, 16)
    net = _net(
        [inp, spec, conv, assertion],
        {0: [], 1: [0], 2: [1], 3: [2]},
    )
    bounds[2] = _box([-1.25] * 16, [1.25] * 16)
    return net, bounds


def _subnormal_dense_toy():
    inp, spec, bounds = _input_pair(1, [-1.0], [1.0])
    dense = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[0.5]], dtype=_F64),
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
    bounds[2] = _box([-0.5], [0.5])
    return net, bounds


class QueryDualReplayV51Tests(unittest.TestCase):
    @staticmethod
    def _prepared(net, bounds, queries):
        return frozen._prepare(
            net,
            bounds,
            start_lid=None,
            query_rows=queries,
            one_hot=None,
            query_bias=None,
            alpha_by_relu=None,
            deadline=frozen._Deadline.build(None, None),
            expected_net_sha256=None,
            expected_bounds_sha256=None,
            expected_query_sha256=None,
            expected_alpha_sha256=None,
        )

    def test_dense_relu_residual_and_dense_conv_fraction_gate(self):
        net, bounds = _point_dense_toy()
        _compare_v3_v51(
            self,
            net,
            bounds,
            one_hot={"indices": [0, 1], "signs": [1.0, -1.0]},
            query_bias=np.asarray([0.25, -0.5], dtype=_F64),
        )

        net, bounds = _single_relu_toy()
        _compare_v3_v51(
            self,
            net,
            bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={
                3: np.asarray([[[0.75], [0.125]]], dtype=_F64)
            },
        )

        net, bounds = _residual_toy()
        _compare_v3_v51(
            self,
            net,
            bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={3: np.asarray([0.5], dtype=_F64)},
        )

        net, bounds = _conv_toy()
        _, result, _ = _compare_v3_v51(
            self,
            net,
            bounds,
            query_rows=np.asarray(
                [[1.0, -1.0], [0.3, 0.7], [-0.25, 1.5]],
                dtype=_F64,
            ),
        )
        record = result.receipt["affine_executions"][0]
        self.assertEqual(record["operator"], "CONV2D")
        self.assertEqual(record["conv_branch"], "dense")
        self.assertEqual(record["scalar_guard_policy_count"], 1)
        self.assertEqual(
            record["componentwise_radius_policy_count"], 0
        )
        self.assertEqual(record["active_count"], 3)
        self.assertEqual(record["fallback_count"], 0)
        self.assertIn("plan_sha256", record)
        self.assertIn("support_sha256", record)

    def test_dense_mixed_zero_rows_are_row_local(self):
        net, bounds = _point_dense_toy()
        queries = np.asarray(
            [[0.0, -0.0], [1.0, -1.0]], dtype=_F64
        )
        old, new, _ = _compare_v3_v51(
            self, net, bounds, query_rows=queries
        )
        record = new.receipt["affine_executions"][0]
        self.assertEqual(record["active_mask"], [False, True])
        self.assertEqual(
            record["scalar_applied_mask"], [False, True]
        )
        self.assertEqual(record["scalar_subtraction_rows"], 1)
        # V3 subtracts a block-wide guard when any sibling row is active;
        # V5.1 intentionally removes that cross-row zero contamination.
        self.assertGreater(new.lower_bounds[0], old.lower_bounds[0])

    def test_conv_mixed_zero_rows_are_row_local(self):
        net, bounds = _conv_toy()
        queries = np.asarray(
            [[0.0, -0.0], [1.0, -1.0]], dtype=_F64
        )
        old, new, _ = _compare_v3_v51(
            self, net, bounds, query_rows=queries
        )
        record = new.receipt["affine_executions"][0]
        self.assertEqual(record["conv_branch"], "dense")
        self.assertEqual(record["active_mask"], [False, True])
        self.assertEqual(
            record["scalar_applied_mask"], [False, True]
        )
        self.assertEqual(record["scalar_subtraction_rows"], 1)
        self.assertGreater(new.lower_bounds[0], old.lower_bounds[0])

    def test_conv_threshold_three_neighbours(self):
        net, bounds = _threshold_conv_toy()
        for nonzero, expected_branch in (
            (1, "sparse"),
            (2, "sparse"),
            (3, "dense"),
        ):
            query = np.zeros((1, 16), dtype=_F64)
            query[0, :nonzero] = 1.0
            with self.subTest(nonzero=nonzero):
                old, new, _ = _compare_v3_v51(
                    self, net, bounds, query_rows=query
                )
                record = new.receipt["affine_executions"][0]
                self.assertEqual(
                    record["conv_branch"], expected_branch
                )
                self.assertEqual(
                    record["threshold_lhs"], 8 * nonzero
                )
                self.assertEqual(record["threshold_rhs"], 16)
                if expected_branch == "sparse":
                    self.assertEqual(
                        record["componentwise_radius_policy_count"],
                        1,
                    )
                    self.assertEqual(
                        new.receipt["conv_plan_count"], 0
                    )
                    np.testing.assert_array_equal(
                        new.lower_bounds.view(np.uint64),
                        old.lower_bounds.view(np.uint64),
                    )
                else:
                    self.assertEqual(
                        record["scalar_guard_policy_count"], 1
                    )
                    self.assertEqual(
                        new.receipt["conv_plan_count"], 1
                    )

    def test_dense_underflow_fallback_is_bound_in_receipt(self):
        net, bounds = _subnormal_dense_toy()
        query = np.asarray([[_ETA]], dtype=_F64)
        _, result, _ = _compare_v3_v51(
            self, net, bounds, query_rows=query
        )
        record = result.receipt["affine_executions"][0]
        self.assertEqual(record["active_mask"], [True])
        self.assertEqual(record["fallback_mask"], [True])
        self.assertEqual(record["fallback_count"], 1)
        self.assertTrue(record["fallback_reasons"][0])
        self.assertIn(
            "coefficient_subnormal",
            record["fallback_reasons"][0],
        )
        self.assertEqual(record["scalar_applied_mask"], [True])
        self.assertIn("catalog_sha256", record)
        self.assertIn("helper_receipt_sha256", record)

    def test_affine_nominal_receipts_match_frozen_v3_bits(self):
        dense_net, dense_bounds = _point_dense_toy()
        dense_queries = np.asarray(
            [[1.0, 0.0], [0.25, -1.5]], dtype=_F64
        )
        dense_prepared = self._prepared(
            dense_net, dense_bounds, dense_queries
        )
        dense_layer = dense_prepared.layers[
            dense_prepared.output_id
        ]
        expected_dense, _ = frozen._matrix_product_with_error(
            dense_prepared.queries, dense_layer.params["weight"]
        )
        dense_result = replay_query_lower_bounds_v51_candidate(
            dense_net,
            dense_bounds,
            query_rows=dense_queries,
        )
        self.assertEqual(
            dense_result.receipt["affine_executions"][0][
                "nominal_sha256"
            ],
            frozen._array_digest(expected_dense),
        )

        conv_net, conv_bounds = _conv_toy()
        conv_queries = np.asarray(
            [[1.0, -1.0], [0.3, 0.7]], dtype=_F64
        )
        conv_prepared = self._prepared(
            conv_net, conv_bounds, conv_queries
        )
        conv_layer = conv_prepared.layers[conv_prepared.output_id]
        expected_conv, _ = frozen._conv_reverse_with_error(
            conv_prepared.queries,
            conv_layer,
            frozen._Deadline.build(None, None),
            frozen._ReplayStats(),
        )
        conv_result = replay_query_lower_bounds_v51_candidate(
            conv_net,
            conv_bounds,
            query_rows=conv_queries,
        )
        self.assertEqual(
            conv_result.receipt["affine_executions"][0][
                "nominal_sha256"
            ],
            frozen._array_digest(expected_conv),
        )

    def test_chunk_spans_and_inconsistent_rehashed_span_mutation(self):
        net, bounds = _point_dense_toy()
        queries = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=_F64,
        )
        result = replay_query_lower_bounds_v51_candidate(
            net,
            bounds,
            query_rows=queries,
            chunk_size=2,
        )
        self.assertTrue(verify_query_dual_replay_v51_candidate(result))
        self.assertFalse(result.receipt["semantic_authority"])
        self.assertEqual(
            result.receipt["integrity_scope"],
            "unkeyed_internal_consistency_only",
        )
        manifest = result.receipt["execution_span_manifest"]
        self.assertEqual(manifest[0]["spans"], [[0, 2], [2, 4], [4, 5]])
        bindings = result.receipt["query_span_bindings"]
        self.assertEqual(
            [
                [binding["query_start"], binding["query_end"]]
                for binding in bindings
            ],
            [[0, 2], [2, 4], [4, 5]],
        )
        self.assertEqual(
            result.receipt["query_span_bindings_sha256"],
            v51._json_sha256(bindings),
        )
        global_block = result.receipt["query_block_sha256"]
        for record in result.receipt["affine_executions"]:
            self.assertEqual(
                record["query_block_sha256"], global_block
            )
            self.assertIn("input_coefficient_sha256", record)
            self.assertIn("query_rows_sha256", record)
            self.assertIn("query_bias_sha256", record)
            self.assertIn("alpha_slice_sha256", record)
            self.assertIn("active_mask_sha256", record)
            self.assertIn("fallback_mask_sha256", record)
            span = next(
                binding
                for binding in bindings
                if binding["query_start"] == record["query_start"]
                and binding["query_end"] == record["query_end"]
            )
            self.assertEqual(
                record["query_rows_sha256"],
                span["query_rows_sha256"],
            )
            self.assertEqual(
                record["query_bias_sha256"],
                span["query_bias_sha256"],
            )
            self.assertEqual(
                record["alpha_slice_sha256"],
                span["alpha_slice_sha256"],
            )
            self.assertEqual(
                record["input_coefficient_sha256"],
                record["query_rows_sha256"],
            )

        body = copy.deepcopy(dict(result.receipt))
        body["affine_executions"][1]["query_start"] = 1
        body["affine_executions"][1]["query_end"] = 3
        _rehash_receipt(body)
        forged = QueryDualReplayV51CandidateResult(
            lower_bounds=result.lower_bounds,
            receipt=MappingProxyType(body),
        )
        self.assertFalse(
            verify_query_dual_replay_v51_candidate(forged)
        )

        residual_net, residual_bounds = _residual_toy()
        residual = replay_query_lower_bounds_v51_candidate(
            residual_net,
            residual_bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={3: np.asarray([0.5], dtype=_F64)},
            chunk_size=1,
        )
        self.assertTrue(
            verify_query_dual_replay_v51_candidate(residual)
        )
        residual_records = residual.receipt["affine_executions"]
        self.assertEqual(
            len(
                {
                    record["query_block_sha256"]
                    for record in residual_records
                }
            ),
            1,
        )
        for binding in residual.receipt["query_span_bindings"]:
            matching = [
                record
                for record in residual_records
                if record["query_start"] == binding["query_start"]
                and record["query_end"] == binding["query_end"]
            ]
            self.assertEqual(len(matching), 2)
            self.assertEqual(
                {
                    (
                        record["query_rows_sha256"],
                        record["query_bias_sha256"],
                        record["alpha_slice_sha256"],
                    )
                    for record in matching
                },
                {
                    (
                        binding["query_rows_sha256"],
                        binding["query_bias_sha256"],
                        binding["alpha_slice_sha256"],
                    )
                },
            )

    def test_inconsistently_rehashed_query_material_substitutions_fail(self):
        dense_net, dense_bounds = _point_dense_toy()
        dense_result = replay_query_lower_bounds_v51_candidate(
            dense_net,
            dense_bounds,
            query_rows=np.asarray(
                [[1.0, 0.0], [0.0, -1.0]], dtype=_F64
            ),
        )
        for field, replacement in (
            ("query_rows_sha256", "1" * 64),
            ("query_bias_sha256", "2" * 64),
            ("input_coefficient_sha256", "3" * 64),
        ):
            body = copy.deepcopy(dict(dense_result.receipt))
            body["affine_executions"][0][field] = replacement
            _rehash_receipt(body)
            forged = QueryDualReplayV51CandidateResult(
                lower_bounds=dense_result.lower_bounds,
                receipt=MappingProxyType(body),
            )
            with self.subTest(field=field):
                self.assertFalse(
                    verify_query_dual_replay_v51_candidate(forged)
                )

        relu_net, relu_bounds = _single_relu_toy()
        relu_result = replay_query_lower_bounds_v51_candidate(
            relu_net,
            relu_bounds,
            query_rows=np.asarray([[1.0], [-1.0]], dtype=_F64),
            alpha_by_relu={
                3: np.asarray([[[0.75], [0.125]]], dtype=_F64)
            },
        )
        body = copy.deepcopy(dict(relu_result.receipt))
        body["affine_executions"][0][
            "alpha_slice_sha256"
        ] = "4" * 64
        _rehash_receipt(body)
        forged = QueryDualReplayV51CandidateResult(
            lower_bounds=relu_result.lower_bounds,
            receipt=MappingProxyType(body),
        )
        self.assertFalse(
            verify_query_dual_replay_v51_candidate(forged)
        )

    def test_receipt_mask_support_and_lower_bound_mutations(self):
        net, bounds = _point_dense_toy()
        result = replay_query_lower_bounds_v51_candidate(
            net,
            bounds,
            query_rows=np.asarray(
                [[0.0, 0.0], [1.0, -1.0]], dtype=_F64
            ),
        )
        for mutation in ("mask", "support", "lower"):
            body = copy.deepcopy(dict(result.receipt))
            values = result.lower_bounds
            if mutation == "mask":
                body["affine_executions"][0]["active_mask"][0] = True
            elif mutation == "support":
                body["affine_executions"][0][
                    "support_sha256"
                ] = "0" * 64
            else:
                changed = np.asarray(values).copy()
                changed[0] = np.nextafter(changed[0], math.inf)
                changed.setflags(write=False)
                values = changed
            forged = QueryDualReplayV51CandidateResult(
                lower_bounds=values,
                receipt=MappingProxyType(body),
            )
            with self.subTest(mutation=mutation):
                self.assertFalse(
                    verify_query_dual_replay_v51_candidate(forged)
                )

        body = copy.deepcopy(dict(result.receipt))
        body["affine_executions"][0][
            "support_sha256"
        ] = "not-even-a-sha"
        _rehash_receipt(body)
        forged = QueryDualReplayV51CandidateResult(
            lower_bounds=result.lower_bounds,
            receipt=MappingProxyType(body),
        )
        self.assertFalse(
            verify_query_dual_replay_v51_candidate(forged)
        )

    def test_private_observer_dense_and_sparse_payloads(self):
        def run_with_observer(net, bounds, queries):
            events = []

            def observer(**event):
                for name, value in event.items():
                    if name != "record":
                        self.assertFalse(value.flags.writeable)
                events.append(
                    {
                        name: (
                            dict(value)
                            if name == "record"
                            else np.asarray(value).copy()
                        )
                        for name, value in event.items()
                    }
                )

            prepared = self._prepared(net, bounds, queries)
            context = v51._V51Context(
                prepared=prepared,
                execution_observer=observer,
            )
            stats = frozen._ReplayStats()
            stats.configure_queries(queries.shape[0])
            lower = v51._replay_block_v51(
                context, 0, queries.shape[0], stats
            )
            return lower, events, context

        dense_net, dense_bounds = _point_dense_toy()
        dense_queries = np.asarray(
            [[0.0, 0.0], [1.0, -1.0]], dtype=_F64
        )
        _, dense_events, dense_context = run_with_observer(
            dense_net, dense_bounds, dense_queries
        )
        self.assertEqual(len(dense_events), 1)
        dense_event = dense_events[0]
        self.assertEqual(
            set(dense_event),
            {
                "record",
                "nominal",
                "scalar_before",
                "scalar_after",
                "scalar_guard",
            },
        )
        expected_after, _ = v51._row_local_subtract(
            dense_event["scalar_before"],
            dense_event["scalar_guard"],
            np.asarray(
                dense_event["record"]["active_mask"],
                dtype=np.bool_,
            ),
            where="observer test",
        )
        np.testing.assert_array_equal(
            dense_event["scalar_after"].view(np.uint64),
            expected_after.view(np.uint64),
        )
        self.assertEqual(len(dense_context.executions), 1)

        sparse_net, sparse_bounds = _threshold_conv_toy()
        sparse_queries = np.zeros((1, 16), dtype=_F64)
        sparse_queries[0, 0] = 1.0
        _, sparse_events, sparse_context = run_with_observer(
            sparse_net, sparse_bounds, sparse_queries
        )
        self.assertEqual(len(sparse_events), 1)
        sparse_event = sparse_events[0]
        self.assertEqual(
            set(sparse_event),
            {
                "record",
                "nominal",
                "scalar_before",
                "scalar_after",
                "componentwise_radius",
                "componentwise_penalty",
            },
        )
        self.assertEqual(
            sparse_event["record"]["conv_branch"], "sparse"
        )
        reference_stats = frozen._ReplayStats()
        expected_sparse_after = frozen._absorb_radius(
            sparse_event["scalar_before"],
            sparse_event["componentwise_radius"],
            frozen._output_box(
                sparse_context.prepared,
                sparse_event["record"]["predecessor_id"],
            ),
            reference_stats,
        )
        np.testing.assert_array_equal(
            sparse_event["scalar_after"].view(np.uint64),
            expected_sparse_after.view(np.uint64),
        )

    def test_sparse_absorb_helper_matches_frozen_bits(self):
        rng = np.random.default_rng(2026072852)
        for case_index in range(32):
            rows = int(rng.integers(1, 6))
            width = int(rng.integers(1, 10))
            scalar = np.ascontiguousarray(
                rng.normal(size=rows), dtype=_F64
            )
            radius = np.ascontiguousarray(
                np.abs(rng.normal(scale=1e-12, size=(rows, width))),
                dtype=_F64,
            )
            if case_index % 4 == 0:
                radius.fill(0.0)
            lower = -np.abs(rng.normal(size=width))
            upper = np.abs(rng.normal(size=width))
            if case_index % 5 == 0:
                lower[0] = 0.0
                upper[0] = 0.0
            box = frozen._Box(
                lb=np.ascontiguousarray(lower, dtype=_F64),
                ub=np.ascontiguousarray(upper, dtype=_F64),
            )
            frozen_stats = frozen._ReplayStats()
            integrated_stats = frozen._ReplayStats()
            expected = frozen._absorb_radius(
                scalar.copy(), radius, box, frozen_stats
            )
            actual, penalty = v51._absorb_radius_with_penalty(
                scalar.copy(), radius, box, integrated_stats
            )
            with self.subTest(case=case_index):
                np.testing.assert_array_equal(
                    actual.view(np.uint64),
                    expected.view(np.uint64),
                )
                self.assertEqual(
                    integrated_stats.coefficient_guards,
                    frozen_stats.coefficient_guards,
                )
                self.assertEqual(
                    integrated_stats.guard_total.hex(),
                    frozen_stats.guard_total.hex(),
                )
                self.assertTrue(np.all(penalty >= 0.0))
                self.assertTrue(np.all(np.isfinite(penalty)))

    def test_deadline_schema_and_input_validation(self):
        net, bounds = _point_dense_toy()
        with self.assertRaises(frozen.QueryDualReplayTimeout):
            replay_query_lower_bounds_v51_candidate(
                net, bounds, one_hot=0, timeout_s=0.0
            )
        with self.assertRaisesRegex(
            frozen.QueryDualReplayError, "INVALID_CHUNK"
        ):
            replay_query_lower_bounds_v51_candidate(
                net, bounds, one_hot=0, chunk_size=0
            )
        with mock.patch.object(
            v51,
            "dense_support_compressed_guard_v51",
            side_effect=QueryDualScalarGuardV51Error(
                "DEADLINE_EXPIRED", "injected helper expiry"
            ),
        ):
            with self.assertRaises(frozen.QueryDualReplayTimeout):
                replay_query_lower_bounds_v51_candidate(
                    net, bounds, one_hot=0
                )
        old = frozen.replay_query_lower_bounds(
            net, bounds, one_hot=0
        )
        new = replay_query_lower_bounds_v51_candidate(
            net, bounds, one_hot=0
        )
        self.assertTrue(old.proof_authority)
        self.assertFalse(new.proof_authority)
        self.assertNotEqual(
            old.receipt.get("schema"), new.receipt.get("schema")
        )

    def test_fixed_random_dense_fraction_and_no_regression(self):
        rng = np.random.default_rng(2026072851)
        audited = 0
        for case_index in range(20):
            input_width = int(rng.integers(1, 7))
            output_width = int(rng.integers(1, 7))
            input_lower = rng.uniform(
                -2.0, 0.0, size=input_width
            )
            input_upper = rng.uniform(
                0.0, 2.0, size=input_width
            )
            inp, spec, bounds = _input_pair(
                input_width, input_lower, input_upper
            )
            weight = np.ascontiguousarray(
                rng.normal(size=(output_width, input_width)),
                dtype=_F64,
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
            lower = np.asarray(input_lower, dtype=_F64)
            upper = np.asarray(input_upper, dtype=_F64)
            positive = np.maximum(weight, 0.0)
            negative = np.minimum(weight, 0.0)
            bounds[2] = _box(
                np.nextafter(
                    positive @ lower + negative @ upper + bias,
                    -math.inf,
                ),
                np.nextafter(
                    positive @ upper + negative @ lower + bias,
                    math.inf,
                ),
            )
            queries = np.ascontiguousarray(
                rng.normal(size=(10, output_width)), dtype=_F64
            )
            with self.subTest(case=case_index):
                _compare_v3_v51(
                    self,
                    net,
                    bounds,
                    query_rows=queries,
                    chunk_size=3,
                )
            audited += queries.shape[0]
        self.assertEqual(audited, 200)


if __name__ == "__main__":
    unittest.main()
