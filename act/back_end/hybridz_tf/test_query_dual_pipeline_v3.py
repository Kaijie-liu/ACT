#!/usr/bin/env python3
"""Controlled transaction and red-team tests for sealed sparse V3."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import time
import unittest
from unittest import mock

import numpy as np
import torch

from act.back_end.hybridz_tf.property_residual_targets import (
    _binary64_sha256,
    plan_sparse_query_rows_from_property_adjoints,
)
from act.back_end.hybridz_tf.query_dual_pipeline import (
    QueryDualPipelineError,
    _array_digest,
    _json_sha256,
    _receipt,
    build_verified_query_dual_feedback,
    validate_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.query_dual_candidates import (
    _receipt_with_sha256 as _candidate_receipt,
    generate_query_dual_candidates,
)
from act.back_end.hybridz_tf import query_dual_pipeline_v3 as v3_module
from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
    build_verified_query_dual_feedback_v3,
    validate_verified_query_dual_feedback_v3,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    create_query_dual_replay_session,
    fraction_replay_lower_bounds,
    replay_query_lower_bounds,
    verify_query_dual_replay_receipt,
)
from act.back_end.hybridz_tf.test_query_dual_pipeline import (
    _IntervalCandidateSolver,
    _pipeline_net,
)
from act.back_end.hybridz_tf.test_query_dual_box_certifier import (
    _grouped_conv_net,
    _input_pair,
    _layer,
    _net,
    _residual_net,
)
from act.util.device_manager import initialize_device


def _toy_selector(**kwargs):
    rows = np.asarray(kwargs["C"], dtype=np.float64)
    thresholds = np.asarray(kwargs["thresholds"], dtype=np.float64)
    kind = str(kwargs["kind"])
    before = kwargs["before"]
    quotas = kwargs["layer_quotas"]
    rival_count = int(rows.shape[0])
    adjoints = {}
    for layer_id in sorted(quotas):
        width = int(before[layer_id].lb.numel())
        values = torch.arange(
            1,
            rival_count * width + 1,
            dtype=torch.float64,
        ).reshape(rival_count, width)
        if layer_id & 1:
            values = torch.flip(values, dims=(1,))
        adjoints[int(layer_id)] = values
    return plan_sparse_query_rows_from_property_adjoints(
        adjoints,
        before,
        layer_quotas=quotas,
        rival_ids=tuple(range(rival_count)),
        rival_hardness=tuple(
            float(rival_count - value) for value in range(rival_count)
        ),
        all_rivals_processed=True,
        property_sha256=_binary64_sha256(
            rows, thresholds, kind=kind
        ),
        pool_per_rival=int(kwargs["pool_per_rival"]),
    )


def _build():
    net = _pipeline_net()
    rows = np.asarray([[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64)
    thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
    bundle = build_verified_query_dual_feedback_v3(
        net,
        rows,
        thresholds,
        target_relu_ids=(5, 7),
        stage_quotas=(1, 1),
        steps=2,
        block_size=1,
        replay_chunk_size=2,
        candidate_device="cpu",
        timeout_s=10.0,
        selector=_toy_selector,
        solver_factory=_IntervalCandidateSolver,
    )
    return net, rows, thresholds, bundle


def _rechain_bundle(bundle):
    previous = bundle.root_certificate.receipt["receipt_sha256"]
    for stage in bundle.stages:
        candidate_sha256 = stage.candidate_receipt["receipt_sha256"]
        stage.receipt["candidate_receipt_sha256"] = candidate_sha256
        stage.receipt["candidate_descriptor_coverage_sha256"] = (
            stage.candidate_receipt["descriptor_coverage_sha256"]
        )
        stage.receipt["parent_chain_sha256"] = previous
        result_chain = _json_sha256(
            {
                "previous": previous,
                "stage_index": int(stage.stage_index),
                "parent_boxes_sha256": stage.parent_boxes_sha256,
                "result_boxes_sha256": stage.result_boxes_sha256,
                "candidate_receipt_sha256": candidate_sha256,
                "bounds_frame_sha256": stage.receipt[
                    "bounds_frame_sha256"
                ],
            }
        )
        stage.receipt["result_chain_sha256"] = result_chain
        stage.receipt.update(_receipt(stage.receipt))
        previous = result_chain

    property_stage = bundle.property_stage
    candidate_sha256 = property_stage.candidate_receipt["receipt_sha256"]
    property_stage.receipt["candidate_receipt_sha256"] = candidate_sha256
    property_stage.receipt[
        "candidate_descriptor_coverage_sha256"
    ] = property_stage.candidate_receipt["descriptor_coverage_sha256"]
    property_stage.receipt["parent_chain_sha256"] = previous
    result_chain = _json_sha256(
        {
            "previous": previous,
            "stage": "property",
            "parent_boxes_sha256": property_stage.parent_boxes_sha256,
            "candidate_receipt_sha256": candidate_sha256,
            "bounds_frame_sha256": property_stage.receipt[
                "bounds_frame_sha256"
            ],
            "property_spec_sha256": property_stage.property_spec_sha256,
            "property_upper_sha256": _array_digest(
                property_stage.property_upper
            ),
        }
    )
    property_stage.receipt["result_chain_sha256"] = result_chain
    property_stage.receipt.update(_receipt(property_stage.receipt))

    bundle.receipt["stage_receipt_sha256"] = [
        stage.receipt["receipt_sha256"] for stage in bundle.stages
    ]
    bundle.receipt["target_candidate_receipt_sha256"] = [
        stage.candidate_receipt["receipt_sha256"]
        for stage in bundle.stages
    ]
    bundle.receipt["property_receipt_sha256"] = property_stage.receipt[
        "receipt_sha256"
    ]
    bundle.receipt["property_candidate_receipt_sha256"] = (
        property_stage.candidate_receipt["receipt_sha256"]
    )
    bundle.receipt["final_stage_chain_sha256"] = result_chain
    bundle.receipt.update(_receipt(bundle.receipt))


def _mutate_candidate_descriptor(receipt, field, value):
    receipt["descriptor_records"][0][field] = value
    coverage = _json_sha256(receipt["descriptor_records"])
    receipt["descriptor_records_sha256"] = coverage
    receipt["descriptor_coverage_sha256"] = coverage


def _rehash_replay_receipt(block):
    receipt = block.replay_receipt
    sealed = receipt["sealed_context"]
    sealed["manifest_crosswalk_sha256"] = _json_sha256(
        sealed["manifest_crosswalk"]
    )
    context_body = dict(sealed)
    context_body.pop("context_sha256", None)
    sealed["context_sha256"] = _json_sha256(context_body)
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _json_sha256(receipt_body)


def _rebind_replay_receipts(bundle):
    for stage in bundle.stages:
        stage.receipt["block_receipt_sha256"] = [
            block.replay_receipt["receipt_sha256"]
            for block in stage.blocks
        ]
    bundle.property_stage.receipt["block_receipt_sha256"] = [
        block.replay_receipt["receipt_sha256"]
        for block in bundle.property_stage.blocks
    ]
    _rechain_bundle(bundle)


def _residual_tightness_net():
    inp, spec = _input_pair(1, [-1.0], [1.0])
    split = _layer(
        2,
        "DENSE",
        2,
        {
            "weight": np.asarray([[1.0], [-1.0]], dtype=np.float64),
            "bias": np.asarray([0.0, 0.0], dtype=np.float64),
        },
    )
    absolute_parts = _layer(3, "RELU", 2)
    shifted_sum = _layer(
        4,
        "DENSE",
        1,
        {
            "weight": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "bias": np.asarray([-0.75], dtype=np.float64),
        },
    )
    output = _layer(5, "RELU", 1)
    assertion = _layer(6, "ASSERT", 1, {"kind": "AUDIT"})
    return _net(
        [inp, spec, split, absolute_parts, shifted_sum, output, assertion],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4],
            6: [5],
        },
    )


def _parallel_residual_tightness_net(width=80):
    width = int(width)
    inp, spec = _input_pair(
        width, np.full(width, -1.0), np.full(width, 1.0)
    )
    split_weight = np.vstack(
        [np.eye(width, dtype=np.float64), -np.eye(width, dtype=np.float64)]
    )
    split = _layer(
        2,
        "DENSE",
        2 * width,
        {
            "weight": split_weight,
            "bias": np.zeros(2 * width, dtype=np.float64),
        },
    )
    absolute_parts = _layer(3, "RELU", 2 * width)
    sum_weight = np.zeros((width, 2 * width), dtype=np.float64)
    indices = np.arange(width)
    sum_weight[indices, indices] = 1.0
    sum_weight[indices, width + indices] = 1.0
    shifted_sum = _layer(
        4,
        "DENSE",
        width,
        {
            "weight": sum_weight,
            "bias": np.full(width, -0.75, dtype=np.float64),
        },
    )
    output = _layer(5, "RELU", width)
    assertion = _layer(6, "ASSERT", width, {"kind": "AUDIT"})
    return _net(
        [inp, spec, split, absolute_parts, shifted_sum, output, assertion],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4],
            6: [5],
        },
    )


class QueryDualPipelineV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prior_device = torch.get_default_device()
        cls._prior_dtype = torch.get_default_dtype()
        initialize_device(device="cpu", dtype="float64")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._prior_dtype)
        torch.set_default_device(cls._prior_device)

    def test_live_sparse_transaction(self):
        net, rows, thresholds, bundle = _build()
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )
        self.assertEqual(
            bundle.receipt["schema"],
            "act.verified_query_dual_feedback.v3",
        )
        self.assertTrue(
            all(
                stage.receipt["selected_coverage_complete"]
                and stage.receipt[
                    "unselected_bounds_bit_identical_parent"
                ]
                for stage in bundle.stages
            )
        )
        self.assertEqual(
            bundle.property_stage.receipt["coverage_complete"], True
        )
        session_ids = {
            block.replay_receipt["sealed_context"][
                "session_nonce_sha256"
            ]
            for stage in bundle.stages
            for block in stage.blocks
        }
        session_ids.update(
            block.replay_receipt["sealed_context"][
                "session_nonce_sha256"
            ]
            for block in bundle.property_stage.blocks
        )
        self.assertEqual(len(session_ids), 1)

    def test_deepcopy_has_no_live_authority(self):
        net, rows, thresholds, bundle = _build()
        copied = replace(bundle)
        self.assertFalse(
            validate_verified_query_dual_feedback(
                copied,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )

    def test_real_single_adjoint_selector_path(self):
        net = _pipeline_net()
        for layer in net.layers:
            for name, value in list(layer.params.items()):
                if isinstance(value, np.ndarray):
                    layer.params[name] = torch.as_tensor(
                        value, dtype=torch.float64
                    )
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        bundle = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            stage_quotas=(1, 1),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
            solver_factory=_IntervalCandidateSolver,
        )
        self.assertEqual(
            bundle.receipt["selector_receipt"][
                "adjoint_solver_calls"
            ],
            1,
        )
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )

    def test_selector_kind_is_canonical_and_blank_is_rejected(self):
        net = _pipeline_net()
        for layer in net.layers:
            for name, value in list(layer.params.items()):
                if isinstance(value, np.ndarray):
                    layer.params[name] = torch.as_tensor(
                        value, dtype=torch.float64
                    )
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        bundle = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            stage_quotas=(1, 1),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            selector_kind="top1_robust",
            timeout_s=10.0,
            solver_factory=_IntervalCandidateSolver,
        )
        self.assertEqual(bundle.receipt["selector_kind"], "TOP1_ROBUST")
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )

        with self.assertRaises(QueryDualPipelineError) as caught:
            build_verified_query_dual_feedback_v3(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                stage_quotas=(1, 1),
                steps=2,
                block_size=1,
                replay_chunk_size=2,
                candidate_device="cpu",
                selector_kind=" \t ",
                timeout_s=10.0,
                selector=_toy_selector,
                solver_factory=_IntervalCandidateSolver,
            )
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_invalid_target_order_and_empty_quotas_fail_before_root(self):
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        invalid_cases = (
            ("reverse_targets", (7, 5), (1, 1), "INVALID_TARGETS"),
            ("all_zero_quotas", (5, 7), (0, 0), "INVALID_CONFIG"),
            ("empty_targets", (), (), "INVALID_CONFIG"),
        )
        for name, targets, quotas, expected_code in invalid_cases:
            with self.subTest(name=name):
                with mock.patch.object(
                    v3_module, "_TRUSTED_CERTIFIER"
                ) as certifier:
                    with self.assertRaises(
                        QueryDualPipelineError
                    ) as caught:
                        build_verified_query_dual_feedback_v3(
                            _pipeline_net(),
                            rows,
                            thresholds,
                            target_relu_ids=targets,
                            stage_quotas=quotas,
                            steps=2,
                            block_size=1,
                            replay_chunk_size=2,
                            candidate_device="cpu",
                            timeout_s=10.0,
                            selector=_toy_selector,
                            solver_factory=_IntervalCandidateSolver,
                        )
                    self.assertEqual(
                        caught.exception.code, expected_code
                    )
                    certifier.assert_not_called()

        net = _pipeline_net()
        mixed = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(5, 7),
            stage_quotas=(0, 1),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        self.assertEqual(
            [stage.receipt["selected_row_ids"] for stage in mixed.stages],
            [[], [0]],
        )
        self.assertEqual(
            [
                stage.candidate_receipt["status"]
                for stage in mixed.stages
            ],
            ["no_queries_fallback", "descriptors_generated"],
        )
        self.assertTrue(
            validate_verified_query_dual_feedback_v3(
                mixed,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
            )
        )

        stable_net = _pipeline_net()
        stable_net.layers[6].params["bias"] = np.asarray(
            [10.0, 10.0], dtype=np.float64
        )
        stable = build_verified_query_dual_feedback_v3(
            stable_net,
            rows,
            thresholds,
            target_relu_ids=(7,),
            stage_quotas=(1,),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        self.assertEqual(
            stable.stages[0].candidate_receipt["status"],
            "no_queries_fallback",
        )
        self.assertEqual(
            stable.stages[0].receipt["eligible_row_ids"], []
        )
        self.assertEqual(
            stable.stages[0].receipt["selected_row_ids"], []
        )
        self.assertEqual(len(stable.property_stage.blocks), rows.shape[0])
        self.assertTrue(
            stable.property_stage.receipt["coverage_complete"]
        )
        self.assertTrue(
            validate_verified_query_dual_feedback_v3(
                stable,
                net=stable_net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(7,),
            )
        )

    def test_selector_receipt_semantic_matrix_stops_before_candidates(self):
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)

        def mutate_layer(field, value):
            def apply(receipt):
                receipt["layers"][0][field] = value

            return apply

        def append_extra_layer(receipt):
            extra = dict(receipt["layers"][0])
            extra["layer_id"] = 999
            receipt["layers"].append(extra)

        mutations = (
            ("status", lambda receipt: receipt.update(status="forged")),
            (
                "targets_selected",
                lambda receipt: receipt.update(
                    targets_selected=receipt["targets_selected"] + 1
                ),
            ),
            ("eligible_count", mutate_layer("eligible_count", 999)),
            ("selected_count", mutate_layer("selected_count", 999)),
            ("omitted_count", mutate_layer("omitted_count", 999)),
            ("quota_bool_alias", mutate_layer("quota", True)),
            (
                "selected_count_bool_alias",
                mutate_layer("selected_count", True),
            ),
            (
                "eligible_rows_sha256",
                mutate_layer("eligible_rows_sha256", "0" * 64),
            ),
            (
                "selected_rows_sha256",
                mutate_layer("selected_rows_sha256", "0" * 64),
            ),
            (
                "omitted_rows_sha256",
                mutate_layer("omitted_rows_sha256", "0" * 64),
            ),
            (
                "partition_complete",
                mutate_layer("partition_complete", False),
            ),
            (
                "partition_disjoint",
                mutate_layer("partition_disjoint", False),
            ),
            ("quota_filled", mutate_layer("quota_filled", False)),
            ("candidate_union", mutate_layer("candidate_union", 0)),
            ("extra_layer", append_extra_layer),
            (
                "selection_sha256",
                lambda receipt: receipt.update(
                    selection_sha256="0" * 64
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(field=name):
                candidate_generator = mock.Mock(
                    side_effect=AssertionError(
                        "candidate generation must not be reached"
                    )
                )

                def selector(**kwargs):
                    plan = _toy_selector(**kwargs)
                    mutate(plan.receipt)
                    return plan

                with self.assertRaises(
                    QueryDualPipelineError
                ) as caught:
                    build_verified_query_dual_feedback_v3(
                        _pipeline_net(),
                        rows,
                        thresholds,
                        target_relu_ids=(5, 7),
                        stage_quotas=(1, 1),
                        steps=2,
                        block_size=1,
                        replay_chunk_size=2,
                        candidate_device="cpu",
                        timeout_s=10.0,
                        selector=selector,
                        solver_factory=_IntervalCandidateSolver,
                        candidate_generator=candidate_generator,
                    )
                self.assertEqual(
                    caught.exception.code, "INVALID_SELECTOR"
                )
                candidate_generator.assert_not_called()

    def test_target_candidate_receipt_semantic_matrix(self):
        mutations = (
            (
                "candidate_only",
                lambda receipt: receipt.update(candidate_only=False),
            ),
            (
                "proof_authority",
                lambda receipt: receipt.update(proof_authority=True),
            ),
            (
                "selected_target_count",
                lambda receipt: receipt.update(
                    selected_target_count=(
                        receipt["selected_target_count"] + 1
                    )
                ),
            ),
            (
                "selected_target_count_bool_alias",
                lambda receipt: receipt.update(
                    selected_target_count=True
                ),
            ),
            (
                "block_size_bool_alias",
                lambda receipt: receipt.update(block_size=True),
            ),
            (
                "property_rows_bool_alias",
                lambda receipt: receipt.update(property_rows=False),
            ),
            (
                "strict_target_improvements_bool_alias",
                lambda receipt: receipt.update(
                    strict_target_improvements=False
                ),
            ),
            (
                "descriptor_block_id_bool_alias",
                lambda receipt: _mutate_candidate_descriptor(
                    receipt, "block_id", False
                ),
            ),
            (
                "descriptor_alpha_index_bool_alias",
                lambda receipt: _mutate_candidate_descriptor(
                    receipt, "alpha_tree_index", False
                ),
            ),
            (
                "selected_target_rows_sha256",
                lambda receipt: receipt.update(
                    selected_target_rows_sha256="0" * 64
                ),
            ),
            (
                "target_partition_complete",
                lambda receipt: receipt.update(
                    target_partition_complete=False
                ),
            ),
            (
                "unselected_policy",
                lambda receipt: receipt.update(
                    unselected_policy="forged"
                ),
            ),
            (
                "unselected_candidate_target_bounds_sha256",
                lambda receipt: receipt.update(
                    unselected_candidate_target_bounds_sha256="0" * 64
                ),
            ),
            (
                "candidate_target_bounds_sha256",
                lambda receipt: receipt.update(
                    candidate_target_bounds_sha256="0" * 64
                ),
            ),
            (
                "descriptor_coverage_sha256",
                lambda receipt: receipt.update(
                    descriptor_coverage_sha256="0" * 64
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(field=name):
                net, rows, thresholds, bundle = _build()
                candidate = bundle.stages[0].candidate_receipt
                mutate(candidate)
                candidate.update(_candidate_receipt(candidate))
                _rechain_bundle(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback_v3(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

    def test_property_candidate_receipt_semantic_matrix(self):
        mutations = (
            (
                "candidate_only",
                lambda receipt: receipt.update(candidate_only=False),
            ),
            (
                "proof_authority",
                lambda receipt: receipt.update(proof_authority=True),
            ),
            (
                "selected_property_count",
                lambda receipt: receipt.update(
                    selected_property_count=(
                        receipt["selected_property_count"] + 1
                    )
                ),
            ),
            (
                "target_width_bool_alias",
                lambda receipt: receipt.update(target_width=False),
            ),
            (
                "strict_property_improvements_bool_alias",
                lambda receipt: receipt.update(
                    strict_property_improvements=False
                ),
            ),
            (
                "descriptor_M_bool_alias",
                lambda receipt: _mutate_candidate_descriptor(
                    receipt, "M", True
                ),
            ),
            (
                "eligible_property_row_ids",
                lambda receipt: receipt.update(
                    eligible_property_row_ids=[1, 0]
                ),
            ),
            (
                "eligible_property_rows_sha256",
                lambda receipt: receipt.update(
                    eligible_property_rows_sha256="0" * 64
                ),
            ),
            (
                "property_coverage_complete",
                lambda receipt: receipt.update(
                    property_coverage_complete=False
                ),
            ),
            (
                "property_baseline_sha256",
                lambda receipt: receipt.update(
                    property_baseline_sha256="0" * 64
                ),
            ),
            (
                "candidate_property_bounds_sha256",
                lambda receipt: receipt.update(
                    candidate_property_bounds_sha256="0" * 64
                ),
            ),
            (
                "descriptor_coverage_sha256",
                lambda receipt: receipt.update(
                    descriptor_coverage_sha256="0" * 64
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(field=name):
                net, rows, thresholds, bundle = _build()
                candidate = bundle.property_stage.candidate_receipt
                mutate(candidate)
                candidate.update(_candidate_receipt(candidate))
                _rechain_bundle(bundle)
                self.assertFalse(
                    validate_verified_query_dual_feedback_v3(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=thresholds,
                        expected_target_relu_ids=(5, 7),
                        require_live_provenance=False,
                    )
                )

    def test_candidate_fallback_status_precedes_success_schema_checks(self):
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)

        with self.assertRaises(QueryDualPipelineError) as target_error:
            build_verified_query_dual_feedback_v3(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                stage_quotas=(1, 1),
                steps=2,
                block_size=1,
                replay_chunk_size=2,
                candidate_device="cpu",
                timeout_s=10.0,
                selector=_toy_selector,
                solver_factory=lambda: object(),
            )
        self.assertEqual(target_error.exception.code, "CANDIDATE_FAILURE")

        def fail_property_only(**kwargs):
            if kwargs["target_relu_lid"] is None:
                kwargs["solver_factory"] = lambda: object()
            return generate_query_dual_candidates(**kwargs)

        with self.assertRaises(QueryDualPipelineError) as property_error:
            build_verified_query_dual_feedback_v3(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                stage_quotas=(1, 1),
                steps=2,
                block_size=1,
                replay_chunk_size=2,
                candidate_device="cpu",
                timeout_s=10.0,
                selector=_toy_selector,
                solver_factory=_IntervalCandidateSolver,
                candidate_generator=fail_property_only,
            )
        self.assertEqual(
            property_error.exception.code, "NO_PROPERTY_CANDIDATE"
        )

    def test_v3_descriptor_receipt_schema_is_intentionally_closed(self):
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)

        def extended_candidate(**kwargs):
            candidate = generate_query_dual_candidates(**kwargs)
            if not candidate.query_descriptors:
                return candidate
            receipt = dict(candidate.receipt)
            receipt["descriptor_records"] = [
                dict(record)
                for record in candidate.receipt["descriptor_records"]
            ]
            receipt["descriptor_records"][0]["backend_tag"] = "toy"
            coverage = _json_sha256(receipt["descriptor_records"])
            receipt["descriptor_records_sha256"] = coverage
            receipt["descriptor_coverage_sha256"] = coverage
            receipt = _candidate_receipt(receipt)
            return replace(candidate, receipt=receipt)

        with self.assertRaises(QueryDualPipelineError) as caught:
            build_verified_query_dual_feedback_v3(
                _pipeline_net(),
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                stage_quotas=(1, 1),
                steps=2,
                block_size=1,
                replay_chunk_size=2,
                candidate_device="cpu",
                timeout_s=10.0,
                selector=_toy_selector,
                solver_factory=_IntervalCandidateSolver,
                candidate_generator=extended_candidate,
            )
        self.assertEqual(caught.exception.code, "CANDIDATE_BINDING")

    def test_stage_declared_semantics_are_consumed(self):
        net, rows, thresholds, bundle = _build()
        stage = bundle.stages[0]
        stage.receipt["relu_key_semantics"] = "postactivation"
        stage.receipt.update(_receipt(stage.receipt))
        bundle.receipt["stage_receipt_sha256"][0] = stage.receipt[
            "receipt_sha256"
        ]
        bundle.receipt.update(_receipt(bundle.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        net, rows, thresholds, bundle = _build()
        stage = bundle.stages[0]
        stage.receipt[
            "candidate_descriptor_coverage_sha256"
        ] = "0" * 64
        stage.receipt.update(_receipt(stage.receipt))
        bundle.receipt["stage_receipt_sha256"][0] = stage.receipt[
            "receipt_sha256"
        ]
        bundle.receipt.update(_receipt(bundle.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        net, rows, thresholds, bundle = _build()
        property_stage = bundle.property_stage
        property_stage.receipt[
            "candidate_descriptor_coverage_sha256"
        ] = "0" * 64
        property_stage.receipt.update(_receipt(property_stage.receipt))
        bundle.receipt["property_receipt_sha256"] = (
            property_stage.receipt["receipt_sha256"]
        )
        bundle.receipt.update(_receipt(bundle.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

    def test_residual_dependency_tightness_reaches_quarter(self):
        net = _residual_tightness_net()
        rows = np.asarray([[1.0]], dtype=np.float64)
        thresholds = np.asarray([0.0], dtype=np.float64)
        bundle = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(5,),
            stage_quotas=(1,),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        stage = bundle.stages[0]
        self.assertLessEqual(float(stage.target_upper[0]), 0.2500001)
        self.assertLessEqual(float(bundle.property_upper[0]), 0.2500001)
        self.assertGreaterEqual(stage.strict_improvements, 1)
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5,),
            )
        )

    def test_one_thousand_fraction_dag_and_conv_objectives(self):
        rng = np.random.default_rng(20260728)
        cases = []

        tight = _residual_tightness_net()
        tight_rows = (
            rng.integers(-32, 33, size=(600, 1)).astype(np.float64)
            / 16.0
        )
        tight_alpha = {
            3: rng.integers(0, 17, size=(600, 2)).astype(np.float64)
            / 16.0,
            5: rng.integers(0, 17, size=(600, 1)).astype(np.float64)
            / 16.0,
        }
        cases.append((tight, tight_rows, tight_alpha))

        dag = _residual_net()
        dag_rows = (
            rng.integers(-32, 33, size=(200, 2)).astype(np.float64)
            / 16.0
        )
        dag_alpha = {
            3: rng.integers(0, 17, size=(200, 2)).astype(np.float64)
            / 16.0
        }
        cases.append((dag, dag_rows, dag_alpha))

        conv, _ = _grouped_conv_net()
        conv_width = len(conv.layers[-1].out_vars)
        conv_rows = (
            rng.integers(
                -8, 9, size=(200, conv_width)
            ).astype(np.float64)
            / 8.0
        )
        cases.append((conv, conv_rows, None))

        checked = 0
        violations = 0
        for net, query_rows, alpha in cases:
            certificate = certify_query_dual_boxes(
                net, conv_channel_chunk=1
            )
            legacy = replay_query_lower_bounds(
                net,
                certificate.bounds,
                query_rows=query_rows,
                alpha_by_relu=alpha,
                chunk_size=64,
                max_workspace_bytes=64 * 1024 * 1024,
            )
            session = create_query_dual_replay_session(
                net,
                certificate,
                [None],
                deadline=time.monotonic() + 30.0,
            )
            frame = session.seal_bounds(
                certificate.bounds, start_lids=(None,)
            )
            pending = session.replay(
                frame,
                query_rows=query_rows,
                alpha_by_relu=alpha,
                chunk_size=64,
                max_workspace_bytes=64 * 1024 * 1024,
            )
            committed = session.commit()[0]
            exact = fraction_replay_lower_bounds(
                net,
                certificate.bounds,
                query_rows=query_rows,
                alpha_by_relu=alpha,
                max_arithmetic_terms=50_000_000,
            )
            self.assertTrue(
                np.array_equal(
                    legacy.lower_bounds, committed.lower_bounds
                )
            )
            self.assertEqual(
                legacy.receipt["lower_bounds_sha256"],
                committed.receipt["lower_bounds_sha256"],
            )
            self.assertTrue(
                np.array_equal(
                    pending.lower_bounds, committed.lower_bounds
                )
            )
            for lower, exact_lower in zip(
                committed.lower_bounds, exact
            ):
                checked += 1
                if Fraction.from_float(float(lower)) > exact_lower:
                    violations += 1
        self.assertEqual(checked, 1000)
        self.assertEqual(violations, 0)

    def test_k64_recovers_eighty_percent_of_full_query_gain(self):
        width = 80
        net = _parallel_residual_tightness_net(width)
        rows = np.eye(width, dtype=np.float64)
        thresholds = np.zeros(width, dtype=np.float64)
        root = certify_query_dual_boxes(net)
        root_upper = (
            root.bounds[5].ub.detach().cpu().numpy().reshape(-1)
        )
        full = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(5,),
            steps=2,
            block_size=1024,
            replay_chunk_size=1024,
            candidate_device="cpu",
            timeout_s=20.0,
            solver_factory=_IntervalCandidateSolver,
        )
        sparse = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(5,),
            stage_quotas=(64,),
            steps=2,
            block_size=1024,
            replay_chunk_size=1024,
            candidate_device="cpu",
            timeout_s=20.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        full_gain = float(
            np.sum(root_upper - full.stages[0].target_upper)
        )
        sparse_gain = float(
            np.sum(root_upper - sparse.stages[0].target_upper)
        )
        self.assertGreater(full_gain, 0.0)
        self.assertGreaterEqual(
            sparse_gain / full_gain,
            0.8 - 1.0e-12,
        )
        self.assertEqual(full.stages[0].strict_improvements, 80)
        self.assertEqual(sparse.stages[0].strict_improvements, 64)

    def test_fully_rehashed_selector_and_unselected_tamper_fail(self):
        # Rebuild from a pristine second bundle so the top-level hash is
        # syntactically valid while the nested selector semantics are false.
        net, rows, thresholds, forged = _build()
        forged_selector = forged.receipt["selector_receipt"]
        forged_selector["layers"][0]["selected_rows"].append(999)
        forged.receipt.update(_receipt(forged.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                forged,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        net, rows, thresholds, bundle = _build()
        stage = next(
            value
            for value in bundle.stages
            if value.receipt["omitted_row_ids"]
        )
        omitted = int(stage.receipt["omitted_row_ids"][0])
        stage.target_lower.setflags(write=True)
        stage.target_lower[omitted] = np.nextafter(
            stage.target_lower[omitted], np.inf
        )
        stage.target_lower.setflags(write=False)
        stage.receipt["target_bounds_sha256"] = _array_digest(
            np.stack([stage.target_lower, stage.target_upper])
        )
        stage.receipt.update(_receipt(stage.receipt))
        hashes = list(bundle.receipt["stage_receipt_sha256"])
        hashes[stage.stage_index] = stage.receipt["receipt_sha256"]
        bundle.receipt["stage_receipt_sha256"] = hashes
        bundle.receipt.update(_receipt(bundle.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

    def test_signed_zero_public_property_arrays_are_bit_bound(self):
        rows = np.asarray([[0.0]], dtype=np.float64)
        zero_threshold = np.asarray([0.0], dtype=np.float64)
        calibration = build_verified_query_dual_feedback_v3(
            _residual_tightness_net(),
            rows,
            zero_threshold,
            target_relu_ids=(5,),
            stage_quotas=(1,),
            steps=2,
            block_size=1,
            replay_chunk_size=2,
            candidate_device="cpu",
            timeout_s=10.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        threshold = calibration.property_upper.copy()

        for public_array in ("bundle", "property_stage"):
            with self.subTest(public_array=public_array):
                net = _residual_tightness_net()
                bundle = build_verified_query_dual_feedback_v3(
                    net,
                    rows,
                    threshold,
                    target_relu_ids=(5,),
                    stage_quotas=(1,),
                    steps=2,
                    block_size=1,
                    replay_chunk_size=2,
                    candidate_device="cpu",
                    timeout_s=10.0,
                    selector=_toy_selector,
                    solver_factory=_IntervalCandidateSolver,
                )
                array = (
                    bundle.property_upper
                    if public_array == "bundle"
                    else bundle.property_stage.property_upper
                )
                self.assertEqual(float(array[0]), 0.0)
                before_bits = int(array.view(np.uint64)[0])
                array.setflags(write=True)
                array.view(np.uint64)[0] ^= np.uint64(
                    0x8000000000000000
                )
                array.setflags(write=False)
                self.assertEqual(float(array[0]), 0.0)
                self.assertNotEqual(
                    int(array.view(np.uint64)[0]), before_bits
                )
                self.assertFalse(
                    validate_verified_query_dual_feedback_v3(
                        bundle,
                        net=net,
                        property_rows=rows,
                        thresholds=threshold,
                        expected_target_relu_ids=(5,),
                    )
                )

    def test_cross_session_block_and_property_reorder_fail(self):
        net, rows, thresholds, first = _build()
        _, _, _, second = _build()
        first_stage = first.stages[0]
        second_stage = second.stages[0]
        object.__setattr__(
            first_stage, "blocks", second_stage.blocks
        )
        first_stage.receipt["block_receipt_sha256"] = [
            block.replay_receipt["receipt_sha256"]
            for block in first_stage.blocks
        ]
        first_stage.receipt["alpha_bridge_sha256"] = [
            block.alpha_bridge_sha256
            for block in first_stage.blocks
        ]
        first_stage.receipt.update(_receipt(first_stage.receipt))
        stage_hashes = list(first.receipt["stage_receipt_sha256"])
        stage_hashes[0] = first_stage.receipt["receipt_sha256"]
        first.receipt["stage_receipt_sha256"] = stage_hashes
        first.receipt.update(_receipt(first.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                first,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        net, rows, thresholds, bundle = _build()
        property_stage = bundle.property_stage
        self.assertGreaterEqual(len(property_stage.blocks), 2)
        object.__setattr__(
            property_stage,
            "blocks",
            tuple(reversed(property_stage.blocks)),
        )
        property_stage.receipt["block_receipt_sha256"] = [
            block.replay_receipt["receipt_sha256"]
            for block in property_stage.blocks
        ]
        property_stage.receipt["alpha_bridge_sha256"] = [
            block.alpha_bridge_sha256
            for block in property_stage.blocks
        ]
        property_stage.receipt.update(
            _receipt(property_stage.receipt)
        )
        bundle.receipt["property_receipt_sha256"] = (
            property_stage.receipt["receipt_sha256"]
        )
        bundle.receipt.update(_receipt(bundle.receipt))
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

    def test_fully_rehashed_crosswalk_and_frame_claims_fail_semantics(self):
        net, rows, thresholds, bundle = _build()
        all_blocks = [
            block for stage in bundle.stages for block in stage.blocks
        ] + list(bundle.property_stage.blocks)
        for block in all_blocks:
            crosswalk = block.replay_receipt["sealed_context"][
                "manifest_crosswalk"
            ]
            crosswalk["root_snapshot_content_sha256"] = "0" * 64
            cone = crosswalk["replay_cones"][0]
            cone["forward_layer_ids"].insert(-1, 999)
            _rehash_replay_receipt(block)
            # A self-hashed standalone receipt cannot derive the sealed root.
            self.assertTrue(
                verify_query_dual_replay_receipt(block.replay_receipt)
            )
        _rebind_replay_receipts(bundle)
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

        net, rows, thresholds, bundle = _build()
        stage = bundle.stages[0]
        forged_frame = "a" * 64
        stage.receipt["bounds_frame_sha256"] = forged_frame
        for block in stage.blocks:
            block.replay_receipt["sealed_context"][
                "bounds_frame_sha256"
            ] = forged_frame
            _rehash_replay_receipt(block)
            self.assertTrue(
                verify_query_dual_replay_receipt(block.replay_receipt)
            )
        _rebind_replay_receipts(bundle)
        self.assertFalse(
            validate_verified_query_dual_feedback_v3(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(5, 7),
                require_live_provenance=False,
            )
        )

    def test_candidate_failure_rolls_back_before_authority(self):
        calls = 0

        def fail_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second-stage failure")
            from act.back_end.hybridz_tf.query_dual_candidates import (
                generate_query_dual_candidates,
            )

            return generate_query_dual_candidates(**kwargs)

        net = _pipeline_net()
        rows = np.asarray(
            [[1.0, -0.3], [-0.5, 1.0]], dtype=np.float64
        )
        thresholds = np.asarray([-0.2, 0.35], dtype=np.float64)
        with self.assertRaises(QueryDualPipelineError):
            build_verified_query_dual_feedback_v3(
                net,
                rows,
                thresholds,
                target_relu_ids=(5, 7),
                stage_quotas=(1, 1),
                steps=2,
                block_size=1,
                replay_chunk_size=2,
                candidate_device="cpu",
                timeout_s=10.0,
                selector=_toy_selector,
                solver_factory=_IntervalCandidateSolver,
                candidate_generator=fail_second,
            )
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
