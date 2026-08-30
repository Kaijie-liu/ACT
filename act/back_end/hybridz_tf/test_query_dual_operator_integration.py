#!/usr/bin/env python3
"""Fail-closed toy audits for query-dual/Operator-HZ integration."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    _intersect_verified_query_dual_box,
    build_operator_hz,
)
from act.back_end.hybridz_tf.query_dual_pipeline import (
    _receipt as _pipeline_receipt,
    build_verified_query_dual_feedback,
    validate_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.test_operator_residual_normal_form import (
    _scalar_relu_toy,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import (
    _validate_verified_query_dual_property_actual_object,
    verify_once,
)
from act.front_end.specs import OutKind
from act.util.device_manager import initialize_device
from act.util.stats import VerifyStatus


def _residual_two_relu_toy() -> SimpleNamespace:
    """``x -> [x,-x] -> ReLU -> sum + .25x -> ReLU -> Dense``."""

    dtype = torch.float64
    device = torch.device("cpu")
    C = np.asarray([[1.0], [-1.0]], dtype=np.float64)
    # These exercise both signs in C*y-threshold.  The exact first maximum is
    # 1.25 - 1.5 = -0.25.  The second has a negative threshold and therefore
    # equals max(-y + 0.25), which is +0.25 at x=0.
    thresholds = np.asarray([1.5, -0.25], dtype=np.float64)
    assertion = {
        "kind": OutKind.LINEAR_LE,
        "C": torch.tensor(C, dtype=dtype, device=device),
        "thresholds": torch.tensor(
            thresholds.reshape(1, 2),
            dtype=dtype,
            device=device,
        ),
        "M": 2,
    }
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=[0],
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype, device=device),
                "ub": torch.tensor([[1.0]], dtype=dtype, device=device),
            },
            in_vars=[0],
            out_vars=[0],
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[1.0], [-1.0]], dtype=dtype, device=device
                ),
                "bias": torch.zeros(2, dtype=dtype, device=device),
                "in_features": 1,
                "out_features": 2,
            },
            in_vars=[0],
            out_vars=[1, 2],
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=[1, 2],
            out_vars=[3, 4],
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[1.0, 1.0]], dtype=dtype, device=device
                ),
                "bias": torch.zeros(1, dtype=dtype, device=device),
                "in_features": 2,
                "out_features": 1,
            },
            in_vars=[3, 4],
            out_vars=[5],
        ),
        Layer(
            id=5,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[0.25]], dtype=dtype, device=device
                ),
                "bias": torch.zeros(1, dtype=dtype, device=device),
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=[0],
            out_vars=[6],
        ),
        Layer(
            id=6,
            kind="ADD",
            params={"x_vars": [5], "y_vars": [6]},
            in_vars=[5, 6],
            out_vars=[7],
        ),
        Layer(
            id=7,
            kind="RELU",
            params={},
            in_vars=[7],
            out_vars=[8],
        ),
        Layer(
            id=8,
            kind="DENSE",
            params={
                "weight": torch.ones(
                    (1, 1), dtype=dtype, device=device
                ),
                "bias": torch.zeros(1, dtype=dtype, device=device),
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=[8],
            out_vars=[9],
        ),
        Layer(
            id=9,
            kind="ASSERT",
            params=assertion,
            in_vars=[9],
            out_vars=[9],
        ),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [1],
        6: [4, 5],
        7: [6],
        8: [7],
        9: [8],
    }
    succs = {
        0: [1],
        1: [2, 5],
        2: [3],
        3: [4],
        4: [6],
        5: [6],
        6: [7],
        7: [8],
        8: [9],
        9: [],
    }
    net = Net(layers=layers, preds=preds, succs=succs)
    widths = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
    }
    facts = {}
    for layer_id, width in widths.items():
        lower = torch.full(
            (1, width), -1.0e30, dtype=dtype, device=device
        )
        upper = torch.full(
            (1, width), 1.0e30, dtype=dtype, device=device
        )
        if layer_id in {0, 1}:
            lower = torch.tensor(
                [[-1.0]], dtype=dtype, device=device
            )
            upper = torch.tensor([[1.0]], dtype=dtype, device=device)
        facts[layer_id] = Fact(Bounds(lower, upper), ConSet())
    return SimpleNamespace(
        net=net,
        facts=facts,
        C=C,
        thresholds=thresholds,
    )


def _build_live_feedback(toy: SimpleNamespace):
    return build_verified_query_dual_feedback(
        toy.net,
        toy.C,
        toy.thresholds,
        target_relu_ids=(7,),
        steps=1,
        block_size=1,
        replay_chunk_size=16,
        candidate_device="cpu",
        timeout_s=20.0,
    )


def _build_operator(toy: SimpleNamespace, feedback):
    return build_operator_hz(
        toy.net,
        toy.facts,
        toy.facts,
        exact_budget=0,
        materialize_add=True,
        property_upper_C=toy.C,
        property_upper_thresholds=toy.thresholds,
        verified_query_dual_feedback=feedback,
    )


def _sparse_exact(left: sp.spmatrix, right: sp.spmatrix) -> bool:
    left = left.tocsr()
    right = right.tocsr()
    return bool(
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _operator_backend_config() -> BackendConfig:
    return BackendConfig(
        solver="hybridz",
        device="cpu",
        dtype="float64",
        hybridz=HybridZConfig(
            timeout=5.0,
            engine="operator_hz_objbound",
            property_tail_upper=True,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
        ),
    )


def _operator_query_backend_config() -> BackendConfig:
    return BackendConfig(
        solver="hybridz",
        device="cpu",
        dtype="float64",
        hybridz=HybridZConfig(
            timeout=5.0,
            engine="operator_hz_objbound",
            operator_exact_budget=0,
            property_tail_upper=True,
            query_dual_feedback_targets=(7,),
            query_dual_feedback_steps=1,
            query_dual_feedback_time_limit=2.0,
            query_dual_feedback_block_size=1,
            query_dual_feedback_device="cpu",
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
        ),
    )


def _operator_property_only_query_backend_config() -> BackendConfig:
    return BackendConfig(
        solver="hybridz",
        device="cpu",
        dtype="float64",
        hybridz=HybridZConfig(
            timeout=5.0,
            engine="operator_hz_objbound",
            operator_exact_budget=0,
            operator_materialize_add=True,
            residual_bound_screen=True,
            property_tail_upper=False,
            query_dual_feedback_targets=(),
            query_dual_feedback_steps=1,
            query_dual_feedback_time_limit=2.0,
            query_dual_feedback_block_size=2,
            query_dual_feedback_device="cpu",
            lp_prefilter_fraction=0.0,
            lp_prefilter_max_seconds=0.0,
        ),
    )


class QueryDualOperatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._prior_device = torch.get_default_device()
        cls._prior_dtype = torch.get_default_dtype()
        initialize_device(device="cpu", dtype="float64")
        cls.toy = _residual_two_relu_toy()
        cls.feedback = _build_live_feedback(cls.toy)
        cls.operator_build = _build_operator(cls.toy, cls.feedback)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_default_dtype(cls._prior_dtype)
        torch.set_default_device(cls._prior_device)

    def test_atomic_box_intersection_and_cross_rejection(self) -> None:
        lower, upper = _intersect_verified_query_dual_box(
            np.asarray([-2.0, -1.0, 0.0]),
            np.asarray([2.0, 3.0, 4.0]),
            np.asarray([-1.5, -2.0, 1.0]),
            np.asarray([1.5, 2.0, 5.0]),
            layer_id=7,
        )
        np.testing.assert_array_equal(lower, [-1.5, -1.0, 1.0])
        np.testing.assert_array_equal(upper, [1.5, 2.0, 4.0])

        invalid_pairs = (
            (
                np.asarray([-2.0]),
                np.asarray([2.0]),
                np.asarray([3.0]),
                np.asarray([4.0]),
            ),
            (
                np.asarray([-2.0]),
                np.asarray([2.0]),
                np.asarray([1.0]),
                np.asarray([-1.0]),
            ),
            (
                np.asarray([-2.0]),
                np.asarray([2.0]),
                np.asarray([np.nan]),
                np.asarray([1.0]),
            ),
            (
                np.asarray([-2.0, -1.0]),
                np.asarray([2.0, 1.0]),
                np.asarray([-1.0]),
                np.asarray([1.0]),
            ),
        )
        for local_l, local_u, query_l, query_u in invalid_pairs:
            with self.subTest(
                local_shape=local_l.shape,
                query_lower=query_l.tolist(),
            ):
                with self.assertRaises(OperatorHZBuildError):
                    _intersect_verified_query_dual_box(
                        local_l,
                        local_u,
                        query_l,
                        query_u,
                        layer_id=7,
                    )

    def test_none_preserves_baseline_metadata_surface(self) -> None:
        toy = _scalar_relu_toy()
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray([[1.0]], dtype=np.float64),
            property_upper_thresholds=np.asarray([2.0], dtype=np.float64),
            verified_query_dual_feedback=None,
        )
        self.assertNotIn(
            "verified_query_dual_feedback",
            build.metadata,
        )
        self.assertNotIn(
            "verified_query_dual_property_constants",
            build.metadata["property_tail_upper"],
        )
        relu = next(
            item
            for item in build.metadata["layers"]
            if item["kind"] == "RELU"
        )
        self.assertNotIn("verified_query_dual_bound", relu)
        self.assertEqual(build.property_upper_row_groups, ((0,),))
        self.assertEqual(build.hz.n_out, 1)

    def test_live_transaction_is_consumed_as_bound_and_constant_rows(self) -> None:
        feedback = self.feedback
        build = self.operator_build
        self.assertTrue(
            validate_verified_query_dual_feedback(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )
        # Positive and negative thresholds must keep the exact
        # C*y-threshold sign through replay and export.
        self.assertLess(float(feedback.property_upper[0]), 0.0)
        self.assertGreater(float(feedback.property_upper[1]), 0.0)
        self.assertGreaterEqual(float(feedback.property_upper[0]), -0.25)
        self.assertGreaterEqual(float(feedback.property_upper[1]), 0.25)

        tail = build.metadata["property_tail_upper"]
        self.assertEqual(
            tail["alternative_plane_kinds"],
            [
                "verified_query_dual_property_constant",
                "verified_query_dual_property_constant",
            ],
        )
        self.assertEqual(
            tail["alternative_plane_rival_ids"],
            [0, 1],
        )
        self.assertEqual(build.property_upper_row_groups, ((0, 2), (1, 3)))
        constant_rows = np.asarray([2, 3], dtype=np.int64)
        actual = np.ascontiguousarray(build.hz.c[constant_rows])
        expected = np.ascontiguousarray(feedback.property_upper)
        self.assertTrue(
            np.array_equal(
                actual.view(np.uint64),
                expected.view(np.uint64),
            )
        )
        self.assertEqual(build.hz.Gc[constant_rows, :].nnz, 0)
        self.assertEqual(build.hz.Gb[constant_rows, :].nnz, 0)

        relu = next(
            item
            for item in build.metadata["layers"]
            if item["layer_id"] == 7
        )
        bound_receipt = relu["verified_query_dual_bound"]
        self.assertTrue(bound_receipt["proof_authority"])
        self.assertGreater(
            bound_receipt["lower_improved_rows"]
            + bound_receipt["upper_improved_rows"],
            0,
        )
        self.assertEqual(
            bound_receipt["transaction_receipt_sha256"],
            feedback.receipt["receipt_sha256"],
        )
        self.assertEqual(
            bound_receipt["final_boxes_sha256"],
            feedback.receipt["final_boxes_sha256"],
        )
        feedback_metadata = build.metadata["verified_query_dual_feedback"]
        self.assertEqual(
            feedback_metadata["live_full_validation_passes"],
            2,
        )
        self.assertGreaterEqual(
            feedback_metadata["validation_and_snapshot_seconds"],
            0.0,
        )

        self.assertTrue(
            _validate_verified_query_dual_property_actual_object(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                operator_build=build,
                tail_receipt=tail,
                property_row_groups=build.property_upper_row_groups,
                alternative_rivals=tail[
                    "alternative_plane_rival_ids"
                ],
                alternative_kinds=tail[
                    "alternative_plane_kinds"
                ],
            )
        )

    def test_actual_object_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.operator_build)
        tail = tampered.metadata["property_tail_upper"]
        tampered.hz.c[2] = np.nextafter(
            tampered.hz.c[2], np.float64(np.inf)
        )
        self.assertFalse(
            _validate_verified_query_dual_property_actual_object(
                self.feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                operator_build=tampered,
                tail_receipt=tail,
                property_row_groups=tampered.property_upper_row_groups,
                alternative_rivals=tail[
                    "alternative_plane_rival_ids"
                ],
                alternative_kinds=tail[
                    "alternative_plane_kinds"
                ],
            )
        )

        tampered = copy.deepcopy(self.operator_build)
        tampered.hz.Gc = tampered.hz.Gc.tolil()
        tampered.hz.Gc[2, 0] = 1.0
        tampered.hz.Gc = tampered.hz.Gc.tocsr()
        tail = tampered.metadata["property_tail_upper"]
        self.assertFalse(
            _validate_verified_query_dual_property_actual_object(
                self.feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                operator_build=tampered,
                tail_receipt=tail,
                property_row_groups=tampered.property_upper_row_groups,
                alternative_rivals=tail[
                    "alternative_plane_rival_ids"
                ],
                alternative_kinds=tail[
                    "alternative_plane_kinds"
                ],
            )
        )

    def test_constants_follow_existing_add_source_alternatives(self) -> None:
        build = build_operator_hz(
            self.toy.net,
            self.toy.facts,
            self.toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=self.toy.C,
            property_upper_thresholds=self.toy.thresholds,
            property_tail_add_source_planes=True,
            verified_query_dual_feedback=self.feedback,
        )
        tail = build.metadata["property_tail_upper"]
        self.assertEqual(tail["add_source_planes"]["status"], "applied")
        self.assertEqual(
            tail["alternative_plane_kinds"],
            [
                "add_source_alpha0",
                "add_source_alpha0",
                "verified_query_dual_property_constant",
                "verified_query_dual_property_constant",
            ],
        )
        self.assertEqual(
            build.property_upper_row_groups,
            ((0, 2, 4), (1, 3, 5)),
        )
        self.assertEqual(build.hz.Gc[[4, 5], :].nnz, 0)
        self.assertTrue(
            _validate_verified_query_dual_property_actual_object(
                self.feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                operator_build=build,
                tail_receipt=tail,
                property_row_groups=build.property_upper_row_groups,
                alternative_rivals=tail[
                    "alternative_plane_rival_ids"
                ],
                alternative_kinds=tail[
                    "alternative_plane_kinds"
                ],
            )
        )

    def test_copy_rehydration_and_block_reordering_are_rejected(self) -> None:
        # ``replace`` rehydrates an equal-content dataclass without copying
        # the root certificate's intentional MappingProxyType fields.
        copied = replace(self.feedback)
        self.assertFalse(
            validate_verified_query_dual_feedback(
                copied,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )
        with self.assertRaisesRegex(
            OperatorHZBuildError, "process-local transaction validator"
        ):
            _build_operator(self.toy, copied)

        reordered_property = replace(
            self.feedback.property_stage,
            blocks=tuple(reversed(self.feedback.property_stage.blocks)),
        )
        reordered = replace(
            self.feedback,
            property_stage=reordered_property,
        )
        self.assertFalse(
            validate_verified_query_dual_feedback(
                reordered,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
                require_live_provenance=False,
            )
        )

    def test_same_live_object_array_and_receipt_tamper_are_rejected(self) -> None:
        feedback = _build_live_feedback(self.toy)
        target_lower = feedback.certified_bounds[7].lb
        original_lower = target_lower.clone()
        target_lower.reshape(-1)[0] += 0.125
        self.assertFalse(
            validate_verified_query_dual_feedback(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )
        with self.assertRaises(OperatorHZBuildError):
            _build_operator(self.toy, feedback)
        target_lower.copy_(original_lower)
        self.assertTrue(
            validate_verified_query_dual_feedback(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )

        original_hash = feedback.receipt["property_upper_sha256"]
        feedback.receipt["property_upper_sha256"] = "0" * 64
        self.assertFalse(
            validate_verified_query_dual_feedback(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )
        with self.assertRaises(OperatorHZBuildError):
            _build_operator(self.toy, feedback)
        feedback.receipt["property_upper_sha256"] = original_hash
        self.assertTrue(
            validate_verified_query_dual_feedback(
                feedback,
                net=self.toy.net,
                property_rows=self.toy.C,
                thresholds=self.toy.thresholds,
                expected_target_relu_ids=(7,),
            )
        )

    def test_post_snapshot_revalidation_closes_builder_toctou(self) -> None:
        feedback = _build_live_feedback(self.toy)
        call_count = 0

        def validate_then_mutate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            valid = validate_verified_query_dual_feedback(*args, **kwargs)
            # Simulate a caller racing the second live validation: the
            # validator observed the original bits, but the public builder
            # must compare those bits again before consuming its snapshot.
            if call_count == 2:
                live_lower = feedback.certified_bounds[7].lb
                live_lower.reshape(-1)[0] += 0.125
            return valid

        with mock.patch(
            "act.back_end.hybridz_tf.query_dual_pipeline."
            "validate_verified_query_dual_feedback",
            side_effect=validate_then_mutate,
        ):
            with self.assertRaisesRegex(
                OperatorHZBuildError,
                "bounds changed while taking the private snapshot",
            ):
                _build_operator(self.toy, feedback)
        self.assertEqual(call_count, 2)

    def test_validated_but_crossing_box_aborts_complete_build(self) -> None:
        crossing = replace(self.feedback)
        crossing.certified_bounds[7] = Bounds(
            lb=torch.tensor([[100.0]], dtype=torch.float64),
            ub=torch.tensor([[101.0]], dtype=torch.float64),
        )
        with mock.patch(
            "act.back_end.hybridz_tf.query_dual_pipeline."
            "validate_verified_query_dual_feedback",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                OperatorHZBuildError, "conflicts at row"
            ):
                _build_operator(self.toy, crossing)

    def test_verifier_revalidates_live_bundle_and_actual_rows(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                self.toy.net,
                backend_cfg=_operator_backend_config(),
                verified_query_dual_feedback=self.feedback,
            )[0]
            self.assertNotEqual(
                result.metadata.get("reason"),
                "hybridz_operator_build_failed",
            )
            operator = result.metadata["operator_hz"]
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(transaction["source"], "explicit_live_object")
            self.assertEqual(transaction["steps"], 1)
            self.assertEqual(transaction["block_size"], 1)
            self.assertEqual(transaction["replay_chunk_size"], 16)
            self.assertEqual(transaction["device"], "cpu")
            self.assertTrue(
                operator["verified_query_dual_feedback"][
                    "process_local_validation"
                ]
            )
            self.assertEqual(
                operator["property_tail_upper"][
                    "verified_query_dual_property_constants"
                ]["constant_row_indices"],
                [2, 3],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verify_once_builds_and_applies_configured_live_transaction(
        self,
    ) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                wraps=build_operator_hz,
            ) as operator_build:
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                    fail_fast_on_query_dual_fallback=True,
                )[0]
            operator_build.assert_called_once()
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(transaction["status"], "applied")
            self.assertTrue(transaction["proof_authority"])
            self.assertEqual(
                transaction["source"], "built_in_verify_once"
            )
            self.assertEqual(
                transaction["pipeline_schema"],
                "act.verified_query_dual_feedback.v2",
            )
            self.assertEqual(
                transaction["target_stage_schema"],
                "act.verified_query_dual_stage.v2",
            )
            self.assertEqual(
                transaction["property_stage_schema"],
                "act.verified_query_dual_property.v2",
            )
            self.assertEqual(
                transaction["candidate_schema"],
                "act.query_dual_candidates.v2",
            )
            self.assertEqual(
                transaction["candidate_protocol"],
                "descriptor_only_v2",
            )
            self.assertEqual(transaction["targets"], [7])
            self.assertEqual(transaction["steps"], 1)
            self.assertEqual(transaction["block_size"], 1)
            self.assertEqual(transaction["replay_chunk_size"], 1)
            self.assertEqual(transaction["device"], "cpu")
            self.assertEqual(transaction["property_rows"], 2)
            self.assertEqual(transaction["target_stage_count"], 1)
            self.assertGreaterEqual(
                transaction["strict_improvements_total"], 1
            )
            pipeline = transaction["pipeline_receipt"]
            operator = result.metadata["operator_hz"][
                "verified_query_dual_feedback"
            ]
            self.assertEqual(
                transaction["operator_transaction_receipt_sha256"],
                pipeline["receipt_sha256"],
            )
            self.assertEqual(
                operator["transaction_receipt_sha256"],
                pipeline["receipt_sha256"],
            )
            self.assertEqual(
                result.metadata["cfg_query_dual_feedback_targets"],
                [7],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verify_once_property_only_replay_uses_exported_c5_frame(
        self,
    ) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                self.toy.net,
                backend_cfg=(
                    _operator_property_only_query_backend_config()
                ),
                fail_fast_on_query_dual_fallback=True,
            )[0]
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(transaction["status"], "applied")
            self.assertTrue(transaction["proof_authority"])
            self.assertEqual(transaction["targets"], [])
            self.assertEqual(
                transaction["application_mode"],
                "property_only_post_operator_bound_frame",
            )
            initial = transaction["pipeline_receipt"][
                "initial_preactivation_frame"
            ]
            exported = result.metadata["operator_hz"][
                "verified_preactivation_frame"
            ]
            self.assertTrue(initial["enabled"])
            self.assertEqual(
                initial["source_receipt_sha256"],
                exported["receipt_sha256"],
            )
            self.assertIn(
                result.metadata.get("reason"),
                {
                    None,
                    "property_only_query_dual_incomplete",
                },
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verify_once_query_failure_rolls_back_but_is_not_hidden(
        self,
    ) -> None:
        from act.back_end.hybridz_tf.query_dual_pipeline import (
            QueryDualPipelineError,
        )

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with (
                mock.patch(
                    "act.back_end.hybridz_tf.query_dual_pipeline."
                    "build_verified_query_dual_feedback",
                    side_effect=QueryDualPipelineError(
                        "AUDIT_INJECTED", "controlled candidate failure"
                    ),
                ),
                mock.patch(
                    "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                    wraps=build_operator_hz,
                ) as operator_build,
            ):
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                )[0]
            operator_build.assert_called_once()
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(
                transaction["status"], "error_fallback_baseline"
            )
            self.assertFalse(transaction["proof_authority"])
            self.assertEqual(
                transaction["error_code"], "AUDIT_INJECTED"
            )
            self.assertEqual(
                transaction["rollback"],
                "complete_query_dual_feature",
            )
            self.assertNotIn(
                "verified_query_dual_feedback",
                result.metadata["operator_hz"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verify_once_query_failure_fail_fast_skips_operator(self) -> None:
        from act.back_end.hybridz_tf.query_dual_pipeline import (
            QueryDualPipelineError,
        )

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with (
                mock.patch(
                    "act.back_end.hybridz_tf.query_dual_pipeline."
                    "build_verified_query_dual_feedback",
                    side_effect=QueryDualPipelineError(
                        "AUDIT_INJECTED", "controlled candidate failure"
                    ),
                ),
                mock.patch(
                    "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                ) as operator_build,
            ):
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                    fail_fast_on_query_dual_fallback=True,
                )[0]

            operator_build.assert_not_called()
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                result.metadata["reason"],
                "query_dual_feedback_not_applied",
            )
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(
                transaction["status"], "error_fallback_baseline"
            )
            self.assertFalse(transaction["proof_authority"])
            self.assertEqual(
                transaction["error_code"], "AUDIT_INJECTED"
            )
            self.assertEqual(
                transaction["error"],
                "AUDIT_INJECTED: controlled candidate failure",
            )
            self.assertNotIn("operator_hz", result.metadata)
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_v1_live_top_fail_fast_rejects_before_operator(self) -> None:
        original_receipt = self.feedback.receipt
        forged_body = copy.deepcopy(dict(original_receipt))
        forged_body["schema"] = "act.verified_query_dual_feedback.v1"
        forged_body["candidate_schema"] = "act.query_dual_candidates.v1"
        forged_body["candidate_protocol"] = "frozen_alpha_replay_v1"
        object.__setattr__(
            self.feedback,
            "receipt",
            _pipeline_receipt(forged_body),
        )

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with (
                mock.patch(
                    "act.back_end.hybridz_tf.query_dual_pipeline."
                    "build_verified_query_dual_feedback",
                    return_value=self.feedback,
                ),
                mock.patch(
                    "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                ) as operator_build,
            ):
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                    fail_fast_on_query_dual_fallback=True,
                )[0]

            operator_build.assert_not_called()
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                result.metadata["reason"],
                "query_dual_feedback_not_applied",
            )
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(
                transaction["status"], "error_fallback_baseline"
            )
            self.assertFalse(transaction["proof_authority"])
            self.assertEqual(transaction["error_type"], "ValueError")
            self.assertIn(
                "descriptor-only V2 pending receipt rejected",
                transaction["error"],
            )
            self.assertNotIn("operator_hz", result.metadata)
        finally:
            object.__setattr__(
                self.feedback,
                "receipt",
                original_receipt,
            )
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_explicit_rehashed_protocol_fails_closed_before_operator(
        self,
    ) -> None:
        original_receipt = self.feedback.receipt
        forged_body = copy.deepcopy(dict(original_receipt))
        forged_body["candidate_protocol"] = "frozen_alpha_replay_v1"
        object.__setattr__(
            self.feedback,
            "receipt",
            _pipeline_receipt(forged_body),
        )

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
            ) as operator_build:
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                    verified_query_dual_feedback=self.feedback,
                    fail_fast_on_query_dual_fallback=True,
                )[0]

            operator_build.assert_not_called()
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                result.metadata["reason"],
                "query_dual_feedback_not_applied",
            )
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(
                transaction["status"], "error_fallback_baseline"
            )
            self.assertEqual(
                transaction["source"], "explicit_live_object"
            )
            self.assertEqual(transaction["error_type"], "ValueError")
            self.assertIn(
                "descriptor-only V2 pending receipt rejected",
                transaction["error"],
            )
            self.assertNotIn("operator_hz", result.metadata)
        finally:
            object.__setattr__(
                self.feedback,
                "receipt",
                original_receipt,
            )
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_explicit_feedback_mismatch_fail_fast_skips_operator(self) -> None:
        mismatched = replace(
            self.feedback,
            target_relu_ids=(3,),
        )
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
            ) as operator_build:
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_query_backend_config(),
                    verified_query_dual_feedback=mismatched,
                    fail_fast_on_query_dual_fallback=True,
                )[0]

            operator_build.assert_not_called()
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(
                transaction["status"], "error_fallback_baseline"
            )
            self.assertEqual(transaction["error_type"], "ValueError")
            self.assertIn(
                "target schedule differs",
                transaction["error"],
            )
            self.assertEqual(
                result.metadata["reason"],
                "query_dual_feedback_not_applied",
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_fail_fast_flag_does_not_change_disabled_query_path(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                wraps=build_operator_hz,
            ) as operator_build:
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_backend_config(),
                    fail_fast_on_query_dual_fallback=True,
                )[0]

            operator_build.assert_called_once()
            transaction = result.metadata[
                "query_dual_feedback_transaction"
            ]
            self.assertEqual(transaction["status"], "disabled")
            self.assertFalse(transaction["proof_authority"])
            self.assertIn("operator_hz", result.metadata)
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_rejects_tampered_exported_constant_row(self) -> None:
        tampered = copy.deepcopy(self.operator_build)
        tampered.hz.c[2] = np.nextafter(
            tampered.hz.c[2], np.float64(np.inf)
        )
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                return_value=tampered,
            ):
                result = verify_once(
                    self.toy.net,
                    backend_cfg=_operator_backend_config(),
                    verified_query_dual_feedback=self.feedback,
                )[0]
            self.assertEqual(
                result.metadata["reason"],
                "hybridz_operator_build_failed",
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_omitted_and_explicit_none_are_bit_compatible(self) -> None:
        toy = _scalar_relu_toy()

        def build(*, explicit_none: bool):
            next_id = 0

            def fresh_ids(count: int, device: str = "cpu"):
                nonlocal next_id
                result = torch.arange(
                    next_id,
                    next_id + int(count),
                    dtype=torch.int64,
                    device=device,
                )
                next_id += int(count)
                return result

            kwargs = {
                "exact_budget": 0,
                "materialize_add": True,
                "property_upper_C": np.asarray(
                    [[1.0]], dtype=np.float64
                ),
                "property_upper_thresholds": np.asarray(
                    [2.0], dtype=np.float64
                ),
            }
            if explicit_none:
                kwargs["verified_query_dual_feedback"] = None
            with (
                mock.patch(
                    "act.back_end.hybridz_tf.operator_hz."
                    "hz_fresh_col_ids",
                    side_effect=fresh_ids,
                ),
                mock.patch(
                    "act.back_end.hybridz_tf.operator_hz.time.monotonic",
                    return_value=100.0,
                ),
            ):
                return build_operator_hz(
                    toy.net,
                    toy.facts,
                    toy.facts,
                    **kwargs,
                )

        omitted = build(explicit_none=False)
        explicit = build(explicit_none=True)
        for field in ("c", "b", "ub", "col_ids", "bcol_ids"):
            np.testing.assert_array_equal(
                getattr(omitted.hz, field),
                getattr(explicit.hz, field),
            )
        for field in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            self.assertTrue(
                _sparse_exact(
                    getattr(omitted.hz, field),
                    getattr(explicit.hz, field),
                )
            )
        self.assertEqual(omitted.metadata, explicit.metadata)
        self.assertEqual(
            omitted.property_upper_row_groups,
            explicit.property_upper_row_groups,
        )


if __name__ == "__main__":
    unittest.main()
