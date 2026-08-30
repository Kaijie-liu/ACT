#!/usr/bin/env python3
"""Toy-first authority tests for C5 -> property-only query-dual replay."""

from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np
import torch

from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZPreactivationFrame,
    build_operator_hz,
    validate_operator_hz_preactivation_frame,
)
from act.back_end.hybridz_tf.query_dual_pipeline import (
    QueryDualPipelineError,
    build_verified_query_dual_feedback,
    validate_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.test_operator_property_correlation import (
    _correlated_add_toy,
)
from act.back_end.hybridz_tf.test_query_dual_pipeline import (
    _IntervalCandidateSolver,
)
from act.util.device_manager import initialize_device


class QueryDualOperatorBoundFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prior_device = torch.get_default_device()
        cls._prior_dtype = torch.get_default_dtype()
        initialize_device(device="cpu", dtype="float64")

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls._prior_dtype)
        torch.set_default_device(cls._prior_device)

    def _build(self, *, bias=0.25):
        net, facts = _correlated_add_toy(bias=bias)
        built = build_operator_hz(
            net,
            facts,
            facts,
            residual_bound_screen=True,
        )
        frame = built.verified_preactivation_frame
        self.assertIsNotNone(frame)
        return net, facts, built, frame

    def test_frame_is_live_network_bound_and_not_rehydratable(self):
        net, _facts, _built, frame = self._build()
        self.assertTrue(
            validate_operator_hz_preactivation_frame(frame, net=net)
        )

        forged = OperatorHZPreactivationFrame(
            bounds=frame.bounds,
            receipt=frame.receipt,
            provenance_nonce=frame.provenance_nonce,
        )
        self.assertFalse(
            validate_operator_hz_preactivation_frame(
                forged, net=net
            )
        )

        tampered_bounds = dict(frame.bounds)
        lower, upper = tampered_bounds[6]
        changed = upper.copy()
        changed.setflags(write=True)
        changed[0] = np.nextafter(changed[0], -np.inf)
        tampered_bounds[6] = (lower, changed)
        tampered = OperatorHZPreactivationFrame(
            bounds=tampered_bounds,
            receipt=frame.receipt,
            provenance_nonce=frame.provenance_nonce,
        )
        self.assertFalse(
            validate_operator_hz_preactivation_frame(
                tampered,
                net=net,
                require_live_provenance=False,
            )
        )

        different_net, _different_facts = _correlated_add_toy(
            bias=0.5
        )
        self.assertFalse(
            validate_operator_hz_preactivation_frame(
                frame, net=different_net
            )
        )

    def test_property_only_replay_consumes_c5_intersection(self):
        net, _facts, _built, frame = self._build()
        rows = np.asarray([[1.0]], dtype=np.float64)
        thresholds = np.asarray([0.3], dtype=np.float64)
        bundle = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(),
            steps=1,
            block_size=4,
            replay_chunk_size=4,
            candidate_device="cpu",
            solver_factory=_IntervalCandidateSolver,
            verified_preactivation_frame=frame,
            timeout_s=2.0,
        )
        self.assertTrue(
            validate_verified_query_dual_feedback(
                bundle,
                net=net,
                property_rows=rows,
                thresholds=thresholds,
                expected_target_relu_ids=(),
            )
        )
        lower = (
            bundle.certified_bounds[6]
            .lb.detach()
            .cpu()
            .double()
            .numpy()
            .reshape(-1)
        )
        upper = (
            bundle.certified_bounds[6]
            .ub.detach()
            .cpu()
            .double()
            .numpy()
            .reshape(-1)
        )
        self.assertLessEqual(float(lower[0]), 0.25)
        self.assertGreaterEqual(float(upper[0]), 0.25)
        self.assertLess(float(upper[0] - lower[0]), 1.0e-12)
        self.assertLess(float(bundle.property_upper[0]), 0.0)
        exact_violation = Fraction(1, 4) - Fraction.from_float(0.3)
        self.assertGreaterEqual(
            Fraction.from_float(float(bundle.property_upper[0])),
            exact_violation,
        )
        audit = bundle.receipt["initial_preactivation_frame"]
        self.assertTrue(audit["enabled"])
        self.assertGreater(
            audit["strict_lower_rows"] + audit["strict_upper_rows"],
            0,
        )

    def test_bound_frame_rejects_target_replay_mixing(self):
        net, _facts, _built, frame = self._build()
        with self.assertRaisesRegex(
            QueryDualPipelineError, "property-only"
        ):
            build_verified_query_dual_feedback(
                net,
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.3], dtype=np.float64),
                target_relu_ids=(6,),
                steps=1,
                block_size=4,
                replay_chunk_size=4,
                candidate_device="cpu",
                solver_factory=_IntervalCandidateSolver,
                verified_preactivation_frame=frame,
            )


if __name__ == "__main__":
    unittest.main()
