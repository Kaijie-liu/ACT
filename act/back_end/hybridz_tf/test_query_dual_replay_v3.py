"""Controlled equivalence and fail-closed tests for sealed replay V3."""

from __future__ import annotations

import copy
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest import mock

import numpy as np

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import query_dual_box_certifier as box_module
from act.back_end.hybridz_tf import query_dual_replay as replay_module
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    QueryDualReplayError,
    QueryDualReplayTimeout,
    create_query_dual_replay_session,
    fraction_replay_lower_bounds,
    replay_query_lower_bounds,
    validate_query_dual_replay_result,
    verify_query_dual_replay_receipt,
)
from act.back_end.hybridz_tf.test_query_dual_box_certifier import (
    _cancellation_net,
    _grouped_conv_net,
    _residual_net,
)


def _deadline() -> float:
    return time.monotonic() + 30.0


def _cloned_bounds(certificate):
    return {
        lid: Bounds(lb=value.lb.clone(), ub=value.ub.clone())
        for lid, value in certificate.bounds.items()
    }


def _assert_v1_v3_equal(test, old, new):
    test.assertTrue(np.array_equal(old.lower_bounds, new.lower_bounds))
    test.assertEqual(
        [float(value).hex() for value in old.lower_bounds],
        [float(value).hex() for value in new.lower_bounds],
    )
    test.assertEqual(
        old.receipt["lower_bounds_sha256"],
        new.receipt["lower_bounds_sha256"],
    )
    test.assertEqual(dict(old.receipt["hashes"]), dict(new.receipt["hashes"]))
    test.assertEqual(new.receipt["schema"], "act.query_dual_replay.v2")
    test.assertTrue(validate_query_dual_replay_result(new))
    test.assertTrue(verify_query_dual_replay_receipt(new.receipt))


class QueryDualReplayV3Tests(unittest.TestCase):
    def _compare(self, net, query_rows, *, alpha=None):
        certificate = certify_query_dual_boxes(net, conv_channel_chunk=1)
        old = replay_query_lower_bounds(
            net,
            certificate.bounds,
            query_rows=query_rows,
            alpha_by_relu=alpha,
            chunk_size=1,
        )
        oracle = fraction_replay_lower_bounds(
            net,
            certificate.bounds,
            query_rows=query_rows,
            alpha_by_relu=alpha,
        )
        session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        frame = session.seal_bounds(certificate.bounds)
        pending = session.replay(
            frame,
            query_rows=query_rows,
            alpha_by_relu=alpha,
            chunk_size=1,
        )
        self.assertFalse(pending.proof_authority)
        new = session.commit()[0]
        _assert_v1_v3_equal(self, old, new)
        for numeric, exact in zip(new.lower_bounds, oracle):
            self.assertLessEqual(Fraction.from_float(float(numeric)), exact)
        sealed = new.receipt["sealed_context"]
        crosswalk = sealed["manifest_crosswalk"]
        self.assertTrue(crosswalk["hashes_are_crosswalked_not_compared"])
        self.assertEqual(
            sealed["root_net_sha256"], crosswalk["root_net_sha256"]
        )
        self.assertEqual(
            sealed["replay_net_sha256"],
            old.receipt["hashes"]["net_sha256"],
        )

    def test_dense_relu_bit_exact_equivalence(self):
        net = _cancellation_net()
        queries = np.asarray([[1.0], [-1.0]], dtype=np.float64)
        self._compare(
            net,
            queries,
            alpha={3: np.asarray([[0.25], [0.75]], dtype=np.float64)},
        )

    def test_add_residual_dag_bit_exact_equivalence(self):
        net = _residual_net()
        queries = np.asarray(
            [[1.0, -1.0], [-0.25, 0.75], [2.0, 0.5]], dtype=np.float64
        )
        alpha = {
            3: np.asarray(
                [[0.2, 0.8], [0.4, 0.6], [0.9, 0.1]],
                dtype=np.float64,
            )
        }
        self._compare(net, queries, alpha=alpha)

    def test_grouped_conv_bit_exact_equivalence(self):
        net, _ = _grouped_conv_net()
        width = int(np.prod((4, 3, 4)))
        queries = np.zeros((3, width), dtype=np.float64)
        queries[0, 0] = 1.0
        queries[1, -1] = -1.0
        queries[2, width // 2] = 0.375
        self._compare(net, queries)

    def test_four_worker_rows_are_bit_exact_and_fraction_sound(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        base = np.asarray(
            [[1.0, -1.0], [-0.25, 0.75], [2.0, 0.5], [-0.5, -1.5]],
            dtype=np.float64,
        )
        queries = np.tile(base, (4, 1))
        alpha = {
            3: np.tile(
                np.asarray(
                    [[0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3]],
                    dtype=np.float64,
                ),
                (4, 1),
            )
        }

        serial_session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        serial_frame = serial_session.seal_bounds(certificate.bounds)
        serial_session.replay(
            serial_frame,
            query_rows=queries,
            alpha_by_relu=alpha,
            chunk_size=1,
            proof_workers=1,
        )
        serial = serial_session.commit()[0]

        parallel_session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        parallel_frame = parallel_session.seal_bounds(certificate.bounds)
        parallel_session.replay(
            parallel_frame,
            query_rows=queries,
            alpha_by_relu=alpha,
            chunk_size=1,
            proof_workers=4,
        )
        parallel = parallel_session.commit()[0]

        self.assertTrue(
            np.array_equal(serial.lower_bounds, parallel.lower_bounds)
        )
        self.assertEqual(
            serial.receipt["lower_bounds_sha256"],
            parallel.receipt["lower_bounds_sha256"],
        )
        workers = parallel.receipt["proof_row_parallelism"]
        self.assertEqual(workers["requested_workers"], 4)
        self.assertEqual(workers["effective_workers"], 4)
        self.assertFalse(workers["partial_authority"])
        self.assertTrue(validate_query_dual_replay_result(parallel))
        oracle = fraction_replay_lower_bounds(
            net,
            certificate.bounds,
            query_rows=queries,
            alpha_by_relu=alpha,
        )
        for numeric, exact in zip(parallel.lower_bounds, oracle):
            self.assertLessEqual(Fraction.from_float(float(numeric)), exact)

    def test_parallel_worker_failure_is_atomic_and_invalidates_session(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        nonce = session._nonce
        frame = session.seal_bounds(certificate.bounds)
        with mock.patch.object(
            replay_module,
            "_replay_block",
            side_effect=QueryDualReplayTimeout(),
        ):
            with self.assertRaises(QueryDualReplayTimeout):
                session.replay(
                    frame,
                    query_rows=np.tile(np.eye(2), (4, 1)),
                    chunk_size=1,
                    proof_workers=4,
                )
        self.assertFalse(session._pending)
        self.assertNotIn(nonce, replay_module._SEALED_SESSION_REGISTRY)
        with self.assertRaises(QueryDualReplayError) as caught:
            session.commit()
        self.assertEqual(caught.exception.code, "INVALID_SESSION")

    def test_parallel_worker_count_validation_fails_closed(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        for invalid in (0, 33, True):
            session = create_query_dual_replay_session(
                net, certificate, [None], deadline=_deadline()
            )
            frame = session.seal_bounds(certificate.bounds)
            with self.assertRaises(QueryDualReplayError) as caught:
                session.replay(
                    frame,
                    query_rows=np.eye(2),
                    proof_workers=invalid,
                )
            self.assertEqual(caught.exception.code, "INVALID_WORKERS")

    def test_unique_cones_and_no_static_manifest_hash_in_stage_loop(self):
        net = _residual_net()
        with mock.patch.object(
            box_module, "_freeze_graph", wraps=box_module._freeze_graph
        ) as root_freeze, mock.patch.object(
            replay_module, "_freeze_layer", wraps=replay_module._freeze_layer
        ) as legacy_layer_freeze, mock.patch.object(
            replay_module,
            "_layer_manifest",
            wraps=replay_module._layer_manifest,
        ) as static_layer_hash:
            certificate = certify_query_dual_boxes(net)
            session = create_query_dual_replay_session(
                net,
                certificate,
                [2, 4, None, 2, 4],
                deadline=_deadline(),
            )
            self.assertEqual(session.unique_context_count, 3)
            # One manifest build per full frozen layer, not once per
            # overlapping cone.
            self.assertEqual(static_layer_hash.call_count, len(net.layers))
            static_layer_hash.reset_mock()
            targeted = session.seal_bounds(
                certificate.bounds, start_lids=[2]
            )
            expected_targeted = {
                lid
                for lid in session._contexts[2].reverse_order
                if session._contexts[2].layers[lid].kind != "INPUT"
            }
            self.assertEqual(set(targeted._bounds), expected_targeted)
            self.assertLess(
                len(targeted._bounds), len(certificate.bounds)
            )
            frame = session.seal_bounds(certificate.bounds)
            with mock.patch.object(
                session,
                "_build_crosswalk",
                wraps=session._build_crosswalk,
            ) as rebuild:
                session.replay(frame, start_lid=2, query_rows=np.eye(2))
                session.replay(frame, start_lid=4, one_hot=[0, 1])
                session.replay(frame, query_rows=np.eye(2))
                self.assertEqual(rebuild.call_count, 0)
                self.assertEqual(static_layer_hash.call_count, 0)
                committed = session.commit()
                self.assertEqual(rebuild.call_count, 1)
                unique_reachable = {
                    lid
                    for cone in session._contexts.values()
                    for lid in cone.reverse_order
                }
                self.assertEqual(
                    static_layer_hash.call_count, len(unique_reachable)
                )
            self.assertEqual(len(committed), 3)
            self.assertEqual(session.static_manifest_commit_validations, 1)
            # Root certification freezes once and the final live bind once.
            self.assertEqual(root_freeze.call_count, 2)
            # V3 never enters the legacy live-layer freezer.
            self.assertEqual(legacy_layer_freeze.call_count, 0)
            replay_query_lower_bounds(
                net, certificate.bounds, start_lid=2, query_rows=np.eye(2)
            )
            replay_query_lower_bounds(
                net, certificate.bounds, start_lid=4, one_hot=[0, 1]
            )
            replay_query_lower_bounds(
                net, certificate.bounds, query_rows=np.eye(2)
            )
            expected_legacy_freezes = sum(
                len(session._contexts[key].reverse_order)
                for key in (2, 4, None)
            )
            self.assertEqual(
                legacy_layer_freeze.call_count, expected_legacy_freezes
            )

    def test_sealed_receipt_recomputes_unique_cone_partition(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        session = create_query_dual_replay_session(
            net, certificate, [2, 4, None], deadline=_deadline()
        )
        frame = session.seal_bounds(
            certificate.bounds, start_lids=(2,)
        )
        session.replay(frame, start_lid=2, query_rows=np.eye(2))
        committed = session.commit()[0]
        self.assertTrue(
            verify_query_dual_replay_receipt(committed.receipt)
        )

        forged = copy.deepcopy(dict(committed.receipt))
        sealed = forged["sealed_context"]
        sealed["unique_cone_count"] += 1
        context_body = dict(sealed)
        context_body.pop("context_sha256")
        sealed["context_sha256"] = replay_module._json_digest(
            context_body
        )
        receipt_body = dict(forged)
        receipt_body.pop("receipt_sha256")
        forged["receipt_sha256"] = replay_module._json_digest(
            receipt_body
        )
        self.assertFalse(verify_query_dual_replay_receipt(forged))

    def test_bounds_queries_alpha_and_pending_are_private_clones(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        stage_bounds = _cloned_bounds(certificate)
        baseline_bounds = _cloned_bounds(certificate)
        queries = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
        alpha = {3: np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)}
        old = replay_query_lower_bounds(
            net,
            baseline_bounds,
            query_rows=queries.copy(),
            alpha_by_relu={3: alpha[3].copy()},
        )
        session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        frame = session.seal_bounds(stage_bounds)
        stage_bounds[1].lb.fill_(-999.0)
        pending = session.replay(
            frame,
            query_rows=queries,
            alpha_by_relu=alpha,
        )
        queries.fill(123.0)
        alpha[3].fill(1.0)
        with self.assertRaises(ValueError):
            pending.lower_bounds.setflags(write=True)
        with self.assertRaises(ValueError):
            pending.lower_bounds[0] = 0.0
        new = session.commit()[0]
        _assert_v1_v3_equal(self, old, new)

    def test_root_start_frame_session_and_live_mutation_fail_closed(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        sealed_graph = certificate._sealed_frozen_graph
        frozen_dense = next(
            layer for layer in sealed_graph.layers if layer.kind == "DENSE"
        )
        frozen_weight = frozen_dense.params["weight"]
        self.assertEqual(frozen_weight.dtype, np.float64)
        self.assertTrue(frozen_weight.flags.c_contiguous)
        self.assertFalse(frozen_weight.flags.writeable)
        with self.assertRaises(ValueError):
            frozen_weight.setflags(write=True)
        with self.assertRaises(TypeError):
            frozen_dense.params["weight"] = np.zeros_like(frozen_weight)
        forged_root = replace(certificate, _sealed_frozen_graph=None)
        self.assertTrue(verify_query_dual_box_certificate(forged_root))
        with self.assertRaises(QueryDualReplayError) as caught:
            create_query_dual_replay_session(
                net, forged_root, [None], deadline=_deadline()
            )
        self.assertEqual(caught.exception.code, "INVALID_ROOT_CERTIFICATE")

        with self.assertRaises(QueryDualReplayError) as caught:
            create_query_dual_replay_session(
                net, certificate, [999], deadline=_deadline()
            )
        self.assertEqual(caught.exception.code, "INVALID_START_LAYER")

        first = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        second = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        foreign_frame = first.seal_bounds(certificate.bounds)
        with self.assertRaises(QueryDualReplayError) as caught:
            second.replay(foreign_frame, one_hot=[0])
        self.assertEqual(caught.exception.code, "INVALID_FRAME")

        live = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        frame = live.seal_bounds(certificate.bounds)
        live.replay(frame, one_hot=[0])
        net.layers[2].params["weight"][0, 0] += 0.125
        with self.assertRaises(QueryDualReplayError) as caught:
            live.commit()
        self.assertEqual(caught.exception.code, "LIVE_NET_MISMATCH")
        with self.assertRaises(QueryDualReplayError) as caught:
            live.commit()
        self.assertEqual(caught.exception.code, "INVALID_SESSION")

    def test_root_semantics_and_manifest_share_one_owned_parameter_read(self):
        net = _residual_net()
        baseline = certify_query_dual_boxes(net)
        live_weight = net.layers[2].params["weight"]
        original_weight = live_weight.copy()
        original_capture = box_module._snapshot_manifest_value
        mutated = False

        def capture_then_mutate(value, *, name):
            nonlocal mutated
            captured = original_capture(value, name=name)
            if name == "layer[2].params['weight']" and not mutated:
                live_weight[0, 0] += 0.5
                mutated = True
            return captured

        with mock.patch.object(
            box_module,
            "_snapshot_manifest_value",
            side_effect=capture_then_mutate,
        ):
            certificate = certify_query_dual_boxes(net)
        self.assertTrue(mutated)
        self.assertEqual(
            certificate.receipt["hashes"]["net_sha256"],
            baseline.receipt["hashes"]["net_sha256"],
        )
        for lid in baseline.bounds:
            self.assertTrue(
                np.array_equal(
                    certificate.bounds[lid].lb.numpy(),
                    baseline.bounds[lid].lb.numpy(),
                )
            )
            self.assertTrue(
                np.array_equal(
                    certificate.bounds[lid].ub.numpy(),
                    baseline.bounds[lid].ub.numpy(),
                )
            )
        self.assertFalse(
            verify_query_dual_box_certificate(certificate, net=net)
        )
        live_weight[...] = original_weight
        self.assertTrue(
            verify_query_dual_box_certificate(certificate, net=net)
        )

    def test_context_identity_and_deadlines_fail_closed(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        with self.assertRaises(QueryDualReplayTimeout):
            create_query_dual_replay_session(
                net,
                certificate,
                [None],
                deadline=time.monotonic() - 1.0,
            )

        expired = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        with mock.patch.object(
            replay_module.time,
            "monotonic",
            return_value=expired._deadline_end_seal + 1.0,
        ):
            with self.assertRaises(QueryDualReplayTimeout):
                expired.seal_bounds(certificate.bounds)

        extended = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        extended._deadline.end += 60.0
        with self.assertRaises(QueryDualReplayError) as caught:
            extended.seal_bounds(certificate.bounds)
        self.assertEqual(caught.exception.code, "INVALID_DEADLINE")

        tampered = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        tampered._contexts = {}
        with self.assertRaises(QueryDualReplayError) as caught:
            tampered.seal_bounds(certificate.bounds)
        self.assertEqual(caught.exception.code, "INVALID_CONTEXT")

    def test_abort_discards_pending_resources_and_never_promotes(self):
        net = _residual_net()
        certificate = certify_query_dual_boxes(net)
        session = create_query_dual_replay_session(
            net, certificate, [None], deadline=_deadline()
        )
        nonce = session._nonce
        frame = session.seal_bounds(certificate.bounds)
        pending = session.replay(frame, one_hot=[0])
        self.assertFalse(pending.proof_authority)
        self.assertIn(nonce, replay_module._SEALED_SESSION_REGISTRY)
        self.assertTrue(session._frames)
        self.assertTrue(session._pending)
        # Models a selector/candidate/coverage exception outside replay.
        session.abort()
        session.abort()
        self.assertNotIn(nonce, replay_module._SEALED_SESSION_REGISTRY)
        self.assertFalse(session._frames)
        self.assertFalse(session._frame_bounds_identities)
        self.assertFalse(session._pending)
        self.assertTrue(session._closed)
        with self.assertRaises(QueryDualReplayError) as caught:
            session.commit()
        self.assertEqual(caught.exception.code, "INVALID_SESSION")


if __name__ == "__main__":
    unittest.main()
