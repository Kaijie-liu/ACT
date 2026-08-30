#!/usr/bin/env python3
"""Controlled toy tests for the root-owned V5.1 replay transaction."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import gc
import hashlib
from types import MappingProxyType
import time
import unittest
from unittest import mock
import weakref

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay_v51_conv as conv_v51
from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf import query_dual_replay_v51 as replay_v51
from act.back_end.hybridz_tf import query_dual_replay_v51_session as session
from act.back_end.hybridz_tf import query_dual_v51_authority as authority
from act.back_end.hybridz_tf.query_dual_blas_contract import (
    QueryDualBlasContract,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.test_query_dual_box_certifier import (
    _input_pair,
    _layer,
    _net,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _five_stage_conv_net():
    """A point-input Conv/Dense chain with four controlled target ReLUs."""

    inp, spec = _input_pair(2, [1.0, 2.0], [1.0, 2.0])
    conv = _layer(
        2,
        "CONV2D",
        2,
        {
            "weight": np.asarray([[[[1.25]]]], dtype=np.float64),
            "bias": np.asarray([0.1], dtype=np.float64),
            "input_shape": (1, 1, 2),
            "output_shape": (1, 1, 2),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "padding_mode": "zeros",
        },
    )
    layers = [inp, spec, conv, _layer(3, "RELU", 2)]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    last = 3
    weight = np.asarray(
        [[1.0, 0.25], [-0.5, 1.0]], dtype=np.float64
    )
    bias = np.asarray([0.1, 0.2], dtype=np.float64)
    for dense_id, relu_id in ((4, 5), (6, 7), (8, 9)):
        layers.extend(
            [
                _layer(
                    dense_id,
                    "DENSE",
                    2,
                    {
                        "weight": weight.copy(),
                        "bias": bias.copy(),
                    },
                ),
                _layer(relu_id, "RELU", 2),
            ]
        )
        preds[dense_id] = [last]
        preds[relu_id] = [dense_id]
        last = relu_id
    layers.extend(
        [
            _layer(
                10,
                "DENSE",
                2,
                {
                    "weight": np.eye(2, dtype=np.float64),
                    "bias": np.zeros(2, dtype=np.float64),
                },
            ),
            _layer(11, "ASSERT", 2, {"kind": "AUDIT"}),
        ]
    )
    preds[10] = [last]
    preds[11] = [10]
    return _net(layers, preds)


def _stage_uses() -> tuple[authority.StageUse, ...]:
    return tuple(
        authority.StageUse(
            use_index=index,
            stage_kind=authority.STAGE_TARGET,
            stage_index=index,
            target_relu_lid=3 + 2 * index,
            cone_start_lid=2 + 2 * index,
        )
        for index in range(4)
    ) + (
        authority.StageUse(
            use_index=4,
            stage_kind=authority.STAGE_PROPERTY,
            stage_index=None,
            target_relu_lid=None,
            cone_start_lid=None,
        ),
    )


def _contains_ndarray(
    value, seen: set[int] | None = None
) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return False
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, MappingProxyType) or isinstance(value, dict):
        return any(
            _contains_ndarray(item, seen)
            for item in value.values()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_ndarray(item, seen) for item in value)
    if is_dataclass(value):
        return any(
            _contains_ndarray(getattr(value, item.name), seen)
            for item in fields(value)
            if item.name not in {"_owner"}
        )
    return False


class QueryDualReplayV51SessionTests(unittest.TestCase):
    def setUp(self):
        self.net = _five_stage_conv_net()
        self.root = certify_query_dual_boxes(self.net)
        self.blas = QueryDualBlasContract(
            required_threads=4,
            content_sha256=_sha("controlled-blas"),
            receipt=MappingProxyType({}),
        )
        self.blas_patch = mock.patch.object(
            session,
            "validate_query_dual_blas_contract",
            return_value=True,
        )
        self.blas_validator = self.blas_patch.start()

    def tearDown(self):
        self.blas_patch.stop()

    def _session(self, *, seconds: float = 60.0):
        return session.create_query_dual_replay_v51_session(
            self.net,
            self.root,
            _stage_uses(),
            deadline=time.monotonic() + seconds,
            blas_contract=self.blas,
        )

    def _frame(self, value=None):
        current = self._session() if value is None else value
        return current, current.seal_bounds(
            self.root.bounds,
            parent_chain_sha256=_sha("one-five-stage-chain"),
        )

    @staticmethod
    def _dense_queries(rows: int = 2) -> np.ndarray:
        return np.ascontiguousarray(
            np.resize(np.eye(2, dtype=np.float64), (rows, 2))
        )

    def _replay_all(
        self,
        value,
        frame,
        *,
        rows: int = 2,
        chunk_size: int = 1,
    ):
        return [
            value.replay(
                frame,
                stage_use_index=index,
                query_rows=self._dense_queries(rows),
                chunk_size=chunk_size,
            )
            for index in range(5)
        ]

    def test_five_stage_frame_commits_candidate_only(self):
        value, frame = self._frame()
        pending = [
            value.replay(
                frame,
                stage_use_index=index,
                query_rows=self._dense_queries(),
                chunk_size=1,
            )
            for index in range(5)
        ]
        self.assertTrue(all(not item.proof_authority for item in pending))
        self.assertEqual(value.unique_context_count, 5)
        self.assertEqual(len(frame._binding.stage_uses), 5)
        results = value.commit()
        self.assertEqual(len(results), 5)
        self.assertEqual(value.static_manifest_commit_validations, 1)
        self.assertTrue(
            all(
                session.validate_query_dual_replay_v51_session_candidate(
                    result
                )
                for result in results
            )
        )
        self.assertTrue(all(not result.proof_authority for result in results))
        copied = copy.copy(results[0])
        self.assertFalse(
            session.validate_query_dual_replay_v51_session_candidate(
                copied
            )
        )

    def test_same_frame_plan_cache_reuse_and_cross_frame_rebuild(self):
        value, first = self._frame()
        original = replay_v51.prepare_dense_conv_v51_plan
        with mock.patch.object(
            replay_v51,
            "prepare_dense_conv_v51_plan",
            wraps=original,
        ) as prepare:
            value.replay(
                first,
                stage_use_index=0,
                query_rows=self._dense_queries(4),
                chunk_size=1,
            )
            self.assertEqual(prepare.call_count, 1)
            builds = value.catalog_build_count
            value.replay(
                first,
                stage_use_index=0,
                query_rows=self._dense_queries(4),
                chunk_size=2,
            )
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(value.catalog_build_count, builds)
            self.assertGreater(value.catalog_hit_count, 0)

            second = value.seal_bounds(
                self.root.bounds,
                parent_chain_sha256=_sha("second-frame"),
            )
            value.replay(
                second,
                stage_use_index=0,
                query_rows=self._dense_queries(),
                chunk_size=1,
            )
            self.assertEqual(prepare.call_count, 2)
            self.assertIsNot(
                next(iter(first._catalog.values())).alias,
                next(iter(second._catalog.values())).alias,
            )
        value.abort()

    def test_cross_frame_catalog_transplant_is_rejected(self):
        value, first = self._frame()
        value.replay(
            first,
            stage_use_index=0,
            query_rows=self._dense_queries(),
            chunk_size=1,
        )
        second = value.seal_bounds(
            self.root.bounds,
            parent_chain_sha256=_sha("other-frame"),
        )
        second._catalog.update(first._catalog)
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as caught:
            value.replay(
                second,
                stage_use_index=0,
                query_rows=self._dense_queries(),
            )
        self.assertEqual(caught.exception.code, "INVALID_CATALOG")

    def test_query_span_gap_and_overlap_are_rejected(self):
        original = authority._mint_compact_guard_ledger
        for mode in ("gap", "overlap"):
            with self.subTest(mode=mode):
                value, frame = self._frame()

                def alter(owner, spans, expectations):
                    changed = list(spans)
                    second = spans[1]
                    start = (
                        second.query_start + 1
                        if mode == "gap"
                        else second.query_start - 1
                    )
                    changed[1] = authority._mint_query_span(
                        owner,
                        stage_use_sha256=second.stage_use_sha256,
                        span_index=second.span_index,
                        query_start=start,
                        query_end=second.query_end,
                        query_total=second.query_total,
                        query_block_sha256=second.query_block_sha256,
                        query_rows_sha256=second.query_rows_sha256,
                        query_bias_sha256=second.query_bias_sha256,
                        alpha_slice_sha256=second.alpha_slice_sha256,
                    )
                    return original(owner, changed, expectations)

                with mock.patch.object(
                    authority,
                    "_mint_compact_guard_ledger",
                    side_effect=alter,
                ):
                    with self.assertRaises(
                        session.QueryDualReplayV51SessionError
                    ) as caught:
                        value.replay(
                            frame,
                            stage_use_index=0,
                            query_rows=self._dense_queries(4),
                            chunk_size=2,
                        )
                self.assertEqual(
                    caught.exception.code,
                    (
                        "QUERY_SPAN_GAP"
                        if mode == "gap"
                        else "QUERY_SPAN_OVERLAP"
                    ),
                )

    def test_branch_policy_active_and_fallback_substitution_fail(self):
        mutations = {
            "branch": lambda record: record.__setitem__(
                "conv_branch", "sparse"
            ),
            "policy": lambda record: record.__setitem__(
                "policy", "frozen_v3_componentwise"
            ),
            "active": lambda record: record["active_mask"].__setitem__(
                0, not record["active_mask"][0]
            ),
            "fallback": lambda record: record[
                "fallback_mask"
            ].__setitem__(0, True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value, frame = self._frame()
                original = value._observe_execution

                def altered(**payload):
                    record = dict(payload["record"])
                    record["active_mask"] = list(record["active_mask"])
                    record["fallback_mask"] = list(
                        record["fallback_mask"]
                    )
                    mutate(record)
                    payload["record"] = MappingProxyType(record)
                    return original(**payload)

                with mock.patch.object(
                    value,
                    "_observe_execution",
                    side_effect=altered,
                ):
                    with self.assertRaises(
                        session.QueryDualReplayV51SessionError
                    ):
                        value.replay(
                            frame,
                            stage_use_index=0,
                            query_rows=self._dense_queries(),
                        )

    def test_missing_and_double_observer_trace_fail(self):
        for mode in ("missing", "double"):
            with self.subTest(mode=mode):
                value, frame = self._frame()
                original = value._observe_execution

                def changed(**payload):
                    if mode == "missing":
                        return None
                    original(**payload)
                    return original(**payload)

                with mock.patch.object(
                    value,
                    "_observe_execution",
                    side_effect=changed,
                ):
                    with self.assertRaises(
                        session.QueryDualReplayV51SessionError
                    ) as caught:
                        value.replay(
                            frame,
                            stage_use_index=0,
                            query_rows=self._dense_queries(),
                        )
                self.assertIn(
                    caught.exception.code,
                    {"MISSING_EXECUTION", "INVALID_EXECUTION"},
                )

    def test_sparse_conv_trace_carries_radius_and_exact_penalty(self):
        value, frame = self._frame()
        value.replay(
            frame,
            stage_use_index=0,
            query_rows=np.zeros((1, 2), dtype=np.float64),
        )
        ledger = value._pending[-1].ledger
        self.assertEqual(len(ledger.traces), 1)
        trace = ledger.traces[0]
        self.assertIsNotNone(trace.componentwise_radius_sha256)
        self.assertIsNotNone(trace.componentwise_penalty_sha256)
        self.assertIsNone(trace.scalar_guard_sha256)
        self.assertTrue(
            authority.validate_compact_guard_ledger_certificate(ledger)
        )
        value.abort()

    def test_concurrent_session_use_poisoning_is_fail_closed(self):
        value, frame = self._frame()
        value._operation_lock.acquire()
        try:
            with self.assertRaises(
                session.QueryDualReplayV51SessionError
            ) as caught:
                value.replay(
                    frame,
                    stage_use_index=0,
                    query_rows=self._dense_queries(),
                )
        finally:
            value._operation_lock.release()
        self.assertEqual(caught.exception.code, "CONCURRENT_SESSION")
        with self.assertRaises(session.QueryDualReplayV51SessionError):
            value.seal_bounds(self.root.bounds)

    def test_one_absolute_deadline_covers_replay_and_authority(self):
        with self.assertRaises(
            session.QueryDualReplayV51SessionTimeout
        ):
            session.create_query_dual_replay_v51_session(
                self.net,
                self.root,
                _stage_uses(),
                deadline=time.monotonic() - 1.0,
                blas_contract=self.blas,
            )

        value, frame = self._frame()
        end = float(value._deadline.end)
        with mock.patch.object(
            frozen.time, "monotonic", return_value=end
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                value.replay(
                    frame,
                    stage_use_index=0,
                    query_rows=self._dense_queries(),
                )

    def test_publication_deadline_removes_all_provisional_results(self):
        value, frame = self._frame()
        self._replay_all(value, frame)
        original_check = value._check
        original_validate = (
            session.validate_query_dual_replay_v51_session_candidate
        )
        check_count = [0]
        captured = []

        def expire_after_last_inner_check(runtime=None):
            result = original_check(runtime)
            check_count[0] += 1
            if check_count[0] == 4:
                value._deadline.end = time.monotonic() - 1.0
            return result

        def capture_result(result):
            captured.append(result)
            return original_validate(result)

        with mock.patch.object(
            value,
            "_check",
            side_effect=expire_after_last_inner_check,
        ), mock.patch.object(
            session,
            "validate_query_dual_replay_v51_session_candidate",
            side_effect=capture_result,
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                value.commit()
        self.assertEqual(len(captured), 5)
        self.assertTrue(
            all(
                not original_validate(result)
                for result in captured
            )
        )

    def test_live_network_mutation_fails_final_commit(self):
        value, frame = self._frame()
        self._replay_all(value, frame)
        self.net.by_id[2].params["weight"][0, 0, 0, 0] += 1.0
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as caught:
            value.commit()
        self.assertEqual(caught.exception.code, "LIVE_NET_MISMATCH")

    def test_blas_contract_is_revalidated_at_commit(self):
        self.blas_validator.side_effect = [True, False]
        value, frame = self._frame()
        self._replay_all(value, frame)
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as caught:
            value.commit()
        self.assertEqual(
            caught.exception.code, "BLAS_CONTRACT_MISMATCH"
        )

    def test_ledger_retains_no_execution_arrays_and_catalog_is_released(self):
        value, frame = self._frame()
        self._replay_all(value, frame, rows=4, chunk_size=1)
        ledger = value._pending[0].ledger
        self.assertFalse(_contains_ndarray(ledger))
        self.assertGreater(frame.catalog_entry_count, 0)
        result = value.commit()[0]
        self.assertEqual(frame.catalog_entry_count, 0)
        self.assertFalse(_contains_ndarray(result.receipt))
        self.assertTrue(
            session.validate_query_dual_replay_v51_session_candidate(
                result
            )
        )

    def test_commit_requires_exact_five_stage_single_frame_schedule(self):
        value, frame = self._frame()
        value.replay(
            frame,
            stage_use_index=0,
            query_rows=self._dense_queries(),
        )
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as partial:
            value.commit()
        self.assertEqual(partial.exception.code, "INCOMPLETE_SESSION")

        value, frame = self._frame()
        for _ in range(2):
            value.replay(
                frame,
                stage_use_index=0,
                query_rows=self._dense_queries(),
            )
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as duplicate:
            value.commit()
        self.assertEqual(duplicate.exception.code, "INCOMPLETE_SESSION")

        value, first = self._frame()
        second = value.seal_bounds(
            self.root.bounds,
            parent_chain_sha256=_sha("other-complete-frame"),
        )
        value.replay(
            first,
            stage_use_index=0,
            query_rows=self._dense_queries(),
        )
        for index in range(1, 5):
            value.replay(
                second,
                stage_use_index=index,
                query_rows=self._dense_queries(),
            )
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as mixed:
            value.commit()
        self.assertEqual(mixed.exception.code, "INCOMPLETE_SESSION")

    def test_session_external_seal_rejects_copy_and_self_resealing(self):
        value = self._session()
        copied = copy.copy(value)
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as copied_error:
            copied.seal_bounds(self.root.bounds)
        self.assertEqual(copied_error.exception.code, "INVALID_SESSION")
        frame = value.seal_bounds(self.root.bounds)
        value.abort()

        value = self._session()
        object.__setattr__(value, "_operation_lock", None)
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as lock:
            value.seal_bounds(self.root.bounds)
        self.assertEqual(lock.exception.code, "INVALID_SESSION")

        value, frame = self._frame()
        replacement = {}
        object.__setattr__(frame, "_catalog", replacement)
        value._frame_identity[frame._frame_nonce] = (
            id(frame._bounds),
            id(frame._binding),
            id(frame._owner),
            id(replacement),
        )
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as frame_error:
            value.replay(
                frame,
                stage_use_index=0,
                query_rows=self._dense_queries(),
            )
        self.assertEqual(frame_error.exception.code, "INVALID_FRAME")

    def test_pending_stage_and_deadline_external_seals_fail_closed(self):
        value, frame = self._frame()
        self._replay_all(value, frame)
        object.__setattr__(
            value._pending[0], "stage_use_index", 4
        )
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as pending:
            value.commit()
        self.assertEqual(pending.exception.code, "INVALID_STAGE")

        value = self._session()
        changed = float(value._deadline.end) + 1000.0
        value._deadline.end = changed
        value._deadline_end = changed
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as deadline:
            value.seal_bounds(self.root.bounds)
        self.assertEqual(deadline.exception.code, "INVALID_SESSION")

    def test_session_and_result_are_pid_and_container_bound(self):
        value, frame = self._frame()
        self._replay_all(value, frame)
        result = value.commit()[0]
        original_receipt = result.receipt
        object.__setattr__(result, "receipt", dict(original_receipt))
        self.assertFalse(
            session.validate_query_dual_replay_v51_session_candidate(
                result
            )
        )
        object.__setattr__(result, "receipt", original_receipt)
        self.assertTrue(
            session.validate_query_dual_replay_v51_session_candidate(
                result
            )
        )
        with mock.patch.object(
            session.os,
            "getpid",
            return_value=session._SESSION_PID + 1,
        ):
            self.assertFalse(
                session.validate_query_dual_replay_v51_session_candidate(
                    result
                )
            )

    def test_public_frozen_errors_are_normalized(self):
        value = self._session()
        malformed = dict(self.root.bounds)
        malformed[2] = object()
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as bounds_error:
            value.seal_bounds(malformed)
        self.assertEqual(bounds_error.exception.code, "MISSING_BOUNDS")

        value, frame = self._frame()
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as query_error:
            value.replay(
                frame,
                stage_use_index=0,
                query_rows=np.ones((1, 3), dtype=np.float64),
            )
        self.assertEqual(query_error.exception.code, "SHAPE_MISMATCH")

    def test_late_create_and_frame_owner_deadlines_are_normalized(self):
        with mock.patch.object(
            frozen._Deadline,
            "check",
            autospec=True,
            side_effect=[None, frozen.QueryDualReplayTimeout()],
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                session.create_query_dual_replay_v51_session(
                    self.net,
                    self.root,
                    _stage_uses(),
                    deadline=time.monotonic() + 60.0,
                    blas_contract=self.blas,
                )

        value = self._session()
        with mock.patch.object(
            authority,
            "_mint_frame_owner",
            side_effect=authority.QueryDualV51AuthorityError(
                "DEADLINE_EXPIRED", "controlled frame-owner deadline"
            ),
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                value.seal_bounds(self.root.bounds)

    def test_cached_conv_and_blas_deadlines_are_normalized(self):
        value, frame = self._frame()
        value.replay(
            frame,
            stage_use_index=0,
            query_rows=self._dense_queries(),
            chunk_size=1,
        )
        self.assertTrue(
            any(
                entry.material_kind == "CONV_PLAN"
                for entry in frame._catalog.values()
            )
        )
        with mock.patch.object(
            conv_v51,
            "_validate_plan",
            side_effect=frozen.QueryDualReplayTimeout(),
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                value.replay(
                    frame,
                    stage_use_index=0,
                    query_rows=self._dense_queries(),
                    chunk_size=1,
                )

        value, frame = self._frame()
        self._replay_all(value, frame)
        self.blas_validator.return_value = False
        with mock.patch.object(
            value._deadline,
            "check",
            side_effect=[None, frozen.QueryDualReplayTimeout()],
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionTimeout
            ):
                value.commit()

    def test_commit_platform_error_is_normalized(self):
        value, frame = self._frame()
        self._replay_all(value, frame)
        self.blas_validator.return_value = True
        with mock.patch.object(
            frozen,
            "_check_numeric_platform",
            side_effect=frozen.QueryDualReplayError(
                "PLATFORM_DRIFT", "controlled platform drift"
            ),
        ):
            with self.assertRaises(
                session.QueryDualReplayV51SessionError
            ) as caught:
                value.commit()
        self.assertEqual(caught.exception.code, "PLATFORM_DRIFT")
        self.assertTrue(value._operation_lock.acquire(blocking=False))
        value._operation_lock.release()

    def test_failed_session_finalizer_releases_orphan_catalog(self):
        value, frame = self._frame()
        value.replay(
            frame,
            stage_use_index=0,
            query_rows=self._dense_queries(),
            chunk_size=1,
        )
        self.assertGreater(frame.catalog_entry_count, 0)
        nonce = value._nonce
        reference = weakref.ref(value)
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ):
            value.replay(
                frame,
                stage_use_index=1,
                query_rows=np.ones((1, 3), dtype=np.float64),
            )
        del value
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(nonce, session._SESSION_SEALS)
        self.assertEqual(frame.catalog_entry_count, 0)

    def test_registry_and_catalog_corruption_have_stable_errors(self):
        value = self._session()
        value._nonce = []
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as registry_error:
            value.seal_bounds(self.root.bounds)
        self.assertEqual(registry_error.exception.code, "INVALID_SESSION")

        value, frame = self._frame()
        value.replay(
            frame,
            stage_use_index=0,
            query_rows=self._dense_queries(),
            chunk_size=1,
        )
        key = next(iter(frame._catalog))
        frame._catalog[key] = object()
        with self.assertRaises(
            session.QueryDualReplayV51SessionError
        ) as catalog_error:
            value.replay(
                frame,
                stage_use_index=0,
                query_rows=self._dense_queries(),
                chunk_size=1,
            )
        self.assertEqual(catalog_error.exception.code, "INVALID_CATALOG")


if __name__ == "__main__":
    unittest.main()
