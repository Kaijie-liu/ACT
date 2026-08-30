#!/usr/bin/env python3
"""Controlled tests for the isolated V5.1 semantic authority sidecar."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
import hashlib
import threading
import time
import unittest
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_v51_authority as authority


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _stage_uses() -> tuple[authority.StageUse, ...]:
    targets = tuple(
        authority.StageUse(
            use_index=index,
            stage_kind=authority.STAGE_TARGET,
            stage_index=index,
            target_relu_lid=100 + index,
            cone_start_lid=90 + index,
        )
        for index in range(4)
    )
    return targets + (
        authority.StageUse(
            use_index=4,
            stage_kind=authority.STAGE_PROPERTY,
            stage_index=None,
            target_relu_lid=None,
            cone_start_lid=None,
        ),
    )


def _binding(
    *,
    frame: str = "frame-a",
    deadline: float | None = None,
    stage_uses: tuple[authority.StageUse, ...] | None = None,
) -> authority.V51FrameBinding:
    end = time.monotonic() + 30.0 if deadline is None else float(deadline)
    return authority.V51FrameBinding(
        session_nonce_sha256=_sha("session"),
        frame_nonce_sha256=_sha(frame),
        frame_content_sha256=_sha(f"{frame}:content"),
        bounds_manifest_sha256=_sha(f"{frame}:bounds"),
        root_receipt_sha256=_sha("root"),
        parent_chain_sha256=_sha(f"{frame}:parent"),
        deadline_monotonic_hex=end.hex(),
        stage_uses=_stage_uses() if stage_uses is None else stage_uses,
    )


def _alias(
    owner,
    *,
    use_index: int = 0,
    layer_id: int = 11,
    predecessor_id: int = 90,
    operator_kind: str = "DENSE",
    branch: str = authority.BRANCH_DENSE,
):
    stage_use = owner.binding.stage_uses[use_index]
    return authority._mint_support_catalog_alias(
        owner,
        stage_use_sha256=stage_use.stage_use_sha256,
        layer_id=layer_id,
        predecessor_id=predecessor_id,
        operator_kind=operator_kind,
        branch=branch,
        box_semantics="output",
        catalog_content_sha256=_sha(
            f"catalog:{use_index}:{layer_id}:{branch}"
        ),
        support_content_sha256=_sha(
            f"support:{use_index}:{layer_id}:{branch}"
        ),
        weight_sha256=_sha(f"weight:{layer_id}"),
        geometry_sha256=_sha(f"geometry:{layer_id}"),
        source_lb_sha256=_sha(f"lb:{predecessor_id}"),
        source_ub_sha256=_sha(f"ub:{predecessor_id}"),
        numeric_platform_sha256=_sha("platform"),
        implementation_sha256=_sha("implementation"),
        branch_evidence_sha256=_sha(f"branch:{branch}"),
    )


def _span(
    owner,
    *,
    span_index: int,
    start: int,
    end: int,
    total: int,
    use_index: int = 0,
    block: str = "block-a",
):
    return authority._mint_query_span(
        owner,
        stage_use_sha256=(
            owner.binding.stage_uses[use_index].stage_use_sha256
        ),
        span_index=span_index,
        query_start=start,
        query_end=end,
        query_total=total,
        query_block_sha256=_sha(block),
        query_rows_sha256=_sha(f"{block}:rows:{start}:{end}"),
        query_bias_sha256=_sha(f"{block}:bias:{start}:{end}"),
        alpha_slice_sha256=_sha(f"{block}:alpha:{start}:{end}"),
    )


def _scalar_partition(
    rows: int,
    *,
    active: tuple[int, ...] = (),
    fallback: tuple[int, ...] = (),
) -> authority.RowPolicyPartition:
    active_values = np.zeros(rows, dtype=np.bool_)
    fallback_values = np.zeros(rows, dtype=np.bool_)
    active_values[list(active)] = True
    fallback_values[list(fallback)] = True
    return authority.RowPolicyPartition(
        row_count=rows,
        scalar_mask=authority._full_mask(rows),
        componentwise_mask=bytes(authority._mask_length(rows)),
        active_mask=authority._mask_from_bool(active_values),
        fallback_mask=authority._mask_from_bool(fallback_values),
    )


def _componentwise_partition(
    rows: int,
) -> authority.RowPolicyPartition:
    return authority.RowPolicyPartition(
        row_count=rows,
        scalar_mask=bytes(authority._mask_length(rows)),
        componentwise_mask=authority._full_mask(rows),
        active_mask=bytes(authority._mask_length(rows)),
        fallback_mask=bytes(authority._mask_length(rows)),
    )


def _expectation(
    owner,
    *,
    execution_index: int,
    span,
    alias,
    partition,
):
    return authority._mint_affine_execution_expectation(
        owner,
        execution_index=execution_index,
        span=span,
        support_alias=alias,
        partition=partition,
        input_coefficient_sha256=_sha(
            f"coefficient:{execution_index}:{span.query_start}"
        ),
    )


def _scalar_trace(owner, expectation):
    rows = expectation.span.row_count
    width = 3
    nominal = np.ascontiguousarray(
        np.arange(rows * width, dtype=np.float64).reshape(rows, width)
    )
    before = np.ascontiguousarray(
        np.linspace(10.0, 10.0 + rows - 1, rows, dtype=np.float64)
    )
    guard = np.zeros(rows, dtype=np.float64)
    active = authority._mask_to_bool(
        expectation.partition.active_mask, row_count=rows
    )
    guard[active] = np.arange(1, np.count_nonzero(active) + 1) * 0.25
    after = before.copy()
    if np.any(active):
        after[active] = np.nextafter(
            before[active] - guard[active], -np.inf
        )
    return authority._mint_compact_absorption_trace(
        owner,
        expectation,
        nominal=nominal,
        scalar_before=before,
        scalar_after=np.ascontiguousarray(after),
        scalar_guard=np.ascontiguousarray(guard),
    )


def _componentwise_trace(owner, expectation):
    rows = expectation.span.row_count
    width = 4
    nominal = np.ascontiguousarray(
        np.arange(rows * width, dtype=np.float64).reshape(rows, width)
    )
    before = np.ascontiguousarray(
        np.linspace(4.0, 4.0 + rows - 1, rows, dtype=np.float64)
    )
    radius = np.full((rows, width), 0.125, dtype=np.float64)
    penalty = np.full(rows, 0.5, dtype=np.float64)
    after = np.nextafter(before - penalty, -np.inf)
    return authority._mint_compact_absorption_trace(
        owner,
        expectation,
        nominal=nominal,
        scalar_before=before,
        scalar_after=np.ascontiguousarray(after),
        componentwise_radius=np.ascontiguousarray(radius),
        componentwise_penalty=np.ascontiguousarray(penalty),
    )


def _ready_scalar_ledger(*, spans_count: int = 1):
    owner = authority._mint_frame_owner(_binding())
    alias = _alias(owner)
    spans = tuple(
        _span(
            owner,
            span_index=index,
            start=index,
            end=index + 1,
            total=spans_count,
        )
        for index in range(spans_count)
    )
    expectations = tuple(
        _expectation(
            owner,
            execution_index=index,
            span=span,
            alias=alias,
            partition=_scalar_partition(1, active=(0,)),
        )
        for index, span in enumerate(spans)
    )
    traces = tuple(
        _scalar_trace(owner, expectation)
        for expectation in expectations
    )
    ledger = authority._mint_compact_guard_ledger(
        owner, spans, expectations
    )
    return owner, alias, spans, expectations, traces, ledger


def _contains_ndarray(value, seen: set[int] | None = None) -> bool:
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
    if isinstance(value, dict) or hasattr(value, "items"):
        try:
            return any(
                _contains_ndarray(key, seen)
                or _contains_ndarray(item, seen)
                for key, item in value.items()
            )
        except TypeError:
            return False
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_ndarray(item, seen) for item in value)
    if is_dataclass(value):
        return any(
            _contains_ndarray(getattr(value, item.name), seen)
            for item in fields(value)
            if item.name not in {"_owner"}
        )
    return False


class QueryDualV51AuthorityTests(unittest.TestCase):
    def test_multi_stage_frame_canonicalizes_five_uses_and_cones(self):
        binding = _binding()
        self.assertTrue(authority.validate_frame_binding(binding))
        self.assertEqual(len(binding.stage_uses), 5)
        self.assertEqual(
            [value.use_index for value in binding.stage_uses],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            binding.stage_uses[-1].stage_kind,
            authority.STAGE_PROPERTY,
        )
        copied = replace(
            binding,
            frame_content_sha256=binding.frame_content_sha256,
        )
        self.assertTrue(authority.validate_frame_binding(copied))
        object.__setattr__(copied, "binding_sha256", _sha("forged"))
        self.assertFalse(authority.validate_frame_binding(copied))

        broken = list(_stage_uses())
        broken[1] = authority.StageUse(
            use_index=7,
            stage_kind=authority.STAGE_TARGET,
            stage_index=1,
            target_relu_lid=101,
            cone_start_lid=91,
        )
        with self.assertRaises(authority.QueryDualV51AuthorityError):
            _binding(stage_uses=tuple(broken))

    def test_frame_owner_is_process_local_and_deadline_is_fail_closed(self):
        owner = authority._mint_frame_owner(_binding())
        self.assertTrue(authority._verify_owner(owner))
        owner_copy = copy.copy(owner)
        self.assertIsNot(owner_copy, owner)
        self.assertFalse(authority._verify_owner(owner_copy))

        expired = _binding(deadline=time.monotonic() - 1.0)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            authority._mint_frame_owner(expired)
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

    def test_authority_objects_are_not_inherited_across_fork_pid(self):
        owner = authority._mint_frame_owner(_binding())
        child_pid = authority._AUTHORITY_PID + 1
        with mock.patch.object(
            authority.os, "getpid", return_value=child_pid
        ):
            self.assertFalse(authority._verify_owner(owner))
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as caught:
                authority._mint_frame_owner(_binding())
        self.assertEqual(caught.exception.code, "PROCESS_MISMATCH")
        self.assertTrue(authority._verify_owner(owner))

    def test_live_registered_objects_keep_mint_time_external_seals(self):
        owner = authority._mint_frame_owner(_binding())
        replacement = _binding(frame="frame-b")
        object.__setattr__(owner, "binding", replacement)
        self.assertFalse(authority._verify_owner(owner))

        owner = authority._mint_frame_owner(_binding(frame="frame-c"))
        span = _span(
            owner, span_index=0, start=0, end=2, total=2
        )
        replacement_rows = _sha("replacement-rows")
        object.__setattr__(
            span, "query_rows_sha256", replacement_rows
        )
        stage_use = owner.binding.stage_uses[0]
        body = authority._span_body(
            owner=owner,
            stage_use=stage_use,
            span_index=span.span_index,
            query_start=span.query_start,
            query_end=span.query_end,
            query_total=span.query_total,
            query_block_sha256=span.query_block_sha256,
            query_rows_sha256=replacement_rows,
            query_bias_sha256=span.query_bias_sha256,
            alpha_slice_sha256=span.alpha_slice_sha256,
            span_nonce_sha256=hashlib.sha256(
                span._nonce.encode("ascii")
            ).hexdigest(),
        )
        object.__setattr__(
            span, "content_sha256", authority._json_sha256(body)
        )
        self.assertFalse(authority.validate_query_span(span))

    def test_catalog_alias_binds_frame_cone_branch_and_cache_identity(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        cached = _alias(owner)
        self.assertIs(alias, cached)
        self.assertTrue(
            authority.validate_support_catalog_alias(alias)
        )
        self.assertEqual(
            alias.cone_start_lid,
            owner.binding.stage_uses[0].cone_start_lid,
        )
        self.assertEqual(alias.branch, authority.BRANCH_DENSE)
        with self.assertRaises(TypeError):
            alias.receipt["branch"] = authority.BRANCH_CONV_DENSE

        copied = copy.copy(alias)
        self.assertFalse(
            authority.validate_support_catalog_alias(copied)
        )
        rehashed = replace(
            alias,
            branch_evidence_sha256=_sha("other-branch"),
            content_sha256=_sha("rehash"),
        )
        self.assertFalse(
            authority.validate_support_catalog_alias(rehashed)
        )

        other_owner = authority._mint_frame_owner(
            _binding(frame="frame-b")
        )
        other_span = _span(
            other_owner, span_index=0, start=0, end=2, total=2
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            _expectation(
                other_owner,
                execution_index=0,
                span=other_span,
                alias=alias,
                partition=_scalar_partition(2),
            )
        self.assertEqual(caught.exception.code, "INVALID_EXPECTATION")

    def test_catalog_cache_hit_checks_absolute_deadline(self):
        binding = _binding()
        owner = authority._mint_frame_owner(binding)
        _alias(owner)
        end = float.fromhex(binding.deadline_monotonic_hex)
        with mock.patch.object(
            authority.time, "monotonic", return_value=end
        ):
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as caught:
                _alias(owner)
            self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

    def test_catalog_validation_cannot_run_past_cached_deadline(self):
        binding = _binding()
        owner = authority._mint_frame_owner(binding)
        cached = _alias(owner)
        deadline = float.fromhex(binding.deadline_monotonic_hex)
        clock = [deadline - 1.0]
        original_validate = authority.validate_support_catalog_alias

        def validate_then_expire(value):
            result = original_validate(value)
            clock[0] = deadline
            return result

        with mock.patch.object(
            authority.time,
            "monotonic",
            side_effect=lambda: clock[0],
        ), mock.patch.object(
            authority,
            "validate_support_catalog_alias",
            side_effect=validate_then_expire,
        ):
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as caught:
                self.assertIs(_alias(owner), cached)
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

    def test_catalog_string_boundaries_return_stable_errors(self):
        owner = authority._mint_frame_owner(_binding())
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as branch:
            _alias(owner, operator_kind="CONV2D", branch=[])
        self.assertEqual(branch.exception.code, "INVALID_BRANCH")

        stage_use = owner.binding.stage_uses[0]
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as semantics:
            authority._mint_support_catalog_alias(
                owner,
                stage_use_sha256=stage_use.stage_use_sha256,
                layer_id=11,
                predecessor_id=90,
                operator_kind="DENSE",
                branch=authority.BRANCH_DENSE,
                box_semantics=[],
                catalog_content_sha256=_sha("catalog"),
                support_content_sha256=_sha("support"),
                weight_sha256=_sha("weight"),
                geometry_sha256=_sha("geometry"),
                source_lb_sha256=_sha("lb"),
                source_ub_sha256=_sha("ub"),
                numeric_platform_sha256=_sha("platform"),
                implementation_sha256=_sha("implementation"),
                branch_evidence_sha256=_sha("branch"),
            )
        self.assertEqual(semantics.exception.code, "INVALID_BINDING")

    def test_query_span_is_process_local_and_binds_all_block_hashes(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=3, total=3
        )
        self.assertTrue(authority.validate_query_span(span))
        self.assertEqual(span.query_start, 0)
        self.assertEqual(span.query_end, 3)
        self.assertEqual(span.row_count, 3)
        self.assertEqual(span.query_block_sha256, _sha("block-a"))
        copied = copy.copy(span)
        self.assertFalse(authority.validate_query_span(copied))
        forged = replace(
            span,
            query_rows_sha256=_sha("replacement"),
            content_sha256=_sha("fully-rehashed-text"),
        )
        self.assertFalse(authority.validate_query_span(forged))

    def test_row_policy_masks_are_exact_disjoint_and_tail_clean(self):
        partition = _scalar_partition(
            10, active=(0, 3, 9), fallback=(3,)
        )
        self.assertTrue(
            authority.validate_row_policy_partition(partition)
        )
        self.assertEqual(partition.scalar_row_count, 10)
        self.assertEqual(partition.componentwise_row_count, 0)
        self.assertEqual(partition.active_row_count, 3)
        self.assertEqual(partition.fallback_row_count, 1)

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as inactive_fallback:
            _scalar_partition(4, active=(0,), fallback=(1,))
        self.assertEqual(
            inactive_fallback.exception.code, "INVALID_MASK"
        )

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as overlap:
            authority.RowPolicyPartition(
                row_count=4,
                scalar_mask=b"\x0f",
                componentwise_mask=b"\x01",
                active_mask=b"\x00",
                fallback_mask=b"\x00",
            )
        self.assertEqual(overlap.exception.code, "POLICY_OVERLAP")

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as gap:
            authority.RowPolicyPartition(
                row_count=4,
                scalar_mask=b"\x03",
                componentwise_mask=b"\x04",
                active_mask=b"\x00",
                fallback_mask=b"\x00",
            )
        self.assertEqual(gap.exception.code, "POLICY_GAP")

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as tail:
            authority.RowPolicyPartition(
                row_count=9,
                scalar_mask=b"\xff\x81",
                componentwise_mask=b"\x00\x00",
                active_mask=b"\x00\x00",
                fallback_mask=b"\x00\x00",
            )
        self.assertEqual(tail.exception.code, "INVALID_MASK")

    def test_expectation_binds_span_masks_branch_and_catalog(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=4, total=4
        )
        alias = _alias(owner)
        partition = _scalar_partition(
            4, active=(0, 2), fallback=(2,)
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=alias,
            partition=partition,
        )
        self.assertTrue(
            authority.validate_affine_execution_expectation(
                expectation
            )
        )
        copied = copy.copy(expectation)
        self.assertFalse(
            authority.validate_affine_execution_expectation(copied)
        )

        sparse_alias = _alias(
            owner,
            layer_id=12,
            predecessor_id=90,
            operator_kind="CONV2D",
            branch=authority.BRANCH_CONV_SPARSE,
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            _expectation(
                owner,
                execution_index=1,
                span=span,
                alias=sparse_alias,
                partition=partition,
            )
        self.assertEqual(caught.exception.code, "POLICY_MISMATCH")

    def test_compact_scalar_trace_keeps_zero_guard_rows_bit_identical(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=4, total=4
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=_alias(owner),
            partition=_scalar_partition(
                4, active=(1, 3), fallback=(3,)
            ),
        )
        trace = _scalar_trace(owner, expectation)
        self.assertTrue(
            authority.validate_compact_absorption_trace(trace)
        )
        self.assertIsNotNone(trace.scalar_guard_sha256)
        self.assertIsNone(trace.componentwise_radius_sha256)
        self.assertFalse(_contains_ndarray(trace))
        copied = copy.copy(trace)
        self.assertFalse(
            authority.validate_compact_absorption_trace(copied)
        )

        nominal = np.zeros((4, 3), dtype=np.float64)
        before = np.asarray([0.0, 2.0, -0.0, 4.0], dtype=np.float64)
        after = before.copy()
        after[0] = np.nextafter(after[0], -np.inf)
        guard = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        after[1] = np.nextafter(before[1] - guard[1], -np.inf)
        after[3] = np.nextafter(before[3] - guard[3], -np.inf)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as changed:
            authority._mint_compact_absorption_trace(
                owner,
                expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(after),
                scalar_guard=np.ascontiguousarray(guard),
            )
        self.assertEqual(changed.exception.code, "ZERO_GUARD_CHANGED")

    def test_active_mask_and_double_policy_substitution_fail(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=2, total=2
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=_alias(owner),
            partition=_scalar_partition(2, active=(0,)),
        )
        nominal = np.zeros((2, 2), dtype=np.float64)
        before = np.ones(2, dtype=np.float64)
        guard = np.asarray([0.0, 0.0], dtype=np.float64)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as active:
            authority._mint_compact_absorption_trace(
                owner,
                expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(before.copy()),
                scalar_guard=np.ascontiguousarray(guard),
            )
        self.assertEqual(
            active.exception.code, "ACTIVE_MASK_MISMATCH"
        )

        guard = np.asarray([0.5, 0.0], dtype=np.float64)
        after = before.copy()
        after[0] = np.nextafter(before[0] - guard[0], -np.inf)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as double:
            authority._mint_compact_absorption_trace(
                owner,
                expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(after),
                scalar_guard=np.ascontiguousarray(guard),
                componentwise_radius=np.ones(
                    (2, 2), dtype=np.float64
                ),
            )
        self.assertEqual(double.exception.code, "DOUBLE_CHARGE")

    def test_absorption_must_match_the_bound_exactly_once(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=2, total=2
        )
        scalar_expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=_alias(owner),
            partition=_scalar_partition(2, active=(0,)),
        )
        nominal = np.zeros((2, 2), dtype=np.float64)
        before = np.ones(2, dtype=np.float64)
        guard = np.asarray([0.5, 0.0], dtype=np.float64)
        undercharged = before.copy()
        undercharged[0] = np.nextafter(0.75, -np.inf)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as scalar:
            authority._mint_compact_absorption_trace(
                owner,
                scalar_expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(undercharged),
                scalar_guard=np.ascontiguousarray(guard),
            )
        self.assertEqual(
            scalar.exception.code, "ABSORPTION_MISMATCH"
        )

        sparse_alias = _alias(
            owner,
            layer_id=13,
            predecessor_id=90,
            operator_kind="CONV2D",
            branch=authority.BRANCH_CONV_SPARSE,
        )
        componentwise_expectation = _expectation(
            owner,
            execution_index=1,
            span=span,
            alias=sparse_alias,
            partition=_componentwise_partition(2),
        )
        radius = np.full((2, 2), 0.125, dtype=np.float64)
        penalty = np.full(2, 0.5, dtype=np.float64)
        exact_after = np.nextafter(before - penalty, -np.inf)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as missing:
            authority._mint_compact_absorption_trace(
                owner,
                componentwise_expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(exact_after),
                componentwise_radius=np.ascontiguousarray(radius),
            )
        self.assertEqual(missing.exception.code, "MISSING_GUARD")

        wrong_after = np.nextafter(before - 0.25, -np.inf)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as componentwise:
            authority._mint_compact_absorption_trace(
                owner,
                componentwise_expectation,
                nominal=np.ascontiguousarray(nominal),
                scalar_before=np.ascontiguousarray(before),
                scalar_after=np.ascontiguousarray(wrong_after),
                componentwise_radius=np.ascontiguousarray(radius),
                componentwise_penalty=np.ascontiguousarray(penalty),
            )
        self.assertEqual(
            componentwise.exception.code, "ABSORPTION_MISMATCH"
        )

    def test_trace_hashes_owned_snapshots_not_later_caller_mutation(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=1, total=1
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=_alias(owner),
            partition=_scalar_partition(1, active=(0,)),
        )
        nominal = np.zeros((1, 1), dtype=np.float64)
        before = np.ones(1, dtype=np.float64)
        guard = np.asarray([0.5], dtype=np.float64)
        correct_after = np.nextafter(before - guard, -np.inf)
        after = correct_after.copy()
        original_metadata = authority._array_metadata

        def mutate_after_snapshot(value, *, nonnegative):
            if value.shape == guard.shape and nonnegative:
                after[0] = before[0]
            return original_metadata(value, nonnegative=nonnegative)

        with mock.patch.object(
            authority,
            "_array_metadata",
            side_effect=mutate_after_snapshot,
        ):
            trace = authority._mint_compact_absorption_trace(
                owner,
                expectation,
                nominal=nominal,
                scalar_before=before,
                scalar_after=after,
                scalar_guard=guard,
            )
        self.assertTrue(
            authority.validate_compact_absorption_trace(trace)
        )
        self.assertEqual(
            trace.scalar_after_sha256,
            authority._array_sha256(correct_after),
        )
        self.assertNotEqual(
            trace.scalar_after_sha256,
            authority._array_sha256(after),
        )

    def test_componentwise_trace_is_compact_and_branch_bound(self):
        owner = authority._mint_frame_owner(_binding())
        span = _span(
            owner, span_index=0, start=0, end=3, total=3
        )
        sparse_alias = _alias(
            owner,
            layer_id=13,
            predecessor_id=90,
            operator_kind="CONV2D",
            branch=authority.BRANCH_CONV_SPARSE,
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=sparse_alias,
            partition=_componentwise_partition(3),
        )
        trace = _componentwise_trace(owner, expectation)
        self.assertTrue(
            authority.validate_compact_absorption_trace(trace)
        )
        self.assertIsNone(trace.scalar_guard_sha256)
        self.assertIsNotNone(trace.componentwise_radius_sha256)
        self.assertFalse(_contains_ndarray(trace))

    def test_query_span_gap_overlap_and_tail_are_rejected(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)

        def attempt(ranges):
            spans = tuple(
                _span(
                    owner,
                    span_index=index,
                    start=start,
                    end=end,
                    total=5,
                )
                for index, (start, end) in enumerate(ranges)
            )
            expectations = tuple(
                _expectation(
                    owner,
                    execution_index=index,
                    span=span,
                    alias=alias,
                    partition=_scalar_partition(span.row_count),
                )
                for index, span in enumerate(spans)
            )
            return authority._mint_compact_guard_ledger(
                owner, spans, expectations
            )

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as gap:
            attempt(((0, 2), (3, 5)))
        self.assertEqual(gap.exception.code, "QUERY_SPAN_GAP")

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as overlap:
            attempt(((0, 3), (2, 5)))
        self.assertEqual(
            overlap.exception.code, "QUERY_SPAN_OVERLAP"
        )

        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as tail:
            attempt(((0, 2), (2, 4)))
        self.assertEqual(tail.exception.code, "QUERY_SPAN_GAP")

    def test_query_spans_must_share_stage_and_query_block(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        first = _span(
            owner, span_index=0, start=0, end=2, total=4
        )
        second = _span(
            owner,
            span_index=1,
            start=2,
            end=4,
            total=4,
            block="block-b",
        )
        expectations = (
            _expectation(
                owner,
                execution_index=0,
                span=first,
                alias=alias,
                partition=_scalar_partition(2),
            ),
            _expectation(
                owner,
                execution_index=1,
                span=second,
                alias=alias,
                partition=_scalar_partition(2),
            ),
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            authority._mint_compact_guard_ledger(
                owner, (first, second), expectations
            )
        self.assertEqual(caught.exception.code, "INVALID_QUERY_SPAN")

    def test_compact_ledger_commits_complete_schedule_without_arrays(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        spans = (
            _span(
                owner, span_index=0, start=0, end=2, total=5
            ),
            _span(
                owner, span_index=1, start=2, end=5, total=5
            ),
        )
        expectations = (
            _expectation(
                owner,
                execution_index=0,
                span=spans[0],
                alias=alias,
                partition=_scalar_partition(
                    2, active=(0,), fallback=(0,)
                ),
            ),
            _expectation(
                owner,
                execution_index=1,
                span=spans[1],
                alias=alias,
                partition=_scalar_partition(
                    3, active=(1, 2), fallback=(2,)
                ),
            ),
        )
        traces = tuple(
            _scalar_trace(owner, expectation)
            for expectation in expectations
        )
        ledger = authority._mint_compact_guard_ledger(
            owner, spans, expectations
        )
        for trace in reversed(traces):
            ledger.record(trace)
        certificate = ledger.commit()
        self.assertFalse(certificate.proof_authority)
        self.assertTrue(
            authority.validate_compact_guard_ledger_certificate(
                certificate
            )
        )
        self.assertFalse(_contains_ndarray(certificate))
        self.assertFalse(certificate.receipt["arrays_retained"])
        self.assertEqual(certificate.receipt["query_total"], 5)
        self.assertEqual(certificate.receipt["span_count"], 2)
        self.assertEqual(certificate.receipt["execution_count"], 2)
        self.assertEqual(
            certificate.receipt["scalar_policy_row_count"], 5
        )
        self.assertEqual(certificate.receipt["active_row_count"], 3)
        self.assertEqual(certificate.receipt["fallback_row_count"], 2)
        with self.assertRaises(TypeError):
            certificate.receipt["proof_authority"] = True

        copied = copy.copy(certificate)
        self.assertFalse(
            authority.validate_compact_guard_ledger_certificate(copied)
        )
        rehashed = replace(
            certificate,
            content_sha256=_sha("certificate-rehash"),
        )
        self.assertFalse(
            authority.validate_compact_guard_ledger_certificate(
                rehashed
            )
        )

        # A self-consistent textual mutation is still not process-local
        # authority: copy the live object, alter a semantic field, then
        # recompute both nested hashes exactly as an attacker could.
        fully_rehashed = copy.copy(certificate)
        forged_receipt = dict(certificate.receipt)
        forged_receipt.pop("receipt_sha256")
        forged_receipt.pop("content_sha256")
        forged_receipt["coverage_complete"] = False
        forged_content_sha = authority._json_sha256(forged_receipt)
        forged_receipt["content_sha256"] = forged_content_sha
        forged_receipt["receipt_sha256"] = authority._json_sha256(
            forged_receipt
        )
        object.__setattr__(
            fully_rehashed,
            "receipt",
            authority._deep_freeze(forged_receipt),
        )
        object.__setattr__(
            fully_rehashed, "content_sha256", forged_content_sha
        )
        self.assertFalse(
            authority.validate_compact_guard_ledger_certificate(
                fully_rehashed
            )
        )

    def test_registry_seals_reject_canonical_mutable_substitutions(self):
        owner, alias, _, _, traces, ledger = _ready_scalar_ledger()
        original_alias_receipt = alias.receipt
        object.__setattr__(alias, "receipt", dict(alias.receipt))
        self.assertFalse(authority.validate_support_catalog_alias(alias))
        object.__setattr__(alias, "receipt", original_alias_receipt)
        self.assertTrue(authority.validate_support_catalog_alias(alias))

        trace = traces[0]
        original_shape = trace.nominal_shape
        object.__setattr__(trace, "nominal_shape", list(original_shape))
        self.assertFalse(
            authority.validate_compact_absorption_trace(trace)
        )
        object.__setattr__(trace, "nominal_shape", original_shape)
        self.assertTrue(authority.validate_compact_absorption_trace(trace))

        ledger.record(trace)
        certificate = ledger.commit()
        original_spans = certificate.spans
        object.__setattr__(certificate, "spans", list(original_spans))
        self.assertFalse(
            authority.validate_compact_guard_ledger_certificate(
                certificate
            )
        )
        object.__setattr__(certificate, "spans", original_spans)
        original_receipt = certificate.receipt
        object.__setattr__(
            certificate, "receipt", dict(original_receipt)
        )
        self.assertFalse(
            authority.validate_compact_guard_ledger_certificate(
                certificate
            )
        )
        object.__setattr__(certificate, "receipt", original_receipt)
        self.assertTrue(
            authority.validate_compact_guard_ledger_certificate(
                certificate
            )
        )

    def test_ledger_identity_schedule_deadline_and_lock_are_sealed(self):
        _, _, _, _, traces, ledger = _ready_scalar_ledger()
        ledger.record(traces[0])
        copied = copy.copy(ledger)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as copied_error:
            copied.commit()
        self.assertEqual(copied_error.exception.code, "INVALID_LEDGER")
        certificate = ledger.commit()
        self.assertTrue(
            authority.validate_compact_guard_ledger_certificate(
                certificate
            )
        )

        _, _, _, expectations, traces, ledger = _ready_scalar_ledger(
            spans_count=2
        )
        object.__setattr__(
            ledger, "_expectations", (expectations[0],)
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as schedule:
            ledger.record(traces[0])
        self.assertEqual(schedule.exception.code, "INVALID_LEDGER")

        _, _, _, _, traces, ledger = _ready_scalar_ledger()
        object.__setattr__(ledger, "_deadline", ledger._deadline + 1000.0)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as deadline:
            ledger.record(traces[0])
        self.assertEqual(deadline.exception.code, "INVALID_LEDGER")

        _, _, _, _, traces, ledger = _ready_scalar_ledger()
        object.__setattr__(ledger, "_lock", None)
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as lock:
            ledger.record(traces[0])
        self.assertEqual(lock.exception.code, "INVALID_LEDGER")

    def test_ledger_deadline_and_recorded_trace_mutation_fail_closed(self):
        owner, _, spans, expectations, traces, ledger = (
            _ready_scalar_ledger()
        )
        deadline = float.fromhex(
            owner.binding.deadline_monotonic_hex
        )
        with mock.patch.object(
            authority.time, "monotonic", return_value=deadline
        ):
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as expired:
                authority._mint_compact_guard_ledger(
                    owner, spans, expectations
                )
        self.assertEqual(expired.exception.code, "DEADLINE_EXPIRED")

        trace = traces[0]
        ledger.record(trace)
        object.__setattr__(trace, "trace_sha256", _sha("changed-trace"))
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as changed:
            ledger.commit()
        self.assertEqual(changed.exception.code, "INVALID_TRACE")

    def test_duplicate_semantic_affine_execution_is_rejected(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        span = _span(
            owner, span_index=0, start=0, end=1, total=1
        )
        partition = _scalar_partition(1, active=(0,))
        coefficient_sha = _sha("same-coefficient")
        expectations = tuple(
            authority._mint_affine_execution_expectation(
                owner,
                execution_index=index,
                span=span,
                support_alias=alias,
                partition=partition,
                input_coefficient_sha256=coefficient_sha,
            )
            for index in range(2)
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as duplicate:
            authority._mint_compact_guard_ledger(
                owner, (span,), expectations
            )
        self.assertEqual(
            duplicate.exception.code, "DUPLICATE_EXECUTION"
        )

    def test_double_and_missing_absorption_fail_closed(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        spans = (
            _span(
                owner, span_index=0, start=0, end=1, total=2
            ),
            _span(
                owner, span_index=1, start=1, end=2, total=2
            ),
        )
        expectations = tuple(
            _expectation(
                owner,
                execution_index=index,
                span=span,
                alias=alias,
                partition=_scalar_partition(1, active=(0,)),
            )
            for index, span in enumerate(spans)
        )
        traces = tuple(
            _scalar_trace(owner, value) for value in expectations
        )

        double_ledger = authority._mint_compact_guard_ledger(
            owner, spans, expectations
        )
        double_ledger.record(traces[0])
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as double:
            double_ledger.record(traces[0])
        self.assertEqual(double.exception.code, "DOUBLE_CHARGE")

        missing_ledger = authority._mint_compact_guard_ledger(
            owner, spans, expectations
        )
        missing_ledger.record(traces[0])
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as missing:
            missing_ledger.commit()
        self.assertEqual(missing.exception.code, "MISSING_CHARGE")

    def test_every_span_requires_an_affine_execution(self):
        owner = authority._mint_frame_owner(_binding())
        alias = _alias(owner)
        spans = (
            _span(
                owner, span_index=0, start=0, end=1, total=2
            ),
            _span(
                owner, span_index=1, start=1, end=2, total=2
            ),
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=spans[0],
            alias=alias,
            partition=_scalar_partition(1),
        )
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            authority._mint_compact_guard_ledger(
                owner, spans, (expectation,)
            )
        self.assertEqual(caught.exception.code, "MISSING_EXECUTION")

    def test_deadline_after_last_record_prevents_certificate(self):
        binding = _binding()
        owner = authority._mint_frame_owner(binding)
        alias = _alias(owner)
        span = _span(
            owner, span_index=0, start=0, end=1, total=1
        )
        expectation = _expectation(
            owner,
            execution_index=0,
            span=span,
            alias=alias,
            partition=_scalar_partition(1, active=(0,)),
        )
        trace = _scalar_trace(owner, expectation)
        ledger = authority._mint_compact_guard_ledger(
            owner, (span,), (expectation,)
        )
        ledger.record(trace)
        end = float.fromhex(binding.deadline_monotonic_hex)
        with mock.patch.object(
            authority.time, "monotonic", return_value=end
        ):
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as caught:
                ledger.commit()
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

    def test_commit_concurrency_poison_and_publication_deadline(self):
        def ready_ledger(binding):
            owner = authority._mint_frame_owner(binding)
            alias = _alias(owner)
            span = _span(
                owner, span_index=0, start=0, end=1, total=1
            )
            expectation = _expectation(
                owner,
                execution_index=0,
                span=span,
                alias=alias,
                partition=_scalar_partition(1, active=(0,)),
            )
            trace = _scalar_trace(owner, expectation)
            ledger = authority._mint_compact_guard_ledger(
                owner, (span,), (expectation,)
            )
            ledger.record(trace)
            return ledger

        ledger = ready_ledger(_binding())
        entered = threading.Event()
        release = threading.Event()
        original_body = authority._ledger_body
        outcome = {}

        def blocked_body(**values):
            entered.set()
            self.assertTrue(release.wait(timeout=5.0))
            return original_body(**values)

        def first_commit():
            try:
                outcome["value"] = ledger.commit()
            except Exception as exc:  # expected poisoned failure
                outcome["error"] = exc

        with mock.patch.object(
            authority, "_ledger_body", side_effect=blocked_body
        ):
            worker = threading.Thread(target=first_commit)
            worker.start()
            self.assertTrue(entered.wait(timeout=5.0))
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as concurrent:
                ledger.commit()
            self.assertEqual(
                concurrent.exception.code, "CONCURRENT_LEDGER"
            )
            release.set()
            worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertNotIn("value", outcome)
        self.assertIsInstance(
            outcome.get("error"),
            authority.QueryDualV51AuthorityError,
        )

        binding = _binding()
        deadline = float.fromhex(binding.deadline_monotonic_hex)
        ledger = ready_ledger(binding)
        clock = [deadline - 1.0]
        original_freeze = authority._deep_freeze

        def expire_during_freeze(value):
            result = original_freeze(value)
            clock[0] = deadline
            return result

        with mock.patch.object(
            authority.time,
            "monotonic",
            side_effect=lambda: clock[0],
        ), mock.patch.object(
            authority,
            "_deep_freeze",
            side_effect=expire_during_freeze,
        ):
            with self.assertRaises(
                authority.QueryDualV51AuthorityError
            ) as expired:
                ledger.commit()
        self.assertEqual(expired.exception.code, "DEADLINE_EXPIRED")

    def test_noncanonical_overflow_hex_is_fail_closed(self):
        with self.assertRaises(
            authority.QueryDualV51AuthorityError
        ) as caught:
            authority._finite_hex(
                "0x1p+999999999999999999",
                name="overflow",
                nonnegative=True,
            )
        self.assertEqual(caught.exception.code, "INVALID_BINDING")


if __name__ == "__main__":
    unittest.main()
