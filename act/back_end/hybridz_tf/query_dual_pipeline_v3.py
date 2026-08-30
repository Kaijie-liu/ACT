#!/usr/bin/env python3
# ===- query_dual_pipeline_v3.py - sealed sparse authority ----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===----------------------------------------------------------------===#
"""Property-covered sealed sparse query-dual transactions.

V3 is intentionally separate from the frozen V2 implementation.  A V3
transaction has one outward root, one property-selected per-layer schedule,
one root-owned sealed replay session, sequential immutable bounds frames, and
one final live-network commit.  Candidate scores and CUDA optimization remain
non-authoritative.  Only committed independent CPU binary64 replay values may
change a target box or a final property upper bound.

The public validator is called through
``query_dual_pipeline.validate_verified_query_dual_feedback``.  Keeping the
dataclasses and process-local authority registry shared lets Operator-HZ
consume V2 and V3 without weakening either protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
import secrets
import time
from typing import Any, Callable, Dict, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import query_dual_pipeline as _v2
from act.back_end.hybridz_tf.property_residual_targets import (
    PropertySparseQueryPlan,
    _binary64_sha256 as _selector_property_sha256,
    select_property_sparse_query_rows,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    QueryDualBoxError,
    QueryDualBoxTimeout,
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
)
from act.back_end.hybridz_tf.query_dual_candidates import (
    QueryDescriptor,
    QueryDualCandidates,
    generate_query_dual_candidates,
    query_dual_stored_alpha_sha256,
    validate_query_dual_candidates,
    verify_query_dual_candidates_receipt,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    QueryDualReplayError,
    QueryDualReplayPendingResult,
    QueryDualReplayResult,
    QueryDualReplayTimeout,
    _build_query_dual_replay_validation_context,
    _query_dual_replay_frame_payload,
    _query_dual_replay_frame_sha256,
    create_query_dual_replay_session,
    validate_query_dual_replay_result,
)
from act.util.device_manager import get_default_device, get_default_dtype


SCHEMA = "act.verified_query_dual_feedback.v3"
STAGE_SCHEMA = "act.verified_query_dual_stage.v3"
PROPERTY_SCHEMA = "act.verified_query_dual_property.v3"
CANDIDATE_SCHEMA = "act.query_dual_candidates.v3"
CANDIDATE_PROTOCOL = "property_sparse_descriptor_only_v3"
REPLAY_SCHEMA = "act.query_dual_replay.v2"
REPLAY_PROTOCOL = "frozen_union_context_v3"
SELECTOR_SCHEMA = "property_sparse_query_selector_v3"
UNSELECTED_POLICY = "bit_identical_immutable_outward_parent_box"
_AUTHORITY_SOURCE = (
    "independent_outward_root_plus_property_selected_"
    "sealed_sparse_cpu_replay"
)
_TRUSTED_CERTIFIER = certify_query_dual_boxes
_TRUSTED_SESSION_FACTORY = create_query_dual_replay_session
_CANDIDATE_NON_AUTHORITATIVE_AUDIT_FIELDS = [
    "lr_alpha",
    "lr_decay",
    "solver",
    "elapsed_seconds",
    "timings",
]
_PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS = [
    "candidate_generator",
    "candidate_solver_factory",
    "selector",
    "selector_receipt",
    "dual_solver_default_device",
    "dual_solver_default_dtype",
    "candidate_cuda_device_name",
]


def _fail(code: str, message: str) -> NoReturn:
    raise _v2.QueryDualPipelineError(code, message)


def _selected_partition(
    lower: np.ndarray,
    upper: np.ndarray,
    selected_rows: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    eligible = tuple(
        int(value)
        for value in np.flatnonzero((lower < 0.0) & (upper > 0.0))
    )
    selected = tuple(int(value) for value in selected_rows)
    if (
        len(set(selected)) != len(selected)
        or any(value not in eligible for value in selected)
    ):
        _fail(
            "INVALID_SELECTION",
            "selected rows must be a unique ordered subset of unstable rows",
        )
    selected_set = set(selected)
    omitted = tuple(value for value in eligible if value not in selected_set)
    return eligible, omitted


def _ordered_rows_sha256(rows: Sequence[int]) -> str:
    return _v2._json_sha256([int(value) for value in rows])


def _binary64_bits_equal(left: Any, right: Any) -> bool:
    """Compare binary64 arrays without collapsing the signs of zero."""

    left_array = np.ascontiguousarray(np.asarray(left, dtype=np.float64))
    right_array = np.ascontiguousarray(np.asarray(right, dtype=np.float64))
    return bool(
        left_array.shape == right_array.shape
        and np.array_equal(
            left_array.view(np.uint64),
            right_array.view(np.uint64),
        )
    )


def _normalise_selector_kind(value: Any) -> str:
    if not isinstance(value, str):
        _fail("INVALID_CONFIG", "selector_kind must be a nonempty string")
    normalized = value.strip().upper()
    if not normalized:
        _fail("INVALID_CONFIG", "selector_kind must be a nonempty string")
    return normalized


def _exact_int(value: Any, expected: int) -> bool:
    """Match an integer receipt scalar without accepting bool aliases."""

    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (Integral, np.integer))
        and int(value) == int(expected)
    )


def _exact_optional_int(
    value: Any,
    expected: Optional[int],
) -> bool:
    if expected is None:
        return value is None
    return _exact_int(value, expected)


def _receipt_row_ids(value: Any, *, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        _fail("INVALID_SELECTOR", f"{name} must be an integer list")
    rows = []
    for raw in value:
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, (Integral, np.integer))
            or int(raw) < 0
        ):
            _fail("INVALID_SELECTOR", f"{name} contains an invalid row id")
        rows.append(int(raw))
    if len(set(rows)) != len(rows):
        _fail("INVALID_SELECTOR", f"{name} contains duplicate row ids")
    return tuple(rows)


def _selector_receipt_semantics(
    receipt: Mapping[str, Any],
    *,
    targets: Tuple[int, ...],
    quotas: Tuple[int, ...],
    root_bounds: Mapping[int, Bounds],
    property_sha256: str,
    expected_selected_by_layer: Optional[
        Mapping[int, Sequence[int]]
    ] = None,
    expected_selection_sha256: Optional[str] = None,
) -> Dict[int, Tuple[int, ...]]:
    expected_quotas = [
        [int(layer_id), int(quota)]
        for layer_id, quota in zip(targets, quotas)
    ]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != SELECTOR_SCHEMA
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("property_sha256") != property_sha256
        or receipt.get("layer_quotas") != expected_quotas
    ):
        _fail("INVALID_SELECTOR", "selector receipt semantics mismatch")
    records = receipt.get("layers")
    if not isinstance(records, list) or len(records) != len(targets):
        _fail(
            "INVALID_SELECTOR",
            "selector layer records do not exactly cover the targets",
        )
    record_ids = []
    for record in records:
        if not isinstance(record, Mapping):
            _fail("INVALID_SELECTOR", "selector layer record is malformed")
        raw_lid = record.get("layer_id")
        if (
            isinstance(raw_lid, (bool, np.bool_))
            or not isinstance(raw_lid, (Integral, np.integer))
        ):
            _fail("INVALID_SELECTOR", "selector layer id is malformed")
        record_ids.append(int(raw_lid))
    if tuple(record_ids) != targets:
        _fail(
            "INVALID_SELECTOR",
            "selector layer records are not in exact target order",
        )
    if expected_selected_by_layer is not None:
        plan_layer_ids = set()
        for raw_lid in expected_selected_by_layer:
            if (
                isinstance(raw_lid, (bool, np.bool_))
                or not isinstance(raw_lid, (Integral, np.integer))
            ):
                _fail(
                    "INVALID_SELECTOR",
                    "selector plan contains a malformed layer id",
                )
            plan_layer_ids.add(int(raw_lid))
        if not plan_layer_ids.issubset(set(targets)):
            _fail(
                "INVALID_SELECTOR",
                "selector plan contains a non-target layer",
            )

    by_layer: Dict[int, Tuple[int, ...]] = {}
    selected_schedule: list[list[int]] = []
    for target_lid, quota, record in zip(targets, quotas, records):
        record_eligible = _receipt_row_ids(
            record.get("eligible_rows"),
            name=f"selector layer {target_lid} eligible_rows",
        )
        selected = _receipt_row_ids(
            record.get("selected_rows"),
            name=f"selector layer {target_lid} selected_rows",
        )
        record_omitted = _receipt_row_ids(
            record.get("omitted_rows"),
            name=f"selector layer {target_lid} omitted_rows",
        )
        if expected_selected_by_layer is not None:
            expected_selected = tuple(
                int(value)
                for value in expected_selected_by_layer.get(target_lid, ())
            )
            if selected != expected_selected:
                _fail(
                    "INVALID_SELECTOR",
                    f"selector plan/receipt rows differ at ReLU {target_lid}",
                )
        lower, upper = _v2._flat_box(
            root_bounds[target_lid], lid=target_lid
        )
        eligible, omitted = _selected_partition(lower, upper, selected)
        by_layer[target_lid] = selected
        selected_schedule.extend(
            [[int(target_lid), int(row)] for row in selected]
        )
        candidate_union = record.get("candidate_union")
        if (
            not _exact_int(record.get("quota"), quota)
            or record_eligible != eligible
            or record_omitted != omitted
            or record.get("eligible_rows_sha256")
            != _ordered_rows_sha256(eligible)
            or record.get("selected_rows_sha256")
            != _ordered_rows_sha256(selected)
            or record.get("omitted_rows_sha256")
            != _ordered_rows_sha256(omitted)
            or not _exact_int(
                record.get("eligible_count"), len(eligible)
            )
            or not _exact_int(
                record.get("selected_count"), len(selected)
            )
            or not _exact_int(
                record.get("omitted_count"), len(omitted)
            )
            or isinstance(candidate_union, (bool, np.bool_))
            or not isinstance(candidate_union, (Integral, np.integer))
            or int(candidate_union) < len(selected)
            or int(candidate_union) > len(eligible)
            or record.get("partition_complete") is not True
            or record.get("partition_disjoint") is not True
            or record.get("quota_filled") is not True
            or len(selected) != min(int(quota), len(eligible))
        ):
            _fail(
                "INVALID_SELECTOR",
                f"selector partition mismatch for ReLU {target_lid}",
            )

    schedule = receipt.get("schedule")
    if not isinstance(schedule, list):
        _fail("INVALID_SELECTOR", "selector schedule must be a list")
    receipt_schedule = []
    for item in schedule:
        if not isinstance(item, Mapping):
            _fail("INVALID_SELECTOR", "selector schedule item is malformed")
        raw_lid = item.get("layer_id")
        raw_row = item.get("row")
        if (
            isinstance(raw_lid, (bool, np.bool_))
            or not isinstance(raw_lid, (Integral, np.integer))
            or isinstance(raw_row, (bool, np.bool_))
            or not isinstance(raw_row, (Integral, np.integer))
            or int(raw_row) < 0
        ):
            _fail("INVALID_SELECTOR", "selector schedule item is malformed")
        receipt_schedule.append([int(raw_lid), int(raw_row)])
    selection_sha256 = _v2._json_sha256(
        {
            "property_sha256": property_sha256,
            "layer_quotas": expected_quotas,
            "selected": selected_schedule,
        }
    )
    expected_status = "selected" if selected_schedule else "no_selected_rows"
    if (
        receipt_schedule != selected_schedule
        or receipt.get("status") != expected_status
        or not _exact_int(
            receipt.get("targets_selected"), len(selected_schedule)
        )
        or receipt.get("selection_sha256") != selection_sha256
        or (
            expected_selection_sha256 is not None
            and expected_selection_sha256 != selection_sha256
        )
    ):
        _fail("INVALID_SELECTOR", "selector schedule semantics mismatch")
    return by_layer


def _selector_plan_semantics(
    plan: PropertySparseQueryPlan,
    *,
    targets: Tuple[int, ...],
    quotas: Tuple[int, ...],
    root_bounds: Mapping[int, Bounds],
    property_sha256: str,
) -> Dict[int, Tuple[int, ...]]:
    if not isinstance(plan, PropertySparseQueryPlan):
        _fail("INVALID_SELECTOR", "selector returned an unexpected object")
    if plan.property_sha256 != property_sha256:
        _fail("INVALID_SELECTOR", "selector plan property hash mismatch")
    try:
        plan_by_layer = plan.rows_by_layer
    except Exception as exc:
        raise _v2.QueryDualPipelineError(
            "INVALID_SELECTOR",
            f"selector plan rows are malformed: {type(exc).__name__}: {exc}",
        ) from exc
    return _selector_receipt_semantics(
        plan.receipt,
        targets=targets,
        quotas=quotas,
        root_bounds=root_bounds,
        property_sha256=property_sha256,
        expected_selected_by_layer=plan_by_layer,
        expected_selection_sha256=plan.selection_sha256,
    )


def _normalise_quotas(
    targets: Tuple[int, ...],
    stage_quotas: Sequence[int],
) -> Tuple[int, ...]:
    if isinstance(stage_quotas, (str, bytes)) or not isinstance(
        stage_quotas, Sequence
    ):
        _fail("INVALID_CONFIG", "stage_quotas must be an explicit sequence")
    quotas = tuple(stage_quotas)
    if len(quotas) != len(targets):
        _fail(
            "INVALID_CONFIG",
            "stage_quotas length must equal target_relu_ids length",
        )
    result = []
    for value in quotas:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Integral, np.integer))
            or int(value) < 0
            or int(value) > 64
        ):
            _fail(
                "INVALID_CONFIG",
                "each sparse stage quota must be an integer in [0,64]",
            )
        result.append(int(value))
    return tuple(result)


def _candidate_indexed_bounds_sha256(
    lower: np.ndarray,
    upper: np.ndarray,
    rows: Sequence[int],
) -> str:
    index = np.asarray(tuple(int(value) for value in rows), dtype=np.int64)
    return _v2._candidate_array_digest(
        np.stack([lower[index], upper[index]])
        if index.size
        else np.zeros((2, 0), dtype=np.float64)
    )


def _candidate_receipt_row_ids(
    value: Any,
    *,
    name: str,
) -> Tuple[int, ...]:
    if not isinstance(value, list):
        _fail("CANDIDATE_BINDING", f"{name} must be an integer list")
    rows = []
    for raw in value:
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, (Integral, np.integer))
            or int(raw) < 0
        ):
            _fail("CANDIDATE_BINDING", f"{name} contains an invalid row id")
        rows.append(int(raw))
    if len(set(rows)) != len(rows):
        _fail("CANDIDATE_BINDING", f"{name} contains duplicate row ids")
    return tuple(rows)


def _candidate_v3_receipt_semantics(
    receipt: Mapping[str, Any],
    bounds: Mapping[int, Bounds],
    *,
    target_relu_lid: Optional[int],
    target_start_lid: Optional[int],
    selected_rows: Tuple[int, ...],
    property_rows: Optional[np.ndarray],
    output_lid: int,
    block_size: int,
    steps: int,
    deadline: float,
    descriptor_count: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], list[Mapping[str, Any]]]:
    if (
        not isinstance(receipt, Mapping)
        or not verify_query_dual_candidates_receipt(receipt)
    ):
        _fail("INVALID_CANDIDATE", "V3 candidate receipt hash is invalid")

    if target_relu_lid is None:
        target_lower = np.zeros(0, dtype=np.float64)
        target_upper = np.zeros(0, dtype=np.float64)
        eligible: Tuple[int, ...] = ()
        omitted: Tuple[int, ...] = ()
    else:
        target_lower, target_upper = _v2._flat_box(
            bounds[target_relu_lid], lid=target_relu_lid
        )
        eligible, omitted = _selected_partition(
            target_lower, target_upper, selected_rows
        )
    target_bounds_sha256 = _v2._candidate_array_digest(
        np.stack([target_lower, target_upper])
        if target_lower.size
        else np.zeros((2, 0), dtype=np.float64)
    )
    selected_parent_sha256 = _candidate_indexed_bounds_sha256(
        target_lower, target_upper, selected_rows
    )
    unselected_parent_sha256 = _candidate_indexed_bounds_sha256(
        target_lower, target_upper, omitted
    )

    if property_rows is None:
        property_count = 0
        property_width = 0
        property_rows_sha256 = None
        property_output_lid: Optional[int] = None
        property_baseline_sha256 = _v2._candidate_array_digest(
            np.zeros((2, 0), dtype=np.float64)
        )
        property_lower_source = "not_requested"
        property_upper_source = "not_requested"
    else:
        property_count = int(property_rows.shape[0])
        property_width = int(property_rows.shape[1])
        output_lower, output_upper = _v2._flat_box(
            bounds[output_lid], lid=output_lid
        )
        if property_width != output_lower.size:
            _fail(
                "CANDIDATE_BINDING",
                "property candidate width differs from the output box",
            )
        positive = np.maximum(property_rows, 0.0)
        negative = np.minimum(property_rows, 0.0)
        property_lower = (
            positive @ output_lower + negative @ output_upper
        )
        property_upper = (
            positive @ output_upper + negative @ output_lower
        )
        property_rows_sha256 = _v2._candidate_array_digest(property_rows)
        property_output_lid = int(output_lid)
        property_baseline_sha256 = _v2._candidate_array_digest(
            np.stack([property_lower, property_upper])
        )
        property_lower_source = (
            "frozen_interval_baseline_not_dual_replayed"
        )
        property_upper_source = (
            "baseline_placeholder_no_candidate_bound"
        )

    expected_property_ids = tuple(range(property_count))
    receipt_eligible = _candidate_receipt_row_ids(
        receipt.get("eligible_target_row_ids"),
        name="eligible_target_row_ids",
    )
    receipt_selected = _candidate_receipt_row_ids(
        receipt.get("selected_target_row_ids"),
        name="selected_target_row_ids",
    )
    receipt_omitted = _candidate_receipt_row_ids(
        receipt.get("omitted_target_row_ids"),
        name="omitted_target_row_ids",
    )
    receipt_eligible_property = _candidate_receipt_row_ids(
        receipt.get("eligible_property_row_ids"),
        name="eligible_property_row_ids",
    )
    receipt_selected_property = _candidate_receipt_row_ids(
        receipt.get("selected_property_row_ids"),
        name="selected_property_row_ids",
    )

    layouts: list[
        Tuple[
            str,
            Optional[int],
            Optional[int],
            Tuple[int, ...],
            str,
            int,
        ]
    ] = []
    for offset in range(0, len(selected_rows), int(block_size)):
        block_rows = selected_rows[offset : offset + int(block_size)]
        layouts.append(
            (
                "relu_unstable_plus_minus_one_hot",
                target_start_lid,
                target_relu_lid,
                block_rows,
                "positive_rows_then_negated_rows",
                2 * len(block_rows),
            )
        )
    for offset in range(0, property_count, int(block_size)):
        block_rows = tuple(
            range(offset, min(offset + int(block_size), property_count))
        )
        layouts.append(
            (
                "final_property_negative_c_upper_only",
                None,
                None,
                block_rows,
                "negated_rows_only_for_property_upper_bounds",
                len(block_rows),
            )
        )
    records_raw = receipt.get("descriptor_records")
    alpha_hashes = receipt.get("alpha_hashes")
    if (
        not isinstance(records_raw, list)
        or not isinstance(alpha_hashes, list)
        or len(records_raw) != int(descriptor_count)
        or len(records_raw) != len(layouts)
        or len(alpha_hashes) != len(layouts)
    ):
        _fail(
            "CANDIDATE_BINDING",
            "candidate descriptor records/count mismatch",
        )
    # V3 freezes its descriptor receipt grammar.  The generic candidate
    # validator is intentionally extensible, but authority promotion here
    # accepts exactly the reviewed V3 fields.
    record_keys = {
        "block_id",
        "query_kind",
        "start_lid",
        "target_relu_lid",
        "row_ids",
        "objective_order",
        "objective_sha256",
        "M",
        "alpha_tree_index",
        "alpha_sha256",
        "bound_source",
    }
    selected_descriptor_rows: list[int] = []
    property_descriptor_rows: list[int] = []
    records: list[Mapping[str, Any]] = []
    for index, (record, layout) in enumerate(zip(records_raw, layouts)):
        if not isinstance(record, Mapping) or set(record) != record_keys:
            _fail(
                "CANDIDATE_BINDING",
                "candidate descriptor record schema mismatch",
            )
        (
            query_kind,
            start_lid,
            target_lid,
            block_rows,
            objective_order,
            objective_count,
        ) = layout
        record_rows = _candidate_receipt_row_ids(
            record.get("row_ids"),
            name=f"descriptor_records[{index}].row_ids",
        )
        alpha_sha256 = record.get("alpha_sha256")
        objective_sha256 = record.get("objective_sha256")
        if (
            not _exact_int(record.get("block_id"), index)
            or record.get("query_kind") != query_kind
            or not _exact_optional_int(
                record.get("start_lid"), start_lid
            )
            or not _exact_optional_int(
                record.get("target_relu_lid"), target_lid
            )
            or record_rows != block_rows
            or record.get("objective_order") != objective_order
            or not _exact_int(record.get("M"), objective_count)
            or not _exact_int(
                record.get("alpha_tree_index"), index
            )
            or record.get("bound_source") != "none_descriptor_only"
            or not isinstance(alpha_sha256, str)
            or len(alpha_sha256) != 64
            or alpha_hashes[index] != alpha_sha256
            or not isinstance(objective_sha256, str)
            or len(objective_sha256) != 64
        ):
            _fail(
                "CANDIDATE_BINDING",
                "candidate descriptor record semantics mismatch",
            )
        if query_kind == "relu_unstable_plus_minus_one_hot":
            selected_descriptor_rows.extend(record_rows)
        else:
            property_descriptor_rows.extend(record_rows)
        records.append(record)

    records_sha256 = _v2._json_sha256(records_raw)
    alpha_hashes_sha256 = _v2._json_sha256(alpha_hashes)
    expected_status = (
        "descriptors_generated" if layouts else "no_queries_fallback"
    )
    candidate_generated = bool(layouts)
    if (
        receipt.get("schema") != CANDIDATE_SCHEMA
        or receipt.get("protocol") != CANDIDATE_PROTOCOL
        or receipt.get("non_authoritative_audit_fields")
        != _CANDIDATE_NON_AUTHORITATIVE_AUDIT_FIELDS
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("return_optimized_required") is not True
        or receipt.get("refresh_forward") is not False
        or receipt.get("bounds_source")
        != "caller_frozen_bounds_private_clone"
        or receipt.get("alpha_storage") != "cpu_stored_binary64_tree"
        or receipt.get("candidate_bound_source")
        != "none_descriptor_only"
        or receipt.get("optimizer_best_margins_used_as_bounds") is not False
        or receipt.get("shared_absolute_deadline") is not True
        or receipt.get("deadline_monotonic") != float(deadline)
        or not _exact_int(receipt.get("steps_requested"), steps)
        or not _exact_int(receipt.get("block_size"), block_size)
        or not _exact_optional_int(
            receipt.get("target_relu_lid"), target_relu_lid
        )
        or receipt.get("property_only") is not (target_relu_lid is None)
        or receipt.get("property_upper_only") is not True
        or receipt.get("selector_authoritative") is not False
        or receipt.get("selected_rows_source")
        != "caller_supplied_non_authoritative_selector"
        or receipt.get("sparse_selection_bound") is not True
        or receipt.get("property_coverage_policy")
        != "all_property_rows"
        or receipt.get("unselected_policy")
        != "bit_identical_immutable_parent_target_bounds"
        or receipt.get("input_bounds_sha256")
        != _v2._candidate_bounds_sha256(bounds)
        or receipt_eligible != eligible
        or receipt_selected != selected_rows
        or receipt_omitted != omitted
        or receipt.get("eligible_target_rows_sha256")
        != _ordered_rows_sha256(eligible)
        or receipt.get("selected_target_rows_sha256")
        != _ordered_rows_sha256(selected_rows)
        or receipt.get("omitted_target_rows_sha256")
        != _ordered_rows_sha256(omitted)
        or not _exact_int(
            receipt.get("eligible_target_count"), len(eligible)
        )
        or not _exact_int(
            receipt.get("selected_target_count"), len(selected_rows)
        )
        or not _exact_int(
            receipt.get("omitted_target_count"), len(omitted)
        )
        or receipt.get("target_partition_complete") is not True
        or receipt.get("target_partition_disjoint") is not True
        or receipt.get("selected_parent_target_bounds_sha256")
        != selected_parent_sha256
        or receipt.get("unselected_parent_target_bounds_sha256")
        != unselected_parent_sha256
        or receipt.get("unselected_candidate_target_bounds_sha256")
        != unselected_parent_sha256
        or receipt.get("unselected_bounds_bit_identical_parent") is not True
        or receipt_eligible_property != expected_property_ids
        or receipt_selected_property != expected_property_ids
        or receipt.get("eligible_property_rows_sha256")
        != _ordered_rows_sha256(expected_property_ids)
        or receipt.get("selected_property_rows_sha256")
        != _ordered_rows_sha256(expected_property_ids)
        or not _exact_int(
            receipt.get("selected_property_count"), property_count
        )
        or not _exact_optional_int(
            receipt.get("target_start_lid"), target_start_lid
        )
        or not _exact_int(
            receipt.get("target_width"), int(target_lower.size)
        )
        or receipt.get("target_bounds_sha256") != target_bounds_sha256
        or not _exact_int(receipt.get("property_rows"), property_count)
        or not _exact_int(receipt.get("property_width"), property_width)
        or receipt.get("property_rows_sha256") != property_rows_sha256
        or not _exact_optional_int(
            receipt.get("property_output_lid"), property_output_lid
        )
        or receipt.get("property_baseline_sha256")
        != property_baseline_sha256
        or receipt.get("property_lower_bound_source")
        != property_lower_source
        or receipt.get("property_upper_bound_source")
        != property_upper_source
        or not _exact_int(
            receipt.get("unstable_target_neurons"), len(eligible)
        )
        or not _exact_int(
            receipt.get("planned_query_blocks"), len(layouts)
        )
        or receipt.get("status") != expected_status
        or receipt.get("candidate_generated") is not candidate_generated
        or receipt.get("whole_batch_complete") is not True
        or receipt.get("caller_bounds_unchanged") is not True
        or not _exact_int(receipt.get("query_blocks"), len(layouts))
        or not _exact_int(receipt.get("alpha_trees"), len(layouts))
        or receipt.get("descriptor_records_sha256") != records_sha256
        or receipt.get("descriptor_coverage_sha256") != records_sha256
        or receipt.get("descriptor_coverage_complete") is not True
        or receipt.get("alpha_hashes_sha256") != alpha_hashes_sha256
        or not _exact_int(receipt.get("strict_target_improvements"), 0)
        or not _exact_int(
            receipt.get("strict_property_improvements"), 0
        )
        or receipt.get("improved_target_indices") != []
        or receipt.get("improved_property_indices") != []
        or receipt.get("candidate_target_bounds_sha256")
        != target_bounds_sha256
        or receipt.get("candidate_property_bounds_sha256")
        != property_baseline_sha256
        or receipt.get("optimizer_margins_exported") is not False
        or receipt.get("optimizer_margins_used_for_improvement") is not False
        or receipt.get("gpu_frozen_alpha_replay") is not False
        or receipt.get("cpu_independent_replay_required") is not True
        or receipt.get("all_candidate_updates_replayed_with_stored_alpha")
        is not False
        or receipt.get("all_bounds_replayed_with_stored_alpha") is not False
        or receipt.get("property_lower_dual_replayed") is not False
        or not _exact_int(receipt.get("completed_blocks_discarded"), 0)
        or receipt.get("selected_descriptor_rows_sha256")
        != _ordered_rows_sha256(selected_descriptor_rows)
        or receipt.get("property_descriptor_rows_sha256")
        != _ordered_rows_sha256(property_descriptor_rows)
        or receipt.get("selected_coverage_complete") is not True
        or receipt.get("property_coverage_complete") is not True
    ):
        _fail("CANDIDATE_BINDING", "V3 candidate receipt semantics mismatch")
    return eligible, omitted, records


@dataclass
class _PendingBlock:
    descriptor: QueryDescriptor
    candidate_record: Mapping[str, Any]
    pending: QueryDualReplayPendingResult
    query_bias: np.ndarray
    expected_objectives: np.ndarray
    expected_kind: str
    expected_target: Optional[int]
    expected_start: Optional[int]
    expected_rows: Tuple[int, ...]
    candidate_alpha_sha256: str
    replay_bounds_sha256: str
    replay_query_sha256: str
    bounds_frame_sha256: str


@dataclass
class _TargetDraft:
    stage_index: int
    target_lid: int
    predecessor: int
    predecessor_kind: str
    parent_boxes_sha256: str
    result_boxes_sha256: str
    candidate_bounds_sha256: str
    candidate: QueryDualCandidates
    blocks: Tuple[_PendingBlock, ...]
    lower: np.ndarray
    upper: np.ndarray
    strict: int
    eligible: Tuple[int, ...]
    selected: Tuple[int, ...]
    omitted: Tuple[int, ...]
    bounds_frame_sha256: str
    parent_chain_sha256: str
    result_chain_sha256: str


@dataclass
class _PropertyDraft:
    parent_boxes_sha256: str
    candidate_bounds_sha256: str
    candidate: QueryDualCandidates
    blocks: Tuple[_PendingBlock, ...]
    property_upper: np.ndarray
    property_spec_sha256: str
    bounds_frame_sha256: str
    parent_chain_sha256: str
    result_chain_sha256: str


def _candidate_v3_common(
    candidate: QueryDualCandidates,
    bounds: Mapping[int, Bounds],
    *,
    target_relu_lid: Optional[int],
    target_start_lid: Optional[int],
    selected_rows: Tuple[int, ...],
    property_rows: Optional[np.ndarray],
    output_lid: int,
    block_size: int,
    steps: int,
    deadline: float,
    expected_status: str,
    failure_code: str,
    failure_context: str,
) -> Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    list[Mapping[str, Any]],
]:
    if (
        not validate_query_dual_candidates(candidate)
        or not verify_query_dual_candidates_receipt(candidate.receipt)
    ):
        _fail("INVALID_CANDIDATE", "V3 candidate object validation failed")
    status = str(candidate.status)
    if status != expected_status:
        if status.startswith("deadline_"):
            raise _v2.QueryDualPipelineTimeout(
                f"V3 {failure_context} candidate deadline fallback"
            )
        _fail(
            failure_code,
            f"{failure_context} candidate status {status!r}, "
            f"expected {expected_status!r}",
        )
    return _candidate_v3_receipt_semantics(
        candidate.receipt,
        bounds,
        target_relu_lid=target_relu_lid,
        target_start_lid=target_start_lid,
        selected_rows=selected_rows,
        property_rows=property_rows,
        output_lid=int(output_lid),
        block_size=int(block_size),
        steps=int(steps),
        deadline=float(deadline),
        descriptor_count=len(candidate.query_descriptors),
    )


def _pending_descriptor(
    *,
    session: Any,
    frame: Any,
    net: Any,
    parent_bounds: Mapping[int, Bounds],
    descriptor: QueryDescriptor,
    candidate_record: Mapping[str, Any],
    alpha_tree: Any,
    query_bias: np.ndarray,
    expected_objectives: np.ndarray,
    expected_kind: str,
    expected_target: Optional[int],
    expected_start: Optional[int],
    expected_rows: Tuple[int, ...],
    chunk_size: int,
    max_workspace_bytes: int,
    deadline: float,
    proof_workers: int = 1,
) -> _PendingBlock:
    _v2._check_deadline(deadline, "before sealed replay")
    objectives = _v2._as_numpy_f64(
        descriptor.objectives,
        name=f"descriptor[{descriptor.block_id}].objectives",
    )
    if (
        descriptor.query_kind != expected_kind
        or descriptor.target_relu_lid != expected_target
        or descriptor.start_lid != expected_start
        or tuple(descriptor.row_ids) != expected_rows
        or descriptor.M != int(expected_objectives.shape[0])
        or objectives.shape != expected_objectives.shape
        or not _binary64_bits_equal(objectives, expected_objectives)
        or descriptor.objective_sha256
        != _v2._candidate_array_digest(expected_objectives)
        or candidate_record.get("objective_sha256")
        != descriptor.objective_sha256
        or candidate_record.get("row_ids") != list(expected_rows)
        or candidate_record.get("alpha_sha256")
        != descriptor.alpha_sha256
    ):
        _fail("OBJECTIVE_BINDING", "V3 candidate descriptor mismatch")
    alpha = _v2._flat_alpha_tree(
        alpha_tree, net=net, start_lid=expected_start
    )
    if query_dual_stored_alpha_sha256(alpha) != descriptor.alpha_sha256:
        _fail("ALPHA_BINDING", "V3 candidate alpha clone mismatch")
    output_lid = (
        _v2._assert_output_id(*_v2._layer_maps(net))
        if expected_start is None
        else int(expected_start)
    )
    replay_query_hash = _v2._replay_query_sha256(
        expected_objectives,
        query_bias,
        expected_start,
        output_lid=output_lid,
    )
    replay_bounds_hash = _v2._replay_bounds_sha256(
        net, parent_bounds, expected_start
    )
    pending = session.replay(
        frame,
        start_lid=expected_start,
        query_rows=objectives,
        query_bias=query_bias,
        alpha_by_relu=alpha,
        expected_bounds_sha256=replay_bounds_hash,
        expected_query_sha256=replay_query_hash,
        chunk_size=int(chunk_size),
        max_workspace_bytes=int(max_workspace_bytes),
        proof_workers=int(proof_workers),
    )
    _v2._check_deadline(deadline, "after sealed replay")
    if (
        not isinstance(pending, QueryDualReplayPendingResult)
        or pending.proof_authority is not False
        or query_dual_stored_alpha_sha256(alpha)
        != descriptor.alpha_sha256
    ):
        _fail("INVALID_REPLAY", "sealed replay returned invalid pending state")
    return _PendingBlock(
        descriptor=descriptor,
        candidate_record=copy.deepcopy(dict(candidate_record)),
        pending=pending,
        query_bias=np.ascontiguousarray(query_bias, dtype=np.float64).copy(),
        expected_objectives=np.ascontiguousarray(
            expected_objectives, dtype=np.float64
        ).copy(),
        expected_kind=str(expected_kind),
        expected_target=expected_target,
        expected_start=expected_start,
        expected_rows=expected_rows,
        candidate_alpha_sha256=str(descriptor.alpha_sha256),
        replay_bounds_sha256=replay_bounds_hash,
        replay_query_sha256=replay_query_hash,
        bounds_frame_sha256=str(frame._content_sha256),
    )


def _committed_block(
    pending: _PendingBlock,
    result: QueryDualReplayResult,
) -> _v2.QueryDualAuthorityBlock:
    if not isinstance(result, QueryDualReplayResult):
        _fail("INVALID_REPLAY", "session commit returned an invalid result")
    hashes = result.receipt.get("hashes", {})
    replay_alpha_hash = str(hashes.get("alpha_sha256", ""))
    replay_net_hash = str(hashes.get("net_sha256", ""))
    sealed = result.receipt.get("sealed_context", {})
    if (
        result.receipt.get("schema") != REPLAY_SCHEMA
        or not isinstance(sealed, Mapping)
        or sealed.get("protocol") != REPLAY_PROTOCOL
        or sealed.get("live_net_commit_bound") is not True
        or sealed.get("bounds_frame_sha256")
        != pending.bounds_frame_sha256
        or not validate_query_dual_replay_result(
            result,
            expected_net_sha256=replay_net_hash,
            expected_bounds_sha256=pending.replay_bounds_sha256,
            expected_query_sha256=pending.replay_query_sha256,
            expected_alpha_sha256=replay_alpha_hash,
        )
        or not _binary64_bits_equal(
            result.lower_bounds, pending.pending.lower_bounds
        )
    ):
        _fail("INVALID_REPLAY", "committed V3 replay binding failed")
    bridge_hash = _v2._json_sha256(
        {
            "candidate_alpha_sha256": pending.candidate_alpha_sha256,
            "replay_alpha_sha256": replay_alpha_hash,
            "objective_sha256": pending.descriptor.objective_sha256,
            "replay_query_sha256": pending.replay_query_sha256,
        }
    )
    return _v2.QueryDualAuthorityBlock(
        block_id=int(pending.descriptor.block_id),
        query_kind=str(pending.descriptor.query_kind),
        start_lid=pending.descriptor.start_lid,
        target_relu_lid=pending.descriptor.target_relu_lid,
        row_ids=tuple(int(value) for value in pending.descriptor.row_ids),
        objective_sha256=str(pending.descriptor.objective_sha256),
        candidate_alpha_sha256=pending.candidate_alpha_sha256,
        replay_query_sha256=pending.replay_query_sha256,
        replay_alpha_sha256=replay_alpha_hash,
        replay_bounds_sha256=pending.replay_bounds_sha256,
        replay_net_sha256=replay_net_hash,
        alpha_bridge_sha256=bridge_hash,
        lower_bounds=result.lower_bounds,
        replay_receipt=result.receipt,
    )


def build_verified_query_dual_feedback_v3(
    net: Any,
    property_rows: Any,
    thresholds: Any,
    *,
    target_relu_ids: Sequence[int],
    stage_quotas: Sequence[int],
    steps: int = 4,
    block_size: int = 1024,
    lr_alpha: float = 0.25,
    lr_decay: float = 0.98,
    replay_chunk_size: int = 1024,
    replay_max_workspace_bytes: int = 512 * 1024 * 1024,
    conv_channel_chunk: int = 32,
    candidate_device: str = "cuda",
    selector_time_limit: float = 1.0,
    selector_max_adjoint_cells: int = 30_000_000,
    selector_pool_per_rival: int = 64,
    selector_kind: str = "TOP1_ROBUST",
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
    solver_factory: Optional[Callable[[], Any]] = None,
    candidate_generator: Callable[..., QueryDualCandidates] = (
        generate_query_dual_candidates
    ),
    selector: Callable[..., PropertySparseQueryPlan] = (
        select_property_sparse_query_rows
    ),
) -> _v2.VerifiedQueryDualFeedback:
    """Build one all-or-nothing V3 sparse replay transaction.

    V3 deliberately requires a finite absolute transaction deadline.  The
    selector and candidate generator are injectable because neither can grant
    authority.  The outward certifier, sealed-session factory, CPU numerical
    replay, and live commit are not injectable.
    """

    effective_deadline, started = _v2._effective_deadline(
        deadline, timeout_s
    )
    if effective_deadline is None:
        _fail("INVALID_CONFIG", "V3 requires a finite transaction deadline")
    rows, threshold = _v2._normalise_property(property_rows, thresholds)
    by_id, preds = _v2._layer_maps(net)
    targets = _v2._normalise_targets(target_relu_ids, by_id)
    quotas = _normalise_quotas(targets, stage_quotas)
    if targets != tuple(sorted(targets)):
        _fail(
            "INVALID_TARGETS",
            "V3 target_relu_ids must be in increasing layer-id order",
        )
    if not any(quota > 0 for quota in quotas):
        _fail(
            "INVALID_CONFIG",
            "V3 requires at least one positive sparse stage quota",
        )
    selector_kind_token = _normalise_selector_kind(selector_kind)
    if not isinstance(candidate_device, str) or candidate_device not in {
        "cpu",
        "cuda",
    }:
        _fail("INVALID_CONFIG", "candidate_device must be 'cpu' or 'cuda'")
    if candidate_device == "cuda" and not torch.cuda.is_available():
        _fail(
            "CANDIDATE_DEVICE_UNAVAILABLE",
            "CUDA candidate generation was requested but CUDA is unavailable",
        )
    requested_candidate_device = torch.device(candidate_device)
    default_candidate_device = get_default_device()
    default_candidate_dtype = get_default_dtype()
    if default_candidate_device.type != requested_candidate_device.type:
        _fail(
            "CANDIDATE_DEVICE_MISMATCH",
            "candidate device differs from DualSolver default device",
        )
    candidate_torch_device = default_candidate_device
    if (
        candidate_generator is generate_query_dual_candidates
        and solver_factory is None
    ):
        _v2._validate_real_solver_net_device(
            net, candidate_torch_device, default_candidate_dtype
        )
    for name, value in (
        ("steps", steps),
        ("block_size", block_size),
        ("replay_chunk_size", replay_chunk_size),
        ("replay_max_workspace_bytes", replay_max_workspace_bytes),
        ("conv_channel_chunk", conv_channel_chunk),
        ("selector_max_adjoint_cells", selector_max_adjoint_cells),
        ("selector_pool_per_rival", selector_pool_per_rival),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Integral, np.integer))
            or int(value) <= 0
        ):
            _fail("INVALID_CONFIG", f"{name} must be a positive integer")
    if (
        not isinstance(selector_time_limit, (int, float))
        or isinstance(selector_time_limit, bool)
        or not math.isfinite(float(selector_time_limit))
        or float(selector_time_limit) <= 0.0
    ):
        _fail(
            "INVALID_CONFIG",
            "selector_time_limit must be finite and positive",
        )
    if any(quota > int(selector_pool_per_rival) for quota in quotas):
        _fail(
            "INVALID_CONFIG",
            "selector_pool_per_rival must cover every fixed stage quota",
        )

    output_lid = _v2._assert_output_id(by_id, preds)
    _v2._check_deadline(effective_deadline, "before V3 root certification")
    try:
        root = _TRUSTED_CERTIFIER(
            net,
            deadline=effective_deadline,
            conv_channel_chunk=int(conv_channel_chunk),
        )
    except QueryDualBoxTimeout as exc:
        raise _v2.QueryDualPipelineTimeout(str(exc)) from exc
    except QueryDualBoxError as exc:
        raise _v2.QueryDualPipelineError(
            "ROOT_CERTIFICATION", str(exc)
        ) from exc
    _v2._check_deadline(effective_deadline, "after V3 root certification")
    if not verify_query_dual_box_certificate(root):
        _fail("ROOT_CERTIFICATION", "V3 root object validation failed")
    current = _v2._clone_bounds(root.bounds)
    if output_lid not in current:
        _fail("ROOT_CERTIFICATION", "root omits ASSERT predecessor")
    output_width = _v2._flat_box(
        current[output_lid], lid=output_lid
    )[0].size
    if rows.shape[1] != output_width:
        _fail("INVALID_PROPERTY", "property width differs from network output")
    root_boxes_hash = _v2._boxes_sha256(net, current)

    predecessors = []
    for target_lid in targets:
        target_preds = preds[target_lid]
        if len(target_preds) != 1:
            _fail("INVALID_TARGETS", f"target ReLU {target_lid} is not unary")
        predecessors.append(int(target_preds[0]))
    try:
        session = _TRUSTED_SESSION_FACTORY(
            net,
            root,
            tuple(predecessors) + (None,),
            deadline=float(effective_deadline),
        )
    except QueryDualReplayTimeout as exc:
        raise _v2.QueryDualPipelineTimeout(str(exc)) from exc
    except QueryDualReplayError as exc:
        raise _v2.QueryDualPipelineError(
            "SEALED_CONTEXT", str(exc)
        ) from exc

    try:
        _v2._check_deadline(
            effective_deadline, "before V3 property selector"
        )
        try:
            plan = selector(
                net=net,
                before=_v2._clone_bounds(current),
                after=_v2._clone_bounds(current),
                C=rows.copy(),
                thresholds=threshold.copy(),
                kind=selector_kind_token,
                output_layer_id=int(output_lid),
                layer_quotas={
                    int(target_lid): int(quota)
                    for target_lid, quota in zip(targets, quotas)
                },
                time_limit=float(selector_time_limit),
                deadline=float(effective_deadline),
                max_adjoint_cells=int(selector_max_adjoint_cells),
                pool_per_rival=int(selector_pool_per_rival),
            )
        except Exception as exc:
            if time.monotonic() >= effective_deadline:
                raise _v2.QueryDualPipelineTimeout(
                    "V3 property selector deadline expired"
                ) from exc
            raise _v2.QueryDualPipelineError(
                "PROPERTY_SELECTOR", f"{type(exc).__name__}: {exc}"
            ) from exc
        _v2._check_deadline(
            effective_deadline, "after V3 property selector"
        )
        selector_property_sha256 = _selector_property_sha256(
            rows,
            threshold,
            kind=selector_kind_token,
        )
        selected_by_layer = _selector_plan_semantics(
            plan,
            targets=targets,
            quotas=quotas,
            root_bounds=current,
            property_sha256=selector_property_sha256,
        )
        selector_selection_sha256 = str(plan.selection_sha256)
        selector_receipt = copy.deepcopy(dict(plan.receipt))
    except Exception:
        session.abort()
        raise

    drafts: list[_TargetDraft] = []
    pending_order: list[_PendingBlock] = []
    previous_chain_sha = str(root.receipt["receipt_sha256"])
    try:
        for stage_index, (
            target_lid,
            quota,
            predecessor,
        ) in enumerate(zip(targets, quotas, predecessors)):
            _v2._check_deadline(
                effective_deadline, f"before V3 target stage {stage_index}"
            )
            parent = _v2._clone_bounds(current)
            parent_hash = _v2._boxes_sha256(net, parent)
            frame = session.seal_bounds(
                parent, start_lids=(int(predecessor),)
            )
            frame_sha = str(frame._content_sha256)
            target_lower, target_upper = _v2._flat_box(
                parent[target_lid], lid=target_lid
            )
            predecessor_kind = _v2._kind(by_id[predecessor])
            if predecessor_kind != "RELU":
                pred_lower, pred_upper = _v2._flat_box(
                    parent[predecessor], lid=predecessor
                )
                if (
                    not _binary64_bits_equal(target_lower, pred_lower)
                    or not _binary64_bits_equal(target_upper, pred_upper)
                ):
                    _fail(
                        "PARENT_FRAME_MISMATCH",
                        f"ReLU {target_lid} preactivation is not bit-identical "
                        "to its predecessor output",
                    )
            selected = selected_by_layer[target_lid]
            eligible, omitted = _selected_partition(
                target_lower, target_upper, selected
            )
            if len(selected) > int(quota):
                _fail(
                    "INVALID_SELECTOR",
                    f"selector exceeded target quota at {target_lid}",
                )
            candidate_kwargs: Dict[str, Any] = {
                "net": net,
                "bounds_dict": _v2._candidate_bounds_on_device(
                    parent, candidate_torch_device
                ),
                "target_relu_lid": int(target_lid),
                "property_rows": None,
                "property_upper_only": True,
                "steps": int(steps),
                "block_size": int(block_size),
                "lr_alpha": float(lr_alpha),
                "lr_decay": float(lr_decay),
                "deadline": float(effective_deadline),
                "descriptor_only": True,
                "selected_target_rows": selected,
            }
            if solver_factory is not None:
                candidate_kwargs["solver_factory"] = solver_factory
            candidate = candidate_generator(**candidate_kwargs)
            _v2._check_deadline(
                effective_deadline,
                f"after V3 target candidate {target_lid}",
            )
            if _v2._boxes_sha256(net, parent) != parent_hash:
                _fail(
                    "PARENT_TOCTOU",
                    f"candidate changed target parent {target_lid}",
                )
            (
                candidate_eligible,
                candidate_omitted,
                records,
            ) = _candidate_v3_common(
                candidate,
                parent,
                target_relu_lid=int(target_lid),
                target_start_lid=int(predecessor),
                selected_rows=selected,
                property_rows=None,
                output_lid=int(output_lid),
                block_size=int(block_size),
                steps=int(steps),
                deadline=float(effective_deadline),
                expected_status=(
                    "descriptors_generated"
                    if selected
                    else "no_queries_fallback"
                ),
                failure_code="CANDIDATE_FAILURE",
                failure_context=f"target {target_lid}",
            )
            if candidate_eligible != eligible or candidate_omitted != omitted:
                _fail(
                    "COVERAGE_ERROR",
                    "candidate sparse target partition mismatch",
                )
            combined_lower = target_lower.copy()
            combined_upper = target_upper.copy()
            stage_pending: list[_PendingBlock] = []
            covered: list[int] = []
            for descriptor_index, descriptor in enumerate(
                candidate.query_descriptors
            ):
                descriptor_rows = tuple(
                    int(value) for value in descriptor.row_ids
                )
                objective = _v2._expected_target_objective(
                    descriptor_rows, target_lower.size
                )
                pending = _pending_descriptor(
                    session=session,
                    frame=frame,
                    net=net,
                    parent_bounds=parent,
                    descriptor=descriptor,
                    candidate_record=records[descriptor_index],
                    alpha_tree=candidate.alpha_trees[
                        descriptor.alpha_tree_index
                    ],
                    query_bias=np.zeros(
                        objective.shape[0], dtype=np.float64
                    ),
                    expected_objectives=objective,
                    expected_kind="relu_unstable_plus_minus_one_hot",
                    expected_target=int(target_lid),
                    expected_start=int(predecessor),
                    expected_rows=descriptor_rows,
                    chunk_size=int(replay_chunk_size),
                    max_workspace_bytes=int(replay_max_workspace_bytes),
                    deadline=float(effective_deadline),
                )
                count = len(descriptor_rows)
                replay_lower = pending.pending.lower_bounds[:count]
                replay_upper = -pending.pending.lower_bounds[count:]
                if (
                    pending.pending.lower_bounds.size != 2 * count
                    or np.any(replay_lower > replay_upper)
                ):
                    _fail(
                        "BOUND_CONFLICT",
                        f"target {target_lid} sparse replay is inconsistent",
                    )
                index = np.asarray(descriptor_rows, dtype=np.int64)
                combined_lower[index] = np.maximum(
                    combined_lower[index], replay_lower
                )
                combined_upper[index] = np.minimum(
                    combined_upper[index], replay_upper
                )
                if np.any(combined_lower > combined_upper):
                    _fail(
                        "BOUND_CONFLICT",
                        f"target {target_lid} sparse intersection is empty",
                    )
                covered.extend(descriptor_rows)
                stage_pending.append(pending)
                pending_order.append(pending)
            if tuple(covered) != selected:
                _fail(
                    "COVERAGE_ERROR",
                    f"target {target_lid} selected coverage/order mismatch",
                )
            omitted_index = np.asarray(omitted, dtype=np.int64)
            if omitted and (
                not _binary64_bits_equal(
                    combined_lower[omitted_index],
                    target_lower[omitted_index],
                )
                or not _binary64_bits_equal(
                    combined_upper[omitted_index],
                    target_upper[omitted_index],
                )
            ):
                _fail(
                    "UNSELECTED_MUTATION",
                    f"target {target_lid} changed an omitted row",
                )
            strict = int(
                np.count_nonzero(
                    (combined_lower > target_lower)
                    | (combined_upper < target_upper)
                )
            )
            next_bounds = _v2._clone_bounds(parent)
            _v2._replace_box(
                next_bounds,
                target_lid,
                combined_lower,
                combined_upper,
            )
            if predecessor_kind != "RELU":
                _v2._replace_box(
                    next_bounds,
                    predecessor,
                    combined_lower,
                    combined_upper,
                )
            result_hash = _v2._boxes_sha256(net, next_bounds)
            result_chain_sha = _v2._json_sha256(
                {
                    "previous": previous_chain_sha,
                    "stage_index": int(stage_index),
                    "parent_boxes_sha256": parent_hash,
                    "result_boxes_sha256": result_hash,
                    "candidate_receipt_sha256": candidate.receipt[
                        "receipt_sha256"
                    ],
                    "bounds_frame_sha256": frame_sha,
                }
            )
            drafts.append(
                _TargetDraft(
                    stage_index=int(stage_index),
                    target_lid=int(target_lid),
                    predecessor=int(predecessor),
                    predecessor_kind=str(predecessor_kind),
                    parent_boxes_sha256=parent_hash,
                    result_boxes_sha256=result_hash,
                    candidate_bounds_sha256=(
                        _v2._candidate_bounds_sha256(parent)
                    ),
                    candidate=candidate,
                    blocks=tuple(stage_pending),
                    lower=combined_lower.copy(),
                    upper=combined_upper.copy(),
                    strict=strict,
                    eligible=eligible,
                    selected=selected,
                    omitted=omitted,
                    bounds_frame_sha256=frame_sha,
                    parent_chain_sha256=previous_chain_sha,
                    result_chain_sha256=result_chain_sha,
                )
            )
            current = next_bounds
            previous_chain_sha = result_chain_sha

        _v2._check_deadline(
            effective_deadline, "before V3 property candidate"
        )
        property_parent = _v2._clone_bounds(current)
        property_parent_hash = _v2._boxes_sha256(net, property_parent)
        property_frame = session.seal_bounds(
            property_parent, start_lids=(None,)
        )
        property_frame_sha = str(property_frame._content_sha256)
        property_kwargs: Dict[str, Any] = {
            "net": net,
            "bounds_dict": _v2._candidate_bounds_on_device(
                property_parent, candidate_torch_device
            ),
            "target_relu_lid": None,
            "property_rows": rows.copy(),
            "property_upper_only": True,
            "steps": int(steps),
            "block_size": int(block_size),
            "lr_alpha": float(lr_alpha),
            "lr_decay": float(lr_decay),
            "deadline": float(effective_deadline),
            "descriptor_only": True,
            "selected_target_rows": (),
        }
        if solver_factory is not None:
            property_kwargs["solver_factory"] = solver_factory
        property_candidate = candidate_generator(**property_kwargs)
        _v2._check_deadline(
            effective_deadline, "after V3 property candidate"
        )
        if _v2._boxes_sha256(net, property_parent) != property_parent_hash:
            _fail("PARENT_TOCTOU", "property candidate changed parent boxes")
        _, _, property_records = _candidate_v3_common(
            property_candidate,
            property_parent,
            target_relu_lid=None,
            target_start_lid=None,
            selected_rows=(),
            property_rows=rows,
            output_lid=int(output_lid),
            block_size=int(block_size),
            steps=int(steps),
            deadline=float(effective_deadline),
            expected_status="descriptors_generated",
            failure_code="NO_PROPERTY_CANDIDATE",
            failure_context="property",
        )
        property_pending: list[_PendingBlock] = []
        property_upper = np.empty(rows.shape[0], dtype=np.float64)
        property_covered: list[int] = []
        for descriptor_index, descriptor in enumerate(
            property_candidate.query_descriptors
        ):
            row_ids = tuple(int(value) for value in descriptor.row_ids)
            index = np.asarray(row_ids, dtype=np.int64)
            if (
                not row_ids
                or np.any(index < 0)
                or np.any(index >= rows.shape[0])
            ):
                _fail("COVERAGE_ERROR", "invalid V3 property row ids")
            objective = -rows[index]
            bias = threshold[index]
            pending = _pending_descriptor(
                session=session,
                frame=property_frame,
                net=net,
                parent_bounds=property_parent,
                descriptor=descriptor,
                candidate_record=property_records[descriptor_index],
                alpha_tree=property_candidate.alpha_trees[
                    descriptor.alpha_tree_index
                ],
                query_bias=bias,
                expected_objectives=objective,
                expected_kind="final_property_negative_c_upper_only",
                expected_target=None,
                expected_start=None,
                expected_rows=row_ids,
                chunk_size=int(replay_chunk_size),
                max_workspace_bytes=int(replay_max_workspace_bytes),
                deadline=float(effective_deadline),
            )
            if pending.pending.lower_bounds.size != len(row_ids):
                _fail(
                    "SHAPE_MISMATCH",
                    "V3 property replay result count mismatch",
                )
            property_upper[index] = -pending.pending.lower_bounds
            property_covered.extend(row_ids)
            property_pending.append(pending)
            pending_order.append(pending)
        if tuple(property_covered) != tuple(range(rows.shape[0])):
            _fail(
                "COVERAGE_ERROR",
                "V3 property rows are not covered exactly once in order",
            )
        if not np.all(np.isfinite(property_upper)):
            _fail("NONFINITE", "V3 property replay is non-finite")
        property_spec_hash = _v2._property_spec_sha256(rows, threshold)
        property_chain_sha = _v2._json_sha256(
            {
                "previous": previous_chain_sha,
                "stage": "property",
                "parent_boxes_sha256": property_parent_hash,
                "candidate_receipt_sha256": property_candidate.receipt[
                    "receipt_sha256"
                ],
                "bounds_frame_sha256": property_frame_sha,
                "property_spec_sha256": property_spec_hash,
                "property_upper_sha256": _v2._array_digest(
                    property_upper
                ),
            }
        )
        property_draft = _PropertyDraft(
            parent_boxes_sha256=property_parent_hash,
            candidate_bounds_sha256=(
                _v2._candidate_bounds_sha256(property_parent)
            ),
            candidate=property_candidate,
            blocks=tuple(property_pending),
            property_upper=property_upper.copy(),
            property_spec_sha256=property_spec_hash,
            bounds_frame_sha256=property_frame_sha,
            parent_chain_sha256=previous_chain_sha,
            result_chain_sha256=property_chain_sha,
        )

        _v2._check_deadline(
            effective_deadline, "before sealed V3 transaction commit"
        )
        committed_results = session.commit()
        _v2._check_deadline(
            effective_deadline, "after sealed V3 transaction commit"
        )
        if len(committed_results) != len(pending_order):
            _fail(
                "INVALID_REPLAY",
                "sealed commit returned an unexpected result count",
            )
        committed_by_pending = {
            id(pending): _committed_block(pending, result)
            for pending, result in zip(pending_order, committed_results)
        }

        stages: list[_v2.QueryDualTargetStage] = []
        for draft in drafts:
            blocks = tuple(
                committed_by_pending[id(value)] for value in draft.blocks
            )
            status = (
                "verified"
                if draft.strict > 0
                else "verified_no_improvement"
            )
            stage_body = {
                "schema": STAGE_SCHEMA,
                "status": status,
                "proof_authority": True,
                "stage_index": draft.stage_index,
                "target_relu_lid": draft.target_lid,
                "predecessor_lid": draft.predecessor,
                "predecessor_kind": draft.predecessor_kind,
                "relu_key_semantics": "preactivation",
                "parent_boxes_sha256": draft.parent_boxes_sha256,
                "result_boxes_sha256": draft.result_boxes_sha256,
                "candidate_bounds_sha256": draft.candidate_bounds_sha256,
                "candidate_receipt_sha256": draft.candidate.receipt[
                    "receipt_sha256"
                ],
                "candidate_schema": CANDIDATE_SCHEMA,
                "candidate_protocol": CANDIDATE_PROTOCOL,
                "candidate_status": draft.candidate.status,
                "candidate_descriptor_coverage_sha256": (
                    draft.candidate.receipt[
                        "descriptor_coverage_sha256"
                    ]
                ),
                "eligible_row_ids": list(draft.eligible),
                "eligible_rows_sha256": _ordered_rows_sha256(
                    draft.eligible
                ),
                "selected_row_ids": list(draft.selected),
                "selected_rows_sha256": _ordered_rows_sha256(
                    draft.selected
                ),
                "omitted_row_ids": list(draft.omitted),
                "omitted_rows_sha256": _ordered_rows_sha256(
                    draft.omitted
                ),
                "partition_complete": True,
                "partition_disjoint": True,
                "selected_coverage_complete": True,
                "eligible_coverage_complete": not bool(draft.omitted),
                "unselected_policy": UNSELECTED_POLICY,
                "unselected_bounds_bit_identical_parent": True,
                "bounds_frame_sha256": draft.bounds_frame_sha256,
                "parent_chain_sha256": draft.parent_chain_sha256,
                "result_chain_sha256": draft.result_chain_sha256,
                "block_receipt_sha256": [
                    block.replay_receipt["receipt_sha256"]
                    for block in blocks
                ],
                "alpha_bridge_sha256": [
                    block.alpha_bridge_sha256 for block in blocks
                ],
                "strict_improvements": int(draft.strict),
                "target_bounds_sha256": _v2._array_digest(
                    np.stack([draft.lower, draft.upper])
                ),
                "commit": "sealed_session_atomic_whole_transaction",
            }
            stage = _v2.QueryDualTargetStage(
                stage_index=draft.stage_index,
                target_relu_lid=draft.target_lid,
                predecessor_lid=draft.predecessor,
                predecessor_kind=draft.predecessor_kind,
                parent_boxes_sha256=draft.parent_boxes_sha256,
                result_boxes_sha256=draft.result_boxes_sha256,
                candidate_bounds_sha256=draft.candidate_bounds_sha256,
                candidate_receipt=draft.candidate.receipt,
                blocks=blocks,
                target_lower=draft.lower,
                target_upper=draft.upper,
                strict_improvements=draft.strict,
                status=status,
                receipt=_v2._receipt(stage_body),
            )
            stages.append(stage)

        property_blocks = tuple(
            committed_by_pending[id(value)]
            for value in property_draft.blocks
        )
        property_body = {
            "schema": PROPERTY_SCHEMA,
            "status": "verified",
            "proof_authority": True,
            "direction": "UPPER",
            "quantity": "C_y_minus_threshold",
            "objective": "-C",
            "replay_query_bias": "+threshold",
            "upper_reconstruction": "-LB(-C_y+threshold)",
            "parent_boxes_sha256": (
                property_draft.parent_boxes_sha256
            ),
            "candidate_bounds_sha256": (
                property_draft.candidate_bounds_sha256
            ),
            "candidate_receipt_sha256": (
                property_candidate.receipt["receipt_sha256"]
            ),
            "candidate_schema": CANDIDATE_SCHEMA,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "candidate_status": property_candidate.status,
            "candidate_descriptor_coverage_sha256": (
                property_candidate.receipt[
                    "descriptor_coverage_sha256"
                ]
            ),
            "bounds_frame_sha256": property_draft.bounds_frame_sha256,
            "parent_chain_sha256": (
                property_draft.parent_chain_sha256
            ),
            "result_chain_sha256": (
                property_draft.result_chain_sha256
            ),
            "block_receipt_sha256": [
                block.replay_receipt["receipt_sha256"]
                for block in property_blocks
            ],
            "alpha_bridge_sha256": [
                block.alpha_bridge_sha256 for block in property_blocks
            ],
            "property_spec_sha256": property_spec_hash,
            "property_upper_sha256": _v2._array_digest(
                property_draft.property_upper
            ),
            "property_rows": int(rows.shape[0]),
            "eligible_property_row_ids": list(range(rows.shape[0])),
            "coverage_complete": True,
            "commit": "sealed_session_atomic_whole_transaction",
        }
        property_stage = _v2.QueryDualPropertyStage(
            parent_boxes_sha256=property_draft.parent_boxes_sha256,
            candidate_bounds_sha256=(
                property_draft.candidate_bounds_sha256
            ),
            candidate_receipt=property_candidate.receipt,
            blocks=property_blocks,
            property_upper=property_draft.property_upper,
            property_spec_sha256=property_spec_hash,
            receipt=_v2._receipt(property_body),
        )

        completed = time.monotonic()
        _v2._check_deadline(
            effective_deadline, "before V3 bundle assembly"
        )
        final_boxes_hash = _v2._boxes_sha256(net, current)
        nonce = secrets.token_hex(32)
        top_body = {
            "schema": SCHEMA,
            "status": "verified",
            "proof_authority": True,
            "authority_source": _AUTHORITY_SOURCE,
            "ordinary_interval_facts_consumed": False,
            "transaction": "all_or_nothing",
            "process_local_identity_capability_required": True,
            "provenance_nonce_sha256": hashlib.sha256(
                nonce.encode("ascii")
            ).hexdigest(),
            "root_receipt_sha256": root.receipt["receipt_sha256"],
            "root_net_sha256": root.receipt["hashes"]["net_sha256"],
            "root_input_sha256": root.receipt["hashes"]["input_sha256"],
            "root_boxes_sha256": root_boxes_hash,
            "final_boxes_sha256": final_boxes_hash,
            "target_relu_ids": list(targets),
            "stage_quotas": list(quotas),
            "selector_schema": SELECTOR_SCHEMA,
            "selector_kind": selector_kind_token,
            "selector_property_sha256": selector_property_sha256,
            "selector_selection_sha256": selector_selection_sha256,
            "selector_receipt": selector_receipt,
            "stage_receipt_sha256": [
                stage.receipt["receipt_sha256"] for stage in stages
            ],
            "property_receipt_sha256": (
                property_stage.receipt["receipt_sha256"]
            ),
            "property_spec_sha256": property_spec_hash,
            "property_upper_sha256": _v2._array_digest(
                property_draft.property_upper
            ),
            "candidate_schema": CANDIDATE_SCHEMA,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "replay_schema": REPLAY_SCHEMA,
            "replay_protocol": REPLAY_PROTOCOL,
            "target_candidate_receipt_sha256": [
                stage.candidate_receipt["receipt_sha256"]
                for stage in stages
            ],
            "property_candidate_receipt_sha256": (
                property_candidate.receipt["receipt_sha256"]
            ),
            "final_stage_chain_sha256": (
                property_draft.result_chain_sha256
            ),
            "candidate_generator": _v2._callable_name(
                candidate_generator
            ),
            "candidate_solver_factory": _v2._callable_name(
                solver_factory
            ),
            "selector": _v2._callable_name(selector),
            "root_certifier": _v2._callable_name(_TRUSTED_CERTIFIER),
            "sealed_session_factory": _v2._callable_name(
                _TRUSTED_SESSION_FACTORY
            ),
            "steps": int(steps),
            "block_size": int(block_size),
            "replay_chunk_size": int(replay_chunk_size),
            "replay_max_workspace_bytes": int(
                replay_max_workspace_bytes
            ),
            "conv_channel_chunk": int(conv_channel_chunk),
            "candidate_device": candidate_device,
            "dual_solver_default_device": str(default_candidate_device),
            "dual_solver_default_dtype": str(default_candidate_dtype),
            "candidate_device_fallback": False,
            "candidate_cuda_device_name": (
                torch.cuda.get_device_name(candidate_torch_device)
                if candidate_device == "cuda"
                else None
            ),
            "selector_time_limit": float(selector_time_limit),
            "selector_max_adjoint_cells": int(
                selector_max_adjoint_cells
            ),
            "selector_pool_per_rival": int(
                selector_pool_per_rival
            ),
            "non_authoritative_audit_fields": list(
                _PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS
            ),
            "unselected_policy": UNSELECTED_POLICY,
            "property_coverage_policy": "all_property_rows",
            "sealed_commit_completed": True,
            "deadline_present": True,
            "deadline_monotonic_hex": float(
                effective_deadline
            ).hex(),
            "started_monotonic_hex": float(started).hex(),
            "completed_monotonic_hex": float(completed).hex(),
            "completed_before_deadline": completed < effective_deadline,
        }
        bundle = _v2.VerifiedQueryDualFeedback(
            root_certificate=root,
            certified_bounds=current,
            target_relu_ids=targets,
            stages=tuple(stages),
            property_stage=property_stage,
            property_upper=property_draft.property_upper,
            receipt=_v2._receipt(top_body),
            provenance_nonce=nonce,
        )
        if not _validate_v3_contents(
            bundle,
            net=net,
            property_rows=rows,
            thresholds=threshold,
            expected_target_relu_ids=targets,
            verify_live_net=False,
        ):
            _fail(
                "INTERNAL_VALIDATION",
                "fresh V3 transaction failed deterministic validation",
            )
        _v2._check_deadline(
            effective_deadline, "after V3 transaction self-validation"
        )
        _v2._register_live(bundle)
        return bundle
    except _v2.QueryDualPipelineError:
        session.abort()
        raise
    except (QueryDualReplayTimeout, QueryDualBoxTimeout) as exc:
        session.abort()
        raise _v2.QueryDualPipelineTimeout(str(exc)) from exc
    except (QueryDualReplayError, QueryDualBoxError) as exc:
        session.abort()
        raise _v2.QueryDualPipelineError(
            "INDEPENDENT_PROOF", str(exc)
        ) from exc
    except Exception as exc:
        session.abort()
        raise _v2.QueryDualPipelineError(
            "TRANSACTION_ABORTED", f"{type(exc).__name__}: {exc}"
        ) from exc


def _sealed_replay_semantics(
    block: _v2.QueryDualAuthorityBlock,
    *,
    frame_sha256: str,
    root_receipt: Mapping[str, Any],
    validation_context: Any,
    expected_start: Optional[int],
) -> Optional[Tuple[str, str]]:
    """Validate the sealed-context envelope and return session/crosswalk ids."""

    try:
        receipt = block.replay_receipt
        sealed = receipt["sealed_context"]
        crosswalk = sealed["manifest_crosswalk"]
        expected_crosswalk = dict(validation_context.crosswalk)
        expected_crosswalk_sha = _v2._json_sha256(expected_crosswalk)
        expected_cones = list(expected_crosswalk["replay_cones"])
        expected_start_record = (
            "ASSERT_PREDECESSOR"
            if expected_start is None
            else int(expected_start)
        )
        matching_cones = [
            entry
            for entry in expected_cones
            if entry["start_layer"] == expected_start_record
        ]
        sealed_without_hash = dict(sealed)
        claimed_context = str(
            sealed_without_hash.pop("context_sha256")
        )
        if (
            receipt.get("schema") != REPLAY_SCHEMA
            or receipt.get("proof_authority") is not True
            or receipt.get("authority_source")
            != (
                "independent_reverse_topological_replay_"
                "sealed_transaction"
            )
            or sealed.get("protocol") != REPLAY_PROTOCOL
            or sealed.get("live_net_commit_bound") is not True
            or sealed.get("live_net_bind")
            != (
                "root_certificate_full_live_verification_"
                "once_at_commit"
            )
            or not _exact_int(
                sealed.get("network_snapshot_freeze_count"), 1
            )
            or not _exact_int(
                sealed.get("unique_cone_count"), len(expected_cones)
            )
            or sealed.get("bounds_frame_sha256") != frame_sha256
            or sealed.get("root_net_sha256")
            != root_receipt["hashes"]["net_sha256"]
            or not isinstance(crosswalk, Mapping)
            or _v2._json_sha256(crosswalk) != expected_crosswalk_sha
            or len(matching_cones) != 1
            or sealed.get("manifest_crosswalk_sha256")
            != expected_crosswalk_sha
            or claimed_context
            != _v2._json_sha256(sealed_without_hash)
            or sealed.get("replay_net_sha256")
            != block.replay_net_sha256
            or sealed.get("replay_net_sha256")
            != matching_cones[0]["replay_net_sha256"]
        ):
            return None
        session_sha = str(sealed.get("session_nonce_sha256", ""))
        crosswalk_sha = str(sealed.get("manifest_crosswalk_sha256", ""))
        if (
            len(session_sha) != 64
            or len(crosswalk_sha) != 64
            or any(
                value not in "0123456789abcdef"
                for value in session_sha + crosswalk_sha
            )
        ):
            return None
        return session_sha, crosswalk_sha
    except (KeyError, TypeError, ValueError):
        return None


def _validate_v3_block(
    block: _v2.QueryDualAuthorityBlock,
    *,
    net: Any,
    parent_bounds: Mapping[int, Bounds],
    expected_objective: np.ndarray,
    expected_bias: np.ndarray,
    expected_kind: str,
    expected_target: Optional[int],
    expected_start: Optional[int],
    expected_rows: Tuple[int, ...],
    candidate_record: Mapping[str, Any],
    frame_sha256: str,
    root_receipt: Mapping[str, Any],
    validation_context: Any,
    replay_chunk_size: int,
    replay_max_workspace_bytes: int,
) -> Optional[Tuple[str, str]]:
    if (
        not _v2._validate_replay_block(
            block,
            net=net,
            parent_bounds=parent_bounds,
            expected_objective=expected_objective,
            expected_bias=expected_bias,
            expected_kind=expected_kind,
            expected_target=expected_target,
            expected_start=expected_start,
            expected_rows=expected_rows,
            candidate_record=candidate_record,
        )
        or block.replay_receipt.get("requested_chunk_size")
        != int(replay_chunk_size)
        or block.replay_receipt.get("max_workspace_bytes")
        != int(replay_max_workspace_bytes)
    ):
        return None
    return _sealed_replay_semantics(
        block,
        frame_sha256=frame_sha256,
        root_receipt=root_receipt,
        validation_context=validation_context,
        expected_start=expected_start,
    )


def _validate_v3_contents(
    bundle: _v2.VerifiedQueryDualFeedback,
    *,
    net: Any,
    property_rows: Any,
    thresholds: Any,
    expected_target_relu_ids: Optional[Sequence[int]],
    verify_live_net: bool,
) -> bool:
    """Deterministically validate V3, optionally rebinding the live network."""

    try:
        if (
            not isinstance(bundle, _v2.VerifiedQueryDualFeedback)
            or bundle.proof_authority is not True
            or not _v2._verify_receipt(bundle.receipt, SCHEMA)
        ):
            return False
        if verify_live_net:
            if not verify_query_dual_box_certificate(
                bundle.root_certificate, net=net
            ):
                return False
        elif not verify_query_dual_box_certificate(
            bundle.root_certificate
        ):
            return False
        rows, threshold = _v2._normalise_property(
            property_rows, thresholds
        )
        by_id, preds = _v2._layer_maps(net)
        output_lid = _v2._assert_output_id(by_id, preds)
        expected_targets = (
            bundle.target_relu_ids
            if expected_target_relu_ids is None
            else _v2._normalise_targets(
                expected_target_relu_ids, by_id
            )
        )
        if bundle.target_relu_ids != expected_targets:
            return False
        expected_replay_starts: list[Optional[int]] = []
        for target_lid in expected_targets:
            target_preds = preds[target_lid]
            if len(target_preds) != 1:
                return False
            predecessor = int(target_preds[0])
            if predecessor not in expected_replay_starts:
                expected_replay_starts.append(predecessor)
        expected_replay_starts.append(None)
        validation_context = _build_query_dual_replay_validation_context(
            bundle.root_certificate,
            tuple(expected_replay_starts),
        )
        receipt = bundle.receipt
        quotas_raw = receipt.get("stage_quotas")
        if not isinstance(quotas_raw, list):
            return False
        quotas = _normalise_quotas(expected_targets, quotas_raw)
        if (
            expected_targets != tuple(sorted(expected_targets))
            or not any(quota > 0 for quota in quotas)
        ):
            return False
        root_receipt = bundle.root_certificate.receipt
        if (
            receipt.get("status") != "verified"
            or receipt.get("proof_authority") is not True
            or receipt.get("authority_source") != _AUTHORITY_SOURCE
            or receipt.get("transaction") != "all_or_nothing"
            or receipt.get("ordinary_interval_facts_consumed") is not False
            or receipt.get(
                "process_local_identity_capability_required"
            )
            is not True
            or receipt.get("candidate_schema") != CANDIDATE_SCHEMA
            or receipt.get("candidate_protocol") != CANDIDATE_PROTOCOL
            or receipt.get("replay_schema") != REPLAY_SCHEMA
            or receipt.get("replay_protocol") != REPLAY_PROTOCOL
            or receipt.get("selector_schema") != SELECTOR_SCHEMA
            or receipt.get("unselected_policy") != UNSELECTED_POLICY
            or receipt.get("property_coverage_policy")
            != "all_property_rows"
            or receipt.get("sealed_commit_completed") is not True
            or receipt.get("candidate_device_fallback") is not False
            or receipt.get("candidate_device") not in {"cpu", "cuda"}
            or receipt.get("non_authoritative_audit_fields")
            != _PIPELINE_NON_AUTHORITATIVE_AUDIT_FIELDS
            or receipt.get("root_certifier")
            != _v2._callable_name(_TRUSTED_CERTIFIER)
            or receipt.get("sealed_session_factory")
            != _v2._callable_name(_TRUSTED_SESSION_FACTORY)
            or receipt.get("target_relu_ids") != list(expected_targets)
            or receipt.get("root_receipt_sha256")
            != root_receipt["receipt_sha256"]
            or receipt.get("root_net_sha256")
            != root_receipt["hashes"]["net_sha256"]
            or receipt.get("root_input_sha256")
            != root_receipt["hashes"]["input_sha256"]
            or receipt.get("conv_channel_chunk")
            != root_receipt.get("conv_channel_chunk")
            or receipt.get("provenance_nonce_sha256")
            != hashlib.sha256(
                bundle.provenance_nonce.encode("ascii")
            ).hexdigest()
        ):
            return False
        for config_key in (
            "steps",
            "block_size",
            "replay_chunk_size",
            "replay_max_workspace_bytes",
            "conv_channel_chunk",
            "selector_max_adjoint_cells",
            "selector_pool_per_rival",
        ):
            value = receipt.get(config_key)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (Integral, np.integer))
                or int(value) <= 0
            ):
                return False
        selector_time_limit = receipt.get("selector_time_limit")
        if (
            isinstance(selector_time_limit, bool)
            or not isinstance(selector_time_limit, (int, float))
            or not math.isfinite(float(selector_time_limit))
            or float(selector_time_limit) <= 0.0
        ):
            return False
        started = float.fromhex(receipt["started_monotonic_hex"])
        completed = float.fromhex(receipt["completed_monotonic_hex"])
        deadline = float.fromhex(receipt["deadline_monotonic_hex"])
        if (
            not all(math.isfinite(value) for value in (started, completed, deadline))
            or completed < started
            or completed >= deadline
            or receipt.get("deadline_present") is not True
            or receipt.get("completed_before_deadline") is not True
        ):
            return False

        selector_receipt = receipt.get("selector_receipt")
        if not isinstance(selector_receipt, Mapping):
            return False
        selector_kind_token = _normalise_selector_kind(
            receipt.get("selector_kind")
        )
        if receipt.get("selector_kind") != selector_kind_token:
            return False
        selector_property_hash = _selector_property_sha256(
            rows,
            threshold,
            kind=selector_kind_token,
        )
        current = _v2._clone_bounds(bundle.root_certificate.bounds)
        selector_selected_by_layer = _selector_receipt_semantics(
            selector_receipt,
            targets=expected_targets,
            quotas=quotas,
            root_bounds=current,
            property_sha256=selector_property_hash,
        )
        selector_selection_hash = str(
            selector_receipt["selection_sha256"]
        )
        selected_schedule = [
            [int(target_lid), int(row)]
            for target_lid in expected_targets
            for row in selector_selected_by_layer[target_lid]
        ]
        if (
            receipt.get("selector_property_sha256")
            != selector_property_hash
            or receipt.get("selector_selection_sha256")
            != selector_selection_hash
        ):
            return False

        root_hash = _v2._boxes_sha256(net, current)
        if (
            receipt.get("root_boxes_sha256") != root_hash
            or len(bundle.stages) != len(expected_targets)
            or receipt.get("stage_receipt_sha256")
            != [
                stage.receipt["receipt_sha256"]
                for stage in bundle.stages
            ]
            or receipt.get("target_candidate_receipt_sha256")
            != [
                stage.candidate_receipt["receipt_sha256"]
                for stage in bundle.stages
            ]
        ):
            return False

        sessions: set[str] = set()
        crosswalks: set[str] = set()
        frame_payloads: list[Tuple[str, Mapping[str, Any]]] = []
        reconstructed_schedule = []
        previous_chain_sha = str(root_receipt["receipt_sha256"])
        for stage_index, (target_lid, quota, stage) in enumerate(
            zip(expected_targets, quotas, bundle.stages)
        ):
            if (
                not isinstance(stage, _v2.QueryDualTargetStage)
                or not _v2._verify_receipt(
                    stage.receipt, STAGE_SCHEMA
                )
                or stage.stage_index != stage_index
                or stage.target_relu_lid != target_lid
                or stage.parent_boxes_sha256
                != _v2._boxes_sha256(net, current)
                or stage.candidate_bounds_sha256
                != _v2._candidate_bounds_sha256(current)
            ):
                return False
            candidate_receipt = stage.candidate_receipt
            target_preds = preds[target_lid]
            if len(target_preds) != 1:
                return False
            predecessor = int(target_preds[0])
            predecessor_kind = _v2._kind(by_id[predecessor])
            frame_payloads.append(
                (
                    str(stage.receipt.get("bounds_frame_sha256", "")),
                    _query_dual_replay_frame_payload(
                        validation_context,
                        current,
                        start_lids=(predecessor,),
                    ),
                )
            )
            base_lower, base_upper = _v2._flat_box(
                current[target_lid], lid=target_lid
            )
            if predecessor_kind != "RELU":
                pred_lower, pred_upper = _v2._flat_box(
                    current[predecessor], lid=predecessor
                )
                if (
                    not _binary64_bits_equal(base_lower, pred_lower)
                    or not _binary64_bits_equal(base_upper, pred_upper)
                ):
                    return False
            selected = _receipt_row_ids(
                stage.receipt.get("selected_row_ids"),
                name=f"stage {stage_index} selected_row_ids",
            )
            eligible, omitted = _selected_partition(
                base_lower, base_upper, selected
            )
            if len(selected) > quota:
                return False
            reconstructed_schedule.extend(
                [[int(target_lid), int(row)] for row in selected]
            )
            candidate_eligible, candidate_omitted, records = (
                _candidate_v3_receipt_semantics(
                    candidate_receipt,
                    current,
                    target_relu_lid=target_lid,
                    target_start_lid=predecessor,
                    selected_rows=selected,
                    property_rows=None,
                    output_lid=int(output_lid),
                    block_size=int(receipt["block_size"]),
                    steps=int(receipt["steps"]),
                    deadline=deadline,
                    descriptor_count=len(stage.blocks),
                )
            )
            stage_eligible = _receipt_row_ids(
                stage.receipt.get("eligible_row_ids"),
                name=f"stage {stage_index} eligible_row_ids",
            )
            stage_omitted = _receipt_row_ids(
                stage.receipt.get("omitted_row_ids"),
                name=f"stage {stage_index} omitted_row_ids",
            )
            if (
                selected != selector_selected_by_layer[target_lid]
                or candidate_eligible != eligible
                or candidate_omitted != omitted
                or len(selected) != min(int(quota), len(eligible))
                or stage_eligible != eligible
                or stage_omitted != omitted
                or stage.receipt.get("eligible_rows_sha256")
                != _ordered_rows_sha256(eligible)
                or stage.receipt.get("selected_rows_sha256")
                != _ordered_rows_sha256(selected)
                or stage.receipt.get("omitted_rows_sha256")
                != _ordered_rows_sha256(omitted)
                or stage.receipt.get("partition_complete") is not True
                or stage.receipt.get("partition_disjoint") is not True
                or stage.receipt.get("selected_coverage_complete")
                is not True
                or stage.receipt.get("eligible_coverage_complete")
                is not (not bool(omitted))
                or stage.receipt.get("unselected_policy")
                != UNSELECTED_POLICY
                or stage.receipt.get(
                    "unselected_bounds_bit_identical_parent"
                )
                is not True
            ):
                return False
            expected_status = (
                "descriptors_generated"
                if selected
                else "no_queries_fallback"
            )
            lower = base_lower.copy()
            upper = base_upper.copy()
            covered = []
            for block_index, block in enumerate(stage.blocks):
                block_rows = tuple(int(value) for value in block.row_ids)
                objective = _v2._expected_target_objective(
                    block_rows, base_lower.size
                )
                context_ids = _validate_v3_block(
                    block,
                    net=net,
                    parent_bounds=current,
                    expected_objective=objective,
                    expected_bias=np.zeros(
                        objective.shape[0], dtype=np.float64
                    ),
                    expected_kind=(
                        "relu_unstable_plus_minus_one_hot"
                    ),
                    expected_target=target_lid,
                    expected_start=predecessor,
                    expected_rows=block_rows,
                    candidate_record=records[block_index],
                    frame_sha256=str(
                        stage.receipt.get("bounds_frame_sha256")
                    ),
                    root_receipt=root_receipt,
                    validation_context=validation_context,
                    replay_chunk_size=int(
                        receipt["replay_chunk_size"]
                    ),
                    replay_max_workspace_bytes=int(
                        receipt["replay_max_workspace_bytes"]
                    ),
                )
                if context_ids is None:
                    return False
                sessions.add(context_ids[0])
                crosswalks.add(context_ids[1])
                count = len(block_rows)
                raw_lower = block.lower_bounds[:count]
                raw_upper = -block.lower_bounds[count:]
                if (
                    block.block_id != block_index
                    or block.lower_bounds.size != 2 * count
                    or np.any(raw_lower > raw_upper)
                    or any(value not in selected for value in block_rows)
                ):
                    return False
                index = np.asarray(block_rows, dtype=np.int64)
                lower[index] = np.maximum(lower[index], raw_lower)
                upper[index] = np.minimum(upper[index], raw_upper)
                if np.any(lower > upper):
                    return False
                covered.extend(block_rows)
            if tuple(covered) != selected:
                return False
            omitted_index = np.asarray(omitted, dtype=np.int64)
            if omitted and (
                not _binary64_bits_equal(
                    lower[omitted_index], base_lower[omitted_index]
                )
                or not _binary64_bits_equal(
                    upper[omitted_index], base_upper[omitted_index]
                )
            ):
                return False
            strict = int(
                np.count_nonzero(
                    (lower > base_lower) | (upper < base_upper)
                )
            )
            next_bounds = _v2._clone_bounds(current)
            _v2._replace_box(
                next_bounds, target_lid, lower, upper
            )
            if predecessor_kind != "RELU":
                _v2._replace_box(
                    next_bounds, predecessor, lower, upper
                )
            result_hash = _v2._boxes_sha256(net, next_bounds)
            result_chain_sha = _v2._json_sha256(
                {
                    "previous": previous_chain_sha,
                    "stage_index": int(stage_index),
                    "parent_boxes_sha256": stage.parent_boxes_sha256,
                    "result_boxes_sha256": result_hash,
                    "candidate_receipt_sha256": candidate_receipt[
                        "receipt_sha256"
                    ],
                    "bounds_frame_sha256": stage.receipt.get(
                        "bounds_frame_sha256"
                    ),
                }
            )
            stage_status = (
                "verified" if strict > 0 else "verified_no_improvement"
            )
            if (
                stage.predecessor_lid != predecessor
                or stage.predecessor_kind != predecessor_kind
                or not _binary64_bits_equal(stage.target_lower, lower)
                or not _binary64_bits_equal(stage.target_upper, upper)
                or stage.strict_improvements != strict
                or stage.status != stage_status
                or stage.result_boxes_sha256 != result_hash
                or stage.receipt.get("status") != stage_status
                or stage.receipt.get("proof_authority") is not True
                or stage.receipt.get("stage_index") != stage_index
                or stage.receipt.get("target_relu_lid") != target_lid
                or stage.receipt.get("predecessor_lid") != predecessor
                or stage.receipt.get("predecessor_kind")
                != predecessor_kind
                or stage.receipt.get("relu_key_semantics")
                != "preactivation"
                or stage.receipt.get("candidate_receipt_sha256")
                != candidate_receipt["receipt_sha256"]
                or stage.receipt.get("candidate_schema")
                != CANDIDATE_SCHEMA
                or stage.receipt.get("candidate_protocol")
                != CANDIDATE_PROTOCOL
                or stage.receipt.get("candidate_status")
                != expected_status
                or stage.receipt.get(
                    "candidate_descriptor_coverage_sha256"
                )
                != candidate_receipt["descriptor_coverage_sha256"]
                or stage.receipt.get("parent_boxes_sha256")
                != stage.parent_boxes_sha256
                or stage.receipt.get("result_boxes_sha256")
                != result_hash
                or stage.receipt.get("candidate_bounds_sha256")
                != stage.candidate_bounds_sha256
                or stage.receipt.get("strict_improvements") != strict
                or stage.receipt.get("target_bounds_sha256")
                != _v2._array_digest(np.stack([lower, upper]))
                or stage.receipt.get("block_receipt_sha256")
                != [
                    block.replay_receipt["receipt_sha256"]
                    for block in stage.blocks
                ]
                or stage.receipt.get("alpha_bridge_sha256")
                != [
                    block.alpha_bridge_sha256
                    for block in stage.blocks
                ]
                or stage.receipt.get("parent_chain_sha256")
                != previous_chain_sha
                or stage.receipt.get("result_chain_sha256")
                != result_chain_sha
                or stage.receipt.get("commit")
                != "sealed_session_atomic_whole_transaction"
            ):
                return False
            current = next_bounds
            previous_chain_sha = result_chain_sha
        if reconstructed_schedule != selected_schedule:
            return False

        property_stage = bundle.property_stage
        if (
            not isinstance(property_stage, _v2.QueryDualPropertyStage)
            or not _v2._verify_receipt(
                property_stage.receipt, PROPERTY_SCHEMA
            )
        ):
            return False
        candidate_receipt = property_stage.candidate_receipt
        parent_hash = _v2._boxes_sha256(net, current)
        candidate_hash = _v2._candidate_bounds_sha256(current)
        spec_hash = _v2._property_spec_sha256(rows, threshold)
        expected_property_ids = tuple(range(rows.shape[0]))
        if (
            property_stage.parent_boxes_sha256 != parent_hash
            or property_stage.candidate_bounds_sha256 != candidate_hash
            or property_stage.property_spec_sha256 != spec_hash
        ):
            return False
        frame_payloads.append(
            (
                str(
                    property_stage.receipt.get(
                        "bounds_frame_sha256", ""
                    )
                ),
                _query_dual_replay_frame_payload(
                    validation_context,
                    current,
                    start_lids=(None,),
                ),
            )
        )
        _, _, records = _candidate_v3_receipt_semantics(
            candidate_receipt,
            current,
            target_relu_lid=None,
            target_start_lid=None,
            selected_rows=(),
            property_rows=rows,
            output_lid=int(output_lid),
            block_size=int(receipt["block_size"]),
            steps=int(receipt["steps"]),
            deadline=deadline,
            descriptor_count=len(property_stage.blocks),
        )
        upper = np.empty(rows.shape[0], dtype=np.float64)
        covered = []
        for block_index, block in enumerate(property_stage.blocks):
            row_ids = tuple(int(value) for value in block.row_ids)
            index = np.asarray(row_ids, dtype=np.int64)
            if (
                block.block_id != block_index
                or not row_ids
                or np.any(index < 0)
                or np.any(index >= rows.shape[0])
            ):
                return False
            objective = -rows[index]
            bias = threshold[index]
            context_ids = _validate_v3_block(
                block,
                net=net,
                parent_bounds=current,
                expected_objective=objective,
                expected_bias=bias,
                expected_kind=(
                    "final_property_negative_c_upper_only"
                ),
                expected_target=None,
                expected_start=None,
                expected_rows=row_ids,
                candidate_record=records[block_index],
                frame_sha256=str(
                    property_stage.receipt.get(
                        "bounds_frame_sha256"
                    )
                ),
                root_receipt=root_receipt,
                validation_context=validation_context,
                replay_chunk_size=int(receipt["replay_chunk_size"]),
                replay_max_workspace_bytes=int(
                    receipt["replay_max_workspace_bytes"]
                ),
            )
            if context_ids is None:
                return False
            sessions.add(context_ids[0])
            crosswalks.add(context_ids[1])
            if block.lower_bounds.size != len(row_ids):
                return False
            upper[index] = -block.lower_bounds
            covered.extend(row_ids)
        final_hash = _v2._boxes_sha256(net, current)
        property_chain_sha = _v2._json_sha256(
            {
                "previous": previous_chain_sha,
                "stage": "property",
                "parent_boxes_sha256": parent_hash,
                "candidate_receipt_sha256": candidate_receipt[
                    "receipt_sha256"
                ],
                "bounds_frame_sha256": property_stage.receipt.get(
                    "bounds_frame_sha256"
                ),
                "property_spec_sha256": spec_hash,
                "property_upper_sha256": _v2._array_digest(upper),
            }
        )
        frames_match_committed_session = False
        if len(sessions) == 1:
            session_sha = next(iter(sessions))
            frames_match_committed_session = all(
                claimed
                == _query_dual_replay_frame_sha256(
                    payload,
                    session_nonce_sha256=session_sha,
                )
                for claimed, payload in frame_payloads
            )
        if (
            tuple(covered) != expected_property_ids
            or len(sessions) != 1
            or len(crosswalks) != 1
            or not frames_match_committed_session
            or not _binary64_bits_equal(
                property_stage.property_upper, upper
            )
            or not _binary64_bits_equal(bundle.property_upper, upper)
            or _v2._boxes_sha256(net, bundle.certified_bounds)
            != final_hash
            or set(bundle.certified_bounds) != set(current)
            or any(
                not _binary64_bits_equal(
                    _v2._flat_box(
                        bundle.certified_bounds[lid], lid=lid
                    )[0],
                    _v2._flat_box(current[lid], lid=lid)[0],
                )
                or not _binary64_bits_equal(
                    _v2._flat_box(
                        bundle.certified_bounds[lid], lid=lid
                    )[1],
                    _v2._flat_box(current[lid], lid=lid)[1],
                )
                for lid in current
            )
            or property_stage.receipt.get("status") != "verified"
            or property_stage.receipt.get("proof_authority") is not True
            or property_stage.receipt.get("direction") != "UPPER"
            or property_stage.receipt.get("quantity")
            != "C_y_minus_threshold"
            or property_stage.receipt.get("objective") != "-C"
            or property_stage.receipt.get("replay_query_bias")
            != "+threshold"
            or property_stage.receipt.get("upper_reconstruction")
            != "-LB(-C_y+threshold)"
            or property_stage.receipt.get("parent_boxes_sha256")
            != parent_hash
            or property_stage.receipt.get(
                "candidate_bounds_sha256"
            )
            != candidate_hash
            or property_stage.receipt.get(
                "candidate_receipt_sha256"
            )
            != candidate_receipt["receipt_sha256"]
            or property_stage.receipt.get("candidate_schema")
            != CANDIDATE_SCHEMA
            or property_stage.receipt.get("candidate_protocol")
            != CANDIDATE_PROTOCOL
            or property_stage.receipt.get("candidate_status")
            != "descriptors_generated"
            or property_stage.receipt.get(
                "candidate_descriptor_coverage_sha256"
            )
            != candidate_receipt["descriptor_coverage_sha256"]
            or property_stage.receipt.get("property_spec_sha256")
            != spec_hash
            or property_stage.receipt.get("property_upper_sha256")
            != _v2._array_digest(upper)
            or property_stage.receipt.get("property_rows")
            != int(rows.shape[0])
            or property_stage.receipt.get(
                "eligible_property_row_ids"
            )
            != list(expected_property_ids)
            or property_stage.receipt.get("coverage_complete") is not True
            or property_stage.receipt.get("block_receipt_sha256")
            != [
                block.replay_receipt["receipt_sha256"]
                for block in property_stage.blocks
            ]
            or property_stage.receipt.get("alpha_bridge_sha256")
            != [
                block.alpha_bridge_sha256
                for block in property_stage.blocks
            ]
            or property_stage.receipt.get("parent_chain_sha256")
            != previous_chain_sha
            or property_stage.receipt.get("result_chain_sha256")
            != property_chain_sha
            or property_stage.receipt.get("commit")
            != "sealed_session_atomic_whole_transaction"
            or receipt.get("final_boxes_sha256") != final_hash
            or receipt.get("property_receipt_sha256")
            != property_stage.receipt["receipt_sha256"]
            or receipt.get("property_spec_sha256") != spec_hash
            or receipt.get("property_upper_sha256")
            != _v2._array_digest(upper)
            or receipt.get("property_candidate_receipt_sha256")
            != candidate_receipt["receipt_sha256"]
            or receipt.get("final_stage_chain_sha256")
            != property_chain_sha
        ):
            return False
        return True
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        QueryDualReplayError,
        QueryDualBoxError,
        _v2.QueryDualPipelineError,
    ):
        return False


def validate_verified_query_dual_feedback_v3(
    bundle: _v2.VerifiedQueryDualFeedback,
    *,
    net: Any,
    property_rows: Any,
    thresholds: Any,
    expected_target_relu_ids: Optional[Sequence[int]] = None,
    require_live_provenance: bool = True,
) -> bool:
    """Validate a process-local V3 bundle before every authority use."""

    try:
        if (
            require_live_provenance
            and not _v2._has_live_capability(bundle)
        ):
            return False
        return _validate_v3_contents(
            bundle,
            net=net,
            property_rows=property_rows,
            thresholds=thresholds,
            expected_target_relu_ids=expected_target_relu_ids,
            verify_live_net=True,
        )
    except (AttributeError, TypeError, ValueError):
        return False


__all__ = [
    "CANDIDATE_PROTOCOL",
    "CANDIDATE_SCHEMA",
    "PROPERTY_SCHEMA",
    "REPLAY_PROTOCOL",
    "REPLAY_SCHEMA",
    "SCHEMA",
    "SELECTOR_SCHEMA",
    "STAGE_SCHEMA",
    "build_verified_query_dual_feedback_v3",
    "validate_verified_query_dual_feedback_v3",
]
