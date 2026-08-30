#===- query_dual_replay_v51.py - Experimental V5.1 replay ------------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Complete, isolated, non-authoritative V5.1 replay candidate.

The frozen V3 graph/parser/ReLU/DAG/scalar machinery is reused without
changing its production entry point.  Dense affine roundoff is dispatched to
``query_dual_scalar_guard_v51``; dense Conv2D uses
``query_dual_replay_v51_conv``; sparse Conv2D remains byte-for-byte on the V3
componentwise-radius branch.

This module is a controlled research candidate.  Its receipt is useful for
mutation tests and internal cross-checks, but its unkeyed hashes cannot
authenticate coordinated replacement of the underlying query material.
Every public result permanently carries ``proof_authority=False``; only the
separate root-owned session may bind live material to process-local evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf.query_dual_replay_v51_conv import (
    DenseConvV51Plan,
    prepare_dense_conv_v51_plan,
    replay_dense_conv_v51,
)
from act.back_end.hybridz_tf.query_dual_scalar_guard_v51 import (
    DenseV51Support,
    QueryDualScalarGuardV51Error,
    check_v51_platform,
    dense_support_compressed_guard_v51,
    prepare_dense_support_v51,
)


SCHEMA = "act.query_dual_replay_v51_candidate.v1"
NUMERIC_PROTOCOL = "wide_support_structural_scalar_guard_v51"


@dataclass(frozen=True)
class QueryDualReplayV51CandidateResult:
    """Immutable research output that can never issue a solver verdict."""

    lower_bounds: np.ndarray = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("the V5.1 candidate cannot issue authority")
        values = np.asarray(self.lower_bounds)
        if (
            values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "V5.1 lower bounds must be immutable finite binary64"
            )


@dataclass
class _V51Context:
    prepared: frozen._Prepared
    dense_supports: Dict[int, DenseV51Support] = field(
        default_factory=dict
    )
    conv_plans: Dict[int, DenseConvV51Plan] = field(
        default_factory=dict
    )
    executions: List[Mapping[str, Any]] = field(default_factory=list)
    execution_observer: Optional[Callable[..., None]] = None
    query_block_manifest: Mapping[str, Any] = field(init=False)
    query_block_sha256: str = field(init=False)
    query_span_bindings: Dict[
        Tuple[int, int], Mapping[str, Any]
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        manifest = _query_block_manifest(self.prepared)
        self.query_block_manifest = MappingProxyType(manifest)
        self.query_block_sha256 = _json_sha256(manifest)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _bool_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    header = json.dumps(
        {"dtype": "|b1", "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.astype(np.uint8, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _query_block_manifest(
    prepared: frozen._Prepared,
) -> Dict[str, Any]:
    alpha_entries = [
        {
            "layer_id": int(layer_id),
            "sha256": frozen._array_digest(
                np.ascontiguousarray(prepared.alpha[layer_id])
            ),
        }
        for layer_id in sorted(prepared.alpha)
    ]
    return {
        "query_total": int(prepared.queries.shape[0]),
        "query_rows_sha256": frozen._array_digest(
            np.ascontiguousarray(prepared.queries)
        ),
        "query_bias_sha256": frozen._array_digest(
            np.ascontiguousarray(prepared.query_bias)
        ),
        "alpha_sha256_by_layer": alpha_entries,
        "alpha_manifest_sha256": _json_sha256(alpha_entries),
    }


def _make_query_span_binding(
    prepared: frozen._Prepared,
    query_start: int,
    query_end: int,
) -> Dict[str, Any]:
    if (
        query_start < 0
        or query_end <= query_start
        or query_end > prepared.queries.shape[0]
    ):
        frozen._fail("V51_QUERY_BINDING", "invalid query span")
    alpha_entries = []
    for layer_id in sorted(prepared.alpha):
        width = prepared.layers[layer_id].width
        block = np.ascontiguousarray(
            frozen._alpha_block(
                prepared,
                layer_id,
                query_start,
                query_end,
                width,
            ),
            dtype=np.float64,
        )
        alpha_entries.append(
            {
                "layer_id": int(layer_id),
                "sha256": frozen._array_digest(block),
            }
        )
    return {
        "query_start": int(query_start),
        "query_end": int(query_end),
        "query_count": int(query_end - query_start),
        "query_rows_sha256": frozen._array_digest(
            np.ascontiguousarray(
                prepared.queries[query_start:query_end]
            )
        ),
        "query_bias_sha256": frozen._array_digest(
            np.ascontiguousarray(
                prepared.query_bias[query_start:query_end]
            )
        ),
        "alpha_slice_sha256": _json_sha256(alpha_entries),
    }


def _query_span_binding(
    context: _V51Context,
    query_start: int,
    query_end: int,
) -> Mapping[str, Any]:
    key = (int(query_start), int(query_end))
    cached = context.query_span_bindings.get(key)
    if cached is not None:
        return cached
    binding = MappingProxyType(
        _make_query_span_binding(
            context.prepared, query_start, query_end
        )
    )
    context.query_span_bindings[key] = binding
    return binding


def _normalise_dense_error(
    error: QueryDualScalarGuardV51Error,
) -> None:
    if error.code == "DEADLINE_EXPIRED":
        raise frozen.QueryDualReplayTimeout(
            "V5.1 Dense helper deadline expired"
        ) from error
    raise frozen.QueryDualReplayError(
        f"V51_DENSE_{error.code}", str(error)
    ) from error


def _active_nonzero_mask(
    active: np.ndarray, guard: np.ndarray
) -> np.ndarray:
    mask = np.asarray(active, dtype=np.bool_).reshape(-1)
    values = np.asarray(guard, dtype=np.float64).reshape(-1)
    if mask.shape != values.shape:
        frozen._fail("V51_GUARD_LEDGER", "guard/activity shape mismatch")
    return np.ascontiguousarray(mask & (values != 0.0))


def _row_local_subtract(
    scalar: np.ndarray,
    guard: np.ndarray,
    active: np.ndarray,
    *,
    where: str,
) -> Tuple[np.ndarray, np.ndarray]:
    applied = _active_nonzero_mask(active, guard)
    output = np.ascontiguousarray(scalar.copy(), dtype=np.float64)
    if np.any(applied):
        output[applied] = frozen._down_add(
            output[applied],
            -np.asarray(guard)[applied],
            where=where,
        )
    frozen._require_finite(output, where=where)
    return output, applied


def _absorb_radius_with_penalty(
    scalar: np.ndarray,
    radius: np.ndarray,
    box: frozen._Box,
    stats: frozen._ReplayStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen-V3 radius absorption with the exact row penalty exposed."""

    if (
        radius.ndim != 2
        or radius.shape[0] != scalar.size
        or radius.shape[1] != box.lb.size
        or np.any(radius < 0.0)
    ):
        frozen._fail(
            "NUMERIC_GUARD", "coefficient radius/box mismatch"
        )
    if not np.any(radius):
        return scalar, np.zeros(scalar.size, dtype=np.float64)
    max_abs = np.maximum(np.abs(box.lb), np.abs(box.ub))
    _, raw_error = frozen._row_dots_with_error(radius, max_abs)
    nominal = np.asarray(radius @ max_abs, dtype=np.float64)
    penalty = frozen._upper_nonnegative_sum(nominal, raw_error)
    zero_rows = ~np.any(
        (radius != 0.0)
        & (max_abs.reshape(1, -1) != 0.0),
        axis=1,
    )
    penalty[zero_rows] = 0.0
    frozen._require_finite(
        penalty, where="coefficient-error box absorption"
    )
    stats.record_guard(penalty, coefficient=True)
    if not np.any(penalty):
        return scalar, penalty
    return (
        frozen._down_add(
            scalar,
            -penalty,
            where="coefficient-error absorption",
        ),
        penalty,
    )


def _observe_execution(
    context: _V51Context,
    record: Mapping[str, Any],
    **arrays: np.ndarray,
) -> None:
    observer = context.execution_observer
    if observer is not None:
        readonly = {}
        for name, value in arrays.items():
            view = np.asarray(value).view()
            view.setflags(write=False)
            readonly[name] = view
        observer(record=record, **readonly)


def _record_execution(
    context: _V51Context,
    *,
    layer_id: int,
    predecessor_id: int,
    operator: str,
    policy: str,
    query_start: int,
    query_end: int,
    coefficient_input: np.ndarray,
    nominal: np.ndarray,
    scalar_guard: Optional[np.ndarray],
    active_mask: np.ndarray,
    fallback_mask: np.ndarray,
    scalar_applied_mask: np.ndarray,
    radius_applied: bool,
    fallback_reasons: Optional[Tuple[Tuple[str, ...], ...]] = None,
    branch: Optional[str] = None,
    nonzero_count: Optional[int] = None,
    dense_count: Optional[int] = None,
    support_sha256: Optional[str] = None,
    catalog_sha256: Optional[str] = None,
    plan_sha256: Optional[str] = None,
    helper_receipt_sha256: Optional[str] = None,
    helper_input_coefficient_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    count = int(query_end - query_start)
    if count <= 0:
        frozen._fail("V51_GUARD_LEDGER", "empty query span")
    active = np.ascontiguousarray(active_mask, dtype=np.bool_).reshape(-1)
    fallback = np.ascontiguousarray(
        fallback_mask, dtype=np.bool_
    ).reshape(-1)
    applied = np.ascontiguousarray(
        scalar_applied_mask, dtype=np.bool_
    ).reshape(-1)
    if (
        active.size != count
        or fallback.size != count
        or applied.size != count
        or np.any(fallback & ~active)
        or np.any(applied & ~active)
    ):
        frozen._fail("V51_GUARD_LEDGER", "invalid row-local masks")
    scalar_policy = int(scalar_guard is not None)
    radius_policy = int(bool(radius_applied))
    if scalar_policy + radius_policy != 1:
        frozen._fail(
            "V51_GUARD_LEDGER",
            f"affine layer {layer_id} needs exactly one guard policy",
        )
    if radius_policy and (np.any(active) or np.any(fallback) or np.any(applied)):
        frozen._fail(
            "V51_GUARD_LEDGER",
            "componentwise branch cannot claim scalar row masks",
        )
    reasons = (
        tuple(() for _ in range(count))
        if fallback_reasons is None
        else tuple(tuple(row) for row in fallback_reasons)
    )
    if (
        len(reasons) != count
        or any(
            row != tuple(sorted(set(row)))
            or any(not isinstance(reason, str) or not reason for reason in row)
            for row in reasons
        )
        or any(bool(reasons[index]) != bool(fallback[index])
               for index in range(count))
    ):
        frozen._fail("V51_GUARD_LEDGER", "invalid fallback reasons")
    span_binding = _query_span_binding(
        context, query_start, query_end
    )
    record: Dict[str, Any] = {
        "execution_index": len(context.executions),
        "layer_id": int(layer_id),
        "predecessor_id": int(predecessor_id),
        "operator": str(operator),
        "policy": str(policy),
        "query_start": int(query_start),
        "query_end": int(query_end),
        "query_count": count,
        "query_block_sha256": context.query_block_sha256,
        "query_rows_sha256": span_binding["query_rows_sha256"],
        "query_bias_sha256": span_binding["query_bias_sha256"],
        "alpha_slice_sha256": span_binding["alpha_slice_sha256"],
        "query_span_binding_sha256": _json_sha256(
            dict(span_binding)
        ),
        "input_coefficient_sha256": frozen._array_digest(
            np.ascontiguousarray(coefficient_input)
        ),
        "nominal_sha256": frozen._array_digest(
            np.ascontiguousarray(nominal)
        ),
        "scalar_guard_policy_count": scalar_policy,
        "componentwise_radius_policy_count": radius_policy,
        "active_mask": [bool(value) for value in active],
        "active_mask_sha256": _bool_digest(active),
        "active_count": int(np.count_nonzero(active)),
        "fallback_mask": [bool(value) for value in fallback],
        "fallback_mask_sha256": _bool_digest(fallback),
        "fallback_count": int(np.count_nonzero(fallback)),
        "fallback_reasons": [list(row) for row in reasons],
        "fallback_reasons_sha256": _json_sha256(reasons),
        "scalar_applied_mask": [bool(value) for value in applied],
        "scalar_applied_mask_sha256": _bool_digest(applied),
        "scalar_subtraction_rows": int(np.count_nonzero(applied)),
    }
    if scalar_guard is not None:
        guard = np.ascontiguousarray(scalar_guard, dtype=np.float64)
        if (
            guard.ndim != 1
            or guard.size != count
            or np.any(guard < 0.0)
            or not np.all(np.isfinite(guard))
            or not np.array_equal(applied, active & (guard != 0.0))
        ):
            frozen._fail("V51_GUARD_LEDGER", "invalid scalar guard")
        record["scalar_guard_sha256"] = frozen._array_digest(guard)
        record["scalar_guard_hex"] = [
            float(value).hex() for value in guard
        ]
    if branch is not None:
        record["conv_branch"] = str(branch)
    if nonzero_count is not None or dense_count is not None:
        if nonzero_count is None or dense_count is None:
            frozen._fail("V51_GUARD_LEDGER", "partial Conv counts")
        record.update(
            {
                "nonzero_count": int(nonzero_count),
                "dense_count": int(dense_count),
                "threshold_lhs": 8 * int(nonzero_count),
                "threshold_rhs": int(dense_count),
            }
        )
    for key, value in (
        ("support_sha256", support_sha256),
        ("catalog_sha256", catalog_sha256),
        ("plan_sha256", plan_sha256),
        ("helper_receipt_sha256", helper_receipt_sha256),
        (
            "helper_input_coefficient_sha256",
            helper_input_coefficient_sha256,
        ),
    ):
        if value is not None:
            record[key] = str(value)
    immutable_record = MappingProxyType(record)
    context.executions.append(immutable_record)
    return immutable_record


def _dense_support(
    context: _V51Context,
    *,
    layer: frozen._FrozenLayer,
    predecessor_id: int,
) -> Tuple[DenseV51Support, np.ndarray]:
    box = frozen._output_box(context.prepared, predecessor_id)
    max_abs = np.ascontiguousarray(
        np.maximum(np.abs(box.lb), np.abs(box.ub)),
        dtype=np.float64,
    )
    cached = context.dense_supports.get(layer.id)
    if cached is not None:
        context.prepared.deadline.check(force=True)
        return cached, max_abs
    try:
        support = prepare_dense_support_v51(
            layer.params["weight"],
            max_abs,
            binding={
                "numeric_protocol": NUMERIC_PROTOCOL,
                "net_sha256": context.prepared.hashes["net_sha256"],
                "bounds_sha256": context.prepared.hashes[
                    "bounds_sha256"
                ],
                "layer_id": str(layer.id),
                "predecessor_id": str(predecessor_id),
                "box_semantics": (
                    "relu_postactivation_from_preactivation_box_v1"
                    if context.prepared.layers[predecessor_id].kind
                    == "RELU"
                    else "output_box"
                ),
            },
            deadline=context.prepared.deadline.end,
        )
    except QueryDualScalarGuardV51Error as error:
        _normalise_dense_error(error)
    context.prepared.deadline.check(force=True)
    context.dense_supports[layer.id] = support
    return support, max_abs


def _conv_plan(
    context: _V51Context,
    *,
    layer: frozen._FrozenLayer,
    predecessor_id: int,
) -> DenseConvV51Plan:
    cached = context.conv_plans.get(layer.id)
    if cached is not None:
        context.prepared.deadline.check(force=True)
        return cached
    plan = prepare_dense_conv_v51_plan(
        layer,
        frozen._output_box(context.prepared, predecessor_id),
        deadline=context.prepared.deadline,
    )
    context.prepared.deadline.check(force=True)
    context.conv_plans[layer.id] = plan
    return plan


def _replay_block_v51(
    context: _V51Context,
    query_start: int,
    query_end: int,
    stats: frozen._ReplayStats,
) -> np.ndarray:
    prepared = context.prepared
    prepared.deadline.check(force=True)
    _query_span_binding(context, query_start, query_end)
    batch = query_end - query_start
    stats.begin_block(query_start, query_end)
    pending: Dict[int, np.ndarray] = {
        prepared.output_id: np.ascontiguousarray(
            prepared.queries[query_start:query_end].copy(),
            dtype=np.float64,
        )
    }
    scalar = np.ascontiguousarray(
        prepared.query_bias[query_start:query_end].copy(),
        dtype=np.float64,
    )
    reached_input = False

    for lid in prepared.reverse_order:
        prepared.deadline.check()
        coefficient = pending.pop(lid, None)
        if coefficient is None:
            continue
        layer = prepared.layers[lid]
        kind = layer.kind
        preds = layer.preds

        if kind == "INPUT_SPEC":
            if lid != prepared.input_spec_id:
                frozen._fail(
                    "INVALID_GRAPH", "unexpected secondary INPUT_SPEC"
                )
            support_value = frozen._input_support_lower(
                coefficient, prepared.bounds[lid], stats
            )
            scalar = frozen._down_add(
                scalar,
                support_value,
                where="V5.1 input support addition",
            )
            reached_input = True
            continue
        if kind == "INPUT":
            frozen._fail("INVALID_GRAPH", "proof path bypassed INPUT_SPEC")

        if kind == "DENSE":
            weight = layer.params["weight"]
            if coefficient.shape != (batch, weight.shape[0]):
                frozen._fail(
                    "SHAPE_MISMATCH", f"DENSE layer {lid} adjoint width"
                )
            bias_contribution = frozen._row_dots_lower(
                coefficient, layer.params["bias"], stats
            )
            scalar = frozen._down_add(
                scalar,
                bias_contribution,
                where=f"V5.1 DENSE {lid} bias",
            )
            pred = preds[0]
            support, max_abs = _dense_support(
                context, layer=layer, predecessor_id=pred
            )
            try:
                guarded = dense_support_compressed_guard_v51(
                    coefficient,
                    weight,
                    max_abs,
                    support,
                    deadline=prepared.deadline.end,
                )
            except QueryDualScalarGuardV51Error as error:
                _normalise_dense_error(error)
            scalar_before_guard = scalar
            scalar_after_guard, applied = _row_local_subtract(
                scalar_before_guard,
                guarded.final_guard,
                guarded.active_mask,
                where=f"V5.1 DENSE {lid} scalar guard",
            )
            stats.record_guard(
                guarded.final_guard, coefficient=True
            )
            scalar = frozen._push_adjoint(
                prepared,
                pending,
                pred,
                guarded.nominal,
                scalar_after_guard,
                stats,
            )
            record = _record_execution(
                context,
                layer_id=lid,
                predecessor_id=pred,
                operator="DENSE",
                policy="v51_wide_or_streamed_scalar_once",
                query_start=query_start,
                query_end=query_end,
                coefficient_input=coefficient,
                nominal=guarded.nominal,
                scalar_guard=guarded.final_guard,
                active_mask=guarded.active_mask,
                fallback_mask=guarded.fallback_mask,
                scalar_applied_mask=applied,
                radius_applied=False,
                fallback_reasons=guarded.fallback_reasons,
                support_sha256=support.support_sha256,
                catalog_sha256=support.diagnostics.sha256,
                helper_receipt_sha256=guarded.diagnostics.sha256,
            )
            _observe_execution(
                context,
                record,
                nominal=guarded.nominal,
                scalar_before=scalar_before_guard,
                scalar_after=scalar_after_guard,
                scalar_guard=guarded.final_guard,
            )
            stats.affine_terms += batch * int(weight.size)
            continue

        if kind == "CONV2D":
            p = layer.params
            out_c, out_h, out_w = p["output_shape"]
            repeated_bias = np.repeat(
                p["bias_channels"], out_h * out_w
            )
            bias_contribution = frozen._row_dots_lower(
                coefficient, repeated_bias, stats
            )
            scalar = frozen._down_add(
                scalar,
                bias_contribution,
                where=f"V5.1 CONV2D {lid} bias",
            )
            pred = preds[0]
            nonzero_count = int(np.count_nonzero(coefficient))
            dense_count = int(coefficient.size)
            if nonzero_count * 8 <= dense_count:
                new_coefficient, radius = frozen._conv_reverse_with_error(
                    coefficient, layer, prepared.deadline, stats
                )
                predecessor_box = frozen._output_box(prepared, pred)
                scalar_before_guard = scalar
                scalar_after_guard, penalty = (
                    _absorb_radius_with_penalty(
                        scalar_before_guard,
                        radius,
                        predecessor_box,
                        stats,
                    )
                )
                scalar = scalar_after_guard
                false_mask = np.zeros(batch, dtype=np.bool_)
                record = _record_execution(
                    context,
                    layer_id=lid,
                    predecessor_id=pred,
                    operator="CONV2D",
                    policy="frozen_v3_componentwise",
                    query_start=query_start,
                    query_end=query_end,
                    coefficient_input=coefficient,
                    nominal=new_coefficient,
                    scalar_guard=None,
                    active_mask=false_mask,
                    fallback_mask=false_mask,
                    scalar_applied_mask=false_mask,
                    radius_applied=True,
                    branch="sparse",
                    nonzero_count=nonzero_count,
                    dense_count=dense_count,
                )
                _observe_execution(
                    context,
                    record,
                    nominal=new_coefficient,
                    scalar_before=scalar_before_guard,
                    scalar_after=scalar_after_guard,
                    componentwise_radius=radius,
                    componentwise_penalty=penalty,
                )
            else:
                plan = _conv_plan(
                    context, layer=layer, predecessor_id=pred
                )
                guarded = replay_dense_conv_v51(
                    coefficient,
                    plan,
                    deadline=prepared.deadline,
                )
                new_coefficient = guarded.coefficient
                scalar_before_guard = scalar
                scalar_after_guard, applied = _row_local_subtract(
                    scalar_before_guard,
                    guarded.scalar_guard,
                    guarded.active_mask,
                    where=f"V5.1 CONV2D {lid} scalar guard",
                )
                scalar = scalar_after_guard
                stats.record_guard(
                    guarded.scalar_guard, coefficient=True
                )
                stats.conv_dense_blocks += 1
                false_fallback = np.zeros(batch, dtype=np.bool_)
                record = _record_execution(
                    context,
                    layer_id=lid,
                    predecessor_id=pred,
                    operator="CONV2D",
                    policy="v51_dense_conv_D_plus_A_once",
                    query_start=query_start,
                    query_end=query_end,
                    coefficient_input=coefficient,
                    nominal=new_coefficient,
                    scalar_guard=guarded.scalar_guard,
                    active_mask=guarded.active_mask,
                    fallback_mask=false_fallback,
                    scalar_applied_mask=applied,
                    radius_applied=False,
                    branch="dense",
                    nonzero_count=nonzero_count,
                    dense_count=dense_count,
                    support_sha256=frozen._array_digest(plan.support),
                    plan_sha256=str(
                        plan.manifest["content_sha256"]
                    ),
                    helper_receipt_sha256=str(
                        guarded.receipt["content_sha256"]
                    ),
                    helper_input_coefficient_sha256=str(
                        guarded.receipt[
                            "coefficient_input_sha256"
                        ]
                    ),
                )
                _observe_execution(
                    context,
                    record,
                    nominal=new_coefficient,
                    scalar_before=scalar_before_guard,
                    scalar_after=scalar_after_guard,
                    scalar_guard=guarded.scalar_guard,
                )
            if new_coefficient.shape[1] != prepared.layers[pred].width:
                frozen._fail(
                    "SHAPE_MISMATCH",
                    f"CONV2D layer {lid} input width",
                )
            scalar = frozen._push_adjoint(
                prepared,
                pending,
                pred,
                new_coefficient,
                scalar,
                stats,
            )
            stats.affine_terms += (
                batch * int(p["weight"].size) * out_h * out_w
            )
            continue

        if kind == "FLATTEN":
            pred = preds[0]
            if coefficient.shape[1] != prepared.layers[pred].width:
                frozen._fail(
                    "SHAPE_MISMATCH",
                    f"FLATTEN layer {lid} is not size preserving",
                )
            scalar = frozen._push_adjoint(
                prepared,
                pending,
                pred,
                coefficient.copy(),
                scalar,
                stats,
            )
            continue

        if kind == "ADD":
            bias_contribution = frozen._row_dots_lower(
                coefficient, layer.params["bias"], stats
            )
            scalar = frozen._down_add(
                scalar,
                bias_contribution,
                where=f"V5.1 ADD {lid} bias",
            )
            for pred in preds:
                if prepared.layers[pred].width != coefficient.shape[1]:
                    frozen._fail(
                        "SHAPE_MISMATCH",
                        f"ADD layer {lid} predecessor width",
                    )
                scalar = frozen._push_adjoint(
                    prepared,
                    pending,
                    pred,
                    coefficient.copy(),
                    scalar,
                    stats,
                )
            continue

        if kind == "RELU":
            box = prepared.bounds[lid]
            if coefficient.shape[1] != box.lb.size:
                frozen._fail(
                    "SHAPE_MISMATCH",
                    f"RELU layer {lid} adjoint width",
                )
            ambiguous = (box.lb < 0.0) & (box.ub > 0.0)
            off = box.ub <= 0.0
            on = (box.lb >= 0.0) & ~off
            alpha = frozen._alpha_block(
                prepared,
                lid,
                query_start,
                query_end,
                coefficient.shape[1],
            )
            factor = np.zeros_like(coefficient)
            factor[:, on] = 1.0
            positive_ambiguous = (
                ambiguous.reshape(1, -1) & (coefficient >= 0.0)
            )
            negative_ambiguous = (
                ambiguous.reshape(1, -1) & (coefficient < 0.0)
            )
            slope, beta, _ = frozen._relu_lines(
                prepared,
                lid,
                stats,
                required_mask=np.any(negative_ambiguous, axis=0),
            )
            factor[positive_ambiguous] = alpha[positive_ambiguous]
            slope_block = np.broadcast_to(
                slope.reshape(1, -1), coefficient.shape
            )
            factor[negative_ambiguous] = slope_block[
                negative_ambiguous
            ]
            new_coefficient, radius = (
                frozen._elementwise_product_with_error(
                    coefficient, factor
                )
            )
            intercept = np.zeros_like(coefficient)
            beta_block = np.broadcast_to(
                beta.reshape(1, -1), coefficient.shape
            )
            intercept[negative_ambiguous] = beta_block[
                negative_ambiguous
            ]
            intercept_contribution = frozen._row_dots_lower(
                coefficient, intercept, stats
            )
            scalar = frozen._down_add(
                scalar,
                intercept_contribution,
                where=f"V5.1 RELU {lid} intercept",
            )
            pred = preds[0]
            scalar = frozen._absorb_radius(
                scalar, radius, box, stats
            )
            scalar = frozen._push_adjoint(
                prepared,
                pending,
                pred,
                new_coefficient,
                scalar,
                stats,
            )
            stats.relu_ambiguous_terms += (
                batch * int(np.count_nonzero(ambiguous))
            )
            continue

        frozen._fail(
            "UNSUPPORTED_OPERATOR",
            f"unhandled layer {lid} kind {kind}",
        )

    if pending:
        frozen._fail(
            "INVALID_GRAPH",
            f"unprocessed adjoints remain at {sorted(pending)}",
        )
    if not reached_input:
        frozen._fail("INVALID_GRAPH", "query does not reach INPUT_SPEC")
    scalar = np.nextafter(scalar, -math.inf)
    frozen._require_finite(scalar, where="V5.1 final lower bound")
    return np.ascontiguousarray(scalar)


def _stats_record(stats: frozen._ReplayStats) -> Mapping[str, Any]:
    guards = (
        np.zeros(0, dtype=np.float64)
        if stats.guard_by_query is None
        else stats.guard_by_query
    )
    return {
        "affine_terms": int(stats.affine_terms),
        "coefficient_guards": int(stats.coefficient_guards),
        "scalar_guards": int(stats.scalar_guards),
        "fraction_endpoint_audits": int(
            stats.fraction_endpoint_audits
        ),
        "relu_ambiguous_terms": int(stats.relu_ambiguous_terms),
        "dag_merges": int(stats.dag_merges),
        "conv_sparse_blocks": int(stats.conv_sparse_blocks),
        "conv_dense_blocks": int(stats.conv_dense_blocks),
        "guard_total_hex": float(stats.guard_total).hex(),
        "guard_max_hex": float(stats.guard_max).hex(),
        "guard_by_query_sha256": frozen._array_digest(guards),
        "guard_by_query_hex": [
            float(value).hex() for value in guards
        ],
    }


def _span_manifest(
    records: List[Mapping[str, Any]],
    query_count: int,
) -> List[Mapping[str, Any]]:
    by_layer: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
    for record in records:
        key = (int(record["layer_id"]), str(record["operator"]))
        by_layer.setdefault(key, []).append(
            (int(record["query_start"]), int(record["query_end"]))
        )
    result: List[Mapping[str, Any]] = []
    for (layer_id, operator), spans in sorted(by_layer.items()):
        cursor = 0
        for start, end in spans:
            if start != cursor or end <= start or end > query_count:
                frozen._fail(
                    "V51_GUARD_LEDGER",
                    f"layer {layer_id} query spans are incomplete",
                )
            cursor = end
        if cursor != query_count:
            frozen._fail(
                "V51_GUARD_LEDGER",
                f"layer {layer_id} query coverage ended at {cursor}",
            )
        result.append(
            {
                "layer_id": layer_id,
                "operator": operator,
                "spans": [[start, end] for start, end in spans],
            }
        )
    return result


def _query_span_bindings_manifest(
    context: _V51Context,
    query_count: int,
) -> List[Mapping[str, Any]]:
    result = [
        dict(context.query_span_bindings[key])
        for key in sorted(context.query_span_bindings)
    ]
    cursor = 0
    for binding in result:
        start = int(binding["query_start"])
        end = int(binding["query_end"])
        if (
            start != cursor
            or end <= start
            or end > query_count
            or int(binding["query_count"]) != end - start
        ):
            frozen._fail(
                "V51_QUERY_BINDING",
                "query span bindings are not a continuous partition",
            )
        cursor = end
    if cursor != query_count:
        frozen._fail(
            "V51_QUERY_BINDING",
            f"query span bindings ended at {cursor}",
        )
    return result


def replay_query_lower_bounds_v51_candidate(
    net: Any,
    certified_bounds: Mapping[Any, Any],
    *,
    start_lid: Optional[int] = None,
    query_rows: Optional[Any] = None,
    one_hot: Optional[Any] = None,
    query_bias: Optional[Any] = None,
    alpha_by_relu: Optional[Mapping[Any, Any]] = None,
    expected_net_sha256: Optional[str] = None,
    expected_bounds_sha256: Optional[str] = None,
    expected_query_sha256: Optional[str] = None,
    expected_alpha_sha256: Optional[str] = None,
    chunk_size: int = 1024,
    max_workspace_bytes: int = 512 * 1024 * 1024,
    deadline: Optional[float] = None,
    timeout_s: Optional[float] = None,
) -> QueryDualReplayV51CandidateResult:
    """Run the complete V5.1 research replay without proof authority."""

    started = time.monotonic()
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        frozen._fail(
            "INVALID_CHUNK", "chunk_size must be a positive integer"
        )
    if (
        isinstance(max_workspace_bytes, bool)
        or not isinstance(max_workspace_bytes, int)
        or max_workspace_bytes <= 0
    ):
        frozen._fail(
            "INVALID_CHUNK",
            "max_workspace_bytes must be a positive integer",
        )
    timer = frozen._Deadline.build(deadline, timeout_s)
    timer.check(force=True)
    frozen_platform = frozen._check_numeric_platform()
    try:
        scalar_platform = check_v51_platform()
    except QueryDualScalarGuardV51Error as error:
        _normalise_dense_error(error)
    timer.check(force=True)
    prepared = frozen._prepare(
        net,
        certified_bounds,
        start_lid=start_lid,
        query_rows=query_rows,
        one_hot=one_hot,
        query_bias=query_bias,
        alpha_by_relu=alpha_by_relu,
        deadline=timer,
        expected_net_sha256=expected_net_sha256,
        expected_bounds_sha256=expected_bounds_sha256,
        expected_query_sha256=expected_query_sha256,
        expected_alpha_sha256=expected_alpha_sha256,
    )
    maximum_width = max(
        layer.width for layer in prepared.layers.values()
    )
    bytes_per_query = max(1, maximum_width * 8 * 12)
    memory_limited = max(
        1, max_workspace_bytes // bytes_per_query
    )
    effective_chunk_size = min(
        chunk_size,
        memory_limited,
        prepared.queries.shape[0],
    )
    stats = frozen._ReplayStats()
    stats.configure_queries(prepared.queries.shape[0])
    values = np.empty(prepared.queries.shape[0], dtype=np.float64)
    context = _V51Context(prepared=prepared)
    for chunk_start in range(
        0, values.size, effective_chunk_size
    ):
        timer.check(force=True)
        chunk_end = min(
            values.size, chunk_start + effective_chunk_size
        )
        values[chunk_start:chunk_end] = _replay_block_v51(
            context, chunk_start, chunk_end, stats
        )
    timer.check(force=True)
    if not np.all(np.isfinite(values)):
        frozen._fail(
            "NONFINITE", "non-finite V5.1 candidate lower bounds"
        )
    immutable_values = frozen._immutable_f64_array(
        values, name="V5.1 candidate lower bounds"
    )
    execution_records = [
        dict(record) for record in context.executions
    ]
    spans = _span_manifest(execution_records, immutable_values.size)
    query_span_bindings = _query_span_bindings_manifest(
        context, immutable_values.size
    )
    body: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "experimental_candidate",
        "proof_authority": False,
        "semantic_authority": False,
        "integrity_scope": "unkeyed_internal_consistency_only",
        "numeric_protocol": NUMERIC_PROTOCOL,
        "authority_integration_complete": False,
        "trusted_assumption": "supplied_bounds_are_certified",
        "hashes": dict(prepared.hashes),
        "output_layer_id": int(prepared.output_id),
        "query_count": int(immutable_values.size),
        "query_block_manifest": dict(
            context.query_block_manifest
        ),
        "query_block_sha256": context.query_block_sha256,
        "query_span_bindings": query_span_bindings,
        "query_span_bindings_sha256": _json_sha256(
            query_span_bindings
        ),
        "requested_chunk_size": int(chunk_size),
        "effective_chunk_size": int(effective_chunk_size),
        "max_workspace_bytes": int(max_workspace_bytes),
        "lower_bounds_sha256": frozen._array_digest(
            immutable_values
        ),
        "lower_bounds_hex": [
            float(value).hex() for value in immutable_values
        ],
        "stats": _stats_record(stats),
        "affine_execution_count": len(execution_records),
        "affine_executions": execution_records,
        "guard_ledger_sha256": _json_sha256(execution_records),
        "execution_span_manifest": spans,
        "execution_span_manifest_sha256": _json_sha256(spans),
        "dense_support_count": len(context.dense_supports),
        "conv_plan_count": len(context.conv_plans),
        "numeric_platform": {
            "frozen_v3": dict(frozen_platform),
            "v51_dense_sha256": scalar_platform.sha256,
            "v51_dense": dict(scalar_platform.items),
        },
        "elapsed_s_hex": float(
            time.monotonic() - started
        ).hex(),
    }
    body["receipt_sha256"] = _json_sha256(body)
    timer.check(force=True)
    return QueryDualReplayV51CandidateResult(
        lower_bounds=immutable_values,
        receipt=MappingProxyType(body),
    )


def _verify_execution_record(
    record: Mapping[str, Any],
    *,
    index: int,
    query_count: int,
    query_block_sha256: str,
    span_binding: Mapping[str, Any],
    output_layer_id: int,
) -> bool:
    start = int(record["query_start"])
    end = int(record["query_end"])
    count = end - start
    active = np.ascontiguousarray(
        record["active_mask"], dtype=np.bool_
    )
    fallback = np.ascontiguousarray(
        record["fallback_mask"], dtype=np.bool_
    )
    applied = np.ascontiguousarray(
        record["scalar_applied_mask"], dtype=np.bool_
    )
    scalar_policy = int(record["scalar_guard_policy_count"])
    radius_policy = int(record["componentwise_radius_policy_count"])
    reasons = tuple(
        tuple(row) for row in record["fallback_reasons"]
    )
    input_coefficient_sha256 = record[
        "input_coefficient_sha256"
    ]
    if (
        int(record["execution_index"]) != index
        or start < 0
        or end <= start
        or end > query_count
        or int(record["query_count"]) != count
        or active.shape != (count,)
        or fallback.shape != (count,)
        or applied.shape != (count,)
        or record["active_mask_sha256"] != _bool_digest(active)
        or record["fallback_mask_sha256"] != _bool_digest(fallback)
        or record["scalar_applied_mask_sha256"]
        != _bool_digest(applied)
        or int(record["active_count"]) != int(np.count_nonzero(active))
        or int(record["fallback_count"])
        != int(np.count_nonzero(fallback))
        or len(reasons) != count
        or record["fallback_reasons_sha256"] != _json_sha256(reasons)
        or any(
            row != tuple(sorted(set(row)))
            or any(
                not isinstance(reason, str) or not reason
                for reason in row
            )
            for row in reasons
        )
        or any(
            bool(reasons[row]) != bool(fallback[row])
            for row in range(count)
        )
        or int(record["scalar_subtraction_rows"])
        != int(np.count_nonzero(applied))
        or np.any(fallback & ~active)
        or np.any(applied & ~active)
        or scalar_policy not in {0, 1}
        or radius_policy not in {0, 1}
        or scalar_policy + radius_policy != 1
        or record["query_block_sha256"]
        != query_block_sha256
        or record["query_rows_sha256"]
        != span_binding["query_rows_sha256"]
        or record["query_bias_sha256"]
        != span_binding["query_bias_sha256"]
        or record["alpha_slice_sha256"]
        != span_binding["alpha_slice_sha256"]
        or record["query_span_binding_sha256"]
        != _json_sha256(dict(span_binding))
        or not _is_sha256(input_coefficient_sha256)
        or not _is_sha256(record["nominal_sha256"])
    ):
        return False
    helper_input = record.get(
        "helper_input_coefficient_sha256"
    )
    if helper_input is not None and (
        not _is_sha256(helper_input)
        or not hmac.compare_digest(
            helper_input, input_coefficient_sha256
        )
    ):
        return False
    operator = record.get("operator")
    policy = record.get("policy")
    if operator == "DENSE":
        if (
            policy != "v51_wide_or_streamed_scalar_once"
            or "conv_branch" in record
            or any(
                not _is_sha256(record.get(field))
                for field in (
                    "support_sha256",
                    "catalog_sha256",
                    "helper_receipt_sha256",
                )
            )
            or any(
                field in record
                for field in (
                    "plan_sha256",
                    "nonzero_count",
                    "dense_count",
                    "threshold_lhs",
                    "threshold_rhs",
                )
            )
        ):
            return False
    elif operator == "CONV2D":
        branch = record.get("conv_branch")
        if branch == "dense":
            if (
                policy != "v51_dense_conv_D_plus_A_once"
                or any(
                    not _is_sha256(record.get(field))
                    for field in (
                        "support_sha256",
                        "plan_sha256",
                        "helper_receipt_sha256",
                        "helper_input_coefficient_sha256",
                    )
                )
            ):
                return False
        elif branch == "sparse":
            if (
                policy != "frozen_v3_componentwise"
                or any(
                    field in record
                    for field in (
                        "support_sha256",
                        "catalog_sha256",
                        "plan_sha256",
                        "helper_receipt_sha256",
                        "helper_input_coefficient_sha256",
                        "scalar_guard_sha256",
                        "scalar_guard_hex",
                    )
                )
            ):
                return False
        else:
            return False
    else:
        return False
    if (
        int(record["layer_id"]) == output_layer_id
        and not hmac.compare_digest(
            input_coefficient_sha256,
            str(span_binding["query_rows_sha256"]),
        )
    ):
        return False
    if scalar_policy:
        guard = np.asarray(
            [
                float.fromhex(value)
                for value in record["scalar_guard_hex"]
            ],
            dtype=np.float64,
        )
        if (
            guard.shape != (count,)
            or np.any(guard < 0.0)
            or not np.all(np.isfinite(guard))
            or record["scalar_guard_sha256"]
            != frozen._array_digest(guard)
            or not np.array_equal(applied, active & (guard != 0.0))
        ):
            return False
    else:
        if np.any(active) or np.any(fallback) or np.any(applied):
            return False
    if "conv_branch" in record:
        lhs = int(record["threshold_lhs"])
        rhs = int(record["threshold_rhs"])
        if (
            lhs != 8 * int(record["nonzero_count"])
            or rhs != int(record["dense_count"])
            or (
                record["conv_branch"] == "sparse"
                and lhs > rhs
            )
            or (
                record["conv_branch"] == "dense"
                and lhs <= rhs
            )
            or record["conv_branch"] not in {"sparse", "dense"}
        ):
            return False
    return True


def verify_query_dual_replay_v51_candidate(
    result: QueryDualReplayV51CandidateResult,
) -> bool:
    """Check unkeyed internal consistency while denying semantic authority.

    This verifier cannot authenticate a coordinated rewrite of both payload
    digests and their enclosing receipt.  The root-owned V5.1 session instead
    reconstructs these bindings from its sealed graph and live observer.
    """

    try:
        if (
            not isinstance(
                result, QueryDualReplayV51CandidateResult
            )
            or result.proof_authority is not False
        ):
            return False
        body = dict(result.receipt)
        claimed = str(body.pop("receipt_sha256"))
        values = np.asarray(result.lower_bounds)
        records = body["affine_executions"]
        query_count = int(body["query_count"])
        output_layer_id = int(body["output_layer_id"])
        query_block_manifest = body["query_block_manifest"]
        query_block_sha256 = body["query_block_sha256"]
        alpha_entries = query_block_manifest[
            "alpha_sha256_by_layer"
        ]
        query_span_bindings = body["query_span_bindings"]
        if (
            body.get("schema") != SCHEMA
            or body.get("status") != "experimental_candidate"
            or body.get("proof_authority") is not False
            or body.get("semantic_authority") is not False
            or body.get("integrity_scope")
            != "unkeyed_internal_consistency_only"
            or body.get("authority_integration_complete") is not False
            or body.get("numeric_protocol") != NUMERIC_PROTOCOL
            or not isinstance(records, list)
            or int(body["affine_execution_count"]) != len(records)
            or body["guard_ledger_sha256"]
            != _json_sha256(records)
            or not hmac.compare_digest(
                _json_sha256(body), claimed
            )
            or values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
            or query_count != values.size
            or body["lower_bounds_sha256"]
            != frozen._array_digest(values)
            or body["lower_bounds_hex"]
            != [float(value).hex() for value in values]
            or not isinstance(query_block_manifest, dict)
            or int(query_block_manifest["query_total"])
            != query_count
            or not _is_sha256(
                query_block_manifest["query_rows_sha256"]
            )
            or not _is_sha256(
                query_block_manifest["query_bias_sha256"]
            )
            or not isinstance(alpha_entries, list)
            or [
                int(entry["layer_id"]) for entry in alpha_entries
            ]
            != sorted(
                {
                    int(entry["layer_id"])
                    for entry in alpha_entries
                }
            )
            or any(
                not _is_sha256(entry["sha256"])
                for entry in alpha_entries
            )
            or query_block_manifest["alpha_manifest_sha256"]
            != _json_sha256(alpha_entries)
            or not _is_sha256(query_block_sha256)
            or query_block_sha256
            != _json_sha256(query_block_manifest)
            or not isinstance(query_span_bindings, list)
            or body["query_span_bindings_sha256"]
            != _json_sha256(query_span_bindings)
        ):
            return False
        span_by_key: Dict[
            Tuple[int, int], Mapping[str, Any]
        ] = {}
        cursor = 0
        required_binding_keys = {
            "query_start",
            "query_end",
            "query_count",
            "query_rows_sha256",
            "query_bias_sha256",
            "alpha_slice_sha256",
        }
        for binding in query_span_bindings:
            if (
                not isinstance(binding, dict)
                or set(binding) != required_binding_keys
            ):
                return False
            start = int(binding["query_start"])
            end = int(binding["query_end"])
            key = (start, end)
            if (
                start != cursor
                or end <= start
                or end > query_count
                or int(binding["query_count"]) != end - start
                or key in span_by_key
                or any(
                    not _is_sha256(binding[field])
                    for field in (
                        "query_rows_sha256",
                        "query_bias_sha256",
                        "alpha_slice_sha256",
                    )
                )
            ):
                return False
            span_by_key[key] = binding
            cursor = end
        if cursor != query_count:
            return False
        if len(query_span_bindings) == 1:
            only = query_span_bindings[0]
            if (
                only["query_rows_sha256"]
                != query_block_manifest["query_rows_sha256"]
                or only["query_bias_sha256"]
                != query_block_manifest["query_bias_sha256"]
            ):
                return False
        for index, record in enumerate(records):
            key = (
                int(record["query_start"]),
                int(record["query_end"]),
            )
            binding = span_by_key.get(key)
            if binding is None:
                return False
            if not _verify_execution_record(
                record,
                index=index,
                query_count=query_count,
                query_block_sha256=query_block_sha256,
                span_binding=binding,
                output_layer_id=output_layer_id,
            ):
                return False
        spans = _span_manifest(records, query_count)
        if (
            body["execution_span_manifest"] != spans
            or body["execution_span_manifest_sha256"]
            != _json_sha256(spans)
        ):
            return False
        return True
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        frozen.QueryDualReplayError,
    ):
        return False


__all__ = [
    "NUMERIC_PROTOCOL",
    "QueryDualReplayV51CandidateResult",
    "replay_query_lower_bounds_v51_candidate",
    "verify_query_dual_replay_v51_candidate",
]
