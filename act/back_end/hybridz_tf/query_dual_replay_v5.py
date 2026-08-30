#===- query_dual_replay_v5.py - Experimental scalar-guard replay ------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
"""Isolated V5 replay candidate with support-compressed affine guards.

This module deliberately does not modify or wrap the sealed V3 authority
session.  It reuses the frozen parser, graph snapshot, ReLU audit, DAG merge,
and scalar-dot machinery, but dispatches Dense and *dense* Conv2D affine
roundoff through experimental scalar guards.  Sparse Conv2D keeps the frozen
componentwise-radius path.

The result is useful only for controlled soundness/tightness/performance
gates.  It can never carry proof authority.  A separate schema, guarded
session, commit ledger, and public validator are required before production
use.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as frozen
from act.back_end.hybridz_tf.query_dual_replay_v5_candidate import (
    DenseConvScalarGuardPlan,
    prepare_dense_conv_scalar_guard,
    replay_dense_conv_scalar_guard,
)
from act.back_end.hybridz_tf.query_dual_scalar_guard import (
    DenseSupport,
    check_scalar_guard_platform,
    dense_support_compressed_guard,
    prepare_dense_support,
)


SCHEMA = "act.query_dual_replay_v5_candidate.v1"
NUMERIC_PROTOCOL = "scalar_compressed_affine_roundoff_v5"


@dataclass(frozen=True)
class QueryDualReplayV5CandidateResult:
    """Integrity-checkable research output which explicitly has no authority."""

    lower_bounds: np.ndarray = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    proof_authority: bool = False

    def __post_init__(self) -> None:
        if self.proof_authority:
            raise ValueError("the isolated V5 candidate cannot issue authority")
        values = np.asarray(self.lower_bounds)
        if (
            values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("V5 candidate lower bounds must be immutable finite f64")


@dataclass
class _V5Context:
    prepared: frozen._Prepared
    dense_supports: Dict[int, DenseSupport] = field(default_factory=dict)
    conv_plans: Dict[int, DenseConvScalarGuardPlan] = field(default_factory=dict)
    executions: List[Mapping[str, Any]] = field(default_factory=list)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _record_execution(
    context: _V5Context,
    *,
    layer_id: int,
    predecessor_id: int,
    operator: str,
    policy: str,
    query_start: int,
    query_end: int,
    nominal: np.ndarray,
    scalar_guard: Optional[np.ndarray],
    radius_applied: bool,
    branch: Optional[str] = None,
    nonzero_count: Optional[int] = None,
    dense_count: Optional[int] = None,
    support_sha256: Optional[str] = None,
) -> None:
    scalar_count = int(scalar_guard is not None)
    radius_count = int(bool(radius_applied))
    if scalar_count + radius_count != 1 or scalar_count * radius_count != 0:
        frozen._fail(
            "V5_GUARD_LEDGER",
            f"affine layer {layer_id} must consume exactly one guard policy",
        )
    record: Dict[str, Any] = {
        "execution_index": len(context.executions),
        "layer_id": int(layer_id),
        "predecessor_id": int(predecessor_id),
        "operator": str(operator),
        "policy": str(policy),
        "query_start": int(query_start),
        "query_end": int(query_end),
        "query_count": int(query_end - query_start),
        "nominal_sha256": frozen._array_digest(nominal),
        "scalar_guard_applied_count": scalar_count,
        "componentwise_radius_applied_count": radius_count,
    }
    if scalar_guard is not None:
        record["scalar_guard_sha256"] = frozen._array_digest(scalar_guard)
        record["scalar_guard_hex"] = [
            float(value).hex() for value in scalar_guard
        ]
    if branch is not None:
        record["conv_branch"] = str(branch)
    if nonzero_count is not None or dense_count is not None:
        if nonzero_count is None or dense_count is None:
            frozen._fail("V5_GUARD_LEDGER", "partial Conv branch counts")
        record.update(
            {
                "nonzero_count": int(nonzero_count),
                "dense_count": int(dense_count),
                "threshold_lhs": 8 * int(nonzero_count),
                "threshold_rhs": int(dense_count),
            }
        )
    if support_sha256 is not None:
        record["support_sha256"] = str(support_sha256)
    context.executions.append(MappingProxyType(record))


def _dense_support(
    context: _V5Context,
    *,
    layer: frozen._FrozenLayer,
    predecessor_id: int,
) -> DenseSupport:
    cached = context.dense_supports.get(layer.id)
    if cached is not None:
        context.prepared.deadline.check(force=True)
        return cached
    box = frozen._output_box(context.prepared, predecessor_id)
    max_abs = np.ascontiguousarray(
        np.maximum(np.abs(box.lb), np.abs(box.ub)), dtype=np.float64
    )
    support = prepare_dense_support(
        layer.params["weight"],
        max_abs,
        binding={
            "numeric_protocol": NUMERIC_PROTOCOL,
            "net_sha256": context.prepared.hashes["net_sha256"],
            "bounds_sha256": context.prepared.hashes["bounds_sha256"],
            "layer_id": str(layer.id),
            "predecessor_id": str(predecessor_id),
            "box_semantics": (
                "relu_postactivation_from_preactivation_box_v1"
                if context.prepared.layers[predecessor_id].kind == "RELU"
                else "output_box"
            ),
        },
        deadline=context.prepared.deadline.end,
    )
    context.prepared.deadline.check(force=True)
    context.dense_supports[layer.id] = support
    return support


def _conv_plan(
    context: _V5Context,
    *,
    layer: frozen._FrozenLayer,
    predecessor_id: int,
) -> DenseConvScalarGuardPlan:
    cached = context.conv_plans.get(layer.id)
    if cached is not None:
        context.prepared.deadline.check(force=True)
        return cached
    plan = prepare_dense_conv_scalar_guard(
        layer,
        frozen._output_box(context.prepared, predecessor_id),
        deadline=context.prepared.deadline,
    )
    context.prepared.deadline.check(force=True)
    context.conv_plans[layer.id] = plan
    return plan


def _replay_block_v5(
    context: _V5Context,
    query_start: int,
    query_end: int,
    stats: frozen._ReplayStats,
) -> np.ndarray:
    """Replay one objective block; only affine guard dispatch differs from V3."""

    prepared = context.prepared
    prepared.deadline.check(force=True)
    batch = query_end - query_start
    stats.begin_block(query_start, query_end)
    pending: Dict[int, np.ndarray] = {
        prepared.output_id: np.ascontiguousarray(
            prepared.queries[query_start:query_end].copy(), dtype=np.float64
        )
    }
    scalar = np.ascontiguousarray(
        prepared.query_bias[query_start:query_end].copy(), dtype=np.float64
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
                frozen._fail("INVALID_GRAPH", "unexpected secondary INPUT_SPEC")
            support = frozen._input_support_lower(
                coefficient, prepared.bounds[lid], stats
            )
            scalar = frozen._down_add(
                scalar, support, where="V5 input support addition"
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
                scalar, bias_contribution, where=f"V5 DENSE {lid} bias"
            )
            pred = preds[0]
            support = _dense_support(
                context, layer=layer, predecessor_id=pred
            )
            guarded = dense_support_compressed_guard(
                coefficient,
                weight,
                support,
                deadline=prepared.deadline.end,
            )
            # Preserve the frozen path's exact-zero behavior.  Calling
            # ``_down_add`` with an identically zero guard would introduce a
            # gratuitous one-ULP decrease even though no roundoff mass exists.
            if np.any(guarded.scalar_guard):
                scalar = frozen._down_add(
                    scalar,
                    -guarded.scalar_guard,
                    where=f"V5 DENSE {lid} scalar roundoff absorption",
                )
            stats.record_guard(guarded.scalar_guard, coefficient=True)
            scalar = frozen._push_adjoint(
                prepared,
                pending,
                pred,
                guarded.nominal,
                scalar,
                stats,
            )
            _record_execution(
                context,
                layer_id=lid,
                predecessor_id=pred,
                operator="DENSE",
                policy="scalar_compressed_once",
                query_start=query_start,
                query_end=query_end,
                nominal=guarded.nominal,
                scalar_guard=guarded.scalar_guard,
                radius_applied=False,
                support_sha256=support.support_sha256,
            )
            stats.affine_terms += batch * int(weight.size)
            continue
        if kind == "CONV2D":
            p = layer.params
            out_c, out_h, out_w = p["output_shape"]
            repeated_bias = np.repeat(p["bias_channels"], out_h * out_w)
            bias_contribution = frozen._row_dots_lower(
                coefficient, repeated_bias, stats
            )
            scalar = frozen._down_add(
                scalar, bias_contribution, where=f"V5 CONV2D {lid} bias"
            )
            pred = preds[0]
            nonzero_count = int(np.count_nonzero(coefficient))
            dense_count = int(coefficient.size)
            if nonzero_count * 8 <= dense_count:
                new_coefficient, radius = frozen._conv_reverse_with_error(
                    coefficient, layer, prepared.deadline, stats
                )
                scalar = frozen._absorb_radius(
                    scalar,
                    radius,
                    frozen._output_box(prepared, pred),
                    stats,
                )
                _record_execution(
                    context,
                    layer_id=lid,
                    predecessor_id=pred,
                    operator="CONV2D",
                    policy="componentwise_v3",
                    query_start=query_start,
                    query_end=query_end,
                    nominal=new_coefficient,
                    scalar_guard=None,
                    radius_applied=True,
                    branch="sparse",
                    nonzero_count=nonzero_count,
                    dense_count=dense_count,
                )
            else:
                guarded = replay_dense_conv_scalar_guard(
                    coefficient,
                    _conv_plan(
                        context, layer=layer, predecessor_id=pred
                    ),
                    deadline=prepared.deadline,
                )
                new_coefficient = guarded.coefficient
                if np.any(guarded.scalar_guard):
                    scalar = frozen._down_add(
                        scalar,
                        -guarded.scalar_guard,
                        where=f"V5 CONV2D {lid} scalar roundoff absorption",
                    )
                stats.record_guard(guarded.scalar_guard, coefficient=True)
                stats.conv_dense_blocks += 1
                _record_execution(
                    context,
                    layer_id=lid,
                    predecessor_id=pred,
                    operator="CONV2D",
                    policy="scalar_conv_v5",
                    query_start=query_start,
                    query_end=query_end,
                    nominal=new_coefficient,
                    scalar_guard=guarded.scalar_guard,
                    radius_applied=False,
                    branch="dense",
                    nonzero_count=nonzero_count,
                    dense_count=dense_count,
                    support_sha256=frozen._array_digest(
                        _conv_plan(
                            context, layer=layer, predecessor_id=pred
                        ).support
                    ),
                )
            if new_coefficient.shape[1] != prepared.layers[pred].width:
                frozen._fail(
                    "SHAPE_MISMATCH", f"CONV2D layer {lid} input width"
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
                prepared, pending, pred, coefficient.copy(), scalar, stats
            )
            continue
        if kind == "ADD":
            bias_contribution = frozen._row_dots_lower(
                coefficient, layer.params["bias"], stats
            )
            scalar = frozen._down_add(
                scalar, bias_contribution, where=f"V5 ADD {lid} bias"
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
                    "SHAPE_MISMATCH", f"RELU layer {lid} adjoint width"
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
            factor[negative_ambiguous] = slope_block[negative_ambiguous]
            new_coefficient, radius = frozen._elementwise_product_with_error(
                coefficient, factor
            )
            intercept = np.zeros_like(coefficient)
            beta_block = np.broadcast_to(
                beta.reshape(1, -1), coefficient.shape
            )
            intercept[negative_ambiguous] = beta_block[negative_ambiguous]
            intercept_contribution = frozen._row_dots_lower(
                coefficient, intercept, stats
            )
            scalar = frozen._down_add(
                scalar,
                intercept_contribution,
                where=f"V5 RELU {lid} intercept",
            )
            pred = preds[0]
            scalar = frozen._absorb_radius(scalar, radius, box, stats)
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
            "UNSUPPORTED_OPERATOR", f"unhandled layer {lid} kind {kind}"
        )

    if pending:
        frozen._fail(
            "INVALID_GRAPH", f"unprocessed adjoints remain at {sorted(pending)}"
        )
    if not reached_input:
        frozen._fail("INVALID_GRAPH", "query does not reach INPUT_SPEC")
    scalar = np.nextafter(scalar, -math.inf)
    frozen._require_finite(scalar, where="V5 final lower bound")
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
        "fraction_endpoint_audits": int(stats.fraction_endpoint_audits),
        "relu_ambiguous_terms": int(stats.relu_ambiguous_terms),
        "dag_merges": int(stats.dag_merges),
        "conv_sparse_blocks": int(stats.conv_sparse_blocks),
        "conv_dense_blocks": int(stats.conv_dense_blocks),
        "guard_total_hex": float(stats.guard_total).hex(),
        "guard_max_hex": float(stats.guard_max).hex(),
        "guard_by_query_sha256": frozen._array_digest(guards),
        "guard_by_query_hex": [float(value).hex() for value in guards],
    }


def replay_query_lower_bounds_v5_candidate(
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
) -> QueryDualReplayV5CandidateResult:
    """Run the isolated V5 numerical candidate with no authority."""

    started = time.monotonic()
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        frozen._fail("INVALID_CHUNK", "chunk_size must be a positive integer")
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
    scalar_platform = check_scalar_guard_platform()
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
    maximum_width = max(layer.width for layer in prepared.layers.values())
    bytes_per_query = max(1, maximum_width * 8 * 12)
    memory_limited = max(1, max_workspace_bytes // bytes_per_query)
    effective_chunk_size = min(
        chunk_size, memory_limited, prepared.queries.shape[0]
    )
    stats = frozen._ReplayStats()
    stats.configure_queries(prepared.queries.shape[0])
    values = np.empty(prepared.queries.shape[0], dtype=np.float64)
    context = _V5Context(prepared=prepared)
    for chunk_start in range(0, values.size, effective_chunk_size):
        timer.check(force=True)
        chunk_end = min(values.size, chunk_start + effective_chunk_size)
        values[chunk_start:chunk_end] = _replay_block_v5(
            context, chunk_start, chunk_end, stats
        )
    timer.check(force=True)
    if not np.all(np.isfinite(values)):
        frozen._fail("NONFINITE", "non-finite V5 candidate lower bounds")
    immutable_values = frozen._immutable_f64_array(
        values, name="V5 candidate lower bounds"
    )
    execution_records = [dict(record) for record in context.executions]
    body: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "experimental_candidate",
        "proof_authority": False,
        "numeric_protocol": NUMERIC_PROTOCOL,
        "authority_integration_complete": False,
        "trusted_assumption": "supplied_bounds_are_certified",
        "hashes": dict(prepared.hashes),
        "query_count": int(immutable_values.size),
        "requested_chunk_size": int(chunk_size),
        "effective_chunk_size": int(effective_chunk_size),
        "max_workspace_bytes": int(max_workspace_bytes),
        "lower_bounds_sha256": frozen._array_digest(immutable_values),
        "lower_bounds_hex": [
            float(value).hex() for value in immutable_values
        ],
        "stats": _stats_record(stats),
        "affine_execution_count": len(execution_records),
        "affine_executions": execution_records,
        "guard_ledger_sha256": _json_sha256(execution_records),
        "dense_support_count": len(context.dense_supports),
        "conv_support_count": len(context.conv_plans),
        "numeric_platform": {
            "frozen_v3": dict(frozen_platform),
            "scalar_guard_sha256": scalar_platform.sha256,
            "scalar_guard": dict(scalar_platform.items),
        },
        "elapsed_s_hex": float(time.monotonic() - started).hex(),
    }
    body["receipt_sha256"] = _json_sha256(body)
    timer.check(force=True)
    return QueryDualReplayV5CandidateResult(
        lower_bounds=immutable_values,
        receipt=MappingProxyType(body),
    )


def verify_query_dual_replay_v5_candidate(
    result: QueryDualReplayV5CandidateResult,
) -> bool:
    """Verify candidate integrity while continuing to deny authority."""

    try:
        if (
            not isinstance(result, QueryDualReplayV5CandidateResult)
            or result.proof_authority is not False
        ):
            return False
        body = dict(result.receipt)
        claimed = str(body.pop("receipt_sha256"))
        values = np.asarray(result.lower_bounds)
        records = body["affine_executions"]
        if (
            body.get("schema") != SCHEMA
            or body.get("status") != "experimental_candidate"
            or body.get("proof_authority") is not False
            or body.get("authority_integration_complete") is not False
            or body.get("numeric_protocol") != NUMERIC_PROTOCOL
            or not isinstance(records, list)
            or body.get("affine_execution_count") != len(records)
            or body.get("guard_ledger_sha256") != _json_sha256(records)
            or not frozen.hmac.compare_digest(_json_sha256(body), claimed)
            or values.dtype != np.float64
            or values.ndim != 1
            or values.flags.writeable
            or not np.all(np.isfinite(values))
            or int(body.get("query_count")) != values.size
            or body.get("lower_bounds_sha256")
            != frozen._array_digest(values)
            or body.get("lower_bounds_hex")
            != [float(value).hex() for value in values]
        ):
            return False
        for index, record in enumerate(records):
            scalar_count = record.get("scalar_guard_applied_count")
            radius_count = record.get("componentwise_radius_applied_count")
            if (
                record.get("execution_index") != index
                or scalar_count not in {0, 1}
                or radius_count not in {0, 1}
                or scalar_count + radius_count != 1
                or scalar_count * radius_count != 0
            ):
                return False
        return True
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False


__all__ = [
    "NUMERIC_PROTOCOL",
    "QueryDualReplayV5CandidateResult",
    "replay_query_lower_bounds_v5_candidate",
    "verify_query_dual_replay_v5_candidate",
]
