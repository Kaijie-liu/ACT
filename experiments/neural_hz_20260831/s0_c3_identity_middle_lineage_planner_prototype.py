"""Isolated pure S0-C3 identity-middle structural planner.

The only accepted core is

    source -> ImplicitConv -> channel-stationary Diagonal* -> common ADD
           -> shared channel-stationary Diagonal* -> shared ImplicitConv
           -> term-local output Diagonal*.

The pre-ADD diagonal product may be genuinely empty.  A branch whose complete
segment before ADD is graph-certified empty is an identity skip and remains
the identical term object.  Every decision is planning-only: this module has
no compile, materialize, execute, rewrite, cache, or publish operation.

Structural authority comes from an ordered path certificate, not from an
operator tuple alone.  The certificate records both variable-producer and
graph-predecessor occurrences, exact value-flow tokens, and a bijection from
each multiplicative event to one live semantic operator occurrence.  Current
operator payloads are snapshotted into a stable certificate digest during
planning.  BIAS is an explicit value transition but consumes no operator
slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
import struct
from typing import Final

import numpy as np

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    BudgetReservation,
    DescriptorRequest,
    PlannerBudget,
    TailPlannerReject,
    _estimate_descriptor as _c1_estimate_descriptor,
    _normalize_support as _c1_normalize_support,
    _reserve_budget as _c1_reserve_budget,
)


C3_RULE_ID: Final = "s0_c3_same_frame_residual_distributive_identity_middle_v1"
C3_DESCRIPTOR_TAG: Final = "s0_c3_current_snapshot_composed_descriptor_v1"
C3_PAYLOAD_PREFIX: Final = b"S0C3D1"
C3_CERTIFICATE_SCHEMA: Final = "s0_c3_ordered_graph_semantic_bijection_v1"

_MULTIPLICATIVE_KINDS: Final = frozenset({"CONV", "SCALE"})
_TRANSITION_KINDS: Final = frozenset({"BIAS", "ADD"})
_BARRIER_KINDS: Final = frozenset(
    {
        "RESHAPE",
        "FLATTEN",
        "SQUEEZE",
        "UNSQUEEZE",
        "NONLINEAR",
        "RELU",
        "POOL",
        "MAXPOOL",
        "AVGPOOL",
        "DENSE",
        "CONVTRANSPOSE",
        "PERMUTE",
        "TRANSPOSE",
        "MATERIALIZE",
    }
)

_SOURCE_SEMANTIC_FIELDS: Final = (
    "c",
    "Gc",
    "Gb",
    "Ac",
    "Ab",
    "b",
    "Auc",
    "Aub",
    "ub",
)

C3_NO_CLAIMS: Final = (
    "pure_structural_and_resource_planning_only",
    "live_operator_occurrences_require_atomic_revalidation_before_use",
    "prospective_emission_is_not_executed_or_measured",
    "artifact_and_complete_strong_root_accounting_are_not_proven_here",
    "exact_set_shadow_and_concurrency_gates_are_not_proven_here",
    "no_target_run_or_score_gain_is_claimed",
)


class C3PlannerReject(ValueError):
    """Deterministic fail-closed rejection from the isolated planner."""


@dataclass(frozen=True)
class GraphEventEvidence:
    """One immutable-shape event in a graph/semantic path certificate.

    ``operator_position`` is the number of semantic operator slots consumed
    before this event.  CONV and SCALE consume exactly the operator at that
    position.  BIAS and ADD do not consume a slot.  ``selected_input`` tells
    which incoming edge this term follows; the full ordered incoming tuples
    remain present so a common ADD snapshot is comparable across terms.
    """

    kind: str
    occurrence_token: object
    producer_occurrence_tokens: tuple[object, ...]
    graph_predecessor_occurrence_tokens: tuple[object, ...]
    input_value_tokens: tuple[object, ...]
    output_value_token: object
    operator_position: int
    selected_input: int
    operator_occurrence: object | None = None
    bias_payload: tuple[float, ...] = ()


@dataclass(frozen=True)
class OrderedPathCertificate:
    """Complete ordered path evidence consumed afresh by every use."""

    schema: str
    entry_producer_occurrence_token: object
    entry_value_token: object
    events: tuple[GraphEventEvidence, ...]


@dataclass(frozen=True)
class C3AffineTermView:
    """Minimal exact term view; source owns all factor predicates."""

    source: object
    operators: tuple[object, ...]
    certificate: OrderedPathCertificate | None


@dataclass(frozen=True)
class C3AffineExprView:
    """Same-frame expression view whose contained identities are retained."""

    terms: tuple[C3AffineTermView, ...]
    bias: object
    n_out: int
    frame_id: object


@dataclass(frozen=True)
class C3BranchUse:
    """One active complete branch selected by the unique rule."""

    term_index: int
    source: object
    inner: ImplicitConv2DOp
    pre_add_diagonals: tuple[DiagonalLinearOp, ...]
    post_add_diagonals: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    output_diagonals: tuple[DiagonalLinearOp, ...]
    selected_rows: tuple[int, ...]
    descriptor_content_key: tuple[str, str]
    pre_add_diagonal_count: int
    certificate: OrderedPathCertificate
    certificate_digest: str
    certificate_snapshot_payload: bytes


@dataclass(frozen=True)
class C3PureLineagePlan:
    """Immutable, prospective, non-executable all-or-nothing plan."""

    rule_id: str
    expression: C3AffineExprView
    original_terms: tuple[C3AffineTermView, ...]
    bias: object
    frame_id: object
    common_add_snapshot_payload: bytes
    certificate_digests: tuple[str, ...]
    requests: tuple[DescriptorRequest, ...]
    uses: tuple[C3BranchUse, ...]
    identity_term_indices: tuple[int, ...]
    zero_support_term_indices: tuple[int, ...]
    reservation: BudgetReservation
    prospective_emission_contributions: int
    emission_executed: bool = False
    emission_status: str = "deferred_to_future_actual_support_transaction"
    no_claims: tuple[str, ...] = C3_NO_CLAIMS


@dataclass(frozen=True)
class C3PlanDecision:
    """Identity-preserving result of a pure planning attempt."""

    accepted: bool
    reason: str
    expression: C3AffineExprView
    terms: tuple[C3AffineTermView, ...]
    bias: object
    plan: C3PureLineagePlan | None
    no_claims: tuple[str, ...] = C3_NO_CLAIMS


@dataclass(frozen=True)
class _ValidatedCertificate:
    digest: str
    snapshot_payload: bytes
    add_event_index: int
    add_operator_position: int
    add_snapshot_payload: bytes
    event_snapshots: tuple[bytes, ...]
    normalized_event_snapshots: tuple[bytes, ...]
    operator_event_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ParsedTerm:
    term_index: int
    term: C3AffineTermView
    source: object
    identity_skip: bool
    inner: ImplicitConv2DOp | None
    pre_add_diagonals: tuple[DiagonalLinearOp, ...]
    post_add_diagonals: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    output_diagonals: tuple[DiagonalLinearOp, ...]
    selected_rows: tuple[int, ...]
    active_selected_rows: tuple[int, ...]
    stable_numeric_payload: bytes | None
    content_key: tuple[str, str] | None
    representative_key: bytes | None
    inner_snapshot: bytes | None
    post_add_snapshots: tuple[bytes, ...]
    outer_snapshot: bytes
    validated_certificate: _ValidatedCertificate
    shared_event_payloads: tuple[bytes, ...]
    shared_operator_occurrences: tuple[object, ...]


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise C3PlannerReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise C3PlannerReject(f"{name}_not_integer") from exc
    if result < 0:
        raise C3PlannerReject(f"{name}_negative")
    return result


def _token_payload(value: object, *, depth: int = 0) -> bytes:
    """Canonicalize only recursively immutable, nonempty identity tokens."""

    if depth > 32:
        raise C3PlannerReject("occurrence_token_too_deep")
    if value is None or isinstance(value, (bool, np.bool_)):
        raise C3PlannerReject("occurrence_token_not_stable")
    if isinstance(value, (int, np.integer)):
        raw = str(int(value)).encode("ascii")
        return b"I" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, str):
        if not value:
            raise C3PlannerReject("occurrence_token_empty")
        raw = value.encode("utf-8")
        return b"S" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, bytes):
        if not value:
            raise C3PlannerReject("occurrence_token_empty")
        return b"Y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, tuple):
        if not value:
            raise C3PlannerReject("occurrence_token_empty")
        parts = tuple(_token_payload(item, depth=depth + 1) for item in value)
        return b"T" + len(parts).to_bytes(8, "big") + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts
        )
    raise C3PlannerReject("occurrence_token_not_stable")


def _sequence_payload(tag: bytes, values: tuple[object, ...]) -> bytes:
    if not isinstance(values, tuple) or not values:
        raise C3PlannerReject("event_incoming_tuple_invalid")
    parts = tuple(_token_payload(value) for value in values)
    return tag + len(parts).to_bytes(8, "big") + b"".join(
        len(part).to_bytes(8, "big") + part for part in parts
    )


def _frame_payload(value: object, *, name: str) -> bytes:
    try:
        return _token_payload(value)
    except C3PlannerReject as exc:
        raise C3PlannerReject(f"{name}_not_stable") from exc


def _float64_array(value, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise C3PlannerReject(f"{name}_rank")
    if raw.dtype.kind not in "fiu":
        raise C3PlannerReject(f"{name}_not_numeric")
    result = np.array(raw, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise C3PlannerReject(f"{name}_nonfinite")
    return result


def _typed_array_payload(tag: bytes, value: np.ndarray) -> bytes:
    shape_parts = tuple(
        _token_payload(int(dimension)) for dimension in value.shape
    )
    shape = b"".join(
        len(part).to_bytes(8, "big") + part for part in shape_parts
    )
    raw = np.asarray(value, order="C").tobytes(order="C")
    return (
        tag
        + len(shape_parts).to_bytes(8, "big")
        + len(shape).to_bytes(8, "big")
        + shape
        + len(raw).to_bytes(8, "big")
        + raw
    )


def _diagonal_snapshot(
    op: object, *, name: str
) -> tuple[bytes, np.ndarray]:
    if not isinstance(op, DiagonalLinearOp) or not hasattr(op, "_diagonal"):
        raise C3PlannerReject(f"{name}_not_diagonal")
    diagonal = _float64_array(op._diagonal, name=name, ndim=1)
    if tuple(op.shape) != (diagonal.size, diagonal.size):
        raise C3PlannerReject(f"{name}_shape")
    return _typed_array_payload(b"D3", diagonal), diagonal


def _conv_snapshot(op: object, *, name: str) -> bytes:
    if not isinstance(op, ImplicitConv2DOp):
        raise C3PlannerReject(f"{name}_not_implicit_conv")
    required = (
        "_kernel",
        "_input_shape",
        "_output_shape",
        "_stride",
        "_padding",
        "_dilation",
        "_groups",
        "_row_mask",
    )
    if any(not hasattr(op, field) for field in required):
        raise C3PlannerReject(f"{name}_payload_missing")
    kernel = _float64_array(op._kernel, name=f"{name}_kernel", ndim=4)
    metadata = (
        tuple(int(value) for value in op._input_shape),
        tuple(int(value) for value in op._output_shape),
        tuple(int(value) for value in op._stride),
        tuple(int(value) for value in op._padding),
        tuple(int(value) for value in op._dilation),
        _strict_nonnegative_int(op._groups, name=f"{name}_groups"),
    )
    if metadata[-1] == 0:
        raise C3PlannerReject(f"{name}_groups_zero")
    meta_parts = tuple(_token_payload(value) for value in metadata)
    meta = b"".join(
        len(part).to_bytes(8, "big") + part for part in meta_parts
    )
    if op._row_mask is None:
        mask = b"N"
    else:
        raw_mask = np.asarray(op._row_mask)
        if raw_mask.dtype.kind != "b":
            raise C3PlannerReject(f"{name}_row_mask_not_boolean")
        normalized = np.array(
            raw_mask, dtype=np.uint8, order="C", copy=True
        ).reshape(-1)
        if normalized.size != int(np.prod(metadata[1])):
            raise C3PlannerReject(f"{name}_row_mask_shape")
        mask = _typed_array_payload(b"M3", normalized)
    kernel_payload = _typed_array_payload(b"K3", kernel)
    return b"C3" + b"".join(
        len(part).to_bytes(8, "big") + part
        for part in (meta, kernel_payload, mask)
    )


def _stationary_channel_vector(
    op: DiagonalLinearOp,
    shape: tuple[int, int, int, int],
    *,
    name: str,
) -> tuple[np.ndarray, bytes]:
    snapshot, diagonal = _diagonal_snapshot(op, name=name)
    batch, channels, height, width = (int(value) for value in shape)
    if diagonal.size != batch * channels * height * width:
        raise C3PlannerReject(f"{name}_shape")
    full = diagonal.reshape(batch, channels, height, width)
    channel = np.array(full[0, :, 0, 0], dtype=np.float64, copy=True)
    expected = np.broadcast_to(channel.reshape(1, channels, 1, 1), full.shape)
    if not np.array_equal(full, expected):
        raise C3PlannerReject(f"{name}_not_channel_stationary")
    return channel, snapshot


def _exact_dyadic_product_payload(
    scales: tuple[np.ndarray, ...], *, channels: int
) -> bytes:
    """Encode the exact-real product of stored binary64 channel values.

    A rounded float64 product is useful as a future numeric compile payload,
    but it is not an exact-real cache authority: two different dyadic products
    can round to the same binary64 value.  Integer significands and powers of
    two give a canonical product while naturally treating an empty product and
    any number of explicit exact-one maps as the same linear map.
    """

    numerators = [1] * channels
    exponents = [0] * channels
    for scale in scales:
        if scale.shape != (channels,):
            raise C3PlannerReject("exact_scale_product_shape")
        for index, raw_value in enumerate(scale):
            numerator, denominator = float(raw_value).as_integer_ratio()
            if denominator <= 0 or denominator & (denominator - 1):
                raise C3PlannerReject("binary64_ratio_not_dyadic")
            numerators[index] *= numerator
            exponents[index] -= denominator.bit_length() - 1
            product = numerators[index]
            if product == 0:
                exponents[index] = 0
                continue
            trailing_power = (
                abs(product) & -abs(product)
            ).bit_length() - 1
            if trailing_power:
                numerators[index] //= 1 << trailing_power
                exponents[index] += trailing_power
    entries = tuple(
        _token_payload((numerator, exponent))
        for numerator, exponent in zip(numerators, exponents)
    )
    return b"R3" + len(entries).to_bytes(8, "big") + b"".join(
        len(entry).to_bytes(8, "big") + entry for entry in entries
    )


def _bias_snapshot(value: object) -> bytes:
    if not isinstance(value, tuple) or not value:
        raise C3PlannerReject("bias_payload_not_nonempty_tuple")
    packed: list[bytes] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise C3PlannerReject("bias_payload_not_numeric")
        number = float(item)
        if not np.isfinite(number):
            raise C3PlannerReject("bias_payload_nonfinite")
        packed.append(struct.pack(">d", number))
    return b"B3" + len(packed).to_bytes(8, "big") + b"".join(packed)


def _event_graph_payload(
    event: GraphEventEvidence,
    *,
    include_selected_input: bool,
    include_operator_position: bool = True,
) -> bytes:
    kind = event.kind.encode("ascii")
    position = _strict_nonnegative_int(
        event.operator_position, name="event_operator_position"
    )
    selected = _strict_nonnegative_int(
        event.selected_input, name="event_selected_input"
    )
    parts = (
        b"K" + len(kind).to_bytes(8, "big") + kind,
        b"O" + _token_payload(event.occurrence_token),
        _sequence_payload(b"P", event.producer_occurrence_tokens),
        _sequence_payload(
            b"G", event.graph_predecessor_occurrence_tokens
        ),
        _sequence_payload(b"I", event.input_value_tokens),
        b"V" + _token_payload(event.output_value_token),
    )
    if include_operator_position:
        parts = (*parts, b"Q" + position.to_bytes(8, "big"))
    if include_selected_input:
        parts = (*parts, b"S" + selected.to_bytes(8, "big"))
    return b"E3" + b"".join(
        len(part).to_bytes(8, "big") + part for part in parts
    )


def _validate_certificate(
    certificate: OrderedPathCertificate | None,
    operators: tuple[object, ...],
    *,
    term_index: int,
) -> _ValidatedCertificate:
    if not isinstance(certificate, OrderedPathCertificate):
        raise C3PlannerReject("semantic_path_certificate_missing")
    if certificate.schema != C3_CERTIFICATE_SCHEMA:
        raise C3PlannerReject("semantic_path_certificate_schema")
    if not isinstance(certificate.events, tuple) or not certificate.events:
        raise C3PlannerReject("semantic_path_events_missing")
    entry_occurrence = _token_payload(
        certificate.entry_producer_occurrence_token
    )
    entry_value = _token_payload(certificate.entry_value_token)

    cursor = 0
    previous_occurrence = entry_occurrence
    previous_value = entry_value
    previous_kind: str | None = None
    seen_occurrences: set[bytes] = {entry_occurrence}
    seen_values: set[bytes] = {entry_value}
    event_snapshots: list[bytes] = []
    normalized_event_snapshots: list[bytes] = []
    operator_event_indices: list[int] = []
    add_indices: list[int] = []
    add_payload: bytes | None = None

    for event_index, event in enumerate(certificate.events):
        if not isinstance(event, GraphEventEvidence):
            raise C3PlannerReject("semantic_path_event_type")
        if not isinstance(event.kind, str):
            raise C3PlannerReject("unknown_graph_event")
        kind = event.kind
        if kind not in (
            _MULTIPLICATIVE_KINDS
            | _TRANSITION_KINDS
            | _BARRIER_KINDS
        ):
            raise C3PlannerReject("unknown_graph_event")
        if kind in _BARRIER_KINDS:
            raise C3PlannerReject(f"core_contains_{kind.lower()}_event")

        occurrence = _token_payload(event.occurrence_token)
        if occurrence in seen_occurrences:
            raise C3PlannerReject("duplicate_graph_event")
        seen_occurrences.add(occurrence)
        producers = tuple(
            _token_payload(value)
            for value in event.producer_occurrence_tokens
        ) if isinstance(event.producer_occurrence_tokens, tuple) else ()
        predecessors = tuple(
            _token_payload(value)
            for value in event.graph_predecessor_occurrence_tokens
        ) if isinstance(event.graph_predecessor_occurrence_tokens, tuple) else ()
        inputs = tuple(
            _token_payload(value) for value in event.input_value_tokens
        ) if isinstance(event.input_value_tokens, tuple) else ()
        if not producers or not predecessors or not inputs:
            raise C3PlannerReject("event_incoming_tuple_invalid")
        if not (len(producers) == len(predecessors) == len(inputs)):
            raise C3PlannerReject("event_incoming_arity_mismatch")
        if producers != predecessors:
            raise C3PlannerReject("producer_predecessor_mismatch")
        if occurrence in producers:
            raise C3PlannerReject("event_occurrence_self_dependency")
        output_value = _token_payload(event.output_value_token)
        if output_value in seen_values:
            raise C3PlannerReject("duplicate_output_value")
        selected = _strict_nonnegative_int(
            event.selected_input, name="event_selected_input"
        )
        if selected >= len(inputs):
            raise C3PlannerReject("event_selected_input_out_of_range")
        if producers[selected] != previous_occurrence:
            raise C3PlannerReject("ordered_event_predecessor_mismatch")
        if inputs[selected] != previous_value:
            raise C3PlannerReject("ordered_event_value_flow_mismatch")
        position = _strict_nonnegative_int(
            event.operator_position, name="event_operator_position"
        )
        if position != cursor:
            raise C3PlannerReject("event_operator_order")

        graph_payload = _event_graph_payload(
            event, include_selected_input=True
        )
        normalized_graph_payload = _event_graph_payload(
            event,
            include_selected_input=False,
            include_operator_position=False,
        )
        if kind in _MULTIPLICATIVE_KINDS:
            if event.bias_payload != ():
                raise C3PlannerReject("multiplicative_event_has_bias_payload")
            if len(inputs) != 1 or selected != 0:
                raise C3PlannerReject("multiplicative_event_arity")
            if cursor >= len(operators):
                raise C3PlannerReject("unaccounted_linear_event")
            semantic_op = operators[cursor]
            expected_type = (
                ImplicitConv2DOp if kind == "CONV" else DiagonalLinearOp
            )
            if not isinstance(semantic_op, expected_type):
                raise C3PlannerReject("unaccounted_linear_event")
            if event.operator_occurrence is not semantic_op:
                raise C3PlannerReject("operator_occurrence_mismatch")
            if kind == "CONV":
                semantic_payload = _conv_snapshot(
                    semantic_op,
                    name=f"term_{term_index}_event_{event_index}_conv",
                )
            else:
                semantic_payload, _ = _diagonal_snapshot(
                    semantic_op,
                    name=f"term_{term_index}_event_{event_index}_scale",
                )
            snapshot = b"M3" + b"".join(
                len(part).to_bytes(8, "big") + part
                for part in (graph_payload, semantic_payload)
            )
            normalized_snapshot = b"M3" + b"".join(
                len(part).to_bytes(8, "big") + part
                for part in (normalized_graph_payload, semantic_payload)
            )
            operator_event_indices.append(event_index)
            cursor += 1
        else:
            if event.operator_occurrence is not None:
                raise C3PlannerReject("transition_occupies_operator_slot")
            if kind == "BIAS":
                if previous_kind != "SCALE":
                    raise C3PlannerReject("bias_without_paired_scale_predecessor")
                if len(inputs) != 1 or selected != 0:
                    raise C3PlannerReject("bias_transition_arity")
                transition_payload = _bias_snapshot(event.bias_payload)
            else:
                if event.bias_payload != ():
                    raise C3PlannerReject("add_event_has_bias_payload")
                if len(inputs) < 2:
                    raise C3PlannerReject("add_event_arity")
                transition_payload = b"A3"
                add_indices.append(event_index)
                add_payload = b"A3" + b"".join(
                    len(part).to_bytes(8, "big") + part
                    for part in (
                        _event_graph_payload(
                            event,
                            include_selected_input=False,
                            include_operator_position=False,
                        ),
                        transition_payload,
                    )
                )
            snapshot = b"T3" + b"".join(
                len(part).to_bytes(8, "big") + part
                for part in (graph_payload, transition_payload)
            )
            normalized_snapshot = b"T3" + b"".join(
                len(part).to_bytes(8, "big") + part
                for part in (normalized_graph_payload, transition_payload)
            )

        event_snapshots.append(snapshot)
        normalized_event_snapshots.append(normalized_snapshot)
        previous_occurrence = occurrence
        previous_value = output_value
        seen_values.add(output_value)
        previous_kind = kind

    if cursor != len(operators):
        raise C3PlannerReject("semantic_operator_without_graph_event")
    if len(operator_event_indices) != len(set(operator_event_indices)):
        raise C3PlannerReject("duplicate_operator_mapping")
    if not add_indices:
        raise C3PlannerReject("common_add_event_missing")
    if len(add_indices) != 1:
        raise C3PlannerReject("nested_add_event")
    assert add_payload is not None
    add_event_index = add_indices[0]
    add_position = certificate.events[add_event_index].operator_position
    schema = certificate.schema.encode("utf-8")
    snapshot_payload = b"S0C3CERT1" + b"".join(
        len(part).to_bytes(8, "big") + part
        for part in (
            schema,
            entry_occurrence,
            entry_value,
            *event_snapshots,
        )
    )
    return _ValidatedCertificate(
        digest=hashlib.sha256(snapshot_payload).hexdigest(),
        snapshot_payload=snapshot_payload,
        add_event_index=add_event_index,
        add_operator_position=int(add_position),
        add_snapshot_payload=add_payload,
        event_snapshots=tuple(event_snapshots),
        normalized_event_snapshots=tuple(normalized_event_snapshots),
        operator_event_indices=tuple(operator_event_indices),
    )


def _source_is_valid(source: object, frame_payload: bytes) -> bool:
    if not (
        hasattr(source, "frame_id")
        and hasattr(source, "exact")
        and source.exact is True
        and all(hasattr(source, field) for field in _SOURCE_SEMANTIC_FIELDS)
    ):
        return False
    try:
        source_frame = _frame_payload(
            source.frame_id, name="source_frame_id"
        )
    except C3PlannerReject:
        return False
    return source_frame == frame_payload


def _parse_term(
    term: C3AffineTermView,
    *,
    term_index: int,
    frame_payload: bytes,
    support: tuple[int, ...],
    n_out: int,
) -> _ParsedTerm:
    if not isinstance(term, C3AffineTermView):
        raise C3PlannerReject("unsupported_term_view")
    if not _source_is_valid(term.source, frame_payload):
        raise C3PlannerReject("source_frame_exact_or_predicate_schema_mismatch")
    if not isinstance(term.operators, tuple):
        raise C3PlannerReject("term_operators_not_tuple")
    operators = term.operators
    validated = _validate_certificate(
        term.certificate, operators, term_index=term_index
    )
    add_position = validated.add_operator_position
    if add_position > len(operators):
        raise C3PlannerReject("add_operator_position_out_of_range")

    cursor = len(operators)
    while cursor and isinstance(operators[cursor - 1], DiagonalLinearOp):
        cursor -= 1
    output_diagonals = tuple(operators[cursor:])
    if not cursor or not isinstance(operators[cursor - 1], ImplicitConv2DOp):
        raise C3PlannerReject("post_add_shared_outer_missing")
    outer_index = cursor - 1
    outer = operators[outer_index]
    if outer_index < add_position:
        raise C3PlannerReject("common_add_not_before_outer")
    post_add_diagonals = tuple(operators[add_position:outer_index])
    if any(
        not isinstance(value, DiagonalLinearOp)
        for value in post_add_diagonals
    ):
        if any(
            isinstance(value, ImplicitConv2DOp)
            for value in operators[add_position:outer_index]
        ):
            raise C3PlannerReject("two_convs_after_common_add")
        raise C3PlannerReject("post_add_suffix_operator_barrier")

    outer_snapshot = _conv_snapshot(
        outer, name=f"term_{term_index}_outer"
    )
    if int(outer.shape[0]) != n_out:
        raise C3PlannerReject("tail_output_shape_mismatch")
    for output_index, output_op in enumerate(output_diagonals):
        _, diagonal = _diagonal_snapshot(
            output_op,
            name=f"term_{term_index}_output_{output_index}",
        )
        if diagonal.size != n_out:
            raise C3PlannerReject("output_diagonal_shape")

    selected = support
    for output_index, output_op in enumerate(reversed(output_diagonals)):
        _, diagonal = _diagonal_snapshot(
            output_op,
            name=f"term_{term_index}_output_reverse_{output_index}",
        )
        selected = tuple(row for row in selected if diagonal[row] != 0.0)
    if outer._row_mask is None:
        active_selected = selected
    else:
        raw_mask = np.asarray(outer._row_mask)
        if raw_mask.dtype.kind != "b" or raw_mask.size != n_out:
            raise C3PlannerReject("outer_row_mask_shape")
        mask = raw_mask.reshape(-1)
        active_selected = tuple(row for row in selected if bool(mask[row]))

    post_snapshots: list[bytes] = []
    post_scales: list[np.ndarray] = []
    for post_index, post_op in enumerate(post_add_diagonals):
        scale, snapshot = _stationary_channel_vector(
            post_op,
            tuple(outer.input_shape),
            name=f"term_{term_index}_post_add_{post_index}",
        )
        post_scales.append(scale)
        post_snapshots.append(snapshot)

    # The source is the already-built prefix.  Consequently the semantic
    # operator tuple begins at the unique candidate branch entry; no caller-
    # chosen cut may hide an operator or reclassify a complete active branch.
    segment = operators[:add_position]
    identity_skip = not segment
    if identity_skip:
        inner = None
        pre_add_diagonals: tuple[DiagonalLinearOp, ...] = ()
        inner_snapshot = None
        stable_payload = None
        content_key = None
        representative_key = None
    else:
        if not isinstance(segment[0], ImplicitConv2DOp):
            raise C3PlannerReject("nonempty_branch_without_inner_conv")
        if any(isinstance(value, ImplicitConv2DOp) for value in segment[1:]):
            raise C3PlannerReject("two_convs_before_common_add")
        if any(
            not isinstance(value, DiagonalLinearOp) for value in segment[1:]
        ):
            raise C3PlannerReject("nonempty_branch_operator_barrier")
        inner = segment[0]
        pre_add_diagonals = tuple(segment[1:])
        if tuple(inner.output_shape) != tuple(outer.input_shape):
            raise C3PlannerReject("tail_intermediate_shape_mismatch")
        inner_snapshot = _conv_snapshot(
            inner, name=f"term_{term_index}_inner"
        )
        sigma = np.ones(int(inner.output_shape[1]), dtype=np.float64)
        all_diagonal_snapshots: list[bytes] = []
        middle_scales: list[np.ndarray] = []
        for middle_index, middle_op in enumerate(
            (*pre_add_diagonals, *post_add_diagonals)
        ):
            scale, snapshot = _stationary_channel_vector(
                middle_op,
                tuple(inner.output_shape),
                name=f"term_{term_index}_middle_{middle_index}",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                sigma = np.multiply(sigma, scale)
            if not np.all(np.isfinite(sigma)):
                raise C3PlannerReject("middle_scale_product_nonfinite")
            middle_scales.append(scale)
            all_diagonal_snapshots.append(snapshot)
        sigma_payload = _typed_array_payload(
            b"S3", np.asarray(sigma, dtype="<f8", order="C")
        )
        exact_product_payload = _exact_dyadic_product_payload(
            tuple(middle_scales), channels=int(inner.output_shape[1])
        )
        stable_payload = C3_PAYLOAD_PREFIX + b"".join(
            len(part).to_bytes(8, "big") + part
            for part in (
                inner_snapshot,
                exact_product_payload,
                sigma_payload,
                outer_snapshot,
            )
        )
        content_key = (
            C3_DESCRIPTOR_TAG,
            hashlib.sha256(stable_payload).hexdigest(),
        )
        representative_key = b"".join(
            len(part).to_bytes(8, "big") + part
            for part in (
                inner_snapshot,
                *all_diagonal_snapshots,
                outer_snapshot,
                validated.snapshot_payload,
            )
        )

    cert = term.certificate
    assert cert is not None
    outer_event_index = validated.operator_event_indices[outer_index]
    shared_event_start = validated.add_event_index
    shared_event_payloads = list(
        validated.normalized_event_snapshots[
            shared_event_start : outer_event_index + 1
        ]
    )
    # The selected incoming ADD edge is use-local.  The normalized complete
    # ADD payload, which contains every ordered input, is the shared snapshot.
    shared_event_payloads[0] = validated.add_snapshot_payload
    shared_operator_occurrences = tuple(
        operators[position] for position in range(add_position, outer_index + 1)
    )

    return _ParsedTerm(
        term_index=term_index,
        term=term,
        source=term.source,
        identity_skip=identity_skip,
        inner=inner,
        pre_add_diagonals=pre_add_diagonals,
        post_add_diagonals=post_add_diagonals,
        outer=outer,
        output_diagonals=output_diagonals,
        selected_rows=selected,
        active_selected_rows=active_selected,
        stable_numeric_payload=stable_payload,
        content_key=content_key,
        representative_key=representative_key,
        inner_snapshot=inner_snapshot,
        post_add_snapshots=tuple(post_snapshots),
        outer_snapshot=outer_snapshot,
        validated_certificate=validated,
        shared_event_payloads=tuple(shared_event_payloads),
        shared_operator_occurrences=shared_operator_occurrences,
    )


def _require_one_shared_suffix(parsed: tuple[_ParsedTerm, ...]) -> None:
    reference = parsed[0]
    for item in parsed[1:]:
        if (
            item.validated_certificate.add_snapshot_payload
            != reference.validated_certificate.add_snapshot_payload
        ):
            raise C3PlannerReject("latest_common_add_occurrence_or_snapshot_mismatch")
        if item.outer is not reference.outer:
            raise C3PlannerReject("shared_outer_occurrence_mismatch")
        if item.outer_snapshot != reference.outer_snapshot:
            raise C3PlannerReject("shared_outer_snapshot_mismatch")
        if len(item.post_add_diagonals) != len(reference.post_add_diagonals):
            raise C3PlannerReject("post_add_suffix_length_mismatch")
        if any(
            left is not right
            for left, right in zip(
                item.post_add_diagonals, reference.post_add_diagonals
            )
        ):
            raise C3PlannerReject("post_add_suffix_occurrence_mismatch")
        if item.post_add_snapshots != reference.post_add_snapshots:
            raise C3PlannerReject("post_add_suffix_snapshot_mismatch")
        if item.shared_event_payloads != reference.shared_event_payloads:
            raise C3PlannerReject("shared_graph_suffix_snapshot_mismatch")
        if len(item.shared_operator_occurrences) != len(
            reference.shared_operator_occurrences
        ) or any(
            left is not right
            for left, right in zip(
                item.shared_operator_occurrences,
                reference.shared_operator_occurrences,
            )
        ):
            raise C3PlannerReject("shared_graph_suffix_occurrence_mismatch")


def _reject_decision(
    expr: C3AffineExprView,
    terms: tuple[C3AffineTermView, ...],
    bias: object,
    reason: str,
) -> C3PlanDecision:
    return C3PlanDecision(
        accepted=False,
        reason=reason,
        expression=expr,
        terms=terms,
        bias=bias,
        plan=None,
    )


def plan_s0_c3_identity_middle_lineage(
    expr: C3AffineExprView,
    output_support,
    *,
    budget: PlannerBudget = PlannerBudget(),
) -> C3PlanDecision:
    """Return the sole pure C3 plan or the untouched expression by identity."""

    original_terms = expr.terms
    original_bias = expr.bias
    try:
        if not isinstance(original_terms, tuple):
            raise C3PlannerReject("expression_terms_not_tuple")
        if not original_terms:
            raise C3PlannerReject("expression_terms_empty")
        n_out = _strict_nonnegative_int(expr.n_out, name="n_out")
        frame_payload = _frame_payload(
            expr.frame_id, name="expression_frame_id"
        )
        try:
            support = _c1_normalize_support(output_support, n_out)
        except TailPlannerReject as exc:
            raise C3PlannerReject(str(exc)) from exc
        if not support:
            raise C3PlannerReject("empty_output_support")

        parsed = tuple(
            _parse_term(
                term,
                term_index=term_index,
                frame_payload=frame_payload,
                support=support,
                n_out=n_out,
            )
            for term_index, term in enumerate(original_terms)
        )
        _require_one_shared_suffix(parsed)

        identity_indices = tuple(
            item.term_index for item in parsed if item.identity_skip
        )
        complete = tuple(item for item in parsed if not item.identity_skip)
        if not complete:
            raise C3PlannerReject("no_complete_c3_branch")
        zero_support = tuple(
            item.term_index
            for item in complete
            if not item.active_selected_rows
        )
        eligible = tuple(
            item for item in complete if item.active_selected_rows
        )
        if not eligible:
            raise C3PlannerReject("no_nonzero_fusible_branch")

        grouped: dict[bytes, list[_ParsedTerm]] = {}
        digest_payloads: dict[str, bytes] = {}
        for item in eligible:
            assert item.stable_numeric_payload is not None
            assert item.content_key is not None
            digest = item.content_key[1]
            prior = digest_payloads.get(digest)
            if prior is not None and prior != item.stable_numeric_payload:
                raise C3PlannerReject("descriptor_content_hash_collision")
            digest_payloads[digest] = item.stable_numeric_payload
            grouped.setdefault(item.stable_numeric_payload, []).append(item)

        requests: list[DescriptorRequest] = []
        uses: list[C3BranchUse] = []
        for _, group in sorted(
            grouped.items(),
            key=lambda pair: (hashlib.sha256(pair[0]).hexdigest(), pair[0]),
        ):
            representative = min(
                group, key=lambda item: item.representative_key or b""
            )
            selected_rows = tuple(
                sorted({row for item in group for row in item.selected_rows})
            )
            assert representative.inner is not None
            assert representative.content_key is not None
            estimate = _c1_estimate_descriptor(
                representative.inner,
                representative.outer,
                selected_rows,
            )
            requests.append(
                DescriptorRequest(
                    content_key=representative.content_key,
                    inner=representative.inner,
                    middle_ops=(
                        *representative.pre_add_diagonals,
                        *representative.post_add_diagonals,
                    ),
                    outer=representative.outer,
                    selected_rows=selected_rows,
                    term_indices=tuple(
                        sorted(item.term_index for item in group)
                    ),
                    use_count=len(group),
                    estimate=estimate,
                )
            )
            for item in group:
                assert item.inner is not None
                assert item.content_key is not None
                certificate = item.term.certificate
                assert certificate is not None
                uses.append(
                    C3BranchUse(
                        term_index=item.term_index,
                        source=item.source,
                        inner=item.inner,
                        pre_add_diagonals=item.pre_add_diagonals,
                        post_add_diagonals=item.post_add_diagonals,
                        outer=item.outer,
                        output_diagonals=item.output_diagonals,
                        selected_rows=item.selected_rows,
                        descriptor_content_key=item.content_key,
                        pre_add_diagonal_count=len(
                            item.pre_add_diagonals
                        ),
                        certificate=certificate,
                        certificate_digest=(
                            item.validated_certificate.digest
                        ),
                        certificate_snapshot_payload=(
                            item.validated_certificate.snapshot_payload
                        ),
                    )
                )

        request_tuple = tuple(requests)
        try:
            reservation = _c1_reserve_budget(request_tuple, budget)
        except TailPlannerReject as exc:
            raise C3PlannerReject(str(exc)) from exc
        plan = C3PureLineagePlan(
            rule_id=C3_RULE_ID,
            expression=expr,
            original_terms=original_terms,
            bias=original_bias,
            frame_id=expr.frame_id,
            common_add_snapshot_payload=(
                parsed[0].validated_certificate.add_snapshot_payload
            ),
            certificate_digests=tuple(
                item.validated_certificate.digest for item in parsed
            ),
            requests=request_tuple,
            uses=tuple(sorted(uses, key=lambda item: item.term_index)),
            identity_term_indices=identity_indices,
            zero_support_term_indices=zero_support,
            reservation=reservation,
            prospective_emission_contributions=sum(
                request.estimate.emission_contributions
                for request in request_tuple
            ),
        )
        return C3PlanDecision(
            accepted=True,
            reason="planned_pure_identity_middle_lineage_only",
            expression=expr,
            terms=original_terms,
            bias=original_bias,
            plan=plan,
        )
    except C3PlannerReject as exc:
        return _reject_decision(expr, original_terms, original_bias, str(exc))
    except Exception as exc:
        return _reject_decision(
            expr,
            original_terms,
            original_bias,
            f"planner_failed_{type(exc).__name__}",
        )


__all__ = [
    "C3AffineExprView",
    "C3AffineTermView",
    "C3BranchUse",
    "C3_CERTIFICATE_SCHEMA",
    "C3_DESCRIPTOR_TAG",
    "C3_NO_CLAIMS",
    "C3_PAYLOAD_PREFIX",
    "C3PlanDecision",
    "C3PlannerReject",
    "C3PureLineagePlan",
    "C3_RULE_ID",
    "GraphEventEvidence",
    "OrderedPathCertificate",
    "PlannerBudget",
    "plan_s0_c3_identity_middle_lineage",
]
