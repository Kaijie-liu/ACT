"""Experiment-only pure lineage planner for preregistered S0-C2-v1.

This module is intentionally disconnected from ACT production imports.  It
recognizes one, and only one, residual-distributive schedule: every complete
nonzero-support branch crosses exactly one latest common ADD and has shape

    prefix -> Conv -> channel diagonal+ -> ADD -> common channel diagonal*
           -> shared Conv -> term-local output diagonal*.

Identity skips and zero-support terms stay as their original objects.  Any
ambiguous branch rejects the complete request.  Success is only an immutable
plan plus conservative S0-C1 estimates; this module has no compile, execute,
rewrite, publication, fallback, or emission interface.

The C1 estimator is reused only for arithmetic.  C2 descriptor grouping uses
fresh current-payload snapshots instead of an operator's constructor-time
``content_key``.  Requests still retain live operators, so a returned plan is
not a frozen compile payload and must be revalidated by a future atomic
integration before it can be consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
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
    _encode_stable as _c1_encode_stable,
    _estimate_descriptor as _c1_estimate_descriptor,
    _normalize_support as _c1_normalize_support,
    _reserve_budget as _c1_reserve_budget,
)


C2_RULE_ID: Final = "s0_c2_same_frame_residual_distributive_lineage_v1"
C2_DESCRIPTOR_TAG: Final = "s0_c2_current_snapshot_composed_descriptor_v1"

# A mark kind outside this closed vocabulary is not silently treated as a
# prefix annotation: unknown lineage is malformed and the transaction closes.
BOUNDARY_KINDS: Final = frozenset(
    {
        "ADD",
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

C2_NO_CLAIMS: Final = (
    "c1_constructor_time_content_key_can_be_stale_after_payload_mutation",
    "c2_groups_by_current_snapshot_but_returned_requests_hold_live_operators",
    "returned_plan_is_not_a_frozen_plan_to_compile_payload",
    "v2_actual_support_emission_reservation_is_deferred_and_not_executed",
    "c1_estimate_does_not_prove_the_frozen_one_quarter_unfused_work_gate",
    "logical_estimates_do_not_prove_actual_artifact_bytes_or_nnz",
    "whole_state_bytes_entries_old_roots_rss_and_concurrency_are_not_proven",
    "no_production_lineage_hit_tiny_iid143_or_formal_gain_is_claimed",
)


class C2PlannerReject(ValueError):
    """Stable fail-closed rejection from the isolated pure planner."""


@dataclass(frozen=True)
class BoundaryMark:
    """Immutable graph cut; occurrence values are equality-only tokens."""

    kind: str
    occurrence_key: object
    operator_index: int


@dataclass(frozen=True)
class LineageAffineTermView:
    """Minimal term view.  ``source`` owns factor/predicate semantics."""

    source: object
    operators: tuple[object, ...]
    boundaries: tuple[BoundaryMark, ...] | None


@dataclass(frozen=True)
class LineageAffineExprView:
    """Minimal same-frame expression view; object identity is significant."""

    terms: tuple[LineageAffineTermView, ...]
    bias: object
    n_out: int
    frame_id: object


@dataclass(frozen=True)
class C2BranchUse:
    """One complete nonzero branch selected by the unique C2 rule."""

    term_index: int
    common_add: BoundaryMark
    prefix: tuple[object, ...]
    inner: ImplicitConv2DOp
    pre_add_diagonals: tuple[DiagonalLinearOp, ...]
    post_add_diagonals: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    output_diagonals: tuple[DiagonalLinearOp, ...]
    selected_rows: tuple[int, ...]
    descriptor_content_key: tuple[str, str]


@dataclass(frozen=True)
class C2PureLineagePlan:
    """Prospective all-or-nothing plan.  It is deliberately not executable."""

    rule_id: str
    expression: LineageAffineExprView
    original_terms: tuple[LineageAffineTermView, ...]
    bias: object
    frame_id: object
    common_add_occurrence_payload: bytes
    common_add_marks: tuple[BoundaryMark, ...]
    requests: tuple[DescriptorRequest, ...]
    uses: tuple[C2BranchUse, ...]
    identity_term_indices: tuple[int, ...]
    zero_support_term_indices: tuple[int, ...]
    reservation: BudgetReservation
    prospective_emission_contributions: int
    emission_executed: bool = False
    emission_status: str = "deferred_to_future_v2_actual_support_transaction"
    no_claims: tuple[str, ...] = C2_NO_CLAIMS


@dataclass(frozen=True)
class C2PlanDecision:
    """Identity-preserving result of a pure planning attempt."""

    accepted: bool
    reason: str
    expression: LineageAffineExprView
    terms: tuple[LineageAffineTermView, ...]
    bias: object
    plan: C2PureLineagePlan | None
    no_claims: tuple[str, ...] = C2_NO_CLAIMS


@dataclass(frozen=True)
class _ParsedTerm:
    term_index: int
    term: LineageAffineTermView
    common_add: BoundaryMark
    occurrence_payload: bytes
    prefix: tuple[object, ...]
    identity_skip: bool
    inner: ImplicitConv2DOp | None
    pre_add_diagonals: tuple[DiagonalLinearOp, ...]
    post_add_diagonals: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    output_diagonals: tuple[DiagonalLinearOp, ...]
    selected_rows: tuple[int, ...]
    active_selected_rows: tuple[int, ...]
    sigma: np.ndarray | None
    inner_snapshot: bytes | None
    post_add_snapshots: tuple[bytes, ...]
    outer_snapshot: bytes
    stable_payload: bytes | None
    content_key: tuple[str, str] | None
    representative_key: bytes | None


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise C2PlannerReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise C2PlannerReject(f"{name}_not_integer") from exc
    if result < 0:
        raise C2PlannerReject(f"{name}_negative")
    return result


def _occurrence_payload(value, *, depth: int = 0) -> bytes:
    """Encode only recursively immutable equality tokens."""

    if depth > 32:
        raise C2PlannerReject("occurrence_key_too_deep")
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise C2PlannerReject("occurrence_key_not_stable")
    if isinstance(value, (int, np.integer)):
        payload = str(int(value)).encode("ascii")
        return b"I" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, str):
        if not value:
            raise C2PlannerReject("occurrence_key_empty")
        payload = value.encode("utf-8")
        return b"S" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, bytes):
        if not value:
            raise C2PlannerReject("occurrence_key_empty")
        return b"Y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, tuple):
        if not value:
            raise C2PlannerReject("occurrence_key_empty")
        pieces = [_occurrence_payload(item, depth=depth + 1) for item in value]
        return b"T" + len(pieces).to_bytes(8, "big") + b"".join(
            len(piece).to_bytes(8, "big") + piece for piece in pieces
        )
    raise C2PlannerReject("occurrence_key_not_stable")


def _float64_array(value, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise C2PlannerReject(f"{name}_rank")
    if raw.dtype.kind not in "fiu":
        raise C2PlannerReject(f"{name}_not_numeric")
    array = np.array(raw, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise C2PlannerReject(f"{name}_nonfinite")
    return array


def _typed_array_payload(tag: bytes, array: np.ndarray) -> bytes:
    shape = _c1_encode_stable(tuple(int(value) for value in array.shape))
    data = np.asarray(array, order="C").tobytes(order="C")
    return (
        tag
        + len(shape).to_bytes(8, "big")
        + shape
        + len(data).to_bytes(8, "big")
        + data
    )


def _diagonal_snapshot(
    op: object, *, name: str
) -> tuple[bytes, np.ndarray]:
    if not isinstance(op, DiagonalLinearOp) or not hasattr(op, "_diagonal"):
        raise C2PlannerReject(f"{name}_not_diagonal")
    diagonal = _float64_array(op._diagonal, name=name, ndim=1)
    if tuple(op.shape) != (diagonal.size, diagonal.size):
        raise C2PlannerReject(f"{name}_shape")
    return _typed_array_payload(b"D2", diagonal), diagonal


def _conv_snapshot(op: object, *, name: str) -> bytes:
    if not isinstance(op, ImplicitConv2DOp):
        raise C2PlannerReject(f"{name}_not_implicit_conv2d")
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
        raise C2PlannerReject(f"{name}_payload_missing")
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
        raise C2PlannerReject(f"{name}_groups_zero")
    meta_payload = _c1_encode_stable(metadata)
    kernel_payload = _typed_array_payload(b"K2", kernel)
    if op._row_mask is None:
        mask_payload = b"N"
    else:
        raw_mask = np.asarray(op._row_mask)
        if raw_mask.dtype.kind != "b":
            raise C2PlannerReject(f"{name}_row_mask_not_boolean")
        mask = np.array(raw_mask, dtype=np.uint8, order="C", copy=True).reshape(-1)
        if mask.size != int(np.prod(metadata[1])):
            raise C2PlannerReject(f"{name}_row_mask_shape")
        mask_payload = _typed_array_payload(b"M2", mask)
    return b"C2" + b"".join(
        len(piece).to_bytes(8, "big") + piece
        for piece in (meta_payload, kernel_payload, mask_payload)
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
        raise C2PlannerReject(f"{name}_shape")
    full = diagonal.reshape(batch, channels, height, width)
    channel = np.array(full[0, :, 0, 0], dtype=np.float64, copy=True)
    expected = np.broadcast_to(channel.reshape(1, channels, 1, 1), full.shape)
    if not np.array_equal(full, expected):
        raise C2PlannerReject(f"{name}_not_channel_stationary")
    return channel, snapshot


def _validate_boundaries(
    boundaries: tuple[BoundaryMark, ...] | None,
    *,
    operator_count: int,
) -> tuple[tuple[BoundaryMark, bytes, int], ...]:
    if boundaries is None:
        raise C2PlannerReject("lineage_missing")
    if not isinstance(boundaries, tuple):
        raise C2PlannerReject("lineage_not_tuple")
    if not boundaries:
        raise C2PlannerReject("lineage_empty")
    validated: list[tuple[BoundaryMark, bytes, int]] = []
    previous_index = -1
    seen: set[tuple[str, bytes, int]] = set()
    for mark in boundaries:
        if not isinstance(mark, BoundaryMark):
            raise C2PlannerReject("lineage_mark_type")
        if not isinstance(mark.kind, str) or mark.kind not in BOUNDARY_KINDS:
            raise C2PlannerReject("lineage_mark_kind")
        occurrence = _occurrence_payload(mark.occurrence_key)
        index = _strict_nonnegative_int(
            mark.operator_index, name="boundary_operator_index"
        )
        if index > operator_count:
            raise C2PlannerReject("boundary_operator_index_out_of_range")
        if index < previous_index:
            raise C2PlannerReject("lineage_mark_order")
        identity = (mark.kind, occurrence, index)
        if identity in seen:
            raise C2PlannerReject("lineage_mark_duplicate")
        seen.add(identity)
        validated.append((mark, occurrence, index))
        previous_index = index
    if validated[-1][0].kind != "ADD":
        raise C2PlannerReject("latest_boundary_not_add")
    return tuple(validated)


def _frame_payload(value: object, *, name: str) -> bytes:
    """Encode a nonempty immutable frame identity or fail closed."""

    try:
        return _occurrence_payload(value)
    except C2PlannerReject as exc:
        raise C2PlannerReject(f"{name}_not_stable") from exc


def _source_is_same_frame(source: object, frame_payload: bytes) -> bool:
    if not (
        hasattr(source, "frame_id")
        and hasattr(source, "exact")
        and source.exact is True
    ):
        return False
    try:
        source_payload = _frame_payload(
            source.frame_id, name="source_frame_id"
        )
    except C2PlannerReject:
        return False
    return source_payload == frame_payload


def _parse_term(
    term: LineageAffineTermView,
    *,
    term_index: int,
    frame_payload: bytes,
    support: tuple[int, ...],
    n_out: int,
) -> _ParsedTerm:
    if not isinstance(term, LineageAffineTermView):
        raise C2PlannerReject("unsupported_term_view")
    if not _source_is_same_frame(term.source, frame_payload):
        raise C2PlannerReject("source_frame_or_exact_mismatch")
    if not isinstance(term.operators, tuple):
        raise C2PlannerReject("term_operators_not_tuple")
    operators = term.operators
    lineage = _validate_boundaries(
        term.boundaries, operator_count=len(operators)
    )
    common_add, occurrence, add_index = lineage[-1]
    previous_index = lineage[-2][2] if len(lineage) > 1 else 0
    if previous_index > add_index:
        raise C2PlannerReject("lineage_mark_order")
    identity_skip = previous_index == add_index

    cursor = len(operators)
    while cursor and isinstance(operators[cursor - 1], DiagonalLinearOp):
        cursor -= 1
    output_diagonals = tuple(operators[cursor:])
    if not cursor or not isinstance(operators[cursor - 1], ImplicitConv2DOp):
        raise C2PlannerReject("post_add_shared_outer_missing")
    outer_index = cursor - 1
    outer = operators[outer_index]
    if outer_index < add_index:
        raise C2PlannerReject("common_add_not_between_convs")
    convs_after_add = sum(
        isinstance(value, ImplicitConv2DOp)
        for value in operators[add_index:outer_index]
    )
    if convs_after_add:
        raise C2PlannerReject("two_convs_same_segment_not_c2")
    post_add_diagonals = tuple(operators[add_index:outer_index])
    if any(
        not isinstance(value, DiagonalLinearOp)
        for value in post_add_diagonals
    ):
        raise C2PlannerReject("post_add_suffix_operator_barrier")

    outer_snapshot = _conv_snapshot(outer, name=f"term_{term_index}_outer")
    if int(outer.shape[0]) != n_out:
        raise C2PlannerReject("tail_output_shape_mismatch")
    for output_index, output_op in enumerate(output_diagonals):
        _, diagonal = _diagonal_snapshot(
            output_op,
            name=f"term_{term_index}_output_{output_index}",
        )
        if diagonal.size != n_out:
            raise C2PlannerReject("output_diagonal_shape")

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
            raise C2PlannerReject("outer_row_mask_shape")
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

    if identity_skip:
        prefix = tuple(operators[:add_index])
        return _ParsedTerm(
            term_index=term_index,
            term=term,
            common_add=common_add,
            occurrence_payload=occurrence,
            prefix=prefix,
            identity_skip=True,
            inner=None,
            pre_add_diagonals=(),
            post_add_diagonals=post_add_diagonals,
            outer=outer,
            output_diagonals=output_diagonals,
            selected_rows=selected,
            active_selected_rows=active_selected,
            sigma=None,
            inner_snapshot=None,
            post_add_snapshots=tuple(post_snapshots),
            outer_snapshot=outer_snapshot,
            stable_payload=None,
            content_key=None,
            representative_key=None,
        )

    segment = operators[previous_index:add_index]
    if (
        len(segment) < 2
        or not isinstance(segment[0], ImplicitConv2DOp)
        or any(not isinstance(value, DiagonalLinearOp) for value in segment[1:])
    ):
        raise C2PlannerReject("nonempty_branch_without_complete_chain")
    inner = segment[0]
    pre_add_diagonals = tuple(segment[1:])
    inner_index = previous_index
    for older_mark, _, mark_index in lineage[:-1]:
        if inner_index < mark_index <= outer_index:
            raise C2PlannerReject(
                f"core_contains_{older_mark.kind.lower()}_cut"
            )
    if tuple(inner.output_shape) != tuple(outer.input_shape):
        raise C2PlannerReject("tail_intermediate_shape_mismatch")

    inner_snapshot = _conv_snapshot(inner, name=f"term_{term_index}_inner")
    sigma = np.ones(int(inner.output_shape[1]), dtype=np.float64)
    diagonal_snapshots: list[bytes] = []
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
            raise C2PlannerReject("middle_scale_product_nonfinite")
        diagonal_snapshots.append(snapshot)

    sigma_payload = _typed_array_payload(
        b"S2", np.asarray(sigma, dtype="<f8", order="C")
    )
    stable_payload = b"S0C2D1" + b"".join(
        len(piece).to_bytes(8, "big") + piece
        for piece in (inner_snapshot, sigma_payload, outer_snapshot)
    )
    content_key = (
        C2_DESCRIPTOR_TAG,
        hashlib.sha256(stable_payload).hexdigest(),
    )
    representative_key = b"".join(
        len(piece).to_bytes(8, "big") + piece
        for piece in (inner_snapshot, *diagonal_snapshots, outer_snapshot)
    )
    return _ParsedTerm(
        term_index=term_index,
        term=term,
        common_add=common_add,
        occurrence_payload=occurrence,
        prefix=tuple(operators[:inner_index]),
        identity_skip=False,
        inner=inner,
        pre_add_diagonals=pre_add_diagonals,
        post_add_diagonals=post_add_diagonals,
        outer=outer,
        output_diagonals=output_diagonals,
        selected_rows=selected,
        active_selected_rows=active_selected,
        sigma=sigma,
        inner_snapshot=inner_snapshot,
        post_add_snapshots=tuple(post_snapshots),
        outer_snapshot=outer_snapshot,
        stable_payload=stable_payload,
        content_key=content_key,
        representative_key=representative_key,
    )


def _require_shared_post_add_suffix(parsed: tuple[_ParsedTerm, ...]) -> None:
    reference = parsed[0]
    for item in parsed[1:]:
        # Object identity proves one graph event; fresh snapshots additionally
        # prevent a stale constructor key from standing in for current value.
        if item.outer is not reference.outer:
            raise C2PlannerReject("shared_outer_occurrence_mismatch")
        if item.outer_snapshot != reference.outer_snapshot:
            raise C2PlannerReject("shared_outer_snapshot_mismatch")
        if len(item.post_add_diagonals) != len(reference.post_add_diagonals):
            raise C2PlannerReject("post_add_suffix_length_mismatch")
        if any(
            left is not right
            for left, right in zip(
                item.post_add_diagonals, reference.post_add_diagonals
            )
        ):
            raise C2PlannerReject("post_add_suffix_occurrence_mismatch")
        if item.post_add_snapshots != reference.post_add_snapshots:
            raise C2PlannerReject("post_add_suffix_snapshot_mismatch")


def _reject_decision(
    expr: LineageAffineExprView,
    terms: tuple[LineageAffineTermView, ...],
    bias: object,
    reason: str,
) -> C2PlanDecision:
    return C2PlanDecision(
        accepted=False,
        reason=reason,
        expression=expr,
        terms=terms,
        bias=bias,
        plan=None,
    )


def plan_s0_c2_residual_distributive_lineage(
    expr: LineageAffineExprView,
    output_support,
    *,
    budget: PlannerBudget = PlannerBudget(),
) -> C2PlanDecision:
    """Return the unique pure C2 plan or the untouched expression by identity."""

    original_terms = expr.terms
    original_bias = expr.bias
    try:
        if not isinstance(original_terms, tuple):
            raise C2PlannerReject("expression_terms_not_tuple")
        if not original_terms:
            raise C2PlannerReject("expression_terms_empty")
        n_out = _strict_nonnegative_int(expr.n_out, name="n_out")
        frame_payload = _frame_payload(
            expr.frame_id, name="expression_frame_id"
        )
        try:
            support = _c1_normalize_support(output_support, n_out)
        except TailPlannerReject as exc:
            raise C2PlannerReject(str(exc)) from exc
        if not support:
            raise C2PlannerReject("empty_output_support")

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
        occurrence = parsed[0].occurrence_payload
        if any(item.occurrence_payload != occurrence for item in parsed[1:]):
            raise C2PlannerReject("latest_common_add_occurrence_mismatch")
        _require_shared_post_add_suffix(parsed)

        identity_indices = tuple(
            item.term_index for item in parsed if item.identity_skip
        )
        complete = tuple(item for item in parsed if not item.identity_skip)
        if not complete:
            raise C2PlannerReject("no_complete_c2_branch")
        zero_support = tuple(
            item.term_index for item in complete if not item.active_selected_rows
        )
        eligible = tuple(item for item in complete if item.active_selected_rows)
        if not eligible:
            raise C2PlannerReject("no_nonzero_fusible_branch")

        grouped: dict[bytes, list[_ParsedTerm]] = {}
        digest_payloads: dict[str, bytes] = {}
        for item in eligible:
            assert item.stable_payload is not None
            assert item.content_key is not None
            digest = item.content_key[1]
            prior = digest_payloads.get(digest)
            if prior is not None and prior != item.stable_payload:
                raise C2PlannerReject("descriptor_content_hash_collision")
            digest_payloads[digest] = item.stable_payload
            grouped.setdefault(item.stable_payload, []).append(item)

        requests: list[DescriptorRequest] = []
        uses: list[C2BranchUse] = []
        for _, group in sorted(
            grouped.items(),
            key=lambda pair: (hashlib.sha256(pair[0]).hexdigest(), pair[0]),
        ):
            representative = min(
                group,
                key=lambda item: item.representative_key or b"",
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
                    term_indices=tuple(sorted(item.term_index for item in group)),
                    use_count=len(group),
                    estimate=estimate,
                )
            )
            for item in group:
                assert item.inner is not None
                assert item.content_key is not None
                uses.append(
                    C2BranchUse(
                        term_index=item.term_index,
                        common_add=item.common_add,
                        prefix=item.prefix,
                        inner=item.inner,
                        pre_add_diagonals=item.pre_add_diagonals,
                        post_add_diagonals=item.post_add_diagonals,
                        outer=item.outer,
                        output_diagonals=item.output_diagonals,
                        selected_rows=item.selected_rows,
                        descriptor_content_key=item.content_key,
                    )
                )

        request_tuple = tuple(requests)
        try:
            reservation = _c1_reserve_budget(request_tuple, budget)
        except TailPlannerReject as exc:
            raise C2PlannerReject(str(exc)) from exc
        plan = C2PureLineagePlan(
            rule_id=C2_RULE_ID,
            expression=expr,
            original_terms=original_terms,
            bias=original_bias,
            frame_id=expr.frame_id,
            common_add_occurrence_payload=occurrence,
            common_add_marks=tuple(item.common_add for item in parsed),
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
        return C2PlanDecision(
            accepted=True,
            reason="planned_pure_lineage_only",
            expression=expr,
            terms=original_terms,
            bias=original_bias,
            plan=plan,
        )
    except C2PlannerReject as exc:
        return _reject_decision(expr, original_terms, original_bias, str(exc))
    except Exception as exc:
        return _reject_decision(
            expr,
            original_terms,
            original_bias,
            f"planner_failed_{type(exc).__name__}",
        )


__all__ = [
    "BOUNDARY_KINDS",
    "BoundaryMark",
    "C2BranchUse",
    "C2_DESCRIPTOR_TAG",
    "C2_NO_CLAIMS",
    "C2PlanDecision",
    "C2PlannerReject",
    "C2PureLineagePlan",
    "C2_RULE_ID",
    "LineageAffineExprView",
    "LineageAffineTermView",
    "PlannerBudget",
    "plan_s0_c2_residual_distributive_lineage",
]
