"""Isolated pure tail planner for S0-C1 composed Conv descriptors.

This module is not imported by ACT.  It models the transaction that a future
``SparseHZAffineExpr`` materializer would perform without mutating an
expression or compiling anything during planning.  Only the contiguous tail

    ImplicitConv2DOp -> channel-stationary DiagonalLinearOp(s)
        -> ImplicitConv2DOp -> optional DiagonalLinearOp(s)

is recognized.  Prefix operators and separate residual terms remain separate;
unsupported operators therefore form hard barriers rather than being crossed.
Unique descriptors are budgeted and sorted by stable content before a caller
may compile them.  A rejection returns the original term tuple and bias object
by identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator
import struct
from typing import Callable

import numpy as np

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)


MIB = 1024 * 1024
GIB = 1024 * MIB


class TailPlannerReject(ValueError):
    """A deterministic fail-closed planner rejection."""


@dataclass(frozen=True)
class AffineTermView:
    """Lightweight stand-in for ``SparseHZAffineTerm``."""

    source: object
    operators: tuple[object, ...]


@dataclass(frozen=True)
class AffineExprView:
    """Lightweight immutable expression view; bias identity is significant."""

    terms: tuple[AffineTermView, ...]
    bias: object
    n_out: int


@dataclass(frozen=True)
class PlannerBudget:
    """Frozen transaction capacities reserved before any compilation."""

    max_unique_descriptors: int = 64
    max_descriptor_contraction_products: int = 200_000_000
    max_descriptor_coefficient_entries: int = 2_000_000
    max_descriptor_resident_bytes: int = 64 * MIB
    max_descriptor_result_nnz: int = 64_000_000
    max_transaction_contraction_products: int = 256_000_000
    max_transaction_total_work: int = 256_000_000
    max_transaction_coefficient_entries: int = 2_000_000
    max_transaction_resident_bytes: int = 64 * MIB
    max_transaction_transient_bytes: int = GIB
    max_transaction_result_nnz: int = 64_000_000


@dataclass(frozen=True)
class DescriptorEstimate:
    """Conservative allocation/work reservation for one unique descriptor."""

    selected_rows: int
    active_selected_rows: int
    contraction_products: int
    emission_contributions: int
    total_work: int
    coefficient_entries: int
    resident_bytes: int
    transient_bytes: int
    result_nnz_upper: int


@dataclass(frozen=True)
class BudgetReservation:
    unique_descriptors: int
    contraction_products: int
    total_work: int
    coefficient_entries: int
    resident_bytes: int
    transient_bytes: int
    result_nnz_upper: int


@dataclass(frozen=True)
class DescriptorRequest:
    """One stable, unique compile request shared by one or more terms."""

    content_key: tuple[str, str]
    inner: ImplicitConv2DOp
    middle_ops: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    selected_rows: tuple[int, ...]
    term_indices: tuple[int, ...]
    use_count: int
    estimate: DescriptorEstimate


@dataclass(frozen=True)
class _TailUse:
    term_index: int
    prefix: tuple[object, ...]
    output_diagonals: tuple[DiagonalLinearOp, ...]
    descriptor_content_key: tuple[str, str]


@dataclass(frozen=True)
class TailPlan:
    """An immutable all-or-nothing rewrite plan."""

    original_terms: tuple[AffineTermView, ...]
    bias: object
    requests: tuple[DescriptorRequest, ...]
    uses: tuple[_TailUse, ...]
    reservation: BudgetReservation


@dataclass(frozen=True)
class PlanDecision:
    accepted: bool
    reason: str
    terms: tuple[AffineTermView, ...]
    bias: object
    plan: TailPlan | None


@dataclass(frozen=True)
class RewriteDecision:
    accepted: bool
    reason: str
    terms: tuple[AffineTermView, ...]
    bias: object
    plan: TailPlan | None
    compiled_unique: tuple[object, ...]


@dataclass(frozen=True)
class _MatchedTail:
    term_index: int
    prefix: tuple[object, ...]
    inner: ImplicitConv2DOp
    middle_ops: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    output_diagonals: tuple[DiagonalLinearOp, ...]
    selected_rows: tuple[int, ...]
    stable_payload: bytes
    content_key: tuple[str, str]
    representative_key: bytes


def _strict_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TailPlannerReject(f"{name}_not_integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise TailPlannerReject(f"{name}_not_integer") from exc
    if result < 0:
        raise TailPlannerReject(f"{name}_negative")
    return result


def _encode_stable(value) -> bytes:
    """Canonical encoding for exact-linear-operator content keys."""

    if value is None:
        return b"N"
    if isinstance(value, (bool, np.bool_)):
        return b"B1" if bool(value) else b"B0"
    if isinstance(value, (int, np.integer)):
        payload = str(int(value)).encode("ascii")
        return b"I" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, (float, np.floating)):
        return b"F" + struct.pack(">d", float(value))
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"S" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, bytes):
        return b"Y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, (tuple, list)):
        pieces = [_encode_stable(item) for item in value]
        return b"T" + len(pieces).to_bytes(8, "big") + b"".join(
            len(piece).to_bytes(8, "big") + piece for piece in pieces
        )
    raise TailPlannerReject("operator_content_key_not_stable")


def _operator_key(operator_value: object) -> bytes:
    if not hasattr(operator_value, "content_key"):
        raise TailPlannerReject("operator_missing_content_key")
    return _encode_stable(operator_value.content_key)


def _normalize_support(support, size: int) -> tuple[int, ...]:
    raw = np.asarray(support)
    if raw.dtype.kind == "b":
        mask = raw.reshape(-1)
        if mask.size != size:
            raise TailPlannerReject("output_support_mask_shape")
        return tuple(int(value) for value in np.flatnonzero(mask))
    if raw.dtype.kind not in "iu":
        raise TailPlannerReject("output_support_not_integer")
    rows = np.asarray(raw, dtype=np.int64).reshape(-1)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= size):
        raise TailPlannerReject("output_support_out_of_range")
    return tuple(sorted({int(value) for value in rows}))


def _diagonal_payload(op: DiagonalLinearOp, *, name: str) -> np.ndarray:
    if not isinstance(op, DiagonalLinearOp) or not hasattr(op, "_diagonal"):
        raise TailPlannerReject(f"{name}_not_diagonal")
    diagonal = np.asarray(op._diagonal, dtype=np.float64).reshape(-1)
    if diagonal.size != int(op.shape[0]) or op.shape[0] != op.shape[1]:
        raise TailPlannerReject(f"{name}_shape")
    if not np.all(np.isfinite(diagonal)):
        raise TailPlannerReject(f"{name}_nonfinite")
    return diagonal


def _channel_stationary_vector(
    op: DiagonalLinearOp,
    shape: tuple[int, int, int, int],
    *,
    name: str,
) -> np.ndarray:
    diagonal = _diagonal_payload(op, name=name)
    batch, channels, height, width = (int(value) for value in shape)
    if diagonal.size != batch * channels * height * width:
        raise TailPlannerReject(f"{name}_shape")
    full = diagonal.reshape(batch, channels, height, width)
    channel = np.array(full[0, :, 0, 0], dtype=np.float64, copy=True)
    expected = np.broadcast_to(channel.reshape(1, channels, 1, 1), full.shape)
    if not np.array_equal(full, expected):
        raise TailPlannerReject(f"{name}_not_channel_stationary")
    return channel


def _backpropagate_output_support(
    selected_rows: tuple[int, ...],
    output_diagonals: tuple[DiagonalLinearOp, ...],
    *,
    output_size: int,
) -> tuple[int, ...]:
    active = selected_rows
    for reverse_index, diagonal_op in enumerate(reversed(output_diagonals)):
        diagonal = _diagonal_payload(
            diagonal_op, name=f"output_diagonal_{reverse_index}"
        )
        if diagonal.size != output_size:
            raise TailPlannerReject("output_diagonal_shape")
        active = tuple(row for row in active if diagonal[row] != 0.0)
    return active


def _match_tail(
    term: AffineTermView,
    *,
    term_index: int,
    output_support: tuple[int, ...],
    n_out: int,
) -> _MatchedTail | None:
    operators = term.operators
    if not isinstance(operators, tuple):
        raise TailPlannerReject("term_operators_not_tuple")

    cursor = len(operators)
    while cursor and isinstance(operators[cursor - 1], DiagonalLinearOp):
        cursor -= 1
    output_diagonals = tuple(operators[cursor:])
    if not cursor or not isinstance(operators[cursor - 1], ImplicitConv2DOp):
        return None
    outer_index = cursor - 1
    outer = operators[outer_index]

    cursor = outer_index
    while cursor and isinstance(operators[cursor - 1], DiagonalLinearOp):
        cursor -= 1
    middle_ops = tuple(operators[cursor:outer_index])
    if not middle_ops or not cursor:
        return None
    if not isinstance(operators[cursor - 1], ImplicitConv2DOp):
        return None
    inner_index = cursor - 1
    inner = operators[inner_index]
    prefix = tuple(operators[:inner_index])

    if tuple(inner.output_shape) != tuple(outer.input_shape):
        raise TailPlannerReject("tail_intermediate_shape_mismatch")
    if int(outer.shape[0]) != n_out:
        raise TailPlannerReject("tail_output_shape_mismatch")
    width = int(inner.shape[0])
    for index, middle in enumerate(middle_ops):
        if tuple(middle.shape) != (width, width):
            raise TailPlannerReject(f"middle_{index}_shape")
    for index, output_diagonal in enumerate(output_diagonals):
        if tuple(output_diagonal.shape) != (n_out, n_out):
            raise TailPlannerReject(f"output_{index}_shape")

    sigma = np.ones(int(inner.output_shape[1]), dtype=np.float64)
    for index, middle in enumerate(middle_ops):
        scale = _channel_stationary_vector(
            middle, inner.output_shape, name=f"middle_{index}"
        )
        with np.errstate(over="ignore", invalid="ignore"):
            sigma = sigma * scale
        if not np.all(np.isfinite(sigma)):
            raise TailPlannerReject("middle_scale_product_nonfinite")

    selected = _backpropagate_output_support(
        output_support,
        output_diagonals,
        output_size=n_out,
    )
    if not selected:
        return None

    sigma_le = np.asarray(sigma, dtype="<f8", order="C")
    stable_payload = b"s0_c1_tail_descriptor_v1\0" + b"".join(
        (
            len(_operator_key(inner)).to_bytes(8, "big"),
            _operator_key(inner),
            len(_operator_key(outer)).to_bytes(8, "big"),
            _operator_key(outer),
            len(sigma_le.tobytes()).to_bytes(8, "big"),
            sigma_le.tobytes(),
        )
    )
    digest = hashlib.sha256(stable_payload).hexdigest()
    representative_key = b"".join(
        len(key).to_bytes(8, "big") + key
        for key in (
            _operator_key(inner),
            *(_operator_key(op) for op in middle_ops),
            _operator_key(outer),
        )
    )
    return _MatchedTail(
        term_index=term_index,
        prefix=prefix,
        inner=inner,
        middle_ops=middle_ops,
        outer=outer,
        output_diagonals=output_diagonals,
        selected_rows=selected,
        stable_payload=stable_payload,
        content_key=("s0_c1_tail_descriptor_v1", digest),
        representative_key=representative_key,
    )


def _reachable_input_pairs(inner: ImplicitConv2DOp, outer: ImplicitConv2DOp):
    input_channels = int(inner.input_shape[1])
    middle_channels = int(inner.output_shape[1])
    output_channels = int(outer.output_shape[1])
    inner_groups = int(inner._groups)
    outer_groups = int(outer._groups)
    inner_output_per_group = middle_channels // inner_groups
    inner_input_per_group = input_channels // inner_groups
    outer_output_per_group = output_channels // outer_groups
    outer_input_per_group = middle_channels // outer_groups
    counts: list[int] = []
    for output_channel in range(output_channels):
        outer_group = output_channel // outer_output_per_group
        middle_start = outer_group * outer_input_per_group
        reachable: set[int] = set()
        for middle_channel in range(
            middle_start, middle_start + outer_input_per_group
        ):
            inner_group = middle_channel // inner_output_per_group
            input_start = inner_group * inner_input_per_group
            reachable.update(range(input_start, input_start + inner_input_per_group))
        counts.append(len(reachable))
    return tuple(counts)


def _estimate_descriptor(
    inner: ImplicitConv2DOp,
    outer: ImplicitConv2DOp,
    selected_rows: tuple[int, ...],
) -> DescriptorEstimate:
    input_channels = int(inner.input_shape[1])
    middle_channels = int(inner.output_shape[1])
    output_channels = int(outer.output_shape[1])
    inner_taps = int(inner._kernel.shape[2] * inner._kernel.shape[3])
    outer_taps = int(outer._kernel.shape[2] * outer._kernel.shape[3])
    inner_input_per_group = int(inner._kernel.shape[1])
    outer_input_per_group = int(outer._kernel.shape[1])
    pair_counts = _reachable_input_pairs(inner, outer)
    pair_count = sum(pair_counts)

    contraction_products = (
        inner_taps
        * outer_taps
        * output_channels
        * outer_input_per_group
        * inner_input_per_group
    )
    coefficient_entries = inner_taps * outer_taps * pair_count
    outer_mask = (
        None
        if outer._row_mask is None
        else np.asarray(outer._row_mask, dtype=bool).reshape(-1)
    )
    _, _, output_h, output_w = outer.output_shape
    output_spatial = int(output_h * output_w)
    output_per_batch = output_channels * output_spatial
    active_rows = 0
    emission = 0
    for row in selected_rows:
        if outer_mask is not None and not bool(outer_mask[row]):
            continue
        active_rows += 1
        _, within_batch = divmod(int(row), output_per_batch)
        output_channel, _ = divmod(within_batch, output_spatial)
        emission += pair_counts[output_channel] * inner_taps * outer_taps

    result_nnz_upper = min(
        emission,
        active_rows * math.prod(inner.input_shape),
    )
    resident_bytes = (
        coefficient_entries * 8
        + pair_count * 2 * np.dtype(np.int64).itemsize
        + (output_channels + 1) * np.dtype(np.int64).itemsize
        + (0 if outer_mask is None else int(outer_mask.nbytes))
    )
    channel_workspace_bytes = (
        inner_taps * middle_channels * input_channels * 8
        + 2 * outer_taps * output_channels * middle_channels * 8
        + output_channels * input_channels * 8
    )
    transient_bytes = resident_bytes + channel_workspace_bytes
    return DescriptorEstimate(
        selected_rows=len(selected_rows),
        active_selected_rows=active_rows,
        contraction_products=int(contraction_products),
        emission_contributions=int(emission),
        total_work=int(contraction_products + emission),
        coefficient_entries=int(coefficient_entries),
        resident_bytes=int(resident_bytes),
        transient_bytes=int(transient_bytes),
        result_nnz_upper=int(result_nnz_upper),
    )


def _validated_budget(budget: PlannerBudget) -> PlannerBudget:
    for field in budget.__dataclass_fields__:
        _strict_nonnegative_int(getattr(budget, field), name=field)
    return budget


def _reserve_budget(
    requests: tuple[DescriptorRequest, ...], budget: PlannerBudget
) -> BudgetReservation:
    budget = _validated_budget(budget)
    if len(requests) > budget.max_unique_descriptors:
        raise TailPlannerReject("unique_descriptor_limit")
    for request in requests:
        estimate = request.estimate
        if (
            estimate.contraction_products
            > budget.max_descriptor_contraction_products
        ):
            raise TailPlannerReject("descriptor_contraction_product_limit")
        if estimate.coefficient_entries > budget.max_descriptor_coefficient_entries:
            raise TailPlannerReject("descriptor_coefficient_entry_limit")
        if estimate.resident_bytes > budget.max_descriptor_resident_bytes:
            raise TailPlannerReject("descriptor_resident_byte_limit")
        if estimate.result_nnz_upper > budget.max_descriptor_result_nnz:
            raise TailPlannerReject("descriptor_result_nnz_limit")

    reservation = BudgetReservation(
        unique_descriptors=len(requests),
        contraction_products=sum(
            request.estimate.contraction_products for request in requests
        ),
        total_work=sum(request.estimate.total_work for request in requests),
        coefficient_entries=sum(
            request.estimate.coefficient_entries for request in requests
        ),
        resident_bytes=sum(request.estimate.resident_bytes for request in requests),
        transient_bytes=sum(
            request.estimate.transient_bytes for request in requests
        ),
        result_nnz_upper=sum(
            request.estimate.result_nnz_upper for request in requests
        ),
    )
    cumulative_gates = (
        (
            reservation.contraction_products,
            budget.max_transaction_contraction_products,
            "transaction_contraction_product_limit",
        ),
        (
            reservation.total_work,
            budget.max_transaction_total_work,
            "transaction_work_limit",
        ),
        (
            reservation.coefficient_entries,
            budget.max_transaction_coefficient_entries,
            "transaction_coefficient_entry_limit",
        ),
        (
            reservation.resident_bytes,
            budget.max_transaction_resident_bytes,
            "transaction_resident_byte_limit",
        ),
        (
            reservation.transient_bytes,
            budget.max_transaction_transient_bytes,
            "transaction_transient_byte_limit",
        ),
        (
            reservation.result_nnz_upper,
            budget.max_transaction_result_nnz,
            "transaction_result_nnz_limit",
        ),
    )
    for value, limit, reason in cumulative_gates:
        if value > limit:
            raise TailPlannerReject(reason)
    return reservation


def plan_composed_tails(
    expr: AffineExprView,
    output_support,
    *,
    budget: PlannerBudget = PlannerBudget(),
) -> PlanDecision:
    """Purely recognize, group, order, and reserve an all-or-nothing plan."""

    original_terms = expr.terms
    original_bias = expr.bias
    try:
        if not isinstance(original_terms, tuple):
            raise TailPlannerReject("expression_terms_not_tuple")
        n_out = _strict_nonnegative_int(expr.n_out, name="n_out")
        support = _normalize_support(output_support, n_out)
        if not support:
            raise TailPlannerReject("empty_output_support")

        matches: list[_MatchedTail] = []
        for term_index, term in enumerate(original_terms):
            if not isinstance(term, AffineTermView):
                raise TailPlannerReject("unsupported_term_view")
            matched = _match_tail(
                term,
                term_index=term_index,
                output_support=support,
                n_out=n_out,
            )
            if matched is not None:
                matches.append(matched)
        if not matches:
            raise TailPlannerReject("no_eligible_tail")

        grouped: dict[bytes, list[_MatchedTail]] = {}
        digest_payloads: dict[str, bytes] = {}
        for matched in matches:
            digest = matched.content_key[1]
            previous_payload = digest_payloads.get(digest)
            if previous_payload is not None and previous_payload != matched.stable_payload:
                raise TailPlannerReject("descriptor_content_hash_collision")
            digest_payloads[digest] = matched.stable_payload
            grouped.setdefault(matched.stable_payload, []).append(matched)

        requests: list[DescriptorRequest] = []
        uses: list[_TailUse] = []
        ordered_groups = sorted(
            grouped.items(),
            key=lambda item: (hashlib.sha256(item[0]).hexdigest(), item[0]),
        )
        for _, group in ordered_groups:
            representative = min(group, key=lambda item: item.representative_key)
            selected_rows = tuple(
                sorted({row for matched in group for row in matched.selected_rows})
            )
            estimate = _estimate_descriptor(
                representative.inner,
                representative.outer,
                selected_rows,
            )
            requests.append(
                DescriptorRequest(
                    content_key=representative.content_key,
                    inner=representative.inner,
                    middle_ops=representative.middle_ops,
                    outer=representative.outer,
                    selected_rows=selected_rows,
                    term_indices=tuple(sorted(item.term_index for item in group)),
                    use_count=len(group),
                    estimate=estimate,
                )
            )
            for matched in group:
                uses.append(
                    _TailUse(
                        term_index=matched.term_index,
                        prefix=matched.prefix,
                        output_diagonals=matched.output_diagonals,
                        descriptor_content_key=matched.content_key,
                    )
                )
        request_tuple = tuple(requests)
        reservation = _reserve_budget(request_tuple, budget)
        plan = TailPlan(
            original_terms=original_terms,
            bias=original_bias,
            requests=request_tuple,
            uses=tuple(sorted(uses, key=lambda item: item.term_index)),
            reservation=reservation,
        )
        return PlanDecision(
            accepted=True,
            reason="planned",
            terms=original_terms,
            bias=original_bias,
            plan=plan,
        )
    except TailPlannerReject as exc:
        return PlanDecision(
            accepted=False,
            reason=str(exc),
            terms=original_terms,
            bias=original_bias,
            plan=None,
        )


def execute_tail_plan(
    decision: PlanDecision,
    compiler: Callable[[DescriptorRequest], object],
) -> RewriteDecision:
    """Compile every unique request once, publishing terms only on success."""

    if not decision.accepted or decision.plan is None:
        return RewriteDecision(
            accepted=False,
            reason=decision.reason,
            terms=decision.terms,
            bias=decision.bias,
            plan=None,
            compiled_unique=(),
        )
    plan = decision.plan
    compiled: dict[tuple[str, str], object] = {}
    compiled_order: list[object] = []
    try:
        for request in plan.requests:
            if request.content_key in compiled:
                raise TailPlannerReject("duplicate_compile_request")
            operator_value = compiler(request)
            if operator_value is None:
                raise TailPlannerReject("compiler_returned_none")
            compiled[request.content_key] = operator_value
            compiled_order.append(operator_value)

        uses = {use.term_index: use for use in plan.uses}
        rewritten: list[AffineTermView] = []
        for term_index, term in enumerate(plan.original_terms):
            use = uses.get(term_index)
            if use is None:
                rewritten.append(term)
                continue
            descriptor = compiled[use.descriptor_content_key]
            rewritten.append(
                AffineTermView(
                    source=term.source,
                    operators=(
                        *use.prefix,
                        descriptor,
                        *use.output_diagonals,
                    ),
                )
            )
        return RewriteDecision(
            accepted=True,
            reason="rewritten",
            terms=tuple(rewritten),
            bias=plan.bias,
            plan=plan,
            compiled_unique=tuple(compiled_order),
        )
    except TailPlannerReject as exc:
        reason = str(exc)
    except Exception as exc:  # fail closed; never publish a partial tuple
        reason = f"compiler_failed_{type(exc).__name__}"
    return RewriteDecision(
        accepted=False,
        reason=reason,
        terms=plan.original_terms,
        bias=plan.bias,
        plan=plan,
        compiled_unique=(),
    )


__all__ = [
    "AffineExprView",
    "AffineTermView",
    "BudgetReservation",
    "DescriptorEstimate",
    "DescriptorRequest",
    "PlanDecision",
    "PlannerBudget",
    "RewriteDecision",
    "TailPlan",
    "TailPlannerReject",
    "execute_tail_plan",
    "plan_composed_tails",
]
