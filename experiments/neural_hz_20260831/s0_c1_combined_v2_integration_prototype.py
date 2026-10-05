"""Experiment-only atomic integration of the pure tail planner and V2.

The two existing isolated components cannot be connected by blindly passing
``DescriptorRequest.content_key`` into V2:

* the planner key trusts constructor-time operator keys, while hardened V2
  derives identity from current semantic payload snapshots;
* planner reservations are estimates, whereas V2 owns the consuming work
  transaction; and
* neither component exposes a savepoint that can undo several already
  committed unique descriptors in an existing transaction.

This minimal adapter therefore treats planner keys only as local plan handles,
recomputes the authoritative V2 identity for every matched term, freezes input
payloads into private clones, and compiles all requests in an unpublished
staging transaction.  The expression and staging transaction are published
only after every request and final identity guard succeeds.  A normal
rejection or ``Exception`` returns the original term tuple and bias by
identity and does not expose partial descriptors or consumed budget.
Process-control ``BaseException`` subclasses unwind the unpublished staging
transaction and are re-raised.

This module is not imported by ACT and performs no spatial Conv CSR expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Callable

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
import experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 as v2
from experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 import (
    ComposedConv2DStencilCandidateV2,
    FrozenV1Limits,
    GateRequestV2,
    TransactionSnapshot,
    V2Transaction,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    AffineExprView,
    AffineTermView,
    DescriptorRequest,
    PlanDecision,
    PlannerBudget,
    plan_composed_tails,
)


ADAPTER_GAPS = (
    "planner_handle_is_not_v2_content_key",
    "v2_has_no_public_key_only_snapshot_api",
    "v2_has_no_committed_transaction_savepoint_or_merge",
    "adapter_guard_and_snapshot_transients_are_not_charged",
    "actual_emission_materializer_not_integrated",
)

ADAPTER_NON_CLAIMS = (
    "whole_state_root_alias_not_proven",
    "old_expression_retention_not_accounted",
    "synthetic_physical_ledger_is_not_a_representation_result",
    "production_add_reshape_lineage_is_not_represented",
    "production_batch_two_plumbing_is_not_proven",
    "support_emission_is_prospective_not_committed",
)


class CombinedIntegrationReject(ValueError):
    """A deterministic adapter-level fail-closed rejection."""


@dataclass(frozen=True)
class CombinedPhysicalLedger:
    """Whole-state physical operands supplied by the synthetic caller."""

    reachable_before_bytes: int | None
    reachable_after_other_bytes: int | None


@dataclass(frozen=True)
class _FrozenRequest:
    planner_handle: tuple[str, str]
    expected_v2_content_key: tuple
    inner: ImplicitConv2DOp
    middle_ops: tuple[DiagonalLinearOp, ...]
    outer: ImplicitConv2DOp
    selected_rows: tuple[int, ...]
    term_indices: tuple[int, ...]


@dataclass(frozen=True)
class _TermRewriteMetadata:
    original_prefix: tuple[object, ...]
    original_output_diagonals: tuple[object, ...]


@dataclass(frozen=True)
class CombinedIntegrationDecision:
    accepted: bool
    reason: str
    expression: AffineExprView
    terms: tuple[AffineTermView, ...]
    bias: object
    planner_decision: PlanDecision | None
    planner_handles: tuple[tuple[str, str], ...]
    planned_selected_rows: tuple[tuple[int, ...], ...]
    prospective_emission_products: tuple[int, ...]
    compiled_unique: tuple[ComposedConv2DStencilCandidateV2, ...]
    actual_content_keys: tuple[tuple, ...]
    transaction: V2Transaction | None
    transaction_snapshot: TransactionSnapshot
    compile_attempts: int
    exception_type: str | None
    adapter_gaps: tuple[str, ...] = ADAPTER_GAPS
    adapter_non_claims: tuple[str, ...] = ADAPTER_NON_CLAIMS


def _raw_array_key(array: np.ndarray) -> tuple:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"combined_adapter_raw_array_v1\x00")
    dtype = contiguous.dtype.str.encode("ascii")
    digest.update(struct.pack(">I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">I", contiguous.ndim))
    for size in contiguous.shape:
        digest.update(struct.pack(">Q", int(size)))
    digest.update(contiguous.tobytes(order="C"))
    return (
        "combined_adapter_raw_array_v1",
        contiguous.dtype.str,
        tuple(int(v) for v in contiguous.shape),
        digest.digest(),
    )


def _object_guard(value: object) -> tuple:
    if isinstance(value, np.ndarray):
        return ("ndarray", id(value), _raw_array_key(value))
    if sp.issparse(value):
        matrix = value.tocsr().copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.eliminate_zeros()
        return (
            "sparse_matrix",
            id(value),
            tuple(int(v) for v in matrix.shape),
            _raw_array_key(matrix.indptr),
            _raw_array_key(matrix.indices),
            _raw_array_key(matrix.data),
        )
    return (
        "object_identity",
        type(value).__module__,
        type(value).__qualname__,
        id(value),
    )


def _operator_guard(operator_value: object) -> tuple:
    if isinstance(operator_value, ImplicitConv2DOp):
        payload = v2._conv_payload(operator_value, name="guard_conv")
        return ("implicit_conv2d", id(operator_value), payload[-1])
    if isinstance(operator_value, DiagonalLinearOp):
        if not hasattr(operator_value, "_diagonal"):
            raise CombinedIntegrationReject("diagonal_payload_missing")
        diagonal = np.asarray(operator_value._diagonal)
        return (
            "diagonal",
            id(operator_value),
            v2._typed_float64_key(diagonal),
        )
    return _object_guard(operator_value)


def _source_guard(source: object) -> tuple:
    """Guard predicate-bearing source identity without interpreting it.

    Production sources carry the value map, equality/inequality predicates,
    frame and exactness bit.  The adapter never rewrites those fields, but it
    must also not publish a descriptor after any of them is repointed or
    mutated during planning.  Absence is canonical for lightweight test
    sources and does not make a source ineligible.
    """

    guarded_fields = []
    for name in (
        "frame",
        "frame_id",
        "exact",
        "predicates",
        "c",
        "Gc",
        "Gb",
        "Ac",
        "Ab",
        "b",
        "Auc",
        "Aub",
        "ub",
    ):
        if not hasattr(source, name):
            guarded_fields.append((name, "absent"))
            continue
        value = getattr(source, name)
        guarded_fields.append((name, "present", id(value), _object_guard(value)))
    return (id(source), tuple(guarded_fields))


def _expression_guard(expr: AffineExprView) -> tuple:
    if not isinstance(expr, AffineExprView):
        raise CombinedIntegrationReject("expression_view_type")
    if not isinstance(expr.terms, tuple):
        raise CombinedIntegrationReject("expression_terms_not_tuple")
    guarded_terms = []
    for term in expr.terms:
        if not isinstance(term, AffineTermView):
            raise CombinedIntegrationReject("term_view_type")
        if not isinstance(term.operators, tuple):
            raise CombinedIntegrationReject("term_operators_not_tuple")
        guarded_terms.append(
            (
                id(term),
                _source_guard(term.source),
                id(term.operators),
                tuple(_operator_guard(op) for op in term.operators),
            )
        )
    return (
        id(expr.terms),
        tuple(guarded_terms),
        id(expr.bias),
        _object_guard(expr.bias),
        int(expr.n_out),
    )


def _require_guard(expr: AffineExprView, expected: tuple, *, reason: str) -> None:
    if _expression_guard(expr) != expected:
        raise CombinedIntegrationReject(reason)


def _tail_core(term: AffineTermView):
    """Repeat only the planner's type-level tail split for identity audit."""

    operators = term.operators
    cursor = len(operators)
    while cursor and isinstance(operators[cursor - 1], DiagonalLinearOp):
        cursor -= 1
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
    return operators[cursor - 1], middle_ops, outer


def _local_diagonal_shadow(operator_value: object):
    """Normalize only an exact affine-tail diagonal; otherwise be a barrier."""

    if isinstance(operator_value, DiagonalLinearOp):
        return operator_value
    if not sp.issparse(operator_value):
        return None
    if operator_value.ndim != 2 or operator_value.dtype.kind not in "biuf":
        return None
    matrix = operator_value.tocsr().astype(np.float64, copy=True)
    if matrix.shape[0] != matrix.shape[1]:
        return None
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        return None
    rows = np.repeat(
        np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr)
    )
    if matrix.nnz and np.any(rows != matrix.indices):
        return None
    return DiagonalLinearOp(np.asarray(matrix.diagonal(), dtype=np.float64))


def _normalize_term_tail(term: AffineTermView):
    """Return a planner-only shadow plus original rewrite identities."""

    operators = term.operators
    cursor = len(operators)
    output_pairs: list[tuple[object, DiagonalLinearOp]] = []
    while cursor:
        shadow = _local_diagonal_shadow(operators[cursor - 1])
        if shadow is None:
            break
        output_pairs.append((operators[cursor - 1], shadow))
        cursor -= 1
    output_pairs.reverse()
    if not cursor or not isinstance(operators[cursor - 1], ImplicitConv2DOp):
        return term, None
    outer_index = cursor - 1

    cursor = outer_index
    middle_pairs: list[tuple[object, DiagonalLinearOp]] = []
    while cursor:
        shadow = _local_diagonal_shadow(operators[cursor - 1])
        if shadow is None:
            break
        middle_pairs.append((operators[cursor - 1], shadow))
        cursor -= 1
    middle_pairs.reverse()
    if not middle_pairs or not cursor or not isinstance(
        operators[cursor - 1], ImplicitConv2DOp
    ):
        return term, None
    inner_index = cursor - 1
    original_prefix = tuple(operators[:inner_index])
    original_outputs = tuple(pair[0] for pair in output_pairs)
    shadow_operators = (
        *original_prefix,
        operators[inner_index],
        *(pair[1] for pair in middle_pairs),
        operators[outer_index],
        *(pair[1] for pair in output_pairs),
    )
    shadow_term = AffineTermView(
        source=term.source,
        operators=tuple(shadow_operators),
    )
    metadata = _TermRewriteMetadata(
        original_prefix=original_prefix,
        original_output_diagonals=original_outputs,
    )
    return shadow_term, metadata


def _normalize_expression_for_planner(expr: AffineExprView):
    shadow_terms: list[AffineTermView] = []
    metadata: list[_TermRewriteMetadata | None] = []
    for term in expr.terms:
        shadow, rewrite = _normalize_term_tail(term)
        shadow_terms.append(shadow)
        metadata.append(rewrite)
    return (
        AffineExprView(
            terms=tuple(shadow_terms),
            bias=expr.bias,
            n_out=expr.n_out,
        ),
        tuple(metadata),
    )


def _semantic_key_and_payload(inner, middle_ops, outer):
    """Build the exact hardened-V2 key from current semantic snapshots."""

    inner_payload = v2._conv_payload(inner, name="adapter_inner")
    outer_payload = v2._conv_payload(outer, name="adapter_outer")
    (
        inner_kernel,
        input_shape,
        middle_shape,
        inner_stride,
        inner_padding,
        inner_dilation,
        inner_groups,
        inner_mask,
        inner_snapshot_key,
    ) = inner_payload
    (
        outer_kernel,
        outer_input_shape,
        output_shape,
        outer_stride,
        outer_padding,
        outer_dilation,
        outer_groups,
        outer_mask,
        outer_snapshot_key,
    ) = outer_payload
    if tuple(middle_shape) != tuple(outer_input_shape):
        raise CombinedIntegrationReject("adapter_intermediate_shape_mismatch")

    middle_clones: list[DiagonalLinearOp] = []
    sigma = np.ones(int(middle_shape[1]), dtype=np.float64)
    for index, middle in enumerate(middle_ops):
        if not isinstance(middle, DiagonalLinearOp) or not hasattr(
            middle, "_diagonal"
        ):
            raise CombinedIntegrationReject(f"adapter_middle_{index}_type")
        diagonal = np.array(
            middle._diagonal, dtype=np.float64, order="C", copy=True
        )
        clone = DiagonalLinearOp(diagonal)
        middle_clones.append(clone)
        scale = v2._stationary_channel_vector(
            clone,
            middle_shape,
            name=f"adapter_middle_{index}",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            sigma = np.multiply(sigma, scale)
        if not np.all(np.isfinite(sigma)):
            raise CombinedIntegrationReject("adapter_middle_product_nonfinite")
    if inner_mask is not None:
        channel_mask = v2._stationary_channel_vector(
            inner_mask,
            middle_shape,
            name="adapter_inner_row_mask",
        )
        if np.any((channel_mask != 0.0) & (channel_mask != 1.0)):
            raise CombinedIntegrationReject("adapter_inner_mask_not_binary")
        sigma = np.multiply(sigma, channel_mask)

    expected_key = (
        "composed_conv2d_stencil_candidate_v2",
        ("algorithm", v2.ALGORITHM_KEY),
        ("inner_operator", inner_snapshot_key),
        ("middle_scale", v2._typed_float64_key(sigma)),
        ("outer_operator", outer_snapshot_key),
    )
    inner_clone = ImplicitConv2DOp(
        np.array(inner_kernel, dtype=np.float64, order="C", copy=True),
        input_shape,
        stride=inner_stride,
        padding=inner_padding,
        dilation=inner_dilation,
        groups=inner_groups,
        row_mask=(
            None
            if inner_mask is None
            else np.array(inner_mask, dtype=bool, copy=True)
        ),
    )
    outer_clone = ImplicitConv2DOp(
        np.array(outer_kernel, dtype=np.float64, order="C", copy=True),
        outer_input_shape,
        stride=outer_stride,
        padding=outer_padding,
        dilation=outer_dilation,
        groups=outer_groups,
        row_mask=(
            None
            if outer_mask is None
            else np.array(outer_mask, dtype=bool, copy=True)
        ),
    )
    if inner_clone.output_shape != outer_clone.input_shape or (
        outer_clone.output_shape != output_shape
    ):
        raise CombinedIntegrationReject("adapter_clone_geometry_mismatch")
    return expected_key, inner_clone, tuple(middle_clones), outer_clone


def _freeze_requests(decision: PlanDecision) -> tuple[_FrozenRequest, ...]:
    if not decision.accepted or decision.plan is None:
        raise CombinedIntegrationReject("planner_plan_missing")
    plan = decision.plan
    request_by_handle = {request.content_key: request for request in plan.requests}
    if len(request_by_handle) != len(plan.requests):
        raise CombinedIntegrationReject("planner_duplicate_request_handle")
    uses_by_handle: dict[tuple[str, str], set[int]] = {}
    for use in plan.uses:
        uses_by_handle.setdefault(use.descriptor_content_key, set()).add(
            use.term_index
        )

    frozen: list[_FrozenRequest] = []
    for request in plan.requests:
        if not isinstance(request, DescriptorRequest):
            raise CombinedIntegrationReject("planner_request_type")
        actual_terms = uses_by_handle.get(request.content_key, set())
        if actual_terms != set(request.term_indices):
            raise CombinedIntegrationReject("planner_request_use_mismatch")
        expected_key, inner_clone, middle_clones, outer_clone = (
            _semantic_key_and_payload(
                request.inner,
                request.middle_ops,
                request.outer,
            )
        )
        # Reconcile every grouped term against current payload bytes.  This is
        # what prevents a stale constructor-time planner key from deduplicating
        # two ImplicitConv objects after one was mutated in place.
        for term_index in request.term_indices:
            if term_index < 0 or term_index >= len(plan.original_terms):
                raise CombinedIntegrationReject("planner_term_index_range")
            core = _tail_core(plan.original_terms[term_index])
            if core is None:
                raise CombinedIntegrationReject("planner_tail_no_longer_matches")
            term_key, _, _, _ = _semantic_key_and_payload(*core)
            if term_key != expected_key:
                raise CombinedIntegrationReject(
                    "planner_v2_snapshot_identity_mismatch"
                )
        frozen.append(
            _FrozenRequest(
                planner_handle=request.content_key,
                expected_v2_content_key=expected_key,
                inner=inner_clone,
                middle_ops=middle_clones,
                outer=outer_clone,
                selected_rows=tuple(int(v) for v in request.selected_rows),
                term_indices=tuple(int(v) for v in request.term_indices),
            )
        )
    return tuple(frozen)


def _empty_snapshot() -> TransactionSnapshot:
    return V2Transaction().snapshot()


def _failure(
    *,
    reason: str,
    expression: AffineExprView,
    terms,
    bias,
    compile_attempts: int,
    exception_type: str | None = None,
) -> CombinedIntegrationDecision:
    return CombinedIntegrationDecision(
        accepted=False,
        reason=reason,
        expression=expression,
        terms=terms,
        bias=bias,
        # A plan owns shadow terms and original operands.  Publishing it on
        # either path would keep replaced objects reachable and invalidate a
        # future whole-state physical comparison.
        planner_decision=None,
        planner_handles=(),
        planned_selected_rows=(),
        prospective_emission_products=(),
        compiled_unique=(),
        actual_content_keys=(),
        transaction=None,
        transaction_snapshot=_empty_snapshot(),
        compile_attempts=compile_attempts,
        exception_type=exception_type,
    )


def integrate_composed_tails_v2(
    expr: AffineExprView,
    output_support,
    *,
    planner_budget: PlannerBudget = PlannerBudget(),
    v1_limits: FrozenV1Limits = FrozenV1Limits(),
    physical: CombinedPhysicalLedger = CombinedPhysicalLedger(None, None),
    after_plan_hook: Callable[[], None] | None = None,
    before_compile_hook: Callable[[int], None] | None = None,
) -> CombinedIntegrationDecision:
    """Plan, freeze, compile, and publish one atomic synthetic transaction."""

    original_terms = expr.terms
    original_bias = expr.bias
    planner_decision = None
    compile_attempts = 0
    try:
        if not isinstance(physical, CombinedPhysicalLedger):
            raise CombinedIntegrationReject("physical_ledger_type")
        physical_before = (
            None
            if physical.reachable_before_bytes is None
            else v2._strict_nonnegative_int(
                physical.reachable_before_bytes,
                name="reachable_before_bytes",
            )
        )
        physical_after_base = (
            None
            if physical.reachable_after_other_bytes is None
            else v2._strict_nonnegative_int(
                physical.reachable_after_other_bytes,
                name="reachable_after_other_bytes",
            )
        )
        initial_guard = _expression_guard(expr)
        planner_expr, rewrite_metadata = _normalize_expression_for_planner(expr)
        planner_decision = plan_composed_tails(
            planner_expr,
            output_support,
            budget=planner_budget,
        )
        _require_guard(
            expr,
            initial_guard,
            reason="planner_mutated_expression",
        )
        if not planner_decision.accepted or planner_decision.plan is None:
            return _failure(
                reason=f"planner_{planner_decision.reason}",
                expression=expr,
                terms=original_terms,
                bias=original_bias,
                compile_attempts=0,
            )

        if planner_decision.plan.bias is not original_bias:
            raise CombinedIntegrationReject("planner_bias_identity_changed")

        frozen_requests = _freeze_requests(planner_decision)
        frozen_guard = _expression_guard(expr)
        if after_plan_hook is not None:
            after_plan_hook()
        _require_guard(
            expr,
            frozen_guard,
            reason="semantic_payload_changed_after_plan",
        )

        staging = V2Transaction()
        planner_to_descriptor: dict[
            tuple[str, str], ComposedConv2DStencilCandidateV2
        ] = {}
        actual_unique: dict[tuple, ComposedConv2DStencilCandidateV2] = {}
        prospective_emissions: list[int] = []
        for request_index, request in enumerate(frozen_requests):
            if before_compile_hook is not None:
                before_compile_hook(request_index)
            _require_guard(
                expr,
                frozen_guard,
                reason="semantic_payload_changed_during_compile",
            )
            compile_attempts += 1
            build = ComposedConv2DStencilCandidateV2.try_build(
                request.inner,
                request.middle_ops,
                request.outer,
                GateRequestV2(
                    selected_rows=request.selected_rows,
                    reachable_before_bytes=physical_before,
                    # V2Transaction now owns the cumulative descriptor
                    # resident ledger and rechecks it under the reserve lock.
                    reachable_after_other_bytes=physical_after_base,
                    transaction=staging,
                    limits=v1_limits,
                ),
            )
            _require_guard(
                expr,
                frozen_guard,
                reason="semantic_payload_changed_during_compile",
            )
            if not build.triggered or build.operator is None:
                raise CombinedIntegrationReject(f"v2_{build.reason}")
            if build.estimate is None or not (
                build.estimate.emission_accounting_deferred
            ):
                raise CombinedIntegrationReject(
                    "v2_emission_boundary_contract_mismatch"
                )
            prospective_emissions.append(
                int(build.estimate.prospective_emission_products)
            )
            descriptor = build.operator
            if descriptor.content_key != request.expected_v2_content_key:
                raise CombinedIntegrationReject("v2_compiled_content_key_mismatch")
            existing = actual_unique.get(descriptor.content_key)
            if existing is not None and existing is not descriptor:
                raise CombinedIntegrationReject("v2_intern_identity_mismatch")
            if existing is None:
                actual_unique[descriptor.content_key] = descriptor
            planner_to_descriptor[request.planner_handle] = descriptor

        _require_guard(
            expr,
            frozen_guard,
            reason="semantic_payload_changed_before_publish",
        )
        plan = planner_decision.plan
        uses = {use.term_index: use for use in plan.uses}
        if len(uses) != len(plan.uses):
            raise CombinedIntegrationReject("planner_duplicate_term_use")
        rewritten: list[AffineTermView] = []
        for term_index, term in enumerate(original_terms):
            use = uses.get(term_index)
            if use is None:
                rewritten.append(term)
                continue
            metadata = rewrite_metadata[term_index]
            if metadata is None:
                raise CombinedIntegrationReject("rewrite_metadata_missing")
            descriptor = planner_to_descriptor.get(
                use.descriptor_content_key
            )
            if descriptor is None:
                raise CombinedIntegrationReject("compiled_handle_missing")
            rewritten.append(
                AffineTermView(
                    source=term.source,
                    operators=(
                        *metadata.original_prefix,
                        descriptor,
                        *metadata.original_output_diagonals,
                    ),
                )
            )
        _require_guard(
            expr,
            frozen_guard,
            reason="semantic_payload_changed_before_publish",
        )
        snapshot = staging.snapshot()
        if snapshot.reservation_open or snapshot.transient_live_bytes != 0:
            raise CombinedIntegrationReject("v2_transaction_not_closed")
        if frozenset(actual_unique) != snapshot.compiled_content_keys:
            raise CombinedIntegrationReject("v2_transaction_key_mismatch")
        rewritten_expression = AffineExprView(
            terms=tuple(rewritten),
            bias=plan.bias,
            n_out=expr.n_out,
        )
        return CombinedIntegrationDecision(
            accepted=True,
            reason="rewritten_atomically",
            expression=rewritten_expression,
            terms=rewritten_expression.terms,
            bias=plan.bias,
            # Keep only stable summaries.  PlanDecision owns original
            # terms/operators and must die before a physical ledger can claim
            # that the replaced objects are unreachable.
            planner_decision=None,
            planner_handles=tuple(
                request.planner_handle for request in frozen_requests
            ),
            planned_selected_rows=tuple(
                request.selected_rows for request in frozen_requests
            ),
            prospective_emission_products=tuple(prospective_emissions),
            compiled_unique=tuple(actual_unique.values()),
            actual_content_keys=tuple(actual_unique),
            transaction=staging,
            transaction_snapshot=snapshot,
            compile_attempts=compile_attempts,
            exception_type=None,
        )
    except CombinedIntegrationReject as exc:
        return _failure(
            reason=str(exc),
            expression=expr,
            terms=original_terms,
            bias=original_bias,
            compile_attempts=compile_attempts,
        )
    except Exception as exc:
        return _failure(
            reason=f"integration_baseexception_{type(exc).__name__}",
            expression=expr,
            terms=original_terms,
            bias=original_bias,
            compile_attempts=compile_attempts,
            exception_type=type(exc).__name__,
        )
    except BaseException:
        # The transaction and frozen clones are unpublished request-local
        # objects.  Unwinding discards them; process-control exceptions must
        # never be converted into a verifier verdict or normal fallback.
        raise


__all__ = [
    "ADAPTER_GAPS",
    "ADAPTER_NON_CLAIMS",
    "CombinedIntegrationDecision",
    "CombinedIntegrationReject",
    "CombinedPhysicalLedger",
    "integrate_composed_tails_v2",
]
