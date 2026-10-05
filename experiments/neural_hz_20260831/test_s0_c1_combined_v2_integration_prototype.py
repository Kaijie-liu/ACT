"""Focused tests for the isolated planner/V2 atomic integration adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from experiments.neural_hz_20260831.composed_conv2d_stencil_candidate_v2 import (
    ComposedConv2DStencilCandidateV2,
    FrozenV1Limits,
    V2Transaction,
)
from experiments.neural_hz_20260831.s0_c1_combined_v2_integration_prototype import (
    ADAPTER_GAPS,
    ADAPTER_NON_CLAIMS,
    CombinedPhysicalLedger,
    integrate_composed_tails_v2,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    AffineExprView,
    AffineTermView,
)


PHYSICAL = CombinedPhysicalLedger(
    reachable_before_bytes=10**12,
    reachable_after_other_bytes=0,
)


@dataclass(frozen=True)
class SourceView:
    name: str
    frame: object
    predicates: object | None = None


class MutableSparseSourceView:
    def __init__(self):
        self.frame_id = 17
        self.exact = True
        self.c = np.array([0.25], dtype=np.float64)
        self.Gc = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
        self.Gb = sp.csr_matrix(np.array([[0.5]], dtype=np.float64))
        self.Ac = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
        self.Ab = sp.csr_matrix(np.array([[-1.0]], dtype=np.float64))
        self.b = np.array([0.0], dtype=np.float64)
        self.Auc = sp.csr_matrix(np.array([[0.25]], dtype=np.float64))
        self.Aub = sp.csr_matrix(np.array([[0.75]], dtype=np.float64))
        self.ub = np.array([1.0], dtype=np.float64)


@dataclass(frozen=True)
class Barrier:
    kind: str


class SyntheticAbort(BaseException):
    pass


def _kernel(out_channels, in_per_group, shape, *, offset=0):
    size = out_channels * in_per_group * shape[0] * shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 13 - 6) / 16.0
    return values.reshape(out_channels, in_per_group, shape[0], shape[1])


def _tail(*, offset=0):
    # The wider middle layer ensures the frozen 4x work gate passes rather
    # than relying on relaxed synthetic work-ratio limits.
    inner = ImplicitConv2DOp(
        _kernel(16, 4, (3, 3), offset=offset),
        (1, 4, 8, 8),
        padding=1,
    )
    channel = np.resize(
        np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64), 16
    )
    stationary = np.broadcast_to(
        channel.reshape(1, 16, 1, 1), inner.output_shape
    ).copy()
    middle = DiagonalLinearOp(stationary.reshape(-1))
    outer = ImplicitConv2DOp(
        _kernel(4, 16, (3, 3), offset=offset + 3),
        inner.output_shape,
        padding=1,
    )
    return inner, middle, outer


def _expr(terms, n_out):
    bias = np.linspace(-1.0, 1.0, n_out, dtype=np.float64)
    return AffineExprView(terms=tuple(terms), bias=bias, n_out=n_out)


def _all_rows(n_out):
    return np.ones(n_out, dtype=bool)


def _assert_atomic_failure(result, expr):
    assert not result.accepted
    assert result.expression is expr
    assert result.terms is expr.terms
    assert result.bias is expr.bias
    assert result.compiled_unique == ()
    assert result.actual_content_keys == ()
    assert result.transaction is None
    assert result.planner_decision is None
    assert result.planner_handles == ()
    assert result.planned_selected_rows == ()
    assert result.prospective_emission_products == ()
    assert result.transaction_snapshot == V2Transaction().snapshot()


def test_shared_tails_compile_one_actual_v2_key_and_preserve_all_identities():
    inner, middle, outer = _tail(offset=0)
    prefix = Barrier("PREFIX_AFFINE")
    output_values = np.ones(outer.shape[0], dtype=np.float64)
    output_values[::7] = 0.5
    output_diagonal = DiagonalLinearOp(output_values)
    frame = object()
    sources = tuple(SourceView(f"source_{index}", frame) for index in range(12))
    terms = tuple(
        AffineTermView(
            source=source,
            operators=(prefix, inner, middle, outer, output_diagonal),
        )
        for source in sources
    )
    expr = _expr(terms, outer.shape[0])

    result = integrate_composed_tails_v2(
        expr,
        _all_rows(expr.n_out),
        physical=PHYSICAL,
    )
    assert result.accepted, result.reason
    assert result.reason == "rewritten_atomically"
    assert result.expression is not expr
    assert result.expression.terms is result.terms
    assert result.expression.bias is expr.bias
    assert result.bias is expr.bias
    assert result.planner_decision is None
    assert len(result.terms) == len(expr.terms)
    assert len(result.compiled_unique) == 1
    assert len(result.actual_content_keys) == 1
    assert result.compile_attempts == 1
    assert result.transaction is not None
    assert len(result.transaction_snapshot.compiled_content_keys) == 1
    assert not result.transaction_snapshot.reservation_open
    assert result.transaction_snapshot.transient_live_bytes == 0
    descriptor = result.compiled_unique[0]
    planner_handle = result.planner_handles[0]
    assert descriptor.content_key == result.actual_content_keys[0]
    assert descriptor.content_key != planner_handle

    for index, rewritten in enumerate(result.terms):
        assert rewritten.source is sources[index]
        assert rewritten.source.frame is frame
        kept_prefix, compiled, kept_output = rewritten.operators
        assert kept_prefix is prefix
        assert compiled is descriptor
        assert kept_output is output_diagonal
    # The input expression was never mutated or repointed.
    assert expr.terms is terms
    assert all(expr.terms[index] is terms[index] for index in range(len(terms)))


def test_equivalent_middle_factorizations_share_one_v2_descriptor():
    inner, middle, outer = _tail(offset=1)
    first_values = np.asarray(middle._diagonal, dtype=np.float64)
    factor_a = DiagonalLinearOp(np.full(first_values.size, 0.5))
    factor_b = DiagonalLinearOp(first_values * 2.0)
    sources = (SourceView("a", object()), SourceView("b", object()))
    expr = _expr(
        (
            AffineTermView(sources[0], (inner, middle, outer)),
            AffineTermView(sources[1], (inner, factor_a, factor_b, outer)),
        ),
        outer.shape[0],
    )
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert result.compile_attempts == 1
    assert len(result.compiled_unique) == 1
    assert result.terms[0].operators[0] is result.compiled_unique[0]
    assert result.terms[1].operators[0] is result.compiled_unique[0]


@pytest.mark.parametrize("kind", ["ADD", "RELU", "POOL", "DENSE"])
def test_barriers_are_never_crossed_and_mixed_residual_term_is_untouched(kind):
    inner, middle, outer = _tail(offset=2)
    eligible_source = SourceView("eligible", object())
    blocked_source = SourceView("blocked", object())
    barrier = Barrier(kind)
    eligible = AffineTermView(eligible_source, (inner, middle, outer))
    blocked = AffineTermView(
        blocked_source, (inner, middle, barrier, outer)
    )
    expr = _expr((eligible, blocked), outer.shape[0])
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert result.terms[0].source is eligible_source
    assert isinstance(
        result.terms[0].operators[0], ComposedConv2DStencilCandidateV2
    )
    assert result.terms[1] is blocked
    assert result.terms[1].source is blocked_source
    assert result.terms[1].operators[2] is barrier

    blocked_only = _expr((blocked,), outer.shape[0])
    rejected = integrate_composed_tails_v2(
        blocked_only,
        _all_rows(blocked_only.n_out),
        physical=PHYSICAL,
    )
    _assert_atomic_failure(rejected, blocked_only)
    assert rejected.reason == "planner_no_eligible_tail"


def test_output_mask_support_is_backpropagated_but_diagonal_is_preserved():
    inner, middle, outer = _tail(offset=3)
    output_values = np.ones(outer.shape[0], dtype=np.float64)
    output_values[::2] = 0.0
    output_diagonal = DiagonalLinearOp(output_values)
    source = SourceView("masked", object())
    expr = _expr(
        (
            AffineTermView(
                source,
                (inner, middle, outer, output_diagonal),
            ),
        ),
        outer.shape[0],
    )
    bias_before = expr.bias.copy()
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert result.planned_selected_rows[0] == tuple(
        np.flatnonzero(output_values)
    )
    assert result.terms[0].operators[-1] is output_diagonal
    assert result.terms[0].source is source
    # Support/keep is a term-map decision.  It never masks affine bias.
    assert result.bias is expr.bias
    assert result.expression.bias is expr.bias
    assert np.array_equal(result.bias, bias_before)


def test_zero_support_term_and_its_unique_predicates_are_never_deleted():
    inner, middle, outer = _tail(offset=13)
    eligible_source = SourceView("eligible", object(), object())
    zero_predicates = object()
    zero_source = SourceView("zero", object(), zero_predicates)
    eligible = AffineTermView(eligible_source, (inner, middle, outer))
    zero_output = DiagonalLinearOp(np.zeros(outer.shape[0], dtype=np.float64))
    zero_term = AffineTermView(
        zero_source, (inner, middle, outer, zero_output)
    )
    expr = _expr((eligible, zero_term), outer.shape[0])
    bias_before = expr.bias.copy()
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert isinstance(
        result.terms[0].operators[0], ComposedConv2DStencilCandidateV2
    )
    assert result.terms[1] is zero_term
    assert result.terms[1].source is zero_source
    assert result.terms[1].source.predicates is zero_predicates
    assert result.terms[1].operators[-1] is zero_output
    assert result.bias is expr.bias
    assert np.array_equal(result.bias, bias_before)

    only_zero = _expr((zero_term,), outer.shape[0])
    rejected = integrate_composed_tails_v2(
        only_zero, _all_rows(only_zero.n_out), physical=PHYSICAL
    )
    _assert_atomic_failure(rejected, only_zero)
    assert rejected.reason == "planner_no_eligible_tail"
    assert rejected.terms[0] is zero_term
    assert rejected.terms[0].source.predicates is zero_predicates


def test_each_support_gets_a_fresh_estimate_but_emission_remains_deferred():
    inner, middle, outer = _tail(offset=14)
    term = AffineTermView(SourceView("support", object()), (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])
    small_support = np.zeros(expr.n_out, dtype=bool)
    small_support[::2] = True
    large_support = np.ones(expr.n_out, dtype=bool)

    small = integrate_composed_tails_v2(
        expr, small_support, physical=PHYSICAL
    )
    large = integrate_composed_tails_v2(
        expr, large_support, physical=PHYSICAL
    )
    assert small.accepted, small.reason
    assert large.accepted, large.reason
    assert small.transaction is not large.transaction
    assert small.prospective_emission_products[0] > 0
    assert large.prospective_emission_products[0] > (
        small.prospective_emission_products[0]
    )
    assert small.transaction_snapshot.contraction_used == (
        large.transaction_snapshot.contraction_used
    )
    assert small.transaction_snapshot.descriptor_resident_used_bytes == (
        large.transaction_snapshot.descriptor_resident_used_bytes
    )
    assert "support_emission_is_prospective_not_committed" in (
        small.adapter_non_claims
    )


def test_canonical_csr_middle_and_output_diagonals_are_local_shadows_only():
    inner, middle, outer = _tail(offset=11)
    middle_csr = sp.diags(
        np.asarray(middle._diagonal, dtype=np.float64), format="csr"
    )
    output_values = np.ones(outer.shape[0], dtype=np.float64)
    output_values[::3] = 0.0
    output_csr = sp.diags(output_values, format="csr")
    middle_before = (
        middle_csr.indptr.copy(),
        middle_csr.indices.copy(),
        middle_csr.data.copy(),
    )
    output_before = (
        output_csr.indptr.copy(),
        output_csr.indices.copy(),
        output_csr.data.copy(),
    )
    source = SourceView("csr", object())
    original_term = AffineTermView(
        source, (inner, middle_csr, outer, output_csr)
    )
    expr = _expr((original_term,), outer.shape[0])
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert isinstance(result.terms[0].operators[0], ComposedConv2DStencilCandidateV2)
    # The output affine operator is the original CSR object, never its shadow.
    assert result.terms[0].operators[1] is output_csr
    assert result.terms[0].source is source
    assert result.planned_selected_rows[0] == tuple(
        np.flatnonzero(output_values)
    )
    assert np.array_equal(middle_csr.indptr, middle_before[0])
    assert np.array_equal(middle_csr.indices, middle_before[1])
    assert np.array_equal(middle_csr.data, middle_before[2])
    assert np.array_equal(output_csr.indptr, output_before[0])
    assert np.array_equal(output_csr.indices, output_before[1])
    assert np.array_equal(output_csr.data, output_before[2])
    assert expr.terms[0] is original_term


def test_non_diagonal_csr_is_a_hard_barrier_not_a_global_phase_conversion():
    inner, middle, outer = _tail(offset=12)
    diagonal = sp.diags(
        np.asarray(middle._diagonal, dtype=np.float64), format="csr"
    )
    non_diagonal = diagonal.tolil(copy=True)
    non_diagonal[0, 1] = 0.25
    non_diagonal = non_diagonal.tocsr()
    blocked = AffineTermView(
        SourceView("blocked_csr", object()),
        (inner, non_diagonal, outer),
    )
    blocked_expr = _expr((blocked,), outer.shape[0])
    rejected = integrate_composed_tails_v2(
        blocked_expr, _all_rows(blocked_expr.n_out), physical=PHYSICAL
    )
    _assert_atomic_failure(rejected, blocked_expr)
    assert rejected.reason == "planner_no_eligible_tail"
    assert rejected.terms[0] is blocked
    assert rejected.terms[0].operators[1] is non_diagonal

    eligible = AffineTermView(
        SourceView("eligible", object()), (inner, middle, outer)
    )
    mixed_expr = _expr((eligible, blocked), outer.shape[0])
    mixed = integrate_composed_tails_v2(
        mixed_expr, _all_rows(mixed_expr.n_out), physical=PHYSICAL
    )
    assert mixed.accepted, mixed.reason
    assert mixed.terms[1] is blocked
    assert mixed.terms[1].operators[1] is non_diagonal


def test_multiple_unique_terms_consume_one_shared_v2_budget_atomically():
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=7)
    n_out = tail_a[-1].shape[0]
    single_expr = _expr((AffineTermView(SourceView("one", object()), tail_a),), n_out)
    single = integrate_composed_tails_v2(
        single_expr, _all_rows(n_out), physical=PHYSICAL
    )
    assert single.accepted, single.reason
    single_work = (
        single.transaction_snapshot.contraction_used
        + single.prospective_emission_products[0]
    )

    source_a = SourceView("a", object())
    source_b = SourceView("b", object())
    first = AffineTermView(source_a, tail_a)
    second = AffineTermView(source_b, tail_b)
    expr = _expr((first, second), n_out)
    limits = replace(FrozenV1Limits(), max_transaction_work=single_work)
    rejected = integrate_composed_tails_v2(
        expr,
        _all_rows(n_out),
        physical=PHYSICAL,
        v1_limits=limits,
    )
    _assert_atomic_failure(rejected, expr)
    assert rejected.reason == "v2_transaction_work_limit"
    assert rejected.compile_attempts == 2
    assert rejected.terms[0] is first and rejected.terms[1] is second
    assert rejected.terms[0].source is source_a
    assert rejected.terms[1].source is source_b


def test_real_compiler_baseexception_after_first_commit_is_re_raised(
    monkeypatch,
):
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=8)
    n_out = tail_a[-1].shape[0]
    terms = (
        AffineTermView(SourceView("a", object()), tail_a),
        AffineTermView(SourceView("b", object()), tail_b),
    )
    expr = _expr(terms, n_out)
    original = ComposedConv2DStencilCandidateV2.try_build
    calls = []

    def injected(inner, middle_ops, outer, gate):
        calls.append((inner, middle_ops, outer))
        if len(calls) == 2:
            raise SyntheticAbort("abort second compile")
        return original(inner, middle_ops, outer, gate)

    monkeypatch.setattr(
        ComposedConv2DStencilCandidateV2,
        "try_build",
        staticmethod(injected),
    )
    with pytest.raises(SyntheticAbort, match="abort second compile"):
        integrate_composed_tails_v2(
            expr, _all_rows(n_out), physical=PHYSICAL
        )
    assert len(calls) == 2
    assert expr.terms is terms
    assert expr.terms[0] is terms[0] and expr.terms[1] is terms[1]


def test_real_compiler_exception_after_first_commit_returns_original(
    monkeypatch,
):
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=8)
    n_out = tail_a[-1].shape[0]
    terms = (
        AffineTermView(SourceView("a", object()), tail_a),
        AffineTermView(SourceView("b", object()), tail_b),
    )
    expr = _expr(terms, n_out)
    original = ComposedConv2DStencilCandidateV2.try_build
    calls = []

    def injected(inner, middle_ops, outer, gate):
        calls.append((inner, middle_ops, outer))
        if len(calls) == 2:
            raise RuntimeError("abort second compile")
        return original(inner, middle_ops, outer, gate)

    monkeypatch.setattr(
        ComposedConv2DStencilCandidateV2,
        "try_build",
        staticmethod(injected),
    )
    result = integrate_composed_tails_v2(
        expr, _all_rows(n_out), physical=PHYSICAL
    )
    _assert_atomic_failure(result, expr)
    assert result.reason == "integration_baseexception_RuntimeError"
    assert result.exception_type == "RuntimeError"
    assert result.compile_attempts == 2
    assert len(calls) == 2


def test_mutation_between_plan_and_compile_fails_before_any_v2_publication():
    inner, middle, outer = _tail(offset=4)
    term = AffineTermView(SourceView("mutable", object()), (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])

    def mutate_after_plan():
        inner._kernel[0, 0, 0, 0] += 0.125

    result = integrate_composed_tails_v2(
        expr,
        _all_rows(expr.n_out),
        physical=PHYSICAL,
        after_plan_hook=mutate_after_plan,
    )
    _assert_atomic_failure(result, expr)
    assert result.reason == "semantic_payload_changed_after_plan"
    assert result.compile_attempts == 0
    assert result.terms[0] is term


def test_source_predicate_mutation_during_plan_window_fails_closed():
    inner, middle, outer = _tail(offset=15)
    predicates = np.array([1.0, -2.0], dtype=np.float64)
    source = SourceView("predicate_mutation", object(), predicates)
    term = AffineTermView(source, (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])

    def mutate_after_plan():
        predicates[0] = 3.0

    result = integrate_composed_tails_v2(
        expr,
        _all_rows(expr.n_out),
        physical=PHYSICAL,
        after_plan_hook=mutate_after_plan,
    )
    _assert_atomic_failure(result, expr)
    assert result.reason == "semantic_payload_changed_after_plan"
    assert result.compile_attempts == 0
    assert result.terms[0] is term
    assert result.terms[0].source is source


def test_source_value_map_mutation_during_plan_window_fails_closed():
    inner, middle, outer = _tail(offset=16)
    source = MutableSparseSourceView()
    term = AffineTermView(source, (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])

    def mutate_after_plan():
        source.Gc.data[0] = 2.0

    result = integrate_composed_tails_v2(
        expr,
        _all_rows(expr.n_out),
        physical=PHYSICAL,
        after_plan_hook=mutate_after_plan,
    )
    _assert_atomic_failure(result, expr)
    assert result.reason == "semantic_payload_changed_after_plan"
    assert result.compile_attempts == 0
    assert result.terms[0] is term
    assert result.terms[0].source is source


def test_stale_planner_dedup_is_reconciled_against_current_payload_snapshots():
    first_tail = _tail(offset=5)
    second_tail = _tail(offset=5)
    # Constructor-time content_key remains equal, but current payload differs.
    second_tail[0]._kernel[0, 0, 0, 0] += 0.25
    assert first_tail[0].content_key == second_tail[0].content_key
    n_out = first_tail[-1].shape[0]
    terms = (
        AffineTermView(SourceView("first", object()), first_tail),
        AffineTermView(SourceView("second", object()), second_tail),
    )
    expr = _expr(terms, n_out)
    result = integrate_composed_tails_v2(
        expr, _all_rows(n_out), physical=PHYSICAL
    )
    _assert_atomic_failure(result, expr)
    assert result.reason == "planner_v2_snapshot_identity_mismatch"
    assert result.compile_attempts == 0


def test_v2_reject_and_planner_reject_both_return_original_identity():
    inner, middle, outer = _tail(offset=6)
    term = AffineTermView(SourceView("source", object()), (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])
    unproven = integrate_composed_tails_v2(expr, _all_rows(expr.n_out))
    _assert_atomic_failure(unproven, expr)
    assert unproven.reason == "v2_physical_metric_unproven"
    assert unproven.compile_attempts == 1

    spatial_values = np.ones(inner.shape[0], dtype=np.float64)
    spatial_values[1] = 0.5
    spatial = DiagonalLinearOp(spatial_values)
    bad_term = AffineTermView(
        SourceView("bad", object()), (inner, spatial, outer)
    )
    bad_expr = _expr((bad_term,), outer.shape[0])
    planner_reject = integrate_composed_tails_v2(
        bad_expr, _all_rows(bad_expr.n_out), physical=PHYSICAL
    )
    _assert_atomic_failure(planner_reject, bad_expr)
    assert planner_reject.reason == "planner_middle_0_not_channel_stationary"
    assert planner_reject.compile_attempts == 0


def test_adapter_never_calls_input_conv_full_csr(monkeypatch):
    inner, middle, outer = _tail(offset=9)
    expr = _expr(
        (AffineTermView(SourceView("guard", object()), (inner, middle, outer)),),
        outer.shape[0],
    )

    def forbidden(_self):
        raise AssertionError("full Conv CSR expansion")

    monkeypatch.setattr(ImplicitConv2DOp, "to_csr_reference", forbidden)
    result = integrate_composed_tails_v2(
        expr, _all_rows(expr.n_out), physical=PHYSICAL
    )
    assert result.accepted, result.reason
    assert len(result.compiled_unique) == 1


def test_adapter_gaps_are_explicit_and_module_is_absent_from_production():
    assert ADAPTER_GAPS == (
        "planner_handle_is_not_v2_content_key",
        "v2_has_no_public_key_only_snapshot_api",
        "v2_has_no_committed_transaction_savepoint_or_merge",
        "adapter_guard_and_snapshot_transients_are_not_charged",
        "actual_emission_materializer_not_integrated",
    )
    assert ADAPTER_NON_CLAIMS == (
        "whole_state_root_alias_not_proven",
        "old_expression_retention_not_accounted",
        "synthetic_physical_ledger_is_not_a_representation_result",
        "production_add_reshape_lineage_is_not_represented",
        "production_batch_two_plumbing_is_not_proven",
        "support_emission_is_prospective_not_committed",
    )
    root = Path(__file__).resolve().parents[2]
    module_name = "s0_c1_combined_v2_integration_prototype"
    for relative in (
        "act/back_end/hybridz_tf/exact_linear_op.py",
        "act/back_end/hybridz_tf/tf_cnn.py",
        "act/back_end/hybridz_tf/tf_mlp.py",
    ):
        assert module_name not in (root / relative).read_text(encoding="utf-8")
