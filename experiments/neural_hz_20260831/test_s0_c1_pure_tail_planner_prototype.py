"""Focused tests for the isolated S0-C1 pure tail planner."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    AffineExprView,
    AffineTermView,
    PlannerBudget,
    TailPlannerReject,
    execute_tail_plan,
    plan_composed_tails,
)


@dataclass(frozen=True)
class Barrier:
    kind: str


@dataclass(frozen=True)
class CompiledDescriptor:
    content_key: tuple[str, str]


def _kernel(out_channels, in_per_group, shape, *, offset=0):
    size = out_channels * in_per_group * shape[0] * shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 11 - 5) / 16.0
    return values.reshape(out_channels, in_per_group, shape[0], shape[1])


def _tail(*, offset=0):
    inner = ImplicitConv2DOp(
        _kernel(4, 2, (3, 3), offset=offset),
        (1, 2, 5, 6),
        padding=1,
    )
    channel = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64)
    stationary = np.broadcast_to(
        channel.reshape(1, 4, 1, 1), inner.output_shape
    ).copy()
    middle = DiagonalLinearOp(stationary.reshape(-1))
    outer = ImplicitConv2DOp(
        _kernel(3, 4, (3, 3), offset=offset + 3),
        inner.output_shape,
        padding=1,
    )
    return inner, middle, outer


def _expr(terms, n_out):
    bias = np.linspace(-1.0, 1.0, n_out, dtype=np.float64)
    return AffineExprView(terms=tuple(terms), bias=bias, n_out=n_out)


def _all_rows(n_out):
    return np.ones(n_out, dtype=bool)


def _compile_counter(calls, *, fail_after=None):
    def compile_request(request):
        calls.append(request.content_key)
        if fail_after is not None and len(calls) > fail_after:
            raise RuntimeError("synthetic compile failure")
        return CompiledDescriptor(request.content_key)

    return compile_request


def test_forty_content_shared_suffixes_reserve_and_compile_once():
    inner, middle, outer = _tail(offset=0)
    # The second half uses separately allocated but content-identical operators,
    # proving grouping is by stable content rather than Python object identity.
    clone_inner, clone_middle, clone_outer = _tail(offset=0)
    terms = []
    sources = []
    for index in range(40):
        source = object()
        sources.append(source)
        operators = (
            (inner, middle, outer)
            if index < 20
            else (clone_inner, clone_middle, clone_outer)
        )
        terms.append(AffineTermView(source=source, operators=operators))
    expr = _expr(terms, outer.shape[0])

    decision = plan_composed_tails(expr, _all_rows(expr.n_out))
    assert decision.accepted, decision.reason
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is not None
    assert len(decision.plan.requests) == 1
    request = decision.plan.requests[0]
    assert request.use_count == 40
    assert request.term_indices == tuple(range(40))
    assert decision.plan.reservation.unique_descriptors == 1
    assert (
        decision.plan.reservation.resident_bytes
        == request.estimate.resident_bytes
    )

    calls = []
    rewritten = execute_tail_plan(decision, _compile_counter(calls))
    assert rewritten.accepted, rewritten.reason
    assert len(calls) == 1
    assert len(rewritten.compiled_unique) == 1
    descriptor = rewritten.compiled_unique[0]
    assert rewritten.bias is expr.bias
    assert len(rewritten.terms) == 40
    for index, term in enumerate(rewritten.terms):
        assert term.source is sources[index]
        assert term.operators == (descriptor,)
        assert term.operators[0] is descriptor


def test_unique_requests_have_stable_content_key_order_independent_of_terms():
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=7)
    n_out = tail_a[-1].shape[0]
    first = _expr(
        (
            AffineTermView(object(), tail_b),
            AffineTermView(object(), tail_a),
        ),
        n_out,
    )
    second = _expr(
        (
            AffineTermView(object(), tail_a),
            AffineTermView(object(), tail_b),
        ),
        n_out,
    )

    first_plan = plan_composed_tails(first, _all_rows(n_out))
    second_plan = plan_composed_tails(second, _all_rows(n_out))
    assert first_plan.accepted and second_plan.accepted
    first_keys = tuple(request.content_key for request in first_plan.plan.requests)
    second_keys = tuple(request.content_key for request in second_plan.plan.requests)
    assert first_keys == second_keys
    assert tuple(key[1] for key in first_keys) == tuple(
        sorted(key[1] for key in first_keys)
    )

    calls = []
    rewritten = execute_tail_plan(first_plan, _compile_counter(calls))
    assert rewritten.accepted
    assert tuple(calls) == first_keys


def test_multiple_unique_descriptors_use_cumulative_transaction_budget():
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=7)
    n_out = tail_a[-1].shape[0]
    expr = _expr(
        (
            AffineTermView(object(), tail_a),
            AffineTermView(object(), tail_b),
        ),
        n_out,
    )
    generous = plan_composed_tails(expr, _all_rows(n_out))
    assert generous.accepted
    estimates = [request.estimate for request in generous.plan.requests]
    assert len(estimates) == 2
    each_fits = max(estimate.total_work for estimate in estimates)
    assert sum(estimate.total_work for estimate in estimates) > each_fits

    tight = replace(
        PlannerBudget(),
        max_transaction_total_work=each_fits,
    )
    rejected = plan_composed_tails(expr, _all_rows(n_out), budget=tight)
    assert not rejected.accepted
    assert rejected.reason == "transaction_work_limit"
    assert rejected.terms is expr.terms
    assert rejected.bias is expr.bias
    calls = []
    rewrite = execute_tail_plan(rejected, _compile_counter(calls))
    assert not rewrite.accepted
    assert calls == []
    assert rewrite.terms is expr.terms
    assert rewrite.bias is expr.bias


def test_compile_failure_rolls_back_original_tuple_and_bias_after_local_work():
    tail_a = _tail(offset=0)
    tail_b = _tail(offset=7)
    n_out = tail_a[-1].shape[0]
    expr = _expr(
        (
            AffineTermView(object(), tail_a),
            AffineTermView(object(), tail_b),
        ),
        n_out,
    )
    plan = plan_composed_tails(expr, _all_rows(n_out))
    assert plan.accepted
    calls = []
    rewrite = execute_tail_plan(
        plan,
        _compile_counter(calls, fail_after=1),
    )
    assert not rewrite.accepted
    assert rewrite.reason == "compiler_failed_RuntimeError"
    assert len(calls) == 2
    assert rewrite.compiled_unique == ()
    assert rewrite.terms is expr.terms
    assert rewrite.bias is expr.bias


def test_output_diagonal_masks_are_backpropagated_and_preserved():
    inner, middle, outer = _tail(offset=0)
    n_out = outer.shape[0]
    first_values = np.ones(n_out, dtype=np.float64)
    first_values[1::2] = 0.0
    second_values = np.ones(n_out, dtype=np.float64)
    second_values[::3] = 0.0
    first = DiagonalLinearOp(first_values)
    second = DiagonalLinearOp(second_values)
    term = AffineTermView(
        source=object(),
        operators=(inner, middle, outer, first, second),
    )
    expr = _expr((term,), n_out)

    decision = plan_composed_tails(expr, _all_rows(n_out))
    assert decision.accepted, decision.reason
    request = decision.plan.requests[0]
    expected = tuple(
        index
        for index in range(n_out)
        if first_values[index] != 0.0 and second_values[index] != 0.0
    )
    assert request.selected_rows == expected
    assert request.estimate.selected_rows == len(expected)

    calls = []
    rewritten = execute_tail_plan(decision, _compile_counter(calls))
    assert rewritten.accepted
    descriptor, kept_first, kept_second = rewritten.terms[0].operators
    assert descriptor is rewritten.compiled_unique[0]
    assert kept_first is first
    assert kept_second is second
    assert rewritten.bias is expr.bias


def test_same_core_different_output_masks_union_support_and_compile_once():
    inner, middle, outer = _tail(offset=0)
    n_out = outer.shape[0]
    even_values = np.zeros(n_out, dtype=np.float64)
    even_values[::2] = 1.0
    odd_values = np.zeros(n_out, dtype=np.float64)
    odd_values[1::2] = -1.0
    even = DiagonalLinearOp(even_values)
    odd = DiagonalLinearOp(odd_values)
    expr = _expr(
        (
            AffineTermView(object(), (inner, middle, outer, even)),
            AffineTermView(object(), (inner, middle, outer, odd)),
        ),
        n_out,
    )
    decision = plan_composed_tails(expr, _all_rows(n_out))
    assert decision.accepted
    assert len(decision.plan.requests) == 1
    assert decision.plan.requests[0].selected_rows == tuple(range(n_out))
    assert decision.plan.requests[0].use_count == 2
    calls = []
    rewritten = execute_tail_plan(decision, _compile_counter(calls))
    assert rewritten.accepted
    assert len(calls) == 1
    assert rewritten.terms[0].operators[-1] is even
    assert rewritten.terms[1].operators[-1] is odd


def test_spatial_middle_diagonal_rejects_transaction_without_new_objects():
    inner, _, outer = _tail(offset=0)
    values = np.ones(inner.shape[0], dtype=np.float64)
    values[1] = 0.5
    spatial = DiagonalLinearOp(values)
    term = AffineTermView(object(), (inner, spatial, outer))
    expr = _expr((term,), outer.shape[0])
    decision = plan_composed_tails(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "middle_0_not_channel_stationary"
    assert decision.terms is expr.terms
    assert decision.terms[0] is term
    assert decision.bias is expr.bias


@pytest.mark.parametrize("barrier_kind", ["ADD", "RELU", "DENSE", "RESHAPE"])
def test_barriers_are_not_crossed(barrier_kind):
    inner, middle, outer = _tail(offset=0)
    barrier = Barrier(barrier_kind)
    blocked_after = AffineTermView(
        object(), (inner, middle, outer, barrier)
    )
    blocked_inside = AffineTermView(
        object(), (inner, middle, barrier, outer)
    )
    for term in (blocked_after, blocked_inside):
        expr = _expr((term,), outer.shape[0])
        decision = plan_composed_tails(expr, _all_rows(expr.n_out))
        assert not decision.accepted
        assert decision.reason == "no_eligible_tail"
        assert decision.terms is expr.terms
        assert decision.bias is expr.bias


def test_separate_residual_terms_remain_ordered_and_unmatched_term_is_untouched():
    inner, middle, outer = _tail(offset=0)
    eligible_source = object()
    blocked_source = object()
    barrier = Barrier("DENSE")
    eligible = AffineTermView(eligible_source, (inner, middle, outer))
    blocked = AffineTermView(blocked_source, (barrier,))
    expr = _expr((eligible, blocked), outer.shape[0])
    decision = plan_composed_tails(expr, _all_rows(expr.n_out))
    assert decision.accepted
    calls = []
    rewritten = execute_tail_plan(decision, _compile_counter(calls))
    assert rewritten.accepted
    assert len(rewritten.terms) == 2
    assert rewritten.terms[0].source is eligible_source
    assert rewritten.terms[1] is blocked
    assert rewritten.terms[1].source is blocked_source
    assert rewritten.terms[1].operators == (barrier,)


def test_empty_support_and_invalid_budget_fail_closed():
    inner, middle, outer = _tail(offset=0)
    term = AffineTermView(object(), (inner, middle, outer))
    expr = _expr((term,), outer.shape[0])
    empty = plan_composed_tails(expr, np.zeros(expr.n_out, dtype=bool))
    assert not empty.accepted
    assert empty.reason == "empty_output_support"
    assert empty.terms is expr.terms and empty.bias is expr.bias

    invalid = plan_composed_tails(
        expr,
        _all_rows(expr.n_out),
        budget=replace(PlannerBudget(), max_unique_descriptors=True),
    )
    assert not invalid.accepted
    assert invalid.reason == "max_unique_descriptors_not_integer"
    assert invalid.terms is expr.terms and invalid.bias is expr.bias


def test_compiler_explicit_none_is_transactional_rejection():
    inner, middle, outer = _tail(offset=0)
    expr = _expr(
        (AffineTermView(object(), (inner, middle, outer)),),
        outer.shape[0],
    )
    plan = plan_composed_tails(expr, _all_rows(expr.n_out))
    assert plan.accepted
    rewrite = execute_tail_plan(plan, lambda _request: None)
    assert not rewrite.accepted
    assert rewrite.reason == "compiler_returned_none"
    assert rewrite.terms is expr.terms
    assert rewrite.bias is expr.bias
