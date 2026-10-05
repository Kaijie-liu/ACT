"""Independent adversarial tests for the isolated S0-C2 pure planner.

These tests deliberately stop at planning.  They do not authorize descriptor
compilation, CSR emission, expression rewriting, or any production claim.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from experiments.neural_hz_20260831 import (
    s0_c2_residual_distributive_lineage_planner_prototype as c2,
)
from experiments.neural_hz_20260831.s0_c2_residual_distributive_lineage_planner_prototype import (
    BoundaryMark,
    C2PureLineagePlan,
    LineageAffineTermView,
    PlannerBudget,
    plan_s0_c2_residual_distributive_lineage,
)
from experiments.neural_hz_20260831.test_s0_c2_residual_distributive_lineage_planner_prototype import (
    _all_rows,
    _expr,
    _full_term,
    _mark,
    _small_chain,
    _source,
)


def _two_unique_descriptor_expression():
    inner_a, pre_a, _, outer = _small_chain(offset=0)
    inner_b, pre_b, _, _ = _small_chain(offset=7)
    terms = (
        _full_term(_source("budget-a"), inner_a, pre_a, outer),
        _full_term(_source("budget-b"), inner_b, pre_b, outer),
    )
    return _expr(terms, outer.shape[0])


def _exact_budget_for_plan(plan) -> PlannerBudget:
    estimates = tuple(request.estimate for request in plan.requests)
    reservation = plan.reservation
    return PlannerBudget(
        max_unique_descriptors=len(plan.requests),
        max_descriptor_contraction_products=max(
            item.contraction_products for item in estimates
        ),
        max_descriptor_coefficient_entries=max(
            item.coefficient_entries for item in estimates
        ),
        max_descriptor_resident_bytes=max(
            item.resident_bytes for item in estimates
        ),
        max_descriptor_result_nnz=max(
            item.result_nnz_upper for item in estimates
        ),
        max_transaction_contraction_products=reservation.contraction_products,
        max_transaction_total_work=reservation.total_work,
        max_transaction_coefficient_entries=reservation.coefficient_entries,
        max_transaction_resident_bytes=reservation.resident_bytes,
        max_transaction_transient_bytes=reservation.transient_bytes,
        max_transaction_result_nnz=reservation.result_nnz_upper,
    )


@pytest.mark.parametrize(
    "invalid_frame",
    [
        None,
        True,
        False,
        [17],
        {"frame": 17},
        np.array([17], dtype=np.int64),
        object(),
    ],
    ids=("none", "true", "false", "list", "dict", "ndarray", "object"),
)
def test_unstable_or_unframed_expression_token_must_fail_closed(invalid_frame):
    """Regression for the independently reproduced frame fail-open."""

    inner, pre, _, outer = _small_chain()
    source = _source("invalid-frame", frame_id=invalid_frame)
    term = _full_term(source, inner, pre, outer)
    expr = _expr((term,), outer.shape[0], frame_id=invalid_frame)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "expression_frame_id_not_stable"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


@pytest.mark.parametrize(
    "invalid_source_frame",
    [None, True, [17], {"frame": 17}, np.array([17]), object()],
    ids=("none", "bool", "list", "dict", "ndarray", "object"),
)
def test_stable_expression_never_accepts_unstable_source_frame(
    invalid_source_frame,
):
    inner, pre, _, outer = _small_chain()
    source = _source("invalid-source-frame", frame_id=invalid_source_frame)
    expr = _expr(
        (_full_term(source, inner, pre, outer),),
        outer.shape[0],
        frame_id=("frame", 17),
    )

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "source_frame_or_exact_mismatch"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias


def test_common_add_occurrence_not_local_index_controls_cross_branch_match():
    inner, pre, _, outer = _small_chain()
    occurrence = ("same-residual-event", 32)
    first = _full_term(
        _source("first"),
        inner,
        pre,
        outer,
        mark=_mark(occurrence=occurrence, index=2),
    )
    prefix_op = object()
    prefix_cut = BoundaryMark("RESHAPE", ("prefix", 7), 1)
    second_add = _mark(occurrence=occurrence, index=3)
    second = LineageAffineTermView(
        source=_source("second"),
        operators=(prefix_op, inner, pre, outer),
        boundaries=(prefix_cut, second_add),
    )
    expr = _expr((first, second), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert len(decision.plan.requests) == 1
    assert decision.plan.requests[0].term_indices == (0, 1)
    assert decision.plan.common_add_marks == (
        first.boundaries[-1],
        second_add,
    )
    assert decision.plan.uses[1].prefix == (prefix_op,)
    assert decision.plan.uses[1].prefix[0] is prefix_op


def test_same_position_prior_boundary_is_only_an_unfused_identity_segment():
    inner, pre, _, outer = _small_chain()
    occurrence = ("common-add", 32)
    complete = _full_term(
        _source("complete"),
        inner,
        pre,
        outer,
        mark=_mark(occurrence=occurrence, index=2),
    )
    immediately_prior = BoundaryMark("RELU", ("prior-relu", 31), 2)
    identity_add = _mark(occurrence=occurrence, index=2)
    identity = LineageAffineTermView(
        source=_source("identity"),
        operators=(inner, pre, outer),
        boundaries=(immediately_prior, identity_add),
    )
    expr = _expr((complete, identity), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.identity_term_indices == (1,)
    assert tuple(use.term_index for use in decision.plan.uses) == (0,)
    assert decision.terms[1] is identity
    assert decision.terms[1].operators == (inner, pre, outer)
    assert decision.terms[1].boundaries == (immediately_prior, identity_add)

    reversed_events = replace(
        identity,
        boundaries=(identity_add, immediately_prior),
    )
    reversed_expr = _expr((complete, reversed_events), outer.shape[0])
    rejected = plan_s0_c2_residual_distributive_lineage(
        reversed_expr, _all_rows(reversed_expr.n_out)
    )
    assert not rejected.accepted
    assert rejected.reason == "latest_boundary_not_add"
    assert rejected.expression is reversed_expr
    assert rejected.terms is reversed_expr.terms


def test_shared_outer_current_mutation_between_terms_rejects(monkeypatch):
    inner, pre, _, outer = _small_chain()
    terms = (
        _full_term(_source("outer-a"), inner, pre, outer),
        _full_term(_source("outer-b"), inner, pre, outer),
    )
    expr = _expr(terms, outer.shape[0])
    original_snapshot = c2._conv_snapshot
    seen_outer = 0

    def mutate_after_first_outer_snapshot(op, *, name):
        nonlocal seen_outer
        snapshot = original_snapshot(op, name=name)
        if op is outer:
            seen_outer += 1
            if seen_outer == 1:
                outer._kernel[0, 0, 0, 0] += 0.125
        return snapshot

    monkeypatch.setattr(c2, "_conv_snapshot", mutate_after_first_outer_snapshot)
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert seen_outer == 2
    assert not decision.accepted
    assert decision.reason == "shared_outer_snapshot_mismatch"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.plan is None


def test_shared_post_diagonal_current_mutation_between_terms_rejects(monkeypatch):
    inner, pre, post, outer = _small_chain(post=True)
    assert post is not None
    terms = (
        _full_term(_source("post-a"), inner, pre, outer, post=post),
        _full_term(_source("post-b"), inner, pre, outer, post=post),
    )
    expr = _expr(terms, outer.shape[0])
    original_stationary = c2._stationary_channel_vector
    seen_post = 0

    def mutate_after_first_post_snapshot(op, shape, *, name):
        nonlocal seen_post
        result = original_stationary(op, shape, name=name)
        if op is post:
            seen_post += 1
            if seen_post == 1:
                full = post._diagonal.reshape(outer.input_shape)
                full[:, 0, :, :] += 0.125
        return result

    monkeypatch.setattr(
        c2, "_stationary_channel_vector", mutate_after_first_post_snapshot
    )
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert seen_post >= 2
    assert not decision.accepted
    assert decision.reason == "post_add_suffix_snapshot_mismatch"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.plan is None


def test_partial_support_row_mask_and_distinct_dout_union_without_subset():
    provisional_inner, provisional_pre, _, provisional_outer = _small_chain()
    n_out = provisional_outer.shape[0]
    row_mask = np.zeros(n_out, dtype=bool)
    row_mask[[2, 3]] = True
    inner, pre, _, outer = _small_chain(row_mask=row_mask)
    assert outer.shape[0] == n_out

    partial = np.zeros(n_out, dtype=np.float64)
    partial[[0, 2]] = 1.0
    masked_only = np.zeros(n_out, dtype=np.float64)
    masked_only[1] = 1.0
    live_other = np.zeros(n_out, dtype=np.float64)
    live_other[3] = 1.0
    terms = (
        _full_term(
            _source("partial"),
            inner,
            pre,
            outer,
            dout=(DiagonalLinearOp(partial),),
        ),
        _full_term(
            _source("masked-only"),
            inner,
            pre,
            outer,
            dout=(DiagonalLinearOp(masked_only),),
        ),
        _full_term(
            _source("live-other"),
            inner,
            pre,
            outer,
            dout=(DiagonalLinearOp(live_other),),
        ),
    )
    expr = _expr(terms, n_out)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.zero_support_term_indices == (1,)
    assert tuple(use.term_index for use in decision.plan.uses) == (0, 2)
    request = decision.plan.requests[0]
    assert request.term_indices == (0, 2)
    assert request.selected_rows == (0, 2, 3)
    assert request.estimate.selected_rows == 3
    assert request.estimate.active_selected_rows == 2
    assert decision.terms[1] is terms[1]
    assert decision.terms[1].source is terms[1].source


_BUDGET_GATES = (
    ("max_unique_descriptors", "unique_descriptor_limit"),
    (
        "max_descriptor_contraction_products",
        "descriptor_contraction_product_limit",
    ),
    (
        "max_descriptor_coefficient_entries",
        "descriptor_coefficient_entry_limit",
    ),
    ("max_descriptor_resident_bytes", "descriptor_resident_byte_limit"),
    ("max_descriptor_result_nnz", "descriptor_result_nnz_limit"),
    (
        "max_transaction_contraction_products",
        "transaction_contraction_product_limit",
    ),
    ("max_transaction_total_work", "transaction_work_limit"),
    (
        "max_transaction_coefficient_entries",
        "transaction_coefficient_entry_limit",
    ),
    ("max_transaction_resident_bytes", "transaction_resident_byte_limit"),
    (
        "max_transaction_transient_bytes",
        "transaction_transient_byte_limit",
    ),
    ("max_transaction_result_nnz", "transaction_result_nnz_limit"),
)


@pytest.mark.parametrize(("field_name", "reason"), _BUDGET_GATES)
def test_every_budget_gate_accepts_equality_and_rejects_one_less_atomically(
    field_name, reason
):
    expr = _two_unique_descriptor_expression()
    initial = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert initial.accepted, initial.reason
    assert len(initial.plan.requests) == 2
    exact_budget = _exact_budget_for_plan(initial.plan)
    at_equality = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out), budget=exact_budget
    )
    assert at_equality.accepted, (field_name, at_equality.reason)
    assert tuple(use.term_index for use in at_equality.plan.uses) == (0, 1)

    exact_value = getattr(exact_budget, field_name)
    assert exact_value > 0
    below = replace(exact_budget, **{field_name: exact_value - 1})
    rejected = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out), budget=below
    )
    assert not rejected.accepted
    assert rejected.reason == reason
    assert rejected.expression is expr
    assert rejected.terms is expr.terms
    assert rejected.bias is expr.bias
    assert rejected.plan is None


def test_mutated_stationary_diagonal_never_dedups_by_stale_constructor_key():
    inner_a, pre_a, _, outer = _small_chain(offset=0)
    inner_b, pre_b, _, _ = _small_chain(offset=0)
    assert inner_a.content_key == inner_b.content_key
    assert pre_a.content_key == pre_b.content_key
    pre_b._diagonal.reshape(inner_b.output_shape)[:, 0, :, :] += 0.125
    assert pre_a.content_key == pre_b.content_key
    terms = (
        _full_term(_source("stale-a"), inner_a, pre_a, outer),
        _full_term(_source("stale-b"), inner_b, pre_b, outer),
    )
    expr = _expr(terms, outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert len(decision.plan.requests) == 2
    assert len({request.content_key for request in decision.plan.requests}) == 2
    assert tuple(use.term_index for use in decision.plan.uses) == (0, 1)


def test_distinct_current_payloads_with_forced_digest_collision_fail_closed(
    monkeypatch,
):
    expr = _two_unique_descriptor_expression()

    class ForcedDigest:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(c2.hashlib, "sha256", lambda payload: ForcedDigest())
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "descriptor_content_hash_collision"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


def test_sources_terms_bias_and_stable_frame_are_preserved_by_identity():
    frame = ("exact-frame", 17)
    inner, pre, _, outer = _small_chain()
    first_source = _source("equal-value-a", frame_id=frame)
    second_source = _source("equal-value-b", frame_id=frame)
    assert np.array_equal(first_source.c, second_source.c)
    assert np.array_equal(first_source.Gc, second_source.Gc)
    assert np.array_equal(first_source.Gb, second_source.Gb)
    first = _full_term(first_source, inner, pre, outer)
    second = _full_term(second_source, inner, pre, outer)
    bias = np.linspace(-0.25, 0.25, outer.shape[0], dtype=np.float64)
    expr = _expr((first, second), outer.shape[0], bias=bias, frame_id=frame)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is bias
    assert decision.plan.expression is expr
    assert decision.plan.original_terms is expr.terms
    assert decision.plan.bias is bias
    assert decision.plan.frame_id is frame
    assert decision.terms[0] is first
    assert decision.terms[1] is second
    assert decision.terms[0].source is first_source
    assert decision.terms[1].source is second_source
    assert first_source.Ac is not second_source.Ac
    assert first_source.Ab is not second_source.Ab
    assert first_source.Auc is not second_source.Auc
    assert first_source.Aub is not second_source.Aub


@pytest.mark.parametrize("exception", [RuntimeError("boom"), ValueError("bad")])
def test_ordinary_internal_exception_fails_closed_with_original_identity(
    monkeypatch, exception
):
    inner, pre, _, outer = _small_chain()
    expr = _expr(
        (_full_term(_source("ordinary"), inner, pre, outer),),
        outer.shape[0],
    )

    def raise_exception(*args, **kwargs):
        raise exception

    monkeypatch.setattr(c2, "_c1_estimate_descriptor", raise_exception)
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == f"planner_failed_{type(exception).__name__}"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_base_exception_is_never_swallowed(monkeypatch, exception_type):
    inner, pre, _, outer = _small_chain()
    expr = _expr(
        (_full_term(_source("base-exception"), inner, pre, outer),),
        outer.shape[0],
    )

    def raise_base_exception(*args, **kwargs):
        raise exception_type()

    monkeypatch.setattr(c2, "_c1_estimate_descriptor", raise_base_exception)
    with pytest.raises(exception_type):
        plan_s0_c2_residual_distributive_lineage(
            expr, _all_rows(expr.n_out)
        )


def test_pure_planner_never_executes_or_expands_csr_and_makes_no_speed_claim(
    monkeypatch,
):
    inner, pre, _, outer = _small_chain()
    expr = _expr(
        (_full_term(_source("pure"), inner, pre, outer),),
        outer.shape[0],
    )
    inner_before = inner._kernel.copy()
    outer_before = outer._kernel.copy()
    pre_before = pre._diagonal.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("pure C2 planner executed a linear operator")

    monkeypatch.setattr(ImplicitConv2DOp, "to_csr_reference", forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, "left_compose", forbidden)
    monkeypatch.setattr(DiagonalLinearOp, "to_csr_reference", forbidden)
    monkeypatch.setattr(DiagonalLinearOp, "left_compose", forbidden)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.emission_executed is False
    assert "deferred" in decision.plan.emission_status
    assert np.array_equal(inner._kernel, inner_before)
    assert np.array_equal(outer._kernel, outer_before)
    assert np.array_equal(pre._diagonal, pre_before)
    plan_fields = {field.name for field in fields(C2PureLineagePlan)}
    assert not plan_fields.intersection(
        {
            "unfused_work",
            "one_quarter_gate_proven",
            "whole_state_before",
            "whole_state_after",
            "concurrency_speedup",
        }
    )
    assert any("one_quarter" in claim for claim in decision.plan.no_claims)
    assert any("whole_state" in claim for claim in decision.plan.no_claims)
    assert any("live_operators" in claim for claim in decision.plan.no_claims)
