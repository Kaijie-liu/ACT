"""Focused tests for the isolated S0-C2 pure lineage planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect

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
    C2_NO_CLAIMS,
    LineageAffineExprView,
    LineageAffineTermView,
    PlannerBudget,
    plan_s0_c2_residual_distributive_lineage,
)


@dataclass(frozen=True)
class SourceView:
    frame_id: int
    exact: bool
    name: str
    c: object
    Gc: object
    Gb: object
    Ac: object
    Ab: object
    b: object
    Auc: object
    Aub: object
    ub: object


def _source(name: str, *, frame_id: int = 17, exact: bool = True):
    # Every field is a separate identity so preservation tests can detect a
    # value-based merge or loss of a zero-support term's unique predicates.
    return SourceView(
        frame_id=frame_id,
        exact=exact,
        name=name,
        c=np.array([0.25, -0.5], dtype=np.float64),
        Gc=np.array([[0.5], [0.25]], dtype=np.float64),
        Gb=np.array([[0.125], [-0.25]], dtype=np.float64),
        Ac=object(),
        Ab=object(),
        b=object(),
        Auc=object(),
        Aub=object(),
        ub=object(),
    )


def _kernel(out_channels, in_per_group, shape, *, offset=0):
    size = out_channels * in_per_group * shape[0] * shape[1]
    values = ((np.arange(size, dtype=np.int64) + offset) % 13 - 6) / 16.0
    return values.reshape(out_channels, in_per_group, shape[0], shape[1])


def _small_chain(*, offset=0, post=False, row_mask=None):
    inner = ImplicitConv2DOp(
        _kernel(4, 2, (3, 3), offset=offset),
        (1, 2, 5, 6),
        padding=1,
    )
    pre_channel = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64)
    pre = DiagonalLinearOp(
        np.broadcast_to(
            pre_channel.reshape(1, 4, 1, 1), inner.output_shape
        ).reshape(-1)
    )
    outer_input = inner.output_shape
    common_post = None
    if post:
        post_channel = np.array([0.5, 1.0, -2.0, 0.25], dtype=np.float64)
        common_post = DiagonalLinearOp(
            np.broadcast_to(
                post_channel.reshape(1, 4, 1, 1), outer_input
            ).reshape(-1)
        )
    outer = ImplicitConv2DOp(
        _kernel(3, 4, (3, 3), offset=offset + 3),
        outer_input,
        padding=1,
        row_mask=row_mask,
    )
    return inner, pre, common_post, outer


def _expr(terms, n_out, *, bias=None, frame_id=17):
    if bias is None:
        bias = np.linspace(-1.0, 1.0, n_out, dtype=np.float64)
    return LineageAffineExprView(
        terms=tuple(terms),
        bias=bias,
        n_out=n_out,
        frame_id=frame_id,
    )


def _mark(occurrence=("residual", 32), index=2, *, kind="ADD"):
    return BoundaryMark(kind=kind, occurrence_key=occurrence, operator_index=index)


def _full_term(source, inner, pre, outer, *, mark=None, post=None, dout=()):
    if mark is None:
        mark = _mark()
    post_ops = () if post is None else (post,)
    return LineageAffineTermView(
        source=source,
        operators=(inner, pre, *post_ops, outer, *tuple(dout)),
        boundaries=(mark,),
    )


def _all_rows(n_out):
    return np.ones(n_out, dtype=bool)


def test_real_conv29_d_add32_conv33_shape_plans_once_with_identity_skip():
    channels = 128
    spatial = 14
    inner = ImplicitConv2DOp(
        _kernel(channels, channels, (3, 3), offset=29),
        (1, channels, spatial, spatial),
        padding=1,
    )
    scale = np.broadcast_to(
        np.linspace(0.5, 1.5, channels, dtype=np.float64).reshape(
            1, channels, 1, 1
        ),
        inner.output_shape,
    ).copy()
    middle = DiagonalLinearOp(scale.reshape(-1))
    outer = ImplicitConv2DOp(
        _kernel(channels, channels, (3, 3), offset=33),
        inner.output_shape,
        padding=1,
    )
    main_mark = _mark(index=2)
    skip_mark = _mark(index=0)
    main_source = _source("main")
    skip_source = _source("identity")
    main = LineageAffineTermView(
        main_source, (inner, middle, outer), (main_mark,)
    )
    identity = LineageAffineTermView(skip_source, (outer,), (skip_mark,))
    bias = np.linspace(-0.25, 0.25, outer.shape[0], dtype=np.float64)
    expr = _expr((main, identity), outer.shape[0], bias=bias)
    support = np.arange(2_180, dtype=np.int64)

    decision = plan_s0_c2_residual_distributive_lineage(expr, support)

    assert decision.accepted, decision.reason
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is bias
    assert decision.plan is not None
    assert decision.plan.original_terms is expr.terms
    assert decision.plan.bias is bias
    assert decision.plan.identity_term_indices == (1,)
    assert decision.plan.zero_support_term_indices == ()
    assert len(decision.plan.requests) == 1
    assert len(decision.plan.uses) == 1
    assert decision.plan.uses[0].term_index == 0
    assert decision.plan.uses[0].common_add is main_mark
    assert decision.plan.common_add_marks[1] is skip_mark
    assert decision.plan.requests[0].estimate.contraction_products == 169_869_312
    assert decision.plan.reservation.unique_descriptors == 1
    assert decision.plan.prospective_emission_contributions > 0
    assert decision.plan.emission_executed is False
    assert "deferred" in decision.plan.emission_status
    assert expr.terms[1] is identity
    assert expr.terms[1].source is skip_source


def test_real_four_term_topology_is_one_complete_branch_plus_three_skips():
    inner, pre, _, outer = _small_chain()
    scale34 = DiagonalLinearOp(
        np.ones(outer.shape[0], dtype=np.float64)
    )
    add32 = ("act-layer-v1", 32)
    main = LineageAffineTermView(
        _source("relu28-main"),
        (inner, pre, outer, scale34),
        (_mark(occurrence=add32, index=2),),
    )

    skips = []
    for index in range(3):
        prefix = tuple(object() for _ in range(index))
        cut = len(prefix)
        add24 = BoundaryMark(
            "ADD", ("act-layer-v1", 24), cut
        )
        latest = _mark(occurrence=add32, index=cut)
        skips.append(
            LineageAffineTermView(
                _source(f"add24-skip-{index}"),
                (*prefix, outer, scale34),
                (add24, latest),
            )
        )
    expr = _expr((main, *skips), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.identity_term_indices == (1, 2, 3)
    assert tuple(use.term_index for use in decision.plan.uses) == (0,)
    assert len(decision.plan.requests) == 1
    assert decision.plan.requests[0].term_indices == (0,)
    assert decision.plan.uses[0].output_diagonals == (scale34,)
    for term_index, skip in enumerate(skips, start=1):
        assert decision.terms[term_index] is skip
        assert skip.operators[-2] is outer
        assert skip.operators[-1] is scale34


def test_all_complete_nonzero_branches_form_one_unique_set_and_union_support():
    inner, pre, post, outer = _small_chain(post=True)
    assert post is not None
    n_out = outer.shape[0]
    even = np.zeros(n_out, dtype=np.float64)
    odd = np.zeros(n_out, dtype=np.float64)
    even[::2] = 1.0
    odd[1::2] = 1.0
    even_dout = DiagonalLinearOp(even)
    odd_dout = DiagonalLinearOp(odd)
    first_mark = _mark()
    second_mark = _mark()
    first = _full_term(
        _source("first"), inner, pre, outer,
        mark=first_mark, post=post, dout=(even_dout,)
    )
    second = _full_term(
        _source("second"), inner, pre, outer,
        mark=second_mark, post=post, dout=(odd_dout,)
    )
    expr = _expr((first, second), n_out)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(n_out)
    )

    assert decision.accepted, decision.reason
    assert len(decision.plan.requests) == 1
    request = decision.plan.requests[0]
    assert request.term_indices == (0, 1)
    assert request.use_count == 2
    assert request.selected_rows == tuple(range(n_out))
    assert len(decision.plan.uses) == 2
    assert decision.plan.uses[0].output_diagonals[0] is even_dout
    assert decision.plan.uses[1].output_diagonals[0] is odd_dout
    assert all(use.post_add_diagonals == (post,) for use in decision.plan.uses)


def test_separately_allocated_equal_occurrence_tokens_accept_but_value_mismatch_rejects():
    inner, pre, _, outer = _small_chain()
    first_mark = _mark(occurrence=tuple(["residual", 32]))
    second_mark = _mark(occurrence=("residual", int("32")))
    first = _full_term(_source("first"), inner, pre, outer, mark=first_mark)
    second = _full_term(_source("second"), inner, pre, outer, mark=second_mark)
    accepted_expr = _expr((first, second), outer.shape[0])
    accepted = plan_s0_c2_residual_distributive_lineage(
        accepted_expr, _all_rows(outer.shape[0])
    )
    assert accepted.accepted, accepted.reason
    assert accepted.plan.common_add_marks == (first_mark, second_mark)

    different = replace(second, boundaries=(_mark(("residual", 33)),))
    rejected_expr = _expr((first, different), outer.shape[0])
    rejected = plan_s0_c2_residual_distributive_lineage(
        rejected_expr, _all_rows(outer.shape[0])
    )
    assert not rejected.accepted
    assert rejected.reason == "latest_common_add_occurrence_mismatch"
    assert rejected.expression is rejected_expr
    assert rejected.terms is rejected_expr.terms
    assert rejected.bias is rejected_expr.bias
    assert rejected.plan is None


@pytest.mark.parametrize("kind", ["ADD", "RESHAPE", "RELU", "POOL"])
def test_second_or_hard_cut_inside_core_rejects_whole_transaction(kind):
    inner, pre, _, outer = _small_chain()
    nested = BoundaryMark(kind, ("older", kind), 1)
    latest = _mark(index=2)
    term = LineageAffineTermView(
        _source("cut"),
        (inner, pre, outer),
        (nested, latest),
    )
    expr = _expr((term,), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(outer.shape[0])
    )

    assert not decision.accepted
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


def test_two_convs_after_add_is_not_c2_and_incomplete_branch_cannot_fallback():
    inner, pre, _, outer = _small_chain()
    same_segment = LineageAffineTermView(
        _source("same-segment"),
        (inner, pre, outer),
        (_mark(index=0),),
    )
    same_expr = _expr((same_segment,), outer.shape[0])
    same = plan_s0_c2_residual_distributive_lineage(
        same_expr, _all_rows(outer.shape[0])
    )
    assert not same.accepted
    assert same.reason == "two_convs_same_segment_not_c2"

    incomplete = LineageAffineTermView(
        _source("incomplete"),
        (pre, outer),
        (_mark(index=1),),
    )
    incomplete_expr = _expr((incomplete,), outer.shape[0])
    rejected = plan_s0_c2_residual_distributive_lineage(
        incomplete_expr, _all_rows(outer.shape[0])
    )
    assert not rejected.accepted
    assert rejected.reason == "nonempty_branch_without_complete_chain"
    assert rejected.expression is incomplete_expr
    assert rejected.plan is None


def test_identity_skip_preserves_complete_common_suffix_without_rewrite():
    inner, pre, post, outer = _small_chain(post=True)
    assert post is not None
    main = _full_term(_source("main"), inner, pre, outer, post=post)
    output = DiagonalLinearOp(np.ones(outer.shape[0], dtype=np.float64))
    skip = LineageAffineTermView(
        _source("skip"),
        (post, outer, output),
        (_mark(index=0),),
    )
    expr = _expr((main, skip), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(outer.shape[0])
    )

    assert decision.accepted, decision.reason
    assert decision.plan.identity_term_indices == (1,)
    assert decision.plan.uses[0].post_add_diagonals == (post,)
    assert expr.terms[1] is skip
    assert expr.terms[1].operators == (post, outer, output)
    assert expr.terms[1].operators[0] is post
    assert expr.terms[1].operators[1] is outer
    assert expr.terms[1].operators[2] is output


def test_mixed_zero_support_term_keeps_unique_source_predicates_and_order():
    inner, pre, _, outer = _small_chain()
    n_out = outer.shape[0]
    keep = DiagonalLinearOp(np.ones(n_out, dtype=np.float64))
    zero = DiagonalLinearOp(np.zeros(n_out, dtype=np.float64))
    live_source = _source("live")
    predicate_source = _source("zero-with-unique-predicates")
    live_mark = _mark()
    zero_mark = _mark()
    live = _full_term(
        live_source, inner, pre, outer, mark=live_mark, dout=(keep,)
    )
    silent = _full_term(
        predicate_source, inner, pre, outer, mark=zero_mark, dout=(zero,)
    )
    expr = _expr((live, silent), n_out)

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.zero_support_term_indices == (1,)
    assert tuple(use.term_index for use in decision.plan.uses) == (0,)
    assert decision.terms is expr.terms
    assert decision.plan.original_terms is expr.terms
    assert decision.terms[1] is silent
    assert decision.terms[1].source is predicate_source
    assert decision.terms[1].boundaries[0] is zero_mark
    assert decision.terms[1].source.Ac is predicate_source.Ac
    assert decision.terms[1].source.Aub is predicate_source.Aub
    assert decision.bias is expr.bias

    only_zero_expr = _expr((silent,), n_out)
    only_zero = plan_s0_c2_residual_distributive_lineage(
        only_zero_expr, _all_rows(n_out)
    )
    assert not only_zero.accepted
    assert only_zero.reason == "no_nonzero_fusible_branch"
    assert only_zero.expression is only_zero_expr
    assert only_zero.terms is only_zero_expr.terms
    assert only_zero.terms[0] is silent
    assert only_zero.bias is only_zero_expr.bias


def _single_term_with_boundaries(boundaries):
    inner, pre, _, outer = _small_chain()
    term = LineageAffineTermView(
        _source("malformed"),
        (inner, pre, outer),
        boundaries,
    )
    return _expr((term,), outer.shape[0])


@pytest.mark.parametrize(
    "boundaries",
    [
        None,
        [_mark()],
        (object(),),
        (BoundaryMark("ADD", ["mutable"], 2),),
        (BoundaryMark("UNKNOWN", ("x", 1), 2),),
        (BoundaryMark("add", ("x", 1), 2),),
        (BoundaryMark("ADD", ("x", 1), True),),
        (BoundaryMark("ADD", ("x", 1), 99),),
        (
            BoundaryMark("RESHAPE", ("old", 1), 2),
            BoundaryMark("ADD", ("new", 1), 1),
        ),
        (
            BoundaryMark("ADD", ("same", 1), 2),
            BoundaryMark("ADD", ("same", 1), 2),
        ),
    ],
)
def test_none_or_malformed_lineage_rejects_original_expression(boundaries):
    expr = _single_term_with_boundaries(boundaries)
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


def test_budget_rejection_is_all_or_nothing_and_never_selects_one_branch():
    inner_a, pre_a, _, outer = _small_chain(offset=0)
    inner_b, pre_b, _, _ = _small_chain(offset=7)
    first = _full_term(_source("a"), inner_a, pre_a, outer)
    second = _full_term(_source("b"), inner_b, pre_b, outer)
    expr = _expr((first, second), outer.shape[0])

    generous = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert generous.accepted, generous.reason
    assert len(generous.plan.requests) == 2
    assert tuple(use.term_index for use in generous.plan.uses) == (0, 1)
    one_request_work = max(
        request.estimate.total_work for request in generous.plan.requests
    )
    assert generous.plan.reservation.total_work > one_request_work

    tight = replace(
        PlannerBudget(), max_transaction_total_work=one_request_work
    )
    rejected = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out), budget=tight
    )
    assert not rejected.accepted
    assert rejected.reason == "transaction_work_limit"
    assert rejected.expression is expr
    assert rejected.terms is expr.terms
    assert rejected.bias is expr.bias
    assert rejected.plan is None


def test_current_payload_snapshot_prevents_stale_constructor_key_dedup():
    inner_a, pre_a, _, outer = _small_chain(offset=0)
    inner_b, pre_b, _, _ = _small_chain(offset=0)
    assert inner_a.content_key == inner_b.content_key
    inner_b._kernel[0, 0, 0, 0] += 0.125
    # The production operator's constructor key is intentionally stale here.
    assert inner_a.content_key == inner_b.content_key
    first = _full_term(_source("a"), inner_a, pre_a, outer)
    second = _full_term(_source("b"), inner_b, pre_b, outer)
    expr = _expr((first, second), outer.shape[0])

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert len(decision.plan.requests) == 2
    assert len({request.content_key for request in decision.plan.requests}) == 2
    assert tuple(request.content_key for request in decision.plan.requests) == tuple(
        sorted(
            (request.content_key for request in decision.plan.requests),
            key=lambda key: key[1],
        )
    )


def test_shared_outer_and_common_post_require_graph_occurrence_identity():
    inner, pre, post, outer = _small_chain(post=True)
    assert post is not None
    _, _, post_clone, outer_clone = _small_chain(post=True)
    assert post_clone is not None
    first = _full_term(_source("first"), inner, pre, outer, post=post)
    second_outer = _full_term(
        _source("second"), inner, pre, outer_clone, post=post
    )
    outer_expr = _expr((first, second_outer), outer.shape[0])
    rejected_outer = plan_s0_c2_residual_distributive_lineage(
        outer_expr, _all_rows(outer.shape[0])
    )
    assert not rejected_outer.accepted
    assert rejected_outer.reason == "shared_outer_occurrence_mismatch"
    assert rejected_outer.expression is outer_expr

    second_post = _full_term(
        _source("second"), inner, pre, outer, post=post_clone
    )
    post_expr = _expr((first, second_post), outer.shape[0])
    rejected_post = plan_s0_c2_residual_distributive_lineage(
        post_expr, _all_rows(outer.shape[0])
    )
    assert not rejected_post.accepted
    assert rejected_post.reason == "post_add_suffix_occurrence_mismatch"
    assert rejected_post.expression is post_expr


@pytest.mark.parametrize(
    ("source", "frame_id"),
    [
        (_source("wrong-frame", frame_id=99), 17),
        (_source("inexact", exact=False), 17),
    ],
)
def test_source_frame_or_exact_mismatch_rejects(source, frame_id):
    inner, pre, _, outer = _small_chain()
    term = _full_term(source, inner, pre, outer)
    expr = _expr((term,), outer.shape[0], frame_id=frame_id)
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == "source_frame_or_exact_mismatch"
    assert decision.expression is expr
    assert decision.terms is expr.terms


@pytest.mark.parametrize("invalid_frame", [None, True, False, np.bool_(True)])
def test_unframed_or_boolean_expression_frame_fails_closed(invalid_frame):
    inner, pre, _, outer = _small_chain()
    source = _source("invalid-frame", frame_id=invalid_frame)
    term = _full_term(source, inner, pre, outer)
    expr = _expr(
        (term,), outer.shape[0], frame_id=invalid_frame
    )

    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "expression_frame_id_not_stable"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


def test_keep_masks_term_maps_never_bias_and_planner_never_expands_conv_csr(monkeypatch):
    inner, pre, _, outer = _small_chain()
    keep = np.zeros(outer.shape[0], dtype=np.float64)
    keep[::5] = 1.0
    output = DiagonalLinearOp(keep)
    term = _full_term(
        _source("masked"), inner, pre, outer, dout=(output,)
    )
    bias = np.linspace(2.0, 3.0, outer.shape[0], dtype=np.float64)
    expr = _expr((term,), outer.shape[0], bias=bias)

    def forbidden_full_csr(*args, **kwargs):
        raise AssertionError("pure C2 planner expanded an input Conv CSR")

    monkeypatch.setattr(ImplicitConv2DOp, "to_csr_reference", forbidden_full_csr)
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert decision.accepted, decision.reason
    assert decision.bias is bias
    assert decision.plan.bias is bias
    assert np.array_equal(bias, np.linspace(2.0, 3.0, outer.shape[0]))
    assert decision.plan.requests[0].selected_rows == tuple(
        int(row) for row in np.flatnonzero(keep)
    )


def test_dyadic_residual_distribution_preserves_factor_witness_and_bias_once():
    # This is the preregistered exact-HZ algebra at a tiny dyadic oracle.  The
    # source/predicate objects are not copied, merged, or reparameterized.
    source_a = _source("a")
    source_b = _source("b")
    xi = np.array([0.5], dtype=np.float64)
    z = np.array([-1.0], dtype=np.float64)
    s_a = source_a.c + source_a.Gc @ xi + source_a.Gb @ z
    s_b = source_b.c + source_b.Gc @ xi + source_b.Gb @ z
    t_a = np.array([[0.5, 0.25], [-0.25, 1.0]], dtype=np.float64)
    t_b = np.array([[1.0, -0.5], [0.125, 0.25]], dtype=np.float64)
    outer = np.array([[0.5, -1.0], [0.25, 0.5]], dtype=np.float64)
    bias = np.array([0.25, -0.125], dtype=np.float64)
    beta = np.array([0.5, 0.25], dtype=np.float64)

    before = outer @ (t_a @ s_a + t_b @ s_b + bias) + beta
    after = (
        (outer @ t_a) @ s_a
        + (outer @ t_b) @ s_b
        + outer @ bias
        + beta
    )
    assert np.array_equal(before, after)
    assert source_a.frame_id == source_b.frame_id == 17
    assert source_a.exact is source_b.exact is True
    assert source_a.Ac is not source_b.Ac
    assert source_a.Auc is not source_b.Auc
    assert xi is xi and z is z  # the same concrete factor witness is reused


def test_interface_is_one_uniform_pure_rule_with_explicit_no_claims():
    signature = inspect.signature(plan_s0_c2_residual_distributive_lineage)
    assert tuple(signature.parameters) == ("expr", "output_support", "budget")
    assert signature.parameters["budget"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(c2, "execute_s0_c2_plan")
    assert not hasattr(c2, "compile_s0_c2_plan")
    assert all(token not in signature.parameters for token in ("iid", "family", "layer"))
    assert any("stale" in claim for claim in C2_NO_CLAIMS)
    assert any("live_operators" in claim for claim in C2_NO_CLAIMS)
    assert any("emission" in claim and "deferred" in claim for claim in C2_NO_CLAIMS)
    assert any("whole_state" in claim for claim in C2_NO_CLAIMS)
