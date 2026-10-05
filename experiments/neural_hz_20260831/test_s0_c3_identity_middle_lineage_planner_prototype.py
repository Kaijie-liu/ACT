"""P0 tests for the isolated S0-C3 identity-middle pure planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from experiments.neural_hz_20260831 import (
    s0_c3_identity_middle_lineage_planner_prototype as c3,
)
from experiments.neural_hz_20260831.s0_c1_pure_tail_planner_prototype import (
    PlannerBudget,
)
from experiments.neural_hz_20260831.s0_c2_residual_distributive_lineage_planner_prototype import (
    BoundaryMark as C2BoundaryMark,
    LineageAffineExprView as C2Expr,
    LineageAffineTermView as C2Term,
    plan_s0_c2_residual_distributive_lineage,
)
from experiments.neural_hz_20260831.s0_c3_identity_middle_lineage_planner_prototype import (
    C3AffineExprView,
    C3AffineTermView,
    C3_CERTIFICATE_SCHEMA,
    C3_DESCRIPTOR_TAG,
    C3_PAYLOAD_PREFIX,
    C3_RULE_ID,
    GraphEventEvidence,
    OrderedPathCertificate,
    plan_s0_c3_identity_middle_lineage,
)


@dataclass(frozen=True)
class SourceView:
    frame_id: object
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


def _source(name: str, *, frame_id: object = ("frame", 17), exact=True):
    return SourceView(
        frame_id=frame_id,
        exact=exact,
        name=name,
        c=np.array([0.25, -0.5]),
        Gc=np.array([[0.5], [0.25]]),
        Gb=np.array([[0.125], [-0.25]]),
        Ac=object(),
        Ab=object(),
        b=object(),
        Auc=object(),
        Aub=object(),
        ub=object(),
    )


def _kernel(out_channels, in_channels, *, offset=0):
    size = out_channels * in_channels * 3 * 3
    values = ((np.arange(size) + offset) % 11 - 5) / 16.0
    return values.reshape(out_channels, in_channels, 3, 3)


def _ops(*, offset=0, post=False, row_mask=None):
    inner = ImplicitConv2DOp(
        _kernel(3, 2, offset=offset), (1, 2, 4, 5), padding=1
    )
    pre_values = np.broadcast_to(
        np.array([1.0, -0.5, 2.0]).reshape(1, 3, 1, 1),
        inner.output_shape,
    ).copy()
    pre = DiagonalLinearOp(pre_values.reshape(-1))
    common_post = None
    if post:
        post_values = np.broadcast_to(
            np.array([0.25, 1.0, -2.0]).reshape(1, 3, 1, 1),
            inner.output_shape,
        ).copy()
        common_post = DiagonalLinearOp(post_values.reshape(-1))
    outer = ImplicitConv2DOp(
        _kernel(2, 3, offset=offset + 2),
        inner.output_shape,
        padding=1,
        row_mask=row_mask,
    )
    return inner, pre, common_post, outer


def _all_rows(size):
    return np.ones(size, dtype=bool)


def _unary_event(
    kind,
    occurrence,
    previous_occurrence,
    input_value,
    output_value,
    position,
    op,
):
    return GraphEventEvidence(
        kind=kind,
        occurrence_token=occurrence,
        producer_occurrence_tokens=(previous_occurrence,),
        graph_predecessor_occurrence_tokens=(previous_occurrence,),
        input_value_tokens=(input_value,),
        output_value_token=output_value,
        operator_position=position,
        selected_input=0,
        operator_occurrence=op,
    )


def _branch_endpoint(name, segment, *, bias_after_pre_scale=False):
    if not segment:
        return ("entry", name), ("entry-value", name)
    last = len(segment) - 1
    if bias_after_pre_scale and isinstance(segment[-1], DiagonalLinearOp):
        return ("branch-bias", name, last), ("branch-bias-value", name, last)
    return ("branch-op", name, last), ("branch-value", name, last)


def _certificate(
    *,
    name,
    prefix=(),
    segment,
    post,
    outer,
    output=(),
    add_producers,
    add_values,
    selected_input,
    add_occurrence=("common-add", 9),
    add_output=("common-add-value", 9),
    bias_after_pre_scale=False,
):
    events = []
    previous_occurrence = ("entry", name)
    previous_value = ("entry-value", name)
    position = 0
    for op in prefix:
        kind = "CONV" if isinstance(op, ImplicitConv2DOp) else "SCALE"
        occurrence = ("prefix-op", name, position)
        output_value = ("prefix-value", name, position)
        events.append(
            _unary_event(
                kind,
                occurrence,
                previous_occurrence,
                previous_value,
                output_value,
                position,
                op,
            )
        )
        previous_occurrence = occurrence
        previous_value = output_value
        position += 1
    for op in segment:
        kind = "CONV" if isinstance(op, ImplicitConv2DOp) else "SCALE"
        occurrence = ("branch-op", name, position)
        output_value = ("branch-value", name, position)
        events.append(
            _unary_event(
                kind,
                occurrence,
                previous_occurrence,
                previous_value,
                output_value,
                position,
                op,
            )
        )
        previous_occurrence = occurrence
        previous_value = output_value
        position += 1
        if bias_after_pre_scale and kind == "SCALE":
            bias_occurrence = ("branch-bias", name, position - 1)
            bias_output = ("branch-bias-value", name, position - 1)
            events.append(
                GraphEventEvidence(
                    kind="BIAS",
                    occurrence_token=bias_occurrence,
                    producer_occurrence_tokens=(previous_occurrence,),
                    graph_predecessor_occurrence_tokens=(previous_occurrence,),
                    input_value_tokens=(previous_value,),
                    output_value_token=bias_output,
                    operator_position=position,
                    selected_input=0,
                    bias_payload=(0.125, -0.25, 0.5),
                )
            )
            previous_occurrence = bias_occurrence
            previous_value = bias_output

    assert add_producers[selected_input] == previous_occurrence
    assert add_values[selected_input] == previous_value
    events.append(
        GraphEventEvidence(
            kind="ADD",
            occurrence_token=add_occurrence,
            producer_occurrence_tokens=tuple(add_producers),
            graph_predecessor_occurrence_tokens=tuple(add_producers),
            input_value_tokens=tuple(add_values),
            output_value_token=add_output,
            operator_position=position,
            selected_input=selected_input,
        )
    )
    previous_occurrence = add_occurrence
    previous_value = add_output

    suffix = (*tuple(post), outer, *tuple(output))
    for suffix_index, op in enumerate(suffix):
        semantic_position = position + suffix_index
        is_output = suffix_index >= len(post) + 1
        if is_output:
            occurrence = ("term-output", name, suffix_index)
            output_value = ("term-output-value", name, suffix_index)
        else:
            occurrence = ("shared-op", suffix_index)
            output_value = ("shared-value", suffix_index)
        kind = "CONV" if isinstance(op, ImplicitConv2DOp) else "SCALE"
        events.append(
            _unary_event(
                kind,
                occurrence,
                previous_occurrence,
                previous_value,
                output_value,
                semantic_position,
                op,
            )
        )
        previous_occurrence = occurrence
        previous_value = output_value

    return OrderedPathCertificate(
        schema=C3_CERTIFICATE_SCHEMA,
        entry_producer_occurrence_token=("entry", name),
        entry_value_token=("entry-value", name),
        events=tuple(events),
    )


def _term(
    source,
    *,
    name,
    prefix=(),
    segment,
    post,
    outer,
    output=(),
    add_producers=None,
    add_values=None,
    selected_input=0,
    bias_after_pre_scale=False,
):
    if add_producers is None or add_values is None:
        endpoint = _branch_endpoint(
            name, segment, bias_after_pre_scale=bias_after_pre_scale
        )
        add_producers = (endpoint[0], ("other-producer", name))
        add_values = (endpoint[1], ("other-value", name))
    cert = _certificate(
        name=name,
        prefix=tuple(prefix),
        segment=tuple(segment),
        post=tuple(post),
        outer=outer,
        output=tuple(output),
        add_producers=tuple(add_producers),
        add_values=tuple(add_values),
        selected_input=selected_input,
        bias_after_pre_scale=bias_after_pre_scale,
    )
    return C3AffineTermView(
        source=source,
        operators=(
            *tuple(prefix),
            *tuple(segment),
            *tuple(post),
            outer,
            *tuple(output),
        ),
        certificate=cert,
    )


def _expr(terms, n_out, *, bias=None, frame_id=("frame", 17)):
    if bias is None:
        bias = np.linspace(-0.25, 0.25, n_out)
    return C3AffineExprView(tuple(terms), bias, n_out, frame_id)


def test_conv_empty_diagonal_and_true_add_identity_skip_plan_atomically():
    inner, _, _, outer = _ops()
    complete_endpoint = _branch_endpoint("main", (inner,))
    skip_endpoint = _branch_endpoint("skip", ())
    add_producers = (complete_endpoint[0], skip_endpoint[0])
    add_values = (complete_endpoint[1], skip_endpoint[1])
    main_source = _source("main")
    skip_source = _source("skip")
    main = _term(
        main_source,
        name="main",
        segment=(inner,),
        post=(),
        outer=outer,
        add_producers=add_producers,
        add_values=add_values,
        selected_input=0,
    )
    skip = _term(
        skip_source,
        name="skip",
        segment=(),
        post=(),
        outer=outer,
        add_producers=add_producers,
        add_values=add_values,
        selected_input=1,
    )
    bias = np.linspace(-1.0, 1.0, outer.shape[0])
    expr = _expr((main, skip), outer.shape[0], bias=bias)

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is bias
    assert decision.plan.original_terms is expr.terms
    assert decision.plan.identity_term_indices == (1,)
    assert decision.plan.zero_support_term_indices == ()
    assert tuple(use.term_index for use in decision.plan.uses) == (0,)
    assert decision.plan.uses[0].pre_add_diagonal_count == 0
    assert decision.plan.uses[0].source is main_source
    assert decision.plan.uses[0].certificate is main.certificate
    assert len(decision.plan.uses[0].certificate_digest) == 64
    assert expr.terms[1] is skip
    assert expr.terms[1].source is skip_source
    assert decision.plan.rule_id == C3_RULE_ID
    assert decision.plan.requests[0].content_key[0] == C3_DESCRIPTOR_TAG
    assert decision.plan.emission_executed is False


def test_caller_cannot_hide_a_first_conv_in_a_freely_chosen_prefix():
    prefix, _, _, _ = _ops(offset=5)
    inner, _, _, outer = _ops()
    endpoint_occurrence = ("branch-op", "prefixed", 1)
    endpoint_value = ("branch-value", "prefixed", 1)
    term = _term(
        _source("prefixed"),
        name="prefixed",
        prefix=(prefix,),
        segment=(inner,),
        post=(),
        outer=outer,
        add_producers=(endpoint_occurrence, ("other-prefix", 1)),
        add_values=(endpoint_value, ("other-prefix-value", 1)),
        selected_input=0,
    )
    expr = _expr((term,), outer.shape[0])

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "two_convs_before_common_add"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias


def test_empty_and_explicit_one_numeric_descriptors_reuse_but_lineage_does_not():
    inner_empty, _, _, outer = _ops(offset=0)
    inner_one, _, _, _ = _ops(offset=0)
    ones = DiagonalLinearOp(np.ones(inner_one.shape[0]))
    empty_endpoint = _branch_endpoint("empty", (inner_empty,))
    one_endpoint = _branch_endpoint("one", (inner_one, ones))
    add_producers = (empty_endpoint[0], one_endpoint[0])
    add_values = (empty_endpoint[1], one_endpoint[1])
    empty = _term(
        _source("empty"),
        name="empty",
        segment=(inner_empty,),
        post=(),
        outer=outer,
        add_producers=add_producers,
        add_values=add_values,
        selected_input=0,
    )
    explicit = _term(
        _source("one"),
        name="one",
        segment=(inner_one, ones),
        post=(),
        outer=outer,
        add_producers=add_producers,
        add_values=add_values,
        selected_input=1,
    )
    expr = _expr((empty, explicit), outer.shape[0])

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert len(decision.plan.requests) == 1
    assert decision.plan.requests[0].term_indices == (0, 1)
    assert decision.plan.requests[0].use_count == 2
    assert tuple(use.pre_add_diagonal_count for use in decision.plan.uses) == (0, 1)
    assert decision.plan.uses[0].certificate is empty.certificate
    assert decision.plan.uses[1].certificate is explicit.certificate
    assert decision.plan.uses[0].certificate_digest != decision.plan.uses[1].certificate_digest
    assert decision.plan.uses[0].descriptor_content_key == decision.plan.uses[1].descriptor_content_key


def test_rounded_equal_but_exact_real_distinct_diagonal_products_do_not_reuse():
    inner_a, _, _, outer = _ops(offset=0)
    inner_b, _, _, _ = _ops(offset=0)
    tenth = np.float64(0.1)
    rounded_square = np.float64(tenth * tenth)
    d_a1 = DiagonalLinearOp(np.full(inner_a.shape[0], tenth))
    d_a2 = DiagonalLinearOp(np.full(inner_a.shape[0], tenth))
    d_b = DiagonalLinearOp(np.full(inner_b.shape[0], rounded_square))
    segment_a = (inner_a, d_a1, d_a2)
    segment_b = (inner_b, d_b)
    endpoint_a = _branch_endpoint("dyadic-a", segment_a)
    endpoint_b = _branch_endpoint("dyadic-b", segment_b)
    producers = (endpoint_a[0], endpoint_b[0])
    values = (endpoint_a[1], endpoint_b[1])
    first = _term(
        _source("dyadic-a"),
        name="dyadic-a",
        segment=segment_a,
        post=(),
        outer=outer,
        add_producers=producers,
        add_values=values,
        selected_input=0,
    )
    second = _term(
        _source("dyadic-b"),
        name="dyadic-b",
        segment=segment_b,
        post=(),
        outer=outer,
        add_producers=producers,
        add_values=values,
        selected_input=1,
    )
    expr = _expr((first, second), outer.shape[0])

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert decision.accepted, decision.reason
    assert np.float64(tenth * tenth) == rounded_square
    assert Fraction.from_float(float(tenth)) ** 2 != Fraction.from_float(
        float(rounded_square)
    )
    assert len(decision.plan.requests) == 2
    assert (
        decision.plan.uses[0].descriptor_content_key
        != decision.plan.uses[1].descriptor_content_key
    )


def test_conv_diagonal_bias_add_post_diagonal_outer_is_explicit_and_accepted():
    inner, pre, post, outer = _ops(post=True)
    assert post is not None
    endpoint = _branch_endpoint(
        "biased", (inner, pre), bias_after_pre_scale=True
    )
    term = _term(
        _source("biased"),
        name="biased",
        segment=(inner, pre),
        post=(post,),
        outer=outer,
        add_producers=(endpoint[0], ("other", 1)),
        add_values=(endpoint[1], ("other-value", 1)),
        selected_input=0,
        bias_after_pre_scale=True,
    )
    expr = _expr((term,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert decision.accepted, decision.reason
    use = decision.plan.uses[0]
    assert use.pre_add_diagonals == (pre,)
    assert use.post_add_diagonals == (post,)
    assert any(event.kind == "BIAS" for event in use.certificate.events)
    assert use.certificate_snapshot_payload.startswith(b"S0C3CERT1")


def test_mixed_zero_support_term_and_predicates_remain_identical():
    inner, _, _, outer = _ops()
    n_out = outer.shape[0]
    live_output = DiagonalLinearOp(np.ones(n_out))
    zero_output = DiagonalLinearOp(np.zeros(n_out))
    first_endpoint = _branch_endpoint("live", (inner,))
    second_endpoint = _branch_endpoint("zero", (inner,))
    add_producers = (first_endpoint[0], second_endpoint[0])
    add_values = (first_endpoint[1], second_endpoint[1])
    live_source = _source("live")
    zero_source = _source("zero")
    live = _term(
        live_source,
        name="live",
        segment=(inner,),
        post=(),
        outer=outer,
        output=(live_output,),
        add_producers=add_producers,
        add_values=add_values,
        selected_input=0,
    )
    zero = _term(
        zero_source,
        name="zero",
        segment=(inner,),
        post=(),
        outer=outer,
        output=(zero_output,),
        add_producers=add_producers,
        add_values=add_values,
        selected_input=1,
    )
    expr = _expr((live, zero), n_out)

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(n_out)
    )

    assert decision.accepted, decision.reason
    assert decision.plan.zero_support_term_indices == (1,)
    assert tuple(use.term_index for use in decision.plan.uses) == (0,)
    assert decision.terms[1] is zero
    assert decision.terms[1].source is zero_source
    for field in ("Ac", "Ab", "b", "Auc", "Aub", "ub"):
        assert getattr(decision.terms[1].source, field) is getattr(zero_source, field)


@pytest.mark.parametrize(
    ("segment_factory", "reason"),
    [
        (lambda inner, pre: (pre,), "nonempty_branch_without_inner_conv"),
        (lambda inner, pre: (inner, inner), "two_convs_before_common_add"),
    ],
)
def test_nonempty_segments_outside_the_one_grammar_reject(segment_factory, reason):
    inner, pre, _, outer = _ops()
    segment = segment_factory(inner, pre)
    endpoint = _branch_endpoint("bad", segment)
    term = _term(
        _source("bad"),
        name="bad",
        segment=segment,
        post=(),
        outer=outer,
        add_producers=(endpoint[0], ("other", 2)),
        add_values=(endpoint[1], ("other-value", 2)),
    )
    expr = _expr((term,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == reason
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias
    assert decision.plan is None


@pytest.mark.parametrize("kind", ["RESHAPE", "NONLINEAR", "POOL", "DENSE"])
def test_recognized_hard_graph_event_rejects_whole_request(kind):
    inner, _, _, outer = _ops()
    term = _term(
        _source("barrier"),
        name="barrier",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    cert = term.certificate
    events = list(cert.events)
    events[0] = replace(events[0], kind=kind)
    malformed = replace(term, certificate=replace(cert, events=tuple(events)))
    expr = _expr((malformed,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == f"core_contains_{kind.lower()}_event"


def test_nested_add_rejects_before_any_plan():
    inner, _, _, outer = _ops()
    term = _term(
        _source("nested"),
        name="nested",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    cert = term.certificate
    events = list(cert.events)
    common = next(index for index, event in enumerate(events) if event.kind == "ADD")
    old = events[common]
    nested_occurrence = ("older-add", 4)
    nested_value = ("older-add-value", 4)
    nested = GraphEventEvidence(
        kind="ADD",
        occurrence_token=nested_occurrence,
        producer_occurrence_tokens=(old.producer_occurrence_tokens[0], ("older-other", 4)),
        graph_predecessor_occurrence_tokens=(old.producer_occurrence_tokens[0], ("older-other", 4)),
        input_value_tokens=(old.input_value_tokens[0], ("older-other-value", 4)),
        output_value_token=nested_value,
        operator_position=old.operator_position,
        selected_input=0,
    )
    events.insert(common, nested)
    events[common + 1] = replace(
        old,
        producer_occurrence_tokens=(nested_occurrence, old.producer_occurrence_tokens[1]),
        graph_predecessor_occurrence_tokens=(nested_occurrence, old.graph_predecessor_occurrence_tokens[1]),
        input_value_tokens=(nested_value, old.input_value_tokens[1]),
    )
    malformed = replace(term, certificate=replace(cert, events=tuple(events)))
    expr = _expr((malformed,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "nested_add_event"


def test_unknown_duplicate_disordered_and_extra_events_fail_closed():
    inner, _, _, outer = _ops()
    base = _term(
        _source("events"),
        name="events",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    cert = base.certificate

    unknown_events = list(cert.events)
    unknown_events[0] = replace(unknown_events[0], kind="MYSTERY")
    unknown = replace(base, certificate=replace(cert, events=tuple(unknown_events)))
    decision = plan_s0_c3_identity_middle_lineage(
        _expr((unknown,), outer.shape[0]), _all_rows(outer.shape[0])
    )
    assert not decision.accepted
    assert decision.reason == "unknown_graph_event"

    duplicate_events = list(cert.events)
    duplicate_events[1] = replace(
        duplicate_events[1], occurrence_token=duplicate_events[0].occurrence_token
    )
    duplicate = replace(base, certificate=replace(cert, events=tuple(duplicate_events)))
    decision = plan_s0_c3_identity_middle_lineage(
        _expr((duplicate,), outer.shape[0]), _all_rows(outer.shape[0])
    )
    assert not decision.accepted
    assert decision.reason == "duplicate_graph_event"

    disorder_events = list(cert.events)
    disorder_events[-1] = replace(disorder_events[-1], operator_position=0)
    disorder = replace(base, certificate=replace(cert, events=tuple(disorder_events)))
    decision = plan_s0_c3_identity_middle_lineage(
        _expr((disorder,), outer.shape[0]), _all_rows(outer.shape[0])
    )
    assert not decision.accepted
    assert decision.reason == "event_operator_order"

    extra_events = list(cert.events)
    add_index = next(index for index, event in enumerate(extra_events) if event.kind == "ADD")
    add = extra_events[add_index]
    extra_events.insert(
        add_index,
        GraphEventEvidence(
            kind="SCALE",
            occurrence_token=("omitted-scale", 30),
            producer_occurrence_tokens=(add.producer_occurrence_tokens[0],),
            graph_predecessor_occurrence_tokens=(add.graph_predecessor_occurrence_tokens[0],),
            input_value_tokens=(add.input_value_tokens[0],),
            output_value_token=("omitted-scale-value", 30),
            operator_position=add.operator_position,
            selected_input=0,
            operator_occurrence=None,
        ),
    )
    extra_events[add_index + 1] = replace(
        add,
        producer_occurrence_tokens=(("omitted-scale", 30), add.producer_occurrence_tokens[1]),
        graph_predecessor_occurrence_tokens=(("omitted-scale", 30), add.graph_predecessor_occurrence_tokens[1]),
        input_value_tokens=(("omitted-scale-value", 30), add.input_value_tokens[1]),
    )
    extra = replace(base, certificate=replace(cert, events=tuple(extra_events)))
    decision = plan_s0_c3_identity_middle_lineage(
        _expr((extra,), outer.shape[0]), _all_rows(outer.shape[0])
    )
    assert not decision.accepted
    assert decision.reason == "unaccounted_linear_event"


def test_bias_without_scale_and_producer_graph_mismatch_reject_stably():
    inner, _, _, outer = _ops()
    term = _term(
        _source("bias"),
        name="bias",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    cert = term.certificate
    events = list(cert.events)
    add_index = next(index for index, event in enumerate(events) if event.kind == "ADD")
    add = events[add_index]
    bias_occurrence = ("bad-bias", 1)
    bias_value = ("bad-bias-value", 1)
    events.insert(
        add_index,
        GraphEventEvidence(
            kind="BIAS",
            occurrence_token=bias_occurrence,
            producer_occurrence_tokens=(add.producer_occurrence_tokens[0],),
            graph_predecessor_occurrence_tokens=(add.graph_predecessor_occurrence_tokens[0],),
            input_value_tokens=(add.input_value_tokens[0],),
            output_value_token=bias_value,
            operator_position=add.operator_position,
            selected_input=0,
            bias_payload=(1.0,),
        ),
    )
    events[add_index + 1] = replace(
        add,
        producer_occurrence_tokens=(bias_occurrence, add.producer_occurrence_tokens[1]),
        graph_predecessor_occurrence_tokens=(bias_occurrence, add.graph_predecessor_occurrence_tokens[1]),
        input_value_tokens=(bias_value, add.input_value_tokens[1]),
    )
    bad_bias = replace(term, certificate=replace(cert, events=tuple(events)))
    expr = _expr((bad_bias,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "bias_without_paired_scale_predecessor"

    mismatch_events = list(cert.events)
    mismatch_events[0] = replace(
        mismatch_events[0],
        graph_predecessor_occurrence_tokens=(("different-pred", 1),),
    )
    mismatch = replace(term, certificate=replace(cert, events=tuple(mismatch_events)))
    expr = _expr((mismatch,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "producer_predecessor_mismatch"


def test_multiplicative_graph_event_cannot_claim_multiple_inputs():
    inner, _, _, outer = _ops()
    term = _term(
        _source("multi-input"),
        name="multi-input",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    first = term.certificate.events[0]
    malformed_first = replace(
        first,
        producer_occurrence_tokens=(
            first.producer_occurrence_tokens[0],
            ("spurious-producer", 1),
        ),
        graph_predecessor_occurrence_tokens=(
            first.graph_predecessor_occurrence_tokens[0],
            ("spurious-producer", 1),
        ),
        input_value_tokens=(
            first.input_value_tokens[0],
            ("spurious-value", 1),
        ),
    )
    malformed = replace(
        term,
        certificate=replace(
            term.certificate,
            events=(malformed_first, *term.certificate.events[1:]),
        ),
    )
    expr = _expr((malformed,), outer.shape[0])

    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )

    assert not decision.accepted
    assert decision.reason == "multiplicative_event_arity"
    assert decision.expression is expr
    assert decision.terms is expr.terms
    assert decision.bias is expr.bias


def test_graph_event_cannot_reuse_entry_occurrence_or_its_input_value():
    inner, _, _, outer = _ops()
    term = _term(
        _source("self-loop"),
        name="self-loop",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    cert = term.certificate
    events = list(cert.events)
    first = events[0]
    add_index = next(index for index, event in enumerate(events) if event.kind == "ADD")
    add = events[add_index]
    events[0] = replace(
        first, occurrence_token=cert.entry_producer_occurrence_token
    )
    events[add_index] = replace(
        add,
        producer_occurrence_tokens=(
            cert.entry_producer_occurrence_token,
            add.producer_occurrence_tokens[1],
        ),
        graph_predecessor_occurrence_tokens=(
            cert.entry_producer_occurrence_token,
            add.graph_predecessor_occurrence_tokens[1],
        ),
    )
    malformed = replace(
        term, certificate=replace(cert, events=tuple(events))
    )
    expr = _expr((malformed,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == "duplicate_graph_event"

    events = list(cert.events)
    first = events[0]
    events[0] = replace(
        first, output_value_token=cert.entry_value_token
    )
    add = events[add_index]
    events[add_index] = replace(
        add,
        input_value_tokens=(
            cert.entry_value_token,
            add.input_value_tokens[1],
        ),
    )
    malformed = replace(
        term, certificate=replace(cert, events=tuple(events))
    )
    expr = _expr((malformed,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == "duplicate_output_value"

    events = list(cert.events)
    first_output = events[0].output_value_token
    events[-1] = replace(events[-1], output_value_token=first_output)
    malformed = replace(
        term, certificate=replace(cert, events=tuple(events))
    )
    expr = _expr((malformed,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == "duplicate_output_value"


def test_certificate_is_mandatory_and_mutable_or_wrong_schema_tokens_reject():
    inner, _, _, outer = _ops()
    term = _term(
        _source("cert"),
        name="cert",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    for malformed, expected in (
        (replace(term, certificate=None), "semantic_path_certificate_missing"),
        (
            replace(term, certificate=replace(term.certificate, schema="wrong")),
            "semantic_path_certificate_schema",
        ),
        (
            replace(
                term,
                certificate=replace(
                    term.certificate,
                    entry_producer_occurrence_token=["mutable"],
                ),
            ),
            "occurrence_token_not_stable",
        ),
    ):
        expr = _expr((malformed,), outer.shape[0])
        decision = plan_s0_c3_identity_middle_lineage(
            expr, _all_rows(expr.n_out)
        )
        assert not decision.accepted
        assert decision.reason == expected
        assert decision.expression is expr
        assert decision.terms is expr.terms
        assert decision.bias is expr.bias


def test_current_payload_mutation_changes_certificate_and_descriptor_digest():
    inner, _, _, outer = _ops()
    term = _term(
        _source("snapshot"),
        name="snapshot",
        segment=(inner,),
        post=(),
        outer=outer,
    )
    expr = _expr((term,), outer.shape[0])
    first = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert first.accepted, first.reason
    first_cert = first.plan.uses[0].certificate_digest
    first_descriptor = first.plan.uses[0].descriptor_content_key

    inner._kernel[0, 0, 0, 0] += 0.125
    second = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert second.accepted, second.reason
    assert second.plan.uses[0].certificate_digest != first_cert
    assert second.plan.uses[0].descriptor_content_key != first_descriptor


def test_equal_valued_distinct_outer_and_post_occurrences_do_not_merge():
    inner, _, post, outer = _ops(post=True)
    _, _, post_clone, outer_clone = _ops(post=True)
    assert post is not None and post_clone is not None
    a_endpoint = _branch_endpoint("a", (inner,))
    b_endpoint = _branch_endpoint("b", (inner,))
    producers = (a_endpoint[0], b_endpoint[0])
    values = (a_endpoint[1], b_endpoint[1])
    first = _term(
        _source("a"), name="a", segment=(inner,), post=(post,), outer=outer,
        add_producers=producers, add_values=values, selected_input=0,
    )
    second_outer = _term(
        _source("b"), name="b", segment=(inner,), post=(post,), outer=outer_clone,
        add_producers=producers, add_values=values, selected_input=1,
    )
    expr = _expr((first, second_outer), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "shared_outer_occurrence_mismatch"

    second_post = _term(
        _source("b"), name="b", segment=(inner,), post=(post_clone,), outer=outer,
        add_producers=producers, add_values=values, selected_input=1,
    )
    expr = _expr((first, second_post), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "post_add_suffix_occurrence_mismatch"


def test_common_add_value_equal_but_distinct_occurrence_rejects():
    inner, _, _, outer = _ops()
    a_endpoint = _branch_endpoint("a", (inner,))
    b_endpoint = _branch_endpoint("b", (inner,))
    producers = (a_endpoint[0], b_endpoint[0])
    values = (a_endpoint[1], b_endpoint[1])
    first = _term(
        _source("a"), name="a", segment=(inner,), post=(), outer=outer,
        add_producers=producers, add_values=values, selected_input=0,
    )
    second = _term(
        _source("b"), name="b", segment=(inner,), post=(), outer=outer,
        add_producers=producers, add_values=values, selected_input=1,
    )
    events = list(second.certificate.events)
    add_index = next(i for i, event in enumerate(events) if event.kind == "ADD")
    events[add_index] = replace(events[add_index], occurrence_token=("different-add", 9))
    following = events[add_index + 1]
    events[add_index + 1] = replace(
        following,
        producer_occurrence_tokens=(("different-add", 9),),
        graph_predecessor_occurrence_tokens=(("different-add", 9),),
    )
    second = replace(second, certificate=replace(second.certificate, events=tuple(events)))
    expr = _expr((first, second), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "latest_common_add_occurrence_or_snapshot_mismatch"


def test_channel_nonstationary_and_malformed_silent_term_reject_complete_request():
    inner, _, _, outer = _ops()
    values = np.ones(inner.shape[0])
    values[1] = 2.0
    nonstationary = DiagonalLinearOp(values)
    endpoint = _branch_endpoint("bad-scale", (inner, nonstationary))
    bad = _term(
        _source("bad-scale"),
        name="bad-scale",
        segment=(inner, nonstationary),
        post=(),
        outer=outer,
        add_producers=(endpoint[0], ("other", 3)),
        add_values=(endpoint[1], ("other-value", 3)),
    )
    expr = _expr((bad,), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert "not_channel_stationary" in decision.reason

    live_endpoint = _branch_endpoint("live", (inner,))
    silent_endpoint = _branch_endpoint("silent", (inner,))
    producers = (live_endpoint[0], silent_endpoint[0])
    vals = (live_endpoint[1], silent_endpoint[1])
    live = _term(
        _source("live"), name="live", segment=(inner,), post=(), outer=outer,
        add_producers=producers, add_values=vals, selected_input=0,
    )
    zero = DiagonalLinearOp(np.zeros(outer.shape[0]))
    silent = _term(
        _source("silent"), name="silent", segment=(inner,), post=(), outer=outer,
        output=(zero,), add_producers=producers, add_values=vals, selected_input=1,
    )
    malformed_event = replace(
        silent.certificate.events[0], occurrence_token=["mutable"]
    )
    malformed_cert = replace(
        silent.certificate,
        events=(malformed_event, *silent.certificate.events[1:]),
    )
    silent = replace(silent, certificate=malformed_cert)
    expr = _expr((live, silent), outer.shape[0])
    decision = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not decision.accepted
    assert decision.reason == "occurrence_token_not_stable"
    assert decision.terms[1] is silent


def _two_unique_expression():
    inner_a, _, _, outer = _ops(offset=0)
    inner_b, _, _, _ = _ops(offset=7)
    a_endpoint = _branch_endpoint("budget-a", (inner_a,))
    b_endpoint = _branch_endpoint("budget-b", (inner_b,))
    producers = (a_endpoint[0], b_endpoint[0])
    values = (a_endpoint[1], b_endpoint[1])
    terms = (
        _term(
            _source("budget-a"), name="budget-a", segment=(inner_a,),
            post=(), outer=outer, add_producers=producers,
            add_values=values, selected_input=0,
        ),
        _term(
            _source("budget-b"), name="budget-b", segment=(inner_b,),
            post=(), outer=outer, add_producers=producers,
            add_values=values, selected_input=1,
        ),
    )
    return _expr(terms, outer.shape[0])


def _exact_budget(plan):
    estimates = tuple(request.estimate for request in plan.requests)
    reservation = plan.reservation
    return PlannerBudget(
        max_unique_descriptors=len(plan.requests),
        max_descriptor_contraction_products=max(x.contraction_products for x in estimates),
        max_descriptor_coefficient_entries=max(x.coefficient_entries for x in estimates),
        max_descriptor_resident_bytes=max(x.resident_bytes for x in estimates),
        max_descriptor_result_nnz=max(x.result_nnz_upper for x in estimates),
        max_transaction_contraction_products=reservation.contraction_products,
        max_transaction_total_work=reservation.total_work,
        max_transaction_coefficient_entries=reservation.coefficient_entries,
        max_transaction_resident_bytes=reservation.resident_bytes,
        max_transaction_transient_bytes=reservation.transient_bytes,
        max_transaction_result_nnz=reservation.result_nnz_upper,
    )


_BUDGET_GATES = (
    ("max_unique_descriptors", "unique_descriptor_limit"),
    ("max_descriptor_contraction_products", "descriptor_contraction_product_limit"),
    ("max_descriptor_coefficient_entries", "descriptor_coefficient_entry_limit"),
    ("max_descriptor_resident_bytes", "descriptor_resident_byte_limit"),
    ("max_descriptor_result_nnz", "descriptor_result_nnz_limit"),
    ("max_transaction_contraction_products", "transaction_contraction_product_limit"),
    ("max_transaction_total_work", "transaction_work_limit"),
    ("max_transaction_coefficient_entries", "transaction_coefficient_entry_limit"),
    ("max_transaction_resident_bytes", "transaction_resident_byte_limit"),
    ("max_transaction_transient_bytes", "transaction_transient_byte_limit"),
    ("max_transaction_result_nnz", "transaction_result_nnz_limit"),
)


@pytest.mark.parametrize(("field_name", "reason"), _BUDGET_GATES)
def test_each_resource_gate_accepts_exact_limit_and_rejects_one_unit_below(
    field_name, reason
):
    expr = _two_unique_expression()
    initial = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert initial.accepted, initial.reason
    exact = _exact_budget(initial.plan)
    at_limit = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out), budget=exact
    )
    assert at_limit.accepted, (field_name, at_limit.reason)
    value = getattr(exact, field_name)
    assert value > 0
    below = replace(exact, **{field_name: value - 1})
    rejected = plan_s0_c3_identity_middle_lineage(
        expr, _all_rows(expr.n_out), budget=below
    )
    assert not rejected.accepted
    assert rejected.reason == reason
    assert rejected.expression is expr
    assert rejected.terms is expr.terms
    assert rejected.bias is expr.bias


def test_ordinary_failure_returns_original_identity_and_async_interrupt_propagates(monkeypatch):
    inner, _, _, outer = _ops()
    term = _term(
        _source("failure"), name="failure", segment=(inner,), post=(), outer=outer
    )
    expr = _expr((term,), outer.shape[0])

    def ordinary(*args, **kwargs):
        raise MemoryError("ordinary allocation failure")

    monkeypatch.setattr(c3, "_conv_snapshot", ordinary)
    rejected = plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))
    assert not rejected.accepted
    assert rejected.reason == "planner_failed_MemoryError"
    assert rejected.expression is expr
    assert rejected.terms is expr.terms
    assert rejected.terms[0] is term
    assert rejected.bias is expr.bias

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(c3, "_conv_snapshot", interrupt)
    with pytest.raises(KeyboardInterrupt):
        plan_s0_c3_identity_middle_lineage(expr, _all_rows(expr.n_out))


def test_c2_frozen_empty_middle_fixture_still_rejects():
    inner, _, _, outer = _ops()
    source = _source("c2-empty")
    term = C2Term(
        source=source,
        operators=(inner, outer),
        boundaries=(
            C2BoundaryMark(
                kind="ADD",
                occurrence_key=("common-add", 9),
                operator_index=1,
            ),
        ),
    )
    expr = C2Expr(
        terms=(term,),
        bias=np.zeros(outer.shape[0]),
        n_out=outer.shape[0],
        frame_id=("frame", 17),
    )
    decision = plan_s0_c2_residual_distributive_lineage(
        expr, _all_rows(expr.n_out)
    )
    assert not decision.accepted
    assert decision.reason == "nonempty_branch_without_complete_chain"


def test_dense_and_csr_dyadic_exact_set_oracles_preserve_bias_once():
    source_a = _source("oracle-a")
    source_b = _source("oracle-b")
    xi = np.array([0.5])
    z = np.array([-1.0])
    witness_a = source_a.c + source_a.Gc @ xi + source_a.Gb @ z
    witness_b = source_b.c + source_b.Gc @ xi + source_b.Gb @ z
    inner_a = np.array([[0.5, 0.25], [-0.25, 1.0]])
    inner_b = np.array([[1.0, -0.5], [0.125, 0.25]])
    diagonal_a = np.diag([1.0, 0.5])
    diagonal_b = np.eye(2)
    common = np.diag([0.5, -1.0])
    outer = np.array([[0.5, -1.0], [0.25, 0.5]])
    bias = np.array([0.25, -0.125])
    beta = np.array([0.5, 0.25])

    before = outer @ common @ (
        diagonal_a @ inner_a @ witness_a
        + diagonal_b @ inner_b @ witness_b
        + bias
    ) + beta
    after = (
        (outer @ common @ diagonal_a @ inner_a) @ witness_a
        + (outer @ common @ diagonal_b @ inner_b) @ witness_b
        + outer @ common @ bias
        + beta
    )
    assert np.array_equal(before, after)

    csr_before = sparse.csr_matrix(outer) @ sparse.csr_matrix(common) @ (
        sparse.csr_matrix(diagonal_a) @ sparse.csr_matrix(inner_a) @ witness_a
        + sparse.csr_matrix(diagonal_b) @ sparse.csr_matrix(inner_b) @ witness_b
        + bias
    ) + beta
    csr_after = (
        sparse.csr_matrix(outer @ common @ diagonal_a @ inner_a) @ witness_a
        + sparse.csr_matrix(outer @ common @ diagonal_b @ inner_b) @ witness_b
        + sparse.csr_matrix(outer @ common) @ bias
        + beta
    )
    assert np.array_equal(np.asarray(csr_before).reshape(-1), np.asarray(csr_after).reshape(-1))
    assert source_a.Ac is not source_b.Ac
    assert source_a.Aub is not source_b.Aub


def test_interface_namespace_and_source_have_no_instance_selectors():
    signature = inspect.signature(plan_s0_c3_identity_middle_lineage)
    assert tuple(signature.parameters) == ("expr", "output_support", "budget")
    assert signature.parameters["budget"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(c3, "execute_s0_c3_plan")
    assert not hasattr(c3, "compile_s0_c3_plan")
    assert C3_RULE_ID.startswith("s0_c3_")
    assert C3_DESCRIPTOR_TAG.startswith("s0_c3_")
    assert C3_PAYLOAD_PREFIX == b"S0C3D1"
    assert not hasattr(OrderedPathCertificate, "accepted")
    source_text = Path(c3.__file__).read_text(encoding="utf-8").lower()
    forbidden = ("i" + "id", "fam" + "ily", "lay" + "er", "mar" + "gin", "ver" + "dict")
    assert all(token not in source_text for token in forbidden)
    assert "s0_c2" not in source_text
