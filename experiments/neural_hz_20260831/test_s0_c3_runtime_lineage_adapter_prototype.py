"""Unit and adversarial tests for the isolated S0-C3 runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import inspect
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from scipy import sparse

from act.back_end.hybridz_tf.exact_linear_op import (
    DiagonalLinearOp,
    ImplicitConv2DOp,
)
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831 import (
    s0_c3_runtime_lineage_adapter_prototype as adapter,
)


@dataclass
class LayerView:
    id: int
    kind: str
    params: dict
    in_vars: list[int]
    out_vars: list[int]


def _source(name: str, width: int, frame=43) -> SparseHZono:
    del name
    continuous = sparse.csr_matrix(
        (
            np.asarray([0.125]),
            (np.asarray([0]), np.asarray([0])),
        ),
        shape=(width, 1),
    )
    return SparseHZono(
        c=np.linspace(-0.25, 0.25, width),
        Gc=continuous,
        Gb=sparse.csr_matrix((width, 1), dtype=np.float64),
        Ac=sparse.csr_matrix((0, 1), dtype=np.float64),
        Ab=sparse.csr_matrix((0, 1), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse.csr_matrix((0, 1), dtype=np.float64),
        Aub=sparse.csr_matrix((0, 1), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        frame_id=frame,
        exact=True,
    )


def _source_with_factor_schema(
    width,
    n_cont,
    n_bin,
    *,
    frame=43,
    equality_value=None,
    inequality_value=None,
):
    gc = (
        sparse.csr_matrix(
            ([0.125], ([0], [0])),
            shape=(width, n_cont),
            dtype=np.float64,
        )
        if width and n_cont
        else sparse.csr_matrix((width, n_cont), dtype=np.float64)
    )
    gb = sparse.csr_matrix((width, n_bin), dtype=np.float64)
    if equality_value is None:
        ac = sparse.csr_matrix((0, n_cont), dtype=np.float64)
        ab = sparse.csr_matrix((0, n_bin), dtype=np.float64)
        b = np.zeros(0, dtype=np.float64)
    else:
        ac = sparse.csr_matrix(
            ([1.0], ([0], [0])), shape=(1, n_cont)
        )
        ab = sparse.csr_matrix((1, n_bin), dtype=np.float64)
        b = np.asarray([equality_value], dtype=np.float64)
    if inequality_value is None:
        auc = sparse.csr_matrix((0, n_cont), dtype=np.float64)
        aub = sparse.csr_matrix((0, n_bin), dtype=np.float64)
        ub = np.zeros(0, dtype=np.float64)
    else:
        auc = sparse.csr_matrix(
            ([1.0], ([0], [n_cont - 1])), shape=(1, n_cont)
        )
        aub = sparse.csr_matrix((1, n_bin), dtype=np.float64)
        ub = np.asarray([inequality_value], dtype=np.float64)
    return SparseHZono(
        c=np.linspace(-0.1, 0.1, width),
        Gc=gc,
        Gb=gb,
        Ac=ac,
        Ab=ab,
        b=b,
        Auc=auc,
        Aub=aub,
        ub=ub,
        frame_id=frame,
        exact=True,
    )


def _vars(start: int, size: int) -> list[int]:
    return list(range(start, start + size))


def _conv_params(op: ImplicitConv2DOp, *, bias=False) -> dict:
    params = {
        "weight": np.array(op._kernel, dtype=np.float32, copy=True),
        "input_shape": tuple(op._input_shape),
        "output_shape": tuple(op._output_shape),
        "kernel_size": tuple(op._kernel.shape[-2:]),
        "stride": tuple(op._stride),
        "padding": tuple(op._padding),
        "dilation": tuple(op._dilation),
        "groups": int(op._groups),
        "in_channels": int(op._input_shape[1]),
        "out_channels": int(op._output_shape[1]),
    }
    if bias:
        params["bias"] = np.linspace(-0.1, 0.1, op._output_shape[1])
    return params


def _successors(preds: dict[int, list[int]]) -> dict[int, list[int]]:
    result = {layer_id: [] for layer_id in preds}
    for layer_id in range(len(preds)):
        for predecessor in preds[layer_id]:
            result[predecessor].append(layer_id)
    return result


def _capture(layers, preds, succs, path, operators, registry, source):
    return adapter.capture_runtime_lineage_events(
        layers,
        preds,
        succs,
        path,
        operators,
        registry=registry,
        source=source,
    )


def _factor_allocator_for_sources(source_by_layer):
    allocator = adapter._begin_private_runtime_factor_allocator(
        object(), source_by_layer
    )
    rooted_frames = set()
    for source in source_by_layer.values():
        if source.frame_id not in rooted_frames:
            adapter._record_private_runtime_factor_frame_root(
                allocator, source
            )
            rooted_frames.add(source.frame_id)
        else:
            adapter._record_private_runtime_factor_source_prefix(
                allocator, source
            )
    return adapter._seal_private_runtime_factor_allocator(allocator)


def _private_arena_and_registry(
    layers, preds, succs, source_by_layer, affine_cache=None
):
    if affine_cache is None:
        affine_cache = {}
    factor_allocator = _factor_allocator_for_sources(source_by_layer)
    arena = adapter._begin_private_runtime_lineage_arena(
        layers,
        preds,
        succs,
        source_by_layer,
        affine_cache,
        factor_allocator,
    )
    for layer_id, source in source_by_layer.items():
        adapter._record_private_runtime_source_boundary(
            layers, preds, succs, arena, layer_id, source
        )
    registry = adapter._begin_private_runtime_lineage_registry(
        layers, preds, succs, arena
    )
    return arena, registry


def _capture_operand(
    layers, preds, succs, path, operators, registry, source
):
    return adapter._capture_private_runtime_operand_lineage(
        layers,
        preds,
        succs,
        path,
        operators,
        registry=registry,
        source=source,
    )


def _record_affine_cache(
    layers,
    preds,
    succs,
    registry,
    producer_layer_id,
    terms,
    *,
    bias=None,
    frame_id=43,
):
    width = len(layers[producer_layer_id].out_vars)
    if bias is None:
        bias = np.zeros(width, dtype=np.float64)
    return adapter._record_private_runtime_affine_expression_cache(
        layers,
        preds,
        succs,
        registry,
        producer_layer_id,
        tuple(terms),
        bias,
        width,
        frame_id,
    )


def _suffix_bias(
    initial_bias, post, graph_bias, outer, output
):
    value = DiagonalLinearOp.matvec(
        post, np.asarray(initial_bias, dtype=np.float64)
    )
    value = value + np.asarray(graph_bias, dtype=np.float64)
    value = ImplicitConv2DOp.matvec(outer, value)
    return DiagonalLinearOp.matvec(output, value)


def _fixture(
    main_operand_multiplicity=1,
    *,
    skip_cached_bias=None,
    outer_row_mask_index=None,
    terminal_successor_kinds=(),
    add_extra_params=None,
    conv_extra_params=None,
    conv_layer_extra_attrs=None,
    torch_graph_weights=False,
):
    in_shape = (1, 2, 2, 2)
    middle_shape = (1, 3, 2, 2)
    inner_kernel = np.arange(6, dtype=np.float64).reshape(3, 2, 1, 1) / 16.0
    outer_kernel = (np.arange(6, dtype=np.float64).reshape(2, 3, 1, 1) - 2) / 8.0
    inner = ImplicitConv2DOp(inner_kernel, in_shape)
    pre = DiagonalLinearOp(
        np.broadcast_to(np.array([1.0, -0.5, 2.0]).reshape(1, 3, 1, 1), middle_shape).reshape(-1)
    )
    post = DiagonalLinearOp(
        np.broadcast_to(np.array([0.25, 1.0, -2.0]).reshape(1, 3, 1, 1), middle_shape).reshape(-1)
    )
    unmasked_outer = ImplicitConv2DOp(outer_kernel, middle_shape)
    outer_row_mask = None
    if outer_row_mask_index is not None:
        outer_row_mask = np.ones(unmasked_outer.shape[0], dtype=bool)
        outer_row_mask[int(outer_row_mask_index)] = False
    outer = ImplicitConv2DOp(
        outer_kernel, middle_shape, row_mask=outer_row_mask
    )
    output = DiagonalLinearOp(np.linspace(0.5, 1.25, outer.shape[0]))

    v0 = _vars(0, inner.shape[1])
    v1 = _vars(100, inner.shape[0])
    v2 = _vars(200, inner.shape[0])
    v3 = _vars(300, inner.shape[0])
    v4 = _vars(400, inner.shape[0])
    v5 = _vars(500, inner.shape[0])
    v6 = _vars(600, inner.shape[0])
    v7 = _vars(700, inner.shape[0])
    v8 = _vars(800, outer.shape[0])
    v9 = _vars(900, outer.shape[0])

    add_params = {"x_vars": v3.copy(), "y_vars": v4.copy()}
    if add_extra_params is not None:
        add_params.update(add_extra_params)
    inner_params = _conv_params(inner)
    if conv_extra_params is not None:
        inner_params.update(conv_extra_params)
    outer_params = _conv_params(outer)
    if torch_graph_weights:
        if adapter.torch is None:
            raise RuntimeError("torch unavailable")
        inner_params["weight"] = adapter.torch.tensor(
            inner_params["weight"], dtype=adapter.torch.float64
        )
        outer_params["weight"] = adapter.torch.tensor(
            outer_params["weight"], dtype=adapter.torch.float64
        )
    layers = [
        LayerView(0, "RELU", {}, [], v0),
        LayerView(1, "CONV2D", inner_params, v0, v1),
        LayerView(2, "SCALE", {"a": pre._diagonal.copy()}, v1, v2),
        LayerView(3, "BIAS", {"c": np.linspace(-0.2, 0.2, len(v3))}, v2, v3),
        LayerView(4, "RELU", {}, [], v4),
        LayerView(5, "ADD", add_params, v3 + v4, v5),
        LayerView(6, "SCALE", {"a": post._diagonal.copy()}, v5, v6),
        LayerView(7, "BIAS", {"c": np.linspace(0.3, -0.3, len(v7))}, v6, v7),
        LayerView(8, "CONV2D", outer_params, v7, v8),
        LayerView(9, "SCALE", {"a": output._diagonal.copy()}, v8, v9),
    ]
    if conv_layer_extra_attrs is not None:
        for name, value in conv_layer_extra_attrs.items():
            setattr(layers[1], name, value)
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [],
        5: [3, 4],
        6: [5],
        7: [6],
        8: [7],
        9: [8],
    }
    for successor_offset, successor_kind in enumerate(
        terminal_successor_kinds, start=10
    ):
        successor_out = _vars(
            1000 + 100 * (successor_offset - 10), len(v9)
        )
        if successor_kind == "SCALE":
            successor_params = {
                "a": np.ones(len(v9), dtype=np.float64)
            }
        else:
            successor_params = {}
        layers.append(
            LayerView(
                successor_offset,
                successor_kind,
                successor_params,
                v9.copy(),
                successor_out,
            )
        )
        preds[successor_offset] = [9]
    succs = _successors(preds)
    main_ops = (inner, pre, post, outer, output)
    skip_ops = (post, outer, output)
    main_source = _source("main", len(v0))
    skip_source = _source("skip", len(v4))
    source_by_layer = {0: main_source, 4: skip_source}
    affine_cache = {}
    arena, registry = _private_arena_and_registry(
        layers, preds, succs, source_by_layer, affine_cache
    )
    main_operands = tuple(
        _capture_operand(
            layers,
            preds,
            succs,
            (0, 1, 2, 3),
            (inner, pre),
            registry,
            main_source,
        )
        for _ in range(main_operand_multiplicity)
    )
    skip_operand = _capture_operand(
        layers, preds, succs, (4,), (), registry, skip_source
    )
    _record_affine_cache(
        layers,
        preds,
        succs,
        registry,
        3,
        main_operands,
        bias=layers[3].params["c"].copy(),
    )
    if skip_cached_bias is None:
        skip_cached_bias = np.zeros(len(v4), dtype=np.float64)
    else:
        skip_cached_bias = np.asarray(
            skip_cached_bias, dtype=np.float64
        ).reshape(-1)
    _record_affine_cache(
        layers,
        preds,
        succs,
        registry,
        4,
        (skip_operand,),
        bias=skip_cached_bias,
    )
    adapter._record_private_runtime_add_operands(
        layers, preds, succs, registry, 5
    )
    combined_cached_bias = (
        np.asarray(layers[3].params["c"], dtype=np.float64)
        + skip_cached_bias
    )
    expected_bias = _suffix_bias(
        combined_cached_bias,
        post,
        layers[7].params["c"],
        unmasked_outer,
        output,
    )
    terminal_main_terms = tuple(
        _capture_operand(
            layers,
            preds,
            succs,
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            main_ops,
            registry,
            main_source,
        )
        for _ in range(main_operand_multiplicity)
    )
    terminal_skip = _capture_operand(
        layers,
        preds,
        succs,
        (4, 5, 6, 7, 8, 9),
        skip_ops,
        registry,
        skip_source,
    )
    _record_affine_cache(
        layers,
        preds,
        succs,
        registry,
        9,
        (*terminal_main_terms, terminal_skip),
        bias=expected_bias,
    )
    adapter._record_private_runtime_terminal_expression(
        layers, preds, succs, registry, 9
    )
    main_terms = tuple(
        adapter.RuntimeLineageTermView(
            main_source,
            main_ops,
            _capture(
                layers,
                preds,
                succs,
                (0, 1, 2, 3, 5, 6, 7, 8, 9),
                main_ops,
                registry,
                main_source,
            ),
        )
        for _ in range(main_operand_multiplicity)
    )
    skip_events = _capture(
        layers,
        preds,
        succs,
        (4, 5, 6, 7, 8, 9),
        skip_ops,
        registry,
        skip_source,
    )
    skip = adapter.RuntimeLineageTermView(skip_source, skip_ops, skip_events)
    expr = adapter.RuntimeLineageExprView(
        (*main_terms, skip), expected_bias, len(v9), 43
    )
    adapter._seal_private_runtime_lineage_registry(registry)
    return SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        expr=expr,
        inner=inner,
        pre=pre,
        post=post,
        outer=outer,
        unmasked_outer=unmasked_outer,
        output=output,
        main_cached_bias=np.asarray(
            layers[3].params["c"], dtype=np.float64
        ),
        skip_cached_bias=skip_cached_bias,
        expected_bias=expected_bias,
        source_by_layer=source_by_layer,
        affine_cache=affine_cache,
        arena=arena,
        registry=registry,
    )


def _plan(fixture):
    return adapter.plan_s0_c3_from_runtime_lineage(
        fixture.layers,
        fixture.preds,
        fixture.succs,
        fixture.expr,
        np.ones(fixture.expr.n_out, dtype=bool),
        registry=fixture.registry,
    )


def _fresh_registry(fixture, source_by_layer=None):
    if source_by_layer is not None and source_by_layer is not fixture.source_by_layer:
        return _private_arena_and_registry(
            fixture.layers, fixture.preds, fixture.succs, source_by_layer
        )[1]
    return adapter._begin_private_runtime_lineage_registry(
        fixture.layers, fixture.preds, fixture.succs, fixture.arena
    )


def _authorize_current_add_and_terminal(fixture, registry):
    adapter._record_private_runtime_add_operands(
        fixture.layers, fixture.preds, fixture.succs, registry, 5
    )
    adapter._record_private_runtime_terminal_expression(
        fixture.layers, fixture.preds, fixture.succs, registry, 9
    )


def _candidate_ending_at(fixture, terminal_layer_id):
    registry = _fresh_registry(fixture)
    _authorize_current_add_and_terminal(fixture, registry)
    main_full_path = (0, 1, 2, 3, 5, 6, 7, 8, 9)
    skip_full_path = (4, 5, 6, 7, 8, 9)
    main_path = main_full_path[
        : main_full_path.index(terminal_layer_id) + 1
    ]
    skip_path = skip_full_path[
        : skip_full_path.index(terminal_layer_id) + 1
    ]
    operator_by_layer = {
        1: fixture.inner,
        2: fixture.pre,
        6: fixture.post,
        8: fixture.outer,
        9: fixture.output,
    }
    main_ops = tuple(
        operator_by_layer[layer_id]
        for layer_id in main_path
        if layer_id in operator_by_layer
    )
    skip_ops = tuple(
        operator_by_layer[layer_id]
        for layer_id in skip_path
        if layer_id in operator_by_layer
    )
    main = adapter.RuntimeLineageTermView(
        fixture.expr.terms[0].source,
        main_ops,
        _capture(
            fixture.layers,
            fixture.preds,
            fixture.succs,
            main_path,
            main_ops,
            registry,
            fixture.expr.terms[0].source,
        ),
    )
    skip = adapter.RuntimeLineageTermView(
        fixture.expr.terms[-1].source,
        skip_ops,
        _capture(
            fixture.layers,
            fixture.preds,
            fixture.succs,
            skip_path,
            skip_ops,
            registry,
            fixture.expr.terms[-1].source,
        ),
    )
    n_out = len(fixture.layers[terminal_layer_id].out_vars)
    fixture.expr = adapter.RuntimeLineageExprView(
        (main, skip), np.zeros(n_out, dtype=np.float64), n_out, 43
    )
    fixture.registry = adapter._seal_private_runtime_lineage_registry(
        registry
    )
    return fixture


def _fresh_current_graph_registry(fixture):
    source_by_layer = {
        layer_id: _source(
            f"current-{layer_id}", source.n_out, source.frame_id
        )
        for layer_id, source in fixture.source_by_layer.items()
    }
    arena, registry = _private_arena_and_registry(
        fixture.layers, fixture.preds, fixture.succs, source_by_layer
    )
    return arena, registry, source_by_layer


def test_current_graph_events_build_the_existing_c3_plan_without_execution():
    fx = _fixture()
    result = _plan(fx)
    assert result.accepted, result.reason
    assert result.expression is fx.expr
    assert result.planner_decision.accepted
    assert result.planner_decision.expression is result.c3_expression
    assert result.planner_decision.plan.identity_term_indices == (1,)
    assert result.planner_decision.plan.emission_executed is False
    assert result.execution_enabled is False
    assert result.default_enabled is False
    assert result.gain == 0
    assert result.formal_baseline == "1870/2413"
    assert len(result.current_state_sha256) == 64


def test_terminal_hook_reads_owner_cache_and_accepts_no_candidate_expression():
    parameters = inspect.signature(
        adapter._record_private_runtime_terminal_expression
    ).parameters
    assert tuple(parameters) == (
        "layers",
        "preds",
        "succs",
        "registry",
        "terminal_layer_id",
    )
    fx = _fixture()
    snapshot = fx.registry.terminal_expression_snapshot
    assert snapshot.cache_entry is fx.affine_cache[9]
    assert snapshot.terms is fx.affine_cache[9].terms
    assert snapshot.terminal_layer_id == 9
    assert snapshot.terminal_consumer_layer_id is None


@pytest.mark.parametrize("terminal_layer_id", [8, 7, 6])
def test_candidate_cannot_truncate_trusted_terminal_suffix(
    terminal_layer_id,
):
    fx = _candidate_ending_at(_fixture(), terminal_layer_id)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_terminal_term_occurrence_mismatch"
    assert result.expression is fx.expr


def test_terminal_boundary_accepts_one_explicit_relu_consumer():
    fx = _fixture(terminal_successor_kinds=("RELU",))
    result = _plan(fx)
    assert result.accepted, result.reason
    assert (
        fx.registry.terminal_expression_snapshot.terminal_consumer_layer_id
        == 10
    )


def test_terminal_boundary_rejects_unrecorded_same_width_linear_suffix():
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_terminal_followed_by_unrecorded_linear_event",
    ):
        _fixture(terminal_successor_kinds=("SCALE",))


def test_terminal_boundary_rejects_a_second_consumer():
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_terminal_consumer_not_unique",
    ):
        _fixture(terminal_successor_kinds=("RELU", "RELU"))


def test_terminal_hook_rejects_a_caller_chosen_middle_cache_entry():
    fx = _fixture()
    registry = _fresh_registry(fx)
    adapter._record_private_runtime_add_operands(
        fx.layers, fx.preds, fx.succs, registry, 5
    )
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_terminal_followed_by_unrecorded_linear_event",
    ):
        adapter._record_private_runtime_terminal_expression(
            fx.layers, fx.preds, fx.succs, registry, 3
        )


def test_arbitrary_finite_candidate_bias_cannot_hit():
    fx = _fixture()
    fx.expr = replace(
        fx.expr,
        bias=np.linspace(100.0, 200.0, fx.expr.n_out),
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_terminal_expression_bias_mismatch"


def test_candidate_bias_cannot_omit_one_add_operand_bias():
    skip_bias = np.linspace(0.07, 0.19, 12)
    fx = _fixture(skip_cached_bias=skip_bias)
    wrong = _suffix_bias(
        fx.main_cached_bias,
        fx.post,
        fx.layers[7].params["c"],
        fx.unmasked_outer,
        fx.output,
    )
    assert not np.array_equal(wrong, fx.expected_bias)
    fx.expr = replace(fx.expr, bias=wrong)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_terminal_expression_bias_mismatch"


def test_three_plus_one_term_count_cannot_multiply_add_operand_bias():
    fx = _three_plus_one_fixture()
    wrong_pre_add = 3.0 * fx.main_cached_bias + fx.skip_cached_bias
    wrong = _suffix_bias(
        wrong_pre_add,
        fx.post,
        fx.layers[7].params["c"],
        fx.unmasked_outer,
        fx.output,
    )
    assert not np.array_equal(wrong, fx.expected_bias)
    fx.expr = replace(fx.expr, bias=wrong)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_terminal_expression_bias_mismatch"


def test_candidate_bias_cannot_reorder_suffix_bias_and_scale():
    fx = _fixture()
    wrong = _suffix_bias(
        fx.main_cached_bias + fx.skip_cached_bias
        + np.asarray(fx.layers[7].params["c"], dtype=np.float64),
        fx.post,
        np.zeros(fx.expr.n_out * 3 // 2, dtype=np.float64),
        fx.unmasked_outer,
        fx.output,
    )
    assert not np.array_equal(wrong, fx.expected_bias)
    fx.expr = replace(fx.expr, bias=wrong)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_terminal_expression_bias_mismatch"


def test_output_support_cannot_hide_a_global_bias_mismatch():
    fx = _fixture()
    wrong = fx.expr.bias.copy()
    wrong[0] += 0.125
    fx.expr = replace(fx.expr, bias=wrong)
    support = np.ones(fx.expr.n_out, dtype=bool)
    support[0] = False
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        fx.expr,
        support,
        registry=fx.registry,
    )
    assert not result.accepted
    assert result.reason == "runtime_terminal_expression_bias_mismatch"


def test_conv_row_mask_cannot_discard_global_bias_or_hide_mismatch():
    fx = _fixture(outer_row_mask_index=0)
    assert fx.outer._row_mask[0] == np.bool_(False)
    assert fx.expected_bias[0] != 0.0
    accepted = _plan(fx)
    assert accepted.accepted, accepted.reason

    wrong = fx.expr.bias.copy()
    wrong[0] = 0.0
    fx.expr = replace(fx.expr, bias=wrong)
    support = np.ones(fx.expr.n_out, dtype=bool)
    support[0] = False
    rejected = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        fx.expr,
        support,
        registry=fx.registry,
    )
    assert not rejected.accepted
    assert rejected.reason == "runtime_terminal_expression_bias_mismatch"


def test_add_selected_inputs_are_branch_specific_and_shared_suffix_is_identical():
    fx = _fixture()
    main_add = next(event for event in fx.expr.terms[0].events if event.kind == "ADD")
    skip_add = next(event for event in fx.expr.terms[1].events if event.kind == "ADD")
    assert main_add.occurrence_layer is skip_add.occurrence_layer is fx.layers[5]
    assert main_add.selected_input == 0
    assert skip_add.selected_input == 1
    for main, skip in zip(fx.expr.terms[0].events[-4:], fx.expr.terms[1].events[-4:]):
        assert main.occurrence_layer is skip.occurrence_layer
        assert main.operator_occurrence is skip.operator_occurrence


@pytest.mark.parametrize(
    "term_indices",
    [
        (0,),
        (0, 0),
        (0, 0, 1),
        (1, 1),
    ],
)
def test_add_expected_input_prefix_multiset_rejects_missing_duplicate_extra_or_cross_input(
    term_indices,
):
    fx = _fixture()
    fx.expr = replace(
        fx.expr,
        terms=tuple(fx.expr.terms[index] for index in term_indices),
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_add_input_multiplicity_mismatch"
    assert result.expression is fx.expr


def _three_plus_one_fixture():
    return _fixture(main_operand_multiplicity=3)


def test_three_terms_from_one_add_input_and_one_from_the_other_are_accepted():
    fx = _three_plus_one_fixture()
    result = _plan(fx)
    assert result.accepted, result.reason
    assert result.expression is fx.expr
    selected = [
        next(event for event in term.events if event.kind == "ADD").selected_input
        for term in fx.expr.terms
    ]
    assert selected == [0, 0, 0, 1]
    snapshot = fx.registry.add_operand_snapshots[0]
    assert snapshot.operands[0] is fx.affine_cache[3].terms
    assert snapshot.operands[1] is fx.affine_cache[4].terms
    assert tuple(len(terms) for terms in snapshot.operands) == (3, 1)


def test_add_hook_api_reads_cache_and_accepts_no_caller_operands_parameter():
    parameters = inspect.signature(
        adapter._record_private_runtime_add_operands
    ).parameters
    assert tuple(parameters) == (
        "layers",
        "preds",
        "succs",
        "registry",
        "add_layer_id",
    )
    one_plus_one = _fixture()
    three_plus_one = _three_plus_one_fixture()
    assert tuple(
        len(one_plus_one.affine_cache[layer_id].terms)
        for layer_id in (3, 4)
    ) == (1, 1)
    assert tuple(
        len(three_plus_one.affine_cache[layer_id].terms)
        for layer_id in (3, 4)
    ) == (3, 1)


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_three_plus_one_registry_enforces_exact_multiplicity(mode):
    fx = _three_plus_one_fixture()
    if mode == "missing":
        terms = (fx.expr.terms[0], fx.expr.terms[1], fx.expr.terms[3])
    else:
        terms = (
            fx.expr.terms[0],
            fx.expr.terms[1],
            fx.expr.terms[2],
            fx.expr.terms[2],
            fx.expr.terms[3],
        )
    fx.expr = replace(fx.expr, terms=terms)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_add_input_multiplicity_mismatch"


@pytest.mark.parametrize("main_only_count", [2, 3, 4])
def test_candidate_capture_cannot_self_authorize_a_missing_add_operand(
    main_only_count,
):
    fx = _fixture()
    registry = _fresh_registry(fx)
    template = fx.expr.terms[0]
    terms = []
    for _ in range(main_only_count):
        events = _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            template.operators,
            registry,
            template.source,
        )
        terms.append(replace(template, events=events))
    assert registry.add_operand_snapshots == []
    fx.expr = replace(fx.expr, terms=tuple(terms))
    fx.registry = adapter._seal_private_runtime_lineage_registry(registry)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_add_operand_snapshot_missing_or_duplicate"
    assert result.expression is fx.expr


@pytest.mark.parametrize("mode", ["empty_input1", "main_as_input1"])
def test_add_hook_requires_both_current_predecessor_operand_caches(mode):
    fx = _fixture()
    registry = _fresh_registry(fx)
    if mode == "empty_input1":
        fx.affine_cache.pop(4)
        reason = "runtime_affine_cache_ledger_incomplete"
    else:
        fx.affine_cache[4] = fx.affine_cache[3]
        reason = "runtime_affine_cache_ledger_cas_mismatch"
    with pytest.raises(adapter.RuntimeLineageReject, match=reason):
        adapter._record_private_runtime_add_operands(
            fx.layers, fx.preds, fx.succs, registry, 5
        )
    assert registry.add_operand_snapshots == []


def test_add_hook_rejects_a_genuinely_unrecorded_second_predecessor_cache():
    fx = _fixture()
    sources = {
        0: _source("missing-cache-main", fx.expr.terms[0].source.n_out, 43),
        4: _source("missing-cache-skip", fx.expr.terms[1].source.n_out, 43),
    }
    _, registry = _private_arena_and_registry(
        fx.layers, fx.preds, fx.succs, sources
    )
    main = _capture_operand(
        fx.layers,
        fx.preds,
        fx.succs,
        (0, 1, 2, 3),
        (fx.inner, fx.pre),
        registry,
        sources[0],
    )
    _record_affine_cache(
        fx.layers,
        fx.preds,
        fx.succs,
        registry,
        3,
        (main,),
        bias=fx.layers[3].params["c"].copy(),
    )
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_add_operand_cache_entry_missing",
    ):
        adapter._record_private_runtime_add_operands(
            fx.layers, fx.preds, fx.succs, registry, 5
        )


@pytest.mark.parametrize("mutation", ["bias", "terms"])
def test_add_hook_rejects_operand_cache_mutation_since_cache_recording(
    mutation,
):
    fx = _fixture()
    registry = _fresh_registry(fx)
    cache_entry = fx.affine_cache[3]
    if mutation == "bias":
        cache_entry.bias[0] += 0.125
    else:
        object.__setattr__(
            cache_entry, "terms", tuple(list(cache_entry.terms))
        )
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_affine_cache_entry_snapshot_mismatch",
    ):
        adapter._record_private_runtime_add_operands(
            fx.layers, fx.preds, fx.succs, registry, 5
        )


def test_outer_add_hook_requires_the_nested_add_snapshot_first():
    width = 2
    v0, v1, v2, v3, v4, v5 = (
        _vars(offset, width) for offset in (0, 10, 20, 30, 40, 50)
    )
    layers = [
        LayerView(0, "RELU", {}, [], v0),
        LayerView(1, "RELU", {}, [], v1),
        LayerView(2, "ADD", {"x_vars": v0, "y_vars": v1}, v0 + v1, v2),
        LayerView(3, "BIAS", {"c": np.zeros(width)}, v2, v3),
        LayerView(4, "RELU", {}, [], v4),
        LayerView(5, "ADD", {"x_vars": v3, "y_vars": v4}, v3 + v4, v5),
    ]
    preds = {0: [], 1: [], 2: [0, 1], 3: [2], 4: [], 5: [3, 4]}
    succs = _successors(preds)
    sources = {
        0: _source("nested-left", width, 91),
        1: _source("nested-right", width, 91),
        4: _source("outer-right", width, 91),
    }
    _, registry = _private_arena_and_registry(
        layers, preds, succs, sources
    )
    nested_left = _capture_operand(
        layers, preds, succs, (0,), (), registry, sources[0]
    )
    nested_right = _capture_operand(
        layers, preds, succs, (1,), (), registry, sources[1]
    )
    outer_left_a = _capture_operand(
        layers, preds, succs, (0, 2, 3), (), registry, sources[0]
    )
    outer_left_b = _capture_operand(
        layers, preds, succs, (1, 2, 3), (), registry, sources[1]
    )
    outer_right = _capture_operand(
        layers, preds, succs, (4,), (), registry, sources[4]
    )
    _record_affine_cache(
        layers, preds, succs, registry, 0, (nested_left,), frame_id=91
    )
    _record_affine_cache(
        layers, preds, succs, registry, 1, (nested_right,), frame_id=91
    )
    _record_affine_cache(
        layers,
        preds,
        succs,
        registry,
        3,
        (outer_left_a, outer_left_b),
        bias=layers[3].params["c"].copy(),
        frame_id=91,
    )
    _record_affine_cache(
        layers, preds, succs, registry, 4, (outer_right,), frame_id=91
    )
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_add_operand_snapshot_missing_or_duplicate",
    ):
        adapter._record_private_runtime_add_operands(
            layers, preds, succs, registry, 5
        )
    adapter._record_private_runtime_add_operands(
        layers, preds, succs, registry, 2
    )
    adapter._record_private_runtime_add_operands(
        layers, preds, succs, registry, 5
    )
    assert [
        snapshot.add_layer_id for snapshot in registry.add_operand_snapshots
    ] == [2, 5]


def test_conv_and_scale_consume_slots_while_bias_and_add_do_not():
    fx = _fixture()
    observations = [
        (event.kind, event.operator_position, event.operator_occurrence is not None)
        for event in fx.expr.terms[0].events
    ]
    assert observations == [
        ("CONV", 0, True),
        ("SCALE", 1, True),
        ("BIAS", 2, False),
        ("ADD", 2, False),
        ("SCALE", 2, True),
        ("BIAS", 3, False),
        ("CONV", 3, True),
        ("SCALE", 4, True),
    ]


def test_publicly_forged_event_has_no_private_authority():
    fx = _fixture()
    forged = replace(fx.expr.terms[0].events[0], _custody=object())
    term = replace(fx.expr.terms[0], events=(forged, *fx.expr.terms[0].events[1:]))
    fx.expr = replace(fx.expr, terms=(term, fx.expr.terms[1]))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_lineage_private_custody_missing"
    assert result.expression is fx.expr


def test_raw_source_mapping_is_not_a_runtime_registry_capability():
    fx = _fixture()
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_private_registry_missing",
    ):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5),
            (fx.inner, fx.pre),
            fx.source_by_layer,
            fx.expr.terms[0].source,
        )


def test_schema_valid_raw_sources_cannot_mint_a_registry_without_an_arena():
    fx = _fixture()
    raw_sources = {
        0: _source("raw-main", fx.expr.terms[0].source.n_out, 43),
        4: _source("raw-skip", fx.expr.terms[1].source.n_out, 43),
    }
    with pytest.raises(
        adapter.RuntimeLineageReject, match="runtime_private_arena_missing"
    ):
        adapter._begin_private_runtime_lineage_registry(
            fx.layers, fx.preds, fx.succs, raw_sources
        )


def test_same_frame_source_object_cannot_cross_request_arenas():
    fx = _fixture()
    other_cache = {0: fx.expr.terms[0].source}
    factor_allocator = _factor_allocator_for_sources(other_cache)
    other_arena = adapter._begin_private_runtime_lineage_arena(
        fx.layers,
        fx.preds,
        fx.succs,
        other_cache,
        {},
        factor_allocator,
    )
    with pytest.raises(
        adapter.RuntimeLineageReject, match="runtime_source_cross_arena_custody"
    ):
        adapter._record_private_runtime_source_boundary(
            fx.layers,
            fx.preds,
            fx.succs,
            other_arena,
            0,
            fx.expr.terms[0].source,
        )


def test_same_frame_fresh_sources_cannot_reuse_another_arena_lineage():
    fx = _fixture()
    other_sources = {
        0: _source("other-main", fx.expr.terms[0].source.n_out, 43),
        4: _source("other-skip", fx.expr.terms[1].source.n_out, 43),
    }
    _, other_registry = _private_arena_and_registry(
        fx.layers, fx.preds, fx.succs, other_sources
    )
    other_registry = adapter._seal_private_runtime_lineage_registry(
        other_registry
    )
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        fx.expr,
        np.ones(fx.expr.n_out, dtype=bool),
        registry=other_registry,
    )
    assert not result.accepted
    assert result.reason in {
        "runtime_lineage_arena_custody_mismatch",
        "runtime_source_boundary_identity_mismatch",
    }
    assert result.expression is fx.expr


def test_factor_allocator_accepts_legal_same_frame_growth_and_distinct_predicates():
    fx = _fixture()
    earlier = _source_with_factor_schema(
        fx.expr.terms[0].source.n_out,
        1,
        1,
        equality_value=0.0,
    )
    later = _source_with_factor_schema(
        fx.expr.terms[1].source.n_out,
        3,
        2,
        inequality_value=0.75,
    )
    cache = {0: earlier, 4: later}
    allocator = adapter._begin_private_runtime_factor_allocator(
        object(), cache
    )
    adapter._record_private_runtime_factor_frame_root(allocator, earlier)
    adapter._record_private_runtime_factor_relu_allocation(
        allocator, 43, 12, 7
    )
    adapter._record_private_runtime_factor_source_prefix(allocator, later)
    allocator = adapter._seal_private_runtime_factor_allocator(allocator)
    arena = adapter._begin_private_runtime_lineage_arena(
        fx.layers, fx.preds, fx.succs, cache, {}, allocator
    )
    for layer_id, source in cache.items():
        adapter._record_private_runtime_source_boundary(
            fx.layers, fx.preds, fx.succs, arena, layer_id, source
        )
    registry = adapter._begin_private_runtime_lineage_registry(
        fx.layers, fx.preds, fx.succs, arena
    )
    assert registry.arena.factor_allocator is allocator
    assert allocator.frame_widths[43] == (3, 2)
    assert earlier.Ac.shape[0] == 1
    assert later.Auc.shape[0] == 1


@pytest.mark.parametrize(("n_cont", "n_bin"), [(2, 1), (1, 2)])
def test_same_integer_frame_with_unregistered_nonprefix_width_is_rejected(
    n_cont, n_bin
):
    earlier = _source_with_factor_schema(4, 1, 1)
    forged = _source_with_factor_schema(4, n_cont, n_bin)
    cache = {0: earlier, 1: forged}
    allocator = adapter._begin_private_runtime_factor_allocator(
        object(), cache
    )
    adapter._record_private_runtime_factor_frame_root(allocator, earlier)
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_factor_source_not_current_prefix",
    ):
        adapter._record_private_runtime_factor_source_prefix(
            allocator, forged
        )


def test_schema_valid_same_frame_source_cannot_be_added_after_allocator_seal():
    fx = _fixture()
    cache = {
        0: _source("factor-root", fx.expr.terms[0].source.n_out),
        4: _source("factor-branch", fx.expr.terms[1].source.n_out),
    }
    allocator = _factor_allocator_for_sources(cache)
    cache[4] = _source("same-int-fake", fx.expr.terms[1].source.n_out)
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_factor_source_ledger_incomplete",
    ):
        adapter._begin_private_runtime_lineage_arena(
            fx.layers, fx.preds, fx.succs, cache, {}, allocator
        )


def test_same_frame_id_from_another_allocator_cannot_cross_request_owner():
    fx = _fixture()
    cache_a = {
        0: _source("allocator-a-main", fx.expr.terms[0].source.n_out),
        4: _source("allocator-a-skip", fx.expr.terms[1].source.n_out),
    }
    cache_b = {
        0: _source("allocator-b-main", fx.expr.terms[0].source.n_out),
        4: _source("allocator-b-skip", fx.expr.terms[1].source.n_out),
    }
    allocator_a = _factor_allocator_for_sources(cache_a)
    _factor_allocator_for_sources(cache_b)
    with pytest.raises(
        adapter.RuntimeLineageReject, match="runtime_owner_cache_missing"
    ):
        adapter._begin_private_runtime_lineage_arena(
            fx.layers, fx.preds, fx.succs, cache_b, {}, allocator_a
        )


def test_factor_slot_prefix_permutation_fails_allocator_cas():
    source = _source_with_factor_schema(4, 2, 1)
    cache = {0: source}
    allocator = _factor_allocator_for_sources(cache)
    prefix = allocator.source_prefixes[0]
    object.__setattr__(
        prefix,
        "continuous_slots",
        tuple(reversed(prefix.continuous_slots)),
    )
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_factor_source_prefix_mismatch",
    ):
        adapter._validate_factor_allocator_current(allocator)


def test_frontier_rebase_has_an_explicit_allocator_event_and_legal_prefix():
    before = _source_with_factor_schema(4, 1, 1, equality_value=0.0)
    after = _source_with_factor_schema(
        4, 3, 1, equality_value=0.0, inequality_value=0.5
    )
    cache = {0: before, 1: after}
    allocator = adapter._begin_private_runtime_factor_allocator(
        object(), cache
    )
    adapter._record_private_runtime_factor_frame_root(allocator, before)
    adapter._record_private_runtime_factor_rebase(
        allocator, 17, before, after
    )
    allocator = adapter._seal_private_runtime_factor_allocator(allocator)
    assert tuple(
        record.kind for record in allocator.allocation_history
    ) == ("ROOT", "REBASE")
    assert allocator.frame_widths[43] == (3, 1)
    assert allocator.relu_slots == {}
    assert allocator.aux_slots == {}
    assert allocator.source_prefixes[-1].source is after


def test_private_capture_from_a_caller_chosen_middle_cut_is_not_runtime_authority():
    fx = _fixture()
    main_source = fx.expr.terms[0].source
    fake_source_map = {1: main_source, 4: fx.expr.terms[1].source}
    hidden_ops = (fx.pre, fx.post, fx.outer, fx.output)
    del hidden_ops
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_source_boundary_kind_not_authorized",
    ):
        factor_allocator = _factor_allocator_for_sources(fake_source_map)
        arena = adapter._begin_private_runtime_lineage_arena(
            fx.layers,
            fx.preds,
            fx.succs,
            fake_source_map,
            {},
            factor_allocator,
        )
        adapter._record_private_runtime_source_boundary(
            fx.layers, fx.preds, fx.succs, arena, 1, main_source
        )


def test_term_source_replacement_cannot_reuse_old_lineage_custody():
    fx = _fixture()
    changed_source = _source(
        "forged-source", fx.expr.terms[0].source.n_out
    )
    changed = replace(fx.expr.terms[0], source=changed_source)
    fx.expr = replace(fx.expr, terms=(changed, fx.expr.terms[1]))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_lineage_source_object_custody_mismatch"


@pytest.mark.parametrize(
    ("field_name", "new_value", "reason"),
    [
        ("exact", False, "runtime_source_not_exact"),
        ("frame_id", 44, "runtime_factor_source_prefix_mismatch"),
    ],
)
def test_current_source_exact_frame_and_entry_width_are_mandatory(
    field_name, new_value, reason
):
    fx = _fixture()
    object.__setattr__(fx.expr.terms[0].source, field_name, new_value)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == reason
    assert result.expression is fx.expr


def test_current_source_n_out_and_factor_rows_must_match_entry_width():
    fx = _fixture()
    fx.expr.terms[0].source.c = fx.expr.terms[0].source.c[:-1]
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_source_value_factor_shape"
    assert result.expression is fx.expr


@pytest.mark.parametrize("mutation", ["same_shape_value", "same_value_object"])
def test_source_is_frozen_from_arena_recording_until_planning(mutation):
    fx = _fixture()
    source = fx.expr.terms[0].source
    if mutation == "same_shape_value":
        source.c[0] += 0.125
    else:
        source.c = source.c.copy()
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_source_arena_snapshot_mismatch"
    assert result.expression is fx.expr


def test_malformed_source_cross_factor_shape_fails_closed():
    fx = _fixture()
    source = fx.expr.terms[0].source
    source.Ac = sparse.csr_matrix((0, 2), dtype=np.float64)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_source_equality_factor_shape"


@pytest.mark.parametrize(
    ("malformation", "reason"),
    [
        ("dense_dtype", "runtime_source_dense_field_dtype"),
        ("sparse_dtype", "runtime_source_Gc_csr_data_dtype"),
        ("nonfinite", "runtime_source_Gc_csr_data_nonfinite"),
        ("explicit_zero", "runtime_source_Gc_csr_explicit_zero"),
        ("index_oob", "runtime_source_Gc_csr_index_out_of_bounds"),
        ("indptr_tail", "runtime_source_Gc_csr_indptr_not_monotone"),
        ("duplicate_index", "runtime_source_Gc_csr_not_canonical"),
    ],
)
def test_malformed_source_csr_structure_and_numeric_schema_fail_closed(
    malformation, reason
):
    fx = _fixture()
    source = fx.expr.terms[0].source
    if malformation == "dense_dtype":
        source.c = source.c.astype(np.float32)
    elif malformation == "sparse_dtype":
        source.Gc = source.Gc.astype(np.float32)
    elif malformation == "nonfinite":
        source.Gc.data[0] = np.nan
    elif malformation == "explicit_zero":
        source.Gc.data[0] = 0.0
    elif malformation == "index_oob":
        source.Gc.indices[0] = source.Gc.shape[1]
    elif malformation == "indptr_tail":
        source.Gc.indptr[-1] = 0
    else:
        source.Gc = sparse.csr_matrix(
            (
                np.asarray([0.125, 0.25]),
                np.asarray([0, 0]),
                np.asarray([0, 2, *([2] * (source.n_out - 1))]),
            ),
            shape=source.Gc.shape,
        )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == reason
    assert result.expression is fx.expr


def test_csr_cached_flags_are_exact_and_never_coerced_through_bool():
    fx = _fixture()
    callbacks = []

    class HostileFlag:
        def __bool__(self):
            callbacks.append("called")
            fx.layers[0].kind = "EVIL_CSR_FLAG_CALLBACK"
            return True

    matrix = fx.expr.terms[0].source.Gc
    matrix.__dict__["_has_sorted_indices"] = HostileFlag()
    matrix.__dict__["_has_canonical_format"] = HostileFlag()
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.endswith("_not_exact_bool")
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


def test_csr_shape_rejects_callback_capable_equal_integer():
    fx = _fixture()
    callbacks = []

    class HostileDimension(int):
        def __int__(self):
            callbacks.append("called")
            fx.layers[0].kind = "EVIL_CSR_SHAPE_CALLBACK"
            return int.__int__(self)

    matrix = fx.expr.terms[0].source.Gc
    rows, columns = matrix.__dict__["_shape"]
    matrix.__dict__["_shape"] = (rows, HostileDimension(columns))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.endswith("_not_exact_integer")
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


def test_non_sparse_hz_source_type_cannot_enter_private_registry():
    fx = _fixture()
    fake = SimpleNamespace(
        exact=True,
        frame_id=43,
        n_out=fx.expr.terms[0].source.n_out,
    )
    source_map = {0: fake, 4: fx.expr.terms[1].source}
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="runtime_source_type_not_sparse_hzono",
    ):
        factor_allocator = _factor_allocator_for_sources(source_map)
        arena = adapter._begin_private_runtime_lineage_arena(
            fx.layers,
            fx.preds,
            fx.succs,
            source_map,
            {},
            factor_allocator,
        )
        adapter._record_private_runtime_source_boundary(
            fx.layers, fx.preds, fx.succs, arena, 0, fake
        )


def test_unknown_mutable_object_in_source_semantic_field_fails_closed():
    fx = _fixture()

    class Mutable:
        def __init__(self):
            self.value = 1

    fx.expr.terms[0].source.Ac = Mutable()
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_source_sparse_field_not_csr"


def test_expression_bias_must_be_finite_one_dimensional_and_exact_width():
    fx = _fixture()
    for bad_bias in (
        np.zeros((2, 4)),
        np.zeros(fx.expr.n_out - 1),
        np.array([np.nan, *np.zeros(fx.expr.n_out - 1)]),
    ):
        bad_expr = replace(fx.expr, bias=bad_bias)
        result = adapter.plan_s0_c3_from_runtime_lineage(
            fx.layers,
            fx.preds,
            fx.succs,
            bad_expr,
            np.ones(fx.expr.n_out, dtype=bool),
            registry=fx.registry,
        )
        assert not result.accepted
        assert result.expression is bad_expr


def test_every_term_operator_chain_must_end_at_expression_width():
    fx = _fixture()
    bad_expr = replace(
        fx.expr,
        n_out=fx.expr.n_out - 1,
        bias=np.zeros(fx.expr.n_out - 1),
    )
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        bad_expr,
        np.ones(bad_expr.n_out, dtype=bool),
        registry=fx.registry,
    )
    assert not result.accepted
    assert result.reason == "runtime_term_output_width_mismatch"
    assert result.expression is bad_expr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda events: events[:-1],
        lambda events: (events[1], events[0], *events[2:]),
        lambda events: (events[0], events[0], *events[1:]),
        lambda events: (replace(events[0], selected_input=1), *events[1:]),
        lambda events: (replace(events[0], operator_position=7), *events[1:]),
        lambda events: (replace(events[0], layer_payload_sha256="0" * 64), *events[1:]),
    ],
)
def test_missing_reordered_duplicate_or_mutated_public_records_fail_custody(mutation):
    fx = _fixture()
    term = replace(fx.expr.terms[0], events=tuple(mutation(fx.expr.terms[0].events)))
    fx.expr = replace(fx.expr, terms=(term, fx.expr.terms[1]))
    result = _plan(fx)
    assert not result.accepted
    assert "custody" in result.reason
    assert result.expression is fx.expr


@pytest.mark.parametrize("kind", ["RELU", "RESHAPE", "DENSE", "MATERIALIZE", "MYSTERY"])
def test_unknown_and_registered_barriers_cannot_be_recorded(kind):
    fx = _fixture()
    fx.layers[2].kind = kind
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(adapter.RuntimeLineageReject, match="unsupported_runtime_event_kind"):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            fx.expr.terms[0].operators,
            registry,
            sources[0],
        )


def test_csr_diagonal_cannot_impersonate_a_scale_operator():
    fx = _fixture()
    csr = sparse.diags(fx.pre._diagonal, format="csr")
    operators = (fx.inner, csr, fx.post, fx.outer, fx.output)
    with pytest.raises(adapter.RuntimeLineageReject, match="scale_operator_not_diagonal_linear_op"):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            operators,
            _fresh_registry(fx),
            fx.expr.terms[0].source,
        )


@pytest.mark.parametrize("drop", [True, False])
def test_missing_or_extra_operator_fails_closed(drop):
    fx = _fixture()
    operators = fx.expr.terms[0].operators[:-1] if drop else (*fx.expr.terms[0].operators, fx.output)
    with pytest.raises(adapter.RuntimeLineageReject, match="operator"):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            tuple(operators),
            _fresh_registry(fx),
            fx.expr.terms[0].source,
        )


def test_conv_numeric_payload_mismatch_rejects_capture():
    fx = _fixture()
    fx.layers[1].params["weight"][0, 0, 0, 0] += np.float32(0.5)
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(adapter.RuntimeLineageReject, match="conv_operator_payload_mismatch"):
        _capture(
            fx.layers, fx.preds, fx.succs, (0, 1, 2, 3, 5),
            (fx.inner, fx.pre), registry, sources[0]
        )


def test_conv_geometry_mismatch_rejects_capture():
    fx = _fixture()
    fx.layers[1].params["padding"] = (1, 1)
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(adapter.RuntimeLineageReject, match="conv_operator_geometry_mismatch"):
        _capture(
            fx.layers, fx.preds, fx.succs, (0, 1, 2, 3, 5),
            (fx.inner, fx.pre), registry, sources[0]
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("kernel_size", (2, 1), "conv_operator_kernel_size_mismatch"),
        ("in_channels", 7, "conv_operator_in_channels_mismatch"),
        ("out_channels", 7, "conv_operator_out_channels_mismatch"),
    ],
)
def test_declared_conv_metadata_must_match_the_live_operator(
    field, value, reason
):
    fx = _fixture()
    fx.layers[1].params[field] = value
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(adapter.RuntimeLineageReject, match=reason):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5),
            (fx.inner, fx.pre),
            registry,
            sources[0],
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"transposed": True},
        {"output_padding": (1, 0)},
        {"padding_mode": "reflect"},
        {"data_format": "NHWC"},
        {"activation": "relu"},
    ],
)
def test_unrepresented_conv_semantics_cannot_enter_narrow_runtime_grammar(
    extra,
):
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="conv2d_semantic_parameter_unsupported",
    ):
        _fixture(conv_extra_params=extra)


@pytest.mark.parametrize(
    "extra",
    [
        {"transposed": True},
        {"output_padding": (1, 0)},
        {"activation": "relu"},
        {"semantic_payload": {"mode": "foreign"}},
    ],
)
def test_unrepresented_layer_raw_fields_fail_before_registry_seal(extra):
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="layer_raw_state_schema_mismatch",
    ):
        _fixture(conv_layer_extra_attrs=extra)


def test_unrepresented_layer_raw_field_added_after_seal_fails_closed():
    fx = _fixture()
    fx.layers[1].semantic_payload = {"mode": "foreign"}
    fx.layers[1].semantic_payload["mode"] = "mutated-after-seal"
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "layer_raw_state_schema_mismatch"
    assert result.expression is fx.expr


def test_nonzero_inline_conv_bias_cannot_hide_an_unrecorded_transition():
    fx = _fixture()
    fx.layers[1].params["bias"] = np.full(
        fx.inner._output_shape[1], 0.125
    )
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="conv_inline_bias_transition_not_explicit",
    ):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 3, 5),
            (fx.inner, fx.pre),
            registry,
            sources[0],
        )


def test_scale_numeric_payload_mismatch_rejects_capture():
    fx = _fixture()
    fx.layers[2].params["a"][0] += 0.25
    _, registry, sources = _fresh_current_graph_registry(fx)
    with pytest.raises(adapter.RuntimeLineageReject, match="scale_operator_payload_mismatch"):
        _capture(
            fx.layers, fx.preds, fx.succs, (0, 1, 2, 3, 5),
            (fx.inner, fx.pre), registry, sources[0]
        )


def test_bias_payload_is_current_complete_and_finite():
    fx = _fixture()
    fx.layers[3].params["c"][0] = np.nan
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert "nonfinite" in result.reason


def test_add_arity_and_complete_operand_values_are_checked():
    fx = _fixture()
    fx.layers[5].params["y_vars"] = fx.layers[5].params["y_vars"][:-1]
    result = _plan(fx)
    assert not result.accepted
    assert "add_operands_do_not_equal_in_vars" in result.reason


@pytest.mark.parametrize(
    "bias",
    [
        np.full(12, 0.25, dtype=np.float64),
        np.asarray([np.nan]),
        "not-a-numeric-bias",
    ],
)
def test_add_optional_bias_must_not_hide_a_value_transition(bias):
    with pytest.raises(adapter.RuntimeLineageReject) as captured:
        _fixture(add_extra_params={"bias": bias})
    assert (
        "add_inline_bias_transition_not_explicit" in str(captured.value)
        or "bias" in str(captured.value)
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"bias": None},
        {"bias": 0.0},
        {"bias": np.zeros(12, dtype=np.float64)},
    ],
)
def test_absent_none_or_strictly_zero_add_bias_is_a_pure_sum(extra):
    fx = _fixture(add_extra_params=extra)
    result = _plan(fx)
    assert result.accepted, result.reason


@pytest.mark.parametrize(
    "bias",
    [
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(11, dtype=np.float64),
        np.zeros((1, 12), dtype=np.float64),
    ],
)
def test_zero_add_bias_wrong_rank_or_width_is_not_implicit_broadcast(bias):
    with pytest.raises(adapter.RuntimeLineageReject) as captured:
        _fixture(add_extra_params={"bias": bias})
    assert (
        "add_bias_rank" in str(captured.value)
        or "add_inline_bias_shape_unsupported" in str(captured.value)
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"axis": 1},
        {"broadcast": True},
        {"broadcast_shape": (1, 3, 2, 2)},
    ],
)
def test_unproved_add_broadcast_metadata_fails_closed(metadata):
    with pytest.raises(
        adapter.RuntimeLineageReject,
        match="add_semantic_parameter_unsupported",
    ):
        _fixture(add_extra_params=metadata)


def test_unary_extra_predecessor_is_not_silently_selected():
    fx = _fixture()
    fx.preds[6] = [5, 4]
    fx.succs = _successors(fx.preds)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason in {
        "graph_predecessor_producer_mismatch",
        "unary_event_arity_or_predecessor_mismatch",
    }


def test_path_input_output_alias_fails_closed():
    fx = _fixture()
    fx.layers[2].out_vars = fx.layers[2].in_vars.copy()
    fx.layers[3].in_vars = fx.layers[2].out_vars.copy()
    result = _plan(fx)
    assert not result.accepted
    assert result.reason in {
        "graph_duplicate_output_variable_producer",
        "runtime_event_input_output_alias",
    }


def test_duplicate_path_occurrence_fails_before_certificate_creation():
    fx = _fixture()
    with pytest.raises(adapter.RuntimeLineageReject, match="duplicate"):
        _capture(
            fx.layers,
            fx.preds,
            fx.succs,
            (0, 1, 2, 1, 2, 3, 5),
            (fx.inner, fx.pre, fx.inner, fx.pre),
            _fresh_registry(fx),
            fx.expr.terms[0].source,
        )


def test_shared_suffix_must_use_the_identical_operator_object():
    fx = _fixture()
    registry = _fresh_registry(fx)
    post_clone = DiagonalLinearOp(fx.post._diagonal.copy())
    skip_ops = (post_clone, fx.outer, fx.output)
    main = fx.expr.terms[0]
    main_events = _capture(
        fx.layers,
        fx.preds,
        fx.succs,
        (0, 1, 2, 3, 5, 6, 7, 8, 9),
        main.operators,
        registry,
        main.source,
    )
    skip_events = _capture(
        fx.layers, fx.preds, fx.succs, (4, 5, 6, 7, 8, 9), skip_ops,
        registry, fx.expr.terms[1].source
    )
    main = replace(main, events=main_events)
    skip = replace(fx.expr.terms[1], operators=skip_ops, events=skip_events)
    fx.expr = replace(fx.expr, terms=(main, skip))
    fx.registry = adapter._seal_private_runtime_lineage_registry(registry)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "shared_suffix_operator_object_mismatch"


def test_one_operator_object_cannot_alias_two_graph_occurrences():
    fx = _fixture()
    fx.layers[6].params["a"] = fx.pre._diagonal.copy()
    aliased_ops = (fx.inner, fx.pre, fx.pre, fx.outer, fx.output)
    skip_ops = (fx.pre, fx.outer, fx.output)
    _, registry, sources = _fresh_current_graph_registry(fx)
    main_operand = _capture_operand(
        fx.layers,
        fx.preds,
        fx.succs,
        (0, 1, 2, 3),
        (fx.inner, fx.pre),
        registry,
        sources[0],
    )
    skip_operand = _capture_operand(
        fx.layers, fx.preds, fx.succs, (4,), (), registry, sources[4]
    )
    _record_affine_cache(
        fx.layers,
        fx.preds,
        fx.succs,
        registry,
        3,
        (main_operand,),
        bias=fx.layers[3].params["c"].copy(),
    )
    _record_affine_cache(
        fx.layers, fx.preds, fx.succs, registry, 4, (skip_operand,)
    )
    adapter._record_private_runtime_add_operands(
        fx.layers, fx.preds, fx.succs, registry, 5
    )
    main_events = _capture(
        fx.layers, fx.preds, fx.succs, (0, 1, 2, 3, 5, 6, 7, 8, 9), aliased_ops,
        registry, sources[0]
    )
    skip_events = _capture(
        fx.layers,
        fx.preds,
        fx.succs,
        (4, 5, 6, 7, 8, 9),
        skip_ops,
        registry,
        sources[4],
    )
    main = adapter.RuntimeLineageTermView(sources[0], aliased_ops, main_events)
    skip = adapter.RuntimeLineageTermView(sources[4], skip_ops, skip_events)
    fx.expr = replace(fx.expr, terms=(main, skip))
    fx.registry = adapter._seal_private_runtime_lineage_registry(registry)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "operator_object_aliases_graph_occurrences"


def test_graph_edge_asymmetry_fails_closed():
    fx = _fixture()
    fx.succs[5] = []
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "predecessor_successor_asymmetry"


def test_duplicate_ssa_producer_outside_all_selected_paths_fails_closed():
    fx = _fixture()
    duplicate = fx.layers[1].out_vars.copy()
    fx.layers.append(
        LayerView(10, "RELU", {}, fx.layers[9].out_vars.copy(), duplicate)
    )
    fx.preds[10] = [9]
    fx.succs = _successors(fx.preds)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "graph_duplicate_output_variable_producer"


def test_only_exact_inputspec_or_assert_wrapper_alias_is_permitted():
    fx = _fixture()
    wrapper_vars = fx.layers[9].out_vars.copy()
    fx.layers.append(
        LayerView(10, "ASSERT", {}, wrapper_vars.copy(), wrapper_vars.copy())
    )
    fx.preds[10] = [9]
    fx.succs = _successors(fx.preds)
    _, registry, sources = _fresh_current_graph_registry(fx)
    source = sources[0]
    # A fresh capture proves the global SSA audit accepted this exact alias;
    # the path itself still stops at layer 5 and remains non-executable.
    events = _capture(
        fx.layers,
        fx.preds,
        fx.succs,
        (0, 1, 2, 3, 5),
        (fx.inner, fx.pre),
        registry,
        source,
    )
    assert events[-1].kind == "ADD"


def test_partial_wrapper_alias_is_rejected_by_global_ssa():
    fx = _fixture()
    wrapper_in = fx.layers[9].out_vars.copy()
    wrapper_out = wrapper_in[:-1]
    fx.layers.append(
        LayerView(10, "ASSERT", {}, wrapper_in, wrapper_out)
    )
    fx.preds[10] = [9]
    fx.succs = _successors(fx.preds)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "graph_duplicate_output_variable_producer"


def test_equal_layer_replacement_is_detected_by_occurrence_identity():
    fx = _fixture()
    old = fx.layers[6]
    fx.layers[6] = LayerView(old.id, old.kind, dict(old.params), list(old.in_vars), list(old.out_vars))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason in {
        "runtime_registry_graph_cas_mismatch",
        "runtime_lineage_occurrence_object_mismatch",
    }


def test_equal_operator_replacement_is_detected_by_occurrence_identity():
    fx = _fixture()
    post_clone = DiagonalLinearOp(fx.post._diagonal.copy())
    main = fx.expr.terms[0]
    main_ops = (fx.inner, fx.pre, post_clone, fx.outer, fx.output)
    fx.expr = replace(fx.expr, terms=(replace(main, operators=main_ops), fx.expr.terms[1]))
    result = _plan(fx)
    assert not result.accepted
    assert "operator_occurrence_mismatch" in result.reason


def test_graph_payload_mutation_after_capture_is_detected():
    fx = _fixture()
    fx.layers[6].params["a"][0] += 0.125
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason in {
        "runtime_registry_graph_cas_mismatch",
        "runtime_lineage_current_snapshot_mismatch",
    }


def test_planner_time_graph_mutation_fails_the_second_current_state_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        fx.layers[7].params["c"][0] += 0.5
        return result

    monkeypatch.setattr(adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return)
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason.startswith("current_state_")


@pytest.mark.parametrize(
    "target",
    [
        "live_kernel",
        "live_diagonal",
        "graph_bias",
        "terminal_bias",
        "terminal_terms",
    ],
)
def test_live_aba_cannot_change_detached_plan_semantics(
    monkeypatch, target
):
    baseline = _plan(_fixture())
    assert baseline.accepted, baseline.reason
    baseline_fingerprint = adapter._planner_decision_semantic_fingerprint(
        baseline.planner_decision
    )
    baseline_descriptor = (
        baseline.planner_decision.plan.requests[0].content_key
    )
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_restore(*args, **kwargs):
        if target == "live_kernel":
            array = fx.inner._kernel
            old = float(array[0, 0, 0, 0])
            array[0, 0, 0, 0] = old + 0.5
            restore = lambda: array.__setitem__((0, 0, 0, 0), old)
        elif target == "live_diagonal":
            array = fx.post._diagonal
            old = float(array[0])
            array[0] = old + 0.5
            restore = lambda: array.__setitem__(0, old)
        elif target == "graph_bias":
            array = fx.layers[7].params["c"]
            old = float(array[0])
            array[0] = old + 0.5
            restore = lambda: array.__setitem__(0, old)
        elif target == "terminal_bias":
            array = fx.affine_cache[9].bias
            old = float(array[0])
            array[0] = old + 0.5
            restore = lambda: array.__setitem__(0, old)
        else:
            entry = fx.affine_cache[9]
            old = entry.terms
            object.__setattr__(entry, "terms", tuple(list(old)))
            restore = lambda: object.__setattr__(entry, "terms", old)
        try:
            return original(*args, **kwargs)
        finally:
            restore()

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        mutate_then_restore,
    )
    result = _plan(fx)
    assert result.accepted, result.reason
    assert (
        result.planner_decision.plan.requests[0].content_key
        == baseline_descriptor
    )
    assert (
        adapter._planner_decision_semantic_fingerprint(
            result.planner_decision
        )
        == baseline_fingerprint
    )


@pytest.mark.parametrize("operator_kind", ["CONV", "SCALE"])
def test_first_round_detached_input_aba_is_rejected_by_semantic_replan(
    monkeypatch, operator_kind
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def mutate_first_snapshot_then_restore(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            return original(expression, *args, **kwargs)
        operator_value = next(
            value
            for value in expression.terms[0].operators
            if (
                operator_kind == "CONV"
                and type(value) is ImplicitConv2DOp
            )
            or (
                operator_kind == "SCALE"
                and type(value) is DiagonalLinearOp
            )
        )
        array = (
            operator_value._kernel
            if operator_kind == "CONV"
            else operator_value._diagonal
        )
        index = (0, 0, 0, 0) if operator_kind == "CONV" else 0
        array.setflags(write=True)
        old = float(array[index])
        array[index] = old + 0.5
        try:
            return original(expression, *args, **kwargs)
        finally:
            array[index] = old
            array.setflags(write=False)

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        mutate_first_snapshot_then_restore,
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason in {
        "runtime_planner_semantic_snapshot_mismatch",
        "runtime_planner_failed:RuntimeLineageReject",
    }


@pytest.mark.parametrize(
    "target",
    [
        "request_inner_wrong_type",
        "request_inner_equal_clone",
        "use_source",
        "use_certificate_equal_clone",
        "use_operator_equal_clone",
        "plan_frame",
        "plan_original_terms_equal_tuple",
        "plan_original_term_equal_clone",
    ],
)
def test_first_planner_return_cannot_forge_nested_plan_binding(
    monkeypatch, target
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def forge_only_first_return(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(expression, *args, **kwargs)
        if calls != 1:
            return result
        plan = result.plan
        assert plan is not None
        request = plan.requests[0]
        use = plan.uses[0]
        if target == "request_inner_wrong_type":
            object.__setattr__(request, "inner", expression.terms[0].operators[1])
        elif target == "request_inner_equal_clone":
            object.__setattr__(
                request,
                "inner",
                adapter._detach_runtime_operator(request.inner),
            )
        elif target == "use_source":
            object.__setattr__(use, "source", object())
        elif target == "use_certificate_equal_clone":
            object.__setattr__(
                use, "certificate", replace(use.certificate)
            )
        elif target == "use_operator_equal_clone":
            object.__setattr__(
                use,
                "outer",
                adapter._detach_runtime_operator(use.outer),
            )
        elif target == "plan_frame":
            object.__setattr__(plan, "frame_id", 999)
        elif target == "plan_original_terms_equal_tuple":
            object.__setattr__(
                plan,
                "original_terms",
                tuple(list(plan.original_terms)),
            )
        else:
            object.__setattr__(
                plan,
                "original_terms",
                (
                    replace(plan.original_terms[0]),
                    *plan.original_terms[1:],
                ),
            )
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        forge_only_first_return,
    )
    result = _plan(fx)
    assert calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert "planner" in result.reason


class _EqualAnything:
    def __init__(self, value=0):
        self.value = value

    def __eq__(self, other):
        del other
        return True

    def __ne__(self, other):
        del other
        return False

    def __hash__(self):
        return 1


@pytest.mark.parametrize(
    "target",
    [
        "request_selected_rows_forged_equality",
        "use_selected_rows_forged_equality",
        "request_term_indices_forged_equality",
        "identity_indices_forged_equality",
        "estimate_equal_float",
        "estimate_unknown_object",
        "reservation_equal_float",
        "reservation_unknown_object",
        "certificate_digest_unknown_object",
        "content_key_unknown_object",
    ],
)
def test_first_planner_return_rejects_hostile_equality_and_nonexact_scalars(
    monkeypatch, target
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def forge_only_first_return(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(expression, *args, **kwargs)
        if calls != 1:
            return result
        plan = result.plan
        assert plan is not None
        request = plan.requests[0]
        use = plan.uses[0]
        if target == "request_selected_rows_forged_equality":
            object.__setattr__(
                request,
                "selected_rows",
                tuple(
                    _EqualAnything(1000 + index)
                    for index, _ in enumerate(request.selected_rows)
                ),
            )
        elif target == "use_selected_rows_forged_equality":
            object.__setattr__(
                use,
                "selected_rows",
                tuple(
                    _EqualAnything(1000 + index)
                    for index, _ in enumerate(use.selected_rows)
                ),
            )
        elif target == "request_term_indices_forged_equality":
            object.__setattr__(
                request,
                "term_indices",
                tuple(_EqualAnything(1000) for _ in request.term_indices),
            )
        elif target == "identity_indices_forged_equality":
            object.__setattr__(
                plan,
                "identity_term_indices",
                tuple(
                    _EqualAnything(1000)
                    for _ in plan.identity_term_indices
                ),
            )
        elif target == "estimate_equal_float":
            object.__setattr__(
                request.estimate,
                "resident_bytes",
                float(request.estimate.resident_bytes),
            )
        elif target == "estimate_unknown_object":
            object.__setattr__(
                request.estimate,
                "resident_bytes",
                _EqualAnything(request.estimate.resident_bytes),
            )
        elif target == "reservation_equal_float":
            object.__setattr__(
                plan.reservation,
                "resident_bytes",
                float(plan.reservation.resident_bytes),
            )
        elif target == "reservation_unknown_object":
            object.__setattr__(
                plan.reservation,
                "resident_bytes",
                _EqualAnything(plan.reservation.resident_bytes),
            )
        elif target == "certificate_digest_unknown_object":
            object.__setattr__(
                plan,
                "certificate_digests",
                (
                    _EqualAnything(plan.certificate_digests[0]),
                    *plan.certificate_digests[1:],
                ),
            )
        else:
            object.__setattr__(
                request,
                "content_key",
                (request.content_key[0], _EqualAnything(request.content_key[1])),
            )
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        forge_only_first_return,
    )
    result = _plan(fx)
    assert calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason == "runtime_planner_failed:RuntimeLineageReject"


@pytest.mark.parametrize(
    "source_field", ("c", "Gc", "Gb", "Ac", "Ab", "b", "Auc", "Aub", "ub")
)
@pytest.mark.parametrize("payload_kind", ("hostile_empty", "nonempty_tuple"))
def test_first_planner_source_proxy_fields_require_canonical_empty_tuples(
    monkeypatch, source_field, payload_kind
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def forge_first_source(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            object.__setattr__(
                expression.terms[0].source,
                source_field,
                _EqualAnything(())
                if payload_kind == "hostile_empty"
                else (0,),
            )
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        forge_first_source,
    )
    result = _plan(fx)
    assert calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason == "runtime_planner_failed:RuntimeLineageReject"


class _WeirdGroup:
    def __index__(self):
        return 1

    def __int__(self):
        return 1

    def __rfloordiv__(self, other):
        del other
        return 0


@pytest.mark.parametrize(
    "forged_group", (np.int64(1), _WeirdGroup())
)
def test_first_planner_conv_group_requires_exact_safe_integer(
    monkeypatch, forged_group
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def forge_first_group(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            object.__setattr__(
                expression.terms[0].operators[0],
                "_groups",
                forged_group,
            )
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        forge_first_group,
    )
    result = _plan(fx)
    assert calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason == "runtime_planner_failed:RuntimeLineageReject"


def test_first_planner_cannot_make_detached_bias_writeable(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def flip_first_bias_writeable(expression, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            expression.bias.setflags(write=True)
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        flip_first_bias_writeable,
    )
    result = _plan(fx)
    assert calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason == "runtime_planner_failed:RuntimeLineageReject"


def test_post_final_guard_catches_third_derive_mutating_retained_plan(
    monkeypatch,
):
    fx = _fixture()
    original_planner = adapter.c3.plan_s0_c3_identity_middle_lineage
    original_derive = adapter._derive_current
    retained_expression = None
    planner_calls = 0
    derive_calls = 0

    def retain_expression(expression, *args, **kwargs):
        nonlocal retained_expression, planner_calls
        planner_calls += 1
        if retained_expression is None:
            retained_expression = expression
        return original_planner(expression, *args, **kwargs)

    def mutate_after_third_derive(*args, **kwargs):
        nonlocal derive_calls
        derive_calls += 1
        result = original_derive(*args, **kwargs)
        if derive_calls == 3:
            assert retained_expression is not None
            retained_expression.bias.setflags(write=True)
            retained_expression.bias[0] += 777.0
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        retain_expression,
    )
    monkeypatch.setattr(adapter, "_derive_current", mutate_after_third_derive)
    result = _plan(fx)
    assert planner_calls == 2
    assert derive_calls == 3
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason in {
        "post_final_request_input_changed_during_planning",
        "runtime_detached_expression_bias_storage_schema",
    }


def test_layer_subclass_getter_is_never_runtime_graph_authority():
    fx = _fixture()
    getter_calls = 0

    class ReentrantLayer(LayerView):
        @property
        def id(self):
            nonlocal getter_calls
            getter_calls += 1
            fx.expr.bias[0] += 888.0
            return object.__getattribute__(self, "__dict__")["id"]

    original_bias = fx.expr.bias.copy()
    fx.layers[-1].__class__ = ReentrantLayer
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert getter_calls == 0
    np.testing.assert_array_equal(fx.expr.bias, original_bias)


def test_frame_tokens_reject_int_subclasses_without_calling_conversion():
    fx = _fixture()
    conversions = 0

    class ReentrantFrame(int):
        def __int__(self):
            nonlocal conversions
            conversions += 1
            fx.expr.bias[0] += 999.0
            return 43

    original_bias = fx.expr.bias.copy()
    bad_expr = replace(fx.expr, frame_id=ReentrantFrame(43))
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        bad_expr,
        np.ones(bad_expr.n_out, dtype=bool),
        registry=fx.registry,
    )
    assert not result.accepted
    assert result.expression is bad_expr
    assert conversions == 0
    np.testing.assert_array_equal(fx.expr.bias, original_bias)


def test_final_live_tensor_instance_shadow_is_rejected_without_callback(
    monkeypatch,
):
    if adapter.torch is None:
        pytest.skip("torch unavailable")
    fx = _fixture(torch_graph_weights=True)
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    planner_calls = 0
    shadow_calls = 0
    weight = fx.layers[1].params["weight"]

    def shadow_detach():
        nonlocal shadow_calls
        shadow_calls += 1
        fx.layers[0].kind = "EVIL"
        return adapter.torch.Tensor.detach(weight)

    def attach_after_second_plan(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        result = original(*args, **kwargs)
        if planner_calls == 2:
            weight.detach = shadow_detach
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        attach_after_second_plan,
    )
    result = _plan(fx)
    assert planner_calls == 2
    assert not result.accepted
    assert result.expression is fx.expr
    assert shadow_calls == 0
    assert fx.layers[0].kind == "RELU"


def test_detached_operator_instance_method_shadow_is_never_invoked(
    monkeypatch,
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    planner_calls = 0
    shadow_calls = 0

    def attach_shadow_then_plan(expression, *args, **kwargs):
        nonlocal planner_calls, shadow_calls
        planner_calls += 1
        if planner_calls == 1:
            operator_value = expression.terms[0].operators[0]

            def shadow_count():
                nonlocal shadow_calls
                shadow_calls += 1
                fx.layers[0].kind = "EVIL"
                return ImplicitConv2DOp._count_expanded_entries(
                    operator_value
                )

            operator_value._count_expanded_entries = shadow_count
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        attach_shadow_then_plan,
    )
    result = _plan(fx)
    assert planner_calls == 1
    assert not result.accepted
    assert result.expression is fx.expr
    assert shadow_calls == 0
    assert fx.layers[0].kind == "RELU"


def test_final_live_operator_group_subclass_rejected_without_conversion(
    monkeypatch,
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    planner_calls = 0
    conversions = 0

    class EvilGroup(int):
        def __int__(self):
            nonlocal conversions
            conversions += 1
            fx.layers[0].kind = "EVIL"
            return 1

    def replace_group_after_second_plan(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        result = original(*args, **kwargs)
        if planner_calls == 2:
            fx.inner._groups = EvilGroup(1)
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        replace_group_after_second_plan,
    )
    result = _plan(fx)
    assert planner_calls == 2
    assert not result.accepted
    assert result.expression is fx.expr
    assert conversions == 0
    assert fx.layers[0].kind == "RELU"


@pytest.mark.parametrize(
    "target",
    (
        "kernel",
        "input_shape",
        "content_key",
        "row_mask",
        "diagonal",
    ),
)
def test_sealed_affine_ledger_rejects_equal_live_operator_alias_replacement(
    target,
):
    fx = _fixture(outer_row_mask_index=0 if target == "row_mask" else None)
    if target == "kernel":
        prior = fx.inner._kernel
        fx.inner._kernel = np.array(prior, copy=True)
        current = fx.inner._kernel
    elif target == "input_shape":
        prior = fx.inner._input_shape
        fx.inner._input_shape = tuple(list(prior))
        current = fx.inner._input_shape
    elif target == "content_key":
        prior = fx.inner._content_key
        fx.inner._content_key = tuple(list(prior))
        current = fx.inner._content_key
    elif target == "row_mask":
        prior = fx.outer._row_mask
        assert prior is not None
        fx.outer._row_mask = np.array(prior, dtype=bool, copy=True)
        current = fx.outer._row_mask
    else:
        prior = fx.pre._diagonal
        fx.pre._diagonal = np.array(prior, copy=True)
        current = fx.pre._diagonal
    assert current is not prior
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert "affine_cache_entry_snapshot_mismatch" in result.reason


def test_terminal_expression_rejects_equal_foreign_bias_object():
    fx = _fixture()
    foreign_bias = np.array(fx.expr.bias, dtype=np.float64, copy=True)
    assert foreign_bias is not fx.expr.bias
    fx.expr = replace(fx.expr, bias=foreign_bias)
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason == (
        "runtime_terminal_expression_bias_identity_mismatch"
    )


@pytest.mark.parametrize(
    "target",
    [
        "graph_param",
        "live_kernel",
        "live_diagonal",
        "source_buffer",
        "source_frame",
        "source_exact",
        "expr_bias",
        "expr_terms",
        "terminal_bias",
        "terminal_terms",
        "operand_cache_bias",
        "operand_cache_terms",
        "allocator_container",
        "registry_container",
        "output_support",
        "budget",
    ],
)
def test_second_planner_persistent_mutation_is_closed_by_final_cas(
    monkeypatch, target
):
    fx = _fixture()
    support = np.ones(fx.expr.n_out, dtype=bool)
    budget = adapter.PlannerBudget()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def mutate_after_second_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls != 2:
            return result
        if target == "graph_param":
            fx.layers[7].params["c"][0] += 0.5
        elif target == "live_kernel":
            fx.inner._kernel[0, 0, 0, 0] += 0.5
        elif target == "live_diagonal":
            fx.post._diagonal[0] += 0.5
        elif target == "source_buffer":
            fx.expr.terms[0].source.Gc.data[0] += 0.5
        elif target == "source_frame":
            object.__setattr__(fx.expr.terms[0].source, "frame_id", 99)
        elif target == "source_exact":
            object.__setattr__(fx.expr.terms[0].source, "exact", False)
        elif target == "expr_bias":
            fx.expr.bias[0] += 0.5
        elif target == "expr_terms":
            object.__setattr__(
                fx.expr, "terms", tuple(list(fx.expr.terms))
            )
        elif target == "terminal_bias":
            fx.affine_cache[9].bias[0] += 0.5
        elif target == "terminal_terms":
            entry = fx.affine_cache[9]
            object.__setattr__(entry, "terms", tuple(list(entry.terms)))
        elif target == "operand_cache_bias":
            fx.affine_cache[3].bias[0] += 0.5
        elif target == "operand_cache_terms":
            entry = fx.affine_cache[3]
            object.__setattr__(entry, "terms", tuple(list(entry.terms)))
        elif target == "allocator_container":
            fx.arena.factor_allocator.frame_widths = dict(
                fx.arena.factor_allocator.frame_widths
            )
        elif target == "registry_container":
            fx.registry.add_operand_snapshots = tuple(
                list(fx.registry.add_operand_snapshots)
            )
        elif target == "output_support":
            support[0] = False
        else:
            object.__setattr__(budget, "max_unique_descriptors", 65)
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        mutate_after_second_plan,
    )
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        fx.expr,
        support,
        registry=fx.registry,
        budget=budget,
    )
    assert calls == 2
    assert not result.accepted
    if target in {"output_support", "budget"}:
        assert result.reason == "final_request_input_changed_during_planning"
    else:
        assert result.reason.startswith("final_state_")


@pytest.mark.parametrize("target", ["graph_param", "output_support", "budget"])
def test_second_planner_mutate_restore_is_semantically_harmless(
    monkeypatch, target
):
    fx = _fixture()
    support = np.ones(fx.expr.n_out, dtype=bool)
    budget = adapter.PlannerBudget()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0

    def mutate_only_during_second_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 2:
            return original(*args, **kwargs)
        if target == "graph_param":
            array = fx.layers[7].params["c"]
            old = float(array[0])
            array[0] = old + 0.5
            restore = lambda: array.__setitem__(0, old)
        elif target == "output_support":
            old = bool(support[0])
            support[0] = not old
            restore = lambda: support.__setitem__(0, old)
        else:
            old = budget.max_unique_descriptors
            object.__setattr__(budget, "max_unique_descriptors", old + 1)
            restore = lambda: object.__setattr__(
                budget, "max_unique_descriptors", old
            )
        try:
            return original(*args, **kwargs)
        finally:
            restore()

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        mutate_only_during_second_plan,
    )
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers,
        fx.preds,
        fx.succs,
        fx.expr,
        support,
        registry=fx.registry,
        budget=budget,
    )
    assert calls == 2
    assert result.accepted, result.reason


@pytest.mark.parametrize(
    "target",
    [
        "detached_kernel",
        "detached_diagonal",
        "expression_terms",
        "term_object",
        "term_operators",
        "certificate_object",
        "certificate_events",
        "certificate_event",
        "source_proxy",
        "decision_terms",
        "plan_original_terms",
        "plan_requests",
        "plan_request_object",
        "request_estimate",
        "plan_uses",
        "plan_use_object",
        "reservation",
    ],
)
def test_second_planner_cannot_persistently_mutate_first_detached_authority(
    monkeypatch, target
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage
    calls = 0
    first_expression = None
    first_decision = None

    def retain_then_mutate_first(expression, *args, **kwargs):
        nonlocal calls, first_expression, first_decision
        calls += 1
        result = original(expression, *args, **kwargs)
        if calls == 1:
            first_expression = expression
            first_decision = result
            return result
        assert first_expression is not None
        assert first_decision is not None
        first_term = first_expression.terms[0]
        plan = first_decision.plan
        if target == "detached_kernel":
            array = first_term.operators[0]._kernel
            array.setflags(write=True)
            array[0, 0, 0, 0] += 0.5
            array.setflags(write=False)
        elif target == "detached_diagonal":
            array = first_term.operators[1]._diagonal
            array.setflags(write=True)
            array[0] += 0.5
            array.setflags(write=False)
        elif target == "expression_terms":
            object.__setattr__(
                first_expression,
                "terms",
                tuple(list(first_expression.terms)),
            )
        elif target == "term_object":
            object.__setattr__(
                first_expression,
                "terms",
                (replace(first_term), *first_expression.terms[1:]),
            )
        elif target == "term_operators":
            object.__setattr__(
                first_term,
                "operators",
                tuple(list(first_term.operators)),
            )
        elif target == "certificate_object":
            object.__setattr__(
                first_term,
                "certificate",
                replace(first_term.certificate),
            )
        elif target == "certificate_events":
            object.__setattr__(
                first_term.certificate,
                "events",
                tuple(list(first_term.certificate.events)),
            )
        elif target == "certificate_event":
            event = first_term.certificate.events[0]
            object.__setattr__(
                event, "selected_input", event.selected_input + 1
            )
        elif target == "source_proxy":
            object.__setattr__(
                first_term.source, "semantic_sha256", "0" * 64
            )
        elif target == "decision_terms":
            object.__setattr__(
                first_decision,
                "terms",
                tuple(list(first_decision.terms)),
            )
        elif target == "plan_original_terms":
            object.__setattr__(
                plan,
                "original_terms",
                tuple(list(plan.original_terms)),
            )
        elif target == "plan_requests":
            object.__setattr__(
                plan, "requests", tuple(list(plan.requests))
            )
        elif target == "plan_request_object":
            object.__setattr__(
                plan,
                "requests",
                (replace(plan.requests[0]), *plan.requests[1:]),
            )
        elif target == "request_estimate":
            estimate = plan.requests[0].estimate
            object.__setattr__(
                estimate,
                "resident_bytes",
                estimate.resident_bytes + 1,
            )
        elif target == "plan_uses":
            object.__setattr__(plan, "uses", tuple(list(plan.uses)))
        elif target == "plan_use_object":
            object.__setattr__(
                plan,
                "uses",
                (replace(plan.uses[0]), *plan.uses[1:]),
            )
        else:
            object.__setattr__(
                plan.reservation,
                "resident_bytes",
                plan.reservation.resident_bytes + 1,
            )
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        retain_then_mutate_first,
    )
    result = _plan(fx)
    assert calls == 2
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason in {
        "runtime_detached_conv_content_cache_mismatch",
        "runtime_detached_diagonal_content_cache_mismatch",
        "runtime_planner_semantic_snapshot_mismatch",
        "final_planner_input_changed_during_planning",
        "final_planner_decision_changed_during_planning",
        "runtime_planner_input_binding_mismatch",
        "runtime_planner_plan_binding_mismatch",
    }


def test_detached_planner_input_is_compact_read_only_and_preserves_alias_classes():
    fx = _fixture()
    derived = adapter._derive_current(
        fx.layers, fx.preds, fx.succs, fx.registry, fx.expr
    )
    detached = adapter._detach_c3_expression(derived.c3_expression)
    assert type(detached.bias) is np.ndarray
    assert detached.bias.dtype == np.dtype(np.float64)
    assert detached.bias.flags.c_contiguous
    assert detached.bias.flags.owndata
    assert detached.bias.flags.writeable is False
    assert not np.shares_memory(detached.bias, fx.expr.bias)
    for live_term, detached_term in zip(
        derived.c3_expression.terms, detached.terms, strict=True
    ):
        assert type(detached_term.source).__name__ == (
            "_DetachedSourceAuthorityProxy"
        )
        assert detached_term.source.semantic_sha256
        assert detached_term.source.c == ()
        for live, copied in zip(
            live_term.operators, detached_term.operators, strict=True
        ):
            if type(live) is ImplicitConv2DOp:
                assert type(copied._kernel) is np.ndarray
                assert copied._kernel.dtype == np.dtype(np.float64)
                assert copied._kernel.flags.c_contiguous
                assert copied._kernel.flags.owndata
                assert copied._kernel.flags.writeable is False
                assert type(copied._groups) is int
                for geometry in (
                    copied._input_shape,
                    copied._output_shape,
                    copied._stride,
                    copied._padding,
                    copied._dilation,
                ):
                    assert type(geometry) is tuple
                    assert all(type(item) is int for item in geometry)
                assert not np.shares_memory(copied._kernel, live._kernel)
                if copied._row_mask is not None:
                    assert type(copied._row_mask) is np.ndarray
                    assert copied._row_mask.dtype == np.dtype(bool)
                    assert copied._row_mask.flags.c_contiguous
                    assert copied._row_mask.flags.owndata
                    assert copied._row_mask.flags.writeable is False
                    assert not np.shares_memory(
                        copied._row_mask, live._row_mask
                    )
            else:
                assert type(copied._diagonal) is np.ndarray
                assert copied._diagonal.dtype == np.dtype(np.float64)
                assert copied._diagonal.flags.c_contiguous
                assert copied._diagonal.flags.owndata
                assert copied._diagonal.flags.writeable is False
                assert not np.shares_memory(
                    copied._diagonal, live._diagonal
                )
    assert detached.terms[0].operators[-3:] == (
        detached.terms[1].operators
    )
    assert adapter._identity_class_pattern(detached) == (
        (0, (1, 2, 3, 4, 5), (1, 2, None, None, 3, None, 4, 5)),
        (6, (3, 4, 5), (None, 3, None, 4, 5)),
    )


def test_source_custody_ledger_holds_only_weak_request_roots():
    adapter._prune_source_arena_owners()
    baseline_roots = len(adapter._SOURCE_ARENA_OWNERS)

    def one_request():
        fx = _fixture()
        source_reference = weakref.ref(fx.expr.terms[0].source)
        arena_reference = weakref.ref(fx.arena)
        result = _plan(fx)
        assert result.accepted, result.reason
        return source_reference, arena_reference

    source_reference, arena_reference = one_request()
    gc.collect()
    adapter._prune_source_arena_owners()
    assert source_reference() is None
    assert arena_reference() is None
    assert len(adapter._SOURCE_ARENA_OWNERS) <= baseline_roots


@pytest.mark.parametrize("outcome", ["rejected", "fatal"])
def test_rejected_and_fatal_requests_do_not_leave_strong_custody_roots(
    monkeypatch, outcome
):
    gc.collect()
    adapter._prune_source_arena_owners()
    baseline_roots = len(adapter._SOURCE_ARENA_OWNERS)
    fx = _fixture()
    source_reference = weakref.ref(fx.expr.terms[0].source)
    arena_reference = weakref.ref(fx.arena)
    if outcome == "fatal":
        def fatal(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(
            adapter.c3, "plan_s0_c3_identity_middle_lineage", fatal
        )
        with pytest.raises(KeyboardInterrupt):
            _plan(fx)
    else:
        result = adapter.plan_s0_c3_from_runtime_lineage(
            fx.layers,
            fx.preds,
            fx.succs,
            fx.expr,
            np.zeros(fx.expr.n_out, dtype=bool),
            registry=fx.registry,
        )
        assert not result.accepted
        del result
    del fx
    gc.collect()
    adapter._prune_source_arena_owners()
    assert source_reference() is None
    assert arena_reference() is None
    assert len(adapter._SOURCE_ARENA_OWNERS) <= baseline_roots


def test_planner_time_same_value_graph_payload_replacement_fails_cas(
    monkeypatch,
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        fx.layers[2].params["a"] = fx.layers[2].params["a"].copy()
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


def test_unknown_mutable_graph_parameter_fails_closed():
    fx = _fixture()

    class Mutable:
        pass

    fx.layers[0].params["opaque"] = Mutable()
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.endswith("unsupported_mutable_type")
    assert result.expression is fx.expr


def test_planner_time_operator_mutation_fails_the_second_current_state_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        fx.outer._kernel[0, 0, 0, 0] += 0.5
        return result

    monkeypatch.setattr(adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return)
    result = _plan(fx)
    assert not result.accepted
    assert result.expression is fx.expr
    assert result.reason.startswith("current_state_")


def test_planner_time_bias_object_replacement_is_detected_by_reference_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(fx.expr, "bias", fx.expr.bias.copy())
        return result

    monkeypatch.setattr(adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")


@pytest.mark.parametrize("buffer_name", ["data", "indices", "indptr"])
def test_planner_time_sparse_source_buffer_mutation_fails_cas(
    monkeypatch, buffer_name
):
    fx = _fixture()
    matrix = fx.expr.terms[0].source.Gc
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        buffer = getattr(matrix, buffer_name)
        if buffer_name == "data":
            buffer[0] += 0.5
        else:
            buffer[0] += 1
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")


def test_planner_time_source_boundary_cache_mutation_fails_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        fx.source_by_layer[8] = _source("new-boundary", fx.expr.n_out)
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("exact", False),
        ("frame_id", 99),
    ],
)
def test_planner_time_source_semantic_mutation_fails_second_cas(
    monkeypatch, field_name, new_value
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(
            fx.expr.terms[0].source, field_name, new_value
        )
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


def test_planner_time_source_n_out_mutation_fails_second_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        source = fx.expr.terms[0].source
        source.c = source.c[:-1]
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


def test_planner_time_unknown_mutable_source_field_cannot_evade_cas(monkeypatch):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    class Mutable:
        def __init__(self):
            self.value = 1

    mutable = Mutable()

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        fx.expr.terms[0].source.Ac = mutable
        mutable.value = 999
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


@pytest.mark.parametrize("mutation", ["snapshot_tuple", "operand_tuple"])
def test_planner_time_add_authority_container_replacement_fails_cas(
    monkeypatch, mutation
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        if mutation == "snapshot_tuple":
            fx.registry.add_operand_snapshots = tuple(
                list(fx.registry.add_operand_snapshots)
            )
        else:
            snapshot = fx.registry.add_operand_snapshots[0]
            object.__setattr__(
                snapshot, "operands", tuple(list(snapshot.operands))
            )
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


@pytest.mark.parametrize("mutation", ["cache_bias", "cache_terms"])
def test_planner_time_affine_cache_entry_mutation_fails_cas(
    monkeypatch, mutation
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        cache_entry = fx.affine_cache[3]
        if mutation == "cache_bias":
            cache_entry.bias[0] += 0.25
        else:
            object.__setattr__(
                cache_entry, "terms", tuple(list(cache_entry.terms))
            )
        return result

    monkeypatch.setattr(
        adapter.c3, "plan_s0_c3_identity_middle_lineage", mutate_then_return
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


@pytest.mark.parametrize(
    "mutation", ["terminal_bias", "terminal_terms", "terminal_snapshot"]
)
def test_planner_time_terminal_authority_mutation_fails_cas(
    monkeypatch, mutation
):
    fx = _fixture()
    original = adapter.c3.plan_s0_c3_identity_middle_lineage

    def mutate_then_return(*args, **kwargs):
        result = original(*args, **kwargs)
        terminal_entry = fx.affine_cache[9]
        if mutation == "terminal_bias":
            terminal_entry.bias[0] += 0.25
        elif mutation == "terminal_terms":
            object.__setattr__(
                terminal_entry,
                "terms",
                tuple(list(terminal_entry.terms)),
            )
        else:
            fx.registry.terminal_expression_snapshot = replace(
                fx.registry.terminal_expression_snapshot
            )
        return result

    monkeypatch.setattr(
        adapter.c3,
        "plan_s0_c3_identity_middle_lineage",
        mutate_then_return,
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason.startswith("current_state_")
    assert result.expression is fx.expr


@pytest.mark.parametrize(
    "mutation",
    ["owner_cache", "source_ledger", "owner_affine_cache", "affine_ledger"],
)
def test_preplan_arena_container_replacement_fails_closed(mutation):
    fx = _fixture()
    if mutation == "owner_cache":
        fx.arena.owner_cache = dict(fx.arena.owner_cache)
        expected = "runtime_owner_cache_cas_mismatch"
    elif mutation == "source_ledger":
        fx.arena.source_entries = tuple(list(fx.arena.source_entries))
        expected = "runtime_arena_source_ledger_cas_mismatch"
    elif mutation == "owner_affine_cache":
        fx.arena.owner_affine_cache = dict(fx.arena.owner_affine_cache)
        expected = "runtime_owner_cache_cas_mismatch"
    else:
        fx.arena.affine_entries = list(fx.arena.affine_entries)
        expected = "runtime_registry_authority_cas_mismatch"
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == expected
    assert result.expression is fx.expr


@pytest.mark.parametrize(
    ("owner_name", "field_name", "reason"),
    [
        ("registry", "sealed", "runtime_registry_sealed_not_exact_bool"),
        ("arena", "closed", "runtime_arena_closed_not_exact_bool"),
        (
            "factor_allocator",
            "sealed",
            "runtime_factor_allocator_sealed_not_exact_bool",
        ),
    ],
)
def test_sealed_authority_flags_reject_callback_capable_bool_objects(
    owner_name, field_name, reason
):
    fx = _fixture()
    callbacks = []

    class HostileBool:
        def __bool__(self):
            callbacks.append("called")
            fx.layers[0].kind = "EVIL_BOOL_CALLBACK"
            return True

    owner = (
        fx.arena.factor_allocator
        if owner_name == "factor_allocator"
        else getattr(fx, owner_name)
    )
    setattr(owner, field_name, HostileBool())
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == reason
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


def test_factor_allocation_record_rejects_hostile_equal_string_kind():
    fx = _fixture()
    callbacks = []

    class HostileKind(str):
        def __eq__(self, other):
            callbacks.append(other)
            fx.layers[0].kind = "EVIL_FACTOR_RECORD_CALLBACK"
            return str.__eq__(self, other)

        def __ne__(self, other):
            return not self.__eq__(other)

    record = fx.arena.factor_allocator.allocation_history[0]
    object.__setattr__(record, "kind", HostileKind("ROOT"))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_factor_allocation_kind_not_exact_string"
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


def test_candidate_event_rejects_hostile_equal_string_kind():
    fx = _fixture()
    callbacks = []

    class HostileKind(str):
        def __eq__(self, other):
            callbacks.append(other)
            fx.layers[0].kind = "EVIL_EVENT_CALLBACK"
            return str.__eq__(self, other)

    event = fx.expr.terms[0].events[0]
    object.__setattr__(event, "kind", HostileKind("CONV"))
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_event_kind_not_exact_string"
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


@pytest.mark.parametrize("target", ["custody", "expected_prefix"])
def test_sealed_lineage_digest_rejects_hostile_equal_string(target):
    fx = _fixture()
    callbacks = []

    class HostileDigest(str):
        def __eq__(self, other):
            callbacks.append(other)
            fx.layers[0].kind = "EVIL_DIGEST_CALLBACK"
            return str.__eq__(self, other)

    if target == "custody":
        authority = fx.expr.terms[0].events[0]._custody
    else:
        authority = fx.registry.add_operand_snapshots[0].expected_prefixes[0]
    object.__setattr__(
        authority,
        "canonical_sha256",
        HostileDigest(authority.canonical_sha256),
    )
    result = _plan(fx)
    assert not result.accepted
    assert "not_exact_string" in result.reason
    assert callbacks == []
    assert fx.layers[0].kind == "RELU"


def test_sealed_operand_path_rejects_equal_list_replacement():
    fx = _fixture()
    lineage = fx.registry.add_operand_snapshots[0].operands[0][0]
    object.__setattr__(
        lineage, "path_layer_ids", list(lineage.path_layer_ids)
    )
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_operand_path_not_exact_tuple"
    assert result.expression is fx.expr


@pytest.mark.parametrize(
    ("target", "field_name", "reason"),
    [
        ("registry", "graph_sha256", "runtime_registry_graph_sha256_not_exact_string"),
        ("arena", "graph_sha256", "runtime_arena_graph_sha256_not_exact_string"),
        ("arena", "frame_payload", "runtime_arena_frame_payload_not_exact_bytes"),
        (
            "source_boundaries",
            "key_payload",
            "runtime_source_boundary_key_payload_not_exact_bytes",
        ),
    ],
)
def test_sealed_scalar_payloads_reject_builtin_subclasses(
    target, field_name, reason
):
    fx = _fixture()
    owner = (
        fx.registry.source_boundaries
        if target == "source_boundaries"
        else getattr(fx, target)
    )
    current = getattr(owner, field_name)
    hostile = type("HostileScalar", (type(current),), {})(current)
    object.__setattr__(owner, field_name, hostile)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == reason


def test_ordinary_planner_exception_fails_closed_by_original_expression_identity(monkeypatch):
    fx = _fixture()

    def fail(*args, **kwargs):
        raise RuntimeError("ordinary")

    monkeypatch.setattr(adapter.c3, "plan_s0_c3_identity_middle_lineage", fail)
    result = _plan(fx)
    assert not result.accepted
    assert result.reason == "runtime_planner_failed:RuntimeError"
    assert result.expression is fx.expr


@pytest.mark.parametrize("fatal", [KeyboardInterrupt(), SystemExit(17)])
def test_baseexception_from_planner_propagates(monkeypatch, fatal):
    fx = _fixture()

    def fail(*args, **kwargs):
        raise fatal

    monkeypatch.setattr(adapter.c3, "plan_s0_c3_identity_middle_lineage", fail)
    with pytest.raises(type(fatal)):
        _plan(fx)


def test_pure_planner_rejection_still_returns_original_runtime_expression():
    fx = _fixture()
    result = adapter.plan_s0_c3_from_runtime_lineage(
        fx.layers, fx.preds, fx.succs, fx.expr,
        np.zeros(fx.expr.n_out, dtype=bool),
        registry=fx.registry,
    )
    assert not result.accepted
    assert result.reason == "c3_planner_rejected:empty_output_support"
    assert result.expression is fx.expr
    assert result.c3_expression is None
    assert result.planner_decision is None


def test_adapter_is_not_imported_by_production_runtime_and_has_no_execution_api():
    root = Path(__file__).resolve().parents[2]
    production = root / "act"
    needle = "s0_c3_runtime_lineage_adapter_prototype"
    assert all(needle not in path.read_text(errors="ignore") for path in production.rglob("*.py"))
    forbidden = {"execute", "materialize", "rewrite", "publish", "verify"}
    assert forbidden.isdisjoint(set(adapter.RuntimeAdapterDecision.__dict__))
    assert "_begin_private_runtime_lineage_registry" not in adapter.__all__
    assert "_begin_private_runtime_lineage_arena" not in adapter.__all__
    assert "_begin_private_runtime_factor_allocator" not in adapter.__all__
    assert "_record_private_runtime_factor_frame_root" not in adapter.__all__
    assert (
        "_record_private_runtime_factor_source_prefix"
        not in adapter.__all__
    )
    assert "_record_private_runtime_factor_rebase" not in adapter.__all__
    assert "_seal_private_runtime_factor_allocator" not in adapter.__all__
    assert "_record_private_runtime_source_boundary" not in adapter.__all__
    assert "_capture_private_runtime_operand_lineage" not in adapter.__all__
    assert (
        "_record_private_runtime_affine_expression_cache"
        not in adapter.__all__
    )
    assert "_record_private_runtime_add_operands" not in adapter.__all__
    assert (
        "_record_private_runtime_terminal_expression"
        not in adapter.__all__
    )
    assert "_seal_private_runtime_lineage_registry" not in adapter.__all__
    assert "no_graph_repair_materialization_verifier_run_or_score_gain" in adapter.RUNTIME_ADAPTER_NO_CLAIMS
