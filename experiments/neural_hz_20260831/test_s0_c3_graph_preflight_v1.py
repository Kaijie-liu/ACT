"""Tests of negative-screen completeness and independent concrete dataflow."""

from types import SimpleNamespace

import pytest
import torch

from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import (
    affine_source_paths, c3_necessary_condition, concrete_forward,
)


def layer(lid, kind, inputs=(), outputs=(), **params):
    return SimpleNamespace(id=lid, kind=kind, in_vars=list(inputs),
                           out_vars=list(outputs), params=params)


def nested_graph():
    layers = [layer(0, "RELU"), layer(1, "CONV2D"), layer(2, "RELU"),
              layer(3, "ADD"), layer(4, "RELU"), layer(5, "CONV2D"),
              layer(6, "ADD"), layer(7, "CONV2D"), layer(8, "BIAS")]
    preds = {0: [], 1: [0], 2: [], 3: [1, 2], 4: [], 5: [4],
             6: [3, 5], 7: [6], 8: [7]}
    return layers, preds


def test_single_add_only_passes_necessary_shape_never_authorizes():
    layers, preds = nested_graph()
    preds[6] = [2, 5]
    result = c3_necessary_condition(layers, preds, 8)
    assert not result["structurally_impossible"]
    assert result["reason_counts"] == {"necessary_shape_only": 2}
    assert not result["runtime_lineage_proved"]
    assert not result["positive_authorization"]


def test_complete_nested_residual_cannot_cut_at_earlier_add():
    layers, preds = nested_graph()
    result = c3_necessary_condition(layers, preds, 8)
    assert result["structurally_impossible"]
    assert result["reason_counts"] == {"nested_add": 2, "necessary_shape_only": 1}
    assert [row["source"] for row in result["paths"]] == [0, 2, 4]
    assert [row["add_occurrences"] for row in result["paths"]] == [[3, 6], [3, 6], [6]]


def test_duplicate_operand_preserves_multiplicity():
    layers, preds = nested_graph()
    preds[6] = [3, 3]
    paths = affine_source_paths(layers, preds, 8)
    assert len(paths) == 4
    assert paths[0] == paths[2] and paths[1] == paths[3]
    assert c3_necessary_condition(layers, preds, 8)["reason_counts"] == {"nested_add": 4}


def test_new_nonlinear_source_is_a_real_permitted_cut():
    layers, preds = nested_graph()
    layers[3].kind = "RELU"
    paths = affine_source_paths(layers, preds, 8)
    assert [path.source for path in paths] == [3, 4]
    assert all(3 not in path.events for path in paths)


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "arity", "budget"])
def test_incomplete_census_cannot_produce_a_negative_result(mutation):
    layers, preds = nested_graph()
    if mutation == "cycle":
        preds[1] = [7]
    elif mutation == "unknown":
        layers[3].kind = "RESHAPE"
    elif mutation == "arity":
        preds[6] = [5]
    with pytest.raises(ValueError):
        affine_source_paths(layers, preds, 8, max_paths=2 if mutation == "budget" else 4096)


def test_forward_probe_detects_omitted_scale_and_matches_repaired_edges_exactly():
    layers = [
        layer(0, "INPUT", (), (0, 1)),
        layer(1, "INPUT_SPEC", (0, 1), (0, 1)),
        layer(2, "SCALE", (0, 1), (2, 3), a=torch.tensor([2., -0.5], dtype=torch.float64)),
        layer(3, "BIAS", (2, 3), (4, 5), c=torch.tensor([0.25, 1.], dtype=torch.float64)),
        layer(4, "ADD", (0, 1, 4, 5), (6, 7), x_vars=[0, 1], y_vars=[4, 5]),
    ]
    sibling = {0: [], 1: [0], 2: [1], 3: [1], 4: [1, 3]}
    repaired = {**sibling, 3: [2]}
    x = torch.tensor([0.5, -1.], dtype=torch.float64)
    variable = concrete_forward(layers, sibling, x, variable_program=True)
    candidate = concrete_forward(layers, repaired, x)
    original = concrete_forward(layers, sibling, x)
    assert all(torch.equal(variable[lid], candidate[lid]) for lid in variable)
    assert torch.equal(candidate[4], torch.tensor([1.75, 0.5], dtype=torch.float64))
    assert not torch.equal(original[4], candidate[4])
