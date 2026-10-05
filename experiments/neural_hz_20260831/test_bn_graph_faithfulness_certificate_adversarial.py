"""Adversarial tests for the isolated BN graph-faithfulness prototype.

These tests exercise only graph/variable evidence, clone-only repair
authorization, and transaction integrity.  They deliberately do not treat
hand-authored path observations as production runtime-lineage evidence; the
prototype explicitly disclaims that adapter.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.neural_hz_20260831 import (
    bn_graph_faithfulness_certificate_prototype as graph_cert,
)


def _layer(layer_id, kind, in_vars, out_vars, *, params=None):
    return SimpleNamespace(
        id=layer_id,
        kind=kind,
        in_vars=list(in_vars),
        out_vars=list(out_vars),
        params={} if params is None else dict(params),
    )


def _successors(preds):
    result = {key: [] for key in preds}
    for layer_id, predecessors in preds.items():
        for predecessor in predecessors:
            if layer_id not in result[predecessor]:
                result[predecessor].append(layer_id)
    return result


def _bn_chain(*, sibling=False, payload_kind="list"):
    if payload_kind == "list":
        scale_payload = [1.25, -0.5]
        bias_payload = [0.125, 0.75]
    elif payload_kind == "numpy":
        scale_payload = np.asarray([1.25, -0.5], dtype=np.float64)
        bias_payload = np.asarray([0.125, 0.75], dtype=np.float64)
    else:  # pragma: no cover - helper misuse
        raise AssertionError(payload_kind)
    layers = [
        _layer(0, "INPUT", (), (0, 1)),
        _layer(1, "INPUT_SPEC", (0, 1), (0, 1)),
        _layer(2, "CONV2D", (0, 1), (2, 3)),
        _layer(
            3,
            "SCALE",
            (2, 3),
            (4, 5),
            params={
                "a": scale_payload,
                "is_batchnorm_decomposition": True,
            },
        ),
        _layer(
            4,
            "BIAS",
            (4, 5),
            (6, 7),
            params={
                "c": bias_payload,
                "is_batchnorm_decomposition": True,
                "paired_with_scale": True,
            },
        ),
        _layer(5, "RELU", (6, 7), (8, 9)),
    ]
    if sibling:
        preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [2], 5: [4]}
    else:
        preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    return layers, preds, _successors(preds)


def _issue_codes(certificate):
    return tuple(issue.code for issue in certificate.issues)


@pytest.mark.parametrize("missing", ["a", "c"])
def test_marked_bn_event_without_required_operator_payload_fails_closed(missing):
    layers, preds, succs = _bn_chain(sibling=True)
    del layers[3 if missing == "a" else 4].params[missing]

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized
    assert plan.replacements == ()


@pytest.mark.parametrize(
    ("target", "payload"),
    [
        ("a", [1.0]),
        ("c", [0.0]),
        ("a", np.asarray([[1.0, 2.0]], dtype=np.float64)),
        ("c", np.asarray(0.0, dtype=np.float64)),
        ("a", [True, False]),
        ("c", [False, True]),
        ("a", [1.0 + 0.0j, 2.0 + 0.0j]),
        ("c", ["0", "1"]),
        ("a", [np.nan, 1.0]),
        ("c", [np.inf, 0.0]),
    ],
)
def test_bn_payload_must_be_finite_numeric_1d_and_exact_width(target, payload):
    layers, preds, succs = _bn_chain(sibling=True)
    layers[3 if target == "a" else 4].params[target] = payload

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized


@pytest.mark.parametrize("event", ["scale", "bias"])
def test_bn_event_input_output_width_mismatch_fails_closed(event):
    layers, preds, succs = _bn_chain(sibling=True)
    if event == "scale":
        layers[3].out_vars = [4]
        layers[3].params["a"] = [1.25, -0.5]
        layers[4].in_vars = [4]
        layers[4].params["c"] = [0.125]
    else:
        layers[4].out_vars = [6]

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized


def test_empty_bn_event_is_not_a_zero_width_repair_candidate():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(
            1,
            "SCALE",
            (),
            (),
            params={"a": [], "is_batchnorm_decomposition": True},
        ),
        _layer(
            2,
            "BIAS",
            (),
            (),
            params={
                "c": [],
                "is_batchnorm_decomposition": True,
                "paired_with_scale": True,
            },
        ),
    ]
    preds = {0: [], 1: [], 2: []}
    succs = _successors(preds)

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized


def test_valid_numpy_and_torch_bn_payloads_are_supported():
    numpy_layers, preds, succs = _bn_chain(payload_kind="numpy")
    numpy_certificate = graph_cert.audit_graph_faithfulness(
        numpy_layers, preds, succs
    )
    assert numpy_certificate.accepted

    torch = pytest.importorskip("torch")
    torch_layers, preds, succs = _bn_chain()
    torch_layers[3].params["a"] = torch.tensor(
        [1.25, -0.5], dtype=torch.float32
    )
    torch_layers[4].params["c"] = torch.tensor(
        [0.125, 0.75], dtype=torch.float32
    )
    torch_certificate = graph_cert.audit_graph_faithfulness(
        torch_layers, preds, succs
    )
    assert torch_certificate.accepted


@pytest.mark.parametrize("target", ["a", "c"])
def test_bn_payload_value_is_bound_into_graph_digest_and_plan_cas(target):
    layers, preds, succs = _bn_chain(sibling=True)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)
    assert plan.authorized

    layer = layers[3 if target == "a" else 4]
    layer.params[target][0] += 0.5
    changed = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert changed.graph_sha256 != plan.source_graph_sha256
    with pytest.raises(graph_cert.GraphFaithfulnessReject, match="digest CAS"):
        graph_cert.apply_repair_plan_to_clone(layers, preds, succs, plan)


def test_bn_payload_dtype_is_bound_into_graph_digest_and_plan_cas():
    layers, preds, succs = _bn_chain(sibling=True, payload_kind="numpy")
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)
    assert plan.authorized

    layers[3].params["a"] = layers[3].params["a"].astype(np.float32)
    changed = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert changed.graph_sha256 != plan.source_graph_sha256
    with pytest.raises(graph_cert.GraphFaithfulnessReject, match="digest CAS"):
        graph_cert.apply_repair_plan_to_clone(layers, preds, succs, plan)


def test_non_alias_cross_layer_variable_redefinition_is_rejected():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "DENSE", (0,), (1,)),
        _layer(2, "DENSE", (0,), (1,)),
        _layer(3, "RELU", (1,), (2,)),
    ]
    preds = {0: [], 1: [0], 2: [0], 3: [2]}
    certificate = graph_cert.audit_graph_faithfulness(
        layers, preds, _successors(preds)
    )

    assert not certificate.accepted


def test_input_spec_alias_whitelist_requires_exact_in_out_identity():
    layers = [
        _layer(0, "INPUT", (), (0, 1)),
        _layer(1, "INPUT_SPEC", (0, 1), (0, 2)),
        _layer(2, "RELU", (0, 2), (3, 4)),
    ]
    preds = {0: [], 1: [0], 2: [1]}
    certificate = graph_cert.audit_graph_faithfulness(
        layers, preds, _successors(preds)
    )

    assert not certificate.accepted


def test_assert_exact_wrapper_alias_is_accepted_but_partial_alias_is_not():
    exact_layers = [
        _layer(0, "INPUT", (), (0, 1)),
        _layer(1, "DENSE", (0, 1), (2, 3)),
        _layer(2, "ASSERT", (2, 3), (2, 3)),
    ]
    preds = {0: [], 1: [0], 2: [1]}
    exact = graph_cert.audit_graph_faithfulness(
        exact_layers, preds, _successors(preds)
    )

    partial_layers = deepcopy(exact_layers)
    partial_layers[2].out_vars = [2, 4]
    partial = graph_cert.audit_graph_faithfulness(
        partial_layers, preds, _successors(preds)
    )

    assert exact.accepted
    assert not partial.accepted
    assert "duplicate_output_variable_definition" in _issue_codes(partial)


def test_duplicate_variable_producer_cannot_hide_inside_bn_repair():
    layers = [
        _layer(0, "INPUT", (), (0, 1)),
        _layer(1, "INPUT_SPEC", (0, 1), (0, 1)),
        _layer(2, "CONV2D", (0, 1), (2, 3)),
        _layer(3, "DENSE", (0, 1), (2, 3)),
        _layer(
            4,
            "SCALE",
            (2, 3),
            (4, 5),
            params={"a": [1.0, 1.0], "is_batchnorm_decomposition": True},
        ),
        _layer(
            5,
            "BIAS",
            (4, 5),
            (6, 7),
            params={
                "c": [0.0, 0.0],
                "is_batchnorm_decomposition": True,
                "paired_with_scale": True,
            },
        ),
        _layer(6, "RELU", (6, 7), (8, 9)),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [3], 5: [3], 6: [5]}
    succs = _successors(preds)

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized
    assert plan.replacements == ()


@pytest.mark.parametrize("kind", ["ADD", "SUB", "MUL", "MATMUL"])
def test_multi_operand_event_without_operand_partitions_fails_closed(kind):
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "CONSTANT", (), (1,)),
        _layer(2, kind, (0, 1), (2,)),
    ]
    preds = {0: [], 1: [], 2: [0, 1]}
    certificate = graph_cert.audit_graph_faithfulness(
        layers, preds, _successors(preds)
    )

    assert not certificate.accepted


def test_repeated_multi_operand_without_partitions_cannot_collapse_to_one_edge():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "SUB", (0, 0), (1,)),
    ]
    preds = {0: [], 1: [0]}
    certificate = graph_cert.audit_graph_faithfulness(
        layers, preds, _successors(preds)
    )

    assert not certificate.accepted


def test_unary_bn_scale_with_two_producer_occurrences_is_not_repairable():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "CONSTANT", (), (1,)),
        _layer(
            2,
            "SCALE",
            (0, 1),
            (2, 3),
            params={"a": [1.0, 2.0], "is_batchnorm_decomposition": True},
        ),
        _layer(
            3,
            "BIAS",
            (2, 3),
            (4, 5),
            params={
                "c": [0.0, 0.0],
                "is_batchnorm_decomposition": True,
                "paired_with_scale": True,
            },
        ),
        _layer(4, "RELU", (4, 5), (6, 7)),
    ]
    preds = {0: [], 1: [], 2: [0, 1], 3: [0, 1], 4: [3]}
    succs = _successors(preds)

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert not plan.authorized


def test_one_invalid_pair_blocks_the_entire_multi_pair_repair():
    layers, preds, _ = _bn_chain(sibling=True)
    layers.extend(
        [
            _layer(
                6,
                "SCALE",
                (8, 9),
                (10, 11),
                params={
                    "a": [1.0, 1.0],
                    "is_batchnorm_decomposition": True,
                },
            ),
            _layer(
                7,
                "BIAS",
                (10, 11),
                (12, 13),
                params={
                    "c": [0.0],
                    "is_batchnorm_decomposition": True,
                    "paired_with_scale": True,
                },
            ),
            _layer(8, "RELU", (12, 13), (14, 15)),
        ]
    )
    preds.update({6: [5], 7: [5], 8: [7]})
    succs = _successors(preds)

    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not plan.authorized
    assert plan.replacements == ()


def test_public_plan_dataclass_cannot_authorize_a_non_bn_edge_repair():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "DENSE", (0,), (1,)),
        _layer(2, "RELU", (1,), (2,)),
    ]
    source_preds = {0: [], 1: [0], 2: [0]}
    source_succs = _successors(source_preds)
    source = graph_cert.audit_graph_faithfulness(
        layers, source_preds, source_succs
    )
    assert not source.accepted

    candidate_preds = {0: [], 1: [0], 2: [1]}
    candidate = graph_cert.audit_graph_faithfulness(
        layers, candidate_preds, _successors(candidate_preds)
    )
    assert candidate.accepted
    forged = graph_cert.BatchNormRepairPlan(
        authorized=True,
        reason="authorized_bn_sibling_to_chain_clone_only",
        source_graph_sha256=source.graph_sha256,
        candidate_graph_sha256=candidate.graph_sha256,
        replacements=(graph_cert.EdgeReplacement(2, (0,), (1,)),),
        candidate_certificate=candidate,
    )

    with pytest.raises(graph_cert.GraphFaithfulnessReject):
        graph_cert.apply_repair_plan_to_clone(
            layers, source_preds, source_succs, forged
        )


@pytest.mark.parametrize("target", ["clone", "certificate"])
def test_validator_cannot_tamper_with_staged_objects_and_still_commit(target):
    layers, preds, succs = _bn_chain(sibling=True)

    def tamper(clone, certificate):
        if target == "clone":
            object.__setattr__(clone, "graph_sha256", "f" * 64)
        else:
            object.__setattr__(certificate, "accepted", False)
        return True

    result = graph_cert.execute_clone_only_repair_transaction(
        layers, preds, succs, candidate_validator=tamper
    )

    assert not result.committed
    assert result.clone is None


def test_validator_caller_graph_mutation_invalidates_transaction_source_cas():
    layers, preds, succs = _bn_chain(sibling=True)

    def mutate_caller_graph(_clone, _certificate):
        preds[5] = [3]
        succs.clear()
        succs.update(_successors(preds))
        return True

    result = graph_cert.execute_clone_only_repair_transaction(
        layers, preds, succs, candidate_validator=mutate_caller_graph
    )

    assert not result.committed
    assert result.clone is None


def test_validator_caller_payload_mutation_invalidates_transaction_source_cas():
    layers, preds, succs = _bn_chain(sibling=True)

    def mutate_caller_payload(_clone, _certificate):
        layers[3].params["a"][0] = 9.0
        return True

    result = graph_cert.execute_clone_only_repair_transaction(
        layers, preds, succs, candidate_validator=mutate_caller_payload
    )

    assert not result.committed
    assert result.clone is None


@pytest.mark.parametrize("exc", [GeneratorExit(), KeyboardInterrupt()])
def test_transaction_never_downgrades_baseexception_from_validator(exc):
    layers, preds, succs = _bn_chain(sibling=True)

    def interrupt(_clone, _certificate):
        raise exc

    with pytest.raises(type(exc)):
        graph_cert.execute_clone_only_repair_transaction(
            layers, preds, succs, candidate_validator=interrupt
        )


def test_path_cannot_skip_passive_bias_event_and_echoes_exact_endpoints():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [3, 5],
        [graph_cert.PathOperatorObservation(3, "diagonal_scale")],
    )

    assert not certificate.accepted
    assert certificate.path_layer_ids == (3, 5)
    assert "path_skips_graph_event" in tuple(
        issue.code for issue in certificate.issues
    )


def test_path_observation_at_passive_event_never_counts_as_linear_coverage():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            graph_cert.PathOperatorObservation(2, "implicit_conv2d"),
            graph_cert.PathOperatorObservation(3, "diagonal_scale"),
            graph_cert.PathOperatorObservation(4, "diagonal_scale"),
        ],
    )

    assert not certificate.accepted
    assert "extra_lineage_operator" in tuple(
        issue.code for issue in certificate.issues
    )


def test_large_variable_evidence_is_stable_bounded_and_address_free():
    missing = tuple(range(10_000, 20_000))
    layers_a = [_layer(0, "RELU", missing, (0,))]
    layers_b = [_layer(0, "RELU", list(missing), (0,))]
    preds = {0: []}
    succs = {0: []}

    first = graph_cert.audit_graph_faithfulness(layers_a, preds, succs)
    second = graph_cert.audit_graph_faithfulness(layers_b, preds, succs)
    issue = first.issues[0]
    encoded = json.dumps(
        graph_cert.certificate_to_jsonable(first), sort_keys=True
    )

    assert not first.accepted
    assert first.graph_sha256 == second.graph_sha256
    assert issue.variable_count == 10_000
    assert len(issue.variable_sha256) == 64
    assert issue.variable_sha256 == second.issues[0].variable_sha256
    assert issue.variable_sample == missing[:4] + missing[-4:]
    assert len(issue.variable_sample) == 8
    assert "0x" not in encoded


def test_graph_digest_is_independent_of_mapping_insertion_order():
    layers, preds, succs = _bn_chain()
    reversed_preds = dict(reversed(tuple(preds.items())))
    reversed_succs = dict(reversed(tuple(succs.items())))

    first = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    second = graph_cert.audit_graph_faithfulness(
        layers, reversed_preds, reversed_succs
    )

    assert first.accepted and second.accepted
    assert first.graph_sha256 == second.graph_sha256


def test_audit_does_not_mutate_payload_arrays_or_graph_containers():
    layers, preds, succs = _bn_chain(sibling=True, payload_kind="numpy")
    before_a = layers[3].params["a"].copy()
    before_c = layers[4].params["c"].copy()
    before_preds = deepcopy(preds)
    before_succs = deepcopy(succs)

    graph_cert.audit_graph_faithfulness(layers, preds, succs)
    graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    np.testing.assert_array_equal(layers[3].params["a"], before_a)
    np.testing.assert_array_equal(layers[4].params["c"], before_c)
    assert preds == before_preds
    assert succs == before_succs
