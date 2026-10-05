"""Focused tests for the isolated BN graph-faithfulness certificate."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import pytest

from experiments.neural_hz_20260831 import (
    bn_graph_faithfulness_certificate_prototype as graph_cert,
)


def _layer(
    layer_id,
    kind,
    in_vars,
    out_vars,
    *,
    params=None,
):
    return SimpleNamespace(
        id=layer_id,
        kind=kind,
        in_vars=list(in_vars),
        out_vars=list(out_vars),
        params={} if params is None else dict(params),
    )


def _bn_chain(*, sibling=False):
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
                "a": [1.25, -0.5],
                "is_batchnorm_decomposition": True,
            },
        ),
        _layer(
            4,
            "BIAS",
            (4, 5),
            (6, 7),
            params={
                "c": [0.125, -0.25],
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
    succs = _successors(preds)
    return layers, preds, succs


def _successors(preds):
    result = {key: [] for key in preds}
    for layer_id, predecessors in preds.items():
        for predecessor in predecessors:
            if layer_id not in result[predecessor]:
                result[predecessor].append(layer_id)
    return result


def _issue_codes(certificate):
    return tuple(issue.code for issue in certificate.issues)


def test_correct_bn_chain_has_complete_graph_variable_certificate():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert certificate.accepted
    assert len(certificate.graph_sha256) == 64
    assert certificate.layer_count == 6
    assert certificate.edge_count == 5
    assert certificate.checked_input_variables == 10
    assert certificate.issues == ()
    assert certificate.formal_baseline == "1870/2413"
    assert certificate.gain == 0
    assert certificate.default_enabled is False
    assert certificate.batchnorm_pairs == (
        graph_cert.BatchNormPair(3, 4, (2,), 2, True),
    )


def test_sibling_bug_is_rejected_as_producer_and_missing_graph_event():
    layers, preds, succs = _bn_chain(sibling=True)
    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert not certificate.accepted
    assert _issue_codes(certificate) == (
        "bn_scale_bias_graph_event_missing",
        "predecessor_producer_mismatch",
    )
    assert certificate.batchnorm_pairs == (
        graph_cert.BatchNormPair(3, 4, (2,), 2, False),
    )


def test_sibling_bug_gets_deterministic_clone_only_repair_plan():
    layers, preds, succs = _bn_chain(sibling=True)
    before_layers = deepcopy(layers)
    before_preds = deepcopy(preds)
    before_succs = deepcopy(succs)

    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert plan.authorized, plan.reason
    assert plan.reason == "authorized_bn_sibling_to_chain_clone_only"
    assert plan.replacements == (
        graph_cert.EdgeReplacement(4, (2,), (3,)),
    )
    assert plan.source_graph_sha256 != plan.candidate_graph_sha256
    assert plan.candidate_certificate is not None
    assert plan.candidate_certificate.accepted
    assert layers == before_layers
    assert preds == before_preds
    assert succs == before_succs


def test_apply_returns_faithful_immutable_private_clone_without_mutation():
    layers, preds, succs = _bn_chain(sibling=True)
    before_preds = deepcopy(preds)
    before_succs = deepcopy(succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    clone, certificate = graph_cert.apply_repair_plan_to_clone(
        layers, preds, succs, plan
    )

    assert certificate.accepted
    assert clone.graph_sha256 == certificate.graph_sha256
    assert clone.predecessor_dict()[4] == [3]
    assert clone.successor_dict()[2] == [3]
    assert clone.successor_dict()[3] == [4]
    assert preds == before_preds
    assert succs == before_succs
    copy_out = clone.predecessor_dict()
    copy_out[4].append(99)
    assert clone.predecessor_dict()[4] == [3]


def test_plan_and_apply_are_bound_to_exact_source_graph_digest():
    layers, preds, succs = _bn_chain(sibling=True)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)
    preds[5] = [3]
    succs = _successors(preds)

    with pytest.raises(graph_cert.GraphFaithfulnessReject, match="digest CAS"):
        graph_cert.apply_repair_plan_to_clone(layers, preds, succs, plan)


def test_already_faithful_graph_does_not_create_a_repair_menu():
    layers, preds, succs = _bn_chain()
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not plan.authorized
    assert plan.reason == "source_graph_already_faithful"
    assert plan.replacements == ()


def test_missing_paired_bias_event_fails_closed_and_cannot_be_repaired():
    layers, preds, _ = _bn_chain()
    del layers[4]
    layers[4].id = 4
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
    succs = _successors(preds)

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert "bn_scale_pair_count_mismatch" in _issue_codes(certificate)
    assert not plan.authorized
    assert plan.replacements == ()


def test_ambiguous_two_bias_consumers_fail_closed():
    layers, _, _ = _bn_chain()
    layers.insert(
        5,
        _layer(
            5,
            "BIAS",
            (4, 5),
            (10, 11),
            params={
                "c": [0.5, -0.5],
                "is_batchnorm_decomposition": True,
                "paired_with_scale": True,
            },
        ),
    )
    layers[6].id = 6
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [3], 6: [4]}
    succs = _successors(preds)

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert not certificate.accepted
    assert "bn_scale_pair_count_mismatch" in _issue_codes(certificate)
    assert _issue_codes(certificate).count("bn_bias_without_unique_scale") == 2


@pytest.mark.parametrize(
    "params",
    [
        {"is_batchnorm_decomposition": 1},
        {"is_batchnorm_decomposition": "true"},
        {"is_batchnorm_decomposition": True, "paired_with_scale": 1},
    ],
)
def test_non_exact_bn_markers_are_malformed_and_never_planned(params):
    layers, preds, succs = _bn_chain(sibling=True)
    layers[4].params = params

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not certificate.accepted
    assert certificate.graph_sha256 == ""
    assert certificate.issues[0].code.startswith("malformed_input:")
    assert not plan.authorized
    assert plan.reason == "malformed_source_graph"


def test_bn_markers_on_unrelated_kind_fail_closed():
    layers, preds, succs = _bn_chain()
    layers[5].params["is_batchnorm_decomposition"] = True

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert not certificate.accepted
    assert "bn_marker_on_unsupported_kind" in _issue_codes(certificate)


def test_paired_marker_without_bn_marker_fails_closed():
    layers, preds, succs = _bn_chain()
    layers[5].params["paired_with_scale"] = True

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert not certificate.accepted
    assert "paired_marker_without_bn_marker" in _issue_codes(certificate)


def test_non_bn_mismatch_blocks_bn_only_repair():
    layers, preds, _ = _bn_chain(sibling=True)
    preds[5] = [3]
    succs = _successors(preds)

    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert not plan.authorized
    assert plan.reason.startswith("repair_rejected:")


def test_asymmetric_source_graph_is_not_silently_normalized():
    layers, preds, succs = _bn_chain(sibling=True)
    succs[2] = [3]

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    assert "predecessor_successor_asymmetry" in _issue_codes(certificate)
    assert not plan.authorized
    assert plan.reason == "source_graph_edges_are_asymmetric"


def test_two_bn_sibling_pairs_are_repaired_all_or_nothing_in_layer_order():
    layers, preds, _ = _bn_chain(sibling=True)
    layers.extend(
        [
            _layer(
                6,
                "SCALE",
                (8, 9),
                (10, 11),
                params={
                    "a": [0.75, 1.5],
                    "is_batchnorm_decomposition": True,
                },
            ),
            _layer(
                7,
                "BIAS",
                (10, 11),
                (12, 13),
                params={
                    "c": [0.25, -0.75],
                    "is_batchnorm_decomposition": True,
                    "paired_with_scale": True,
                },
            ),
            _layer(8, "RELU", (12, 13), (14, 15)),
        ]
    )
    preds = {
        0: [], 1: [0], 2: [1], 3: [2], 4: [2], 5: [4],
        6: [5], 7: [5], 8: [7],
    }
    succs = _successors(preds)

    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)
    clone, certificate = graph_cert.apply_repair_plan_to_clone(
        layers, preds, succs, plan
    )

    assert plan.replacements == (
        graph_cert.EdgeReplacement(4, (2,), (3,)),
        graph_cert.EdgeReplacement(7, (5,), (6,)),
    )
    assert clone.predecessor_dict()[4] == [3]
    assert clone.predecessor_dict()[7] == [6]
    assert certificate.accepted


def test_ordered_duplicate_multi_operand_predecessors_are_preserved():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(
            1,
            "ADD",
            (0, 0),
            (1,),
            params={"x_vars": [0], "y_vars": [0]},
        ),
    ]
    preds = {0: [], 1: [0, 0]}
    succs = {0: [1], 1: []}

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert certificate.accepted
    assert certificate.edge_count == 2


def test_one_multi_operand_spanning_two_producers_is_explicitly_ambiguous():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "CONSTANT", (), (1,)),
        _layer(
            2,
            "ADD",
            (0, 1, 1),
            (2,),
            params={"x_vars": [0, 1], "y_vars": [1]},
        ),
    ]
    # This matches the old expanded producer list exactly; it must still be
    # rejected because one logical x operand has no unique graph occurrence.
    preds = {0: [], 1: [], 2: [0, 1, 1]}
    succs = {0: [2], 1: [2], 2: []}

    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)

    assert not certificate.accepted
    assert _issue_codes(certificate) == ("multi_operand_producer_ambiguity",)
    issue = certificate.issues[0]
    assert issue.expected == (1, 1)
    assert issue.observed == (2, 1)
    assert issue.related_layer_ids == (0, 1)


def test_latest_aliasing_producer_is_the_authoritative_event():
    layers = [
        _layer(0, "INPUT", (), (0, 1)),
        _layer(1, "INPUT_SPEC", (0, 1), (0, 1)),
        _layer(2, "DENSE", (0, 1), (2,)),
    ]
    faithful = graph_cert.audit_graph_faithfulness(
        layers,
        {0: [], 1: [0], 2: [1]},
        {0: [1], 1: [2], 2: []},
    )
    stale = graph_cert.audit_graph_faithfulness(
        layers,
        {0: [], 1: [0], 2: [0]},
        {0: [1, 2], 1: [], 2: []},
    )

    assert faithful.accepted
    assert not stale.accepted
    assert stale.issues[0].expected == (1,)


def _path_observation(layer_id, semantic_kind):
    return graph_cert.PathOperatorObservation(layer_id, semantic_kind)


def _path_issue_codes(certificate):
    return tuple(issue.code for issue in certificate.issues)


def test_path_operator_lineage_accounts_for_scale_and_records_bias_transition():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(2, "implicit_conv2d"),
            _path_observation(3, "diagonal_scale"),
        ],
    )

    assert certificate.accepted
    assert certificate.bias_transition_layer_ids == (4,)
    assert certificate.expected_operators == certificate.observed_operators
    assert certificate.gain == 0
    assert certificate.default_enabled is False


def test_path_lineage_missing_scale_is_unaccounted_linear_event():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [_path_observation(2, "implicit_conv2d")],
    )

    assert not certificate.accepted
    assert _path_issue_codes(certificate) == ("unaccounted_linear_event",)
    issue = certificate.issues[0]
    assert issue.occurrence_layer_id == 3
    assert issue.expected_semantic_kind == "diagonal_scale"


def test_bias_cannot_masquerade_as_multiplicative_lineage_operator():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(2, "implicit_conv2d"),
            _path_observation(3, "diagonal_scale"),
            _path_observation(4, "diagonal_bias"),
        ],
    )

    assert not certificate.accepted
    assert "extra_lineage_operator" in _path_issue_codes(certificate)


def test_wrong_operator_semantics_are_rejected_at_same_occurrence():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(2, "implicit_conv2d"),
            _path_observation(3, "implicit_conv2d"),
        ],
    )

    assert _path_issue_codes(certificate) == (
        "operator_semantic_kind_mismatch",
    )


def test_operator_order_mismatch_is_rejected_even_when_set_is_complete():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(3, "diagonal_scale"),
            _path_observation(2, "implicit_conv2d"),
        ],
    )

    assert _path_issue_codes(certificate) == (
        "lineage_operator_order_mismatch",
    )


def test_duplicate_operator_occurrence_is_rejected():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(2, "implicit_conv2d"),
            _path_observation(3, "diagonal_scale"),
            _path_observation(3, "diagonal_scale"),
        ],
    )

    assert "duplicate_lineage_operator_occurrence" in _path_issue_codes(certificate)


def test_path_cannot_skip_the_scale_graph_event():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 4, 5],
        [_path_observation(2, "implicit_conv2d")],
    )

    assert not certificate.accepted
    assert "path_skips_graph_event" in _path_issue_codes(certificate)


def test_path_lineage_never_licenses_an_unfaithful_source_graph():
    layers, preds, succs = _bn_chain(sibling=True)
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 4, 5],
        [_path_observation(2, "implicit_conv2d")],
    )

    assert not certificate.accepted
    assert _path_issue_codes(certificate) == ("source_graph_not_faithful",)


def test_unregistered_path_event_is_rejected_instead_of_silently_ignored():
    layers = [
        _layer(0, "INPUT", (), (0,)),
        _layer(1, "AVGPOOL2D", (0,), (1,)),
        _layer(2, "RELU", (1,), (2,)),
    ]
    preds = {0: [], 1: [0], 2: [1]}
    succs = _successors(preds)

    certificate = graph_cert.audit_path_operator_lineage(
        layers, preds, succs, [0, 1, 2], []
    )

    assert not certificate.accepted
    assert _path_issue_codes(certificate) == ("unsupported_path_event_kind",)


def test_path_lineage_json_explicitly_disclaims_runtime_adapter():
    layers, preds, succs = _bn_chain()
    certificate = graph_cert.audit_path_operator_lineage(
        layers,
        preds,
        succs,
        [2, 3, 4, 5],
        [
            _path_observation(2, "implicit_conv2d"),
            _path_observation(3, "diagonal_scale"),
        ],
    )

    payload = graph_cert.path_operator_certificate_to_jsonable(certificate)

    assert payload["accepted"] is True
    assert payload["runtime_lineage_adapter_proven"] is False
    assert payload["gain"] == 0


def test_transaction_commits_only_after_exact_true_validator():
    layers, preds, succs = _bn_chain(sibling=True)

    accepted = graph_cert.execute_clone_only_repair_transaction(
        layers,
        preds,
        succs,
        candidate_validator=lambda clone, certificate: (
            certificate.accepted and clone.predecessor_dict()[4] == [3]
        ),
    )
    rejected = graph_cert.execute_clone_only_repair_transaction(
        layers,
        preds,
        succs,
        candidate_validator=lambda _clone, _certificate: 1,
    )

    assert accepted.committed
    assert accepted.clone is not None
    assert not rejected.committed
    assert rejected.reason == "candidate_validator_rejected"
    assert rejected.clone is None


def test_ordinary_exception_discards_staging_and_preserves_callers():
    layers, preds, succs = _bn_chain(sibling=True)
    before = deepcopy((layers, preds, succs))

    def fail(_clone, _certificate):
        raise RuntimeError("ordinary validator failure")

    result = graph_cert.execute_clone_only_repair_transaction(
        layers, preds, succs, candidate_validator=fail
    )

    assert not result.committed
    assert result.reason == "transaction_rejected:RuntimeError"
    assert result.clone is None
    assert (layers, preds, succs) == before


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(7)])
def test_baseexception_propagates_after_private_staging_without_publication(exc):
    layers, preds, succs = _bn_chain(sibling=True)
    before = deepcopy((layers, preds, succs))

    def interrupt(_clone, _certificate):
        raise exc

    with pytest.raises(type(exc)):
        graph_cert.execute_clone_only_repair_transaction(
            layers, preds, succs, candidate_validator=interrupt
        )
    assert (layers, preds, succs) == before


def test_baseexception_from_input_snapshot_is_not_downgraded_to_rejection():
    class InterruptingLayer:
        @property
        def id(self):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        graph_cert.audit_graph_faithfulness(
            [InterruptingLayer()], {0: []}, {0: []}
        )


def test_json_records_are_stable_address_free_and_explicitly_no_gain():
    layers, preds, succs = _bn_chain(sibling=True)
    certificate = graph_cert.audit_graph_faithfulness(layers, preds, succs)
    plan = graph_cert.plan_batchnorm_graph_repair(layers, preds, succs)

    encoded_certificate = json.dumps(
        graph_cert.certificate_to_jsonable(certificate), sort_keys=True
    )
    encoded_plan = json.dumps(
        graph_cert.repair_plan_to_jsonable(plan), sort_keys=True
    )

    assert "0x" not in encoded_certificate + encoded_plan
    assert '"gain": 0' in encoded_certificate
    assert '"default_enabled": false' in encoded_plan
    assert "full_13_family_2413_replay_is_required_before_enablement" in encoded_plan


def test_prototype_has_no_production_import_or_mutation_interface():
    source = inspect.getsource(graph_cert)

    assert "from act." not in source
    assert "import act." not in source
    assert "1870/2413" in source
    assert not hasattr(graph_cert, "enable")
    assert not hasattr(graph_cert, "patch_net")
