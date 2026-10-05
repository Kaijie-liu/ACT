"""Sixteen bounded branch-accounting tests; no new row or kernel algorithm.

Scalar fixtures independently enumerate every source-to-node path. One tiny
complete nonconvex source checks all native/source/graph/inverse arrays against
the C128 control. The root worker retains the four unchanged full-size cases.
"""
import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.neural_hz_20260831 import c130_nodewise_branch_v1 as branch
from experiments.neural_hz_20260831 import c130_birth_emission_v1 as birth
from experiments.neural_hz_20260831.c128_birth_emission_v1 import lift as control_lift
from experiments.neural_hz_20260831.c129_complete_source_v1 import packet, source_arrays
from experiments.neural_hz_20260831.c130_complete_source_v1 import normalize_metadata
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct, SCHEMA
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression


def _fixture():
    nodes = [dict(kind="source", width=1, parents=[]),
             dict(kind="source", width=1, parents=[]),
             dict(kind="sum", width=1, parents=[0, 1])]
    counts = []
    for node, continuous, support in zip(nodes, (100, 0, 2), (0, 1000, 0), strict=True):
        counts.append(dict(kind=node["kind"], width=1, auxiliaries=1,
            continuous_edges=continuous, binary_edges=0, center_edges=0,
            support_work=support, encoding_work_upper=16*(continuous+1)))
    return nodes, counts


def _plan():
    nodes, counts = _fixture()
    return branch.plan(nodes, counts, 3, 1, enabled=True)


def _all_paths(nodes):
    paths = []
    for index, node in enumerate(nodes):
        paths.append([(index,)] if not node["parents"] else
                     [(*path, index) for parent in node["parents"] for path in paths[parent]])
    return [path for group in paths for path in group]


def _counter(rows=0, coefficients=0, power=None, omitted=None):
    return dict(logical_rows=rows, logical_coefficients=coefficients,
        once_checked_logical_power_elements=coefficients if power is None else power,
        omitted_post_magnitude_elements=coefficients if omitted is None else omitted)


def _advance(before, rows, coefficients, extra=0):
    return {key: before[key]+amount for key, amount in zip(branch.COUNTERS,
        (rows, coefficients, coefficients, coefficients+extra), strict=True)}


def _prepared(last):
    return dict(logical_input_rows=last["logical_rows"],
        logical_input_coefficients=last["logical_coefficients"],
        once_checked_logical_power_elements=last["once_checked_logical_power_elements"],
        actual_omitted_post_magnitude_coefficients=last["omitted_post_magnitude_elements"],
        omitted_post_finite_coefficient_checks=last["omitted_post_magnitude_elements"],
        duplicate_comparison_credit_per_logical_coefficient=2,
        original_input_finite_and_complete_inverse_checks_retained=True,
        generic_RHS_finite_check_retained=True,
        original_power_bounds_checked_before_signed_copy=True,
        generic_emit_radix_power_validation_retained=True)


def _report(extra=0):
    cert = _plan()
    previous = _counter(1, 2, omitted=2+extra)
    branch.observe_predicates(cert, _counter(), previous)
    for index, node in enumerate(cert["nodes"]):
        after = _advance(previous, node["planned_rows"], node["planned_coefficients"], extra)
        branch.observe_node(cert, index, previous, after)
        previous = after
    prepared = _prepared(previous)
    branch.finish(cert, prepared)
    counts = [dict(kind=node["kind"], **{key: node[key] for key in branch.COUNT_FIELDS},
                   encoding_work_upper=node["encoding_work_upper"]) for node in cert["nodes"]]
    whole = cert["whole_base_without_proof_fee"]+cert["proof_work"]
    path = cert["new_branch_base_without_proof_fee"]+cert["proof_work"]
    return dict(nodewise_branch_certificate=cert, node_counts=counts, prepared_encoding=prepared,
        whole_base_work=whole, branch_base_work=path, total_work_upper=whole+7,
        largest_branch_work_upper=path+7, alias_quotient=dict(coupled_extra_work=7),
        original_branch_encoding_price_retained=False, nodewise_original_emission_credit_proved=True,
        old_predicate_work_upper=cert["old_predicate_whole_work"],
        support_work=sum(node["support_work"] for node in cert["nodes"]),
        affine_work_upper=sum(node["encoding_work_upper"]+node["support_work"] for node in cert["nodes"]))


def _first_observation():
    cert = _plan()
    before = _counter(1, 2)
    branch.observe_predicates(cert, _counter(), before)
    return cert, before, _advance(before, 1, 101)


def test_default_off_does_not_read_payload_and_nonboolean_enable_is_rejected():
    assert branch.plan(None, None, None, None) is None
    assert birth.lift(None, None) is None
    for enabled in (1, 0, None, "true", np.bool_(True)):
        with pytest.raises(ValueError):
            branch.plan(None, None, None, None, enabled=enabled)
        with pytest.raises(ValueError):
            birth.lift(None, None, enabled=enabled)


def test_all_parent_paths_change_maximizer_and_global_subtraction_is_wrong():
    nodes, counts = _fixture()
    cert = _plan()
    paths = _all_paths(nodes)
    old_local = [count["encoding_work_upper"]//16*12+count["support_work"] for count in counts]
    credit = [4*(count["continuous_edges"]+count["binary_edges"]+count["auxiliaries"]) for count in counts]
    new_local = [old-cut for old, cut in zip(old_local, credit, strict=True)]
    old_path = max(paths, key=lambda path: sum(old_local[index] for index in path))
    new_path = max(paths, key=lambda path: sum(new_local[index] for index in path))
    assert old_path == (0, 2) and new_path == (1, 2)
    assert cert["old_maximizing_path"] == list(old_path)
    assert cert["new_maximizing_path"] == list(new_path)
    assert cert["old_branch_base_without_proof_fee"] == 1248+36
    assert cert["new_branch_base_without_proof_fee"] == 1032+36
    assert 1248-sum(credit) == 828 < 1032
    assert cert["old_predicate_branch_work"] == 36
    assert cert["old_predicate_whole_work"] == 28


def test_every_parent_incidence_is_paid_including_repeated_parents_before_reads():
    nodes, counts = _fixture()
    nodes[2]["parents"] = [0, 1, 0, 1]
    cert = branch.plan(nodes, counts, 3, 1, enabled=True)
    assert cert["parent_entries"] == 4
    assert cert["proof_work"] == 2048+512*3+64*4
    assert cert["nodes"][2]["parents"] == [0, 1, 0, 1]
    assert cert["new_branch_base_without_proof_fee"] == 1068
    # An invalid parent VALUE must not be inspected when header payment fails.
    nodes[2]["parents"][0] = object()
    with pytest.raises(MemoryError, match="prepayment"):
        branch.plan(nodes, counts, 3, 1, enabled=True, max_work=cert["proof_work"]-1)


def test_whole_and_branch_caps_both_include_the_complete_proof_fee():
    nodes, counts = _fixture()
    cert = _plan()
    whole = cert["whole_base_without_proof_fee"]+cert["proof_work"]
    path = cert["new_branch_base_without_proof_fee"]+cert["proof_work"]
    exact = branch.plan(nodes, counts, 3, 1, enabled=True, max_work=whole, max_branch_work=path)
    assert exact == cert
    for kwargs in (dict(max_work=whole-1), dict(max_branch_work=path-1)):
        with pytest.raises(MemoryError):
            branch.plan(nodes, counts, 3, 1, enabled=True, **kwargs)


def test_noninteger_negative_and_increased_domains_are_not_coerced():
    for value in (-1, True, 1.5, np.int64(1)):
        nodes, counts = _fixture()
        counts[0]["continuous_edges"] = value
        with pytest.raises(ValueError):
            branch.plan(nodes, counts, 3, 1, enabled=True)
    nodes, counts = _fixture()
    for entries, rows in ((True, 0), (-1, 0), (3, 4), (3, .5)):
        with pytest.raises(ValueError):
            branch.plan(nodes, counts, entries, rows, enabled=True)
    for kwargs in (dict(max_work=256000001), dict(max_branch_work=200000001), dict(max_work=True)):
        with pytest.raises(ValueError):
            branch.plan(nodes, counts, 3, 1, enabled=True, **kwargs)


def test_topology_kind_population_and_original_sixteen_unit_counts_are_strict():
    mutations = [lambda nodes, counts: nodes[2].update(parents=[2]),
                 lambda nodes, counts: nodes[2].update(parents=[-1]),
                 lambda nodes, counts: nodes[2].update(parents=[True]),
                 lambda nodes, counts: nodes[0].update(parents=[0]),
                 lambda nodes, counts: nodes[2].update(parents=[]),
                 lambda nodes, counts: counts[0].update(encoding_work_upper=1212),
                 lambda nodes, counts: counts[0].update(auxiliaries=2),
                 lambda nodes, counts: counts[0].update(center_edges=2),
                 lambda nodes, counts: counts.pop()]
    for mutate in mutations:
        nodes, counts = _fixture()
        mutate(nodes, counts)
        with pytest.raises(ValueError):
            branch.plan(nodes, counts, 3, 1, enabled=True)


def test_planning_is_detached_and_never_mutates_original_node_count_inputs():
    nodes, counts = _fixture()
    before = copy.deepcopy((nodes, counts))
    cert = branch.plan(nodes, counts, 3, 1, enabled=True)
    assert (nodes, counts) == before
    nodes[2]["parents"][0] = 1
    counts[0]["continuous_edges"] = 9
    assert cert["nodes"][2]["parents"] == [0, 1]
    assert cert["nodes"][0]["planned_coefficients"] == 101
    assert not cert["complete_original_emission_proved"]
    assert not cert["runtime_speedup_claimed"] and cert["formal_gain"] == 0


def test_four_actual_counters_are_recorded_at_idle_boundaries_and_fully_verified():
    encoder = SimpleNamespace(pending_uid=None, in_auxiliary=False, heads_discarded=False,
                              **_counter(3, 8))
    assert branch.snapshot(encoder) == _counter(3, 8)
    for name, invalid in (("pending_uid", 0), ("in_auxiliary", True), ("heads_discarded", True)):
        changed = copy.copy(encoder)
        setattr(changed, name, invalid)
        with pytest.raises(ValueError):
            branch.snapshot(changed)
    report = _report()
    assert branch.verify_certificate(report, _fixture()[0]) is True
    cert = report["nodewise_branch_certificate"]
    assert cert["complete_original_emission_proved"] and cert["observed_node_count"] == 3
    for node, coefficients in zip(cert["nodes"], (101, 1, 3), strict=True):
        assert node["observed"] == _counter(1, coefficients)


def test_radix_extra_post_omissions_are_allowed_but_never_credited_as_logical():
    report = _report(extra=7)
    assert branch.verify_certificate(report) is True
    cert = report["nodewise_branch_certificate"]
    assert cert["new_branch_base_without_proof_fee"] == _plan()["new_branch_base_without_proof_fee"]
    for node in cert["nodes"]:
        actual = node["observed"]
        assert actual["omitted_post_magnitude_elements"] == actual["logical_coefficients"]+7
        assert actual["once_checked_logical_power_elements"] == actual["logical_coefficients"]


def test_original_power_delta_must_be_exact_not_only_a_global_lower_bound():
    for offset in (-1, 1):
        cert, before, after = _first_observation()
        after["once_checked_logical_power_elements"] += offset
        with pytest.raises(ValueError, match="local credit"):
            branch.observe_node(cert, 0, before, after)
        assert cert["observed_node_count"] == 0


def test_original_logical_row_delta_matches_each_original_node_population():
    cert, before, after = _first_observation()
    after["logical_rows"] += 1
    with pytest.raises(ValueError, match="local credit"):
        branch.observe_node(cert, 0, before, after)
    with pytest.raises(ValueError):
        branch.observe_node(cert, 1, before, after)
    with pytest.raises(ValueError):
        branch.finish(cert, _prepared(after))


def test_original_logical_coefficient_delta_cannot_borrow_other_node_work():
    cert, before, after = _first_observation()
    after["logical_coefficients"] -= 1
    with pytest.raises(ValueError, match="local credit"):
        branch.observe_node(cert, 0, before, after)
    changed = dict(before, logical_coefficients=before["logical_coefficients"]+1)
    with pytest.raises(ValueError, match="unattributed"):
        branch.observe_node(cert, 0, changed, after)


def test_post_omission_delta_cannot_be_less_than_actual_logical_coefficients():
    cert, before, after = _first_observation()
    after["omitted_post_magnitude_elements"] -= 1
    with pytest.raises(ValueError, match="local credit"):
        branch.observe_node(cert, 0, before, after)
    with pytest.raises(ValueError):
        branch.observe_predicates(_plan(), _counter(1, 1), _counter(2, 3))


def test_unchanged_global_totals_do_not_authenticate_stale_local_attribution():
    report = _report()
    original_global = copy.deepcopy(report["prepared_encoding"])
    cert = report["nodewise_branch_certificate"]
    for key in branch.COUNTERS[1:]:
        cert["nodes"][0]["after"][key] -= 1
        cert["nodes"][1]["before"][key] -= 1
    assert report["prepared_encoding"] == original_global
    assert cert["nodes"][-1]["after"] == _report()["nodewise_branch_certificate"]["nodes"][-1]["after"]
    with pytest.raises(ValueError, match="local credit"):
        branch.verify_certificate(report)


def test_saved_certificate_rechecks_paths_fees_coupled_work_and_original_guards():
    original = _report()
    mutations = [lambda r: r["nodewise_branch_certificate"].update(proof_work=0),
                 lambda r: r["nodewise_branch_certificate"].update(new_maximizing_path=[0, 2]),
                 lambda r: r["nodewise_branch_certificate"].update(parent_entries=1),
                 lambda r: r["nodewise_branch_certificate"]["nodes"][0].update(new_path_work=0),
                 lambda r: r.update(total_work_upper=r["whole_base_work"]),
                 lambda r: r.update(largest_branch_work_upper=r["branch_base_work"]),
                 lambda r: r["prepared_encoding"].update(original_input_finite_and_complete_inverse_checks_retained=False),
                 lambda r: r["prepared_encoding"].update(generic_RHS_finite_check_retained=False),
                 lambda r: r["prepared_encoding"].update(original_power_bounds_checked_before_signed_copy=False),
                 lambda r: r["prepared_encoding"].update(generic_emit_radix_power_validation_retained=False),
                 lambda r: r["prepared_encoding"].update(duplicate_comparison_credit_per_logical_coefficient=1)]
    for mutate in mutations:
        report = copy.deepcopy(original)
        mutate(report)
        with pytest.raises(ValueError):
            branch.verify_certificate(report)
    nodes, _ = _fixture()
    nodes[2]["parents"] = [1, 0]
    with pytest.raises(ValueError, match="metadata"):
        branch.verify_certificate(original, nodes)


def test_tiny_complete_nonconvex_source_all_bytes_metadata_owners_and_inverse_match_control():
    expr = expression(c=1, k=1, h=3)
    keep = np.ones(expr.n_out, dtype=bool)
    before = {name: (a.dtype, a.shape, a.tobytes(order="C")) for name, a in source_arrays(expr).items()}
    reserve = 2_000_000
    control = control_lift(expr, keep, enabled=True, max_work=reserve, max_branch_work=reserve)
    actual = birth.lift(expr, keep, enabled=True, max_work=reserve, max_branch_work=reserve)
    old_meta, old_arrays = packet(control)
    new_meta, new_arrays = packet(actual)
    assert branch.verify_certificate(new_meta["report"], new_meta["nodes"]) is True
    assert normalize_metadata(old_meta, source_reserve=reserve) == normalize_metadata(new_meta, source_reserve=reserve)
    assert set(old_arrays) == set(new_arrays)
    for name, left in old_arrays.items():
        right = new_arrays[name]
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C"), name
    points = []
    for built in (control, actual):
        fields, construction = built["fields"], built["construction"]
        hz = fields["hz"]
        assert hz.n_bin == 1 and hz.n_eq > 1 and hz.n_ineq == 1 and hz.exact
        assert np.array_equal(fields["owners"], actual_words(hz, fields["old_n_cont"],
            fields["logical_n_cont"], construction["eq_uids"], construction["ineq_uids"]))
        point = [Fraction(i % 5-2, 8) for i in range(hz.n_cont)]
        points.append(reconstruct(point, fields["eq_roots"], fields["eq_scales"],
            old_n_cont=fields["old_n_cont"], old_n_eq=fields["old_n_eq"], n_cont=hz.n_cont, schema=SCHEMA))
    assert points[0] == points[1]
    for name, a in source_arrays(expr).items():
        assert (a.dtype, a.shape, a.tobytes(order="C")) == before[name]
    for left, right in zip(control["construction"]["nodes"], actual["construction"]["nodes"], strict=True):
        assert left.get("source") is right.get("source") and left.get("op") is right.get("op")
    cert = new_meta["report"]["nodewise_branch_certificate"]
    assert new_meta["report"]["whole_base_work"] == old_meta["report"]["whole_base_work"]+cert["proof_work"]
    assert cert["old_branch_base_without_proof_fee"] == old_meta["report"]["branch_base_work"]
    assert cert["runtime_speedup_claimed"] is False and cert["formal_gain"] == 0
