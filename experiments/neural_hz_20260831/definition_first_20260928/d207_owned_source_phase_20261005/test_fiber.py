"""Frozen supplied-structure checks, not model verification or an attack."""
from fractions import Fraction as Q
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d207_owned_source_phase_20261005.fiber import (
    Fiber, Form, Predicate, MAX_WORK, MAX_BRANCH_WORK, MAX_ENTRIES,
)


POINTS = ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (Q(1, 3), -Q(1, 2)))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-2, 3))


def _control(aliases=False):
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    pre = root.affine(root.sources(), ((1, 3), (3, 4)), (Q(1, 4), Q(1, 4)))
    scale = root.affine(root.sources(), ((0, Q(1, 4)),), (Q(3, 4),))
    parent = root
    if aliases:
        parent, pre = root.alias(pre, ("pre0", "pre1"))
    fiber, output = parent.relu_bank(pre, ("r0", "r1"))
    return root, fiber, output, pre, scale


def _successor_form(fiber, output, pre, scale):
    joined = fiber.concat(output, fiber.select(pre, (0,)), scale)
    return fiber.affine(joined, ((Q(73, 16), -Q(15, 8), -Q(13, 16), -Q(195, 32)),),
                        (-Q(1, 16),))


def _indices(fiber, kind):
    return tuple(i for i, factor in enumerate(fiber.factors) if factor.kind == kind)


def _row_holds(row, point):
    value = row.form.at(point)
    return value == row.rhs if row.relation == "eq" else value <= row.rhs


def _record(name, value):
    run = Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"])
    expected = Path(__file__).resolve().parents[2] / "results/d207_owned_source_phase_20261005_v1"
    assert run == expected and run.is_dir() and not run.is_symlink()
    with (run / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def test_default_off_and_rational_inputs():
    with pytest.raises(ValueError):
        Fiber.box(((-1, 1),))
    for value in (0.5, True, Q(1, 1 << 513)):
        with pytest.raises(ValueError):
            Form(value)
    root = Fiber.box(((-1, 1),), enabled=True)
    assert root.bounds(root.sources()) == (-1, 1)
    with pytest.raises(ValueError):
        Fiber(root._frame, root.factors, root.predicates)


def test_hz_embedding_predicates_and_decoder():
    equality = Predicate(Form(0, ((0, Q(1)), (2, Q(-1)))), "eq", 0)
    inequality = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
                     predicates=(equality, inequality), decoder_matrix=((2, 0), (0, 3)),
                     decoder_bias=(1, -1), enabled=True)
    point = root.complete((1, 0), (1,))
    assert root.contains(point) and root.decode(point) == (3, -1)
    assert root.evaluate(root.signed_phases(), point) == (1,)
    assert not root.contains((0, 0, 1))
    assert root.predicates[0] is equality and root.predicates[1] is inequality


def test_shared_affine_add_concat():
    root, fiber, output, pre, scale = _control()
    joined = fiber.concat(output, pre, scale, root.sources())
    negative = fiber.affine(joined, tuple(tuple(-int(i == j) for j in range(7)) for i in range(7)),
                            (0,) * 7)
    cancelled = fiber.add(joined, negative)
    assert fiber.support(cancelled, (1,) * 7) == 0
    assert fiber.evaluate(cancelled, fiber.complete((0, 0))) == (0,) * 7
    assert len(fiber.factors) == 6


def test_foreign_and_sibling_views_rejected():
    root = Fiber.box(((-1, 1),), enabled=True)
    foreign = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        root.add(root.sources(), foreign.sources())
    left, lview = root.alias(root.sources(), ("left",))
    right, rview = root.alias(root.sources(), ("right",))
    with pytest.raises(ValueError):
        left.add(lview, rview)
    with pytest.raises(ValueError):
        right.support(lview, (1,))
    assert left.bounds(root.sources()) == (-1, 1)


def test_alias_identity_and_defining_equation():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    pre = root.affine(root.sources(), ((1, 3),), (Q(1, 4),))
    child, alias = root.alias(pre, ("keep_alias",))
    assert child.factors[:2] == root.factors
    assert child.factors[2].kind == "alias" and child.factors[2].producer == pre.forms[0]
    assert child.predicates[-1].relation == "eq"
    point = child.complete((0, 0))
    assert child.evaluate(alias, point) == (Q(1, 4),)
    invalid = list(point)
    invalid[2] += Q(1, 100)
    assert not child.contains(invalid)
    assert child.bounds(alias) == root.bounds(pre)


def test_nested_alias_canonical_equivalence():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    original = root.affine(root.sources(), ((1, 3), (3, 4)), (Q(1, 4), Q(1, 4)))
    first, av = root.alias(original, ("a0", "a1"))
    second, bv = first.alias(av, ("b0", "b1"))
    for direction in DIRECTIONS:
        assert second.support(bv, direction) == root.support(original, direction)
    assert len(_indices(second, "alias")) == 4
    state, q = second.relu_bank(bv, ("g0", "g1"))
    assert state.banks[-1].pairs[0].scale == Form(Q(3, 4), ((1, Q(1, 4)),))
    assert state.evaluate(q, state.complete((1, -1))) == (0, 0)


def test_canonical_shared_scale_positive_control():
    _, fiber, _, _, _ = _control()
    pair = fiber.banks[0].pairs[0]
    assert pair.caps == (Q(13, 8), Q(41, 8))
    assert pair.scale == Form(Q(3, 4), ((1, Q(1, 4)),))
    assert pair.lower == Q(1, 2) and len(pair.rows) == 10
    assert pair.source_m[0] * pair.source_m[1] < Q(1, 4)
    assert len(fiber.predicates) == 18


def test_zero_common_deficit_negative_control():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    pre = root.affine(root.sources(), ((Q(2, 3), 0), (-Q(2, 3), -2)), (Q(1, 2), Q(1, 2)))
    fiber, output = root.relu_bank(pre, ("n0", "n1"))
    pair = fiber.banks[0].pairs[0]
    assert pair.caps == (Q(9, 4), Q(13, 4))
    assert pair.scale == Form(1) and pair.lower == 1
    assert pair.source_h == pair.constant_h and pair.source_m == pair.constant_m
    assert fiber.support_certificates(output, (1, -1))[1:] == (
        fiber.support_certificates(output, (1, -1))[1],) * 2


def test_all_original_relu_graphs_and_zero_labels():
    root = Fiber.box(((-1, 1),), enabled=True)
    fiber, output = root.relu(root.sources(), "g")
    assert len(fiber.predicates) == 4 and len(_indices(fiber, "phase")) == 1
    for x in (-1, 0, 1):
        point = fiber.complete((x,))
        assert fiber.evaluate(output, point) == (max(0, x),)
        assert all(_row_holds(row, point) for row in fiber.predicates)
    zero = list(fiber.complete((0,)))
    zero[_indices(fiber, "phase")[0]] = 1
    assert fiber.contains(zero)
    positive = list(fiber.complete((1,)))
    positive[_indices(fiber, "phase")[0]] = -1
    assert not fiber.contains(positive)


def test_source_phase_row_excludes_strong_fractional_control():
    _, fiber, _, _, _ = _control()
    point = list(fiber.complete((0, -Q(1, 3))))
    for index in _indices(fiber, "phase"):
        point[index] = 0
    for index, value in zip(_indices(fiber, "relu"), (Q(41, 40), Q(23, 40))):
        point[index] = value
    pair = fiber.banks[0].pairs[0]
    assert all(_row_holds(row, point) for row in fiber.predicates[:8])
    violations = tuple(row.form.at(point) - row.rhs for row in pair.rows)
    assert max(violations) == Q(29, 480)
    assert not fiber.contains(point)


def test_intrinsic_query_stabilizes_successor():
    _, fiber, output, pre, scale = _control()
    next_pre = _successor_form(fiber, output, pre, scale)
    certificates = fiber.support_certificates(next_pre, (1,))
    assert certificates[2] == -Q(1, 16)
    assert certificates[0] > 0 and certificates[1] > 0
    child, value = fiber.relu(next_pre, "child")
    assert type(child) is Fiber and child.bounds(value) == (0, 0)
    assert child.banks[-1].gates[0].stable == -1
    for point in POINTS:
        assert child.evaluate(value, child.complete(point)) == (0,)
    _record("source_phase_positive_control.json", {
        "preactivation_upper_certificates": [str(v) for v in certificates],
        "successor_bounds": [str(v) for v in child.bounds(value)],
        "cost_scope": "complete supplied-control exercise including repeated checks",
        "cost": child.cost_report(), "actual_model": False, "formal_gain": 0})


def test_exact_alias_preserves_successor_proof():
    _, direct, q0, f0, s0 = _control()
    _, alias, q1, f1, s1 = _control(aliases=True)
    p0, p1 = _successor_form(direct, q0, f0, s0), _successor_form(alias, q1, f1, s1)
    assert direct.support_certificates(p0, (1,)) == alias.support_certificates(p1, (1,))
    final, output = alias.relu(p1, "alias_child")
    assert final.bounds(output) == (0, 0)
    assert len(_indices(final, "alias")) == 2
    assert len(_indices(final, "phase")) == 3
    assert all(final.evaluate(output, final.complete(point)) == (0,) for point in POINTS)
    _record("exact_alias_control.json", {
        "direct": [str(v) for v in direct.support_certificates(p0, (1,))],
        "with_aliases": [str(v) for v in alias.support_certificates(p1, (1,))],
        "retained_aliases": 2, "retained_bits_with_successor": 3,
        "source_phase_proof_preserved": True, "actual_model": False, "formal_gain": 0})


def test_alias_after_relu_and_recursive_second_bank():
    root, first, output, pre, scale = _control()
    aliased, same_q = first.alias(output, ("q_alias0", "q_alias1"))
    next_pre = _successor_form(aliased, same_q, pre, scale)
    child, zero = aliased.relu(next_pre, "after_q_alias")
    assert child.bounds(zero) == (0, 0)
    mixed = child.affine(child.concat(same_q, root.sources()),
                         ((1, Q(1, 2), -1, 0), (Q(1, 2), -1, 0, 1)),
                         (-Q(1, 4), Q(1, 8)))
    final, second = child.relu_bank(mixed, ("second0", "second1"))
    assert len(final.banks) == 3 and final.cost_report()["aliases"] == 2
    for point in POINTS:
        assignment = final.complete(point)
        actual = final.evaluate(second, assignment)
        expected = tuple(max(0, v) for v in final.evaluate(mixed, assignment))
        assert actual == expected
        for direction in DIRECTIONS:
            truth = sum(a * b for a, b in zip(direction, actual))
            assert all(bound >= truth for bound in final.support_certificates(second, direction))


def test_complete_two_hundred_gate_bank():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    matrix = tuple((1, 3) if i % 2 == 0 else (3, 4) for i in range(200))
    pre = root.affine(root.sources(), matrix, (Q(1, 4),) * 200)
    fiber, output = root.relu_bank(pre, tuple("g%d" % i for i in range(200)))
    report = fiber.cost_report()
    assert report["binary"] == 200 and report["continuous"] == 202
    assert report["banks"] == 1 and report["pairs"] == 100
    assert report["predicates"] == 1800 and len(output.forms) == 200
    assert len({id(f.identity) for f in fiber.factors}) == 402
    assignment = fiber.complete((0, 0))
    assert len(assignment) == 402 and fiber.contains(assignment)
    assert fiber.evaluate(output, assignment) == (Q(1, 4),) * 200
    assert fiber.support(output, (1,) * 200) >= 50
    assert fiber.cost_report()["binary"] == 200
    _record("complete_200_gate_control.json", {
        "supplied_structure": "100 repetitions of the fixed ordinary two-source pair",
        "cost_scope": "complete supplied-bank exercise including membership and query",
        "cost": fiber.cost_report(), "actual_model": False, "formal_gain": 0})


def test_odd_bank_keeps_unpaired_gate():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    pre = root.affine(root.sources(), ((1, 3), (3, 4), (1, -1)), (Q(1, 4), Q(1, 4), 0))
    fiber, output = root.relu_bank(pre, ("a", "b", "tail"))
    assert len(fiber.banks[0].pairs) == 1 and len(fiber.banks[0].gates) == 3
    assert len(fiber.predicates) == 22
    assert fiber.evaluate(output, fiber.complete((1, -1))) == (0, 0, 2)
    assert fiber.support(fiber.select(output, (2,)), (1,)) == 2


def test_stable_gates_keep_original_bits():
    root = Fiber.box(((-1, 1),), enabled=True)
    pre = root.affine(root.sources(), ((0,), (0,), (0,)), (1, -1, 0))
    fiber, output = root.relu_bank(pre, ("positive", "negative", "zero"))
    assert len(_indices(fiber, "phase")) == 3 and len(fiber.predicates) == 12
    assignment = list(fiber.complete((0,)))
    assert fiber.evaluate(output, assignment) == (1, 0, 0)
    assignment[fiber.banks[0].gates[2].phase_index] = 1
    assert fiber.contains(assignment)
    assignment[fiber.banks[0].gates[0].phase_index] = -1
    assert not fiber.contains(assignment)


def test_initial_binary_and_mixed_predicates_survive():
    predicate = Predicate(Form(0, ((0, Q(1)), (2, Q(-1)))), "eq", 0)
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("old",), predicates=(predicate,), enabled=True)
    mixed = root.concat(root.sources(), root.signed_phases())
    pre = root.affine(mixed, ((1, 3, Q(1, 4)), (3, 4, -Q(1, 4))), (Q(1, 4), Q(1, 4)))
    fiber, output = root.relu_bank(pre, ("new0", "new1"))
    assert fiber.factors[:3] == root.factors and fiber.predicates[0] is predicate
    for source, phases in (((1, 0), (1,)), ((-1, 0), (-1,))):
        assignment = fiber.complete(source, phases)
        assert fiber.contains(assignment) and fiber.decode(assignment) == source
        values = fiber.evaluate(output, assignment)
        assert all(b >= sum(values) for b in fiber.support_certificates(output, (1, 1)))


def test_fixed_queries_are_sound_on_declared_points():
    _, fiber, output, _, _ = _control(aliases=True)
    bounds = tuple(fiber.support_certificates(output, direction) for direction in DIRECTIONS)
    for point in POINTS:
        values = fiber.evaluate(output, fiber.complete(point))
        for direction, certificates in zip(DIRECTIONS, bounds):
            actual = sum(a * b for a, b in zip(direction, values))
            assert all(bound >= actual for bound in certificates)


def test_parent_predicates_are_never_dropped():
    predicate = Predicate(Form(0, ((0, Q(1)),)), "le", 0)
    root = Fiber.box(((-1, 1),), predicates=(predicate,), enabled=True)
    aliased, av = root.alias(root.sources(), ("alias",))
    fiber, q = aliased.relu(av, "gate")
    assert fiber.predicates[0] is predicate
    assert fiber.contains(fiber.complete((-1,)))
    invalid = (Q(1), Q(1), Q(1), Q(1))
    assert not fiber.contains(invalid)
    with pytest.raises(ValueError):
        fiber.evaluate(q, invalid)
    with pytest.raises(ValueError):
        fiber.decode(invalid)


def test_fixed_three_certificates_and_shared_work():
    root, fiber, output, _, _ = _control()
    before = root.cost_report()["work_used"]
    certificates = fiber.support_certificates(output, (1, -1))
    after = root.cost_report()["work_used"]
    assert len(certificates) == 3 and all(isinstance(v, Q) for v in certificates)
    assert after > before and fiber._frame.work is root._frame.work
    assert fiber.support(output, (1, -1)) == min(certificates)
    child, view = fiber.alias(output, ("a0", "a1"))
    assert child._frame.work is root._frame.work
    assert child.support_certificates(view, (1, -1)) == certificates


def test_reject_fractional_bits_and_false_alias_state():
    _, fiber, output, _, _ = _control(aliases=True)
    valid = fiber.complete((0, 0))
    bad_bit = list(valid)
    bad_bit[_indices(fiber, "phase")[0]] = 0
    assert not fiber.contains(bad_bit)
    bad_alias = list(valid)
    bad_alias[_indices(fiber, "alias")[0]] += Q(1, 100)
    assert not fiber.contains(bad_alias)
    bad_output = list(valid)
    bad_output[_indices(fiber, "relu")[0]] = Q(1, 5)
    assert not fiber.contains(bad_output)
    with pytest.raises(ValueError):
        fiber.evaluate(output, bad_alias)


def test_nonpositive_pair_caps_keep_exact_graph():
    root = Fiber.box(((-1, 1),), enabled=True)
    pre = root.affine(root.sources(), ((Q(1, 2),), (1,)), (-Q(1, 10), 0))
    fiber, output = root.relu_bank(pre, ("a", "b"))
    assert not fiber.banks[0].pairs and len(fiber.predicates) == 8
    assert all(g.stable == 0 for g in fiber.banks[0].gates)
    for x in (-1, 0, 1):
        point = fiber.complete((x,))
        assert fiber.evaluate(output, point) == (max(0, Q(x, 2) - Q(1, 10)), max(0, x))
    certificates = fiber.support_certificates(output, (1, -1))
    assert certificates[0] == certificates[1] == certificates[2]


def test_unsupported_inputs_and_resource_limits_fail_closed():
    for kwargs in (dict(max_work=MAX_WORK + 1), dict(max_branch_work=MAX_BRANCH_WORK + 1),
                   dict(max_entries=MAX_ENTRIES + 1), dict(max_entries=1), dict(max_work=1)):
        with pytest.raises(ValueError):
            Fiber.box(((-1, 1),), enabled=True, **kwargs)
    for terms in (((-1, Q(1)),), ((0, Q(1)), (0, Q(1))), ((0, Q(0)),)):
        with pytest.raises(ValueError):
            Form(0, terms)
    root = Fiber.box(((-1, 1),), enabled=True, max_work=200, max_branch_work=200)
    source = root.sources()
    with pytest.raises(ValueError):
        root.support(source, (1, 2))
    with pytest.raises(ValueError):
        root.affine(source, ((1, 0),), (0,))
    with pytest.raises(ValueError):
        for _ in range(100):
            root.support(source, (1,))
    with pytest.raises(ValueError):
        root.support(source, (0,))


def test_complete_logical_cost_and_readout_accounting():
    root, fiber, output, _, _ = _control(aliases=True)
    before = fiber.cost_report()
    assert before["factors"] == 8 and before["binary"] == 2 and before["aliases"] == 2
    assert before["predicates"] == 20
    assert before["predicate_nnz"] == sum(len(p.form.terms) for p in fiber.predicates)
    assert before["entries"] >= before["predicate_nnz"] + before["factors"]
    fiber.concat(output, root.sources())
    after = fiber.cost_report()
    assert after["readouts"] > before["readouts"] and after["entries"] > before["entries"]
    assert after["work_used"] > before["work_used"]
    assert after["branch_work_used"] == after["work_used"]
    assert not after["physical_qualified"] and not after["native_qualified"]
    assert after["formal_gain"] == 0
