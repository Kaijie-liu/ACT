"""Preregistered six-certificate domain tests; no model or native qualification."""

from dataclasses import replace
from fractions import Fraction as Q
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d214_parametric_relation_consumption_20261005.fiber import (
    Fiber, Form, Predicate, DomainError, BudgetError,
    MAX_WORK, MAX_BRANCH_WORK, MAX_ENTRIES, MAX_BITS,
)


POINTS = ((-1, 1, -1), (-1, 1, 1), (0, 0, 0),
          (Q(1, 3), -Q(1, 2), 0),
          (Q(1, 2), -Q(1, 4), -Q(1, 4)),
          (Q(9, 10), -Q(2, 5), Q(1, 24)))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-2, 3))


def _record(name, data):
    run = Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"])
    expected = (Path(__file__).resolve().parents[2]
                / "results/d214_parametric_relation_consumption_20261005_v1")
    assert run == expected and run.is_dir() and not run.is_symlink()
    with (run / name).open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _holds(row, assignment):
    value = row.form.at(assignment)
    return value == row.rhs if row.relation == "eq" else value <= row.rhs


def _form_json(form):
    return {"constant": str(form.constant),
            "terms": [[index, str(coefficient)] for index, coefficient in form.terms]}


def _pair_json(pair):
    return {
        "caps": [str(v) for v in pair.caps],
        "parent_proof_bounds": [str(p.bound) for p in pair.parent_proofs],
        "parent_le_weights": [[[list(key), str(v)] for key, v in p.le_weights]
                              for p in pair.parent_proofs],
        "common_form": _form_json(pair.common_form),
        "common_weights": [[list(key), str(v)] for key, v in pair.common_weights],
        "box_upper": str(pair.box_upper), "T_int": str(pair.T_int), "T": str(pair.T),
        "h": [_form_json(form) for form in pair.h], "m": [str(v) for v in pair.m],
        "row_indices": list(pair.row_indices),
    }


def _simple_pair(with_t=False):
    width = 3 if with_t else 2
    root = Fiber.box(((-1, 1),) * width, enabled=True)
    sources = root.sources()
    matrix = ((1, Q(1, 12)), (1, -Q(1, 12)))
    if with_t:
        matrix = tuple(row + (0,) for row in matrix)
    pre = root.affine(sources, matrix)
    state, q = root.relu_bank(pre, ("simple0", "simple1"))
    x = state.select(sources, (0,))
    f = state.affine(state.concat(q, x), ((1, -Q(13, 35), -Q(13, 35)),),
                     (-Q(13, 35),))
    child_pre = state.affine(f, ((1,),), (-Q(1, 140),))
    return root, state, sources, pre, q, f, child_pre


def _legacy_first():
    root = Fiber.box(((-1, 1),) * 3, names=("x", "y", "t"), enabled=True)
    sources = root.sources()
    pre = root.affine(sources, ((1, 3, 0), (3, 4, 0)), (Q(1, 4),) * 2)
    scale = root.affine(sources, ((0, Q(1, 4), 0),), (Q(3, 4),))
    first, q = root.relu_bank(pre, ("legacy0", "legacy1"))
    j = first.affine(first.concat(q, first.select(pre, (0,)), scale),
                     ((Q(73, 16), -Q(15, 8), -Q(13, 16), -Q(195, 32)),))
    t = first.select(sources, (2,))
    return root, first, sources, pre, scale, q, j, t


def _proof_value(fiber, proof, assignment):
    total = Q(0)
    for (kind, index), weight in proof.le_weights:
        if kind == "le":
            row = fiber.predicates[index]
            slack = row.rhs - row.form.at(assignment)
        elif kind == "lo":
            slack = assignment[index] - fiber.factors[index].lower
        else:
            assert kind == "hi"
            slack = fiber.factors[index].upper - assignment[index]
        total += weight * slack
    for index, weight in proof.eq_weights:
        row = fiber.predicates[index]
        total += weight * (row.rhs - row.form.at(assignment))
    return total


def test_phase_budget_strict_complete_readout():
    root, state, sources, pre, q, f, child_pre = _simple_pair()
    pair = state._phase_banks[0].pairs[0]
    assert pair.caps == (Q(5, 8),) * 2
    assert pair.common_form == Form(Q(4, 5), ((0, -Q(4, 5)),))
    assert pair.box_upper == pair.T == Q(8, 5) and pair.T_int == Q(28, 15)
    assert pair.h == (Form(Q(13, 35), ((0, Q(13, 35)),)),) * 2
    assert pair.m == (Q(13, 35),) * 2 and len(pair.row_indices) == 4
    proofs = state.support_proofs(child_pre, (1,))
    bounds = tuple(proof.bound for proof in proofs)
    assert bounds[:5] == (Q(1, 3), Q(309, 5740), Q(5419, 177940),
                          Q(5419, 177940), -Q(1, 140))
    assert len(bounds) == 6 and bounds[5] == bounds[3]
    assert all(state.verify_proof(proof) for proof in proofs)
    fake = list(state.complete((-Q(1, 2), Q(3, 4))))
    first, second = state.banks[0].gates
    fake[first.q_index], fake[second.q_index] = Q(1, 5), Q(0)
    fake[first.phase_index], fake[second.phase_index] = -Q(1, 5), Q(-1)
    old = state._proof_banks[0]
    old_indices = tuple(i for rows in old.gate_rows + old.old_pair_rows for i in rows)
    assert len(old_indices) == 18
    assert all(_holds(state.predicates[i], fake) for i in old_indices)
    assert all(_holds(row, fake) for closure in old.closure_pairs for row in closure.pair.rows)
    assert all(factor.lower <= value <= factor.upper for factor, value in zip(state.factors, fake))
    assert f.forms[0].at(fake) == Q(1, 70) and child_pre.forms[0].at(fake) == Q(1, 140)
    row = state.predicates[pair.row_indices[1]]
    assert row.form.at(fake) - row.rhs == Q(1, 10)
    assert not state.contains(fake)  # Fractional phase diagnostic, never ADV.
    final, child = state.relu(child_pre, "simple_child")
    packet = final.concat(sources, pre, q, f, child_pre, child)
    assert len(packet.forms) == 9 and final.bounds(child) == (0, 0)
    assert len(final.factors) == 8 and sum(v.kind == "phase" for v in final.factors) == 3
    for point in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (-Q(1, 2), Q(3, 4))):
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert len(final.evaluate(packet, assignment)) == 9
        assert final.evaluate(child, assignment) == (0,)
        assert final.evaluate(child_pre, assignment)[0] <= -Q(1, 140)
    assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))
    _record("phase_budget_separation.json", {
        "scope": "complete supplied ordinary mixed pair with live source and successor",
        "six_bounds": [str(v) for v in bounds], "phase_pair": _pair_json(pair),
        "old_eighteen_rows_accept_fractional_diagnostic": True,
        "new_budget_violation": "1/10", "new_consumed_violation": "1/70",
        "fractional_diagnostic_is_adv": False, "successor_bounds": ["0", "0"],
        "complete_ports": 9, "cost": final.cost_report(),
        "actual_model": False, "native_qualified": False, "formal_gain": 0,
    })


def test_d209_positive_cap_multilayer_preserved():
    _, first, _, _, _, _, j, t = _legacy_first()
    pre = first.affine(first.concat(t, j), ((1, Q(3, 64)), (3, Q(1, 16))),
                       (Q(1, 2), Q(1, 4)))
    second, q = first.relu_bank(pre, ("positive0", "positive1"))
    # Keep the exact previously registered property, not new measured bounds.
    old_ell, old_p = Q(884539, 610304), Q(18305, 32768)
    scale = second.affine(j, ((Q(1, 64),),), (1,))
    energy = second.affine(scale, ((Q(7, 8),),))
    w = second.affine(second.concat(q, second.select(pre, (0,)), energy),
                      ((old_ell + old_p, -old_ell / 2, -old_p, -old_ell),))
    child_pre = second.affine(w, ((32,),), (-Q(1, 32),))
    proofs = second.support_proofs(child_pre, (1,))
    bounds = tuple(proof.bound for proof in proofs)
    assert len(bounds) == 6 and all(second.verify_proof(proof) for proof in proofs)
    # Captured stdout is intentionally emitted before the capability assertion:
    # tighter birth bounds need not preserve a fixed h/M query envelope.
    print(json.dumps({
        "diagnostic": "d209_positive_cap_preservation_before_assert",
        "six_bounds": [str(v) for v in bounds], "required_upper": "-1/32",
        "old_property_ell": str(old_ell),
        "actual_gate_bounds": [[str(g.lower), str(g.upper)] for g in second.banks[-1].gates],
        "first_negative_j_bounds": [str(v) for v in first.support_certificates(j, (-1,))],
        "phase_pair": _pair_json(second._phase_banks[-1].pairs[0]),
        "closure_caps": [str(v) for v in second._proof_banks[-1].closure_pairs[0].pair.caps],
        "cost": second.cost_report(), "actual_model": False, "formal_gain": 0,
    }, sort_keys=True, allow_nan=False))
    assert bounds[5] == -Q(1, 32) and min(bounds) <= -Q(1, 32)
    final, child = second.relu(child_pre, "positive_child")
    assert final.bounds(child) == (0, 0)
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment) and final.evaluate(child, assignment) == (0,)


def test_d209_nonpositive_cap_multilayer_preserved():
    root, first, sources, pre, scale, q, j, t = _legacy_first()
    next_pre = first.affine(first.concat(t, j), ((Q(1, 2), Q(1, 16)), (1, 0)),
                            (-Q(1, 8), Q(1, 4)))
    second, next_q = first.relu_bank(next_pre, ("ordered0", "ordered1"))
    child_pre = second.affine(next_q, ((1, -Q(1, 2)),), (-Q(1, 8),))
    bounds = second.support_certificates(child_pre, (1,))
    pair = second._phase_banks[-1].pairs[0]
    assert pair.caps[0] == 0 and pair.parent_proofs[0].bound <= -Q(1, 4)
    assert pair.common_form == Form() and pair.T == 1 and not pair.intrinsic_proofs
    assert min(bounds) <= -Q(1, 8)
    final, child = second.relu(child_pre, "order_child")
    packet = final.concat(sources, pre, scale, q, j, next_pre, next_q, child_pre, child)
    assert len(packet.forms) == 15 and final.bounds(child) == (0, 0)
    assert len(final.factors) == 13 and sum(v.kind == "phase" for v in final.factors) == 5
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert final.evaluate(child, assignment) == (0,)
    zeros = final.complete((Q(1, 2), -Q(1, 4), -Q(1, 4)))
    for gate in (final.banks[0].gates[0], final.banks[1].gates[1]):
        assert gate.input.at(zeros) == 0
        alternate = list(zeros)
        alternate[gate.phase_index] = 1
        assert final.contains(alternate)
    assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))


def test_fifth_certificate_drives_next_birth():
    root, first, sources, pre, q, f, _ = _simple_pair(with_t=True)
    t = first.select(sources, (2,))
    next_pre = first.affine(first.concat(t, f),
                            ((Q(1, 2), Q(1, 16)), (1, Q(3, 32))),
                            (Q(1, 4), Q(1, 4)))
    differences = first.affine(next_pre, ((1, -Q(1, 2)), (-Q(1, 2), 1)))
    cap_proofs = tuple(first.support_proofs(first.select(differences, (i,)), (1,)) for i in (0, 1))
    assert tuple(proofs[4].bound for proofs in cap_proofs) == (Q(1, 8), Q(7, 8))
    assert all(all(p.bound > proofs[4].bound for p in proofs[:4]) for proofs in cap_proofs)
    second, second_q = first.relu_bank(next_pre, ("next0", "next1"))
    pair = second._phase_banks[-1].pairs[0]
    assert pair.caps == (Q(1, 8), Q(7, 8))
    for born, choices in zip(pair.parent_proofs, cap_proofs):
        assert born.bound == choices[4].bound and born.subject == choices[4].subject
        assert born.le_weights == choices[4].le_weights and born.eq_weights == choices[4].eq_weights
        assert second.verify_proof(born)
    expected_common = second.affine(f, ((-Q(1, 14),),)).forms[0]
    assert pair.common_form == expected_common and pair.common_weights
    shared_ids = {key for key, _ in pair.common_weights}
    assert ("le", first._phase_banks[0].pairs[0].row_indices[1]) in shared_ids
    assert ("le", first._proof_banks[0].gate_rows[0][2]) in shared_ids
    assert pair.T == 1
    f2 = second.affine(second.concat(second_q, f),
                       ((1, -Q(1, 2), -Q(1, 112)),), (-Q(1, 8),))
    final_pre = second.affine(f2, ((1,),), (-Q(1, 140),))
    final_proofs = second.support_proofs(final_pre, (1,))
    assert len(final_proofs) == 6 and final_proofs[4].bound == -Q(1, 140)
    assert final_proofs[5].bound == -Q(1, 140)
    assert all(second.verify_proof(proof) for proof in final_proofs)
    final, child = second.relu(final_pre, "recursive_child")
    packet = final.concat(sources, pre, q, f, next_pre, second_q, f2, final_pre, child)
    assert len(packet.forms) == 15 and final.bounds(child) == (0, 0)
    assert len(final.factors) == 13 and sum(v.kind == "phase" for v in final.factors) == 5
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert final.evaluate(child, assignment) == (0,)
        assert len(final.evaluate(packet, assignment)) == 15
    assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))
    _record("multilayer_closure.json", {
        "scope": "fifth certificate selected by actual uniform next-bank birth and consumed again",
        "six_parent_cap_bounds": [[str(p.bound) for p in choices] for choices in cap_proofs],
        "next_phase_pair": _pair_json(pair),
        "next_child_six_bounds": [str(p.bound) for p in final_proofs],
        "successor_bounds": ["0", "0"], "retained_bits": 5, "complete_ports": 15,
        "legacy_positive_preservation_evidence": "captured stdout and pytest result of its separate mandatory test",
        "cost": final.cost_report(), "actual_model": False, "native_qualified": False,
        "formal_gain": 0,
    })


def test_intrinsic_t_bound_identity_and_selection():
    _, first, _, _, _, _, j, t = _legacy_first()
    pre = first.affine(first.concat(j, t), ((1, 0), (1, Q(1, 8))), (1, 1))
    state, output = first.relu_bank(pre, ("intrinsic0", "intrinsic1"))
    pair = state._phase_banks[-1].pairs[0]
    assert pair.caps == (Q(9, 16), Q(5, 8))
    assert pair.common_form == state.affine(j, ((-Q(4, 5),),)).forms[0]
    assert pair.box_upper == Q(1481, 80)
    assert pair.T_int == Q(1871, 116)
    assert pair.T_int < pair.box_upper and pair.T == pair.T_int > 1
    assert pair.selected_T_proof in pair.intrinsic_proofs
    assert len(pair.lower_proofs) == len(pair.intrinsic_proofs) == len(pair.residual_proofs) == 2
    gates = state.banks[-1].gates
    for i in (0, 1):
        upper = (pair.caps[i] + pair.caps[1 - i] / 2) / Q(3, 4)
        lower_proof, intrinsic = pair.lower_proofs[i], pair.intrinsic_proofs[i]
        negative = state.affine(state.select(pre, (i,)), ((-1,),))
        assert lower_proof.subject == negative.forms[0] and lower_proof.bound == -gates[i].lower
        assert intrinsic.subject == pair.common_form
        assert intrinsic.bound == 1 + lower_proof.bound / upper
        assert pair.residual_proofs[i].bound == 0
    all_proofs = (pair.parent_proofs + pair.residual_proofs + pair.lower_proofs
                  + pair.intrinsic_proofs + (pair.box_proof, pair.selected_T_proof))
    assert all(state.verify_proof(proof) for proof in all_proofs)
    for point in POINTS:
        assignment = state.complete(point)
        value = pair.common_form.at(assignment)
        assert 0 <= value <= pair.T
        for proof in all_proofs:
            gap = proof.bound - proof.subject.at(assignment)
            assert gap == _proof_value(state, proof, assignment) and gap >= 0
        assert state.evaluate(output, assignment) == tuple(max(0, v) for v in state.evaluate(pre, assignment))


def test_small_box_bound_does_not_assume_domination():
    _, state, _, _, _, _, _, _ = _legacy_first()
    pair = state._phase_banks[0].pairs[0]
    assert pair.box_upper == Q(40, 41) < 1 and pair.T == 1
    point = list(state.complete((-Q(1, 4), -Q(9, 2000), 0)))
    first, second = state.banks[0].gates
    point[first.q_index], point[second.q_index] = Q(13, 16), Q(0)
    point[first.phase_index], point[second.phase_index] = Q(0), Q(-1)
    assert pair.common_form.at(point) == Q(49, 100)
    assert all(_holds(row, point) for row in pair.rows)
    old = state._proof_banks[0]
    assert all(_holds(state.predicates[index], point) for rows in old.gate_rows for index in rows)
    old_source = state.banks[0].pairs[0].rows[4]
    assert not _holds(old_source, point)
    assert old_source in state.predicates  # The weaker new lift does not replace it.
    assert not state.contains(point)
    q = state._view(tuple(Form(0, ((gate.q_index, Q(1)),)) for gate in state.banks[0].gates))
    for direction in DIRECTIONS:
        bounds = state.support_certificates(q, direction)
        assert len(bounds) == 6 and state.support(q, direction) == min(bounds)
        assert min(bounds) <= min(bounds[:4])


def test_zero_caps_stable_and_zero_phase_labels():
    root = Fiber.box(((-1, 1),), enabled=True)
    pre = root.affine(root.sources(), ((Q(1, 2),), (1,)), (-Q(1, 10), 0))
    state, q = root.relu_bank(pre, ("zero_cap0", "zero_cap1"))
    pair = state._phase_banks[0].pairs[0]
    assert pair.caps == (0, Q(4, 5))
    assert pair.T == pair.T_int == 1 and pair.common_form == Form()
    assert not pair.residual_proofs and not pair.lower_proofs and not pair.intrinsic_proofs
    assert pair.box_proof is None and pair.selected_T_proof is None
    assert pair.h[0] == Form() and pair.m == (Q(1, 2),) * 2
    assert state.support(q, (1, -Q(1, 2))) <= 0
    for x in (-1, 0, Q(1, 5), 1):
        point = state.complete((x,))
        assert state.contains(point)
        for gate in state.banks[0].gates:
            if gate.input.at(point) == 0:
                alternate = list(point)
                alternate[gate.phase_index] = 1
                assert state.contains(alternate)
    stable_pre = root.affine(root.sources(), ((1,), (1,), (0,)), (2, -2, 0))
    stable, stable_q = root.relu_bank(stable_pre, ("positive", "negative", "zero"))
    assert len(stable.predicates) == 12 and not stable._phase_banks[0].pairs
    point = stable.complete((0,))
    assert stable.evaluate(stable_q, point) == (2, 0, 0)
    point = list(point)
    point[stable.banks[0].gates[-1].phase_index] = 1
    assert stable.contains(point)
    point[stable.banks[0].gates[-1].phase_index] = 0
    assert not stable.contains(point)


def test_full_input_decoder_and_interfaces():
    eq = Predicate(Form(0, ((0, Q(1)), (2, -Q(1)))), "eq")
    le = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
                     predicates=(eq, le), decoder_matrix=((2, -1), (-3, 4)),
                     decoder_bias=(Q(1, 3), -Q(1, 7)), enabled=True)
    source = root.sources()
    pre = root.affine(source, ((1, Q(1, 12)), (1, -Q(1, 12))))
    alias, av = root.alias(pre, ("port0", "port1"))
    state, q = alias.relu_bank(av, ("read0", "read1"))
    packet = state.concat(source, root.signed_phases(), pre, av, q)
    identity = tuple(tuple(int(i == j) for j in range(9)) for i in range(9))
    opposite = tuple(tuple(-v for v in row) for row in identity)
    zero = state.add(state.affine(packet, identity), state.affine(packet, opposite))
    assert state.support(zero, (1,) * 9) == 0
    assert state.predicates[0] is eq and state.predicates[1] is le
    assert all(a.identity is b.identity for a, b in zip(root.factors, state.factors))
    assert state._frame.decoder is root._frame.decoder
    assert len(state._phase_banks) == len(state.banks) and type(state) is Fiber
    for x, y, phase in ((1, 0, 1), (-1, Q(1, 2), -1)):
        point = state.complete((x, y), (phase,))
        assert state.contains(point) and state.evaluate(zero, point) == (0,) * 9
        assert state.decode(point) == (2 * x - y + Q(1, 3), -3 * x + 4 * y - Q(1, 7))
        values = state.evaluate(packet, point)
        selected = state.select(packet, (8, 0, 4, 2))
        assert state.evaluate(selected, point) == tuple(values[i] for i in (8, 0, 4, 2))


def test_owned_proofs_and_alias_equalities():
    root = Fiber.box(((-1, 1),), enabled=True)
    form = root.affine(root.sources(), ((2,),), (1,))
    first, av = root.alias(form, ("a",))
    second, bv = first.alias(av, ("b",))
    for direction in ((1,), (-1,)):
        proofs = second.support_proofs(bv, direction)
        assert len(proofs) == 6
        assert all(second.verify_proof(p) and len(p.eq_weights) == 2 for p in proofs)
        assert all(all(value == direction[0] for _, value in p.eq_weights) for p in proofs)
    proof = second.support_proof(bv, (1,))
    for invalid in (replace(proof, bound=proof.bound + 1), replace(proof, eq_weights=()),
                    replace(proof, frame=object()), replace(proof, le_weights=())):
        with pytest.raises(DomainError):
            second.verify_proof(invalid)
    sibling, other = root.alias(form, ("sibling",))
    with pytest.raises(DomainError):
        sibling.verify_proof(proof)
    with pytest.raises(DomainError):
        sibling.support(bv, (1,))
    foreign = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(DomainError):
        second.add(bv, foreign.sources())
    ancestor = root.support_proof(root.sources(), (1,))
    assert second.verify_proof(ancestor)
    with pytest.raises(DomainError):
        root.verify_proof(proof)
    damaged = list(second.complete((0,)))
    damaged[-1] += Q(1, 100)
    assert not second.contains(damaged)
    assert second.cost_report()["alias_equation_references"] == 2


def test_complete_two_hundred_gate_members_and_cost():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    matrix = tuple((1, Q(1, 12)) if i % 2 == 0 else (1, -Q(1, 12)) for i in range(200))
    pre = root.affine(root.sources(), matrix)
    state, output = root.relu_bank(pre, tuple("bank%d" % i for i in range(200)))
    assert len(output.forms) == 200 and len(state.factors) == 402
    assert len({id(f.identity) for f in state.factors}) == 402
    for point in ((0, 0), (Q(1, 2), -Q(3, 4))):
        assignment = state.complete(point)
        assert state.contains(assignment) and state.decode(assignment) == point
        assert state.evaluate(output, assignment) == tuple(max(0, v) for v in state.evaluate(pre, assignment))
        assert all(assignment[g.phase_index] in (-1, 1) for g in state.banks[0].gates)
    proofs = state.support_proofs(output, tuple(1 if i % 2 == 0 else -Q(13, 35) for i in range(200)))
    assert len(proofs) == 6 and all(state.verify_proof(p) for p in proofs)
    report = state.cost_report()
    assert report["continuous"] == 202 and report["binary"] == 200
    assert report["pairs"] == report["closure_pairs"] == report["phase_pairs"] == 100
    assert report["predicates"] == 2200 and report["predicate_nnz"] == 6000
    assert report["phase_row_references"] == 400 and report["phase_product_factors"] == 0
    assert report["phase_parent_proofs"] == report["phase_residual_proofs"] == 200
    assert report["phase_lower_proofs"] == report["phase_intrinsic_proofs"] == 200
    assert report["phase_box_proofs"] == 100 and report["fixed_query_certificates"] == 6
    assert report["parametric_rule"]
    assert report["parametric_product_factors"] == report["parametric_relation_rows"] == 0
    assert report["work_used"] == report["branch_work_used"] <= MAX_BRANCH_WORK
    assert report["entries"] <= MAX_ENTRIES
    assert not any(report[key] for key in ("native_qualified", "physical_qualified",
                                          "actual_model_qualified", "gpu_qualified",
                                          "new_set_class_qualified"))
    _record("complete_bank_cost.json", {
        "scope": "complete supplied 200-gate bank: 100 repetitions of the registered pair",
        "not_actual_or_heterogeneous_cnn_bank": True, "retained_bits": 200,
        "complete_outputs": 200, "six_bounds": [str(p.bound) for p in proofs],
        "cost": report, "actual_model": False, "native_qualified": False,
        "gpu_qualified": False, "formal_gain": 0,
    })


def test_cumulative_budgets_fail_closed():
    with pytest.raises(DomainError):
        Fiber.box(((-1, 1),))
    assert (MAX_WORK, MAX_BRANCH_WORK, MAX_ENTRIES, MAX_BITS) == (256_000_000, 200_000_000, 64_000_000, 512)
    for overrides in ({"max_work": MAX_WORK + 1}, {"max_branch_work": MAX_BRANCH_WORK + 1},
                      {"max_entries": MAX_ENTRIES + 1}, {"max_work": True}):
        with pytest.raises(DomainError):
            Fiber.box(((-1, 1),), enabled=True, **overrides)
    with pytest.raises(BudgetError):
        Fiber.box(((-1, 1),), enabled=True, max_entries=1)
    root = Fiber.box(((-1, 1),), enabled=True, max_branch_work=3000)
    source = root.sources()
    alias, view = root.alias(source, ("alias",))
    assert alias._work is root._work
    with pytest.raises(BudgetError):
        for _ in range(1000):
            alias.support_proofs(view, (1,))
    with pytest.raises(BudgetError):
        root.support(source, (1,))
    with pytest.raises(BudgetError):
        alias.relu(view, "after_exhaustion")
    for value in (True, 0.5, Q(1, 1 << 513)):
        with pytest.raises(DomainError):
            Form(value)


def test_all_five_proofs_sound_and_paid():
    _, state, _, _, q, f, _ = _simple_pair()
    packet = state.concat(q, f)
    before = state.cost_report()
    proofs = state.support_proofs(packet, (Q(2, 3), -Q(5, 4), Q(1, 7)))
    after = state.cost_report()
    assert len(proofs) == after["fixed_query_certificates"] == 6
    assert after["work_used"] > before["work_used"] and after["entries"] > before["entries"]
    assert after["work_used"] == after["branch_work_used"]
    assert after["phase_parent_proofs"] == after["phase_lower_proofs"] == 2
    assert after["phase_intrinsic_proofs"] == after["phase_residual_proofs"] == 2
    assert after["phase_row_references"] == 4 and after["phase_product_factors"] == 0
    assert after["parametric_product_factors"] == after["parametric_relation_rows"] == 0
    for proof in proofs:
        assert state.verify_proof(proof)
        assert all(value > 0 for _, value in proof.le_weights)
        for point in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (-Q(1, 2), Q(3, 4))):
            assignment = state.complete(point)
            gap = proof.bound - proof.subject.at(assignment)
            assert gap == _proof_value(state, proof, assignment) and gap >= 0
    best = state.support_proof(packet, (Q(2, 3), -Q(5, 4), Q(1, 7)))
    assert best.bound == min(p.bound for p in proofs)



def test_mixed_weight_interval_family_and_scaling():
    _, state, sources, pre, q, _, _ = _simple_pair()
    closure = state._proof_banks[0].closure_pairs[0]
    pair = closure.pair
    gate = state.banks[0].gates[0]
    rows = state._proof_banks[0].gate_rows[0]
    assert pair.caps == (Q(5, 8),) * 2 and pair.lower == Q(1, 2)
    assert pair.scale == Form(Q(3, 4), ((0, Q(1, 4)),))
    ell, cap = -gate.lower, pair.caps[0]
    r_min = ell / (cap * pair.lower + ell)
    assert r_min == Q(52, 67)
    scale = state.affine(sources, ((Q(1, 4), 0),), (Q(3, 4),))
    packet = state.concat(q, pre, scale)
    cases = []
    for r in (Q(52, 67), Q(4, 5), Q(9, 10), Q(1)):
        for multiplier in (Q(1), Q(3, 2), Q(5)):
            view = state.affine(packet, ((
                multiplier, -multiplier * r / 2,
                -multiplier * (1 - r), 0, -multiplier * r * cap,
            ),))
            before = state.cost_report()
            proofs = state.support_proofs(view, (1,))
            after = state.cost_report()
            assert len(proofs) == 6 and all(state.verify_proof(p) for p in proofs)
            proof = proofs[5]
            assert proof.bound == 0 and min(p.bound for p in proofs) == 0
            correction = r * cap * pair.lower - (1 - r) * ell
            assert correction >= 0
            expected = {
                ("le", rows[3]): multiplier * (1 - r),
                ("le", closure.row_indices[4]): multiplier * r,
                ("hi", gate.phase_index): multiplier * correction / 2,
            }
            expected = {key: value for key, value in expected.items() if value}
            assert dict(proof.le_weights) == expected and not proof.eq_weights
            assert proof.subject == view.forms[0] and proof.frame is state._frame
            assert after["work_used"] > before["work_used"]
            assert after["entries"] > before["entries"]
            for point in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1),
                          (-Q(1, 2), Q(3, 4))):
                assignment = state.complete(point)
                assert state.contains(assignment) and state.decode(assignment) == point
                gap = proof.bound - proof.subject.at(assignment)
                assert gap == _proof_value(state, proof, assignment) and gap >= 0
            # This is a real graph member attaining the zero bound, not an LP point.
            assert state.evaluate(view, state.complete((1, 1))) == (0,)
            raw = state.affine(q, ((multiplier, -multiplier * r / 2),))
            raw_proofs = state.support_proofs(raw, (1,))
            expected_raw = multiplier * (Q(13, 12) - Q(11, 24) * r)
            assert len(raw_proofs) == 6 and raw_proofs[5].bound == expected_raw
            assert all(state.verify_proof(p) for p in raw_proofs)
            assert min(p.bound for p in raw_proofs[:5]) == multiplier * Q(195, 268)
            assert state.evaluate(raw, state.complete((1, 1))) == (expected_raw,)
            if r == r_min:
                assert raw_proofs[5].bound == min(p.bound for p in raw_proofs[:5])
            else:
                assert raw_proofs[5].bound < min(p.bound for p in raw_proofs[:5])
            raw_after = state.cost_report()
            assert raw_after["work_used"] > after["work_used"]
            cases.append({
                "r": str(r), "multiplier": str(multiplier),
                "six_bounds": [str(p.bound) for p in proofs],
                "complete_subject": _form_json(proof.subject),
                "proofs": [{
                    "bound": str(p.bound),
                    "subject": _form_json(p.subject),
                    "le_weights": [[list(key), str(value)] for key, value in p.le_weights],
                    "eq_weights": [[index, str(value)] for index, value in p.eq_weights],
                    "same_frame": p.frame is state._frame,
                    "owned_factor_count": len(p.identities),
                    "owned_predicate_count": len(p.predicates),
                } for p in proofs],
                "complete_cost_before": before, "complete_cost_after": after,
                "raw_six_bounds": [str(p.bound) for p in raw_proofs],
                "raw_exact_support": str(expected_raw),
                "raw_actual_attainment": ["1", "1"],
                "raw_strict_against_first_five": r > r_min,
                "complete_cost_after_raw": raw_after,
            })
    assert len(cases) == 12
    report = state.cost_report()
    assert report["fixed_query_certificates"] == 6 and report["parametric_rule"]
    assert report["parametric_product_factors"] == report["parametric_relation_rows"] == 0
    _record("parametric_family.json", {
        "scope": "fixed-parent parameter family, full live source readout, exact signed phases",
        "r_min": str(r_min), "registered_cases": cases,
        "parameters": ["52/67", "4/5", "9/10", "1"],
        "multipliers": ["1", "3/2", "5"], "cost": report,
        "global_rebirth_monotonicity_claimed": False, "new_set_class": False,
        "actual_model": False, "native_qualified": False, "formal_gain": 0,
    })


def test_swapped_pair_and_live_skip_consumption():
    root, state, sources, pre, q, _, _ = _simple_pair()
    closure = state._proof_banks[0].closure_pairs[0]
    scale = state.affine(sources, ((Q(1, 4), 0),), (Q(3, 4),))
    cap, multiplier = Q(5, 8), Q(3, 2)
    for r in (Q(52, 67), Q(4, 5), Q(9, 10), Q(1)):
        # The second output is positive, with a live shared affine skip behind an EQ.
        skip = state.affine(state.concat(pre, scale), ((0, 1 - r, r * cap),))
        alias, av = state.alias(skip, ("swapped_live_skip",))
        view = alias.affine(alias.concat(q, av, sources),
                            ((-multiplier * r / 2, multiplier, -multiplier, 0, 0),))
        proofs = alias.support_proofs(view, (1,))
        assert len(proofs) == 6 and proofs[5].bound == 0
        assert all(alias.verify_proof(p) for p in proofs)
        assert proofs[5].eq_weights
        second_gate = alias.banks[0].gates[1]
        assert ("le", closure.row_indices[9]) in dict(proofs[5].le_weights)
        correction = r * cap * Q(1, 2) - (1 - r) * Q(13, 12)
        if correction:
            assert dict(proofs[5].le_weights)[("hi", second_gate.phase_index)] == multiplier * correction / 2
        child_pre = alias.affine(view, ((1,),), (-Q(1, 100),))
        final, child = alias.relu(child_pre, "swapped_successor")
        packet = final.concat(sources, pre, q, av, view, child_pre, child)
        assert len(packet.forms) == 10 and final.bounds(child) == (0, 0)
        assert final._frame.decoder is root._frame.decoder
        assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))
        assert len(final._phase_banks) == len(final.banks) == 2
        for point in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)):
            assignment = final.complete(point)
            assert final.contains(assignment) and final.decode(assignment) == point
            assert len(final.evaluate(packet, assignment)) == 10
            assert final.evaluate(view, assignment)[0] <= 0
            assert final.evaluate(child, assignment) == (0,)
        assert final.evaluate(view, final.complete((1, -1))) == (0,)


def test_endpoint_separation_and_actual_successor():
    root, state, sources, pre, q, _, _ = _simple_pair()
    readout = state.affine(q, ((1, -Q(1, 2)),))
    proofs = state.support_proofs(readout, (1,))
    expected = (Q(13, 12), Q(65, 82), Q(195, 268),
                Q(195, 268), Q(26, 35), Q(5, 8))
    assert tuple(p.bound for p in proofs) == expected
    assert all(state.verify_proof(p) for p in proofs)
    assert all(p.bound > proofs[5].bound for p in proofs[:5])
    attained = state.complete((1, 1))
    assert state.contains(attained) and state.evaluate(readout, attained) == (Q(5, 8),)
    child_pre = state.affine(readout, ((1,),), (-Q(5, 8) - Q(1, 100),))
    child_proofs = state.support_proofs(child_pre, (1,))
    assert tuple(p.bound for p in child_proofs) == tuple(v - Q(5, 8) - Q(1, 100) for v in expected)
    assert child_proofs[5].bound == -Q(1, 100)
    final, child = state.relu(child_pre, "endpoint_successor")
    packet = final.concat(sources, pre, q, readout, child_pre, child)
    assert len(packet.forms) == 9 and final.bounds(child) == (0, 0)
    assert len(final.factors) == 8 and sum(f.kind == "phase" for f in final.factors) == 3
    for point in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1),
                  (-Q(1, 2), Q(3, 4))):
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert final.evaluate(child, assignment) == (0,)
        assert final.evaluate(child_pre, assignment)[0] <= -Q(1, 100)
        assert len(final.evaluate(packet, assignment)) == 9
    assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))
    assert final._frame.decoder is root._frame.decoder
    report = final.cost_report()
    assert report["parametric_product_factors"] == report["parametric_relation_rows"] == 0


def test_invalid_parameter_region_uses_owned_closure():
    _, state, sources, pre, q, _, _ = _simple_pair()
    closure = state._proof_banks[0].closure_pairs[0]
    scale = state.affine(sources, ((Q(1, 4), 0),), (Q(3, 4),))
    packet = state.concat(q, pre, scale)
    for r in (Q(1, 2), Q(3, 2)):
        assert not Q(52, 67) <= r <= 1
        view = state.affine(packet, ((1, -r / 2, -(1 - r), 0, -r * Q(5, 8)),))
        proofs = state.support_proofs(view, (1,))
        assert len(proofs) == 6
        old, new = proofs[3], proofs[5]
        assert new.bound == old.bound and new.subject == old.subject
        assert new.le_weights == old.le_weights and new.eq_weights == old.eq_weights
        assert new.frame is old.frame is state._frame
        assert all(state.verify_proof(p) for p in proofs)
        assert all(value > 0 for _, value in new.le_weights)
        assert ("le", closure.row_indices[4]) in dict(new.le_weights)
        for point in ((-1, -1), (0, 0), (1, 1), (-Q(1, 2), Q(3, 4))):
            assignment = state.complete(point)
            assert state.contains(assignment)
            gap = new.bound - new.subject.at(assignment)
            assert gap == _proof_value(state, new, assignment) and gap >= 0
