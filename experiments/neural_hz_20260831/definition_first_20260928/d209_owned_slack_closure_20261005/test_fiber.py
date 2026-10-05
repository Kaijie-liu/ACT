"""Preregistered supplied-structure mathematics, not model/ADV qualification.

No constraint or cap is injected into either layered control.  Every bank and
every proof is born through the same default-off owned-domain interface.
"""

from dataclasses import replace
from fractions import Fraction as Q
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d209_owned_slack_closure_20261005.fiber import (
    Fiber, Form, Predicate, DomainError, BudgetError, MAX_WORK,
    MAX_BRANCH_WORK, MAX_ENTRIES, MAX_BITS,
)


POINTS = (
    (-1, 1, -1), (-1, 1, 1), (-1, -1, -1), (-1, -1, 1),
    (0, 0, 0), (Q(1, 3), -Q(1, 2), 0),
    (Q(1, 2), -Q(1, 4), -Q(1, 4)),
    (Q(9, 10), -Q(2, 5), Q(1, 24)),
)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-2, 3))


def _holds(row, assignment):
    value = row.form.at(assignment)
    return value == row.rhs if row.relation == "eq" else value <= row.rhs


def _record(name, value):
    run = Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"])
    expected = (Path(__file__).resolve().parents[2]
                / "results/d209_owned_slack_closure_20261005_v1")
    assert run == expected and run.is_dir() and not run.is_symlink()
    with (run / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _first(aliases=False):
    root = Fiber.box(((-1, 1),) * 3, names=("x", "y", "t"), enabled=True)
    sources = root.sources()
    pre = root.affine(sources, ((1, 3, 0), (3, 4, 0)), (Q(1, 4),) * 2)
    scale = root.affine(sources, ((0, Q(1, 4), 0),), (Q(3, 4),))
    parent = root
    if aliases:
        parent, pre = root.alias(pre, ("pre_alias0", "pre_alias1"))
    first, output = parent.relu_bank(pre, ("first0", "first1"))
    joint = first.concat(output, first.select(pre, (0,)), scale)
    j = first.affine(joint, ((Q(73, 16), -Q(15, 8), -Q(13, 16), -Q(195, 32)),))
    t = first.select(sources, (2,))
    return root, first, sources, pre, scale, output, j, t


def _negative_control():
    root, first, sources, pre, scale, output, j, t = _first()
    next_pre = first.affine(first.concat(t, j),
                            ((Q(1, 2), Q(1, 16)), (1, 0)),
                            (-Q(1, 8), Q(1, 4)))
    second, second_q = first.relu_bank(next_pre, ("second0", "second1"))
    child_pre = second.affine(second_q, ((1, -Q(1, 2)),), (-Q(1, 8),))
    return (root, first, second, sources, pre, scale, output, j,
            next_pre, second_q, child_pre)


def _positive_control():
    root, first, sources, pre, scale, output, j, t = _first()
    next_pre = first.affine(first.concat(t, j),
                            ((1, Q(3, 64)), (3, Q(1, 16))),
                            (Q(1, 2), Q(1, 4)))
    second, second_q = first.relu_bank(next_pre, ("positive0", "positive1"))
    # These are preregistered readout coefficients, not externally supplied
    # birth bounds, scales, predicates, or query certificates.
    ell = Q(884539, 610304)
    lower = Q(2615, 4096)
    ci = Q(7, 8)
    coefficient = ci * lower
    born = second._proof_banks[-1].closure_pairs[0]
    assert -second.banks[-1].gates[0].lower == ell
    assert born.pair.caps == (ci, Q(5, 2))
    assert born.pair.lower == lower
    source_scale = second.affine(j, ((Q(1, 64),),), (1,))
    assert source_scale.forms[0] == born.pair.scale
    energy = second.affine(source_scale, ((ci,),))
    joined = second.concat(second_q, second.select(next_pre, (0,)), energy)
    witness = second.affine(joined, ((ell + coefficient, -ell / 2,
                                      -coefficient, -ell),))
    child_pre = second.affine(witness, ((32,),), (-Q(1, 32),))
    return (root, first, second, sources, pre, scale, output, j,
            next_pre, second_q, source_scale, witness, child_pre)


def _identity_value(fiber, proof, assignment):
    """Independent scalar reconstruction of the public sparse proof identity."""
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


def test_default_off_and_hz_embedding():
    with pytest.raises(DomainError):
        Fiber.box(((-1, 1),))
    equality = Predicate(Form(0, ((0, Q(1)), (2, Q(-1)))), "eq")
    inequality = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
                     predicates=(equality, inequality),
                     decoder_matrix=((2, 0), (0, 3)), decoder_bias=(1, -1),
                     enabled=True)
    point = root.complete((1, 0), (1,))
    assert root.contains(point) and root.decode(point) == (3, -1)
    assert root.evaluate(root.signed_phases(), point) == (1,)
    assert root.predicates[0] is equality and root.predicates[1] is inequality
    assert not root.contains((0, 0, 1))
    with pytest.raises(DomainError):
        Fiber(root._frame, root.factors, root.predicates)
    report = root.cost_report()
    assert report["fixed_query_certificates"] == 4
    assert not report["native_qualified"] and report["formal_gain"] == 0


def test_original_graph_and_phase_semantics():
    root = Fiber.box(((-1, 1),), enabled=True)
    pre = root.affine(root.sources(), ((1,), (1,), (1,), (0,)), (0, 2, -2, 0))
    fiber, q = root.relu_bank(pre, ("cross", "positive", "negative", "zero"))
    assert len(fiber.predicates) == 16 and len(fiber.banks[0].gates) == 4
    assert [gate.stable for gate in fiber.banks[0].gates] == [0, 1, -1, -1]
    assert sum(f.kind == "phase" for f in fiber.factors) == 4
    for x in (-1, 0, 1):
        point = fiber.complete((x,))
        expected = tuple(max(0, v) for v in fiber.evaluate(pre, point))
        assert fiber.evaluate(q, point) == expected
        assert all(_holds(row, point) for row in fiber.predicates)
        for index in (0, 3) if x == 0 else (3,):
            alternate = list(point)
            alternate[fiber.banks[0].gates[index].phase_index] = 1
            assert fiber.contains(alternate)
    wrong = list(fiber.complete((1,)))
    wrong[fiber.banks[0].gates[0].phase_index] = -1
    assert not fiber.contains(wrong)
    wrong[fiber.banks[0].gates[0].phase_index] = 0
    assert not fiber.contains(wrong)


def test_exact_alias_and_signed_eq_witness():
    root = Fiber.box(((-1, 1),), enabled=True)
    affine = root.affine(root.sources(), ((2,),), (1,))
    first, alias = root.alias(affine, ("affine_alias",))
    second, twice = first.alias(alias, ("second_alias",))
    for direction in ((1,), (-1,)):
        proofs = second.support_proofs(twice, direction)
        assert tuple(p.bound for p in proofs) == (root.support(affine, direction),) * 4
        for proof in proofs:
            assert second.verify_proof(proof)
            assert len(proof.eq_weights) == 2
            assert all(weight == direction[0] for _, weight in proof.eq_weights)
    point = second.complete((Q(1, 3),))
    assert second.evaluate(twice, point) == (Q(5, 3),)
    damaged = list(point)
    damaged[-1] += Q(1, 100)
    assert not second.contains(damaged)
    direct = _first()
    expanded = _first(aliases=True)
    assert direct[1].support_certificates(direct[6], (1,)) == expanded[1].support_certificates(expanded[6], (1,))
    assert expanded[1].cost_report()["aliases"] == 2
    alias_proofs = expanded[1].support_proofs(expanded[6], (1,))
    assert any(p.eq_weights for p in alias_proofs)
    assert all(expanded[1].verify_proof(p) for p in alias_proofs)


def test_query_witnesses_reconstruct_all_four_bounds():
    _, fiber, _, _, _, output, j, _ = _first()
    joined = fiber.concat(output, j)
    direction = (Q(2, 3), -Q(5, 4), Q(1, 7))
    proofs = fiber.support_proofs(joined, direction)
    assert len(proofs) == 4
    assert tuple(p.bound for p in proofs) == fiber.support_certificates(joined, direction)
    assert min(p.bound for p in proofs) == fiber.support(joined, direction)
    for proof in proofs:
        assert fiber.verify_proof(proof)
        assert proof.subject == proofs[0].subject
        assert all(weight > 0 for _, weight in proof.le_weights)
        for point in POINTS:
            assignment = fiber.complete(point)
            gap = proof.bound - proof.subject.at(assignment)
            assert gap == _identity_value(fiber, proof, assignment) and gap >= 0


def test_reject_modified_foreign_and_missing_proofs():
    root = Fiber.box(((-1, 1),), enabled=True)
    proof = root.support_proof(root.sources(), (1,))
    key, weight = proof.le_weights[0]
    malformed = (
        replace(proof, bound=proof.bound + 1),
        replace(proof, subject=Form(0)),
        replace(proof, le_weights=()),
        replace(proof, le_weights=((key, -weight),)),
        replace(proof, le_weights=((key, weight), (key, weight))),
        replace(proof, le_weights=((("le", 0), weight),)),
        replace(proof, identities=(object(),)),
        replace(proof, frame=object()),
        replace(proof, predicates=(Predicate(Form()),)),
    )
    for invalid in malformed:
        with pytest.raises(DomainError):
            root.verify_proof(invalid)
    foreign = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(DomainError):
        foreign.verify_proof(proof)
    with pytest.raises(DomainError):
        root.verify_proof(None)
    alias, view = root.alias(root.sources(), ("alias",))
    eqproof = alias.support_proof(view, (1,))
    assert eqproof.eq_weights
    with pytest.raises(DomainError):
        alias.verify_proof(replace(eqproof, eq_weights=()))
    with pytest.raises(DomainError):
        alias.verify_proof(replace(eqproof, predicates=tuple(
            Predicate(row.form, row.relation, row.rhs) for row in eqproof.predicates)))


def test_root_box_slack_matches_source_pair():
    _, first, _, _, _, _, j, _ = _first()
    old = first.banks[0].pairs[0]
    closure = first._proof_banks[0].closure_pairs[0]
    assert old.caps == closure.pair.caps == (Q(13, 8), Q(41, 8))
    assert old.scale == closure.pair.scale == Form(Q(3, 4), ((1, Q(1, 4)),))
    assert old.lower == closure.pair.lower == Q(1, 2)
    assert old.rows == closure.pair.rows
    assert len(first.predicates) == 18 and len(closure.row_indices) == 10
    assert closure.common_weights and len(closure.scale_proofs) == 2
    assert all(first.verify_proof(p) for p in closure.parent_proofs + closure.scale_proofs)
    certificates = first.support_certificates(j, (1,))
    assert certificates[2:] == (0, 0)
    assert certificates[0] > 0 and certificates[1] > 0
    assert first.support_certificates(j, (-1,)) == (
        Q(1481, 64), Q(25385, 1216), Q(193129, 9536), Q(193129, 9536))


def test_two_layer_order_closure_is_intrinsic():
    (root, first, second, sources, pre, scale, output, j,
     next_pre, second_q, child_pre) = _negative_control()
    closure = second._proof_banks[-1].closure_pairs[0]
    assert tuple(p.bound for p in closure.parent_proofs)[0] == -Q(1, 4)
    assert closure.pair.caps[0] == 0 and closure.pair.scale == Form(1)
    assert closure.pair.lower == 1 and not closure.common_weights
    assert not closure.scale_proofs and closure.pair.source_h[0] == Form()
    assert closure.pair.source_m[0] == Q(1, 2)
    old = second.banks[-1].pairs[0]
    assert old.caps == (Q(985, 1024), Q(3657, 2048)) and old.scale == Form(1)
    proofs = second.support_proofs(child_pre, (1,))
    bounds = tuple(p.bound for p in proofs)
    assert all(bound >= Q(1, 4) for bound in bounds[:3])
    assert bounds[3] == -Q(1, 8)
    final, child = second.relu(child_pre, "order_child")
    assert type(final) is Fiber and final.bounds(child) == (0, 0)
    assert final.banks[-1].gates[0].stable == -1
    all_ports = final.concat(sources, pre, scale, output, j, next_pre,
                             second_q, child_pre, child)
    assert len(all_ports.forms) == 15 and len(final.factors) == 13
    assert sum(f.kind == "phase" for f in final.factors) == 5
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert final.evaluate(child, assignment) == (0,)
        assert final.evaluate(child_pre, assignment)[0] <= -Q(1, 8)
        assert len(final.evaluate(all_ports, assignment)) == 15
        assert all(second.verify_proof(proof) for proof in proofs)
    assert second.evaluate(next_pre, second.complete((-1, 1, -1))) == (-Q(5, 8), -Q(3, 4))
    assert second.evaluate(next_pre, second.complete((-1, 1, 1))) == (Q(3, 8), Q(5, 4))
    zeros = final.complete((Q(1, 2), -Q(1, 4), -Q(1, 4)))
    for gate in (final.banks[0].gates[0], final.banks[1].gates[1]):
        assert gate.input.at(zeros) == 0
        alternate = list(zeros)
        alternate[gate.phase_index] = 1
        assert final.contains(alternate)
    assert all(a.identity is b.identity for a, b in zip(root.factors, final.factors))
    _record("closure_repair.json", {
        "scope": "supplied ordinary two-layer order control; no injected predicates/caps",
        "preactivation_certificates": [str(v) for v in bounds],
        "parent_difference_certificates": [str(p.bound) for p in closure.parent_proofs],
        "successor_bounds": ["0", "0"], "complete_ports": 15,
        "original_bits_retained": 5, "cost": final.cost_report(),
        "actual_model": False, "native_qualified": False,
        "gpu_qualified": False, "formal_gain": 0,
    })


def test_positive_cap_shared_slack_closure():
    (_, first, second, _, _, _, _, j, next_pre, second_q,
     source_scale, witness, child_pre) = _positive_control()
    meta = second._proof_banks[-1]
    closure = meta.closure_pairs[0]
    assert tuple(p.bound for p in closure.parent_proofs) == (Q(7, 8), Q(5, 2))
    assert closure.common_form == second.affine(j, ((-Q(1, 64),),)).forms[0]
    assert closure.common_box_upper == Q(1481, 4096) and closure.tau == 1
    assert closure.pair.lower == Q(2615, 4096)
    assert closure.common_weights and len(closure.scale_proofs) == 2
    assert all(second.verify_proof(p) for p in closure.parent_proofs + closure.scale_proofs)
    old = second.banks[-1].pairs[0]
    assert old.caps == (Q(4825, 4096), Q(26685, 8192))
    old_scale = second.affine(j, ((Q(64, 5444),),), (Q(4203, 5444),))
    assert old.scale == old_scale.forms[0] and old.lower == Q(1, 2)

    source = (Q(9, 10), -Q(2, 5), Q(1, 24))
    true = second.complete(source)
    assert second.evaluate(j, true) == (-Q(4129, 640),)
    assert second.evaluate(next_pre, true) == (Q(29399, 122880), -Q(289, 10240))
    ell, ci = Q(884539, 610304), Q(7, 8)
    g1 = next_pre.forms[0].at(true)
    q1 = ci * (g1 + ell) / (ci + ell)
    beta1 = q1 / ci
    fake = list(true)
    gates = second.banks[-1].gates
    fake[gates[0].q_index], fake[gates[1].q_index] = q1, Q(0)
    fake[gates[0].phase_index], fake[gates[1].phase_index] = 2 * beta1 - 1, Q(-1)
    assert Q(63, 100) < q1 < Q(65, 100)
    assert Q(18, 25) < beta1 < Q(26, 35)
    assert all(f.lower <= v <= f.upper for f, v in zip(second.factors, fake))
    assert all(_holds(row, fake) for row in first.predicates)
    old_indices = tuple(i for rows in meta.gate_rows + meta.old_pair_rows for i in rows)
    assert len(old_indices) == 18 and all(_holds(second.predicates[i], fake) for i in old_indices)
    # Six tight-constant labelled rows, including both nonnegative amplitudes.
    q2, beta2 = Q(0), Q(0)
    tight_constant_slacks = (
        q1, q2, Q(17, 6) * beta1 - q1, Q(47, 12) * beta2 - q2,
        ci * beta1 + q2 / 2 - q1, Q(5, 2) * beta2 + q1 / 2 - q2,
    )
    assert len(tight_constant_slacks) == 6 and min(tight_constant_slacks) >= 0
    new_row = second.predicates[closure.row_indices[4]]
    row_violation = new_row.form.at(fake) - new_row.rhs
    assert row_violation > 0
    w_value = witness.forms[0].at(fake)
    assert w_value > Q(1, 600)
    fake_child_pre = child_pre.forms[0].at(fake)
    assert fake_child_pre > Q(53, 2400)
    # This is deliberately not a native witness: fractional original signed
    # phase labels are a relaxation diagnostic, never an ADV/SAT result.
    assert fake[gates[0].phase_index] not in (-1, 1)
    assert not second.contains(fake)

    proofs = second.support_proofs(child_pre, (1,))
    bounds = tuple(p.bound for p in proofs)
    assert all(bound > Q(53, 2400) for bound in bounds[:3])
    assert bounds[3] == -Q(1, 32)
    for proof in proofs:
        assert second.verify_proof(proof)
        assert proof.bound - proof.subject.at(fake) == _identity_value(second, proof, fake)
    final, child = second.relu(child_pre, "positive_child")
    assert final.bounds(child) == (0, 0) and final.banks[-1].gates[0].stable == -1
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment)
        assert final.evaluate(witness, assignment)[0] <= 0
        assert final.evaluate(child, assignment) == (0,)
        assert all(assignment[i] in (-1, 1) for i, f in enumerate(final.factors) if f.kind == "phase")
    _record("positive_slack_control.json", {
        "scope": "supplied mixed two-layer positive-cap control; no injected predicates/caps",
        "preactivation_certificates": [str(v) for v in bounds],
        "parent_difference_certificates": [str(p.bound) for p in closure.parent_proofs],
        "common_box_upper": str(closure.common_box_upper), "tau": str(closure.tau),
        "scale_lower": str(closure.pair.lower), "common_slack_terms": len(closure.common_weights),
        "fractional_diagnostic_source": [str(v) for v in source],
        "fractional_q1": str(q1), "fractional_beta1": str(beta1),
        "old_eighteen_rows_accept": True, "tight_constant_six_rows_accept": True,
        "new_source_phase_violation": str(row_violation), "witness_value": str(w_value),
        "fractional_successor_preactivation": str(fake_child_pre),
        "fractional_point_is_native": False, "is_adv_or_sat": False,
        "successor_bounds": ["0", "0"], "cost": final.cost_report(),
        "actual_model": False, "native_qualified": False,
        "gpu_qualified": False, "formal_gain": 0,
    })


def test_three_bank_mixed_queries_are_sound():
    _, _, second, sources, _, _, _, _, _, second_q, _ = _negative_control()
    joint = second.concat(second_q, sources)
    pre = second.affine(joint, ((1, -Q(2, 3), Q(1, 5), 0, Q(1, 7)),
                                (-Q(3, 4), 1, 0, Q(2, 5), -Q(1, 6))),
                        (-Q(1, 8), Q(1, 9)))
    final, output = second.relu_bank(pre, ("third0", "third1"))
    assert len(final.banks) == 3 and type(final) is Fiber
    proofs = tuple(final.support_proofs(output, direction) for direction in DIRECTIONS)
    for direction, certificates in zip(DIRECTIONS, proofs):
        assert len(certificates) == 4 and all(final.verify_proof(p) for p in certificates)
        for point in POINTS:
            assignment = final.complete(point)
            truth = final.evaluate(output, assignment)
            assert truth == tuple(max(0, v) for v in final.evaluate(pre, assignment))
            target = sum(a * b for a, b in zip(direction, truth))
            assert all(p.bound >= target for p in certificates)


def test_all_modes_are_paid_and_retained():
    _, first, _, _, _, output, j, _ = _first()
    before = first.cost_report()
    proofs = first.support_proofs(first.concat(output, j), (1, -1, 1))
    after = first.cost_report()
    assert len(proofs) == 4 and after["fixed_query_certificates"] == 4
    assert after["work_used"] > before["work_used"]
    assert after["entries"] > before["entries"]
    assert after["branch_work_used"] == after["work_used"]
    assert after["stored_parent_proofs"] == 2 and after["stored_scale_proofs"] == 2
    assert after["stored_proof_le_terms"] > 0 and after["common_slack_terms"] > 0
    assert after["closure_row_references"] == 10
    assert len(first.predicates) == 18 and len(first.banks[0].pairs[0].rows) == 10
    alias, view = first.alias(output, ("account0", "account1"))
    assert alias._work is first._work
    assert alias.cost_report()["alias_equation_references"] == 2
    assert alias.support_certificates(view, (1, -1)) == first.support_certificates(output, (1, -1))


def test_complete_two_hundred_gate_accounting():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    matrix = tuple((1, 3) if i % 2 == 0 else (3, 4) for i in range(200))
    pre = root.affine(root.sources(), matrix, (Q(1, 4),) * 200)
    fiber, output = root.relu_bank(pre, tuple("g%d" % i for i in range(200)))
    point = fiber.complete((0, 0))
    assert len(point) == 402 and fiber.contains(point)
    assert fiber.evaluate(output, point) == (Q(1, 4),) * 200
    proofs = fiber.support_proofs(output, (1,) * 200)
    assert len(proofs) == 4 and min(p.bound for p in proofs) >= 50
    assert all(fiber.verify_proof(p) for p in proofs)
    report = fiber.cost_report()
    assert report["factors"] == 402 and report["continuous"] == 202 and report["binary"] == 200
    assert report["banks"] == 1 and report["pairs"] == report["closure_pairs"] == 100
    assert report["predicates"] == 1800 and report["predicate_nnz"] == 4600
    assert report["closure_row_references"] == 1000
    assert report["stored_parent_proofs"] == report["stored_scale_proofs"] == 200
    assert report["common_slack_terms"] >= 100 and report["stored_proof_le_terms"] >= 400
    assert len({id(f.identity) for f in fiber.factors}) == 402
    assert report["whole_work_cap"] == MAX_WORK == 256_000_000
    assert report["branch_work_cap"] == MAX_BRANCH_WORK == 200_000_000
    assert report["entries_cap"] == MAX_ENTRIES == 64_000_000
    assert report["work_used"] == report["branch_work_used"] <= MAX_BRANCH_WORK
    assert report["entries"] <= MAX_ENTRIES
    assert not any(report[key] for key in ("physical_qualified", "native_qualified",
                                          "actual_model_qualified", "gpu_qualified",
                                          "new_set_class_qualified"))
    _record("complete_200_gate_control.json", {
        "supplied_structure": "100 repetitions of the fixed ordinary two-source pair",
        "not_heterogeneous_or_actual_native_bank": True,
        "cost_scope": "complete exercise including construction, membership and all four proofs",
        "query_certificates": [str(p.bound) for p in proofs], "cost": report,
        "actual_model": False, "native_qualified": False,
        "gpu_qualified": False, "formal_gain": 0,
    })


def test_full_interface_and_decoder_preserved():
    eq = Predicate(Form(0, ((0, Q(1)), (2, Q(-1)))), "eq")
    le = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("input_phase",),
                     predicates=(eq, le), decoder_matrix=((2, -1), (-3, 4)),
                     decoder_bias=(Q(1, 3), -Q(1, 7)), enabled=True)
    sources = root.sources()
    pre = root.affine(sources, ((1, 3), (3, 4)), (Q(1, 4),) * 2)
    aliased, av = root.alias(pre, ("port0", "port1"))
    first, q = aliased.relu_bank(av, ("read0", "read1"))
    packet = first.concat(sources, root.signed_phases(), pre, av, q)
    identity = tuple(tuple(int(i == j) for j in range(9)) for i in range(9))
    negative = tuple(tuple(-v for v in row) for row in identity)
    copy = first.affine(packet, identity)
    zero = first.add(copy, first.affine(packet, negative))
    assert first.support(zero, (1,) * 9) == 0
    selected = first.select(packet, (8, 0, 4, 2))
    for source, phase in (((1, 0), (1,)), ((-1, Q(1, 2)), (-1,))):
        point = first.complete(source, phase)
        assert first.contains(point) and first.evaluate(zero, point) == (0,) * 9
        values = first.evaluate(packet, point)
        assert first.evaluate(selected, point) == tuple(values[i] for i in (8, 0, 4, 2))
        assert first.decode(point) == (2 * source[0] - source[1] + Q(1, 3),
                                       -3 * source[0] + 4 * source[1] - Q(1, 7))
    assert first.predicates[0] is eq and first.predicates[1] is le
    assert all(a.identity is b.identity for a, b in zip(root.factors, first.factors))
    assert first._frame.decoder is root._frame.decoder


def test_nonpositive_caps_keep_original_bits():
    root = Fiber.box(((-1, 1),), enabled=True)
    pre = root.affine(root.sources(), ((Q(1, 2),), (1,)), (-Q(1, 10), 0))
    fiber, output = root.relu_bank(pre, ("ordered0", "ordered1"))
    closure = fiber._proof_banks[0].closure_pairs[0]
    assert tuple(p.bound for p in closure.parent_proofs) == (-Q(1, 10), Q(4, 5))
    assert closure.pair.caps == (0, Q(4, 5)) and not fiber.banks[0].pairs
    assert closure.pair.scale == Form(1) and closure.pair.lower == 1
    assert not closure.common_weights and not closure.scale_proofs
    assert closure.pair.source_h[0] == Form() and closure.pair.source_m[0] == Q(1, 2)
    assert sum(f.kind == "phase" for f in fiber.factors) == 2
    assert len(fiber._proof_banks[0].gate_rows) == 2
    assert fiber.support(output, (1, -Q(1, 2))) == 0
    for x in (-1, 0, Q(1, 5), 1):
        point = fiber.complete((x,))
        assert fiber.contains(point)
        truth = fiber.evaluate(output, point)
        assert truth[0] <= truth[1] / 2
        for gate in fiber.banks[0].gates:
            if gate.input.at(point) == 0:
                alternate = list(point)
                alternate[gate.phase_index] = 1
                assert fiber.contains(alternate)


def test_zero_and_no_shared_slack():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    pre = root.affine(root.sources(), ((Q(2, 3), 0), (-Q(2, 3), -2)), (Q(1, 2),) * 2)
    fiber, output = root.relu_bank(pre, ("unshared0", "unshared1"))
    closure = fiber._proof_banks[0].closure_pairs[0]
    assert closure.pair.caps == (Q(9, 4), Q(13, 4))
    assert not closure.common_weights and closure.common_form == Form()
    assert closure.common_box_upper == 0 and closure.tau == 1
    assert closure.pair.scale == Form(1) and closure.pair.lower == 1
    assert len(closure.scale_proofs) == 2
    for direction in DIRECTIONS:
        bounds = fiber.support_certificates(output, direction)
        assert bounds[1] == bounds[2] == bounds[3]
    zero_root = Fiber.box(((0, 0),), enabled=True)
    zero, q = zero_root.relu(zero_root.sources(), "zero_gate")
    assert not zero._proof_banks[0].closure_pairs
    assert zero.support_certificates(q, (1,)) == (0, 0, 0, 0)
    point = zero.complete((0,))
    for sign in (-1, 1):
        labelled = list(point)
        labelled[zero.banks[0].gates[0].phase_index] = sign
        assert zero.contains(labelled)


def test_resources_and_exact_inputs_fail_closed():
    assert MAX_BITS == 512
    for bad in (0.5, True, Q(1, 1 << 513)):
        with pytest.raises(DomainError):
            Form(bad)
    for kwargs in ({"max_work": MAX_WORK + 1},
                   {"max_branch_work": MAX_BRANCH_WORK + 1},
                   {"max_entries": MAX_ENTRIES + 1}, {"max_work": True}):
        with pytest.raises(DomainError):
            Fiber.box(((-1, 1),), enabled=True, **kwargs)
    with pytest.raises(BudgetError):
        Fiber.box(((-1, 1),), enabled=True, max_entries=1)
    root = Fiber.box(((-1, 1),), enabled=True, max_branch_work=1000)
    source = root.sources()
    with pytest.raises(BudgetError):
        for _ in range(1000):
            root.support_proofs(source, (1,))
    assert root._work.failed and root._work.branch_used > root._work.max_branch_work
    with pytest.raises(BudgetError):
        root.support(source, (1,))
    with pytest.raises(BudgetError):
        root.alias(source, ("after_exhaustion",))
    fresh = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(DomainError):
        fresh.affine(fresh.sources(), ((0.5,),))
    with pytest.raises(DomainError):
        fresh.support_proofs(fresh.sources(), (True,))
    with pytest.raises(DomainError):
        fresh.affine(fresh.sources(), ((1, 2),))


def test_initial_predicates_and_sibling_identity():
    eq = Predicate(Form(0, ((0, Q(1)), (1, Q(-1)))), "eq")
    le = Predicate(Form(0, ((0, Q(1)),)), "le", Q(3, 4))
    root = Fiber.box(((-1, 1), (-1, 1)), predicates=(eq, le), enabled=True)
    sources = root.sources()
    root_proof = root.support_proof(root.select(sources, (0,)), (1,))
    left, lv = root.alias(sources, ("left0", "left1"))
    right, rv = root.alias(sources, ("right0", "right1"))
    assert left.verify_proof(root_proof) and right.verify_proof(root_proof)
    left_proof = left.support_proof(lv, (1, -1))
    with pytest.raises(DomainError):
        right.verify_proof(left_proof)
    with pytest.raises(DomainError):
        left.add(lv, rv)
    with pytest.raises(DomainError):
        right.support(lv, (1, 0))
    foreign = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    with pytest.raises(DomainError):
        left.concat(lv, foreign.sources())
    pre = left.affine(lv, ((1, 2), (-2, 1)), (Q(1, 3), -Q(1, 5)))
    final, output = left.relu_bank(pre, ("predicate0", "predicate1"))
    assert final.predicates[0] is eq and final.predicates[1] is le
    with pytest.raises(DomainError):
        final.complete((1, 1))
    with pytest.raises(DomainError):
        final.complete((0, Q(1, 2)))
    valid = final.complete((Q(1, 2), Q(1, 2)))
    assert final.contains(valid) and final.decode(valid) == (Q(1, 2), Q(1, 2))
    assert len(final.evaluate(output, valid)) == 2
    assert final._work is left._work is right._work is root._work
    assert final.verify_proof(left_proof)
