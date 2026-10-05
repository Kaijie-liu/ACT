"""Frozen-interface equivalence and accounting; no real-model qualification."""

from dataclasses import replace
from fractions import Fraction as Q
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d209_owned_slack_closure_20261005.fiber import Fiber as PreviousFiber
from experiments.neural_hz_20260831.definition_first_20260928.d211_shared_proof_frame_20261005.fiber import (
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
                / "results/d211_shared_proof_frame_20261005_v1")
    assert run == expected and run.is_dir() and not run.is_symlink()
    with (run / name).open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _holds(row, assignment):
    value = row.form.at(assignment)
    return value == row.rhs if row.relation == "eq" else value <= row.rhs


def _control(cls, positive):
    root = cls.box(((-1, 1),) * 3, names=("x", "y", "t"), enabled=True)
    sources = root.sources()
    pre = root.affine(sources, ((1, 3, 0), (3, 4, 0)), (Q(1, 4),) * 2)
    scale = root.affine(sources, ((0, Q(1, 4), 0),), (Q(3, 4),))
    first, q = root.relu_bank(pre, ("first0", "first1"))
    j = first.affine(first.concat(q, first.select(pre, (0,)), scale),
                     ((Q(73, 16), -Q(15, 8), -Q(13, 16), -Q(195, 32)),))
    t = first.select(sources, (2,))
    if positive:
        matrix, bias = ((1, Q(3, 64)), (3, Q(1, 16))), (Q(1, 2), Q(1, 4))
    else:
        matrix, bias = ((Q(1, 2), Q(1, 16)), (1, 0)), (-Q(1, 8), Q(1, 4))
    second_pre = first.affine(first.concat(t, j), matrix, bias)
    second, second_q = first.relu_bank(second_pre, ("second0", "second1"))
    witness = None
    if positive:
        ell, ci, lower = Q(884539, 610304), Q(7, 8), Q(2615, 4096)
        born = second._proof_banks[-1].closure_pairs[0]
        assert -second.banks[-1].gates[0].lower == ell
        assert born.pair.caps == (ci, Q(5, 2)) and born.pair.lower == lower
        derived_scale = second.affine(j, ((Q(1, 64),),), (1,))
        assert derived_scale.forms[0] == born.pair.scale
        energy = second.affine(derived_scale, ((ci,),))
        joint = second.concat(second_q, second.select(second_pre, (0,)), energy)
        witness = second.affine(joint, ((ell + ci * lower, -ell / 2,
                                         -ci * lower, -ell),))
        child_pre = second.affine(witness, ((32,),), (-Q(1, 32),))
    else:
        child_pre = second.affine(second_q, ((1, -Q(1, 2)),), (-Q(1, 8),))
    return dict(root=root, first=first, second=second, sources=sources,
                pre=pre, scale=scale, q=q, j=j, second_pre=second_pre,
                second_q=second_q, witness=witness, child_pre=child_pre)


def _complete_packet(control, state, child):
    return state.concat(control["sources"], control["pre"], control["scale"],
                        control["q"], control["j"], control["second_pre"],
                        control["second_q"], control["child_pre"], child)


def _state_shape_equal(one, two):
    assert len(one.factors) == len(two.factors)
    assert tuple((f.kind, f.lower, f.upper, f.producer) for f in one.factors) == tuple(
        (f.kind, f.lower, f.upper, f.producer) for f in two.factors)
    assert one.predicates == two.predicates
    assert len(one.banks) == len(two.banks)
    for old, new in zip(one.banks, two.banks):
        assert old == new


def test_default_off_and_hz_embedding():
    with pytest.raises(DomainError):
        Fiber.box(((-1, 1),))
    eq = Predicate(Form(0, ((0, Q(1)), (2, -Q(1)))), "eq")
    le = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
                     predicates=(eq, le), decoder_matrix=((2, -1),),
                     decoder_bias=(Q(1, 3),), enabled=True)
    point = root.complete((1, 0), (1,))
    assert root.contains(point) and root.decode(point) == (Q(7, 3),)
    assert root.evaluate(root.signed_phases(), point) == (1,)
    assert root.predicates[0] is eq and root.predicates[1] is le
    assert not root.contains((0, 0, 1))
    with pytest.raises(DomainError):
        Fiber(root._frame, root.factors, root.predicates)
    report = root.cost_report()
    assert report["shared_owner_frames"] is True and report["owner_depth"] == 0
    assert report["query_prefix_identity_copies"] == report["query_prefix_predicate_copies"] == 0
    assert report["fixed_query_certificates"] == 4 and report["formal_gain"] == 0


def test_all_four_bounds_match_d209():
    recorded = {}
    for positive in (False, True):
        old, new = _control(PreviousFiber, positive), _control(Fiber, positive)
        _state_shape_equal(old["second"], new["second"])
        for name in ("j", "child_pre"):
            for direction in ((1,), (-1,)):
                assert old["second"].support_certificates(old[name], direction) == new["second"].support_certificates(new[name], direction)
        for direction in DIRECTIONS:
            assert old["second"].support_certificates(old["second_q"], direction) == new["second"].support_certificates(new["second_q"], direction)
        bounds = new["second"].support_certificates(new["child_pre"], (1,))
        assert len(bounds) == 4 and bounds[-1] == (-Q(1, 32) if positive else -Q(1, 8))
        final_old, child_old = old["second"].relu(old["child_pre"], "child")
        final_new, child_new = new["second"].relu(new["child_pre"], "child")
        packet_old, packet_new = _complete_packet(old, final_old, child_old), _complete_packet(new, final_new, child_new)
        assert packet_old.forms == packet_new.forms and len(packet_new.forms) == 15
        _state_shape_equal(final_old, final_new)
        for point in POINTS:
            assignment_old, assignment_new = final_old.complete(point), final_new.complete(point)
            assert assignment_old == assignment_new
            assert final_old.evaluate(packet_old, assignment_old) == final_new.evaluate(packet_new, assignment_new)
            assert final_new.evaluate(child_new, assignment_new) == (0,)
        recorded["positive_cap" if positive else "nonpositive_cap"] = {
            "four_bounds": [str(v) for v in bounds], "all_four_equal_to_d209": True,
            "full_ports": 15, "retained_bits": 5, "full_predicates_equal": True,
        }
    _record("pair_equivalence.json", {
        "controls": recorded, "equivalence_scope": "same complete supplied states and readouts",
        "new_precision_over_d209": False, "actual_model": False,
        "native_qualified": False, "gpu_qualified": False, "formal_gain": 0,
    })


def test_positive_cap_pair_equivalence():
    control = _control(Fiber, True)
    first, second = control["first"], control["second"]
    closure = second._proof_banks[-1].closure_pairs[0]
    assert tuple(p.bound for p in closure.parent_proofs) == (Q(7, 8), Q(5, 2))
    assert closure.common_box_upper == Q(1481, 4096) and closure.tau == 1
    assert closure.pair.lower == Q(2615, 4096)
    assert closure.common_form == second.affine(control["j"], ((-Q(1, 64),),)).forms[0]
    assert all(second.verify_proof(p) for p in closure.parent_proofs + closure.scale_proofs)
    source = (Q(9, 10), -Q(2, 5), Q(1, 24))
    true = second.complete(source)
    assert second.evaluate(control["second_pre"], true) == (Q(29399, 122880), -Q(289, 10240))
    ell, ci = Q(884539, 610304), Q(7, 8)
    q1 = ci * (control["second_pre"].forms[0].at(true) + ell) / (ci + ell)
    beta1 = q1 / ci
    gates, fake = second.banks[-1].gates, list(true)
    fake[gates[0].q_index], fake[gates[1].q_index] = q1, Q(0)
    fake[gates[0].phase_index], fake[gates[1].phase_index] = 2 * beta1 - 1, Q(-1)
    meta = second._proof_banks[-1]
    indices = tuple(i for rows in meta.gate_rows + meta.old_pair_rows for i in rows)
    assert len(indices) == 18 and all(_holds(second.predicates[i], fake) for i in indices)
    assert all(_holds(row, fake) for row in first.predicates)
    assert all(f.lower <= v <= f.upper for f, v in zip(second.factors, fake))
    assert min(q1, Q(0), Q(17, 6) * beta1 - q1, Q(0),
               ci * beta1 - q1, q1 / 2) >= 0
    assert not _holds(second.predicates[closure.row_indices[4]], fake)
    assert control["witness"].forms[0].at(fake) > Q(1, 600)
    assert control["child_pre"].forms[0].at(fake) > Q(53, 2400)
    assert fake[gates[0].phase_index] not in (-1, 1) and not second.contains(fake)
    proofs = second.support_proofs(control["child_pre"], (1,))
    assert all(p.bound > Q(53, 2400) for p in proofs[:3])
    assert proofs[3].bound == -Q(1, 32)
    assert all(p.owner is second and second.verify_proof(p) for p in proofs)
    final, child = second.relu(control["child_pre"], "positive_child")
    assert final.bounds(child) == (0, 0)
    assert all(final.evaluate(child, final.complete(point)) == (0,) for point in POINTS)


def test_negative_cap_closure_equivalence():
    old, new = _control(PreviousFiber, False), _control(Fiber, False)
    second = new["second"]
    closure = second._proof_banks[-1].closure_pairs[0]
    assert closure.parent_proofs[0].bound == -Q(1, 4) and closure.pair.caps[0] == 0
    assert closure.pair.scale == Form(1) and not closure.common_weights
    bounds = second.support_certificates(new["child_pre"], (1,))
    assert bounds == old["second"].support_certificates(old["child_pre"], (1,))
    assert all(value >= Q(1, 4) for value in bounds[:3]) and bounds[3] == -Q(1, 8)
    final, child = second.relu(new["child_pre"], "order_child")
    assert final.bounds(child) == (0, 0) and type(final) is Fiber
    assert len(final.factors) == 13 and sum(f.kind == "phase" for f in final.factors) == 5
    zero = final.complete((Q(1, 2), -Q(1, 4), -Q(1, 4)))
    for gate in (final.banks[0].gates[0], final.banks[1].gates[1]):
        assert gate.input.at(zero) == 0
        alternate = list(zero)
        alternate[gate.phase_index] = 1
        assert final.contains(alternate)


def test_shared_owner_preserves_original_identity():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    sources = root.sources()
    alias, view = root.alias(sources, ("a0", "a1"))
    state, q = alias.relu_bank(view, ("g0", "g1"))
    assert sources.owner is root and view.owner is alias and q.owner is state
    assert not hasattr(sources, "identities")
    proof = state.support_proof(q, (1, -1))
    assert proof.owner is state and proof.frame is state._frame
    assert not hasattr(proof, "identities") and not hasattr(proof, "predicates")
    assert all(a.identity is b.identity for a, b in zip(root.factors, state.factors))
    assert all(a is b for a, b in zip(alias.predicates, state.predicates))
    for p in state.support_proofs(q, (1, -1)):
        assert p.owner is state and state.verify_proof(p)
    assert state.cost_report()["owner_depth"] == 2
    assert state.evaluate(sources, state.complete((Q(1, 3), -Q(1, 2)))) == (Q(1, 3), -Q(1, 2))


def test_foreign_and_sibling_proofs_rejected():
    root = Fiber.box(((-1, 1),), enabled=True)
    source = root.sources()
    left, lv = root.alias(source, ("left",))
    right, rv = root.alias(source, ("right",))
    proof = left.support_proof(lv, (1,))
    with pytest.raises(DomainError):
        right.verify_proof(proof)
    with pytest.raises(DomainError):
        right.support(lv, (1,))
    with pytest.raises(DomainError):
        left.concat(lv, rv)
    foreign = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(DomainError):
        foreign.verify_proof(proof)
    with pytest.raises(DomainError):
        left.sparse_affine(foreign.sources(), (((0, 1),),))
    with pytest.raises(DomainError):
        left.verify_proof(replace(proof, owner=right))
    root_proof = root.support_proof(source, (1,))
    assert left.verify_proof(root_proof) and right.verify_proof(root_proof)


def test_modified_and_missing_proofs_rejected():
    root = Fiber.box(((-1, 1),), enabled=True)
    proof = root.support_proof(root.sources(), (1,))
    key, weight = proof.le_weights[0]
    invalids = (replace(proof, bound=proof.bound + 1),
                replace(proof, subject=Form()), replace(proof, le_weights=()),
                replace(proof, le_weights=((key, -weight),)),
                replace(proof, le_weights=((key, weight), (key, weight))),
                replace(proof, le_weights=((("le", 0), weight),)),
                replace(proof, frame=object()), replace(proof, owner=object()))
    for invalid in invalids:
        with pytest.raises(DomainError):
            root.verify_proof(invalid)
    with pytest.raises(DomainError):
        root.verify_proof(None)
    alias, view = root.alias(root.sources(), ("alias",))
    eqproof = alias.support_proof(view, (1,))
    assert eqproof.eq_weights
    with pytest.raises(DomainError):
        alias.verify_proof(replace(eqproof, eq_weights=()))
    with pytest.raises(DomainError):
        alias.verify_proof(replace(eqproof, eq_weights=((0, -eqproof.eq_weights[0][1]),)))


def test_ancestor_proof_keeps_birth_bounds():
    root = Fiber.box(((-1, 1),), enabled=True)
    source = root.sources()
    root_proof = root.support_proof(source, (1,))
    alias, av = root.alias(root.affine(source, ((2,),), (1,)), ("alias",))
    alias_proof = alias.support_proof(av, (1,))
    final, q = alias.relu(av, "relu")
    assert final.verify_proof(root_proof) and final.verify_proof(alias_proof)
    assert root_proof.owner is root and alias_proof.owner is alias
    assert root_proof.bound == 1 and alias_proof.bound == 3
    assert final.evaluate(q, final.complete((-1,))) == (0,)
    # A descendant cannot grant an ancestor proof access to later rows/factors.
    with pytest.raises(DomainError):
        final.verify_proof(replace(root_proof, subject=q.forms[0]))
    with pytest.raises(DomainError):
        final.verify_proof(replace(root_proof, le_weights=((("hi", 1), Q(1)),)))
    with pytest.raises(DomainError):
        final.verify_proof(replace(root_proof, le_weights=(), eq_weights=((0, Q(1)),)))
    with pytest.raises(DomainError):
        root.verify_proof(alias_proof)


def test_exact_alias_equivalence_and_signed_eq():
    states = []
    for cls in (PreviousFiber, Fiber):
        root = cls.box(((-1, 1), (-1, 1)), enabled=True)
        original = root.affine(root.sources(), ((2, -1), (1, 3)), (1, -Q(1, 4)))
        first, av = root.alias(original, ("a0", "a1"))
        second, bv = first.alias(av, ("b0", "b1"))
        states.append((root, original, second, bv))
    for direction in DIRECTIONS:
        old, new = states
        assert old[2].support_certificates(old[3], direction) == new[2].support_certificates(new[3], direction)
        assert new[2].support(new[3], direction) == new[0].support(new[1], direction)
        assert all(new[2].verify_proof(p) for p in new[2].support_proofs(new[3], direction))
    state, view = states[1][2:]
    for direction, expected in (((1, 0), 1), ((-1, 0), -1)):
        for proof in state.support_proofs(view, direction):
            assert len(proof.eq_weights) == 2
            assert all(weight == expected for _, weight in proof.eq_weights)
    point = state.complete((Q(1, 3), -Q(1, 2)))
    damaged = list(point)
    damaged[-1] += Q(1, 100)
    assert not state.contains(damaged)


def test_sparse_affine_matches_dense():
    root = Fiber.box(((-1, 1),) * 4, enabled=True)
    source = root.sources()
    rows = (((0, 2), (2, -Q(3, 2))), (), ((1, -1), (3, Q(2, 5))))
    dense = ((2, 0, -Q(3, 2), 0), (0, 0, 0, 0), (0, -1, 0, Q(2, 5)))
    bias = (Q(1, 7), -2, Q(3, 11))
    sparse_view, dense_view = root.sparse_affine(source, rows, bias), root.affine(source, dense, bias)
    assert sparse_view.forms == dense_view.forms and sparse_view.owner is root
    assert root.sparse_affine(source, ((),)).forms == root.affine(source, ((0, 0, 0, 0),)).forms
    assert root.sparse_affine(source, (), ()).forms == ()
    for direction in ((1, 1, 1), (-2, 3, -1)):
        assert root.support_certificates(sparse_view, direction) == root.support_certificates(dense_view, direction)
    # Input-index sparsity and underlying latent-index cancellation are distinct.
    duplicate_values = root.concat(root.select(source, (0,)), root.select(source, (0,)))
    cancelled = root.sparse_affine(duplicate_values, (((0, 1), (1, -1)),))
    assert cancelled.forms == (Form(),) and root.support(cancelled, (1,)) == 0
    first, alias = root.alias(sparse_view, ("s0", "s1", "s2"))
    again = first.sparse_affine(alias, (((0, 2), (2, -1)),))
    dense_again = first.affine(alias, ((2, 0, -1),))
    assert again.forms == dense_again.forms
    for point in ((0, 0, 0, 0), (Q(1, 2), -Q(1, 3), Q(2, 3), -Q(3, 4))):
        assignment = first.complete(point)
        assert first.evaluate(sparse_view, assignment) == first.evaluate(dense_view, assignment)
        assert first.evaluate(again, assignment) == first.evaluate(dense_again, assignment)


def test_sparse_affine_rejects_invalid_rows():
    root = Fiber.box(((-1, 1),) * 2, enabled=True)
    source = root.sources()
    invalid_rows = (
        (((0, 1), (0, 2)),), (((1, 1), (0, 2)),), (((2, 1),),),
        (((-1, 1),),), (((True, 1),),), (((0.0, 1),),),
        (((0, 0),),), (((0, 0.5),),), (((0, True),),),
        (((0, Q(1, 1 << 513)),),),
    )
    for rows in invalid_rows:
        with pytest.raises(DomainError):
            root.sparse_affine(source, rows)
    for bias in ((), (0, 0), (0.5,), (True,)):
        with pytest.raises(DomainError):
            root.sparse_affine(source, (((0, 1),),), bias)
    foreign = Fiber.box(((-1, 1),) * 2, enabled=True)
    with pytest.raises(DomainError):
        root.sparse_affine(foreign.sources(), (((0, 1),),))


def test_full_interfaces_bits_and_decoder():
    eq = Predicate(Form(0, ((0, Q(1)), (2, -Q(1)))), "eq")
    le = Predicate(Form(0, ((1, Q(1)),)), "le", Q(1, 2))
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("initial",),
                     predicates=(eq, le), decoder_matrix=((2, -1), (-3, 4)),
                     decoder_bias=(Q(1, 3), -Q(1, 7)), enabled=True)
    source = root.sources()
    pre = root.sparse_affine(source, (((0, 1), (1, 3)), ((0, 3), (1, 4))), (Q(1, 4),) * 2)
    alias, av = root.alias(pre, ("f0", "f1"))
    state, q = alias.relu_bank(av, ("g0", "g1"))
    packet = state.concat(source, root.signed_phases(), pre, av, q)
    assert len(packet.forms) == 9
    rows = tuple(((i, 1),) for i in range(9))
    opposite = tuple(((i, -1),) for i in range(9))
    zero = state.add(state.sparse_affine(packet, rows), state.sparse_affine(packet, opposite))
    assert state.support(zero, (1,) * 9) == 0
    assert state.predicates[0] is eq and state.predicates[1] is le
    assert all(a.identity is b.identity for a, b in zip(root.factors, state.factors))
    assert state._frame.decoder is root._frame.decoder
    for x, y, sigma in ((1, 0, 1), (-1, Q(1, 2), -1)):
        point = state.complete((x, y), (sigma,))
        assert state.contains(point) and state.evaluate(zero, point) == (0,) * 9
        assert state.decode(point) == (2 * x - y + Q(1, 3), -3 * x + 4 * y - Q(1, 7))
        selected = state.select(packet, (8, 0, 4, 2))
        values = state.evaluate(packet, point)
        assert state.evaluate(selected, point) == tuple(values[i] for i in (8, 0, 4, 2))


def test_shared_frame_cost_strictly_decreases():
    reports, values, predicates = [], [], []
    for cls in (PreviousFiber, Fiber):
        root = cls.box(((-1, 1),) * 64, enabled=True)
        source = root.sources()
        matrix = ((1, 3) + (0,) * 62, (3, 4) + (0,) * 62)
        pre = root.affine(source, matrix, (Q(1, 4),) * 2)
        state, q = root.relu_bank(pre, ("g0", "g1"))
        values.append(tuple(state.support_certificates(q, direction) for direction in DIRECTIONS))
        point = state.complete((0,) * 64)
        assert state.contains(point) and state.evaluate(q, point) == (Q(1, 4),) * 2
        assert state.decode(point) == (0,) * 64
        predicates.append(state.predicates)
        reports.append(state.cost_report())
    old, new = reports
    assert values[0] == values[1] and predicates[0] == predicates[1]
    for key in ("factors", "continuous", "binary", "predicates", "predicate_nnz",
                "banks", "pairs", "closure_pairs", "closure_row_references",
                "stored_parent_proofs", "stored_scale_proofs", "fixed_query_certificates"):
        assert old[key] == new[key]
    assert new["entries"] < old["entries"] and new["work_used"] < old["work_used"]
    assert new["whole_work_cap"] == old["whole_work_cap"] == MAX_WORK
    assert new["branch_work_cap"] == old["branch_work_cap"] == MAX_BRANCH_WORK
    assert new["entries_cap"] == old["entries_cap"] == MAX_ENTRIES
    assert new["query_prefix_identity_copies"] == new["query_prefix_predicate_copies"] == 0
    _record("shared_frame_cost.json", {
        "scope": "identical complete supplied 64-source exercise and six full query directions",
        "d209": old, "d211": new, "identical_four_bounds_and_predicates": True,
        "strict_logical_entries_and_work_decrease": True,
        "resource_caps_changed": False, "actual_model": False,
        "physical_qualified": False, "gpu_qualified": False, "formal_gain": 0,
    })


def test_wide_source_complete_two_hundred_gates():
    root = Fiber.box(((-1, 1),) * 3072, enabled=True)
    source = root.sources()
    rows = tuple(((2 * (i // 2), 1 if i % 2 == 0 else 3),
                  (2 * (i // 2) + 1, 3 if i % 2 == 0 else 4),
                  (2 * (i // 2) + 2, Q(1, 2) if i % 2 == 0 else -Q(1, 2)))
                 for i in range(200))
    bias = tuple(Q(i % 7 - 3, 16) for i in range(200))
    pre = root.sparse_affine(source, rows, bias)
    assert len(pre.forms) == 200 and all(len(form.terms) == 3 for form in pre.forms)
    state, q = root.relu_bank(pre, tuple("wide%d" % i for i in range(200)))
    assert len(q.forms) == 200 and len(state.factors) == 3472
    assert sum(f.kind == "phase" for f in state.factors) == 200
    assert len({id(f.identity) for f in state.factors}) == 3472
    zero = state.complete((0,) * 3072)
    actual = state.evaluate(q, zero)
    assert actual == tuple(max(0, offset) for offset in bias)
    assert state.decode(zero) == (0,) * 3072
    all_bounds = []
    for i in range(200):
        scalar = state.select(q, (i,))
        bounds = state.support_certificates(scalar, (1,))
        assert len(bounds) == 4 and min(bounds) >= actual[i]
        all_bounds.append(bounds)
    complete = state.concat(source, q)
    assert len(complete.forms) == 3272 and len(state.evaluate(complete, zero)) == 3272
    for direction in ((1,) * 200, tuple(1 if i % 3 else -2 for i in range(200))):
        proofs = state.support_proofs(q, direction)
        target = sum(a * b for a, b in zip(direction, actual))
        assert all(p.bound >= target and state.verify_proof(p) for p in proofs)
    report = state.cost_report()
    assert report["continuous"] == 3272 and report["binary"] == 200
    assert report["banks"] == 1 and report["pairs"] == report["closure_pairs"] == 100
    assert report["stored_parent_proofs"] == report["stored_scale_proofs"] == 200
    assert report["closure_row_references"] == 1000
    assert report["predicates"] >= 1800 and report["predicate_nnz"] >= 4600
    assert report["query_prefix_identity_copies"] == report["query_prefix_predicate_copies"] == 0
    assert report["work_used"] == report["branch_work_used"] <= MAX_BRANCH_WORK
    assert report["entries"] <= MAX_ENTRIES
    assert not any(report[key] for key in ("native_qualified", "physical_qualified",
                                          "actual_model_qualified", "gpu_qualified",
                                          "new_set_class_qualified"))
    _record("sparse_affine_control.json", {
        "scope": "supplied 3072-source synthetic sparse bank; complete 200 outputs, not a real CNN bank",
        "not_actual_14400_gate_admission": True, "original_sources": 3072,
        "complete_gates": 200, "input_stencil_nnz": 600,
        "bias_classes": 7, "preserved_original_bits": 200,
        "per_output_four_certificate_queries": 200,
        "per_output_four_bounds": [[str(v) for v in bounds] for bounds in all_bounds],
        "complete_live_ports": 3272, "cost": report,
        "actual_model": False, "native_qualified": False,
        "gpu_qualified": False, "formal_gain": 0,
    })


def test_cumulative_budgets_fail_closed():
    assert (MAX_WORK, MAX_BRANCH_WORK, MAX_ENTRIES, MAX_BITS) == (256_000_000, 200_000_000, 64_000_000, 512)
    for overrides in ({"max_work": MAX_WORK + 1}, {"max_branch_work": MAX_BRANCH_WORK + 1},
                      {"max_entries": MAX_ENTRIES + 1}, {"max_work": True}):
        with pytest.raises(DomainError):
            Fiber.box(((-1, 1),), enabled=True, **overrides)
    with pytest.raises(BudgetError):
        Fiber.box(((-1, 1),), enabled=True, max_entries=1)
    root = Fiber.box(((-1, 1),), enabled=True, max_branch_work=3000)
    source = root.sources()
    child, view = root.alias(source, ("alias",))
    assert child._work is root._work
    with pytest.raises(BudgetError):
        for _ in range(1000):
            child.support_proofs(view, (1,))
    assert root._work.failed
    with pytest.raises(BudgetError):
        root.support(source, (1,))
    with pytest.raises(BudgetError):
        child.sparse_affine(view, (((0, 1),),))
    for value in (True, 0.5, Q(1, 1 << 513)):
        with pytest.raises(DomainError):
            Form(value)


def test_stable_zero_and_unshared_boundaries():
    for cls in (PreviousFiber, Fiber):
        root = cls.box(((-1, 1),), enabled=True)
        pre = root.affine(root.sources(), ((1,), (1,), (1,), (0,)), (0, 2, -2, 0))
        state, q = root.relu_bank(pre, ("cross", "positive", "negative", "zero"))
        assert len(state.predicates) == 16 and sum(f.kind == "phase" for f in state.factors) == 4
        point = state.complete((0,))
        for position in (0, 3):
            alternate = list(point)
            alternate[state.banks[0].gates[position].phase_index] = 1
            assert state.contains(alternate)
        assert state.evaluate(q, point) == (0, 2, 0, 0)
    certificates = []
    for cls in (PreviousFiber, Fiber):
        root = cls.box(((-1, 1), (-1, 1)), enabled=True)
        pre = root.affine(root.sources(), ((Q(2, 3), 0), (-Q(2, 3), -2)), (Q(1, 2),) * 2)
        state, q = root.relu_bank(pre, ("unshared0", "unshared1"))
        closure = state._proof_banks[0].closure_pairs[0]
        assert closure.common_form == Form() and not closure.common_weights
        assert closure.common_box_upper == 0 and closure.tau == 1
        assert closure.pair.scale == Form(1) and closure.pair.lower == 1
        certificates.append(tuple(state.support_certificates(q, direction) for direction in DIRECTIONS))
    assert certificates[0] == certificates[1]
