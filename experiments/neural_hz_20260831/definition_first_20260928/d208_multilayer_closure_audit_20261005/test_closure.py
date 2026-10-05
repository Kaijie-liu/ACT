"""Frozen counterexample to query precision closure, not a new verifier."""
from fractions import Fraction as Q
import json
import os
from pathlib import Path

from experiments.neural_hz_20260831.definition_first_20260928.d207_owned_source_phase_20261005.fiber import (
    Fiber, Form,
)


POINTS = ((-1, 1, -1), (-1, 1, 1), (-1, -1, -1), (-1, -1, 1),
          (0, 0, 0), (Q(1, 3), -Q(1, 2), 0), (Q(1, 2), -Q(1, 4), -Q(1, 4)))


def _prefix():
    root = Fiber.box(((-1, 1),) * 3, names=("x", "y", "t"), enabled=True)
    sources = root.sources()
    pre = root.affine(sources, ((1, 3, 0), (3, 4, 0)), (Q(1, 4), Q(1, 4)))
    scale = root.affine(sources, ((0, Q(1, 4), 0),), (Q(3, 4),))
    first, q = root.relu_bank(pre, ("first0", "first1"))
    j = first.affine(first.concat(q, first.select(pre, (0,)), scale),
                     ((Q(73, 16), -Q(15, 8), -Q(13, 16), -Q(195, 32)),))
    t = first.select(sources, (2,))
    g = first.affine(first.concat(t, j), ((Q(1, 2), Q(1, 16)), (1, 0)),
                     (-Q(1, 8), Q(1, 4)))
    return root, first, sources, pre, scale, q, j, g


def _whole():
    root, first, sources, pre, scale, q, j, g = _prefix()
    second, r = first.relu_bank(g, ("second0", "second1"))
    p = second.affine(r, ((1, -Q(1, 2)),), (-Q(1, 8),))
    final, z = second.relu(p, "child")
    interface = final.concat(sources, pre, scale, q, j, g, r, p, z)
    return root, first, second, final, interface, j, g, r, p, z


def _record(name, value):
    run = Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"])
    expected = Path(__file__).resolve().parents[2] / "results/d208_multilayer_closure_audit_20261005_v1"
    assert run == expected and run.is_dir() and not run.is_symlink()
    assert name in ("closure_counterexample.json", "direction_family.json")
    with (run / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def test_two_layer_counterexample_retains_full_interface():
    root, first, second, final, interface, j, g, r, p, z = _whole()
    assert len(interface.forms) == 15 and len(final.factors) == 13
    assert final.factors[:len(first.factors)] == first.factors
    assert final.factors[:len(second.factors)] == second.factors
    assert all(a is b for a, b in zip(first.predicates, final.predicates))
    assert final.cost_report()["binary"] == 5 and len(final.predicates) == 40
    assert all(gate.stable == 0 for gate in second.banks[-1].gates)
    for point in POINTS:
        assignment = final.complete(point)
        assert final.contains(assignment) and final.decode(assignment) == point
        assert len(final.evaluate(interface, assignment)) == 15
        assert final.evaluate(j, assignment)[0] <= 0
        assert final.evaluate(g, assignment)[0] <= final.evaluate(g, assignment)[1] / 2 - Q(1, 4)
        assert final.evaluate(r, assignment)[0] <= final.evaluate(r, assignment)[1] / 2
        assert final.evaluate(p, assignment)[0] <= -Q(1, 8)
        assert final.evaluate(z, assignment) == (0,)
    assert first.evaluate(g, first.complete((-1, 1, -1))) == (-Q(5, 8), -Q(3, 4))
    assert first.evaluate(g, first.complete((-1, 1, 1))) == (Q(3, 8), Q(5, 4))


def test_same_domain_proves_the_lost_preactivation_relation():
    root, first, sources, pre, scale, q, j, g = _prefix()
    difference = first.affine(g, ((1, -Q(1, 2)),))
    assert first.support(j, (1,)) == 0
    assert first.support(difference, (1,)) == -Q(1, 4)
    assert first.bounds(first.select(g, (0,)))[1] == Q(3, 8)
    assert first.bounds(first.select(g, (1,))) == (-Q(3, 4), Q(5, 4))
    assert len(first.predicates) == 18  # No injected J or ordering predicate.


def test_fixed_queries_lose_the_relation_at_next_relu():
    root, first, second, final, interface, j, g, r, p, z = _whole()
    pair = second.banks[-1].pairs[0]
    assert pair.caps == (Q(985, 1024), Q(3657, 2048))
    assert pair.scale == Form(1) and pair.lower == 1
    assert pair.source_h == pair.constant_h and pair.source_m == pair.constant_m
    certificates = second.support_certificates(p, (1,))
    assert len(certificates) == 3 and all(value >= Q(1, 4) for value in certificates)
    assert final.banks[-1].gates[0].stable == 0
    assert final.bounds(z)[1] > 0
    difference = first.affine(g, ((1, -Q(1, 2)),))
    _record("closure_counterexample.json", {
        "kind": "negative_precision_closure_control",
        "first_relation_upper": str(first.support(j, (1,))),
        "same_parent_difference_upper": str(first.support(difference, (1,))),
        "second_pair_box_caps": [str(v) for v in pair.caps],
        "second_pair_scale": "1",
        "child_preactivation_upper_certificates": [str(v) for v in certificates],
        "child_stable": final.banks[-1].gates[0].stable,
        "child_abstract_bounds": [str(v) for v in final.bounds(z)],
        "paper_global_child_value": "0",
        "paper_global_claim_is_not_inferred_from_fixed_points": True,
        "cost_scope": "complete supplied-control exercise including repeated queries",
        "cost": final.cost_report(), "actual_model": False,
        "gpu_qualified": False, "new_domain_implemented": False, "formal_gain": 0})


def test_direction_family_and_phase_preservation():
    root, first, second, final, interface, j, g, r, p, z = _whole()
    family = []
    for multiplier in (1, 2, 3):
        certificates = second.support_certificates(p, (multiplier,))
        assert all(value >= multiplier * Q(1, 4) for value in certificates)
        for point in POINTS:
            assert multiplier * final.evaluate(p, final.complete(point))[0] <= -multiplier * Q(1, 8)
        family.append({"multiplier": multiplier, "upper_certificates": [str(v) for v in certificates],
                       "paper_true_upper": str(-multiplier * Q(1, 8))})
    assignment = list(final.complete((Q(1, 2), -Q(1, 4), -Q(1, 4))))
    first_zero = first.banks[0].gates[0].phase_index
    second_zero = second.banks[-1].gates[1].phase_index
    assert final.evaluate(first._view((first.banks[0].gates[0].input,)), assignment) == (0,)
    assert final.evaluate(final.select(g, (1,)), assignment) == (0,)
    for index in (first_zero, second_zero):
        assignment[index] = 1
        assert final.contains(assignment)
        assignment[index] = -1
        assert final.contains(assignment)
    assignment[first_zero] = 0
    assert not final.contains(assignment)
    _record("direction_family.json", {"family": family, "retained_original_bits": 5,
        "zero_labels_preserved": True, "fractional_phase_rejected": True,
        "actual_model": False, "formal_gain": 0})
