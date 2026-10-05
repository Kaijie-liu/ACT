"""Four exact-rational curvature controls, with no new numerical solver calls.

The pair witnesses certify membership in the intersection of the ten labelled
two-gate hulls, not in a full five-gate hull. Fractional comparison points are
not concrete inputs or attacks. All source identities and old signed bits stay.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product as cartesian_product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct


RESULTS = Path(__file__).resolve().parents[2] / 'results'
KNOTS = (F(0), F(1, 4), F(1, 2), F(3, 4), F(1))


def _col(index):
    return ef.Form(F(0), ((index, F(1)),))


def _evaluate(form, point):
    return form.bias + sum((coefficient*point[index]
                            for index, coefficient in form.terms), F(0))


def _holds(system, point):
    return (len(point) == len(system.bounds)
            and all(lo <= value <= hi for value, (lo, hi) in zip(point, system.bounds))
            and all(_evaluate(row, point) == 0 for row in system.eq)
            and all(_evaluate(row, point) <= 0 for row in system.le))


def _qindex(gate):
    assert gate.q.bias == 0 and len(gate.q.terms) == 1
    index, coefficient = gate.q.terms[0]
    assert coefficient == 1
    return index


def _gate(system, preactivation):
    lo, hi = ef.box(system, preactivation)
    system, q = ef.variable(system, max(F(0), lo), max(F(0), hi))
    bit = len(system.bounds)
    system, signed = ef.variable(system, F(-1), F(1))
    active = (signed+1)*F(1, 2)
    rows = (-q, preactivation-q, q-max(F(0), hi)*active,
            q-preactivation+min(F(0), lo)*(1-active))
    system = replace(system, binary=system.binary+(bit,), le=system.le+rows)
    return system, ct.Gate(preactivation, q, bit, lo, hi)


def _fixture(eps=F(0)):
    width = 3 if eps else 2
    system = ef.System(((F(-1), F(1)),)*width, (), (ef.Form(),), (), 11900+width)
    x, y = _col(0), _col(1)
    z = _col(2) if eps else ef.Form()
    f, h = x+y*F(1, 5)+F(1, 10), -x*F(1, 4)+y+F(1, 5)
    forms = (f, f*F(3, 4)+h*F(1, 4)+eps*z,
             (f+h)*F(1, 2), f*F(1, 4)+h*F(3, 4)-eps*z, h)
    gates = []
    for preactivation in forms:
        system, gate = _gate(system, preactivation)
        gates.append(gate)
    gates = tuple(gates)
    p, q1, _, q3, q = tuple(gate.q for gate in gates)
    Y = (p+q)*F(1, 2)+(f+h)*F(1, 4)-q1-q3
    system, child = _gate(system, Y+(f+h)*F(1, 4)+F(1, 2))
    return system, gates, child, f, h, Y, width


def _canonical(system, gates, source, zero_labels=None):
    point = list(source)
    choices = {} if zero_labels is None else zero_labels
    for gate in gates:
        assert _qindex(gate) == len(point) and gate.bit == len(point)+1
        preactivation = _evaluate(gate.g, point)
        point.append(max(F(0), preactivation))
        signed = F(1) if preactivation > 0 else F(-1)
        if preactivation == 0:
            signed = choices.get(gate.bit, F(-1))
            assert signed in (F(-1), F(1))
        point.append(signed)
    assert _holds(system, point)
    return tuple(point)


def _source(f, h, width):
    values = (F(20, 21)*f-F(4, 21)*h-F(2, 35),
              F(5, 21)*f+F(20, 21)*h-F(3, 14))
    return values + ((F(0),) if width == 3 else ())


def _fractional(gates, child, width):
    point = list(_source(F(0), F(0), width))
    for gate, value in zip(gates, (F(1, 5), F(1, 20), F(1, 20), F(1, 20), F(1, 5))):
        assert _qindex(gate) == len(point)
        point.extend((value, F(0)))
    assert _qindex(child) == len(point)
    point.extend((F(3, 5), F(1)))
    return tuple(point)


def _active_run():
    path = Path(os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'])
    if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
            or path.resolve() != path or not path.is_relative_to(RESULTS)
            or path == RESULTS):
        raise ValueError('untrusted active component evidence directory')
    return path


def _json_fraction(value):
    if isinstance(value, F):
        return [value.numerator, value.denominator]
    raise TypeError('non-plain control evidence')


def test_curvature_canonical_retention():
    original, gates, child, f, h, _, width = _fixture()
    extended, receipt = ct.append_curvature(
        original, gates, KNOTS, frames=(original.frame,)*len(gates), enabled=True)
    assert extended.bounds == original.bounds and extended.binary == original.binary
    assert extended.eq == original.eq and extended.le[:len(original.le)] == original.le
    assert receipt['new_columns'] == receipt['new_eq'] == 0
    assert receipt['new_le'] == 4 and len(extended.le)-len(original.le) == 4
    assert receipt['nnz'] == sum(len(row.terms) for row in extended.le[len(original.le):]) == 15
    assert receipt['entries'] == ef.entries(extended)
    assert tuple(receipt['residual_boxes']) == ((F(0), F(0)),)*3
    expected_budget = 2*gates[0].q+2*gates[4].q+f+h-4*gates[1].q-4*gates[3].q
    assert receipt['budget_row'] == expected_budget == extended.le[-1]
    assert receipt['native_binding_qualified'] is False
    assert receipt['complete_physical_qualification'] is False
    lowered, child_receipt = ct.append_budget_child(
        extended, child, tuple(g.q for g in gates), receipt['budget_row'],
        frame=original.frame, enabled=True)
    assert lowered.bounds == original.bounds and lowered.binary == original.binary
    assert lowered.eq == original.eq and lowered.le[:len(extended.le)] == extended.le
    assert child_receipt['lambda'] == F(1, 4)
    assert child_receipt['upper'] == (f+h)*F(1, 4)+F(1, 2)
    assert child_receipt['row'] == child.q-child_receipt['upper'] == lowered.le[-1]
    for x, y in cartesian_product((F(-1), F(0), F(1)), repeat=2):
        point = _canonical(original, gates+(child,), (x, y))
        assert _holds(lowered, point)
    # Exactly the zero preactivation face: every original zero label is legal.
    for signs in cartesian_product((F(-1), F(1)), repeat=len(gates)):
        choices = dict(zip((g.bit for g in gates), signs))
        point = _canonical(original, gates+(child,), _source(F(0), F(0), width), choices)
        assert all(_evaluate(g.g, point) == 0 for g in gates)
        assert tuple(point[g.bit] for g in gates) == signs
        assert _holds(lowered, point)
    for sign in (F(-1), F(1)):
        point = _canonical(original, gates+(child,), _source(F(-1, 2), F(-1, 2), width),
                           {child.bit: sign})
        assert _evaluate(child.g, point) == 0 and point[child.bit] == sign
        assert _holds(lowered, point)


def test_curvature_contract_rejections():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected '+name)

    poison = Poison()
    result, receipt = ct.append_curvature(
        poison, poison, poison, frames=poison, enabled=False, max_entries=poison)
    assert result is poison and receipt == {'enabled': False}
    result, receipt = ct.append_budget_child(
        poison, poison, poison, poison, frame=poison, enabled=False, max_entries=poison)
    assert result is poison and receipt == {'enabled': False}
    original, gates, child, _, _, _, _ = _fixture()
    good = dict(frames=(original.frame,)*len(gates), enabled=True)
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            ct.append_curvature(original, gates, KNOTS, **dict(good, enabled=flag))
        with pytest.raises(ValueError):
            ct.append_budget_child(original, child, tuple(g.q for g in gates), ef.Form(),
                                   frame=original.frame, enabled=flag)
    for frames in ((original.frame,)*4, (0,)*5, (True,)*5):
        with pytest.raises(ValueError):
            ct.append_curvature(original, gates, KNOTS, **dict(good, frames=frames))
    for knots in (KNOTS[:-1], (F(0), F(1, 4), F(1, 4), F(3, 4), F(1)),
                  (F(-1),)+KNOTS[1:], KNOTS[:-1]+(F(2),)):
        with pytest.raises(ValueError):
            ct.append_curvature(original, gates, knots, **good)
    with pytest.raises(ValueError):
        ct.append_curvature(replace(original, le=original.le[1:]), gates, KNOTS, **good)
    with pytest.raises(ValueError):
        ct.append_curvature(replace(original, binary=original.binary[1:]), gates, KNOTS, **good)
    for bad in (replace(gates[0], bit=True), replace(gates[0], hi=F(1 << 512))):
        with pytest.raises(ValueError):
            ct.append_curvature(original, (bad,)+gates[1:], KNOTS, **good)
    for cap in (0, 1, True, 64_000_001):
        with pytest.raises(ValueError):
            ct.append_curvature(original, gates, KNOTS, **good, max_entries=cap)
    extended, proof = ct.append_curvature(original, gates, KNOTS, **good)
    with pytest.raises(ValueError):
        ct.append_budget_child(original, child, tuple(g.q for g in gates), proof['budget_row'],
                               frame=original.frame, enabled=True)
    with pytest.raises(ValueError):
        ct.append_budget_child(extended, child, tuple(g.q for g in gates), proof['budget_row'],
                               frame=original.frame+1, enabled=True)
    with pytest.raises(ValueError):
        ct.append_budget_child(extended, child, tuple(g.q for g in gates), proof['budget_row'],
                               frame=original.frame, enabled=True, max_entries=1)
    negative, negative_child = _gate(extended, child.g-1)
    with pytest.raises(ValueError):
        ct.append_budget_child(negative, negative_child, tuple(g.q for g in gates), proof['budget_row'],
                               frame=negative.frame, enabled=True)
    for knots, weights in (((F(1, 2), F(1, 2)), (F(1), F(1))),
                           ((F(1, 2),), (F(1 << 512),))):
        with pytest.raises(ValueError):
            ct.support(knots, weights)
    assert original == _fixture()[0]


def test_curvature_pair_hull_forward():
    report = dict(status='started', solver_calls=0, controls=[],
                  full_group_hull_claim=False, trained_model_qualified=False,
                  native_binding_qualified=False, formal_gain=0)
    # Exclusive active-run output, including assertion failures; never an old RUN.
    with (_active_run() / 'curvature_control.json').open('x', encoding='utf-8') as stream:
        try:
            pairs = ((0, 4, F(2, 5), F(2, 5)),
                     (0, 1, F(2, 5), F(-4, 5)),
                     (0, 2, F(2, 5), F(-1, 5)),
                     (0, 3, F(2, 5), F(0)),
                     (1, 4, F(0), F(2, 5)),
                     (2, 4, F(-1, 5), F(2, 5)),
                     (3, 4, F(-4, 5), F(2, 5)),
                     (1, 2, F(1, 10), F(1, 10)),
                     (1, 3, F(1, 10), F(1, 10)),
                     (2, 3, F(1, 10), F(1, 10)))
            assert {(i, j) for i, j, _, _ in pairs} == {
                (i, j) for i in range(5) for j in range(i+1, 5)}
            for eps in (F(0), F(1, 20)):
                original, gates, child, f, h, Y, width = _fixture(eps)
                fake = _fractional(gates, child, width)
                assert _holds(original, fake)
                assert _evaluate(child.g, fake) == fake[_qindex(child)] == F(3, 5)
                assert fake[child.bit] == 1 and _evaluate(Y, fake) == F(1, 10)
                pair_evidence = []
                for i, j, fstar, hstar in pairs:
                    plus = _canonical(original, gates+(child,), _source(fstar, hstar, width))
                    minus = _canonical(original, gates+(child,), _source(-fstar, -hstar, width))
                    for point in (plus, minus):
                        assert all(F(-1) < point[k] < F(1) for k in range(width))
                    columns = tuple(range(width))+tuple(
                        index for k in (i, j) for index in (_qindex(gates[k]), gates[k].bit))
                    assert all((plus[k]+minus[k])/2 == fake[k] for k in columns)
                    assert all(_evaluate(gates[k].g, plus) > 0
                               and _evaluate(gates[k].g, minus) < 0 for k in (i, j))
                    pair_evidence.append(dict(pair=(i, j), fstar=fstar, hstar=hstar,
                                              source_plus=plus[:width], source_minus=minus[:width],
                                              weights=(F(1, 2), F(1, 2))))
                S = 2*gates[0].q+2*gates[4].q-f-h
                assert _evaluate(S, fake) == F(4, 5)
                for knot, gate in zip(KNOTS[1:-1], gates[1:-1]):
                    defect = (1-knot)*gates[0].q+knot*gates[4].q-gate.q
                    assert _evaluate(defect, fake) == F(3, 20)
                    assert F(0) <= _evaluate(defect, fake) <= knot*(1-knot)*_evaluate(S, fake)
                extended, proof = ct.append_curvature(
                    original, gates, KNOTS, frames=(original.frame,)*len(gates), enabled=True)
                assert tuple(proof['residual_boxes']) == ((-eps, eps), (F(0), F(0)), (-eps, eps))
                expected = 4*Y-4*eps
                assert proof['budget_row'] == expected
                assert all(_evaluate(row, fake) <= 0 for row in extended.le[len(original.le):-1])
                assert _evaluate(proof['budget_row'], fake) == F(2, 5)-4*eps > 0
                lowered, child_proof = ct.append_budget_child(
                    extended, child, tuple(g.q for g in gates), proof['budget_row'],
                    frame=original.frame, enabled=True)
                assert child_proof['lambda'] == F(1, 4)
                assert child_proof['upper'] == (f+h)*F(1, 4)+F(1, 2)+eps
                gap = _evaluate(child_proof['row'], fake)
                assert gap == F(1, 10)-eps > 0 and not _holds(lowered, fake)
                on = _canonical(original, gates+(child,), _source(F(-2, 5), F(6, 5), width))
                off_source = (F(-1), F(-1))+((F(0),) if width == 3 else ())
                off = _canonical(original, gates+(child,), off_source)
                assert _holds(lowered, on) and _holds(lowered, off)
                assert _evaluate(child.g, on) == on[_qindex(child)] == F(7, 10)
                assert _evaluate(child.g, off) == F(-13, 40) and off[_qindex(child)] == 0
                assert F(-13, 40) < F(3, 5) < F(7, 10)
                canonical_grid_count = 0
                for source in cartesian_product((F(-1), F(0), F(1)), repeat=width):
                    point = _canonical(original, gates+(child,), source)
                    assert _holds(lowered, point)
                    canonical_grid_count += 1
                assert canonical_grid_count == (27 if eps else 9)
                parent_zero_labels = 0
                for signs in cartesian_product((F(-1), F(1)), repeat=len(gates)):
                    choices = dict(zip((gate.bit for gate in gates), signs))
                    point = _canonical(original, gates+(child,), _source(F(0), F(0), width), choices)
                    assert all(_evaluate(gate.g, point) == 0 for gate in gates)
                    assert tuple(point[gate.bit] for gate in gates) == signs
                    assert _holds(lowered, point)
                    parent_zero_labels += 1
                child_zero_labels = 0
                for sign in (F(-1), F(1)):
                    point = _canonical(original, gates+(child,),
                                       _source(F(-1, 2), F(-1, 2), width), {child.bit: sign})
                    assert all(_evaluate(gate.g, point) < 0 for gate in gates)
                    assert _evaluate(child.g, point) == 0 and point[child.bit] == sign
                    assert _holds(lowered, point)
                    child_zero_labels += 1
                assert parent_zero_labels == 32 and child_zero_labels == 2
                report['controls'].append(dict(epsilon=eps, labelled_pair_count=len(pairs),
                    pair_witnesses=pair_evidence, old_point=fake, false_child=F(3, 5),
                    true_child_preactivations=(F(-13, 40), F(7, 10)), child_row_violation=gap,
                    exact_child_scalar_bounds_alone_do_not_exclude=True,
                    canonical_grid_count=canonical_grid_count,
                    parent_zero_label_assignments=parent_zero_labels,
                    child_zero_label_assignments=child_zero_labels,
                    old_signed_bits_retained=True,
                    new_group_le=proof['new_le'], group_nnz=proof['nnz'],
                    group_new_columns=proof['new_columns'], child_new_le=child_proof['new_le']))
            report['status'] = 'passed'
        except BaseException as error:
            report['status'] = 'failed'
            report['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            json.dump(report, stream, default=_json_fraction, sort_keys=True, indent=2)
            stream.write('\n')


def test_curvature_prefix_support():
    cases = (((F(1, 4), F(1, 2), F(3, 4)), (F(2), F(-3), F(1))),
             ((F(1, 5), F(2, 5), F(4, 5)), (F(-2), F(3, 2), F(-1, 7))),
             ((F(1, 6), F(1, 3), F(3, 5), F(5, 6)), (F(3), F(-1), F(-2), F(4))),
             ((F(1, 2),), (F(-3, 5),)),
             ((F(1, 3), F(2, 3)), (F(0), F(0))))
    for knots, weights in cases:
        lower, upper = ct.support(knots, weights)
        values = tuple(sum((w*(min(t, k)-t*k) for t, w in zip(knots, weights)), F(0))
                       for k in (F(0),)+knots+(F(1),))
        assert (lower, upper) == (min(values), max(values))
        assert isinstance(lower, F) and isinstance(upper, F) and lower <= 0 <= upper
        reverse = ct.support(knots, tuple(-w for w in weights))
        assert reverse == (-upper, -lower)
        for f, h in cartesian_product((F(-1), F(-1, 3), F(0), F(2, 5), F(1)), repeat=2):
            p, q = max(F(0), f), max(F(0), h)
            defects = tuple((1-t)*p+t*q-max(F(0), (1-t)*f+t*h) for t in knots)
            readout = sum((w*e for w, e in zip(weights, defects)), F(0))
            S = abs(f)+abs(h)
            assert lower*S <= readout <= upper*S
