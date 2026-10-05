"""Six exact-rational, no-solver controls of the signed phase projection.

Finite fixture grids check formulas and original zero labels.  They are not an
input/phase-splitting verifier, and no relaxed witness is reported as an ADV.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product as cartesian_product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d125_signed_phase_component_20261002 import phase_envelope as pe


ef = pe.ef


def _col(index):
    return ef.Form(F(0), ((index, F(1)),))


def _value(form, point):
    return form.bias + sum((coefficient * point[index]
                            for index, coefficient in form.terms), F(0))


def _holds(system, point, *, integral=True):
    return (len(point) == len(system.bounds)
            and all(lo <= value <= hi for value, (lo, hi) in zip(point, system.bounds))
            and (not integral or all(point[index] in (F(-1), F(1))
                                     for index in system.binary))
            and all(_value(row, point) == 0 for row in system.eq)
            and all(_value(row, point) <= 0 for row in system.le))


def _gate(system, g, bounds=None):
    lo, hi = ef.box(system, g) if bounds is None else bounds
    system, q = ef.variable(system, max(F(0), lo), max(F(0), hi))
    bit = len(system.bounds)
    system, signed = ef.variable(system, F(-1), F(1))
    active = (signed + 1) * F(1, 2)
    rows = (-q, g-q, q-max(F(0), hi)*active,
            q-g+min(F(0), lo)*(1-active))
    system = replace(system, binary=system.binary+(bit,), le=system.le+rows)
    return system, pe.Gate(g, q, bit, lo, hi)


def _fixture(*, weights=(F(1, 2), F(-1, 2)), d=None,
             preactivations=None, paper_bounds=False, extra_source=False):
    bounds = ((F(-1), F(1)),) * 2
    binary, eq, le = (), (), ()
    if extra_source:
        bounds += ((F(-2), F(2)), (F(-1), F(1)))
        binary = (3,)
        eq = (_col(2)-_col(0)-_col(1),)
        le = (_col(2)-2,)
    system = ef.System(bounds, binary, eq, le, 12501)
    x, y = _col(0), _col(1)
    d = F(1, 10)-2*x+y*F(1, 20) if d is None else d
    forms = (x+y*F(1, 10), x-y*F(1, 10)) if preactivations is None else preactivations
    parents = []
    for form in forms:
        system, gate = _gate(system, form)
        parents.append(gate)
    parents = tuple(parents)
    g = d + sum((weight*gate.q for weight, gate in zip(weights, parents)), ef.Form())
    target_bounds = (F(-41, 20), F(43, 20)) if paper_bounds else None
    system, target = _gate(system, g, target_bounds)
    return system, parents, target, d, target.q+x


def _apply(system, parents, target, readouts=(), **changes):
    kwargs = dict(tau=(0,)*len(parents), kappa=(F(2),)*len(parents),
                  readouts=readouts, frames=(system.frame,)*(len(parents)+1),
                  enabled=True)
    kwargs.update(changes)
    return pe.project(system, parents, target, **kwargs)


def _canonical(system, gates, source, labels=None):
    labels = {} if labels is None else labels
    point = list(source)
    for gate in gates:
        assert gate.q == _col(len(point)) and gate.bit == len(point)+1
        g = _value(gate.g, point)
        sign = F(1) if g > 0 else F(-1)
        if g == 0:
            sign = labels.get(gate.bit, F(-1))
        point.extend((max(F(0), g), sign))
    assert _holds(system, point)
    return tuple(point)


def _project_point(result, point):
    return tuple(point[index] for index in result.retained_ids)


def _remap(form, retained_ids):
    positions = {old: new for new, old in enumerate(retained_ids)}
    return ef.Form(form.bias, tuple((positions[index], value)
                                   for index, value in form.terms))


def test_phase_positive_forward_control():
    original, parents, target, h, readout = _fixture(paper_bounds=True)
    result = _apply(original, parents, target, (readout,))
    proof = result.receipt
    assert proof['rho'] == (F(7, 20), F(1, 4))
    assert proof['T_plus'] == proof['T_minus'] == F(1, 4)
    assert proof['R_plus'] == F(7, 80) and proof['R_minus'] == F(1, 16)
    assert proof['E_plus'] == F(7, 60) and proof['E_minus'] == F(11, 120)
    assert proof['h'] == h and proof['coefficients'] == (F(1, 2), F(-1, 2))
    assert len(original.bounds)-len(original.binary) == 5
    assert len(result.system.bounds)-len(result.system.binary) == 3
    assert (len(original.le), len(result.system.le)) == (12, 8)
    assert sum(len(row.terms) for row in original.le) == 34
    assert sum(len(row.terms) for row in result.system.le) == 23
    expected_upper = F(129, 239)*(h+F(39, 20))
    assert proof['forward_upper'] == _remap(expected_upper, result.retained_ids)
    j_upper = proof['forward_upper']+_remap(_col(0), result.retained_ids)
    assert ef.box(result.system, j_upper)[1] == F(2899, 2390)
    assert F(49, 40)-F(2899, 2390) == F(23, 1912)
    assert F(49, 40)-F(29, 24) == F(1, 60)

    # The two scalar upper lines meet here; this is a direct feasible point,
    # not an LP invocation and not a true-network counterexample.
    outer_old_coordinates = (F(-113, 120), F(1), F(0), F(-1),
                             F(0), F(-1), F(43, 20), F(1))
    outer = _project_point(result, outer_old_coordinates)
    assert _holds(result.system, outer)
    assert _value(result.readouts[0], outer) == F(29, 24)

    # Fractional old phases occur only in this explicitly marked LP diagnostic.
    old_lp = (F(1, 40), F(0), F(9, 16), F(1, 44),
              F(1, 40), F(1), F(16297, 13440), F(43, 336))
    assert _holds(original, old_lp, integral=False)
    assert not _holds(original, old_lp)
    assert _value(target.g, old_lp) == F(51, 160)
    assert _value(readout, old_lp) == F(16633, 13440)
    assert F(16633, 13440)-F(29, 24) == F(131, 4480)
    pre = _value(readout-F(49, 40), old_lp)
    assert pre == F(169, 13440) > 0
    common_bounds = (F(-89, 40), F(77, 40))
    old_next, _ = _gate(original, readout-F(49, 40), common_bounds)
    assert _holds(old_next, old_lp+(pre, F(1)), integral=False)
    new_next, _ = _gate(result.system, result.readouts[0]-F(49, 40), common_bounds)
    assert _holds(new_next, outer+(F(0), F(-1)))
    forward_next_upper = ef.box(result.system, j_upper-F(49, 40))[1]
    assert forward_next_upper == -F(23, 1912)
    assert max(F(0), forward_next_upper) == 0
    for source in cartesian_product((F(-1), F(0), F(1)), repeat=2):
        point = _canonical(original, parents+(target,), source)
        reduced = _project_point(result, point)
        assert _holds(result.system, reduced)
        assert pe.decode(result, reduced) == point
        assert _holds(new_next, reduced+(F(0), F(-1)))


def test_phase_zero_rho_labels():
    x = _col(0)
    original, parents, target, _, readout = _fixture(
        weights=(F(1, 4), F(-1, 4)), d=-2*x,
        preactivations=(x, x))
    result = _apply(original, parents, target, (readout,))
    proof = result.receipt
    assert proof['rho'] == (F(0), F(0))
    assert proof['E_plus'] == proof['E_minus'] == proof['R_minus'] == 0
    assert proof['exact_when_rho_zero'] is True
    assert len(result.system.binary) == len(original.binary) == 3
    for x_value in (F(-1), F(-1, 2), F(0), F(1, 2), F(1)):
        q = max(F(0), x_value)
        for r in (F(0), F(1, 2), F(1), F(3, 2), F(2)):
            for a1, a2, beta in cartesian_product((F(-1), F(1)), repeat=3):
                old_point = (x_value, F(0), q, a1, q, a2, r, beta)
                new_point = _project_point(result, old_point)
                assert _holds(result.system, new_point) == _holds(original, old_point)
                if _holds(result.system, new_point):
                    assert pe.decode(result, new_point) == old_point
    # All three gates are zero: none of their eight original labelings vanish.
    for signs in cartesian_product((F(-1), F(1)), repeat=3):
        point = (F(0), F(0), F(0), signs[0], F(0), signs[1], F(0), signs[2])
        assert _holds(original, point)
        assert _holds(result.system, _project_point(result, point))

    # A tighter old target lower bound restricts the original source even
    # when rho=0. Target relation equivalence must not imply full-System
    # equivalence after that extra restriction has been dropped.
    tight_target = replace(target, lo=F(-1))
    active = (_col(target.bit)+1)*F(1, 2)
    tight_rows = (-target.q, target.g-target.q,
                  target.q-target.hi*active,
                  target.q-target.g-(1-active))
    restricted = replace(original, le=original.le[:-4]+tight_rows)
    scoped = _apply(restricted, parents, tight_target, (readout,))
    assert scoped.receipt['rho_zero'] is True
    assert scoped.receipt['exact_target_relation_when_rho_zero'] is True
    assert scoped.receipt['exact_when_rho_zero'] is False
    point = _canonical(original, parents+(target,), (F(1), F(0)))
    assert not _holds(restricted, point)
    reduced = _project_point(scoped, point)
    assert _holds(scoped.system, reduced)
    with pytest.raises(ValueError):
        pe.decode(scoped, reduced)


def test_phase_signed_reference_decomposition():
    for weights in ((F(1, 2), F(-1, 2)), (F(-1, 2), F(1, 2)),
                    (F(1, 2), F(1, 3)), (F(-1, 2), F(-1, 3))):
        original, parents, target, d, readout = _fixture(weights=weights)
        for tau in cartesian_product((0, 1), repeat=2):
            result = _apply(original, parents, target, (readout,), tau=tau)
            proof = result.receipt
            h = d+sum((weight*reference*gate.g for weight, reference, gate
                       in zip(weights, tau, parents)), ef.Form())
            terms = tuple((1-2*reference)*gate.g for reference, gate in zip(tau, parents))
            rho = tuple(max(F(0), ef.box(original, h+2*term)[1]) for term in terms)
            tp = sum((weight/2 for weight in weights if weight > 0), F(0))
            tm = sum((-weight/2 for weight in weights if weight < 0), F(0))
            rp = sum((weight*error/2 for weight, error in zip(weights, rho)
                      if weight > 0), F(0))
            rm = sum((-weight*error/2 for weight, error in zip(weights, rho)
                      if weight < 0), F(0))
            assert proof['h'] == h and proof['t'] == terms
            assert proof['rho'] == rho and proof['coefficients'] == weights
            assert (proof['T_plus'], proof['T_minus']) == (tp, tm)
            assert (proof['R_plus'], proof['R_minus']) == (rp, rm)
            assert proof['E_plus'] == rp/(1-tp)
            assert proof['E_minus'] == rm+tm*rp/(1-tp)
            for source in cartesian_product((F(-1), F(0), F(1)), repeat=2):
                point = _canonical(original, parents+(target,), source)
                for reference, gate, term in zip(tau, parents, terms):
                    assert _value(gate.q, point) == (
                        reference*_value(gate.g, point)+max(F(0), _value(term, point)))
                reduced = _project_point(result, point)
                assert _holds(result.system, reduced)
                assert pe.decode(result, reduced) == point


def test_phase_decoder_rejects_lossy_point():
    original, parents, target, _, readout = _fixture(paper_bounds=True)
    result = _apply(original, parents, target, (readout,))
    spurious_old = (F(1, 40), F(0), F(1, 40), F(1),
                    F(1, 40), F(1), F(1, 10), F(1))
    spurious = _project_point(result, spurious_old)
    assert _holds(result.system, spurious) and not _holds(original, spurious_old)
    with pytest.raises(ValueError):
        pe.decode(result, spurious)
    true_point = _canonical(original, parents+(target,), (F(1, 40), F(0)))
    assert _value(target.q, true_point) == F(1, 20)
    reduced = _project_point(result, true_point)
    assert pe.decode(result, reduced) == true_point
    for bad in (reduced[:-1], (F(2),)+reduced[1:]):
        with pytest.raises(ValueError):
            pe.decode(result, bad)
    fractional_bit = list(reduced)
    fractional_bit[result.system.binary[0]] = F(0)
    with pytest.raises(ValueError):
        pe.decode(result, tuple(fractional_bit))


def test_phase_preserves_shared_hz_state():
    original, parents, target, _, readout = _fixture(extra_source=True)
    visible = readout+_col(2)+F(1, 7)*_col(3)
    result = _apply(original, parents, target, (visible, _col(2)))
    assert result.original is original and result.parents == parents
    assert result.system.frame == original.frame
    removed = tuple(gate.q.terms[0][0] for gate in parents)
    assert result.removed_ids == removed
    assert result.retained_ids == tuple(i for i in range(len(original.bounds)) if i not in removed)
    assert result.system.bounds == tuple(original.bounds[i] for i in result.retained_ids)
    assert tuple(result.retained_ids[i] for i in result.system.binary) == original.binary
    assert result.system.eq == tuple(_remap(row, result.retained_ids) for row in original.eq)
    assert _remap(original.le[0], result.retained_ids) in result.system.le
    assert result.readouts == (_remap(visible, result.retained_ids),
                               _remap(_col(2), result.retained_ids))
    proof = result.receipt
    assert proof['retained_original_system'] is True
    assert proof['original_entries'] == ef.entries(original)
    assert proof['emitted_entries'] == ef.entries(result.system)
    assert proof['original_plus_emitted_entries'] == (
        proof['original_entries']+proof['emitted_entries'])
    assert proof['original_plus_emitted_entries'] <= proof['entry_upper']
    assert proof['original_bits_deleted'] == 0
    for key in ('external_consumers_qualified', 'native_binding_qualified',
                'actual_model_qualified', 'complete_physical_qualification',
                'whole_work_qualified', 'gpu_qualified'):
        assert proof[key] is False
    point = _canonical(original, parents+(target,), (F(0), F(0), F(0), F(1)))
    assert pe.decode(result, _project_point(result, point)) == point
    broken_eq = list(_project_point(result, point))
    broken_eq[2] = F(1)
    with pytest.raises(ValueError):
        pe.decode(result, tuple(broken_eq))
    # A deleted q's stronger variable bounds must survive as source rows,
    # rather than silently vanish during remapping.
    parent_output = parents[0].q.terms[0][0]
    for interval, expected_row, rejected_source in (
            ((F(0), F(1, 2)), parents[0].g-F(1, 2),
             (F(1), F(0), F(1), F(1))),
            ((F(1, 4), F(11, 10)), F(1, 4)-parents[0].g,
             (F(0), F(0), F(0), F(1)))):
        bounds = list(original.bounds)
        bounds[parent_output] = interval
        tighter = replace(original, bounds=tuple(bounds))
        projected = _apply(tighter, parents, target, (visible,))
        assert _remap(expected_row, projected.retained_ids) in projected.system.le
        true_point = _canonical(tighter, parents+(target,),
                                (F(1, 4), F(0), F(1, 4), F(1)))
        assert pe.decode(projected, _project_point(projected, true_point)) == true_point
        rejected = _canonical(original, parents+(target,), rejected_source)
        assert not _holds(projected.system, _project_point(projected, rejected))
    assert original == _fixture(extra_source=True)[0]


def test_phase_fail_closed_guards_and_budgets():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected '+name)

    poison = Poison()
    disabled = pe.project(poison, poison, poison, tau=poison, kappa=poison,
                          readouts=poison, frames=poison, enabled=False,
                          max_entries=poison)
    assert disabled.system is poison and disabled.receipt == {'enabled': False}
    original, parents, target, _, readout = _fixture(paper_bounds=True)
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            _apply(original, parents, target, (readout,), enabled=flag)
    for frames in ((original.frame,), (0,)*3, (True,)*3):
        with pytest.raises(ValueError):
            _apply(original, parents, target, (readout,), frames=frames)
    for tau in ((True, 0), (F(0), 0), (2, 0), (0,)):
        with pytest.raises(ValueError):
            _apply(original, parents, target, (readout,), tau=tau)
    for kappa in ((F(0), F(2)), (F(-1), F(2)), (2, F(2)),
                  (F(1, 2), F(2)), (F(1 << 512), F(2))):
        with pytest.raises(ValueError):
            _apply(original, parents, target, (readout,), kappa=kappa)
    for cap in (0, 1, True, 64_000_001):
        with pytest.raises(ValueError):
            _apply(original, parents, target, (readout,), max_entries=cap)
    with pytest.raises(ValueError):
        _apply(original, parents, target, (parents[0].q,))
    with pytest.raises(ValueError):
        _apply(replace(original, le=original.le+(parents[0].q-F(11, 10),)),
               parents, target, (readout,))
    with pytest.raises(ValueError):
        _apply(replace(original, eq=(parents[0].q-parents[1].q,)),
               parents, target, (readout,))
    with pytest.raises(ValueError):
        _apply(replace(original, le=original.le[1:]), parents, target, (readout,))
    with pytest.raises(ValueError):
        _apply(replace(original, binary=original.binary[1:]), parents, target, (readout,))
    with pytest.raises(ValueError):
        _apply(original, (parents[0], parents[0]), target, (readout,))
    assert original == _fixture(paper_bounds=True)[0]
