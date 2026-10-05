"""Four no-solver controls of signed mixed consumers and original phases.

These exact-rational fixtures do not stand in for the separately registered
original-model source/consumer population.  No test deletes an old predicate.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product as cartesian_product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct
from experiments.neural_hz_20260831.definition_first_20260928.d120_mixed_consumer_source_20261002 import mixed_transfer as mt


KNOTS = (F(0), F(1, 4), F(1, 2), F(3, 4), F(1))


def _col(index):
    return ef.Form(F(0), ((index, F(1)),))


def _evaluate(form, point):
    return form.bias + sum((value * point[index] for index, value in form.terms), F(0))


def _holds(system, point):
    return (len(point) == len(system.bounds)
            and all(lo <= value <= hi for value, (lo, hi) in zip(point, system.bounds))
            and all(point[index] in (F(-1), F(1)) for index in system.binary)
            and all(_evaluate(row, point) == 0 for row in system.eq)
            and all(_evaluate(row, point) <= 0 for row in system.le))


def _gate(system, preactivation):
    lo, hi = ef.box(system, preactivation)
    system, q = ef.variable(system, max(F(0), lo), max(F(0), hi))
    bit = len(system.bounds)
    system, signed = ef.variable(system, F(-1), F(1))
    active = (signed + 1) * F(1, 2)
    rows = (-q, preactivation - q, q - max(F(0), hi) * active,
            q - preactivation + min(F(0), lo) * (1 - active))
    system = replace(system, binary=system.binary + (bit,), le=system.le + rows)
    return system, ct.Gate(preactivation, q, bit, lo, hi)


def _fixture(eps=F(0), endpoint_weights=(F(0), F(0)), bias=F(0)):
    original = ef.System(((F(-1), F(1)),) * 3, (), (ef.Form(),), (), 12001)
    x, y, z = (_col(index) for index in range(3))
    preactivations = (x, F(3, 4) * x + F(1, 4) * y + eps * z,
                      (x + y) * F(1, 2),
                      F(1, 4) * x + F(3, 4) * y - eps * z, y)
    gates = []
    for preactivation in preactivations:
        original, gate = _gate(original, preactivation)
        gates.append(gate)
    gates = tuple(gates)
    original, outside = _gate(original, z)
    # The unrelated original output is real fan-in, not a removable fixture tag.
    rest = z + F(1, 3) * outside.q + bias
    weights = (endpoint_weights[0], F(1), F(-2), F(1), endpoint_weights[1])
    readout = rest + sum((weight * gate.q for weight, gate in zip(weights, gates)), ef.Form())
    original, child = _gate(original, readout)
    return original, gates, outside, child, readout, rest, weights


def _apply(original, gates, readout, child=None, **extra):
    kwargs = dict(frames=(original.frame,) * len(gates),
                  readout_frame=original.frame, enabled=True, child=child)
    kwargs.update(extra)
    return mt.append_mixed_consumer(original, gates, KNOTS, readout, **kwargs)


def _canonical(system, gates, source, zero_labels=None):
    choices = {} if zero_labels is None else zero_labels
    point = list(source)
    for gate in gates:
        assert gate.q == _col(len(point)) and gate.bit == len(point) + 1
        preactivation = _evaluate(gate.g, point)
        point.append(max(F(0), preactivation))
        signed = F(1) if preactivation > 0 else F(-1)
        if preactivation == 0:
            signed = choices.get(gate.bit, F(-1))
            assert signed in (F(-1), F(1))
        point.append(signed)
    assert _holds(system, point)
    return tuple(point)


def _preserved(original, extended, receipt, new_rows):
    assert extended.bounds == original.bounds and extended.binary == original.binary
    assert extended.eq == original.eq and extended.le[:len(original.le)] == original.le
    assert receipt['new_columns'] == receipt['new_eq'] == receipt['original_bits_deleted'] == 0
    assert receipt['new_le'] == new_rows == len(extended.le) - len(original.le)
    assert receipt['rows'] == extended.le[len(original.le):]
    assert receipt['entries'] == ef.entries(extended) <= receipt['entry_upper']
    assert receipt['nnz'] == sum(len(row.terms) for row in receipt['rows'])
    assert receipt['solver_calls'] == 0
    for key in ('native_binding_qualified', 'actual_model_qualified',
                'complete_physical_qualification', 'whole_work_qualified', 'gpu_qualified'):
        assert receipt[key] is False


def test_mixed_readout_and_rest():
    for endpoints in ((F(0), F(0)), (F(1, 3), F(-1, 5))):
        original, gates, outside, child, readout, rest, weights = _fixture(
            endpoint_weights=endpoints, bias=F(1, 7))
        extended, proof = _apply(original, gates, readout)
        _preserved(original, extended, proof, 2)
        assert proof['weights'] == weights and proof['rest'] == rest
        # Direct hand calculation for (1,-2,1), not an oracle call to support.
        assert proof['kernel_lower'] == F(-1, 4) and proof['kernel_upper'] == 0
        amplitude = 2 * gates[0].q + 2 * gates[-1].q - _col(0) - _col(1)
        affine = endpoints[0] * gates[0].q + endpoints[1] * gates[-1].q
        assert proof['affine'] == affine and proof['amplitude'] == amplitude
        assert proof['upper'] == rest + affine + amplitude * F(1, 4)
        assert proof['lower'] == rest + affine
        assert proof['residual_boxes'] == ((F(0), F(0)),) * 3
        assert proof['rows'] == (readout - proof['upper'], proof['lower'] - readout)
        assert proof['child_upper'] is None and proof['child_branch'] is None
        assert dict(proof['rest'].terms)[outside.q.terms[0][0]] == F(1, 3)
        for source in cartesian_product((F(-1), F(0), F(1)), repeat=3):
            point = _canonical(original, gates + (outside, child), source)
            assert _holds(extended, point)
            assert (_evaluate(proof['lower'], point) <= _evaluate(readout, point)
                    <= _evaluate(proof['upper'], point))
        point = _canonical(original, gates + (outside, child), (F(0), F(0), F(1)))
        assert _evaluate(readout, point) == _evaluate(proof['upper'], point) == F(31, 21)
    # No positive multiple of (2,-4,0,-4,2) has this mixed nonzero middle weight.
    assert weights[2] == -2


def test_mixed_residual_sign_symmetry():
    eps = F(1, 20)
    original, gates, outside, child, readout, rest, weights = _fixture(
        eps=eps, endpoint_weights=(F(1, 3), F(-1, 5)), bias=F(1, 7))
    extended, proof = _apply(original, gates, readout)
    negative, reverse = _apply(original, gates, -readout)
    _preserved(original, extended, proof, 2)
    _preserved(original, negative, reverse, 2)
    assert proof['residual_boxes'] == ((-eps, eps), (F(0), F(0)), (-eps, eps))
    assert proof['weighted_boxes'] == proof['residual_boxes']
    assert sum(proof['positive_secants'], ef.Form()) == ef.Form(eps)
    assert sum(proof['negative_secants'], ef.Form()) == ef.Form(eps)
    affine = F(1, 3) * gates[0].q - F(1, 5) * gates[-1].q
    amplitude = 2 * gates[0].q + 2 * gates[-1].q - _col(0) - _col(1)
    assert proof['upper'] == rest + affine + F(1, 4) * amplitude + eps
    assert proof['lower'] == rest + affine - eps
    assert reverse['weights'] == tuple(-weight for weight in weights)
    assert reverse['rest'] == -rest
    assert reverse['upper'] == -proof['lower'] and reverse['lower'] == -proof['upper']
    for source in cartesian_product((F(-1), F(0), F(1)), repeat=3):
        point = _canonical(original, gates + (outside, child), source)
        assert _holds(extended, point) and _holds(negative, point)
        assert (_evaluate(proof['lower'], point) <= _evaluate(readout, point)
                <= _evaluate(proof['upper'], point))


def test_mixed_successor_and_original_phases():
    original, gates, outside, child, readout, _, _ = _fixture()
    extended, proof = _apply(original, gates, readout, child=child)
    _preserved(original, extended, proof, 3)
    assert proof['child_branch'] == 'crossing'
    assert proof['upper_box'] == (F(-3, 2), F(17, 6))
    assert proof['child_upper'] == F(17, 26) * proof['upper'] + F(51, 52)
    assert proof['rows'][-1] == child.q - proof['child_upper']
    assert ef.box(original, readout) != proof['upper_box']
    all_gates = gates + (outside, child)
    for source in cartesian_product((F(-1), F(0), F(1)), repeat=3):
        assert _holds(extended, _canonical(original, all_gates, source))
    # Seven original gates are all at zero here: preserve all 128 legal labels.
    for signs in cartesian_product((F(-1), F(1)), repeat=len(all_gates)):
        labels = dict(zip((gate.bit for gate in all_gates), signs))
        point = _canonical(original, all_gates, (F(0), F(0), F(0)), labels)
        assert all(_evaluate(gate.g, point) == 0 for gate in all_gates)
        assert tuple(point[gate.bit] for gate in all_gates) == signs
        assert _holds(extended, point)
    for bias, branch in ((F(4), 'nonnegative'), (F(-4), 'nonpositive')):
        stable, group, other, next_gate, value, _, _ = _fixture(bias=bias)
        tightened, record = _apply(stable, group, value, child=next_gate)
        _preserved(stable, tightened, record, 3)
        assert record['child_branch'] == branch
        assert record['child_upper'] == (record['upper'] if bias > 0 else ef.Form())
        for source in cartesian_product((F(-1), F(0), F(1)), repeat=3):
            assert _holds(tightened, _canonical(stable, group + (other, next_gate), source))


def test_mixed_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected ' + name)

    poison = Poison()
    result, proof = mt.append_mixed_consumer(
        poison, poison, poison, poison, frames=poison, readout_frame=poison,
        child=poison, enabled=False, max_entries=poison)
    assert result is poison and proof == {'enabled': False}
    original, gates, _, child, readout, _, _ = _fixture()
    good = dict(frames=(original.frame,) * len(gates),
                readout_frame=original.frame, enabled=True)
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, gates, KNOTS, readout,
                                     **dict(good, enabled=flag))
    for frames in ((original.frame,) * 4, (0,) * 5, (True,) * 5):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, gates, KNOTS, readout,
                                     **dict(good, frames=frames))
    for frame in (True, original.frame + 1):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, gates, KNOTS, readout,
                                     **dict(good, readout_frame=frame))
    for knots in (KNOTS[:-1], (F(0), F(1, 4), F(1, 4), F(3, 4), F(1)),
                  (F(-1),) + KNOTS[1:], KNOTS[:-1] + (F(2),),
                  (0,) + KNOTS[1:]):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, gates, knots, readout, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, (gates[0],) + gates[:-1], KNOTS, readout, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(replace(original, le=original.le[1:]), gates,
                                 KNOTS, readout, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(replace(original, binary=original.binary[1:]), gates,
                                 KNOTS, readout, **good)
    for bad in (replace(gates[0], bit=True), replace(gates[0], lo=0),
                replace(gates[0], hi=F(1 << 512))):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, (bad,) + gates[1:], KNOTS, readout, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, gates, KNOTS, _col(len(original.bounds)), **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, gates, KNOTS, readout + 1, child=child, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, gates, KNOTS, gates[0].g, child=gates[0], **good)
    for cap in (0, 1, True, 64_000_001, ef.entries(original)):
        with pytest.raises(ValueError):
            mt.append_mixed_consumer(original, gates, KNOTS, readout,
                                     max_entries=cap, **good)
    _, receipt = mt.append_mixed_consumer(original, gates, KNOTS, readout, **good)
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, gates, KNOTS, readout,
                                 max_entries=receipt['entry_upper'] - 1, **good)
    # Inputs individually fit 512 bits, but the support scan sum does not.
    huge = F(1 << 511)
    wide_readout = huge * gates[1].q + huge * gates[2].q
    with pytest.raises(ValueError):
        mt.append_mixed_consumer(original, gates, KNOTS, wide_readout, **good)
    assert original == _fixture()[0]
