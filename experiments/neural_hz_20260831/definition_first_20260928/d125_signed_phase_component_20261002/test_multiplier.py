"""Six exact-rational controls of the common-multiplier subfamily only."""
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d125_signed_phase_component_20261002 import multiplier as mm
from experiments.neural_hz_20260831.definition_first_20260928.d125_signed_phase_component_20261002 import phase_envelope as pe


ef = pe.ef


def _col(index):
    return ef.Form(F(0), ((index, F(1)),))


def _system(bounds=((F(-1), F(1)), (F(-1), F(1)))):
    return ef.System(bounds, (), (), (), 12502)


def _apply(system, h, terms, weights, **changes):
    kwargs = dict(enabled=True)
    kwargs.update(changes)
    return mm.common_multiplier(system, h, terms, weights, **kwargs)


def _objective(system, h, terms, weights, kappa):
    positive = sum((weight for weight in weights if weight > 0), F(0))
    assert kappa > positive
    numerator = sum((weight*max(F(0), ef.box(system, h+kappa*term)[1])
                     for weight, term in zip(weights, terms) if weight > 0), F(0))
    return numerator/(kappa-positive)


def _finite_witness(proof, system, h, terms, weights, expected):
    assert proof['enabled'] is True and proof['attained'] is True
    assert proof['boundary'] == 'finite'
    assert isinstance(proof['kappa'], F)
    assert proof['kappa'] > sum((weight for weight in weights if weight > 0), F(0))
    assert proof['infimum'] == proof['finite_value'] == expected
    assert _objective(system, h, terms, weights, proof['kappa']) == expected


def test_multiplier_genuine_interior_optimum():
    system = _system()
    x, y = _col(0), _col(1)
    h = F(1, 10)-2*x+y*F(1, 20)
    terms = (x+y*F(1, 10), x-y*F(1, 10))
    weights = (F(1, 2), F(-1, 2))
    proof = _apply(system, h, terms, weights)
    _finite_witness(proof, system, h, terms, weights, F(7, 60))
    assert proof['kappa'] == 2 and F(2) in proof['breakpoints']
    assert proof['independent_cap'] == F(11, 20)
    for kappa in (F(3, 4), F(1), F(3, 2), F(5, 2), F(4)):
        assert _objective(system, h, terms, weights, kappa) > F(7, 60)


def test_multiplier_boundary_infima():
    system = _system()
    h, terms, weights = ef.Form(F(1)), (ef.Form(F(1)),), (F(1),)
    proof = _apply(system, h, terms, weights)
    assert proof['infimum'] == proof['independent_cap'] == F(1)
    assert proof['kappa'] is None and proof['finite_value'] is None
    assert proof['attained'] is False and proof['boundary'] == 'infinity'
    for kappa in (F(3, 2), F(2), F(3), F(10)):
        assert _objective(system, h, terms, weights, kappa) == (kappa+1)/(kappa-1) > 1
    # At A=1, N(A)=2: the right limit is +infinity, not an optimum.
    assert max(F(0), ef.box(system, h+terms[0])[1]) == 2
    # N(A)=0 gives a finite lower limit, but a neighboring affine segment
    # actually attains it.  A forbidden boundary point is never the witness.
    h = ef.Form(F(-1))
    lower = _apply(system, h, terms, weights)
    _finite_witness(lower, system, h, terms, weights, F(1))


def test_multiplier_constant_plateau_witness():
    system = _system()
    h, terms, weights = ef.Form(F(-1)), (ef.Form(),), (F(1),)
    proof = _apply(system, h, terms, weights)
    _finite_witness(proof, system, h, terms, weights, F(0))
    assert proof['independent_cap'] == 0
    # An identically zero clipped B segment is not an infinite list of roots.
    assert isinstance(proof['breakpoints'], tuple)
    assert len(proof['breakpoints']) < 5
    for kappa in (F(3, 2), F(2), F(7)):
        assert _objective(system, h, terms, weights, kappa) == 0


def test_multiplier_zero_weight():
    system = _system()
    h = ef.Form(F(1, 3))
    terms, weights = (_col(0), -_col(1)), (F(0), F(-2))
    proof = _apply(system, h, terms, weights)
    _finite_witness(proof, system, h, terms, weights, F(0))
    assert proof['independent_cap'] == 0

    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected '+name)

    poison = Poison()
    assert mm.common_multiplier(poison, poison, poison, poison,
                                enabled=False, max_entries=poison) == {'enabled': False}
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            _apply(system, h, terms, weights, enabled=flag)
    for cap in (0, 1, True, 64_000_001):
        with pytest.raises(ValueError):
            _apply(system, h, terms, weights, max_entries=cap)
    with pytest.raises(ValueError):
        _apply(system, h, terms, (F(0), F(1 << 512)))


def test_multiplier_constant_and_zero_coefficients():
    system = _system()
    assert ef.Form(F(-2), ((0, F(0)),)) == ef.Form(F(-2))
    h, terms, weights = ef.Form(F(-2)), (ef.Form(F(1)),), (F(1),)
    proof = _apply(system, h, terms, weights)
    _finite_witness(proof, system, h, terms, weights, F(0))
    assert F(1) < proof['kappa'] <= F(2)
    assert F(2) in proof['breakpoints']
    assert proof['independent_cap'] == 1
    degenerate = _system(((F(2), F(2)),))
    h, terms = _col(0)-2, (_col(0)-1,)
    proof = _apply(degenerate, h, terms, weights)
    assert proof['infimum'] == proof['independent_cap'] == 1
    assert proof['boundary'] == 'infinity' and proof['attained'] is False
    assert proof['kappa'] is None
    for kappa in (F(2), F(3)):
        assert _objective(degenerate, h, terms, weights, kappa) == kappa/(kappa-1)


def test_multiplier_general_source_box():
    system = _system(((F(2), F(4)), (F(-3), F(-1))))
    x, y = _col(0), _col(1)
    h = F(31, 5)-2*x+y*F(1, 20)
    terms = (x+y*F(1, 10)-F(14, 5), x-y*F(1, 10)-F(16, 5))
    weights = (F(1, 2), F(-1, 2))
    assert tuple(ef.box(system, term) for term in terms) == ((F(-11, 10), F(11, 10)),)*2
    assert ef.box(system, h) == (F(-39, 20), F(43, 20))
    proof = _apply(system, h, terms, weights)
    _finite_witness(proof, system, h, terms, weights, F(7, 60))
    assert proof['kappa'] == 2 and proof['independent_cap'] == F(11, 20)
    assert tuple(max(F(0), ef.box(system, h+2*term)[1]) for term in terms) == (F(7, 20), F(1, 4))
    with pytest.raises(ValueError):
        _apply(system, h, terms, (F(1, 2),))
    with pytest.raises(ValueError):
        _apply(system, h, terms, (F(1, 2), -0.5))
