"""Fixed owned-simplex mathematics and full small neural source control.

No trained model or GPU is admitted. Terminal LP assignments are diagnostic
fractional witnesses, never concrete adversarial inputs. All objectives and
source rows are fixed before the first solve.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product as cartesian_product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import test_endpoint_forward as prior_helpers
from experiments.neural_hz_20260831.definition_first_20260928.d116_owned_simplex_transfer_20261002 import owned_simplex as owned


RESULTS = Path(__file__).resolve().parents[2] / 'results'


def _col(i):
    return ef.Form(F(0), ((i, F(1)),))


def _eval(form, point):
    return form.bias + sum((value*point[i] for i, value in form.terms), F(0))


def _holds(system, point):
    return (len(point) == len(system.bounds)
            and all(lo <= v <= hi for v, (lo, hi) in zip(point, system.bounds))
            and all(_eval(r, point) == 0 for r in system.eq)
            and all(_eval(r, point) <= 0 for r in system.le))


def _local():
    system = ef.System(((F(0), F(1)),)*9 + ((F(-1), F(1)),)*2,
                       (9, 10), (), (), 11600)
    v = tuple(_col(i) for i in range(3))
    t = (v[0], _col(3), _col(4))
    u = tuple(_col(i) for i in range(5, 8))
    alpha, beta = (_col(9)+1)*F(1, 2), (_col(10)+1)*F(1, 2)
    args = dict(v=v, t=t, u=u, alpha=alpha, beta=beta, delta=_col(8), B=F(1))
    system = replace(system, le=owned.reference_rows(**args))
    return system, args


def _explicit_extension(system, args):
    """Same simplex information only; no extra joint source/triangle lifting."""
    v, t, u = args['v'], args['t'], args['u']
    a, b, d, B = args['alpha'], args['beta'], args['delta'], args['B']
    s, specs = [u[0]], []
    for i in (1, 2):
        index = len(system.bounds)
        system, value = ef.variable(system, F(0), B)
        s.append(value)
        specs.append((index, (a, b, v[i])))
    rows = []
    for i in (1, 2):
        rows.extend((-s[i], s[i]-t[i], s[i]-u[i], t[i]+u[i]-v[i]-s[i]))
    S, V, T, U = (sum(vv, ef.Form()) for vv in (s, v, t, u))
    rows.extend((S-B*d, T-S-B*(a-d), U-S-B*(b-d),
                 V-T-U+S-B*(1-a-b+d)))
    return replace(system, le=system.le+tuple(rows)), specs


def _canonical_local(a, b, values):
    return list(values)+[a*values[1], a*values[2]]+[
        b*v for v in values]+[a*b, 2*a-1, 2*b-1]


def test_owned_simplex_exact_extension():
    original, args = _local()
    extended, receipt = owned.append_owned_simplex(
        original, **args, frames=(original.frame,)*7, enabled=True)
    assert extended.bounds == original.bounds
    assert extended.binary == original.binary and extended.eq == original.eq
    assert extended.le[:len(original.le)] == original.le
    assert receipt['new_le'] == 6 and receipt['new_columns'] == 0
    assert len(extended.le)-len(original.le) == 6
    for a, b in cartesian_product((F(0), F(1)), repeat=2):
        for first in (F(0), F(1, 5)*a):
            values = (first, F(1, 4), F(3, 10))
            point = _canonical_local(a, b, values)
            assert _holds(original, point) and _holds(extended, point)
            assert all(point[i] in (F(-1), F(1)) for i in original.binary)
    explicit, specs = _explicit_extension(original, args)
    assert len(explicit.bounds) == len(original.bounds)+2
    assert len(explicit.le) == len(original.le)+12
    for a, b in cartesian_product((F(0), F(1)), repeat=2):
        point = _canonical_local(a, b, (a/5, F(1, 4), F(3, 10)))
        for index, factors in specs:
            assert index == len(point)
            value = F(1)
            for form in factors:
                value *= _eval(form, point)
            point.append(value)
        assert _holds(explicit, point)


def test_owned_simplex_contract_rejections():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected '+name)

    poison = Poison()
    result, receipt = owned.append_owned_simplex(
        poison, poison, poison, poison, poison, poison, poison, poison,
        frames=poison, enabled=False, max_entries=poison)
    assert result is poison and receipt == {'enabled': False}
    original, args = _local()
    good = dict(frames=(original.frame,)*7, enabled=True)
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            owned.append_owned_simplex(original, **args, **dict(good, enabled=flag))
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **args, **dict(good, frames=(0,)*7))
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, v=args['v'][:2]), **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, B=F(0)), **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, alpha=args['beta']), **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, alpha=ef.Form(F(1, 2))), **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(replace(original, le=()), **args, **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, t=(args['u'][0],)+args['t'][1:]), **good)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **args, **good, max_entries=1)
    with pytest.raises(ValueError):
        owned.append_owned_simplex(original, **dict(args, B=F(1 << 513)), **good)
    assert original == _local()[0]


def test_owned_simplex_strict_local_face():
    original, args = _local()
    point = [F(1, 5), F(1, 4), F(3, 10), F(1, 5), F(1, 20),
             F(3, 20), F(1, 5), F(1, 20), F(1, 4), F(0), F(0)]
    assert _holds(original, point)
    assert _eval(args['t'][0]-args['v'][0], point) == 0
    extended, _ = owned.append_owned_simplex(
        original, **args, frames=(original.frame,)*7, enabled=True)
    assert _eval(extended.le[-6], point) == F(1, 20)
    assert not _holds(extended, point)
    # Shared s1=u1 and s2>=t2+u2-v2 already require total s>=3/10.
    required = _eval(args['u'][0]+args['t'][1]+args['u'][1]-args['v'][1], point)
    assert required == F(3, 10) > _eval(args['B']*args['delta'], point)


def _gate(system, preactivation):
    lo, hi = ef.box(system, preactivation)
    qindex = len(system.bounds)
    system, q = ef.variable(system, max(F(0), lo), max(F(0), hi))
    bindex = len(system.bounds)
    system, signed = ef.variable(system, F(-1), F(1))
    system = replace(system, binary=system.binary+(bindex,))
    bit = (signed+1)*F(1, 2)
    rows = (-q, preactivation-q, q-max(F(0), hi)*bit,
            q-preactivation+min(F(0), lo)*(1-bit))
    system = replace(system, le=system.le+rows)
    return system, dict(g=preactivation, q=q, bit=bit, qindex=qindex,
                        bindex=bindex, lo=lo, hi=hi)


def _product(system, x, y, specs):
    index = len(system.bounds)
    system, value = ef.product(system, x, y)
    if len(system.bounds) != index:
        assert len(system.bounds) == index+1 and value == _col(index)
        specs.append((index, (x, y)))
    return system, value


def _conditional(form, bit, mapping):
    return form.bias*bit+sum((coefficient*mapping[index]
                              for index, coefficient in form.terms), ef.Form())


def _triangle(gate):
    lo, hi = gate['lo'], gate['hi']
    assert lo < 0 < hi
    slope = hi/(hi-lo)
    return (-gate['q'], gate['g']-gate['q'],
            gate['q']-slope*gate['g']+slope*lo)


def _two_phase_rows(v, t, u, a, b, d, lo, hi):
    assert lo < hi
    z = (v-lo)*(1/(hi-lo))
    tn, un = (t-lo*a)*(1/(hi-lo)), (u-lo*b)*(1/(hi-lo))
    return (tn+un-d-z, tn+d-un-a, un+d-tn-b, a+b+z-tn-un-d-1)


def _neural():
    system = ef.System(((F(-1), F(1)),)*2, (), (), (), 11601)
    x, y = _col(0), _col(1)
    pres = (x+y*F(1, 4)+F(1, 5), -x+y*F(1, 5)+F(3, 10), -y+x*F(1, 10)+F(1, 4))
    gates = []
    for pre in pres:
        system, gate = _gate(system, pre)
        gates.append(gate)
    q1, q2, q3 = tuple(g['q'] for g in gates)
    h = q1-F(3, 5)*q2+F(2, 5)*q3+x*F(1, 10)-F(1, 5)
    system, gate_p = _gate(system, h)
    gates.append(gate_p)
    p = gate_p['q']
    j = p+F(2, 5)*q1-F(4, 5)*q2+F(3, 5)*q3+y*F(1, 10)-F(1, 4)
    system, gate_w = _gate(system, j)
    gates.append(gate_w)
    w = gate_w['q']
    a, b = gates[0]['bit'], gate_p['bit']
    B, v = F(251, 100), (q1, q2, q3)
    # Authenticate the paper budget by recomputing the ordinary secant sum.
    secants = tuple(g['hi']/(g['hi']-g['lo'])*(g['g']-g['lo']) for g in gates[:3])
    assert ef.box(system, sum(secants, ef.Form()))[1] == B
    assert (gate_p['lo'], gate_p['hi']) == (F(-6, 5), F(189, 100))
    assert (gate_w['lo'], gate_w['hi']) == (F(-31, 20), F(313, 100))
    system = replace(system, le=system.le+(sum(v, ef.Form())-B,))
    specs = []
    system, delta = _product(system, a, b, specs)
    source_forms = (x, y, q1, q2, q3, p)
    mappings = []
    for bit, owned_value in ((a, q1), (b, p)):
        mapping = {}
        for form in source_forms:
            if form == owned_value:
                value = form
            else:
                system, value = _product(system, bit, form, specs)
            index, coefficient = form.terms[0]
            assert form.bias == 0 and len(form.terms) == 1 and coefficient == 1
            mapping[index] = value
        mappings.append(mapping)
    ma, mb = mappings
    t, u = tuple(ma[g['qindex']] for g in gates[:3]), tuple(mb[g['qindex']] for g in gates[:3])
    rows = []
    for bit, mapping in ((a, ma), (b, mb)):
        for gate in gates[:4]:
            for row in _triangle(gate):
                on = _conditional(row, bit, mapping)
                rows.extend((on, row-on))
    for form in source_forms:
        index = form.terms[0][0]
        lo, hi = ef.box(system, form)
        rows.extend(_two_phase_rows(form, ma[index], mb[index], a, b, delta, lo, hi))
    args = dict(v=v, t=t, u=u, alpha=a, beta=b, delta=delta, B=B)
    rows.extend(owned.reference_rows(**args))
    rows.append(t[0]+t[1]-F(29, 20)*a)
    system = replace(system, eq=system.eq+(
        q1-_conditional(gates[0]['g'], a, ma), p-_conditional(h, b, mb)),
        le=system.le+tuple(rows))
    on_j = _conditional(j, b, mb)
    off_j = j-on_j
    L0, U0, L1, U1 = F(-31, 20), F(31, 25), F(-31, 20), F(313, 100)
    k0, k1 = U0/(U0-L0), U1/(U1-L1)
    system = replace(system, le=system.le+(
        w-k0*off_j+k0*L0*(1-b)-k1*on_j+k1*L1*b,))
    directions = (('p', p), ('w', w), ('w_minus_p', w-p),
                  ('w_minus_q1', w-q1), ('w_minus_q2', w-q2),
                  ('w_minus_q3', w-q3), ('w_minus_p_plus_q1_minus_q2', w-p+q1-q2))
    return system, args, gates, specs, directions


def _true_points(system, gates, specs, x, y):
    # Only finite mathematical identity fixtures; never candidate search.
    choices = []
    original = [x, y]
    for gate in gates:
        value = _eval(gate['g'], original)
        choices.append((F(-1), F(1)) if value == 0 else (F(1) if value > 0 else F(-1),))
        assert gate['qindex'] == len(original)
        original.extend((max(F(0), value), F(0)))
    for labels in cartesian_product(*choices):
        point = list(original)
        for gate, label in zip(gates, labels):
            point[gate['bindex']] = label
        for index, factors in specs:
            assert index == len(point)
            value = F(1)
            for form in factors:
                value *= _eval(form, point)
            point.append(value)
        assert all(point[i] in (F(-1), F(1)) for i in system.binary)
        yield point


def _run_dir():
    run = Path(os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'])
    if (not run.is_absolute() or run.is_symlink() or not run.is_dir()
            or run.resolve() != run or run == RESULTS or not run.is_relative_to(RESULTS)):
        raise ValueError('untrusted active component run')
    return run


def test_owned_simplex_bound_neural_control():
    record = dict(scope='fixed exact-rational neural block, not native trained model',
                  formal_gain=0, model_qualified=False, gpu_qualified=False,
                  native_binding_qualified=False, complete_physical_qualification=False,
                  solver_rescue_registered=False, local_point_imposed=False,
                  full_prefix_hull=False, known_small_perspective_reference=True)
    try:
        A, args, gates, specs, directions = _neural()
        C, receipt = owned.append_owned_simplex(A, **args, frames=(A.frame,)*7, enabled=True)
        E, extra_specs = _explicit_extension(A, args)
        assert A.binary == C.binary == E.binary and len(A.binary) == 5
        assert C.eq == A.eq and E.eq == A.eq
        assert C.le[:len(A.le)] == E.le[:len(A.le)] == A.le
        fixtures = ((F(-1, 2), F(0)), (F(-1, 10), F(-9, 10)),
                    (F(0), F(0)), (F(1, 2), F(0)), (F(-1, 5), F(0)),
                    (F(4, 87), F(0)), (F(3, 10), F(0)), (F(0), F(1, 4)),
                    (F(17, 150), F(0)))
        checked = 0
        for x, y in fixtures:
            for point in _true_points(A, gates, specs, x, y):
                assert _holds(A, point) and _holds(C, point)
                extended = list(point)
                for index, factors in extra_specs:
                    assert index == len(extended)
                    value = F(1)
                    for form in factors:
                        value *= _eval(form, extended)
                    extended.append(value)
                assert _holds(E, extended)
                checked += 1
        record['canonical_points_checked'] = checked
        record['zero_phase_labels_checked'] = True
        record['receipt'] = receipt
        record['arms'] = {name: dict(columns=len(s.bounds), original_signed_bits=s.binary,
            eq=len(s.eq), le=len(s.le), nnz=sum(len(r.terms) for r in s.eq+s.le),
            retained_entries=ef.entries(s)) for name, s in (('A', A), ('C', C), ('E', E))}
        observations = []
        record['directions'] = observations
        for name, form in directions:
            for sign in (F(1), F(-1)):
                result = dict(name=name, sign=int(sign))
                observations.append(result)
                for arm, system in (('A', A), ('C', C), ('E', E)):
                    value = prior_helpers._maximum(system, sign*form)
                    result[arm] = value
                    assert value['success'], (name, sign, arm, value)
                    assert value['max_equality_residual'] <= 1e-7
                    assert value['max_inequality_residual'] <= 1e-7
                result['A_minus_C'] = result['A']['upper']-result['C']['upper']
                result['C_minus_E'] = result['C']['upper']-result['E']['upper']
                assert result['A_minus_C'] >= -1e-7
                assert abs(result['C_minus_E']) <= 1e-7
        assert len(observations) == 14
        record['fixed_terminal_solves'] = 42
        record['directions_with_gain'] = sum(item['A_minus_C'] > 1e-7 for item in observations)
        record['maximum_gain'] = max(item['A_minus_C'] for item in observations)
        record['assertions_passed'] = True
    except BaseException as error:
        record['assertions_passed'] = False
        record['failure'] = type(error).__name__+': '+str(error)
        raise
    finally:
        with (_run_dir() / 'source_bound_joint_budget_control.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
