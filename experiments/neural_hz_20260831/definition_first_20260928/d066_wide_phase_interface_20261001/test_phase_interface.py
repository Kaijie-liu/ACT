"""Four fixed mathematical groups, collected only by the frozen full gate.

No model sampling or adversarial search. Direct reference expansion does not
use the candidate's cached bases, sparse support corrections or majorants.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms
from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss
from experiments.neural_hz_20260831.definition_first_20260928.d066_wide_phase_interface_20261001 import phase_interface as pi

Z, O, M = F(0), F(1), 2**24


def _context(bounds):
    frame = ms.make_frame('original mathematical source', enabled=True)
    values = tuple(ms.make_value(frame, 'source ' + str(i), enabled=True)
                   for i in range(len(bounds)))
    context = ms.source_context(frame, tuple((v, i, *interval)
        for i, (v, interval) in enumerate(zip(values, bounds))), enabled=True)
    return frame, context


def _point(context, bias, coefficients):
    return ss.affine(context, (bias, bias),
        tuple((s, c, c) for s, c in zip(context.sources, coefficients)), enabled=True)


def _gates(frame, forms, weights):
    result = []
    for i, (form, weight) in enumerate(zip(forms, weights)):
        output = ms.make_value(frame, 'original gate ' + str(i), enabled=True)
        phase = ms.make_phase(output, i, 'original phase ' + str(i), enabled=True)
        result.append((form, output, phase, weight))
    return tuple(result)


def _value(frame, label):
    return ms.make_value(frame, label, enabled=True)


def _mid(interval):
    return (interval[0] + interval[1]) / 2


def _nominal(form):
    return _mid(form.bias), {s: (lo + hi) / 2 for s, lo, hi in form.terms}


def _sum(terms):
    bias, coefficients = Z, {}
    for scale, (b, values) in terms:
        bias += scale * b
        for source, amount in values.items():
            coefficients[source] = coefficients.get(source, Z) + scale * amount
    return bias, coefficients


def _support(form):
    bias, coefficients = form
    return bias + sum((max(a*s.lower, a*s.upper) for s, a in coefficients.items()), Z)


def _secant(form):
    lower, upper = -_support(_sum(((-O, form),))), _support(form)
    if lower >= Z:
        return form
    if upper <= Z:
        return Z, {}
    ideal = upper / (upper - lower)
    quantized = F((ideal * M).__floor__(), M)
    return _sum(((quantized, form), (O, ((O-quantized)*upper, {}))))


def _radius(form):
    return ((form.bias[1]-form.bias[0])/2 +
        sum(((hi-lo)*max(abs(s.lower), abs(s.upper))/2 for s, lo, hi in form.terms), Z))


def _error(v, gates):
    total = _radius(v)
    for form, _, _, weight in gates:
        eps = _radius(form)
        cap = max(Z, _support(_nominal(form)) + eps)
        total += (weight[1]-weight[0])*cap/2 + abs(_mid(weight))*eps
    return total


def _direct(v, gates, first, second, sign):
    nominal = tuple(_nominal(g[0]) for g in gates)
    majorants = tuple(_secant(g) for g in nominal)
    weights = tuple(sign*_mid(g[3]) for g in gates)
    result = []
    for state in ((0, 0), (1, 0), (0, 1), (1, 1)):
        terms = [(sign, _nominal(v)), (weights[first]*state[0], nominal[first]),
                 (weights[second]*state[1], nominal[second])]
        for k, weight in enumerate(weights):
            if k in (first, second):
                continue
            if weight > Z:
                terms.append((weight, majorants[k]))
            elif state == (1, 1):
                terms.append((weight, nominal[k]))
        result.append(_support(_sum(terms)))
    return tuple(result)


def _parameter(interval, choice):
    return _mid(interval) if choice == 2 else interval[choice]


def _eval(form, assignment, choice=2):
    return _parameter(form.bias, choice) + sum((
        _parameter((lo, hi), choice)*assignment[s] for s, lo, hi in form.terms), Z)


def _bits(pre):
    return (0,) if pre < Z else ((1,) if pre > Z else (0, 1))


def _envelope(table, alpha, beta, delta):
    def value(endpoints):
        v00, v10, v01, v11 = endpoints
        return v00+(v10-v00)*alpha+(v01-v00)*beta+(v11-v10-v01+v00)*delta
    return value(table.lower), value(table.upper)


def _rows_hold(compiled, values, phases, overlaps):
    return all(sum((a*values[t] for t, a in vt), Z)
               +sum((a*phases[t] for t, a in pt), Z)
               +sum((a*overlaps[t] for t, a in ot), Z) <= rhs
               for vt, pt, ot, rhs in compiled.rows)


def test_shared_phase_forward_control():
    frame, ctx = _context(((F(-1), O),)*3)
    forms = (_point(ctx, F(1, 2), (O, O, Z)),
             _point(ctx, F(1, 2), (Z, O, O)),
             _point(ctx, F(1, 2), (O, Z, O)))
    gates = _gates(frame, forms, ((O, O), (O, O), (-O, -O)))
    v = _point(ctx, Z, (Z, -O, Z))
    f, g = _value(frame, 'original F'), _value(frame, 'original G')
    prepared_f = pi.prepare(v, gates, f, enabled=True)
    gates_g = tuple((form, q, bit, ((-F(1, 2),)*2 if k == 2 else w))
        for k, (form, q, bit, w) in enumerate(gates))
    prepared_g = pi.prepare(v, gates_g, g, enabled=True)
    tf, tg = (pi.condition(p, 0, 1, enabled=True) for p in (prepared_f, prepared_g))
    assert tf.upper == (O, F(3, 2), F(3, 2), F(3, 2))
    assert tg.upper == (O, F(3, 2), F(3, 2), F(11, 4))
    assert prepared_f.gates == gates and prepared_f.v is v and prepared_f.error == Z
    assert (tf.context, tf.alpha, tf.beta) == (ctx, gates[0][2], gates[1][2])
    r, s = _value(frame, 'original r'), _value(frame, 'original s')
    rb = ms.make_phase(r, 3, 'own r phase', enabled=True)
    sb = ms.make_phase(s, 4, 'own s phase', enabled=True)
    tr = pi.relu(tf, -F(1, 2), r, rb, enabled=True)
    ts = pi.relu(tg, -O, s, sb, enabled=True)
    assert tr.upper == (F(1, 2), O, O, O)
    assert ts.upper == (Z, F(1, 2), F(1, 2), F(7, 4))
    combined = pi.affine((tr, ts), (F(3), F(2)), Z,
                         _value(frame, 'original weighted readout'), enabled=True)
    assert combined.upper == (F(3, 2), F(4), F(4), F(13, 2))
    assert combined.upper[3]-combined.upper[1]-combined.upper[2]+combined.upper[0] == Z
    cancelled = pi.affine((tr, tr), (O, -O), F(7),
                          _value(frame, 'original cancellation'), enabled=True)
    assert cancelled.lower == cancelled.upper == (F(7),)*4
    compiled = pi.compile_rows((tr, ts), enabled=True)
    assert len(compiled.overlaps) == 1 and len(compiled.rows) == 8
    delta_token = compiled.overlaps[0]
    fake_r, fake_s, half = F(19, 20), F(17, 20), F(1, 2)
    assert _envelope(tr, half, half, Z)[0] <= fake_r <= _envelope(tr, half, half, Z)[1]
    assert _envelope(ts, half, half, half)[0] <= fake_s <= _envelope(ts, half, half, half)[1]
    # Shared feasibility would require delta <=1/10 and delta >=7/15.
    assert F(7, 15) > F(1, 10)
    assert F(3)*fake_r+F(2)*fake_s > F(3, 2)+F(5, 2)*(half+half)
    for delta in (Z, F(1, 10), F(7, 15), half):
        assert not _rows_hold(compiled, {r: fake_r, s: fake_s},
            {tf.alpha: half, tf.beta: half}, {delta_token: delta})
    # Check original integer concretization; zero labels are covered below.
    for point in product((F(-1), Z, O), repeat=3):
        assignment = dict(zip(ctx.sources, point))
        pre = tuple(_eval(form, assignment) for form in forms)
        q = tuple(max(Z, a) for a in pre)
        fv, gv = q[0]+q[1]-q[2]-point[1], q[0]+q[1]-q[2]/2-point[1]
        rv, sv = max(Z, fv-F(1, 2)), max(Z, gv-O)
        deficit = q[0]+q[1]-pre[0]-pre[1]
        assert 3*rv+2*sv+F(5, 3)*deficit <= F(13, 2)
        for a, b in product(_bits(pre[0]), _bits(pre[1])):
            assert _rows_hold(compiled, {r: rv, s: sv},
                {tf.alpha: F(a), tf.beta: F(b)}, {delta_token: F(a*b)})
    # Every complete parent-pair hull contains the original strict comparison point.
    witnesses = (((0, 1), (F(9, 10), F(3, 4), F(9, 10))),
                 ((0, 2), (F(9, 10), F(3, 4), F(1, 5))),
                 ((1, 2), (F(1, 5), F(3, 4), F(9, 10))))
    mean = (Z, -F(1, 10), Z)
    targets = (F(43, 40), F(43, 40), F(4, 5))
    for pair, positive in witnesses:
        negative = tuple(2*m-a for m, a in zip(mean, positive))
        assert all(-O < x < O for x in positive+negative)
        for k in pair:
            a = _eval(forms[k], dict(zip(ctx.sources, positive)))
            b = _eval(forms[k], dict(zip(ctx.sources, negative)))
            assert a > Z > b and a/2 == targets[k]


def test_wide_signed_cached_support():
    frame, ctx = _context(((F(-2), O), (O, F(3)), (F(2), F(2))))
    forms = (_point(ctx, F(1, 3), (F(2), -O, F(1, 2))),
             _point(ctx, -F(1, 5), (-F(3, 4), F(2), -F(1, 3))),
             _point(ctx, -F(1, 4), (-O, F(1, 2), F(1, 3))),
             _point(ctx, F(2, 7), (F(1, 5), -F(2, 3), F(1, 2))),
             _point(ctx, Z, (Z, Z, Z)), _point(ctx, F(4), (Z, Z, O)))
    base = _gates(frame, forms, ((O, O),)*6)
    v = _point(ctx, F(3, 10), (F(1, 3), -F(1, 2), F(2, 5)))
    receiver = _value(frame, 'original wide signed receiver')
    for weights in ((O, F(-2), F(2), -O, F(3), -F(1, 2)),
                    (-O, F(-2), F(-2), -O, F(-3), -O), (Z,)*6):
        gates = tuple((form, q, phase, (w, w))
            for (form, q, phase, _), w in zip(base, weights))
        prepared = pi.prepare(v, gates, receiver, enabled=True)
        assert prepared.gates == gates and prepared.error == Z
        for i, j in ((0, 1), (2, 3), (4, 5)):
            table = pi.condition(prepared, i, j, enabled=True)
            assert table.upper == _direct(v, gates, i, j, O)
            assert table.lower == tuple(-x for x in _direct(v, gates, i, j, -O))
            assert table.alpha is gates[i][2] and table.beta is gates[j][2]
            for point in ((F(-2), O, F(2)), (Z, F(2), F(2)), (O, F(3), F(2))):
                assignment = dict(zip(ctx.sources, point))
                pre = tuple(_eval(form, assignment) for form in forms)
                actual = _eval(v, assignment)+sum((w*max(Z, p) for w, p in zip(weights, pre)), Z)
                for a, b in product(_bits(pre[i]), _bits(pre[j])):
                    slot = a+2*b
                    assert table.lower[slot] <= actual <= table.upper[slot]
        both = pi.compile_rows((pi.condition(prepared, 0, 1, enabled=True),
                                pi.condition(prepared, 2, 3, enabled=True)), enabled=True)
        assert len(both.overlaps) == 2


def test_interval_actual_phase_and_quantization():
    frame, ctx = _context(((F(-1), O),))
    source = ctx.sources[0]
    forms = (ss.affine(ctx, (-F(1, 5), F(2, 5)), ((source, O, O),), enabled=True),
             ss.affine(ctx, (-F(1, 10), F(1, 10)), ((source, -O, -O),), enabled=True),
             _point(ctx, F(1, 5), (F(1, 3),)))
    weights = ((-O, F(2)), (F(-2), -F(1, 2)), (O, F(3, 2)))
    gates = _gates(frame, forms, weights)
    v = ss.affine(ctx, (-F(1, 8), F(1, 4)),
                  ((source, F(1, 4), F(5, 12)),), enabled=True)
    prepared = pi.prepare(v, gates, _value(frame, 'original interval consumer'), enabled=True)
    assert prepared.error == _error(v, gates) > Z
    table = pi.condition(prepared, 0, 1, enabled=True)
    assert table.upper == tuple(x+prepared.error for x in _direct(v, gates, 0, 1, O))
    assert table.lower == tuple(-x-prepared.error for x in _direct(v, gates, 0, 1, -O))
    opposite_reference, zero_bits = False, set()
    for point, choice in product((F(-1), Z, O), (0, 1, 2)):
        assignment = {source: point}
        pre = tuple(_eval(form, assignment, choice) for form in forms)
        nominal = _eval(forms[0], assignment)
        opposite_reference |= pre[0] < Z < nominal
        actual = _eval(v, assignment, choice)+sum((
            _parameter(weight, choice)*max(Z, p) for weight, p in zip(weights, pre)), Z)
        for a, b in product(_bits(pre[0]), _bits(pre[1])):
            if pre[1] == Z:
                zero_bits.add(b)
            assert table.lower[a+2*b] <= actual <= table.upper[a+2*b]
    assert opposite_reference and zero_bits == {0, 1}
    # Slopes are dyadic, sound at l/0/u, and within the precise secant error bound.
    exact, inexact = False, False
    for (lower, upper), slope in zip(prepared.midpoint_bounds, prepared.slopes):
        if lower < Z < upper:
            ideal = upper/(upper-lower)
            assert slope == F((ideal*M).__floor__(), M)
            assert (slope.denominator & (slope.denominator-1)) == 0
            exact |= slope == ideal
            inexact |= slope != ideal
            for g in (lower, Z, upper):
                value, secant = slope*g+(O-slope)*upper, ideal*g+(O-ideal)*upper
                assert value >= max(Z, g)
                assert Z <= value-secant < (upper-lower)/M
    assert exact and inexact
    # Stable and boundary cases take the stated rule, retaining every phase.
    frame, ctx = _context(((F(-1), O),))
    stable_forms = tuple(_point(ctx, bias, (coefficient,)) for bias, coefficient in
        ((F(2), O), (F(-2), O), (Z, Z), (O, O), (-O, O)))
    stable_gates = _gates(frame, stable_forms, ((O, O),)*5)
    stable = pi.prepare(_point(ctx, Z, (Z,)), stable_gates,
                        _value(frame, 'original stable receiver'), enabled=True)
    assert stable.gates == stable_gates and len(stable.midpoint_bounds) == 5
    assert stable.slopes == (O, Z, O, O, Z)


def test_identity_default_off_and_limits():
    assert pi.prepare(None, None, None) is None
    assert pi.condition(None, None, None) is None
    assert pi.affine(None, None, None, None) is None
    assert pi.relu(None, None, None, None) is None
    assert pi.compile_rows(None) is None
    with pytest.raises(pi.KernelError):
        pi.prepare(None, None, None, enabled=1)
    frame, ctx = _context(((F(-1), O),))
    forms = (_point(ctx, F(1, 4), (O,)), _point(ctx, F(1, 3), (-O,)))
    gates = _gates(frame, forms, ((O, O), (-O, -O)))
    v, receiver = _point(ctx, Z, (O,)), _value(frame, 'original receiver')
    before = (ctx.sources, tuple(gates), tuple(form.terms for form in forms))
    prepared = pi.prepare(v, gates, receiver, enabled=True)
    table = pi.condition(prepared, 0, 1, enabled=True)
    for i, j in ((1, 0), (0, 0), (-1, 1), (False, 1), (0, 2)):
        with pytest.raises(pi.KernelError):
            pi.condition(prepared, i, j, enabled=True)
    frame2, ctx2 = _context(((F(-1), O),))
    alien = _value(frame2, 'same label is not same original')
    with pytest.raises(pi.KernelError):
        pi.affine((table,), (O,), Z, alien, enabled=True)
    with pytest.raises(pi.KernelError):
        pi.affine((table, replace(table, context=ctx2)), (O, O), Z, receiver, enabled=True)
    with pytest.raises(pi.KernelError):
        pi.compile_rows((table, replace(table, alpha=table.beta, beta=table.alpha)), enabled=True)
    child, other = _value(frame, 'original child'), _value(frame, 'other original child')
    wrong_phase = ms.make_phase(other, 5, 'wrong owner', enabled=True)
    with pytest.raises(pi.KernelError):
        pi.relu(table, Z, child, wrong_phase, enabled=True)
    with pytest.raises(pi.KernelError):
        pi.prepare(v, (gates[0], gates[0]), receiver, enabled=True)
    with pytest.raises(pi.KernelError):
        pi.prepare(v, gates*(pi.MAX_SUPPORT+1), receiver, enabled=True)
    too_large = F(1 << 513)
    with pytest.raises(pi.KernelError):
        pi.prepare(replace(v, bias=(too_large, too_large)), gates, receiver, enabled=True)
    with pytest.raises(pi.KernelError):
        pi.affine((table,), (too_large,), Z, receiver, enabled=True)
    assert before == (ctx.sources, tuple(gates), tuple(form.terms for form in forms))
    assert prepared.context is ctx and table.receiver is receiver
    assert gates[0][2].original_output is gates[0][1]
    assert gates[1][2].original_output is gates[1][1]
