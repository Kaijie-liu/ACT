"""Four fixed mathematical groups; first execution belongs to the frozen gate.

The independent support reference uses vertices of a small box and intersections
of H=0 with its edges. It does not use the candidate's sorting/tree algorithm,
solve a phase subproblem, sample a model, or claim that a box witness is an ADV.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms
from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss
from experiments.neural_hz_20260831.definition_first_20260928.d066_wide_phase_interface_20261001 import phase_interface as pi
from experiments.neural_hz_20260831.definition_first_20260928.d070_joint_negative_component_20261001 import hinge_support as hs
from experiments.neural_hz_20260831.definition_first_20260928.d070_joint_negative_component_20261001 import negative_interface as ni

Z, O, M = F(0), F(1), 2**24


def _linear(bias, terms, point):
    return bias + sum((coefficient*point[ordinal] for ordinal, coefficient in terms), Z)


def _geometric(bounds, a_bias, a_terms, h_bias, h_terms):
    """Exact in these small dimensions: clipped-box vertices suffice."""
    ordinals = tuple(item[0] for item in bounds)
    candidates = [dict(zip(ordinals, coordinates))
        for coordinates in product(*((lo, hi) for _, lo, hi in bounds))]
    h = dict(h_terms)
    for ordinal, lower, upper in bounds:
        coefficient = h.get(ordinal, Z)
        if coefficient == Z or lower == upper:
            continue
        others = tuple(item for item in bounds if item[0] != ordinal)
        for endpoints in product(*((lo, hi) for _, lo, hi in others)):
            point = dict(zip((item[0] for item in others), endpoints))
            rest = h_bias + sum((a*point[j] for j, a in h_terms if j != ordinal), Z)
            crossing = -rest/coefficient
            if lower <= crossing <= upper:
                point[ordinal] = crossing
                candidates.append(point)
    return max(_linear(a_bias, a_terms, point)
               -max(Z, _linear(h_bias, h_terms, point)) for point in candidates)


def _updated(terms, updates):
    coefficients = dict(terms)
    for ordinal, amount in updates:
        coefficients[ordinal] = coefficients.get(ordinal, Z)+amount
    return tuple(sorted(coefficients.items()))


def _tree(node):
    if node is None:
        return 0, 0, Z, Z, ()
    lh, ln, lc, lw, left = _tree(node.left)
    rh, rn, rc, rw, right = _tree(node.right)
    assert abs(lh-rh) <= 1 and node.height == 1+max(lh, rh)
    assert node.capacity > Z and node.weighted == node.rho*node.capacity
    assert node.size == 1+ln+rn
    assert node.total_capacity == lc+node.capacity+rc
    assert node.total_weighted == lw+node.weighted+rw
    order = left+((node.rho, node.ordinal),)+right
    assert order == tuple(sorted(order, key=lambda item: (-item[0], item[1])))
    return node.height, node.size, node.total_capacity, node.total_weighted, order


def _context(bounds):
    frame = ms.make_frame('original joint-negative source', enabled=True)
    values = tuple(ms.make_value(frame, 'source '+str(i), enabled=True)
                   for i in range(len(bounds)))
    context = ms.source_context(frame, tuple((v, i, *interval)
        for i, (v, interval) in enumerate(zip(values, bounds))), enabled=True)
    return frame, context


def _point(context, bias, coefficients):
    return ss.affine(context, (bias, bias), tuple((s, c, c)
        for s, c in zip(context.sources, coefficients)), enabled=True)


def _gates(frame, forms, weights):
    result = []
    for i, (form, weight) in enumerate(zip(forms, weights)):
        output = ms.make_value(frame, 'original gate '+str(i), enabled=True)
        phase = ms.make_phase(output, i, 'original phase '+str(i), enabled=True)
        result.append((form, output, phase, weight))
    return tuple(result)


def _value(frame, name):
    return ms.make_value(frame, name, enabled=True)


def _mid(interval):
    return (interval[0]+interval[1])/2


def _nominal(form):
    return _mid(form.bias), {s.ordinal: (lo+hi)/2 for s, lo, hi in form.terms}


def _sum(terms):
    bias, coefficients = Z, {}
    for scale, (constant, values) in terms:
        bias += scale*constant
        for source, amount in values.items():
            coefficients[source] = coefficients.get(source, Z)+scale*amount
    return bias, coefficients


def _box(context):
    return tuple((s.ordinal, s.lower, s.upper) for s in context.sources)


def _affine_support(form, bounds):
    bias, coefficients = form
    return bias+sum((max(coefficients.get(i, Z)*lo,
                         coefficients.get(i, Z)*hi) for i, lo, hi in bounds), Z)


def _secant(form, bounds):
    lower = -_affine_support(_sum(((-O, form),)), bounds)
    upper = _affine_support(form, bounds)
    if lower >= Z:
        return form
    if upper <= Z:
        return Z, {}
    ideal = upper/(upper-lower)
    quantized = F((ideal*M).__floor__(), M)
    return _sum(((quantized, form), (O, ((O-quantized)*upper, {}))))


def _direct(v, gates, first, second, sign):
    """Full affine expansion plus geometry, independent of both cached bases."""
    bounds = _box(v.context)
    forms = tuple(_nominal(item[0]) for item in gates)
    weights = tuple(sign*_mid(item[3]) for item in gates)
    majorants = tuple(_secant(form, bounds) for form in forms)
    values = []
    for state in ((0, 0), (1, 0), (0, 1), (1, 1)):
        positive = [(sign, _nominal(v)), (weights[first]*state[0], forms[first]),
                    (weights[second]*state[1], forms[second])]
        negative = []
        for k, weight in enumerate(weights):
            if k in (first, second):
                continue
            if weight > Z:
                positive.append((weight, majorants[k]))
            elif weight < Z:
                negative.append((-weight, forms[k]))
        a, h = _sum(positive), _sum(negative)
        values.append(_geometric(bounds, a[0], tuple(sorted(a[1].items())),
                                 h[0], tuple(sorted(h[1].items()))))
    return tuple(values)


def _radius(form):
    return (form.bias[1]-form.bias[0])/2 + sum((
        (hi-lo)*max(abs(s.lower), abs(s.upper))/2 for s, lo, hi in form.terms), Z)


def _error(v, gates):
    total = _radius(v)
    for form, _, _, weight in gates:
        epsilon = _radius(form)
        cap = max(Z, _affine_support(_nominal(form), _box(form.context))+epsilon)
        total += (weight[1]-weight[0])*cap/2+abs(_mid(weight))*epsilon
    return total


def _parameter(interval, choice):
    return _mid(interval) if choice == 2 else interval[choice]


def _eval(form, assignment, choice=2):
    return _parameter(form.bias, choice)+sum((
        _parameter((lo, hi), choice)*assignment[s] for s, lo, hi in form.terms), Z)


def _bits(value):
    return (0,) if value < Z else ((1,) if value > Z else (0, 1))


def _nested(new, old):
    assert new.context is old.context and new.alpha is old.alpha and new.beta is old.beta
    assert all(a >= b for a, b in zip(new.lower, old.lower))
    assert all(a <= b for a, b in zip(new.upper, old.upper))


def _rows_hold(compiled, values, phases, overlaps):
    return all(sum((a*values[t] for t, a in vt), Z)
               +sum((a*phases[t] for t, a in pt), Z)
               +sum((a*overlaps[t] for t, a in ot), Z) <= rhs
               for vt, pt, ot, rhs in compiled.rows)


def test_primal_support_and_sparse_updates():
    bounds = ((0, F(-2), O), (3, Z, F(3)), (7, F(2), F(2)), (9, -O, F(2)))
    a_bias, a_terms = F(3, 10), ((0, F(1, 3)), (3, -F(1, 2)), (7, F(2, 5)))
    h_bias, h_terms = -F(1, 5), ((0, F(2)), (3, -O), (7, F(1, 3)))
    base = hs.prepare(bounds, a_bias, a_terms, h_bias, h_terms, enabled=True)
    root, original_tree = base.root, _tree(base.root)
    original_coordinates = tuple(base.coordinates.items())
    changes = ((Z, (), Z, ()),
        (F(1, 7), ((0, -F(1, 3)), (7, -F(1, 4)), (9, F(2, 5))),
         -F(2, 3), ((0, F(-4)), (3, O), (7, -F(1, 3)), (9, F(3, 4)))),
        (-F(1, 8), ((3, F(3, 2)),), F(2, 5), ((0, F(-2)), (3, F(2)))),
        (Z, (), F(20), ()), (Z, (), F(-20), ()))
    for da, updates_a, dh, updates_h in changes:
        result = hs.query(base, a_bias_delta=da, a_updates=updates_a,
                          h_bias_delta=dh, h_updates=updates_h, enabled=True)
        a, h = _updated(a_terms, updates_a), _updated(h_terms, updates_h)
        assert result.value == _geometric(bounds, a_bias+da, a, h_bias+dh, h)
        point_items = hs.decode(result, enabled=True)
        assert tuple(i for i, _ in point_items) == tuple(i for i, _, _ in bounds)
        point = dict(point_items)
        assert all(lo <= point[i] <= hi for i, lo, hi in bounds)
        assert result.value == _linear(a_bias+da, a, point)-max(Z, _linear(h_bias+dh, h, point))
        _tree(result.root)
        assert result.base is base and base.root is root
        assert _tree(base.root) == original_tree
        assert tuple(base.coordinates.items()) == original_coordinates
    # rho=1 and rho=0 ties: the selected filling witness lies on the hinge;
    # other maximizers may be vertices.
    tied_bounds = ((0, Z, O), (1, Z, O), (2, Z, O))
    tied = hs.prepare(tied_bounds, F(1, 8), ((0, O), (1, O), (2, Z)),
                      -F(1, 2), ((0, O), (1, O), (2, -O)), enabled=True)
    result = hs.query(tied, enabled=True)
    assert result.value == F(13, 8)
    assert result.value == _geometric(tied_bounds, F(1, 8), ((0, O), (1, O), (2, Z)),
                                      -F(1, 2), ((0, O), (1, O), (2, -O)))
    tied_point = dict(hs.decode(result, enabled=True))
    assert _linear(-F(1, 2), ((0, O), (1, O), (2, -O)), tied_point) == Z
    _tree(result.root)
    # No denominator nodes: independent affine coordinates and a constant H.
    constant = hs.prepare(bounds, F(1, 3), a_terms, -F(1, 2), (), enabled=True)
    assert constant.root is None
    assert hs.query(constant, enabled=True).value == _geometric(bounds, F(1, 3), a_terms, -F(1, 2), ())
    with pytest.raises((AttributeError, TypeError)):
        base.root.height = 0
    with pytest.raises(TypeError):
        base.coordinates[0] = None


def test_shared_negative_forward_control():
    frame, ctx = _context(((-O, O),)*3)
    forms = (_point(ctx, F(1, 10), (O, Z, Z)),
             _point(ctx, F(1, 10), (F(4, 5), O, Z)),
             _point(ctx, F(1, 5), (F(4, 5), Z, O)))
    gates = _gates(frame, forms, ((-O, -O), (F(1, 2),)*2, (F(1, 2),)*2))
    v = _point(ctx, F(1, 20), (F(1, 2), Z, Z))
    receiver = _value(frame, 'original F')
    prepared = ni.prepare(v, gates, receiver, enabled=True)
    new = ni.condition(prepared, 1, 2, enabled=True)
    old = pi.condition(prepared.original, 1, 2, enabled=True)
    assert prepared.original.gates == gates and prepared.original.v is v
    assert prepared.original.error == Z
    assert new.upper == (Z, F(51, 100), F(14, 25), F(7, 5))
    assert old.upper == (F(11, 20), F(3, 2), F(31, 20), F(7, 5))
    assert new.upper[3]-new.upper[1]-new.upper[2]+new.upper[0] == F(33, 100)
    assert old.upper[3]-old.upper[1]-old.upper[2]+old.upper[0] == -F(11, 10)
    _nested(new, old)
    child = _value(frame, 'original next ReLU')
    own_phase = ms.make_phase(child, 3, 'original next phase', enabled=True)
    new_child = ni.relu(new, -F(1, 4), child, own_phase, enabled=True)
    old_child = pi.relu(old, -F(1, 4), child, own_phase, enabled=True)
    _nested(new_child, old_child)
    assert new_child.upper == (Z, F(13, 50), F(31, 100), F(23, 20))
    assert max(new_child.upper) == F(23, 20) < max(old_child.upper) == F(13, 10)
    downstream = _value(frame, 'original mixed residual readout')
    mixed_new = ni.affine((new, new_child), (O, F(-2)), F(1, 10), downstream, enabled=True)
    mixed_old = pi.affine((old, old_child), (O, F(-2)), F(1, 10), downstream, enabled=True)
    _nested(mixed_new, mixed_old)
    cancelled = ni.affine((new, new), (O, -O), F(7),
                          _value(frame, 'original cancelling readout'), enabled=True)
    assert cancelled.lower == cancelled.upper == (F(7),)*4
    compiled = ni.compile_rows((new, new_child), enabled=True)
    assert len(compiled.overlaps) == 1 and len(compiled.rows) == 8
    assert new.alpha is gates[1][2] and new.beta is gates[2][2]
    attaining = ((-F(1, 10), -O, -O), (-F(1, 10), O, -O),
                 (-F(1, 10), -O, O), (O, O, O))
    for slot, coordinates in enumerate(attaining):
        assignment = dict(zip(ctx.sources, coordinates))
        pre = tuple(_eval(form, assignment) for form in forms)
        value = _eval(v, assignment)+sum((_mid(g[3])*max(Z, p)
                                         for g, p in zip(gates, pre)), Z)
        assert value == new.upper[slot]
        assert pre[1] != Z and pre[2] != Z
        assert int(pre[1] > Z)+2*int(pre[2] > Z) == slot
    points = attaining+((Z, Z, Z), (F(1, 2),)*3, (Z, -F(1, 10), -F(1, 5)))
    zero_choices = set()
    child_signs = set()
    for coordinates in points:
        assignment = dict(zip(ctx.sources, coordinates))
        pre = tuple(_eval(form, assignment) for form in forms)
        value = _eval(v, assignment)+sum((_mid(g[3])*max(Z, p)
                                         for g, p in zip(gates, pre)), Z)
        child_pre = value-F(1, 4)
        child_signs.add(child_pre > Z)
        for a, b in product(_bits(pre[1]), _bits(pre[2])):
            if pre[1] == pre[2] == Z:
                zero_choices.add((a, b))
            assert _rows_hold(compiled, {receiver: value, child: max(Z, child_pre)},
                {new.alpha: F(a), new.beta: F(b)}, {compiled.overlaps[0]: F(a*b)})
    assert zero_choices == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert child_signs == {False, True}


def test_interval_signed_wide_dominance():
    frame, ctx = _context(((F(-2), O), (Z, F(3)), (F(2), F(2))))
    forms = (_point(ctx, F(1, 3), (F(2), -O, F(1, 2))),
             _point(ctx, -F(1, 5), (-F(3, 4), F(2), -F(1, 3))),
             _point(ctx, -F(1, 4), (-O, F(1, 2), F(1, 3))),
             _point(ctx, F(2, 7), (F(1, 5), -F(2, 3), F(1, 2))),
             _point(ctx, Z, (Z, Z, Z)), _point(ctx, F(4), (Z, Z, O)))
    original = _gates(frame, forms, ((O, O),)*6)
    v = _point(ctx, F(3, 10), (F(1, 3), -F(1, 2), F(2, 5)))
    receiver = _value(frame, 'original wide consumer')
    for weights in ((O, F(-2), F(2), -O, F(3), -F(1, 2)),
                    (O, F(2), O, O, F(3), O),
                    (-O, F(-2), F(-2), -O, F(-3), -O), (Z,)*6):
        gates = tuple((g, q, phase, (w, w))
                      for (g, q, phase, _), w in zip(original, weights))
        prepared = ni.prepare(v, gates, receiver, enabled=True)
        assert prepared.original.gates == gates and prepared.original.error == Z
        for i, j in ((0, 1), (2, 3), (4, 5)):
            new = ni.condition(prepared, i, j, enabled=True)
            old = pi.condition(prepared.original, i, j, enabled=True)
            _nested(new, old)
            assert new.upper == _direct(v, gates, i, j, O)
            assert new.lower == tuple(-a for a in _direct(v, gates, i, j, -O))
            for coordinates in ((F(-2), O, F(2)), (Z, F(2), F(2)), (O, F(3), F(2))):
                assignment = dict(zip(ctx.sources, coordinates))
                pre = tuple(_eval(form, assignment) for form in forms)
                actual = _eval(v, assignment)+sum((w*max(Z, p) for w, p in zip(weights, pre)), Z)
                for a, b in product(_bits(pre[i]), _bits(pre[j])):
                    assert new.lower[a+2*b] <= actual <= new.upper[a+2*b]
    # Interval coefficients: actual phase may differ from the midpoint's phase.
    frame, ctx = _context(((-O, O),))
    source = ctx.sources[0]
    forms = (ss.affine(ctx, (-F(1, 5), F(2, 5)), ((source, O, O),), enabled=True),
             ss.affine(ctx, (-F(1, 10), F(1, 10)), ((source, -O, -O),), enabled=True),
             _point(ctx, F(1, 5), (F(1, 3),)))
    gates = _gates(frame, forms, ((-O, F(2)), (F(-2), -F(1, 2)), (O, F(3, 2))))
    v = ss.affine(ctx, (-F(1, 8), F(1, 4)), ((source, F(1, 4), F(5, 12)),), enabled=True)
    prepared = ni.prepare(v, gates, _value(frame, 'original interval consumer'), enabled=True)
    new = ni.condition(prepared, 0, 1, enabled=True)
    old = pi.condition(prepared.original, 0, 1, enabled=True)
    epsilon = _error(v, gates)
    assert prepared.original.error == epsilon > Z
    assert new.upper == tuple(a+epsilon for a in _direct(v, gates, 0, 1, O))
    assert new.lower == tuple(-a-epsilon for a in _direct(v, gates, 0, 1, -O))
    _nested(new, old)
    opposite, zero_bits = False, set()
    for point, choice in product((-O, Z, O), (0, 1, 2)):
        assignment = {source: point}
        pre = tuple(_eval(form, assignment, choice) for form in forms)
        opposite |= pre[0] < Z < _eval(forms[0], assignment)
        actual = _eval(v, assignment, choice)+sum((
            _parameter(g[3], choice)*max(Z, p) for g, p in zip(gates, pre)), Z)
        for a, b in product(_bits(pre[0]), _bits(pre[1])):
            if pre[1] == Z:
                zero_bits.add(b)
            assert new.lower[a+2*b] <= actual <= new.upper[a+2*b]
    assert opposite and zero_bits == {0, 1}


def test_identity_default_off_and_limits():
    assert hs.prepare(None, None, None, None, None) is None
    assert hs.query(None, a_bias_delta=None, a_updates=None, h_bias_delta=None, h_updates=None) is None
    assert hs.decode(None) is None
    assert ni.prepare(None, None, None) is None
    assert ni.condition(None, None, None) is None
    assert ni.affine(None, None, None, None) is None
    assert ni.relu(None, None, None, None) is None
    assert ni.compile_rows(None) is None
    for operation in (lambda: hs.prepare(None, None, None, None, None, enabled=1),
                      lambda: ni.prepare(None, None, None, enabled=1)):
        with pytest.raises(hs.KernelError):
            operation()
    bounds = ((0, -O, O), (3, Z, O))
    base = hs.prepare(bounds, Z, ((0, O),), Z, ((3, O),), enabled=True)
    for bad in (list(bounds), ((True, -O, O),), ((-1, -O, O),),
                ((0, O, -O),), ((0, -1, O),), ((0, -O, O), (0, Z, O)),
                tuple(reversed(bounds)), ((0, -O),)):
        with pytest.raises(hs.KernelError):
            hs.prepare(bad, Z, (), Z, (), enabled=True)
    for bad in ([(0, O)], ((0, O), (0, O)), ((4, O),), ((True, O),),
                ((3, O), (0, O)), ((0, 1),), ((0, O, O),)):
        with pytest.raises(hs.KernelError):
            hs.prepare(bounds, Z, bad, Z, (), enabled=True)
        with pytest.raises(hs.KernelError):
            hs.query(base, a_updates=bad, enabled=True)
        with pytest.raises(hs.KernelError):
            hs.query(base, h_updates=bad, enabled=True)
    with pytest.raises(hs.KernelError):
        hs.query(object(), enabled=True)
    with pytest.raises(hs.KernelError):
        hs.decode(base, enabled=True)
    huge = F(1 << 513)
    for operation in (lambda: hs.prepare(bounds, huge, (), Z, (), enabled=True),
                      lambda: hs.query(base, a_bias_delta=huge, enabled=True),
                      lambda: hs.query(base, h_updates=((0, huge),), enabled=True),
                      lambda: hs.prepare(bounds, Z, ((0, O),)*(hs.MAX_SUPPORT+1), Z, (), enabled=True),
                      lambda: hs.query(base, a_updates=((0, O),)*(hs.MAX_SUPPORT+1), enabled=True)):
        with pytest.raises(hs.KernelError):
            operation()
    # Every input is small enough, but the required arithmetic is not.
    big = F(1 << 300)
    with pytest.raises(hs.KernelError):
        hs.prepare(((0, Z, big),), Z, ((0, big),), Z, (), enabled=True)
    denominator = 1 << 260
    with pytest.raises(hs.KernelError):
        hs.prepare(((0, Z, O), (1, Z, O)), Z,
            ((0, F(denominator, denominator-1)),
             (1, F(denominator-2, denominator-3))),
            Z, ((0, O), (1, O)), enabled=True)
    frame, ctx = _context(((-O, O),))
    forms = (_point(ctx, F(1, 4), (O,)), _point(ctx, F(1, 3), (-O,)))
    gates = _gates(frame, forms, ((O, O), (-O, -O)))
    v, receiver = _point(ctx, Z, (O,)), _value(frame, 'original receiver')
    before = (ctx.sources, gates, tuple(form.terms for form in forms))
    prepared = ni.prepare(v, gates, receiver, enabled=True)
    table = ni.condition(prepared, 0, 1, enabled=True)
    for i, j in ((1, 0), (0, 0), (-1, 1), (False, 1), (0, 2)):
        with pytest.raises(ni.KernelError):
            ni.condition(prepared, i, j, enabled=True)
    with pytest.raises(ni.KernelError):
        ni.condition(prepared.original, 0, 1, enabled=True)
    with pytest.raises(ni.KernelError):
        ni.prepare(v, (gates[0], gates[0]), receiver, enabled=True)
    with pytest.raises(ni.KernelError):
        ni.prepare(v, gates*(ni.MAX_SUPPORT+1), receiver, enabled=True)
    frame2, ctx2 = _context(((-O, O),))
    with pytest.raises(ni.KernelError):
        ni.affine((table,), (O,), Z, _value(frame2, 'alien readout'), enabled=True)
    with pytest.raises(ni.KernelError):
        ni.compile_rows((table, replace(table, context=ctx2)), enabled=True)
    with pytest.raises(ni.KernelError):
        ni.compile_rows((replace(table, alpha=table.beta, beta=table.alpha),), enabled=True)
    child, other = _value(frame, 'child'), _value(frame, 'other child')
    wrong = ms.make_phase(other, 5, 'wrong owner', enabled=True)
    with pytest.raises(ni.KernelError):
        ni.relu(table, Z, child, wrong, enabled=True)
    assert before == (ctx.sources, gates, tuple(form.terms for form in forms))
