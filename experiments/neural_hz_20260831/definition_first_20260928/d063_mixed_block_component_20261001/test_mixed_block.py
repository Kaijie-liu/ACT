"""Four fixed mathematical fixtures for the default-off D062 block rule.

Formal identities and boxes below are declared proof premises, not native
model/decoder authentication.  The finite rational cases are neither attacks
nor model sampling.  Authoring is static: collection and execution belong only
to the root's complete, frozen mathematical qualification run.
"""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms
from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss
from experiments.neural_hz_20260831.definition_first_20260928.d063_mixed_block_component_20261001 import mixed_block as mb


ZERO, ONE = F(0), F(1)


def _frame(bounds):
    frame = ms.make_frame("original mathematical frontier", enabled=True)
    values = tuple(ms.make_value(frame, "source " + str(i), enabled=True)
                   for i in range(len(bounds)))
    context = ms.source_context(frame, tuple((value, i, lo, hi)
                                for i, (value, (lo, hi)) in enumerate(zip(values, bounds))),
                                enabled=True)
    return frame, context, values


def _point(context, bias, coefficients):
    return ss.affine(context, (bias, bias),
                     tuple((s, a, a) for s, a in zip(context.sources, coefficients)),
                     enabled=True)


def _gates(frame, forms, weights):
    result = []
    for i, (form, weight) in enumerate(zip(forms, weights)):
        q = ms.make_value(frame, "original gate " + str(i), enabled=True)
        alpha = ms.make_phase(q, i, "original phase " + str(i), enabled=True)
        result.append((form, q, alpha, weight))
    return tuple(result)


def _mid(interval):
    return (interval[0] + interval[1]) / 2


def _nominal(form):
    return _mid(form.bias), {source: (lo + hi) / 2 for source, lo, hi in form.terms}


def _combine(terms):
    # Deliberately rebuild complete planes: independent of the candidate's
    # cached absolute-value increment implementation.
    bias, coefficients = ZERO, {}
    for scale, (constant, values) in terms:
        bias += scale * constant
        for source, coefficient in values.items():
            coefficients[source] = coefficients.get(source, ZERO) + scale * coefficient
    return bias, coefficients


def _support(form):
    bias, coefficients = form
    return bias + sum((max(a * s.lower, a * s.upper)
                       for s, a in coefficients.items()), ZERO)


def _reference_direction(v, gates, sign):
    base = _combine(((sign, _nominal(v)),))
    positives, negatives = [], []
    for preactivation, _, _, weight in gates:
        coefficient = sign * _mid(weight)
        if coefficient > ZERO:
            positives.append(_combine(((coefficient, _nominal(preactivation)),)))
        elif coefficient < ZERO:
            negatives.append((-coefficient, _nominal(preactivation)))
    if not positives:
        value = _support(base)
        return value, ((value,),)
    count = (len(positives) + 1) // 2
    negative = _combine(tuple(negatives))
    common = _combine(((F(1, count), base),))
    corrected = _combine(((F(1, count), base), (-F(1, count), negative)))
    groups = []
    for i in range(0, len(positives), 2):
        first = positives[i]
        if i + 1 == len(positives):
            planes = (common, _combine(((ONE, corrected), (ONE, first))))
        else:
            second = positives[i + 1]
            planes = (common,
                      _combine(((ONE, common), (ONE, first))),
                      _combine(((ONE, common), (ONE, second))),
                      _combine(((ONE, corrected), (ONE, first), (ONE, second))))
        groups.append(tuple(_support(plane) for plane in planes))
    return sum((max(group) for group in groups), ZERO), tuple(groups)


def _radius(form):
    return ((form.bias[1] - form.bias[0]) / 2
            + sum(((hi - lo) * max(abs(s.lower), abs(s.upper)) / 2
                   for s, lo, hi in form.terms), ZERO))


def _error_reference(v, gates):
    total = _radius(v)
    for preactivation, _, _, weight in gates:
        epsilon = _radius(preactivation)
        actual_upper = max(ZERO, _support(_nominal(preactivation)) + epsilon)
        total += ((weight[1] - weight[0]) * actual_upper / 2
                  + abs(_mid(weight)) * epsilon)
    return total


def _parameter(interval, choice):
    return _mid(interval) if choice == 2 else interval[choice]


def _concrete(form, values, choice=2):
    return (_parameter(form.bias, choice)
            + sum((_parameter((lo, hi), choice) * values[source]
                   for source, lo, hi in form.terms), ZERO))


def _bits(value):
    return (ZERO,) if value < ZERO else ((ONE,) if value > ZERO else (ZERO, ONE))


def _rows_hold(rows, values, phases):
    return all(sum((a * values[token] for token, a in value_terms), ZERO)
               + sum((a * phases[token] for token, a in phase_terms), ZERO) <= rhs
               for value_terms, phase_terms, rhs in rows)


def test_complete_pair_control_and_forward():
    frame, ctx, _ = _frame(((F(-1), ONE),) * 3)
    x, y, z = ctx.sources
    forms = (_point(ctx, F(1, 10), (ONE, ONE, ZERO)),
             _point(ctx, F(1, 10), (ZERO, ONE, ONE)),
             _point(ctx, F(1, 10), (ONE, ZERO, ONE)))
    gates = _gates(frame, forms, ((ONE, ONE), (ONE, ONE), (-ONE, -ONE)))
    receiver = ms.make_value(frame, "original F consumer", enabled=True)
    v = _point(ctx, ZERO, (ZERO, -ONE, ZERO))
    certificate = mb.generate(v, gates, receiver, enabled=True)
    assert certificate.context is ctx and certificate.v is v
    assert certificate.receiver is receiver and certificate.gates == gates
    assert (certificate.lower, certificate.upper, certificate.error) == (F(-1), F(11, 10), ZERO)
    assert certificate.upper_groups == ((ONE, F(11, 10), F(11, 10), F(11, 10)),)
    assert certificate.lower_groups == ((ONE, F(9, 10)),)
    assert certificate.rows == ((((receiver, ONE),), (), F(11, 10)),
                                (((receiver, -ONE),), (), ONE))
    assert certificate.observation.context is ctx
    assert certificate.observation.readout.terms == ((receiver, ONE),)
    assert certificate.observation.lower.bias == certificate.lower
    assert certificate.observation.upper.bias == certificate.upper
    assert ms.compile_rows(certificate.observation, enabled=True) == certificate.rows

    outputs = (F(17, 20), F(17, 20), F(1, 8))
    witnesses = (((0, 1), (F(4, 5), F(4, 5), F(4, 5))),
                 ((0, 2), (F(4, 5), F(4, 5), F(-13, 20))),
                 ((1, 2), (F(-13, 20), F(4, 5), F(4, 5))))
    for pair, active in witnesses:
        inactive = tuple(-value for value in active)
        assert all(-ONE < value < ONE for value in active + inactive)
        assert tuple((a + b) / 2 for a, b in zip(active, inactive)) == (ZERO,) * 3
        for index in pair:
            positive = _concrete(forms[index], dict(zip(ctx.sources, active)))
            negative = _concrete(forms[index], dict(zip(ctx.sources, inactive)))
            assert positive > ZERO > negative
            assert positive / 2 == outputs[index]
            assert _bits(positive) == (ONE,) and _bits(negative) == (ZERO,)
            assert (ONE + ZERO) / 2 == F(1, 2)
    fake = outputs[0] + outputs[1] - outputs[2]
    assert fake == F(63, 40) and fake - certificate.upper == F(19, 40)
    assert not _rows_hold(certificate.rows, {receiver: fake}, {})
    assert max(ZERO, certificate.upper - F(1, 2)) == F(3, 5) < F(7, 10)
    assert max(ZERO, fake - F(1, 2)) == F(43, 40) > F(7, 10)
    # A real next gate is crossing, and the certified upper endpoint is attained.
    expected = (((ZERO, ZERO, ZERO), F(1, 10)),
                ((F(4, 5),) * 3, F(9, 10)), ((ONE,) * 3, F(11, 10)))
    child_signs = set()
    for point, value in expected:
        assignment = dict(zip(ctx.sources, point))
        pre = tuple(_concrete(form, assignment) for form in forms)
        actual = max(ZERO, pre[0]) + max(ZERO, pre[1]) - max(ZERO, pre[2]) - point[1]
        assert actual == value and certificate.lower <= actual <= certificate.upper
        child_signs.add(actual - F(1, 2) > ZERO)
        assert max(ZERO, actual - F(1, 2)) <= F(3, 5)
    assert child_signs == {False, True}
    assert tuple(gate[2].original_output for gate in certificate.gates) == tuple(gate[1] for gate in gates)


def test_cached_support_and_signed_groups():
    frame, ctx, _ = _frame(((F(-2), ONE), (ONE, F(3)), (F(2), F(2))))
    forms = (_point(ctx, F(1, 3), (F(2), -ONE, F(1, 2))),
             _point(ctx, F(-1, 5), (F(-3, 4), F(2), F(-1, 3))),
             _point(ctx, F(-1, 4), (-ONE, F(1, 2), F(1, 3))),
             _point(ctx, F(2, 7), (F(1, 5), F(-2, 3), F(1, 2))),
             _point(ctx, ZERO, (ZERO, ZERO, ZERO)))
    v = _point(ctx, F(3, 10), (F(1, 3), F(-1, 2), F(2, 5)))
    base_gates = _gates(frame, forms, ((ONE, ONE),) * len(forms))
    receiver = ms.make_value(frame, "original mixed receiver", enabled=True)
    weights = ((ONE, F(-2), F(2), -ONE, F(3)),
               (-ONE, F(-2), F(-2), -ONE, F(-3)),
               (ZERO,) * 5, (ONE, ZERO, ZERO, ZERO, ZERO))
    for amounts in weights:
        gates = tuple((form, q, phase, (amount, amount))
                      for (form, q, phase, _), amount in zip(base_gates, amounts))
        certificate = mb.generate(v, gates, receiver, enabled=True)
        upper, upper_groups = _reference_direction(v, gates, ONE)
        minus_lower, lower_groups = _reference_direction(v, gates, -ONE)
        assert certificate.error == ZERO
        assert certificate.upper_groups == upper_groups
        assert certificate.lower_groups == lower_groups
        assert (certificate.lower, certificate.upper) == (-minus_lower, upper)
        assert certificate.gates == gates  # Even weight-zero and constant-zero originals remain.
        if amounts == weights[0]:
            assert tuple(map(len, certificate.upper_groups)) == (4, 2)
            assert tuple(map(len, certificate.lower_groups)) == (4,)
        if all(amount == ZERO for amount in amounts):
            assert tuple(map(len, certificate.upper_groups)) == (1,)
            assert tuple(map(len, certificate.lower_groups)) == (1,)
        for point in ((F(-2), ONE, F(2)), (ONE, F(3), F(2)), (ZERO, F(2), F(2))):
            assignment = dict(zip(ctx.sources, point))
            actual = _concrete(v, assignment) + sum((amount * max(ZERO, _concrete(form, assignment))
                         for form, amount in zip(forms, amounts)), ZERO)
            assert certificate.lower <= actual <= certificate.upper
    # The first two positive increments in the odd grouping cancel BOTH
    # nonfixed source coefficients before absolute values are taken.
    combined = _combine(((ONE, _nominal(forms[0])), (F(2), _nominal(forms[2]))))
    assert combined[1][ctx.sources[0]] == combined[1][ctx.sources[1]] == ZERO


def test_interval_parameters_and_zero_phases():
    frame, ctx, _ = _frame(((F(-1), ONE), (ZERO, F(2))))
    x, y = ctx.sources
    v = ss.affine(ctx, (F(1, 10), F(3, 10)),
                  ((x, F(-1, 10), F(1, 10)), (y, F(1, 5), F(2, 5))), enabled=True)
    forms = (ss.affine(ctx, (F(-1, 5), F(1, 5)),
                       ((x, F(1, 2), F(3, 2)), (y, F(-1, 4), F(1, 4))), enabled=True),
             ss.affine(ctx, (ZERO, F(1, 5)),
                       ((x, -ONE, F(-1, 2)), (y, F(1, 4), F(3, 4))), enabled=True),
             _point(ctx, F(-1, 10), (F(1, 4), F(-1, 2))))
    gates = _gates(frame, forms, ((-ONE, ONE), (F(1, 2), F(3, 2)), (F(-3, 2), F(-1, 2))))
    receiver = ms.make_value(frame, "original interval receiver", enabled=True)
    certificate = mb.generate(v, gates, receiver, enabled=True)
    error = _error_reference(v, gates)
    assert error == F(39, 8) == certificate.error
    upper, up_groups = _reference_direction(v, gates, ONE)
    minus_lower, lo_groups = _reference_direction(v, gates, -ONE)
    assert certificate.upper_groups == up_groups and certificate.lower_groups == lo_groups
    assert (certificate.lower, certificate.upper) == (-minus_lower - error, upper + error)
    # Endpoints and interior coefficient choices, including actual positive
    # and negative weights whose reference midpoint is zero.
    for parameter_choice, weight_choice in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0)):
        for point in ((F(-1), ZERO), (F(-1), F(2)), (ONE, ZERO), (ONE, F(2)), (ZERO, ONE)):
            assignment = dict(zip(ctx.sources, point))
            actual = _concrete(v, assignment, parameter_choice)
            for preactivation, _, _, weight in gates:
                actual += (_parameter(weight, weight_choice)
                           * max(ZERO, _concrete(preactivation, assignment, parameter_choice)))
            assert certificate.lower <= actual <= certificate.upper
            assert _rows_hold(certificate.rows, {receiver: actual}, {})

    frame, ctx, _ = _frame(((F(-1), ONE),) * 2)
    forms = (_point(ctx, ZERO, (ONE, ZERO)), _point(ctx, ZERO, (ZERO, ONE)),
             _point(ctx, ZERO, (ZERO, ZERO)))
    gates = _gates(frame, forms, ((ONE, ONE), (-ONE, -ONE), (ONE, ONE)))
    receiver = ms.make_value(frame, "original zero-point receiver", enabled=True)
    certificate = mb.generate(_point(ctx, ZERO, (ZERO, ZERO)), gates, receiver, enabled=True)
    legal = set()
    for bits in product(_bits(ZERO), repeat=3):
        phases = {gate[2]: bit for gate, bit in zip(gates, bits)}
        assert _rows_hold(certificate.rows, {receiver: ZERO}, phases)
        legal.add(bits)
    assert len(legal) == 8 and certificate.gates == gates
    assert all(gate[2].original_output is gate[1] for gate in certificate.gates)


def test_identity_limits_and_default_off():
    class Poison:
        def __iter__(self):
            raise AssertionError("disabled call inspected input")

        def __len__(self):
            raise AssertionError("disabled call inspected input")

        def __getattribute__(self, name):
            raise AssertionError("disabled call inspected input")

    poison = Poison()
    assert mb.generate(poison, poison, poison) is None
    with pytest.raises(mb.KernelError):
        mb.generate(poison, poison, poison, enabled=1)
    with pytest.raises(mb.KernelError):
        mb.generate(poison, poison, poison, enabled=None)

    frame, ctx, originals = _frame(((F(-1), ONE),))
    v = _point(ctx, ZERO, (ZERO,))
    forms = (_point(ctx, F(1, 10), (ONE,)), _point(ctx, F(-1, 5), (F(-1, 2),)))
    gates = _gates(frame, forms, ((ONE, ONE), (-ONE, -ONE)))
    receiver = ms.make_value(frame, "original receiver", enabled=True)
    assert mb.generate(v, gates, receiver, enabled=True).context is ctx
    with pytest.raises(mb.KernelError):
        mb.generate(v, list(gates), receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, tuple(reversed(gates)), receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, (gates[0], gates[0]), receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, ((forms[0], gates[0][1], gates[1][2], (ONE, ONE)),), receiver, enabled=True)
    for forbidden_receiver in (gates[0][1], originals[0]):
        with pytest.raises(mb.KernelError):
            mb.generate(v, gates, forbidden_receiver, enabled=True)
    other_ctx = ms.source_context(frame, ((originals[0], 0, F(-1), ONE),), enabled=True)
    other_form = _point(other_ctx, ZERO, (ONE,))
    with pytest.raises(mb.KernelError):
        mb.generate(v, ((other_form,) + gates[0][1:],), receiver, enabled=True)
    foreign_frame, foreign_ctx, _ = _frame(((F(-1), ONE),))
    foreign_receiver = ms.make_value(foreign_frame, "foreign receiver", enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, gates, foreign_receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(_point(foreign_ctx, ZERO, (ZERO,)), gates, receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, ((forms[0], gates[0][1], gates[0][2], (ONE, ZERO)),), receiver, enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(v, ((forms[0], gates[0][1], gates[0][2], (0.0, 1.0)),), receiver, enabled=True)

    cycle_frame = ms.make_frame("self-reference rejected", enabled=True)
    cycle_q = ms.make_value(cycle_frame, "original self q", enabled=True)
    cycle_phase = ms.make_phase(cycle_q, 0, "original self phase", enabled=True)
    cycle_ctx = ms.source_context(cycle_frame, ((cycle_q, 0, ZERO, ONE),), enabled=True)
    cycle_form = _point(cycle_ctx, ZERO, (ONE,))
    cycle_receiver = ms.make_value(cycle_frame, "other receiver", enabled=True)
    with pytest.raises(mb.KernelError):
        mb.generate(_point(cycle_ctx, ZERO, (ZERO,)),
                    ((cycle_form, cycle_q, cycle_phase, (ONE, ONE)),),
                    cycle_receiver, enabled=True)

    oversized = F(1 << mb.MAX_BITS)
    forged = ss._Affine(ctx, (oversized, oversized), ())
    with pytest.raises(mb.KernelError):
        mb.generate(forged, gates, receiver, enabled=True)
    # Inputs individually fit; the weighted nominal constant needs >512 bits.
    large = F(1 << (mb.MAX_BITS - 2))
    large_form = _point(ctx, large, (ZERO,))
    with pytest.raises(mb.KernelError):
        mb.generate(v, ((large_form, gates[0][1], gates[0][2], (F(4), F(4))),),
                    receiver, enabled=True)
    # Repeated occurrences must count BEFORE merging/deduplicating identities.
    # This creates references, not thousands of formal gates or source values.
    repeated = (gates[0],) * mb.MAX_SUPPORT
    with pytest.raises(mb.KernelError, match="aggregate"):
        mb.generate(v, repeated, receiver, enabled=True)
