"""Four frozen-entry mathematical tests; no model, search or verifier run."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import permutations, product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d056_phase_triangle_component_20260930 import phase_triangle as pt

ss, ms, mp = pt.ss, pt.ms, pt.mp
ZERO, ONE, HALF = F(0), F(1), F(1, 2)


def _fixture(biases=(F(-3, 10), F(-9, 10), F(-3, 10)),
             coefficients=((ONE, F(-1, 3)), (ONE, ONE), (F(-1, 3), ONE)),
             bounds=((ZERO, ONE), (ZERO, ONE))):
    frame = ms.make_frame('shared original mathematical source', enabled=True)
    x = tuple(ms.make_value(frame, 'source ' + str(i), enabled=True) for i in range(len(bounds)))
    ctx = ms.source_context(frame, tuple((v, i, lo, hi) for i, (v, (lo, hi))
                                        in enumerate(zip(x, bounds))), enabled=True)
    q = tuple(ms.make_value(frame, 'original gate ' + str(i), enabled=True) for i in range(3))
    b = tuple(ms.make_phase(v, i, 'original phase ' + str(i), enabled=True) for i, v in enumerate(q))
    forms = tuple(ss.affine(ctx, (bias, bias), tuple((s, a, a) for s, a in zip(ctx.sources, row)),
                            enabled=True) for bias, row in zip(biases, coefficients))
    return forms, q, b, x


def _left(row, values, bits):
    return sum((a * values[t] for t, a in row[0]), ZERO) + sum((a * bits[t] for t, a in row[1]), ZERO)


def _pre(form, values):
    assert form.bias[0] == form.bias[1]
    assert all(lo == hi for _, lo, hi in form.terms)
    return form.bias[0] + sum((lo * values[s.original_value] for s, lo, _ in form.terms), ZERO)


def _hold(rows, values, bits):
    return all(_left(row, values, bits) <= row[2] for row in rows)


def _physical(forms, q, x, values):
    residual = tuple(values[v] - _pre(f, values) for f, v in zip(forms, q))
    return (2 * sum((values[v] for v in q), ZERO) - sum((values[v] for v in x), ZERO)
            + F(12, 19) * (residual[0] + residual[2]) + F(26, 27) * residual[1])


def test_d054_physical_control():
    forms, q, b, x = _fixture()
    before = (len(b[0].frame._values), len(b[0].frame._phases))
    result = pt.generate(forms, q, b, enabled=True)
    assert result.status == 'odd_cycle' and len(result.rows) == 1
    assert tuple(pair.delta for pair in result.pairs) == (F(2, 3), F(-2, 3), F(2, 3))
    row = result.rows[0]
    assert dict(row[0]) == {q[0]: F(2), q[1]: F(2), q[2]: F(2), x[0]: -ONE, x[1]: -ONE}
    assert dict(row[1]) == {b[0]: F(-2, 5), b[1]: F(-13, 15), b[2]: F(-2, 5)}
    assert row[2] == ZERO and len(row[0]) + len(row[1]) == 8
    bits = dict.fromkeys(b, HALF)
    values = dict(zip(x + q, (HALF, HALF, F(1, 3), F(2, 5), F(1, 3))))
    for i, j in permutations(range(3), 2):
        pair = ss.generate(forms[i], forms[j], q[i], q[j], b[i], b[j], enabled=True)
        assert _hold(pair.rows, values, bits)
    assert _left(row, values, bits) - row[2] == F(3, 10)
    assert _physical(forms, q, x, values) == F(308, 171)
    # Explicit full single-gate source-labelled hull witnesses, not scalar M.
    active = ((F(59, 60), F(1, 20)), (F(17, 20), F(17, 20)), (F(1, 20), F(59, 60)))
    for i, coordinates in enumerate(active):
        av = dict(zip(x, coordinates))
        iv = dict(zip(x, tuple(ONE - c for c in coordinates)))
        assert _pre(forms[i], av) > ZERO > _pre(forms[i], iv)
        assert _pre(forms[i], av) / 2 == values[q[i]]
    strict = dict(values)
    for v in q:
        strict[v] -= F(1, 300)
    assert _physical(forms, q, x, strict) - F(5, 3) == F(824, 7695)
    actual = dict.fromkeys(x, ONE)
    actual.update((v, max(ZERO, _pre(f, actual))) for f, v in zip(forms, q))
    assert _hold(result.rows, actual, dict.fromkeys(b, ONE))
    assert _physical(forms, q, x, actual) == F(5, 3)
    for order in permutations(range(3)):
        reordered = pt.generate(tuple(forms[i] for i in order), tuple(q[i] for i in order),
                                tuple(b[i] for i in order), enabled=True)
        assert reordered == result
    assert (len(b[0].frame._values), len(b[0].frame._phases)) == before


def test_signed_triangle_projection():
    forms, q, b, x = _fixture()
    # Independent formula on fixed rational means, all sign orientations and
    # both equal/unequal magnitudes. This finite proof fixture is not runtime
    # phase enumeration or a new model experiment.
    for magnitudes in ((F(1, 3), F(2, 5), F(3, 7)), (F(2, 3),) * 3):
        for signs in product((-1, 1), repeat=3):
            d = tuple(s * a for s, a in zip(signs, magnitudes))
            planes = pt._planes(b, d)
            if sum(s < 0 for s in signs) % 2 == 0:
                assert planes == ()
                continue
            assert 1 <= len(planes) <= 4
            tau = min(magnitudes)
            for mean in product((ZERO, HALF, ONE), repeat=3):
                assignment = dict(zip(b, mean))
                upper = min(p.bias + sum((a * assignment[t] for t, a in p.terms), ZERO) for p in planes)
                if signs.count(-1) == 3:
                    reference = tau * (ONE - sum(mean, ZERO))
                else:
                    edge = pt.EDGES[signs.index(-1)]
                    reference = tau * mean[next(i for i in range(3) if i not in edge)]
                for (i, j), amount in zip(pt.EDGES, d):
                    remainder = abs(amount) - tau
                    reference += (remainder * min(mean[i], mean[j]) if amount > ZERO
                                  else -remainder * max(ZERO, mean[i] + mean[j] - ONE))
                assert upper == reference
                if all(m in (ZERO, ONE) for m in mean):
                    quadratic = sum((a * mean[i] * mean[j] for (i, j), a in zip(pt.EDGES, d)), ZERO)
                    assert quadratic <= upper
    assert pt._planes(b, (ONE, ZERO, -ONE)) == ()
    # Ordinary three-negative cycle with nonzero biases and mixed weights.
    forms, q, b, x = _fixture((F(1, 10),) * 3,
        ((ONE, F(-1, 3), ZERO), (ZERO, ONE, F(-1, 3)), (F(-1, 3), ZERO, ONE)),
        ((-ONE, ONE),) * 3)
    result = pt.generate(forms, q, b, enabled=True)
    assert result.status == 'odd_cycle' and len(result.rows) == 1
    assert tuple(p.delta for p in result.pairs) == (F(-2, 3),) * 3
    row = result.rows[0]
    assert dict(row[0]) == dict.fromkeys(q, F(2))
    assert dict(row[1]) == dict.fromkeys(b, F(-11, 5)) and row[2] == F(2, 3)
    values = dict.fromkeys(x, ZERO)
    values.update(dict.fromkeys(q, F(7, 10)))
    bits = dict.fromkeys(b, HALF)
    assert all(_hold(p.rows, values, bits) for p in result.pairs)
    assert _left(row, values, bits) - row[2] == F(7, 30)


def test_nonpoint_and_zero_semantics():
    forms, q, b, x = _fixture()
    # Nonpoint bias enclosure is retained in the support, not midpoint truth.
    width = F(1, 100)
    interval_forms = tuple(ss.affine(f.context, (f.bias[0] - width, f.bias[1] + width),
                                    f.terms, enabled=True) for f in forms)
    result = pt.generate(interval_forms, q, b, enabled=True)
    assert result.status == 'odd_cycle'
    for offsets in product((-width, ZERO, width), repeat=3):
        for coordinates in product((ZERO, HALF, ONE), repeat=2):
            values = dict(zip(x, coordinates))
            pre = tuple(_pre(f, values) + offset for f, offset in zip(forms, offsets))
            values.update((v, max(ZERO, f)) for v, f in zip(q, pre))
            legal = tuple((ZERO, ONE) if f == ZERO else ((ONE,) if f > ZERO else (ZERO,)) for f in pre)
            for choice in product(*legal):
                assert _hold(result.rows, values, dict(zip(b, choice)))
    # All-zero preactivations have ALL eight legal phase choices; no stable
    # filter may choose a sign, even though every observed output is zero.
    forms, q, b, x = _fixture((ZERO,) * 3,
        ((ONE, -ONE, ZERO), (ZERO, ONE, -ONE), (-ONE, ZERO, ONE)), ((ZERO, ONE),) * 3)
    result = pt.generate(forms, q, b, enabled=True)
    assert result.status == 'odd_cycle'
    values = dict.fromkeys(x + q, ZERO)
    for choice in product((ZERO, ONE), repeat=3):
        assert _hold(result.rows, values, dict(zip(b, choice)))
    positive = ss.affine(forms[0].context, (F(3), F(3)), forms[0].terms, enabled=True)
    stable = pt.generate((positive, *forms[1:]), q, b, enabled=True)
    assert stable.status == 'strict_stable' and stable.rows == () and stable.pairs == ()
    assert len(b[0].frame._phases) == 3
    touch = ss.affine(forms[0].context, (ONE, ONE), forms[0].terms, enabled=True)
    touching = pt.generate((touch, *forms[1:]), q, b, enabled=True)
    assert touching.bounds[0][0] == ZERO and touching.status != 'strict_stable'


def test_identity_limits_and_default_off():
    assert pt.generate(object(), object(), object()) is None
    forms, q, b, x = _fixture()
    with pytest.raises(pt.KernelError):
        pt.generate(forms, q, b, enabled=1)
    with pytest.raises(pt.KernelError):
        pt.generate(list(forms), q, b, enabled=True)
    with pytest.raises(pt.KernelError):
        pt.generate(forms, (q[0], q[0], q[2]), b, enabled=True)
    with pytest.raises(pt.KernelError):
        pt.generate(forms, q, (b[1], b[0], b[2]), enabled=True)
    other, oq, ob, ox = _fixture()
    with pytest.raises(pt.KernelError):
        pt.generate((forms[0], forms[1], other[2]), q, b, enabled=True)
    with pytest.raises(pt.KernelError):
        pt.generate(forms, q, (b[0], b[1], ob[2]), enabled=True)
    bad = replace(forms[0], bias=(F(1 << 512), F(1 << 512)))
    with pytest.raises(pt.KernelError):
        pt.generate((bad, *forms[1:]), q, b, enabled=True)
    oversized = replace(forms[0], terms=(forms[0].terms[0],) * 32768)
    with pytest.raises(pt.KernelError, match='aggregate triangle input'):
        pt.generate((oversized, *forms[1:]), q, b, enabled=True)
    circular_ctx = ms.source_context(b[0].frame, ((q[2], 0, ZERO, ONE),), enabled=True)
    circular = tuple(ss.affine(circular_ctx, (-HALF, -HALF),
                    ((circular_ctx.sources[0], ONE, ONE),), enabled=True) for _ in range(3))
    with pytest.raises(pt.KernelError, match='frontier'):
        pt.generate(circular, q, b, enabled=True)
