"""Four fixed rational tests for the default-off D052 certificate component.

These are analytic fixtures, not attacks, parameter searches or a native-model
qualification.  Collection/import/execution is reserved for the root's single
post-freeze mathematical gate; this file was authored by static inspection.
"""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms
from experiments.neural_hz_20260831.definition_first_20260928.d053_static_source_component_20260930 import static_source as ss


ZERO, ONE = F(0), F(1)


def _fixture(bounds=((ZERO, ONE), (ZERO, ONE))):
    frame = ms.make_frame("original mathematical frontier", enabled=True)
    originals = tuple(ms.make_value(frame, "source " + str(i), enabled=True)
                      for i in range(len(bounds)))
    context = ms.source_context(frame, tuple((value, i, lo, hi)
                                for i, (value, (lo, hi)) in enumerate(zip(originals, bounds))),
                                enabled=True)
    q = ms.make_value(frame, "original q", enabled=True)
    p = ms.make_value(frame, "original p", enabled=True)
    alpha = ms.make_phase(q, 0, "original alpha", enabled=True)
    beta = ms.make_phase(p, 1, "original beta", enabled=True)
    return frame, context, originals, q, p, alpha, beta


def _point(context, bias, terms=()):
    return ss.affine(context, (bias, bias), tuple((source, value, value)
                     for source, value in terms), enabled=True)


def _generate(f, g, tokens):
    return ss.generate(f, g, *tokens, enabled=True)


def _check_form(value, bias, terms=()):
    assert value.bias == bias
    assert dict(value.source_terms) == dict(terms)
    assert value.phase_terms == ()


def _lhs(row, values, phases):
    value_terms, phase_terms, _ = row
    return (sum((a * values[token] for token, a in value_terms), ZERO)
            + sum((a * phases[token] for token, a in phase_terms), ZERO))


def _rows_hold(certificate, values, phases):
    return all(_lhs(row, values, phases) <= row[2] for row in certificate.rows)


def _bits(preactivation):
    return (ZERO,) if preactivation < ZERO else ((ONE,) if preactivation > ZERO else (ZERO, ONE))


def _upper_reference(original, minus, plus):
    # Independent direct four-product support on the original interval form.
    intervals = {s: (lo, hi) for s, lo, hi in original.terms}
    subtract, add = dict(minus.source_terms), dict(plus.source_terms)
    result = original.bias[1] - minus.bias + plus.bias
    for source in intervals.keys() | subtract.keys() | add.keys():
        lo, hi = intervals.get(source, (ZERO, ZERO))
        offset = add.get(source, ZERO) - subtract.get(source, ZERO)
        result += max((lo + offset) * source.lower, (lo + offset) * source.upper,
                      (hi + offset) * source.lower, (hi + offset) * source.upper)
    return result


def test_exact_matching_and_projection():
    _, ctx, originals, q, p, alpha, beta = _fixture()
    s, t = ctx.sources
    f = _point(ctx, F(-9, 10), ((s, ONE), (t, ONE)))
    cases = ((F(11, 10), F(-9, 10), F(1, 10), F(1, 5), F(1, 5)),
             (F(9, 10), F(-11, 10), F(1, 5), F(1, 10), F(1, 10)),
             (ONE, F(-1), F(1, 10), F(1, 10), F(1, 10)))
    for cs, ct, k0, k1, upper in cases:
        g = _point(ctx, F(1, 10), ((s, cs), (t, ct)))
        cert = _generate(f, g, (q, p, alpha, beta))
        assert cert.f is f and cert.g is g and cert.context is ctx
        assert cert.alpha is alpha and cert.beta is beta
        assert (cert.kappa0, cert.kappa1, cert.upper, cert.delta) == (k0, k1, upper, k1 - k0)
        _check_form(cert.P, ZERO, ((s, min(ONE, cs)),))
        _check_form(cert.N, ZERO, ((t, min(ONE, -ct)),))
        assert len(cert.rows) == (1 if k0 == k1 else 2)
        for value_terms, phase_terms, rhs in cert.rows:
            assert dict(value_terms) == {q: ONE, p: ONE, originals[0]: -min(ONE, cs)}
            assert set(dict(phase_terms)) <= {alpha, beta}
            assert type(rhs) is F
        if k1 < k0:
            assert cert.rows[1][2] == F(1, 10)
            assert dict(cert.rows[1][1]) == {alpha: F(-1, 10)}
        # Exact projection of the SINGLE-consumer linear lift, not a claim
        # about fractional nonlinear gates or their complete joint hull.
        for a, b in product((ZERO, F(1, 2), ONE), repeat=2):
            theta_lo, theta_hi = max(ZERO, a + b - ONE), min(a, b)
            assert theta_lo <= theta_hi
            theta = theta_hi if cert.delta >= ZERO else theta_lo
            boundary = k0 * a + upper * b + cert.delta * theta
            for offset in (F(-1, 100), ZERO, F(1, 100)):
                candidate = boundary + offset
                values = {originals[0]: ZERO, originals[1]: ZERO, q: candidate, p: ZERO}
                assert _rows_hold(cert, values, {alpha: a, beta: b}) == (offset <= ZERO)

    # Non-symmetric boxes, a reversed f direction, a g-only source, and a
    # fixed original source all contribute to the authenticated support.
    _, ctx, originals, q, p, alpha, beta = _fixture(((F(-2), ONE), (ONE, F(3)),
                                                    (F(2), F(2)), (F(-1), F(2))))
    x, y, fixed, extra = ctx.sources
    f = _point(ctx, F(1, 4), ((x, F(-2)), (y, F(1, 2)), (fixed, F(3, 4))))
    g = _point(ctx, F(-1, 3), ((x, F(-1)), (y, F(-1, 4)),
                              (fixed, F(1, 2)), (extra, F(2))))
    cert = _generate(f, g, (q, p, alpha, beta))
    assert any(source is fixed for source, _, _ in f.terms)
    _check_form(cert.P, ONE, ((x, F(-1)),))
    _check_form(cert.N, F(-1, 4), ((y, F(1, 4)),))
    assert (cert.kappa0, cert.kappa1, cert.upper, cert.delta) == (F(17, 4), F(27, 4), F(41, 12), F(5, 2))
    assert cert.rows[0][2] == ONE
    assert dict(cert.rows[0][0]) == {q: ONE, p: ONE, originals[0]: ONE}
    zero = ms.form(ctx, ZERO, (), (), enabled=True)
    assert cert.kappa0 == _upper_reference(f, cert.P, zero)
    assert cert.kappa1 == _upper_reference(f, cert.N, zero)
    assert cert.upper == _upper_reference(g, cert.P, cert.N)


def test_nonpoint_parameters_and_zero_phases():
    _, ctx, originals, q, p, alpha, beta = _fixture(((F(-1), ONE), (ZERO, F(2))))
    x, y = ctx.sources
    f = ss.affine(ctx, (F(-1, 10), F(1, 10)),
                  ((x, F(-1, 2), F(3, 2)), (y, F(-1, 4), F(1, 4))), enabled=True)
    g = ss.affine(ctx, (ZERO, F(1, 5)),
                  ((x, F(-3, 2), F(1, 2)), (y, F(1, 2), ONE)), enabled=True)
    cert = _generate(f, g, (q, p, alpha, beta))
    _check_form(cert.P, ZERO)
    _check_form(cert.N, F(1, 2), ((x, F(1, 2)),))
    assert (cert.kappa0, cert.kappa1, cert.upper, cert.delta) == (F(21, 10), F(11, 10), F(37, 10), F(-1))
    zero = ms.form(ctx, ZERO, (), (), enabled=True)
    assert cert.kappa0 == _upper_reference(f, cert.P, zero)
    assert cert.kappa1 == _upper_reference(f, cert.N, zero)
    assert cert.upper == _upper_reference(g, cert.P, cert.N)
    # Fixed endpoint and interior parameter fixtures; midpoint directions do
    # not replace these real coefficient choices, including sign reversals.
    parameters = ((F(-1, 10), F(-1, 2), F(-1, 4), ZERO, F(-3, 2), F(1, 2)),
                  (F(1, 10), F(3, 2), F(1, 4), F(1, 5), F(1, 2), ONE),
                  (F(1, 10), F(-1, 2), F(1, 4), ZERO, F(1, 2), F(1, 2)),
                  (ZERO, F(1, 2), ZERO, ZERO, F(-1, 2), F(3, 4)))
    sources = ((F(-1), ZERO), (F(-1), F(2)), (ONE, ZERO), (ONE, F(2)), (ZERO, ZERO))
    zero_choices = set()
    for fc, fx, fy, gc, gx, gy in parameters:
        for xv, yv in sources:
            fv, gv = fc + fx * xv + fy * yv, gc + gx * xv + gy * yv
            qv, pv = max(ZERO, fv), max(ZERO, gv)
            assignment = {x: xv, y: yv}
            P = ms.evaluate(cert.P, assignment, enabled=True)
            N = ms.evaluate(cert.N, assignment, enabled=True)
            assert P >= ZERO and N >= ZERO
            assert fv - P <= cert.kappa0 and fv - N <= cert.kappa1
            assert gv - P + N <= cert.upper
            values = {originals[0]: xv, originals[1]: yv, q: qv, p: pv}
            for a, b in product(_bits(fv), _bits(gv)):
                assert _rows_hold(cert, values, {alpha: a, beta: b})
                if fv == gv == ZERO:
                    zero_choices.add((a, b))
    assert zero_choices == {(ZERO, ZERO), (ZERO, ONE), (ONE, ZERO), (ONE, ONE)}


def test_strict_control_and_forward_use():
    _, ctx, originals, q, p, alpha, beta = _fixture()
    s, t = ctx.sources
    f = _point(ctx, F(-9, 10), ((s, ONE), (t, ONE)))
    g = _point(ctx, F(1, 10), ((s, F(11, 10)), (t, F(-9, 10))))
    cert = _generate(f, g, (q, p, alpha, beta))
    assert (cert.kappa0, cert.kappa1, cert.upper) == (F(1, 10), F(1, 5), F(1, 5))
    old_s = old_t = old_alpha = old_beta = F(1, 2)
    old_q, old_p = F(2, 5), F(81, 200)
    # Two different, complete source-labelled single-gate hull witnesses.
    qa, qi = (F(17, 20), F(17, 20)), (F(3, 20), F(3, 20))
    pa, pi = (F(17, 20), F(1, 4)), (F(3, 20), F(3, 4))
    for active, inactive in ((qa, qi), (pa, pi)):
        assert all(ZERO < v < ONE for v in active + inactive)
        assert tuple((a + b) / 2 for a, b in zip(active, inactive)) == (old_s, old_t)
    assert sum(qa, ZERO) - F(9, 10) == F(4, 5) > ZERO
    assert sum(qi, ZERO) - F(9, 10) == F(-3, 5) < ZERO
    assert old_q == F(1, 2) * F(4, 5)
    assert F(11, 10) * pa[0] - F(9, 10) * pa[1] + F(1, 10) == F(81, 100) > ZERO
    assert F(11, 10) * pi[0] - F(9, 10) * pi[1] + F(1, 10) == F(-41, 100) < ZERO
    assert old_p == F(1, 2) * F(81, 100)
    old_f, old_g = F(1, 10), F(1, 5)
    difference, delta = old_q - old_p, old_f - old_g
    correction = difference - delta
    assert (difference, delta, correction) == (F(-1, 200), F(-1, 10), F(19, 200))
    lo, hi = F(-11, 10), F(9, 10)
    assert lo * old_beta <= difference <= hi * old_alpha
    assert -hi * (ONE - old_beta) <= correction <= -lo * (ONE - old_alpha)
    # The fixed D049 source ordering and its reverse both admit this point.
    assert -F(11, 10) * (ONE - old_t) <= difference <= F(9, 10) * old_t
    assert old_f <= old_q <= old_s / 10 + old_t
    assert old_g <= old_p <= F(3, 10) * old_s + F(9, 10) * (ONE - old_t)
    assert old_q <= old_s + old_t / 10
    assert old_p <= F(11, 10) * old_s + (ONE - old_t) / 10
    values = {originals[0]: old_s, originals[1]: old_t, q: old_q, p: old_p}
    assert not _rows_hold(cert, values, {alpha: old_alpha, beta: old_beta})
    residual = old_q + old_p - old_s
    old_Z = residual + F(2, 9) * (old_q - old_f) + F(1, 4) * (old_p - old_g)
    assert old_Z == F(203, 480) and old_Z - F(2, 5) == F(11, 480)
    assert old_q - old_f <= F(9, 10) * (ONE - old_alpha)
    assert old_p - old_g <= F(4, 5) * (ONE - old_beta)

    # Reuse the GENERATED first row, not an independent interval for F.
    # p<=6*beta/5 and h=-F/2+p/40+1/100 yield F+h<=psi+1/100,
    # psi=(alpha+beta)/5. The shared psi eliminates all new gamma*oldbit.
    old_h = -residual / 2 + old_p / 40 + F(1, 100)
    assert old_h == F(-1059, 8000) < ZERO
    assert old_Z + max(ZERO, old_h) > F(41, 100)
    assert old_Z - F(41, 100) == F(31, 2400)
    fixtures = ((ZERO, ZERO), (ONE, ONE), (ONE, ZERO), (ZERO, ONE),
                (F(3, 10), F(2, 5)), (F(71, 200), F(109, 200)))
    zero_choices, child_signs = set(), set()
    for sv, tv in fixtures:
        fv, gv = sv + tv - F(9, 10), F(11, 10) * sv - F(9, 10) * tv + F(1, 10)
        qv, pv = max(ZERO, fv), max(ZERO, gv)
        Fv = qv + pv - sv
        hv = -Fv / 2 + pv / 40 + F(1, 100)
        rv = max(ZERO, hv)
        child_signs.add(hv > ZERO)
        value_map = {originals[0]: sv, originals[1]: tv, q: qv, p: pv}
        for a, b, gamma in product(_bits(fv), _bits(gv), _bits(hv)):
            assert _rows_hold(cert, value_map, {alpha: a, beta: b})
            psi = (a + b) / 5
            assert Fv <= psi and pv <= F(6, 5) * b
            assert Fv + hv <= psi + F(1, 100)
            assert Fv + rv <= psi + gamma / 100
            assert qv - fv <= F(9, 10) * (ONE - a)
            assert pv - gv <= F(4, 5) * (ONE - b)
            physical = Fv + F(2, 9) * (qv - fv) + (pv - gv) / 4
            assert physical <= F(2, 5)
            assert physical + rv <= F(2, 5) + gamma / 100 <= F(41, 100)
            if fv == gv == ZERO:
                zero_choices.add((a, b))
        if (sv, tv) == (ONE, ONE):
            assert (qv, pv, physical) == (F(11, 10), F(3, 10), F(2, 5))
            assert hv == F(-73, 400)
        if (sv, tv) == (F(3, 10), F(2, 5)):
            assert fv != ZERO and gv != ZERO and hv == F(507, 4000)
    assert child_signs == {False, True}
    assert zero_choices == {(ZERO, ZERO), (ZERO, ONE), (ONE, ZERO), (ONE, ONE)}


def test_disabled_identity_and_rejection():
    assert ss.affine(None, None, None) is None
    assert ss.generate(None, None, None, None, None, None) is None
    with pytest.raises(ss.KernelError):
        ss.affine(None, None, None, enabled=1)
    _, ctx, originals, q, p, alpha, beta = _fixture()
    s, t = ctx.sources
    f = _point(ctx, ZERO, ((s, ONE),))
    g = _point(ctx, ZERO, ((t, ONE),))
    cert = _generate(f, g, (q, p, alpha, beta))
    with pytest.raises(AttributeError):
        cert.delta = ONE
    for bad in ((ONE, ZERO), (0, ONE), (0.0, ONE), (F(1 << 512), F(1 << 512)),
                [ZERO, ONE], (ZERO,)):
        with pytest.raises(ss.KernelError):
            ss.affine(ctx, bad, (), enabled=True)
    for bad_terms in ([(s, ONE, ONE)], ((s, ONE),), ((s, ONE, ZERO),),
                      ((s, 1, ONE),), ((alpha, ONE, ONE),),
                      ((s, ZERO, ZERO), (s, ONE, ONE)),
                      ((s, ONE, ONE),) * 65_537):
        with pytest.raises(ss.KernelError):
            ss.affine(ctx, (ZERO, ZERO), bad_terms, enabled=True)
    other = ms.source_context(ctx.frame, ((originals[0], 0, ZERO, ONE),), enabled=True)
    with pytest.raises(ss.KernelError):
        _point(ctx, ZERO, ((other.sources[0], ONE),))
    other_f = _point(other, ZERO, ((other.sources[0], ONE),))
    for arguments in ((f, other_f, q, p, alpha, beta),
                      (f, g, q, q, alpha, alpha),
                      (f, g, q, p, beta, alpha),
                      (None, g, q, p, alpha, beta)):
        with pytest.raises(ss.KernelError):
            ss.generate(*arguments, enabled=True)
    # Same labels and interval values do not authorize any identity merge.
    canonical = ss.affine(ctx, (ZERO, ZERO), ((t, F(2), F(2)), (s, ONE, ONE)), enabled=True)
    assert canonical.terms == ((s, ONE, ONE), (t, F(2), F(2)))
    for values, phases, _ in cert.rows:
        assert set(dict(values)) <= {q, p, *originals}
        assert set(dict(phases)) <= {alpha, beta}
        assert all(token is not s and token is not t for token, _ in values)
    # A source token wrapping the target output is not accepted as a new,
    # independent root or a circular local declaration.
    circular = ms.source_context(ctx.frame, ((q, 7, ZERO, ONE),), enabled=True)
    circular_f = _point(circular, ZERO, ((circular.sources[0], ONE),))
    circular_g = _point(circular, ZERO)
    with pytest.raises(ss.KernelError):
        _generate(circular_f, circular_g, (q, p, alpha, beta))
    # Input endpoints fit, but midpoint/intermediate arithmetic cannot
    # silently exceed the component's rational-bit guard.
    huge = _point(ctx, ZERO, ((s, F(1 << 511)),))
    with pytest.raises(ss.KernelError):
        _generate(huge, g, (q, p, alpha, beta))
