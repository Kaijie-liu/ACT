"""Four fixed exact-rational mathematical tests; never run before freeze.

Finite fixture grids check proved operators, not runtime phase/input search.
No token establishes native model, phase-column, or decoder binding.
"""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d046_multiphase_envelopes_20260930 import multiphase as mp
from experiments.neural_hz_20260831.definition_first_20260928.d049_mixed_source_envelopes_20260930 import mixed_source as ms


ZERO, ONE = F(0), F(1)


def _fixture(lower=ZERO, upper=ONE):
    frame = ms.make_frame("shared mathematical frame", enabled=True)
    xv = ms.make_value(frame, "original continuous x", enabled=True)
    qv = ms.make_value(frame, "original gate q", enabled=True)
    pv = ms.make_value(frame, "original gate p", enabled=True)
    context = ms.source_context(frame, ((xv, 0, lower, upper),), enabled=True)
    alpha = ms.make_phase(qv, 0, "original alpha", enabled=True)
    eta = ms.make_phase(pv, 1, "original eta", enabled=True)
    return frame, context, context.sources[0], xv, qv, pv, alpha, eta


def _form(context, bias=ZERO, sources=(), phases=()):
    return ms.form(context, bias, sources, phases, enabled=True)


def _check_form(actual, bias, sources=(), phases=()):
    assert actual.bias == bias
    assert dict(actual.source_terms) == dict(sources)
    assert dict(actual.phase_terms) == dict(phases)


def _evaluate(form, assignment):
    return ms.evaluate(form, assignment, enabled=True)


def _legal_bits(preactivation):
    if preactivation < 0:
        return (ZERO,)
    if preactivation > 0:
        return (ONE,)
    return ZERO, ONE


def _assert_rows(observation, values, phases):
    rows = ms.compile_rows(observation, enabled=True)
    assert type(rows) is tuple and len(rows) == 2
    for value_terms, phase_terms, rhs in rows:
        lhs = sum((coefficient * values[token] for token, coefficient in value_terms), ZERO)
        lhs += sum((coefficient * phases[token] for token, coefficient in phase_terms), ZERO)
        assert lhs <= rhs


def test_mixed_majorant_minorant():
    _, ctx, x, _, _, _, alpha, eta = _fixture()
    first = _form(ctx, F(-1, 2), ((x, ONE),), ((alpha, ONE), (eta, F(2))))
    upper = ms.positive_majorant(first, enabled=True)
    lower = ms.positive_minorant(first, enabled=True)
    _check_form(upper, ZERO, ((x, F(1, 2)),), ((alpha, ONE), (eta, F(2))))
    _check_form(lower, ZERO, (), ((alpha, F(1, 2)), (eta, F(3, 2))))
    # This case selects the continuous baseline theta=1 and retains x.
    second = _form(ctx, F(-1, 4), ((x, ONE),), ((alpha, ONE),))
    upper2 = ms.positive_majorant(second, enabled=True)
    lower2 = ms.positive_minorant(second, enabled=True)
    _check_form(upper2, ZERO, ((x, F(3, 4)),), ((alpha, ONE),))
    _check_form(lower2, F(-1, 4), ((x, ONE),), ((alpha, F(3, 4)),))
    # Negative continuous coefficients require both scale and bias recovery.
    third = _form(ctx, F(3, 4), ((x, F(-1)),), ((alpha, ONE),))
    upper3 = ms.positive_majorant(third, enabled=True)
    lower3 = ms.positive_minorant(third, enabled=True)
    _check_form(upper3, F(3, 4), ((x, F(-3, 4)),), ((alpha, ONE),))
    _check_form(lower3, F(3, 4), ((x, F(-1)),), ((alpha, F(3, 4)),))
    triples = ((first, lower, upper), (second, lower2, upper2), (third, lower3, upper3))
    for value, a, b in product((ZERO, F(1, 4), F(1, 2), F(3, 4), ONE), (ZERO, ONE), (ZERO, ONE)):
        assignment = {x: value, alpha: a, eta: b}
        for original, lo, hi in triples:
            relu = max(ZERO, _evaluate(original, assignment))
            assert _evaluate(lo, assignment) <= relu <= _evaluate(hi, assignment)
    # Upper is also sound on fractional bits; singleton lower is NOT promised
    # to be pointwise below the hinge there.
    for value, a, b in product((ZERO, F(1, 2), ONE), repeat=3):
        assignment = {x: value, alpha: a, eta: b}
        assert max(ZERO, _evaluate(first, assignment)) <= _evaluate(upper, assignment)
    fractional = {x: ZERO, alpha: F(1, 2), eta: ZERO}
    assert _evaluate(lower, fractional) == F(1, 4)
    assert max(ZERO, _evaluate(first, fractional)) == ZERO


def test_phase_only_and_typed_identity():
    frame, ctx, x, xv, _, _, alpha, eta = _fixture(F(-1), ONE)
    for bias, coefficients in ((F(-1, 2), (ONE, F(2))),
                               (F(1, 4), (F(-1), F(2))),
                               (F(-2), (F(-1, 3), F(1, 7))),
                               (F(2), (ZERO, ZERO))):
        phase_terms = ((alpha, coefficients[0]), (eta, coefficients[1]))
        mixed = _form(ctx, bias, (), phase_terms)
        original = mp.form(frame, "phase", bias, phase_terms, enabled=True)
        for mixed_op, original_op in ((ms.positive_majorant, mp.positive_majorant),
                                      (ms.positive_minorant, mp.positive_minorant)):
            actual, expected = mixed_op(mixed, enabled=True), original_op(original, enabled=True)
            _check_form(actual, expected.bias, (), expected.terms)
    # Source ordinal 0 and original phase ordinal 0 remain distinct types.
    typed = _form(ctx, ZERO, ((x, ONE),), ((alpha, ONE),))
    assert len(typed.source_terms) == len(typed.phase_terms) == 1
    assert _evaluate(typed, {x: F(1, 4), alpha: F(1, 2)}) == F(3, 4)
    duplicate_context = ms.source_context(frame, ((xv, 0, F(-1), ONE),), enabled=True)
    other = _form(duplicate_context, ZERO, ((duplicate_context.sources[0], ONE),))
    with pytest.raises(ms.KernelError):
        ms.linear(ctx, ((ONE, other),), enabled=True)
    # Fixed original sources fold exactly but remain in the context.
    fixed_value = ms.make_value(frame, "fixed existing source", enabled=True)
    fixed_ctx = ms.source_context(frame, ((fixed_value, 8, F(2), F(2)),), enabled=True)
    fixed = _form(fixed_ctx, F(1, 2), ((fixed_ctx.sources[0], F(3)),))
    _check_form(fixed, F(13, 2))
    assert len(fixed_ctx.sources) == 1
    # Equal diagnostic labels and bounds never establish source identity.
    av = ms.make_value(frame, "same label", enabled=True)
    bv = ms.make_value(frame, "same label", enabled=True)
    separate = ms.source_context(frame, ((av, 3, ZERO, ONE), (bv, 4, ZERO, ONE)), enabled=True)
    distinction = _form(separate, ZERO, ((separate.sources[0], ONE), (separate.sources[1], F(-1))))
    assert len(distinction.source_terms) == 2


def test_source_relational_control_and_forward_composition():
    frame = ms.make_frame("nonzero-bias common-source control", enabled=True)
    xv, yv = (ms.make_value(frame, name, enabled=True) for name in ("original x", "original y"))
    ctx = ms.source_context(frame, ((xv, 0, F(-1), ONE), (yv, 1, F(-1), ONE)), enabled=True)
    x, y = ctx.sources
    fv, qv, pv = (ms.make_value(frame, name, enabled=True) for name in ("original f", "original q", "original p"))
    alpha = ms.make_phase(qv, 0, "original alpha", enabled=True)
    eta = ms.make_phase(pv, 1, "original eta", enabled=True)
    delta = _form(ctx, F(-1, 200), ((x, F(1, 4)), (y, F(-6, 25))))
    f = _form(ctx, F(1, 200), ((x, F(3, 4)), (y, F(6, 25))))
    delta_readout = ms.readout(frame, ZERO, ((xv, ONE), (fv, F(-1))), enabled=True)
    f_readout = ms.readout(frame, ZERO, ((fv, ONE),), enabled=True)
    delta_observation = ms.observe(delta_readout, delta, delta, enabled=True)
    f_observation = ms.observe(f_readout, f, f, enabled=True)
    difference, companion = ms.relu_pair(delta_observation, f_observation, qv, pv, enabled=True)
    _check_form(difference.upper, F(97, 400), ((x, F(1, 400)), (y, F(-6, 25))))

    old_x, old_y, old_q, old_a = F(1, 4), ZERO, F(9, 20), F(11, 20)
    old_p, old_e = F(77, 400), ONE
    # Complete q source-labelled hull: two true graph points with the same y.
    active, inactive = F(9, 11), F(-4, 9)
    assert active > 0 > inactive
    assert old_a * active + (1 - old_a) * inactive == old_x
    assert old_a * active == old_q
    assert old_a * ZERO + (1 - old_a) * ZERO == old_y
    # p's full source-labelled hull contains this actual single graph point.
    assert old_p == F(3, 4) * old_x + F(6, 25) * old_y + F(1, 200) > 0
    assert old_e == ONE
    d = old_q - old_p
    pre_difference = old_x - old_p
    correction = d - pre_difference
    lo, hi = F(-99, 200), F(97, 200)
    assert (d, pre_difference, correction) == (F(103, 400), F(23, 400), F(1, 5))
    assert lo * old_e <= d <= hi * old_a
    assert -hi * (1 - old_e) <= correction <= -lo * (1 - old_a)
    source_assignment = {x: old_x, y: old_y}
    assert _evaluate(difference.upper, source_assignment) == F(389, 1600)
    assert d - _evaluate(difference.upper, source_assignment) == F(23, 1600)
    assert d - old_x / 400 + F(6, 25) * old_y == F(411, 1600) > F(97, 400)

    h2 = ms.affine(ctx, ((F(11, 10), difference), (F(1, 2), companion)), bias=F(1, 10), enabled=True)
    w2 = ms.affine(ctx, ((F(-1, 2), difference), (F(1, 2), companion)), bias=F(1, 10), enabled=True)
    assert dict(h2.readout.terms) == {qv: F(11, 10), pv: F(-3, 5)}
    assert dict(w2.readout.terms) == {qv: F(-1, 2), pv: ONE}
    delta2 = ms.affine(ctx, ((F(8, 5), difference),), enabled=True)
    r2v, t2v = (ms.make_value(frame, name, enabled=True) for name in ("original r2", "original t2"))
    beta = ms.make_phase(r2v, 2, "original beta", enabled=True)
    tau = ms.make_phase(t2v, 3, "original tau", enabled=True)
    difference2, companion2 = ms.relu_pair(delta2, w2, r2v, t2v, enabled=True)
    _check_form(difference2.upper, F(97, 250), ((x, F(1, 250)), (y, F(-48, 125))))
    old_r2 = F(1, 10) + F(11, 10) * old_q - F(3, 5) * old_p
    old_t2 = F(1, 10) - old_q / 2 + old_p
    assert (old_r2, old_t2) == (F(959, 2000), F(27, 400))
    assert old_r2 > 0 and old_t2 > 0  # The old point extends to exact child graphs.
    assert old_r2 - old_t2 - old_x / 250 == F(411, 1000) > F(97, 250)

    fixtures = ((ONE, F(-1)), (F(-1), ONE), (ZERO, ZERO),
                (ZERO, F(-1, 48)), (F(-1, 10), ONE), (F(3, 10), F(-1)))
    for x_value, y_value in fixtures:
        f_value = F(3, 4) * x_value + F(6, 25) * y_value + F(1, 200)
        q_value, p_value = max(ZERO, x_value), max(ZERO, f_value)
        h_value = F(1, 10) + F(11, 10) * q_value - F(3, 5) * p_value
        w_value = F(1, 10) - q_value / 2 + p_value
        r_value, t_value = max(ZERO, h_value), max(ZERO, w_value)
        sources = {x: x_value, y: y_value}
        values = {xv: x_value, yv: y_value, fv: f_value, qv: q_value,
                  pv: p_value, r2v: r_value, t2v: t_value}
        for a, e, b, t in product(_legal_bits(x_value), _legal_bits(f_value),
                                   _legal_bits(h_value), _legal_bits(w_value)):
            phases = {alpha: a, eta: e, beta: b, tau: t}
            for observation, actual in ((difference, q_value - p_value), (companion, p_value),
                                         (difference2, r_value - t_value), (companion2, t_value)):
                assert _evaluate(observation.lower, sources) <= actual <= _evaluate(observation.upper, sources)
                _assert_rows(observation, values, phases)
        if (x_value, y_value) == (ONE, F(-1)):
            assert q_value - p_value - x_value / 400 + F(6, 25) * y_value == F(97, 400)
            assert r_value - t_value - x_value / 250 + F(48, 125) * y_value == F(97, 250)
        if (x_value, y_value) == (ZERO, F(-1, 48)):
            assert x_value == f_value == q_value == p_value == ZERO
            assert len(_legal_bits(x_value)) * len(_legal_bits(f_value)) == 4
        if (x_value, y_value) == (F(-1, 10), ONE):
            assert h_value == F(-1, 500)
        if (x_value, y_value) == (F(3, 10), F(-1)):
            assert w_value == F(-1, 20)


def test_disabled_and_rejected_premises_and_rows():
    assert ms.source_context(None, None) is None
    assert ms.form(None, None, None, None) is None
    assert ms.positive_majorant(None) is None
    assert ms.positive_minorant(None) is None
    assert ms.observe(None, None, None) is None
    assert ms.affine(None, None) is None
    assert ms.relu_pair(None, None, None, None) is None
    assert ms.compile_rows(None) is None
    assert ms.evaluate(None, None) is None
    frame, ctx, x, xv, qv, _, alpha, eta = _fixture()
    with pytest.raises(ms.KernelError):
        ms.positive_majorant(None, enabled=1)
    for bias in (0.5, 1, F(1 << 512)):
        with pytest.raises(ms.KernelError):
            _form(ctx, bias)
    with pytest.raises(ms.KernelError):
        ms.source_context(frame, ((xv, 0, ONE, ZERO),), enabled=True)
    with pytest.raises(ms.KernelError):
        ms.source_context(frame, ((xv, 0, ZERO, ONE), (qv, 0, ZERO, ONE)), enabled=True)
    with pytest.raises(ms.KernelError):
        _form(ctx, ZERO, ((alpha, ONE),))
    with pytest.raises(ms.KernelError):
        _form(ctx, ZERO, (), ((x, ONE),))
    with pytest.raises(ms.KernelError):
        ms.form(ctx, ZERO, [(x, ONE)], (), enabled=True)
    with pytest.raises(ms.KernelError):
        _form(ctx, ZERO, ((x, ZERO),) * 65_537)
    huge = F(1 << 511)
    overflow = _form(ctx, ZERO, ((x, huge),), ((alpha, huge),))
    with pytest.raises(ms.KernelError):
        ms.positive_majorant(overflow, enabled=True)
    value = _form(ctx, ZERO, ((x, ONE),), ((alpha, ONE),))
    # Each form and the collection length fit separately, but aggregate input
    # occurrences exceed 65536 and must be rejected before combination.
    with pytest.raises(ms.KernelError):
        ms.linear(ctx, ((ONE, value),) * 32_769, enabled=True)
    for assignment in ({}, {x: F(2), alpha: ZERO}, {x: ZERO, alpha: F(2)},
                       {x: 0.0, alpha: ZERO}):
        with pytest.raises(ms.KernelError):
            _evaluate(value, assignment)
    lower = _form(ctx, F(-3), ((x, F(-1)),), ((alpha, F(-1)),))
    upper = _form(ctx, F(3), ((x, F(2)),), ((alpha, F(-1)),))
    readout = ms.readout(frame, ONE, ((qv, F(2)), (xv, ONE)), enabled=True)
    observation = ms.observe(readout, lower, upper, enabled=True)
    rows = ms.compile_rows(observation, enabled=True)
    upper_values, upper_phases, upper_rhs = rows[0]
    lower_values, lower_phases, lower_rhs = rows[1]
    # Source x and readout x are the SAME original value, not two assignments.
    assert dict(upper_values) == {qv: F(2), xv: F(-1)}
    assert dict(upper_phases) == {alpha: ONE}
    assert upper_rhs == F(2)
    assert dict(lower_values) == {qv: F(-2), xv: F(-2)}
    assert dict(lower_phases) == {alpha: F(-1)}
    assert lower_rhs == F(4)
    _assert_rows(observation, {xv: F(1, 4), qv: F(1, 2)}, {alpha: F(3, 4)})
    with pytest.raises(ms.KernelError):
        ms.observe(readout, upper, lower, enabled=True)
    other_frame = ms.make_frame("unrelated frame", enabled=True)
    other_value = ms.make_value(other_frame, "not the original readout", enabled=True)
    unrelated = ms.readout(other_frame, ZERO, ((other_value, ONE),), enabled=True)
    with pytest.raises(ms.KernelError):
        ms.observe(unrelated, lower, upper, enabled=True)
    with pytest.raises(ms.KernelError):
        ms.make_phase(qv, 2, "duplicate original gate", enabled=True)
