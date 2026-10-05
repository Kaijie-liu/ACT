"""Eight preregistered mathematical tests; do not import before root freeze.

Finite truth grids here are test oracles, never a candidate phase-search path.
Formal tokens are not model certificates.  The fixtures state their original
source/gate premises; no benchmark, native HZ binding, or score is inferred.
"""

from fractions import Fraction as F
from itertools import product

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d044_relational_generator_20260930 import relational as old
from experiments.neural_hz_20260831.definition_first_20260928.d046_multiphase_envelopes_20260930 import multiphase as mp


def _pf(frame, bias=F(0), terms=()):
    return mp.form(frame, "phase", F(bias), tuple(terms), enabled=True)


def _vf(frame, bias=F(0), terms=()):
    return mp.form(frame, "value", F(bias), tuple(terms), enabled=True)


def _value_observation(frame, value, lower, upper):
    return mp.observe(_vf(frame, terms=((value, F(1)),)), lower, upper,
                      enabled=True)


def _evaluate(form, assignments):
    return mp.evaluate(form, assignments, enabled=True)


def _phases(pre):
    return (F(0),) if pre < 0 else (F(1),) if pre > 0 else (F(0), F(1))


def _network(x, y, z):
    q, p = max(F(0), x), max(F(0), y)
    w = z + q / 10 - p / 5
    h = z - F(2, 5) * q - F(7, 10) * p + F(1, 4)
    return {"x": x, "y": y, "z": z, "g": x, "f": y, "q": q, "p": p,
            "w": w, "h": h, "t": max(F(0), w), "r": max(F(0), h),
            "alpha": F(int(x > 0)), "eta": F(int(y > 0)),
            "tau": F(int(w > 0)), "beta": F(int(h > 0))}


def _control():
    frame = mp.make_frame("one original network", enabled=True)
    values = {name: mp.make_value(frame, name, enabled=True)
              for name in ("x", "y", "z", "q", "p", "w", "h", "t", "r")}
    phases = {name: mp.make_phase(values[output], ordinal, name, enabled=True)
              for ordinal, (name, output) in enumerate(
                  (("alpha", "q"), ("eta", "p"), ("tau", "t"), ("beta", "r")))}
    q = _value_observation(frame, values["q"], _pf(frame),
                           _pf(frame, terms=((phases["alpha"], F(1)),)))
    p = _value_observation(frame, values["p"], _pf(frame),
                           _pf(frame, terms=((phases["eta"], F(1)),)))
    z = _value_observation(frame, values["z"], _pf(frame, -1), _pf(frame, 1))
    # Normalize the actual shared readout before intervalization.  Do not
    # subtract independent h/w bounds and claim to recover the shared z.
    difference = mp.affine(frame, ((F(-1, 2), q), (F(-1, 2), p)), F(1, 4),
                           enabled=True)
    companion = mp.affine(frame, ((F(1), z), (F(1, 10), q), (F(-1, 5), p)),
                          enabled=True)
    after = mp.relu_pair(difference, companion, values["r"], values["t"],
                         enabled=True)
    return frame, values, phases, z, difference, companion, after


def _old_point():
    return {"x": F(1, 2), "y": F(-1, 2), "z": F(-1, 40),
            "g": F(1, 2), "f": F(-1, 2), "q": F(3, 4), "p": F(1, 4),
            "alpha": F(3, 4), "eta": F(1, 4), "w": F(0), "t": F(7, 20),
            "tau": F(1, 2), "h": F(-1, 4), "r": F(0), "beta": F(0)}


def _phase_assignment(phases, values):
    return {phase: values[name] for name, phase in phases.items()}


def _contains(observation, actual, assignment):
    assert _evaluate(observation.lower, assignment) <= actual
    assert actual <= _evaluate(observation.upper, assignment)


def _physical(values):
    return (values["t"] - values["r"] + (values["q"] - values["x"]) / 4
            + (values["p"] - values["y"]) / 2)


def test_default_off_and_identity():
    assert mp.make_frame(None) is None
    assert mp.make_value(None, None) is None
    assert mp.make_phase(None, None, None) is None
    assert mp.form(None, None, None, None) is None
    assert mp.observe(None, None, None) is None
    assert mp.affine(None, None) is None
    assert mp.positive_majorant(None) is None
    assert mp.positive_minorant(None) is None
    assert mp.relu_pair(None, None, None, None) is None
    assert mp.compile_rows(None) is None
    assert mp.evaluate(None, None) is None
    frame = mp.make_frame("same label", enabled=True)
    other = mp.make_frame("same label", enabled=True)
    assert frame is not other
    q = mp.make_value(frame, "q", enabled=True)
    q_again = mp.make_value(frame, "q", enabled=True)
    assert q is not q_again
    readout = _vf(frame, terms=((q, F(1)), (q_again, F(-1))))
    assert len(readout.terms) == 2
    alpha = mp.make_phase(q, 0, "old bit", enabled=True)
    eta = mp.make_phase(q_again, 1, "old bit", enabled=True)
    combined = _pf(frame, terms=((alpha, F(1)), (eta, F(-1))))
    assert len(combined.terms) == 2
    foreign = mp.make_value(other, "q", enabled=True)
    with pytest.raises(mp.KernelError):
        _vf(frame, terms=((foreign, F(1)),))


def test_majorant_and_minorant_soundness():
    frame = mp.make_frame("finite hinge fixtures", enabled=True)
    phases = tuple(mp.make_phase(mp.make_value(frame, str(i), enabled=True),
                                 i, str(i), enabled=True) for i in range(3))
    fixtures = ((F(-1, 2), ()),
                (F(-1, 2), (F(1),)),
                (F(1, 4), (F(-1),)),
                (F(-1, 2), (F(1), F(1))),
                (F(-3, 4), (F(-3, 4), F(1, 2), F(5, 4))))
    for bias, coefficients in fixtures:
        used = phases[:len(coefficients)]
        original = _pf(frame, bias, tuple(zip(used, coefficients)))
        major = mp.positive_majorant(original, enabled=True)
        minor = mp.positive_minorant(original, enabled=True)
        for point in product((F(0), F(1)), repeat=len(used)):
            assignment = dict(zip(used, point))
            hinge = max(F(0), bias + sum((a * b for a, b in zip(coefficients, point)), F(0)))
            assert _evaluate(minor, assignment) <= hinge <= _evaluate(major, assignment)
        # Only the majorant is claimed on the continuous cube.
        for point in product((F(0), F(1, 4), F(1, 2), F(1)), repeat=len(used)):
            assignment = dict(zip(used, point))
            hinge = max(F(0), bias + sum((a * b for a, b in zip(coefficients, point)), F(0)))
            assert hinge <= _evaluate(major, assignment)
    counter = _pf(frame, F(-1, 2), ((phases[0], F(1)), (phases[1], F(1))))
    fractional = {phases[0]: F(1, 4), phases[1]: F(1, 4)}
    assert max(F(0), _evaluate(counter, fractional)) == 0
    assert _evaluate(mp.positive_minorant(counter, enabled=True), fractional) == F(1, 4)


def test_single_anchor_matches_d044():
    frame = mp.make_frame("single original bit", enabled=True)
    q, h, w, r, t = (mp.make_value(frame, name, enabled=True)
                     for name in ("anchor q", "h", "w", "r", "t"))
    alpha = mp.make_phase(q, 0, "alpha", enabled=True)
    old_frame = old.make_frame("same mathematical fixture", enabled=True)
    oq, oh, ow, or_, ot = (old.make_symbol(old_frame, name, enabled=True)
                           for name in ("q", "h", "w", "r", "t"))
    old_anchor = old.make_anchor(oq, "alpha", enabled=True)
    fixtures = ((((F(-3, 4), F(1, 2)), (F(1, 4), F(5, 4))),
                 ((F(-1), F(3, 4)), (F(1, 8), F(3, 2)))),
                (((F(-1), F(-1, 4)), (F(-2), F(2))),
                 ((F(1, 4), F(1)), (F(-1), F(0)))))
    for delta, companion in fixtures:
        lower = _pf(frame, delta[0][0], ((alpha, delta[1][0] - delta[0][0]),))
        upper = _pf(frame, delta[0][1], ((alpha, delta[1][1] - delta[0][1]),))
        difference = mp.observe(_vf(frame, terms=((h, F(1)), (w, F(-1)))),
                                lower, upper, enabled=True)
        lower_w = _pf(frame, companion[0][0], ((alpha, companion[1][0] - companion[0][0]),))
        upper_w = _pf(frame, companion[0][1], ((alpha, companion[1][1] - companion[0][1]),))
        before_w = _value_observation(frame, w, lower_w, upper_w)
        actual = mp.relu_pair(difference, before_w, r, t, enabled=True)
        old_d = old.observation(old_anchor, ((oh, F(1)), (ow, F(-1))), F(0),
                                delta, enabled=True)
        old_w = old.observation(old_anchor, ((ow, F(1)),), F(0), companion, enabled=True)
        expected = old.relu_pair(old.pair(old_d, old_w, enabled=True), or_, ot, enabled=True)
        for state in (0, 1):
            assignment = {alpha: F(state)}
            for current, legacy in ((actual[0], expected.difference),
                                     (actual[1], expected.companion)):
                assert (_evaluate(current.lower, assignment), _evaluate(current.upper, assignment)) == legacy.bounds[state]


def test_mixed_phase_affine_transfer():
    frame = mp.make_frame("mixed coefficients", enabled=True)
    q, p = (mp.make_value(frame, name, enabled=True) for name in ("q", "p"))
    alpha = mp.make_phase(q, 0, "alpha", enabled=True)
    eta = mp.make_phase(p, 1, "eta", enabled=True)
    first = _value_observation(frame, q, _pf(frame, -1, ((alpha, F(1, 2)),)),
                               _pf(frame, 2, ((alpha, F(1)),)))
    second = _value_observation(frame, p, _pf(frame, -2, ((eta, F(1)),)),
                                _pf(frame, 1, ((eta, F(-1, 2)),)))
    mixed = mp.affine(frame, ((F(3, 2), first), (F(-2), second)), F(1, 4), enabled=True)
    assert mixed.lower.bias == F(-13, 4)
    assert dict(mixed.lower.terms) == {alpha: F(3, 4), eta: F(1)}
    assert mixed.upper.bias == F(29, 4)
    assert dict(mixed.upper.terms) == {alpha: F(3, 2), eta: F(-2)}
    assert dict(mixed.readout.terms) == {q: F(3, 2), p: F(-2)}
    cancelled = mp.affine(frame, ((F(7), first), (F(-7), first)), F(1, 8), enabled=True)
    assert cancelled.readout.terms == cancelled.lower.terms == cancelled.upper.terms == ()
    assert cancelled.readout.bias == cancelled.lower.bias == cancelled.upper.bias == F(1, 8)
    signed = _pf(frame, F(1, 4), ((eta, F(2)), (alpha, F(-1))))
    major = mp.positive_majorant(signed, enabled=True)
    minor = mp.positive_minorant(signed, enabled=True)
    assert major.bias == minor.bias == F(1, 4)
    assert dict(major.terms) == {alpha: F(-1, 4), eta: F(2)}
    assert dict(minor.terms) == {alpha: F(-1, 4), eta: F(5, 4)}
    duplicate = _pf(frame, F(1, 4), ((alpha, F(2)), (alpha, F(-2))))
    assert duplicate.terms == ()
    assert mp.positive_majorant(duplicate, enabled=True).bias == F(1, 4)


def test_strong_hull_control():
    _, _, phases, _, _, _, after = _control()
    sources = ((F(3, 8), F(1), F(-1), F(3, 5)),
               (F(3, 8), F(1), F(-1), F(-4, 5)),
               (F(1, 8), F(-1), F(1), F(9, 10)),
               (F(1, 8), F(-1), F(1), F(-1, 2)))
    prefix = tuple((weight, _network(x, y, z)) for weight, x, y, z in sources)
    target = _old_point()
    for weight, values in prefix:
        assert weight > 0
        assert all(F(-1) <= values[name] <= F(1) for name in ("x", "y", "z"))
        assert all(values[name] != 0 for name in ("g", "f", "w"))
    assert sum((weight for weight, _ in prefix), F(0)) == 1
    for name in ("x", "y", "z", "g", "f", "q", "p", "w", "t", "alpha", "eta", "tau"):
        assert sum((weight * values[name] for weight, values in prefix), F(0)) == target[name]
    # The continuous full-prefix point is itself a graph point of the final gate.
    assert target["h"] == target["z"] - F(2, 5) * target["q"] - F(7, 10) * target["p"] + F(1, 4)
    assert target["r"] == max(F(0), target["h"]) == 0 and target["beta"] == 0
    gate = {"g": ("q", "alpha"), "f": ("p", "eta"),
            "w": ("t", "tau"), "h": ("r", "beta")}
    pairs = (("g", "f", F(-2), F(2)),
             ("g", "w", F(-2), F(21, 10)),
             ("f", "w", F(-21, 10), F(11, 5)),
             ("g", "h", F(-9, 4), F(57, 20)),
             ("f", "h", F(-9, 4), F(57, 20)),
             ("w", "h", F(-1, 4), F(3, 4)))
    for first, second, lo, hi in pairs:
        output_i, bit_i = gate[first]
        output_j, bit_j = gate[second]
        delta = target[first] - target[second]
        d = target[output_i] - target[output_j]
        e = d - delta
        assert lo * target[bit_j] <= d <= hi * target[bit_i]
        assert -hi * (1 - target[bit_j]) <= e <= -lo * (1 - target[bit_i])
    difference = target["t"] - target["r"]
    assert difference <= F(1, 4) + target["alpha"] / 2
    assert difference <= F(1, 4) + target["eta"] / 2
    assert difference <= F(3, 4) * target["tau"]
    assignment = _phase_assignment(phases, target)
    assert -_evaluate(after[0].lower, assignment) == F(5, 16)
    assert difference - F(5, 16) == F(3, 80)


def test_physical_separation_and_attainment():
    _, _, phases, _, _, _, after = _control()
    assert _physical(_old_point()) == F(63, 80) > F(3, 4)
    attain = _network(F(-1), F(-1), F(-1, 2))
    assert all(attain[name] != 0 for name in ("g", "f", "w", "h"))
    assert _physical(attain) == F(3, 4)
    for x, y, z in product((F(-1), F(0), F(1)), repeat=3):
        values = _network(x, y, z)
        assert _physical(values) <= F(3, 4)
        # Keep each original zero-point phase choice; never identify bits.
        for bits in product(*(_phases(values[name]) for name in ("g", "f", "w", "h"))):
            labeled = dict(values)
            labeled.update(zip(("alpha", "eta", "tau", "beta"), bits))
            assignment = _phase_assignment(phases, labeled)
            _contains(after[0], values["r"] - values["t"], assignment)
            _contains(after[1], values["t"], assignment)
            combined = (-_evaluate(after[0].lower, assignment)
                        + (1 - labeled["alpha"]) / 4 + (1 - labeled["eta"]) / 2)
            assert combined == F(3, 4)
    threshold = F(123, 160)
    assert threshold - _physical(attain) == _physical(_old_point()) - threshold == F(3, 160)


def test_second_mixed_block():
    frame, values, phases, z_obs, _, _, after = _control()
    r2 = mp.make_value(frame, "original r2", enabled=True)
    t2 = mp.make_value(frame, "original t2", enabled=True)
    beta2 = mp.make_phase(r2, 4, "original beta2", enabled=True)
    tau2 = mp.make_phase(t2, 5, "original tau2", enabled=True)
    # Actual w2=.1+t-.5r+.2z and h2=.1+1.1r-.6t+.2z.
    # Reuse the SAME original alpha/eta in e=r-t, not beta2/tau2.
    d2 = mp.affine(frame, ((F(8, 5), after[0]),), enabled=True)
    w2 = mp.affine(frame, ((F(-1, 2), after[0]), (F(1, 2), after[1]),
                            (F(1, 5), z_obs)), F(1, 10), enabled=True)
    next_pair = mp.relu_pair(d2, w2, r2, t2, enabled=True)
    assert next_pair[0].lower.bias == 0
    assert dict(next_pair[0].lower.terms) == {phases["alpha"]: F(-2, 5),
                                             phases["eta"]: F(-4, 5)}
    for observation in next_pair:
        for bound in (observation.lower, observation.upper):
            assert set(dict(bound.terms)) <= {phases["alpha"], phases["eta"]}
            assert beta2 not in dict(bound.terms) and tau2 not in dict(bound.terms)
    assert dict(next_pair[0].readout.terms) == {r2: F(1), t2: F(-1)}
    for x, y, z in product((F(-1), F(0), F(1)), repeat=3):
        original = _network(x, y, z)
        actual_w2 = F(1, 10) + original["t"] - original["r"] / 2 + z / 5
        actual_h2 = F(1, 10) + F(11, 10) * original["r"] - F(3, 5) * original["t"] + z / 5
        assert actual_h2 - actual_w2 == F(8, 5) * (original["r"] - original["t"])
        actual_t2, actual_r2 = max(F(0), actual_w2), max(F(0), actual_h2)
        assignment = _phase_assignment(phases, original)
        _contains(next_pair[0], actual_r2 - actual_t2, assignment)
        _contains(next_pair[1], actual_t2, assignment)
    # This witness is already excluded by block 1; no new block-2 separation is claimed.
    relaxed = _old_point()
    actual_w2 = F(1, 10) + relaxed["t"] - relaxed["r"] / 2 + relaxed["z"] / 5
    actual_h2 = F(1, 10) + F(11, 10) * relaxed["r"] - F(3, 5) * relaxed["t"] + relaxed["z"] / 5
    assert (actual_w2, actual_h2) == (F(89, 200), F(-23, 200))
    assert max(F(0), actual_w2) - max(F(0), actual_h2) <= F(1, 2)


def test_rows_and_rejected_premises():
    frame = mp.make_frame("row fixture", enabled=True)
    q, p = (mp.make_value(frame, name, enabled=True) for name in ("q", "p"))
    alpha = mp.make_phase(q, 0, "alpha", enabled=True)
    eta = mp.make_phase(p, 1, "eta", enabled=True)
    readout = _vf(frame, F(3, 7), ((q, F(2)), (p, F(-1))))
    lower = _pf(frame, F(-2), ((alpha, F(1, 2)), (eta, F(-1, 4))))
    upper = _pf(frame, F(3), ((alpha, F(3, 4)), (eta, F(1, 8))))
    observation = mp.observe(readout, lower, upper, enabled=True)
    rows = mp.compile_rows(observation, enabled=True)
    assert len(rows) == 2
    assert dict(rows[0][0]) == {q: F(2), p: F(-1)}
    assert dict(rows[0][1]) == {alpha: F(-3, 4), eta: F(-1, 8)}
    assert rows[0][2] == F(18, 7)
    assert dict(rows[1][0]) == {q: F(-2), p: F(1)}
    assert dict(rows[1][1]) == {alpha: F(1, 2), eta: F(-1, 4)}
    assert rows[1][2] == F(17, 7)
    other = mp.make_frame("row fixture", enabled=True)
    foreign = mp.make_value(other, "q", enabled=True)
    foreign_obs = _value_observation(other, foreign, _pf(other, -1), _pf(other, 1))
    with pytest.raises(mp.KernelError):
        mp.affine(frame, ((F(0), foreign_obs),), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.make_phase(p, 0, "duplicate original ordinal", enabled=True)
    with pytest.raises(mp.KernelError):
        mp.form(frame, "invalid kind", F(0), (), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.form(frame, "phase", 0.0, (), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.form(frame, "phase", F(0), ((alpha, 1),), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.observe(readout, _pf(frame, 1), _pf(frame, 0), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.observe(readout, _pf(other, -1), upper, enabled=True)
    with pytest.raises(mp.KernelError):
        mp.relu_pair(observation, observation, foreign, p, enabled=True)
    with pytest.raises(mp.KernelError):
        mp.make_frame("wrong opt-in type", enabled=1)
    needs_alpha = _pf(frame, terms=((alpha, F(1)),))
    with pytest.raises(mp.KernelError):
        mp.evaluate(needs_alpha, {}, enabled=True)
    with pytest.raises(mp.KernelError):
        mp.evaluate(needs_alpha, {alpha: F(3, 2)}, enabled=True)
    # Finite resource rejection is part of this component, not a new search.
    with pytest.raises(mp.KernelError):
        mp.form(frame, "phase", F(1 << 512), (), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.form(frame, "phase", F(0),
                ((alpha, F(1 << 511)), (alpha, F(1 << 511))), enabled=True)
    with pytest.raises(mp.KernelError):
        mp.form(frame, "phase", F(0), ((alpha, F(1)),) * 65_537, enabled=True)
    # A predicate might imply alpha<=eta; this API does not certify that premise.
    with pytest.raises(mp.KernelError):
        mp.observe(readout, _pf(frame, terms=((alpha, F(1)),)),
                   _pf(frame, terms=((eta, F(1)),)), enabled=True)
