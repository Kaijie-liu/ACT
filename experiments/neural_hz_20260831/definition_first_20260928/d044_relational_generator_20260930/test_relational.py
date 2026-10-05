"""Exactly ten small, nonparameterized tests; run only after root freeze.

The rational fixtures establish their own conditional premises on paper.  They
do not treat token labels as source certificates, benchmark results, or a model
binding.  This module must not be imported before the single supervised run.
"""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d044_relational_generator_20260930 import relational as rel


def _context():
    frame = rel.make_frame("original shared frame", enabled=True)
    q = rel.make_symbol(frame, "q", enabled=True)
    p = rel.make_symbol(frame, "p", enabled=True)
    anchor = rel.make_anchor(q, "q original alpha", enabled=True)
    return frame, q, p, anchor


def _source(anchor, symbol, lo, hi):
    interval = (F(lo), F(hi))
    return rel.observation(anchor, ((symbol, F(1)),), F(0),
                           (interval, interval), enabled=True)


def _phases(preactivation):
    return (0,) if preactivation < 0 else (1,) if preactivation > 0 else (0, 1)


def _contains(observation, value, state):
    lo, hi = observation.bounds[state]
    assert lo <= value <= hi


def _network(x, y, z, f_bias=F(-1, 4)):
    g, f = x, F(9, 10) * x + z / 10 + f_bias
    q, p = max(F(0), g), max(F(0), f)
    w = y + q / 5 - p / 10
    h = y + F(2, 5) * q - p + F(3, 20) - z / 10
    return {"x": x, "y": y, "z": z, "g": g, "f": f,
            "q": q, "p": p, "w": w, "h": h,
            "t": max(F(0), w), "r": max(F(0), h)}


def _control(f_bias=F(-1, 4)):
    frame, q, p, anchor = _context()
    y = rel.make_symbol(frame, "original y", enabled=True)
    z = rel.make_symbol(frame, "original z", enabled=True)
    r = rel.make_symbol(frame, "original r", enabled=True)
    t = rel.make_symbol(frame, "original t", enabled=True)
    y_obs, z_obs = _source(anchor, y, -1, 1), _source(anchor, z, -1, 1)
    seed = rel.seed_pair(anchor, q, p, (F(-1), F(1)),
                         (F(-1) + f_bias, F(1) + f_bias),
                         (F(-1, 5) - f_bias, F(1, 5) - f_bias), enabled=True)
    pre = rel.paired_affine(((F(2, 5), F(-1)),), ((F(1, 5), F(-1, 10)),),
                            (seed,), F(3, 20), F(0),
                            left_extras=((F(1), y_obs), (F(-1, 10), z_obs)),
                            right_extras=((F(1), y_obs),), enabled=True)
    post = rel.relu_pair(pre, r, t, enabled=True)
    return {"frame": frame, "anchor": anchor, "q": q, "p": p,
            "y": y, "z": z, "r": r, "t": t, "y_obs": y_obs,
            "seed": seed, "pre": pre, "post": post}


def test_seed_grid_contains_original_values():
    _, q, p, anchor = _context()
    seeded = rel.seed_pair(anchor, q, p, (F(-1), F(1)), (F(-3, 4), F(3, 4)),
                           (F(-1, 2), F(3, 4)), enabled=True)
    for g in (F(-1), F(-1, 2), F(0), F(1, 2), F(1)):
        for f in (F(-3, 4), F(-1, 2), F(0), F(1, 2), F(3, 4)):
            if not F(-1, 2) <= g - f <= F(3, 4):
                continue
            actual_q, actual_p = max(F(0), g), max(F(0), f)
            for alpha in _phases(g):
                for eta in _phases(f):
                    assert eta in (0, 1)
                    _contains(seeded.companion, actual_p, alpha)
                    _contains(seeded.difference, actual_q - actual_p, alpha)
    assert seeded.anchor is anchor


def test_mixed_affine_and_exact_identity_cancellation():
    frame, q, p, anchor = _context()
    x_obs = _source(anchor, q, -1, 2)
    p_obs = _source(anchor, p, 1, 3)
    y = rel.make_symbol(frame, "shared shortcut", enabled=True)
    y_first, y_again = _source(anchor, y, -2, 2), _source(anchor, y, -1, 1)
    mixed = rel.affine(anchor, ((F(2), x_obs), (F(-3), p_obs),
                                (F(7), y_first), (F(-7), y_again)), F(1, 4), enabled=True)
    assert mixed.bounds == ((F(-43, 4), F(5, 4)),) * 2
    assert y not in dict(mixed.terms)
    d_obs = rel.observation(anchor, ((q, F(1)), (p, F(-1))), F(0),
                            ((F(-4), F(1)),) * 2, enabled=True)
    exact_zero = rel.affine(anchor, ((F(1), d_obs), (F(1), p_obs),
                                     (F(-1), x_obs)), enabled=True)
    assert exact_zero.terms == ()
    assert exact_zero.bounds == ((F(0), F(0)),) * 2


def test_companion_is_required_for_unequal_weights():
    frame, q, p, _ = _context()
    original_anchor_output = rel.make_symbol(frame, "independent original gate", enabled=True)
    anchor = rel.make_anchor(original_anchor_output, "old original bit", enabled=True)
    difference = rel.observation(anchor, ((q, F(1)), (p, F(-1))), F(0),
                                 ((F(0), F(0)),) * 2, enabled=True)
    companion = _source(anchor, p, 1, 2)
    original_pair = rel.pair(difference, companion, enabled=True)
    output = rel.paired_affine(((F(1), F(0)),), ((F(0), F(2)),),
                               (original_pair,), F(0), F(0), enabled=True)
    assert output.difference.bounds == ((F(-2), F(-1)),) * 2
    for original_value in (F(1), F(2)):
        actual = original_value - 2 * original_value
        assert actual != F(0)  # A difference-only zero replacement is unsound.
        _contains(output.difference, actual, 0)
        _contains(output.difference, actual, 1)
    with pytest.raises(rel.KernelError):
        rel.pair(difference, None, enabled=True)


def test_relu_crossing_and_zero_original_phase_choices():
    frame, q, p, anchor = _context()
    difference = rel.observation(anchor, ((q, F(1)), (p, F(-1))), F(0),
                                 ((F(-1, 2), F(1, 2)), (F(0), F(0))), enabled=True)
    companion = _source(anchor, p, -1, 1)
    before = rel.pair(difference, companion, enabled=True)
    r, t = (rel.make_symbol(frame, label, enabled=True) for label in ("original r", "original t"))
    after = rel.relu_pair(before, r, t, enabled=True)
    for h in (F(-1), F(-1, 2), F(0), F(1, 2), F(1)):
        for w in (F(-1), F(-1, 2), F(0), F(1, 2), F(1)):
            for alpha in (0, 1):
                lo, hi = difference.bounds[alpha]
                if not lo <= h - w <= hi:
                    continue
                for beta in _phases(h):
                    for tau in _phases(w):
                        assert beta in (0, 1) and tau in (0, 1)
                        _contains(after.difference, max(F(0), h) - max(F(0), w), alpha)
                        _contains(after.companion, max(F(0), w), alpha)
    zero_seed = rel.seed_pair(anchor, q, p, (F(0), F(0)), (F(0), F(0)),
                              (F(0), F(0)), enabled=True)
    assert zero_seed.difference.bounds == ((F(0), F(0)),) * 2
    assert zero_seed.anchor is anchor and after.anchor is anchor


def test_different_frame_anchor_and_bit_binding_are_rejected():
    frame, q, p, anchor = _context()
    q_obs = _source(anchor, q, -1, 1)
    other_bit = rel.make_anchor(p, "other original bit", enabled=True)
    p_other = _source(other_bit, p, -1, 1)
    with pytest.raises(rel.KernelError):
        rel.affine(anchor, ((F(1), q_obs), (F(0), p_other)), enabled=True)
    with pytest.raises(rel.KernelError):
        rel.seed_pair(other_bit, q, p, (F(-1), F(1)), (F(-1), F(1)),
                      (F(-2), F(2)), enabled=True)
    other_frame = rel.make_frame(frame.label, enabled=True)
    foreign = rel.make_symbol(other_frame, q.label, enabled=True)
    with pytest.raises(rel.KernelError):
        _source(anchor, foreign, -1, 1)
    same_label = rel.make_symbol(frame, q.label, enabled=True)
    first, second = _source(anchor, q, 1, 1), _source(anchor, same_label, 2, 2)
    unequal = rel.affine(anchor, ((F(1), first), (F(-1), second)), enabled=True)
    assert len(unequal.terms) == 2  # Identical strings do not merge source identities.
    assert unequal.bounds == ((F(-1), F(-1)),) * 2
    before = rel.pair(q_obs, q_obs, enabled=True)
    with pytest.raises(rel.KernelError):
        rel.relu_pair(before, foreign, p, enabled=True)


def test_generic_control_exact_seed_and_paired_endpoints():
    values = _control()
    seed, before, after = values["seed"], values["pre"], values["post"]
    assert seed.difference.bounds == ((F(0), F(0)), (F(0), F(9, 20)))
    assert seed.companion.bounds == ((F(0), F(0)), (F(0), F(3, 4)))
    assert before.difference.bounds == ((F(1, 20), F(1, 4)), (F(-19, 40), F(17, 50)))
    assert before.companion.bounds == ((F(-1), F(1)), (F(-1), F(233, 200)))
    assert after.difference.bounds == ((F(0), F(1, 4)), (F(-19, 40), F(17, 50)))
    assert after.companion.bounds == ((F(0), F(1)), (F(0), F(233, 200)))
    assert values["y"] not in dict(before.difference.terms)
    assert dict(before.difference.terms) == {values["q"]: F(1, 5),
                                            values["p"]: F(-9, 10), values["z"]: F(-1, 10)}
    # A fixed second mathematical fixture, not a runtime model/instance branch.
    non_ordered = _control(F(-1, 20))
    assert non_ordered["seed"].difference.bounds == ((F(-3, 20), F(0)),
                                                     (F(-3, 20), F(1, 4)))
    assert non_ordered["seed"].companion.bounds == ((F(0), F(3, 20)),
                                                    (F(0), F(19, 20)))
    assert non_ordered["pre"].difference.bounds == ((F(-17, 200), F(1, 4)),
                                                    (F(-129, 200), F(3, 10)))
    assert non_ordered["post"].difference.bounds == non_ordered["pre"].difference.bounds
    no_dominance = _network(F(-1, 25), F(0), F(1), F(-1, 20))
    assert no_dominance["q"] == 0 and no_dominance["p"] == F(7, 500) > 0
    reverse = _network(F(1), F(0), F(-1), F(-1, 20))
    assert reverse["q"] == 1 and reverse["p"] == F(3, 4) < reverse["q"]


def test_generic_control_physical_separation_and_true_attainment():
    values = _control()
    coefficient = -values["post"].difference.bounds[1][0]
    assert coefficient == F(19, 40)
    old = {"g": F(-9, 10), "f": F(-53, 50), "w": F(0), "h": F(3, 20),
           "q": F(0), "p": F(0), "t": F(1, 4), "r": F(1, 5)}
    assert old["t"] - old["r"] + coefficient * F(9, 10) == F(191, 400)
    assert F(191, 400) > coefficient
    upper, lower = rel.compile_rows(values["post"].difference, enabled=True)
    symbol_values = {values["r"]: old["r"], values["t"]: old["t"]}
    assert sum(amount * symbol_values[symbol] for symbol, amount in lower[0]) > lower[3]
    assert upper[1] is lower[1] is values["anchor"]
    gates = {"g": ("q", F(0)), "f": ("p", F(0)),
             "w": ("t", F(3, 10)), "h": ("r", F(1, 2))}
    difference_bounds = (("g", "f", F(1, 20), F(9, 20)),
                         ("g", "w", F(-2), F(15, 8)),
                         ("f", "w", F(-9, 4), F(13, 8)),
                         ("g", "h", F(-9, 4), F(23, 10)),
                         ("f", "h", F(-5, 2), F(41, 20)),
                         ("h", "w", F(-17, 40), F(59, 180)))
    for left, right, lo, hi in difference_bounds:
        out_left, bit_left = gates[left]
        out_right, bit_right = gates[right]
        d = old[out_left] - old[out_right]
        e = d - (old[left] - old[right])
        assert lo * bit_right <= d <= hi * bit_left
        assert -hi * (1 - bit_right) <= e <= -lo * (1 - bit_left)

    variant_mixtures = (((F(1, 2), F(22, 25)), (F(1, 2), F(-29, 50))),
                        ((F(1), F(-3, 20)),))
    variant_prefix = []
    for mixture in variant_mixtures:
        points = [(weight, _network(F(-19, 20), y, F(1), F(-1, 20)))
                  for weight, y in mixture]
        assert all(-1 <= point["y"] <= 1 and point["g"] < 0 and point["f"] < 0
                   and point["w"] != 0 for _, point in points)
        y = sum(weight * point["y"] for weight, point in points)
        t = sum(weight * point["t"] for weight, point in points)
        tau = sum(weight * int(point["w"] > 0) for weight, point in points)
        variant_prefix.append((y, t, tau))
    assert tuple(variant_prefix) == ((F(3, 20), F(11, 25), F(1, 2)),
                                     (F(-3, 20), F(0), F(0)))
    assert sum(point[0] for point in variant_prefix) / 2 == 0
    assert sum(point[1] for point in variant_prefix) / 2 == F(11, 50)
    assert sum(point[2] for point in variant_prefix) / 2 == F(1, 4)
    assert sum(max(F(0), point[0] + F(1, 20)) for point in variant_prefix) / 2 == F(1, 10)
    # Whole prefix hull, followed by last-gate hull over that continuous P.
    active = ((F(1, 2), F(23, 25)), (F(1, 2), F(-21, 50)))
    inactive = ((F(1, 10), F(2, 5)), (F(9, 10), F(-29, 90)))
    prefix_points = []
    for mixture in (active, inactive):
        y = sum(weight * source_y for weight, source_y in mixture)
        t = sum(weight * max(F(0), source_y) for weight, source_y in mixture)
        tau = sum(weight * int(source_y > 0) for weight, source_y in mixture)
        prefix_points.append((y, t, tau))
    assert tuple(prefix_points) == ((F(1, 4), F(23, 50), F(1, 2)),
                                    (F(-1, 4), F(1, 25), F(1, 10)))
    assert sum(max(F(0), point[0] + F(3, 20)) for point in prefix_points) / 2 == old["r"]
    actual = _network(F(-1), F(-1, 2), F(0))
    assert all(actual[name] != 0 for name in ("g", "f", "w", "h"))
    assert actual["t"] - actual["r"] + coefficient * (actual["q"] - actual["x"]) == coefficient
    non_ordered = _control(F(-1, 20))
    (l0, _), (l1, _) = non_ordered["post"].difference.bounds
    intercept, slope = -l0, l0 - l1
    assert (intercept, slope, intercept + slope) == (F(17, 200), F(14, 25), F(129, 200))
    old_variant_z = F(11, 50) - F(1, 10) + slope * F(19, 20)
    assert old_variant_z == F(163, 250)
    assert old_variant_z - (intercept + slope) == F(7, 1000)
    # The same global-pair comparator, now for the non-ordered fixed fixture.
    variant = {"g": F(-19, 20), "f": F(-161, 200), "w": F(0), "h": F(1, 20),
               "q": F(0), "p": F(0), "t": F(11, 50), "r": F(1, 10)}
    variant_gates = {"g": ("q", F(0)), "f": ("p", F(0)),
                     "w": ("t", F(1, 4)), "h": ("r", F(1, 2))}
    variant_bounds = (("g", "f", F(-3, 20), F(1, 4)),
                      ("g", "w", F(-2), F(379, 200)),
                      ("f", "w", F(-41, 20), F(369, 200)),
                      ("g", "h", F(-9, 4), F(5, 2)),
                      ("f", "h", F(-23, 10), F(49, 20)),
                      ("h", "w", F(-121, 200), F(17, 60)))
    for left, right, lo, hi in variant_bounds:
        out_left, bit_left = variant_gates[left]
        out_right, bit_right = variant_gates[right]
        d = variant[out_left] - variant[out_right]
        e = d - (variant[left] - variant[right])
        assert lo * bit_right <= d <= hi * bit_left
        assert -hi * (1 - bit_right) <= e <= -lo * (1 - bit_left)


def test_additional_mixed_block_is_composable():
    values = _control()
    r2 = rel.make_symbol(values["frame"], "existing r2", enabled=True)
    t2 = rel.make_symbol(values["frame"], "existing t2", enabled=True)
    extra = ((F(1, 5), values["y_obs"]),)
    before = rel.paired_affine(((F(11, 10), F(-3, 5)),), ((F(-1, 2), F(1)),),
                               (values["post"],), F(1, 10), F(1, 10),
                               left_extras=extra, right_extras=extra, enabled=True)
    assert before.difference.bounds == ((F(0), F(2, 5)), (F(-19, 25), F(68, 125)))
    assert dict(before.difference.terms) == {values["r"]: F(8, 5), values["t"]: F(-8, 5)}
    after = rel.relu_pair(before, r2, t2, enabled=True)
    assert after.anchor is values["anchor"]
    assert after.difference.bounds == before.difference.bounds
    old_r2, old_t2 = F(17, 100), F(1, 4)
    assert old_t2 - old_r2 + F(19, 25) * F(9, 10) == F(191, 250) > F(19, 25)
    actual = _network(F(-1), F(-3, 4), F(0))
    w2 = F(1, 10) + actual["t"] - actual["r"] / 2 + actual["y"] / 5
    h2 = F(1, 10) + F(11, 10) * actual["r"] - F(3, 5) * actual["t"] + actual["y"] / 5
    assert h2 == w2 == F(-1, 20)
    assert max(F(0), w2) - max(F(0), h2) + F(19, 25) * (actual["q"] - actual["x"]) == F(19, 25)


def test_default_off_and_malformed_or_contradictory_premises():
    assert rel.make_frame(None) is None
    assert rel.make_symbol(None, None) is None
    assert rel.make_anchor(None, None) is None
    assert rel.observation(None, None, None, None) is None
    assert rel.pair(None, None) is None
    assert rel.seed_pair(None, None, None, None, None, None) is None
    assert rel.affine(None, None) is None
    assert rel.paired_affine(None, None, None, None, None) is None
    assert rel.relu_pair(None, None, None) is None
    assert rel.compile_rows(None) is None
    _, q, p, anchor = _context()
    with pytest.raises(rel.KernelError):
        rel.make_frame("bad enabled", enabled=1)
    with pytest.raises(rel.KernelError):
        rel.seed_pair(anchor, q, p, None, (F(-1), F(1)), (F(-2), F(2)), enabled=True)
    with pytest.raises(rel.KernelError):
        rel.seed_pair(anchor, q, p, (F(-1), F(1)), (F(-1), F(1)),
                      (F(3), F(4)), enabled=True)
    with pytest.raises(rel.KernelError):
        rel.seed_pair(anchor, q, p, (F(1), F(-1)), (F(-1), F(1)),
                      (F(-2), F(2)), enabled=True)
    with pytest.raises(rel.KernelError):
        rel.observation(anchor, ((q, 1.0),), F(0), ((F(0), F(1)),) * 2, enabled=True)
    with pytest.raises(rel.KernelError):
        rel.observation(anchor, (), F(1), ((F(0), F(0)),) * 2, enabled=True)


def test_compiled_rows_and_constant_bias_are_exact():
    _, q, p, anchor = _context()
    observed = rel.observation(anchor, ((q, F(2)), (p, F(-3))), F(1, 4),
                               ((F(-1), F(2)), (F(3), F(4))), enabled=True)
    upper, lower = rel.compile_rows(observed, enabled=True)
    assert dict(upper[0]) == {q: F(2), p: F(-3)}
    assert dict(lower[0]) == {q: F(-2), p: F(3)}
    assert upper[1] is lower[1] is anchor
    assert upper[2:] == (F(-2), F(7, 4))
    assert lower[2:] == (F(4), F(5, 4))
    constant = rel.affine(anchor, (), F(3, 7), enabled=True)
    assert constant.bounds == ((F(3, 7), F(3, 7)),) * 2
    for terms, bit, coefficient, rhs in rel.compile_rows(constant, enabled=True):
        assert terms == () and bit is anchor
        assert coefficient == rhs == F(0)
