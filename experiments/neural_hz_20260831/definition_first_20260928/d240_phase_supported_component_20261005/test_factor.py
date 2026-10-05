"""Sixteen fixed exact tests of the declared D240 mathematical component.

Finite grids and convex combinations below are independent test witnesses,
never candidate input, bounds, or a runtime phase algorithm.  In particular,
the two precision controls use a nonnegative combination of COMPILED rows,
not an LP, a guessed auxiliary assignment, or a model-verification claim.
Do not import, compile, collect, or execute before the main-agent freeze.
"""

from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d240_phase_supported_component_20261005 import factor


ZERO, ONE = F(0), F(1)
RUN = (Path(__file__).resolve().parents[2]
       / "results/d240_phase_supported_component_20261005_v1")
_NAMES = (
    "default_off_and_identity", "integer_graph_extension",
    "symmetric_certificate", "asymmetric_certificate",
    "strong_prefix_witnesses", "amplitude_anchor_contract",
    "complete_residual_payment", "unsupported_guard_rejected",
    "original_zero_labels", "retained_source_predicates",
    "difference_only_alias", "exact_arithmetic_rejection",
    "sticky_resource_limits", "shared_state_immutable",
    "registered_parameter_family", "complete_record",
)
_EVIDENCE, _CASES, _SNAPSHOTS = {}, {}, {}
_BUDGET = None


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    """The sole exclusive writer, independent of every mathematical test."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = factor.Budget()
    return _BUDGET


def _expr(constant=0, terms=()):
    merged = {}
    for index, amount in terms:
        merged[index] = merged.get(index, ZERO) + F(amount)
    return F(constant), {i: a for i, a in merged.items() if a}


def _sum(*weighted):
    constant, terms = ZERO, {}
    for weight, (bias, entries) in weighted:
        weight = F(weight)
        constant += weight * bias
        for index, amount in entries.items():
            terms[index] = terms.get(index, ZERO) + weight * amount
    return _expr(constant, terms.items())


def _col(index):
    return _expr(0, ((index, ONE),))


def _affine(value):
    return _expr(value.constant, value.terms)


def _row(expression, rhs=0):
    constant, terms = expression
    return factor.Row(tuple(sorted(terms.items())), F(rhs) - constant)


def _value(expression, values):
    constant, terms = expression
    return constant + sum((a * values[i] for i, a in terms.items()), ZERO)


def _lookup(rows, expression, rhs=0):
    expected = _row(expression, rhs)
    matches = [row for row in rows if row == expected]
    assert matches, (expected, "required compiled row is absent")
    return matches[0]


def _nonnegative_sum(weighted_rows):
    terms, rhs = {}, ZERO
    for weight, row in weighted_rows:
        weight = F(weight)
        assert weight >= ZERO
        rhs += weight * row.rhs
        for index, amount in row.terms:
            terms[index] = terms.get(index, ZERO) + weight * amount
    return factor.Row(tuple(sorted((i, a) for i, a in terms.items() if a)), rhs)


def _symbols(block):
    n = block.n_columns - 8
    result = {"x1": _col(0), "x2": _col(1),
              "q1": _col(n), "q2": _col(n + 1),
              "y1": _col(n + 2), "y2": _col(n + 3),
              "alpha1": _expr(F(1, 2), ((n + 4, F(1, 2)),)),
              "alpha2": _expr(F(1, 2), ((n + 5, F(1, 2)),)),
              "c12": _col(block.n_columns), "c21": _col(block.n_columns + 1),
              "v1": _col(block.n_columns + 2), "v2": _col(block.n_columns + 3)}
    for i in (1, 2):
        result["p%d" % i] = _sum((1, result["alpha%d" % i]),
                                 (-1, result["q%d" % i]))
        result["n%d" % i] = _sum((1, _expr(1)),
                                 (-1, result["alpha%d" % i]),
                                 (-1, result["q%d" % i]),
                                 (1, result["x%d" % i]))
    result["D"] = _sum((2, result["q1"]), (2, result["q2"]),
                        (-1, result["x1"]), (-1, result["x2"]))
    result["T"] = _sum((1, result["q1"]), (-1, result["q2"]),
                        (1, result["c12"]), (-1, result["c21"]))
    result["U"] = _sum((1, result["q1"]), (1, result["q2"]),
                        (-1, result["c12"]), (-1, result["c21"]))
    return result


def _readout(block, h=ZERO):
    s = _symbols(block)
    return _sum((1 - F(2, 5) * h, s["D"]), (F(2, 5), s["y1"]),
                (F(-2, 5), s["y2"]), (F(-1, 5), s["x1"]),
                (F(1, 5), s["x2"]))


def _arguments(h=ZERO, residual_sign=ONE):
    a, b = F(3, 4) + h, F(3, 4) - h
    bounds = ((-1, 1), (-1, 1), (-1, 1))
    coefficients = ((a, b, residual_sign / 10, 1, -1),
                    (a, b, residual_sign / 10, -1, 1))
    return bounds, coefficients, (0, 0)


def _make(arguments=None, *, budget=None, **kwargs):
    if arguments is None:
        arguments = _arguments()
    return factor.make_block(*arguments, enabled=True,
                             budget=_budget() if budget is None else budget, **kwargs)


def _snapshot(block):
    return (block.n_columns, block.column_bounds, block.eq, block.le, block.frame)


def _case(h=ZERO):
    h = F(h)
    if h not in _CASES:
        block = _make(_arguments(h))
        _SNAPSHOTS[h] = _snapshot(block)
        compiled = factor.attach(block, anchor_kind="x", tau=1,
                                 enabled=True, frame=block.frame)
        _CASES[h] = block, compiled
    return _CASES[h]


def _defect(compiled):
    s = _symbols(compiled.parent)
    return _sum((1, s["y1"]), (-1, s["y2"]),
                (-compiled.tau, s["alpha1"]), (compiled.tau, s["alpha2"]),
                (-compiled.a, s["q1"]), (compiled.a, s["c21"]),
                (-compiled.b, s["c12"]), (compiled.b, s["q2"]),
                (-1, s["v1"]), (1, s["v2"]))


def _certificate(compiled, h=ZERO):
    """A fixed exact combination of actual rows, with no solving/search."""
    block, lam, k = compiled.parent, F(2, 5), F(3, 4)
    s = _symbols(block)
    assert compiled.a == k + h and compiled.b == k - h
    assert compiled.r_bounds == (F(-1, 10), F(1, 10))
    assert compiled.e_bounds == (ZERO, ZERO)
    rows = [(lam, _lookup(compiled.additional_le,
                         _sum((1, _defect(compiled)), (-2, s["p2"]))))]
    rows.append((lam * k, _lookup(compiled.additional_le,
                 _sum((1, s["T"]), (-1, s["p2"]), (-1, s["n2"])))))
    rows.append((lam * h, _lookup(compiled.additional_le,
                 _sum((1, s["U"]), (-1, s["D"])))))
    # Four actual product facets give v1-v2 <= 1/10 on the SAME r.
    r, lo, hi = _affine(compiled.r), F(-1, 10), F(1, 10)
    mc = (
        (_sum((1, s["v1"]), (-hi, s["alpha1"])), ZERO),
        (_sum((1, s["v1"]), (-1, r), (-lo, s["alpha1"])), -lo),
        (_sum((-1, s["v2"]), (lo, s["alpha2"])), ZERO),
        (_sum((1, r), (-1, s["v2"]), (hi, s["alpha2"])), hi),
    )
    rows.extend((lam / 2, _lookup(compiled.additional_le, expression, rhs))
                for expression, rhs in mc)
    coefficients = (("p1", F(4, 5)), ("n1", F(6, 5)),
                    ("p2", F(1, 10)), ("n2", F(1, 2)))
    rows.extend((weight, _lookup(block.le, _sum((-1, s[name]))))
                for name, weight in coefficients)
    combined = _nonnegative_sum(rows)
    expected = _row(_readout(block, h), F(51, 25))
    assert combined == expected
    assert all(index < block.n_columns for index, _ in combined.terms)
    return combined, sum(weight > 0 for weight, _ in rows)


def _average(points, weights):
    assert len(points) == len(weights) and sum(weights, ZERO) == ONE
    assert all(weight >= ZERO for weight in weights)
    return tuple(sum((w * p[i] for w, p in zip(weights, points)), ZERO)
                 for i in range(len(points[0])))


def _fake(h=ZERO):
    u = F(99, 100)
    return (ZERO, ZERO, ZERO, u / 2, u / 2,
            (F(7, 10) + h / 5) * u, (F(1, 2) + h / 5) * u,
            ZERO, ZERO, ZERO, ZERO)


def _prefix_and_joint_witnesses(block, h=ZERO):
    """Fixed true-source mixtures; no candidate consumes these witnesses."""
    u, target = F(99, 100), _fake(h)
    corners = tuple(block.evaluate((x1 * u, x2 * u, ZERO), frame=block.frame)
                    for x1, x2 in ((-1, -1), (1, -1), (-1, 1), (1, 1)))
    deltas = (F(2, 5), (F(6, 5) * h) / (F(1, 2) + 2 * h))
    for j, delta in enumerate(deltas):
        mean = _average(corners, (delta, F(1, 2) - delta,
                                  F(1, 2) - delta, delta))
        # Every source, both parent outputs/bits, and this original child.
        retained = (0, 1, 2, 3, 4, 7, 8, 5 + j, 9 + j)
        assert all(mean[i] == target[i] for i in retained)
    atoms = []
    for a1, a2 in ((ONE, F(1, 2)), (ZERO, F(1, 2)), (ZERO, ZERO), (ONE, ONE)):
        x1, x2, q1, q2 = u * (2 * a1 - 1), u * (2 * a2 - 1), u * a1, u * a2
        z = (F(3, 4) + h) * x1 + (F(3, 4) - h) * x2
        g1, g2 = z + q1 - q2, z - q1 + q2
        assert g1 != 0 and g2 != 0
        point = (x1, x2, ZERO, q1, q2, max(ZERO, g1), max(ZERO, g2),
                 2 * a1 - 1, 2 * a2 - 1, F(1 if g1 > 0 else -1),
                 F(1 if g2 > 0 else -1))
        assert block.satisfied(point, frame=block.frame)
        parent_weights = ((1-a1)*(1-a2), a1*(1-a2), (1-a1)*a2, a1*a2)
        parent_mean = _average(corners, parent_weights)
        assert all(parent_mean[i] == point[i] for i in (0, 1, 2, 3, 4, 7, 8))
        atoms.append(point)
    joint_mean = _average(tuple(atoms), (F(1, 5), F(1, 5), F(3, 10), F(3, 10)))
    assert joint_mean == target
    assert F(3, 2) * u > ONE  # Actual original-source witnesses enter both tails.
    return 4, 4


def test_01_default_off_and_identity():
    with pytest.raises(factor.Rejected):
        factor.make_block(*_arguments())
    block = _make()
    with pytest.raises(factor.Rejected):
        factor.attach(block, frame=block.frame)
    with pytest.raises(factor.Rejected):
        factor.attach(block, enabled=True, frame=object())
    with pytest.raises(factor.Rejected):
        block.evaluate((0, 0, 0), frame=object())
    compiled = factor.attach(block, enabled=True, frame=block.frame)
    assert compiled.parent is block and compiled.frame is block.frame
    with pytest.raises(factor.Rejected):
        compiled.canonical_extension(block.evaluate((0, 0, 0), frame=block.frame),
                                     frame=object())
    _record(1, default_off=True, foreign_frame_rejected=True,
            declared_graph_only=True, native_or_model_binding=False)


def test_02_integer_graph_extension():
    block, compiled = _case()
    count = 0
    for source in product((-ONE, ZERO, ONE), repeat=3):
        original = block.evaluate(source, frame=block.frame)
        assert block.satisfied(original, frame=block.frame, integral=True)
        extended = compiled.canonical_extension(original, frame=block.frame)
        assert extended[:block.n_columns] == original
        assert compiled.satisfied(extended, frame=block.frame, integral=True)
        alpha1, alpha2 = (original[7] + 1) / 2, (original[8] + 1) / 2
        r = source[2] / 10
        assert extended[-4:] == (alpha1 * source[1], alpha2 * source[0], alpha1 * r, alpha2 * r)
        assert compiled.decode(extended, frame=block.frame) == source
        count += 1
    assert count == 27 and compiled.n_columns == block.n_columns + 4
    _record(2, integer_grid_states=count, all_true_products_and_inputs_retained=True,
            no_candidate_grid_enumeration=True)


def test_03_symmetric_certificate():
    block, compiled = _case()
    certificate, terms = _certificate(compiled)
    fake = _fake()
    assert block.satisfied(fake, frame=block.frame)
    value = _value(_readout(block), fake)
    assert value == F(1287, 625) and value - certificate.rhs == F(12, 625)
    assert value - F(41, 20) == F(23, 2500)
    true = block.evaluate((1, -1, 1), frame=block.frame)
    assert _value(_readout(block), true) == F(51, 25)
    assert certificate.rhs - F(41, 20) == F(-1, 100)
    assert len(compiled.additional_le) == 21
    assert sum(len(row.terms) for row in compiled.additional_le) == 76
    _record(3, upper="51/25", old_physical_readout=str(value), gap="12/625",
            nonnegative_compiled_rows=terms, all_auxiliary_extensions_excluded=True,
            added_columns=4, added_le=21, added_nnz=76, actual_network_cert=False)


def test_04_asymmetric_certificate():
    h = F(1, 100)
    block, compiled = _case(h)
    certificate, terms = _certificate(compiled, h)
    fake = _fake(h)
    assert block.satisfied(fake, frame=block.frame)
    value = _value(_readout(block, h), fake)
    assert value == F(25641, 12500) and value - certificate.rhs == F(141, 12500)
    assert value - F(41, 20) == F(4, 3125)
    true = block.evaluate((1, -1, 1), frame=block.frame)
    extended = compiled.canonical_extension(true, frame=block.frame)
    assert compiled.satisfied(extended, frame=block.frame, integral=True)
    assert _value(_readout(block, h), true) == certificate.rhs
    assert sum(len(row.terms) for row in compiled.additional_le) == 76
    _record(4, upper="51/25", old_physical_readout=str(value), gap="141/12500",
            old_next_output="4/3125", new_next_output="0",
            nonnegative_compiled_rows=terms, all_auxiliary_extensions_excluded=True,
            exact_shared_source_binding_within_declared_block=True)


def test_05_strong_prefix_witnesses():
    counts = []
    for h in (ZERO, F(1, 100)):
        block, _ = _case(h)
        counts.append(_prefix_and_joint_witnesses(block, h))
    _record(5, cases=2, true_corners_per_case=4, joint_parent_atoms_per_case=4,
            all_original_labels_match=True, real_source_clip_tails=True,
            complete_single_child_prefixes=True, joint_children_over_convex_parents=True,
            full_original_four_gate_hull_claim=False)


def test_06_amplitude_anchor_contract():
    arguments = (((-1, 1), (-1, 1), (-1, 1)),
                 ((0, 0, F(1, 10), F(7, 4), F(-1, 4)),
                  (0, 0, F(1, 10), F(-1, 4), F(7, 4))), (F(-1, 5), F(-1, 5)))
    block = _make(arguments)
    compiled = factor.attach(block, anchor_kind="q", enabled=True, frame=block.frame)
    assert (compiled.a, compiled.b) == (F(3, 4), F(3, 4))
    assert compiled.r_bounds == (F(-3, 10), F(-1, 10))
    assert len(compiled.additional_le) == 18 and compiled.capacity_rows == ()
    assert sum(len(row.terms) for row in compiled.additional_le) == 56
    for source in product((-ONE, ZERO, ONE), repeat=3):
        original = block.evaluate(source, frame=block.frame)
        extended = compiled.canonical_extension(original, frame=block.frame)
        a1, a2 = (original[7]+1)/2, (original[8]+1)/2
        assert extended[-4:-2] == (a1*original[4], a2*original[3])
        assert compiled.satisfied(extended, frame=block.frame, integral=True)
    tail = block.evaluate((1, 1, 1), frame=block.frame)
    assert _value(_affine(compiled.z), tail) == F(7, 5)
    _record(6, anchor="q", states=27, added_le=18, added_nnz=56,
            actual_clip_tail="7/5", x_only_capacities_not_borrowed=True,
            actual_model_qualification=False)


def test_07_complete_residual_payment():
    arguments = (((-1, 1),)*4,
                 ((F(1, 2), F(1, 4), 0, F(1, 8), F(23, 20), -1),
                  (F(1, 2), F(1, 4), F(1, 10), F(-1, 8), F(-19, 20), 1)),
                 (F(1, 20), 0))
    block = _make(arguments)
    compiled = factor.attach(block, enabled=True, frame=block.frame)
    assert compiled.e == factor.Affine(F(1, 40), ((2, F(-1, 20)), (3, F(1, 8)), (4, F(1, 20))))
    assert compiled.r == factor.Affine(F(1, 40), ((2, F(1, 20)), (4, F(1, 10))))
    assert compiled.e_bounds == (F(-3, 20), F(1, 4))
    assert compiled.r_bounds == (F(-1, 40), F(7, 40))
    s, delta = _symbols(block), _defect(compiled)
    assert compiled.additional_le[compiled.defect_rows[0]] == _row(
        _sum((1, delta), (-2, s["p2"])), F(1, 2))
    assert compiled.additional_le[compiled.defect_rows[1]] == _row(
        _sum((-1, delta), (-2, s["p1"])), F(3, 10))
    for source in product((-ONE, ONE), repeat=4):
        original = block.evaluate(source, frame=block.frame)
        extended = compiled.canonical_extension(original, frame=block.frame)
        assert compiled.satisfied(extended, frame=block.frame, integral=True)
        assert _value(_affine(compiled.e), original) == (
            F(1, 40)-source[2]/20+source[3]/8+max(ZERO, source[0])/20)
    _record(7, complete_e_bounds=["-3/20", "1/4"], complete_r_bounds=["-1/40", "7/40"],
            bias_skip_and_unmatched_parent_terms_paid=True, integer_states=16)


def test_08_unsupported_guard_rejected():
    cases = (
        ("x", (((-1, 1), (-1, 1)), ((2, 0, 1, -1), (2, 0, -1, 1)), (0, 0))),
        ("q", (((-1, 1), (-1, 1)), ((0, 0, 3, -1), (0, 0, 1, 1)), (0, 0))),
    )
    for anchor, arguments in cases:
        block = _make(arguments)
        before = _snapshot(block)
        with pytest.raises(factor.Rejected):
            factor.attach(block, anchor_kind=anchor, enabled=True, frame=block.frame)
        assert _snapshot(block) == before
        true = block.evaluate((1, -1), frame=block.frame)
        assert true[4] - true[5] == 2  # False alias would require 3, p1=p2=0.
    _record(8, rejected_certificates=2, unsupported_true_state_not_cut=True,
            no_failed_anchor_fallback=True, original_state_unchanged=True)


def test_09_original_zero_labels():
    block, compiled = _case()
    count = 0
    for phases in product((-1, 1), repeat=4):
        original = block.evaluate((0, 0, 0), phases=phases, frame=block.frame)
        assert original[7:] == tuple(F(v) for v in phases)
        extended = compiled.canonical_extension(original, frame=block.frame)
        assert extended[-4:] == (ZERO,)*4
        assert compiled.satisfied(extended, frame=block.frame, integral=True)
        count += 1
    assert count == 16
    with pytest.raises(factor.Rejected):
        block.evaluate((1, -1, 0), phases=(-1, -1, 1, -1), frame=block.frame)
    with pytest.raises(factor.Rejected):
        compiled.canonical_extension(_fake(), frame=block.frame)
    _record(9, legal_zero_assignments=16, wrong_nonzero_phase_rejected=True,
            fractional_query_not_decoded_as_input=True)


def test_10_retained_source_predicates():
    eq = (factor.Row(((2, ONE),), ZERO),)
    le = (factor.Row(((0, ONE), (1, ONE), (3, F(1, 4))), F(1, 2)),)
    arguments = (((-1, 1),)*4,
                 ((F(1, 2), F(1, 2), F(1, 20), F(1, 20), 1, -1),
                  (F(1, 2), F(1, 2), F(1, 20), F(1, 20), -1, 1)), (0, 0))
    block = _make(arguments, source_kinds=("continuous", "continuous", "continuous", "binary"),
                  eq=eq, le=le)
    compiled = factor.attach(block, enabled=True, frame=block.frame)
    assert block.eq == eq and compiled.eq == block.eq
    assert block.le[:1] == le and compiled.le[:len(block.le)] == block.le
    for binary in (-ONE, ONE):
        original = block.evaluate((1, -1, 0, binary), frame=block.frame)
        extended = compiled.canonical_extension(original, frame=block.frame)
        assert compiled.decode(extended, frame=block.frame) == (ONE, -ONE, ZERO, binary)
    with pytest.raises(factor.Rejected):
        block.evaluate((F(3, 4), F(3, 4), 0, 1), frame=block.frame)
    with pytest.raises(factor.Rejected):
        block.evaluate((1, -1, F(1, 2), 1), frame=block.frame)
    fractional = (ZERO,)*8 + (F(-1),)*4
    assert block.satisfied(fractional, frame=block.frame)
    assert not block.satisfied(fractional, frame=block.frame, integral=True)
    with pytest.raises(factor.Rejected):
        compiled.canonical_extension(fractional, frame=block.frame)
    _record(10, source_eq=1, source_le=1, original_extra_binary=1,
            complete_sources_and_predicates_retained=True, invalid_decoders_rejected=True)


def test_11_difference_only_alias():
    block, compiled = _case()
    true = block.evaluate((1, 1, 1), frame=block.frame)
    extended = compiled.canonical_extension(true, frame=block.frame)
    assert compiled.satisfied(extended, frame=block.frame, integral=True)
    z = _value(_affine(compiled.z), true)
    clip = min(F(2), max(ZERO, z + 1))
    assert z == F(8, 5) and clip == 2 and z + 1 == F(13, 5)
    assert true[7] == true[8] == ONE and true[5] == true[6]
    opposite = _make(_arguments(residual_sign=-ONE))
    reverse = factor.attach(opposite, enabled=True, frame=opposite.frame)
    assert reverse.r_bounds == compiled.r_bounds and reverse.r != compiled.r
    left = compiled.canonical_extension(block.evaluate((1, -1, 1), frame=block.frame), frame=block.frame)
    right = reverse.canonical_extension(opposite.evaluate((1, -1, 1), frame=opposite.frame), frame=opposite.frame)
    assert left[-2:] == (F(1, 10), ZERO) and right[-2:] == (F(-1, 10), ZERO)
    assert reverse.parent is opposite and reverse.frame is not compiled.frame
    _record(11, true_clip="2", incorrect_individual_alias="13/5",
            only_common_difference_eliminated=True, equal_ranges_not_equal_sources=True)


def test_12_exact_arithmetic_rejection():
    invalid = (
        (((F(-1, 5), F(4, 5)), (-1, 1)), ((0, 0, 1, -1), (0, 0, -1, 1)), (0, 0)),
        (((-1, 1), (-1, 1)), ((True, 0, 1, -1), (0, 0, -1, 1)), (0, 0)),
        (((-1, 1), (-1, 1)), ((0.5, 0, 1, -1), (0, 0, -1, 1)), (0, 0)),
        (((-1, 1), (-1, 1)), ((0, 0, 1), (0, 0, -1, 1)), (0, 0)),
        (((-1, 1), (-1, 1)), ((0, 0, 1, -1), (0, 0, -1, 1)), (False, 0)),
    )
    for arguments in invalid:
        with pytest.raises(factor.Rejected):
            _make(arguments, budget=factor.Budget())
    block, compiled = _case()
    for tau in (0, -1, True, 0.5):
        with pytest.raises(factor.Rejected):
            factor.attach(block, tau=tau, enabled=True, frame=block.frame)
    with pytest.raises(factor.Rejected):
        factor.attach(block, anchor_kind="choose_by_margin", enabled=True, frame=block.frame)
    with pytest.raises(factor.Rejected):
        _make(eq=(factor.Row(((100, ONE),), ZERO),), budget=factor.Budget())
    assert all(isinstance(a, F) for row in compiled.le for _, a in row.terms)
    assert all(isinstance(row.rhs, F) for row in compiled.eq + compiled.le)
    _record(12, malformed_graphs=5, invalid_tau=4, no_float_or_bool_coercion=True,
            non_zero_preserving_normalization_rejected=True, exact_fraction_rows=True)


def test_13_sticky_resource_limits():
    baseline_budget = factor.Budget()
    reference = _make(budget=baseline_budget)
    build_work = baseline_budget.work
    limited = factor.Budget(max_work=build_work + 1)
    block = _make(budget=limited)
    assert limited.work == build_work
    with pytest.raises(factor.Rejected):
        factor.attach(block, enabled=True, frame=block.frame)
    assert limited.failed
    spent = (limited.work, limited.entries)
    with pytest.raises(factor.Rejected):
        block.evaluate((0, 0, 0), frame=block.frame)
    assert (limited.work, limited.entries) == spent
    bits = factor.Budget(max_bits=512)
    bit_block = _make(budget=bits)
    with pytest.raises(factor.Rejected):
        factor.attach(bit_block, tau=1 << 512, enabled=True, frame=bit_block.frame)
    assert bits.failed
    bit_spent = (bits.work, bits.entries)
    with pytest.raises(factor.Rejected):
        factor.attach(bit_block, enabled=True, frame=bit_block.frame)
    assert (bits.work, bits.entries) == bit_spent
    assert reference.budget is baseline_budget
    _record(13, deterministic_build_work=build_work, shared_sticky_budget=True,
            hard_bit_limit=512, no_precision_or_budget_rescue=True,
            exhausted_work=spent[0], exhausted_entries=spent[1], bit_work=bit_spent[0])


def test_14_shared_state_immutable():
    block, compiled = _case()
    before = _snapshot(block)
    work_before, entries_before = block.budget.work, block.budget.entries
    original = block.evaluate((F(1, 3), F(-2, 5), F(1, 7)), frame=block.frame)
    extended = compiled.canonical_extension(original, frame=block.frame)
    assert compiled.satisfied(extended, frame=block.frame, integral=True)
    assert compiled.parent is block and compiled.budget is block.budget is _budget()
    assert _snapshot(block) == before == _SNAPSHOTS[ZERO]
    assert compiled.eq == block.eq and compiled.le[:len(block.le)] == block.le
    assert compiled.n_columns == block.n_columns + 4
    assert tuple(p.column for p in compiled.products) == tuple(range(block.n_columns, block.n_columns + 4))
    assert block.budget.work > work_before and block.budget.entries >= entries_before
    original_nnz = sum(len(row.terms) for row in block.eq + block.le)
    total_nnz = sum(len(row.terms) for row in compiled.eq + compiled.le)
    assert (block.n_columns, len(block.eq), len(block.le), original_nnz) == (11, 0, 38, 70)
    assert (compiled.n_columns, len(compiled.eq), len(compiled.le), total_nnz) == (15, 0, 59, 146)
    assert len(block.binary_columns) == 4
    _record(14, unchanged_original_graph=True, all_original_columns_preserved=True,
            common_budget_for_construction_extension_and_checking=True,
            original_columns=block.n_columns, total_columns=compiled.n_columns,
            original_eq=len(block.eq), total_eq=len(compiled.eq),
            original_le=len(block.le), total_le=len(compiled.le),
            original_nnz=original_nnz, total_nnz=total_nnz,
            original_binary_columns=len(block.binary_columns),
            count_scope="logical sparse rows; not full physical storage")


def test_15_registered_parameter_family():
    values = []
    for h in (F(1, 200), F(1, 100), F(1, 50)):
        assert ZERO < h < F(4, 165)
        block, compiled = _case(h)
        certificate, _ = _certificate(compiled, h)
        _prefix_and_joint_witnesses(block, h)
        fake_value = _value(_readout(block, h), _fake(h))
        assert fake_value == F(99, 100) * (F(52, 25) - 4 * h / 5)
        assert fake_value > certificate.rhs
        if h < F(23, 1980):
            assert fake_value > F(41, 20)
        true = block.evaluate((1, -1, 1), frame=block.frame)
        assert _value(_readout(block, h), true) == certificate.rhs
        values.append({"h": str(h), "false_F": str(fake_value),
                       "gap": str(fake_value - certificate.rhs)})
    _record(15, fixed_asymmetric_cases=values, no_parameter_or_query_search=True,
            source_coefficient_equality_not_required=True,
            arbitrary_conv_applicability_claim=False)


def test_16_complete_record():
    for h, (block, compiled) in _CASES.items():
        assert _snapshot(block) == _SNAPSHOTS[h]
        assert compiled.parent is block and compiled.frame is block.frame
        assert compiled.budget is _budget()
    assert not _budget().failed
    _record(16, immutable_cached_blocks=len(_CASES), lifetime_work=_budget().work,
            lifetime_entries=_budget().entries,
            cost_scope="shared logical Budget; not native/GPU/complete physical qualification")
    assert tuple(_EVIDENCE) == _NAMES
    _record_file("summary.json", {
        "component": "phase_supported_reference", "test_count": 16,
        "records": _EVIDENCE, "mathematical_component_only": True,
        "candidate_uses_geometry_or_phase_search": False,
        "candidate_uses_support_or_solver": False,
        "ordinary_model_executed": False, "gpu_executed": False,
        "actual_model_binding_qualified": False, "native_HZ_admitted": False,
        "validated_adv": False, "formal_score_changed": False,
        "new_abstract_domain_qualified": False, "publication_novelty_claim": False,
    })
