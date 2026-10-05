"""Twenty fixed tests of the owned, same-H D243 mathematical component.

The rational points and mixtures are test witnesses, never candidate input
selection or bounds.  The precision proof combines stored inequalities with
stored source equations; it excludes every auxiliary extension, without an LP.
No import, compilation, collection, or execution is allowed before freeze.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d243_operator_source_relation_20261005 import source_relation as sr


ZERO, ONE = F(0), F(1)
RUN = (Path(__file__).resolve().parents[2]
       / "results/d243_operator_source_relation_20261005_v1")
_NAMES = (
    "default_off_and_frame_identity", "append_seal_and_lineage",
    "affine_add_and_full_input_decoder", "conv_complete_population_and_padding",
    "two_layer_padding_boundary", "raw_bn_enclosure_and_shared_error",
    "same_h_symmetric_certificate", "same_h_asymmetric_certificate",
    "strong_reference_witnesses", "complete_residual_and_derived_tau",
    "same_range_distinct_source_identity", "whole_h_integer_extensions",
    "original_zero_phase_labels", "binary_source_and_retained_predicates",
    "difference_only_alias", "unsupported_contract_rejected",
    "exact_arithmetic_and_mutation_rejection", "sticky_shared_budget",
    "complete_rows_and_cost", "immutable_state_and_summary",
)
_EVIDENCE, _CASES, _SNAPSHOTS = {}, {}, {}
_BUDGET = None


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    """Only evidence writing is relocatable by an inherited test runner."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = sr.Budget()
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
    return sr.Row(tuple(sorted(terms.items())), F(rhs) - constant)


def _value(expression, values):
    constant, terms = expression
    return constant + sum((a * values[i] for i, a in terms.items()), ZERO)


def _lookup(rows, expression, rhs=0):
    expected = _row(expression, rhs)
    matches = [row for row in rows if row == expected]
    assert matches, (expected, "required stored row missing")
    return matches[0]


def _eq_lookup(rows, expression, rhs=0):
    expected = _row(expression, rhs)
    negative = sr.Row(tuple((i, -a) for i, a in expected.terms), -expected.rhs)
    for row in rows:
        if row == expected:
            return ONE, row
        if row == negative:
            return -ONE, row
    raise AssertionError((expected, "required source equation missing"))


def _combine(weighted, *, nonnegative):
    rhs, terms = ZERO, {}
    for weight, row in weighted:
        weight = F(weight)
        if nonnegative:
            assert weight >= ZERO
        rhs += weight * row.rhs
        for index, amount in row.terms:
            terms[index] = terms.get(index, ZERO) + weight * amount
    return sr.Row(tuple(sorted((i, a) for i, a in terms.items() if a)), rhs)


def _average(points, weights):
    assert len(points) == len(weights) and sum(weights, ZERO) == ONE
    assert all(w >= ZERO for w in weights)
    return tuple(sum((w * p[i] for w, p in zip(weights, points)), ZERO)
                 for i in range(len(points[0])))


def _source(bounds=((-1, 1),) * 3, *, budget=None, **kwargs):
    return sr.Source(bounds, enabled=True,
                     budget=_budget() if budget is None else budget, **kwargs)


def _build(h=ZERO, *, budget=None):
    """Full declared graph, including aliases, skip, and terminal readout EQ."""
    h, lam = F(h), F(2, 5)
    source = _source(budget=budget)
    x1, x2, t = source.inputs
    f1 = source.affine(((x1, ONE),))
    f2 = source.affine(((x2, ONE),))
    parents = (source.relu(f1), source.relu(f2))
    a, b = F(3, 4) + h, F(3, 4) - h
    z = source.affine(((x1, a), (x2, b), (t, F(1, 10))))
    d = source.affine(((parents[0].q, ONE), (parents[1].q, -ONE)))
    g1 = source.add(z, d)
    g2 = source.affine(((z, ONE), (d, -ONE)))
    children = (source.relu(g1), source.relu(g2))
    skip = source.add(z, parents[0].q)
    readout = source.affine(((parents[0].q, 2 * (1-lam*h)),
                             (parents[1].q, 2 * (1-lam*h)),
                             (x1, -(1-lam*h)-lam/2),
                             (x2, -(1-lam*h)+lam/2),
                             (children[0].q, lam), (children[1].q, -lam)))
    state = source.seal()
    return dict(h=h, source=source, H=state, x=(x1, x2, t),
                f=(f1, f2), parents=parents, z=z, d=d, g=(g1, g2),
                children=children, skip=skip, readout=readout)


def _attach(case):
    state = case["H"]
    return sr.attach(state, parents=case["parents"], children=case["children"],
                     enabled=True, frame=state.frame)


def _snapshot(state):
    material = state.materialize(frame=state.frame)
    return (state.frame, state.n_columns, state.column_bounds,
            state.binary_columns, state.input_columns, state.error_columns,
            state.definitions, material.eq, material.le)


def _case(h=ZERO):
    h = F(h)
    if h not in _CASES:
        case = _build(h)
        _SNAPSHOTS[h] = _snapshot(case["H"])
        case["R"] = _attach(case)
        _CASES[h] = case
    return _CASES[h]


def _symbols(case):
    relation, state = case["R"], case["H"]
    x1, x2, q1, q2 = map(_affine, relation.normalized)
    result = dict(x1=x1, x2=x2, q1=q1, q2=q2,
                  y1=_col(case["children"][0].q.column),
                  y2=_col(case["children"][1].q.column))
    for i, gate in enumerate(case["parents"], 1):
        result["alpha%d" % i] = _expr(F(1, 2), ((gate.phase_column, F(1, 2)),))
    for j, name in enumerate(("c12", "c21", "v1", "v2")):
        result[name] = _col(state.n_columns+j)
    for i in (1, 2):
        result["p%d" % i] = _sum((1, result["alpha%d" % i]),
                                 (-1, result["q%d" % i]))
        result["n%d" % i] = _sum((1, _expr(1)),
                                 (-1, result["alpha%d" % i]),
                                 (-1, result["q%d" % i]), (1, result["x%d" % i]))
    result["D"] = _sum((2, q1), (2, q2), (-1, x1), (-1, x2))
    result["T"] = _sum((1, q1), (-1, q2), (1, result["c12"]), (-1, result["c21"]))
    result["U"] = _sum((1, q1), (1, q2), (-1, result["c12"]), (-1, result["c21"]))
    return result


def _certificate(case):
    """Nonnegative stored LE + actual source EQ -> ONE terminal column."""
    state, relation, h = case["H"], case["R"], case["h"]
    material = state.materialize(frame=state.frame)
    s, lam, k = _symbols(case), F(2, 5), F(3, 4)
    assert relation.tau == ONE and (relation.a, relation.b) == (k+h, k-h)
    assert relation.r_bounds == (F(-1, 10), F(1, 10))
    assert relation.e_bounds == (ZERO, ZERO)
    delta = _sum((1, s["y1"]), (-1, s["y2"]), (-1, _affine(relation.K)))
    weighted = [(lam, _lookup(relation.additional_le,
                             _sum((1, delta), (-2, s["p2"]))))]
    weighted.append((lam*k, _lookup(relation.additional_le,
                     _sum((1, s["T"]), (-1, s["p2"]), (-1, s["n2"])))))
    weighted.append((lam*h, _lookup(relation.additional_le,
                     _sum((1, s["U"]), (-1, s["D"])))))
    r, lo, hi = _affine(relation.r), F(-1, 10), F(1, 10)
    mc = ((_sum((1, s["v1"]), (-hi, s["alpha1"])), ZERO),
          (_sum((1, s["v1"]), (-1, r), (-lo, s["alpha1"])), -lo),
          (_sum((-1, s["v2"]), (lo, s["alpha2"])), ZERO),
          (_sum((1, r), (-1, s["v2"]), (hi, s["alpha2"])), hi))
    weighted.extend((lam/2, _lookup(relation.additional_le, ex, rhs)) for ex, rhs in mc)
    for name, weight in (("p1", F(4, 5)), ("n1", F(6, 5)),
                         ("p2", F(1, 10)), ("n2", F(1, 2))):
        weighted.append((weight, _lookup(material.le, _sum((-1, s[name])))))
    combined = _combine(weighted, nonnegative=True)
    physical = _sum((1-lam*h, s["D"]), (lam, s["y1"]), (-lam, s["y2"]),
                    (-lam/2, s["x1"]), (lam/2, s["x2"]))
    assert combined == _row(physical, F(51, 25))
    equalities = []
    for i, coefficient in enumerate((-(1-lam*h)-lam/2, -(1-lam*h)+lam/2)):
        # The normalized source is an original identity alias, not a fresh box.
        alias = _sum((1, s["x%d" % (i+1)]), (-1, _col(case["x"][i].column)))
        if alias != _expr():
            orientation, eq = _eq_lookup(material.eq, alias)
            equalities.append((-coefficient*orientation, eq))
    original_readout = _sum((1-lam*h, _sum((2, s["q1"]), (2, s["q2"]),
                             (-1, _col(case["x"][0].column)),
                             (-1, _col(case["x"][1].column)))),
                            (lam, s["y1"]), (-lam, s["y2"]),
                            (-lam/2, _col(case["x"][0].column)),
                            (lam/2, _col(case["x"][1].column)))
    orientation, terminal_eq = _eq_lookup(material.eq,
        _sum((1, _col(case["readout"].column)), (-1, original_readout)))
    equalities.append((orientation, terminal_eq))
    final = _combine([(ONE, combined)] + equalities, nonnegative=False)
    assert final == sr.Row(((case["readout"].column, ONE),), F(51, 25))
    assert all(index < state.n_columns for index, _ in final.terms)
    return final, sum(w > 0 for w, _ in weighted), len(equalities)


def _relaxed(case, inputs, qs, ys, phases):
    """Independent fixed-fixture evaluator, including every original slot."""
    values = {}
    for handle, value in zip(case["x"], inputs):
        values[handle.column] = F(value)
    for i, gate in enumerate(case["parents"]):
        values[case["f"][i].column] = F(inputs[i])
        values[gate.q.column] = F(qs[i])
        values[gate.phase_column] = F(phases[i])
    h, lam = case["h"], F(2, 5)
    z = (F(3, 4)+h)*inputs[0] + (F(3, 4)-h)*inputs[1] + F(1, 10)*inputs[2]
    d = qs[0]-qs[1]
    values[case["z"].column], values[case["d"].column] = z, d
    for i, gate in enumerate(case["children"]):
        values[case["g"][i].column] = z + (d if i == 0 else -d)
        values[gate.q.column], values[gate.phase_column] = F(ys[i]), F(phases[i+2])
    values[case["skip"].column] = z+qs[0]
    D = 2*qs[0]+2*qs[1]-inputs[0]-inputs[1]
    values[case["readout"].column] = (1-lam*h)*D+lam*(ys[0]-ys[1])-lam/2*(inputs[0]-inputs[1])
    assert set(values) == set(range(case["H"].n_columns))
    return tuple(values[i] for i in range(case["H"].n_columns))


def _fake(case):
    u, h = F(99, 100), case["h"]
    return _relaxed(case, (ZERO, ZERO, ZERO), (u/2, u/2),
                    ((F(7, 10)+h/5)*u, (F(1, 2)+h/5)*u), (ZERO,)*4)


def _witnesses(case):
    state, u, h = case["H"], F(99, 100), case["h"]
    target = _fake(case)
    corners = tuple(state.canonical((x*u, y*u, ZERO), frame=state.frame)
                    for x, y in ((-1, -1), (1, -1), (-1, 1), (1, 1)))
    base = tuple(state.input_columns) + tuple(g.q.column for g in case["parents"]) + tuple(g.phase_column for g in case["parents"])
    base += tuple(v.column for v in case["f"]) + (case["z"].column, case["d"].column, case["skip"].column)
    for j, delta in enumerate((F(2, 5), (F(6, 5)*h)/(F(1, 2)+2*h))):
        mean = _average(corners, (delta, F(1, 2)-delta, F(1, 2)-delta, delta))
        gate = case["children"][j]
        retained = base + (gate.f.column, gate.q.column, gate.phase_column)
        assert all(mean[i] == target[i] for i in retained)
    atoms = []
    for a1, a2 in ((ONE, F(1, 2)), (ZERO, F(1, 2)), (ZERO, ZERO), (ONE, ONE)):
        inputs, qs = (u*(2*a1-1), u*(2*a2-1), ZERO), (u*a1, u*a2)
        z = (F(3, 4)+h)*inputs[0] + (F(3, 4)-h)*inputs[1]
        guards = (z+qs[0]-qs[1], z-qs[0]+qs[1])
        assert all(g != 0 for g in guards)
        point = _relaxed(case, inputs, qs, tuple(max(ZERO, g) for g in guards),
                         (2*a1-1, 2*a2-1) + tuple(F(1 if g > 0 else -1) for g in guards))
        assert state.satisfied(point, frame=state.frame)
        parent_mean = _average(corners, ((1-a1)*(1-a2), a1*(1-a2), (1-a1)*a2, a1*a2))
        assert all(parent_mean[i] == point[i] for i in base)
        atoms.append(point)
    assert _average(atoms, (F(1, 5), F(1, 5), F(3, 10), F(3, 10))) == target
    return len(corners), len(atoms)


def test_01_default_off_and_frame_identity():
    with pytest.raises(sr.Rejected):
        sr.Source(((-1, 1),))
    private = _build(budget=sr.Budget())
    with pytest.raises(sr.Rejected):
        sr.attach(private["H"], parents=private["parents"], children=private["children"], frame=private["H"].frame)
    private = _build(budget=sr.Budget())
    with pytest.raises(sr.Rejected):
        private["H"].canonical((ZERO,)*3, frame=object())
    private = _build(budget=sr.Budget())
    relation = _attach(private)
    with pytest.raises(sr.Rejected):
        relation.materialize(frame=object())
    left, right = _source(budget=sr.Budget()), _source(budget=sr.Budget())
    with pytest.raises(sr.Rejected):
        left.affine(((right.inputs[0], ONE),))
    _record(1, default_off=True, frame_identity_required=True, foreign_handle_rejected=True)


def test_02_append_seal_and_lineage():
    source = _source(((-1, 1), (-1, 1)))
    x, y = source.inputs
    first = source.add(x, y)
    second = source.affine(((first, F(2, 3)), (x, F(-1, 5))), F(1, 7))
    state = source.seal()
    point = state.canonical((F(1, 3), F(-2, 5)), frame=state.frame)
    assert point[first.column] == F(-1, 15)
    assert point[second.column] == F(-2, 45)-F(1, 15)+F(1, 7)
    assert state.satisfied(point, frame=state.frame, integral=True)
    old = _snapshot(state)
    assert old == _snapshot(state)
    private = _source(budget=sr.Budget())
    published = private.seal()
    with pytest.raises(sr.Rejected):
        private.seal()
    assert not private.budget.failed
    point = published.canonical((ZERO,)*3, frame=published.frame)
    assert published.satisfied(point, frame=published.frame, integral=True)
    private = _source(budget=sr.Budget())
    own = private.inputs[0]
    published = private.seal()
    with pytest.raises(sr.Rejected):
        private.affine(((own, ONE),))
    assert not private.budget.failed
    point = published.canonical((ZERO,)*3, frame=published.frame)
    assert published.satisfied(point, frame=published.frame, integral=True)
    _record(2, append_only=True, seal_once=True, complete_lineage=True,
            published_h_survives_invalid_append=True)


def test_03_affine_add_and_full_input_decoder():
    source = _source(((F(-3, 2), F(5, 4)), (F(1, 3), F(7, 6)), (-1, 1)))
    x, y, z = source.inputs
    a = source.affine(((x, F(2, 3)), (y, F(-3, 5))), F(1, 7))
    b = source.add(a, z)
    unused = source.affine(((x, F(-4, 7)), (z, F(1, 9))), F(2, 5))
    state = source.seal()
    inputs = (F(-1, 4), F(5, 6), F(2, 7))
    point = state.canonical(inputs, frame=state.frame)
    assert point[a.column] == F(2, 3)*inputs[0]-F(3, 5)*inputs[1]+F(1, 7)
    assert point[b.column] == point[a.column]+inputs[2]
    assert point[unused.column] == -F(4, 7)*inputs[0]+inputs[2]/9+F(2, 5)
    assert state.decode(point, frame=state.frame) == inputs
    material = state.materialize(frame=state.frame)
    assert all(sum((w*point[i] for i, w in row.terms), ZERO) == row.rhs for row in material.eq)
    _record(3, every_original_input_decoded=True, unselected_consumer_retained=True,
            exact_eq_count=len(material.eq))


def test_04_conv_complete_population_and_padding():
    source = _source(((-1, 1),)*6)
    tensor = source.tensor((1, 2, 3), source.inputs)
    kernel = ((((ONE, F(-1, 2)), (F(1, 3), F(2, 3))),),)
    output = source.conv2d(tensor, kernel, (F(1, 5),), padding=(0, 1, 1, 0))
    state = source.seal()
    inputs = (F(1, 2), F(-1, 3), F(2, 5), F(-3, 4), F(1, 7), F(4, 5))
    point = state.canonical(inputs, frame=state.frame)
    assert output.shape == (1, 2, 3) and len(output.values) == 6
    expected = []
    # Independent literal all-position geometry; no test value is given to Conv.
    for row in range(2):
        for col in range(3):
            value = F(1, 5)
            for kr in range(2):
                for kc in range(2):
                    ir, ic = row+kr, col+kc-1
                    if 0 <= ir < 2 and 0 <= ic < 3:
                        value += kernel[0][0][kr][kc]*inputs[3*ir+ic]
            expected.append(value)
    assert tuple(point[v.column] for v in output.values) == tuple(expected)
    assert state.n_eq == 6 and state.satisfied(point, frame=state.frame, integral=True)
    assert all(state.eq_row(i, frame=state.frame) == state.materialize(frame=state.frame).eq[i]
               for i in range(state.n_eq))
    grouped_source = _source(((-1, 1),)*10)
    grouped_input = grouped_source.tensor((2, 1, 5), grouped_source.inputs)
    grouped_kernel = ((((ONE, F(-1, 2)),),), (((F(3, 4), F(5, 4)),),))
    grouped = grouped_source.conv2d(grouped_input, grouped_kernel, groups=2,
                                    stride=(1, 2), dilation=(1, 2))
    grouped_h = grouped_source.seal()
    grouped_values = tuple(F((-1 if i % 2 else 1)*(i+1), 10) for i in range(10))
    grouped_point = grouped_h.canonical(grouped_values, frame=grouped_h.frame)
    assert grouped.shape == (2, 1, 2) and len(grouped.values) == 4
    grouped_expected = (grouped_values[0]-grouped_values[2]/2,
                        grouped_values[2]-grouped_values[4]/2,
                        F(3, 4)*grouped_values[5]+F(5, 4)*grouped_values[7],
                        F(3, 4)*grouped_values[7]+F(5, 4)*grouped_values[9])
    assert tuple(grouped_point[v.column] for v in grouped.values) == grouped_expected
    assert grouped_h.n_eq == 4 and grouped_h.satisfied(grouped_point, frame=grouped_h.frame, integral=True)
    _record(4, declared_outputs=6, checked_outputs=6, complete_kernel_geometry=True,
            grouped_stride_dilation_outputs=4, local_window_selection=False)


def test_05_two_layer_padding_boundary():
    source = _source(((-1, 1),)*2)
    first = source.tensor((1, 1, 2), source.inputs)
    kernel = ((((ONE, ONE, ONE),),),)
    middle = source.conv2d(first, kernel, padding=(0, 1, 0, 1))
    last = source.conv2d(middle, kernel, padding=(0, 1, 0, 1))
    state = source.seal()
    inputs = (F(1, 5), F(-2, 5))
    point = state.canonical(inputs, frame=state.frame)
    assert tuple(point[v.column] for v in middle.values) == (F(-1, 5),)*2
    assert tuple(point[v.column] for v in last.values) == (F(-2, 5),)*2
    assert point[last.values[0].column] != 3*inputs[0]+2*inputs[1]
    assert state.n_eq == 4
    _record(5, two_layer_boundary="2*x0+2*x1", naive_shared_five_tap_rejected=True)


def test_06_raw_bn_enclosure_and_shared_error():
    source = _source(((-1, 1),)*2)
    tensor = source.tensor((1, 1, 2), source.inputs)
    out = source.bn2d(tensor, mean=(F(1, 4),), beta=(F(-1, 8),),
                      scale_intervals=((F(3, 4), F(5, 4)),), ahat=(ONE,))
    skip = source.add(out.values[0], out.values[0])
    cancellation = source.affine(((out.values[0], ONE), (out.values[0], -ONE)))
    state = source.seal()
    assert len(state.error_columns) == 2 and len(set(state.error_columns)) == 2
    inputs, errors, actual_a = (F(1, 2), F(-1, 2)), (F(1, 10), F(-3, 10)), F(9, 8)
    point = state.canonical(inputs, errors=errors, frame=state.frame)
    for i, value in enumerate(out.values):
        assert point[value.column] == actual_a*(inputs[i]-F(1, 4))-F(1, 8)
    assert point[skip.column] == 2*point[out.values[0].column]
    assert point[cancellation.column] == 0
    assert state.decode(point, frame=state.frame) == inputs
    # A common BN output used by two children cancels in d but remains in r.
    source = _source()
    x1, x2, t = source.inputs
    parents = (source.relu(x1), source.relu(x2))
    residual = source.bn(t, mean=ZERO, beta=ZERO,
                         scale_interval=(F(1, 20), F(3, 20)), ahat=F(1, 10))
    z = source.affine(((x1, F(3, 4)), (x2, F(3, 4)), (residual, ONE)))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, sign),
                                               (parents[1].q, -sign)))) for sign in (ONE, -ONE))
    state = source.seal()
    relation = sr.attach(state, parents=parents, children=children, enabled=True, frame=state.frame)
    assert len(state.error_columns) == 1
    assert relation.e_bounds == (ZERO, ZERO)
    assert dict(relation.r.terms)[state.error_columns[0]] == F(1, 20)
    point = state.canonical((F(1, 2), F(-1, 2), F(1, 3)), errors=(F(2, 5),), frame=state.frame)
    assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(6, declared_scale_interval_not_onnx_sqrt=True, per_scalar_errors=2,
            shared_consumer_error=True, full_bn_parameter_correlation_claimed=False)


def test_07_same_h_symmetric_certificate():
    case = _case()
    row, le_count, eq_count = _certificate(case)
    fake = _fake(case)
    assert case["H"].satisfied(fake, frame=case["H"].frame)
    assert fake[case["readout"].column] == F(1287, 625) > row.rhs
    point = case["H"].canonical((ONE, -ONE, ONE), frame=case["H"].frame)
    assert point[case["readout"].column] == row.rhs == F(51, 25)
    extended = case["R"].canonical_extension(point, frame=case["H"].frame)
    assert case["R"].satisfied(extended, frame=case["H"].frame, integral=True)
    _record(7, physical_upper="51/25", old_fractional_readout="1287/625",
            positive_le_terms=le_count, stored_eq_terms=eq_count,
            every_auxiliary_extension_excluded=True)


def test_08_same_h_asymmetric_certificate():
    case = _case(F(1, 100))
    row, le_count, eq_count = _certificate(case)
    fake = _fake(case)
    assert case["H"].satisfied(fake, frame=case["H"].frame)
    assert fake[case["readout"].column] == F(25641, 12500)
    assert fake[case["readout"].column]-row.rhs == F(141, 12500)
    point = case["H"].canonical((ONE, -ONE, ONE), frame=case["H"].frame)
    assert point[case["readout"].column] == F(51, 25)
    assert row.terms == ((case["readout"].column, ONE),)
    _record(8, physical_upper="51/25", old_fractional_readout="25641/12500",
            strict_gap="141/12500", positive_le_terms=le_count,
            stored_eq_terms=eq_count, no_auxiliary_guess=True)


def test_09_strong_reference_witnesses():
    counts = tuple(_witnesses(_case(h)) for h in (ZERO, F(1, 100)))
    assert counts == ((4, 4), (4, 4))
    _record(9, full_labelled_single_child_prefixes=True,
            joint_children_over_convexified_full_parents=True,
            common_original_sources_and_bits=True,
            all_cross_layer_rlt_or_full_network_hull_claimed=False)


def test_10_complete_residual_and_derived_tau():
    source = _source(((-1, 1),)*4)
    x1, x2, t, u = source.inputs
    parents = (source.relu(x1), source.relu(x2))
    g1 = source.affine(((x1, F(1, 2)), (x2, F(1, 4)), (u, F(1, 8)),
                        (parents[0].q, F(23, 20)), (parents[1].q, -ONE)), F(1, 20))
    g2 = source.affine(((x1, F(1, 2)), (x2, F(1, 4)), (t, F(1, 10)),
                        (u, F(-1, 8)), (parents[0].q, F(-19, 20)), (parents[1].q, ONE)))
    children = (source.relu(g1), source.relu(g2))
    state = source.seal()
    relation = sr.attach(state, parents=parents, children=children, enabled=True, frame=state.frame)
    assert relation.tau == F(41, 40)
    assert relation.r_bounds == (F(-1, 40), F(7, 40))
    assert relation.e_bounds == (F(-3, 20), F(1, 4))
    for inputs in ((F(1, 2), F(-1, 3), F(2, 5), F(-3, 4)), (ONE, ONE, -ONE, ONE), (-ONE, -ONE, ONE, -ONE)):
        point = state.canonical(inputs, frame=state.frame)
        q1, q2 = point[parents[0].q.column], point[parents[1].q.column]
        assert _value(_affine(relation.r), point) == F(1, 40)+inputs[2]/20+q1/10
        assert _value(_affine(relation.e), point) == F(1, 40)-inputs[2]/20+inputs[3]/8+(q1+q2)/40
        assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(10, derived_tau="41/40", e_bounds=["-3/20", "1/4"],
            r_bounds=["-1/40", "7/40"], complete_q_bias_and_source_remainders=True)


def test_11_same_range_distinct_source_identity():
    source = _source()
    x1, x2, t = source.inputs
    parents = (source.relu(x1), source.relu(x2))
    groups = []
    for sign in (ONE, -ONE):
        z = source.affine(((x1, F(3, 4)), (x2, F(3, 4)), (t, sign/10)))
        groups.append(tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s),
                                                       (parents[1].q, -s)))) for s in (ONE, -ONE)))
    state = source.seal()
    relations = tuple(sr.attach(state, parents=parents, children=children, enabled=True, frame=state.frame)
                      for children in groups)
    left, right = relations
    assert left.parent is right.parent is state and left.frame is right.frame
    assert left.r_bounds == right.r_bounds == (F(-1, 10), F(1, 10))
    assert left.r != right.r
    point = state.canonical((F(1, 2), F(-1, 2), F(3, 5)), frame=state.frame)
    lex, rex = (r.canonical_extension(point, frame=state.frame) for r in relations)
    assert lex[-2] == F(3, 50) and rex[-2] == F(-3, 50)
    assert all(r.satisfied(ex, frame=state.frame, integral=True) for r, ex in zip(relations, (lex, rex)))
    _record(11, same_h=True, same_range_not_same_source=True, both_consumer_groups_retained=True)


def test_12_whole_h_integer_extensions():
    case = _case()
    state, relation = case["H"], case["R"]
    checked = 0
    for inputs in product((F(-3, 4), ZERO, F(2, 3)), repeat=3):
        point = state.canonical(tuple(inputs), frame=state.frame)
        extended = relation.canonical_extension(point, frame=state.frame)
        assert extended[:state.n_columns] == point
        assert state.satisfied(point, frame=state.frame, integral=True)
        assert relation.satisfied(extended, frame=state.frame, integral=True)
        assert relation.decode(extended, frame=state.frame) == tuple(inputs)
        alpha = tuple((point[g.phase_column]+1)/2 for g in case["parents"])
        r = _value(_affine(relation.r), point)
        assert extended[-4:] == (alpha[0]*inputs[1], alpha[1]*inputs[0], alpha[0]*r, alpha[1]*r)
        checked += 1
    # Unequal asymmetric bounds: scaling does not center either ReLU zero.
    source = _source(((-2, 1), (-1, 3), (-1, 1)))
    f1, f2, t = source.inputs
    parents = (source.relu(f1), source.relu(f2))
    z = source.affine(((f1, F(3, 8)), (f2, F(1, 4)), (t, F(1, 10))))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, sign/2),
                                               (parents[1].q, -sign/3)))) for sign in (ONE, -ONE))
    scaled_h = source.seal()
    scaled = sr.attach(scaled_h, parents=parents, children=children, enabled=True, frame=scaled_h.frame)
    assert scaled.tau == ONE and scaled.a == scaled.b == F(3, 4)
    scaled_inputs = ((F(-2), F(3), ZERO), (ONE, -ONE, ONE),
                     (F(-1, 2), F(3, 4), F(-1, 2)), (ZERO, ZERO, ZERO))
    for inputs in scaled_inputs:
        point = scaled_h.canonical(inputs, frame=scaled_h.frame)
        expected = (inputs[0]/2, inputs[1]/3, max(ZERO, inputs[0])/2, max(ZERO, inputs[1])/3)
        assert tuple(_value(_affine(v), point) for v in scaled.normalized) == expected
        extended = scaled.canonical_extension(point, frame=scaled_h.frame)
        assert scaled.satisfied(extended, frame=scaled_h.frame, integral=True)
        assert scaled.decode(extended, frame=scaled_h.frame) == inputs
    _record(12, fixed_integer_points=checked, original_projection_preserved=True,
            asymmetric_parent_scale_points=len(scaled_inputs), scales=[2, 3],
            normalization_translation=False, concrete_network_adv_claimed=False)


def test_13_original_zero_phase_labels():
    case = _case()
    state, relation = case["H"], case["R"]
    bits = tuple(g.phase_column for g in case["parents"]+case["children"])
    for signs in product((-1, 1), repeat=4):
        point = state.canonical((ZERO,)*3, zero_phases=tuple(zip(bits, signs)), frame=state.frame)
        assert tuple(point[i] for i in bits) == signs
        assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    private = _build(budget=sr.Budget())
    with pytest.raises(sr.Rejected):
        private["H"].canonical((ONE, -ONE, ZERO),
            zero_phases=((private["parents"][0].phase_column, -1),), frame=private["H"].frame)
    _record(13, original_zero_label_extensions=16, nonzero_wrong_phase_rejected=True,
            virtual_clip_bits=0)


def test_14_binary_source_and_retained_predicates():
    equality = sr.Row(((2, ONE),), ZERO)
    inequality = sr.Row(((0, ONE), (1, ONE), (3, F(1, 4))), F(1, 2))
    source = _source(((-1, 1),)*4, source_kinds=("continuous", "continuous", "continuous", "binary"),
                     eq=(equality,), le=(inequality,))
    x1, x2, t, binary = source.inputs
    parents = (source.relu(x1), source.relu(x2))
    z = source.affine(((x1, F(3, 4)), (x2, F(3, 4)), (binary, F(1, 10))))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, sign),
                                               (parents[1].q, -sign)))) for sign in (ONE, -ONE))
    source.predicate(sr.Row(((z.column, ONE),), F(3, 2)))
    state = source.seal()
    relation = sr.attach(state, parents=parents, children=children, enabled=True, frame=state.frame)
    integer_points = []
    for sign in (-ONE, ONE):
        inputs = (F(1, 4), F(-1, 4), ZERO, sign)
        point = state.canonical(inputs, frame=state.frame)
        extended = relation.canonical_extension(point, frame=state.frame)
        assert relation.decode(extended, frame=state.frame) == inputs
        assert relation.satisfied(extended, frame=state.frame, integral=True)
        integer_points.append(point)
    material = relation.materialize(frame=state.frame)
    assert equality in material.eq and inequality in material.le
    assert binary.column in state.binary_columns and len(state.binary_columns) == 5
    relaxed = _average(integer_points, (F(1, 2), F(1, 2)))
    assert relaxed[binary.column] == ZERO
    assert state.satisfied(relaxed, frame=state.frame)
    assert not state.satisfied(relaxed, frame=state.frame, integral=True)
    _record(14, original_eq_and_le_retained=True, original_binary_inputs=1,
            all_input_coordinates_decoded=True)


def test_15_difference_only_alias():
    case = _case()
    state, relation = case["H"], case["R"]
    point = state.canonical((ONE, ONE, ONE), frame=state.frame)
    z = _value(_affine(relation.z), point)
    assert z == F(8, 5)
    clip = min(F(2), max(ZERO, z+1))
    assert clip == 2 and clip != z+1
    alpha = tuple((point[g.phase_column]+1)/2 for g in case["parents"])
    assert alpha == (ONE, ONE)
    assert (alpha[0]-alpha[1])*clip == (alpha[0]-alpha[1])*(z+1)
    assert alpha[0]*clip != alpha[0]*(z+1)
    assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(15, genuine_crossing_reference_tail=True,
            difference_annihilation_only=True, individual_clip_alias_forbidden=True)


def test_16_unsupported_contract_rejected():
    for mode in ("guard", "tau", "same_parent"):
        source = _source(budget=sr.Budget())
        x1, x2, t = source.inputs
        parents = (source.relu(x1), source.relu(x2))
        z = source.affine(((x1, F(2) if mode == "guard" else F(1, 2)), (t, F(1, 10))))
        sign = -ONE if mode == "tau" else ONE
        children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s*sign),
                                                   (parents[1].q, -s*sign)))) for s in (ONE, -ONE))
        state = source.seal()
        chosen = (parents[0], parents[0]) if mode == "same_parent" else parents
        before = state.budget.work
        with pytest.raises(sr.Rejected):
            sr.attach(state, parents=chosen, children=children, enabled=True, frame=state.frame)
        assert not state.budget.failed and state.budget.work > before
        point = state.canonical((ZERO,)*3, frame=state.frame)
        assert state.satisfied(point, frame=state.frame, integral=True)
    _record(16, unsupported_x_guard_rejected=True, nonpositive_tau_rejected=True,
            repeated_parent_rejected=True, fallback_menu=False,
            rejected_attachment_preserves_original_h=True)


def test_17_exact_arithmetic_and_mutation_rejection():
    for bounds in (((0.0, 1),), ((False, 1),), ((1, -1),)):
        with pytest.raises(sr.Rejected):
            sr.Source(bounds, enabled=True, budget=sr.Budget())
    source = _source(budget=sr.Budget())
    with pytest.raises(sr.Rejected):
        source.affine(((source.inputs[0], 0.5),))
    source = _source(budget=sr.Budget())
    tensor = source.tensor((1, 1, 3), source.inputs)
    with pytest.raises(sr.Rejected):
        source.conv2d(tensor, (((ONE,),),))
    private = _build(budget=sr.Budget())
    relation = _attach(private)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        relation.tau = F(2)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        private["H"].n_columns = 0
    _record(17, floats_bools_bad_geometry_rejected=True, public_frozen_state=True,
            exact_fraction_semantics=True)


def test_18_sticky_shared_budget():
    budget = sr.Budget(max_work=0)
    with pytest.raises(sr.Rejected):
        sr.Source(((-1, 1),), enabled=True, budget=budget)
    assert budget.failed
    with pytest.raises(sr.Rejected):
        sr.Source(((-1, 1),), enabled=True, budget=budget)
    budget = sr.Budget()
    with pytest.raises(sr.Rejected):
        sr.Source(((ZERO, F(1 << 513)),), enabled=True, budget=budget)
    assert budget.failed
    budget = sr.Budget()
    unfinished = _source(((-1, 1),)*2, budget=budget)
    tensor = unfinished.tensor((2, 1, 1), unfinished.inputs)
    with pytest.raises(sr.Rejected):
        unfinished.bn2d(tensor, mean=(ZERO, ZERO), beta=(ZERO, ZERO),
                         scale_intervals=((ONE, ONE), (F(2), ONE)))
    assert unfinished.n_columns > 2 and budget.failed
    with pytest.raises(sr.Rejected):
        unfinished.seal()
    before = _budget().work
    case = _case()
    case["H"].materialize(frame=case["H"].frame)
    case["R"].materialize(frame=case["H"].frame)
    assert _budget().work > before and case["R"].budget is case["H"].budget is _budget()
    _record(18, exhaustion_sticky=True, arithmetic_512_bit_limit=True,
            incomplete_builder_failure_sticky=True,
            full_row_access_charged=True, positive_budget_shared=True)


def test_19_complete_rows_and_cost():
    case = _case()
    state, relation = case["H"], case["R"]
    old, full = state.materialize(frame=state.frame), relation.materialize(frame=state.frame)
    assert relation.n_columns == state.n_columns+4
    assert len(relation.additional_le) == 21
    assert sum(len(row.terms) for row in relation.additional_le) == 76
    assert tuple(full.eq) == tuple(old.eq)
    assert tuple(full.le[:len(old.le)]) == tuple(old.le)
    product_bounds = tuple(row for i in range(state.n_columns, relation.n_columns)
                           for row in (sr.Row(((i, -ONE),), -relation.column_bounds[i][0]),
                                       sr.Row(((i, ONE),), relation.column_bounds[i][1])))
    assert tuple(full.le[len(old.le):len(old.le)+8]) == product_bounds
    assert tuple(full.le[len(old.le)+8:]) == relation.additional_le
    assert full.binary_columns == old.binary_columns == state.binary_columns
    assert len(full.le) == len(old.le)+29 and len(full.eq) == state.n_eq
    assert len(old.le) == state.n_le
    assert all(state.le_row(i, frame=state.frame) == old.le[i] for i in range(state.n_le))
    _record(19, original_columns=state.n_columns, full_columns=relation.n_columns,
            original_eq=len(old.eq), full_eq=len(full.eq), original_le=len(old.le), full_le=len(full.le),
            original_nnz=sum(len(row.terms) for row in old.eq+old.le),
            full_nnz=sum(len(row.terms) for row in full.eq+full.le),
            increment_columns=4, increment_le=21, increment_nnz=76,
            materialized_increment_le=29, materialized_increment_nnz=84,
            original_binary_columns=len(state.binary_columns),
            logical_sparse_cost_not_native_or_gpu_peak=True)


def test_20_immutable_state_and_summary():
    for h, case in _CASES.items():
        assert _snapshot(case["H"]) == _SNAPSHOTS[h]
        assert case["R"].parent is case["H"] and case["R"].frame is case["H"].frame
    assert not _budget().failed
    _record(20, original_state_unchanged=True, fixed_three_model_population_not_executed=True,
            no_solver_no_phase_pool=True)
    assert tuple(_EVIDENCE) == _NAMES and len(_EVIDENCE) == 20
    _record_file("summary.json", dict(
        schema="d243_operator_source_relation_v1", tests=20,
        required_tests=4189, required_test_files=223,
        mathematical_stage_only=True, same_h_declared_relation_verified=True,
        all_auxiliary_projection_certificate=True,
        source_component_qualified=False, actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        complete_physical_qualification=False, gpu_computation_completed=False,
        new_domain_qualified=False, new_capability_qualified=False,
        new_set_class=False, formal_gain=0, independent_e0_gain=0,
        new_benchmark_solves=0, inherited_tests_not_replaced=True,
        shared_work=_budget().work, shared_entries=_budget().entries,
        evidence=_EVIDENCE))
