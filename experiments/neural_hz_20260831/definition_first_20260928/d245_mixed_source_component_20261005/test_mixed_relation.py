"""Twenty frozen mathematical tests of the same-H mixed source relation.

All rational grids and convex mixtures below are independent proof witnesses,
not candidate search, model inputs, or verified adversarial examples.  The main
certificate combines stored LE and EQ, eliminating every auxiliary product.
Do not import, collect, compile, or execute this file before the main freeze.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d245_mixed_source_component_20261005 import mixed_relation as mr


ZERO, ONE = F(0), F(1)
RUN = (Path(__file__).resolve().parents[2]
       / "results/d245_mixed_source_component_20261005_v1")
_NAMES = (
    "default_off_and_frame_identity",
    "shared_source_recovery_without_direct_parent_edges",
    "stored_row_physical_projection_certificate",
    "complete_single_child_prefix_witnesses",
    "joint_children_over_convexified_parents",
    "all_qualified_old_tau_certificates",
    "fixed_integer_extensions_and_decoder",
    "original_zero_phase_labels",
    "complete_nonzero_residual_and_derived_tau",
    "same_range_distinct_source_identity",
    "asymmetric_parent_normalization",
    "bn_error_and_shared_consumers",
    "binary_source_and_retained_predicates",
    "difference_only_tail_alias",
    "rank_and_dependent_parent_rejection",
    "guard_and_invalid_input_rejection",
    "sticky_shared_budget",
    "complete_materialized_cost",
    "original_state_immutability",
    "summary_and_evidence_boundary",
)
_EVIDENCE, _CASES, _SNAPSHOTS = {}, {}, {}
_BUDGET = None


def _record(number, **values):
    name = _NAMES[number-1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    """Only the exclusive evidence writer is relocatable by inheritance."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN/name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = mr.Budget()
    return _BUDGET


def _expr(constant=0, terms=()):
    merged = {}
    for index, amount in terms:
        merged[index] = merged.get(index, ZERO)+F(amount)
    return F(constant), {i: a for i, a in merged.items() if a}


def _sum(*weighted):
    constant, terms = ZERO, {}
    for weight, (bias, entries) in weighted:
        weight = F(weight)
        constant += weight*bias
        for index, amount in entries.items():
            terms[index] = terms.get(index, ZERO)+weight*amount
    return _expr(constant, terms.items())


def _col(index):
    return _expr(0, ((index, ONE),))


def _affine(value):
    return _expr(value.constant, value.terms)


def _row(expression, rhs=0):
    constant, terms = expression
    return mr.Row(tuple(sorted(terms.items())), F(rhs)-constant)


def _value(expression, values):
    constant, terms = expression
    return constant+sum((a*values[i] for i, a in terms.items()), ZERO)


def _lookup(rows, expression, rhs=0):
    expected = _row(expression, rhs)
    for row in rows:
        if row == expected:
            return row
    raise AssertionError((expected, "required stored inequality missing"))


def _eq_lookup(rows, expression, rhs=0):
    expected = _row(expression, rhs)
    negative = mr.Row(tuple((i, -a) for i, a in expected.terms), -expected.rhs)
    for row in rows:
        if row == expected:
            return ONE, row
        if row == negative:
            return -ONE, row
    raise AssertionError((expected, "required stored source equation missing"))


def _combine(weighted, *, nonnegative):
    rhs, terms = ZERO, {}
    for weight, row in weighted:
        weight = F(weight)
        if nonnegative:
            assert weight >= ZERO
        rhs += weight*row.rhs
        for index, amount in row.terms:
            terms[index] = terms.get(index, ZERO)+weight*amount
    return mr.Row(tuple(sorted((i, a) for i, a in terms.items() if a)), rhs)


def _average(points, weights):
    assert len(points) == len(weights) and sum(weights, ZERO) == ONE
    assert all(w >= ZERO for w in weights)
    return tuple(sum((w*p[i] for w, p in zip(weights, points)), ZERO)
                 for i in range(len(points[0])))


def _source(bounds=((0, 1),)*4+((-1, 1),), *, budget=None, **kwargs):
    return mr.Source(bounds, enabled=True,
                     budget=_budget() if budget is None else budget, **kwargs)


def _attach(state, parents, children):
    return mr.attach(state, parents=parents, children=children,
                     enabled=True, frame=state.frame)


def _build(*, budget=None):
    source = _source(budget=budget)
    A1, A2, A3, A4, t = source.inputs
    f = (source.affine(((A1, ONE), (A2, -ONE))),
         source.affine(((A3, ONE), (A4, -ONE))))
    parents = tuple(source.relu(v) for v in f)
    # Neither z nor F reads either parent f: both use the actual source skip.
    z = source.affine(((A1, F(4, 5)), (A2, F(-4, 5)),
                       (A3, F(4, 5)), (A4, F(-4, 5)),
                       (parents[0].q, F(3, 50)), (parents[1].q, F(3, 50)),
                       (t, F(1, 10))))
    d = source.affine(((parents[0].q, ONE), (parents[1].q, -ONE)))
    g = (source.add(z, d), source.affine(((z, ONE), (d, -ONE))))
    children = tuple(source.relu(v) for v in g)
    skip = source.add(z, parents[0].q)
    readout = source.affine(((parents[0].q, F(2)), (parents[1].q, F(2)),
                             (A1, F(-6, 5)), (A2, F(6, 5)),
                             (A3, F(-4, 5)), (A4, F(4, 5)),
                             (children[0].q, F(2, 5)), (children[1].q, F(-2, 5))))
    state = source.seal()
    return dict(source=source, H=state, inputs=source.inputs, f=f,
                parents=parents, z=z, d=d, g=g, children=children,
                skip=skip, readout=readout)


def _snapshot(state):
    material = state.materialize(frame=state.frame)
    return (state.frame, state.n_columns, state.column_bounds,
            state.binary_columns, state.input_columns, state.error_columns,
            state.definitions, material.eq, material.le)


def _case():
    if "main" not in _CASES:
        case = _build()
        _SNAPSHOTS["main"] = _snapshot(case["H"])
        case["R"] = _attach(case["H"], case["parents"], case["children"])
        _CASES["main"] = case
    return _CASES["main"]


def _symbols(case):
    state, relation = case["H"], case["R"]
    x1, x2, q1, q2 = map(_affine, relation.normalized)
    s = dict(x1=x1, x2=x2, q1=q1, q2=q2,
             y1=_col(case["children"][0].q.column),
             y2=_col(case["children"][1].q.column))
    for i, gate in enumerate(case["parents"], 1):
        s["alpha%d" % i] = _expr(F(1, 2), ((gate.phase_column, F(1, 2)),))
    for j, name in enumerate(("c12", "c21", "d12", "d21", "v1", "v2")):
        s[name] = _col(state.n_columns+j)
    for i in (1, 2):
        s["p%d" % i] = _sum((1, s["alpha%d" % i]), (-1, s["q%d" % i]))
        s["n%d" % i] = _sum((1, _expr(1)), (-1, s["alpha%d" % i]),
                              (-1, s["q%d" % i]), (1, s["x%d" % i]))
    s["D"] = _sum((2, q1), (2, q2), (-1, x1), (-1, x2))
    s["T"] = _sum((1, q1), (-1, q2), (1, s["c12"]), (-1, s["c21"]))
    s["U"] = _sum((1, q1), (1, q2), (-1, s["c12"]), (-1, s["c21"]))
    return s


def _certificate(case):
    """Actual nonnegative LE combination, then actual EQ elimination."""
    state, relation = case["H"], case["R"]
    material, s = state.materialize(frame=state.frame), _symbols(case)
    lam, a, b = F(2, 5), F(4, 5), F(3, 50)
    assert relation.tau == ONE and relation.a == (a, a) and relation.b == (b, b)
    delta = _sum((1, s["y1"]), (-1, s["y2"]), (-1, _affine(relation.K)))
    weighted = [(lam, _lookup(relation.additional_le,
                             _sum((1, delta), (-2, s["p2"]))))]
    weighted.append((lam*a, _lookup(relation.additional_le,
        _sum((1, s["T"]), (-1, s["p2"]), (-1, s["n2"])))))
    # V=q1-q2+d12-d21 <= 1, derived from three stored rows, not assumed.
    weighted.extend(((lam*b, _lookup(relation.additional_le,
                                     _sum((1, s["d12"]), (-1, s["q2"])))),
                     (lam*b, _lookup(relation.additional_le, _sum((-1, s["d21"])))),
                     (lam*b, _lookup(material.le, s["q1"], ONE))))
    r, lo, hi = _affine(relation.r), F(-1, 10), F(1, 10)
    assert relation.r_bounds == (lo, hi) and relation.e_bounds == (ZERO, ZERO)
    mc = ((_sum((1, s["v1"]), (-hi, s["alpha1"])), ZERO),
          (_sum((1, s["v1"]), (-1, r), (-lo, s["alpha1"])), -lo),
          (_sum((-1, s["v2"]), (lo, s["alpha2"])), ZERO),
          (_sum((1, r), (-1, s["v2"]), (hi, s["alpha2"])), hi))
    weighted.extend((lam/2, _lookup(relation.additional_le, ex, rhs)) for ex, rhs in mc)
    for name, weight in (("p1", F(4, 5)), ("n1", F(6, 5)),
                         ("p2", F(2, 25)), ("n2", F(12, 25))):
        weighted.append((weight, _lookup(material.le, _sum((-1, s[name])))))
    combined = _combine(weighted, nonnegative=True)
    physical = _sum((1, s["D"]), (lam, s["y1"]), (-lam, s["y2"]),
                    (-lam/2, s["x1"]), (lam/2, s["x2"]))
    assert combined == _row(physical, F(258, 125))
    A = tuple(_col(v.column) for v in case["inputs"][:4])
    differences = (_sum((1, A[0]), (-1, A[1])), _sum((1, A[2]), (-1, A[3])))
    equalities = []
    for i, coefficient in enumerate((F(-6, 5), F(-4, 5))):
        orientation, eq = _eq_lookup(material.eq,
            _sum((1, s["x%d" % (i+1)]), (-1, differences[i])))
        equalities.append((-coefficient*orientation, eq))
    source_readout = _sum((2, s["q1"]), (2, s["q2"]),
                          (F(-6, 5), differences[0]), (F(-4, 5), differences[1]),
                          (lam, s["y1"]), (-lam, s["y2"]))
    orientation, terminal = _eq_lookup(material.eq,
        _sum((1, _col(case["readout"].column)), (-1, source_readout)))
    equalities.append((orientation, terminal))
    final = _combine([(ONE, combined)]+equalities, nonnegative=False)
    assert final == mr.Row(((case["readout"].column, ONE),), F(258, 125))
    assert all(i < state.n_columns for i, _ in final.terms)
    return final, len(weighted), len(equalities)


def _transport(x1, x2, t=ZERO):
    return ((1+x1)/2, (1-x1)/2, (1+x2)/2, (1-x2)/2, F(t))


def _relaxed(case, inputs, qs, ys, phases):
    """Independent evaluation of every original slot of the fixed fixture."""
    inputs, qs, ys, phases = map(tuple, (inputs, qs, ys, phases))
    values = {v.column: F(x) for v, x in zip(case["inputs"], inputs)}
    xs = (inputs[0]-inputs[1], inputs[2]-inputs[3])
    for i, gate in enumerate(case["parents"]):
        values[case["f"][i].column] = xs[i]
        values[gate.q.column], values[gate.phase_column] = F(qs[i]), F(phases[i])
    z = F(4, 5)*(xs[0]+xs[1])+F(3, 50)*(qs[0]+qs[1])+inputs[4]/10
    d = qs[0]-qs[1]
    values[case["z"].column], values[case["d"].column] = z, d
    for i, gate in enumerate(case["children"]):
        values[gate.f.column] = z+(d if i == 0 else -d)
        values[gate.q.column], values[gate.phase_column] = F(ys[i]), F(phases[i+2])
    values[case["skip"].column] = z+qs[0]
    values[case["readout"].column] = 2*(qs[0]+qs[1])-F(6, 5)*xs[0]-F(4, 5)*xs[1]+F(2, 5)*(ys[0]-ys[1])
    assert set(values) == set(range(case["H"].n_columns))
    return tuple(values[i] for i in range(case["H"].n_columns))


def _fake(case):
    u = F(199, 200)
    return _relaxed(case, (F(1, 2),)*4+(ZERO,), (u/2, u/2),
                    (u*F(397, 500), u*F(297, 500)), (ZERO,)*4)


def _corners_and_base(case):
    state, u = case["H"], F(199, 200)
    corners = tuple(state.canonical(_transport(s*u, t*u), frame=state.frame)
                    for s, t in ((-1, -1), (1, -1), (-1, 1), (1, 1)))
    base = tuple(state.input_columns)+tuple(v.column for v in case["f"])
    base += tuple(g.q.column for g in case["parents"])+tuple(g.phase_column for g in case["parents"])
    base += (case["z"].column, case["d"].column, case["skip"].column)
    return corners, base


def _mc(alpha, value, lo, hi, observed):
    return max(lo*alpha, value-hi*(1-alpha)) <= observed <= min(hi*alpha, value-lo*(1-alpha))


def test_01_default_off_and_frame_identity():
    with pytest.raises(mr.Rejected):
        mr.Source(((0, 1),))
    private = _build(budget=mr.Budget())
    state = private["H"]
    for enabled in (False, 1):
        with pytest.raises(mr.Rejected):
            mr.attach(state, parents=private["parents"], children=private["children"],
                      enabled=enabled, frame=state.frame)
    with pytest.raises(mr.Rejected):
        mr.attach(state, parents=private["parents"], children=private["children"],
                  enabled=True, frame=object())
    relation = _attach(state, private["parents"], private["children"])
    with pytest.raises(mr.Rejected):
        relation.materialize(frame=object())
    other = _build(budget=mr.Budget())
    with pytest.raises(mr.Rejected):
        mr.attach(state, parents=other["parents"], children=private["children"],
                  enabled=True, frame=state.frame)
    assert not state.budget.failed
    _record(1, default_off=True, same_owned_frame_required=True,
            foreign_gate_rejected=True, old_state_not_poisoned=True)


def test_02_shared_source_recovery_without_direct_parent_edges():
    case = _case()
    state, relation = case["H"], case["R"]
    assert relation.parent is state and relation.frame is state.frame
    assert relation.a == (F(4, 5),)*2 and relation.b == (F(3, 50),)*2
    assert relation.scales == (ONE, ONE) and relation.tau == ONE
    assert relation.gram == ((F(1, 2), ZERO), (ZERO, F(1, 2)))
    assert relation.gram_rhs == (F(2, 5),)*2 and relation.gram_det == F(1, 4)
    assert relation.mixed_bounds == (F(-9, 10), F(24, 25))
    assert relation.r == mr.Affine(ZERO, ((case["inputs"][4].column, F(1, 10)),))
    for child in case["children"]:
        assert child.f.column != case["f"][0].column and child.f.column != case["f"][1].column
    for column in (case["z"].column, case["readout"].column):
        form = state.definitions[column].data
        assert not ({i for i, _ in form.terms} & {v.column for v in case["f"]})
    assert {v.column for v in case["inputs"]} <= set(relation.source_columns)
    _record(2, nonidentity_parent_eq_recovered=True, gram_det="1/4",
            child_and_readout_use_original_source_skip=True,
            mixed_bounds=["-9/10", "24/25"], pure_x_q_fallback=False)


def test_03_stored_row_physical_projection_certificate():
    case = _case()
    row, le_terms, eq_terms = _certificate(case)
    state, relation = case["H"], case["R"]
    fake = _fake(case)
    assert state.satisfied(fake, frame=state.frame)
    assert fake[case["readout"].column] == F(2587, 1250)
    assert fake[case["readout"].column]-row.rhs == F(7, 1250)
    assert max(ZERO, fake[case["readout"].column]-F(413, 200)) == F(23, 5000)
    point = state.canonical((ONE, ZERO, ZERO, ONE, ONE), frame=state.frame)
    assert point[case["readout"].column] == row.rhs
    assert row.rhs-F(413, 200) == F(-1, 1000)
    extended = relation.canonical_extension(point, frame=state.frame)
    assert relation.satisfied(extended, frame=state.frame, integral=True)
    _record(3, physical_upper="258/125", attained=True,
            false_readout="2587/1250", strict_gap="7/1250",
            positive_le_terms=le_terms, stored_eq_terms=eq_terms,
            every_auxiliary_extension_excluded=True, numerical_solver_used=False)


def test_04_complete_single_child_prefix_witnesses():
    case = _case()
    corners, base = _corners_and_base(case)
    target, u = _fake(case), F(199, 200)
    for j, delta in enumerate((F(2, 5), F(16, 165))):
        mean = _average(corners, (delta, F(1, 2)-delta, F(1, 2)-delta, delta))
        gate = case["children"][j]
        retained = base+(gate.f.column, gate.q.column, gate.phase_column)
        assert all(mean[i] == target[i] for i in retained)
        assert mean[gate.q.column] == u*(F(53, 100)+F(33, 50)*delta)
        assert mean[gate.phase_column] == ZERO
    assert all(F(1, 400) <= point[i] <= F(399, 400)
               for point in corners for i in case["H"].input_columns[:4])
    _record(4, full_source_labelled_single_child_prefixes=True,
            original_source_means_match=True, original_child_labels_match=True,
            separate_prefix_decompositions_not_common_graph_witness=True)


def test_05_joint_children_over_convexified_parents():
    case = _case()
    state, u = case["H"], F(199, 200)
    corners, base = _corners_and_base(case)
    atoms = []
    for a1, a2 in ((ONE, F(1, 2)), (ZERO, F(1, 2)), (ZERO, ZERO), (ONE, ONE)):
        xs, qs = (u*(2*a1-1), u*(2*a2-1)), (u*a1, u*a2)
        z = F(4, 5)*sum(xs)+F(3, 50)*sum(qs)
        guards = (z+qs[0]-qs[1], z-qs[0]+qs[1])
        assert all(v != ZERO for v in guards)
        atom = _relaxed(case, _transport(*xs), qs, tuple(max(ZERO, g) for g in guards),
                        (2*a1-1, 2*a2-1)+tuple(F(1 if g > 0 else -1) for g in guards))
        assert state.satisfied(atom, frame=state.frame)
        parent_mean = _average(corners, ((1-a1)*(1-a2), a1*(1-a2), (1-a1)*a2, a1*a2))
        assert all(parent_mean[i] == atom[i] for i in base)
        atoms.append(atom)
    assert _average(atoms, (F(1, 5), F(1, 5), F(3, 10), F(3, 10))) == _fake(case)
    _record(5, complete_convexified_parent_hull_witness=True,
            both_actual_children_joint=True, source_and_original_labels_preserved=True,
            full_original_network_hull_or_all_rlt_claimed=False)


def test_06_all_qualified_old_tau_certificates():
    u, alpha, q, p = F(199, 200), F(1, 2), F(199, 400), F(1, 400)
    diff, slope = u/5, 2*p+2
    # These are exact boundary plus nonnegative-slope certificates, not a scan.
    x_tau, x_lo, x_hi = F(51, 50), F(-1, 10), F(11, 50)
    x_residual, v1, v2 = F(3, 50)*u, F(1039, 10000), F(-1, 20)
    assert (F(-4, 5)+x_lo, F(4, 5)+x_hi) == (F(-9, 10), x_tau)
    assert max(x_lo*alpha, x_residual-x_hi*(1-alpha)) == F(-1, 20)
    assert min(x_hi*alpha, x_residual-x_lo*(1-alpha)) == F(1097, 10000)
    assert _mc(alpha, x_residual, x_lo, x_hi, v1) and _mc(alpha, x_residual, x_lo, x_hi, v2)
    assert _mc(alpha, ZERO, -ONE, ONE, ZERO)
    assert ZERO <= 2*p and u <= 2*u  # Both T rows and U<=D.
    x_K = v1-v2
    x_payment = 2*x_tau*p+2*(x_tau-1)
    assert diff-x_K == x_payment == F(451, 10000)
    assert -diff+x_K <= x_payment and slope == F(401, 200) > ZERO
    # For tau=x_tau+s, s>=0, both payments equal x_payment+slope*s;
    # K and every MC witness above are independent of tau.
    q_tau, q_lo, q_hi = F(44, 25), F(-17, 10), F(17, 10)
    assert q_hi+F(3, 50) == q_tau and q_lo == -q_hi
    assert _mc(alpha, q, ZERO, ONE, F(199, 800))
    assert _mc(alpha, ZERO, q_lo, q_hi, ZERO)
    q_K = F(3, 50)*(q-F(199, 800))+F(3, 50)*(F(199, 800)-q)
    q_payment = 2*q_tau*p+2*(q_tau-1)
    assert q_K == ZERO and q_payment == F(1911, 1250) > diff
    assert -diff+q_K <= q_payment
    _record(6, pure_x_all_tau_from="51/50", pure_q_all_tau_from="44/25",
            common_payment_slope="401/200", analytical_halfline_proof=True,
            searched_tau_values=0, additional_cross_residual_consistency_claimed=False)


def test_07_fixed_integer_extensions_and_decoder():
    case = _case()
    state, relation = case["H"], case["R"]
    checked = 0
    for x1, x2, t in product((F(-3, 4), ZERO, F(2, 3)), repeat=3):
        inputs = _transport(x1, x2, t)
        point = state.canonical(inputs, frame=state.frame)
        extended = relation.canonical_extension(point, frame=state.frame)
        assert extended[:state.n_columns] == point
        assert relation.satisfied(extended, frame=state.frame, integral=True)
        assert relation.decode(extended, frame=state.frame) == inputs
        alpha = tuple((point[g.phase_column]+1)/2 for g in case["parents"])
        q1, q2 = (point[g.q.column] for g in case["parents"])
        assert extended[-6:] == (alpha[0]*x2, alpha[1]*x1, alpha[0]*q2,
                                 alpha[1]*q1, alpha[0]*t/10, alpha[1]*t/10)
        checked += 1
    assert checked == 27
    _record(7, fixed_integer_points=checked, every_input_decoded=True,
            original_integer_projection_preserved=True, validated_adv_claimed=False)


def test_08_original_zero_phase_labels():
    case = _case()
    state, relation = case["H"], case["R"]
    bits = tuple(g.phase_column for g in case["parents"]+case["children"])
    for signs in product((-1, 1), repeat=4):
        point = state.canonical((F(1, 2),)*4+(ZERO,), zero_phases=tuple(zip(bits, signs)), frame=state.frame)
        assert tuple(point[i] for i in bits) == signs
        assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    private = _build(budget=mr.Budget())
    with pytest.raises(mr.Rejected):
        private["H"].canonical((ONE, ZERO, ZERO, ONE, ZERO),
            zero_phases=((private["parents"][0].phase_column, -1),), frame=private["H"].frame)
    _record(8, original_zero_label_extensions=16, nonzero_wrong_label_rejected=True,
            new_binary_columns=0)


def test_09_complete_nonzero_residual_and_derived_tau():
    source = _source(((0, 1),)*4+((-1, 1),)*2)
    A1, A2, A3, A4, t, u = source.inputs
    f = (source.affine(((A1, ONE), (A2, -ONE))), source.affine(((A3, ONE), (A4, -ONE))))
    parents = tuple(source.relu(v) for v in f)
    common = ((A1, F(1, 2)), (A2, F(-1, 2)), (A3, F(1, 4)), (A4, F(-1, 4)))
    g1 = source.affine(common+((u, F(1, 8)), (parents[0].q, F(23, 20)), (parents[1].q, -ONE)), F(1, 20))
    g2 = source.affine(common+((t, F(1, 10)), (u, F(-1, 8)),
                              (parents[0].q, F(-19, 20)), (parents[1].q, ONE)))
    children = (source.relu(g1), source.relu(g2))
    state = source.seal()
    relation = _attach(state, parents, children)
    assert relation.a == (F(1, 2), F(1, 4)) and relation.b == (F(1, 10), ZERO)
    assert relation.tau == F(41, 40)
    assert relation.r_bounds == (F(-1, 40), F(3, 40))
    assert relation.e_bounds == (F(-3, 20), F(1, 4))
    for inputs in (_transport(F(1, 2), F(-1, 3), F(2, 5))+(F(-3, 4),),
                   (ONE, ZERO, ONE, ZERO, -ONE, ONE), (ZERO, ONE, ZERO, ONE, ONE, -ONE)):
        point = state.canonical(inputs, frame=state.frame)
        q1, q2 = (point[g.q.column] for g in parents)
        assert _value(_affine(relation.r), point) == F(1, 40)+inputs[4]/20
        assert _value(_affine(relation.e), point) == F(1, 40)-inputs[4]/20+inputs[5]/8+(q1+q2)/40
        assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(9, derived_tau="41/40", r_bounds=["-1/40", "3/40"],
            e_bounds=["-3/20", "1/4"], full_q_bias_and_source_residuals=True)


def test_10_same_range_distinct_source_identity():
    source = _source()
    A1, A2, A3, A4, t = source.inputs
    f = (source.affine(((A1, ONE), (A2, -ONE))), source.affine(((A3, ONE), (A4, -ONE))))
    parents = tuple(source.relu(v) for v in f)
    groups = []
    for sign in (ONE, -ONE):
        z = source.affine(((A1, F(4, 5)), (A2, F(-4, 5)), (A3, F(4, 5)), (A4, F(-4, 5)),
                           (parents[0].q, F(3, 50)), (parents[1].q, F(3, 50)), (t, sign/10)))
        groups.append(tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s),
                                                       (parents[1].q, -s)))) for s in (ONE, -ONE)))
    state = source.seal()
    relations = tuple(_attach(state, parents, children) for children in groups)
    assert relations[0].r != relations[1].r
    assert relations[0].r_bounds == relations[1].r_bounds == (F(-1, 10), F(1, 10))
    point = state.canonical(_transport(F(1, 2), F(-1, 2), F(3, 5)), frame=state.frame)
    extensions = tuple(r.canonical_extension(point, frame=state.frame) for r in relations)
    assert extensions[0][-2] == F(3, 50) and extensions[1][-2] == F(-3, 50)
    assert all(r.parent is state and r.satisfied(v, frame=state.frame, integral=True)
               for r, v in zip(relations, extensions))
    _record(10, same_h_multiple_consumers=True, equal_range_not_independent_source=True)


def test_11_asymmetric_parent_normalization():
    source = _source(((-2, 1), (-1, 3), (-1, 1)))
    f1, f2, t = source.inputs
    parents = (source.relu(f1), source.relu(f2))
    z = source.affine(((f1, F(2, 5)), (f2, F(4, 15)),
                       (parents[0].q, F(3, 100)), (parents[1].q, F(1, 50)), (t, F(1, 10))))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s/2),
                                               (parents[1].q, -s/3)))) for s in (ONE, -ONE))
    state = source.seal()
    relation = _attach(state, parents, children)
    assert relation.scales == (F(2), F(3)) and relation.tau == ONE
    assert relation.a == (F(4, 5),)*2 and relation.b == (F(3, 50),)*2
    for inputs in ((F(-2), F(3), ZERO), (ONE, -ONE, ONE),
                   (F(-1, 2), F(3, 4), F(-1, 2)), (ZERO, ZERO, ZERO)):
        point = state.canonical(inputs, frame=state.frame)
        expected = (inputs[0]/2, inputs[1]/3, max(ZERO, inputs[0])/2, max(ZERO, inputs[1])/3)
        assert tuple(_value(_affine(v), point) for v in relation.normalized) == expected
        extended = relation.canonical_extension(point, frame=state.frame)
        assert relation.satisfied(extended, frame=state.frame, integral=True)
        assert relation.decode(extended, frame=state.frame) == inputs
    _record(11, scales=[2, 3], asymmetric_bounds=True, normalization_translation=False)


def test_12_bn_error_and_shared_consumers():
    source = _source()
    A1, A2, A3, A4, t = source.inputs
    f = (source.affine(((A1, ONE), (A2, -ONE))), source.affine(((A3, ONE), (A4, -ONE))))
    parents = tuple(source.relu(v) for v in f)
    residual = source.bn(t, mean=ZERO, beta=ZERO,
                         scale_interval=(F(3, 40), F(1, 8)), ahat=F(1, 10))
    z = source.affine(((A1, F(4, 5)), (A2, F(-4, 5)), (A3, F(4, 5)), (A4, F(-4, 5)),
                       (parents[0].q, F(3, 50)), (parents[1].q, F(3, 50)), (residual, ONE)))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s),
                                               (parents[1].q, -s)))) for s in (ONE, -ONE))
    doubled = source.add(residual, residual)
    canceled = source.affine(((residual, ONE), (residual, -ONE)))
    state = source.seal()
    relation = _attach(state, parents, children)
    assert len(state.error_columns) == 1 and relation.e_bounds == (ZERO, ZERO)
    assert dict(relation.r.terms)[state.error_columns[0]] == F(1, 40)
    assert relation.r_bounds == (F(-1, 8), F(1, 8))
    inputs = _transport(F(1, 2), F(-1, 2), F(1, 3))
    point = state.canonical(inputs, errors=(F(1, 3),), frame=state.frame)
    assert point[residual.column] == F(1, 8)*inputs[4]
    assert point[doubled.column] == 2*point[residual.column] and point[canceled.column] == ZERO
    assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(12, declared_bn_scale_enclosure=True, shared_scalar_error_retained=True,
            no_onnx_sqrt_or_channel_error_idealization_claimed=True)


def test_13_binary_source_and_retained_predicates():
    eq = mr.Row(((4, ONE),), ZERO)
    le = mr.Row(((0, ONE), (2, ONE), (5, F(1, 4))), F(3, 2))
    source = _source(((0, 1),)*4+((-1, 1),)*2,
                     source_kinds=("continuous",)*5+("binary",), eq=(eq,), le=(le,))
    A1, A2, A3, A4, t, binary = source.inputs
    f = (source.affine(((A1, ONE), (A2, -ONE))), source.affine(((A3, ONE), (A4, -ONE))))
    parents = tuple(source.relu(v) for v in f)
    z = source.affine(((A1, F(4, 5)), (A2, F(-4, 5)), (A3, F(4, 5)), (A4, F(-4, 5)),
                       (parents[0].q, F(3, 50)), (parents[1].q, F(3, 50)), (binary, F(1, 10))))
    children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s),
                                               (parents[1].q, -s)))) for s in (ONE, -ONE))
    extra = mr.Row(((z.column, ONE),), F(2))
    source.predicate(extra)
    state = source.seal()
    relation = _attach(state, parents, children)
    points = []
    for sign in (-ONE, ONE):
        inputs = (F(3, 4), F(1, 4), F(1, 4), F(3, 4), ZERO, sign)
        point = state.canonical(inputs, frame=state.frame)
        extended = relation.canonical_extension(point, frame=state.frame)
        assert relation.decode(extended, frame=state.frame) == inputs
        assert relation.satisfied(extended, frame=state.frame, integral=True)
        points.append(point)
    material = relation.materialize(frame=state.frame)
    assert eq in material.eq and le in material.le and extra in material.le
    assert binary.column in state.binary_columns and len(state.binary_columns) == 5
    mean = _average(points, (F(1, 2),)*2)
    assert state.satisfied(mean, frame=state.frame) and not state.satisfied(mean, frame=state.frame, integral=True)
    _record(13, original_binary_sources=1, all_original_predicates_retained=True,
            fractional_source_not_integer_decoder_witness=True)


def test_14_difference_only_tail_alias():
    case = _case()
    state, relation = case["H"], case["R"]
    point = state.canonical((ONE, ZERO, ONE, ZERO, ONE), frame=state.frame)
    z = _value(_affine(relation.z), point)
    assert z == F(91, 50)
    clip = min(F(2), max(ZERO, z+1))
    assert clip == 2 and clip != z+1
    alpha = tuple((point[g.phase_column]+1)/2 for g in case["parents"])
    assert alpha == (ONE, ONE)
    assert (alpha[0]-alpha[1])*clip == (alpha[0]-alpha[1])*(z+1)
    assert alpha[0]*clip != alpha[0]*(z+1)
    assert relation.satisfied(relation.canonical_extension(point, frame=state.frame), frame=state.frame, integral=True)
    _record(14, actual_reference_tail_crossing=True, difference_only_annihilation=True,
            individual_observation_alias_rejected=True, virtual_phase_bits=0)


def test_15_rank_and_dependent_parent_rejection():
    for mode in ("rank", "dependency", "same_parent"):
        source = _source(((-1, 1),)*3, budget=mr.Budget())
        x, y, t = source.inputs
        first = source.relu(source.affine(((x, ONE),)))
        f2 = source.affine(((x, F(2)),)) if mode == "rank" else source.affine(((first.q, ONE), (y, ONE)))
        second = source.relu(f2)
        parents = (first, first) if mode == "same_parent" else (first, second)
        z = source.affine(((x, F(1, 4)), (y, F(1, 4)), (t, F(1, 20))))
        children = tuple(source.relu(source.affine(((z, ONE), (first.q, s),
                                                   (second.q, -s)))) for s in (ONE, -ONE))
        state = source.seal()
        before = state.budget.work
        with pytest.raises(mr.Rejected):
            _attach(state, parents, children)
        assert state.budget.work > before and not state.budget.failed
        point = state.canonical((ZERO,)*3, frame=state.frame)
        assert state.satisfied(point, frame=state.frame, integral=True)
    _record(15, rank_deficient_rejected=True, selected_parent_dependency_rejected=True,
            repeated_parent_rejected=True, original_h_survives_rejection=True)


def test_16_guard_and_invalid_input_rejection():
    for mode in ("guard", "tau", "stable"):
        bounds = ((0, 1), (-1, 1), (-1, 1)) if mode == "stable" else ((-1, 1),)*3
        source = _source(bounds, budget=mr.Budget())
        x, y, t = source.inputs
        parents = (source.relu(x), source.relu(y))
        z = source.affine(((x, F(2) if mode == "guard" else F(1, 2)), (t, F(1, 10))))
        orientation = -ONE if mode == "tau" else ONE
        children = tuple(source.relu(source.affine(((z, ONE), (parents[0].q, s*orientation),
                                                   (parents[1].q, -s*orientation)))) for s in (ONE, -ONE))
        state = source.seal()
        with pytest.raises(mr.Rejected):
            _attach(state, parents, children)
        assert not state.budget.failed
        point = state.canonical((ZERO,)*3, frame=state.frame)
        assert state.satisfied(point, frame=state.frame, integral=True)
    for bounds in (((0.0, 1),), ((False, 1),), ((1, -1),)):
        with pytest.raises(mr.Rejected):
            mr.Source(bounds, enabled=True, budget=mr.Budget())
    _record(16, mixed_guard_nonpositive_tau_stable_parent_rejected=True,
            floats_bools_bad_bounds_rejected=True, no_alternate_rule_fallback=True)


def test_17_sticky_shared_budget():
    budget = mr.Budget(max_work=0)
    with pytest.raises(mr.Rejected):
        mr.Source(((-1, 1),), enabled=True, budget=budget)
    assert budget.failed
    with pytest.raises(mr.Rejected):
        mr.Source(((-1, 1),), enabled=True, budget=budget)
    budget = mr.Budget()
    with pytest.raises(mr.Rejected):
        mr.Source(((ZERO, F(1 << 513)),), enabled=True, budget=budget)
    assert budget.failed
    unfinished = _source(((-1, 1),), budget=mr.Budget())
    with pytest.raises(mr.Rejected):
        unfinished.affine(((unfinished.inputs[0], 0.5),))
    assert unfinished.budget.failed
    with pytest.raises(mr.Rejected):
        unfinished.seal()
    before = _budget().work
    case = _case()
    case["R"].materialize(frame=case["H"].frame)
    assert _budget().work > before and not _budget().failed
    assert case["R"].budget is case["H"].budget is _budget()
    _record(17, resource_failure_sticky=True, unfinished_builder_failure_sticky=True,
            exact_512_bit_limit=True, ordinary_positive_fixtures_share_budget=True)


def test_18_complete_materialized_cost():
    case = _case()
    state, relation = case["H"], case["R"]
    old, full = state.materialize(frame=state.frame), relation.materialize(frame=state.frame)
    assert relation.n_columns == state.n_columns+6
    assert len(relation.additional_le) == 29
    assert sum(len(row.terms) for row in relation.additional_le) == 96
    assert len(relation.products) == 6 and relation.defect_rows == (24, 25)
    assert relation.capacity_rows == (26, 27, 28)
    assert relation.product_rows == tuple(tuple(range(4*j, 4*j+4)) for j in range(6))
    assert full.eq == old.eq and full.le[:len(old.le)] == old.le
    bounds = tuple(row for i in range(state.n_columns, relation.n_columns)
                   for row in (mr.Row(((i, -ONE),), -relation.column_bounds[i][0]),
                               mr.Row(((i, ONE),), relation.column_bounds[i][1])))
    assert full.le[len(old.le):len(old.le)+12] == bounds
    assert full.le[len(old.le)+12:] == relation.additional_le
    assert len(full.le)-len(old.le) == 41
    assert sum(len(row.terms) for row in full.le)-sum(len(row.terms) for row in old.le) == 108
    assert full.binary_columns == old.binary_columns == state.binary_columns
    _record(18, original_columns=state.n_columns, full_columns=relation.n_columns,
            original_eq=len(old.eq), full_eq=len(full.eq), original_le=len(old.le), full_le=len(full.le),
            original_nnz=sum(len(row.terms) for row in old.eq+old.le),
            full_nnz=sum(len(row.terms) for row in full.eq+full.le),
            relation_columns=6, relation_le=29, relation_nnz=96,
            materialized_increment_le=41, materialized_increment_nnz=108,
            original_binary_columns=len(state.binary_columns),
            physical_native_gpu_cost_qualification=False)


def test_19_original_state_immutability():
    case = _case()
    state, relation = case["H"], case["R"]
    assert _snapshot(state) == _SNAPSHOTS["main"]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        relation.tau = F(2)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        state.n_columns = 0
    private = _build(budget=mr.Budget())
    with pytest.raises(mr.Rejected):
        private["source"].affine(((private["inputs"][0], ONE),))
    with pytest.raises(mr.Rejected):
        private["source"].seal()
    assert not private["H"].budget.failed
    point = private["H"].canonical((F(1, 2),)*4+(ZERO,), frame=private["H"].frame)
    assert private["H"].satisfied(point, frame=private["H"].frame, integral=True)
    _record(19, old_eq_le_columns_decoder_unchanged=True,
            published_state_survives_invalid_append=True, immutable_relation=True)


def test_20_summary_and_evidence_boundary():
    assert not _budget().failed
    _record(20, fixed_three_actual_models_not_executed=True,
            solver_phase_pool_attack_rescue_used=False,
            same_h_mathematical_relation_only=True)
    assert tuple(_EVIDENCE) == _NAMES and len(_EVIDENCE) == 20
    _record_file("summary.json", dict(
        schema="d245_mixed_source_component_v1", tests=20,
        required_tests=4209, required_test_files=224,
        mathematical_stage_only=True, same_h_declared_relation_verified=True,
        all_auxiliary_projection_certificate=True, all_qualified_old_tau_compared=True,
        shared_source_gram_recovery_verified=True,
        source_component_qualified=False, actual_model_binding_qualified=False,
        actual_phase_column_binding_verified=False, native_HZ_admitted=False,
        complete_physical_qualification=False, gpu_computation_completed=False,
        new_domain_qualified=False, new_capability_qualified=False, new_set_class=False,
        formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        inherited_tests_not_replaced=True,
        shared_work=_budget().work, shared_entries=_budget().entries,
        evidence=_EVIDENCE))
