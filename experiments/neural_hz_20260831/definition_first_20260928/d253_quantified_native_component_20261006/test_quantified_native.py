"""Frozen native-row mathematics; never import/run before the root freeze.

The fixed mixtures are proof witnesses, not attacks or model qualification.
Only test 14 invokes the ordinary terminal LP, twice in one fixed direction;
the exact nonnegative stored-row certificate is the primary proof.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

from act.back_end.solver.solver_hz import SparseHZono, _lower_hz_milp
from experiments.neural_hz_20260831.definition_first_20260928.d253_quantified_native_component_20261006 import native_quantified as nm
from experiments.neural_hz_20260831.definition_first_20260928.d249_native_mixed_terminal_20261006 import native_mixed as legacy

ZERO, ONE = F(0), F(1)
RUN = Path(__file__).resolve().parents[2] / "results/d253_quantified_native_component_20261006_v1"
_NAMES = (
    "default_off_and_snapshot",
    "guarded_rows_exactly_match_legacy",
    "outside_guard_stored_certificate",
    "complete_single_child_witnesses",
    "joint_children_convexified_parents",
    "event_tail_formulas_and_asymmetry",
    "integer_extensions_and_zero_labels",
    "complete_residual_and_source_identity",
    "preserved_predicates_and_decoder",
    "whole_population_and_native_widths",
    "compact_and_malformed_graphs",
    "unsupported_inputs_and_rank_dependency",
    "outward_rounding_and_snapshot_integrity",
    "complete_cost_and_ordinary_terminal",
    "shared_sticky_budget",
    "summary_and_qualification_boundary",
)
_EVIDENCE, _CASES = {}, {}
_BUDGET = None


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = nm.Budget()
    return _BUDGET


def _expr(constant=0, terms=()):
    merged = {}
    for i, a in terms:
        merged[i] = merged.get(i, ZERO) + F(a)
    return F(constant), {i: a for i, a in merged.items() if a}


def _sum(*pieces):
    c, terms = ZERO, []
    for w, (b, row) in pieces:
        c += F(w) * b
        terms.extend((i, F(w) * a) for i, a in row.items())
    return _expr(c, terms)


def _col(i):
    return _expr(0, ((i, ONE),))


def _af(value):
    return _expr(value.constant, value.terms)


def _row(expr, rhs=0):
    c, terms = expr
    return nm.base.Row(tuple(sorted(terms.items())), F(rhs) - c)


def _at(expr, values):
    c, terms = expr
    return c + sum((a * values[i] for i, a in terms.items()), ZERO)


def _combine(weighted, *, nonnegative=True):
    entries, rhs = [], ZERO
    for w, row in weighted:
        w = F(w)
        if nonnegative:
            assert w >= 0
        entries.extend((i, w * a) for i, a in row.terms)
        rhs += w * row.rhs
    return _row(_expr(0, entries), rhs)


def _lookup(rows, expr, rhs=0):
    expected = _row(expr, rhs)
    assert expected in rows, (expected, rows)
    return expected


def _scaled_lookup(rows, expr, rhs=0):
    target = _row(expr, rhs)
    for row in rows:
        if tuple(i for i, _ in row.terms) != tuple(i for i, _ in target.terms):
            continue
        if not row.terms:
            continue
        scale = target.terms[0][1] / row.terms[0][1]
        if scale > 0 and _combine(((scale, row),)) == target:
            return scale, row
    raise AssertionError(("missing proportional original row", target))


def _nf(bias=0, continuous=(), binary=()):
    return nm.NativeAffine(F(bias), tuple(sorted(_expr(0, continuous)[1].items())),
                           tuple(sorted(_expr(0, binary)[1].items())))


def _nadd(*pieces, constant=0):
    c, cont, bits = F(constant), [], []
    for w, value in pieces:
        w = F(w)
        c += w * value.bias
        cont.extend((i, w * a) for i, a in value.continuous)
        bits.extend((i, w * a) for i, a in value.binary)
    return _nf(c, cont, bits)


def _nv(value, cont, bits):
    return value.bias + sum((a * cont[i] for i, a in value.continuous), ZERO) + sum(
        (a * bits[i] for i, a in value.binary), ZERO)


def _flat(value, nc):
    return _expr(value.bias, value.continuous + tuple((nc + i, a) for i, a in value.binary))


def _matrix(forms, width, kind):
    data, ii, jj = [], [], []
    for row, form in enumerate(forms):
        for col, value in getattr(form, kind):
            floating = float(value)
            assert F.from_float(floating) == value
            ii.append(row)
            jj.append(col)
            data.append(floating)
    matrix = sp.csr_matrix((data, (ii, jj)), shape=(len(forms), width), dtype=np.float64)
    matrix.sort_indices()
    return matrix


def _fixture(*, compact=False, groups=1, mode="main", parent_L=F(-1, 2), predicates=False):
    """Independent direct native construction, not D243.make_block/Source."""
    nc, nb = 5 + (4 if compact else 8) * groups, 4 * groups
    inputs = tuple(_nf(F(1, 2), ((i, F(1, 2)),)) for i in range(4)) + (_nf(0, ((4, ONE),)),)
    equations, rhs, guards, upper, metadata, populations = [], [], [], [], [], []
    outputs = list(inputs)

    def gate(f, number, L, Q):
        eta = 5 + number * (1 if compact else 2) + (0 if compact else 1)
        s = eta if compact else eta - 1
        z = number
        q = _nf(Q, ((eta, -Q),))
        if compact:
            start = len(guards)
            guards.extend((_nf(0, ((eta, -ONE),), ((z, ONE),)),
                           _nadd((ONE, f), (ONE, _nf(0, ((eta, Q),)))),
                           _nadd((-ONE, f), (ONE, _nf(0, ((eta, -Q),), ((z, L),))))))
            upper.extend((ZERO, Q, -L - Q))
            # Move each affine constant, including f.bias, to its RHS.
            for i in range(start, start + 3):
                upper[i] -= guards[i].bias
                guards[i] = _nf(0, guards[i].continuous, guards[i].binary)
            graph = nm.Graph("compact", (s, eta, z), None, (start, start + 1, start + 2))
        else:
            eq = _nadd((-ONE, f), (ONE, _nf(0, ((s, L), (eta, -Q)), ((z, L),))))
            row = len(equations)
            equations.append(_nf(0, eq.continuous, eq.binary))
            rhs.append(-Q - eq.bias)
            start = len(guards)
            guards.extend((_nf(0, ((s, -ONE),), ((z, -ONE),)),
                           _nf(0, ((eta, -ONE),), ((z, ONE),))))
            upper.extend((ZERO, ZERO))
            graph = nm.Graph("extended", (s, eta, z), row, (start, start + 1))
        metadata.append({"f": f, "q": q, "graph": graph, "L": L, "Q": Q})
        return q, graph

    for j in range(groups):
        f1 = _nf(0, ((0, F(1, 2)), (1, F(-1, 2))))
        f2 = _nf(0, ((2, F(1, 2)), (3, F(-1, 2))))
        if mode == "rank":
            f2 = f1
        if mode == "rounding":
            f2 = _nf(0, ((0, F(1, 4)), (1, F(1, 4)), (2, F(1, 4))))
        q1, p1 = gate(f1, 4 * j, parent_L, F(1, 2))
        if mode == "dependent":
            f2 = _nadd((F(1, 2), q1), (ONE, _nf(0, ((2, F(1, 4)),))))
        q2, p2 = gate(f2, 4 * j + 1, parent_L, F(1, 2))
        a, b, eps = F(7, 8), F(1, 16), F(3, 32)
        if mode == "guarded":
            a = F(13, 16)
        if mode == "negative_t":
            eps = -eps
        if mode == "guard":
            a, b, eps = ONE, F(1, 8), F(1, 8)
        if mode == "rounding":
            a, b, eps = F(1, 2), F(1, 32), F(1, 32)
        if mode == "residual":
            a, b, eps = F(3, 4), F(1, 32), F(1, 32)
        z = _nadd((a, f1), (a, f2), (b, q1), (b, q2), (eps, inputs[4]))
        if mode in ("tails", "tails_shift"):
            z = _nadd((F(5, 4), f1), (F(3, 4), f2), (F(-1, 8), q1),
                      (F(1, 4), q2), (F(1, 16), inputs[4]),
                      constant=F(-1, 2) if mode == "tails_shift" else ZERO)
        if mode == "rounding":
            z = _nadd((ONE, z), (ONE, _nf(0, ((2, F(1, 64)),))))
        if mode == "residual":
            z = _nadd((ONE, z), (ONE, _nf(F(1, 128), ((0, F(1, 128)), (1, F(1, 128))))))
        g1 = _nadd((ONE, z), (ONE, q1), (-ONE, q2))
        g2 = _nadd((ONE, z), (-ONE, q1), (ONE, q2))
        if mode == "zero_tau":
            g2 = g1
        if mode == "residual":
            g1 = _nadd((ONE, g1), (F(1, 16), q1), constant=F(1, 128))
        y1, c1 = gate(g1, 4 * j + 2, F(-2), F(2))
        y2, c2 = gate(g2, 4 * j + 3, F(-2), F(2))
        objective = _nadd((F(2), q1), (F(2), q2), (F(-19, 16), f1),
                         (F(-13, 16), f2), (F(3, 8), y1), (F(-3, 8), y2))
        outputs.extend((f1, f2, q1, q2, y1, y2, z, objective, _nadd((ONE, z), (ONE, q1))))
        populations.append(((p1, p2), (c1, c2)))
    if predicates:
        equations.append(_nf(0, ((0, ONE), (1, ONE))))
        rhs.append(ZERO)
        guards.append(_nf(0, ((4, ONE),)))
        upper.append(F(3, 4))
    hz = SparseHZono(np.array([float(v.bias) for v in outputs], dtype=np.float64),
        _matrix(outputs, nc, "continuous"), _matrix(outputs, nb, "binary"),
        _matrix(equations, nc, "continuous"), _matrix(equations, nb, "binary"),
        np.array([float(v) for v in rhs], dtype=np.float64),
        _matrix(guards, nc, "continuous"), _matrix(guards, nb, "binary"),
        np.array([float(v) for v in upper], dtype=np.float64), frame_id=253, exact=True)
    return {"hz": hz, "decoder": inputs, "population": tuple(populations), "gates": tuple(metadata),
            "outputs": tuple(outputs), "F_index": 12, "compact": compact}


def _attach(case, *, budget=None, widths=None):
    hz = case["hz"]
    snap = nm.capture(hz, case["decoder"], global_widths=widths or (hz.n_cont, hz.n_bin),
                      budget=budget if budget is not None else _budget(), enabled=True)
    result = nm.attach(snap, population=case["population"], enabled=True)
    return snap, result


def _case(name="main", **kwargs):
    if name not in _CASES:
        case = _fixture(**kwargs)
        case["snapshot"], case["result"] = _attach(case)
        _CASES[name] = case
    return _CASES[name]


def _transport(x1, x2, t=0):
    return ((1 + F(x1)) / 2, (1 - F(x1)) / 2, (1 + F(x2)) / 2, (1 - F(x2)) / 2, F(t))


def _assignment(case, inputs, *, amplitudes=None, active=None, widths=None):
    hz = case["hz"]
    nc, nb = widths or (hz.n_cont, hz.n_bin)
    cont, bits = [ZERO] * nc, [ONE] * nb
    cont[:4] = [2 * F(v) - 1 for v in inputs[:4]]
    cont[4] = F(inputs[4])
    for i, gate in enumerate(case["gates"]):
        f = _nv(gate["f"], cont, bits)
        q = max(ZERO, f) if amplitudes is None else F(amplitudes[i])
        alpha = (ONE if f > 0 else ZERO) if active is None else F(active[i])
        s, eta, z = gate["graph"].slots
        bits[z] = 1 - 2 * alpha
        cont[eta] = 1 - q / gate["Q"]
        if not case["compact"]:
            cont[s] = (f - q) / gate["L"] - bits[z]
    return tuple(cont + bits)


def _native_rows(hz, equality):
    ac, ab, rhs = (hz.Ac, hz.Ab, hz.b) if equality else (hz.Auc, hz.Aub, hz.ub)
    rows = []
    for i, bound in enumerate(rhs):
        terms = []
        for matrix, offset in ((ac, 0), (ab, hz.n_cont)):
            for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
                terms.append((offset + int(matrix.indices[k]), F.from_float(float(matrix.data[k]))))
        rows.append(_row(_expr(0, terms), F.from_float(float(bound))))
    return tuple(rows)


def _old_holds(hz, values, *, integral=False):
    assert len(values) == hz.n_cont + hz.n_bin
    if any(not -1 <= v <= 1 for v in values):
        return False
    if integral and any(v not in (-1, 1) for v in values[hz.n_cont:]):
        return False
    return all(_at(_expr(0, row.terms), values) == row.rhs for row in _native_rows(hz, True)) and all(
        _at(_expr(0, row.terms), values) <= row.rhs for row in _native_rows(hz, False))


def _bytes(hz):
    out = [hz.c.tobytes(), hz.b.tobytes(), hz.ub.tobytes()]
    for mat in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        out.extend((mat.shape, mat.data.tobytes(), mat.indices.tobytes(), mat.indptr.tobytes()))
    return tuple(out), hz.frame_id, hz.exact


def _average(weighted):
    assert sum((w for w, _ in weighted), ZERO) == 1
    return tuple(sum((w * v[i] for w, v in weighted), ZERO) for i in range(len(weighted[0][1])))


def _readout(case, index, values):
    return _at(_flat(case["outputs"][index], case["hz"].n_cont), values)


def _fake(case):
    u = F(127, 128)
    return _assignment(case, _transport(0, 0),
                       amplitudes=(u / 2, u / 2, F(107, 128) * u, F(75, 128) * u),
                       active=(F(1, 2),) * 4)


def test_01_default_off_and_snapshot():
    assert nm.capture(object(), object(), global_widths=object()) is None
    assert nm.attach(object(), population=object()) is None
    with pytest.raises(nm.Rejected):
        nm.capture(None, (), global_widths=(0, 0), enabled=1)
    case = _case()
    snap = case["snapshot"]
    assert snap.hz is not case["hz"] and snap.original_dimensions == (13, 4)
    assert _bytes(snap.hz) == _bytes(case["hz"])
    assert snap.budget is _budget() and snap.n_columns == 17
    assert not snap.hz.c.flags.writeable and not snap.hz.Ac.data.flags.writeable
    rel = case["result"].relations[0]
    assert rel.scales == (ONE, ONE) and rel.tau == 1
    assert rel.a == (F(7, 8),) * 2 and rel.b == (F(1, 16),) * 2
    assert rel.r_bounds == (F(-3, 32), F(3, 32)) and rel.e_bounds == (ZERO, ZERO)
    for i in range(2):
        assert _af(rel.x[i]) == _flat(case["gates"][i]["f"], 13)
        assert _af(rel.q[i]) == _flat(case["gates"][i]["q"], 13)
        assert _af(rel.alphas[i]) == _expr(F(1, 2), ((13 + i, F(-1, 2)),))
    _record(1, default_off=True, native_dimensions=[13, 4], original_signed_bits=True,
            independent_block_built=False)


def test_02_guarded_rows_exactly_match_legacy():
    case = _case("guarded", mode="guarded")
    old_snapshot = legacy.capture(case["hz"], case["decoder"], global_widths=(13, 4),
                                  budget=_budget(), enabled=True)
    old = legacy.attach(old_snapshot, population=case["population"], enabled=True)
    new = case["result"]
    rel = new.relations[0]
    assert rel.event_bounds == ((F(-29, 32), F(31, 32)),) * 2
    assert rel.tail_errors == ((ZERO, ZERO),) * 2
    assert new.exact_rows == old.exact_rows and new.stored_rows == old.stored_rows
    assert new.row_errors == old.row_errors and new.installed_rhs == old.installed_rhs
    assert _bytes(new.hz) == _bytes(old.hz)
    value = _assignment(case, _transport(1, -1, 1))
    assert new.canonical_extension(value) == old.canonical_extension(value)
    assert new.decode(new.canonical_extension(value)) == _transport(1, -1, 1)
    _record(2, all_29_exact_and_stored_rows_equal=True, no_guard_specific_path=True,
            legacy_test_module_imported_or_mutated=False)


def test_03_outside_guard_stored_certificate():
    case = _case()
    result, hz = case["result"], case["hz"]
    rel = result.relations[0]
    assert rel.event_bounds == ((F(-31, 32), F(33, 32)),) * 2
    assert rel.tail_errors == ((ZERO, F(1, 32)),) * 2
    assert result.exact_rows == result.stored_rows and all(v == 0 for v in result.row_errors)
    q, alpha = tuple(map(_af, rel.q)), tuple(map(_af, rel.alphas))
    _, _, d12, d21, v1, v2 = tuple(_af(p.value) for p in rel.products)
    p = tuple(_sum((1, aa), (-1, qq)) for aa, qq in zip(alpha, q))
    lam, a, b, eps, eta = F(3, 8), F(7, 8), F(1, 16), F(3, 32), F(1, 32)
    rows = result.stored_rows
    weighted = [(lam, rows[24]), (lam * a, rows[26]),
                (lam * b, _lookup(rows, _sum((1, d12), (-1, q[1])))),
                (lam * b, _lookup(rows, _sum((-1, d21))))]
    old_le = _native_rows(hz, False) + tuple(_row(_col(i), 1) for i in range(hz.n_cont + hz.n_bin)) + tuple(
        _row(_sum((-1, _col(i))), 1) for i in range(hz.n_cont + hz.n_bin))
    scale, row = _scaled_lookup(old_le, q[0], 1)
    weighted.append((lam * b * scale, row))
    residual_rows = (
        _lookup(rows, _sum((1, v1), (-eps, alpha[0]))),
        _lookup(rows, _sum((1, v1), (-1, _af(rel.r)), (eps, alpha[0])), eps),
        _lookup(rows, _sum((-1, v2), (-eps, alpha[1]))),
        _lookup(rows, _sum((1, _af(rel.r)), (-1, v2), (eps, alpha[1])), eps),
    )
    weighted.extend((lam / 2, row) for row in residual_rows)
    # The new tail is paid by the ACTUAL original signed bound -z2 <= 1.
    scale, row = _scaled_lookup(old_le, alpha[1], 1)
    weighted.append((lam * eta * scale, row))
    p_weights, n_weights = (F(13, 16), F(7, 64)), (F(19, 16), F(31, 64))
    eq_weighted, old_eq = [], _native_rows(hz, True)
    for i in range(2):
        scale, row = _scaled_lookup(old_le, _sum((-1, p[i])))
        weighted.append((p_weights[i] * scale, row))
        graph = case["gates"][i]["graph"]
        weighted.append((n_weights[i] / 2, _lookup(old_le, _col(graph.slots[0]), 1)))
        eq_weighted.append((n_weights[i], old_eq[graph.eq_row]))
    certificate = _combine(((1, _combine(weighted)), *eq_weighted), nonnegative=False)
    expected = _row(_flat(case["outputs"][case["F_index"]], hz.n_cont), F(265, 128))
    assert certificate == expected
    assert all(i < result.snapshot.n_columns for i, _ in certificate.terms)
    true = result.canonical_extension(_assignment(case, _transport(1, -1, 1)))
    assert _readout(case, case["F_index"], true) == F(527, 256)
    fake = _fake(case)
    assert _old_holds(hz, fake) and _readout(case, case["F_index"], fake) == F(8509, 4096)
    assert F(8509, 4096) - F(265, 128) == F(29, 4096)
    assert F(8509, 4096) - F(1061, 512) == F(21, 4096)
    assert F(265, 128) - F(1061, 512) == F(-1, 512)
    _record(3, exact_stored_certificate=True, upper="265/128", old_feasible="8509/4096",
            physical_gap="29/4096", all_aux_eliminated=True, bound_claimed_tight=False,
            successor_threshold="1061/512", old_successor="21/4096",
            tail_original_signed_bound_multiplier="3/512")


def test_04_complete_single_child_witnesses():
    case, u = _case(), F(127, 128)
    fake = _fake(case)
    witnesses = []
    for child, delta in ((0, F(3, 8)), (1, F(7, 104))):
        weights = (delta, F(1, 2) - delta, F(1, 2) - delta, delta)
        assert all(w >= 0 for w in weights)
        atoms = []
        for (a1, a2), weight in zip(((0, 0), (1, 0), (0, 1), (1, 1)), weights):
            value = _assignment(case, _transport(u * (2 * a1 - 1), u * (2 * a2 - 1)))
            assert _old_holds(case["hz"], value, integral=True)
            atoms.append((weight, value))
        mean = _average(atoms)
        indices = tuple(range(5)) + (5, 6, 7, 8, 9 + child)
        assert all(_readout(case, i, mean) == _readout(case, i, fake) for i in indices)
        assert all(mean[13 + i] == fake[13 + i] for i in (0, 1, 2 + child))
        assert _readout(case, 9 + child, mean) / u == F(17, 32) + F(13, 16) * delta
        witnesses.append(str(delta))
    _record(4, complete_input_and_original_labels=True, prefix_deltas=witnesses,
            separate_prefix_witnesses_not_joint_distribution=True)


def test_05_joint_children_convexified_parents():
    case, u = _case(), F(127, 128)
    atoms = []
    for alphas in ((ONE, F(1, 2)), (ZERO, F(1, 2)), (ZERO, ZERO), (ONE, ONE)):
        x1, x2 = (u * (2 * aa - 1) for aa in alphas)
        q1, q2 = (u * aa for aa in alphas)
        z = F(7, 8) * (x1 + x2) + F(1, 16) * (q1 + q2)
        g1, g2 = z + q1 - q2, z - q1 + q2
        assert g1 != 0 and g2 != 0
        value = _assignment(case, _transport(x1, x2), amplitudes=(q1, q2, max(0, g1), max(0, g2)),
                            active=(*alphas, F(g1 > 0), F(g2 > 0)))
        assert _old_holds(case["hz"], value)
        parent_atoms = []
        for a1, a2 in product((0, 1), repeat=2):
            weight = (alphas[0] if a1 else 1 - alphas[0]) * (alphas[1] if a2 else 1 - alphas[1])
            point = _assignment(case, _transport(u * (2 * a1 - 1), u * (2 * a2 - 1)))
            assert _old_holds(case["hz"], point, integral=True)
            parent_atoms.append((weight, point))
        parent_mean = _average(parent_atoms)
        assert all(_readout(case, i, value) == _readout(case, i, parent_mean) for i in range(9))
        assert value[13:15] == parent_mean[13:15]
        atoms.append((F(1, 4), value))
    mean, fake = _average(atoms), _fake(case)
    assert all(_readout(case, i, mean) == _readout(case, i, fake) for i in range(len(case["outputs"])))
    assert mean[13:17] == fake[13:17]
    _record(5, complete_convexified_parent_witnesses=True, actual_joint_child_graph=True,
            original_source_means_preserved=True, fake_point_is_not_ADV=True,
            full_cross_layer_RLT_beaten=False, all_retuned_hardmixed_beaten=False)


def test_06_event_tail_formulas_and_asymmetry():
    cases = (
        ("tails", ((F(-13, 16), F(19, 16)), (F(-21, 16), F(17, 16))),
         ((ZERO, F(3, 16)), (F(5, 16), F(1, 16)))),
        ("tails_shift", ((F(-21, 16), F(11, 16)), (F(-29, 16), F(9, 16))),
         ((F(5, 16), ZERO), (F(13, 16), ZERO))),
    )
    for name, events, tails in cases:
        case = _case(name, mode=name)
        result, rel = case["result"], case["result"].relations[0]
        assert rel.a == (F(5, 4), F(3, 4)) and rel.b == (F(-1, 8), F(1, 4))
        assert rel.event_bounds == events and rel.tail_errors == tails
        assert rel.mixed_bounds == (min(lo for lo, _ in events), max(hi for _, hi in events))
        for (lo, hi), (minus, plus) in zip(events, tails):
            assert (minus, plus) == (max(-lo - rel.tau, 0), max(hi - rel.tau, 0))
        alpha, q, yy = tuple(map(_af, rel.alphas)), tuple(map(_af, rel.q)), tuple(map(_af, rel.y))
        delta = _sum((1, yy[0]), (-1, yy[1]), (-1, _af(rel.K)))
        p = tuple(_sum((1, aa), (-1, qq)) for aa, qq in zip(alpha, q))
        assert result.exact_rows[24] == _row(_sum(
            (1, delta), (-2 * rel.tau, p[1]), (-tails[0][0], alpha[0]),
            (-tails[1][1], alpha[1])), 2 * max(rel.e_bounds[1], 0))
        assert result.exact_rows[25] == _row(_sum(
            (-1, delta), (-2 * rel.tau, p[0]), (-tails[0][1], alpha[0]),
            (-tails[1][0], alpha[1])), -2 * min(rel.e_bounds[0], 0))
        for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1),
                       _transport(1, 1, 1), _transport(-1, -1, -1)):
            value = _assignment(case, inputs)
            assert _old_holds(case["hz"], value, integral=True)
            assert result.satisfied(result.canonical_extension(value), integral=True)
    _record(6, asymmetric_event_bounds=True, upper_and_lower_tails=True,
            stored_formula_signs_checked=True, six_existing_products_unchanged=True)


def test_07_integer_extensions_and_zero_labels():
    case, count = _case(), 0
    for x1, x2, t in product((-1, 0, 1), repeat=3):
        inputs = _transport(x1, x2, t)
        value = _assignment(case, inputs)
        assert _old_holds(case["hz"], value, integral=True)
        extended = case["result"].canonical_extension(value)
        assert case["result"].satisfied(extended, integral=True)
        assert case["result"].decode(extended) == inputs
        count += 1
    for labels in product((ZERO, ONE), repeat=4):
        value = _assignment(case, _transport(0, 0), active=labels)
        assert _old_holds(case["hz"], value, integral=True)
        assert case["result"].satisfied(case["result"].canonical_extension(value), integral=True)
    with pytest.raises(nm.Rejected):
        case["result"].canonical_extension(_fake(case))
    asym = _case("asymmetric_scale", parent_L=F(-1))
    rel = asym["result"].relations[0]
    assert rel.scales == (F(2), F(2)) and rel.tau == 2
    for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1)):
        extended = asym["result"].canonical_extension(_assignment(asym, inputs))
        assert asym["result"].decode(extended) == inputs
    _record(7, integer_points=count, original_zero_labels=16, no_new_binary=True,
            normalization="positive_scale_not_translation", samples_are_not_ADV=True)


def test_08_complete_residual_and_source_identity():
    case = _case("residual", mode="residual")
    rel = case["result"].relations[0]
    assert rel.tau == F(65, 64) and rel.a == (F(3, 4),) * 2
    assert rel.b == (F(1, 16), F(1, 32))
    assert rel.r_bounds == (F(-9, 256), F(15, 256))
    assert rel.e_bounds == (F(1, 256), F(9, 256))
    expected_r = _expr(F(3, 256), ((0, F(1, 128)), (1, F(1, 128)), (4, F(1, 32))))
    expected_e = _sum((F(1, 64), _af(rel.q[0])), (F(1, 64), _af(rel.q[1])), (1, _expr(F(1, 256))))
    assert _af(rel.r) == expected_r and _af(rel.e) == expected_e
    for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1), (F(1), F(1), ZERO, ZERO, ZERO)):
        extended = case["result"].canonical_extension(_assignment(case, inputs))
        assert case["result"].decode(extended) == inputs
    negative = _case("negative_t", mode="negative_t")
    main_rel, neg_rel = _case()["result"].relations[0], negative["result"].relations[0]
    assert main_rel.r_bounds == neg_rel.r_bounds and main_rel.event_bounds == neg_rel.event_bounds
    assert _af(main_rel.r) != _af(neg_rel.r)
    assert _af(neg_rel.r) == _sum((-1, _af(main_rel.r)))
    inputs = _transport(1, -1, 1)
    val = negative["result"].canonical_extension(_assignment(negative, inputs))
    assert _at(_af(neg_rel.products[4].value), val) == F(-3, 32)
    _record(8, complete_r_e=True, source_sum_not_cancelled=True, derived_tau="65/64",
            same_range_opposite_source_not_aliased=True)


def test_09_preserved_predicates_and_decoder():
    case = _case("predicates", predicates=True)
    old, new = case["snapshot"].hz, case["result"].hz
    assert np.array_equal(old.c, new.c) and np.array_equal(old.b, new.b)
    assert np.array_equal(old.ub, new.ub[:len(old.ub)])
    for original, copied in ((old.Gc, new.Gc), (old.Ac, new.Ac), (old.Auc, new.Auc[:len(old.ub)])):
        assert (original != copied[:, :old.n_cont]).nnz == 0
        assert copied[:, old.n_cont:].nnz == 0
    for original, copied in ((old.Gb, new.Gb), (old.Ab, new.Ab), (old.Aub, new.Aub[:len(old.ub)])):
        assert (original != copied).nnz == 0
    inputs = _transport(F(1, 2), F(-1, 2))
    extended = case["result"].canonical_extension(_assignment(case, inputs))
    assert case["result"].decode(extended) == inputs
    with pytest.raises(nm.Rejected):
        case["result"].canonical_extension(_assignment(case, _transport(0, 0, 1)))
    _record(9, complete_old_eq_le_readouts=True, decoder_coordinates=5, skip_preserved=True)


def test_10_whole_population_and_native_widths():
    case = _fixture(groups=2)
    snap, result = _attach(case, widths=(23, 10))
    assert snap.original_dimensions == (21, 8) and snap.n_columns == 33
    assert len(result.relations) == 2 and result.n_columns == 45
    assert result.hz.n_cont == 35 and result.hz.n_bin == 10 and len(result.exact_rows) == 58
    value = _assignment(case, _transport(1, -1, 1), widths=(23, 10))
    assert result.decode(result.canonical_extension(value)) == _transport(1, -1, 1)
    before = _bytes(snap.hz)
    with pytest.raises(nm.Rejected):
        nm.attach(snap, population=(case["population"][0], case["population"][0]), enabled=True)
    wrong = (case["population"][1][0], (case["population"][1][1][0], case["population"][1][1][0]))
    with pytest.raises(nm.Rejected):
        nm.attach(snap, population=(case["population"][0], wrong), enabled=True)
    assert _bytes(snap.hz) == before and snap.budget._failure is None
    assert result.satisfied(result.canonical_extension(value), integral=True)
    _record(10, complete_pairs=2, reserved_old_factors=[23, 10], aux=12, partial_batch_return=False)


def test_11_compact_and_malformed_graphs():
    compact = _case("compact", compact=True)
    assert compact["hz"].n_cont == 9 and len(compact["hz"].b) == 0
    for x1, x2 in product((-1, 0, 1), repeat=2):
        value = _assignment(compact, _transport(x1, x2))
        assert _old_holds(compact["hz"], value, integral=True)
        compact["result"].canonical_extension(value)
    bad = _fixture(compact=True)
    bad["hz"].ub[bad["gates"][0]["graph"].le_rows[2]] += 1 / 64
    snap = nm.capture(bad["hz"], bad["decoder"], global_widths=(9, 4), budget=nm.Budget(), enabled=True)
    with pytest.raises(nm.Rejected, match="nonzero native gate error"):
        nm.attach(snap, population=bad["population"], enabled=True)
    case = _case()
    graph = replace(case["population"][0][0][0], eq_row=1)
    with pytest.raises(nm.Rejected):
        nm.attach(case["snapshot"], population=(((graph, case["population"][0][0][1]), case["population"][0][1]),), enabled=True)
    _record(11, exact_compact=True, nonzero_error_rejected=True, actual_row_identity_checked=True)


def test_12_unsupported_inputs_and_rank_dependency():
    for mode in ("rank", "dependent", "zero_tau"):
        case = _fixture(mode=mode)
        snap = nm.capture(case["hz"], case["decoder"], global_widths=(13, 4), budget=nm.Budget(), enabled=True)
        before = _bytes(snap.hz)
        with pytest.raises(nm.Rejected):
            nm.attach(snap, population=case["population"], enabled=True)
        assert _bytes(snap.hz) == before and snap.budget._failure is None
    case = _fixture()
    for decoder, widths in ((case["decoder"], (12, 4)), (list(case["decoder"]), (13, 4))):
        with pytest.raises(nm.Rejected):
            nm.capture(case["hz"], decoder, global_widths=widths, budget=nm.Budget(), enabled=True)
    foreign = legacy.capture(case["hz"], case["decoder"], global_widths=(13, 4),
                             budget=nm.Budget(), enabled=True)
    with pytest.raises(nm.Rejected, match="owned"):
        nm.attach(foreign, population=case["population"], enabled=True)
    assert foreign.budget._failure is None
    # An ordinary formerly unsupported range is now paid, not retried.
    broad = _case("guard", mode="guard")
    rr = broad["result"].relations[0]
    assert rr.tail_errors == ((F(1, 8), F(1, 4)),) * 2
    point = _assignment(broad, _transport(1, -1, 1))
    assert broad["result"].satisfied(broad["result"].canonical_extension(point), integral=True)
    _record(12, rank_dependency_nonpositive_tau_rejected=True, foreign_owner_rejected=True,
            truncated_global_width_rejected=True, ordinary_guard_paid_without_retry=True)


def test_13_outward_rounding_and_snapshot_integrity():
    case = _case("rounding", mode="rounding")
    rel, result = case["result"].relations[0], case["result"]
    assert rel.a == (F(1, 2), F(25, 48)) and rel.r_bounds == (F(-5, 96), F(5, 96))
    assert any(error > 0 for error in result.row_errors)
    for exact, stored, error, installed in zip(result.exact_rows, result.stored_rows,
                                              result.row_errors, result.installed_rhs):
        left, right = dict(exact.terms), dict(stored.terms)
        actual_error = sum((abs(left.get(i, ZERO) - right.get(i, ZERO)) for i in set(left) | set(right)), ZERO)
        assert error == actual_error and stored.rhs == installed and installed >= exact.rhs + error
    for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1), (F(1, 2),) * 4 + (ZERO,)):
        extended = result.canonical_extension(_assignment(case, inputs))
        assert result.satisfied(extended, integral=True)
    isolated = _fixture()
    snap, copied = _attach(isolated)
    old = _bytes(snap.hz)
    isolated["hz"].c[0] += 1
    assert _bytes(snap.hz) == old
    with pytest.raises(ValueError):
        snap.hz.c[0] = 0
    assert copied.satisfied(copied.canonical_extension(_assignment(isolated, _transport(0, 0))), integral=True)
    tampered = _fixture()
    ss, rr = _attach(tampered, budget=nm.Budget())
    ss.hz.frame_id += 1
    with pytest.raises(nm.Rejected, match="mutated"):
        rr.terminal_model()
    bad = _fixture()
    bad["hz"].Ac = sp.vstack((bad["hz"].Ac, sp.csr_matrix(([1.0], ([0], [4])), shape=(1, 13))), format="csr")
    bad["hz"].Ab = sp.vstack((bad["hz"].Ab, sp.csr_matrix(([0.1, 0.2], ([0, 0], [0, 1])), shape=(1, 4))), format="csr")
    bad["hz"].b = np.concatenate((bad["hz"].b, np.array([0.0], dtype=np.float64)))
    _, rejected = _attach(bad, budget=nm.Budget())
    with pytest.raises(nm.Rejected, match="translation is not exact"):
        rejected.terminal_model()
    assert rejected.budget._failure is None
    _record(13, nondyadic_gram="25/48", all_coefficient_errors_paid=True,
            source_isolation=True, frame_tamper_rejected=True,
            inexact_normal_signed_translation_rejected=True, second_solver_path=False)


def _normal_lp(hz, model, objective_index):
    count = len(hz.b)
    objective = -model.value_matrix.getrow(objective_index).toarray().ravel()
    answer = linprog(objective, A_ub=model.A[count:], b_ub=model.row_ub[count:],
        A_eq=model.A[:count], b_eq=model.row_ub[:count],
        bounds=list(zip(model.var_lb, model.var_ub)), method="highs", options={"time_limit": 5.0})
    assert answer.success, answer.message
    return float(model.value_center[objective_index] - answer.fun)


def test_14_complete_cost_and_ordinary_terminal():
    case = _case()
    result, old = case["result"], case["snapshot"].hz
    assert result.hz.n_cont == old.n_cont + 6 and result.hz.n_bin == old.n_bin == 4
    assert len(result.hz.ub) == len(old.ub) + 29 and len(result.hz.b) == len(old.b)
    assert len(result.exact_rows) == 29 and len(result.relations[0].products) == 6
    assert sum(len(row.terms) for row in result.stored_rows) == 104
    model = result.terminal_model()
    assert tuple(model.cont_source) == tuple(range(19)) and tuple(model.bin_source) == tuple(range(4))
    assert not model.bin_fixes and not model.cont_eliminations
    assert tuple(model.integrality) == (0,) * 19 + (1,) * 4
    plain = _lower_hz_milp(old, prune_unused=False, coalesce_rows=False,
                         project_inactive_cont=False, fix_implied_binary=False)
    old_bound = _normal_lp(old, plain, case["F_index"])
    new_bound = _normal_lp(result.hz, model, case["F_index"])
    assert old_bound >= float(F(8509, 4096)) - 1e-8
    assert new_bound <= float(F(265, 128)) + 1e-8
    assert new_bound >= float(F(527, 256)) - 1e-8
    assert new_bound < float(F(1061, 512))
    _record(14, new_continuous=6, new_LE=29, actual_new_nnz=104,
            complete_old_binary=4, complete_EQ=len(result.hz.b), complete_LE=len(result.hz.ub),
            complete_predicate_nnz=int(result.hz.Ac.nnz + result.hz.Ab.nnz + result.hz.Auc.nnz + result.hz.Aub.nnz),
            fixed_normal_LPs=2, old_LP=old_bound, new_LP=new_bound, bound_claimed_tight=False,
            adaptive_solver_or_rescue=False, physical_resource_qualified=False)


def test_15_shared_sticky_budget():
    case = _fixture()
    for keywords in ({"max_work": 0}, {"max_entries": 0}, {"max_branch": 0}):
        budget = nm.Budget(**keywords)
        with pytest.raises(nm.Rejected):
            nm.capture(case["hz"], case["decoder"], global_widths=(13, 4), budget=budget, enabled=True)
        assert budget._failure is not None
        with pytest.raises(nm.Rejected):
            nm.capture(case["hz"], case["decoder"], global_widths=(13, 4), budget=budget, enabled=True)
    with pytest.raises(nm.Rejected):
        nm.Budget(max_bits=513)
    before = (_budget().work, _budget().entries)
    case = _case()
    case["result"].canonical_extension(_assignment(case, _transport(0, 0)))
    assert _budget().work > before[0] and _budget().entries > before[1]
    assert _budget()._failure is None
    _record(15, ordinary_positive_fixtures_share_budget=True, sticky_resource_failure=True,
            work=_budget().work, entries=_budget().entries)


def test_16_summary_and_qualification_boundary():
    assert set(_EVIDENCE) == set(_NAMES[:-1])
    _record(16, original_baseline=1870, original_population=2413, independent_score=61,
            formal_gain=0, new_solves=0, native_HZ_admitted=False, new_domain_qualified=False,
            new_capability_qualified=False, actual_model_qualified=False, online_qualified=False,
            gpu_qualified=False, physical_resource_qualified=False, domain_definition_changed=False,
            local_mathematical_tests_complete=True, ordinary_terminal_not_rescue=True)
    _record_file("summary.json", {"schema": "d253_quantified_native_v1", "status": "PASS",
        "tests": _EVIDENCE, "test_count": 16,
        "shared_budget": {"work": _budget().work, "entries": _budget().entries},
        "scope": "detached quantified native mathematical fixture, not actual-model admission"})
