"""Static mathematical tests of actual production rebase and native transport.

The only LP calls are the two fixed ordinary terminal comparisons in test 14.
Constructed native inputs are not model admission or verified adversarial data.
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
import torch
from scipy.optimize import linprog

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono, _lower_hz_milp, sparse_hz_rebase_image_exact
from experiments.neural_hz_20260831.definition_first_20260928.d255_rebase_relation_transport_20261006 import native_rebased as nm
from experiments.neural_hz_20260831.definition_first_20260928.d254_quantified_native_component_20261006 import native_quantified as legacy

ZERO, ONE = F(0), F(1)
RUN = Path(__file__).resolve().parents[2] / "results/d255_rebase_relation_transport_20261006_v1"
_NAMES = (
    "default_off_and_empty_bindings",
    "production_rebase_receipt",
    "rebased_main_relation_restored",
    "original_signed_witnesses_and_decoder",
    "stored_offset_preserved",
    "multiepoch_normalization",
    "binary_source_and_constant_link",
    "old_predicates_outputs_and_highwater",
    "reordered_row_binding",
    "receipt_identity_rejections",
    "structure_and_private_dependencies",
    "fill_and_bit_budget_fail_closed",
    "rounding_and_terminal_preservation",
    "physical_certificate_and_fixed_lp",
    "shared_budget_and_atomic_batch",
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



def _state(nc, nb, outputs, equations=(), rhs=(), guards=(), upper=(), *, frame=255):
    return SparseHZono(np.array([float(v.bias) for v in outputs], dtype=np.float64),
        _matrix(outputs, nc, "continuous"), _matrix(outputs, nb, "binary"),
        _matrix(equations, nc, "continuous"), _matrix(equations, nb, "binary"),
        np.array([float(v) for v in rhs], dtype=np.float64),
        _matrix(guards, nc, "continuous"), _matrix(guards, nb, "binary"),
        np.array([float(v) for v in upper], dtype=np.float64), frame_id=frame, exact=True)


def _raw_forms(ac, ab, constants):
    result = []
    for row, constant in enumerate(constants):
        parts = []
        for mat in (ac, ab):
            parts.append(tuple((int(mat.indices[k]), F.from_float(float(mat.data[k])))
                               for k in range(mat.indptr[row], mat.indptr[row + 1])))
        result.append(_nf(F.from_float(float(constant)), *parts))
    return tuple(result)


def _outputs(hz):
    return _raw_forms(hz.Gc, hz.Gb, hz.c)


def _produce_rebase(before, lower, upper):
    bounds = Bounds(torch.tensor([float(v) for v in lower], dtype=torch.float64),
                    torch.tensor([float(v) for v in upper], dtype=torch.float64))
    return sparse_hz_rebase_image_exact(before, bounds)


class _Builder:
    def __init__(self, budget, *, predicates=False):
        self.budget, self.nc, self.nb = budget, 5, 0
        self.inputs = tuple(_nf(F(1, 2), ((i, F(1, 2)),)) for i in range(4)) + (
            _nf(0, ((4, ONE),)),)
        self.eq, self.rhs, self.le, self.upper = [], [], [], []
        self.gates, self.events, self.rebases, self.bindings = [], [], [], []
        if predicates:
            self.eq.append(_nf(0, ((0, ONE), (1, ONE))))
            self.rhs.append(ZERO)
            self.le.append(_nf(0, ((4, ONE),)))
            self.upper.append(F(3, 4))

    def gate(self, f, L=F(-1, 2), Q=F(1, 2)):
        s, eta, bit = self.nc, self.nc + 1, self.nb
        self.nc += 2
        self.nb += 1
        q = _nf(Q, ((eta, -Q),))
        eq = _nadd((-ONE, f), (ONE, _nf(0, ((s, L), (eta, -Q)), ((bit, L),))))
        row, start = len(self.eq), len(self.le)
        self.eq.append(_nf(0, eq.continuous, eq.binary))
        self.rhs.append(-Q - eq.bias)
        self.le.extend((_nf(0, ((s, -ONE),), ((bit, -ONE),)),
                        _nf(0, ((eta, -ONE),), ((bit, ONE),))))
        self.upper.extend((ZERO, ZERO))
        graph = nm.Graph("extended", (s, eta, bit), row, (start, start + 1))
        self.events.append(("gate", len(self.gates)))
        self.gates.append({"f": f, "q": q, "graph": graph, "L": L, "Q": Q})
        return q, graph

    def state(self, outputs):
        return _state(self.nc, self.nb, outputs, self.eq, self.rhs, self.le, self.upper)

    def rebase(self, outputs):
        before = self.state(outputs)
        after = _produce_rebase(before, (ZERO,) * len(outputs), (ONE,) * len(outputs))
        receipt = nm.record_rebase(before, after, budget=self.budget, enabled=True)
        self.nc, self.nb = after.n_cont, after.n_bin
        self.eq = list(_raw_forms(after.Ac, after.Ab, np.zeros(len(after.b))))
        self.rhs = [F.from_float(float(v)) for v in after.b]
        self.le = list(_raw_forms(after.Auc, after.Aub, np.zeros(len(after.ub))))
        self.upper = [F.from_float(float(v)) for v in after.ub]
        self.events.append(("rebase", len(self.rebases)))
        self.rebases.append((before, after, receipt))
        self.bindings.append(nm.RebaseBinding(receipt, receipt.epoch.eq_rows))
        return _outputs(after)


def _fixture(*, epochs=1, mode="main", predicates=False, budget=None, hidden=False):
    meter = budget if budget is not None else _budget()
    builder = _Builder(meter, predicates=predicates)
    f1 = _nf(0, ((0, F(1, 2)), (1, F(-1, 2))))
    f2 = _nf(0, ((2, F(1, 2)), (3, F(-1, 2))))
    if mode == "rounding":
        f2 = _nf(0, ((0, F(1, 4)), (1, F(1, 4)), (2, F(1, 4))))
    q1, p1 = builder.gate(f1)
    if hidden:
        q1, = builder.rebase((q1,))
        f2 = _nadd((F(1, 2), q1), (ONE, _nf(0, ((2, F(1, 4)),))))
    q2, p2 = builder.gate(f2)
    if not hidden:
        for _ in range(epochs):
            q1, q2 = builder.rebase((q1, q2))
    a, b, eps = F(7, 8), F(1, 16), F(3, 32)
    if mode == "rounding":
        a, b, eps = F(1, 2), F(1, 32), F(1, 32)
    z = _nadd((a, f1), (a, f2), (b, q1), (b, q2), (eps, builder.inputs[4]))
    if mode == "rounding":
        z = _nadd((ONE, z), (ONE, _nf(0, ((2, F(1, 64)),))))
    y1, c1 = builder.gate(_nadd((ONE, z), (ONE, q1), (-ONE, q2)), F(-2), F(2))
    y2, c2 = builder.gate(_nadd((ONE, z), (-ONE, q1), (ONE, q2)), F(-2), F(2))
    objective = _nadd((F(2), q1), (F(2), q2), (F(-19, 16), f1), (F(-13, 16), f2),
                     (F(3, 8), y1), (F(-3, 8), y2))
    outputs = builder.inputs + (f1, f2, q1, q2, y1, y2, z, objective,
                                _nadd((ONE, z), (ONE, q1)))
    return {"hz": builder.state(outputs), "decoder": builder.inputs,
            "population": (((p1, p2), (c1, c2)),), "gates": tuple(builder.gates),
            "events": tuple(builder.events), "rebases": tuple(builder.rebases),
            "bindings": tuple(builder.bindings), "budget": meter,
            "outputs": outputs, "F_index": 12}


def _attach(case, *, widths=None, bindings=None):
    hz = case["hz"]
    snapshot = nm.capture(hz, case["decoder"], global_widths=widths or (hz.n_cont, hz.n_bin),
        bindings=case["bindings"] if bindings is None else bindings,
        budget=case["budget"], enabled=True)
    return snapshot, nm.attach(snapshot, population=case["population"], enabled=True)


def _case(name="main", **kwargs):
    if name not in _CASES:
        case = _fixture(**kwargs)
        case["snapshot"], case["result"] = _attach(case)
        _CASES[name] = case
    return _CASES[name]


def _transport(x1, x2, t=0):
    return ((1 + F(x1)) / 2, (1 - F(x1)) / 2,
            (1 + F(x2)) / 2, (1 - F(x2)) / 2, F(t))


def _assignment(case, inputs, *, amplitudes=None, active=None, widths=None):
    hz = case["hz"]
    nc, nb = widths or (hz.n_cont, hz.n_bin)
    cont, bits = [ZERO] * nc, [ONE] * nb
    cont[:4] = [2 * F(v) - 1 for v in inputs[:4]]
    cont[4] = F(inputs[4])
    for kind, index in case["events"]:
        if kind == "gate":
            gate = case["gates"][index]
            f = _nv(gate["f"], cont, bits)
            q = max(ZERO, f) if amplitudes is None else F(amplitudes[index])
            alpha = (ONE if f > 0 else ZERO) if active is None else F(active[index])
            s, eta, bit = gate["graph"].slots
            bits[bit] = 1 - 2 * alpha
            cont[eta] = 1 - q / gate["Q"]
            cont[s] = (f - q) / gate["L"] - bits[bit]
        else:
            _, _, receipt = case["rebases"][index]
            for col, (row, rhs) in zip(range(receipt.epoch.old_cont, receipt.epoch.new_cont),
                                       receipt.link_rows):
                pivot = dict(row.continuous)[col]
                rest = _nf(row.bias, tuple((i, a) for i, a in row.continuous if i != col), row.binary)
                cont[col] = (rhs - _nv(rest, cont, bits)) / pivot
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


def _readout(case, index, values):
    return _at(_flat(case["outputs"][index], case["hz"].n_cont), values)


def _fake(case):
    u = F(127, 128)
    return _assignment(case, _transport(0, 0),
                       amplitudes=(u / 2, u / 2, F(107, 128) * u, F(75, 128) * u),
                       active=(F(1, 2),) * 4)


def _certificate(case):
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
    # Each actual link is q_old - q_display = 0. Keeping its complete
    # RHS is necessary; the main dyadic case has zero stored offset.
    for binding in case["bindings"]:
        for index in binding.eq_rows:
            eq_weighted.append((F(-2), old_eq[index]))
    certificate = _combine(((1, _combine(weighted)), *eq_weighted), nonnegative=False)
    expected = _row(_flat(case["outputs"][case["F_index"]], hz.n_cont), F(265, 128))
    assert certificate == expected
    assert all(i < result.snapshot.n_columns for i, _ in certificate.terms)
    return certificate



def test_01_default_off_and_empty_bindings():
    assert nm.record_rebase(object(), object()) is None
    assert nm.capture(object(), object(), global_widths=object()) is None
    assert nm.normalize(object(), object()) is None
    assert nm.attach(object(), population=object()) is None
    with pytest.raises(nm.Rejected):
        nm.record_rebase(None, None, enabled=1)
    case = _case("unrebased", epochs=0)
    snap, new = case["snapshot"], case["result"]
    assert snap.definitions == () and dict(snap.normal_forms) == {}
    previous = legacy.capture(case["hz"], case["decoder"], global_widths=(13, 4),
                              budget=_budget(), enabled=True)
    old = legacy.attach(previous, population=case["population"], enabled=True)
    assert new.exact_rows == old.exact_rows and new.stored_rows == old.stored_rows
    assert _bytes(new.hz) == _bytes(old.hz)
    assert _af(nm.normalize(snap, case["outputs"][11], enabled=True)) == _flat(case["outputs"][11], 13)
    _record(1, default_off=True, empty_bindings_preserve_all_legacy_rows=True,
            old_test_modules_imported_or_mutated=False)


def test_02_production_rebase_receipt():
    case = _case()
    before, after, receipt = case["rebases"][0]
    assert (before.n_cont, after.n_cont, before.n_bin, after.n_bin) == (9, 11, 2, 2)
    assert before.n_out == after.n_out == 2
    assert receipt.epoch == nm.LinkEpoch(9, 11, (2, 3))
    assert receipt.offsets == (ZERO, ZERO) and receipt.all_added_eq == 2
    assert receipt.budget is case["budget"] is _budget()
    assert receipt.before_fingerprint != receipt.after_fingerprint
    assert receipt.frame_id == before.frame_id == after.frame_id
    for j, (form, rhs) in enumerate(receipt.link_rows):
        eta = case["gates"][j]["graph"].slots[1]
        assert form == _nf(0, ((eta, F(-1, 2)), (9 + j, F(-1, 2)))) and rhs == 0
    assert _native_rows(after, True)[:2] == tuple(
        _row(_expr(0, tuple((i + 2 if i >= 9 else i, a) for i, a in row.terms)), row.rhs)
        for row in _native_rows(before, True))
    assert _native_rows(after, False) == tuple(
        _row(_expr(0, tuple((i + 2 if i >= 9 else i, a) for i, a in row.terms)), row.rhs)
        for row in _native_rows(before, False))
    assert len(case["snapshot"].definitions) == 2
    with pytest.raises(TypeError):
        case["snapshot"].normal_forms[9] = None
    _record(2, actual_production_function_called=True, image_columns=2, link_EQ=2,
            complete_before_layer_outputs=2, reliable_bounds_independently_chosen=True,
            receipt_proves_ONNX_identity=False, receipt_proves_bound_soundness=False)


def test_03_rebased_main_relation_restored():
    case, direct = _case(), _case("unrebased", epochs=0)
    old = legacy.capture(case["hz"], case["decoder"], global_widths=(15, 4),
                         budget=_budget(), enabled=True)
    with pytest.raises(nm.Rejected, match="tau"):
        legacy.attach(old, population=case["population"], enabled=True)
    assert old.budget._failure is None
    result, rel = case["result"], case["result"].relations[0]
    assert rel.tau == 1 and rel.scales == (ONE, ONE)
    assert rel.a == (F(7, 8),) * 2 and rel.b == (F(1, 16),) * 2
    assert rel.r_bounds == (F(-3, 32), F(3, 32)) and rel.e_bounds == (ZERO, ZERO)
    assert rel.event_bounds == ((F(-31, 32), F(33, 32)),) * 2
    assert rel.tail_errors == ((ZERO, F(1, 32)),) * 2
    # Image columns remain in H, but no normalized factor row depends on them.
    projected = []
    for row in result.exact_rows:
        assert all(i not in (9, 10) for i, _ in row.terms)
        projected.append(_row(_expr(0, tuple((i - 2 if i >= 11 else i, a)
                                            for i, a in row.terms)), row.rhs))
    assert tuple(projected) == direct["result"].exact_rows
    certificate = _certificate(case)
    fake = _fake(case)
    assert _old_holds(case["hz"], fake)
    assert _at(_expr(0, certificate.terms), fake) - certificate.rhs == F(29, 4096)
    _record(3, old_direct_eta_rule_rejects_tau_zero=True, normalized_tau="1",
            same_mathematical_relation_restored=True, all_aux_eliminated=True,
            actual_link_EQ_used=True, fake_is_ADV=False)


def test_04_original_signed_witnesses_and_decoder():
    case, count = _case(), 0
    result = case["result"]
    for x1, x2, t in product((-1, 0, 1), repeat=3):
        inputs = _transport(x1, x2, t)
        old = _assignment(case, inputs)
        assert _old_holds(case["hz"], old, integral=True)
        extended = result.canonical_extension(old)
        assert extended[:len(old)] == old and result.satisfied(extended, integral=True)
        assert result.decode(extended) == inputs
        count += 1
    for labels in product((0, 1), repeat=4):
        old = _assignment(case, _transport(0, 0), active=labels)
        assert _old_holds(case["hz"], old, integral=True)
        extended = result.canonical_extension(old)
        assert result.satisfied(extended, integral=True)
        assert result.decode(extended) == _transport(0, 0)
    _record(4, fixed_integer_inputs=count, all_zero_label_combinations=16,
            original_bits_retained=4, complete_decoder_coordinates=5,
            candidate_sampling_or_attack=False)


def _offset_pair():
    before = _state(1, 0, (_nf(F.from_float(0.1), ((0, F(1, 16)),)),))
    return before, _produce_rebase(before, (ZERO,), (ONE,))


def test_05_stored_offset_preserved():
    before, after = _offset_pair()
    receipt = nm.record_rebase(before, after, budget=_budget(), enabled=True)
    assert receipt.offsets == (F(-1, 1 << 55),)
    assert receipt.link_rows[0][1] == F.from_float(0.4)
    snap = nm.capture(after, (_nf(0, ((0, ONE),)),), global_widths=(2, 0),
        bindings=(nm.RebaseBinding(receipt, receipt.epoch.eq_rows),), budget=_budget(), enabled=True)
    display = _outputs(after)[0]
    normal = nm.normalize(snap, display, enabled=True)
    expected = _expr(F.from_float(0.1) - F(1, 1 << 55), ((0, F(1, 16)),))
    assert _af(normal) == expected
    assert _af(normal) != _flat(_outputs(before)[0], 2)
    for source in (-1, 0, 1):
        image = 2 * (F(source, 16) - F.from_float(0.4))
        values = (F(source), image)
        assert _old_holds(after, values, integral=True)
        assert _at(_af(normal), values) == _at(_flat(display, 2), values)
    _record(5, production_stored_offset="-1/36028797018963968",
            stored_RHS_exactly_kept=True, before_after_set_equality_claimed=False)


def test_06_multiepoch_normalization():
    case = _case("two_epochs", epochs=2)
    assert tuple(r.epoch for _, _, r in case["rebases"]) == (
        nm.LinkEpoch(9, 11, (2, 3)), nm.LinkEpoch(11, 13, (4, 5)))
    assert len(case["snapshot"].definitions) == 4
    first, second = case["snapshot"].definitions[0], case["snapshot"].definitions[2]
    assert _af(second.raw) == _col(9)
    assert second.normal == first.normal
    assert _af(nm.normalize(case["snapshot"], case["outputs"][7], enabled=True)) == _flat(
        case["gates"][0]["q"], case["hz"].n_cont)
    for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1), _transport(0, 0)):
        value = _assignment(case, inputs)
        assert case["result"].satisfied(case["result"].canonical_extension(value), integral=True)
    _certificate(case)
    with pytest.raises(nm.Rejected):
        nm.capture(case["hz"], case["decoder"], global_widths=(17, 4),
                   bindings=tuple(reversed(case["bindings"])), budget=_budget(), enabled=True)
    assert _budget()._failure is None
    _record(6, ordered_epochs=2, preserved_image_columns=4, complete_link_EQ=4,
            cached_normal_forms_exact=True, reversed_epochs_rejected=True)


def test_07_binary_source_and_constant_link():
    before = _state(2, 1, (_nf(0, ((0, F(1, 4)),), ((0, F(1, 4)),)),
                            _nf(0, ((0, ONE), (1, -ONE)))),
                    (_nf(0, ((0, ONE), (1, -ONE))),), (ZERO,))
    after = _produce_rebase(before, (F(-1, 2), ZERO), (F(1, 2), ZERO))
    receipt = nm.record_rebase(before, after, budget=_budget(), enabled=True)
    assert receipt.epoch == nm.LinkEpoch(2, 3, (1,))
    assert receipt.all_added_eq == 2 and len(after.b) == 3
    assert receipt.offsets == (ZERO, ZERO)
    decoder = (_nf(0, ((0, ONE),)), _nf(0, ((1, ONE),)), _nf(0, (), ((0, ONE),)))
    snap = nm.capture(after, decoder, global_widths=(3, 1),
        bindings=(nm.RebaseBinding(receipt, (1,)),), budget=_budget(), enabled=True)
    assert _af(snap.definitions[0].normal) == _expr(0, ((0, F(1, 2)), (3, F(1, 2))))
    normalized = nm.normalize(snap, _outputs(after)[0], enabled=True)
    assert _af(normalized) == _expr(0, ((0, F(1, 4)), (3, F(1, 4))))
    for source, bit in product((-1, 0, 1), (-1, 1)):
        values = (F(source), F(source), F(source + bit, 2), F(bit))
        assert _old_holds(after, values, integral=True)
        assert _at(_af(normalized), values) == F(source + bit, 4)
        assert tuple(_at(_af(v), values) for v in snap.decoder) == (source, source, bit)
    _record(7, original_binary_source=True, no_binary_pivot=True,
            variable_links=1, all_added_EQ=2, nonpivot_constant_link_preserved=True)


def test_08_old_predicates_outputs_and_highwater():
    case = _case("predicates", predicates=True)
    old, new = case["hz"], case["result"].hz
    for name in ("Ac", "Ab", "Auc", "Aub", "Gc", "Gb"):
        left, right = getattr(old, name), getattr(new, name)
        assert (left != right[:left.shape[0], :left.shape[1]]).nnz == 0
    assert np.array_equal(new.b, old.b) and np.array_equal(new.ub[:len(old.ub)], old.ub)
    assert np.array_equal(new.c, old.c)
    value = _assignment(case, _transport(F(1, 2), F(-1, 2), F(1, 2)))
    assert _old_holds(old, value, integral=True)
    assert case["result"].decode(case["result"].canonical_extension(value)) == _transport(
        F(1, 2), F(-1, 2), F(1, 2))
    main = _case()
    snap, extended = _attach(main, widths=(17, 5))
    point = _assignment(main, _transport(1, -1, 1), widths=(17, 5))
    assert snap.original_dimensions == (15, 4) and (snap.hz.n_cont, snap.hz.n_bin) == (17, 5)
    assert extended.satisfied(extended.canonical_extension(point), integral=True)
    assert extended.decode(extended.canonical_extension(point)) == _transport(1, -1, 1)
    assert extended.hz.n_cont == 23 and extended.hz.n_bin == 5
    _record(8, complete_old_EQ_LE_outputs_preserved=True, decoder_unchanged=True,
            global_highwater=[17, 5], unused_original_columns_retained=True)


def test_09_reordered_row_binding():
    original = _case()
    case = dict(original)
    hz = original["hz"]
    order = tuple(reversed(range(len(hz.b))))
    positions = {old: new for new, old in enumerate(order)}
    case["hz"] = replace(hz, Ac=hz.Ac[list(order)].copy(), Ab=hz.Ab[list(order)].copy(),
                         b=hz.b[list(order)].copy())
    remap = lambda graph: replace(graph, eq_row=positions[graph.eq_row])
    case["population"] = tuple((tuple(map(remap, pp)), tuple(map(remap, cc)))
                                for pp, cc in original["population"])
    case["gates"] = tuple({**g, "graph": remap(g["graph"])} for g in original["gates"])
    case["bindings"] = tuple(nm.RebaseBinding(b.receipt, tuple(positions[i] for i in b.eq_rows))
                              for b in original["bindings"])
    snap, result = _attach(case)
    assert result.exact_rows == original["result"].exact_rows
    assert tuple(d.eq_row for d in snap.definitions) == (3, 2)
    point = _assignment(case, _transport(1, -1, 1))
    assert result.satisfied(result.canonical_extension(point), integral=True)
    with pytest.raises(nm.Rejected, match="content"):
        nm.capture(case["hz"], case["decoder"], global_widths=(15, 4),
                   bindings=original["bindings"], budget=_budget(), enabled=True)
    assert _budget()._failure is None
    _record(9, actual_EQ_content_rebound=True, old_row_numbers_not_trusted=True,
            complete_normalized_relation_unchanged=True)


def test_10_receipt_identity_rejections():
    case = _case()
    before, after, receipt = case["rebases"][0]
    with pytest.raises(nm.Rejected):
        nm.record_rebase(before, replace(after, frame_id=after.frame_id + 1),
                         budget=nm.Budget(), enabled=True)
    wrong_rhs = after.b.copy()
    wrong_rhs[0] += 1 / 64
    with pytest.raises(nm.Rejected):
        nm.record_rebase(before, replace(after, b=wrong_rhs), budget=nm.Budget(), enabled=True)
    with pytest.raises(nm.Rejected, match="shared Budget"):
        nm.capture(case["hz"], case["decoder"], global_widths=(15, 4),
                   bindings=case["bindings"], budget=nm.Budget(), enabled=True)
    malformed = (
        (nm.RebaseBinding(receipt, (2,)),),
        (nm.RebaseBinding(receipt, (2, 2)),),
        (nm.RebaseBinding(object(), (2, 3)),),
    )
    for bindings in malformed:
        with pytest.raises(nm.Rejected):
            nm.capture(case["hz"], case["decoder"], global_widths=(15, 4),
                       bindings=bindings, budget=_budget(), enabled=True)
    with pytest.raises(nm.Rejected, match="owned"):
        nm.normalize(object(), _nf(), enabled=True)
    assert _budget()._failure is None
    point = _assignment(case, _transport(0, 0))
    assert case["result"].satisfied(case["result"].canonical_extension(point), integral=True)
    _record(10, frame_prefix_content_owner_and_budget_checked=True,
            duplicate_missing_link_binding_rejected=True, old_state_not_poisoned=True)


def test_11_structure_and_private_dependencies():
    case = _case()
    before, after, _ = case["rebases"][0]
    changed = after.Ac.tolil(copy=True)
    changed[2, 10] = 1 / 8
    with pytest.raises(nm.Rejected, match="linking row"):
        nm.record_rebase(before, replace(after, Ac=changed.tocsr()), budget=nm.Budget(), enabled=True)
    hidden = _fixture(hidden=True, budget=nm.Budget())
    snap = nm.capture(hidden["hz"], hidden["decoder"],
        global_widths=(hidden["hz"].n_cont, hidden["hz"].n_bin),
        bindings=hidden["bindings"], budget=hidden["budget"], enabled=True)
    with pytest.raises(nm.Rejected, match="normalized source"):
        nm.attach(snap, population=hidden["population"], enabled=True)
    assert hidden["budget"]._failure is None
    # A genuine additional graph may share an image column as its eta.
    # Such a column cannot also be an eligible normalization pivot.
    private = _fixture(budget=nm.Budget())
    hz, f = private["hz"], private["gates"][0]["f"]
    eqs = list(_raw_forms(hz.Ac, hz.Ab, np.zeros(len(hz.b))))
    les = list(_raw_forms(hz.Auc, hz.Aub, np.zeros(len(hz.ub))))
    eqs.append(_nadd((-1, f), (1, _nf(0, ((15, F(-1, 2)), (9, F(-1, 2))), ((4, F(-1, 2)),)))))
    les.extend((_nf(0, ((15, -ONE),), ((4, -ONE),)), _nf(0, ((9, -ONE),), ((4, ONE),))))
    private["hz"] = _state(16, 5, private["outputs"], eqs,
        tuple(F.from_float(float(v)) for v in hz.b) + (F(-1, 2),), les,
        tuple(F.from_float(float(v)) for v in hz.ub) + (ZERO, ZERO))
    replacement = nm.Graph("extended", (15, 9, 4), len(hz.b), (len(hz.ub), len(hz.ub) + 1))
    parents, children = private["population"][0]
    private["population"] = (((replacement, parents[1]), children),)
    snap = nm.capture(private["hz"], private["decoder"], global_widths=(16, 5),
                      bindings=private["bindings"], budget=private["budget"], enabled=True)
    with pytest.raises(nm.Rejected, match="gate-private"):
        nm.attach(snap, population=private["population"], enabled=True)
    assert private["budget"]._failure is None
    _record(11, same_epoch_extra_support_rejected=True,
            hidden_parent_dependency_rechecked=True, selected_private_pivot_rejected=True)


def test_12_fill_and_bit_budget_fail_closed():
    before, after = _offset_pair()
    for limits in ({"max_work": 0}, {"max_entries": 0}, {"max_branch": 0}, {"max_bits": 16}):
        budget = nm.Budget(**limits)
        with pytest.raises(nm.Rejected):
            nm.record_rebase(before, after, budget=budget, enabled=True)
        assert budget._failure is not None
        with pytest.raises(nm.Rejected):
            nm.record_rebase(before, after, budget=budget, enabled=True)
    with pytest.raises(nm.Rejected):
        nm.Budget(max_bits=513)
    case = _case()
    work, entries = _budget().work, _budget().entries
    nm.normalize(case["snapshot"], case["outputs"][11], enabled=True)
    assert _budget().work > work and _budget().entries > entries
    _record(12, resource_failures_sticky=True, ordinary_stored_float_bit_limit_checked=True,
            complete_normalization_occurrences_charged=True,
            synthetic_65536_boundary_executed=False, default_caps_unchanged=True)


def test_13_rounding_and_terminal_preservation():
    case = _case("rounding", mode="rounding")
    result, rel = case["result"], case["result"].relations[0]
    assert rel.a == (F(1, 2), F(25, 48)) and rel.r_bounds == (F(-5, 96), F(5, 96))
    assert any(error > 0 for error in result.row_errors)
    for exact, stored, error, installed in zip(result.exact_rows, result.stored_rows,
                                              result.row_errors, result.installed_rhs):
        left, right = dict(exact.terms), dict(stored.terms)
        actual = sum((abs(left.get(i, ZERO) - right.get(i, ZERO)) for i in set(left) | set(right)), ZERO)
        assert error == actual and installed == stored.rhs and installed >= exact.rhs + actual
    for inputs in (_transport(1, -1, 1), _transport(-1, 1, -1), _transport(0, 0)):
        value = result.canonical_extension(_assignment(case, inputs))
        assert result.satisfied(value, integral=True) and result.decode(value) == inputs
    isolated = _fixture()
    snap, rr = _attach(isolated)
    frozen = _bytes(snap.hz)
    isolated["hz"].c[0] += 1
    assert _bytes(snap.hz) == frozen
    with pytest.raises(ValueError):
        snap.hz.Ac.data[0] = 0
    assert rr.satisfied(rr.canonical_extension(_assignment(isolated, _transport(0, 0))), integral=True)
    tampered = _fixture(budget=nm.Budget())
    ss, tt = _attach(tampered)
    ss.hz.frame_id += 1
    with pytest.raises(nm.Rejected, match="mutated"):
        tt.terminal_model()
    assert tampered["budget"]._failure is None
    _record(13, all_stored_coefficient_rounding_paid=True, old_H_isolated_and_readonly=True,
            tampered_terminal_rejected=True, outward_aux_uniqueness_claimed=False)


def _normal_lp(hz, model, objective_index):
    count = len(hz.b)
    objective = -model.value_matrix.getrow(objective_index).toarray().ravel()
    answer = linprog(objective, A_ub=model.A[count:], b_ub=model.row_ub[count:],
        A_eq=model.A[:count], b_eq=model.row_ub[:count],
        bounds=list(zip(model.var_lb, model.var_ub)), method="highs", options={"time_limit": 5.0})
    assert answer.success, answer.message
    return float(model.value_center[objective_index] - answer.fun)



def test_14_physical_certificate_and_fixed_lp():
    case = _case()
    result, old = case["result"], case["snapshot"].hz
    certificate = _certificate(case)
    assert certificate == _row(_flat(case["outputs"][case["F_index"]], 15), F(265, 128))
    fake = _fake(case)
    assert _old_holds(old, fake) and _readout(case, 12, fake) == F(8509, 4096)
    assert F(8509, 4096) - F(265, 128) == F(29, 4096)
    assert F(8509, 4096) - F(1061, 512) == F(21, 4096)
    true = result.canonical_extension(_assignment(case, _transport(1, -1, 1)))
    assert _readout(case, 12, true) == F(527, 256)
    assert (old.n_cont, old.n_bin, len(old.b), len(old.ub)) == (15, 4, 6, 8)
    assert (result.hz.n_cont, result.hz.n_bin) == (21, 4)
    assert len(result.hz.b) == 6 and len(result.hz.ub) == 37
    assert len(result.exact_rows) == 29 and len(result.relations[0].products) == 6
    assert sum(len(row.terms) for row in result.stored_rows) == 104
    model = result.terminal_model()
    assert tuple(model.cont_source) == tuple(range(21)) and tuple(model.bin_source) == tuple(range(4))
    assert not model.bin_fixes and not model.cont_eliminations
    assert tuple(model.integrality) == (0,) * 21 + (1,) * 4
    plain = _lower_hz_milp(old, prune_unused=False, coalesce_rows=False,
                         project_inactive_cont=False, fix_implied_binary=False)
    old_bound = _normal_lp(old, plain, case["F_index"])
    new_bound = _normal_lp(result.hz, model, case["F_index"])
    assert old_bound >= float(F(8509, 4096)) - 1e-8
    assert new_bound <= float(F(265, 128)) + 1e-8
    assert new_bound >= float(F(527, 256)) - 1e-8
    assert new_bound < float(F(1061, 512)) and new_bound < old_bound
    _record(14, stored_nonnegative_LE_and_actual_EQ_certificate=True, upper="265/128",
            old_feasible="8509/4096", aux_free_gap="29/4096", bound_claimed_tight=False,
            fixed_normal_LPs=2, old_LP=old_bound, new_LP=new_bound,
            preserved_image_columns=2, complete_EQ=6, complete_LE=37,
            new_continuous=6, new_LE=29, new_nnz=104,
            complete_predicate_nnz=int(result.hz.Ac.nnz + result.hz.Ab.nnz + result.hz.Auc.nnz + result.hz.Aub.nnz),
            native_columns=[21, 4], adaptive_solver_or_rescue=False)


def test_15_shared_budget_and_atomic_batch():
    case = _case()
    assert all(c["budget"] is _budget() for c in _CASES.values())
    assert all(receipt.budget is _budget() for c in _CASES.values()
               for _, _, receipt in c["rebases"])
    before = _bytes(case["snapshot"].hz)
    work, entries = _budget().work, _budget().entries
    parents, children = case["population"][0]
    bad_second = (parents, (children[0], children[0]))
    with pytest.raises(nm.Rejected):
        nm.attach(case["snapshot"], population=(case["population"][0], bad_second), enabled=True)
    assert _bytes(case["snapshot"].hz) == before
    assert _budget().work > work and _budget().entries > entries
    assert _budget()._failure is None
    point = _assignment(case, _transport(0, 0))
    assert case["result"].satisfied(case["result"].canonical_extension(point), integral=True)
    _record(15, ordinary_positive_fixtures_share_budget=True,
            receipt_capture_budget_same_object=True, failing_batch_returns_no_partial_result=True,
            original_published_state_usable=True, work=_budget().work, entries=_budget().entries)


def test_16_summary_and_qualification_boundary():
    assert set(_EVIDENCE) == set(_NAMES[:-1])
    _record(16, original_baseline=1870, original_population=2413, independent_score=61,
            formal_gain=0, new_solves=0, native_HZ_admitted=False, new_domain_qualified=False,
            new_capability_qualified=False, actual_model_qualified=False, online_qualified=False,
            gpu_qualified=False, physical_resource_qualified=False, domain_definition_changed=False,
            production_operator_called_on_constructed_inputs=True, actual_three_models_run=False,
            rebase_native_transport_passed=False, local_mathematical_tests_complete=True)
    _record_file("summary.json", {"schema": "d255_rebase_relation_v1", "status": "PASS",
        "tests": _EVIDENCE, "test_count": 16,
        "shared_budget": {"work": _budget().work, "entries": _budget().entries},
        "scope": "content-certified production rebase on constructed native inputs; no model admission"})

