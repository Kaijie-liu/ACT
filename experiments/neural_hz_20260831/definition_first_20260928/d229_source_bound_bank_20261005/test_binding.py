"""Eight frozen native mathematical tests; no model or solver qualification.

The native fixtures use the original affine/ReLU operators.  Local-bank
vertices are never interpreted as conditioned original-input atoms.
Do not import, collect, compile, or execute this file before its freeze.
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

from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import (
    sparse_hz_apply_relu_exact, sparse_hz_apply_relu_compact_exact,
)
from experiments.neural_hz_20260831.definition_first_20260928.d229_source_bound_bank_20261005 import binding as bridge


ZERO, ONE = F(0), F(1)
RUN = (Path(__file__).resolve().parents[2]
       / "results/d229_source_bound_bank_20261005_v1")
_NAMES = (
    "default_off_and_phase_orientation",
    "full_actual_residual_binding",
    "pullback_rows_and_original_state_preservation",
    "real_integer_states_have_joint_extensions",
    "zero_labels_and_fractional_view",
    "compact_error_and_wrong_guard_rejected",
    "complete_receiver_bound_and_cost",
    "asymmetric_shared_source_and_summary",
)
_EVIDENCE = {}


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    """Independent evidence writer, relocatable without changing mathematics."""
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _fraction(value):
    return F.from_float(float(value))


def _form(hz, index):
    terms = []
    for matrix in (hz.Gc, hz.Gb):
        begin, end = matrix.indptr[index:index + 2]
        terms.append(tuple((int(matrix.indices[j]), _fraction(matrix.data[j]))
                           for j in range(int(begin), int(end))))
    return bridge.NativeAffine(_fraction(hz.c[index]), terms[0], terms[1])


def _combine(bias, scaled):
    constant = F(bias)
    continuous, binary = {}, {}
    for scale, form in scaled:
        constant += scale * form.bias
        for terms, destination in ((form.continuous, continuous), (form.binary, binary)):
            for index, coefficient in terms:
                destination[index] = destination.get(index, ZERO) + scale * coefficient
    return bridge.NativeAffine(
        constant,
        tuple(sorted((i, coefficient) for i, coefficient in continuous.items() if coefficient)),
        tuple(sorted((i, coefficient) for i, coefficient in binary.items() if coefficient)),
    )


def _value(form, continuous, binary):
    return (form.bias
            + sum((amount * continuous[index] for index, amount in form.continuous), ZERO)
            + sum((amount * binary[index] for index, amount in form.binary), ZERO))


def _cube(form):
    radius = sum((abs(value) for _, value in form.continuous + form.binary), ZERO)
    return form.bias - radius, form.bias + radius


def _matrix_row(matrix, index, values):
    begin, end = matrix.indptr[index:index + 2]
    return sum((_fraction(matrix.data[j]) * values[int(matrix.indices[j])]
                for j in range(int(begin), int(end))), ZERO)


def _native_holds(hz, continuous, binary, integer=True):
    assert len(continuous) == hz.n_cont and len(binary) == hz.n_bin
    assert all(-ONE <= value <= ONE for value in continuous + binary)
    if integer:
        assert all(value in (-ONE, ONE) for value in binary)
    return (all(_matrix_row(hz.Ac, i, continuous) + _matrix_row(hz.Ab, i, binary)
                == _fraction(rhs) for i, rhs in enumerate(hz.b))
            and all(_matrix_row(hz.Auc, i, continuous) + _matrix_row(hz.Aub, i, binary)
                    <= _fraction(rhs) for i, rhs in enumerate(hz.ub)))


def _snapshot(hz):
    return (hz.frame_id, hz.exact,
            tuple((value.shape, str(value.dtype), value.tobytes())
                  for value in (hz.c, hz.b, hz.ub)),
            tuple((matrix.shape, matrix.data.tobytes(), matrix.indices.tobytes(),
                   matrix.indptr.tobytes())
                  for matrix in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)))


def _pad(matrix, columns):
    assert matrix.shape[1] <= columns
    return sp.hstack((matrix, sp.csr_matrix((matrix.shape[0], columns-matrix.shape[1]))),
                     format="csr")


def _fixture(compact=False, asymmetric=False, upstream_binary=False):
    upstream = int(upstream_binary)
    source_gb = (sp.csr_matrix(([0.125], ([0], [0])), shape=(2, 1))
                 if upstream else sp.csr_matrix((2, 0)))
    source = SparseHZono(np.zeros(2), sp.eye(2, format="csr"), source_gb,
                        sp.csr_matrix((0, 2)), sp.csr_matrix((0, upstream)),
                        np.zeros(0), frame_id=22901)
    if asymmetric:
        parent_weights = np.array([[1., 0.25], [0.125, 1.]])
        parent_bias = np.array([0.125, -0.25])
        parent_bounds = ((F(-2), F(2)),) * 2
        child_weights = ((F(3, 4), F(5, 4)), (F(3, 2), F(-1, 2)))
        child_bias = (F(-1, 2), F(-1, 8))
        child_bounds = ((F(-3), F(4)),) * 2
    else:
        assert not upstream
        parent_weights, parent_bias = np.eye(2), np.zeros(2)
        parent_bounds = ((F(-1), F(1)),) * 2
        child_weights = ((ONE, ONE), (ONE, -ONE))
        child_bias = (F(-3, 4), F(-1, 4))
        child_bounds = ((F(-3, 4), F(5, 4)), (F(-5, 4), F(3, 4)))
    parent_pre = sparse_hz_linear(source, parent_weights, parent_bias)
    width = 1 if compact else 2
    parent_slots = tuple((2 + width*i, 2 + width*i + int(not compact), upstream+i)
                         for i in range(2))
    kernel = sparse_hz_apply_relu_compact_exact if compact else sparse_hz_apply_relu_exact
    parents = kernel(parent_pre, [float(pair[0]) for pair in parent_bounds],
                     [float(pair[1]) for pair in parent_bounds], parent_slots,
                     2 + 2*width, upstream+2)
    child_pre = sparse_hz_linear(parents, np.array(child_weights, dtype=np.float64),
                                np.array(child_bias, dtype=np.float64))
    child_slots = tuple((parents.n_cont + width*i,
                         parents.n_cont + width*i + int(not compact), upstream+2+i)
                        for i in range(2))
    children = kernel(child_pre, [float(pair[0]) for pair in child_bounds],
                      [float(pair[1]) for pair in child_bounds], child_slots,
                      parents.n_cont+2*width, upstream+4)
    if compact:
        graphs = tuple(bridge.Graph("compact", slots, None, rows)
                       for slots, rows in zip(parent_slots+child_slots,
                                              ((0, 2, 4), (1, 3, 5),
                                               (6, 8, 10), (7, 9, 11))))
    else:
        graphs = tuple(bridge.Graph("extended", slots, index, rows)
                       for index, (slots, rows) in enumerate(zip(parent_slots+child_slots,
                                                ((0, 2), (1, 3), (4, 6), (5, 7)))))
    base = SparseHZono(
        np.concatenate((children.c, parents.c, source.c)),
        sp.vstack((children.Gc, _pad(parents.Gc, children.n_cont),
                   _pad(source.Gc, children.n_cont)), format="csr"),
        sp.vstack((children.Gb, _pad(parents.Gb, children.n_bin),
                   _pad(source.Gb, children.n_bin)), format="csr"),
        children.Ac, children.Ab, children.b, children.Auc, children.Aub,
        children.ub, frame_id=children.frame_id, exact=children.exact,
    )
    # Preserve y1,y2,q1,q2,source1,source2 and both complete consumer rows.
    receiver_weights = np.array([2., -1.5, 0.25, -0.5, 0.125, -0.25])
    skip_weights = np.array([0., 0., 1., 1., 1., 1.])
    weights = np.vstack((np.eye(6), receiver_weights, skip_weights))
    hz = sparse_hz_linear(base, weights, np.array([0.]*6 + [0.1875, 0.]))
    return dict(hz=hz, graphs=graphs, parent_g=tuple(_form(parent_pre, i) for i in range(2)),
                child_g=tuple(_form(child_pre, i) for i in range(2)),
                parent_bounds=parent_bounds, child_bounds=child_bounds,
                parent_slots=parent_slots, child_slots=child_slots,
                child_weights=child_weights, child_bias=child_bias,
                compact=compact, upstream=upstream)


def _reference(fixture, mismatch=True):
    if mismatch:
        return (((F(1, 8), ZERO, F(3, 4), ONE),
                 (ZERO, F(-1, 8), ONE, F(-3, 4))), (F(-1, 2), F(-1, 8)))
    return (tuple((ZERO, ZERO) + row for row in fixture["child_weights"]),
            fixture["child_bias"])


def _bind(fixture, mismatch=True, budget=None):
    coefficients, biases = _reference(fixture, mismatch)
    return bridge.bind(fixture["hz"], fixture["graphs"], coefficients, biases,
                       enabled=True, budget=budget)


def _states(fixture, x, upstream_bit=ONE):
    hz = fixture["hz"]
    old_binary = (F(upstream_bit),) if fixture["upstream"] else ()
    parent_g = tuple(_value(form, x, old_binary) for form in fixture["parent_g"])
    parent_q = tuple(max(ZERO, value) for value in parent_g)
    child_g = tuple(sum((a*q for a, q in zip(row, parent_q)), ZERO)+bias
                    for row, bias in zip(fixture["child_weights"], fixture["child_bias"]))
    all_g = parent_g + child_g
    choices = tuple((-ONE,) if value > 0 else (ONE,) if value < 0 else (-ONE, ONE)
                    for value in all_g)
    for phase_bits in product(*choices):
        continuous = [ZERO] * hz.n_cont
        continuous[:2] = tuple(F(value) for value in x)
        binary = old_binary + phase_bits
        for value, bit, bounds, slots in zip(all_g, phase_bits,
                fixture["parent_bounds"] + fixture["child_bounds"],
                fixture["parent_slots"] + fixture["child_slots"]):
            lower, upper = bounds
            s, eta, _ = slots
            if not fixture["compact"]:
                continuous[s] = ONE if bit == -ONE else value/(lower/2)-ONE
            continuous[eta] = ONE-max(ZERO, value)/(upper/2)
        assert tuple(_value(form, continuous, binary) for form in fixture["child_g"]) == child_g
        yield tuple(continuous), tuple(binary), all_g


def _physical(bound, continuous, binary):
    return tuple(_value(form, continuous, binary) for form in bound.physical)


def _weights(bank, points):
    weights = [ZERO] * len(bank.vertices)
    for physical, weight in points:
        weights[bank.vertices.index(tuple(physical))] += weight
    assert sum(weights, ZERO) == ONE
    return tuple(weights)


def _pulled_holds(pulled, continuous, binary, weights):
    assert len(weights) == pulled.n_lambda and len(continuous) == pulled.old_n_cont
    extended = tuple(continuous) + tuple(2*weight-ONE for weight in weights)
    assert len(extended) == pulled.n_cont and len(binary) == pulled.n_bin
    assert all(-ONE <= value <= ONE for value in extended)
    return (all(_value(row.affine, extended, binary) == row.rhs for row in pulled.eq)
            and all(_value(row.affine, extended, binary) <= row.rhs for row in pulled.le))


def _identity_rows(bound, pulled):
    """Independently compare symbolic coefficients, then a rational evaluation."""
    continuous = tuple(F(i-3, 17) for i in range(bound.old_n_cont))
    binary = tuple(F(i-2, 7) for i in range(bound.old_n_bin))
    weights = (ONE/len(bound.bank.vertices),) * len(bound.bank.vertices)
    physical = _physical(bound, continuous, binary)
    phases = tuple(-binary[index] for index in bound.phase_columns)
    values = physical + phases + weights
    extended = continuous + tuple(2*weight-ONE for weight in weights)
    compiled = bound.bank.compile(frame=bound.source_hz)
    assert len(compiled.eq) == len(pulled.eq) and len(compiled.le) == len(pulled.le)
    for original, substituted in zip(compiled.eq+compiled.le, pulled.eq+pulled.le):
        scaled = []
        for index, coefficient in original.terms:
            if index < 8:
                form = bound.physical[index]
            elif index < 12:
                form = bridge.NativeAffine(ZERO, (), ((bound.phase_columns[index-8], -ONE),))
            else:
                form = bridge.NativeAffine(F(1, 2), ((bound.old_n_cont+index-12, F(1, 2)),), ())
            scaled.append((coefficient, form))
        symbolic = _combine(ZERO, scaled)
        assert substituted.affine == bridge.NativeAffine(ZERO, symbolic.continuous, symbolic.binary)
        assert substituted.rhs == original.rhs-symbolic.bias
        left = sum((coefficient*values[index] for index, coefficient in original.terms), ZERO)
        right = _value(substituted.affine, extended, binary)
        assert left-original.rhs == right-substituted.rhs
        assert substituted.affine.bias == ZERO
    return len(compiled.eq)+len(compiled.le)


def _residual_extension(bound, physical):
    """Independent interpolation on residual intervals at a parent grid point."""
    base = physical[:4]
    intervals = []
    for j, (lower, upper) in enumerate(bound.bank.spec.residual_bounds):
        row, bias = bound.bank.spec.coefficients[j], bound.bank.spec.biases[j]
        assert row[4:] == ((ONE, ZERO) if j == 0 else (ZERO, ONE))
        offset = sum((a*v for a, v in zip(row[:4], base)), ZERO)+bias
        knots = {lower, upper}
        if lower <= -offset <= upper:
            knots.add(-offset)
        knots = sorted(knots)
        value = physical[4+j]
        assert lower <= value <= upper
        if value in knots:
            intervals.append(((value, ONE),))
        else:
            lo = max(knot for knot in knots if knot < value)
            hi = min(knot for knot in knots if knot > value)
            intervals.append(((lo, (hi-value)/(hi-lo)), (hi, (value-lo)/(hi-lo))))
    points = []
    for (v1, w1), (v2, w2) in product(*intervals):
        local = base + (v1, v2)
        outputs = tuple(max(ZERO, sum((a*v for a, v in zip(row, local)), ZERO)+bias)
                        for row, bias in zip(bound.bank.spec.coefficients, bound.bank.spec.biases))
        points.append((local+outputs, w1*w2))
    weights = _weights(bound.bank, points)
    reconstructed = tuple(sum((weight*vertex[i] for weight, vertex in zip(weights, bound.bank.vertices)), ZERO)
                          for i in range(8))
    assert reconstructed == physical
    return weights


def test_01_default_off_and_phase_orientation():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("disabled bind inspected its input")

    poison = Unreadable()
    with pytest.raises(bridge.Rejected):
        bridge.bind(poison, poison, poison, poison)
    with pytest.raises(bridge.Rejected):
        bridge.bind(poison, poison, poison, poison, enabled=1)
    fixture = _fixture()
    before = _snapshot(fixture["hz"])
    bound = _bind(fixture, mismatch=False)
    assert bound.source_hz is fixture["hz"]
    assert bound.phase_columns == (0, 1, 2, 3)
    continuous, binary, pre = next(_states(fixture, (ONE, ONE)))
    assert binary == (-ONE, -ONE, -ONE, ONE)
    physical = _physical(bound, continuous, binary)
    logical = tuple(-binary[index] for index in bound.phase_columns)
    assert logical == (ONE, ONE, ONE, -ONE)
    assert all((phase > 0) == (value > 0) for phase, value in zip(logical, pre))
    weights = _weights(bound.bank, ((physical, ONE),))
    assert bound.bank.compile(frame=fixture["hz"]).satisfied(physical, logical, weights)
    with pytest.raises(bridge.Rejected):
        bridge.pullback(bound, frame=object())
    assert _snapshot(fixture["hz"]) == before
    _record(1, default_off=True, native_active=-1, bank_active=1,
            original_phase_columns=list(bound.phase_columns), source_unchanged=True)


def test_02_full_actual_residual_binding():
    fixture = _fixture()
    bound = _bind(fixture)
    coefficients, biases = _reference(fixture)
    expected = tuple(_combine(-bias, ((ONE, actual),)
                             + tuple((-coefficient, form)
                                     for coefficient, form in zip(row, bound.physical[:4])))
                     for actual, row, bias in zip(fixture["child_g"], coefficients, biases))
    assert bound.physical[4:6] == expected
    assert expected == (
        bridge.NativeAffine(F(-1, 8), ((0, F(-1, 8)), (3, F(-1, 8))), ()),
        bridge.NativeAffine(F(-1, 4), ((1, F(1, 8)), (5, F(1, 8))), ()),
    )
    assert bound.bank.spec.residual_bounds == ((F(-3, 8), F(1, 8)), (F(-1, 2), ZERO))
    assert bound.bank.spec.residual_bounds == tuple(_cube(form) for form in expected)
    assert bound.bank.spec.biases == biases
    assert bound.bank.spec.coefficients == (coefficients[0]+(ONE, ZERO), coefficients[1]+(ZERO, ONE))
    for j, (row, bias) in enumerate(zip(coefficients, biases)):
        restored = _combine(bias, tuple(zip(row, bound.physical[:4]))+((ONE, bound.physical[4+j]),))
        assert restored == fixture["child_g"][j]
    _record(2, complete_bias_and_coefficient_difference=True, residual_coordinates=2,
            residual_bounds=[[str(value) for value in pair] for pair in bound.bank.spec.residual_bounds],
            independent_physical_noise_added=False)


def test_03_pullback_rows_and_original_state_preservation():
    fixture = _fixture()
    fixture["hz"] = replace(fixture["hz"], exact=False)
    before = _snapshot(fixture["hz"])
    bound = _bind(fixture)
    pulled = bridge.pullback(bound, frame=fixture["hz"])
    checked = _identity_rows(bound, pulled)
    size = len(bound.bank.vertices)
    assert pulled.n_cont == fixture["hz"].n_cont+size
    assert pulled.old_n_cont == fixture["hz"].n_cont
    assert pulled.n_bin == fixture["hz"].n_bin == 4
    assert pulled.n_lambda == size and len(pulled.eq) == 9 and len(pulled.le) == size+8
    assert bound.source_hz.exact is False and _snapshot(fixture["hz"]) == before
    assert all(0 <= index < pulled.n_cont for row in pulled.eq+pulled.le
               for index, _ in row.affine.continuous)
    assert all(0 <= index < pulled.n_bin for row in pulled.eq+pulled.le
               for index, _ in row.affine.binary)
    _record(3, source_byte_state_preserved=True, false_exact_flag_preserved=True,
            all_original_outputs=fixture["hz"].n_out, affine_identity_rows=checked,
            lambda_map="(1+new_continuous)/2", added_continuous=size,
            eq=len(pulled.eq), le=len(pulled.le), new_binary=0,
            runtime_allocator_published=False)


def test_04_real_integer_states_have_joint_extensions():
    fixture = _fixture()
    bound = _bind(fixture)
    pulled = bridge.pullback(bound, frame=fixture["hz"])
    count = 0
    for x in product((F(-1), ZERO, ONE), repeat=2):
        for continuous, binary, _ in _states(fixture, x):
            assert _native_holds(fixture["hz"], continuous, binary)
            physical = _physical(bound, continuous, binary)
            weights = _residual_extension(bound, physical)
            assert _pulled_holds(pulled, continuous, binary, weights)
            assert continuous[:2] == x
            count += 1
    assert count == 16
    _record(4, original_input_grid_points=9, integer_labelled_states=count,
            shared_lambda_extension=True, independent_interval_interpolation=True,
            solver_calls=0, vertex_conditioned_original_source_required=False,
            full_state_enumeration=False, model_witness=False)


def test_05_zero_labels_and_fractional_view():
    fixture = _fixture()
    bound = _bind(fixture, mismatch=False)
    pulled = bridge.pullback(bound, frame=fixture["hz"])
    assert bound.bank.spec.residual_bounds == ((ZERO, ZERO), (ZERO, ZERO))
    count = 0
    for x in ((F(1, 2), F(1, 4)), (ZERO, ZERO)):
        for continuous, binary, _ in _states(fixture, x):
            physical = _physical(bound, continuous, binary)
            weights = _weights(bound.bank, ((physical, ONE),))
            assert _native_holds(fixture["hz"], continuous, binary)
            assert _pulled_holds(pulled, continuous, binary, weights)
            count += 1
    assert count == 8
    states = [next(_states(fixture, x)) for x in ((ONE, ONE), (-ONE, -ONE))]
    continuous = tuple((a+b)/2 for a, b in zip(states[0][0], states[1][0]))
    binary = tuple((a+b)/2 for a, b in zip(states[0][1], states[1][1]))
    points = tuple((_physical(bound, state[0], state[1]), F(1, 2)) for state in states)
    weights = _weights(bound.bank, points)
    assert _native_holds(fixture["hz"], continuous, binary, integer=False)
    assert _pulled_holds(pulled, continuous, binary, weights)
    physical = _physical(bound, continuous, binary)
    assert physical[0] == ZERO and physical[2] == F(1, 2)
    assert any(bit not in (-ONE, ONE) for bit in binary)
    _record(5, legal_zero_label_states=count, fractional_lp_view_kept=True,
            fractional_view_not_true_integer_graph=True, original_bits_not_relaxed_in_domain=True)


def test_06_compact_error_and_wrong_guard_rejected():
    compact = _fixture(compact=True)
    _bind(compact)
    rhs = compact["hz"].ub.copy()
    rhs[compact["graphs"][0].le_rows[2]] += 0.125
    altered = dict(compact, hz=replace(compact["hz"], ub=rhs))
    with pytest.raises(bridge.Rejected):
        _bind(altered)
    fixture = _fixture()
    rhs = fixture["hz"].ub.copy()
    rhs[fixture["graphs"][0].le_rows[0]] = 0.125
    with pytest.raises(bridge.Rejected):
        _bind(dict(fixture, hz=replace(fixture["hz"], ub=rhs)))
    bad_graphs = list(fixture["graphs"])
    bad_graphs[0] = replace(bad_graphs[0], le_rows=(1, 2))
    with pytest.raises(bridge.Rejected):
        _bind(dict(fixture, graphs=tuple(bad_graphs)))
    bound = _bind(fixture)
    original = fixture["hz"].c[0]
    fixture["hz"].c[0] += 0.125
    try:
        with pytest.raises(bridge.Rejected):
            bridge.pullback(bound, frame=fixture["hz"])
        with pytest.raises(bridge.Rejected):
            bridge.receiver_bounds(bound, 6, (0,)*8, frame=fixture["hz"])
    finally:
        fixture["hz"].c[0] = original
    _record(6, exact_compact_accepted=True, nonzero_compact_error_rejected=True,
            wrong_guards_rejected=True, source_mutation_rejected=True,
            midpoint_substitution=False)


def test_07_complete_receiver_bound_and_cost():
    fixture = _fixture()
    before = _snapshot(fixture["hz"])
    budget = bridge.Budget()
    bound = _bind(fixture, budget=budget)
    at_bind = budget.work, budget.entries
    pulled = bridge.pullback(bound, frame=fixture["hz"])
    assert bound.budget is budget and bound.bank.budget is budget
    assert budget.work > at_bind[0] and budget.entries > at_bind[1]
    coefficients = (ZERO, ZERO, F(1, 4), F(-1, 2), ZERO, ZERO, F(2), F(-3, 2))
    result = bridge.receiver_bounds(bound, 6, coefficients, frame=fixture["hz"])
    expected = _combine(ZERO, ((ONE, _form(fixture["hz"], 6)),)
                        + tuple((-weight, form) for weight, form in zip(coefficients, bound.physical)))
    assert result.remainder == expected == bridge.NativeAffine(
        F(3, 16), ((0, F(1, 8)), (1, F(-1, 4))), ())
    lo, hi = _cube(expected)
    assert (lo, hi) == (F(-3, 16), F(9, 16))
    local_upper = bound.bank.support(coefficients, frame=fixture["hz"]).value
    local_lower = -bound.bank.support(tuple(-value for value in coefficients), frame=fixture["hz"]).value
    assert (result.lower, result.upper) == (local_lower+lo, local_upper+hi)
    skip = bridge.receiver_bounds(bound, 7, (ZERO,)*8, frame=fixture["hz"])
    assert skip.remainder == _form(fixture["hz"], 7)
    assert (skip.lower, skip.upper) == _cube(skip.remainder) == (F(-2), F(4))
    for x in product((F(-1), ZERO, ONE), repeat=2):
        continuous, binary, _ = next(_states(fixture, x))
        actual = _value(_form(fixture["hz"], 6), continuous, binary)
        assert result.lower <= actual <= result.upper
    exhausted = bridge.Budget(max_work=0)
    with pytest.raises(bridge.Rejected):
        _bind(fixture, budget=exhausted)
    assert exhausted.failed
    coefficients_bad, biases = _reference(fixture)
    coefficients_bad = ((F(1 << 512),) + coefficients_bad[0][1:], coefficients_bad[1])
    with pytest.raises(bridge.Rejected):
        bridge.bind(fixture["hz"], fixture["graphs"], coefficients_bad, biases, enabled=True)
    assert _snapshot(fixture["hz"]) == before
    nnz = sum(len(row.affine.continuous)+len(row.affine.binary) for row in pulled.eq+pulled.le)
    _record(7, complete_receiver_bounds=[str(result.lower), str(result.upper)],
            complete_remaining_cube=[str(lo), str(hi)], skip_bounds=[str(skip.lower), str(skip.upper)],
            local_support_bounds=[str(local_lower), str(local_upper)],
            cumulative_work=budget.work, cumulative_entries=budget.entries,
            pulled_nnz=nnz, whole_source_unchanged=True, shared_budget=True,
            budget_and_bit_limit_rejected=True, complete_physical_qualification=False)


def test_08_asymmetric_shared_source_and_summary():
    fixture = _fixture(asymmetric=True, upstream_binary=True)
    before = _snapshot(fixture["hz"])
    bound = _bind(fixture)
    pulled = bridge.pullback(bound, frame=fixture["hz"])
    assert bound.phase_columns == (1, 2, 3, 4)
    assert bound.old_n_bin == pulled.n_bin == 5
    assert bound.physical[:2] == fixture["parent_g"]
    assert dict(bound.physical[0].binary)[0] == F(1, 8)
    assert dict(bound.physical[1].binary)[0] == F(1, 64)
    checked = _identity_rows(bound, pulled)
    coefficients = (F(1, 3), F(-1, 4), F(1, 2), F(-2, 3), F(1, 5),
                    F(-1, 7), F(3, 4), F(-1, 2))
    result = bridge.receiver_bounds(bound, 6, coefficients, frame=fixture["hz"])
    expected = _combine(ZERO, ((ONE, _form(fixture["hz"], 6)),)
                        + tuple((-weight, form) for weight, form in zip(coefficients, bound.physical)))
    assert result.remainder == expected
    point_count = 0
    for upstream_bit in (-ONE, ONE):
        for x in ((F(-1, 2), F(1, 4)), (F(1, 3), F(-2, 5)), (ZERO, ZERO)):
            for continuous, binary, _ in _states(fixture, x, upstream_bit):
                assert _native_holds(fixture["hz"], continuous, binary)
                actual = _value(_form(fixture["hz"], 6), continuous, binary)
                assert result.lower <= actual <= result.upper
                point_count += 1
    assert _snapshot(fixture["hz"]) == before
    _record(8, nonparallel_shared_source=True, mixed_biased_children=True,
            retained_upstream_binary=1, original_binary_coefficients_not_flipped=True,
            affine_identity_rows=checked, concrete_mathematical_states=point_count,
            complete_receiver_bounds=[str(result.lower), str(result.upper)],
            source_unchanged=True)
    missing = [name for name in _NAMES if name not in _EVIDENCE]
    _record_file("summary.json", {
        "schema": "d229_source_bound_bank_mathematical_tests_v1",
        "required_tests": list(_NAMES), "completed_tests": _EVIDENCE,
        "missing_tests": missing, "mathematical_native_fixture_only": True,
        "actual_model_run": False, "actual_model_binding_qualified": False,
        "native_runtime_installation_qualified": False, "complete_physical_qualification": False,
        "gpu_qualified": False, "solver_calls": 0, "formal_gain": 0,
        "independent_e0_gain": 0,
    })
    assert not missing
