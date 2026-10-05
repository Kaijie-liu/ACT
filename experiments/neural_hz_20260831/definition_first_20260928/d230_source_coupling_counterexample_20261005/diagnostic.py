"""Default-off mathematical counterexample to independent residual mean binding.

This constructs four ORIGINAL source coordinates and four original native
extended ReLU graphs.  It is not a model run, native installation, solver,
attack, improvement, or new abstract domain.  Stored dyadic float64 fixture
coefficients are read as their exact binary rationals.

The fractional point satisfies the original H's continuous relaxation.  It
is NOT a concrete network state.  Its local-bank extension follows from the
already proved D228 complete labelled graph hull: this diagnostic checks
the two graph atoms, not a numerical lambda decomposition.  No local atom
is asserted to have an original-source witness.

Do not import, collect, compile, or execute before the D230 freeze.
"""

from fractions import Fraction as F

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_linear
from act.back_end.hybridz_tf.tf_mlp import sparse_hz_apply_relu_exact
from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as joint
from experiments.neural_hz_20260831.definition_first_20260928.d229_source_bound_bank_20261005 import binding as bridge


ZERO, ONE = F(0), F(1)
SOURCE_SCALE = (ONE, ONE, F(1, 4), F(1, 8))
PARENT_BOUNDS = ((-ONE, ONE),) * 2
CHILD_BOUNDS = ((F(-3, 2), F(1, 2)), (F(-13, 8), F(5, 8)))
PARENT_SLOTS = ((4, 5, 0), (6, 7, 1))
CHILD_SLOTS = ((8, 9, 2), (10, 11, 3))
REFERENCE = ((ZERO, ZERO, ONE, F(1, 2)),
             (ZERO, ZERO, F(1, 2), ONE))
BIASES = (F(-5, 4), F(-5, 4))
READOUT = (ZERO, ZERO, F(-1, 4), F(-1, 4), ZERO, ZERO, ONE, ONE)


def _fraction(value):
    return F.from_float(float(value))


def _matrix_row(matrix, index, values):
    start, stop = matrix.indptr[index:index + 2]
    return sum((_fraction(matrix.data[j]) * values[int(matrix.indices[j])]
                for j in range(int(start), int(stop))), ZERO)


def _output(hz, index, continuous, binary):
    return (_fraction(hz.c[index]) + _matrix_row(hz.Gc, index, continuous)
            + _matrix_row(hz.Gb, index, binary))


def _affine_value(form, continuous, binary):
    return (form.bias
            + sum((a * continuous[i] for i, a in form.continuous), ZERO)
            + sum((a * binary[i] for i, a in form.binary), ZERO))


def _native_holds(hz, continuous, binary, *, integer):
    assert len(continuous) == hz.n_cont and len(binary) == hz.n_bin
    if not all(-ONE <= value <= ONE for value in continuous + binary):
        return False
    if integer and not all(value in (-ONE, ONE) for value in binary):
        return False
    return (all(_matrix_row(hz.Ac, i, continuous)
                + _matrix_row(hz.Ab, i, binary) == _fraction(rhs)
                for i, rhs in enumerate(hz.b))
            and all(_matrix_row(hz.Auc, i, continuous)
                    + _matrix_row(hz.Aub, i, binary) <= _fraction(rhs)
                    for i, rhs in enumerate(hz.ub)))


def _snapshot(hz):
    return (hz.frame_id, hz.exact,
            tuple((a.shape, str(a.dtype), a.tobytes())
                  for a in (hz.c, hz.b, hz.ub)),
            tuple((a.shape, a.data.tobytes(), a.indices.tobytes(), a.indptr.tobytes())
                  for a in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub)))


def _pad(matrix, columns):
    return sp.hstack((matrix, sp.csr_matrix((matrix.shape[0], columns-matrix.shape[1]))),
                     format="csr")


def _fixture():
    source = SparseHZono(
        np.zeros(4), sp.diags([1., 1., 0.25, 0.125], format="csr"),
        sp.csr_matrix((4, 0)), sp.csr_matrix((0, 4)),
        sp.csr_matrix((0, 0)), np.zeros(0), frame_id=23001,
    )
    parent_pre = sparse_hz_linear(source, np.array([[1., 0., 0., 0.],
                                                   [0., 1., 0., 0.]]), np.zeros(2))
    parents = sparse_hz_apply_relu_exact(
        parent_pre, [-1., -1.], [1., 1.], PARENT_SLOTS, 8, 2,
    )
    frontier = SparseHZono(
        np.concatenate((parents.c, source.c)),
        sp.vstack((parents.Gc, _pad(source.Gc, parents.n_cont)), format="csr"),
        sp.vstack((parents.Gb, _pad(source.Gb, parents.n_bin)), format="csr"),
        parents.Ac, parents.Ab, parents.b, parents.Auc, parents.Aub, parents.ub,
        frame_id=parents.frame_id, exact=parents.exact,
    )
    child_pre = sparse_hz_linear(
        frontier, np.array([[1., 0.5, 0., 0., 1., 0.],
                            [0.5, 1., 0., 0., -1., 1.]]),
        np.array([-1.25, -1.25]),
    )
    children = sparse_hz_apply_relu_exact(
        child_pre, [-1.5, -1.625], [0.5, 0.625], CHILD_SLOTS, 12, 4,
    )
    base = SparseHZono(
        np.concatenate((children.c, parents.c, source.c)),
        sp.vstack((children.Gc, _pad(parents.Gc, 12), _pad(source.Gc, 12)),
                  format="csr"),
        sp.vstack((children.Gb, _pad(parents.Gb, 4), _pad(source.Gb, 4)),
                  format="csr"),
        children.Ac, children.Ab, children.b,
        children.Auc, children.Aub, children.ub,
        frame_id=children.frame_id, exact=children.exact,
    )
    # Keep y1,y2,q1,q2,x1,x2,z,t, then the COMPLETE physical readout F.
    hz = sparse_hz_linear(
        base, np.vstack((np.eye(8), np.array([1., 1., -0.25, -0.25, 0., 0., 0., 0.]))),
        np.zeros(9),
    )
    graphs = tuple(bridge.Graph("extended", slots, i, rows)
                   for i, (slots, rows) in enumerate(zip(
                       PARENT_SLOTS + CHILD_SLOTS,
                       ((0, 2), (1, 3), (4, 6), (5, 7)))))
    return hz, graphs


def _preactivations(source, q):
    x1, x2, z, t = source
    return (x1, x2, q[0] + q[1]/2 + z - F(5, 4),
            q[0]/2 + q[1] - z + t - F(5, 4))


def _state_for_outputs(source, outputs, binary):
    """Fill original extended-ReLU columns; no graph claim for fractional bits."""
    continuous = [ZERO] * 12
    continuous[:4] = tuple(x/scale for x, scale in zip(source, SOURCE_SCALE))
    pre = _preactivations(source, outputs[:2])
    for value, output, bit, bounds, slots in zip(
            pre, outputs, binary, PARENT_BOUNDS + CHILD_BOUNDS,
            PARENT_SLOTS + CHILD_SLOTS):
        lower, upper = bounds
        s, eta, _ = slots
        continuous[s] = (value-output)/(lower/2)-bit
        continuous[eta] = ONE-output/(upper/2)
    return tuple(continuous), tuple(binary)


def _actual_state(hz, source):
    q = tuple(max(ZERO, x) for x in source[:2])
    pre = _preactivations(source, q)
    y = tuple(max(ZERO, g) for g in pre[2:])
    assert all(g != ZERO for g in pre)  # This control has strict guards.
    bits = tuple(-ONE if g > ZERO else ONE for g in pre)
    continuous, binary = _state_for_outputs(source, q+y, bits)
    assert _native_holds(hz, continuous, binary, integer=True)
    expected = y+q+source+(y[0]+y[1]-(q[0]+q[1])/4,)
    values = tuple(_output(hz, i, continuous, binary) for i in range(9))
    assert values == expected
    return continuous, binary, values


def _box_support(bias, coefficients, bounds):
    return bias + sum((max(a*lo, a*hi)
                       for a, (lo, hi) in zip(coefficients, bounds)), ZERO)


def _local_atom(spec, physical, phases):
    f1, f2, q1, q2, v1, v2, y1, y2 = physical
    assert (q1, q2) == (max(ZERO, f1), max(ZERO, f2))
    for value, (lo, hi) in zip((f1, f2, v1, v2),
                              spec.parent_bounds + spec.residual_bounds):
        assert lo <= value <= hi
    child_pre = tuple(bias + sum((a*x for a, x in zip(row, physical[:6])), ZERO)
                      for row, bias in zip(spec.coefficients, spec.biases))
    assert (y1, y2) == tuple(max(ZERO, g) for g in child_pre)
    pre = (f1, f2) + child_pre
    assert all(g != ZERO for g in pre)
    assert phases == tuple(ONE if g > ZERO else -ONE for g in pre)
    return child_pre


def _strings(values):
    return [str(value) for value in values]


def run(*, enabled=False):
    """One opt-in diagnostic.  No files are written and no solver is invoked."""
    if enabled is not True:
        raise bridge.Rejected("D230 mathematical diagnostic is disabled")
    hz, graphs = _fixture()
    before = _snapshot(hz)
    budget = bridge.Budget()
    bound = bridge.bind(hz, graphs, REFERENCE, BIASES, enabled=True, budget=budget)
    assert bound.source_hz is hz and bound.budget is budget and bound.bank.budget is budget
    assert bound.bank.spec.residual_bounds == ((F(-1, 4), F(1, 4)),
                                               (F(-3, 8), F(3, 8)))
    receiver = bridge.receiver_bounds(bound, 8, READOUT, frame=hz)
    assert receiver.upper == F(5, 8)
    assert receiver.remainder == bridge.NativeAffine(ZERO, (), ())

    # D062's no-negative-term identity is pointwise exact.  All ORIGINAL
    # residual contributions remain in these four affine supports.
    full_box = ((ZERO, ONE), (ZERO, ONE), (F(-1, 4), F(1, 4)),
                (F(-1, 8), F(1, 8)))
    planes = ((ZERO, (F(-1, 4), F(-1, 4), ZERO, ZERO)),
              (F(-5, 4), (F(3, 4), F(1, 4), ONE, ZERO)),
              (F(-5, 4), (F(1, 4), F(3, 4), -ONE, ONE)),
              (F(-5, 2), (F(5, 4), F(5, 4), ZERO, ONE)))
    supports = tuple(_box_support(bias, coefficients, full_box)
                     for bias, coefficients in planes)
    assert supports == (ZERO, ZERO, F(1, 8), F(1, 8))
    attainment = (ONE, ONE, ZERO, F(1, 8))
    _, _, attained_values = _actual_state(hz, attainment)
    assert attained_values[8] == max(supports) == F(1, 8)

    # Positive MECHANISM control, not a new candidate: D228 already permits
    # both children to use the same fixed ORIGINAL residual coordinates.
    # Keep (z,t), rather than boxing (z,-z+t) independently.  These choices
    # are declared from the fixture, never fitted to a query or LP point.
    shared_coefficients = ((ZERO, ZERO, ONE, F(1, 2), ONE, ZERO),
                           (ZERO, ZERO, F(1, 2), ONE, -ONE, ONE))
    shared_spec = joint.Spec(
        PARENT_BOUNDS, ((F(-1, 4), F(1, 4)), (F(-1, 8), F(1, 8))),
        shared_coefficients, BIASES,
    )
    shared_identity = joint.Binding(
        hz, ("shared:f1", "shared:f2", "shared:q1", "shared:q2",
             "shared:z", "shared:t", "shared:y1", "shared:y2"),
        ("shared:phase:0", "shared:phase:1", "shared:phase:2", "shared:phase:3"),
    )
    shared_bank = joint.build(shared_spec, shared_identity, enabled=True, budget=budget)
    assert shared_bank.budget is budget
    shared_upper = shared_bank.support(READOUT, frame=hz).value
    assert shared_upper == max(supports) == F(1, 8)

    fake_source = (F(1, 2), F(1, 2), ZERO, ZERO)
    fake_qy = (F(1, 2), F(1, 2), F(1, 5), F(1, 5))
    fake_bits = (-ONE, -ONE, ZERO, ZERO)
    fake_cont, fake_bin = _state_for_outputs(fake_source, fake_qy, fake_bits)
    assert _native_holds(hz, fake_cont, fake_bin, integer=False)
    assert not _native_holds(hz, fake_cont, fake_bin, integer=True)
    assert fake_cont[8:] == (F(14, 15), F(1, 5), F(56, 65), F(9, 25))
    fake_outputs = tuple(_output(hz, i, fake_cont, fake_bin) for i in range(9))
    assert fake_outputs == (F(1, 5), F(1, 5), F(1, 2), F(1, 2)) + fake_source + (F(3, 20),)
    fake_physical = tuple(_affine_value(form, fake_cont, fake_bin)
                          for form in bound.physical)
    assert fake_physical == (F(1, 2), F(1, 2), F(1, 2), F(1, 2),
                             ZERO, ZERO, F(1, 5), F(1, 5))

    local_a = (F(19, 20), F(19, 20), F(19, 20), F(19, 20),
               F(9, 40), F(9, 40), F(2, 5), F(2, 5))
    local_b = (F(1, 20), F(1, 20), F(1, 20), F(1, 20),
               F(-9, 40), F(-9, 40), ZERO, ZERO)
    phases_a, phases_b = (ONE, ONE, ONE, ONE), (ONE, ONE, -ONE, -ONE)
    assert _local_atom(bound.bank.spec, local_a, phases_a) == (F(2, 5), F(2, 5))
    assert _local_atom(bound.bank.spec, local_b, phases_b) == (F(-7, 5), F(-7, 5))
    assert tuple((a+b)/2 for a, b in zip(local_a, local_b)) == fake_physical
    assert tuple((a+b)/2 for a, b in zip(phases_a, phases_b)) == tuple(-b for b in fake_bin)
    # An original source atom would require t=v1+v2.  These local atoms
    # cannot lift individually; their source-compatible mean can still lift
    # into the LOCAL labelled convex hull by D228's complete-hull theorem.
    assert local_a[4]+local_a[5] == F(9, 20) > SOURCE_SCALE[3]
    assert local_b[4]+local_b[5] == F(-9, 20) < -SOURCE_SCALE[3]

    source_witnesses = []
    for child_index, direction in ((0, ONE), (1, -ONE)):
        high = (F(19, 20), F(19, 20), direction*F(9, 40), ZERO)
        low = (F(1, 20), F(1, 20), -direction*F(9, 40), ZERO)
        _, high_bits, high_values = _actual_state(hz, high)
        _, low_bits, low_values = _actual_state(hz, low)
        means = tuple((a+b)/2 for a, b in zip(high_values, low_values))
        assert means[2:8] == (F(1, 2), F(1, 2)) + fake_source
        assert means[child_index] == F(1, 5)
        assert tuple((a+b)/2 for a, b in zip(high_bits[:2], low_bits[:2])) == (-ONE, -ONE)
        assert (high_bits[2+child_index]+low_bits[2+child_index])/2 == ZERO
        source_witnesses.append(dict(child=child_index+1, high=_strings(high),
                                     low=_strings(low), weight="1/2"))

    assert fake_outputs[8]-max(supports) == F(1, 40)
    threshold = F(11, 80)
    assert max(supports)-threshold == F(-1, 80)
    assert fake_outputs[8]-threshold == F(1, 80)
    assert _snapshot(hz) == before and not budget.failed
    return {
        "diagnostic": "source_mean_binding_counterexample",
        "mathematical_diagnostic_only": True,
        "d229_complete_receiver_upper": str(receiver.upper),
        "d229_complete_remainder": "0",
        "d062_four_plane_supports": _strings(supports),
        "exact_true_support": "1/8",
        "shared_source_coordinate_control": {
            "coordinate_choice": "fixed original (z,t); no fit or query search",
            "residual_bounds": [["-1/4", "1/4"], ["-1/8", "1/8"]],
            "child_coefficients": [_strings(row) for row in shared_coefficients],
            "upper": str(shared_upper),
            "bank_vertices": len(shared_bank.vertices),
            "restored_old_reference_only": True,
            "unchanged_d228_component": True,
            "native_binding_or_factor_installed": False,
        },
        "true_attainment_source": _strings(attainment),
        "fake_physical": _strings(fake_physical),
        "fake_native_continuous": _strings(fake_cont),
        "fake_native_binary": _strings(fake_bin),
        "fake_readout": "3/20",
        "fake_above_true_support": "1/40",
        "original_h_relaxation_fake_feasible": True,
        "fake_is_concrete_network_state": False,
        "local_graph_atoms": [_strings(local_a), _strings(local_b)],
        "local_atom_signed_phases": [_strings(phases_a), _strings(phases_b)],
        "local_atom_weights": ["1/2", "1/2"],
        "individual_original_source_witnesses": source_witnesses,
        "local_lambda_extension": "D228 complete labelled hull theorem; not numerically constructed",
        "local_atoms_have_original_source_witnesses": False,
        "all_fixed_w_lower_bound": "3/20",
        "all_fixed_w_scope": "this fixed Bank and original relaxed H, without added D062 cuts",
        "all_fixed_w_reason": "h_B(w)>=w*P and cube(R_w)>=R_w(P), hence their sum>=F(P)",
        "next_relu_threshold": str(threshold),
        "next_relu_true_upper": "0",
        "next_relu_fake_value": "1/80",
        "next_relu_installed_or_executed": False,
        "native_source_unchanged": True,
        "native_dimensions": {"outputs": len(hz.c), "continuous": hz.n_cont,
                              "binary": hz.n_bin, "eq": len(hz.b), "le": len(hz.ub)},
        "bank_vertices": len(bound.bank.vertices),
        "candidate_logical_cost": {"work": budget.work, "entries": budget.entries,
                                   "max_bits": budget.max_bits,
                                   "scope": "shared D228/D229 logical counters, not complete diagnostic cost"},
        "ordinary_model_executed": False,
        "gpu_executed": False,
        "native_factor_installed": False,
        "validated_adv": False,
        "formal_score_changed": False,
    }
