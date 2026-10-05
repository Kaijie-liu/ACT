"""Twenty-four exact, finite regression diagnostics for the D003 reference.

The ordinary evaluator below uses max-ReLU, not the candidate gate evaluator.
The native evidence is independently written from the displayed small graph;
it does not inspect candidate nodes, forms, bounds, rows, or compiler helpers.
These fixed rational witnesses are diagnostics, not a proof or solver result.
No phase enumeration, random sampling, split, or solver is used here.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d003_domain_v1 import (
    make_builder,
)


def _mixed_fixture():
    """Retain the builder as well as the complete frozen source for accounting."""
    builder = make_builder(2, 1, enabled=True)
    x, free = builder.inputs(), builder.binary_inputs()
    signed = builder.affine(free, ((2,),), (-1,))
    pre = builder.affine(
        builder.concat(x, signed),
        ((1, F(-1, 2), F(1, 4)), (-1, 1, F(-1, 2))),
        (F(1, 4), F(-1, 4)),
    )
    first = builder.relu(pre)
    mixed = builder.affine(
        builder.concat(first, x), ((1, -2, F(1, 2), -1),), (F(1, 8),)
    )
    second = builder.relu(mixed)
    skip = builder.affine(x, ((F(1, 4), F(-1, 2)),), (0,))
    residual = builder.add(second, skip)
    equality = builder.affine(builder.concat(x, free), ((1, 1, -1),), (0,))
    builder.constrain(equality, "eq", (0,))
    builder.constrain(pre, "le", (F(3, 2), F(3, 2)))
    builder.constrain(residual, "le", (1,))
    outputs = builder.concat(residual, first, x, free, signed, mixed)
    return builder, builder.freeze(outputs)


def _direct_mixed(inputs, free_bit):
    """Ordinary affine/ReLU/residual evaluation, independently written."""
    x, y = map(F, inputs)
    b = F(free_bit)
    signed = 2 * b - 1
    a = x - y / 2 + signed / 4 + F(1, 4)
    c = -x + y - signed / 2 - F(1, 4)
    r, s = max(F(0), a), max(F(0), c)
    mixed = r - 2 * s + x / 2 - y + F(1, 8)
    t = max(F(0), mixed)
    skip = x / 4 - y / 2
    residual = t + skip
    equality = x + y - b
    return (x, y, b, signed, a, c, r, s, mixed, t, skip, residual, equality)


def _direct_bits(nodes, free_bit, zero_phase=0):
    return (free_bit,) + tuple(
        1 if nodes[index] > 0 else 0 if nodes[index] < 0 else zero_phase
        for index in (4, 5, 8)
    )


def _direct_feasible(nodes):
    return (
        all(-1 <= value <= 1 for value in nodes[:2])
        and nodes[2] in (0, 1)
        and nodes[12] == 0
        and nodes[4] <= F(3, 2)
        and nodes[5] <= F(3, 2)
        and nodes[11] <= 1
    )


def _form(terms, constant=0):
    return tuple(sorted((index, F(value)) for index, value in terms.items())), F(constant)


def _row(terms, rhs=0, relation="le"):
    return _form(terms)[0], F(rhs), relation


def _independent_native():
    """Full independent evidence, retained by the prototype memory diagnostic.

    Variables: x,y,r,s,t,b,beta_a,beta_c,beta_m.  Interval propagation gives
    a in [-3/2,2], c in [-11/4,9/4], mixed in [-47/8,29/8].
    """
    forms = (
        _form({0: 1}),
        _form({1: 1}),
        _form({5: 1}),
        _form({5: 2}, -1),
        _form({0: 1, 1: F(-1, 2), 5: F(1, 2)}),
        _form({0: -1, 1: 1, 5: -1}, F(1, 4)),
        _form({2: 1}),
        _form({3: 1}),
        _form({0: F(1, 2), 1: -1, 2: 1, 3: -2}, F(1, 8)),
        _form({4: 1}),
        _form({0: F(1, 4), 1: F(-1, 2)}),
        _form({0: F(1, 4), 1: F(-1, 2), 4: 1}),
        _form({0: 1, 1: 1, 5: -1}),
    )
    rows = (
        _row({2: -1}),
        _row({0: 1, 1: F(-1, 2), 2: -1, 5: F(1, 2)}),
        _row({2: 1, 6: -2}),
        _row({0: -1, 1: F(1, 2), 2: 1, 5: F(-1, 2), 6: F(3, 2)}, F(3, 2)),
        _row({3: -1}),
        _row({0: -1, 1: 1, 3: -1, 5: -1}, F(-1, 4)),
        _row({3: 1, 7: F(-9, 4)}),
        _row({0: 1, 1: -1, 3: 1, 5: 1, 7: F(11, 4)}, 3),
        _row({4: -1}),
        _row({0: F(1, 2), 1: -1, 2: 1, 3: -2, 4: -1}, F(-1, 8)),
        _row({4: 1, 8: F(-29, 8)}),
        _row({0: F(-1, 2), 1: 1, 2: -1, 3: 2, 4: 1, 8: F(47, 8)}, 6),
        _row({0: 1, 1: 1, 5: -1}, 0, "eq"),
        _row({0: 1, 1: F(-1, 2), 5: F(1, 2)}, F(3, 2)),
        _row({0: -1, 1: 1, 5: -1}, F(5, 4)),
        _row({0: F(1, 4), 1: F(-1, 2), 4: 1}, 1),
    )
    outputs = (11, 6, 7, 0, 1, 2, 3, 8)
    return {
        "forms": forms,
        "outputs": outputs,
        "output_forms": tuple(forms[index] for index in outputs),
        "rows": rows,
        "continuous_bounds": ((F(-1), F(1)), (F(-1), F(1)),
                              (F(0), F(2)), (F(0), F(9, 4)), (F(0), F(29, 8))),
        "binary_ids": (0, 1, 2, 3),
        "binary_columns": (5, 6, 7, 8),
        "gates": ((6, 4, 1, 2, 6, F(-3, 2), F(2)),
                  (7, 5, 2, 3, 7, F(-11, 4), F(9, 4)),
                  (9, 8, 3, 4, 8, F(-47, 8), F(29, 8))),
        "counts": {"n_inputs": 2, "n_original_binary": 1, "n_gates": 3,
                   "n_bin": 4, "n_cont": 5, "n_nodes": 13, "n_predicates": 4,
                   "n_rows": 16, "row_nnz": 50, "row_coefficients": 66,
                   "node_form_nnz": 25, "node_form_coefficients": 38,
                   "output_form_nnz": 13, "affine_edges": 18},
    }


def _apply_form(form, assignment):
    terms, constant = form
    return constant + sum((coefficient * assignment[index] for index, coefficient in terms), F(0))


def _independent_accepts(evidence, assignment):
    if len(assignment) != 9:
        return False
    if any(not lower <= assignment[index] <= upper
           for index, (lower, upper) in enumerate(evidence["continuous_bounds"])):
        return False
    if any(assignment[index] not in (0, 1) for index in evidence["binary_columns"]):
        return False
    for terms, rhs, relation in evidence["rows"]:
        lhs = _apply_form((terms, F(0)), assignment)
        if (lhs != rhs if relation == "eq" else lhs > rhs):
            return False
    return True


def test_01_default_is_strictly_off():
    assert make_builder(2) is None
    for not_true in (False, None, 0, 1, "true"):
        assert make_builder(2, enabled=not_true) is None
    assert make_builder(-1, -1, enabled=False) is None
    assert make_builder(2, enabled=True) is not None


def test_02_exact_mixed_affine_relu_matches_ordinary_evaluator():
    _, element = _mixed_fixture()
    for inputs, bit in (((F(1, 2), F(1, 2)), 1), ((F(-1, 2), F(1, 2)), 0)):
        expected = _direct_mixed(inputs, bit)
        evaluated = element.evaluate(inputs, _direct_bits(expected, bit))
        assert evaluated.feasible
        assert evaluated.node_values == expected
        assert evaluated.outputs == tuple(expected[index] for index in (11, 6, 7, 0, 1, 2, 3, 8))
        assert all(type(value) is F for value in evaluated.node_values)


def test_03_reconvergent_residual_keeps_one_shared_frame():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    left = builder.relu(builder.affine(x, ((2,),), (-1,)))
    right = builder.relu(builder.affine(x, ((-1,),), (F(1, 4),)))
    output = builder.add(builder.add(left, right), x)
    element = builder.freeze(output)
    evaluated = element.evaluate((F(3, 4),), (1, 0))
    assert evaluated.inputs == (F(3, 4),)
    assert evaluated.outputs == (F(5, 4),)
    assert evaluated.feasible
    assert element.n_inputs == 1
    assert element.lower().n_cont == 3


def test_04_concat_preserves_order_duplicates_and_shared_ancestors():
    builder = make_builder(2, enabled=True)
    x = builder.inputs()
    repeated = builder.concat(x[1], x, x[1], x[0])
    assert repeated.nodes == (1, 0, 1, 1, 0)
    mixed = builder.affine(repeated, ((1, 2, -3, 4, -1),), (F(1, 3),))
    element = builder.freeze(builder.concat(mixed, repeated))
    result = element.evaluate((F(1, 2), F(-1, 4)), ())
    assert result.outputs == (F(1, 3), F(-1, 4), F(1, 2), F(-1, 4), F(-1, 4), F(1, 2))
    assert len(element.nodes) == 3
    native = element.lower()
    assert native.n_cont == 2
    assert native.output_values(native.assignment(result, ())) == result.outputs


def test_05_original_free_binary_and_hz_signed_recoding_have_no_guard():
    builder = make_builder(1, 2, enabled=True)
    free = builder.binary_inputs()
    signed = builder.affine(free, ((2, 0), (0, 2)), (-1, -1))
    element = builder.freeze(builder.concat(builder.inputs(), free, signed))
    native = element.lower()
    assert element.n_bin == native.n_bin == 2
    assert not element.gates and not native.rows
    for bits, signed_values in (((0, 1), (-1, 1)), ((1, 0), (1, -1))):
        result = element.evaluate((F(1, 3),), bits)
        assert result.feasible
        assert result.outputs == (F(1, 3),) + bits + signed_values
        assert native.satisfies(native.assignment(result, bits))


def test_06_eq_le_and_intermediate_predicates_remain_visible():
    _, element = _mixed_fixture()
    assert [(p.node, p.relation, p.rhs) for p in element.predicates] == [
        (12, "eq", F(0)), (4, "le", F(3, 2)),
        (5, "le", F(3, 2)), (11, "le", F(1)),
    ]
    for inputs, bit in (((F(1, 4), 0), 0), ((-1, 1), 0), ((1, -1), 0)):
        nodes = _direct_mixed(inputs, bit)
        bits = _direct_bits(nodes, bit)
        result = element.evaluate(inputs, bits)
        assert not result.feasible
        assert not element.lower().satisfies(element.lower().assignment(result, bits))


def test_07_unselected_nodes_guards_and_predicates_are_not_refunded():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    unused_pre = builder.affine(x, ((1,),), (1,))
    unused_gate = builder.relu(unused_pre)
    builder.constrain(unused_gate, "le", (F(1, 2),))
    element = builder.freeze(x)
    assert len(element.nodes) == 3
    assert len(element.gates) == len(element.predicates) == 1
    native = element.lower()
    assert (native.n_cont, native.n_bin, len(native.rows)) == (2, 1, 5)
    assert len(native.node_forms) == 3
    assert not element.evaluate((0,), (0,)).feasible
    assert not element.evaluate((0,), (1,)).feasible
    result = element.evaluate((-1,), (0,))
    assert result.feasible and native.satisfies(native.assignment(result, (0,)))


def test_08_exact_zero_keeps_both_legal_phase_witnesses():
    builder = make_builder(1, enabled=True)
    element = builder.freeze(builder.relu(builder.inputs()))
    native = element.lower()
    first = element.evaluate((0,), (0,))
    second = element.evaluate((0,), (1,))
    assert first.feasible and second.feasible
    assert first.node_values == second.node_values == (F(0), F(0))
    a, b = native.assignment(first, (0,)), native.assignment(second, (1,))
    assert a != b
    assert native.satisfies(a) and native.satisfies(b)
    assert native.output_values(a) == native.output_values(b) == (F(0),)
    assert len(element.gates) == 1 and len(native.rows) == 4


def test_09_stable_positive_and_negative_relus_keep_all_phase_bits():
    builder = make_builder(1, enabled=True)
    pre = builder.affine(builder.inputs(), ((1,), (-1,)), (2, -2))
    element = builder.freeze(builder.relu(pre))
    native = element.lower()
    result = element.evaluate((F(1, 2),), (1, 0))
    assert result.feasible and result.outputs == (F(5, 2), F(0))
    assert (element.n_bin, native.n_bin, len(element.gates), len(native.rows)) == (2, 2, 2, 8)
    assert native.satisfies(native.assignment(result, (1, 0)))
    assert not element.evaluate((F(1, 2),), (0, 0)).feasible
    assert not element.evaluate((F(1, 2),), (1, 1)).feasible
    assert [(g.lower, g.upper) for g in native.gates] == [(F(1), F(3)), (F(-3), F(-1))]


def test_10_complete_native_matches_independently_written_four_rows():
    _, element = _mixed_fixture()
    native, expected = element.lower(), _independent_native()
    assert tuple((row.terms, row.rhs, row.relation) for row in native.rows) == expected["rows"]
    assert tuple((form.terms, form.constant) for form in native.node_forms) == expected["forms"]
    assert tuple((form.terms, form.constant) for form in native.output_forms) == expected["output_forms"]
    assert native.continuous_bounds == expected["continuous_bounds"]
    assert native.input_bounds == expected["continuous_bounds"][:2]
    assert native.binary_ids == expected["binary_ids"]
    assert native.binary_columns == expected["binary_columns"]
    assert tuple((g.node, g.preactivation, g.bit, g.value_var, g.bit_var, g.lower, g.upper)
                 for g in native.gates) == expected["gates"]
    assert native.counts == expected["counts"]


def test_11_all_nodes_native_rows_outputs_and_original_witness_agree():
    _, element = _mixed_fixture()
    native, evidence = element.lower(), _independent_native()
    witnesses = (((0, 0), 0), ((F(1, 2), F(1, 2)), 1),
                 ((F(-1, 2), F(1, 2)), 0), ((1, -1), 0))
    for inputs, free_bit in witnesses:
        nodes = _direct_mixed(inputs, free_bit)
        bits = _direct_bits(nodes, free_bit)
        evaluated = element.evaluate(inputs, bits)
        assignment = native.assignment(evaluated, bits)
        independent_assignment = (nodes[0], nodes[1], nodes[6], nodes[7], nodes[9]) + tuple(map(F, bits))
        assert assignment == independent_assignment
        assert assignment[:2] == evaluated.inputs == tuple(map(F, inputs))
        assert tuple(_apply_form(form, assignment) for form in evidence["forms"]) == nodes == evaluated.node_values
        assert native.output_values(assignment) == evaluated.outputs
        assert evaluated.outputs == tuple(nodes[index] for index in evidence["outputs"])
        assert native.satisfies(assignment) == _independent_accepts(evidence, assignment) == evaluated.feasible == _direct_feasible(nodes)


def test_12_tampered_native_gate_values_and_phase_are_rejected():
    _, element = _mixed_fixture()
    native, evidence = element.lower(), _independent_native()
    inputs = (F(1, 2), F(1, 2))
    nodes = _direct_mixed(inputs, 1)
    bits = _direct_bits(nodes, 1)
    assignment = native.assignment(element.evaluate(inputs, bits), bits)
    for column in (2, 3, 4):
        tampered = list(assignment)
        tampered[column] += F(1, 16)
        assert not native.satisfies(tuple(tampered))
        assert not _independent_accepts(evidence, tuple(tampered))
    tampered = list(assignment)
    tampered[7] = 1
    assert not native.satisfies(tuple(tampered))
    assert not _independent_accepts(evidence, tuple(tampered))


def test_13_input_box_and_binary_integrality_are_checked_without_rows():
    builder = make_builder(1, 1, enabled=True)
    element = builder.freeze(builder.concat(builder.inputs(), builder.binary_inputs()))
    native = element.lower()
    assert not native.rows
    assert native.satisfies((F(1), F(0)))
    assert not native.satisfies((F(2), F(0)))
    assert not native.satisfies((F(0), F(-1)))
    assert not native.satisfies((F(0), F(2)))
    assert not native.satisfies((F(0), F(1, 2)))
    assert not element.evaluate((2,), (0,)).feasible
    assert not element.evaluate((0,), (F(1, 2),)).feasible


def test_14_cross_frame_handles_cannot_be_combined_or_frozen():
    left, right = make_builder(1, enabled=True), make_builder(1, enabled=True)
    x, foreign = left.inputs(), right.inputs()
    with pytest.raises(ValueError):
        left.add(x, foreign)
    with pytest.raises(ValueError):
        left.concat(x, foreign)
    with pytest.raises(ValueError):
        left.affine(foreign, ((1,),), (0,))
    with pytest.raises(ValueError):
        left.relu(foreign)
    with pytest.raises(ValueError):
        left.constrain(foreign, "le", (0,))
    with pytest.raises(ValueError):
        left.freeze(foreign)
    left_element, right_element = left.freeze(x), right.freeze(foreign)
    assert left_element.evaluate((0,), ()).feasible
    with pytest.raises(ValueError):
        left_element.lower().assignment(right_element.evaluate((0,), ()), ())


def test_15_freeze_is_immutable_and_closes_the_builder():
    builder, element = _mixed_fixture()
    native = element.lower()
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        element.outputs = ()
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        element.nodes[0].constant = F(1)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        native.rows[0].rhs = F(1)
    with pytest.raises(TypeError):
        element.nodes[0] = element.nodes[1]
    with pytest.raises(ValueError):
        builder.relu(builder.inputs())
    with pytest.raises(ValueError):
        builder.freeze(builder.inputs())


def test_16_vector_matrix_predicate_and_evaluation_shapes_fail_closed():
    builder = make_builder(2, 1, enabled=True)
    x = builder.inputs()
    with pytest.raises(ValueError):
        builder.affine(x, ((1,),), (0,))
    with pytest.raises(ValueError):
        builder.affine(x, ((1, 1),), (0, 0))
    with pytest.raises(ValueError):
        builder.add(x, x[0])
    with pytest.raises(ValueError):
        builder.constrain(x, "le", (0,))
    with pytest.raises(ValueError):
        builder.constrain(x, "ge", (0, 0))
    element = builder.freeze(x)
    with pytest.raises(ValueError):
        element.evaluate((0,), (0,))
    with pytest.raises(ValueError):
        element.evaluate((0, 0), ())
    with pytest.raises(ValueError):
        element.evaluate((0, 0), (0, 1))
    native = element.lower()
    with pytest.raises(ValueError):
        native.satisfies((0, 0))
    with pytest.raises(ValueError):
        native.output_values((0, 0))
    result = element.evaluate((0, 0), (0,))
    with pytest.raises(ValueError):
        native.assignment(result, (1,))


def test_17_exact_scalar_contract_rejects_floats_and_bool_numbers():
    builder = make_builder(1, 1, enabled=True)
    x = builder.inputs()
    for value in (0.5, True):
        with pytest.raises(TypeError):
            builder.affine(x, ((value,),), (0,))
        with pytest.raises(TypeError):
            builder.affine(x, ((1,),), (value,))
        with pytest.raises(TypeError):
            builder.constrain(x, "le", (value,))
    element = builder.freeze(x)
    for value in (0.5, True):
        with pytest.raises(TypeError):
            element.evaluate((value,), (0,))
        with pytest.raises(TypeError):
            element.evaluate((0,), (value,))
    assert element.evaluate((F(1, 3),), (F(1),)).feasible


def test_18_original_input_and_free_binary_dimension_caps():
    builder = make_builder(16, 8, enabled=True)
    element = builder.freeze(builder.concat(builder.inputs(), builder.binary_inputs()))
    assert (element.n_inputs, element.n_binary, element.n_bin) == (16, 8, 8)
    assert element.evaluate((0,) * 16, (0,) * 8).feasible
    for n_inputs, n_binary in ((17, 0), (1, 9), (-1, 0), (1, -1)):
        with pytest.raises(ValueError):
            make_builder(n_inputs, n_binary, enabled=True)


def test_19_total_binary_cap_keeps_original_and_all_unused_relu_bits():
    builder = make_builder(1, 8, enabled=True)
    x = builder.inputs()
    for _ in range(56):
        builder.relu(x)
    with pytest.raises(ValueError):
        builder.relu(x)
    element = builder.freeze(x)
    native = element.lower()
    assert (element.n_bin, len(element.gates), len(element.nodes)) == (64, 56, 65)
    assert (native.n_cont, native.n_bin, len(native.rows)) == (57, 64, 224)
    assert element.evaluate((0,), (0,) * 64).feasible


def test_20_node_cap_rejects_before_appending_a_partial_operation():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    for _ in range(255):
        builder.affine(x, ((1,),), (0,))
    with pytest.raises(ValueError):
        builder.affine(x, ((1,),), (0,))
    element = builder.freeze(x)
    assert len(element.nodes) == 256
    assert len(element.evaluate((F(1, 3),), ()).node_values) == 256
    assert len(element.lower().node_forms) == 256


def test_21_predicate_fanin_and_total_affine_edge_caps():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    for _ in range(128):
        builder.constrain(x, "le", (1,))
    with pytest.raises(ValueError):
        builder.constrain(x, "le", (1,))
    with pytest.raises(ValueError):
        builder.affine(builder.concat(*([x] * 33)), ((1,) * 33,), (0,))
    assert len(builder.concat(*([x] * 256)).nodes) == 256
    with pytest.raises(ValueError):
        builder.concat(*([x] * 257))
    element = builder.freeze(x)
    assert len(element.predicates) == len(element.lower().rows) == 128

    edges = make_builder(16, enabled=True)
    original = edges.inputs()
    identity = tuple(tuple(int(i == j) for j in range(16)) for i in range(16))
    copied = edges.affine(original, identity, (0,) * 16)
    wide = edges.concat(original, copied)
    for _ in range(127):
        edges.affine(wide, ((1,) * 32,), (0,))
    edges.affine(original, ((1,) * 16,), (0,))
    with pytest.raises(ValueError):
        edges.affine(original, ((1,) * 16,), (0,))
    completed = edges.freeze(original)
    assert completed.lower().counts["affine_edges"] == 4096


def test_22_stored_reduced_rational_numerator_and_denominator_caps():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    accepted = builder.affine(x, ((0,), (0,)), (F(1 << 511), F(1, 1 << 511)))
    for value in (F(1 << 512), F(1, 1 << 512)):
        with pytest.raises(ValueError):
            builder.affine(x, ((value,),), (0,))
        with pytest.raises(ValueError):
            builder.constrain(x, "le", (value,))
    element = builder.freeze(accepted)
    assert element.evaluate((0,), ()).outputs == (F(1 << 511), F(1, 1 << 511))


def test_23_derived_rational_growth_fails_closed_in_evaluation_and_lowering():
    builder = make_builder(1, enabled=True)
    large = builder.affine(builder.inputs(), ((1 << 511,),), (0,))
    doubled = builder.affine(large, ((2,),), (0,))
    element = builder.freeze(doubled)
    with pytest.raises(ValueError):
        element.evaluate((1,), ())
    with pytest.raises(ValueError):
        element.lower()
    with pytest.raises(ValueError):
        element.evaluate((F(1, 1 << 512),), ())


def test_24_rejected_multirow_mutations_leave_no_partial_nodes_or_predicates():
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    with pytest.raises(ValueError):
        builder.affine(x, ((1,), (1, 2)), (0, 0))
    with pytest.raises(TypeError):
        builder.affine(x, ((1,), (1,)), (0, 0.5))
    twice = builder.concat(x, x)
    with pytest.raises(TypeError):
        builder.constrain(twice, "le", (0, True))
    element = builder.freeze(x)
    assert len(element.nodes) == 1
    assert not element.gates and not element.predicates
    assert element.evaluate((1,), ()).feasible
    assert not element.lower().rows
