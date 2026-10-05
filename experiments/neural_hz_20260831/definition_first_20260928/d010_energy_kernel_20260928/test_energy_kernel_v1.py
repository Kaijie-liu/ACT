"""Twenty-four fixed exact-rational D010 diagnostics, not mathematical proofs.

No solver, random samples, phase enumeration, source decoding, or benchmark
claim. Ordinary graph values, four-row gates, complete native forms, and the
positive-control row are independently specified below. The fixture retains
the D003 builder and complete source/evidence for the root's memory accounting.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d003_domain_v1 import (
    make_builder,
)
from experiments.neural_hz_20260831.definition_first_20260928.d010_energy_kernel_20260928 import (
    energy_kernel_v1 as ek,
)


def _bank():
    return ek.make_bank(((1, 1), (1, F(-3, 2)), (1, F(1, 2))),
                        (0, 0, 0), (-1, -1), (1, 1), enabled=True)


def _row(terms, rhs=0, relation="le"):
    return tuple((i, F(c)) for i, c in terms), F(rhs), relation


def _independent_native():
    """x1,x2,r1,r2,r3,b,beta1,beta2,beta3; twelve gates plus original EQ/LE."""
    rows = (
        _row(((2, -1),)),
        _row(((0, 1), (1, 1), (2, -1))),
        _row(((2, 1), (6, -2))),
        _row(((0, -1), (1, -1), (2, 1), (6, 2)), 2),
        _row(((3, -1),)),
        _row(((0, 1), (1, F(-3, 2)), (3, -1))),
        _row(((3, 1), (7, F(-5, 2)))),
        _row(((0, -1), (1, F(3, 2)), (3, 1), (7, F(5, 2))), F(5, 2)),
        _row(((4, -1),)),
        _row(((0, 1), (1, F(1, 2)), (4, -1))),
        _row(((4, 1), (8, F(-3, 2)))),
        _row(((0, -1), (1, F(-1, 2)), (4, 1), (8, F(3, 2))), F(3, 2)),
        _row(((0, 1), (1, 1), (5, -1)), 0, "eq"),
        _row(((3, 1),), 2),
    )
    bounds = ((F(-1), F(1)), (F(-1), F(1)),
              (F(0), F(2)), (F(0), F(5, 2)), (F(0), F(3, 2)))
    forms = (
        (((0, F(1)),), F(0)),
        (((1, F(1)),), F(0)),
        (((5, F(1)),), F(0)),
        (((0, F(1)), (1, F(1))), F(0)),
        (((0, F(1)), (1, F(-3, 2))), F(0)),
        (((0, F(1)), (1, F(1, 2))), F(0)),
        (((2, F(1)),), F(0)),
        (((3, F(1)),), F(0)),
        (((4, F(1)),), F(0)),
        (((0, F(-3)), (2, F(2)), (3, F(2)), (4, F(2))), F(0)),
        (((0, F(1)), (1, F(1)), (5, F(-1))), F(0)),
    )
    outputs = (9, 6, 7, 8, 0, 1, 2)
    return {"rows": rows, "bounds": bounds, "forms": forms,
            "outputs": outputs, "output_forms": tuple(forms[i] for i in outputs),
            "binary_ids": (0, 1, 2, 3), "binary_columns": (5, 6, 7, 8)}


def _independent_mass_row():
    return _row(((0, -3), (2, 2), (3, 2), (4, 2)), F(9, 2))


def _independent_resource():
    """Ordinary rational algebra on its own raw source, not core proof fields."""
    weights = ((F(1), F(1)), (F(1), F(-3, 2)), (F(1), F(1, 2)))
    bias, lower, upper, mass_weights = (F(0),) * 3, (F(-1),) * 2, (F(1),) * 2, (F(1),) * 3
    midpoint = tuple((lo + hi) / 2 for lo, hi in zip(lower, upper))
    radius = tuple((hi - lo) / 2 for lo, hi in zip(lower, upper))
    center = tuple(b + sum((a * x for a, x in zip(row, midpoint)), F(0))
                   for row, b in zip(weights, bias))
    normalized = tuple(tuple(a * rad for a, rad in zip(row, radius)) for row in weights)
    gram = tuple(tuple(sum((mass_weights[i] * normalized[i][j] * normalized[i][l]
                           for i in range(3)), F(0)) for l in range(2)) for j in range(2))
    bound = sum((abs(coefficient) for row in gram for coefficient in row), F(0))
    return {"weights": weights, "bias": bias, "lower": lower, "upper": upper,
            "mass_weights": mass_weights, "midpoint": midpoint, "radius": radius,
            "center": center, "normalized": normalized, "gram": gram, "B": bound,
            "c": sum((w * abs(z) for w, z in zip(mass_weights, center)), F(0)),
            "W0": sum(mass_weights, F(0)), "q": F(9, 2),
            "row": _independent_mass_row()}


def _fixed_samples():
    """Eight individually prescribed witnesses, including invalid EQ and LE."""
    return (
        ((F(0), F(0)), (0, 0, 0, 0)),
        ((F(0), F(0)), (0, 1, 0, 1)),
        ((F(1, 2), F(1, 2)), (1, 1, 0, 1)),
        ((F(-1, 2), F(1, 2)), (0, 0, 0, 0)),
        ((F(1, 2), F(-1, 2)), (0, 1, 1, 1)),
        ((F(1), F(0)), (1, 1, 1, 1)),
        ((F(1, 2), F(1, 2)), (0, 1, 0, 1)),
        ((F(1), F(-1)), (0, 0, 1, 1)),
    )


def _fixture():
    bank = _bank()
    builder = make_builder(2, 1, enabled=True)
    inputs, free = builder.inputs(), builder.binary_inputs()
    pre = builder.affine(inputs, bank.weights, bank.bias)
    relu = builder.relu(pre)
    output = builder.affine(builder.concat(relu, inputs[0]), ((2, 2, 2, -3),), (0,))
    equality = builder.affine(builder.concat(inputs, free), ((1, 1, -1),), (0,))
    builder.constrain(equality, "eq", (0,))
    builder.constrain(relu[1], "le", (2,))
    element = builder.freeze(builder.concat(output, relu, inputs, free))
    certificate = ek.certify(bank, (1, 1, 1), F(9, 2))
    row = ek.compile_row(bank, certificate, enabled=True)
    energy = ek.gram_energy(bank, (1, 1, 1), enabled=True)
    post = ek.relu_energy(energy, enabled=True)
    forward_facts = {
        "relu": post,
        "affine": ek.affine_energy(post, ((1, -1, 0), (0, 1, 1)),
                                   (1, -1), (1, 2), enabled=True),
        "add": ek.add_energy(post, post, enabled=True),
        "concat": ek.concat_energy(post, post, enabled=True),
    }
    independent = _independent_native()
    return {
        "builder": builder, "element": element, "native": element.lower(),
        "bank": bank, "certificate": certificate, "row": row, "energy": energy,
        "forward_facts": forward_facts, "independent": independent,
        "independent_rows": independent["rows"],
        "independent_bounds": independent["bounds"],
        "independent_forms": independent["forms"],
        "independent_output_forms": independent["output_forms"],
        "independent_mass_row": _independent_mass_row(),
        "independent_resource": _independent_resource(),
        "fixed_samples": _fixed_samples(),
    }


def _direct(inputs, bits):
    """Independent affine/max evaluator retaining all source-node values."""
    x1, x2 = map(F, inputs)
    free, beta1, beta2, beta3 = map(F, bits)
    f1, f2, f3 = x1 + x2, x1 - 3 * x2 / 2, x1 + x2 / 2
    r1, r2, r3 = max(F(0), f1), max(F(0), f2), max(F(0), f3)
    output, equality = 2 * (r1 + r2 + r3) - 3 * x1, x1 + x2 - free
    nodes = (x1, x2, free, f1, f2, f3, r1, r2, r3, output, equality)
    phases_valid = all(
        beta in (0, 1) and (f == 0 or (beta == 1 if f > 0 else beta == 0))
        for f, beta in ((f1, beta1), (f2, beta2), (f3, beta3))
    )
    return {
        "inputs": (x1, x2), "bits": (free, beta1, beta2, beta3),
        "node_values": nodes, "preactivations": (f1, f2, f3),
        "relu": (r1, r2, r3), "outputs": (output, r1, r2, r3, x1, x2, free),
        "equality": equality, "readout": output,
        "feasible": (-1 <= x1 <= 1 and -1 <= x2 <= 1 and free in (0, 1)
                     and phases_valid and equality == 0 and r2 <= 2),
    }


def _row_accepts(rows, assignment):
    for row in rows:
        terms, rhs = row[:2]
        relation = row[2] if len(row) == 3 else "le"
        lhs = sum((coefficient * assignment[index] for index, coefficient in terms), F(0))
        if (lhs != rhs if relation == "eq" else lhs > rhs):
            return False
    return True


def _ordinary_accepts(assignment, integral=True):
    evidence = _independent_native()
    if len(assignment) != 9:
        return False
    if any(not lower <= assignment[i] <= upper
           for i, (lower, upper) in enumerate(evidence["bounds"])):
        return False
    if any(not 0 <= assignment[i] <= 1 for i in (5, 6, 7, 8)):
        return False
    if integral and any(assignment[i] not in (0, 1) for i in (5, 6, 7, 8)):
        return False
    return _row_accepts(evidence["rows"], assignment)


def _apply_form(form, assignment):
    return form[1] + sum((coefficient * assignment[index] for index, coefficient in form[0]), F(0))


def _energy_value(fact, values):
    return sum((w * (value - center) ** 2
                for w, value, center in zip(fact.weights, values, fact.center)), F(0))


def _fake_assignment():
    return (F(0), F(0), F(1), F(5, 4), F(3, 4), F(0), F(1, 2), F(1, 2), F(1, 2))


def test_01_default_off_is_strict_and_never_inspects_arguments():
    class Unread:
        def __iter__(self):
            raise AssertionError("disabled energy kernel inspected an argument")

    unread = Unread()
    for flag in (False, None, 0, 1, "true"):
        assert ek.make_bank(unread, unread, unread, unread, enabled=flag) is None
        assert ek.compile_row(unread, unread, enabled=flag) is None
        assert ek.gram_energy(unread, unread, enabled=flag) is None
        assert ek.relu_energy(unread, enabled=flag) is None
        assert ek.affine_energy(unread, unread, unread, unread, enabled=flag) is None
        assert ek.add_energy(unread, unread, enabled=flag) is None
        assert ek.concat_energy(unread, unread, enabled=flag) is None
    assert ek.make_bank(unread, unread, unread, unread) is None
    assert ek.compile_row(unread, unread) is None
    assert ek.gram_energy(unread, unread) is None
    assert ek.relu_energy(unread) is None
    assert ek.affine_energy(unread, unread, unread, unread) is None
    assert ek.add_energy(unread, unread) is None
    assert ek.concat_energy(unread, unread) is None


def test_02_source_is_copied_and_bank_certificate_row_energy_are_frozen():
    weights, bias, lower, upper = [[2, -1]], [3], [1, 2], [5, 4]
    bank = ek.make_bank(weights, bias, lower, upper, enabled=True)
    weights[0][0], bias[0], lower[0], upper[0] = 99, 99, -99, 99
    assert bank.weights == ((F(2), F(-1)),) and bank.bias == (F(3),)
    assert bank.lower == (F(1), F(2)) and bank.upper == (F(5), F(4))
    assert (bank.rows, bank.cols) == (1, 2)
    fixture = _fixture()
    for artifact, field, value in ((bank, "bias", (0,)),
                                   (fixture["certificate"], "B", F(0)),
                                   (fixture["row"], "rhs", F(0)),
                                   (fixture["energy"], "bound", F(0))):
        with pytest.raises(FrozenInstanceError):
            setattr(artifact, field, value)


def test_03_gram_certificate_matches_independent_source_algebra_and_literals():
    fixture = _fixture()
    cert, reference = fixture["certificate"], fixture["independent_resource"]
    assert fixture["bank"].weights == reference["weights"]
    assert fixture["bank"].bias == reference["bias"]
    assert fixture["bank"].lower == reference["lower"]
    assert fixture["bank"].upper == reference["upper"]
    assert cert.bank is fixture["bank"] and cert.weights == reference["mass_weights"]
    assert cert.center == reference["center"] == (0, 0, 0)
    assert cert.radius == reference["radius"] == (1, 1)
    assert cert.gram == reference["gram"] == ((F(3), F(0)), (F(0), F(7, 2)))
    assert cert.B == reference["B"] == F(13, 2)
    assert cert.c == reference["c"] == 0 and cert.q == reference["q"] == F(9, 2)
    assert cert.q ** 2 == F(81, 4) and reference["W0"] * cert.B == F(78, 4)
    # Absolute values AFTER Gram cancellation: entrywise absolute products give 25/2 instead.
    absolute_products = sum((abs(w * row[j] * row[l])
                             for w, row in zip(reference["mass_weights"], reference["normalized"])
                             for j in range(2) for l in range(2)), F(0))
    assert absolute_products == F(25, 2) > cert.B


def test_04_compiled_row_is_exact_repeatable_and_adds_no_variables():
    fixture = _fixture()
    bank, cert, row = fixture["bank"], fixture["certificate"], fixture["row"]
    assert (row.terms, row.rhs, "le") == fixture["independent_mass_row"]
    assert ek.compile_row(bank, cert, enabled=True) == row
    assert len(row.terms) == 4 and tuple(i for i, _ in row.terms) == (0, 2, 3, 4)
    assert row.satisfies((0, 0, 0, 0, 0))
    assert not row.satisfies(_fake_assignment()[:5])
    looser = ek.certify(bank, (1, 1, 1), 5)
    assert ek.compile_row(bank, looser, enabled=True).rhs == 5
    with pytest.raises(ValueError):
        ek.certify(bank, (1, 1, 1), 4)


def test_05_individual_ideal_phase_mixtures_accept_the_rejected_fractional_point():
    fixture, fake = _fixture(), _fake_assignment()
    # Separate exact neuron witnesses, explicitly listed, not a shared joint mixture.
    first = ((F(1), F(1), F(2), F(1)), (F(-1), F(-1), F(0), F(0)))
    second = ((F(1), F(-1), F(5, 2), F(1)), (F(-1), F(1), F(0), F(0)))
    third = ((F(1), F(1), F(3, 2), F(1)), (F(-1), F(-1), F(0), F(0)))
    for coefficients, points, expected in (
        ((F(1), F(1)), first, (0, 0, F(1), F(1, 2))),
        ((F(1), F(-3, 2)), second, (0, 0, F(5, 4), F(1, 2))),
        ((F(1), F(1, 2)), third, (0, 0, F(3, 4), F(1, 2))),
    ):
        for x1, x2, r, beta in points:
            pre = coefficients[0] * x1 + coefficients[1] * x2
            assert r == max(F(0), pre)
            assert pre >= 0 if beta == 1 else pre <= 0
        assert tuple((a + b) / 2 for a, b in zip(*points)) == expected
    assert _ordinary_accepts(fake, integral=False)
    assert not _ordinary_accepts(fake, integral=True)
    assert not _row_accepts((fixture["independent_mass_row"],), fake)
    assert _apply_form(fixture["independent_output_forms"][0], fake) == 6


def test_06_d007_control_is_checked_without_claiming_finite_samples_prove_all_cuts():
    rows = _independent_resource()["weights"]
    magnitude = (F(2), F(5, 2), F(3, 2))
    assert tuple(4 * rows[0][j] + rows[1][j] - 5 * rows[2][j] for j in range(2)) == (0, 0)
    assert rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0] == F(-5, 2)
    weighted = (F(8), F(5, 2), F(15, 2))
    assert all(weighted[i] <= sum(weighted) - weighted[i] for i in range(3))
    # D009's universal result follows from the vector triangle inequality.
    # These three fixed exact/defect instances only diagnose that formula.
    for a, center in (((F(4), F(1), F(-5)), F(0)),
                      ((F(1), F(-2), F(3)), F(0)),
                      ((F(1, 2), F(0), F(-7, 3)), F(2))):
        residual = tuple(sum((a[i] * rows[i][j] for i in range(3)), F(0)) for j in range(2))
        epsilon = abs(center) + sum(map(abs, residual), F(0))
        masses = tuple(abs(a[i]) * magnitude[i] for i in range(3))
        assert all(masses[i] <= sum(masses) - masses[i] + abs(center) + epsilon for i in range(3))
    # Full-box paper control, deliberately not the predicate-restricted fixture.
    assert abs(F(2)) + abs(F(-1, 2)) + abs(F(3, 2)) == 4
    assert abs(F(0)) + abs(F(5, 2)) + abs(F(1, 2)) == 3
    assert F(9, 2) < sum(magnitude)


def test_07_fixed_inputs_match_all_nodes_inputs_outputs_and_native_witnesses():
    fixture = _fixture()
    for inputs, bits in fixture["fixed_samples"]:
        direct = _direct(inputs, bits)
        evaluated = fixture["element"].evaluate(inputs, bits)
        assignment = fixture["native"].assignment(evaluated, bits)
        assert evaluated.node_values == direct["node_values"]
        assert evaluated.inputs == direct["inputs"] and evaluated.outputs == direct["outputs"]
        assert evaluated.feasible == direct["feasible"]
        assert assignment == direct["inputs"] + direct["relu"] + direct["bits"]
        assert tuple(_apply_form(form, assignment) for form in fixture["independent_forms"]) == direct["node_values"]
        assert fixture["native"].output_values(assignment) == direct["outputs"]
        assert fixture["native"].satisfies(assignment) == _ordinary_accepts(assignment) == direct["feasible"]
        assert fixture["row"].satisfies(assignment[:5])
        assert _row_accepts((fixture["independent_mass_row"],), assignment)


def test_08_full_native_rows_bounds_forms_predicates_and_original_bits_survive():
    fixture = _fixture()
    native, element = fixture["native"], fixture["element"]
    assert tuple((row.terms, row.rhs, row.relation) for row in native.rows) == fixture["independent_rows"]
    assert native.continuous_bounds == fixture["independent_bounds"]
    assert tuple((form.terms, form.constant) for form in native.node_forms) == fixture["independent_forms"]
    assert tuple((form.terms, form.constant) for form in native.output_forms) == fixture["independent_output_forms"]
    assert element.outputs == (9, 6, 7, 8, 0, 1, 2)
    assert (element.n_inputs, element.n_binary, element.n_bin, len(element.nodes), len(element.predicates)) == (2, 1, 4, 11, 2)
    assert native.binary_ids == (0, 1, 2, 3) and native.binary_columns == (5, 6, 7, 8)
    assert len(native.rows) == 14 and sum(len(row.terms) for row in native.rows) == 34
    strengthened_rows = fixture["independent_rows"] + (fixture["independent_mass_row"],)
    assert len(strengthened_rows) == 15 and sum(len(row[0]) for row in strengthened_rows) == 38
    assert strengthened_rows[:14] == fixture["independent_rows"]


def test_09_distinct_zero_phase_witnesses_remain_original_independent_bits():
    fixture = _fixture()
    low = fixture["element"].evaluate((0, 0), (0, 0, 0, 0))
    mixed = fixture["element"].evaluate((0, 0), (0, 1, 0, 1))
    assert low.feasible and mixed.feasible and low.node_values == mixed.node_values
    assert fixture["native"].assignment(low, (0, 0, 0, 0))[5:] == (0, 0, 0, 0)
    assert fixture["native"].assignment(mixed, (0, 1, 0, 1))[5:] == (0, 1, 0, 1)
    assert fixture["row"].satisfies((0, 0, 0, 0, 0))
    assert all(i < 5 for i, _ in fixture["row"].terms)


def test_10_original_eq_le_and_tampered_native_values_are_nonvacuously_rejected():
    fixture = _fixture()
    invalid_eq = _direct((F(1, 2), F(1, 2)), (0, 1, 0, 1))
    invalid_le = _direct((1, -1), (0, 0, 1, 1))
    assert invalid_eq["equality"] == 1 and invalid_eq["relu"][1] <= 2
    assert invalid_le["equality"] == 0 and invalid_le["relu"][1] == F(5, 2)
    for direct in (invalid_eq, invalid_le):
        assignment = direct["inputs"] + direct["relu"] + direct["bits"]
        assert not direct["feasible"] and not _ordinary_accepts(assignment)
        assert not fixture["native"].satisfies(assignment)
        assert fixture["row"].satisfies(assignment[:5])
    damaged = (F(0), F(0), F(1, 4), F(0), F(0), F(0), F(0), F(0), F(0))
    assert fixture["row"].satisfies(damaged[:5])
    assert not _ordinary_accepts(damaged) and not fixture["native"].satisfies(damaged)


def test_11_asymmetric_box_bias_center_and_flattened_compensation_are_exact():
    bank = ek.make_bank(((2, -1),), (3,), (1, 2), (5, 4), enabled=True)
    cert = ek.certify(bank, (1,), 5)
    assert cert.center == (F(6),) and cert.radius == (F(2), F(1))
    assert cert.gram == ((F(16), F(-4)), (F(-4), F(1)))
    assert (cert.B, cert.c, cert.q) == (25, 6, 5)
    row = ek.compile_row(bank, cert, enabled=True)
    assert (row.terms, row.rhs, "le") == _row(((0, -2), (1, 1), (2, 2)), 14)
    assert row.satisfies((5, 2, 11)) and row.satisfies((1, 4, 1))
    assert not _row_accepts((_row(((0, -2), (1, 1), (2, 2)), 11),), (5, 2, 11))
    fact = ek.gram_energy(bank, (1,), enabled=True)
    assert fact.center == (6,) and fact.bound == 25 and fact.owner is bank.owner
    assert _energy_value(fact, (11,)) == 25


def test_12_zero_radius_and_constant_banks_keep_the_correct_compensation():
    bank = ek.make_bank(((2, -3),), (1,), (4, -1), (4, 1), enabled=True)
    cert = ek.certify(bank, (1,), 3)
    assert cert.radius == (0, 1) and cert.center == (9,)
    assert cert.gram == ((F(0), F(0)), (F(0), F(9)))
    assert cert.B == 9 and cert.c == 9
    row = ek.compile_row(bank, cert, enabled=True)
    assert (row.terms, row.rhs, "le") == _row(((0, -2), (1, 3), (2, 2)), 13)
    assert row.satisfies((4, -1, 12))
    fixed = ek.make_bank(((1, 1), (1, -1)), (1, -2), (2, 3), (2, 3), enabled=True)
    fixed_cert = ek.certify(fixed, (1, 1), 0)
    assert fixed_cert.radius == (0, 0) and fixed_cert.center == (6, -3)
    assert fixed_cert.B == 0 and fixed_cert.c == 9
    fixed_row = ek.compile_row(fixed, fixed_cert, enabled=True)
    assert (fixed_row.terms, fixed_row.rhs, "le") == _row(((0, -2), (2, 2), (3, 2)), 8)
    assert fixed_row.satisfies((2, 3, 6, 0))
    assert ek.relu_energy(ek.gram_energy(fixed, (1, 1), enabled=True), enabled=True).center == (6, 0)


def test_13_nonnegative_mass_weights_allow_zeros_but_reject_negative_weights_or_q():
    bank = _bank()
    cert = ek.certify(bank, (1, 2, 0), 7)
    assert cert.gram == ((F(3), F(-2)), (F(-2), F(11, 2)))
    assert cert.B == F(25, 2)
    row = ek.compile_row(bank, cert, enabled=True)
    assert (row.terms, row.rhs, "le") == _row(((0, -3), (1, 2), (2, 2), (3, 4)), 7)
    zero = ek.certify(bank, (0, 0, 0), 0)
    empty = ek.compile_row(bank, zero, enabled=True)
    assert zero.B == zero.c == 0 and empty.terms == () and empty.rhs == 0
    assert empty.satisfies((0, 0, 99, -99, 0))
    with pytest.raises(ValueError):
        ek.certify(bank, (1, 2, 0), 6)
    with pytest.raises(ValueError):
        ek.certify(bank, (1, -1, 1), 9)
    with pytest.raises(ValueError):
        ek.gram_energy(bank, (1, -1, 1), enabled=True)
    with pytest.raises(ValueError):
        ek.certify(bank, (0, 0, 0), -1)


def test_14_relu_propagates_diagonal_energy_without_claiming_general_quadratic_contraction():
    fixture = _fixture()
    original, post = fixture["energy"], fixture["forward_facts"]["relu"]
    assert original.kind == "gram" and original.parents == ()
    assert original.center == (0, 0, 0) and original.weights == (1, 1, 1)
    assert original.bound == F(13, 2) and original.factor == 1
    assert post.kind == "relu" and post.parents[0] is original
    assert post.bound == original.bound and post.weights == original.weights
    assert post.center == (0, 0, 0) and post.bank is fixture["bank"]
    for inputs, bits in fixture["fixed_samples"]:
        direct = _direct(inputs, bits)
        assert _energy_value(post, direct["relu"]) <= _energy_value(original, direct["preactivations"]) <= original.bound
        magnitude = tuple(2 * r - f for r, f in zip(direct["relu"], direct["preactivations"]))
        assert sum(value * value for value in magnitude) == sum(value * value for value in direct["preactivations"])
    # The rank-one non-diagonal Q=ones does not contract: no such rule is used.
    assert (F(1) + F(-1)) ** 2 == 0 < (max(F(1), F(0)) + max(F(-1), F(0))) ** 2


def test_15_affine_energy_uses_a_checked_diagonal_majorant_and_exact_center():
    fixture = _fixture()
    post, affine = fixture["forward_facts"]["relu"], fixture["forward_facts"]["affine"]
    expected_gram = ((F(1), F(-1), F(0)), (F(-1), F(3), F(2)), (F(0), F(2), F(2)))
    assert tuple(sum(map(abs, row), F(0)) for row in expected_gram) == (2, 6, 4)
    assert affine.kind == "affine" and affine.parents[0] is post
    assert affine.matrix == ((1, -1, 0), (0, 1, 1)) and affine.bias == (1, -1)
    assert affine.weights == (1, 2) and affine.center == (1, -1)
    assert affine.factor == 6 and affine.bound == 39 and affine.owner is post.owner
    for inputs, bits in fixture["fixed_samples"]:
        r1, r2, r3 = _direct(inputs, bits)["relu"]
        assert _energy_value(affine, (r1 - r2 + 1, r2 + r3 - 1)) <= affine.bound
    biased = ek.gram_energy(ek.make_bank(((1,), (-1,)), (2, -3), (-1,), (1,), enabled=True), (1, 1), enabled=True)
    image = ek.affine_energy(biased, ((2, -1),), (4,), (1,), enabled=True)
    assert image.center == (11,) and image.factor == 6 and image.bound == 12


def test_16_affine_rejects_unweighted_dependency_but_allows_zero_directions():
    bank = ek.make_bank(((1,), (2,)), (0, 0), (-1,), (1,), enabled=True)
    fact = ek.gram_energy(bank, (1, 0), enabled=True)
    assert fact.weights == (1, 0) and fact.bound == 1
    accepted = ek.affine_energy(fact, ((3, 0),), (2,), (1,), enabled=True)
    assert accepted.factor == 9 and accepted.bound == 9 and accepted.center == (2,)
    with pytest.raises(ValueError):
        ek.affine_energy(fact, ((0, 1),), (0,), (1,), enabled=True)
    all_zero = ek.affine_energy(fact, ((0, 0),), (7,), (1,), enabled=True)
    assert all_zero.factor == 0 and all_zero.bound == 0 and all_zero.center == (7,)
    with pytest.raises(ValueError):
        ek.affine_energy(fact, ((1, 0),), (0,), (-1,), enabled=True)


def test_17_shared_add_uses_the_same_parent_and_does_not_assume_independence():
    fixture = _fixture()
    post, added = fixture["forward_facts"]["relu"], fixture["forward_facts"]["add"]
    assert added.parents[0] is post and added.parents[1] is post
    assert added.bank is fixture["bank"] and added.owner is post.owner
    assert added.center == (0, 0, 0) and added.weights == (1, 1, 1)
    assert added.bound == 26 and added.factor == 2 and added.width == 3
    for inputs, bits in fixture["fixed_samples"]:
        values = _direct(inputs, bits)["relu"]
        assert _energy_value(added, tuple(2 * value for value in values)) <= added.bound
    negated = ek.affine_energy(post, ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
                               (0, 0, 0), (1, 1, 1), enabled=True)
    canceled = ek.add_energy(post, negated, enabled=True)
    assert canceled.bound == 26  # Sound resource loss, not a claim that shared values are independent.
    assert canceled.parents[0] is post and canceled.parents[1].parents[0] is post
    builder = make_builder(1, enabled=True)
    x = builder.inputs()
    negative = builder.affine(x, ((-1,),), (0,))
    exact = builder.freeze(builder.add(x, negative))
    assert exact.evaluate((F(1, 3),), ()).outputs == (0,)


def test_18_concat_keeps_shared_ancestry_and_block_diagonal_energy():
    fixture = _fixture()
    post, joined = fixture["forward_facts"]["relu"], fixture["forward_facts"]["concat"]
    assert joined.parents[0] is post and joined.parents[1] is post
    assert joined.bank is post.bank and joined.owner is post.owner
    assert joined.width == 6 and joined.center == (0, 0, 0, 0, 0, 0)
    assert joined.weights == (1, 1, 1, 1, 1, 1) and joined.bound == 13
    assert joined.factor == 1 and joined.matrix == joined.bias == ()
    for inputs, bits in fixture["fixed_samples"]:
        values = _direct(inputs, bits)["relu"]
        assert _energy_value(joined, values + values) == 2 * _energy_value(post, values) <= joined.bound


def test_19_cross_bank_certificates_and_forward_proofs_are_rejected():
    first, second = _bank(), _bank()
    assert first.weights == second.weights and first is not second and first.owner is not second.owner
    cert = ek.certify(first, (1, 1, 1), F(9, 2))
    with pytest.raises(ValueError):
        ek.compile_row(second, cert, enabled=True)
    left = ek.gram_energy(first, (1, 1, 1), enabled=True)
    right = ek.gram_energy(second, (1, 1, 1), enabled=True)
    with pytest.raises(ValueError):
        ek.add_energy(left, right, enabled=True)
    with pytest.raises(ValueError):
        ek.concat_energy(left, right, enabled=True)
    forged = replace(ek.relu_energy(left, enabled=True), bank=second)
    with pytest.raises(ValueError):
        ek.relu_energy(forged, enabled=True)
    with pytest.raises(ValueError):
        ek.gram_energy(replace(first, owner=None), (1, 1, 1), enabled=True)


def test_20_tampered_mass_and_recursive_energy_payloads_are_recomputed():
    fixture = _fixture()
    bank, cert, energy = fixture["bank"], fixture["certificate"], fixture["energy"]
    for damaged in (replace(cert, B=F(6)),
                    replace(cert, c=F(1)),
                    replace(cert, q=F(4)),
                    replace(cert, center=(F(1), F(0), F(0))),
                    replace(cert, radius=(F(0), F(1))),
                    replace(cert, gram=((F(3), F(1)), (F(1), F(7, 2)))),
                    replace(cert, weights=(F(2), F(1), F(1)))):
        with pytest.raises(ValueError):
            ek.compile_row(bank, damaged, enabled=True)
    for damaged in (replace(energy, bound=F(0)),
                    replace(energy, center=(F(1), F(0), F(0))),
                    replace(energy, weights=(F(1), F(0), F(1))),
                    replace(energy, factor=F(2))):
        with pytest.raises(ValueError):
            ek.relu_energy(damaged, enabled=True)
    post = fixture["forward_facts"]["relu"]
    hidden_damage = replace(post, parents=(replace(energy, bound=F(0)),))
    with pytest.raises(ValueError):
        ek.concat_energy(hidden_damage, post, enabled=True)
    affine = fixture["forward_facts"]["affine"]
    with pytest.raises(ValueError):
        ek.relu_energy(replace(affine, factor=F(5)), enabled=True)
    assert fixture["row"] == ek.compile_row(bank, cert, enabled=True)


def test_21_exact_types_and_rows_only_contract_are_explicit():
    fixture = _fixture()
    bank, row, energy = fixture["bank"], fixture["row"], fixture["energy"]
    for invalid in (True, 0.5):
        with pytest.raises(TypeError):
            ek.make_bank(((invalid,),), (0,), (-1,), (1,), enabled=True)
        with pytest.raises(TypeError):
            ek.make_bank(((1,),), (invalid,), (-1,), (1,), enabled=True)
        with pytest.raises(TypeError):
            ek.make_bank(((1,),), (0,), (invalid,), (1,), enabled=True)
        with pytest.raises(TypeError):
            ek.make_bank(((1,),), (0,), (-1,), (invalid,), enabled=True)
        with pytest.raises(TypeError):
            ek.certify(bank, (1, invalid, 1), 9)
        with pytest.raises(TypeError):
            ek.certify(bank, (1, 1, 1), invalid)
        with pytest.raises(TypeError):
            ek.gram_energy(bank, (1, invalid, 1), enabled=True)
        with pytest.raises(TypeError):
            ek.affine_energy(energy, ((invalid, 0, 0),), (0,), (1,), enabled=True)
        with pytest.raises(TypeError):
            row.satisfies((invalid, 0, 0, 0, 0))
    # It is only a linear-row helper: neither its box nor its ReLU graph is checked.
    assert row.satisfies((2, 0, 0, 0, 0))
    assert row.satisfies((0, 0, 1, 0, 0))


def test_22_source_shapes_row_arities_and_forward_shapes_fail_closed():
    for weights, bias, lower, upper in (
        ((), (), (-1,), (1,)),
        (((),), (0,), (), ()),
        (((1, 0), (1,)), (0, 0), (-1, -1), (1, 1)),
        (((1,),), (), (-1,), (1,)),
        (((1,),), (0,), (-1,), (1, 2)),
        (((1,),), (0,), (2,), (1,)),
    ):
        with pytest.raises(ValueError):
            ek.make_bank(weights, bias, lower, upper, enabled=True)
    fixture = _fixture()
    bank, energy = fixture["bank"], fixture["energy"]
    with pytest.raises(ValueError):
        ek.certify(bank, (1, 1), 9)
    with pytest.raises(ValueError):
        ek.gram_energy(bank, (1, 1, 1, 0), enabled=True)
    with pytest.raises(ValueError):
        fixture["row"].satisfies((0, 0, 0, 0))
    with pytest.raises(ValueError):
        fixture["row"].satisfies((0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        ek.affine_energy(energy, ((1, 0),), (0,), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.affine_energy(energy, ((1, 0, 0),), (), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.affine_energy(energy, ((1, 0, 0),), (0,), (1, 1), enabled=True)
    differently_weighted = ek.gram_energy(bank, (1, 0, 1), enabled=True)
    with pytest.raises(ValueError):
        ek.add_energy(energy, differently_weighted, enabled=True)
    with pytest.raises(ValueError):
        ek.add_energy(energy, fixture["forward_facts"]["concat"], enabled=True)


def test_23_bank_vector_and_proof_dag_caps_are_bounded_with_shared_parent_reuse():
    boundary = ek.make_bank(tuple((0,) * 32 for _ in range(64)), (0,) * 64,
                            (-1,) * 32, (1,) * 32, enabled=True)
    assert (boundary.rows, boundary.cols) == (64, 32)
    with pytest.raises(ValueError):
        ek.make_bank(tuple((0,) for _ in range(65)), (0,) * 65, (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.make_bank(((0,) * 33,), (0,), (-1,) * 33, (1,) * 33, enabled=True)
    # The vector cap does not need repeated dense 32-port Gram work.
    wide_bank = ek.make_bank(tuple((0,) for _ in range(64)), (0,) * 64,
                             (-1,), (1,), enabled=True)
    wide = ek.gram_energy(wide_bank, (0,) * 64, enabled=True)
    with pytest.raises(ValueError):
        ek.concat_energy(wide, wide, enabled=True)
    with pytest.raises(ValueError):
        ek.affine_energy(wide, tuple((0,) * 64 for _ in range(65)), (0,) * 65, (1,) * 65, enabled=True)
    small = ek.make_bank(((1,),), (0,), (-1,), (1,), enabled=True)
    leaf = ek.gram_energy(small, (1,), enabled=True)
    chain = leaf
    for _ in range(31):  # A fixed proof-depth boundary, not an input/phase search.
        chain = ek.relu_energy(chain, enabled=True)
    with pytest.raises(ValueError):
        ek.relu_energy(chain, enabled=True)
    shared = leaf
    for _ in range(8):
        shared = ek.add_energy(shared, shared, enabled=True)
        assert shared.parents[0] is shared.parents[1]
    # Thirty-two distinct leaves and a balanced proof tree have 63 unique nodes.
    level = tuple(ek.gram_energy(small, (1,), enabled=True) for _ in range(32))
    while len(level) > 1:
        level = tuple(ek.add_energy(level[i], level[i + 1], enabled=True)
                      for i in range(0, len(level), 2))
    sixty_four = ek.relu_energy(level[0], enabled=True)
    with pytest.raises(ValueError):
        ek.relu_energy(sixty_four, enabled=True)


def test_24_stored_and_derived_512_bit_overflows_decline_only_the_certificate():
    with pytest.raises(ValueError):
        ek.make_bank(((2 ** 512,),), (0,), (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.make_bank(((F(1, 2 ** 512),),), (0,), (-1,), (1,), enabled=True)
    large_source = ek.make_bank(((2 ** 511,),), (0,), (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.gram_energy(large_source, (1,), enabled=True)
    zero_source = ek.make_bank(((0,),), (0,), (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        ek.certify(zero_source, (0,), 2 ** 256)
    large_weight = ek.certify(zero_source, (2 ** 511,), 0)
    with pytest.raises(ValueError):
        ek.compile_row(zero_source, large_weight, enabled=True)
    powered = ek.make_bank(((2 ** 254,),), (0,), (-1,), (1,), enabled=True)
    original = ek.gram_energy(powered, (1,), enabled=True)
    doubled = ek.add_energy(original, original, enabled=True)
    assert original.bound == 2 ** 508 and doubled.bound == 2 ** 510
    with pytest.raises(ValueError):
        ek.add_energy(doubled, doubled, enabled=True)
    with pytest.raises(ValueError):
        ek.affine_energy(original, ((2 ** 256,),), (0,), (1,), enabled=True)
    fixture = _fixture()
    before = (fixture["native"].rows, fixture["native"].binary_ids, fixture["element"].predicates)
    with pytest.raises(ValueError):
        ek.certify(fixture["bank"], (F(1, 2 ** 512), 1, 1), F(9, 2))
    assert (fixture["native"].rows, fixture["native"].binary_ids, fixture["element"].predicates) == before
