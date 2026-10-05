"""Twenty fixed, exact-rational D008 diagnostics; no solver or phase search.

The ordinary evaluator, native rows, bounds, and bundle rows below are written
independently of the candidate compiler. Finite witnesses are diagnostics, not
a proof, benchmark result, or admission of an original network. The fixture
retains its complete D003 source/builder for the root's full-cost accounting.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d003_domain_v1 import (
    make_builder,
)
from experiments.neural_hz_20260831.definition_first_20260928.d008_bundle_kernel_20260928.bundle_kernel_v1 import (
    certify,
    compile_rows,
    make_bank,
    select_certificate,
)


def _bank():
    return make_bank(
        ((1, F(1, 2)), (F(-1, 2), 1), (F(1, 2), F(3, 2))),
        (0, 0, 0), (-1, -1), (1, 1), enabled=True,
    )


def _row(terms, rhs=0, relation="le"):
    return tuple((index, F(coefficient)) for index, coefficient in terms), F(rhs), relation


def _independent_native():
    """Native order x1,x2,r1,r2,r3,b,beta1,beta2,beta3; all 14 rows."""
    rows = (
        _row(((2, -1),)),
        _row(((0, 1), (1, F(1, 2)), (2, -1))),
        _row(((2, 1), (6, F(-3, 2)))),
        _row(((0, -1), (1, F(-1, 2)), (2, 1), (6, F(3, 2))), F(3, 2)),
        _row(((3, -1),)),
        _row(((0, F(-1, 2)), (1, 1), (3, -1))),
        _row(((3, 1), (7, F(-3, 2)))),
        _row(((0, F(1, 2)), (1, -1), (3, 1), (7, F(3, 2))), F(3, 2)),
        _row(((4, -1),)),
        _row(((0, F(1, 2)), (1, F(3, 2)), (4, -1))),
        _row(((4, 1), (8, -2))),
        _row(((0, F(-1, 2)), (1, F(-3, 2)), (4, 1), (8, 2)), 2),
        _row(((0, 1), (1, 1), (5, -1)), 0, "eq"),
        _row(((0, F(-1, 2)), (1, 1), (2, 1), (3, -1), (4, -1)), 2),
    )
    bounds = ((F(-1), F(1)), (F(-1), F(1)),
              (F(0), F(3, 2)), (F(0), F(3, 2)), (F(0), F(2)))
    forms = (
        (((0, F(1)),), F(0)),
        (((1, F(1)),), F(0)),
        (((5, F(1)),), F(0)),
        (((0, F(1)), (1, F(1, 2))), F(0)),
        (((0, F(-1, 2)), (1, F(1))), F(0)),
        (((0, F(1, 2)), (1, F(3, 2))), F(0)),
        (((2, F(1)),), F(0)),
        (((3, F(1)),), F(0)),
        (((4, F(1)),), F(0)),
        (((0, F(-1, 2)), (1, F(1)), (2, F(1)), (3, F(-1)), (4, F(-1))), F(0)),
        (((0, F(1)), (1, F(1)), (5, F(-1))), F(0)),
    )
    outputs = (9, 6, 7, 8, 0, 1, 2)
    return {"rows": rows, "bounds": bounds, "forms": forms,
            "outputs": outputs, "output_forms": tuple(forms[i] for i in outputs),
            "binary_ids": (0, 1, 2, 3), "binary_columns": (5, 6, 7, 8)}


def _independent_bundle_rows():
    """Three triangle rows plus the retained homogeneous magnitude anchor.

    Divide each displayed inequality by its first nonzero coefficient's
    absolute value, without reversing its sign. Coordinates are x1,x2,r1,r2,r3.
    """
    return (
        _row(((0, -1), (1, 2), (2, 2), (3, -2), (4, -2))),
        _row(((0, 1), (1, F(1, 2)), (2, -1), (3, 1), (4, -1))),
        _row(((2, -1), (3, -1), (4, 1))),
        _row(((0, 1), (1, 3), (2, -2), (3, -2), (4, -2))),
    )


def _fixed_samples():
    """Individually specified inputs/bits; no generated phase combinations."""
    return (
        ((F(0), F(0)), (0, 0, 0, 0)),
        ((F(0), F(0)), (0, 1, 0, 1)),
        ((F(1, 2), F(1, 2)), (1, 1, 1, 1)),
        ((F(-1, 2), F(1, 2)), (0, 0, 1, 1)),
        ((F(1), F(-1)), (0, 1, 0, 0)),
        ((F(1, 2), F(1, 2)), (0, 1, 1, 1)),
    )


def _fixture():
    bank = _bank()
    builder = make_builder(2, 1, enabled=True)
    inputs, free = builder.inputs(), builder.binary_inputs()
    pre = builder.affine(inputs, bank.weights, bank.bias)
    relu = builder.relu(pre)
    output = builder.affine(builder.concat(relu, pre[1]), ((1, -1, -1, 1),), (0,))
    equality = builder.affine(builder.concat(inputs, free), ((1, 1, -1),), (0,))
    builder.constrain(equality, "eq", (0,))
    builder.constrain(output, "le", (2,))
    element = builder.freeze(builder.concat(output, relu, inputs, free))
    certificate = certify(bank, (1, 1, -1))
    independent = _independent_native()
    return {
        "builder": builder, "element": element, "native": element.lower(),
        "bank": bank, "certificate": certificate,
        "closure": compile_rows(bank, (certificate,), enabled=True),
        "independent": independent, "independent_rows": independent["rows"],
        "independent_bounds": independent["bounds"],
        "independent_forms": independent["forms"],
        "independent_output": independent["outputs"],
        "independent_output_forms": independent["output_forms"],
        "independent_bundle_rows": _independent_bundle_rows(),
        "fixed_samples": _fixed_samples(),
    }


def _direct(inputs, bits):
    """Independent ordinary affine/max-ReLU/full-source evaluation."""
    x1, x2 = map(F, inputs)
    free, beta1, beta2, beta3 = map(F, bits)
    f1, f2, f3 = x1 + x2 / 2, -x1 / 2 + x2, x1 / 2 + 3 * x2 / 2
    r1, r2, r3 = max(F(0), f1), max(F(0), f2), max(F(0), f3)
    output, equality = r1 - r2 - r3 + f2, x1 + x2 - free
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
                     and phases_valid and equality == 0 and output <= 2),
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


def _compiled_rows(closure):
    return tuple((row.terms, row.rhs, "le") for row in closure.rows)


def _apply_form(form, assignment):
    return form[1] + sum((coefficient * assignment[index] for index, coefficient in form[0]), F(0))


def test_01_default_off_is_strict_and_does_not_inspect_arguments():
    class Unread:
        def __iter__(self):
            raise AssertionError("disabled kernel inspected its input")

    unread = Unread()
    for flag in (False, None, 0, 1, "true"):
        assert make_bank(unread, unread, unread, unread, enabled=flag) is None
        assert select_certificate(unread, enabled=flag) is None
        assert compile_rows(unread, unread, enabled=flag) is None
    assert make_bank(unread, unread, unread, unread) is None
    assert select_certificate(unread) is None
    assert compile_rows(unread, unread) is None


def test_02_bank_copies_exact_source_and_all_artifacts_are_frozen():
    weights, bias, lower, upper = [[1, F(1, 2)]], [0], [-1, -1], [1, 1]
    bank = make_bank(weights, bias, lower, upper, enabled=True)
    weights[0][0], bias[0], lower[0], upper[0] = 9, 9, -9, 9
    assert bank.weights == ((F(1), F(1, 2)),)
    assert bank.bias == (F(0),) and bank.lower == (F(-1), F(-1))
    assert bank.upper == (F(1), F(1)) and (bank.rows, bank.cols) == (1, 2)
    fixture = _fixture()
    for artifact, field, value in ((bank, "bias", (1,)),
                                   (fixture["certificate"], "epsilon", F(1)),
                                   (fixture["closure"], "rows", ()),
                                   (fixture["closure"].rows[0], "rhs", F(1))):
        with pytest.raises(FrozenInstanceError):
            setattr(artifact, field, value)


def test_03_exact_relation_certificate_has_complete_zero_defect_evidence():
    bank = _bank()
    certificate = certify(bank, (1, 1, -1))
    assert certificate.bank is bank
    assert certificate.a == (F(1), F(1), F(-1))
    assert certificate.support == (0, 1, 2)
    assert certificate.d == (F(0), F(0))
    assert (certificate.c, certificate.emin, certificate.emax, certificate.epsilon) == (0, 0, 0, 0)


def test_04_selection_is_deterministic_and_matches_the_mixed_relation():
    bank = _bank()
    first = select_certificate(bank, enabled=True)
    second = select_certificate(bank, enabled=True)
    assert len(first) == 1 and first == second
    assert first[0].bank is bank and first[0].a == (F(-1), F(-1), F(1))
    assert first[0].support == (0, 1, 2) and first[0].epsilon == 0
    assert set(_compiled_rows(compile_rows(bank, first, enabled=True))) == set(_independent_bundle_rows())


def test_05_compiler_matches_handwritten_triangle_rows_and_anchor():
    fixture = _fixture()
    closure = fixture["closure"]
    assert closure.bank is fixture["bank"]
    assert closure.certs == (fixture["certificate"],)
    assert set(_compiled_rows(closure)) == set(fixture["independent_bundle_rows"])
    assert len(closure.rows) == 4 and sum(len(row.terms) for row in closure.rows) == 18
    assert closure.counts["n_ports"] == 2 and closure.counts["n_activations"] == 3
    assert closure.counts["n_rows"] == 4 and closure.counts["row_nnz"] == 18
    assert closure.counts["row_coefficients"] == 22
    assert all(index < 5 for row in closure.rows for index, _ in row.terms)


def test_06_fixed_individual_hull_witness_is_excluded_by_joint_rows():
    fixture = _fixture()
    fake = (F(0), F(0), F(1, 2), F(0), F(0), F(0), F(1, 2), F(0), F(0))
    # Neuron1 uses the equal mixture at x=(1,0),(-1,0); neurons2/3 use zero.
    neuron1_points = ((F(1), F(0), F(1), F(1)), (F(-1), F(0), F(0), F(0)))
    assert tuple((left + right) / 2 for left, right in zip(*neuron1_points)) == (0, 0, F(1, 2), F(1, 2))
    assert _ordinary_accepts(fake, integral=False)
    assert not _ordinary_accepts(fake, integral=True)
    assert not _row_accepts(fixture["independent_bundle_rows"], fake)
    assert not fixture["closure"].satisfies_rows(fake[:2], fake[2:5])
    assert _apply_form(fixture["independent_output_forms"][0], fake) == F(1, 2)


def test_07_fixed_exact_inputs_match_all_nodes_outputs_and_original_witnesses():
    fixture = _fixture()
    for inputs, bits in fixture["fixed_samples"]:
        direct = _direct(inputs, bits)
        evaluated = fixture["element"].evaluate(inputs, bits)
        assignment = fixture["native"].assignment(evaluated, bits)
        assert evaluated.node_values == direct["node_values"]
        assert evaluated.outputs == direct["outputs"] and evaluated.inputs == direct["inputs"]
        assert evaluated.feasible == direct["feasible"]
        assert assignment == direct["inputs"] + direct["relu"] + direct["bits"]
        assert tuple(_apply_form(form, assignment) for form in fixture["independent_forms"]) == direct["node_values"]
        assert fixture["native"].output_values(assignment) == direct["outputs"]
        assert fixture["native"].satisfies(assignment) == _ordinary_accepts(assignment) == direct["feasible"]
        assert fixture["closure"].satisfies_rows(inputs, direct["relu"])
        assert _row_accepts(fixture["independent_bundle_rows"], assignment)


def test_08_complete_native_reference_retains_original_free_bit_and_predicates():
    fixture = _fixture()
    native, element = fixture["native"], fixture["element"]
    assert tuple((row.terms, row.rhs, row.relation) for row in native.rows) == fixture["independent_rows"]
    assert native.continuous_bounds == fixture["independent_bounds"]
    assert tuple((form.terms, form.constant) for form in native.node_forms) == fixture["independent_forms"]
    assert tuple((form.terms, form.constant) for form in native.output_forms) == fixture["independent_output_forms"]
    assert element.outputs == fixture["independent_output"]
    assert (element.n_inputs, element.n_binary, element.n_bin, len(element.nodes), len(element.predicates)) == (2, 1, 4, 11, 2)
    assert native.binary_ids == (0, 1, 2, 3) and native.binary_columns == (5, 6, 7, 8)
    assert len(native.rows) == 14 and sum(len(row.terms) for row in native.rows) == 38
    assert len(native.rows) + len(fixture["closure"].rows) == 18
    assert not _direct((F(1, 2), F(1, 2)), (0, 1, 1, 1))["feasible"]


def test_09_zero_boundary_witnesses_keep_their_distinct_original_phases():
    fixture = _fixture()
    zero_low = fixture["element"].evaluate((0, 0), (0, 0, 0, 0))
    zero_mixed = fixture["element"].evaluate((0, 0), (0, 1, 0, 1))
    zero_high = fixture["element"].evaluate((0, 0), (0, 1, 1, 1))
    assert zero_low.feasible and zero_mixed.feasible and zero_high.feasible
    assert zero_low.node_values == zero_mixed.node_values == zero_high.node_values
    assert fixture["native"].assignment(zero_low, (0, 0, 0, 0))[5:] == (0, 0, 0, 0)
    assert fixture["native"].assignment(zero_mixed, (0, 1, 0, 1))[5:] == (0, 1, 0, 1)
    assert fixture["closure"].satisfies_rows((0, 0), (0, 0, 0))


def test_10_nonzero_bias_emits_compensated_members_and_nontrivial_anchor():
    bank = make_bank(((1,), (-1,)), (2, 1), (-1,), (1,), enabled=True)
    certificate = certify(bank, (1, 1))
    assert certificate.d == (0,) and certificate.c == 3 and certificate.epsilon == 0
    closure = compile_rows(bank, (certificate,), enabled=True)
    expected = (_row(((0, -1), (1, 1), (2, -1)), 2),
                _row(((0, 1), (1, -1), (2, 1)), 1),
                _row(((1, -1), (2, -1)), -3))
    assert set(_compiled_rows(closure)) == set(expected)
    assert closure.satisfies_rows((0,), (2, 1))
    assert not closure.satisfies_rows((0,), (1, 1))


def test_11_asymmetric_box_residual_center_and_bias_compensation_are_exact():
    bank = make_bank(((1, F(1, 4)), (-1, F(1, 4))), (2, 1), (-1, 2), (2, 6), enabled=True)
    certificate = certify(bank, (1, 1))
    assert certificate.d == (0, F(1, 2))
    assert (certificate.c, certificate.emin, certificate.emax, certificate.epsilon) == (5, -1, 1, 1)
    closure = compile_rows(bank, (certificate,), enabled=True)
    expected = (_row(((0, -1), (2, 1), (3, -1)), F(7, 2)),
                _row(((0, 1), (2, -1), (3, 1)), F(5, 2)),
                _row(((1, 1), (2, -4), (3, -4)), -14))
    assert set(_compiled_rows(closure)) == set(expected)
    assert closure.satisfies_rows((-1, 2), (F(3, 2), F(5, 2)))
    assert closure.satisfies_rows((2, 6), (F(11, 2), F(1, 2)))
    assert not closure.satisfies_rows((0, 4), (0, 0))


def test_12_repeated_compilation_and_scaled_certificates_have_identical_rows():
    bank = _bank()
    certificate = certify(bank, (1, 1, -1))
    first = compile_rows(bank, (certificate,), enabled=True)
    repeated = compile_rows(bank, first.certs, enabled=True)
    scaled = compile_rows(bank, (certify(bank, (-2, -2, 2)),), enabled=True)
    assert first.rows == repeated.rows == scaled.rows
    assert first.counts == repeated.counts
    assert len(set(_compiled_rows(first))) == len(first.rows)


def test_13_equal_but_independent_bank_identity_cannot_reuse_a_certificate():
    first, second = _bank(), _bank()
    assert first.weights == second.weights and first is not second
    certificate = certify(first, (1, 1, -1))
    with pytest.raises(ValueError):
        compile_rows(second, (certificate,), enabled=True)
    assert compile_rows(first, (certificate,), enabled=True).bank is first


def test_14_tampered_certificate_proof_fields_are_rejected():
    bank = _bank()
    certificate = certify(bank, (1, 1, -1))
    for damaged in (replace(certificate, epsilon=F(1)),
                    replace(certificate, c=F(1)),
                    replace(certificate, d=(F(1), F(0))),
                    replace(certificate, emin=F(-1)),
                    replace(certificate, emax=F(1)),
                    replace(certificate, support=(0, 1)),
                    replace(certificate, a=(F(1), F(1), F(1)))):
        with pytest.raises(ValueError):
            compile_rows(bank, (damaged,), enabled=True)


def test_15_bank_shapes_bounds_and_certificate_arity_fail_closed():
    for weights, bias, lower, upper in (
        (((1, 0), (1,)), (0, 0), (-1, -1), (1, 1)),
        (((1,),), (), (-1,), (1,)),
        (((1,),), (0,), (), (1,)),
        (((1,),), (0,), (2,), (1,)),
        (((1,),), (0,), (-1,), (1, 2)),
    ):
        with pytest.raises(ValueError):
            make_bank(weights, bias, lower, upper, enabled=True)
    with pytest.raises(ValueError):
        certify(_bank(), (1, -1))
    with pytest.raises(ValueError):
        certify(_bank(), (1, 1, -1, 0))


def test_16_exact_scalar_types_and_rows_only_helper_contract():
    bank = _bank()
    closure = compile_rows(bank, (certify(bank, (1, 1, -1)),), enabled=True)
    for invalid in (True, 0.5):
        with pytest.raises(TypeError):
            make_bank(((invalid,),), (0,), (-1,), (1,), enabled=True)
        with pytest.raises(TypeError):
            make_bank(((1,),), (invalid,), (-1,), (1,), enabled=True)
        with pytest.raises(TypeError):
            certify(bank, (1, invalid, -1))
        with pytest.raises(TypeError):
            closure.satisfies_rows((invalid, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        closure.satisfies_rows((0,), (0, 0, 0))
    with pytest.raises(ValueError):
        closure.satisfies_rows((0, 0), (0, 0))
    # This helper checks only its added rows, not boxes, old guards, or bits.
    assert closure.satisfies_rows((2, 0), (2, 0, 1))
    assert closure.satisfies_rows((0, 0), (1, 1, 1))


def test_17_bank_dimension_caps_and_certificate_support_limit():
    boundary = make_bank(tuple((1,) * 32 for _ in range(64)), (0,) * 64,
                         (-1,) * 32, (1,) * 32, enabled=True)
    assert (boundary.rows, boundary.cols) == (64, 32)
    with pytest.raises(ValueError):
        make_bank(tuple((1,) for _ in range(65)), (0,) * 65, (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        make_bank(((1,) * 33,), (0,), (-1,) * 33, (1,) * 33, enabled=True)
    certificate = certify(boundary, (1,) * 32 + (0,) * 32)
    assert len(certificate.support) == 32
    with pytest.raises(ValueError):
        certify(boundary, (1,) * 33 + (0,) * 31)
    closure = compile_rows(boundary, (certificate,), enabled=True)
    assert len(closure.rows) <= 33 and sum(len(row.terms) for row in closure.rows) <= 2112


def test_18_stored_and_derived_rational_overflows_are_rejected():
    with pytest.raises(ValueError):
        make_bank(((2 ** 512,),), (0,), (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        certify(_bank(), (F(1, 2 ** 512), 1, -1))
    large_bank = make_bank(((2 ** 511,),), (0,), (-1,), (1,), enabled=True)
    with pytest.raises(ValueError):
        certify(large_bank, (2,))
    zero_bank = make_bank(((0,),), (0,), (-1,), (1,), enabled=True)
    large_certificate = certify(zero_bank, (2 ** 511,))
    with pytest.raises(ValueError):
        compile_rows(zero_bank, (large_certificate,), enabled=True)


def test_19_full_rank_selection_and_empty_zero_ledgers_add_no_rows():
    bank = make_bank(((1, 0), (0, 1)), (0, 0), (-1, -1), (1, 1), enabled=True)
    assert select_certificate(bank, enabled=True) == ()
    empty = compile_rows(bank, (), enabled=True)
    zero = compile_rows(bank, (certify(bank, (0, 0)),), enabled=True)
    assert empty.rows == zero.rows == ()
    assert empty.satisfies_rows((0, 0), (0, 0))
    assert zero.certs[0].support == () and zero.certs[0].epsilon == 0


def test_20_certificate_budget_rejects_multiple_entries_without_changing_source():
    fixture = _fixture()
    bank, certificate, native = fixture["bank"], fixture["certificate"], fixture["native"]
    before = (bank.weights, bank.bias, bank.lower, bank.upper, native.rows, native.binary_ids)
    with pytest.raises(ValueError):
        compile_rows(bank, (certificate, certificate), enabled=True)
    assert (bank.weights, bank.bias, bank.lower, bank.upper, native.rows, native.binary_ids) == before
    closure = compile_rows(bank, (certificate,), enabled=True)
    assert closure.rows == fixture["closure"].rows
    assert closure.counts["n_certificates"] == 1 and closure.counts["assignment_size"] == 5
