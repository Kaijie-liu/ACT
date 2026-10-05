"""Twenty-four fixed rational definition tests; no optimizer or model execution."""
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d180_reference_observed_component_20261004 import reference_fiber as nf


def _identity(size):
    return tuple(tuple(F(int(i == j)) for j in range(size)) for i in range(size))


def _phase01(bits):
    return tuple((bit + 1) // 2 for bit in bits)


def _extension(parent, g, child, source, old_bits=(), old_errors=(), bits=None):
    """Actual ReLU image of one checked parent, possibly an abstract member."""
    values = parent.evaluate(g, source, old_bits, old_errors)
    bits = tuple(1 if value >= 0 else -1 for value in values) if bits is None else tuple(bits)
    bank = child.banks[-1]
    correction = tuple(F((bit + 1) // 2 - int(bar > 0)) * (value - bar)
                       for value, bar, bit in zip(values, bank.reference, bits))
    delta = tuple(sum((coefficient * value for coefficient, value in zip(row, correction)), F(0))
                  for row in bank.carrier)
    return old_bits + bits, old_errors + delta, tuple(max(F(0), value) for value in values)


def _five(cls=nf.Fiber):
    parent = cls.box(((-1, 1),) * 5, enabled=True)
    matrix = tuple(tuple(F(int(i == j)) + F(int(j == (i + 1) % 5), 4)
                         for j in range(5)) for i in range(5))
    d = parent.affine(parent.sources(), matrix, (0,) * 5)
    g = parent.affine(d, _identity(5), (F(1, 2), -F(1, 2), F(1, 2), -F(1, 2), F(1, 2)))
    consumers = ((1, -1, -1, 1, F(1, 2)), (1,) * 5)
    child, output = parent.relu_packet(g, consumers, tuple("five:" + str(i) for i in range(5)))
    return parent, d, g, child, output


def _lossy():
    root, d, g, parent, output = _five()
    source = tuple(F(value, 1025) for value in (-3, 12, -48, 192, -768))
    bits, errors = (1, -1, 1, -1, -1), (F(19, 40), F(19, 20))
    assert parent.contains(source, bits, errors)
    return root, g, parent, output, source, bits, errors


def _abstract_child():
    root, g, parent, packet, source, old_bits, old_errors = _lossy()
    next_g = parent.affine(packet, ((0, 1),), (F(1, 4),))
    child, output = parent.relu_packet(next_g, ((1,),), ("abstract-next",))
    bits, errors, _ = _extension(parent, next_g, child, source, old_bits, old_errors)
    return parent, packet, next_g, child, output, source, bits, errors


def test_default_off_and_rational_inputs():
    with pytest.raises(nf.DisabledError):
        nf.Fiber.box(((-1, 1),))
    with pytest.raises(nf.DisabledError):
        nf.Fiber.box(((-1, 1),), enabled=1)
    for value in (True, 0.5, float("nan"), float("inf"), 1 << 513):
        with pytest.raises(ValueError):
            nf.PhaseAffine(value)
    with pytest.raises(ValueError):
        nf.Fiber.box(((1, -1),), enabled=True)
    assert nf.Fiber.box(((0, 0),), enabled=True).contains((0,), (), ())


def test_hz_embedding_preserves_predicates_and_decoder():
    parent, view = nf.Fiber.embed_hz(((-1, 1),), (3,), ((2,),), ((5,),),
        phase_names=("original",), signed_predicates=(((1,), (1,), "le", 1), ((0,), (1,), "eq", -1)),
        decoder_matrix=((2,),), decoder_bias=(1,), enabled=True)
    assert isinstance(parent, nf.Fiber) and not parent.banks
    assert {row.sense for row in parent.predicates} == {"eq", "le"}
    assert parent.evaluate(view, (F(1, 4),), (-1,), ()) == (-F(3, 2),)
    assert parent.decode((F(1, 4),), (-1,), ()) == (F(3, 2),)
    assert not parent.contains((F(1, 4),), (1,), ())
    child, output = parent.relu_packet(view, ((1,),), ("embedded-next",))
    bits, errors, _ = _extension(parent, view, child, (F(1, 4),), (-1,))
    assert child.predicates[:2] == parent.predicates
    assert child.evaluate(output, (F(1, 4),), bits, errors) == (0,)
    assert child.decode((F(1, 4),), bits, errors) == (F(3, 2),)


def test_single_gate_nonconvexity_and_zero_labels():
    parent = nf.Fiber.box(((-1, 1),), enabled=True)
    g = parent.affine(parent.sources(), ((1,),), (F(1, 2),))
    child, output = parent.relu_packet(g, ((1,),), ("single",))
    assert child.references == (1,)
    for bit, delta in ((-1, F(1, 2)), (1, F(0))):
        assert child.evaluate(output, (-F(1, 2),), (bit,), (delta,)) == (0,)
    assert child.evaluate(output, (-1,), (-1,), (1,)) == (0,)
    assert child.evaluate(output, (1,), (1,), (0,)) == (F(3, 2),)
    # The midpoint (x=0,q=3/4) is excluded for BOTH original labels.
    assert not child.contains((0,), (-1,), (F(3, 4),))
    assert not child.contains((0,), (1,), (F(1, 4),))
    assert not child.contains((0,), (0,), (F(1, 2),))
    negative_g = parent.affine(parent.sources(), ((1,),), (-F(1, 2),))
    negative, negative_q = parent.relu_packet(negative_g, ((1,),), ("negative-reference",))
    assert negative.references == (0,)
    for bit, delta in ((-1, F(0)), (1, F(1, 2))):
        assert negative.evaluate(negative_q, (F(1, 2),), (bit,), (delta,)) == (0,)


def test_complete_carrier_and_mass_deduplication():
    parent = nf.Fiber.box(((-1, 1),) * 3, enabled=True)
    g = parent.affine(parent.sources(), _identity(3), (F(1, 4), -F(1, 4), F(1, 4)))
    row = (1, -1, F(1, 2))
    child, output = parent.relu_packet(g, (row,), ("a0", "a1", "a2"))
    assert child.banks[-1].carrier == (row, (1, 1, 1)) and child.banks[-1].mass_index == 1
    assert len(child.errors) == 2 and len(output.forms) == 1
    other, packet = parent.relu_packet(g, ((1, 1, 1), row), ("b0", "b1", "b2"))
    assert other.banks[-1].mass_index == 0 and len(other.errors) == len(packet.forms) == 2
    bits, errors, _ = _extension(parent, g, other, (0, 0, 0))
    assert other.evaluate(packet, (0, 0, 0), bits, errors) == (F(1, 2), F(3, 8))
    assert other.evaluate(other.amplitudes(), (0, 0, 0), bits, errors) == errors == (0, 0)


def test_original_identities_survive_extension():
    parent = nf.Fiber.box(((-1, 1),), phase_names=("root-phase",), enabled=True)
    source = parent.sources()
    first, q = parent.relu_packet(source, ((1,),), ("first",))
    second, r = first.relu_packet(q, ((1,),), ("second",))
    assert second.frame is first.frame is parent.frame
    assert second.phases[0] is parent.phases[0] and second.phases[1] is first.phases[1]
    assert second.errors[0] is first.errors[0] and second.banks[0] is first.banks[0]
    bits1, errors1, _ = _extension(parent, source, first, (F(1, 2),), (-1,))
    bits2, errors2, _ = _extension(first, q, second, (F(1, 2),), bits1, errors1)
    assert second.evaluate(r, (F(1, 2),), bits2, errors2) == (F(1, 2),)
    assert second.decode((F(1, 2),), bits2, errors2) == (F(1, 2),)


def test_shared_skip_add_and_concat():
    parent = nf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1,),), (F(1, 2),))
    child, q = parent.relu_packet(g, ((1,),), ("skip",))
    skip = child.add(q, child.affine(source, ((3,),), (2,)))
    merged = child.concat(skip, q, source)
    zero = child.affine(merged, ((1, -1, -3),), (-2,))
    assert not any((zero.forms[0].constant,) + zero.forms[0].source + zero.forms[0].phase + zero.forms[0].error)
    assert child.bounds(zero) == (0, 0)
    bits, errors, _ = _extension(parent, g, child, (F(1, 4),))
    assert child.evaluate(merged, (F(1, 4),), bits, errors) == (F(7, 2), F(3, 4), F(1, 4))
    assert merged.forms[0].error == merged.forms[1].error and len(child.errors) == 1


def test_independent_sibling_readouts_fail_closed():
    parent = nf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    left, first = parent.relu_packet(source, ((1,),), ("left",))
    right, second = parent.relu_packet(source, ((1,),), ("right",))
    with pytest.raises(ValueError):
        left.add(first, second)
    with pytest.raises(ValueError):
        right.joint_certificate(first, (1,))
    with pytest.raises(ValueError):
        parent.concat(first)
    foreign = nf.Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        parent.add(source, foreign.sources())


def test_actual_extensions_cover_mixed_two_gate_phases():
    parent = nf.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    g = parent.affine(parent.sources(), ((1, F(1, 4)), (-F(1, 3), 1)), (F(1, 8), -F(1, 5)))
    child, output = parent.relu_packet(g, ((1, -F(1, 2)),), ("mixed0", "mixed1"))
    observed = set()
    for source in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        bits, errors, q = _extension(parent, g, child, source)
        observed.add(bits)
        assert child.contains(source, bits, errors) and child.finite_rows_hold(source, bits, errors)
        assert child.evaluate(output, source, bits, errors) == (q[0] - q[1] / 2,)
        assert child.decode(source, bits, errors) == source
    assert observed == {(-1, -1), (-1, 1), (1, -1), (1, 1)}


def test_reference_chart_rejects_d157_false_packet():
    _, _, _, old, old_packet = _five(nf.cf.Fiber)
    _, _, _, child, packet = _five()
    source, bits, false_a = (0,) * 5, (1, -1, 1, -1, 1), (0, F(8, 5))
    assert old.evaluate(old_packet, source, bits, false_a) == (F(1, 4), F(31, 10))
    assert not child.contains(source, bits, false_a)
    assert not child.finite_rows_hold(source, bits, false_a)
    assert child.evaluate(packet, source, bits, (0, 0)) == (F(1, 4), F(3, 2))


def test_complementary_reference_chart_is_exact():
    parent, _, g, child, packet = _five()
    source = (-1, 1, -1, 1, -1)
    bits, errors, _ = _extension(parent, g, child, source)
    assert bits == (-1, 1, -1, 1, -1)
    assert child.evaluate(packet, source, bits, errors) == (0, F(1, 2))
    assert not child.contains(source, bits, (errors[0] + F(1, 1000), errors[1]))


def test_nonreference_native_loss_remains_visible():
    root, g, parent, packet, source, bits, errors = _lossy()
    assert parent.evaluate(packet, source, bits, errors) == (F(1, 10), F(6, 5))
    true_bits, true_errors, q = _extension(root, g, parent, source)
    assert true_bits == bits and true_errors == (F(3, 8), F(3, 4))
    assert parent.evaluate(packet, source, true_bits, true_errors) == (0, 1)
    assert q == (F(1, 2), 0, F(1, 2), 0, 0)
    assert parent.decode(source, bits, errors) == source


def test_finite_rows_do_not_replace_common_witness():
    _, _, parent, _, source, bits, errors = _lossy()
    false_delta = (errors[0], errors[1] + F(1, 1000))
    # Native common-witness range requires delta_1=2*delta_0 here.
    assert parent.finite_rows_hold(source, bits, false_delta)
    assert not parent.contains(source, bits, false_delta)
    with pytest.raises(ValueError):
        parent.decode(source, bits, false_delta)


def test_all_active_and_inactive_packets_are_exact():
    parent, _, g, child, packet = _five()
    for source in ((1,) * 5, (-1,) * 5):
        bits, errors, q = _extension(parent, g, child, source)
        assert len(set(bits)) == 1 and child.finite_rows_hold(source, bits, errors)
        actual = (q[0] - q[1] - q[2] + q[3] + q[4] / 2, sum(q, F(0)))
        assert child.evaluate(packet, source, bits, errors) == actual
        false_delta = (errors[0] + F(1, 1000), errors[1])
        assert not child.contains(source, bits, false_delta)
        assert not child.finite_rows_hold(source, bits, false_delta)


def test_canonical_readout_retains_source_coefficients():
    parent, d, _, child, packet = _five()
    bank = child.banks[0]
    for form, row in zip(packet.forms, bank.carrier):
        expected = tuple(sum((row[i] * int(bank.reference[i] > 0) * d.forms[i].source[j]
                              for i in range(5)), F(0)) for j in range(5))
        assert form.source == expected and any(expected)
    assert all(not any(form.source) for form in child.amplitudes().forms)
    shifted = nf.Fiber.box(((2, 4),), enabled=True)
    g = shifted.affine(shifted.sources(), ((1,),), (-2,))
    exact, q = shifted.relu_packet(g, ((1,),), ("nonzero-midpoint",))
    assert exact.banks[0].reference == (1,)
    assert q.forms[0] == nf.Form(-3, (1,), (1,), (1,))
    assert exact.evaluate(q, (3,), (1,), (0,)) == (1,)


def test_shifted_energy_rows_accept_true_extensions():
    parent, _, g, child, _ = _five()
    bank = child.banks[0]
    begin, end = {label: (lo, hi) for label, lo, hi in bank.row_ranges}["observed_energy"]
    maximum = bank.energy.extrema(child.work)[1]
    expected = []
    for j, row in enumerate(bank.carrier):
        squared = sum((value * value for value in row), F(0))
        scale = nf.sqrt_upper(maximum / squared, child.work)
        for left, right in ((scale, 0), (-scale, 0), (0, scale), (0, -scale), (scale, -scale), (-scale, scale)):
            # nu=-(lambda-mu) cancels every Hd term in the LHS.
            lhs = child.amplitudes().forms[j].scale(2 * (left - right), child.work)
            lhs = lhs.plus(bank.source_projection[j].scale(2 * right, child.work), child.work)
            constant, phases = bank.energy.constant, list(bank.energy.phase)
            for coefficient, bar, index in zip(row, bank.reference, bank.phase_indices):
                h = (right - left) * coefficient * int(bar > 0)
                active, inactive = (left * coefficient + h) ** 2, (right * coefficient + h) ** 2
                constant += inactive
                phases[index] += active - inactive
            form = nf.Form(0, lhs.source, tuple(a - b for a, b in zip(lhs.phase, phases)), lhs.error)
            expected.append(nf.Predicate(form, "le", constant - lhs.constant))
    assert child.predicates[begin:end] == tuple(expected)
    for source in ((0,) * 5, (-1, 1, -1, 1, -1)):
        bits, errors, _ = _extension(parent, g, child, source)
        assert child.finite_rows_hold(source, bits, errors)


def test_range_rows_keep_negative_phase_coefficients():
    _, _, _, child, _ = _five()
    bank = child.banks[0]
    begin, end = {label: (lo, hi) for label, lo, hi in bank.row_ranges}["delta_range"]
    radius = nf.sqrt_upper(bank.energy.extrema(child.work)[1], child.work)
    radius = max((radius,) + tuple(abs(endpoint - bar) for bar, pair in
                                 zip(bank.reference, bank.preactivation_bounds) for endpoint in pair))
    for j, row in enumerate(bank.carrier):
        expected_phase = tuple(radius * abs(value) * (2 * int(bar > 0) - 1)
                               for value, bar in zip(row, bank.reference))
        expected_rhs = sum((radius * abs(value) * int(bar > 0)
                            for value, bar in zip(row, bank.reference)), F(0))
        for offset, sign in ((0, 1), (1, -1)):
            predicate = child.predicates[begin + 2 * j + offset]
            assert predicate.form.source == (0,) * 5 and predicate.form.phase == expected_phase
            assert predicate.form.error == tuple(sign * int(k == j) for k in range(2))
            assert predicate.rhs == expected_rhs
        assert any(value < 0 for value in expected_phase) and any(value > 0 for value in expected_phase)
    assert end - begin == 4


def test_flip_energy_is_zero_at_reference_phases():
    parent = nf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1,),), (F(1, 2),))
    first, q = parent.relu_packet(g, ((1,),), ("flip",))
    flip = first._flip_energy(((1,),), first.banks[0])
    assert flip == nf.PhaseAffine(1, (-1,)) and flip.at((1,), first.work) == 0
    cancellation = first.affine(first.concat(q, source), ((1, -1),), (-F(1, 2),))
    second, _ = first.relu_packet(cancellation, ((1,),), ("cancel-next",))
    assert second.banks[-1].energy == nf.PhaseAffine(F(5, 2), (-F(5, 2), 0))
    assert second.banks[-1].energy.at((1, 0), second.work) == 0
    ordinary, _ = first.relu_packet(q, ((1,),), ("ordinary-next",))
    assert ordinary.banks[-1].energy == nf.PhaseAffine(F(27, 4), (-F(15, 4), 0))
    assert ordinary.banks[-1].energy.at((1, 1), ordinary.work) == 3


def test_two_layer_reference_slice_and_shared_source():
    parent, d, g, first, packet = _five()
    objective = first.affine(first.concat(packet, d), ((2, 1, -3, 0, 1, 0, -2),), (-2,))
    assert objective.forms[0].source == (0,) * 5
    next_g = first.affine(objective, ((1,),), (-F(1, 4),))
    child, result = first.relu_packet(next_g, ((1,),), ("reference-next",))
    assert child.banks[-1].energy.at((1, 0, 1, 0, 1, 0), child.work) == 0
    for source in ((0,) * 5, (F(1, 8),) * 5):
        old_bits, old_errors, _ = _extension(parent, g, first, source)
        assert old_bits == (1, -1, 1, -1, 1) and old_errors == (0, 0)
        assert first.evaluate(objective, source, old_bits, old_errors) == (0,)
        bits, errors, _ = _extension(first, next_g, child, source, old_bits, old_errors)
        assert child.evaluate(result, source, bits, errors) == (0,)
        assert child.decode(source, bits, errors) == source
    assert child.errors[:2] == first.errors and child.banks[0] is first.banks[0]


def test_abstract_parent_extension_is_sound():
    parent, packet, next_g, child, output, source, bits, errors = _abstract_child()
    assert parent.evaluate(packet, source, bits[:-1], errors[:-1]) == (F(1, 10), F(6, 5))
    assert child.evaluate(output, source, bits, errors) == (F(29, 20),)
    assert child.finite_rows_hold(source, bits, errors) and child.decode(source, bits, errors) == source
    bank = child.banks[-1]
    assert bank.reference[0] > 0 and child.references[-1] == 1
    assert any(bank.observed_projection[0].error[:2])
    assert errors[-1] == 0


def test_every_inherited_bank_is_checked():
    parent, _, _, child, _, source, bits, errors = _abstract_child()
    invalid_parent = (errors[0], errors[1] + F(1, 1000), errors[2])
    assert child.finite_rows_hold(source, bits, invalid_parent)
    assert not child.contains(source, bits, invalid_parent)
    assert not parent.contains(source, bits[:-1], invalid_parent[:-1])
    invalid_child = errors[:-1] + (F(1, 1000),)
    assert not child.contains(source, bits, invalid_child)
    assert not child.finite_rows_hold(source, bits, invalid_child)


def test_inherited_joint_positive_control_is_preserved():
    parent = nf.Fiber.box(((-1, 1),) * 16, enabled=True)
    source = parent.sources()
    matrix = tuple(tuple(F(int(i == j)) - F(1, 8) for j in range(16)) for i in range(16))
    g = parent.affine(source, matrix, (-F(1, 2),) * 16)
    consumer = (F(1),) * 15 + (-F(1, 4),)
    child, output = parent.relu_packet(g, (consumer,), tuple("hh:" + str(i) for i in range(16)))
    objective = child.affine(child.concat(output, source), ((1,) + (F(9, 22),) * 16,), (0,))
    certificate = child.joint_certificate(objective, (1,))
    assert certificate == nf.PhaseAffine(F(72, 11), (0,) * 16)
    assert len(child.support_certificates(objective, (1,))) == 3
    next_g = child.affine(objective, ((1,),), (-F(33, 5),))
    assert child.bounds(next_g)[1] == -F(3, 55)
    next_state, result = child.relu_packet(next_g, ((1,),), ("hh-zero",))
    assert next_state.banks[-1].preactivation_bounds[0][1] == -F(3, 55)
    assert next_state.bounds(result) == (0, 0)
    bits1, errors1, _ = _extension(parent, g, child, (0,) * 16)
    bits2, errors2, _ = _extension(child, next_g, next_state, (0,) * 16, bits1, errors1)
    assert next_state.evaluate(result, (0,) * 16, bits2, errors2) == (0,)


def test_all_fixed_support_certificates_are_sound():
    parent, packet, _, child, output, source, bits, errors = _abstract_child()
    target = child.affine(child.concat(output, packet, parent.sources()),
                          ((1, -F(1, 3), F(2, 5), F(1, 10), -F(1, 5), F(3, 10), 0, -F(1, 10)),), (F(1, 7),))
    value = child.evaluate(target, source, bits, errors)[0]
    for direction in (1, -1):
        certificates = child.support_certificates(target, (direction,))
        assert len(certificates) == 3
        for certificate in certificates:
            assert direction * value <= certificate.at(_phase01(bits), child.work)
    assert any(child.banks[-1].observed_projection[0].error[:2])
    assert parent.contains(source, bits[:-1], errors[:-1])


def test_complete_row_and_logical_cost_accounting():
    _, _, _, child, output = _five()
    bank = child.banks[0]
    assert len(child.predicates) == 2 * 5 + 18 * 2 + 2 * 2 == 50
    assert tuple((label, end - begin) for label, begin, end in bank.row_ranges) == (
        ("guards", 10), ("caps", 8), ("mass", 4), ("energy", 12), ("observed_energy", 12), ("delta_range", 4))
    assert bank.row_ranges[0][1] == 0 and bank.row_ranges[-1][2] == 50
    before = child.cost(output)
    identities = (child.phases, child.errors, child.banks, child.predicates)
    child.support_certificates(output, (1, -1))
    after = child.cost(output)
    assert before["observed_projection_coefficients"] == 2 * (1 + 5 + 5 + 2)
    assert before["support_certificate_count"] == 3 and before["d158_joint_query_implemented"] is True
    assert before["phase_factors"] == 5 and before["residual_factors"] == 2
    assert before["complete_physical_qualification"] is False and before["terminal_is_outer_approximation"] is True
    assert all(a is b for a, b in zip(identities, (child.phases, child.errors, child.banks, child.predicates)))
    assert {k: v for k, v in before.items() if k != "algebra_work_used"} == {
        k: v for k, v in after.items() if k != "algebra_work_used"}
    assert after["algebra_work_used"] > before["algebra_work_used"]


def test_unsupported_values_and_resource_limits_fail_closed():
    parent = nf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    for action in (lambda: parent.relu(source, ("bad",)), lambda: parent.error_bank("bad", 1, 1),
                   lambda: parent.interval_affine(source, ((1,),), ((1,),), (0,), (0,), name="bad")):
        with pytest.raises(nf.DisabledError):
            action()
    with pytest.raises(TypeError):
        parent.relu_packet(source, ((1,),), ("bad",), energy=1)
    with pytest.raises(ValueError):
        parent.relu_packet(source, ((1,),) * 26, ("too-many-rows",))
    with pytest.raises(ValueError):
        nf.Fiber.box(((-1, 1),) * 33, enabled=True)
    child, output = parent.relu_packet(source, ((1,),), ("bounded",))
    assert not child.contains((True,), (-1,), (0,))
    assert not child.contains((0,), (0,), (0,))
    with pytest.raises(ValueError):
        child.support_certificates(output, (0.5,))
    assert nf.MAX_WORK == nf.cf.MAX_WORK and nf.MAX_BITS == nf.cf.MAX_BITS
    child.work.charge(nf.MAX_WORK - child.work.used)
    with pytest.raises(ValueError):
        child.support_certificates(output, (1,))
    assert not child.contains((0,), (-1,), (0,))
