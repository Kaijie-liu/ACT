"""Twenty frozen-population mathematical tests; no model or numerical solver.

All witnesses below are explicit rational assignments.  A native abstract
member is not called a concrete network witness unless it is obtained by the
actual ReLU operation on a checked parent member.
"""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d157_consumer_fiber_component_20261004 import consumer_fiber as cf


def _identity(size):
    return tuple(tuple(F(int(i == j)) for j in range(size)) for i in range(size))


def _extension(parent, preactivation, child, source, old_bits=(), old_values=(), bits=None):
    """One actual forward image of a checked parent, with the same source."""
    values = parent.evaluate(preactivation, source, old_bits, old_values)
    new_bits = tuple(1 if value >= 0 else -1 for value in values) if bits is None else tuple(bits)
    bank = child.banks[-1]
    active_deviation = tuple((value - center) if bit == 1 else F(0)
                             for value, center, bit in zip(values, bank.reference, new_bits))
    amplitudes = tuple(sum((coefficient * value for coefficient, value in zip(row, active_deviation)), F(0))
                       for row in bank.carrier)
    return old_bits + new_bits, old_values + amplitudes, tuple(max(F(0), value) for value in values)


def _three_gate_parent():
    parent = cf.Fiber.box(((-1, 1),) * 3, enabled=True)
    g = parent.affine(parent.sources(), _identity(3), (F(1, 4), -F(1, 4), F(1, 4)))
    child, output = parent.relu_packet(g, ((1, -1, F(1, 2)),), ("three:0", "three:1", "three:2"))
    return parent, g, child, output


def _sum_forms(forms, coefficients, work):
    result = forms[0].scale(coefficients[0], work)
    for form, coefficient in zip(forms[1:], coefficients[1:]):
        result = result.plus(form.scale(coefficient, work), work)
    return result


def test_opt_in_and_rational_domain_rejection():
    with pytest.raises(cf.DisabledError):
        cf.Fiber.box(((-1, 1),))
    with pytest.raises(cf.DisabledError):
        cf.Fiber.box(((-1, 1),), enabled=1)
    for invalid in (True, 0.5, float("nan"), float("inf"), 1 << 513):
        with pytest.raises(ValueError):
            cf.PhaseAffine(invalid)
    with pytest.raises(ValueError):
        cf.Fiber.box(((1, -1),), enabled=True)


def test_hz_embedding_and_source_decoder():
    parent, value = cf.Fiber.embed_hz(((-1, 1),), (3,), ((2,),), ((5,),),
        phase_names=("original-sigma",), signed_predicates=(((1,), (1,), "le", 1),),
        decoder_matrix=((2,),), decoder_bias=(1,), enabled=True)
    assert value.forms[0].constant == -2 and value.forms[0].phase == (10,)
    assert parent.evaluate(value, (F(1, 4),), (-1,), ()) == (-F(3, 2),)
    assert parent.decode((F(1, 4),), (-1,), ()) == (F(3, 2),)
    assert not parent.contains((F(1, 4),), (1,), ())
    child, output = parent.relu_packet(value, ((1,),), ("embedded-child",))
    bits, amplitudes, _ = _extension(parent, value, child, (F(1, 4),), (-1,))
    assert child.phases[0] is parent.phases[0]
    assert child.evaluate(output, (F(1, 4),), bits, amplitudes) == (0,)
    assert child.decode((F(1, 4),), bits, amplitudes) == (F(3, 2),)


def test_complete_packet_projection_and_mass_deduplication():
    parent, g, child, output = _three_gate_parent()
    bank = child.banks[-1]
    assert bank.consumers == ((1, -1, F(1, 2)),)
    assert bank.carrier == ((1, -1, F(1, 2)), (1, 1, 1))
    assert bank.mass_index == 1 and len(child.errors) == 2
    assert bank.coordinates == (0, 1)
    bits, amplitudes, q = _extension(parent, g, child, (0, 0, 0))
    assert child.evaluate(output, (0, 0, 0), bits, amplitudes) == (F(3, 8),)
    assert child.evaluate(child.bank_readout(), (0, 0, 0), bits, amplitudes) == (F(3, 8), F(1, 2))
    assert child.evaluate(child.amplitudes(), (0, 0, 0), bits, amplitudes) == amplitudes
    assert q == (F(1, 4), 0, F(1, 4))
    existing, values = parent.relu_packet(g, ((1, 1, 1), (1, -1, F(1, 2))),
        ("dedup:0", "dedup:1", "dedup:2"))
    assert existing.banks[-1].mass_index == 0
    assert len(existing.banks[-1].carrier) == len(values.forms) == len(existing.errors) == 2


def test_original_phase_identities_and_zero_labels():
    parent = cf.Fiber.box(((-1, 1),), phase_names=("original",), enabled=True)
    g = parent.affine(parent.sources(), ((1,),), (F(1, 2),))
    child, output = parent.relu_packet(g, ((1,),), ("zero-child",))
    assert child.phases[0] is parent.phases[0]
    assert child.references == (0, 1)
    for original_bit in (-1, 1):
        # g=0 and nonzero gbar: each legal original label needs its own a.
        assert child.evaluate(output, (-F(1, 2),), (original_bit, -1), (0,)) == (0,)
        assert child.evaluate(output, (-F(1, 2),), (original_bit, 1), (-F(1, 2),)) == (0,)
    assert not child.contains((-F(1, 2),), (-1, 0), (-F(1, 4),))
    # The visible graph is nonconvex, not merely an integer-labelled box.
    assert child.evaluate(output, (-1,), (-1, -1), (0,)) == (0,)
    assert child.evaluate(output, (1,), (-1, 1), (1,)) == (F(3, 2),)
    assert not child.contains((0,), (-1, -1), (F(3, 4),))
    assert not child.contains((0,), (-1, 1), (F(1, 4),))


def test_actual_witnesses_for_all_two_gate_phase_combinations():
    parent = cf.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    g = parent.affine(parent.sources(), ((1, F(1, 4)), (-F(1, 3), 1)), (F(1, 8), -F(1, 5)))
    child, output = parent.relu_packet(g, ((1, -F(1, 2)),), ("mixed:0", "mixed:1"))
    observed = set()
    for source in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        bits, amplitudes, q = _extension(parent, g, child, source)
        observed.add(bits)
        assert child.contains(source, bits, amplitudes)
        assert child.finite_rows_hold(source, bits, amplitudes)
        assert child.evaluate(output, source, bits, amplitudes) == (q[0] - q[1] / 2,)
        assert child.decode(source, bits, amplitudes) == source
    assert observed == {(-1, -1), (-1, 1), (1, -1), (1, 1)}


def test_parent_extension_and_shared_skip():
    parent = cf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1,),), (F(1, 4),))
    child, output = parent.relu_packet(g, ((1,),), ("skip-child",))
    joined = child.concat(output, source, output)
    twice = child.add(output, output)
    skip = child.add(output, source)
    bits, amplitudes, _ = _extension(parent, g, child, (F(1, 4),))
    assert child.evaluate(joined, (F(1, 4),), bits, amplitudes) == (F(1, 2), F(1, 4), F(1, 2))
    assert child.evaluate(twice, (F(1, 4),), bits, amplitudes) == (1,)
    assert child.evaluate(skip, (F(1, 4),), bits, amplitudes) == (F(3, 4),)
    assert joined.forms[0].error == joined.forms[2].error
    assert len(child.errors) == 1 and child.frame is parent.frame


def test_independent_siblings_are_rejected():
    parent = cf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    left, first = parent.relu_packet(source, ((1,),), ("left",))
    right, second = parent.relu_packet(source, ((1,),), ("right",))
    with pytest.raises(ValueError):
        left.add(first, second)
    with pytest.raises(ValueError):
        right.concat(source, first)
    with pytest.raises(ValueError):
        parent.concat(first)
    foreign = cf.Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        parent.add(source, foreign.sources())


def test_all_active_native_and_finite_endpoint_exactness():
    parent = cf.Fiber.box(((-1, 1),) * 3, enabled=True)
    g = parent.affine(parent.sources(), _identity(3), (F(1, 4),) * 3)
    child, output = parent.relu_packet(g, ((1, -1, F(1, 2)),), ("on:0", "on:1", "on:2"))
    source = (F(1, 4), F(1, 8), F(1, 2))
    bits, amplitudes, q = _extension(parent, g, child, source)
    assert bits == (1, 1, 1)
    assert child.evaluate(output, source, bits, amplitudes) == (F(1, 2),)
    assert child.finite_rows_hold(source, bits, amplitudes)
    false_amplitudes = (amplitudes[0] + F(1, 64), amplitudes[1])
    assert not child.contains(source, bits, false_amplitudes)
    assert not child.finite_rows_hold(source, bits, false_amplitudes)
    assert q == (F(1, 2), F(3, 8), F(3, 4))


def test_all_inactive_native_and_finite_endpoint_exactness():
    parent = cf.Fiber.box(((-1, 1),) * 3, enabled=True)
    g = parent.affine(parent.sources(), _identity(3), (-F(1, 4),) * 3)
    child, output = parent.relu_packet(g, ((1, -1, F(1, 2)),), ("off:0", "off:1", "off:2"))
    source, bits = (0, 0, 0), (-1, -1, -1)
    assert child.evaluate(output, source, bits, (0, 0)) == (0,)
    assert child.finite_rows_hold(source, bits, (0, 0))
    assert not child.contains(source, bits, (F(1, 64), 0))
    assert not child.finite_rows_hold(source, bits, (F(1, 64), 0))


def test_second_block_uses_shared_parent_amplitudes():
    parent = cf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    first_g = parent.affine(source, ((1,),), (F(1, 4),))
    first, q = parent.relu_packet(first_g, ((1,),), ("first",))
    second_g = first.affine(first.concat(q, source), ((1, -F(1, 2)),), (-F(1, 8),))
    second, r = first.relu_packet(second_g, ((1,),), ("second",))
    point = (F(1, 2),)
    first_bits, first_amplitudes, _ = _extension(parent, first_g, first, point)
    bits, amplitudes, _ = _extension(first, second_g, second, point, first_bits, first_amplitudes)
    assert second.phases[0] is first.phases[0] and second.errors[0] is first.errors[0]
    assert second.evaluate(second.concat(q, r, source), point, bits, amplitudes) == (F(3, 4), F(3, 8), F(1, 2))
    assert second.decode(point, bits, amplitudes) == point
    assert len(second.banks) == 2 and len(second.errors) == 2


def test_d156_mixed_phase_source_binding_loss():
    # Appending the uniform mass row repairs the OLD two-gate counterexample.
    parent = cf.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    g = parent.affine(parent.sources(), ((1, F(1, 4)), (-F(1, 3), 1)), (F(1, 4), -F(1, 4)))
    child, output = parent.relu_packet(g, ((1, -1),), ("old:0", "old:1"))
    point, bits = (0, -F(1, 4)), (1, -1)
    assert child.evaluate(output, point, bits, (-F(1, 16), -F(1, 16))) == (F(3, 16),)
    assert not child.contains(point, bits, (F(1, 16), F(1, 16)))
    # Three gates still have a genuine consumer kernel; this is an abstract,
    # not a concrete-network, member at the same original source and bits.
    _, _, lossy, visible = _three_gate_parent()
    # The implementation squares an outward rational sqrt(3), not a float.
    assert lossy.banks[-1].energy.phase == (0, 0, 0)
    assert 3 <= lossy.banks[-1].energy.constant < F(49, 16)
    assert lossy.evaluate(visible, (0, 0, 0), (1, -1, 1), (0, 0)) == (F(3, 8),)
    fake = (F(1, 32), -F(1, 32))
    assert lossy.evaluate(lossy.bank_readout(), (0, 0, 0), (1, -1, 1), fake) == (F(13, 32), F(15, 32))
    assert sum((value * value for value in (F(3, 32), F(1, 32), -F(4, 32))), F(0)) == F(13, 512)


def test_finite_outer_is_not_native_membership():
    _, _, child, _ = _three_gate_parent()
    source, bits, amplitudes = (0, 0, 0), (1, -1, 1), (F(1, 32), F(1, 32))
    # Inactive C column is (-1,1).  Its range forbids these equal-sign a's,
    # although each fixed coordinate direction has ample finite slack.
    assert child.finite_rows_hold(source, bits, amplitudes)
    assert not child.contains(source, bits, amplitudes)
    with pytest.raises(ValueError):
        child.decode(source, bits, amplitudes)
    assert not child.contains(source, (0, -1, 1), amplitudes)


def test_householder_generated_rows_prove_mixed_successor_zero():
    parent = cf.Fiber.box(((-1, 1),) * 16, enabled=True)
    matrix = tuple(tuple(F(int(i == j)) - F(1, 8) for j in range(16)) for i in range(16))
    g = parent.affine(parent.sources(), matrix, (-F(1, 2),) * 16)
    consumer = (F(1),) * 15 + (-F(1, 4),)
    child, output = parent.relu_packet(g, (consumer,), tuple("hh:" + str(i) for i in range(16)))
    bank = child.banks[-1]
    assert bank.carrier == (consumer, (F(1),) * 16)
    assert len(child.errors) == 2 and len(child.phases) == 16
    assert bank.energy == cf.PhaseAffine(16, (0,) * 16)
    assert bank.preactivation_bounds == ((-F(13, 4), F(9, 4)),) * 16
    # Indices come from the uniform generated-row metadata, not a row search.
    ranges = {label: (start, end) for label, start, end in bank.row_ranges}
    caps_start, _ = ranges["caps"]
    mass_start, _ = ranges["mass"]
    energy_start, _ = ranges["energy"]
    energy_row = child.predicates[energy_start + 6 * bank.mass_index + 4]
    cap_row = child.predicates[caps_start + 4 * bank.mass_index + 1]
    dominance_row = child.predicates[mass_start + 1]
    certificate = _sum_forms((energy_row.form, cap_row.form, dominance_row.form),
                             (F(9, 44), F(2, 11), F(1)), child.work)
    rhs = F(9, 44) * energy_row.rhs + F(2, 11) * cap_row.rhs + dominance_row.rhs
    objective = child.affine(child.concat(output, child.sources()),
                            ((1,) + (F(9, 22),) * 16,), (0,))
    assert certificate == objective.forms[0]
    assert rhs == F(72, 11) < F(33, 5)
    # Evaluate the actual generated outer rows at the OLD fractional LP point.
    fractional_a = (F(413, 40), F(56, 5))
    assert not child.finite_rows_hold((0,) * 16, (0,) * 16, fractional_a, relax_phases=True)
    old_J = F(531, 80)
    assert old_J - F(33, 5) == F(3, 80)
    # A real network member and its exact next activation remain available.
    bits, amplitudes, _ = _extension(parent, g, child, (0,) * 16)
    next_g = child.affine(objective, ((1,),), (-F(33, 5),))
    next_state, z = child.relu_packet(next_g, ((1,),), ("hh:successor",))
    new_bits, new_amplitudes, _ = _extension(child, next_g, next_state, (0,) * 16, bits, amplitudes)
    assert next_state.evaluate(z, (0,) * 16, new_bits, new_amplitudes) == (0,)


def test_phase_affine_budget_keeps_negative_coefficients():
    parent = cf.Fiber.box(((-1, 1),), enabled=True)
    g = parent.affine(parent.sources(), ((1,),), (F(1, 4),))
    first, q = parent.relu_packet(g, ((1,),), ("reference-one",))
    second, _ = first.relu_packet(q, ((1,),), ("next-reference",))
    assert first.references == (1,) and second.references == (1, 1)
    assert first.banks[-1].reference == (F(1, 4),)
    assert first.banks[-1].energy == cf.PhaseAffine(1, (0,))
    energy = second.banks[-1].energy
    assert energy == cf.PhaseAffine(F(17, 8), (-F(1, 8), 0))
    assert energy.extrema(second.work) == (2, F(17, 8))


def test_recursive_budget_covers_abstract_parent_members():
    _, _, parent, output = _three_gate_parent()
    source, old_bits, old_amplitudes = (0, 0, 0), (1, -1, 1), (F(1, 32), -F(1, 32))
    assert parent.contains(source, old_bits, old_amplitudes)
    g = parent.affine(output, ((1,),), (-F(25, 64),))
    assert parent.evaluate(g, source, old_bits, old_amplitudes) == (F(1, 64),)
    assert parent.evaluate(g, source, old_bits, (0, 0)) == (-F(1, 64),)
    child, result = parent.relu_packet(g, ((1,),), ("abstract-successor",))
    bits, amplitudes, _ = _extension(parent, g, child, source, old_bits, old_amplitudes)
    assert bits == (1, -1, 1, 1)
    assert amplitudes == old_amplitudes + (F(1, 32),)
    assert child.evaluate(result, source, bits, amplitudes) == (F(1, 64),)
    assert child.finite_rows_hold(source, bits, amplitudes)
    assert child.decode(source, bits, amplitudes) == source


def test_affine_cancellation_and_predicate_preservation():
    parent = cf.Fiber.box(((-1, 1), (-1, 1)), enabled=True,
        predicates=(cf.SourceConstraint((1, -1), (), "eq", 0),
                    cf.SourceConstraint((1, 0), (), "le", F(1, 2))))
    source = parent.sources()
    first_map = parent.affine(source, ((1, 2), (-3, 1)), (4, -2))
    composed = parent.affine(first_map, ((2, -1),), (3,))
    direct = parent.affine(source, ((5, 3),), (13,))
    assert composed.forms == direct.forms
    child, output = parent.relu_packet(source, ((1, -F(1, 2)),), ("pred:0", "pred:1"))
    bits, amplitudes, _ = _extension(parent, source, child, (F(1, 4), F(1, 4)))
    assert child.contains((F(1, 4), F(1, 4)), bits, amplitudes)
    assert not child.contains((F(1, 4), 0), bits, amplitudes)
    assert not child.contains((1, 1), bits, amplitudes)
    negative = child.affine(output, ((-1,),), (0,))
    zero = child.add(output, negative)
    assert child.bounds(zero) == (0, 0)
    assert not any(zero.forms[0].source + zero.forms[0].phase + zero.forms[0].error)


def test_invalid_shapes_identities_and_bits_fail_closed():
    parent = cf.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    source = parent.sources()
    for bad_matrix in ((), ((1,),), ((1, 0), (1,))):
        with pytest.raises(ValueError):
            parent.relu_packet(source, bad_matrix, ("shape:0", "shape:1"))
    with pytest.raises(ValueError):
        parent.relu_packet(source, ((1, 0),), ("duplicate", "duplicate"))
    with pytest.raises(ValueError):
        parent.relu_packet(source, ((1, 0),), ("source:0", "new"))
    child, _ = parent.relu_packet(source, ((1, -1),), ("valid:0", "valid:1"))
    assert not child.contains((0, 0), (0, 1), (0, 0))
    assert not child.contains((0,), (1, 1), (0, 0))
    assert not child.contains((0, 0), (1, 1), (0,))
    assert not child.contains((0, 0), (True, 1), (0, 0))
    with pytest.raises(ValueError):
        child.decode((0, 0), (0, 1), (0, 0))


def test_work_and_population_caps_fail_closed():
    with pytest.raises(ValueError):
        cf.Fiber.box(((-1, 1),) * (cf.MAX_SOURCES + 1), enabled=True)
    with pytest.raises(ValueError):
        cf.Fiber.box(((-1, 1),), phase_names=tuple("p:" + str(i) for i in range(cf.MAX_PHASES + 1)), enabled=True)
    parent = cf.Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    with pytest.raises(ValueError):
        parent.work.charge(cf.MAX_WORK + 1)
    parent.work.charge(cf.MAX_WORK - parent.work.used)
    with pytest.raises(ValueError):
        parent.affine(source, ((1,),), (0,))


def test_full_cost_counts_matrices_source_phases_and_rows():
    parent = cf.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    source = parent.sources()
    child, output = parent.relu_packet(source, ((1, -F(1, 2)),), ("cost:0", "cost:1"))
    cost = child.cost(output, source)
    assert cost["source_factors"] == 2 and cost["phase_factors"] == 2
    assert cost["residual_factors"] == 2 and cost["banks"] == 1
    assert cost["predicate_rows"] == len(child.predicates) == 26
    assert cost["decoder_coefficients"] == 6
    assert cost["source_bound_entries"] == 4
    assert cost["source_reference_entries"] == 2 and cost["phase_reference_entries"] == 2
    assert cost["live_readouts"] == 3
    assert cost["live_readout_coefficients"] == 21
    assert cost["predicate_coefficients"] == 26 * 8
    assert cost["carrier_coefficients"] == 4
    assert cost["consumer_coefficients"] == 2
    assert cost["source_projection_coefficients"] == 14
    assert cost["energy_coefficients"] == 3
    assert cost["bank_coordinate_entries"] == 2
    assert cost["bank_phase_entries"] == 2
    assert cost["bank_reference_entries"] == 2
    assert cost["bank_bound_entries"] == 4
    assert cost["bank_row_range_entries"] == 12
    assert cost["complete_physical_qualification"] is False
    assert cost["terminal_is_outer_approximation"] is True


def test_non_unit_source_box_and_fail_closed_certificates():
    parent = cf.Fiber.box(((F(99, 100), F(101, 100)),), enabled=True)
    source = parent.sources()
    child, output = parent.relu_packet(source, ((1,),), ("narrow",))
    assert parent.frame.reference == (1,)
    assert child.banks[-1].reference == (1,)
    assert child.banks[-1].energy == cf.PhaseAffine(F(1, 10000), (0,))
    assert child.evaluate(output, (F(201, 200),), (1,), (F(1, 200),)) == (F(201, 200),)
    assert child.finite_rows_hold((F(201, 200),), (1,), (F(1, 200),))
    assert not child.contains((F(201, 200),), (1,), (0,))
    # No caller-supplied optimistic energy/bounds certificate is accepted.
    with pytest.raises(TypeError):
        parent.relu_packet(source, ((1,),), ("forged-energy",), energy=cf.PhaseAffine(0))
    with pytest.raises(TypeError):
        parent.relu_packet(source, ((1,),), ("forged-bound",), bounds=((1, 1),))
