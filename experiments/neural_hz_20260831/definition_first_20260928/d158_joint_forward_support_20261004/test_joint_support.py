"""Twelve fixed forward-query tests; no solver or certificate-weight search."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d157_consumer_fiber_component_20261004 import consumer_fiber as cf
from experiments.neural_hz_20260831.definition_first_20260928.d158_joint_forward_support_20261004.joint_support import Fiber


def _identity(size):
    return tuple(tuple(F(int(i == j)) for j in range(size)) for i in range(size))


def _extension(parent, g, child, source, old_bits=(), old_amplitudes=()):
    """A real forward ReLU image of this checked (possibly abstract) parent."""
    values = parent.evaluate(g, source, old_bits, old_amplitudes)
    bits = tuple(1 if value >= 0 else -1 for value in values)
    bank = child.banks[-1]
    deviation = tuple(value - bar if bit == 1 else F(0)
                      for value, bar, bit in zip(values, bank.reference, bits))
    amplitudes = tuple(sum((coefficient * value for coefficient, value in zip(row, deviation)), F(0))
                       for row in bank.carrier)
    return old_bits + bits, old_amplitudes + amplitudes


def _householder(last_bias=-F(1, 2)):
    parent = Fiber.box(((-1, 1),) * 16, enabled=True)
    source = parent.sources()
    matrix = tuple(tuple(F(int(i == j)) - F(1, 8) for j in range(16)) for i in range(16))
    g = parent.affine(source, matrix, (-F(1, 2),) * 15 + (last_bias,))
    consumer = (F(1),) * 15 + (-F(1, 4),)
    child, output = parent.relu_packet(g, (consumer,), tuple("hh:" + str(i) for i in range(16)))
    objective = child.affine(child.concat(output, source), ((1,) + (F(9, 22),) * 16,), (0,))
    return parent, source, g, child, output, objective


def _lossy_parent():
    parent = Fiber.box(((-1, 1),) * 3, enabled=True)
    g = parent.affine(parent.sources(), _identity(3), (F(1, 4), -F(1, 4), F(1, 4)))
    child, output = parent.relu_packet(g, ((1, -1, F(1, 2)),), ("loss:0", "loss:1", "loss:2"))
    return parent, g, child, output


def test_default_off_and_hz_embedding():
    with pytest.raises(cf.DisabledError):
        Fiber.box(((-1, 1),))
    parent, value = Fiber.embed_hz(((-1, 1),), (3,), ((2,),), ((5,),),
        phase_names=("original-sigma",), signed_predicates=(((1,), (1,), "le", 1),),
        decoder_matrix=((2,),), decoder_bias=(1,), enabled=True)
    assert isinstance(parent, Fiber)
    assert parent.evaluate(value, (F(1, 4),), (-1,), ()) == (-F(3, 2),)
    assert parent.decode((F(1, 4),), (-1,), ()) == (F(3, 2),)
    assert not parent.contains((F(1, 4),), (1,), ())
    assert parent.joint_certificate(value, (1,)) == cf.PhaseAffine(0, (10,))
    assert parent.bounds(value) == (-4, 10)
    assert len(parent.support_certificates(value, (1,))) == 2


def test_householder_automatic_bound_and_stable_successor():
    parent, _, g, child, _, objective = _householder()
    assert child.joint_certificate(objective, (1,)) == cf.PhaseAffine(F(72, 11), (0,) * 16)
    preactivation = child.affine(objective, ((1,),), (-F(33, 5),))
    assert child.bounds(preactivation)[1] == -F(3, 55)
    certificates = child.support_certificates(preactivation, (1,))
    assert certificates[0].extrema(child.work)[1] > certificates[1].extrema(child.work)[1]
    next_state, result = child.relu_packet(preactivation, ((1,),), ("whole-box-zero",))
    assert isinstance(next_state, Fiber)
    assert next_state.banks[-1].preactivation_bounds[0][1] == -F(3, 55)
    assert next_state.bounds(result) == (0, 0)
    assert len(next_state.phases) == 17
    assert all(next_state.phases[i] is child.phases[i] for i in range(16))
    # A real member is retained, but the zero conclusion above is a whole-state bound.
    point = (0,) * 16
    old_bits, old_amplitudes = _extension(parent, g, child, point)
    bits, amplitudes = _extension(child, preactivation, next_state, point, old_bits, old_amplitudes)
    assert next_state.evaluate(result, point, bits, amplitudes) == (0,)
    assert next_state.finite_rows_hold(point, bits, amplitudes)
    assert not next_state.finite_rows_hold(point, old_bits + (1,), old_amplitudes + (F(33, 5),))
    assert not next_state.finite_rows_hold(point, old_bits + (-1,), old_amplitudes + (1,))


def test_nonuniform_negative_references():
    parent, _, g, child, _, objective = _householder(-F(3, 4))
    certificate = child.joint_certificate(objective, (1,))
    assert certificate == cf.PhaseAffine(F(72, 11), (0,) * 15 + (-F(1, 4),))
    assert child.bounds(objective)[1] == F(72, 11)
    assert child.banks[-1].reference[-1] == -F(3, 4)
    point = (-1,) * 16
    bits, amplitudes = _extension(parent, g, child, point)
    assert bits == (1,) * 16
    actual = child.evaluate(objective, point, bits, amplitudes)[0]
    assert actual <= certificate.at((1,) * 16, child.work)
    assert certificate.at((1,) * 16, child.work) == F(72, 11) - F(1, 4)


def test_mixed_bias_keeps_signed_phase_coefficients():
    parent = Fiber.box(((-1, 1),) * 4, enabled=True)
    g = parent.affine(parent.sources(), _identity(4), (F(1, 4), -F(1, 4), F(1, 4), -F(1, 4)))
    child, mass = parent.relu_packet(g, ((1, 1, 1, 1),), ("mixed:0", "mixed:1", "mixed:2", "mixed:3"))
    target = child.affine(child.concat(mass, parent.sources()), ((1,) + (-F(5, 12),) * 4,), (0,))
    assert child.banks[-1].energy == cf.PhaseAffine(4, (0,) * 4)
    assert child.joint_certificate(target, (1,)) == cf.PhaseAffine(
        F(5, 3), (F(5, 12), -F(1, 12), F(5, 12), -F(1, 12)))
    assert child.bounds(target)[1] == F(5, 2)
    bits, amplitudes = _extension(parent, g, child, (0,) * 4)
    assert child.evaluate(target, (0,) * 4, bits, amplitudes) == (F(1, 2),)


def test_negative_consumer_uses_mass_lower_bound():
    parent = Fiber.box(((1, 2),), enabled=True)
    source = parent.sources()
    child, negative = parent.relu_packet(source, ((-1,),), ("positive-gate",))
    # The negative consumer must use L_Q=beta, not simply drop the mass term.
    certificate = child.joint_certificate(negative, (1,))
    assert certificate == cf.PhaseAffine(0, (-1,))
    assert certificate.at((1,), child.work) == -1
    strengthened = child.add(negative, child.phases01())
    assert child.joint_certificate(strengthened, (1,)) == cf.PhaseAffine(0, (0,))
    assert child.bounds(strengthened)[1] == 0
    point = (F(5, 4),)
    bits, amplitudes = _extension(parent, source, child, point)
    assert child.evaluate(negative, point, bits, amplitudes) == (-F(5, 4),)
    assert child.evaluate(strengthened, point, bits, amplitudes) == (-F(1, 4),)


def test_duplicate_consumers_preserve_norm_cancellation():
    parent = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    child, outputs = parent.relu_packet(parent.sources(), ((1, 0), (1, 0)), ("duplicate:0", "duplicate:1"))
    difference = child.affine(outputs, ((1, -1),), (0,))
    certificates = child.support_certificates(difference, (1,))
    assert len(certificates) == 2
    assert certificates[0].extrema(child.work)[1] == 0
    assert certificates[1].extrema(child.work)[1] > 0
    assert child.support(difference, (1,)) == 0
    assert child.bounds(difference) == (0, 0)
    # The scalar minimum preserves the entire old certificate, not mixed coefficients.
    assert certificates[0] == cf.PhaseAffine(0, (0, 0))


def test_shared_skip_and_affine_cancellation():
    parent = Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1,),), (F(1, 2),))
    child, q = parent.relu_packet(g, ((1,),), ("skip",))
    skip = child.add(q, child.affine(source, ((3,),), (2,)))
    zero = child.affine(child.concat(skip, q, source), ((1, -1, -3),), (-2,))
    assert child.joint_certificate(zero, (1,)) == cf.PhaseAffine(0, (0,))
    assert child.bounds(zero) == (0, 0)
    assert not any(zero.forms[0].source + zero.forms[0].phase + zero.forms[0].error)
    point = (F(1, 4),)
    bits, amplitudes = _extension(parent, g, child, point)
    assert child.evaluate(skip, point, bits, amplitudes) == (F(7, 2),)
    assert child.frame is parent.frame and len(child.errors) == 1


def test_two_bank_substitution_preserves_parent_identity():
    parent = Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1,),), (F(1, 4),))
    first, q = parent.relu_packet(g, ((1,),), ("first",))
    next_g = first.affine(first.concat(q, source), ((1, -F(1, 2)),), (-F(1, 8),))
    second, r = first.relu_packet(next_g, ((1,),), ("second",))
    target = second.affine(second.concat(r, q, source), ((1, -F(1, 3), F(1, 5)),), (0,))
    assert isinstance(second, Fiber)
    assert second.banks[0] is first.banks[0]
    assert second.phases[0] is first.phases[0] and second.errors[0] is first.errors[0]
    assert len(second.banks) == len(second.errors) == 2
    certificate = second.joint_certificate(target, (1,))
    lower, upper = second.bounds(target)
    for point in ((-F(3, 4),), (0,), (F(1, 2),)):
        old_bits, old_amplitudes = _extension(parent, g, first, point)
        bits, amplitudes = _extension(first, next_g, second, point, old_bits, old_amplitudes)
        actual = second.evaluate(target, point, bits, amplitudes)[0]
        assert lower <= actual <= upper
        assert actual <= certificate.at(tuple((bit + 1) // 2 for bit in bits), second.work)
        assert second.decode(point, bits, amplitudes) == point


def test_abstract_parent_members_remain_sound():
    _, _, parent, output = _lossy_parent()
    point, old_bits, fake_amplitudes = (0, 0, 0), (1, -1, 1), (F(1, 32), -F(1, 32))
    assert parent.contains(point, old_bits, fake_amplitudes)
    assert parent.evaluate(output, point, old_bits, fake_amplitudes) == (F(13, 32),)
    assert parent.evaluate(output, point, old_bits, (0, 0)) == (F(3, 8),)
    certificate = parent.joint_certificate(output, (1,))
    assert certificate.at((1, 0, 1), parent.work) >= F(13, 32)
    assert parent.bounds(output)[1] >= F(13, 32)
    g = parent.affine(output, ((1,),), (-F(25, 64),))
    child, result = parent.relu_packet(g, ((1,),), ("abstract-child",))
    bits, amplitudes = _extension(parent, g, child, point, old_bits, fake_amplitudes)
    assert child.evaluate(result, point, bits, amplitudes) == (F(1, 64),)
    assert child.bounds(result)[1] >= F(1, 64)
    assert child.decode(point, bits, amplitudes) == point
    # A different finite point fails the native mixed-phase range condition.
    other_amplitudes = (F(1, 32), F(1, 32))
    assert parent.finite_rows_hold(point, old_bits, other_amplitudes)
    assert not parent.contains(point, old_bits, other_amplitudes)


def test_nonunit_box_and_zero_mass():
    parent = Fiber.box(((F(99, 100), F(101, 100)),), enabled=True)
    child, q = parent.relu_packet(parent.sources(), ((1,),), ("narrow",))
    assert child.banks[-1].reference == (1,)
    assert child.banks[-1].energy == cf.PhaseAffine(F(1, 10000), (0,))
    assert child.bounds(q)[1] == F(101, 100)
    assert child.evaluate(q, (F(201, 200),), (1,), (F(1, 200),)) == (F(201, 200),)
    root = Fiber.box(((-1, 1),), enabled=True)
    negative = root.affine(root.sources(), ((1,),), (-2,))
    inactive, output = root.relu_packet(negative, ((-2,),), ("inactive",))
    assert inactive.bounds(output) == (0, 0)
    assert inactive.evaluate(output, (0,), (-1,), (0, 0)) == (0,)
    assert not inactive.finite_rows_hold((0,), (1,), (-4, 2))
    zero = root.affine(root.sources(), ((0,),), (0,))
    touching, output = root.relu_packet(zero, ((1,),), ("zero",))
    assert touching.bounds(output) == (0, 0)
    assert touching.banks[-1].energy == cf.PhaseAffine(0, (0,))
    for label in (-1, 1):
        assert touching.evaluate(output, (0,), (label,), (0,)) == (0,)


def test_identity_bits_and_work_fail_closed():
    parent = Fiber.box(((-1, 1),), phase_names=("old",), enabled=True)
    source = parent.sources()
    left, first = parent.relu_packet(source, ((1,),), ("left",))
    right, second = parent.relu_packet(source, ((1,),), ("right",))
    assert left.phases[0] is parent.phases[0]
    with pytest.raises(ValueError):
        left.joint_certificate(second, (1,))
    with pytest.raises(ValueError):
        right.support(first, (1,))
    with pytest.raises(ValueError):
        left.joint_certificate(first, ())
    with pytest.raises(ValueError):
        left.joint_certificate(first, (0.5,))
    assert not left.contains((0,), (-1, 0), (0,))
    with pytest.raises(ValueError):
        cf.PhaseAffine(1 << 513)
    left.work.charge(cf.MAX_WORK - left.work.used)
    with pytest.raises(ValueError):
        left.support(first, (1,))


def test_query_cost_adds_no_domain_coordinates():
    _, _, state, output = _lossy_parent()
    source = state.sources()
    banks, predicates, phases, amplitudes = state.banks, state.predicates, state.phases, state.errors
    before = state.cost(output, source)
    assert before["query_support_rule"] == "fixed_reverse_consumer_mass_v1"
    certificate = state.joint_certificate(output, (1,))
    assert isinstance(certificate, cf.PhaseAffine)
    assert len(state.support_certificates(output, (1,))) == 2
    state.bounds(output)
    after = state.cost(output, source)
    assert state.banks is banks and state.predicates is predicates
    assert state.phases is phases and state.errors is amplitudes
    assert {key: value for key, value in before.items() if key != "algebra_work_used"} == {
        key: value for key, value in after.items() if key != "algebra_work_used"}
    assert after["algebra_work_used"] > before["algebra_work_used"]
    assert after["complete_physical_qualification"] is False
    assert after["terminal_is_outer_approximation"] is True
