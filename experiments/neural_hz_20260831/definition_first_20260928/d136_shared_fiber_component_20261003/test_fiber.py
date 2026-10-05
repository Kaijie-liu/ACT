"""Twelve exact mathematical reference tests; not benchmark or GPU admission."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d136_shared_fiber_component_20261003.fiber import (
    DisabledError, Fiber, SourceConstraint, causal_support,
)


def _global_control():
    element = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    source = element.sources()
    f1 = element.affine(source, ((1, F(1, 10)),), (F(1, 4),))
    element, q1 = element.relu(f1, "q1")
    ypart = element.affine(source, ((0, F(1, 4)),), (-F(1, 8),))
    g2 = element.add(ypart, element.affine(q1, ((F(1, 2),),), (0,)))
    element, q2 = element.relu(g2, "q2")
    joint = element.concat(q1, q2)
    objective = element.affine(joint, ((-1, 2),), (0,))
    return element, source, q1, q2, objective


def test_opt_in_and_bad_bounds_fail_closed():
    with pytest.raises(DisabledError):
        Fiber.box(((-1, 1),))
    with pytest.raises(ValueError):
        Fiber.box(((1, -1),), enabled=True)
    with pytest.raises(ValueError):
        Fiber.box(((-1.0, 1),), enabled=True)
    with pytest.raises(ValueError):
        Fiber.box(((0, 1 << 512),), enabled=True)
    element = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        element.relu(element.sources(), "bad", bounds=(0, 1))
    with pytest.raises(ValueError):
        causal_support((1,), ((1,),), (1,))
    with pytest.raises(ValueError):
        causal_support((-1,), ((0,),), (1,))


def test_affine_composition_is_exact():
    element = Fiber.box(((-1, 1), (-2, 2)), enabled=True)
    source = element.sources()
    first = element.affine(source, ((1, 2), (-1, 1)), (3, -2))
    composed = element.affine(first, ((2, -3),), (4,))
    direct = element.affine(source, ((5, 1),), (16,))
    assert composed.forms == direct.forms
    assert element.evaluate(composed, (F(1, 2), -1), (), ()) == (F(35, 2),)


def test_add_cancellation_and_shared_consumers():
    element = Fiber.box(((-1, 1),), enabled=True)
    element, q = element.relu(element.sources(), "gate")
    negative = element.affine(q, ((-1,),), (0,))
    cancelled = element.add(q, negative)
    other = element.affine(q, ((3,),), (2,))
    assert element.bounds(cancelled) == (0, 0)
    assert q.amplitudes[0] is other.amplitudes[0]
    assert element.evaluate(other, (F(1, 2),), (1,), (F(1, 2),)) == (F(7, 2),)


def test_concat_preserves_one_joint_amplitude():
    element = Fiber.box(((-1, 1),), enabled=True)
    element, q = element.relu(element.sources(), "gate")
    joined = element.concat(q, q)
    assert len(element.amplitudes) == 1
    assert joined.forms[0] == joined.forms[1]
    assert element.support(joined, (1, -1)) == 0
    assert element.evaluate(joined, (1,), (1,), (1,)) == (1, 1)


def test_misaligned_frames_and_unjoined_branches_rejected():
    base = Fiber.box(((-1, 1),), enabled=True)
    other = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        base.add(base.sources(), other.sources())
    left, lq = base.relu(base.sources(), "left")
    right, rq = base.relu(base.sources(), "right")
    with pytest.raises(ValueError):
        left.concat(lq, rq)
    assert left.frame is right.frame
    assert left.amplitudes[0] is not right.amplitudes[0]


def test_relu_zero_keeps_both_phase_labels():
    element = Fiber.box(((-1, 1),), enabled=True)
    element, q = element.relu(element.sources(), "gate")
    assert element.contains((0,), (-1,), (0,))
    assert element.contains((0,), (1,), (0,))
    assert not element.contains((0,), (0,), (0,))
    assert element.evaluate(q, (0,), (-1,), (0,)) == (0,)
    assert element.evaluate(q, (0,), (1,), (0,)) == (0,)


def test_relu_graph_is_nonconvex():
    element = Fiber.box(((-1, 1),), enabled=True)
    element, _ = element.relu(element.sources(), "gate")
    assert element.contains((-1,), (-1,), (0,))
    assert element.contains((1,), (1,), (1,))
    assert not element.contains((0,), (-1,), (F(1, 2),))
    assert not element.contains((0,), (1,), (F(1, 2),))
    assert not element.contains((0,), (0,), (F(1, 2),))


def test_stable_relu_keeps_original_phase_identity():
    base = Fiber.box(((-1, 1),), enabled=True)
    negative = base.affine(base.sources(), ((1,),), (-2,))
    element, q = base.relu(negative, "off")
    original_phase = element.phases[0]
    element, qq = element.relu(q, "zero")
    assert element.phases[0] is original_phase
    assert len(element.phases) == len(element.amplitudes) == 2
    assert element.contains((0,), (-1, -1), (0, 0))
    assert element.contains((0,), (-1, 1), (0, 0))
    assert not element.contains((0,), (1, -1), (0, 0))
    assert element.bounds(qq) == (0, 0)


def test_source_predicates_and_decoder_preserved():
    element = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
                        predicates=(SourceConstraint((1, -1), (0,), "eq", 0),
                                    SourceConstraint((1, 0), (0,), "le", F(1, 2)),
                                    SourceConstraint((0, 0), (1,), "eq", 1)),
                        decoder_matrix=((2, 0), (0, 3)), decoder_bias=(1, -1), enabled=True)
    source = element.sources()
    x = element.affine(source, ((1, 0),), (0,))
    original = element.phases[0]
    frame = element.frame
    element, _ = element.relu(x, "gate")
    assert element.frame is frame and element.phases[0] is original
    assert element.contains((F(1, 2), F(1, 2)), (1, 1), (F(1, 2),))
    assert not element.contains((F(1, 2), 0), (1, 1), (F(1, 2),))
    assert not element.contains((1, 1), (1, 1), (1,))
    assert not element.contains((0, 0), (-1, -1), (0,))
    assert element.decode((F(1, 2), F(1, 2)), (1, 1), (F(1, 2),)) == (2, F(1, 2))
    with pytest.raises(ValueError):
        element.decode((1, 1), (1, 1), (1,))


def test_exact_fiber_support_recurrence():
    result = causal_support((F(9, 16), F(1, 4)), ((0, 0), (F(1, 2), 0)), (-1, 3))
    assert result.coefficients == (F(1, 2), 3)
    assert result.value == F(33, 32)
    assert result.value < F(51, 32)
    assert causal_support((1, 2), ((0, 0), (1, 0)), (-1, -2)).value == 0
    assert causal_support((), (), ()).value == 0
    c, matrix, weights = (F(9, 16), F(1, 4)), ((0, 0), (F(1, 2), 0)), (-1, 3)
    k = result.coefficients
    assert all(k[i] >= 0 and k[i] - sum(matrix[j][i] * k[j] for j in range(2)) >= weights[i]
               for i in range(2))
    assert result.value == sum(a * b for a, b in zip(c, k))


def test_global_control_and_stable_successor():
    element, source, q1, q2, objective = _global_control()
    assert element.caps == (F(27, 20), F(1, 8))
    assert element.lower == ((0, 0), (F(1, 2), 0))
    assert element.support(objective, (1,)) == F(1, 4)
    predecessor_phases = element.phases
    preactivation = element.affine(objective, ((1,),), (-F(1, 2),))
    assert element.bounds(preactivation)[1] == -F(1, 4)
    element, successor = element.relu(preactivation, "successor")
    assert element.phases[:2] == predecessor_phases
    assert len(element.phases) == 3
    assert element.bounds(successor) == (0, 0)
    shared_zero = element.add(successor, element.affine(successor, ((2,),), (0,)))
    assert element.bounds(shared_zero) == (0, 0)
    state = ((-F(1, 2), 1), (-1, 1, -1), (0, F(1, 8), 0))
    assert element.contains(*state)
    assert element.evaluate(successor, *state) == (0,)
    assert element.evaluate(objective, *state) == (F(1, 4),)
    assert element.evaluate(source, *state) == state[0]
    assert element.evaluate(element.concat(q1, q2), *state) == (0, F(1, 8))


def test_old_triangle_feasible_point_is_only_comparison():
    x, y = -F(1, 2), F(1)
    q1, q2, beta1, beta2 = F(0), F(16, 47), F(0), F(20, 47)
    g1, g2 = x + y / 10 + F(1, 4), y / 4 - F(1, 8) + q1 / 2
    for g, q, beta, lower, upper in ((g1, q1, beta1, -F(17, 20), F(27, 20)),
                                   (g2, q2, beta2, -F(3, 8), F(4, 5))):
        assert 0 <= beta <= 1
        assert q >= 0 and q >= g
        assert q <= upper * beta
        assert q <= g - lower * (1 - beta)
    assert -q1 + 2 * q2 == F(32, 47) > F(1, 2)
    assert q2 > F(1, 8) + q1 / 2
    element, _, _, _, objective = _global_control()
    assert not element.contains((x, y), (2 * beta1 - 1, 2 * beta2 - 1), (q1, q2))
    assert element.support(objective, (1,)) == F(1, 4)
    assert element.contains((x, y), (-1, 1), (0, F(1, 8)))
