"""Fifteen frozen mathematical controls; no source worker or new optimizer."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d149_overlap_phase_fiber_20261004 import fiber as old
from experiments.neural_hz_20260831.definition_first_20260928.d150_stable_phase_charts_20261004.stable_fiber import (
    DisabledError, Fiber, PhaseAffine,
)


def _mixed(*, decoder_matrix=None, decoder_bias=None):
    root = Fiber.box(((-1, 1), (-1, 1)), decoder_matrix=decoder_matrix,
                     decoder_bias=decoder_bias, enabled=True)
    source = root.sources()
    g = root.affine(source, ((1, 1), (1, -1)), (3, F(1, 2)))
    state, q = root.relu(g, ("stable", "crossing"))
    return root, source, g, state, q


def _norm_values(state, norm, point, bits, errors):
    phase01 = tuple(F(bit + 1, 2) for bit in bits)
    return tuple(form.at(tuple(map(F, point)), phase01, tuple(map(F, errors)), state.work)
                 for form in norm.forms)


def _squared(values):
    return sum((value * value for value in values), F(0))


def test_opt_in_and_inherited_nonconvex_embedding():
    with pytest.raises(DisabledError):
        Fiber.box(((-1, 1),))
    state, value = Fiber.embed_hz(((-1, 1),), (3,), ((2,),), ((5,),),
        phase_names=("original",), signed_predicates=(((1,), (1,), "le", 1),), enabled=True)
    assert isinstance(state, Fiber) and not state.affine_norms
    assert state.evaluate(value, (F(1, 4),), (-1,), ()) == (-F(3, 2),)
    assert not state.contains((F(1, 4),), (1,), ())
    root = Fiber.box(((-1, 1),), enabled=True)
    g = root.affine(root.sources(), ((1,),), (-F(1, 2),))
    child, q = root.relu(g, ("biased",))
    assert child.evaluate(q, (-1,), (-1,), (F(1, 2),)) == (0,)
    assert child.evaluate(q, (1,), (1,), (F(1, 2),)) == (F(1, 2),)
    assert not child.contains((0,), (-1,), (F(1, 4),))
    assert not child.contains((0,), (1,), (F(3, 4),))
    assert not child.contains((0,), (0,), (F(1, 2),))


def test_direct_predicate_bounds_without_search():
    root = Fiber.box(((-1, 1),), enabled=True)
    x = root.sources()
    upper = root.affine(x, ((1,),), (2,))
    state = root.constrain(upper, "le", F(9, 4))
    lower = state.affine(x, ((-1,),), (1,))
    state = state.constrain(lower, "le", F(5, 4))
    shifted = state.affine(x, ((1,),), (F(1, 2),))
    assert state.bounds(shifted) == (F(1, 4), F(3, 4))
    equality = state.affine(x, ((-1,),), (2,))
    point_state = state.constrain(equality, "eq", F(7, 4))
    assert point_state.bounds(shifted) == (F(3, 4), F(3, 4))
    twice = root.affine(x, ((2,),), (0,))
    no_rescaling_search = root.constrain(twice, "le", 0)
    assert no_rescaling_search.bounds(x) == (-1, 1)
    two = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    xy = two.sources()
    x_only, y_only = two.affine(xy, ((1, 0),), (0,)), two.affine(xy, ((0, 1),), (0,))
    two = two.constrain(x_only, "le", 0).constrain(y_only, "le", 0)
    total = two.affine(xy, ((1, 1),), (0,))
    assert two.bounds(total) == (-2, 2)  # No combinations of different rows.
    inconsistent = root.constrain(x, "le", -2)
    with pytest.raises(ValueError):
        inconsistent.bounds(x)


def test_stable_active_has_no_new_residual():
    root = Fiber.box(((1, 2),), enabled=True)
    source = root.sources()
    state, q = root.relu(source, ("positive",))
    assert not state.errors and not state.banks and not state.affine_norms
    assert len(state.phases) == 1 and state.references == (1,)
    assert q.forms[0].source == (1,) and q.forms[0].phase == (0,)
    assert state.evaluate(q, (F(3, 2),), (1,), ()) == (F(3, 2),)
    difference = state.add(q, state.affine(source, ((-1,),), (0,)))
    assert state.bounds(difference) == (0, 0)
    legacy = old.Fiber.box(((1, 2),), enabled=True)
    legacy, legacy_q = legacy.relu(legacy.sources(), ("positive",))
    assert legacy.evaluate(legacy_q, (F(3, 2),), (1,), (F(1, 4),)) == (F(7, 4),)
    assert not state.contains((F(3, 2),), (1,), (F(1, 4),))


def test_stable_inactive_has_no_new_residual():
    root = Fiber.box(((-2, -1),), enabled=True)
    state, q = root.relu(root.sources(), ("negative",))
    assert not state.errors and not state.banks and not state.affine_norms
    assert state.references == (0,) and len(state.phases) == 1
    assert state.evaluate(q, (-F(3, 2),), (-1,), ()) == (0,)
    assert not state.contains((-F(3, 2),), (1,), ())
    assert state.bounds(q) == (0, 0)


def test_zero_preserves_two_phase_labels():
    for bounds in (((0, 1),), ((-1, 0),), ((0, 0),)):
        root = Fiber.box(bounds, phase_names=("original",), enabled=True)
        state, q = root.relu(root.sources(), ("zero",))
        assert state.phases[0] is root.phases[0]
        assert len(state.phases) == 2 and not state.errors
        for original in (-1, 1):
            for new_phase in (-1, 1):
                assert state.evaluate(q, (0,), (original, new_phase), ()) == (0,)
    again, repeated = state.relu(q, ("zero-again",))
    for first in (-1, 1):
        for second in (-1, 1):
            assert again.evaluate(repeated, (0,), (1, first, second), ()) == (0,)


def test_relu_idempotence_preserves_existing_residual():
    root = Fiber.box(((-1, 1),), enabled=True)
    g = root.affine(root.sources(), ((1,),), (F(1, 2),))
    first, q = root.relu(g, ("first",))
    assert first.bounds(q)[0] == 0
    second, repeated = first.relu(q, ("second",))
    assert second.errors == first.errors and second.banks == first.banks
    assert len(second.errors) == 1 and len(second.phases) == 2
    difference = second.add(repeated, second.affine(q, ((-1,),), (0,)))
    assert second.bounds(difference) == (0, 0)
    # Sound on every parent abstract member, including a non-network amplitude.
    assert second.evaluate(repeated, (0,), (1, 1), (F(1, 2),)) == (1,)
    for first_phase, error in ((-1, F(1, 4)), (1, -F(1, 4))):
        for second_phase in (-1, 1):
            assert second.evaluate(repeated, (-F(1, 2),), (first_phase, second_phase), (error,)) == (0,)


def test_mixed_projection_retains_spent_energy():
    _, source, _, state, q = _mixed()
    assert len(state.phases) == 2 and len(state.errors) == 1 and len(state.affine_norms) == 1
    assert q.forms[0].source == (1, 1) and q.forms[0].error == (0,)
    assert state.banks[-1].whole.radii == (PhaseAffine(1, (0, 0)),)
    point, bits, error = (1, 1), (1, 1), (1,)
    # Removing conditioned norms would wrongly admit this ordinary physical point.
    assert old.Fiber.contains(state, point, bits, error)
    assert _norm_values(state, state.affine_norms[0], point, bits, error) == (1, 1)
    assert not state.contains(point, bits, error)
    joint = state.affine(state.concat(q, source), ((0, 1, 0, 1),), (0,))
    assert old.Fiber.support(state, joint, (1,)) == F(5, 2)
    assert state.support(joint, (1,)) == old.sqrt_upper(2, state.work) + F(1, 2) < 2
    preactivation = state.affine(joint, ((1,),), (-2,))
    successor, output = state.relu(preactivation, ("successor",))
    assert successor.errors == state.errors and successor.affine_norms == state.affine_norms
    assert successor.bounds(output) == (0, 0)
    assert successor.evaluate(output, (0, 0), (1, 1, -1), (0,)) == (0,)


def test_mixed_local_groups_not_collapsed_by_crossing_indices():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    g = root.affine(root.sources(), ((1, 0), (0, 1), (1, -1)), (2, 2, F(3, 2)))
    state, q = root.relu(g, ("sx", "sy", "c"), groups=((0, 2), (1, 2)))
    assert len(state.errors) == 1 and len(state.affine_norms) == 3
    assert state.affine_norms[1].forms != state.affine_norms[2].forms
    assert state.affine_norms[1].forms[-1].error == state.affine_norms[2].forms[-1].error == (1,)
    for point, failing, other in (((1, 0), 1, 2), ((0, 1), 2, 1)):
        bits, error = (1, 1, 1), (F(11, 10),)
        assert old.Fiber.contains(state, point, bits, error)
        whole = state.affine_norms[0]
        assert _squared(_norm_values(state, whole, point, bits, error)) <= whole.radii[0].constant ** 2
        bad, good = state.affine_norms[failing], state.affine_norms[other]
        assert _squared(_norm_values(state, bad, point, bits, error)) > bad.radii[0].constant ** 2
        assert _squared(_norm_values(state, good, point, bits, error)) <= good.radii[0].constant ** 2
        assert not state.contains(point, bits, error)
    assert len(q.forms) == 3 and all(form.error == (0,) for form in q.forms[:2])


def test_crossing_rule_matches_parent_candidate():
    fresh = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    legacy = old.Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    matrix, bias = ((1, -1), (1, 1)), (F(1, 4), -F(1, 2))
    g = fresh.affine(fresh.sources(), matrix, bias)
    old_g = legacy.affine(legacy.sources(), matrix, bias)
    fresh, q = fresh.relu(g, ("one", "two"), groups=((0,), (1,)))
    legacy, old_q = legacy.relu(old_g, ("one", "two"), groups=((0,), (1,)))
    assert isinstance(fresh, Fiber) and not fresh.affine_norms
    assert q.forms == old_q.forms and fresh.references == legacy.references
    assert fresh.banks == legacy.banks and fresh.predicates == legacy.predicates
    assert tuple(bit.name for bit in fresh.phases) == tuple(bit.name for bit in legacy.phases)


def test_actual_crossing_witness_and_decoder():
    root, _, g, state, q = _mixed(decoder_matrix=((2, 0), (0, 3)), decoder_bias=(1, -1))
    for point in ((F(1, 2), -F(1, 2)), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        values = root.evaluate(g, point, (), ())
        bits = tuple(1 if value >= 0 else -1 for value in values)
        error = (F(bits[1], 2) * (values[1] - F(1, 2)),)
        assert state.evaluate(q, point, bits, error) == tuple(max(F(0), value) for value in values)
        assert state.decode(point, bits, error) == (2 * point[0] + 1, 3 * point[1] - 1)
    assert not state.contains((0, 0), (0, 1), (0,))


def test_shared_skip_after_stable_projection():
    _, source, _, state, q = _mixed()
    doubled = state.add(q, q)
    assert doubled.forms[0].error == (0,) and doubled.forms[1].error == (2,)
    stable_pre = state.affine(state.concat(q, source), ((1, 0, 1, -1),), (0,))
    assert state.bounds(stable_pre) == (1, 5)
    child, output = state.relu(stable_pre, ("exact-next",))
    assert child.errors[0] is state.errors[0] and child.affine_norms == state.affine_norms
    joint = child.concat(output, q, source)
    assert child.evaluate(joint, (0, 0), (1, 1, 1), (0,)) == (3, 3, F(1, 2), 0, 0)
    zero = child.add(output, child.affine(stable_pre, ((-1,),), (0,)))
    assert child.bounds(zero) == (0, 0)


def test_negative_phase_radius_and_nominal_reference():
    root = Fiber.box(((-1, 1),), enabled=True)
    g = root.affine(root.sources(), ((1,),), (F(1, 2),))
    first, q = root.relu(g, ("first",))
    pre = first.affine(q, ((1,),), (-F(1, 4),))
    assert first.bounds(pre) == (-F(1, 4), F(5, 4))
    second, output = first.relu(pre, ("second",))
    assert second.references == (1, 1)
    assert second.banks[-1].whole.radii == (PhaseAffine(F(3, 4), (-F(1, 4), 0)),)
    assert output.forms[0].constant == -F(1, 4)
    assert output.forms[0].phase == (F(1, 4), F(1, 4))
    assert second.evaluate(output, (0,), (1, 1), (0, 0)) == (F(1, 4),)


def test_interval_bridge_survives_conditioned_norms():
    _, _, _, state, q = _mixed()
    crossing = state.affine(q, ((0, 1),), (0,))
    child, mapped = state.interval_affine(crossing, ((1,),), ((2,),), (0,), (1,), name="parameter")
    assert isinstance(child, Fiber) and child.affine_norms == state.affine_norms
    assert child.banks[0] is state.banks[0] and child.errors[0] is state.errors[0]
    assert len(child.errors) == 2 and len(child.banks) == 2
    assert child.evaluate(mapped, (0, 0), (1, 1), (0, F(3, 4))) == (2,)
    assert not child.contains((1, 1), (1, 1), (1, 0))
    cancelled = child.add(mapped, child.affine(mapped, ((-1,),), (0,)))
    assert child.bounds(cancelled) == (0, 0)


def test_conditional_constraints_survive_descendants():
    _, _, _, state, q = _mixed()
    stable = state.affine(q, ((1, 0),), (0,))
    first, _ = state.relu(stable, ("stable-again",))
    assert first.affine_norms[0] is state.affine_norms[0]
    assert not first.contains((1, 1), (1, 1, 1), (1,))
    crossing = first.affine(q, ((0, 1),), (-F(1, 4),))
    second, output = first.relu(crossing, ("cross-again",))
    assert second.affine_norms[0] is state.affine_norms[0]
    assert len(second.errors) == 2 and second.errors[0] is state.errors[0]
    assert second.evaluate(output, (0, 0), (1, 1, 1, 1), (0, 0)) == (F(1, 4),)
    false_errors = (1, F(1, 2))
    assert old.Fiber.contains(second, (1, 1), (1, 1, 1, 1), false_errors)
    assert not second.contains((1, 1), (1, 1, 1, 1), false_errors)


def test_complete_logical_cost_includes_affine_norms():
    _, _, _, state, q = _mixed()
    record = state.cost(q)
    assert record["phase_factors"] == 2 and record["residual_factors"] == 1
    assert record["norm_constraints"] == record["norm_incidence"] == 1
    assert record["affine_norms"] == record["affine_norm_constraints"] == 1
    assert record["affine_norm_rows"] == record["affine_norm_incidence"] == 2
    assert record["affine_norm_map_slots"] == 12
    assert record["affine_norm_radius_coefficients"] == 3
    assert record["predicate_rows"] == 10 and record["predicate_coefficients"] == 70
    assert record["live_readout_coefficients"] == 12
    assert record["projected_stable_coordinates"] == 1
    assert record["substituted_norm_rows"] == 2 and record["substituted_map_slots"] == 98
    assert record["projection_temporary_peak_errors"] == 2
    assert record["projection_temporary_peak_logical_entries"] > record["live_readout_coefficients"]
    repeated = state.cost(state.concat(q, q))
    assert repeated["affine_norm_map_slots"] == 12 and repeated["live_readout_coefficients"] == 24
    assert repeated["algebra_work_used"] > record["algebra_work_used"] > 0
    assert record["complete_physical_qualification"] is False
