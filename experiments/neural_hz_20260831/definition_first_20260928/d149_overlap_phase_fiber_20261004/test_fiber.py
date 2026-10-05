"""Twenty fixed mathematical tests; no model, numerical search or new solver."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d149_overlap_phase_fiber_20261004.fiber import (
    DisabledError, Fiber, PhaseAffine, SourceConstraint, rational, sqrt_upper,
)


def _householder():
    state = Fiber.box(((-1, 1),) * 16, enabled=True)
    source = state.sources()
    matrix = tuple(tuple(F(int(i == j)) - F(1, 8) for j in range(16)) for i in range(16))
    g = state.affine(source, matrix, (-F(1, 2),) * 16)
    child, q = state.relu(g, tuple("householder:" + str(i) for i in range(16)))
    return state, source, g, child, q


def _sum_squares(values):
    return sum((value * value for value in values), F(0))


def test_opt_in_and_rational_domain_rejection():
    with pytest.raises(DisabledError):
        Fiber.box(((-1, 1),))
    for invalid in (0.5, True, float("nan"), float("inf"), 1 << 513):
        with pytest.raises(ValueError):
            rational(invalid)
    with pytest.raises(ValueError):
        Fiber.box(((1, -1),), enabled=True)
    with pytest.raises(ValueError):
        sqrt_upper(-1)
    upper = sqrt_upper(F(2))
    assert upper * upper >= 2 and upper < F(3, 2)
    assert sqrt_upper(F(9, 16)) == F(3, 4)
    base = Fiber.box(((F(99, 100), F(101, 100)),), enabled=True)
    child, q = base.relu(base.sources(), ("narrow",))
    assert base.frame.reference == (F(1),)
    assert child.banks[-1].whole.radii[0] == PhaseAffine(F(1, 200), (0,))
    assert q.forms[0].constant == -F(1, 2)
    assert q.forms[0].phase == (F(1),)
    with pytest.raises(ValueError):
        base.relu(base.sources(), ("badbound",), bounds=((1, 1),))


def test_hz_embedding_and_decoder():
    state, value = Fiber.embed_hz(((-1, 1),), (3,), ((2,),), ((5,),),
        phase_names=("original-sigma",), signed_predicates=(((1,), (1,), "le", 1),),
        decoder_matrix=((2,),), decoder_bias=(1,), enabled=True)
    assert value.forms[0].constant == -2
    assert value.forms[0].phase == (10,)
    assert state.references == (0,)
    assert state.evaluate(value, (F(1, 4),), (-1,), ()) == (-F(3, 2),)
    assert state.decode((F(1, 4),), (-1,), ()) == (F(3, 2),)
    assert not state.contains((F(1, 4),), (1,), ())
    assert state.phases[0].name == "original-sigma"


def test_affine_composition_and_cancellation():
    state = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    x = state.sources()
    first = state.affine(x, ((1, 2), (-3, 1)), (4, -2))
    second = state.affine(first, ((2, -1),), (3,))
    direct = state.affine(x, ((5, 3),), (13,))
    assert second.forms == direct.forms
    state, e = state.error_bank("shared", 2, 1, local=(((0,), 1), ((1,), 1)))
    neg = state.affine(e, ((-1, 0), (0, -1)), (0, 0))
    zero = state.add(e, neg)
    assert state.support(zero, (1, -3)) == 0
    assert all(not any(form.error) for form in zero.forms)
    assert len(state.errors) == 2


def test_shared_add_concat_and_parent_extension():
    parent = Fiber.box(((-1, 1),), enabled=True)
    x = parent.sources()
    child, q = parent.relu(x, ("first",))
    combined = child.concat(q, x)
    twice = child.add(q, q)
    assert len(child.errors) == 1
    assert twice.forms[0].error == (2,)
    assert combined.forms[0].error == (1,) and combined.forms[1].error == (0,)
    assert child.evaluate(combined, (1,), (1,), (F(1, 2),)) == (1, 1)
    grandchild, z = child.relu(twice, ("second",))
    assert grandchild.phases[0] is child.phases[0]
    assert grandchild.errors[0] is child.errors[0]
    assert grandchild.concat(q, x, z).forms[0].error == (1, 0)


def test_independent_branches_are_rejected():
    parent = Fiber.box(((-1, 1),), enabled=True)
    source = parent.sources()
    left, qleft = parent.relu(source, ("left",))
    right, qright = parent.relu(source, ("right",))
    with pytest.raises(ValueError):
        left.add(qleft, qright)
    with pytest.raises(ValueError):
        right.concat(source, qleft)
    branch_a = parent.constrain(source, "le", F(1, 2))
    branch_b = parent.constrain(source, "le", F(3, 4))
    with pytest.raises(ValueError):
        branch_a.concat(branch_b.sources())
    foreign = Fiber.box(((-1, 1),), enabled=True)
    with pytest.raises(ValueError):
        parent.add(source, foreign.sources())


def test_relu_retains_zero_and_original_phase_identity():
    parent = Fiber.box(((-1, 1),), phase_names=("original",), enabled=True)
    g = parent.affine(parent.sources(), ((1,),), (F(1, 2),))
    child, q = parent.relu(g, ("zero-touch",))
    assert child.phases[0] is parent.phases[0]
    assert child.references == (0, 1)
    # Same real zero, both original labels, distinct required reflected errors.
    for old_phase in (-1, 1):
        assert child.evaluate(q, (-F(1, 2),), (old_phase, -1), (F(1, 4),)) == (0,)
        assert child.evaluate(q, (-F(1, 2),), (old_phase, 1), (-F(1, 4),)) == (0,)
    positive = parent.affine(parent.sources(), ((0,),), (2,))
    stable, output = parent.relu(positive, ("stable-active",))
    assert len(stable.phases) == len(parent.phases) + 1
    assert stable.evaluate(output, (0,), (-1, 1), (0,)) == (2,)
    assert not stable.contains((0,), (-1, -1), (0,))


def test_native_relation_remains_nonconvex():
    parent = Fiber.box(((-1, 1),), enabled=True)
    g = parent.affine(parent.sources(), ((1,),), (-F(1, 2),))
    child, q = parent.relu(g, ("biased",))
    assert child.evaluate(q, (-1,), (-1,), (F(1, 2),)) == (0,)
    assert child.evaluate(q, (1,), (1,), (F(1, 2),)) == (F(1, 2),)
    # The physical midpoint x=0,q=1/4 admits neither integral phase extension.
    assert not child.contains((0,), (-1,), (F(1, 4),))
    assert not child.contains((0,), (1,), (F(3, 4),))
    assert not child.contains((0,), (0,), (F(1, 2),))


def test_phase_center_columns_and_degree_one_update():
    parent, g = Fiber.embed_hz(((-1, 1),), (4,), ((2,),), ((3,),),
                               phase_names=("old",), enabled=True)
    child, q = parent.relu(g, ("new",))
    assert g.forms[0].constant == 1
    assert q.forms[0].constant == 0
    assert q.forms[0].source == (1,)
    assert q.forms[0].phase == (3, 1)
    assert q.forms[0].error == (1,)
    assert child.references == (0, 1)
    assert child.banks[-1].whole.radii[0] == PhaseAffine(1, (3, 0))
    descendant, again = child.relu(q, ("again",))
    assert descendant.references == (0, 1, 1)
    assert again.forms[0].constant == -F(1, 2)
    assert again.forms[0].phase == (F(3, 2), F(1, 2), 1)
    # New center uses the old bit's nominal one; mismatch has a NEGATIVE coefficient.
    radius = descendant.banks[-1].whole.radii[0]
    assert radius == PhaseAffine(F(3, 2), (3, -F(1, 2), 0))
    assert radius.extrema(descendant.work)[0] == 1


def test_overlap_groups_share_residual_coordinates():
    parent = Fiber.box(((-1, 1),), enabled=True)
    g = parent.affine(parent.sources(), ((1,), (2,), (3,)), (0, 0, 0))
    child, q = parent.relu(g, ("a", "b", "c"), groups=((0, 1), (1, 2)))
    bank = child.banks[-1]
    assert len(child.errors) == 3 and bank.coordinates == (0, 1, 2)
    assert bank.local[0].indices == (0, 1) and bank.local[1].indices == (1, 2)
    assert bank.coordinates[bank.local[0].indices[1]] == bank.coordinates[bank.local[1].indices[0]]
    assert q.forms[1].error == (0, 1, 0)
    repeated = child.concat(q, q)
    assert repeated.forms[1].error == repeated.forms[4].error
    assert len(child.errors) == 3
    # Strict native physical strengthening, not merely an identity bookkeeping test.
    control = Fiber.box(((-1, 1),) * 4, enabled=True)
    biased = control.affine(control.sources(),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        (F(1, 2),) * 4)
    names = ("p0", "p1", "p2", "p3")
    whole_only, old_q = control.relu(biased, names)
    overlapping, new_q = control.relu(biased, names, groups=((0, 1), (1, 2), (2, 3)))
    point, bits, false_error = (0,) * 4, (1,) * 4, (F(9, 10), 0, 0, 0)
    assert whole_only.banks[-1].whole.radii[0] == PhaseAffine(1, (0,) * 4)
    assert overlapping.banks[-1].local[0].radii[0].constant < F(9, 10)
    assert whole_only.evaluate(old_q, point, bits, false_error) == (F(7, 5), F(1, 2), F(1, 2), F(1, 2))
    assert not overlapping.contains(point, bits, false_error)
    # Here every g_i=1/2, so guards force these bits: re-labelling cannot rescue q.
    assert whole_only.evaluate(old_q, point, bits, (0,) * 4) == (F(1, 2),) * 4
    assert overlapping.evaluate(new_q, point, bits, (0,) * 4) == (F(1, 2),) * 4


def test_incomplete_or_duplicate_covers_are_rejected():
    parent = Fiber.box(((-1, 1),), enabled=True)
    g = parent.affine(parent.sources(), ((1,), (2,), (3,)), (0, 0, 0))
    for groups in (((0, 0), (1, 2)), ((0, 1),), ((0, 1), (0, 1), (2,)), ((0, 3), (1, 2))):
        with pytest.raises(ValueError):
            parent.relu(g, ("a", "b", "c"), groups=groups)
    with pytest.raises(ValueError):
        parent.error_bank("incomplete", 3, 1, local=(((0, 1), 1),))
    with pytest.raises(ValueError):
        parent.error_bank("duplicate", 2, 1, local=(((0, 0, 1), 1),))


def test_cover_query_cannot_replace_global_query():
    parent = Fiber.box(((-1, 1),), enabled=True)
    state, e = parent.error_bank("unit-ball", 2, 1, local=(((0,), 1), ((1,), 1)))
    whole, cover = state.support_certificates(e, (1, 1))
    assert whole.constant * whole.constant >= 2
    assert whole.constant < F(3, 2) and cover == PhaseAffine(2)
    assert state.support(e, (1, 1)) == whole.constant
    assert not state.contains((0,), (), (1, 1))
    # Propagation keeps the original whole radius as well as the looser cover radius.
    child, _ = state.relu(e, ("first", "second"), groups=((0,), (1,)))
    assert child.banks[-1].whole.radii == (
        PhaseAffine(F(1, 2), (0, 0)), PhaseAffine(1, (0, 0)))


def test_relu_concrete_witness_for_overlapping_groups():
    parent = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    source = parent.sources()
    g = parent.affine(source, ((1, 0), (0, 1), (1, -1)), (F(1, 4), -F(1, 2), F(3, 4)))
    child, q = parent.relu(g, ("one", "two", "three"), groups=((0, 1), (1, 2)))
    for point in ((F(1, 2), -F(1, 4)), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        values = parent.evaluate(g, point, (), ())
        bits = tuple(1 if value >= 0 else -1 for value in values)
        errors = tuple(F(bit, 2) * (value - form.constant)
                       for bit, value, form in zip(bits, values, g.forms))
        assert child.contains(point, bits, errors)
        assert child.evaluate(q, point, bits, errors) == tuple(max(F(0), value) for value in values)
        assert child.decode(point, bits, errors) == point


def test_d148_householder_physical_cut():
    parent, source, g, state, q = _householder()
    assert all(parent.bounds(parent._readout((form,))) == (-F(13, 4), F(9, 4)) for form in g.forms)
    assert state.banks[-1].whole.radii == (PhaseAffine(2, (0,) * 16),)
    joint = state.concat(q, source, state.phases01())
    # Source and phase terms cancel BEFORE the shared norm support is taken.
    direction = (F(1),) * 16 + (F(1, 2),) * 16 + (F(1, 2),) * 16
    whole, cover = state.support_certificates(joint, direction)
    assert whole == cover == PhaseAffine(8, (0,) * 16)
    Q, minimum_N = F(36, 5), F(16, 5)
    assert Q + minimum_N / 2 == F(44, 5) > 8
    # Combining this displayed row with retained q_i <= (9/4)*gamma_i is algebra,
    # not a new optimizer or a claim that support() optimizes all predicates.
    assert 8 / (1 + F(2, 9)) == F(72, 11) < F(33, 5)
    assert F(59, 4) * F(9, 20) == F(531, 80) > F(33, 5)
    assert not state.contains((0,) * 16, (0,) * 16, (F(7, 10),) * 16)


def test_d148_native_old_point_rejected():
    _, _, _, state, q = _householder()
    source, bits = (-F(3, 4),) * 16, (1,) * 16
    old_error, new_error = (F(17, 40),) * 16, (F(37, 40),) * 16
    assert _sum_squares(old_error) == F(289, 100) <= 4
    assert _sum_squares(new_error) == F(1369, 100) > 4
    assert not state.contains(source, bits, new_error)
    assert state.evaluate(q, source, bits, (F(3, 8),) * 16) == (F(1, 4),) * 16
    assert F(59, 4) * F(4, 5) - F(54, 11) == F(379, 55) > F(33, 5)


def test_fractional_noncontainment_is_not_native_membership():
    parent = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    g = parent.affine(parent.sources(), ((F(3, 4), 0), (0, F(3, 4))), (F(1, 2), F(1, 2)))
    state, _ = parent.relu(g, ("a", "b"))
    # D148's deliberately looser B=11/10 comparison is not this query's exact radius.
    a, beta, bar = (F(1, 20), F(159, 100)), (F(1, 2), F(1)), (F(1, 2), F(1, 2))
    new_residual = tuple(value - (2 * bit - 1) * center for value, bit, center in zip(a, beta, bar))
    old_residual = tuple(value - center for value, center in zip(a, bar))
    assert _sum_squares(new_residual) == F(5953, 5000) <= F(121, 100)
    assert _sum_squares(old_residual) == F(6953, 5000) > F(121, 100)
    assert not state.contains((-F(3, 5), 0), (0, 1), tuple(value / 2 for value in new_residual))
    assert not state.contains((-F(3, 5), 0), beta, tuple(value / 2 for value in new_residual))


def test_second_relu_and_skip_composition():
    root = Fiber.box(((-1, 1), (-1, 1)), enabled=True)
    source = root.sources()
    g = root.affine(source, ((1, 0), (0, 1)), (F(1, 4), -F(1, 4)))
    first, q = root.relu(g, ("q1", "q2"), groups=((0,), (1,)))
    assert first.references == (1, 0)
    mixed = first.affine(first.concat(q, source), ((1, -F(1, 2), F(1, 10), 0),), (-F(3, 16),))
    second, output = first.relu(mixed, ("next",))
    assert second.references == (1, 0, 1)
    assert output.forms[0].constant == -F(1, 8)
    assert output.forms[0].phase[-1] == F(1, 16)
    point, bits = (F(1, 2), -F(1, 4)), (1, -1, 1)
    # Nominal child gbar=1/16, actual preactivation49/80, so error11/40.
    errors = (F(1, 4), F(1, 8), F(11, 40))
    readout = second.concat(output, q, source)
    assert second.evaluate(readout, point, bits, errors) == (F(49, 80), F(3, 4), 0, F(1, 2), -F(1, 4))
    assert second.phases[:2] == first.phases and second.errors[:2] == first.errors
    assert second.banks[0] is first.banks[0]
    assert second.predicates[:len(first.predicates)] == first.predicates
    assert any(value < 0 for value in second.banks[-1].whole.radii[0].phase)


def test_predicates_survive_transformations():
    root = Fiber.box(((-1, 1), (-1, 1)), predicates=(
        SourceConstraint((1, -1), (), "eq", 0), SourceConstraint((1, 0), (), "le", F(1, 2))), enabled=True)
    source = root.sources()
    readout = root.affine(source, ((1, 0),), (F(1, 2),))
    child, q = root.relu(readout, ("q",))
    assert child.predicates[:2] == root.predicates
    assert child.evaluate(q, (F(1, 4), F(1, 4)), (1,), (F(1, 8),)) == (F(3, 4),)
    assert not child.contains((F(1, 4), 0), (1,), (F(1, 8),))
    assert not child.contains((F(3, 4), F(3, 4)), (1,), (F(3, 8),))
    bounded = child.constrain(q, "le", F(1, 2))
    assert not bounded.contains((F(1, 4), F(1, 4)), (1,), (F(1, 8),))
    assert bounded.predicates[:len(child.predicates)] == child.predicates


def test_interval_coefficients_keep_error_bank():
    root = Fiber.box(((-1, 1),), enabled=True)
    source = root.sources()
    state, g = root.interval_affine(source, ((1,),), ((3,),), (0,), (2,), name="parameter-error")
    assert len(state.errors) == 1 and len(state.banks) == 1
    assert state.banks[0].whole.radii[0] == PhaseAffine(2)
    assert state.banks[0].local[0].indices == (0,)
    assert state.evaluate(g, (F(1, 2),), (), (F(3, 2),)) == (F(7, 2),)
    twice = state.add(g, g)
    assert twice.forms[0].error == (2,)
    zero = state.add(g, state.affine(g, ((-1,),), (0,)))
    assert state.bounds(zero) == (0, 0)
    descendant, _ = state.relu(g, ("gate",))
    assert descendant.errors[0] is state.errors[0]
    assert descendant.banks[0] is state.banks[0]
    with pytest.raises(ValueError):
        root.interval_affine(source, ((3,),), ((1,),), (0,), (2,), name="bad")


def test_exact_coefficients_do_not_invent_error():
    root = Fiber.box(((2, 3),), enabled=True)
    source = root.sources()
    state, g = root.interval_affine(source, ((2,),), ((2,),), (1,), (1,), name="exact")
    assert state is root and not state.errors and not state.banks
    assert g.forms[0].source == (2,) and g.forms[0].constant == 1
    assert state.evaluate(g, (F(5, 2),), (), ()) == (6,)
    child, q = state.relu(g, ("nonunit",))
    assert child.references == (1,)
    assert child.banks[-1].whole.radii == (PhaseAffine(F(1, 2), (0,)),)
    assert q.forms[0].constant == -F(5, 2) and q.forms[0].phase == (6,)
    assert child.evaluate(q, (3,), (1,), (F(1, 2),)) == (7,)


def test_full_cost_counts_norm_incidence():
    root = Fiber.box(((-1, 1), (-1, 1)), phase_names=("original",),
        predicates=(SourceConstraint((1, 0), (0,), "le", 1),), enabled=True)
    state, e = root.error_bank("overlap", 3, 3, local=(((0, 1), 2), ((1, 2), 2)))
    record = state.cost(e)
    assert record["source_factors"] == 2 and record["phase_factors"] == 1
    assert record["residual_factors"] == 3 and record["banks"] == 1
    assert record["norm_groups"] == record["norm_constraints"] == 3
    assert record["norm_incidence"] == 7
    assert record["radius_coefficients"] == 6
    assert record["predicate_rows"] == 1 and record["predicate_coefficients"] == 5
    assert record["decoder_coefficients"] == 6
    assert record["live_readout_coefficients"] == 21
    assert record["live_readout_nnz"] == 3
    assert record["phase_reference_entries"] == 1 and record["source_reference_entries"] == 2
    repeated = state.cost(state.concat(e, e))
    assert repeated["residual_factors"] == 3 and repeated["norm_incidence"] == 7
    assert repeated["live_readout_coefficients"] == 42
    assert repeated["algebra_work_used"] > record["algebra_work_used"] > 0
    assert repeated["complete_physical_qualification"] is False
