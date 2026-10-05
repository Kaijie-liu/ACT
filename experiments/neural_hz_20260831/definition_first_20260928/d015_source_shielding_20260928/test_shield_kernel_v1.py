"""Unrun draft tests; root must freeze the source census before execution."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928.shield_kernel_v1 import (
    BudgetExceeded, KernelDisabled, KernelError, WorkBudget,
    add, affine_add, affine_scale, div, dyadic_enclose, evaluate_center,
    mul, neg, nonnegative_part, orient_error, pair_conflict, point,
    source_box_bounds, sqrt_interval,
)


def iv(lower, upper=None):
    return F(lower), F(lower if upper is None else upper)


def guard_ok(f, lower_expression, r, beta, lower, upper):
    return (r >= 0 and r >= lower_expression and r <= upper * beta
            and r <= f - lower * (1 - beta))


def test_default_is_disabled():
    budget = WorkBudget()
    with pytest.raises(KernelDisabled):
        point(0, budget)
    assert budget.used == 0


def test_budget_charges_before_work():
    budget = WorkBudget(enabled=True, limit=3)
    budget.charge(3)
    with pytest.raises(BudgetExceeded):
        budget.charge(1)
    assert budget.used == 3
    with pytest.raises(KernelError):
        budget.charge(-1)


def test_exact_types_and_512_bit_cap():
    budget = WorkBudget(enabled=True)
    assert point(1 << 511, budget) == iv(1 << 511)
    with pytest.raises(KernelError):
        point(1 << 512, budget)
    with pytest.raises(KernelError):
        point(F(1, 1 << 512), budget)
    with pytest.raises(KernelError):
        point(0.5, budget)
    with pytest.raises(KernelError):
        point(True, budget)
    with pytest.raises(KernelError):
        add((0, 1), iv(0), budget)


def test_add_and_neg_are_exact():
    budget = WorkBudget(enabled=True)
    assert add(iv(-2, 3), iv(F(1, 3), F(2, 3)), budget) == iv(F(-5, 3), F(11, 3))
    assert neg(iv(-2, 3), budget) == iv(-3, 2)


def test_multiply_crossing_signs():
    budget = WorkBudget(enabled=True)
    assert mul(iv(-2, 3), iv(-4, 5), budget) == iv(-12, 15)
    assert nonnegative_part(iv(-4, 5), budget) == iv(0, 5)


def test_divide_positive_interval():
    budget = WorkBudget(enabled=True)
    assert div(iv(2, 4), iv(2, 3), budget) == iv(F(2, 3), 2)


def test_divide_negative_interval():
    budget = WorkBudget(enabled=True)
    assert div(iv(2, 4), iv(-4, -2), budget) == iv(-2, F(-1, 2))


def test_division_touching_zero_rejects():
    budget = WorkBudget(enabled=True)
    with pytest.raises(KernelError):
        div(iv(1), iv(0, 2), budget)
    with pytest.raises(KernelError):
        div(iv(1), iv(-2, 0), budget)
    with pytest.raises(KernelError):
        div(iv(1), iv(-2, 2), budget)


def test_square_root_exact_and_invalid_points():
    budget = WorkBudget(enabled=True)
    assert sqrt_interval(F(9, 16), budget) == iv(F(3, 4))
    assert sqrt_interval(F(0), budget) == iv(0)
    with pytest.raises(KernelError):
        sqrt_interval(F(-1), budget)


def test_square_root_enclosure_proof():
    budget = WorkBudget(enabled=True)
    lower, upper = sqrt_interval(F(2), budget)
    assert lower * lower <= F(2)
    assert upper * upper >= F(2)
    assert upper - lower == F(1, 1 << 64)


def test_dyadic_rounding_contains_nonrepresentable_point():
    budget = WorkBudget(enabled=True)
    lower, upper = dyadic_enclose(iv(F(1, 3)), budget)
    assert lower <= F(1, 3) <= upper
    assert upper - lower == F(1, 1 << 64)
    assert dyadic_enclose(iv(F(3, 8)), budget) == iv(F(3, 8))


def test_dyadic_rounding_negative_endpoints():
    budget = WorkBudget(enabled=True)
    lower, upper = dyadic_enclose(iv(F(-2, 3), F(-1, 3)), budget)
    assert lower <= F(-2, 3)
    assert upper >= F(-1, 3)
    assert lower > F(-2, 3) - F(1, 1 << 64)
    assert upper < F(-1, 3) + F(1, 1 << 64)


def test_negative_bn_scale_preserves_signed_source():
    budget = WorkBudget(enabled=True)
    form, box = (iv(1), {7: iv(1)}), {7: iv(0, 1)}
    scale = div(iv(-3), sqrt_interval(F(4), budget), budget)
    scaled = affine_scale(form, scale, budget)
    assert source_box_bounds(scaled, box, budget) == iv(-3, F(-3, 2))
    assert evaluate_center(scaled, box, budget) == iv(F(-9, 4))


def test_shared_source_correlation_certifies_conflict():
    budget = WorkBudget(enabled=True)
    box = {11: iv(0, 1)}
    left, right = (iv(F(-3, 4)), {11: iv(1)}), (iv(F(1, 4)), {11: iv(-1)})
    assert source_box_bounds(left, box, budget)[1] > 0
    assert source_box_bounds(right, box, budget)[1] > 0
    assert pair_conflict(left, right, box, budget)
    assert affine_add(left, right, budget) == (iv(F(-1, 2)), {})
    independent_right = (iv(F(1, 4)), {12: iv(-1)})
    assert not pair_conflict(left, independent_right,
                             {11: iv(0, 1), 12: iv(0, 1)}, budget)


def test_orientation_handles_uncertain_and_zero_centers():
    budget = WorkBudget(enabled=True)
    tau, oriented, cap = orient_error((iv(0), {4: iv(1)}), {4: iv(0, 2)}, budget)
    assert tau == 1
    assert oriented == (iv(0), {4: iv(-1)})
    assert cap == iv(0)
    tau, oriented, cap = orient_error((iv(-1, 1), {}), {}, budget)
    assert tau == 0
    assert oriented == (iv(-1, 1), {})
    assert cap == iv(0, 1)
    assert orient_error((iv(0), {}), {}, budget)[0] == 1


def test_partial_negative_paper_control():
    budget = WorkBudget(enabled=True)
    baseline = iv(F(-1, 2))
    contribution_1 = mul(iv(1), iv(1), budget)
    contribution_2 = mul(iv(F(1, 4)), iv(1), budget)
    bound_for_3 = add(baseline, contribution_2, budget)
    bound_for_4 = add(baseline, add(contribution_1, contribution_2, budget), budget)
    assert bound_for_3 == iv(F(-1, 4))
    assert bound_for_3[1] <= 0
    assert bound_for_4 == iv(F(3, 4))
    assert bound_for_4[1] > 0


def test_original_phase_guard_at_zero_rewritten_value():
    lower, upper = F(-5, 2), F(3, 4)
    assert guard_ok(F(-1, 2), F(0), F(0), F(0), lower, upper)
    assert not guard_ok(F(-1, 2), F(0), F(0), F(1), lower, upper)
    assert guard_ok(F(0), F(0), F(0), F(0), lower, upper)
    assert guard_ok(F(0), F(0), F(0), F(1), lower, upper)


def test_rewritten_lower_row_strictly_tightens_paper_lp_point():
    lower, upper = F(-5, 2), F(3, 4)
    e1, e2, e3, e4 = F(1, 2), F(1), F(1, 2), F(0)
    assert e1 + e3 <= 1
    original = F(-1, 2) + e1 + e2 / 4 - e3 - e4
    rewritten = original + e3
    assert original == F(-1, 4)
    assert rewritten == F(1, 4)
    assert guard_ok(original, original, F(0), F(0), lower, upper)
    assert not guard_ok(original, rewritten, F(0), F(0), lower, upper)
