import numpy as np
import pytest

from experiments.neural_hz_20260831 import c6_support_affine_plan_v1 as v1, c6_support_affine_plan_v2 as v2
from experiments.neural_hz_20260831 import test_c6_support_affine_plan_v1 as original_tests
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import *


@pytest.fixture(autouse=True)
def use_v2_for_the_complete_original_suite(monkeypatch):
    monkeypatch.setattr(original_tests, 'plan', v2.plan)
    monkeypatch.setattr(original_tests, 'SupportEngine', v2.SupportEngine)


def test_same_certificate_and_reduced_integer_visits():
    expr, op = fixture()
    keep = np.ones(expr.n_out, dtype=bool)
    before = v1.SupportEngine
    old, new = v1.plan(expr, keep), v2.plan(expr, keep)
    assert v1.SupportEngine is before
    assert old.report['terms'] == new.report['terms']
    assert old.report['uncached_total_product_upper_bound'] == new.report['uncached_total_product_upper_bound']
    assert new.report['support_integer_visits'] < old.report['support_integer_visits']
    assert old.support_sha256 == new.support_sha256


def test_budget_rejection_restores_research_engine_binding():
    expr, op = fixture()
    before = v1.SupportEngine
    with pytest.raises(MemoryError):
        v2.plan(expr, np.ones(expr.n_out, dtype=bool), max_visits=0)
    assert v1.SupportEngine is before
