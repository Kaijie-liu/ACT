"""Fixed exact mathematical controls; never decode or execute a model."""
from fractions import Fraction as F
from itertools import product
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import shield_kernel_v1 as k
from experiments.neural_hz_20260831.definition_first_20260928.d120_mixed_consumer_source_20261002 import source_query as q


def test_source_template_support():
    budget = k.WorkBudget(enabled=True)
    weights = (F(0), F(1), F(-2), F(1), F(0))
    plan = q.template(weights, enabled=True, budget=budget)
    assert plan['knots'] == tuple(F(i, 4) for i in range(5))
    assert (plan['L'], plan['U']) == (F(-1, 4), F(0))
    assert (plan['L_ind'], plan['U_ind']) == (F(-1, 2), F(3, 8))
    assert plan['A_f'] == plan['A_h'] == 0
    for weights in ((F(2), F(-3), F(1), F(-1)),
                    (F(-1), F(2), F(-4), F(3), F(1)),
                    (F(1), F(0), F(0), F(0), F(-2))):
        plan = q.template(weights, enabled=True, budget=budget)
        values = [sum(w * (min(t, s) - t * s)
                      for t, w in zip(plan['knots'][1:-1], weights[1:-1]))
                  for s in plan['knots']]
        assert (plan['L'], plan['U']) == (min(values), max(values))
        assert plan['L_ind'] <= plan['L'] <= 0 <= plan['U'] <= plan['U_ind']


def test_source_endpoint_and_residual_bounds():
    budget = k.WorkBudget(enabled=True)
    assert q.endpoint_range(F(-3), F(1), (F(-2), F(4)), budget) == (F(-8), F(0))
    assert q.endpoint_range(F(2), F(-1), (F(-2), F(4)), budget) == (F(0), F(4))
    assert q.endpoint_range(F(2), F(-1), (F(1), F(4)), budget) == (F(1), F(4))
    assert q.endpoint_range(F(2), F(-1), (F(-4), F(-1)), budget) == (F(1), F(4))
    eps = F(1, 10)
    weights = (F(1), F(-2), F(3), F(-1), F(2))
    plan = q.template(weights, enabled=True, budget=budget)
    source_bounds = ((F(-1), F(1)), (F(-1) - eps, F(1) + eps),
                     (F(-1), F(1)), (F(-1) - eps, F(1) + eps), (F(-1), F(1)))
    residual_bounds = ((-eps, eps), (F(0), F(0)), (-eps, eps))
    result = q.group_bounds(plan, source_bounds, residual_bounds, enabled=True, budget=budget)
    for x, y, z in product((F(-1), F(0), F(1)), repeat=3):
        source = (x, F(3, 4) * x + F(1, 4) * y + eps * z,
                  (x + y) / 2, F(1, 4) * x + F(3, 4) * y - eps * z, y)
        value = sum(w * max(F(0), g) for w, g in zip(weights, source))
        for arm in ('original', 'independent', 'common'):
            assert result[arm][0] <= value <= result[arm][1]


def test_source_real_weight_comparison():
    # These are ordinary signed coefficient controls, not purported trained
    # model data. The actual original-model comparison is the separate worker.
    budget = k.WorkBudget(enabled=True)
    plan = q.template((F(0), F(1), F(-2), F(1), F(0)), enabled=True, budget=budget)
    bounds = ((F(-1), F(1)),) * 5
    residuals = ((F(0), F(0)),) * 3
    result = q.group_bounds(plan, bounds, residuals, enabled=True, budget=budget)
    assert result['common'] == (F(0), F(1, 2))
    assert result['independent'] == (F(-3, 4), F(1))
    assert result['original'] == (F(-2), F(2))
    for alpha in ((F(2), F(3)), (F(-3), F(-2)), (F(-1), F(2))):
        transformed = {name: q.post_affine(result[name], alpha, (F(-1, 4), F(1, 3)), budget)
                       for name in ('original', 'independent', 'common')}
        assert (transformed['original'][0] <= transformed['independent'][0]
                <= transformed['common'][0] <= transformed['common'][1]
                <= transformed['independent'][1] <= transformed['original'][1])
    positive = q.template((F(1), F(1), F(1), F(1)), enabled=True, budget=budget)
    stable = q.group_bounds(positive, ((F(1), F(2)),) * 4,
                           ((F(0), F(0)),) * 2, enabled=True, budget=budget)
    assert all(stable[name][0] <= 4 <= stable[name][1]
               for name in ('original', 'independent', 'common'))


def test_source_query_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled access')

    assert q.template(Poison()) == {'enabled': False}
    assert q.group_bounds(Poison(), Poison(), Poison()) == {'enabled': False}
    budget = k.WorkBudget(enabled=True)
    with pytest.raises(ValueError):
        q.template((F(0),) * 5, enabled=1, budget=budget)
    with pytest.raises(ValueError):
        q.template((0,) * 5, enabled=True, budget=budget)
    with pytest.raises(ValueError):
        q.template((F(0),) * 3, enabled=True, budget=budget)
    with pytest.raises(k.KernelError):
        q.template((F(1 << 513), F(0), F(0), F(0)), enabled=True, budget=budget)
    with pytest.raises(k.BudgetExceeded):
        q.template((F(0),) * 5, enabled=True, budget=k.WorkBudget(enabled=True, limit=1))
    plan = q.template((F(0), F(1), F(-2), F(1), F(0)), enabled=True, budget=budget)
    sources, residuals = ((F(-1), F(1)),) * 5, ((F(0), F(0)),) * 3
    forged = dict(plan, L=F(0))
    with pytest.raises(ValueError):
        q.group_bounds(forged, sources, residuals, enabled=True, budget=budget)
    with pytest.raises(ValueError):
        q.group_bounds(plan, sources[:-1], residuals, enabled=True, budget=budget)
    with pytest.raises(k.KernelError):
        q.group_bounds(plan, ((F(1), F(-1)),) * 5, residuals, enabled=True, budget=budget)
