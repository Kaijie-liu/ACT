"""Eight preregistered rational controls, never a floating-point oracle."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import attention as att
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ei
from experiments.neural_hz_20260831.definition_first_20260928.d130_import_isolation_20261002 import fast_query as fq


def _points(*points):
    return tuple((F(s), F(v)) for s, v in points)


def _overlap(first, second):
    assert max(first[0], second[0]) <= min(first[1], second[1])


def test_fast_positive_geometry():
    budget = ei.Budget()
    # A full-dimensional polygon whose exact maximum is strictly inside
    # its upper edge, not among the vertices: exp(s)*(1-s), maximized at 0.
    vertices = _points((-1, 2), (1, 0), (-1, 1), (1, -1))
    prepared = fq.prepare_polygon(vertices, budget)
    result = fq.threshold(prepared, F(0), F(0), budget)
    assert prepared.upper == _points((-1, 2), (1, 0))
    assert result['branch'] == 'stationary'
    assert result['maximizer'] == (F(0), F(1))
    assert result['h_interval'] == (F(1), F(1))
    assert result['exact_polygon_support'] is True
    assert result['rectangle_interval'][0] > 5
    # A strict kink at the highest vertex must also survive tail selection.
    kink = fq.prepare_polygon(_points((-2, 1), (0, 2), (2, -4),
                                      (-2, -5), (2, -5)), budget)
    answer = fq.threshold(kink, F(0), F(0), budget)
    assert answer['branch'] == 'vertex' and answer['maximizer'] == (F(0), F(2))
    assert answer['h_interval'] == (F(2), F(2))
    # More than one tail edge: the binary search must not stop at the first.
    chain = _points((0, 10), (1, 9), (2, 7), (3, 4), (4, 0), (5, -5))
    multi = fq.prepare_polygon(chain, budget)
    answer = fq.threshold(multi, F(2), F(5), budget)
    assert answer['maximizer'] == (F(8, 3), F(5))
    assert answer['branch'] == 'stationary'
    exact, _ = att._maximum_interval(chain, F(2), F(5), budget)
    _overlap(answer['h_interval'], exact)
    assert answer['h_interval'][1] < answer['rectangle_interval'][0]


def test_fast_strong_product_reference():
    budget = ei.Budget()
    system = ef.System(((F(-1), F(1)), (F(-1), F(1))), (), (), (), 12801)
    x, y = ef.Form(F(0), ((0, F(1)),)), ef.Form(F(0), ((1, F(1)),))
    value = F(3, 4)-x+y*F(1, 4)
    vertices = att._polygon(system, x, value, (0, 1), budget)
    tokens = (fq.prepare_polygon(_points((0, 0)), budget),
              fq.prepare_polygon(vertices, budget))
    t = F(5673, 10000)
    rows = tuple(fq.threshold(token, t, F(1), budget) for token in tokens)
    assert sum(row['h_interval'][1] for row in rows) < 0
    assert sum(row['rectangle_interval'][0] for row in rows) > 0
    # The complete D127 test population separately rechecks the scalar
    # McCormick/simplex/energy feasibility.  Here its SAME true probability
    # graph, not an arbitrary rational p, still gives Tv above the threshold.
    elo, ehi = fq.exp16_bounds(F(1, 2), budget)
    p = F(1)/(1+ehi), F(1)/(1+elo)
    assert F(37754, 100000) < p[0] <= p[1] < F(37755, 100000)
    tv = tuple(q+F(1, 4)-4*(q-F(1, 2))**2 for q in p)
    assert tv[0] > t and tv[1] >= tv[0]
    term = total = F(1)
    for k in range(1, 6):
        term = term*t/k
        total += term
    assert t*total > 1  # The true optimum Omega satisfies Omega*exp(Omega)=1.
    assert rows[0]['exact_polygon_support'] is True
    assert rows[1]['exact_polygon_support'] is True
    for row in rows:
        assert row['native_binding_qualified'] is False
        assert row['model_binding_qualified'] is False
        assert row['adv_witness_returned'] is False
    # The earlier threshold-1/2 control also remains certified.
    simpler = att._polygon(system, x, -x+y*F(1, 4), (0, 1), budget)
    token = fq.prepare_polygon(simpler, budget)
    first = fq.threshold(tokens[0], F(1, 2), F(1), budget)
    second = fq.threshold(token, F(1, 2), F(1), budget)
    assert first['h_interval'][1]+second['h_interval'][1] < 0
    assert system.frame == 12801 and system.binary == ()


def test_fast_negative_outer_loss():
    budget = ei.Budget()
    vertices = _points((0, -1), (1, F(-1, 10)))
    prepared = fq.prepare_polygon(vertices, budget)
    result = fq.threshold(prepared, F(0), F(0), budget)
    exact, _ = att._maximum_interval(vertices, F(0), F(0), budget)
    assert result['branch'] == 'negative_rectangle'
    assert result['exact_polygon_support'] is False
    assert result['h_interval'] == (F(-1, 10), F(-1, 10))
    assert result['rectangle_interval'] == result['h_interval']
    # The exact maximum is -e/10, strictly below the chosen -1/10 bound.
    assert exact[1] < result['h_interval'][0]
    assert exact[0] > F(-3, 10) and exact[1] < F(-2, 10)
    zero = fq.threshold(prepared, prepared.value_max, F(0), budget)
    assert zero['h_interval'] == (F(0), F(0))
    assert zero['exact_polygon_support'] is True
    # Concave upper chain with TWO negative-objective local maxima.  This
    # is a direct counterexample to using the positive unimodality rule here.
    chain = _points((0, -1), (1, F(-1, 3)), (2, F(-1, 7)),
                    (3, F(-1, 27)), (4, F(-1, 64)))
    multi = fq.prepare_polygon(chain, budget)
    assert multi.upper == chain
    for index in (1, 3):
        left = chain[index][1]+multi.slopes[index-1]
        right = chain[index][1]+multi.slopes[index]
        assert left > 0 > right
    assert fq.threshold(multi, F(0), F(4), budget)['branch'] == 'negative_rectangle'


def test_fast_degenerate_shift():
    budget = ei.Budget()
    point = fq.prepare_polygon(_points((2, 3)), budget)
    assert fq.threshold(point, F(3), F(2), budget)['h_interval'] == (F(0), F(0))
    vertical = fq.prepare_polygon(_points((2, -3), (2, 4), (2, 1)), budget)
    assert vertical.upper == _points((2, 4))
    assert vertical.value_min == -3 and vertical.value_max == 4
    assert fq.threshold(vertical, F(3), F(2), budget)['h_interval'] == (F(1), F(1))
    assert fq.threshold(vertical, F(5), F(2), budget)['exact_polygon_support'] is True
    flat = fq.prepare_polygon(_points((0, 1), (1, 1), (3, 1)), budget)
    assert len(flat.upper) == 2
    assert fq.threshold(flat, F(0), F(3), budget)['maximizer'] == (F(3), F(1))
    first = fq.prepare_polygon(_points((-1, 2), (1, 0)), budget)
    shifted = fq.prepare_polygon(_points((6, 2), (8, 0)), budget)
    a = fq.threshold(first, F(1, 4), F(1), budget)
    b = fq.threshold(shifted, F(1, 4), F(8), budget)
    assert a['h_interval'] == b['h_interval']
    assert a['rectangle_interval'] == b['rectangle_interval']
    old_work = budget.work
    fq.threshold(first, F(1, 4), F(1), budget)
    assert budget.work > old_work


def test_fast_bisection_uncertainty():
    budget = ei.Budget()
    exact_tokens = (fq.prepare_polygon(_points((0, 0)), budget),
                    fq.prepare_polygon(_points((0, 2)), budget))
    result = fq.bound(exact_tokens, budget)
    assert result['lo'] == result['hi'] == 1
    assert result['precision_certified'] is True
    assert result['rectangle_lo'] == result['rectangle_hi'] == 1
    # Equal exp(-1) terms cancel in real arithmetic, but outward interval
    # addition cannot certify the midpoint sign.  Keeping hi is mandatory.
    uncertain_tokens = tuple(fq.prepare_polygon(_points(point), budget)
                             for point in ((0, -1), (0, 1), (1, 0)))
    uncertain = fq.bound(uncertain_tokens, budget)
    assert uncertain['uncertain'] is True
    assert uncertain['precision_certified'] is False
    assert uncertain['steps_completed'] == 0
    assert uncertain['lo'] == -1 and uncertain['hi'] == 1
    assert uncertain['last_h_interval'][0] < 0 < uncertain['last_h_interval'][1]
    assert uncertain['quantity'] == 'root_of_sum_H_not_actual_attention_lower'
    assert uncertain['formal_gain'] == 0 and uncertain['adv_witness_returned'] is False
    # Non-degenerate certified bisection, preserving the matched reference.
    line = fq.prepare_polygon(_points((-1, 1), (1, -1)), budget)
    zero = fq.prepare_polygon(_points((0, 0)), budget)
    refined = fq.bound((zero, line), budget)
    assert refined['hi'] <= refined['rectangle_hi']
    assert refined['hi'] < F(1, 2)
    if refined['precision_certified']:
        initial = refined['initial_interval']
        assert refined['hi']-refined['lo'] <= (initial[1]-initial[0])/4096


def test_fast_fail_closed():
    budget = ei.Budget()
    prepared = fq.prepare_polygon(_points((0, 0), (1, 1)), budget)
    with pytest.raises(ValueError):
        fq.threshold(replace(prepared, value_max=F(100)), F(0), F(1), budget)
    with pytest.raises(ValueError):
        fq.threshold(replace(prepared), F(0), F(1), budget)
    with pytest.raises(ValueError):
        fq.threshold(prepared, F(0), F(1), ei.Budget())
    for bad in ([], (), ((F(0),),), ((0, F(1)),), ((F(0), F(1 << 512)),)):
        with pytest.raises(ValueError):
            fq.prepare_polygon(bad, budget)
    for bad in (0, True, 0.5, None):
        with pytest.raises(ValueError):
            fq.threshold(prepared, bad, F(1), budget)
    with pytest.raises(ValueError):
        fq.threshold(prepared, F(0), F(100), budget)
    for bad in (0, -1, 13, True, F(1)):
        with pytest.raises(ValueError):
            fq.bound((prepared,), budget, steps=bad)
    with pytest.raises(ValueError):
        fq.prepare_polygon(_points((0, 0)), ei.Budget(max_work=1))
    with pytest.raises(ValueError):
        fq.bound([], budget)
    # Frozen dataclass bypasses must not make altered public metadata trusted.
    object.__setattr__(prepared, 'value_max', F(100))
    with pytest.raises(ValueError):
        fq.threshold(prepared, F(0), F(1), budget)


def test_fast_exponential16_certificate():
    budget = ei.Budget()
    assert fq.exp16_bounds(F(0), budget) == (F(1), F(1))
    for x, low, high in ((F(1), F(2718, 1000), F(2719, 1000)),
                         (F(1, 2), F(16487, 10000), F(16488, 10000)),
                         (F(-1), F(3678, 10000), F(3679, 10000)),
                         (F(8), F(2980), F(2982)),
                         (F(64), F(2**64), F(3**64))):
        interval = fq.exp16_bounds(x, budget)
        assert low < interval[0] <= interval[1] < high
        _overlap(interval, ei.exp_bounds(x, budget=budget))
        inverse = fq.exp16_bounds(-x, budget)
        if x > 0:
            assert F(0) < inverse[0] <= 1/interval[1] <= 1/interval[0] <= inverse[1]
        else:
            assert F(0) < interval[0] <= 1/inverse[1] <= 1/inverse[0] <= interval[1]
    for x in (F(-64), F(-1, 3), F(1, 7), F(64)):
        interval = fq.exp16_bounds(x, budget)
        bits = 168 if x < 0 else 72
        for endpoint in interval:
            assert endpoint.denominator & (endpoint.denominator-1) == 0
            assert endpoint*(1 << bits) == int(endpoint*(1 << bits))
            assert max(endpoint.numerator.bit_length(), endpoint.denominator.bit_length()) <= 512
        _overlap(interval, ei.exp_bounds(x, budget=budget))
    # The truncated positive sum is NOT itself an upper bound.  The omitted
    # term and geometric tail must raise the certified upper above sum_0^16.
    term = partial = F(1)
    for k in range(1, 17):
        term = term*F(1, 2)/k
        partial += term
    assert fq.exp16_bounds(F(1, 2), budget)[1] > partial
    assert fq.TERMS == 16 and ei.TERMS == 64


def test_fast_exponential16_fail_closed():
    budget = ei.Budget()
    for bad in (0, 1, True, 0.5, float('inf'), float('nan'), None,
                F(65), F(-65), F(1 << 512), F(1, 1 << 512)):
        with pytest.raises(ValueError):
            fq.exp16_bounds(bad, budget)
    with pytest.raises(ValueError):
        fq.exp16_bounds(F(1), object())
    with pytest.raises(ValueError):
        fq.exp16_bounds(F(1), ei.Budget(max_work=1))
    altered = ei.Budget()
    altered.work = -1
    with pytest.raises(ValueError):
        fq.exp16_bounds(F(0), altered)
    altered.work = 0
    altered.max_work = ei.MAX_WORK+1
    with pytest.raises(ValueError):
        fq.exp16_bounds(F(0), altered)
