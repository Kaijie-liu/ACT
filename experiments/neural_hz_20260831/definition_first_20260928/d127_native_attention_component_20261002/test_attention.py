"""Six same-source attention controls, with exact rational reference algebra."""
from dataclasses import replace
from fractions import Fraction as F
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import attention as att
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ex


RESULTS = Path(__file__).resolve().parents[2] / 'results'


def _active_run():
    path = Path(os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'])
    if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
            or path.resolve() != path or not path.is_relative_to(RESULTS)
            or path == RESULTS):
        raise ValueError('untrusted active component evidence directory')
    return path


def _json_fraction(value):
    if type(value) is F:
        return f'{value.numerator}/{value.denominator}'
    raise TypeError('non-plain control evidence')


def _col(index):
    return ef.Form(F(0), ((index, F(1)),))


def _system(bounds=((F(-1), F(1)), (F(-1), F(1)))):
    return ef.System(bounds, (), (), (), 12701)


def _build(system, scores, values, **changes):
    options = dict(frames=(system.frame,)*len(scores), enabled=True)
    options.update(changes)
    return att.build(system, scores, values, **options)


def _query(fiber, threshold, direction=(F(1),), **changes):
    return att.certify(fiber, direction, threshold, **changes)


def _iv(value):
    return value, value


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _neg(a):
    return -a[1], -a[0]


def _sub(a, b):
    return _add(a, _neg(b))


def _scale(a, value):
    endpoints = a[0]*value, a[1]*value
    return min(endpoints), max(endpoints)


def _mul(a, b):
    corners = tuple(x*y for x in a for y in b)
    return min(corners), max(corners)


def _square(a):
    ends = a[0]*a[0], a[1]*a[1]
    return (F(0) if a[0] <= 0 <= a[1] else min(ends)), max(ends)


def _strict_mc(p, lower, upper, product, value, value_lower, value_upper):
    # Four McCormick residuals, all evaluated outward over the whole p/l/u
    # intervals.  This helper is only used where strict slack is available.
    rows = (
        _sub(_add(_scale(lower, value-value_lower),
                  _scale(p, value_lower)), product),
        _sub(_add(_scale(upper, value-value_upper),
                  _scale(p, value_upper)), product),
        _sub(product, _add(_scale(upper, value-value_lower),
                          _scale(p, value_lower))),
        _sub(product, _add(_scale(lower, value-value_upper),
                          _scale(p, value_upper))),
    )
    assert all(row[1] < 0 for row in rows)


def test_attention_joint_polygon():
    system = _system()
    x, y = _col(0), _col(1)
    fiber = _build(system, (ef.Form(), x),
                   ((ef.Form(),), (-x+y*F(1, 4),)))
    proof = _query(fiber, F(1, 2))
    assert proof['certified'] is True and proof['exact_product'] is True
    assert proof['quantity'] == 'shifted_product_box_F'
    assert proof['f_interval'][1] < 0
    vertices = proof['polygon_vertices']
    assert vertices[0] == ((F(0), F(0)),)
    expected = {(-F(1), F(3, 4)), (-F(1), F(5, 4)),
                (F(1), -F(5, 4)), (F(1), -F(3, 4))}
    assert len(vertices[1]) == 4 and set(vertices[1]) == expected
    assert len(proof['token_intervals']) == 2
    for field in ('source_metadata_is_model_binding', 'native_binding_qualified',
                  'actual_model_qualified', 'complete_physical_qualification',
                  'gpu_qualified', 'whole_work_qualified', 'adv_witness_returned'):
        assert proof[field] is False
    assert proof['original_system_retained'] is True
    assert proof['original_bits_deleted'] == 0
    # This is stronger than the independent score/value rectangle for this
    # example: its upper bound 5e/(4(1+e)) is above 1/2 because e>2/3.
    assert ex.exp_bounds(F(1))[0] > F(2, 3)
    assert fiber.system is system

    shifted_box = _system(((F(2), F(4)), (F(-3), F(-1))))
    score, value = x-3, -(x-3)+(y+2)*F(1, 4)
    translated = _build(shifted_box, (ef.Form(), score),
                        ((ef.Form(),), (value,)))
    assert set(_query(translated, F(1, 2))['polygon_vertices'][1]) == expected

    line = _build(system, (ef.Form(), x), ((ef.Form(),), (-x,)))
    line_proof = _query(line, F(1, 2))
    assert line_proof['certified'] is True
    assert set(line_proof['polygon_vertices'][1]) == {(-F(1), F(1)), (F(1), -F(1))}
    point_system = _system(((F(2), F(2)),))
    point = _build(point_system, (_col(0),), ((ef.Form(F(3)),),))
    point_proof = _query(point, F(3))
    assert point_proof['polygon_vertices'] == (((F(2), F(3)),),)
    assert point_proof['f_interval'] == (F(0), F(0))
    assert point_proof['certified'] is True


def test_attention_interior_stationary_control():
    system = _system()
    x, y = _col(0), _col(1)
    # On the upper edge y=1 and at threshold 1/4 the second term is
    # exp(x)*(1-x); its strict interior maximum is 1 at x=0.  Both endpoint
    # terms are <=2/e<3/4.  A corners-only scan therefore fails this test.
    fiber = _build(system, (ef.Form(), x),
                   ((ef.Form(F(1, 4)),), (1-x+y*F(1, 4),)))
    proof = _query(fiber, F(1, 4))
    unshifted = _mul(proof['f_interval'], ex.exp_bounds(proof['shift']))
    assert unshifted[0] > F(3, 4)
    assert unshifted[0] <= 1 <= unshifted[1]
    assert proof['candidate_points'] > sum(map(len, proof['polygon_vertices']))
    assert proof['certified'] is False

    strong = _build(system, (ef.Form(), x),
                    ((ef.Form(),), (F(3, 4)-x+y*F(1, 4),)))
    assert _query(strong, F(5673, 10000))['certified'] is True
    assert ex.exp_bounds(F(1, 2))[1] < 2
    unproved = _query(strong, F(1, 2))
    assert unproved['certified'] is False and unproved['f_interval'][0] > 0


def test_attention_strong_product_reference():
    # p is the exact sigmoid(-1/2), enclosed via the exponential certificate;
    # it is NOT replaced by an arbitrary rational probability witness.
    elo, ehi = ex.exp_bounds(F(1, 2))
    p = (F(1)/(1+ehi), F(1)/(1+elo))
    elo, ehi = ex.exp_bounds(F(1))
    lower = (F(1)/(1+ehi), F(1)/(1+elo))
    upper = _sub(_iv(F(1)), lower)
    complement = _sub(_iv(F(1)), p)
    assert F(37754, 100000) < p[0] <= p[1] < F(37755, 100000)
    assert lower[1] < p[0] and p[1] < upper[0]
    assert lower[1] < complement[0] and complement[1] < upper[0]
    tx = _add(_iv(F(-1, 4)), _scale(_square(_sub(p, _iv(F(1, 2)))), F(4)))
    tv = _sub(p, tx)
    tx_complement = _sub(_iv(F(-1, 2)), tx)
    tv_complement = _sub(_iv(F(3, 2)), tv)
    _strict_mc(p, lower, upper, tx, F(-1, 2), F(-1), F(1))
    _strict_mc(p, lower, upper, tv, F(3, 2), F(-1, 2), F(2))
    _strict_mc(complement, lower, upper, tx_complement, F(-1, 2), F(-1), F(1))
    _strict_mc(complement, lower, upper, tv_complement, F(3, 2), F(-1, 2), F(2))
    for product in (tx, tx_complement):
        assert -upper[0] < product[0] <= product[1] < upper[0]
    for product in (tv, tv_complement):
        assert -upper[0]/2 < product[0] <= product[1] < 2*upper[0]
    # y=1 is an interval endpoint: Ty=p and Ty_complement=1-p exactly.
    # Two product rows are equalities, and the other two reduce to the
    # already certified lower<=p<=upper (likewise for 1-p).  The definitions
    # of complemented products also establish all simplex product sums.
    energy = _add(tx, _iv(F(1, 4)))
    squared_norm = _add(_square(_sub(p, _iv(F(1, 2)))),
                        _square(_sub(complement, _iv(F(1, 2)))))
    assert energy == _scale(squared_norm, F(2))
    threshold = F(5673, 10000)
    assert tv[0] > threshold
    # Positive Taylor partial sum proves t*exp(t)>1, hence Omega<t.
    term = total = F(1)
    for k in range(1, 6):
        term = term*threshold/k
        total += term
    assert threshold*total > 1
    x, y = _col(0), _col(1)
    fiber = _build(_system(), (ef.Form(), x),
                   ((ef.Form(),), (F(3, 4)-x+y*F(1, 4),)))
    proof = _query(fiber, threshold)
    assert proof['certified'] is True and proof['f_interval'][1] < 0
    report = dict(
        schema='d127_native_attention_control_v1',
        status='control_assertions_passed',
        proof=proof, threshold=threshold,
        exact_sigmoid_probability_bracket=p,
        probability_range_lower_bracket=lower,
        probability_range_upper_bracket=upper,
        relaxed_tx_interval=tx, relaxed_tv_interval=tv,
        relaxed_source_point=(F(-1, 2), F(1)),
        energy_interval=energy,
        true_threshold_positive_taylor_certificate=threshold*total,
        comparison='exact_probability_graph_plus_specified_scalar_mccormick_and_energy',
        full_rlt_or_taylor_dominance=False, concrete_network_adv=False,
        actual_model_qualified=False, formal_gain=0)
    # Save this actual control receipt once, in the supervisor-selected new
    # RUN.  No fallback to a historical run and no overwrite are permitted.
    with (_active_run() / 'native_attention_control.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, default=_json_fraction, sort_keys=True, indent=2)
        stream.write('\n')


def test_attention_common_score_shift():
    system = _system()
    x, y = _col(0), _col(1)
    scores = (ef.Form(), x)
    # Two simultaneous outputs use the same scores and source.  Querying
    # opposite directions never produces two independent native fibers.
    values = ((ef.Form(), ef.Form()),
              (-x+y*F(1, 4), x-y*F(1, 4)))
    fiber = _build(system, scores, values)
    translated = _build(system, tuple(score+7 for score in scores), values)
    first = _query(fiber, F(1, 2), (F(1), F(0)))
    second = _query(translated, F(1, 2), (F(1), F(0)))
    assert first['certified'] is True and second['certified'] is True
    assert first['f_interval'] == second['f_interval']
    assert first['token_intervals'] == second['token_intervals']
    # The two components cancel before geometry is formed, because they
    # refer to the same token and latent source, not independent boxes.
    joint = _query(fiber, F(0), (F(1), F(1)))
    assert joint['certified'] is True and joint['f_interval'] == (F(0), F(0))
    assert fiber.scores is scores and fiber.values is values
    assert translated.values is values and translated.system is system
    assert second['shift']-first['shift'] == 7


def test_attention_shared_source_scope():
    system = _system()
    x, y = _col(0), _col(1)
    # Both tokens have the same score x; their values cancel identically.
    # The product-box F is positive, although the actual native output is 0.
    # It cannot be interpreted as a lower bound on the correlated maximum.
    shared = _build(system, (x, x), ((x,), (-x,)))
    proof = _query(shared, F(0))
    assert shared.exact_product is False and proof['exact_product'] is False
    assert proof['certified'] is False and proof['f_interval'][0] > 0
    assert proof['quantity'] == 'shifted_product_box_F'
    assert shared.system is system
    with pytest.raises(ValueError):
        _query(replace(shared, exact_product=True), F(0))

    independent = _build(system, (x, y), ((x,), (-y,)))
    assert independent.exact_product is True
    equality = (x-y,)
    inequality = (x-_col(2),)
    constrained = ef.System(system.bounds+((F(-1), F(1)),), (2,),
                            equality, inequality, system.frame)
    guarded = _build(constrained, (x, y), ((x,), (-y,)))
    guarded_proof = _query(guarded, F(2))
    assert guarded.exact_product is False and guarded_proof['exact_product'] is False
    assert guarded.system is constrained
    assert guarded.system.eq is equality and guarded.system.le is inequality
    assert guarded.system.binary == (2,)
    assert guarded_proof['certified'] is True


def test_attention_opt_in_identity_and_limits():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected '+name)

    poison = Poison()
    assert att.build(poison, poison, poison, frames=poison,
                     max_entries=poison) is None
    assert att.build(poison, poison, poison, frames=poison,
                     enabled=False, max_entries=poison) is None
    system = ef.System(((F(-1), F(1)), (F(-1), F(1))), (1,), (), (), 12702)
    x = _col(0)
    scores, values, frames = (ef.Form(), x), ((ef.Form(),), (-x,)), (12702, 12702)
    fiber = _build(system, scores, values, frames=frames)
    assert fiber.system is system and fiber.frames is frames
    assert fiber.scores is scores and fiber.values is values
    assert fiber.system.binary == (1,) and fiber.exact_product is True
    for forged in (replace(fiber, exact_product=False),
                   replace(fiber, used_columns=((), ())),
                   replace(fiber, frames=(12702, 12703)),
                   replace(fiber, scores=(ef.Form(), _col(1))),
                   replace(fiber, entries=fiber.entries-1),
                   replace(fiber, entry_upper=fiber.entry_upper-1),
                   replace(fiber, max_entries=True)):
        with pytest.raises(ValueError):
            _query(forged, F(1, 2))
    for enabled in (0, 1, None):
        with pytest.raises(ValueError):
            _build(system, scores, values, enabled=enabled)
    for cap in (0, 1, True, 64_000_001):
        with pytest.raises(ValueError):
            _build(system, scores, values, max_entries=cap)
    for wrong in ((12702,), (12702, 12703), (True, 12702)):
        with pytest.raises(ValueError):
            _build(system, scores, values, frames=wrong)
    with pytest.raises(ValueError):
        _build(system, (_col(1),), ((x,),))
    with pytest.raises(ValueError):
        _build(system, (_col(2),), ((x,),))
    with pytest.raises(ValueError):
        _build(system, scores, ((ef.Form(),), (x, x)))
    with pytest.raises(ValueError):
        _query(fiber, F(1, 2), (F(1), F(0)))
    for bad in (0.5, 1, True, F(1 << 512), F(1, 1 << 512)):
        with pytest.raises(ValueError):
            _query(fiber, bad)
    for wrong in ((1,), (0.5,), [F(1)]):
        with pytest.raises(ValueError):
            _query(fiber, F(1, 2), wrong)
    for cap in (0, 1, True, 256_000_001):
        with pytest.raises(ValueError):
            _query(fiber, F(1, 2), max_work=cap)
    assert fiber.entry_upper-1 > fiber.entries
    with pytest.raises(ValueError):
        _query(fiber, F(1, 2), max_work=fiber.entry_upper-1)
    completed = _query(fiber, F(1, 2))
    assert completed['query_entry_cap'] == min(fiber.max_entries, 256_000_000)
    assert completed['work'] <= 256_000_000
    assert completed['entries'] <= completed['entry_upper'] <= completed['query_entry_cap']
    # This finite source is valid for the mathematical fiber, but the
    # component's certified exponential range is insufficient for its span.
    wide = _build(_system(((F(-100), F(100)),)), (ef.Form(), _col(0)),
                  ((ef.Form(),), (_col(0),)))
    with pytest.raises(ValueError):
        _query(wide, F(1, 2))
