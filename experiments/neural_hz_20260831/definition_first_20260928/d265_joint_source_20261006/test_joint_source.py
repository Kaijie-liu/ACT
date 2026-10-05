"""Four preregistered tests, including complete small source-graph fixtures.

The Fraction forward evaluator is test-only: it checks actual graph points,
not candidate-generated bounds, and is never imported by the source worker.
This is not a benchmark or an original-network verification result.
"""
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as accounting
from experiments.neural_hz_20260831.definition_first_20260928.d261_mask_matched_reference_20261006 import source_bounds as sb
from experiments.neural_hz_20260831.definition_first_20260928.d265_joint_source_20261006 import source_arithmetic as ar
from experiments.neural_hz_20260831.definition_first_20260928.d265_joint_source_20261006 import jp_certificate as jp
from experiments.neural_hz_20260831.definition_first_20260928.d265_joint_source_20261006 import joint_pair


RUN = Path(__file__).resolve().parents[2] / 'results/d265_joint_source_20261006_v1'
_NAMES = ('grouped_source_and_bias', 'fixed_gram_and_residual',
          'jp_certificate_controls', 'default_off_budget_and_summary')
_EVIDENCE = {}
_BUDGET = None


class _Meter:
    def __init__(self, *, max_work=256_000_000, max_entries=64_000_000):
        self.owner = accounting.Budget(max_work=max_work, max_entries=max_entries)
        self.meter = self.owner._branch()
        self.max_bits = self.owner.max_bits

    def charge(self, amount, entries=0):
        self.meter.charge(work=amount, entries=entries)


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = _Meter()
    return _BUDGET


def _record(number, **values):
    name = _NAMES[number-1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    assert name == 'summary.json'
    assert Path(os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN']) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _iv(lo, hi=None):
    return ar.I(np.asarray(lo, dtype=np.float64),
                np.asarray(lo if hi is None else hi, dtype=np.float64))


def _f(value):
    return F.from_float(float(value))


def _fixture(projected):
    """Complete small A -> parent -> residual -> child source, two geometries."""
    meter = _budget()
    channels, size = 3, 3
    astride, asize = (2, 5) if projected else (1, 3)
    states = {'A': dict(kind='relu', input='first_pre', node=0,
        shape=(1, channels, asize, asize), bounds=_iv((0.,)*channels, (1.,)*channels))}

    def conv(port, source, weight, bias, stride=1, pad=1):
        incoming = states[source]
        k = weight.shape[2]
        side = (incoming['shape'][2]+2*pad-k)//stride+1
        ledger = sb.conv_channel(weight, bias, incoming['bounds'],
                                 (pad,)*4, meter, enabled=True)
        states[port] = dict(kind='conv', node=len(states), input=source,
            shape=(1, channels, side, side), weights=weight,
            weight_shape=tuple(weight.shape), strides=(stride,)*2,
            pads=(pad,)*4, dilations=(1, 1), group=1, ledger=ledger, bounds=ledger.bounds)

    def bn(port, source, bias):
        incoming = states[source]
        carrier = sb.bn_carrier(incoming['bounds'], _iv((1.,)*channels),
            _iv((float(bias),)*channels), budget=meter, enabled=True)
        states[port] = dict(kind='batchnorm', node=len(states), input=source,
            shape=incoming['shape'], carrier=carrier, bounds=carrier.bounds)

    def relu(port, source):
        incoming = states[source]
        states[port] = dict(kind='relu', node=len(states), input=source,
            shape=incoming['shape'], bounds=sb.relu(incoming['bounds'], budget=meter, enabled=True))

    parent = np.zeros((channels, channels, 3, 3), dtype=np.float64)
    middle, outer = np.zeros_like(parent), np.zeros_like(parent)
    for i in range(channels):
        parent[i, i, 1, 1] = 1.
        parent[i, (i+1) % channels, 1, 1] = -1.
        parent[i, i, 0, 1] = 0.125
        middle[i, i, 1, 1] = 1.
        middle[i, (i+1) % channels, 1, 2] = 0.25
        outer[i, i, 1, 1] = 1.
        outer[i, (i+1) % channels, 1, 1] = -0.25
        outer[i, i, 0, 1] = -0.125
    conv('P', 'A', parent, np.zeros(channels), astride)
    bn('PB', 'P', -0.0625)
    relu('Q', 'PB')
    conv('M', 'Q', middle, np.full(channels, -0.125))
    bn('MB', 'M', 0.0625)
    skip = 'A'
    if projected:
        weight = np.eye(channels, dtype=np.float64).reshape(channels, channels, 1, 1)
        conv('S', 'A', weight, np.full(channels, 0.0625), 2, 0)
        bn('SB', 'S', -0.125)
        skip = 'SB'
    states['ADD'] = dict(kind='add', node=len(states), inputs=('MB', skip),
        shape=states['MB']['shape'],
        bounds=sb.add(states['MB']['bounds'], states[skip]['bounds'], budget=meter, enabled=True))
    conv('O', 'ADD', outer, np.full(channels, -0.25))
    bn('OB', 'O', 0.125)
    relu('Y', 'OB')
    phases = [dict(node=0, preactivation='first_pre', output='A', shape=states['A']['shape']),
              dict(node=3, preactivation='PB', output='Q', shape=states['Q']['shape']),
              dict(node=states['Y']['node'], preactivation='OB', output='Y', shape=states['Y']['shape'])]
    return states, phases


def _forward_fraction(states, variant):
    """Exact evaluation of every original scalar, including its own BN error."""
    _, channels, height, width = states['A']['shape']
    values = {'A': {(c, y, x): F((c+2*y+x+variant) % 3, 2)
                    for c in range(channels) for y in range(height) for x in range(width)}}
    for port, state in states.items():
        if port == 'A':
            continue
        _, out_channels, height, width = state['shape']
        result = {}
        for c in range(out_channels):
            for y in range(height):
                for x in range(width):
                    key = c, y, x
                    if state['kind'] == 'conv':
                        w = state['weights']
                        source = values[state['input']]
                        value = _f(state['ledger'].bias[c])
                        for ci, ky, kx in product(range(w.shape[1]), range(w.shape[2]), range(w.shape[3])):
                            sy = y*state['strides'][0]+ky-state['pads'][0]
                            sx = x*state['strides'][1]+kx-state['pads'][1]
                            value += _f(w[c, ci, ky, kx])*source.get((ci, sy, sx), F(0))
                    elif state['kind'] == 'batchnorm':
                        carrier = state['carrier']
                        eps = (-1)**(c+y+x+variant)
                        value = (_f(carrier.nominal_a[c])*values[state['input']][key]
                                 + _f(carrier.nominal_b[c])+eps*_f(carrier.error[c]))
                    elif state['kind'] == 'relu':
                        value = max(F(0), values[state['input']][key])
                    else:
                        value = sum((values[name][key] for name in state['inputs']), F(0))
                    result[key] = value
        values[port] = result
    return values


def test_01_grouped_source_and_bias():
    meter = _budget()
    # Negative BN scale and two independent, same-direction source terms.
    crest = _iv((-.25,), (.5,))
    t = ar.add(ar.scale(crest, -1.5, meter), _iv((.25,)), meter)
    t = ar.add(t, _iv((-.0625,), (.0625,)), meter)
    rc, ec, ra, ea = ar.normalize_group(_iv((.5,)), _iv((-.25,)), t, meter)
    result = ar.support(ar.statistics(ra, ea, meter), rc, ec, meter)
    lo, hi = map(_f, result['r_bounds'])
    mu = F(*result['mu'])
    for cc, epsilon in product((F(-1, 4), F(1, 2)), (-1, 1)):
        actual_t = F(-3, 2)*cc+F(1, 4)+F(1, 16)*epsilon
        r, e = actual_t/2, -actual_t/4
        assert lo <= r <= hi
        assert abs(r-mu)+2*max(e, 0) <= _f(result['upper'])
    # The production fused weighted path, including interval coefficients,
    # both credit signs and nonzero source centers, gets its own exact oracle.
    uncertain_m = _iv((.25, -.5), (.5, -.25))
    uncertain_h = _iv((.125, .125), (.25, .25))
    source = _iv((1., -2.), (3., 0.))
    cr, ce, grouped = ar.group_stats(uncertain_m, uncertain_h, source, meter)
    combined = ar.support(grouped, cr, ce, meter)
    endpoint_cases = 0
    for m1, m2, h1, h2, t1, t2 in product((F(1, 4), F(1, 2)),
            (F(-1, 2), F(-1, 4)), (F(1, 8), F(1, 4)),
            (F(1, 8), F(1, 4)), (F(1), F(3)), (F(-2), F(0))):
        rr, ee = m1*t1+m2*t2, h1*t1+h2*t2
        assert abs(rr-F(*combined['mu']))+2*max(ee, 0) <= _f(combined['upper'])
        endpoint_cases += 1
    assert endpoint_cases == 64 and grouped[2] > 0 and grouped[3] > 0
    fixtures = []
    for projected in (False, True):
        states, phases = _fixture(projected)
        audit = joint_pair.audit(states, phases, sb, meter, enabled=True)
        assert audit['counts']['registered'] == 18
        assert len(audit['boundary_classes']) == 9 and len(audit['records']) == 18
        assert sum(audit['counts'][key] for key in ('ineligible', 'excluded', 'not_excluded')) == 18
        eligible = [row for row in audit['records'] if row['status'] != 'ineligible']
        assert eligible, 'complete small ordinary source must exercise the registered path'
        by_id = {cls['class_id']: cls for cls in audit['boundary_classes']}
        for variant in (0, 1):
            values = _forward_fraction(states, variant)
            for row in eligible:
                y, x = by_id[row['class_id']]['representative']
                i, j = row['pair']
                order = row['parent_order']
                scales = row['parent_scales']
                xs = tuple(values['PB'][(p, y, x)]/_f(s) for p, s in zip(order, scales))
                qs = tuple(values['Q'][(p, y, x)]/_f(s) for p, s in zip(order, scales))
                ys = (values['Y'][(i, y, x)], values['Y'][(j, y, x)])
                certificate = row['certificate']
                lhs = sum((F(*w)*v for w, v in zip(certificate['row_coefficients'], xs+qs+ys)), F(0))
                assert lhs <= F(*certificate['row_rhs'])
        fixtures.append(dict(projected=projected, records=18, eligible=len(eligible),
                             exact_graph_assignments=2, bn_error_terms_evaluated=True))
    _record(1, negative_scale_and_bias_retained=True, same_direction_group_enclosed=True,
            weighted_interval_endpoint_cases=endpoint_cases,
            complete_source_fixtures=fixtures, true_point_checks_are_not_strict_gain=True)


def test_02_fixed_gram_and_residual():
    meter = _budget()
    p1, p2 = _iv((1., 0., .125, 0.)), _iv((0., 1., 0., .25))
    z = _iv((.5, -.25, 0., 0.))
    chosen = ar.gram(p1, p2, z, meter)
    assert chosen is not None
    exact = (F(32, 65), F(-4, 17))
    assert all(abs(_f(a)-b) < F(1, 10**10) for a, b in zip(chosen, exact))
    residual = ar.sub(ar.sub(z, ar.scale(p1, chosen[0], meter), meter),
                      ar.scale(p2, chosen[1], meter), meter)
    erow = _iv((.125, .125, 0., 0.))
    result = ar.support(ar.statistics(residual, erow, meter), _iv(0.), _iv(0.), meter)
    for signs in product((-1, 1), repeat=4):
        p, q, zz, ee = [sum((_f(v)*s for v, s in zip(row.lo, signs)), F(0))
                        for row in (p1, p2, z, erow)]
        rr = zz-_f(chosen[0])*p-_f(chosen[1])*q
        assert _f(result['r_bounds'][0]) <= rr <= _f(result['r_bounds'][1])
        assert abs(rr-F(*result['mu']))+2*max(ee, 0) <= _f(result['upper'])
    assert ar.gram(p1, p1, z, meter) is None
    reduced = ar.sum_axis(_iv(((1., -.5), (2., .25))), 0, meter)
    assert all(_f(lo) <= wanted <= _f(hi) for lo, hi, wanted in
               zip(reduced.lo, reduced.hi, (F(3), F(-1, 4))))
    _record(2, exact_gram_solution=list(map(str, exact)), fixed_choice=chosen,
            original_parent_error_axes_retained=True, complete_residual_paid=True,
            singular_gram_has_no_second_choice=True, mask_specific_optimality_claimed=False)


def test_03_jp_certificate_controls():
    meter = _budget()
    parameters = ((.875, .875), (.0625, .0625), 1., (-.09375, .09375), .09375, .1875, (3., 3.))
    before = meter.owner.work, meter.owner.entries
    result = jp.certify(*parameters, budget=meter, enabled=True)
    assert (meter.owner.work-before[0], meter.owner.entries-before[1]) == (952, 369)
    assert F(*result['row_rhs']) == F(79, 16)
    assert result['payment_credit'] and not result['redundant']
    for x1, x2, s, t in product((F(-1), F(-1, 2), F(0), F(1, 2), F(1)),
                                (F(-1), F(-1, 2), F(0), F(1, 2), F(1)), (-1, 1), (-1, 1)):
        q1, q2 = max(x1, 0), max(x2, 0)
        r, e = F(3, 64)*(s-t), F(3, 128)*(s+t)
        z = F(7, 8)*(x1+x2)+F(1, 16)*(q1+q2)+r
        d = q1-q2+e
        point = (x1, x2, q1, q2, max(z+d, 0), max(z-d, 0))
        assert sum((F(*w)*v for w, v in zip(result['row_coefficients'], point)), F(0)) <= F(*result['row_rhs'])
    negative = jp.certify((.875, .875), (.0625, .0625), 1., (-2., 2.),
        3., 5., (93./16., 93./16.), budget=meter, enabled=True)
    assert negative['payment_credit'] and negative['redundant']
    assert F(*negative['support_witness']['rhs_minus_support']) == F(1, 16)
    _record(3, positive_control_rhs='79/16', true_graph_points_checked=100,
            broad_residual_credit_is_redundant=True, redundant_margin='1/16',
            fixed_certificate_work=952, fixed_certificate_entries=369,
            not_excluded_is_not_a_capability_claim=True)


def test_04_default_off_budget_and_summary():
    meter = _budget()
    before = meter.owner.work, meter.owner.entries
    assert joint_pair.audit(None, None, None, meter) is None
    assert jp.certify(None, None, None, None, None, None, None, budget=meter) is None
    assert (meter.owner.work, meter.owner.entries) == before
    arguments = ((.875, .875), (.0625, .0625), 1., (-.09375, .09375), .09375, .1875, (3., 3.))
    for limited in (_Meter(max_work=16), _Meter(max_entries=8)):
        with pytest.raises(accounting.Rejected):
            jp.certify(*arguments, budget=limited, enabled=True)
        assert limited.owner.failed
        with pytest.raises(accounting.Rejected):
            jp.certify(*arguments, budget=limited, enabled=True)
    with pytest.raises(ar.Rejected):
        ar.statistics(_iv((float('nan'),)), _iv((0.,)), _Meter())
    with pytest.raises(ar.Rejected):
        ar.div(_iv(1.), _iv(-1., 1.), _Meter())
    assert tuple(_EVIDENCE) == _NAMES[:-1] and not meter.owner.failed
    _record(4, default_off_no_work=True, shared_budget=True, resource_failure_sticky=True,
            finite_checked=True, solver_model_gpu_calls=0)
    _record_file('summary.json', dict(schema='d265_joint_source_v1', tests=4,
        local_joint_source_math_completed=True, joint_source_math_passed=False,
        inherited_mathematical_population=4285, required_tests=4289, required_test_files=232,
        records=_EVIDENCE, whole_work_used=meter.owner.work, numeric_entries=meter.owner.entries,
        source_audit_stage_registered=True, source_component_qualified=False,
        actual_model_binding_qualified=False, native_HZ_admitted=False,
        gpu_computation_completed=False, complete_physical_qualification=False,
        new_domain_qualified=False, new_capability_qualified=False,
        formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        baseline_solved=1870, independent_solved=61))
