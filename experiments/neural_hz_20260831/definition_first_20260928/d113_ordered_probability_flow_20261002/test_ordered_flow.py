"""Four bounded ordered-flow controls, not trained-model or GPU admission.

The paper point establishes non-implication by the specified finite endpoint
relaxation, not by every valid IQC or the full Softmax graph. The production
fixture separately reports a pure-prefix baseline to avoid attributing known
stochastic-order inequalities to a new domain or to shared product allocation.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product as cartesian_product
import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.tf_transformer import (
    _softmax_box, _softmax_ratio_inequalities, _sparse_simplex,
)
from act.back_end.solver.solver_hz import (
    SparseHZono, sparse_empty, sparse_hz_is_point, sparse_hz_lift_bounds,
    sparse_hz_linear, sparse_hz_softmax_value_relaxation,
)
from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import test_endpoint_forward as prior_helpers
from experiments.neural_hz_20260831.definition_first_20260928.d113_ordered_probability_flow_20261002 import ordered_flow as flow


RESULTS = Path(__file__).resolve().parents[2] / 'results'
D = (F(-3, 4), F(-1, 4), F(1, 4), F(3, 4))


def _column(index):
    return ef.Form(F(0), ((index, F(1)),))


def _active_run():
    """Only the supervisor-selected existing isolated evidence directory."""
    path = Path(os.environ['NEURAL_HZ_ACTIVE_COMPONENT_RUN'])
    if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
            or path.resolve() != path or not path.is_relative_to(RESULTS)
            or path == RESULTS):
        raise ValueError('untrusted active component evidence directory')
    return path


def _production_fixture():
    width, frame = 15, 11304
    source = SparseHZono(
        c=np.array([0., 0., 0., 0., 1.5]),
        Gc=sp.csr_matrix((np.array([2.25, 2.25, 2.25, 2.25, .5]),
                         (np.arange(5), np.arange(5))), shape=(5, width)),
        Gb=sparse_empty(5, 0), Ac=sparse_empty(0, width),
        Ab=sparse_empty(0, 0), b=np.zeros(0), frame_id=frame)
    matrix = np.zeros((8, 5))
    for i in range(4):
        matrix[i, i] = matrix[4+i, i] = 1.0
    scores = sparse_hz_linear(source, sp.csr_matrix(matrix),
                              np.array([float(d) for d in D] + [0.]*4))
    score_bounds = prior_helpers._bounds(
        [-2.25+float(d) for d in D] + [-2.25]*4,
        [2.25+float(d) for d in D] + [2.25]*4)
    probability_bounds = _softmax_box(score_bounds, 4)
    simplex, simplex_rhs = _sparse_simplex(4, 8)
    ratio, ratio_rhs = _softmax_ratio_inequalities(score_bounds, 4)
    probabilities = sparse_hz_lift_bounds(
        scores, probability_bounds, tuple(range(5, 13)), width,
        output_equalities=simplex, equality_rhs=simplex_rhs,
        output_inequalities=ratio, inequality_rhs=ratio_rhs)
    value_matrix = np.zeros((4, 5))
    value_matrix[0, 4] = value_matrix[1, 4] = 1.0
    values = sparse_hz_linear(source, sp.csr_matrix(value_matrix))
    value_bounds = prior_helpers._bounds([1., 1., 0., 0.], [2., 2., 0., 0.])
    assert not sparse_hz_is_point(values)
    fused = sparse_hz_softmax_value_relaxation(
        probabilities, values, probability_bounds, value_bounds,
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
        (13, 14), width, scores=scores, score_bounds=score_bounds)
    base = prior_helpers._system((source, scores, probabilities, values, fused))
    score_forms = prior_helpers._readouts(scores)
    probability_forms = prior_helpers._readouts(probabilities)
    value_forms = prior_helpers._readouts(values)
    outputs = prior_helpers._readouts(fused)
    pb = tuple((prior_helpers._literal(lo), prior_helpers._literal(hi)) for lo, hi in zip(
        probability_bounds.lb.detach().cpu().numpy().reshape(-1),
        probability_bounds.ub.detach().cpu().numpy().reshape(-1)))
    args = dict(s=score_forms[:4], t=score_forms[4:],
                p=probability_forms[:4], q=probability_forms[4:],
                values=tuple((v,) for v in value_forms),
                yp=outputs[:1], yq=outputs[1:], prob_bounds=(pb[:4], pb[4:]))
    A = ef.append_value_readout(base, args['p'], args['values'], args['yp'])
    A = ef.append_value_readout(A, args['q'], args['values'], args['yq'])
    assert A.eq[:len(base.eq)] == base.eq and A.le[:len(base.le)] == base.le
    assert all(ef.box(A, s-t) == (d, d) for s, t, d in zip(args['s'], args['t'], D))
    return A, args, dict(
        frame=frame, tokens=4, output_channels=1, source_columns=5,
        original_columns=width, value_bank='(v,v,0,0), v in [1,2]',
        source_box='[-9/4,9/4]^4 x [1,2]', value_is_point=False,
        production_wrapper='sparse_hz_softmax_value_relaxation',
        probability_equalities=probabilities.n_eq,
        probability_inequalities=probabilities.n_ineq,
        fused_equalities=fused.n_eq, fused_inequalities=fused.n_ineq)


def _known_sector(system, args):
    """Fixed necessary tangent rows, explicitly not a complete IQC oracle."""
    z = tuple(a-b for a, b in zip(args['p'], args['q']))
    mu = min(lo for endpoint in args['prob_bounds'] for lo, _ in endpoint)
    L = F(1, 2)
    norm_d = sum((d*d for d in D), F(0))
    dot_d = sum((d*zi for d, zi in zip(D, z)), ef.Form())
    rows = []
    for point in cartesian_product((F(-1, 4), F(0), F(1, 4)), repeat=4):
        row = sum((2*v*zi for v, zi in zip(point, z)), ef.Form())
        row = row - sum((v*v for v in point), F(0)) - (L+mu)*dot_d + mu*L*norm_d
        rows.append(row)
    assert len(rows) == 81
    rows.append(-dot_d)
    cap = (max(D)-min(D))/4
    for zi in z:
        rows.extend((zi-cap, -zi-cap))
    return replace(system, le=system.le+tuple(rows)), mu


def test_flow_exact_extension():
    x, signed_bit, yp0, yp1, yq0, yq1 = tuple(_column(i) for i in range(6))
    original = ef.System(((F(-1), F(1)), (F(-1), F(1))) + ((F(-4), F(4)),)*4,
                         (1,), (yp0-yq0, yp1-yq1), (x-1,), 11300)
    probs = (ef.Form(F(1, 4)),)*4
    values = ((x+1, 1-x), (2*x-1, x), (x*F(1, 2), x+2), (3-x, -2*x))
    args = dict(s=(x,)*4, t=(x,)*4, p=probs, q=probs, values=values,
                yp=(yp0, yp1), yq=(yq0, yq1),
                prob_bounds=(((F(1, 4), F(1, 4)),)*4,)*2)
    extended, receipt = flow.append_ordered_flow(original, **args,
                                                frames=(original.frame,)*7, enabled=True)
    assert tuple(receipt['order']) == (0, 1, 2, 3)
    assert len(receipt['flow_indices']) == 3
    assert receipt['old_columns'] == 6
    assert extended.binary == (1,) and extended.bounds[:6] == original.bounds
    assert extended.eq[:len(original.eq)] == original.eq
    assert extended.le[:len(original.le)] == original.le
    for source in (F(-1), F(-2, 5), F(0), F(3, 7), F(1)):
        for label in (F(-1), F(1)):
            point = [F(0)] * len(extended.bounds)
            y0, y1 = F(5, 8)*source+F(3, 4), F(3, 4)-source/4
            point[:6] = [source, label, y0, y1, y0, y1]
            assert all(lo <= value <= hi for value, (lo, hi) in zip(point, extended.bounds))
            assert all(prior_helpers._eval(row, point) == 0 for row in extended.eq)
            assert all(prior_helpers._eval(row, point) <= 0 for row in extended.le)


def test_flow_rejects_uncertified_order():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected ' + name)

    poison = Poison()
    result, receipt = flow.append_ordered_flow(poison, poison, poison, poison, poison,
        poison, poison, poison, poison, frames=poison, enabled=False, max_entries=poison)
    assert result is poison and receipt == {'enabled': False}
    x = _column(0)
    system = ef.System(((F(-1), F(1)),), (), (), (), 11301)
    zero, half = ef.Form(), ef.Form(F(1, 2))
    args = dict(s=(x, -x), t=(zero, zero), p=(half, half), q=(half, half),
                values=((zero,), (zero,)), yp=(zero,), yq=(zero,),
                prob_bounds=(((F(1, 2), F(1, 2)),)*2,)*2)
    # Midpoints tie, but neither proposed whole-source ordering is certified.
    with pytest.raises(ValueError):
        flow.append_ordered_flow(system, **args, frames=(system.frame,)*7, enabled=True)
    valid = dict(args, s=(zero, zero))
    for flag in (0, 1, None):
        with pytest.raises(ValueError):
            flow.append_ordered_flow(system, **valid, frames=(system.frame,)*7, enabled=flag)
    with pytest.raises(ValueError):
        flow.append_ordered_flow(system, **valid, frames=(system.frame,)*6+(system.frame+1,), enabled=True)
    with pytest.raises(ValueError):
        flow.append_ordered_flow(system, **valid, frames=(system.frame,)*7, enabled=True, max_entries=1)
    assert system == ef.System(((F(-1), F(1)),), (), (), (), 11301)


def test_flow_not_implied_by_endpoint_relaxation():
    lo, hi = F(1, 512), F(511, 512)
    original = ef.System(((lo, hi),)*8+((F(-2), F(2)),)*2, (), (), (), 11302)
    p, q = tuple(_column(i) for i in range(4)), tuple(_column(i) for i in range(4, 8))
    yp, yq = _column(8), _column(9)
    args = dict(s=tuple(ef.Form(d) for d in D), t=(ef.Form(),)*4,
                p=p, q=q, values=((ef.Form(F(1)),),)*2+((ef.Form(),),)*2,
                yp=(yp,), yq=(yq,), prob_bounds=(((lo, hi),)*4,)*2)
    endpoint, er = ef.append_endpoint(original, **args, frames=(original.frame,)*7, enabled=True)
    p_values, q_values = tuple(F(v, 80) for v in (19, 22, 18, 21)), (F(1, 4),)*4
    point = [F(0)] * len(endpoint.bounds)
    point[:10] = list(p_values+q_values+(F(41, 80), F(1, 2)))
    point[er['c_index']] = F(0)
    for index in er['lambda_indices']:
        point[index] = F(1, 8)
    assert all(lo <= value <= hi for value, (lo, hi) in zip(point, endpoint.bounds))
    assert all(prior_helpers._eval(row, point) == 0 for row in endpoint.eq)
    assert all(prior_helpers._eval(row, point) <= 0 for row in endpoint.le)
    z = tuple(a-b for a, b in zip(p_values, q_values))
    dot = sum((a*b for a, b in zip(D, z)), F(0))
    norm = sum((zi*zi for zi in z), F(0))
    assert dot == F(1, 160) and norm == F(1, 640)
    S = F(1, 2)
    assert dot >= 2*norm and dot == (2/S)*norm
    assert all(abs(zi) <= (max(D)-min(D))/4 for zi in z)
    extended, fr = flow.append_ordered_flow(endpoint, **args,
                                            frames=(original.frame,)*7, enabled=True)
    assert extended.eq[:len(endpoint.eq)] == endpoint.eq
    assert extended.le[:len(endpoint.le)] == endpoint.le
    assert tuple(fr['order']) == (0, 1, 2, 3)
    flows = tuple(_column(i) for i in fr['flow_indices'])
    previous = ef.Form()
    for i, current in enumerate(flows):
        assert current-previous+p[i]-q[i] in extended.eq
        previous = current
    # Summing the first two retained EQs forces F_2=-1/80 for ANY aux values;
    # its declared nonnegative bound makes extension impossible, without LP.
    required_F2 = sum((q_values[i]-p_values[i] for i in range(2)), F(0))
    assert required_F2 == F(-1, 80)
    assert extended.bounds[fr['flow_indices'][1]][0] >= 0


def test_flow_production_strong_control():
    record = dict(scope='fixed production Softmax-PV component, not a trained model',
        formal_gain=0, native_binding_qualified=False, actual_model_qualified=False,
        complete_physical_qualification=False, gpu_qualified=False,
        complete_IQC_closure=False, child_relu_executed=False,
        known_grouped_value_reference_registered=True)
    try:
        A, args, metadata = _production_fixture()
        record['production'] = metadata
        B, endpoint_receipt = ef.append_endpoint(A, **args, frames=(A.frame,)*7, enabled=True)
        B, mu = _known_sector(B, args)
        z = tuple(p-q for p, q in zip(args['p'], args['q']))
        prefix_rows = tuple(sum(z[:k], ef.Form()) for k in range(1, 4))
        B_prefix = replace(B, le=B.le+prefix_rows)
        # Strong attribution control: ordinary shared-value factorization,
        # not a new probability-flow theorem. For this fixed V=(v,v,0,0),
        # DeltaY=(z_0+z_1)*v and the known prefix row gives z_0+z_1 <= 0.
        p_bounds, q_bounds = args['prob_bounds']
        group_lower = max(F(-1),
            sum((p_bounds[i][0]-q_bounds[i][1] for i in range(2)), F(0)),
            sum((q_bounds[i][0]-p_bounds[i][1] for i in range(2, 4)), F(0)))
        B_group, grouped_product = ef.product(B_prefix, sum(z[:2], ef.Form()),
            args['values'][0][0], (group_lower, F(0)))
        B_group = replace(B_group, eq=B_group.eq +
            (args['yp'][0]-args['yq'][0]-grouped_product,))
        C, flow_receipt = flow.append_ordered_flow(B, **args, frames=(B.frame,)*7, enabled=True)
        assert C.eq[:len(B.eq)] == B.eq and C.le[:len(B.le)] == B.le
        assert C.binary == B.binary and C.bounds[:len(B.bounds)] == B.bounds
        delta = args['yp'][0]-args['yq'][0]
        for name, system in (('A', A), ('B', B), ('B_prefix', B_prefix),
                             ('B_group', B_group), ('C', C)):
            record[name] = prior_helpers._maximum(system, delta)
        record['endpoint_receipt'] = endpoint_receipt
        record['flow_receipt'] = flow_receipt
        record['known_sector'] = dict(mu=[mu.numerator, mu.denominator],
            L=[1, 2], fixed_tangents=81, monotonicity_rows=1,
            coordinate_increment_rows=8, pure_prefix_rows=3)
        for name in ('A', 'B', 'B_prefix', 'B_group', 'C'):
            assert record[name]['success'], record[name]
            assert record[name]['max_equality_residual'] <= 1e-7
            assert record[name]['max_inequality_residual'] <= 1e-7
        upper_B, upper_prefix, upper_C = (record[name]['upper'] for name in ('B', 'B_prefix', 'C'))
        record['B_minus_C'] = upper_B-upper_C
        record['B_prefix_minus_C'] = upper_prefix-upper_C
        record['B_group_minus_C'] = record['B_group']['upper']-upper_C
        record['known_group_absorbs_reported_gain'] = record['B_group']['upper']-upper_C <= 1e-7
        record['pure_prefix_absorbs_nonpositive_bound'] = upper_prefix <= 1e-7
        record['pure_prefix_absorbs_reported_gain'] = upper_prefix-upper_C <= 1e-7
        assert upper_C <= 1e-7
        assert record['B_group']['upper'] <= 1e-7
        assert abs(record['B_group_minus_C']) <= 1e-7
        assert upper_B-upper_C > 1/2000
        record['assertions_passed'] = True
    except BaseException as error:
        record['assertions_passed'] = False
        record['failure'] = type(error).__name__+': '+str(error)
        raise
    finally:
        # The supervisor creates the unique run. Neither old D112 tests nor its
        # evidence-writing helper _control are called from this module.
        with (_active_run() / 'production_flow_control.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
