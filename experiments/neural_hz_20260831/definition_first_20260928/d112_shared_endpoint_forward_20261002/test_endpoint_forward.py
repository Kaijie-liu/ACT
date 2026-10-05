"""Four frozen mathematical controls; no benchmark ADV or new-domain claim.

The two production fixtures call the unchanged public fused Softmax-value
wrapper.  A includes common probability/value binding before either candidate
is added.  B adds the known two-token incremental bound and one shared value
product.  C is B intersected with D110; strict superiority over B is NOT assumed.
Float production coefficients are read literally as exact binary rationals.
"""
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf.tf_transformer import (
    _softmax_box, _softmax_ratio_inequalities, _sparse_simplex,
)
from act.back_end.solver.solver_hz import (
    SparseHZono, sparse_empty, sparse_hz_is_point, sparse_hz_lift_bounds,
    sparse_hz_linear, sparse_hz_softmax_value_relaxation,
)
from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef


RUN = Path(__file__).resolve().parents[2] / 'results/d112_shared_endpoint_forward_20261002_v1'


def _eval(form, values):
    return form.bias + sum((coef * values[i] for i, coef in form.terms), F(0))


def _literal(value):
    return F.from_float(float(value))


def _bounds(lo, hi):
    return Bounds(torch.tensor(lo, dtype=torch.float64),
                  torch.tensor(hi, dtype=torch.float64))


def _row(matrix, row, bias=0.0):
    begin, end = matrix.indptr[row:row+2]
    return ef.Form(_literal(bias), tuple(
        (int(matrix.indices[k]), _literal(matrix.data[k]))
        for k in range(begin, end) if matrix.data[k] != 0.0))


def _readouts(hz):
    assert hz.n_bin == 0
    return tuple(_row(hz.Gc, i, hz.c[i]) for i in range(hz.n_out))


def _system(parts):
    first = parts[0]
    eq, le = [], []
    for hz in parts:
        assert hz.frame_id == first.frame_id and hz.n_cont == first.n_cont
        assert hz.n_bin == 0
        eq.extend(_row(hz.Ac, i, -hz.b[i]) for i in range(hz.n_eq))
        le.extend(_row(hz.Auc, i, -hz.ub[i]) for i in range(hz.n_ineq))
    return ef.System(((F(-1), F(1)),) * first.n_cont, (), tuple(eq),
                     tuple(le), first.frame_id)


def _production_fixture(dynamic):
    """One shared frame, two actual nonpoint value banks, no model invocation."""
    eps = F(1, 32)
    width, frame = 8, 11201 + int(dynamic)
    source = SparseHZono(
        c=np.array([0.0, 1.5 if dynamic else 0.0]),
        Gc=sp.csr_matrix(([0.5, 0.5 if dynamic else 1.0],
                         ([0, 1], [0, 1])), shape=(2, width)),
        Gb=sparse_empty(2, 0), Ac=sparse_empty(0, width),
        Ab=sparse_empty(0, 0), b=np.zeros(0), frame_id=frame)
    scores = sparse_hz_linear(source, sp.csr_matrix(
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        np.array([float(eps), 0.0, 0.0, 0.0]))
    score_bounds = _bounds([-.5 + float(eps), 0.0, -.5, 0.0],
                           [.5 + float(eps), 0.0, .5, 0.0])
    probability_bounds = _softmax_box(score_bounds, 2)
    simplex, simplex_rhs = _sparse_simplex(2, 4)
    ratio, ratio_rhs = _softmax_ratio_inequalities(score_bounds, 2)
    probabilities = sparse_hz_lift_bounds(
        scores, probability_bounds, (2, 3, 4, 5), width,
        output_equalities=simplex, equality_rhs=simplex_rhs,
        output_inequalities=ratio, inequality_rhs=ratio_rhs)
    if dynamic:
        values = sparse_hz_linear(source, sp.csr_matrix([[0., 1.], [0., 0.]]))
        value_bounds = _bounds([1., 0.], [2., 0.])
    else:
        values = sparse_hz_linear(source, sp.csr_matrix([[0., .125], [0., .125]]),
                                  np.array([1., 0.]))
        value_bounds = _bounds([.875, -.125], [1.125, .125])
    assert not sparse_hz_is_point(values)
    fused = sparse_hz_softmax_value_relaxation(
        probabilities, values, probability_bounds, value_bounds,
        np.array([[0, 1], [2, 3]], dtype=np.int64),
        np.array([[0, 1], [0, 1]], dtype=np.int64), (6, 7), width,
        scores=scores, score_bounds=score_bounds)
    assert fused.frame_id == frame and fused.n_cont == width
    base = _system((source, scores, probabilities, values, fused))
    score_forms, prob_forms = _readouts(scores), _readouts(probabilities)
    value_forms, output_forms = _readouts(values), _readouts(fused)
    pb_lo = probability_bounds.lb.detach().cpu().numpy().reshape(-1)
    pb_hi = probability_bounds.ub.detach().cpu().numpy().reshape(-1)
    prob_box = tuple((_literal(lo), _literal(hi)) for lo, hi in zip(pb_lo, pb_hi))
    payload = dict(s=score_forms[:2], t=score_forms[2:],
                   p=prob_forms[:2], q=prob_forms[2:],
                   values=tuple((v,) for v in value_forms),
                   yp=output_forms[:1], yq=output_forms[1:],
                   prob_bounds=(prob_box[:2], prob_box[2:]))
    A = ef.append_value_readout(base, payload['p'], payload['values'], payload['yp'])
    A = ef.append_value_readout(A, payload['q'], payload['values'], payload['yq'])
    assert A.eq[:len(base.eq)] == base.eq and A.le[:len(base.le)] == base.le
    return A, payload, dict(
        wrapper='sparse_hz_softmax_value_relaxation',
        value_bank='(v,0), v in [1,2]' if dynamic else '(1+w/8,w/8), w in [-1,1]',
        same_frame=frame, source_columns=2, original_columns=width,
        probability_equalities=probabilities.n_eq,
        probability_inequalities=probabilities.n_ineq,
        fused_equalities=fused.n_eq, fused_inequalities=fused.n_ineq,
        baseline_equalities=len(A.eq), baseline_inequalities=len(A.le),
        value_is_point=False, original_model_executed=False)


def _maximum(system, form):
    count = len(system.bounds)
    objective = np.zeros(count)
    for i, coef in form.terms:
        objective[i] = -float(coef)

    def matrix(rows):
        out = np.zeros((len(rows), count))
        for k, row in enumerate(rows):
            for i, coef in row.terms:
                out[k, i] = float(coef)
        return out

    result = linprog(objective,
        A_ub=matrix(system.le) if system.le else None,
        b_ub=np.array([-float(row.bias) for row in system.le]) if system.le else None,
        A_eq=matrix(system.eq) if system.eq else None,
        b_eq=np.array([-float(row.bias) for row in system.eq]) if system.eq else None,
        bounds=[(float(lo), float(hi)) for lo, hi in system.bounds], method='highs')
    record = dict(success=bool(result.success), status=int(result.status),
                  message=str(result.message), columns=count,
                  equalities=len(system.eq), inequalities=len(system.le))
    if result.success:
        point = tuple(F.from_float(float(v)) for v in result.x)
        record.update(upper=float(form.bias)-float(result.fun),
            witness=[float(v) for v in result.x],
            max_equality_residual=max((abs(float(_eval(row, point))) for row in system.eq), default=0.0),
            max_inequality_residual=max((float(_eval(row, point)) for row in system.le), default=0.0))
    return record


def _control(dynamic, filename):
    record = dict(scope='direct production wrapper mathematical fixture',
                  formal_gain=0, native_binding_qualified=False,
                  complete_physical_qualification=False,
                  actual_model_qualified=False, stronger_than_known_increment=False)
    try:
        A, payload, metadata = _production_fixture(dynamic)
        record['production'] = metadata
        delta = payload['yp'][0] - payload['yq'][0]
        z = payload['p'][0] - payload['q'][0]
        increment = F(1, 128)
        B = replace(A, le=A.le + (-z, z-increment))
        B, shared = ef.product(B, z, payload['values'][0][0]-payload['values'][1][0],
                               x_bounds=(F(0), increment))
        B = replace(B, eq=B.eq + (delta-shared,))
        endpoint, receipt_A = ef.append_endpoint(
            A, **payload, frames=(A.frame,)*7, enabled=True)
        C, receipt_B = ef.append_endpoint(
            B, **payload, frames=(B.frame,)*7, enabled=True)
        for name, system in (('A_fused_and_shared_binding', A),
                             ('B_known_increment_and_shared_value', B),
                             ('C_B_plus_D110', C), ('D110_only', endpoint)):
            record[name] = _maximum(system, delta)
        record['endpoint_receipts'] = {'A': receipt_A, 'B': receipt_B}
        record['known_increment_upper'] = float(F(1, 64) if dynamic else increment)
        for key in ('A_fused_and_shared_binding', 'B_known_increment_and_shared_value',
                    'C_B_plus_D110', 'D110_only'):
            answer = record[key]
            assert answer['success'], answer
            assert answer['max_equality_residual'] <= 1e-7
            assert answer['max_inequality_residual'] <= 1e-7
        old = record['A_fused_and_shared_binding']['upper']
        known = record['B_known_increment_and_shared_value']['upper']
        new = record['D110_only']['upper']
        combined = record['C_B_plus_D110']['upper']
        record['strict_gain_over_A'] = old-new
        record['increment_over_known_B'] = known-combined
        record['analytic_child_threshold'] = .01
        record['analytic_child_upper_D110'] = max(0.0, new-.01)
        record['child_relu_executed'] = False
        assert new < old - 1/2000
        assert new <= float(F(1, 48) if dynamic else F(1, 96)) + 1e-7
        assert known <= record['known_increment_upper'] + 1e-7
        assert combined <= known + 1e-7
        assert C.binary == A.binary and endpoint.binary == A.binary
        assert C.eq[:len(B.eq)] == B.eq and C.le[:len(B.le)] == B.le
        record['assertions_passed'] = True
    except BaseException as error:
        record['assertions_passed'] = False
        record['failure'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        # The one-shot supervisor owns this directory. Never create or append
        # to an old run; success and failure controls use distinct x-only files.
        with (RUN / filename).open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')


def test_endpoint_contract_exact_rows():
    x, bit, yp, yq = (ef.Form(F(0), ((i, F(1)),)) for i in range(4))
    original = ef.System(((F(-1), F(1)), (F(-1), F(1)),
                          (F(-2), F(2)), (F(-2), F(2))),
                         (1,), (yp-yq,), (x-1,), 11200)
    args = dict(s=(x+F(1, 4), x+F(1, 4)), t=(x, x),
                p=(ef.Form(F(1, 2)),)*2, q=(ef.Form(F(1, 2)),)*2,
                values=((x+1,), (x-1,)), yp=(yp,), yq=(yq,),
                prob_bounds=(((F(1, 2), F(1, 2)),)*2,)*2)
    extended, receipt = ef.append_endpoint(original, **args,
                                           frames=(original.frame,)*7, enabled=True)
    assert receipt['enabled'] and receipt['old_columns'] == 4
    assert len(receipt['lambda_indices']) == 2
    assert extended.bounds[:4] == original.bounds and extended.binary == (1,)
    assert extended.eq[:len(original.eq)] == original.eq
    assert extended.le[:len(original.le)] == original.le
    for source in (F(-1), F(-1, 3), F(0), F(2, 5), F(1)):
        for signed_bit in (F(-1), F(1)):
            point = [F(0)] * len(extended.bounds)
            point[:4] = [source, signed_bit, source, source]
            point[receipt['c_index']] = F(1, 4)
            for i in receipt['lambda_indices']:
                point[i] = F(1, 2)
            assert all(lo <= value <= hi for value, (lo, hi) in zip(point, extended.bounds))
            assert all(_eval(row, point) == 0 for row in extended.eq)
            assert all(_eval(row, point) <= 0 for row in extended.le)
    z = ef.Form(F(0), ((2, F(1)),))
    rows = ef.mc_rows(z, x, bit, F(-1), F(1), F(0), F(2))
    for a in (F(-1), F(-1, 3), F(0), F(1)):
        for b in (F(0), F(3, 5), F(2)):
            assert all(_eval(row, (a, b, a*b)) <= 0 for row in rows)


def test_endpoint_fail_closed():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('disabled inspected ' + name)

    poison = Poison()
    same, receipt = ef.append_endpoint(poison, poison, poison, poison, poison,
        poison, poison, poison, poison, frames=poison, enabled=False, max_entries=poison)
    assert same is poison and receipt == {'enabled': False}
    system = ef.System((), (), (), (), 11203)
    half, zero = ef.Form(F(1, 2)), ef.Form()
    args = dict(s=(zero, zero), t=(zero, zero), p=(half, half), q=(half, half),
                values=((zero,), (zero,)), yp=(zero,), yq=(zero,),
                prob_bounds=(((F(1, 2), F(1, 2)),)*2,)*2)
    for flag in (0, 1, None, 'yes'):
        with pytest.raises(ValueError):
            ef.append_endpoint(system, **args, frames=(system.frame,)*7, enabled=flag)
    with pytest.raises(ValueError):
        ef.append_endpoint(system, **args, frames=(system.frame,)*6+(system.frame+1,), enabled=True)
    with pytest.raises(ValueError):
        ef.append_endpoint(system, **args, frames=(system.frame,)*7, enabled=True, max_entries=1)
    with pytest.raises(ValueError):
        ef.append_endpoint(system, **dict(args, prob_bounds=(((F(0), F(1)),)*2,)*2),
                           frames=(system.frame,)*7, enabled=True)
    with pytest.raises(ValueError):
        ef.append_endpoint(system, **dict(args, s=(zero,)), frames=(system.frame,)*7, enabled=True)
    with pytest.raises(ValueError):
        ef.Form(F(1 << 512))
    with pytest.raises(ValueError):
        ef.Form(F(1 << 511)) * F(2)
    assert system == ef.System((), (), (), (), 11203)


def test_endpoint_production_forward_control():
    _control(False, 'production_forward_control.json')


def test_endpoint_shared_value_control():
    _control(True, 'shared_value_control.json')
