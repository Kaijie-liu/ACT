"""Four frozen mathematical controls, never a model or phase-search worker."""
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as accounting
from experiments.neural_hz_20260831.definition_first_20260928.d266_joint_guard_capacity_20261006 import joint_guard as jg


RUN = Path(__file__).resolve().parents[2] / 'results/d266_joint_guard_capacity_20261006_v1'
_NAMES = ('guard_identity_and_valid_points', 'strict_open_family_and_consumer',
          'wide_residual_and_old_dominance', 'default_off_and_evidence')
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


def _f(v):
    return F.from_float(v)


def _relu(x):
    return max(F(0), x)


def _lhs(row, point):
    return sum((F(*w)*x for w, x in zip(row['row_coefficients'], point)), F(0))


def _mc(x, beta, lower, upper, w):
    assert lower <= x <= upper and 0 <= beta <= 1
    assert w >= lower*beta
    assert w <= upper*beta
    assert w >= x-upper*(1-beta)
    assert w <= x-lower*(1-beta)


def _graph_lp(x, q, beta, lower, upper):
    assert lower <= x <= upper and lower < 0 < upper
    assert 0 <= beta <= 1 and q >= 0 and q >= x
    assert q <= upper*beta
    assert q <= x-lower*(1-beta)


def _six_product_reference(a, b, rrange, erange, x, q, phases, r, y, values):
    x1, x2 = x
    q1, q2 = q
    al1, al2 = phases
    c12, c21, d12, d21, v1, v2 = values
    _mc(x2, al1, -1, 1, c12)
    _mc(x1, al2, -1, 1, c21)
    _mc(q2, al1, 0, 1, d12)
    _mc(q1, al2, 0, 1, d21)
    _mc(r, al1, *rrange, v1)
    _mc(r, al2, *rrange, v2)
    p1, p2 = al1-q1, al2-q2
    n1, n2 = 1-al1-q1+x1, 1-al2-q2+x2
    tt = q1-q2+c12-c21
    uu = q1+q2-c12-c21
    dabs = 2*q1-x1+2*q2-x2
    assert tt <= p2+n2 and -tt <= p1+n1 and uu <= dabs
    k = (al1-al2+a[0]*(q1-c21)+a[1]*(c12-q2)
         + b[0]*(q1-d21)+b[1]*(d12-q2)+v1-v2)
    lr, ur = rrange
    le, ue = erange
    l10 = lr+min(0,a[0]+b[0])-max(0,a[1])
    u10 = ur+max(0,a[0]+b[0])-min(0,a[1])
    l01 = lr-max(0,a[0])+min(0,a[1]+b[1])
    u01 = ur-min(0,a[0])+max(0,a[1]+b[1])
    upper = 2*p2+2*max(ue,0)+max(-l10-1,0)*al1+max(u01-1,0)*al2
    lower = 2*p1-2*min(le,0)+max(u10-1,0)*al1+max(-l01-1,0)*al2
    assert y[0]-y[1]-k <= upper
    assert -y[0]+y[1]+k <= lower
    return k, upper, lower


def test_01_guard_identity_and_valid_points():
    meter = _budget()
    cases = (((.875,.875),(.0625,.0625)), ((.5,-.25),(-.125,.25)),
             ((-.75,.5),(.25,-.5)), ((2.,-2.),(-.5,.125)))
    graph_count = guard_count = 0
    for (af,bf), tau in product(cases, (.5,1.)):
        row = jg.certify(af,bf,tau,(-.125,.375),.5,.75,.25,(12.,12.),
                         budget=meter,enabled=True)
        a,b,t = tuple(map(_f,af)),tuple(map(_f,bf)),_f(tau)
        mu,c = F(1,8),t+F(1,8)
        a10 = -min(0,a[0]+b[0])+max(0,a[1])-c
        a01 = -min(0,a[0])+max(0,a[1]+b[1])+mu-t
        assert F(*row['row_rhs']) <= F(*row['old_row_rhs'])
        for x1,x2,s,tt in product((F(-1),F(-1,2),F(0),F(1,2),F(1)),
                                  (F(-1),F(-1,2),F(0),F(1,2),F(1)),
                                  (-1,0,1),(-1,0,1)):
            q1,q2 = _relu(x1),_relu(x2)
            rho,e = F(1,8)*(s-tt),F(1,8)*(s+tt)
            r = mu+rho
            z = a[0]*x1+a[1]*x2+b[0]*q1+b[1]*q2+r
            d = t*(q1-q2)+e
            point = x1,x2,q1,q2,_relu(z+d),_relu(z-d)
            assert _lhs(row,point) <= F(*row['row_rhs'])
            graph_count += 1
            labels1 = (0,1) if x1 == 0 else (int(x1>0),)
            labels2 = (0,1) if x2 == 0 else (int(x2>0),)
            for al1,al2 in product(labels1,labels2):
                h = al1-al2
                clip = min(max(z+t,0),2*t)
                ee = h*rho+h*(clip-z-t)
                rhs = max(rho,a10) if h==1 else max(-rho,a01) if h==-1 else F(0)
                assert ee <= rhs
                assert ee+2*_relu(e) <= F(*row['Gstar'])
                guard_count += 1
    _record(1, exact_graph_points=graph_count, legal_phase_checks=guard_count,
            nonzero_residual_center=True, mixed_sign_parameters=True,
            finite_oracle_is_not_a_proof=True)


def test_02_strict_open_family_and_consumer():
    meter = _budget()
    a,b = _f(.85),_f(.06)
    x,q,ph = (F(0),F(0)),(F(99,200),F(99,200)),(F(1,2),F(1,2))
    y = F(16137,20000),F(11187,20000)
    point = x+q+y
    records = []
    for rr in (F(15,128),F(1,8),F(65,512)):
        row = jg.certify((.85,.85),(.06,.06),1.,(-float(rr),float(rr)),
                         float(rr),float(2*rr),float(rr/2),(4.,4.),
                         budget=meter,enabled=True)
        values = F(1,200),F(-1,200),F(99,200),F(0),rr/2,-rr/2
        _six_product_reference((a,a),(b,b),(-rr,rr),(-rr/2,rr/2),x,q,ph,F(0),y,values)
        for xx,qq,al in zip(x,q,ph):
            _graph_lp(xx,qq,al,-1,1)
        z = b*sum(q)
        gl,gu = -2*a+b-1-rr,2*a+b+1+rr
        for yy in y:
            _graph_lp(z,yy,F(1,2),gl,gu)
        lhs = _lhs(row,point)
        assert lhs <= F(*row['old_row_rhs']) and lhs > F(*row['row_rhs'])
        assert F(*row['Gstar']) == rr
        # Same physical row, fixed nonnegative combination with Dabs <= 2.
        s0 = F(*row['row_coefficients'][2])/2
        assert 1-F(2,5)*s0 >= 0
        bound = F(2,5)*F(*row['row_rhs'])+2*(1-F(2,5)*s0)
        assert bound == 2+F(2,5)*(b+rr) and bound < F(83,40)
        displayed = 2*sum(q)+F(2,5)*(y[0]-y[1])
        assert displayed == F(2079,1000)
        assert _relu(displayed-F(83,40)) == F(1,250)
        records.append(dict(R=str(rr),old_rhs=str(F(*row['old_row_rhs'])),
            new_rhs=str(F(*row['row_rhs'])),strict_gap=str(lhs-F(*row['row_rhs'])),
            consumer_upper=str(bound),stored_float_coefficients_checked_exactly=True))
    _record(2, cases=records, complete_six_product_MC_X_QG_and_old_JP=True,
            original_single_gate_LP=True, next_relu_certified_zero=True,
            displayed_old_point_is_not_ADV=True, full_source_RLT_compared=False)


def test_03_wide_residual_and_old_dominance():
    meter = _budget()
    row = jg.certify((.875,.875),(.0625,.0625),1.,(-2.,2.),3.,5.,1.5,
                     (93./16.,93./16.),budget=meter,enabled=True)
    assert F(*row['old_row_rhs']) == F(93,8)
    assert F(*row['row_rhs']) == F(125,16)
    assert F(*row['Gstar']) == 3 and not row['redundant']
    assert F(*row['support_witness']['upper']) == F(185,16)
    checks = 0
    for x1,x2,q3,q4,q5,s1,s2 in product((-1,0,1),(-1,0,1),
            (F(0),F(1,2),F(1)),(F(0),F(1,2),F(1)),(F(0),F(1,2),F(1)),
            (F(0),F(1,3),F(2,3)),(F(0),F(1,3),F(2,3))):
        q1,q2 = _relu(x1),_relu(x2)
        u,v = F(2,3)*(q3+q4)+s1-1,F(2,3)*(q4+q5)+s2-1
        r,e = F(3,2)*(u-v),F(3,4)*(u+v)
        z = F(7,8)*(x1+x2)+F(1,16)*(q1+q2)+r
        d = q1-q2+e
        assert F(-17,4) <= z+d <= F(71,16)
        assert F(-17,4) <= z-d <= F(71,16)
        assert _lhs(row,(x1,x2,q1,q2,_relu(z+d),_relu(z-d))) <= F(*row['row_rhs'])
        checks += 1
    # Same complete original source equalities, stronger ordinary child bounds.
    x,q,ph = (F(0),F(0)),(F(1,2),F(1,2)),(F(1,2),F(1,2))
    q3,q4,q5,s1,s2 = F(39,40),F(39,40),F(1,2),F(13,20),F(131,360)
    u,v = F(2,3)*(q3+q4)+s1-1,F(2,3)*(q4+q5)+s2-1
    r,e = F(3,2)*(u-v),F(3,4)*(u+v)
    assert (u,v,r,e) == (F(19,20),F(25,72),F(217,240),F(467,480))
    y = F(25,8),F(0)
    for xx,qq,al in zip(x,q,ph):
        _graph_lp(xx,qq,al,-1,1)
    z,d = F(1,16)+r,e
    assert (z+d,z-d) == (F(931,480),F(-1,160))
    _graph_lp(z+d,y[0],F(5,7),F(-17,4),F(71,16))
    _graph_lp(z-d,y[1],F(0),F(-17,4),F(71,16))
    kval,up,lo = _six_product_reference((F(7,8),)*2,(F(1,16),)*2,
        (F(-2),F(2)),(F(-3,2),F(3,2)),x,q,ph,r,y,
        (F(0),F(0),F(1,4),F(1,4),r/2,r/2))
    assert kval == 0 and up == F(157,32) and y[0] <= up
    lhs = _lhs(row,x+q+y)
    assert lhs == F(63,8) and lhs <= F(*row['old_row_rhs'])
    assert lhs-F(*row['row_rhs']) == F(1,16)
    assert F(*row['row_rhs']) < F(251,32)
    assert _relu(lhs-F(251,32)) == F(1,32)
    # Explicitly stronger comparison: this wide point does NOT beat D052.
    d052_bound = F(1537,480)-F(3,16)*F(5,7)
    assert y[0]-d052_bound == F(191,3360)
    # Old hard-guard regime is exactly recovered, rather than weakened.
    recovered = jg.certify((.5,.5),(0.,0.),1.,(-.125,.125),.125,.25,.0625,
                           (3.,3.),budget=meter,enabled=True)
    assert recovered['row_rhs'] == recovered['old_row_rhs']
    _record(3, wide_true_graph_checks=checks, wide_old_rhs='93/8',wide_new_rhs='125/16',
            complete_old_reference_accepts_wide_point=True,wide_strict_gap='1/16',
            wide_consumer_certified_zero=True,wide_old_consumer_value='1/32',
            stronger_D052_rejects_wide_point=True,D052_rejection_gap='191/3360',
            full_single_child_source_hull_compared=False,zero_eta_recovers_old_JP=True)


def test_04_default_off_and_evidence():
    meter = _budget()
    before = meter.owner.work,meter.owner.entries
    assert jg.certify(None,None,None,None,None,None,None,None,budget=meter) is None
    assert (meter.owner.work,meter.owner.entries) == before
    args = ((.875,.875),(.0625,.0625),1.,(-2.,2.),3.,5.,1.5,(6.,6.))
    for limited in (_Meter(max_work=16),_Meter(max_entries=8)):
        with pytest.raises(accounting.Rejected):
            jg.certify(*args,budget=limited,enabled=True)
        assert limited.owner.failed
        with pytest.raises(accounting.Rejected):
            jg.certify(*args,budget=limited,enabled=True)
    with pytest.raises(jg.Rejected):
        jg.certify(*args,budget=meter,enabled=1)
    bad = list(args)
    bad[6] = float('nan')
    with pytest.raises(jg.Rejected):
        jg.certify(*bad,budget=meter,enabled=True)
    bad[6] = 1e-300
    with pytest.raises(jg.Rejected):
        jg.certify(*bad,budget=meter,enabled=True)
    assert tuple(_EVIDENCE) == _NAMES[:-1] and not meter.owner.failed
    _record(4,default_off_no_work=True,resource_failure_sticky=True,
            nonfinite_and_oversized_rational_rejected=True,solver_model_gpu_calls=0)
    _record_file('summary.json',dict(schema='d266_joint_guard_capacity_v1',tests=4,
        local_joint_guard_math_completed=True,joint_guard_math_passed=False,
        inherited_mathematical_population=4289,required_tests=4293,required_test_files=233,
        records=_EVIDENCE,whole_work_used=meter.owner.work,numeric_entries=meter.owner.entries,
        source_audit_stage_registered=False,source_component_qualified=False,
        actual_model_binding_qualified=False,native_HZ_admitted=False,
        gpu_computation_completed=False,complete_physical_qualification=False,
        new_domain_qualified=False,new_capability_qualified=False,
        formal_gain=0,independent_e0_gain=0,new_benchmark_solves=0,
        baseline_solved=1870,independent_solved=61))
