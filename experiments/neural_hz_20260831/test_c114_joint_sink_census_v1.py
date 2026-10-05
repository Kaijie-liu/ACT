# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinary shared/disjoint sink graphs; exact independent projection guards."""
from copy import deepcopy
from fractions import Fraction as F
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c114_joint_sink_census_v1 import analyse


def fixture(shared=True):
    base=10
    aux=[dict(slot=10+i,rhs=0.,gauge=0,coefficients=((10+i,1.),(i,-1.))) for i in range(8)]
    aux += [dict(slot=18,rhs=0.,gauge=0,coefficients=((18,1.),*((10+i,-.25) for i in range(4)))),
            dict(slot=19,rhs=0.,gauge=0,coefficients=((19,1.),*((10+i+(0 if shared else 4),-.25*(-1)**i) for i in range(4))))]
    outputs=[dict(slot=8+i,rhs=0.,gauge=0,coefficients=((8+i,1.),(18,-.5),(19,-.5*(-1)**i))) for i in range(2)]
    return aux,outputs,base


def run(aux=None,outputs=None,base=None,**kw):
    if aux is None:aux,outputs,base=fixture()
    return analyse(aux,outputs,base,pool=WorkPool(4_000_000),enabled=True,**kw)


def project(aux,outputs,base):
    definitions={r['slot']:r for r in aux}
    def expand(col):
        if col<base:return {col:F(1)}
        row=definitions[col];entries=dict(row['coefficients']);pivot=F(entries.pop(col))
        result={-1:F(row['rhs'])/pivot}
        for parent,value in entries.items():
            for key,term in expand(parent).items():result[key]=result.get(key,F(0))-F(value)*term/pivot
        return {k:v for k,v in result.items() if v}
    result=[]
    for row in outputs:
        poly={-1:-F(row['rhs'])}
        for col,value in row['coefficients']:
            for key,term in expand(col).items():poly[key]=poly.get(key,F(0))+F(value)*term
        result.append({k:v/F(2)**row['gauge'] for k,v in poly.items() if v})
    return result


def test_default_off():
    pool=WorkPool(0)
    assert analyse(None,None,None,pool=pool) is None and pool.used==0


def test_joint_exact_gain_and_independent_original_projection():
    aux,out,base=fixture();report,_=run(aux,out,base)
    assert report['eligible_sinks']==2 and report['groups']==1
    assert report['individually_positive_groups']==1
    assert all(c['no_collision_single_nnz_delta']==1 for c in report['factors'])
    group=report['group_certificates'][0]
    assert group['nnz_delta']==-10 and group['new_local_nnz']==6
    assert project(aux,out,base)==project(aux[:-2],group['projected_native_rows'],base)


def test_disjoint_support_lower_bound_closes_without_full_expansion():
    aux,out,base=fixture(False);pool=WorkPool(4_000_000)
    report,_=analyse(aux,out,base,pool=pool,enabled=True)
    assert report['group_reasons']=={'necessary_nnz_lower_bound':1}
    assert report['group_certificates'][0]['new_nnz_lower']==18
    assert 'c114_complete_exact_expansion' not in pool.parts


def test_input_immutable():
    aux,out,base=fixture();before=deepcopy((aux,out));run(aux,out,base)
    assert (aux,out)==before


def test_nonredundant_box_rejected():
    aux,out,base=fixture();aux[-1]['rhs']=1.
    with pytest.raises(ValueError,match='box'):run(aux,out,base)


def test_nontriangular_parent_rejected():
    aux,out,base=fixture();aux[-2]['coefficients']=((18,1.),(19,-.25))
    with pytest.raises(ValueError,match='triangular'):run(aux,out,base)


def test_native_window_rejected():
    aux,out,base=fixture();out[0]['coefficients']=((8,1.),(18,2.**-30),(19,-.5))
    with pytest.raises(ValueError,match='window'):run(aux,out,base)


def test_alias_full_identity_checked():
    aux,out,base=fixture()
    with pytest.raises(ValueError,match='identity'):run(aux,out,base,aliases=[dict(old=11,representative=10,scale=1.)])


def test_complete_root_alias_substitution():
    aux,out,base=fixture();aux[1]['coefficients']=((11,1.),(0,1.))
    report,_=run(aux,out,base,aliases=[dict(old=11,representative=10,scale=-1.)])
    assert report['independently_checked_prior_root_aliases']==1
    assert report['effective_auxiliary_rows']==9


def test_complete_consumer_sets_not_partial_overlap():
    aux,out,base=fixture();out[1]['coefficients']=((9,1.),(18,-.5))
    report,_=run(aux,out,base)
    assert report['groups']==2 and report['group_size_histogram']=={1:2}


def test_rhs_exact_inverse_projection():
    aux,out,base=fixture()
    for r in aux[-2:]:
        r['coefficients']=tuple((c,v if c==r['slot'] else v/2) for c,v in r['coefficients'])
        r['rhs']=.25
    report,_=run(aux,out,base)
    group=report['group_certificates'][0]
    assert group['strict_nnz_reduction_proved']
    assert project(aux,out,base)==project(aux[:-2],group['projected_native_rows'],base)


def test_resource_fail_closed_before_work():
    aux,out,base=fixture()
    with pytest.raises(MemoryError):analyse(aux,out,base,pool=WorkPool(0),enabled=True)
