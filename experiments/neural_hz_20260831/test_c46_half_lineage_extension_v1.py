"""Exact algebra fixtures, not independently admitted benchmark SplicedStates."""
from fractions import Fraction as F
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import incidence_oracle
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import ReversibleLineage,encode_splice
from experiments.neural_hz_20260831.c38_all_consumer_coefficients_v1 import discover
from experiments.neural_hz_20260831.c39_half_alias_census_v1 import select_halves,audit_rows
from experiments.neural_hz_20260831.c40_compact_half_gauge_v1 import materialize
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import pack_descriptors
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c46_half_lineage_extension_v1 import _OldRows,extend
from experiments.neural_hz_20260831.c46_ordered_fixture_v1 import build


def fixture(consumer='eq',alias='unit'):
    # x,y,z=x/2,v=-y/2,w,u,t,a. Old C32 unit u's definition has
    # already been removed; its consumer may itself be deleted by C40.
    ac=np.zeros((4,8));ac[0,[0,2]]=[-.5,1.];ac[1,[1,3]]=[.5,1.]
    ac[2,[2,3,4]]=[.5,.25,.25];ac[3,[2,3,6]]=[.25,.125,.5]
    ab=np.zeros((4,1));ab[2,0]=.125
    auc=np.zeros((2,8));auc[0,[2,3,4,6]]=[.25,-.25,.25,.5];auc[1,[4,6]]=[.5,-.25]
    aub=np.array([[.125],[0.]])
    gc=np.zeros((2,8));gc[0,4]=1.;gc[1,6]=1.
    old=SparseHZono(np.zeros(2),sp.csr_matrix(gc),sp.csr_matrix((2,1)),sp.csr_matrix(ac),sp.csr_matrix(ab),
        np.array([0.,-0.,0.,.125]),sp.csr_matrix(auc),sp.csr_matrix(aub),np.array([.5,1.]),frame_id=321,exact=True)
    pool=WorkPool(256_000_000);tables=dict(definitions=np.array([0,1]),eq=np.arange(10,14),le=np.arange(20,22))
    state=SimpleNamespace(hz=old,original_fields=dict(old_n_cont=2,logical_n_cont=4,old_n_eq=0,eq_roots=np.array([0,1],np.int64)))
    counts=incidence_oracle(old,tables['eq'],tables['le'],2,4,pool=pool)
    cohort,_=discover(state,old,tables,counts,pool=pool);half,_=select_halves(cohort,pool=pool)
    proof,rows=audit_rows(old,tables,counts,half,old_nc=2,logical_nc=4,pool=pool,branch=pool)
    assert proof['all_joint_row_arithmetic_proved'] and len(half)==2
    new,_,desc,runs,_=materialize(old,old,half,rows,pool=pool,enabled=True)
    degrees=np.array([int(half[half['column']==t['column']]['degree'][0]) for t in desc])
    words=pack_descriptors(desc,pool=pool)
    # Hand-built, validated lineage PROTOCOL fixture only. The independent
    # equations below prove its unit/alias meaning; no source receipt is forged.
    row,le={'eq':(4,False),'deleted':(0,False),'le':(0,True),'plain':(1,True)}[consumer]
    roots=np.array([0,1,3,encode_splice(2,row,le,1.,1,0),4,-6 if alias=='unit' else -3],np.int64)
    scales=np.ones(6,np.int64);scales.view(np.float64)[3]=.125;scales.view(np.float64)[5]=.5 if alias=='unit' else -.5
    lineage=ReversibleLineage(roots,scales,np.array([5],np.int32),np.array([(33<<20)|77],np.uint64),
        np.zeros(0,np.uint64),2,0)
    lineage.seal=lineage.fingerprint();lineage.validate()
    return old,new,lineage,words,runs,degrees


def call(data,point,pool=None):
    old,new,lineage,words,runs,degrees=data
    return extend(new,lineage,words,runs,degrees,source_digest(old),lineage.seal,point,
        input_n_cont=2,pool=pool if pool is not None else WorkPool(256_000_000),enabled=True)


def residuals(hz,point,binary,inequality=False):
    m=hz.Auc if inequality else hz.Ac;b=hz.Aub if inequality else hz.Ab;rhs=hz.ub if inequality else hz.b
    result=[]
    for row in range(m.shape[0]):
        a,z=m.indptr[row:row+2];ba,bz=b.indptr[row:row+2]
        result.append(sum((F(float(v))*point[int(c)] for c,v in zip(m.indices[a:z],m.data[a:z])),F(0))+
            sum((F(float(v))*binary[int(c)] for c,v in zip(b.indices[ba:bz],b.data[ba:bz])),F(0))-F(float(rhs[row])))
    return result


def test_default_off_never_reads_inputs():
    assert extend(*([object()]*8),input_n_cont=object(),pool=object()) is None


def test_every_old_EQ_LE_continuous_binary_row_and_RHS_matches_original():
    old,new,_,words,runs,degrees=fixture();pool=WorkPool(256_000_000)
    oracle=_OldRows(new,words,runs,degrees,source_digest(old),pool=pool)
    for le in (False,True):
        for continuous in (False,True):
            matrix=getattr(old,('Au' if le else 'A')+('c' if continuous else 'b'))
            for row in range(matrix.shape[0]):
                a,b=matrix.indptr[row:row+2]
                want={int(c):F(float(v)) for c,v in zip(matrix.indices[a:b],matrix.data[a:b])}
                assert dict(oracle.terms(row,inequality=le,continuous=continuous))==want
                assert oracle.rhs(row,inequality=le)==F(float((old.ub if le else old.b)[row]))
    proof=oracle.finish();assert not proof['complete_old_HZ_materialized'] and proof['deleted_definition_reads']>0
    assert all(value is not old for value in vars(oracle).values())


@pytest.mark.parametrize('consumer',['eq','deleted','le','plain'])
@pytest.mark.parametrize('alias',['unit','half'])
def test_complete_half_unit_alias_composition_and_original_nonconvex_feasibility(consumer,alias):
    data=fixture(consumer,alias);old,new,lineage,_,_,_=data
    old_hash,new_hash=source_digest(old),source_digest(new)
    for x,y,b in product([F(-1,4),F(0),F(1,4)],[F(-1,4),F(0),F(1,4)],[F(-1),F(1)]):
        z=x/2;v=-y/2;w=-2*z-v-b/2;t=F(1,4)-z/2-v/4
        point=np.array([x,y,1.,-.75,w,0.,t,0.],np.float64)
        assert all(r==0 for r in residuals(new,list(map(lambda a:F(float(a)),point)),[b]))
        assert all(r<=0 for r in residuals(new,list(map(lambda a:F(float(a)),point)),[b],True))
        got,proof=call(data,point)
        restored=list(map(lambda a:F(float(a)),point));restored[2]=z;restored[3]=v
        reference=lineage.reconstruct_fraction(old,restored,pool=WorkPool(256_000_000))
        assert got==reference and got[:2]==[x,y] and all(abs(a)<=1 for a in got)
        assert all(r==0 for r in residuals(old,got,[b])) and all(r<=0 for r in residuals(old,got,[b],True))
        # Independently specified PRE-unit equations, not the adapter formula.
        prefix={'eq':F(1,4)*z+F(1,8)*v,'deleted':-x/2+z,
            'le':z/4-v/4+w/4,'plain':w/2}[consumer]
        assert prefix+got[5]==F(1,8)
        if consumer=='eq':assert -got[5]+t/2==0
        elif consumer=='deleted':assert -got[5]==-F(1,8)
        elif consumer=='le':assert -got[5]+t/2<=F(3,8)
        else:assert -got[5]-t/4<=F(7,8)
        assert got[7]==(got[5]/2 if alias=='unit' else -z/2)
        assert proof['half_factors_reconstructed']==2 and proof['old_unit_factors_reconstructed']==1
        assert proof['old_alias_factors_reconstructed']==1 and proof['binary_factors_never_rewritten']
        assert not proof['source_and_native_admission_proved'] and not proof['point_feasibility_or_concrete_network_validation_proved']
    assert (source_digest(old),source_digest(new))==(old_hash,new_hash)


@pytest.mark.parametrize('bad',['old_hash','new_value','degrees','run','word','lineage_hash','lineage_value',
    'shape','nan','box','budget','input','half_dependency'])
def test_corrupt_or_incomplete_composition_fails_closed(bad):
    data=fixture();old,new,lin,words,runs,degrees=data;original_sha=source_digest(old);lineage_sha=lin.seal
    point=np.zeros(new.n_cont,np.float64);pool=WorkPool(0 if bad=='budget' else 256_000_000);input_n=2
    if bad=='old_hash':original_sha='0'*64
    elif bad=='new_value':new.Ac.data[0]+=.125
    elif bad=='degrees':degrees[0]+=1
    elif bad=='run':runs=runs[:1]
    elif bad=='word':words=words.copy();words[1]|=np.uint64(1<<50)
    elif bad=='lineage_hash':lineage_sha='0'*64
    elif bad=='lineage_value':lin.eq_scales[0]+=1
    elif bad=='shape':point=point[:-1]
    elif bad=='nan':point[0]=np.nan
    elif bad=='box':point[0]=1.01
    elif bad=='input':input_n=3
    elif bad=='half_dependency':
        lin.eq_roots[0]=-1;lin.eq_scales.view(np.float64)[0]=.5
        lin.seal=lin.fingerprint();lineage_sha=lin.seal
    with pytest.raises((ValueError,MemoryError,IndexError)):
        extend(new,lin,words,runs,degrees,original_sha,lineage_sha,point,input_n_cont=input_n,pool=pool,enabled=True)


def test_changed_new_payload_after_row_read_cannot_finish():
    old,new,_,words,runs,degrees=fixture();pool=WorkPool(256_000_000)
    oracle=_OldRows(new,words,runs,degrees,source_digest(old),pool=pool)
    list(oracle.terms(3));new.b[0]+=.125
    with pytest.raises(ValueError,match='changed'):oracle.finish()


@pytest.mark.parametrize('count',[1,3])
@pytest.mark.parametrize('sign,alias_sign',[(-1,-1),(-1,1),(1,-1),(1,1)])
@pytest.mark.parametrize('unit_sign',[-1,1])
@pytest.mark.parametrize('pivot',[.5,1.,2.])
@pytest.mark.parametrize('offset',[-.125,0.,.125])
def test_ordered_unit_consumer_becomes_deleted_half_definition(count,sign,alias_sign,unit_sign,pivot,offset):
    pool=WorkPool(256_000_000);pre,old,new,lin,words,runs,degrees,writer=build(count,sign,alias_sign,pool=pool,
        unit_sign=unit_sign,pivot=pivot,offset=offset)
    for x in (F(-1,8),F(0),F(1,8)):
        point=np.zeros(new.n_cont,np.float64);point[:count]=float(x)
        for i in range(count):point[count+4*i+2]=-2*unit_sign*sign*float(x)
        binary=[F(-1 if i%2 else 1) for i in range(count)]
        assert all(v==0 for v in residuals(new,list(map(lambda v:F(float(v)),point)),binary))
        assert all(v<=0 for v in residuals(new,list(map(lambda v:F(float(v)),point)),binary,True))
        got,proof=extend(new,lin,words,runs,degrees,source_digest(old),lin.seal,point,
            input_n_cont=count,pool=pool,enabled=True)
        assert got[:count]==[x]*count and proof['deleted_definition_reads']==count
        assert all(v==0 for v in residuals(pre,got,binary)) and all(v<=0 for v in residuals(pre,got,binary,True))
        for i in range(count):
            u,z,r,a=(count+4*i+j for j in range(4))
            assert got[u]==sign*x/2+F(offset) and got[z]==unit_sign*(got[u]-F(offset)) and got[a]==alias_sign*got[u]/2
        assert writer['actual_predicate_nnz_delta']==-2*count
