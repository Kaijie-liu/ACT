"""Full nonconvex/inequality retention, physical rows, ownership and inverse."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import pytest
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX
from experiments.neural_hz_20260831.c22_uid_runs_v1 import pack
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c62_local_equations_v1 import encode,SCHEMA as LOCAL
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import make,audit,extend,recover,SCHEMA,row


def fixture(outputs=16):
    old=4;nc=5+outputs;neq=1+outputs
    ac=sp.lil_matrix((neq,nc));ab=sp.lil_matrix((neq,1));ac[0,0]=1.;ab[0,0]=-.25
    for i in range(outputs):
        ac[i+1,:4]=[-1.,-2.,-1.,-2.];ac[i+1,5+i]=8.
    gc=sp.csr_matrix(([1.],([0],[nc-1])),shape=(1,nc))
    auc=sp.csr_matrix(([1.,-1.],([0,1],[2,3])),shape=(2,nc))
    hz=SparseHZono(np.array([.375]),gc,sp.csr_matrix((1,1)),ac.tocsr(),ab.tocsr(),np.zeros(neq),
        auc,sp.csr_matrix((2,1)),np.array([.75,.5]),frame_id=123,exact=True)
    roots=np.r_[0,0,np.arange(1,neq)].astype(np.int64);scales=np.zeros(len(roots),np.int64)
    tag,num=encode(1,(1,-1));roots[1]=tag;scales[1]=np.array(num,np.float64).view(np.int64).item()
    fields=dict(hz=hz,old_n_cont=old,old_n_bin=1,old_n_eq=1,logical_n_cont=nc,
        expression=None,keep=np.ones(1,bool),eq_roots=roots,eq_scales=scales,
        ineq_roots=np.arange(2,dtype=np.int64),ineq_scales=np.zeros(2,np.int64),def_rows=np.empty(0,np.int64),
        radix_gauges=np.empty(0,np.int64),uid_slabs=np.array([pack(3,0,1+outputs)],np.uint64),owners=np.zeros(1+outputs,np.int64),
        report=dict(radix_uid_base=4+outputs,new_lineage_schema=LOCAL,actual_radix_work=0))
    eu,lu=closed_uid_tables(SimpleNamespace(**fields))
    for matrix,uids in ((hz.Ac,eu),(hz.Auc,lu)):
        for r,uid in enumerate(uids):
            for c in row(matrix,r)[0]:
                if old<=c<nc:fields['owners'][c-old]+=RADIX+int(uid)
    cols=[0,1,2,3,nc];vals=[-1.,-2.,-1.,-2.,8.];ptr=[0,5];pivots=[nc]
    for i in range(outputs):cols.extend([5+i,nc]);vals.extend([8.,-8.]);ptr.append(len(cols));pivots.append(5+i)
    packet=dict(indptr=np.array(ptr,np.int32),columns=np.array(cols,np.int32),native=np.array(vals,np.float64),
        rhs=np.zeros(1+outputs),pivots=np.array(pivots,np.int32),gauges=np.zeros(1+outputs,np.int32),
        ab_indptr=np.zeros(2+outputs,np.int32),new_factors=1)
    return fields,[packet]


def integer_owner_oracle(fields,state):
    h=state['fields']['hz'];eu,lu=closed_uid_tables(SimpleNamespace(**fields))
    eu=np.r_[eu,state['auxiliary_records'][:,1]];out=[0]*h.n_cont
    for matrix,uids in ((h.Ac,eu),(h.Auc,lu)):
        for r,uid in enumerate(uids):
            for c in row(matrix,r)[0]:out[c]+=RADIX+int(uid)
    return out


@pytest.mark.parametrize('outputs',[8,16,32])
@pytest.mark.parametrize('sign',[-1,1])
def test_full_nonconvex_rows_and_nonzero_bidirectional_inverse(outputs,sign):
    fields,packets=fixture(outputs);pool=WorkPool(256_000_000)
    state=make(fields,packets,b'original full source proof',b'circuit full proof',pool=pool,enabled=True)
    got=audit(fields,state,packets,pool=pool,enabled=True)
    assert got['old_scalar_equations_retained']==1 and got['original_binary_factors']==1
    assert got['whole_predicate_nnz_after']<got['whole_predicate_nnz_before']
    point=[F(sign,4),F(1,8),F(-1,8),F(1,4),F(1,16)]
    y=(point[0]+2*point[1]+point[2]+2*point[3])/8
    point.extend([y]*outputs);new=extend(state,point,pool=pool)
    assert recover(state,new,pool=pool)==point
    h=state['fields']['hz']
    for r in range(h.n_eq):
        value=sum((F(float(v))*new[int(c)] for c,v in zip(*row(h.Ac,r))),F(0))
        value+=sum((F(float(v))*sign for c,v in zip(*row(h.Ab,r))),F(0))
        assert value==F(float(h.b[r]))
    oracle=integer_owner_oracle(fields,state);old=fields['old_n_cont'];main=fields['logical_n_cont']
    assert state['fields']['owners'].tolist()==oracle[old:main]
    assert state['auxiliary_records'][:,5].tolist()==oracle[fields['hz'].n_cont:]
    assert state['fields']['expression'] is fields['expression']
    assert state['schema']==SCHEMA and not state['native_or_LIVE_admission']


@pytest.mark.parametrize('change',['owner','coefficient','block','binary','inverse'])
def test_actual_complete_state_corruption_rejects(change):
    fields,packets=fixture();pool=WorkPool(256_000_000)
    state=make(fields,packets,b'source',b'circuit',pool=pool,enabled=True)
    if change=='owner':state['auxiliary_records'][0,5]+=1
    if change=='coefficient':state['fields']['hz'].Ac.data[-1]*=2
    if change=='block':state['block_records'][0,2]+=1
    if change=='binary':
        state['fields']['hz'].Ab=state['fields']['hz'].Ab.copy();state['fields']['hz'].Ab.data[0]*=2
    if change=='inverse':state['auxiliary_records'][0,3]+=1
    with pytest.raises(ValueError):audit(fields,state,packets,pool=pool,enabled=True)


def test_no_source_or_packet_numeric_owner_is_retained_unnecessarily():
    fields,packets=fixture();pool=WorkPool(256_000_000)
    state=make(fields,packets,b'source',b'circuit',pool=pool,enabled=True)
    assert state['fields']['hz'] is not fields['hz']
    assert not np.shares_memory(state['fields']['hz'].Ac.data,fields['hz'].Ac.data)
    assert not np.shares_memory(state['fields']['hz'].Ac.data,packets[0]['native'])
    assert np.shares_memory(state['fields']['hz'].Gc.data,fields['hz'].Gc.data)
    assert np.shares_memory(state['fields']['hz'].Auc.data,fields['hz'].Auc.data)
    assert state['fields']['eq_roots'] is fields['eq_roots']


def test_default_off_and_global_budget():
    pool=WorkPool(0)
    assert make(None,None,None,None,pool=pool) is None
    assert audit(None,None,None,pool=pool) is None
    fields,packets=fixture()
    with pytest.raises(MemoryError):make(fields,packets,b's',b'c',pool=pool,enabled=True)


def test_duplicate_original_root_is_not_applied_twice():
    fields,packets=fixture();duplicate=dict(packets[0]);duplicate['pivots']=duplicate['pivots'].copy()
    duplicate['pivots'][0]+=1;duplicate['columns']=duplicate['columns'].copy();duplicate['columns'][duplicate['columns']==fields['hz'].n_cont]+=1
    with pytest.raises(ValueError):make(fields,[*packets,duplicate],b's',b'c',pool=WorkPool(256_000_000),enabled=True)
