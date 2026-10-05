"""Ordinary masks: independent actual numerical packet is bounded by topology."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct,prepare
from experiments.neural_hz_20260831.c89_quotient_budget_v1 import bill,lower
from experiments.neural_hz_20260831.c92_topology_first_v1 import upper_bill,select


@pytest.mark.parametrize('c,k',[(1,1),(2,3),(8,8),(8,16)])
@pytest.mark.parametrize('mode',['full','partial_input','partial_output','boundary','zero'])
def test_conditional_bound_covers_actual_dense_rows(c,k,mode):
    pool=WorkPool(256_000_000)
    weights=np.ones((k,c,3,3),np.float32)/8
    _,data=transform(weights,pool=pool,enabled=True);prepared=prepare(data,pool=pool)
    assert prepared.dense
    ids=np.arange(16*c).reshape(c,4,4);out=np.arange(16*c,16*c+4*k).reshape(k,2,2)
    if mode=='partial_input':ids[ids%3!=0]=-1
    if mode=='partial_output':out[out%3!=0]=-1
    if mode=='boundary':ids[:,:2,:]=-1;out[:,-1,:]=-1
    if mode=='zero':ids[:]=-1;out[:]=-1
    direct=int((out>=0).sum())
    for channel,y,x in np.ndindex(out.shape):
        if out[channel,y,x]>=0:direct+=int((ids[:,y:y+3,x:x+3]>=0).sum())
    result=upper_bill(ids,out,direct,pool=pool,enabled=True)
    r,p=construct(prepared,ids,np.zeros(ids.shape,np.int32),out,np.full(out.shape,12,np.int32),
        16*c+4*k,pool=pool,enabled=True)
    native,_=lower(r,p,np.arange(16*c+4*k),[(1,0)]*(16*c+4*k),pool=pool,enabled=True)
    b=bill(native,new_factors=r['new_factors'],direct_nnz=direct);u=result['bill']
    assert r['new_factors']==u['new_factors'] and r['rows']==u['rows']
    assert r['nnz']<=u['nnz_upper'] and b['declared_new_bytes']<=u['declared_new_bytes_upper']
    assert b['entry_delta']<=u['entry_delta_upper']
    assert not result['numeric_admission']


def test_alias_is_not_dense_count_admission():
    ids=np.arange(16).reshape(1,4,4);ids[0,1,1]=ids[0,0,0]
    r=upper_bill(ids,np.arange(16,20).reshape(1,2,2),40,pool=WorkPool(10000),enabled=True)
    assert not r['topology_qualified'] and not r['numeric_admission']


def test_plan_is_cumulative_and_not_numeric_receipt():
    cost=dict(topology_qualified=True,bill=dict(byte_saving_lower=100,new_factors=8000,
        new_emission_work_upper=1000,entry_delta_upper=-10))
    r=select([dict(cost=cost),dict(cost=cost)],existing_aux=1000,existing_work=0,existing_entries=0,
        pool=WorkPool(10000),enabled=True)
    assert r['selected_positions']==[0] and r['whole_auxiliary_reserve_used']==9000
    assert not r['numeric_admission']


def test_default_off_and_precharge():
    ids=np.arange(16).reshape(1,4,4);out=np.arange(16,20).reshape(1,2,2);pool=WorkPool(0)
    assert upper_bill(ids,out,40,pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):upper_bill(ids,out,40,pool=pool,enabled=True)
