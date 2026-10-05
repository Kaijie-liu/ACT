"""Prove tighter unchanged row tariffs for ordinary unique dense Conv tiles."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c95_word_filter_v1 import prepare_words
from experiments.neural_hz_20260831.c96_word_row_v1 import construct
from experiments.neural_hz_20260831.c93_packed_mask_v1 import upper_bill
from experiments.neural_hz_20260831.c99_unique_row_bound_v1 import minimum_final_nnz


@pytest.mark.parametrize('channels,outputs',[(1,1),(1,4),(3,4),(16,32)])
@pytest.mark.parametrize('mode',['full','padded','partial_input','partial_output'])
def test_guaranteed_final_words_and_full_actual_row_cost(channels,outputs,mode):
    ids=np.arange(channels*16,dtype=np.int64).reshape(channels,4,4)
    if mode=='padded':ids[:,0,:]=-1;ids[:,:,0]=-1
    if mode=='partial_input':ids[np.indices(ids.shape).sum(axis=0)%3==0]=-1
    out=np.arange(channels*16,channels*16+outputs*4,dtype=np.int64).reshape(outputs,2,2)
    if mode=='partial_output':out[:,1,1]=-1
    a=np.sum((ids.reshape(channels,16)>=0)*(1<<np.arange(16)),axis=1).astype(np.uint16)
    o=np.sum((out.reshape(outputs,4)>=0)*(1<<np.arange(4)),axis=1).astype(np.uint8)
    weights=np.broadcast_to(np.array([[1,2,4],[2,4,8],[4,8,16]],np.float32)/32,(outputs,channels,3,3)).copy()
    _,prepared=prepare_words(weights,pool=WorkPool(256_000_000),enabled=True)
    assert prepared.dense
    pool=WorkPool(256_000_000)
    report,p=construct(prepared,ids,np.zeros(ids.shape,np.int32),out,np.full(out.shape,20,np.int32),
        channels*16+outputs*4,pool=pool,enabled=True)
    lower=minimum_final_nnz(a,o,pool=WorkPool(256_000_000),enabled=True)
    upper=upper_bill(a,o,10000000,pool=WorkPool(256_000_000),enabled=True)['bill']
    assert lower['guaranteed_final_nnz']<=report['nnz']
    assert lower['auxiliary_final_nnz']==p['indptr'][report['new_factors']]
    row_work=sum(pool.parts.get(k,0) for k in
        ('c96_exact_row_prefix_and_numeric','c96_no_credit_for_coalesced_away_guards'))
    new_upper=64*upper['rows']+16*upper['nnz_upper']-8*lower['guaranteed_final_nnz']
    assert row_work<=new_upper


def test_default_off_and_required_mask_dtype():
    assert minimum_final_nnz(None,None,pool=None) is None
    with pytest.raises(ValueError):minimum_final_nnz(np.zeros(2),np.zeros(2),pool=None,enabled=True)
