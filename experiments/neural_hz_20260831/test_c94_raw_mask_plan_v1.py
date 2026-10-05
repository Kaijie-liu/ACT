"""Independent direct-pair expansion for ordinary dense source masks."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c94_raw_mask_plan_v1 import dense_direct,selected_source_check


@pytest.mark.parametrize('c,k',[(1,1),(2,3),(8,8),(16,8)])
@pytest.mark.parametrize('mode',['full','partial_input','partial_output','boundary','zero'])
def test_dense_direct_is_exact_all_original_pairs(c,k,mode):
    a=np.ones((c,4,4),bool);o=np.ones((k,2,2),bool)
    if mode=='partial_input':a[:]=np.arange(a.size).reshape(a.shape)%3==0
    if mode=='partial_output':o[:]=np.arange(o.size).reshape(o.shape)%3==0
    if mode=='boundary':a[:,:2,:]=False;o[:,-1,:]=False
    if mode=='zero':a[:]=False;o[:]=False
    am=np.array([sum(1<<i for i,v in enumerate(row.flat) if v) for row in a],np.uint16)
    om=np.array([sum(1<<i for i,v in enumerate(row.flat) if v) for row in o],np.uint8)
    direct=int(o.sum())
    for output,i,j in np.ndindex(o.shape):
        if o[output,i,j]:
            for channel,y,x in np.ndindex(c,3,3):direct+=int(a[channel,i+y,j+x])
    assert dense_direct(am,om,pool=WorkPool(1000000),enabled=True)==direct


def test_default_off_and_precharge():
    a=np.array([65535],np.uint16);o=np.array([15],np.uint8);pool=WorkPool(0)
    assert dense_direct(a,o,pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):dense_direct(a,o,pool=pool,enabled=True)


def test_invalid_footprint_not_admitted():
    with pytest.raises(ValueError):
        dense_direct(np.array([1],np.uint16),np.array([16],np.uint8),pool=WorkPool(10000),enabled=True)


@pytest.mark.parametrize('mode',['good','parent_alias','output_alias','rhs','binary','count'])
def test_selected_actual_source_postconditions(mode):
    from types import SimpleNamespace
    import scipy.sparse as sp
    ac=sp.eye(20,format='lil');ab=sp.lil_matrix((20,2));rhs=np.zeros(20)
    for i,j in np.ndindex(2,2):
        for y,x in np.ndindex(3,3):ac[16+2*i+j,4*(i+y)+j+x]=-1.
    roots=np.arange(20)
    if mode=='parent_alias':roots[0]=-2
    if mode=='output_alias':roots[16]=-2
    if mode=='rhs':rhs[16]=1.
    if mode=='binary':ab[16,0]=1.
    fields=dict(hz=SimpleNamespace(Ac=ac.tocsr(),Ab=ab.tocsr(),b=rhs),old_n_cont=0,
        logical_n_cont=20,old_n_eq=0,eq_roots=roots)
    maps=dict(ids=np.arange(16).reshape(1,4,4),outs=np.arange(16,20).reshape(1,2,2))
    expected=41 if mode=='count' else 40
    if mode=='good':
        got=selected_source_check(fields,maps,expected,pool=WorkPool(10000),enabled=True)
        assert got['actual_current_direct_nnz']==40 and not got['numeric_or_LIVE_admission']
    else:
        with pytest.raises(ValueError):selected_source_check(fields,maps,expected,pool=WorkPool(10000),enabled=True)
