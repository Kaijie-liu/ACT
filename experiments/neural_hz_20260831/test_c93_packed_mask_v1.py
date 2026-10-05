"""Independent ordinary footprint expansion and unchanged C92 integer bills."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c92_topology_first_v1 import upper_bill as previous
from experiments.neural_hz_20260831.c93_packed_mask_v1 import footprints,upper_bill


@pytest.mark.parametrize('c,k,h,w',[(1,1,3,3),(2,3,4,5),(8,8,5,6),(8,16,8,8)])
@pytest.mark.parametrize('padding',[0,1])
@pytest.mark.parametrize('mode',['full','partial_input','partial_output','boundary','zero'])
def test_complete_footprints_and_every_bill_field(c,k,h,w,padding,mode):
    oh,ow=h+2*padding-2,w+2*padding-2
    a=np.ones((c,h,w),bool);o=np.ones((k,oh,ow),bool)
    if mode=='partial_input':a[:]=(np.arange(a.size).reshape(a.shape)%3==0)
    if mode=='partial_output':o[:]=(np.arange(o.size).reshape(o.shape)%3==0)
    if mode=='boundary':a[:,:,:2]=False;o[:,-1,:]=False
    if mode=='zero':a[:]=False;o[:]=False
    pool=WorkPool(256_000_000);got=footprints(a,o,(padding,padding),pool=pool,enabled=True)
    for t,(y,x) in enumerate(got['positions'].tolist()):
        ids=np.full((c,4,4),-1,np.int64);out=np.full((k,2,2),-1,np.int64)
        for channel,i,j in np.ndindex(ids.shape):
            sy,sx=y-padding+i,x-padding+j
            if 0<=sy<h and 0<=sx<w and a[channel,sy,sx]:ids[channel,i,j]=(channel*h+sy)*w+sx
        for channel,i,j in np.ndindex(out.shape):
            if y+i<oh and x+j<ow and o[channel,y+i,x+j]:out[channel,i,j]=c*h*w+(channel*oh+y+i)*ow+x+j
        for channel in range(c):
            assert int(got['input_masks'][t,channel])==sum(1<<i for i,v in enumerate(ids[channel].flat) if v>=0)
        for channel in range(k):
            assert int(got['output_masks'][t,channel])==sum(1<<i for i,v in enumerate(out[channel].flat) if v>=0)
        direct=int((out>=0).sum())
        for channel,i,j in np.ndindex(out.shape):
            if out[channel,i,j]>=0:direct+=int((ids[:,i:i+3,j:j+3]>=0).sum())
        old=previous(ids,out,direct,pool=pool,enabled=True)
        new=upper_bill(got['input_masks'][t],got['output_masks'][t],direct,pool=pool,enabled=True)
        assert old==new


def test_precharge_and_default_off():
    a=np.ones((1,4,4),bool);o=np.ones((1,2,2),bool);pool=WorkPool(0)
    assert footprints(a,o,(0,0),pool=pool) is None and pool.used==0
    with pytest.raises(MemoryError):footprints(a,o,(0,0),pool=pool,enabled=True)


def test_numerical_admission_remains_false():
    result=upper_bill(np.array([65535],np.uint16),np.array([15],np.uint8),100000,
        pool=WorkPool(10000),enabled=True)
    assert result['topology_qualified'] and not result['numeric_admission']
