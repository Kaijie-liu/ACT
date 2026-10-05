"""Producer-owned original powers, generic emission and conservative counters."""
import numpy as np
import pytest
from experiments.neural_hz_20260831 import c103_prepared_row_v1 as new
from experiments.neural_hz_20260831 import c69_prepared_row_v1 as old
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.mark.parametrize('entry',['encode','emit','prepare'])
@pytest.mark.parametrize('power',[np.array([np.iinfo(np.uint64).max],np.uint64),
                                  np.array([4097],np.uint64),np.array([-4097],np.int64),
                                  np.array([.5])])
def test_original_power_never_wraps_or_narrows_into_admission(entry,power):
    cc=np.array([0]);cv=np.array([.5]);bc=np.empty(0,np.int64);bv=np.empty(0);bp=np.empty(0,np.int64)
    for module in (old,new):
        enc=module.make_encoder(2,1,16,head_pool=WorkPool(1000000),enabled=True)
        with pytest.raises(ValueError):
            if entry=='encode':enc.encode(cc,cv,bc,bv,0.,cp=power)
            elif entry=='emit':enc.emit(cc,cv,power,bc,bv,bp,0.)
            else:module._prepare(cv,power,bv,bp,0.,pool=WorkPool(1000000))
        assert not enc.eq and not enc.ineq


@pytest.mark.parametrize('entry',['encode','emit','prepare'])
def test_one_local_power_constructor_per_complete_input(entry,monkeypatch):
    checks=[];original=new._bounded_powers
    def bounded(power,shape):
        checks.append((np.asarray(power).dtype if power is not None else None,shape))
        return original(power,shape)
    def repeated(*a,**k):raise AssertionError('fitting preparation repeated generic exponent_data')
    monkeypatch.setattr(new,'_bounded_powers',bounded)
    monkeypatch.setattr(new,'exponent_data',repeated)
    cc=np.array([0,1]);cv=np.array([.25,-.75]);bc=np.array([0]);bv=np.array([.5])
    cp=np.array([3,-4],np.int32);bp=np.array([2],np.uint64)
    enc=new.make_encoder(2,1,16,head_pool=WorkPool(1000000),enabled=True)
    if entry=='encode':enc.encode(cc,cv,bc,bv,.125,cp=cp,bp=bp)
    elif entry=='emit':enc.emit(cc,cv,cp,bc,bv,bp,.125)
    else:new._prepare(cv,cp,bv,bp,.125,pool=WorkPool(1000000))
    assert checks==[(cp.dtype,(2,)),(bp.dtype,(1,))]
    assert enc.once_checked_logical_power_elements==(3 if entry=='encode' else 0)


def test_broadcast_copy_preserves_original_powers():
    source=np.array(3,np.uint64)
    values=new._bounded_powers(source,(4,))
    assert values.dtype==np.int64 and values.tolist()==[3]*4
    source[...] = 4
    assert values.tolist()==[3]*4
    assert new._bounded_powers(None,(4,)).tolist()==[0]*4
