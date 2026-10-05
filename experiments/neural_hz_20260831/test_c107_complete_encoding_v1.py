# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete ordinary new entrance, including RHS and every physical row byte."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831 import c104_prepared_row_v1 as old
from experiments.neural_hz_20260831 import c107_prepared_row_v1 as new
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.mark.parametrize('nc,nb',[(3,0),(33,0),(289,0),(33,8),(0,8),(0,0)])
@pytest.mark.parametrize('offset',[-10,0,10])
def test_complete_actual_entrance_rows_counters_and_RHS(nc,nb,offset):
    cc=np.arange(nc,dtype=np.int64);bc=np.arange(nb,dtype=np.int32)
    cv=((cc%17)+1).astype(np.float64)/64;bv=((bc%7)+1).astype(np.float64)/32
    cv[::2]*=-1;bv[::2]*=-1
    cp=((cc%9)-4+offset).astype(np.int64);bp=((bc%7)-3+offset).astype(np.int64)
    a=old.make_encoder(nc,nb,nc+nb+64,head_pool=WorkPool(256_000_000),enabled=True)
    b=new.make_encoder(nc,nb,nc+nb+64,head_pool=WorkPool(256_000_000),enabled=True)
    before=[x.tobytes() for x in (cc,cv,cp,bc,bv,bp)]
    out=a.encode(cc,cv,bc,bv,.125,cp=cp,bp=bp)
    assert b.encode(cc,cv,bc,bv,.125,cp=cp,bp=bp)==out
    assert a.head_pool.parts==b.head_pool.parts and a.head_pool.used==b.head_pool.used
    assert a.eq_heads==b.eq_heads and a.once_checked_logical_power_elements==b.once_checked_logical_power_elements==nc+nb
    assert a.entries==b.entries and a.def_rows==b.def_rows and len(a.eq)==len(b.eq)==1
    for x,y in zip(a.eq[0][:4],b.eq[0][:4]):assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
    assert np.float64(a.eq[0][4]).tobytes()==np.float64(b.eq[0][4]).tobytes()
    assert F(float(b.eq[0][4]))==F(1,8)*F(2)**out[1]
    for values,powers,result in ((cv,cp,b.eq[0][1]),(bv,bp,b.eq[0][3])):
        for v,p,r in zip(values,powers,result):assert F(float(r))==F(float(v))*F(2)**(int(p)+out[1])
    assert before==[x.tobytes() for x in (cc,cv,cp,bc,bv,bp)]
