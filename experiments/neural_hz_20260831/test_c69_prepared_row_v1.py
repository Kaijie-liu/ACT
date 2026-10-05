from fractions import Fraction as F
import math
import pickle
import numpy as np
import pytest

from experiments.neural_hz_20260831 import c69_prepared_row_v1 as t
from experiments.neural_hz_20260831.c9_radix_predicate_v1 import RowEncoder
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c9_radix_predicate_v1 import fixture


def encoder(nc=4,nb=2,entries=128,**kw):
    return t.make_encoder(nc,nb,entries,head_pool=WorkPool(256_000_000),enabled=True,**kw)


def same_rows(new,old):
    assert new.def_rows==old.def_rows
    assert (new.extra_work,new.entries,new.packed_rows,new.relays)==(old.extra_work,old.entries,old.packed_rows,old.relays)
    for rows,expected,heads in ((new.eq,old.eq,new.eq_heads),(new.ineq,old.ineq,new.ineq_heads)):
        assert len(rows)==len(expected)==len(heads)
        for got,want,code in zip(rows,expected,heads):
            for a,b in zip(got[:4],want[:4]):assert a.dtype==b.dtype and a.tobytes()==b.tobytes()
            assert np.float64(got[4]).tobytes()==np.float64(want[4]).tobytes()
            decoded=t.decode_head(code)
            expected_head=None if not len(got[1]) or math.frexp(abs(float(got[1][0])))[0]!=.5 else float(got[1][0])
            assert decoded==expected_head
            for values in (got[1],got[3]):
                assert all(2.**-20<=abs(float(v))<=2.**40 for v in values)


def test_default_off_never_reads_inputs():
    assert t.make_encoder(object(),object(),object(),head_pool=object()) is None
    assert t.checked_changed_head(object(),pool=object()) is None


@pytest.mark.parametrize('exponent',[-900,-100,-20,0,40,100,900])
@pytest.mark.parametrize('mantissa',[.5,np.nextafter(.5,1.),.75,np.nextafter(1.,0.)])
@pytest.mark.parametrize('sign',[-1,1])
def test_actual_original_extrema_prove_every_shifted_boundary(exponent,mantissa,sign):
    values=np.array([math.ldexp(sign*mantissa,exponent)])
    new,old=encoder(),RowEncoder(4,2,128)
    args=(np.array([0]),values,np.zeros(0,np.int64),np.zeros(0),0.)
    assert new.encode(*args)==old.encode(*args)
    same_rows(new,old)
    assert new.omitted_post_magnitude_elements==1
    assert new.head_pool.used==16


@pytest.mark.parametrize('gap',[20,60,100,300,900])
@pytest.mark.parametrize('false_constant',[False,True])
def test_complete_radix_eq_ineq_binary_false_rows_unchanged(gap,false_constant):
    h=fixture(gap,false_constant)
    new,old=encoder(h.n_cont,h.n_bin,1000),RowEncoder(h.n_cont,h.n_bin,1000)
    for kind,m,bm,rhs in ((False,h.Ac,h.Ab,h.b),(True,h.Auc,h.Aub,h.ub)):
        for row in range(m.shape[0]):
            a,b=m.indptr[row:row+2];c,d=bm.indptr[row:row+2]
            args=(m.indices[a:b],m.data[a:b],bm.indices[c:d],bm.data[c:d],float(rhs[row]))
            assert new.encode(*args,inequality=kind)==old.encode(*args,inequality=kind)
    same_rows(new,old)


@pytest.mark.parametrize('cp',[-4096,-23,0,31,4096])
@pytest.mark.parametrize('bp',[-31,0,23])
def test_implicit_powers_and_binary_only_extrema_match_exact_reference(cp,bp):
    new,old=encoder(),RowEncoder(4,2,128)
    args=(np.array([0,1]),np.array([-.125,.5]),np.array([0]),np.array([-.75]),0.)
    outcomes=[]
    for e in (new,old):
        try:outcomes.append(('ok',e.encode(*args,cp=cp,bp=bp)))
        except (ValueError,MemoryError,FloatingPointError,OverflowError) as exc:outcomes.append(('reject',None))
    assert outcomes[0]==outcomes[1]
    if outcomes[0][0]=='ok':same_rows(new,old)


@pytest.mark.parametrize('values',[[],[1.],[-2.**-20],[2.**40],[.75],[-.1]])
def test_changed_head_is_reclassified_not_stale(values):
    a=np.array(values,np.float64);p=WorkPool(256_000_000)
    code=t.checked_changed_head(a,pool=p,enabled=True)
    expected=None if not len(a) or math.frexp(abs(float(a[0])))[0]!=.5 else float(a[0])
    assert t.decode_head(code)==expected and p.used==16


def test_same_owned_buffer_changed_after_alias_or_collision_needs_new_head():
    a=np.array([.75,1.]);p=WorkPool(256_000_000)
    old=t.checked_changed_head(a,pool=p,enabled=True)
    a[0]=1.
    new=t.checked_changed_head(a,pool=p,enabled=True)
    assert old==0 and t.decode_head(new)==1.
    a[:]=a[::-1];a[0]=-.5
    assert t.decode_head(t.checked_changed_head(a,pool=p,enabled=True))==-.5


@pytest.mark.parametrize('bad',[0.,float('nan'),float('inf'),2.**-21,2.**41])
def test_changed_head_rejects_unproved_range(bad):
    with pytest.raises(ValueError):t.checked_changed_head(np.array([bad]),pool=WorkPool(256_000_000),enabled=True)


@pytest.mark.parametrize('bad',['nan','zero','rhs','coords','power','entry','head_work','unbound_shift'])
def test_fail_closed_and_no_alternative_emission(bad):
    e=encoder();cc=np.array([0,1]);cv=np.array([.25,-.5]);bc=np.zeros(0,np.int64);bv=np.zeros(0);rhs=0.;cp=None
    if bad=='nan':cv[0]=np.nan
    elif bad=='zero':cv[0]=0.
    elif bad=='rhs':rhs=np.inf
    elif bad=='coords':cc=np.array([0,0])
    elif bad=='power':cp=.5
    elif bad=='entry':e=encoder(entries=0,max_extra_entries=0)
    elif bad=='head_work':e=t.make_encoder(4,2,128,head_pool=WorkPool(0),enabled=True)
    if bad=='unbound_shift':
        with pytest.raises(ValueError,match='unbound'):e.emit(cc,cv,np.zeros(2,np.int64),bc,bv,np.zeros(0,np.int64),rhs,known_shift=0)
    else:
        with pytest.raises((ValueError,MemoryError,FloatingPointError)):e.encode(cc,cv,bc,bv,rhs,cp=cp)
    assert not e.eq and not e.ineq and not e.eq_heads and not e.ineq_heads


def test_private_prepared_payload_is_one_use_and_not_portable():
    p=t._prepare(np.array([1.]),np.array([0]),np.zeros(0),np.zeros(0,np.int64),0.,pool=WorkPool(256_000_000))
    e=encoder()
    with pytest.raises(TypeError):pickle.dumps(p)
    e._store_prepared(np.array([0]),np.zeros(0,np.int64),p)
    with pytest.raises(ValueError):e._store_prepared(np.array([0]),np.zeros(0,np.int64),p)
    with pytest.raises(ValueError):t._Prepared(object(),None,None,None,None,None)


def test_no_full_post_scaling_abs_map_is_called(monkeypatch):
    old_abs=t.np.abs;calls=[]
    def traced(values,*args,**kw):
        calls.append(np.asarray(values).size);return old_abs(values,*args,**kw)
    monkeypatch.setattr(t.np,'abs',traced)
    e=encoder();e.encode(np.array([0,1]),np.array([.125,.5]),np.array([0]),np.array([.75]),.125)
    # Exactly the ORIGINAL exponent_data absolute map; no second output map.
    assert calls==[3]
    row=e.eq[0]
    assert sum(F(float(v)) for v in row[1])==F(5,8)
