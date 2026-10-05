"""Ordinary full-row bytes/inverse and complete source identity, no score proxy."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831 import c97_prepared_row_v1 as old
from experiments.neural_hz_20260831 import c104_prepared_row_v1 as new
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c98_birth_emission_v1 import lift as original_lift
from experiments.neural_hz_20260831.c104_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c91_physical_archive_v1 import fingerprint
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete


@pytest.mark.parametrize('nc,nb',[(3,0),(33,0),(289,0),(33,8),(0,8),(0,0)])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('power',[-31,0,31])
def test_full_ordinary_rows_identical_and_each_coefficient_exact(nc,nb,reverse,power):
    values=[((np.arange(n)%17)+1).astype(np.float64)/64 for n in (nc,nb)]
    powers=[((np.arange(n)%9)-4+power).astype(np.int64) for n in (nc,nb)]
    for v in values:v[::2]*=-1
    if reverse:values=[v[::-1] for v in values];powers=[p[::-1] for p in powers]
    cv,bv=values;cp,bp=powers
    p,q=WorkPool(256_000_000),WorkPool(256_000_000)
    before=old._prepare(cv,cp,bv,bp,.125,pool=p)
    after=new._prepare(cv,cp,bv,bp,.125,pool=q)
    assert p.parts==q.parts and p.used==q.used
    assert (before.shift,before.head)==(after.shift,after.head)
    assert np.float64(before.rhs).tobytes()==np.float64(after.rhs).tobytes()
    for inputs,ps,a,b in ((cv,cp,before.continuous,after.continuous),(bv,bp,before.binary,after.binary)):
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        assert b.flags.owndata and not np.shares_memory(inputs,b)
        for v,k,out in zip(inputs,ps,b):assert F(float(out))==F(float(v))*F(2)**(int(k)+after.shift)


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint','circuit_conv'])
def test_complete_source_including_report_owner_inverse_and_circuits_matches(kind):
    if kind=='circuit_conv':expr=expression();keep=np.ones(expr.n_out,bool)
    else:
        _,saved=complete(kind);expr,keep=saved['expression'],saved['keep']
    if kind!='circuit_conv':
        for fn in (original_lift,lift):
            with pytest.raises(ValueError,match='no strictly smaller whole circuit plan'):fn(expr,keep,enabled=True)
        return
    expected=original_lift(expr,keep,enabled=True)
    actual=lift(expr,keep,enabled=True)
    a,b=expected['state'],actual['state']
    assert fingerprint(a)==fingerprint(b)
    assert a['fields']['report']==b['fields']['report']
    for k in ('eq_uids','ineq_uids'):assert np.array_equal(expected['construction'][k],actual['construction'][k])
    assert len(expected['construction']['circuits'])==len(actual['construction']['circuits'])
    assert a['fields']['expression'] is b['fields']['expression'] is expr
    assert b['fields']['hz'].n_bin==a['fields']['hz'].n_bin
    assert b['fields']['hz'].exact and b['fields']['hz'].frame_id==expr.frame_id
    assert not b['native_or_LIVE_admission']


def test_default_off_source_does_not_inspect_inputs():
    assert lift(object(),object()) is None
