"""Ordinary same-frame affine twins; no instance-specific or solver repair tests."""
from fractions import Fraction as F
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c84_affine_recipe_census_v1 import census,normalized,independently_equal


def fixture(width=2,scale=1.,different=False,live=False,binary=False):
    n=width+3
    rows=np.zeros((2,n)); rows[:, :width]=np.arange(1,width+1)*.125
    rows[0,width]=1.; rows[1,width+1]=scale; rows[1,:width]*=scale
    rhs=np.array([.125,.125*scale])
    if different: rows[1,0]+=.125*scale
    ab=np.zeros((2,1)); ab[1,0]=float(binary)
    gc=np.zeros((1,n)); gc[0,width+2]=1.
    if live: gc[0,width+1]=1.
    return SimpleNamespace(Ac=sp.csr_matrix(rows),Ab=sp.csr_matrix(ab),
        Gc=sp.csr_matrix(gc),b=rhs,n_eq=2,n_cont=n,n_bin=1),width


@pytest.mark.parametrize('width,scale',[(2,1.),(3,2.),(9,.5),(27,4.)])
def test_exact_twins_and_independent_inverse(width,scale):
    hz,first=fixture(width,scale)
    result,_=census(hz,first=first,limit=hz.n_cont,pool=WorkPool(256_000_000),enabled=True)
    assert result['exact_recipe_twins']==1 and result['candidate_rows']==2
    # Both actual original equations define the same value at arbitrary shared inputs.
    x=[F(i%3-1,4) for i in range(width)]
    out=[]
    for row in range(2):
        a,b=hz.Ac.indptr[row:row+2];vals=hz.Ac.data[a:b]
        out.append((F(float(hz.b[row]))-sum(F(float(v))*q for v,q in zip(vals[:-1],x)))/F(float(vals[-1])))
    assert out[0]==out[1]


@pytest.mark.parametrize('kind',['coefficient','constant','parent','output','binary'])
def test_common_neural_non_twins_not_merged(kind):
    hz,first=fixture(different=kind=='coefficient',live=kind=='output',binary=kind=='binary')
    if kind=='constant': hz.b[1]+=.25
    if kind=='parent':
        rows=hz.Ac.toarray();rows[1,0]=0;rows[1,2]=.125;hz.Ac=sp.csr_matrix(rows)
    r,_=census(hz,first=first,limit=hz.n_cont,pool=WorkPool(256_000_000),enabled=True)
    assert r['exact_recipe_twins']==0


def test_explicit_opt_in_and_prepaid_budget():
    hz,first=fixture();p=WorkPool(0)
    assert census(hz,first=first,limit=hz.n_cont,pool=p) is None and p.used==0
    with pytest.raises(MemoryError): census(hz,first=first,limit=hz.n_cont,pool=p,enabled=True)


def test_scalar_rows_not_reopening_closed_class():
    hz,first=fixture(1)
    r,_=census(hz,first=first,limit=hz.n_cont,pool=WorkPool(256_000_000),enabled=True)
    assert r['candidate_rows']==0 and r['rejections']['fewer_than_two_parents']==2


def test_ambiguous_direct_definition_rejected():
    hz,first=fixture();a=hz.Ac.toarray();a[1,first]=a[1,first+1];a[1,first+1]=0;hz.Ac=sp.csr_matrix(a)
    r,_=census(hz,first=first,limit=hz.n_cont,pool=WorkPool(256_000_000),enabled=True)
    assert r['candidate_rows']==0 and r['rejections']['nonunique_direct_definition']==2


def test_independent_row_oracle_does_not_trust_hash():
    hz,_=fixture(different=True)
    with pytest.raises(ValueError,match='different exact defining coefficient'):
        independently_equal(hz.Ac,hz.b,0,1,WorkPool(256_000_000))


def test_normalization_keeps_absolute_latent_identity():
    values=np.array([.25,.5,2.]);cols=np.array([0,1,3],np.int32)
    key,_=normalized(cols,values,.125)
    other,_=normalized(np.array([1,2,4],np.int32),values,.125)
    assert key!=other
