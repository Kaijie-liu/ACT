from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest

from experiments.neural_hz_20260831.c21_exact_products_v1 import window_products
from experiments.neural_hz_20260831.c10_fused_rows_v1 import window_products as old_products, WorkPool


@pytest.mark.parametrize('a,b', [(1., .3), (.3, .5), (-.3, -.5), (2., -.3), (.3, .7),
    (2.**-20, 2.**-60), (2.**40, 2.**-60), (2.**40, 1.),
    (1.+2.**-52, 1.-2.**-52), (float(2**26+1), float(2**26+1)*2.**-28)])
def test_exact_product_and_window_match_old_and_fraction(a, b):
    values, ratios = np.array([a]), np.array([b])
    before = values.tobytes(), ratios.tobytes()
    good, result, stats = window_products(values, ratios, pool=WorkPool(0, 0))
    old_good, old_result = old_products(values, ratios)
    assert result.tobytes() == old_result.tobytes() and np.array_equal(good, old_good)
    assert bool(good[0]) == (F(float(result[0])) == F(a)*F(b) and 2.**-20 <= abs(result[0]) <= 2.**40)
    assert before == (values.tobytes(), ratios.tobytes())


def test_complete_deterministic_signed_exponent_mantissa_corpus():
    # Synthetic proof enumeration, not NN input sampling/attack.
    pairs = [(s*np.ldexp(m, e), t*np.ldexp(q, f))
        for s,t,m,q,e,f in product((-1.,1.), (-1.,1.), (1.,1.5,1.+2.**-52),
            (1.,.75,1.-2.**-52), (-20,-1,0,26,39), (-60,-1,0))
        if 2.**-20 <= abs(np.ldexp(m,e)) <= 2.**40 and 2.**-60 <= abs(np.ldexp(q,f)) <= 1.]
    a, b = np.array(pairs).T.copy()
    good, result, stats = window_products(a, b, pool=WorkPool(0, 0))
    og, op = old_products(a, b)
    assert np.array_equal(good, og) and result.tobytes() == op.tobytes()
    assert np.array_equal(good, [F(float(z)) == F(float(x))*F(float(y)) and 2.**-20 <= abs(z) <= 2.**40
        for x,y,z in zip(a,b,result)])


@pytest.mark.parametrize('a,b,fast,general', [([.3,.7],[.5,-.25],True,0),
    ([1.,2.],[.3,.7],False,0), ([1.,.3,.25],[.5,.7,.3],False,1)])
def test_mixed_rows_and_exact_work_accounting(a,b,fast,general):
    pool = WorkPool(0,0)
    good, result, stats = window_products(np.array(a),np.array(b),pool=pool)
    n = len(a)
    assert stats == {'hits': n, 'general': general, 'right_batch_fast': fast}
    assert pool.used == 12*n+4 + (0 if fast else 4*n) + 64*general


@pytest.mark.parametrize('a,b', [(0.,.5),(np.nan,.5),(np.inf,.5),(-np.inf,.5),
    (1.,np.nan),(1.,np.inf),(2.**-21,.5),(1.,2.**-61),(1.,2.),(2.**41,.5)])
def test_invalid_operands_rejected_even_when_other_operand_dyadic(a,b):
    with pytest.raises(ValueError): window_products(np.array([a]),np.array([b]),pool=WorkPool(0,0))


@pytest.mark.parametrize('stage', ['common','left','general'])
def test_cap_rejects_before_named_operation_and_no_mutation(stage,monkeypatch):
    import experiments.neural_hz_20260831.c21_exact_products_v1 as module
    a,b=np.array([.3]),np.array([.7])
    before=a.tobytes(),b.tobytes()
    budget={'common':15,'left':19,'general':83}[stage]
    pool=WorkPool(0,0,max_work=budget)
    if stage=='common':
        monkeypatch.setattr(np,'abs',lambda *a,**k: (_ for _ in ()).throw(AssertionError('before common reserve')))
    if stage=='general':
        monkeypatch.setattr(module,'Fraction',lambda *a,**k: (_ for _ in ()).throw(AssertionError('before exact reserve')))
    if stage=='left':
        actual=np.frexp
        calls=[]
        def guarded(x):
            calls.append(1)
            if len(calls)>1: raise AssertionError('left classification before reserve')
            return actual(x)
        monkeypatch.setattr(np,'frexp',guarded)
    with pytest.raises(MemoryError): window_products(a,b,pool=pool)
    assert (a.tobytes(),b.tobytes())==before
    assert pool.used=={'common':0,'left':16,'general':20}[stage]


@pytest.mark.parametrize('bad', ['dtype','shape','cache'])
def test_no_unverified_cached_mantissa_interface(bad):
    a,b=np.array([.3]),np.array([.7])
    if bad=='dtype': a=a.astype(np.float32)
    if bad=='shape': b=b[:,None]
    with pytest.raises((ValueError,TypeError)):
        window_products(a,b,pool=WorkPool(0,0),**({'ratio_odd':np.array([1])} if bad=='cache' else {}))


def test_empty_vector():
    pool=WorkPool(0,0)
    g,p,s=window_products(np.empty(0),np.empty(0),pool=pool)
    assert not g.size and not p.size and pool.used==0
