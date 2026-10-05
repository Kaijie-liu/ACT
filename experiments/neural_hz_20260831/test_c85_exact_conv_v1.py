"""Exact ordinary Conv algebra, original bit weights and shared affine factors."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import (
    T,H,R,direct,prove_basis,equation,reconstruct_actual,tile_input_equations,recover_filter)
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform


def test_all_input_kernel_polynomial_coordinates():
    assert prove_basis()['output_coefficients_proved']==576


@pytest.mark.parametrize('channels',[1,3,8,16])
def test_every_weight_against_independent_fraction(channels):
    w=((np.arange(2*channels*9,dtype=np.float32)%29)-14).reshape(2,channels,3,3)/64
    report,data=transform(w,pool=WorkPool(256_000_000),enabled=True)
    assert report['all_coefficients_exact_binary64'] and report['independent_original_integer_kernel_recovery']
    for k,c,a,b in np.ndindex(data['native'].shape):
        expected=sum((F(int(H[a,i])*int(H[b,j]),4)*F(float(w[k,c,i,j]))
                      for i in range(3) for j in range(3)),F(0))
        assert F(float(data['native'][k,c,a,b]))==expected


@pytest.mark.parametrize('mode',['ordinary','shared','padding','offset_radius'])
def test_actual_input_equations_and_inverse(mode):
    ids=np.arange(16).reshape(4,4);centers=np.zeros((4,4),object);scales=np.ones((4,4),object)
    if mode=='shared':ids[1]=ids[0]
    if mode=='padding':ids[0]=-1;scales[0]=0
    if mode=='offset_radius':
        centers=np.array([F(i%3-1,8) for i in range(16)],object).reshape(4,4)
        scales=np.array([F((i%5)+1,8) for i in range(16)],object).reshape(4,4)
    rows=tile_input_equations(ids,centers,scales,16)
    x=[F(i%3-1,2) for i in range(16)];point=reconstruct_actual(rows,x)
    d=np.array([[F(0) if ids[i,j]<0 else centers[i,j]+scales[i,j]*x[ids[i,j]]
                 for j in range(4)] for i in range(4)],object)
    expected=T.astype(object)@d@T.T.astype(object)
    for row,want in zip(rows,expected.flat,strict=True):
        assert row['pivot']*point[row['slot']]==want
        coeff=dict(row['coefficients']);p=coeff.pop(row['slot'])
        assert abs(row['rhs'])+sum(map(abs,coeff.values()),F(0))<=p


@pytest.mark.parametrize('scale',[2.**-16,2.**-8,1.,2.**8])
def test_signed_binary32_exponents_and_zero(scale):
    w=(np.array([0,-1,2,3,-4,5,6,7,-8],np.float32)*scale).reshape(1,1,3,3)
    report,data=transform(w,pool=WorkPool(256_000_000),enabled=True)
    assert report['all_coefficients_exact_binary64'] and data is not None
    assert np.array_equal(recover_filter(data['numerator']),4*np.ldexp(w.astype(np.float64),-(data['exponent'][...,None,None]+2)).astype(np.int64))


def test_overlapping_tiles_keep_original_ids():
    grid=np.arange(24).reshape(4,6);z=np.zeros((4,4),object);s=np.ones((4,4),object)
    rows=tile_input_equations(grid[:,:4],z,s,24)+tile_input_equations(grid[:,2:],z,s,40)
    x=[F(i%5-2,4) for i in range(24)];point=reconstruct_actual(rows,x)
    assert point[:24]==x and len(point)==56
    assert set(c for row in rows[:16] for c,v in row['coefficients'][:-1]) & set(c for row in rows[16:] for c,v in row['coefficients'][:-1]) == set(grid[:,2:4].flat)


def test_actual_equation_box_failure_not_silently_accepted():
    row=equation([(0,F(2)),(1,F(-1))],F(1),2)
    changed={**row,'coefficients':((0,F(-2)),(1,F(1)),(2,F(1)))}
    with pytest.raises(ValueError,match='box'):reconstruct_actual([changed],[F(1),F(-1)])


def test_explicit_opt_in_and_complete_precharge():
    w=np.ones((1,1,3,3),np.float32);p=WorkPool(0)
    assert transform(w,pool=p) is None and p.used==0
    with pytest.raises(MemoryError):transform(w,pool=p,enabled=True)
