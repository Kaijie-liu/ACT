"""Synthetic source program for a genuinely issued C32 -> C40 binding.

The 2**-20 consumer is the existing normal-window boundary, not a changed
threshold. It keeps a repeated half relation in C31's original quotient.
No benchmark is loaded and no private source permit is constructed here.
"""
import numpy as np
import scipy.sparse as sp
from fractions import Fraction as F
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source


def fixture(sign=1):
    if type(sign) is not int or sign not in (-1,1):
        raise ValueError('synthetic signed half fixture')
    hz=source(12)
    hz.c[:]=0.;hz.c[6]=.25
    hz.Gc.data[:]=.25
    first=ImplicitConv2DOp(np.array([[[[sign*.5]]]],np.float64),(1,1,3,4))
    dense=np.zeros((6,12),np.float64)
    dense[:,6]=[2.**-20,.125,-.125,.25,.125,-.25]
    dense[:,1]=[.125,.25,.125,-.125,.25,.125]
    dense[:,11]=[.125,-.125,.25,.125,-.125,.25]
    op=sp.csr_matrix(dense)
    expr=cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(hz,(first,op)),),
        np.array([0.,.5,-1.5,0.,.5,0.]),6,hz.frame_id)
    return expr,op


def execute(monkeypatch,*,sign=1,layer=78):
    from experiments.neural_hz_20260831 import test_c32_live_splice_v1 as toy
    monkeypatch.setattr(toy,'fixture',lambda:fixture(sign))
    return toy.execute(monkeypatch,layer_id=layer)


def _dot(matrix,row,point):
    a,b=map(int,matrix.indptr[row:row+2])
    return sum((F(float(v))*point[int(c)] for c,v in
        zip(matrix.indices[a:b],matrix.data[a:b])),F(0))


def feasible_point(old,tf,*,layer,input_sign,free,pool):
    """Independent triangular affine evaluation + symbolic native ReLU.

    No optimizer, search or source-success flags. Every old/pre/new predicate
    is checked separately by the caller with Fraction arithmetic.
    """
    if input_sign not in (-1,1) or free not in (-.5,0.,.5):raise ValueError('fixed synthetic point domain')
    f=old.original_fields;pre=f['hz'];lin=old.lineage
    pool.charge('c47_independent_small_source_point',65536)
    point=[F(0)]*old.hz.n_cont;point[0]=F(input_sign);point[1]=F(free)
    binary=[F(1)]*old.hz.n_bin;binary[0]=F(input_sign);binary[1]=F(-1)
    for row in range(f['old_n_eq'],pre.n_eq):
        a,b=map(int,pre.Ac.indptr[row:row+2]);col=int(pre.Ac.indices[b-1]);pivot=F(float(pre.Ac.data[b-1]))
        if point[col]!=0 or not f['old_n_cont']<=col<pre.n_cont:
            raise ValueError('synthetic oracle lost a unique topological defining slot')
        point[col]=(F(float(pre.b[row]))-_dot(pre.Ac,row,point)-_dot(pre.Ab,row,binary))/pivot
    for at in np.flatnonzero(lin.eq_roots<0):
        col=lin.old_n_cont+int(at)-lin.old_n_eq;parent=-int(lin.eq_roots[at])-1
        point[col]=F(float(lin.eq_scales.view(np.float64)[at]))*point[parent]
    values=[F(float(pre.c[r]))+_dot(pre.Gc,r,point)+_dot(pre.Gb,r,binary) for r in range(pre.n_out)]
    lower=[-2.,.2,-2.,-2.,.1,-2.];upper=[2.,2.,-1.,2.,2.,2.]
    if any(not F(l)<=v<=F(u) for l,v,u in zip(lower,values,upper)):
        raise ValueError('synthetic point violates supplied native phase bounds')
    from act.back_end.core import Bounds
    from act.back_end.hybridz_tf.tf_mlp import _sparse_relu_bounds
    import torch
    bounds=Bounds(torch.tensor(lower,dtype=torch.float64),torch.tensor(upper,dtype=torch.float64))
    alpha,beta=_sparse_relu_bounds(pre,bounds,forced_stable_negative=np.asarray(upper)<=0.)
    for row in (0,3,5):
        xi1,xi2,z=tf._sparse_relu_slots[(pre.frame_id,layer,row)]
        value=values[row];binary[z]=F(-1 if value>=0 else 1)
        point[xi2]=1-2*max(F(0),value)/F(float(beta[row]))
        point[xi1]=2*min(F(0),value)/F(float(alpha[row]))-binary[z]
    if any(abs(v)>1 for v in point) or any(abs(v)!=1 for v in binary):
        raise ValueError('synthetic full point is outside HZ box/binary domain')
    return point,binary,values
