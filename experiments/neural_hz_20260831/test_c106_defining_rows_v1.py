# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinary complete row semantics, not a separate numerical acceptance path."""
from collections import Counter
from fractions import Fraction as F
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c7_factored_hz_v1 import restricted_row
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import box_exponent
from experiments.neural_hz_20260831.c106_defining_rows_v1 import bind_node


def parent(n,masked=False):
    support=np.ones(n,bool)
    if masked:support[::5]=False
    return dict(needed=support,support=support,slots=np.arange(n,dtype=np.int64)+100,
        exponents=(np.arange(n)%13).astype(np.int32))


def original(node,nodes,row,pivot):
    if node['kind']=='op':
        p=nodes[node['parents'][0]];indices,values=restricted_row(node['op'],row,p['needed'])
        columns=p['slots'][indices];powers=p['exponents'][indices].astype(np.int64)
    else:
        terms=[(nodes[p]['slots'][row],nodes[p]['exponents'][row],number)
            for p,number in sorted(Counter(node['parents']).items()) if nodes[p]['support'][row]]
        columns=np.array([t[0] for t in terms],np.int64)
        powers=np.array([t[1] for t in terms],np.int64)
        values=np.array([t[2] for t in terms],np.float64)
    unit=box_exponent(values,powers)
    return np.append(columns,pivot),np.append(-values,1.).astype(np.float64),np.append(powers,unit),unit


def check(node,nodes,rows):
    fn=bind_node(node,nodes)
    for row in rows:
        try:old=original(node,nodes,row,10000+row)
        except ValueError:
            with pytest.raises(ValueError):fn(row,10000+row)
            continue
        new=fn(row,10000+row)
        assert old[3]==new[3]
        for a,b in zip(old[:3],new[:3]):
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            assert b.flags.owndata
        radius=sum(abs(F(float(v)))*F(2)**int(e) for v,e in zip(new[1][:-1],new[2][:-1]))
        assert radius<=F(2)**new[3]


@pytest.mark.parametrize('groups,stride,padding,dilation',[
    (1,1,0,1),(1,1,1,1),(2,1,1,1),(4,1,1,1),(1,2,1,1),(2,1,2,2)])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('sparse_kernel',[False,True])
def test_complete_convolution_rows(groups,stride,padding,dilation,masked,sparse_kernel):
    w=((np.arange(8*(4//groups)*9)%19)-9).reshape(8,4//groups,3,3).astype(np.float64)/32
    if sparse_kernel:w[...,::2,::2]=0
    op=ImplicitConv2DOp(w,(2,4,7,8),stride=stride,padding=padding,dilation=dilation,groups=groups)
    if masked:
        mask=np.ones(op.shape[0],bool);mask[::7]=False
        op=op.mask_rows(mask)
    p=parent(op.shape[1],masked);node=dict(kind='op',op=op,parents=(0,))
    check(node,[p],list(dict.fromkeys([0,1,7,op.shape[0]//2,op.shape[0]-1])))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('indices64',[False,True])
def test_complete_CSR_rows(dtype,masked,indices64):
    a=np.arange(15*17).reshape(15,17).astype(dtype)%11-5;a[:,::3]=0
    op=sp.csr_matrix(a)
    if indices64:op.indices=op.indices.astype(np.int64);op.indptr=op.indptr.astype(np.int64)
    check(dict(kind='op',op=op,parents=(0,)),[parent(17,masked)],range(15))


@pytest.mark.parametrize('repeat',[1,2,3])
@pytest.mark.parametrize('masked',[False,True])
def test_shared_sum_repeated_parents(repeat,masked):
    ps=[parent(7,masked),parent(7,False),parent(7,False)]
    for i,p in enumerate(ps):p['slots']+=i*100;p['exponents']+=i*3
    check(dict(kind='sum',parents=tuple([0]*repeat+[1,2,1])),ps,range(7))


@pytest.mark.parametrize('issue',['coordinate','shape','nonfinite'])
def test_unproved_original_row_is_rejected(issue):
    op=sp.csr_matrix([[1.,2.,3.]])
    p=parent(3)
    if issue=='coordinate':op.indices[1]=99
    if issue=='shape':p['slots']=p['slots'][:-1]
    if issue=='nonfinite':op.data[0]=np.nan
    with pytest.raises(ValueError):bind_node(dict(kind='op',op=op,parents=(0,)),[p])(0,10000)
