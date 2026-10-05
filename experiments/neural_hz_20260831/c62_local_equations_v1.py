"""Versioned two-word local inverse equations, not legacy alias tags."""
from fractions import Fraction as F
import math
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,word,fraction,in_window

SCHEMA='c62_local_edge_equations_v1'
FLAG=1<<63
PARENT=(1<<32)-1
USED=FLAG|PARENT|(63<<32)


def encode(parent,ratio):
    if type(parent) is not int or not 0<=parent<2**32:raise ValueError('original parent outside encoding')
    m,e=word(*ratio)
    if not m or abs(m).bit_length()>53 or abs(fraction((m,e)))>1:raise ValueError('exact native contractive local ratio required')
    q=max(0,-20-(abs(m).bit_length()-1+e));n=word(m,e+q)
    if q>40 or not in_window(n) or not in_window((1,q)):raise ValueError('local equation operands outside unchanged window')
    value=math.ldexp(float(n[0]),n[1])
    if F(value)!=fraction(n):raise ValueError('inexact native numerator')
    tag=FLAG|parent|(q<<32)
    return tag-(1<<64),value


def decode(tag,numerator,*,column,n_cont,schema):
    if schema!=SCHEMA or not 0<=column<n_cont<2**32:raise ValueError('explicit source-bound equation schema/frame required')
    if type(tag) not in (int,np.int64) or not -(1<<63)<=int(tag)<0:raise ValueError('removed local equation tag required')
    bits=int(tag)+(1<<64);parent=bits&PARENT;q=(bits>>32)&63
    if bits&~USED or not parent<column or q>40:raise ValueError('reserved bits/parent/denominator differ')
    n=native_word(numerator)
    if not n[0] or not in_window(n) or not in_window((1,q)):raise ValueError('bounded local operands required')
    r=word(n[0],n[1]-q)
    if abs(fraction(r))>1:raise ValueError('local inverse box is not redundant')
    return parent,r


def reconstruct(values,roots,scales,*,old_n_cont,old_n_eq,n_cont,schema):
    if schema!=SCHEMA or len(values)!=n_cont or roots.shape!=scales.shape:raise ValueError('full original frame/local lineage required')
    result=[F(v) for v in values]
    if any(abs(v)>1 for v in result):raise ValueError('original latent box violated')
    for rank in range(len(roots)-old_n_eq):
        i=old_n_eq+rank;v=old_n_cont+rank
        if roots[i]<0:
            parent,ratio=decode(roots[i],scales.view(np.float64)[i],column=v,n_cont=n_cont,schema=schema)
            result[v]=fraction(ratio)*result[parent]
    return result
