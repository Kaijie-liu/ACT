"""Bounded exact dyadic arithmetic and owned packed scalar pool, no native cast."""
import math
import numpy as np

MAX_BITS=512
ZERO=(0,0)
ONE=(1,0)


def canonical(m,e):
    m=int(m);e=int(e)
    if not m:return ZERO
    low=(abs(m)&-abs(m)).bit_length()-1;m>>=low;e+=low
    if abs(m).bit_length()>MAX_BITS:raise MemoryError('exact scalar mantissa exceeds preregistered512 bits')
    # Comparison against powers of two needs no rounded real value.
    floor=abs(m).bit_length()-1+e
    if floor < -20 or floor>40 or (floor==40 and abs(m)!=1):
        raise ValueError('exact scalar outside unchanged coefficient magnitude window')
    return m,e


def source(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('finite exact source scalar required')
    n,d=value.as_integer_ratio()
    return canonical(n,-(d.bit_length()-1))


def multiply(a,b,pool):
    pool.charge('c54_exact_integer_product',64*max(1,(abs(a[0]).bit_length()+63)//64)*max(1,(abs(b[0]).bit_length()+63)//64))
    return canonical(a[0]*b[0],a[1]+b[1])


def add(a,b,pool):
    if not a[0]:return b
    if not b[0]:return a
    e=min(a[1],b[1]);span=max(abs(a[0]).bit_length()+a[1]-e,abs(b[0]).bit_length()+b[1]-e)
    pool.charge('c54_exact_integer_shared_root_sum',32*max(1,(span+63)//64))
    return canonical((a[0]<<(a[1]-e))+(b[0]<<(b[1]-e)),e)


def unit_bounded(a):
    m,e=a
    return bool(m) and (abs(m)<<e<=1 if e>=0 else abs(m)<=1<<-e)


def quotient_by_power(a,pivot):
    if pivot[0]!=1:raise ValueError('positive power-of-two defining pivot required')
    return canonical(-a[0],a[1]-pivot[1])


class Pool:
    def __init__(self,work):
        self.work=work;self.ids={};self.values=[]
    def intern(self,value):
        value=canonical(*value)
        if value not in self.ids:
            self.work.charge('c54_intern_canonical_scalar',32+max(1,(abs(value[0]).bit_length()+63)//64))
            self.ids[value]=len(self.values);self.values.append(value)
        return self.ids[value]
    def pack(self):
        limbs=[];starts=[0];exponents=[];signs=[]
        for m,e in self.values:
            number=abs(m)
            while number:limbs.append(number & ((1<<64)-1));number>>=64
            starts.append(len(limbs));exponents.append(e);signs.append((m>0)-(m<0))
        self.work.charge('c54_complete_owned_scalar_pool_packing',8*(len(limbs)+len(starts)))
        result=dict(indptr=np.array(starts,np.int32),limbs=np.array(limbs,np.uint64),
            exponent=np.array(exponents,np.int32),sign=np.array(signs,np.int8))
        self.ids.clear();self.values.clear()
        return result


def unpack(table):
    if type(table) is not dict or set(table)!={'indptr','limbs','exponent','sign'}:raise ValueError('complete scalar pool required')
    for n,dtype in [('indptr',np.int32),('limbs',np.uint64),('exponent',np.int32),('sign',np.int8)]:
        a=table[n]
        if type(a) is not np.ndarray or a.dtype!=np.dtype(dtype) or a.ndim!=1 or not a.flags.c_contiguous:raise ValueError('owned scalar pool layout differs')
    ptr=table['indptr'];limbs=table['limbs'];size=len(table['sign'])
    if len(ptr)!=size+1 or len(table['exponent'])!=size or ptr[0]!=0 or ptr[-1]!=len(limbs) or np.any(np.diff(ptr)<0):
        raise ValueError('complete scalar pool spans differ')
    result=[]
    for i in range(size):
        a,b=map(int,ptr[i:i+2]);sign=int(table['sign'][i]);e=int(table['exponent'][i])
        if b-a>8 or sign not in (-1,0,1):raise ValueError('noncanonical exact scalar word')
        m=sum(int(limbs[j])<<(64*(j-a)) for j in range(a,b))*sign
        if (not m and (a!=b or sign!=0 or e!=0)) or (m and (not int(limbs[b-1]) or not int(limbs[a])&1)):
            raise ValueError('nonminimal or zero scalar limbs')
        if canonical(m,e)!=(m,e):raise ValueError('noncanonical scalar')
        result.append((m,e))
    if len(set(result))!=len(result):raise ValueError('duplicate exact scalar pool values')
    return result
