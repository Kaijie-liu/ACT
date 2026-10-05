"""Explicit immutable integer-value sequences; not a disguised Python list.

Only closed, value-based IR consumers may use this prototype. Mutations,
scalar Python-object identity and arbitrary list-subclass behavior are NOT
promised. All nontrivial reads/constructions require an explicit work scope.
This descriptor issues no source/HZ/native admission receipt.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import chain
import operator

_BUDGET=ContextVar('c44_id_work',default=None)
_KEY=object();MAX_IDS=64_000_000;MAX_RUNS=4096


def charge(name,n):
    pool=_BUDGET.get()
    if pool is None:raise ValueError('compact ID operation lacks an explicit paid scope')
    pool.charge(name,n)


@contextmanager
def paid(pool):
    token=_BUDGET.set(pool)
    try:yield
    finally:_BUDGET.reset(token)


def _new(runs):
    charge('id_runs_construction',32+32*len(runs));out=[];size=0
    for start,count,step in runs:
        if any(type(v) is not int for v in (start,count,step)) or count<0 or step==0:
            raise ValueError('integer run geometry required')
        if not count:continue
        last=start+(count-1)*step
        if min(start,last)<0 or max(start,last)>2**63-1:raise ValueError('ID outside nonnegative signed64 domain')
        if count==1:step=1
        if out and out[-1][2]==step and out[-1][0]+out[-1][1]*step==start:
            a,n,d=out[-1];out[-1]=(a,n+count,d)
        else:out.append((start,count,step))
        size+=count
        if size>MAX_IDS or len(out)>MAX_RUNS:raise MemoryError('compact ID descriptor ceiling')
    value=object.__new__(VarIds);object.__setattr__(value,'runs',tuple(out));object.__setattr__(value,'size',size)
    return value


def span(start,count,*,enabled=False):
    if not enabled:return None
    return _new(((start,count,1),))


def snapshot(values,*,enabled=False):
    if not enabled:return None
    if type(values) is VarIds:return values.copy()
    if type(values) is not list:raise ValueError('only exact integer lists or compact IDs may be snapshotted')
    charge('id_literal_snapshot',4*len(values));runs=[]
    for value in values:
        if type(value) is not int or value<0:raise ValueError('nonnegative Python integer IDs required')
        if runs and runs[-1][0]+runs[-1][1]==value:
            a,n,d=runs[-1];runs[-1]=(a,n+1,d)
        else:runs.append((value,1,1))
        if len(runs)>MAX_RUNS:raise MemoryError('compact ID run ceiling')
    return _new(runs)


class VarIds(Sequence):
    __hash__=None
    def __new__(cls,*a,**k):raise TypeError('use explicit opt-in ID factories')
    def __setattr__(self,name,value):raise TypeError('compact variable IDs are immutable')
    def validate(self):
        if set(vars(self))!={'runs','size'} or type(self.runs) is not tuple or len(self.runs)>MAX_RUNS:
            raise ValueError('unregistered compact ID fields')
        charge('id_descriptor_validation',16+16*len(self.runs));n=0
        for t in self.runs:
            if type(t) is not tuple or len(t)!=3 or any(type(v) is not int for v in t):raise ValueError('malformed ID run')
            a,k,d=t
            if k<=0 or d==0 or min(a,a+(k-1)*d)<0 or max(a,a+(k-1)*d)>2**63-1:raise ValueError('invalid ID run')
            n+=k
        if type(self.size) is not int or self.size!=n or not 0<=n<=MAX_IDS:raise ValueError('incomplete ID descriptor size')
    def __len__(self):return self.size
    def __iter__(self):
        self.validate();charge('id_value_iteration',self.size)
        return chain.from_iterable(range(a,a+n*d,d) for a,n,d in self.runs)
    def __repr__(self):return f'VarIds(runs={self.runs!r}, size={self.size})'
    def copy(self):
        self.validate();return _new(self.runs)
    def __getitem__(self,index):
        self.validate();charge('id_index_or_slice',16+8*len(self.runs))
        if not isinstance(index,slice):
            i=operator.index(index)
            if i<0:i+=self.size
            if not 0<=i<self.size:raise IndexError('compact ID index out of range')
            for a,n,d in self.runs:
                if i<n:return a+i*d
                i-=n
            raise ValueError('incomplete ID index routing')
        first,stop,stride=index.indices(self.size);total=len(range(first,stop,stride))
        if not total:return _new(())
        positioned=[];offset=0
        for a,n,d in self.runs:positioned.append((offset,a,n,d));offset+=n
        if stride<0:positioned.reverse()
        out=[]
        for offset,a,n,d in positioned:
            if stride>0:
                lo=max(0,-((first-offset)//stride));hi=min(total-1,(offset+n-1-first)//stride)
            else:
                width=-stride;lo=max(0,-((offset+n-1-first)//width));hi=min(total-1,(first-offset)//width)
            if lo<=hi:out.append((a+(first+lo*stride-offset)*d,hi-lo+1,d*stride))
        return _new(out)
    def __add__(self,other):
        if type(other) is list:other=snapshot(other,enabled=True)
        if type(other) is not VarIds:return NotImplemented
        self.validate();other.validate();return _new(self.runs+other.runs)
    def __radd__(self,other):
        if type(other) is not list:return NotImplemented
        return snapshot(other,enabled=True)+self
    def __mul__(self,times):
        self.validate();n=max(0,operator.index(times))
        if n*self.size>MAX_IDS or n*len(self.runs)>MAX_RUNS:raise MemoryError('compact repeated ID ceiling')
        return _new(self.runs*n)
    __rmul__=__mul__
    def __contains__(self,value):
        if type(value) is not int:raise ValueError('only Python integer ID queries are registered')
        self.validate();charge('id_membership',8*len(self.runs))
        return any(value in range(a,a+n*d,d) for a,n,d in self.runs)
    def index(self,value,start=0,stop=None):
        if type(value) is not int:raise ValueError('only Python integer ID queries are registered')
        self.validate();charge('id_position_query',16+12*len(self.runs))
        lower,upper,_=slice(start,stop,1).indices(self.size);offset=0
        for a,n,d in self.runs:
            if value in range(a,a+n*d,d):
                at=offset+(value-a)//d
                if lower<=at<upper:return at
            offset+=n
        raise ValueError(f'{value!r} is not in compact ID sequence')
    def maximum(self):
        self.validate();charge('id_maximum',8*len(self.runs))
        if not self.runs:raise ValueError('max() arg is an empty sequence')
        return max(max(a,a+(n-1)*d) for a,n,d in self.runs)
    def __eq__(self,other):
        self.validate()
        if type(other) is list:
            if len(other)!=self.size:return False
            charge('id_literal_comparison',self.size)
            return all(type(b) is int and a==b for a,b in zip(self,other))
        if type(other) is not VarIds:return NotImplemented
        other.validate()
        if self.size!=other.size:return False
        charge('id_run_comparison',16*(len(self.runs)+len(other.runs)))
        i=j=u=v=0
        while i<len(self.runs):
            a,n,d=self.runs[i];b,m,e=other.runs[j];k=min(n-u,m-v)
            if a+u*d!=b+v*e or (k>1 and d!=e):return False
            u+=k;v+=k
            if u==n:i+=1;u=0
            if v==m:j+=1;v=0
        return True
    def __reduce__(self):
        self.validate();return _restore,(self.runs,self.size)


def _restore(runs,size):
    out=_new(runs)
    if out.size!=size:raise ValueError('serialized ID size differs from exact expansion')
    return out
