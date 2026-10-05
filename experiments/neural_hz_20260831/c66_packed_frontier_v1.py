"""Packed ordinary birth metadata; same C65 equations and exact recurrence."""
from collections.abc import Mapping
import numpy as np
from experiments.neural_hz_20260831.c65_owned_normal_tracker_v1 import BirthTracker as OriginalTracker


class ScalarColumns(Mapping):
    """Numeric values share the one existing actual parent/key dictionary."""
    __slots__=('keys','values','convert')
    def __init__(self,keys,n,dtype,convert):
        self.keys=keys;self.values=np.empty(n,dtype=dtype);self.convert=convert
    def __iter__(self):return iter(self.keys)
    def __len__(self):return len(self.keys)
    def __getitem__(self,key):
        if key not in self.keys:raise KeyError(key)
        return self.convert(self.values[key])
    def __setitem__(self,key,value):
        if key not in self.keys:raise KeyError(key)
        self.values[key]=value


class WordColumns(Mapping):
    """Only original native <=53-bit words; DP products remain Python integers."""
    __slots__=('keys','values')
    def __init__(self,keys,n):
        self.keys=keys;self.values=np.empty((n,2),dtype=np.int64)
    def __iter__(self):return iter(self.keys)
    def __len__(self):return len(self.keys)
    def __getitem__(self,key):
        if key not in self.keys:raise KeyError(key)
        # Convert BEFORE any product: NumPy int64 multiplication is not exact
        # for the composed512-bit word domain.
        return int(self.values[key,0]),int(self.values[key,1])
    def __setitem__(self,key,value):
        if key not in self.keys:raise KeyError(key)
        self.values[key,0]=value[0];self.values[key,1]=value[1]


class BirthTracker(OriginalTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pool.charge('c66_packed_frontier_bindings',1024)
        # No zero fill, new raw-key scan, new key index or extra per-row pass.
        # The unchanged32*MAIN frontier allowance covers the same bounded
        # scalar fields; one parent dictionary remains the sole key authority.
        n=len(self.raw)
        self.local=WordColumns(self.parents,n)
        self.defining=ScalarColumns(self.parents,n,np.int64,int)
        self.tags=ScalarColumns(self.parents,n,np.int64,int)
        self.numerators=ScalarColumns(self.parents,n,np.float64,float)
