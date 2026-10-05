"""One-use prepared row emission with a derived window and dyadic head code.

Default-off row primitive; source admission requires the complete new producer proof.
The complete ORIGINAL range analysis chooses the shift; reversible scaling
binds that decision to the emitted coefficients. No post-scaling full-vector
absolute-magnitude/min/max pass, and no caller-supplied shift certificate.
"""

import math
from experiments.neural_hz_20260831._c107_canonical_encoding_v1 import prepare_checked as _native_checked
import numpy as np
from experiments.neural_hz_20260831._c104_fused_row_v1 import prepare as _native_prepare
from experiments.neural_hz_20260831.c9_radix_predicate_v1 import RowEncoder,exponent_data
from experiments.neural_hz_20260831.c7_factored_hz_v1 import scaled_exact

_KEY=object()


class _Prepared:
    __slots__=('key','used','continuous','binary','rhs','shift','head')
    def __init__(self,key,continuous,binary,rhs,shift,head):
        if key is not _KEY:raise ValueError('only complete row preparation can issue a payload')
        self.key,self.used=key,False
        self.continuous,self.binary,self.rhs,self.shift,self.head=continuous,binary,rhs,shift,head
    def __reduce__(self):raise TypeError('unpublished one-use prepared payload is not an archive certificate')



def _scale_finite_prepared_coefficients(values, shifts):
    """Private consumer of _prepare_bounded's complete finite-input proof.

    Only _prepare_bounded calls this, after classifying exact original operands.
    Retained inverse equality implies finite output for those finite operands.
    This function is not an admission certificate or a generic scaler.
    """
    values = np.asarray(values, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.int64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        out = np.ldexp(values, shifts)
        back = np.ldexp(out, -shifts)
    if not np.array_equal(back, values):
        raise ValueError('power-of-two scaling is not exactly reversible')
    return out


def _bounded_powers(power,shape):
    """Validate ORIGINAL type/range before broadcast and signed copying."""
    if power is None:return np.zeros(shape,dtype=np.int64)
    array=np.asarray(power)
    if array.dtype.kind not in 'iu' or np.any(array < -4096) or np.any(array > 4096):
        raise ValueError('bounded integer coefficient powers required')
    return np.broadcast_to(array,shape).astype(np.int64)


def _prepare(cv,cp,bv,bp,rhs,*,pool):
    """Generic emission establishes its own complete original power bounds."""
    cp=_bounded_powers(cp,np.shape(cv));bp=_bounded_powers(bp,np.shape(bv))
    return _prepare_bounded(cv,cp,bv,bp,rhs,pool=pool)


def _prepare_bounded(cv,cp,bv,bp,rhs,*,pool):
    """Private stage after encode/_prepare constructed bounded int64 powers."""
    pool.charge('prepared_row_decision_and_head',16)
    cv=np.asarray(cv,dtype=np.float64);bv=np.asarray(bv,dtype=np.float64)
    if cv.ndim!=1 or bv.ndim!=1 or cv.size+bv.size>64_000_000:
        raise ValueError('finite nonzero bounded coefficient vector required')
    continuous=np.empty(cv.shape,dtype=np.float64)
    binary=np.empty(bv.shape,dtype=np.float64)
    decision=_native_prepare(cv,cp,bv,bp,continuous,binary)
    if decision is None:return None
    shift,head=decision
    # Complete original generic RHS finite/ldexp/inverse semantics unchanged.
    result_rhs=float(scaled_exact([rhs],shift)[0])
    return _Prepared(_KEY,continuous,binary,result_rhs,shift,head)


def decode_head(code):
    if type(code) is not int or not 0<=code<=122:raise ValueError('invalid fixed-window head code')
    if not code:return None
    exponent,negative=divmod(code-1,2)
    return math.ldexp(-1. if negative else 1.,exponent-20)


def checked_changed_head(values,*,pool,enabled=False):
    """Recertify ONLY the head after an already exact canonical row rewrite.

It is not a proof for the other coefficients, coordinates, box or incidence.
A future row-revision protocol must call this on every changed/reordered row;
pre-rewrite head codes must never be silently reused.
"""
    if not enabled:return None
    pool.charge('changed_row_head_reclassification',16)
    if type(values) is not np.ndarray or values.ndim!=1 or values.dtype!=np.dtype(np.float64):
        raise ValueError('canonical rewritten float64 row required')
    if not len(values):return 0
    value=float(values[0]);absolute=abs(value)
    if not 2.**-20<=absolute<=2.**40:raise ValueError('changed row head outside unchanged window')
    mantissa,exponent=math.frexp(absolute)
    return 1+2*(exponent-1+20)+int(value<0.) if mantissa==.5 else 0


class _Encoder(RowEncoder):
    def __init__(self,*args,head_pool,**kwargs):
        super().__init__(*args,**kwargs)
        self.head_pool=head_pool
        self.eq_heads,self.ineq_heads=[],[]
        self.omitted_post_magnitude_elements=0
        self.once_checked_logical_power_elements=0

    def _store_prepared(self,cc,bc,payload,*,inequality=False):
        if type(payload) is not _Prepared or payload.key is not _KEY or payload.used:
            raise ValueError('missing/already-consumed complete prepared row')
        payload.used=True
        if len(cc)!=len(payload.continuous) or len(bc)!=len(payload.binary):
            raise ValueError('prepared payload coordinate shape changed')
        needed=int(len(payload.continuous)+len(payload.binary)+1)
        if self.entries+needed>self.base_entries+self.max_extra_entries:
            raise MemoryError('unchanged row entry reserve exhausted')
        rows=self.ineq if inequality else self.eq
        heads=self.ineq_heads if inequality else self.eq_heads
        index=len(rows)
        rows.append((np.asarray(cc,dtype=np.int64).copy(),payload.continuous,
            np.asarray(bc,dtype=np.int64).copy(),payload.binary,payload.rhs))
        heads.append(payload.head)
        self.entries+=needed
        self.omitted_post_magnitude_elements+=needed-1
        return index,payload.shift

    def emit(self,cc,cv,cp,bc,bv,bp,rhs,*,inequality=False,known_shift=None):
        if known_shift is not None:
            raise ValueError('an unbound numeric known_shift cannot authorize prepared emission')
        prepared=_prepare(cv,cp,bv,bp,rhs,pool=self.head_pool)
        if prepared is None:raise ValueError('local radix definition exceeds unchanged window')
        return self._store_prepared(cc,bc,prepared,inequality=inequality)

    def encode(self,cc,cv,bc,bv,rhs,*,cp=None,bp=None,inequality=False):
        cc,cv,bc,bv=map(np.asarray,(cc,cv,bc,bv))
        cp=_bounded_powers(cp,cv.shape);bp=_bounded_powers(bp,bv.shape)
        # All original powers above remain checked before their signed copy.
        # Actual coordinate/RHS/finite/window/inverse checks run in this entrance.
        self.head_pool.charge('prepared_row_decision_and_head',16)
        raw=_native_checked(cc,np.asarray(cv,dtype=np.float64),cp,
            bc,np.asarray(bv,dtype=np.float64),bp,self.nc,self.nb,float(rhs))
        prepared=None if raw is None else _Prepared(_KEY,*raw)
        self.head_pool.charge('c97_actual_logical_power_counter',1)
        self.once_checked_logical_power_elements+=int(cv.size+bv.size)
        if prepared is not None:return self._store_prepared(cc,bc,prepared,inequality=inequality)
        # The original radix algorithm, caps, masks, relays and exact RHS rule
        # are unchanged. No alternate window or larger-cap rescue.
        self.packed_rows+=1
        values,powers=np.concatenate((cv,bv)),np.concatenate((cp,bp))
        _,exponents=exponent_data(values,powers)
        buckets=(exponents-int(exponents.min()))//24
        self.charge(values.size*((int(values.size)-1).bit_length()+2))
        terms=[]
        for bucket in np.unique(buckets):
            self.charge(values.size)
            cmask,bmask=buckets[:cv.size]==bucket,buckets[cv.size:]==bucket
            terms.append(self.auxiliary(cc[cmask],cv[cmask],cp[cmask],bc[bmask],bv[bmask],bp[bmask]))
        root=terms[0]
        for term in terms[1:]:
            high=max(root[1],term[1])
            root,term=self.relay(root,high),self.relay(term,high)
            ordered=sorted((root,term))
            root=self.auxiliary(np.array([v[0] for v in ordered]),np.ones(2),
                np.array([v[1] for v in ordered]),np.empty(0,dtype=np.int64),np.empty(0),np.empty(0,dtype=np.int64))
        slot,unit=root
        root_rhs=float(scaled_exact([rhs],-unit)[0])
        self.charge(24)
        index,shift=self.emit(np.array([slot]),np.ones(1),np.zeros(1,dtype=np.int64),
            np.empty(0,dtype=np.int64),np.empty(0),np.empty(0,dtype=np.int64),root_rhs,inequality=inequality)
        return index,shift-unit


def make_encoder(*args,enabled=False,**kwargs):
    if not enabled:return None
    return _Encoder(*args,**kwargs)

