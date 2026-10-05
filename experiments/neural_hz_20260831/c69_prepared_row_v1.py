"""One-use prepared row emission with a derived window and dyadic head code.

Default-off row primitive only. Not a fresh HZ generator/native admission.
The complete ORIGINAL range analysis chooses the shift; reversible scaling
binds that decision to the emitted coefficients. No post-scaling full-vector
absolute-magnitude/min/max pass, and no caller-supplied shift certificate.
"""

import math
import numpy as np
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
    """Private consumer of exponent_data's complete finite-input proof.

    Only _prepare calls this, after classifying the exact original operands.
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


def _prepare(cv,cp,bv,bp,rhs,*,pool):
    # Extra control, head extraction/code construction, one-use metadata and
    # publication bookkeeping. Charge even a rejected/pending-radix attempt.
    pool.charge('prepared_row_decision_and_head',16)
    values,powers=np.concatenate((cv,bv)),np.concatenate((cp,bp))
    mantissas,exponents=exponent_data(values,powers)
    if not len(exponents):shift=0
    else:
        lower=-19-int(exponents.min())
        maximum=int(exponents.max())
        top=float(mantissas[exponents==maximum].max())
        upper=(41 if top==.5 else 40)-maximum
        if lower>upper:return None
        shift=min(max(0,lower),upper)
    # m in[.5,1): lower proves every scaled coefficient>=2^-20;
    # (maximum,top) proves every scaled coefficient<=2^40.
    # Both bounds include the ORIGINAL implicit input powers. These exact
    # ldexp/inverse checks bind that reasoning to the actual new numeric rows.
    continuous=_scale_finite_prepared_coefficients(cv,cp+shift)
    binary=_scale_finite_prepared_coefficients(bv,bp+shift)
    result_rhs=float(scaled_exact([rhs],shift)[0])
    head=0
    if len(cv) and mantissas[0]==.5:
        exponent=int(exponents[0])+shift-1
        if not -20<=exponent<=40:raise ValueError('derived head exponent outside proved window')
        head=1+2*(exponent+20)+int(float(cv[0])<0.)
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
        for power in (cp,bp):
            if power is not None:
                array=np.asarray(power)
                if array.dtype.kind not in 'iu' or np.any(array < -4096) or np.any(array >4096):
                    raise ValueError('bounded integer coefficient powers required')
        cp=np.zeros(cv.size,dtype=np.int64) if cp is None else np.broadcast_to(cp,cv.shape).astype(np.int64)
        bp=np.zeros(bv.size,dtype=np.int64) if bp is None else np.broadcast_to(bp,bv.shape).astype(np.int64)
        if (cc.dtype.kind not in 'iu' or bc.dtype.kind not in 'iu'
                or cc.shape!=cv.shape or bc.shape!=bv.shape or cv.ndim!=1 or bv.ndim!=1
                or np.any(cc<0) or np.any(cc>=self.nc) or np.any(bc<0) or np.any(bc>=self.nb)
                or np.any(np.diff(cc)<=0) or np.any(np.diff(bc)<=0) or not np.isfinite(rhs)):
            raise ValueError('invalid original row coordinates')
        prepared=_prepare(cv,cp,bv,bp,rhs,pool=self.head_pool)
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
