"""C32 reversible edits on single-use fresh C31 prepared-generator maps.

No existing Closed/source/archive map may be mutated. Only generate() issues
an ordinary public permit, before the fresh draft is published. Primitive
plans still need complete proof; this is not a native admission certificate.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
import struct
import sys
import weakref
import numpy as np

from experiments.neural_hz_20260831.c17_packed_ownership_v1 import RADIX, UID_LIMIT
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan

MASK = UID_LIMIT - 1
SPLICE = 1 << 62
REDIRECT = 1 << 61
SPLICE_MASK = (1 << 63) - 1


def encode_splice(definition, consumer, inequality, pivot, sign, old_scale):
    if (type(definition) is not int or type(consumer) is not int
            or not 0 <= definition < UID_LIMIT or not 0 <= consumer < UID_LIMIT
            or type(inequality) is not bool or sign not in (-1,1)
            or not math.isfinite(pivot) or pivot <= 0 or math.frexp(pivot)[0] != .5):
        raise ValueError('invalid exact unit splice descriptor')
    exponent = math.frexp(pivot)[1] - 1
    if not -20 <= exponent <= 40: raise ValueError('pivot outside unchanged normal window')
    if type(old_scale) is not int or not -8192 <= old_scale <= 8191:
        raise ValueError('old scale does not fit the explicit reversible tag domain')
    return SPLICE | ((old_scale+8192) << 48) | (definition << 28) | ((sign == -1) << 27) | ((exponent+20) << 21) | (inequality << 20) | consumer


def decode(root):
    root = int(root)
    if root >= SPLICE:
        if root & ~SPLICE_MASK or ((root >> 21) & 63) > 60:
            raise ValueError('unregistered/reserved splice tag')
        return ('splice', (root >> 28) & MASK, root & MASK, bool(root & UID_LIMIT),
            math.ldexp(1., ((root >> 21) & 63)-20), -1 if root & (1 << 27) else 1)
    if root >= REDIRECT:
        if root & ~(REDIRECT | ((1 << 40)-1)): raise ValueError('unregistered redirect tag')
        return ('redirect', root & MASK)
    if not -UID_LIMIT <= root < UID_LIMIT: raise ValueError('unregistered physical/alias tag')
    return ('alias',-root-1) if root < 0 else ('row',root)


@dataclass
class ReversibleLineage:
    eq_roots: np.ndarray
    eq_scales: np.ndarray
    columns: np.ndarray
    retired: np.ndarray
    tails: np.ndarray
    old_n_cont: int
    old_n_eq: int
    seal: str = ''

    def fingerprint(self):
        if set(vars(self)) != {'eq_roots','eq_scales','columns','retired','tails','old_n_cont','old_n_eq','seal'}:
            raise ValueError('unregistered draft lineage payload')
        h = hashlib.sha256(str((self.old_n_cont,self.old_n_eq)).encode())
        for name,dtype in [('eq_roots',np.int64),('eq_scales',np.int64),('columns',np.int32),('retired',np.uint64),('tails',np.uint64)]:
            value = getattr(self,name)
            if type(value) is not np.ndarray or value.ndim != 1 or value.dtype != np.dtype(dtype):
                raise ValueError('wrong tagged lineage storage')
            if not value.flags.c_contiguous:
                raise ValueError('noncontiguous tagged lineage storage')
            h.update(name.encode())
            h.update(memoryview(value).cast('B'))
        return h.hexdigest()

    def validate(self):
        if self.fingerprint() != self.seal: raise ValueError('provisional lineage mutated')
        if self.eq_roots.shape != self.eq_scales.shape: raise ValueError('incomplete lineage maps')
        if (np.any(np.diff(self.columns.astype(np.int64)) <= 0)
                or np.any(self.columns < self.old_n_cont)
                or np.any(self.columns >= self.old_n_cont+len(self.eq_roots)-self.old_n_eq)
                or len(self.retired) != len(self.columns)
                or any(int(v) >= RADIX for v in self.retired)
                or np.any(np.diff((self.retired >> np.uint64(20)).astype(np.int64)) <= 0)
                or np.any(self.tails >= np.uint64(1 << 60))
                or np.any(np.diff(self.tails.astype(np.int64)) <= 0)):
            raise ValueError('incomplete/noncanonical sparse transplantation metadata')
        previous = -1
        for col in self.columns:
            info = decode(self.eq_roots[self.old_n_eq+int(col)-self.old_n_cont])
            if info[0] != 'splice' or info[1] <= previous: raise ValueError('splice definitions are not ordered')
            previous = info[1]

    def numeric_roots(self):
        self.validate()
        return {k:getattr(self,k) for k in ('eq_roots','eq_scales','columns','retired','tails')}

    def eq_row(self, old_row, *, pool):
        """Rank deletion through existing tags; no full physical-row map copy."""
        pool.charge('transplant_deleted_row_search', 12*max(1,len(self.columns).bit_length())+8)
        lo,hi = 0,len(self.columns)
        while lo < hi:
            mid = (lo+hi)//2
            info = decode(self.eq_roots[self.old_n_eq+int(self.columns[mid])-self.old_n_cont])
            if info[1] < old_row: lo = mid+1
            else: hi = mid
        if lo < len(self.columns):
            value = decode(self.eq_roots[self.old_n_eq+int(self.columns[lo])-self.old_n_cont])
            if value[1] == old_row: return None
        return old_row-lo

    def retired_to(self, uid, *, pool):
        pool.charge('transplant_retired_UID_search',8*max(1,len(self.retired).bit_length())+8)
        lo,hi = 0,len(self.retired)
        while lo < hi:
            mid=(lo+hi)//2
            if int(self.retired[mid]) >> 20 < uid: lo=mid+1
            else: hi=mid
        return (int(self.retired[lo]) & MASK) if lo<len(self.retired) and int(self.retired[lo])>>20 == uid else None

    def owner_query(self,index,original,*,pool):
        if type(index) is not int or not 0 <= index < len(self.eq_roots)-self.old_n_eq:
            raise ValueError('ownership query outside MAIN')
        pool.charge('transplant_MAIN_tag_query',8)
        if decode(self.eq_roots[self.old_n_eq+index])[0] == 'splice': return 0
        value = original.query(index,pool=pool)
        pool.charge('transplant_tail_search',16*max(1,len(self.tails).bit_length())+16)
        bounds=[]
        for key in (index<<40,(index+1)<<40):
            lo,hi=0,len(self.tails)
            while lo<hi:
                mid=(lo+hi)//2
                if int(self.tails[mid])<key: lo=mid+1
                else: hi=mid
            bounds.append(lo)
        before_count=value//RADIX
        for raw in self.tails[bounds[0]:bounds[1]]:
            pool.charge('transplant_known_tail_UID_change',12)
            raw=int(raw)
            value += (raw & MASK)-((raw>>20)&MASK)
        count,total=divmod(value,RADIX)
        if value<0 or count!=before_count or total>count*MASK:
            raise ValueError('tail transplantation changed count/range')
        return value

    def iter_words(self,original,*,pool):
        """Complete ordered ownership reader; no dense new owner vector."""
        cursor=0
        for index,value in enumerate(original.iter_words(pool=pool)):
            pool.charge('transplant_stream_tag_and_range',20)
            info=decode(self.eq_roots[self.old_n_eq+index])
            count=value//RADIX
            while cursor<len(self.tails) and int(self.tails[cursor])>>40==index:
                pool.charge('transplant_stream_known_tail_UID_change',12)
                raw=int(self.tails[cursor]); value+=(raw&MASK)-((raw>>20)&MASK); cursor+=1
            if info[0]=='splice':
                value=0
            elif value<0 or value//RADIX!=count or value%RADIX>count*MASK:
                raise ValueError('transplanted stream outside count/UID domain')
            yield value
        if cursor!=len(self.tails): raise ValueError('unconsumed tail update')

    def reconstruct_fraction(self,hz,continuous,*,pool):
        """Use surviving rows only, then compose legacy independent aliases."""
        self.validate()
        if len(continuous)!=hz.n_cont: raise ValueError('global frame width changed')
        result=[F(v) for v in continuous]
        if any(abs(v)>1 for v in result): raise ValueError('point outside latent box')
        for col in self.columns:
            col=int(col); at=self.old_n_eq+col-self.old_n_cont
            _,d,c,inequality,pivot,sign=decode(self.eq_roots[at])
            target=c if inequality else self.eq_row(c,pool=pool)
            matrix=hz.Auc if inequality else hz.Ac
            if target is None or not 0<=target<matrix.shape[0]: raise ValueError('missing surviving reconstruction row')
            a,b=map(int,matrix.indptr[target:target+2]); cols=matrix.indices[a:b]; values=matrix.data[a:b]
            cut=int(np.searchsorted(cols,col))
            prefix=sum((F(float(v))*result[int(k)] for k,v in zip(cols[:cut],values[:cut])),F(0))
            result[col]=(F(float(self.eq_scales.view(np.float64)[at]))-sign*prefix)/F(pivot)
            if abs(result[col])>1: raise ValueError('extension outside proved MAIN box')
        for at in np.flatnonzero(self.eq_roots<0):
            col=self.old_n_cont+int(at)-self.old_n_eq
            parent=-int(self.eq_roots[at])-1
            ratio=float(self.eq_scales.view(np.float64)[at])
            if not 0<=parent<col or not 2.**-60<=abs(ratio)<=1.: raise ValueError('invalid legacy alias extension')
            result[col]=F(ratio)*result[parent]
        return result


def compile_owned(eq_roots,eq_scales,plans,*,old_n_cont,old_n_eq,pool,ownership=None,enabled=False):
    """Edit only fresh exclusive maps; failures poison the unpublished draft.

The sparse-update pool is NOT a complete integrated-generation ledger.
Full-map seal hashing/validation, source authentication, plan discovery and
independent mathematical checks remain explicitly separate diagnostic work.
No affordable complete live path or new HZ is certified by this primitive.
"""
    if not enabled: return None
    if (type(eq_roots) is not np.ndarray or eq_roots.dtype!=np.dtype(np.int64)
            or type(eq_scales) is not np.ndarray or eq_scales.dtype!=np.dtype(np.int64)
            or eq_roots.ndim!=1 or eq_scales.shape!=eq_roots.shape):
        raise ValueError('complete original int64 maps required')
    n=len(plans); main=len(eq_roots)-old_n_eq
    if not n: return None
    if type(ownership) is not _FreshPermit or ownership.used or ownership.roots() is not eq_roots or ownership.scales() is not eq_scales:
        raise ValueError('single-use fresh generator ownership is missing')
    if (not eq_roots.flags.owndata or not eq_scales.flags.owndata
            or not eq_roots.flags.c_contiguous or not eq_scales.flags.c_contiguous
            or sys.getrefcount(eq_roots)!=3 or sys.getrefcount(eq_scales)!=3):
        raise ValueError('lineage has another strong owner/view or is not an owned contiguous allocation')
    ownership.used=True
    pool.charge('fresh_lineage_ownership_checks',32)
    roots,scales=eq_roots,eq_scales
    pool.charge('transplant_pair_metadata',96*n)
    columns=[]; retired=[]; tails=[]; definitions=set(); consumers=set(); previous=-1
    for p in plans:
        if type(p) is not Plan or not old_n_cont<=p.column<old_n_cont+main or p.column<=previous:
            raise ValueError('complete sorted MAIN plans required')
        previous=p.column; at=old_n_eq+p.column-old_n_cont
        if int(eq_roots[at])!=p.definition or p.definition in definitions:
            raise ValueError('missing/reused original defining row')
        if (not 0<=p.producer_uid<UID_LIMIT or not 0<=p.consumer_uid<UID_LIMIT
                or p.producer_uid==p.consumer_uid or not math.isfinite(p.offset)):
            raise ValueError('invalid disjoint UID/offset')
        key=(p.inequality,p.consumer)
        if key in consumers: raise ValueError('shared selected consumer')
        definitions.add(p.definition); consumers.add(key)
        pool.charge('reversible_original_scale_and_row',16)
        roots[at]=encode_splice(p.definition,p.consumer,p.inequality,p.pivot,p.sign,int(scales[at]))
        scales.view(np.float64)[at]=p.offset
        if p.consumer_main is not None:
            target=old_n_eq+p.consumer_main-old_n_cont
            if not old_n_eq<=target<len(roots) or int(eq_roots[target])!=p.consumer or p.inequality:
                raise ValueError('invalid consumer MAIN redirect')
            roots[target]=REDIRECT|(p.consumer<<20)|p.producer_uid
        columns.append(p.column); retired.append((p.consumer_uid<<20)|p.producer_uid)
        pool.charge('transplant_tail_incidence_scan',12*len(p.tail))
        for col in p.tail:
            if type(col) is not int or col<=p.column: raise ValueError('consumer tail is not strictly above pivot')
            if old_n_cont<=col<old_n_cont+main:
                tails.append(((col-old_n_cont)<<40)|(p.consumer_uid<<20)|p.producer_uid)
    if any(not kind and row in definitions for kind,row in consumers):
        raise ValueError('selected-definition dependencies')
    if len(set(v>>20 for v in retired))!=n or len(set(v&MASK for v in retired))!=n:
        raise ValueError('UID reused by simultaneous selected rows')
    pool.charge('transplant_sparse_arrays',2*n+len(tails))
    pool.charge('transplant_sparse_sort',4*(n*max(1,(n-1).bit_length())+len(tails)*max(1,(len(tails)-1).bit_length())))
    result=ReversibleLineage(roots,scales,np.asarray(columns,np.int32),np.sort(np.asarray(retired,np.uint64),kind='stable'),
        np.sort(np.asarray(tails,np.uint64),kind='stable'),old_n_cont,old_n_eq)
    pool.charge('transplant_sparse_validation',12*n+8*len(tails))
    result.seal=result.fingerprint(); result.validate()
    return result


_FRESH_KEY=object()


class _FreshPermit:
    __slots__=('roots','scales','used')
    def __init__(self,key,roots,scales):
        if key is not _FRESH_KEY: raise ValueError('only fresh generation issues an ownership permit')
        self.roots,self.scales,self.used=weakref.ref(roots),weakref.ref(scales),False
    def __reduce__(self): raise TypeError('fresh ownership permits cannot be persisted or replayed')


def generate(expr,keep_rows,*,enabled=False,**kwargs):
    """Create every numeric map from the original expression, never an archive."""
    if not enabled:return None
    from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift
    draft=lift(expr,keep_rows,enabled=True,**kwargs)
    return draft,_FreshPermit(_FRESH_KEY,draft.eq_roots,draft.eq_scales)


def original_entry(root,scale):
    root,scale=int(root),int(scale)
    info=decode(root)
    if info[0]=='splice':
        return (root>>28)&MASK,((root>>48)&16383)-8192
    if info[0]=='redirect':
        return (root>>20)&MASK,scale
    return root,scale


def inverse_digest(roots,scales,extra_arrays=()):
    """Hash the complete ORIGINAL map image without allocating either old map.

    This is an authentication pass, not a hidden candidate-construction pass.
    It reads every tag, streams unchanged byte spans and emits only scalar
    eight-byte replacements. It must match an independently bound old digest.
    """
    if (type(roots) is not np.ndarray or roots.dtype!=np.dtype(np.int64)
            or type(scales) is not np.ndarray or scales.dtype!=np.dtype(np.int64)
            or roots.ndim!=1 or roots.shape!=scales.shape
            or not roots.flags.c_contiguous or not scales.flags.c_contiguous):
        raise ValueError('complete contiguous native-int64 maps required')
    # Sparse patch indices are transient Python scalars, never dense old maps.
    edits=[i for i,v in enumerate(roots) if int(v)>=REDIRECT]
    h=hashlib.sha256()
    for which,a in enumerate((roots,scales)):
        h.update(repr((a.dtype.str,a.shape)).encode())
        raw=memoryview(a).cast('B'); cursor=0
        for i in edits:
            h.update(raw[cursor:8*i])
            old=original_entry(roots[i],scales[i])[which]
            h.update(struct.pack('=q',old)); cursor=8*(i+1)
        h.update(raw[cursor:])
    for value in extra_arrays:
        a=np.asarray(value)
        if not a.flags.c_contiguous:raise ValueError('original proof arrays must be contiguous')
        h.update(repr((a.dtype.str,a.shape)).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()

