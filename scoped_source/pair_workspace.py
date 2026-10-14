"""Optional pair-scoped owned vectors; exact arithmetic, not an RSS guarantee.

Only the fully checked V3 closure licenses implicit zeros. Global identities,
all row obligations and all arithmetic check points remain in force. No native
solver, tensor library or GPU dependency is imported.
"""
from collections.abc import Mapping, MutableMapping
from fractions import Fraction as F
import struct
from types import MappingProxyType

from scoped_source.hz_request_store import POLICY, rational, require
from scoped_source.bounded_runtime import (Owner, RootTable, ExactMap, node_owner,
    scalar_size, encode_scalar, decode_scalar)

U32 = struct.Struct('<I')
CANDIDATE_WIDTH = 8
CHECK_WIDTH = 2
PROFILE = 'PAIR_SCOPED_CANDIDATE8_CHECK2_V1'


class Scope:
    """Private checked closure, global<->local IDs stored in owned buffers."""
    def __setattr__(self, name, value):
        require(not getattr(self, '_sealed', False), 'immutable pair scope')
        object.__setattr__(self, name, value)

    def __delattr__(self,name): raise AttributeError('immutable pair scope')

    def __init__(self, owner, pair):
        owner.check(); f=owner.forest
        f.rows(pair,'eq')  # Closed, complete unordered-pair admission.
        common=f.roles['common']; guards={(a,b):v for a,b,v in f.roles['pairs']}
        self.owner=owner; self.pair=pair; self.closed=False
        self.views=frozenset((*f.chain(common),guards[pair],
            *f.chain(f.roles['experts'][pair[0]],common),
            *f.chain(f.roles['experts'][pair[1]],common)))
        banks={}
        for kind,count in (('nodes',len(f.nodes)),('factors',len(f.factors))):
            forward=owner.allocate(4*count); selected=0
            for j in range(count):
                owner.work.visit('row_visits')
                if self._member(kind,j):
                    selected+=1; forward.put(4*j,U32.pack(selected))
            reverse=owner.allocate(4*selected)
            for j in range(count):
                k=U32.unpack(forward.get(4*j,4))[0]
                if k: reverse.put(4*(k-1),U32.pack(j))
            banks[kind]=(count,selected,forward,reverse)
        self.banks=MappingProxyType(banks)
        self._blocks=tuple(block for _,_,forward,reverse in banks.values() for block in (forward,reverse))
        self._sealed=True

    def _member(self,kind,j):
        f=self.owner.forest
        vi=node_owner(f,j) if kind=='nodes' else f.factors[j][3]
        return vi in self.views

    def bind(self,owner,pair):
        require(not self.closed,'closed pair scope')
        self.owner.check()
        require(owner is self.owner and pair==self.pair,'pair scope owner/binding')

    def local(self,kind,j):
        self.bind(self.owner,self.pair); require(kind in ('nodes','factors'),'scope domain')
        count,selected,forward,reverse=self.banks[kind]
        require(forward.owner is self.owner and reverse.owner is self.owner,'scope buffer ownership')
        require(type(j) is int and 0<=j<count,'global scoped coordinate')
        k=U32.unpack(forward.get(4*j,4))[0]
        require(bool(k)==self._member(kind,j) and k<=selected,'scope membership binding')
        if not k: return None
        require(U32.unpack(reverse.get(4*(k-1),4))[0]==j,'scope inverse binding')
        return k-1

    def global_id(self,kind,k):
        self.bind(self.owner,self.pair); require(kind in ('nodes','factors'),'scope domain')
        _,selected,_,reverse=self.banks[kind]
        require(type(k) is int and 0<=k<selected,'local scoped coordinate')
        j=U32.unpack(reverse.get(4*k,4))[0]
        require(self.local(kind,j)==k,'scope forward binding'); return j

    def close(self):
        object.__setattr__(self,'closed',True)
        for block in self._blocks: block.release()


class ScopedMap(MutableMapping):
    """Global-ID map, private slots; transfer invalidates the former handle."""
    def __init__(self,scope,kind):
        require(type(scope) is Scope and kind in ('nodes','factors'),'closed scoped map')
        scope.bind(scope.owner,scope.pair)
        self.scope=scope; self.kind=kind; self.active=True
        self._slots=ExactMap(scope.owner,scope.banks[kind][1])

    def _local(self,key):
        require(self.active,'consumed/released scoped handle')
        return self.scope.local(self.kind,key)

    def __getitem__(self,key):
        k=self._local(key)
        if k is None: raise KeyError(key)
        return self._slots[k]

    def __setitem__(self,key,value):
        k=self._local(key)
        require(type(value) is F,'exact scoped value')
        self.scope.owner.work.exact(value)  # Also check omitted zeros.
        if k is None:
            require(value==0,'nonzero outside checked pair scope'); return
        self._slots[k]=value

    def __delitem__(self,key):
        self[key]; del self._slots[self._local(key)]

    def __iter__(self):
        require(self.active,'consumed/released scoped handle')
        for k in self._slots: yield self.scope.global_id(self.kind,k)

    def __len__(self): return sum(1 for _ in self)

    def take(self):
        require(self.active,'single private ownership transfer')
        self.scope.bind(self.scope.owner,self.scope.pair)
        other=object.__new__(ScopedMap)
        other.scope=self.scope; other.kind=self.kind; other._slots=self._slots; other.active=True
        self._slots=None; self.active=False; return other

    def close(self):
        if self.active:
            self.active=False; self._slots.close(); self._slots=None


class SparseRoots(Mapping):
    """At most four immutable exact roots; no borrowed mutable storage."""
    def __setattr__(self,name,value):
        require(not getattr(self,'_sealed',False),'immutable sparse root descriptor')
        object.__setattr__(self,name,value)

    def __delattr__(self,name): raise AttributeError('immutable sparse root descriptor')

    def __init__(self,owner,values):
        require(type(values) is dict and len(values)<=4,'four-root objective capacity')
        self.owner=owner; self.count=len(values); self.width=4+scalar_size(POLICY['work_bits'])
        self.block=owner.allocate(self.width*self.count)
        for k,(j,q) in enumerate(sorted(values.items())):
            require(type(j) is int and 0<=j<len(owner.forest.nodes),'objective root identity')
            self.block.put(k*self.width,U32.pack(j)+encode_scalar(q,POLICY['work_bits']))
        self.block.seal(); self._sealed=True

    def __getitem__(self,j):
        require(type(j) is int and 0<=j<len(self.owner.forest.nodes),'objective root coordinate')
        for k in range(self.count):
            raw=self.block.get(k*self.width,self.width)
            if U32.unpack_from(raw)[0]==j: return decode_scalar(raw[4:],POLICY['work_bits'])
        raise KeyError(j)

    def __iter__(self):
        for k in range(self.count): yield U32.unpack(self.block.get(k*self.width,4))[0]

    def __len__(self): self.owner.check(); return self.count
    def close(self): self.block.release()


class PairWorkspace:
    def __init__(self,forest,work,*,capacity=POLICY['total_bytes']):
        self.owner=Owner(forest,work,capacity=capacity)
        self._maps=[]; self._descriptors=[]; self.scope=None
        self.blocks_checked=0; self.maximum_block=0
        try: self.roots=RootTable(self.owner)
        except BaseException: self.owner.close(); raise

    def bind(self,forest,work):
        self.owner.check()
        require(self.owner.forest is forest and self.owner.work is work,'workspace identity')

    def select(self,pair):
        self.owner.check(); require(self.scope is None,'one scoped pair per live tile')
        self.scope=Scope(self.owner,pair)

    def mapping(self,kind,values=()):
        require(self.scope is not None,'select checked pair first')
        self.scope.bind(self.owner,self.scope.pair)
        m=ScopedMap(self.scope,kind); self._maps.append(m)
        for k,v in (values.items() if hasattr(values,'items') else values): m[k]=v
        return m

    def objective(self,task):
        self.scope.bind(self.owner,task.pair)
        w=self.owner.work; f=self.owner.forest; d=rational(task.offset); alpha={}
        for expert,weights in zip(task.pair,(task.left,task.right),strict=True):
            vi=f.roles['experts'][expert]
            require(len(weights)==len(f.views[vi]['output']),'complete expert output properties')
            for k,weight in enumerate(weights):
                w.visit('row_visits'); q=rational(weight)
                if q:
                    c,n=self.roots.row((vi,'output',k))
                    d=w.exact(d+w.exact(c*q)); alpha[n]=w.exact(alpha.get(n,F(0))+q)
        roots=SparseRoots(self.owner,alpha); self._descriptors.append(roots)
        return d,roots

    def pullback(self,columns):
        require(0<len(columns)<=CHECK_WIDTH,'bounded scoped adjoint columns')
        alpha=[]; out=[]; w=self.owner.work; f=self.owner.forest
        try:
            for c in columns:
                require(type(c) is ScopedMap and c.scope is self.scope and c.kind=='nodes',
                        'exclusive local node accumulators')
                a=c.take(); alpha.append(a)
                out.append(self.mapping('factors'))
            for n in range(len(f.nodes)-1,-1,-1):
                w.visit('row_visits')
                weights=[(k,a.get(n,F(0))) for k,a in enumerate(alpha)]
                weights=[(k,v) for k,v in weights if v]
                if not weights: continue
                _,local,refs=f.nodes[n]
                for j,q in local:
                    w.visit('coefficient_visits')
                    for k,v in weights: out[k][j]=w.exact(out[k].get(j,F(0))+w.exact(v*q))
                for p,q in refs:
                    w.visit('coefficient_visits')
                    for k,v in weights: alpha[k][p]=w.exact(alpha[k].get(p,F(0))+w.exact(v*q))
            return out
        finally:
            for a in alpha: a.close()

    def clear_block(self):
        for m in self._maps: m.close()
        self._maps.clear()

    def clear_tile(self):
        self.clear_block()
        for r in self._descriptors: r.close()
        self._descriptors.clear()
        if self.scope is not None: self.scope.close(); self.scope=None

    def close(self):
        self.clear_tile(); self.owner.close()
