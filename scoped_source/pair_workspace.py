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
        self.owner=owner; self.pair=pair; self.closed=False; self.traversal_ready=False
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

    def prepare_traversal(self,owner,pair):
        """Recheck the complete canonical maps, then seal before publication.

        No candidate-provided roster or cached claim licenses omitted nodes.
        This is an in-process ownership contract, not a hostile Python sandbox.
        """
        self.bind(owner,pair); w=owner.work; f=owner.forest
        w.visit('row_visits')
        require(not self.traversal_ready,'single traversal preparation')
        expected=set(); common=f.roles['common']; guards=[]
        for a,b,vi in f.roles['pairs']:
            w.visit('row_visits')
            if (a,b)==pair: guards.append(vi)
        require(len(guards)==1,'unique traversal guard'); expected.add(guards[0])
        for chain in (f.chain(common),f.chain(f.roles['experts'][pair[0]],common),
                      f.chain(f.roles['experts'][pair[1]],common)):
            for vi in chain:
                w.visit('row_visits'); expected.add(vi)
        require(self.views==expected,'traversal ancestry binding')
        for kind,actual in (('nodes',len(f.nodes)),('factors',len(f.factors))):
            w.visit('row_visits')
            count,selected,forward,reverse=self.banks[kind]
            require(type(count) is int and type(selected) is int and count==actual
                    and 0<=selected<=count and forward.owner is owner and reverse.owner is owner
                    and forward.size==4*count and reverse.size==4*selected,
                    'complete traversal map binding/size')
            seen=0
            for j in range(count):
                w.visit('row_visits'); k=self.local(kind,j)
                if k is not None:
                    require(k==seen,'canonical traversal rank/order'); seen+=1
            require(seen==selected,'complete traversal roster')
            forward.seal(); reverse.seal()  # Copies and overlap are arena-accounted.
        object.__setattr__(self,'traversal_ready',True)
        w.tick()

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


class PairLease:
    """One sealed Scope, never a reusable candidate or cross-session receipt.

    Strong byte references prevent identity reuse; revoke drops all of them.
    This retains the existing in-process API boundary, not a Python sandbox.
    """
    __slots__=('scope','owner','forest','work','session','plan','pair','records','closed')

    def __setattr__(self,name,value): raise AttributeError('immutable pair lease')
    def __delattr__(self,name): raise AttributeError('immutable pair lease')

    def __init__(self,workspace):
        s=workspace.scope; o=workspace.owner; o.check()
        records=[]
        for b in s._blocks:
            o.work.visit('row_visits')
            require(b.owner is o and b._data is None and type(b._sealed) is bytes
                    and len(b._sealed)==b.size,'complete sealed pair lease')
            records.append((b,b.size,b._sealed))
        for name,value in dict(scope=s,owner=o,forest=o.forest,work=o.work,
                session=workspace._session,plan=workspace._tiles,pair=s.pair,
                records=tuple(records),closed=False).items():
            object.__setattr__(self,name,value)

    def check(self,workspace):
        o=workspace.owner; o.work.visit('row_visits'); o.check()
        require(not self.closed and o is self.owner and o.forest is self.forest
                and o.work is self.work and workspace.scope is self.scope
                and workspace._session is self.session and workspace._tiles is self.plan
                and workspace._pair==self.pair,'pair lease invocation identity')
        self.scope.bind(o,self.pair)
        require(len(self.records)==len(self.scope._blocks)==4,'four pair lease maps')
        for (b,size,raw),actual in zip(self.records,self.scope._blocks,strict=True):
            o.work.visit('row_visits')
            require(actual is b and b.owner is o and b.size==size and b._data is None
                    and b._sealed is raw and len(raw)==size,'original sealed map identity')

    def revoke(self):
        if self.closed: return
        object.__setattr__(self,'closed',True)
        records=self.records; scope=self.scope; owner=self.owner
        # The lease must not retain bytes after arena accounting is released.
        object.__setattr__(self,'records',())
        for name in ('scope','owner','forest','work','session','plan'):
            object.__setattr__(self,name,None)
        object.__setattr__(scope,'closed',True)
        for block,size,_ in records:
            # Cleanup uses our admitted allocation metadata, even after a
            # rejected owner/size substitution. No foreign block is released.
            block.owner=owner; block.size=size; block.release()


class PairWorkspace:
    def __setattr__(self,name,value):
        if name in ('scoped_traversal','retain_pair_scope') and hasattr(self,name):
            raise AttributeError('immutable traversal choice')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):
        if name in ('scoped_traversal','retain_pair_scope'): raise AttributeError('immutable traversal choice')
        object.__delattr__(self,name)

    def __init__(self,forest,work,*,capacity=POLICY['total_bytes'],scoped_traversal=False,
                 retain_pair_scope=False):
        require(type(scoped_traversal) is bool,'explicit scoped traversal opt-in')
        require(type(retain_pair_scope) is bool,'explicit pair lifetime opt-in')
        self.scoped_traversal=scoped_traversal
        self.retain_pair_scope=retain_pair_scope
        self.owner=Owner(forest,work,capacity=capacity)
        self._maps=[]; self._descriptors=[]; self.scope=None; self._pair=None
        self._session=None; self._tiles=None; self._next=0; self._live=None; self._lease=None
        self.lifetime_stats=dict(scopes_created=0,scopes_reused=0,scopes_retired=0,
                                scope_work=0,global_seal_work=0,lease_work=0,plan_work=0)
        self.traversal_stats=dict(prepared_tiles=0,setup_work=0,selector_work=0,
                                 scoped_calls=0,global_calls=0,planned_iteration_saving=0)
        self.blocks_checked=0; self.maximum_block=0
        try: self.roots=RootTable(self.owner)
        except BaseException: self.owner.close(); raise

    def bind(self,forest,work):
        self.owner.check()
        require(self.owner.forest is forest and self.owner.work is work,'workspace identity')

    def select(self,pair):
        require(not self.retain_pair_scope,'retained scopes require private Session plan')
        self._select(pair)

    def _select(self,pair):
        self.owner.check(); require(self.scope is None,'one scoped pair per live tile')
        before=sum(self.owner.work.counts.values())
        try: self.scope=Scope(self.owner,pair)
        finally: self.lifetime_stats['scope_work']+=sum(self.owner.work.counts.values())-before
        self._pair=pair
        self.lifetime_stats['scopes_created']+=1
        if self.scoped_traversal:
            before=sum(self.owner.work.counts.values())
            try: self.scope.prepare_traversal(self.owner,pair)
            finally: self.traversal_stats['setup_work']+=sum(self.owner.work.counts.values())-before
            self.traversal_stats['prepared_tiles']+=1

    def configure(self,session):
        from scoped_source.scaled_endpoints import Session
        require(self.retain_pair_scope and self._session is None and type(session) is Session
                and session._Session__workspace is self,'one private Session lifetime')
        self._session=session; self._tiles=session._Session__tiles

    def _session_check(self,session,index):
        before=sum(self.owner.work.counts.values())
        try: self.owner.work.visit('row_visits')
        finally: self.lifetime_stats['plan_work']+=sum(self.owner.work.counts.values())-before
        require(session is self._session and session._Session__workspace is self
                and session._Session__tiles is self._tiles
                and type(index) is int and index==self._next and 0<=index<len(self._tiles),
                'same private plan and next tile')
        self.bind(session._Session__forest,session._Session__work)

    def _lease_check(self):
        before=sum(self.owner.work.counts.values())
        try:
            require(type(self._lease) is PairLease,'complete pair lease required')
            self._lease.check(self)
        finally:
            self.lifetime_stats['lease_work']+=sum(self.owner.work.counts.values())-before

    def select_tile(self,session,index):
        self._session_check(session,index); require(self._live is None,'no retained live tile')
        pair=self._tiles[index][0].pair
        if self.scope is None:
            self._select(pair)
            if not self.scoped_traversal:
                before=sum(self.owner.work.counts.values())
                try:
                    # Global order does not require scoped canonical ranks.
                    for block in self.scope._blocks: block.seal()
                finally:
                    self.lifetime_stats['global_seal_work']+=sum(self.owner.work.counts.values())-before
            before=sum(self.owner.work.counts.values())
            try: self._lease=PairLease(self)
            finally: self.lifetime_stats['lease_work']+=sum(self.owner.work.counts.values())-before
        else:
            require(pair==self._pair,'previous pair must retire at its last tile')
            self.lifetime_stats['scopes_reused']+=1
        self._lease_check(); self._live=index

    def check_tile(self,session,index):
        self._session_check(session,index)
        require(self._live==index,'prepared retained tile'); self._lease_check()

    def finish_tile(self,session,index):
        self.check_tile(session,index)
        self._clear_temporaries()
        if index+1==len(self._tiles) or self._tiles[index+1][0].pair!=self._pair:
            self._clear_scope()
        self._live=None; self._next+=1; self.owner.work.tick()

    def _node_ids(self,columns):
        if self.retain_pair_scope:
            require(self._live is not None,'active tile required for retained traversal')
            self._lease_check()
        n=len(self.owner.forest.nodes)
        if not self.scoped_traversal: return range(n-1,-1,-1)
        before=sum(self.owner.work.counts.values())
        try: self.owner.work.visit('row_visits')
        finally: self.traversal_stats['selector_work']+=sum(self.owner.work.counts.values())-before
        self.scope.bind(self.owner,self._pair)
        require(self.scope.traversal_ready and all(b._data is None and b._sealed is not None
                for b in self.scope._blocks),'sealed traversal maps')
        s=self.scope.banks['nodes'][1]
        saving=(columns+1)*(n-s)-3*s
        # Setup has already been paid. This is not an end-to-end win guarantee.
        if saving>1:
            self.traversal_stats['scoped_calls']+=1
            self.traversal_stats['planned_iteration_saving']+=saving
            return (self.scope.global_id('nodes',k) for k in range(s-1,-1,-1))
        self.traversal_stats['global_calls']+=1
        return range(n-1,-1,-1)

    def mapping(self,kind,values=()):
        require(not self.retain_pair_scope or self._live is not None,'active tile required for retained map')
        require(self.scope is not None,'select checked pair first')
        self.scope.bind(self.owner,self.scope.pair)
        m=ScopedMap(self.scope,kind); self._maps.append(m)
        for k,v in (values.items() if hasattr(values,'items') else values): m[k]=v
        return m

    def objective(self,task):
        require(not self.retain_pair_scope or self._live is not None,'active tile required for retained objective')
        self.scope.bind(self.owner,task.pair)
        from scoped_source.bounded_runtime import shared_source, shared_objective
        if shared_source(self.owner.forest):
            d,roots=shared_objective(self,task); self._descriptors.append(roots); return d,roots
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
            for n in self._node_ids(len(columns)):
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

    def _clear_temporaries(self):
        self.clear_block()
        for r in self._descriptors: r.close()
        self._descriptors.clear()

    def _clear_scope(self):
        if self._lease is not None:
            self._lease.revoke(); self._lease=None
        elif self.scope is not None: self.scope.close()
        if self.scope is not None: self.lifetime_stats['scopes_retired']+=1
        self.scope=None
        self._pair=None

    def clear_tile(self):
        self._clear_temporaries(); self._clear_scope(); self._live=None

    def close(self):
        self.clear_tile(); self._session=None; self._tiles=None; self.owner.close()
