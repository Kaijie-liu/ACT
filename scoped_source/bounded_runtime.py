"""Opt-in owned payloads, not a bound on CPython/native process RSS.

Exact arithmetic temporaries and the upstream checked forest remain outside
this arena. No solver, dataset or tensor library is imported here.
"""
from collections.abc import MutableMapping
from fractions import Fraction as F
import struct
import time

from scoped_source.hz_request_store import POLICY, require, rational
from scoped_source.check_scaled_references import checked

HEADER = struct.Struct('<BBHHH8s')
ROOT = struct.Struct('<dII')  # center, node owner, present


def node_owner(forest, n):
    """Checked wire metadata, without decoding a source coefficient row."""
    from scoped_source.wire_forest import CheckedWireForest
    return (forest.nodes.descriptor(n)[0] if type(forest) is CheckedWireForest
            else forest.nodes[n][0])


def scalar_size(bits, dyadic=False):
    require(type(bits) is int and 1 <= bits <= POLICY['work_bits'], 'fixed scalar bit ceiling')
    return HEADER.size + ((bits+7)//8)*(1 if dyadic else 2)


def encode_scalar(value, bits, *, dyadic=False):
    """Fixed width, canonical magnitude encoding. No float conversion."""
    require(type(value) is F, 'exact Fraction input')
    require(max(value.numerator.bit_length(), value.denominator.bit_length()) <= bits,
            'fixed scalar capacity')
    size=scalar_size(bits,dyadic); width=(bits+7)//8
    if not value: return bytes(size)
    n=abs(value.numerator); d=value.denominator; is_dyadic=d & (d-1)==0
    require(not dyadic or is_dyadic, 'non-dyadic scalar in dyadic profile')
    tag=1 if is_dyadic else 2
    tail=(d.bit_length()-1) if is_dyadic else d.bit_length()
    head=HEADER.pack(tag,int(value<0),n.bit_length(),tail,0,bytes(8))
    raw=head+n.to_bytes(width,'little')
    if not dyadic: raw+=(0 if is_dyadic else d).to_bytes(width,'little')
    return raw


def decode_scalar(raw, bits, *, dyadic=False):
    require(type(raw) is bytes and len(raw)==scalar_size(bits,dyadic), 'fixed scalar length/type')
    tag,sign,nb,db,extra,pad=HEADER.unpack_from(raw)
    require(sign in (0,1) and extra==0 and pad==bytes(8), 'scalar header canonicality')
    if tag==0:
        require(not any(raw),'canonical zero'); return F(0)
    require(tag in (1,2) and 1<=nb<=bits and (tag!=2 or not dyadic), 'scalar tag/magnitude')
    width=(bits+7)//8; n=int.from_bytes(raw[HEADER.size:HEADER.size+width],'little')
    require(n.bit_length()==nb, 'scalar numerator bits')
    if tag==1:
        require(db<bits and (db==0 or n%2==1), 'canonical dyadic denominator')
        require(dyadic or not any(raw[HEADER.size+width:]), 'unused denominator padding')
        d=1<<db
    else:
        require(1<=db<=bits,'rational denominator bits')
        d=int.from_bytes(raw[HEADER.size+width:],'little')
        require(d.bit_length()==db and d & (d-1)!=0, 'general rational denominator')
    q=F(-n if sign else n,d)
    require(q.numerator==(-n if sign else n) and q.denominator==d, 'reduced scalar')
    return q


class Owner:
    """One invocation; payload accounting never refunds cumulative work/bytes."""
    def __setattr__(self,name,value):
        if name in ('forest','work','capacity') and hasattr(self,name):
            raise AttributeError('immutable owner binding/ceiling')
        object.__setattr__(self,name,value)

    def __delattr__(self,name): raise AttributeError('owned invocation fields')

    def __init__(self,forest,work,*,capacity=POLICY['total_bytes']):
        checked(forest,work)
        require(type(capacity) is int and 0<capacity<=POLICY['total_bytes'], 'arena ceiling')
        self.forest=forest; self.work=work; self.capacity=capacity; self.closed=False
        self._blocks=[]; self._tile=None; self._issuer=object()
        self.stats=dict(live_bytes=0,peak_bytes=0,allocated_bytes=0,copied_bytes=0,
                        released_bytes=0,allocations=0,cleanup_seconds=0.)

    def check(self):
        require(not self.closed,'closed owner'); checked(self.forest,self.work)

    def allocate(self,size):
        self.check(); require(type(size) is int and 0<=size<=self.capacity, 'allocation size')
        require(len(self._blocks)<4096 and self.stats['live_bytes']+size<=self.capacity,
                'arena preallocation ceiling')
        self.work.visit('row_visits')
        b=Block(self,size,self._issuer); self._blocks.append(b)
        self.stats['live_bytes']+=size; self.stats['allocated_bytes']+=size
        self.stats['allocations']+=1; self.stats['peak_bytes']=max(self.stats['peak_bytes'],self.stats['live_bytes'])
        self.work.tick(); return b

    def close(self):
        start=time.monotonic()
        self._tile=None; self.closed=True
        while self._blocks: self._blocks[-1].release()
        self.stats['cleanup_seconds']+=time.monotonic()-start


class Block:
    """No buffer protocol or borrowed memoryview; only copies escape."""
    def __init__(self,owner,size,issuer=None):
        require(type(owner) is Owner and issuer is owner._issuer,'owner-admitted buffer only')
        self.owner=owner; self.size=size; self._data=bytearray(size); self._sealed=None

    def _check(self,offset,size):
        self.owner.check(); require(self._data is not None or self._sealed is not None,'released block')
        require(type(offset) is int and type(size) is int and 0<=offset<=self.size-size,'buffer range')

    def put(self,offset,raw):
        require(type(raw) is bytes,'immutable write input'); self._check(offset,len(raw))
        require(self._data is not None,'sealed block'); self.owner.work.visit('coefficient_visits')
        self._data[offset:offset+len(raw)]=raw; self.owner.stats['copied_bytes']+=len(raw)

    def get(self,offset,size):
        self._check(offset,size); self.owner.work.visit('coefficient_visits')
        with memoryview(self._data if self._data is not None else self._sealed) as view:
            raw=bytes(view[offset:offset+size])
        self.owner.stats['copied_bytes']+=size; return raw

    def seal(self):
        self._check(0,self.size); require(self._data is not None,'single seal')
        # Immutable publication copy temporarily coexists with the fill buffer.
        require(self.owner.stats['live_bytes']+self.size<=self.owner.capacity,'seal overlap ceiling')
        self.owner.work.visit('row_visits'); self.owner.stats['allocated_bytes']+=self.size
        self.owner.stats['allocations']+=1
        self.owner.stats['peak_bytes']=max(self.owner.stats['peak_bytes'],self.owner.stats['live_bytes']+self.size)
        self._sealed=bytes(self._data); self._data=None
        self.owner.stats['copied_bytes']+=self.size; self.owner.stats['released_bytes']+=self.size
        self.owner.work.tick(); return self._sealed

    def release(self):
        if self._data is not None or self._sealed is not None:
            self._data=None; self._sealed=None
            self.owner.stats['live_bytes']-=self.size; self.owner.stats['released_bytes']+=self.size
            self.owner._blocks.remove(self)


class ExactMap(MutableMapping):
    """Fixed integer-indexed working rational slots, including explicit zeros."""
    def __init__(self,owner,count,*,dyadic=False):
        require(type(count) is int and 0<=count<=POLICY['nnz'],'exact map count')
        self.owner=owner; self.count=count; self.dyadic=dyadic
        self.width=scalar_size(POLICY['work_bits'],dyadic)
        self.block=owner.allocate(count*(self.width+1))

    def _offset(self,key):
        require(type(key) is int and 0<=key<self.count,'exact coordinate identity')
        return key*(self.width+1)

    def __getitem__(self,key):
        offset=self._offset(key)
        if self.block.get(offset,1)==b'\0': raise KeyError(key)
        return decode_scalar(self.block.get(offset+1,self.width),POLICY['work_bits'],dyadic=self.dyadic)

    def __setitem__(self,key,value):
        offset=self._offset(key); raw=encode_scalar(value,POLICY['work_bits'],dyadic=self.dyadic)
        self.block.put(offset,b'\1'+raw)

    def __delitem__(self,key):
        self[key]; self.block.put(self._offset(key),bytes(self.width+1))

    def __iter__(self):
        for k in range(self.count):
            if self.block.get(self._offset(k),1)!=b'\0': yield k

    def __len__(self): return sum(1 for _ in self)
    def close(self): self.block.release()


class RootTable:
    """Unique checked node roots + ordered membership IDs, no dual merging.

    A node used with different centers is refused by this optional profile.
    The old forest path remains available; no assumption weakens its checks.
    """
    def __init__(self,owner):
        owner.check(); self.owner=owner; f=owner.forest
        self.roots=owner.allocate(len(f.nodes)*ROOT.size); self.members={}
        for vi,view in enumerate(f.views):
            for kind in ('output','eq','ine'):
                block=owner.allocate(4*len(view[kind])); self.members[vi,kind]=block
                for k,(c,n) in enumerate(view[kind]):
                    owner.work.visit('row_visits'); owner_id=node_owner(f,n)
                    raw=ROOT.pack(float(c),owner_id,1)
                    require(F(float(c))==c,'root exact binary64 center')
                    old=self.roots.get(n*ROOT.size,ROOT.size)
                    require(old==bytes(ROOT.size) or old==raw,'root center/owner alias')
                    self.roots.put(n*ROOT.size,raw); block.put(k*4,struct.pack('<I',n))

    def row(self,row_id):
        self.owner.check(); vi,kind,k=row_id
        block=self.members[vi,kind]; n=struct.unpack('<I',block.get(k*4,4))[0]
        c,owner_id,present=ROOT.unpack(self.roots.get(n*ROOT.size,ROOT.size))
        require(present==1 and node_owner(self.owner.forest,n)==owner_id,'root ownership')
        # Every lookup remains bound to its specific ordered source membership.
        require((F(c),n)==self.owner.forest.row(row_id),'ordered root membership')
        return F(c),n


class Workspace:
    def __init__(self,forest,work,*,capacity=POLICY['total_bytes']):
        self.owner=Owner(forest,work,capacity=capacity); self._maps=[]
        try: self.roots=RootTable(self.owner)
        except BaseException: self.owner.close(); raise

    def bind(self,forest,work):
        self.owner.check(); require(self.owner.forest is forest and self.owner.work is work,'workspace identity')

    def mapping(self,kind,values=()):
        count=len(self.owner.forest.nodes if kind=='nodes' else self.owner.forest.factors)
        require(kind in ('nodes','factors'),'map domain')
        m=ExactMap(self.owner,count); self._maps.append(m)
        entries=values.items() if hasattr(values,'items') else values
        for k,v in entries: m[k]=v
        return m

    def objective(self,task):
        w=self.owner.work; f=self.owner.forest; d=rational(task.offset); alpha=self.mapping('nodes')
        for expert,weights in zip(task.pair,(task.left,task.right),strict=True):
            vi=f.roles['experts'][expert]
            require(len(weights)==len(f.views[vi]['output']),'complete expert output properties')
            for k,weight in enumerate(weights):
                w.visit('row_visits'); q=rational(weight)
                if q:
                    c,n=self.roots.row((vi,'output',k))
                    d=w.exact(d+w.exact(c*q)); alpha[n]=w.exact(alpha.get(n,F(0))+q)
        return d,alpha

    def clear_tile(self):
        for m in self._maps: m.close()
        self._maps.clear()

    def close(self):
        self.clear_tile(); self.owner.close()
