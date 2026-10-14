"""Closed direct-source binary format; metadata is not a legacy forest JSON.

Explicit binary64 coefficients, unique roots and ordered membership indices.
No first-layer tensor alias compression or full-process memory claim.
"""
from collections.abc import Sequence
from fractions import Fraction as F
import math
import struct

from scoped_source.hz_request_store import POLICY,compact,decode,digest,identity,require,sha

SCHEMA='DIRECT_SCALED_SOURCE_V1'
MAGIC=b'HZDS0001'
PREFIX=struct.Struct('<8s7I')  # metadata, nodes, locals, refs, roots, uses, scales
NODE=struct.Struct('<5I')
ENTRY=struct.Struct('<Id')
ROOT=struct.Struct('<dI')
INDEX=struct.Struct('<I')
SCALAR=struct.Struct('<d')


def number(q):
    require(type(q) is F,'exact source coefficient')
    x=float(q)
    require(math.isfinite(x) and F(x)==q,'lossless source binary64')
    return x


def value(x):
    require(math.isfinite(x) and (x!=0 or math.copysign(1,x)>0),'canonical source binary64')
    return F(x)


class RootBank(Sequence):
    """Validate immutable whole root values once, charged to the same Work."""
    __slots__=('_values',)

    def __init__(self,values,work):
        require(type(values) is tuple,'immutable root bank input')
        for i,r in enumerate(values):
            work.visit('row_visits')
            require(type(r) is tuple and len(r)==2 and type(r[0]) is F and type(r[1]) is int and r[1]==i,'root bank value/order')
        object.__setattr__(self,'_values',values)

    def __setattr__(self,k,v): raise AttributeError('immutable root bank')
    def __delattr__(self,k): raise AttributeError('immutable root bank')
    def __len__(self): return len(self._values)
    def __getitem__(self,k): return self._values[k]


class RootSequence(Sequence):
    """Immutable ordered uses of whole roots, including centers and node IDs."""
    __slots__=('_roots','_ids')

    def __init__(self,roots,ids):
        require(type(roots) is RootBank and type(ids) is tuple
                and all(type(i) is int and 0<=i<len(roots) for i in ids),'closed root sequence')
        object.__setattr__(self,'_roots',roots); object.__setattr__(self,'_ids',ids)

    def __setattr__(self,k,v): raise AttributeError('immutable root membership')
    def __delattr__(self,k): raise AttributeError('immutable root membership')
    def __len__(self): return len(self._ids)
    def __getitem__(self,k):
        if isinstance(k,slice): return tuple(self._roots[i] for i in self._ids[k])
        return self._roots[self._ids[k]]
    def __eq__(self,other):
        if type(other) is RootSequence and self._roots is other._roots: return self._ids==other._ids
        return isinstance(other,(tuple,list,RootSequence)) and len(self)==len(other) and all(a==b for a,b in zip(self,other))


def encode(builder,roles,cut):
    """Encode the newly constructed graph, never serialize the old document."""
    w=builder.work; w.tick(); nodes=builder.nodes; uses=[]; roots=[None]*len(nodes); views=[]
    for v in builder.views:
        row=dict(v)
        for kind in ('output','eq','ine'):
            start=len(uses)
            for c,n in v[kind]:
                w.visit('row_visits'); require(0<=n<len(nodes),'root node identity')
                require(roots[n] is None or roots[n]==(c,n),'root center alias')
                roots[n]=(c,n); uses.append(n)
            row[kind]=[start,len(uses)-start]
        views.append(row)
    require(all(r is not None for r in roots) and len(uses)<=POLICY['rows'],'complete roots/unchanged membership cap')
    local=sum(len(n[1]) for n in nodes); refs=sum(len(n[2]) for n in nodes)
    m=dict(schema=SCHEMA,source=builder.reader.receipt()['source_sha256'],request=identity(builder.reader.request),
           frame=1,cut=cut,policy=dict(POLICY),factors=builder.factors,views=views,roles=roles,
           source_receipt=builder.reader.receipt())
    header=compact(m)
    require(len(header)<=POLICY['header_bytes'] and len(nodes)<=POLICY['rows'] and
            local<=POLICY['nnz'] and refs<=65536,'direct sections capacity')
    size=PREFIX.size+len(header)+len(nodes)*NODE.size+(local+refs)*ENTRY.size+len(roots)*ROOT.size+len(uses)*4+len(builder.scales)*8
    require(size<=POLICY['total_bytes'],'direct physical source capacity')
    raw=bytearray(size); PREFIX.pack_into(raw,0,MAGIC,len(header),len(nodes),local,refs,len(roots),len(uses),len(builder.scales))
    p=PREFIX.size; raw[p:p+len(header)]=header; p+=len(header); li=ri=0
    for owner,l,r in nodes:
        w.visit('row_visits'); NODE.pack_into(raw,p,owner,li,len(l),ri,len(r)); p+=NODE.size; li+=len(l); ri+=len(r)
    for z in (1,2):
        for n in nodes:
            for j,q in n[z]:
                w.visit('coefficient_visits'); ENTRY.pack_into(raw,p,j,number(q)); p+=ENTRY.size
    for c,n in roots:
        w.visit('row_visits'); ROOT.pack_into(raw,p,number(c),n); p+=ROOT.size
    for n in uses:
        w.visit('row_visits'); INDEX.pack_into(raw,p,n); p+=INDEX.size
    for q in builder.scales:
        w.visit('coefficient_visits'); SCALAR.pack_into(raw,p,number(q)); p+=8
    require(p==size,'exact source extent'); w.tick(); result=bytes(raw); w.tick(); return result


def intake(raw,expected,work):
    work.tick(); require(type(raw) is bytes and PREFIX.size<=len(raw)<=POLICY['total_bytes']
                         and digest(raw)==sha(expected),'direct source identity/extent')
    magic,h,n,l,r,q,u,s=PREFIX.unpack_from(raw)
    require(magic==MAGIC and 0<h<=POLICY['header_bytes'] and 0<n==q<=POLICY['rows']
            and l<=POLICY['nnz'] and r<=65536 and u<=POLICY['rows'] and 0<s<=POLICY['factors'],'direct counts/unchanged caps')
    size=PREFIX.size+h+n*NODE.size+(l+r)*ENTRY.size+q*ROOT.size+u*4+s*8
    require(size==len(raw),'direct section lengths')
    hr=raw[PREFIX.size:PREFIX.size+h]; m=decode(hr)
    require(type(m) is dict and compact(m)==hr and set(m)=={'schema','source','request','frame','cut','policy','factors','views','roles','source_receipt'}
            and m['schema']==SCHEMA,'direct canonical metadata')
    p=PREFIX.size+h; lp=p+n*NODE.size; rp=lp+l*ENTRY.size; rootp=rp+r*ENTRY.size
    usep=rootp+q*ROOT.size; sp=usep+u*4
    nodes=[]; li=ri=0
    for i in range(n):
        work.visit('row_visits'); owner,ls,lc,rs,rc=NODE.unpack_from(raw,p); p+=NODE.size
        require(ls==li and rs==ri and lc<=l-li and rc<=r-ri,'canonical contiguous node sections')
        parts=[]
        for start,count,base in ((ls,lc,lp),(rs,rc,rp)):
            part=[]
            for k in range(start,start+count):
                work.visit('coefficient_visits'); j,x=ENTRY.unpack_from(raw,base+k*ENTRY.size); part.append((j,value(x)))
            parts.append(tuple(part))
        nodes.append((owner,*parts)); li+=lc; ri+=rc
    require(li==l and ri==r,'complete node sections')
    roots=[]
    for i in range(q):
        work.visit('row_visits'); c,j=ROOT.unpack_from(raw,rootp+i*ROOT.size)
        require(j==i,'unique root node order'); roots.append((value(c),j))
    roots=RootBank(tuple(roots),work); uses=[]
    for i in range(u):
        work.visit('row_visits'); j=INDEX.unpack_from(raw,usep+4*i)[0]
        require(j<q,'root membership range'); uses.append(j)
    require(set(uses)==set(range(q)),'all root values used')
    scales=[]
    for i in range(s):
        work.visit('coefficient_visits'); x=value(SCALAR.unpack_from(raw,sp+i*8)[0]); require(x>=0,'nonnegative scale'); scales.append(x)
    require(type(m['views']) is list and 0<len(m['views'])<=POLICY['views'],'direct views capacity')
    views=[]; offset=0
    for v in m['views']:
        require(type(v) is dict and set(v)=={'name','scope','parent','op','factors','output','eq','ine'},'direct view fields'); out=dict(v)
        for kind in ('output','eq','ine'):
            span=v[kind]; require(type(span) is list and len(span)==2 and all(type(j) is int for j in span),'membership span')
            start,count=span
            require(start==offset and 0<=count<=POLICY['view_rows'] and start+count<=u,'ordered complete membership span')
            out[kind]=RootSequence(roots,tuple(uses[start:start+count])); offset+=count
        views.append(out)
    require(offset==u,'no unclaimed memberships'); work.tick()
    return m,tuple(nodes),views,tuple(scales),dict(unique_roots=q,memberships=u,physical_bytes=len(raw))
