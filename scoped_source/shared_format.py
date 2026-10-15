"""Paged source references and interned complete memberships, lossless/stdlib.

Physical aliases never replace the old expanded logical resource checks.
Pages are retained immutable; neither writer nor reader joins the full forest.
"""
from fractions import Fraction as F
import struct

from scoped_source.hz_request_store import POLICY, compact, decode, digest, require, sha, identity
from scoped_source.direct_source_format import (PREFIX, NODE, ENTRY, ROOT, SCALAR, INDEX,
    RootBank, RootSequence, number, value)
from scoped_source.shared_source import SourceLocal

MAGIC = b'HZSH0001'
SCHEMA = 'SHARED_SOURCE_REFERENCE_V1'
PAGE = 65536


class Packet:
    __slots__ = ('pages', 'index', 'identity', 'size')
    def __setattr__(self, k, v): raise AttributeError('immutable forest pages')
    def __delattr__(self, k): raise AttributeError('immutable forest pages')

    def __init__(self, pages, work):
        require(type(pages) is tuple and 0 < len(pages) < POLICY['members'], 'bounded forest pages')
        total = 0; refs = []
        for i, raw in enumerate(pages):
            work.tick(); require(type(raw) is bytes and 0 < len(raw) <= PAGE
                                 and (i == len(pages)-1 or len(raw) == PAGE), 'canonical full pages')
            total += len(raw); require(total <= POLICY['total_bytes'], 'forest total byte ceiling')
            refs.append(dict(file=f'{i:04d}.bin', bytes=len(raw), sha256=digest(raw)))
        index = compact(dict(schema=SCHEMA, bytes=total, pages=refs))
        require(len(index) <= POLICY['member_bytes'], 'page index capacity')
        for k, v in dict(pages=pages, index=index, identity=digest(index), size=total).items():
            object.__setattr__(self, k, v)

    def publish(self, put):
        for i, raw in enumerate(self.pages): put(f'forest/{i:04d}.bin', raw)
        return put('forest/index.json', self.index)


def load(root, index, expected, work):
    from scoped_source.factored_io import read
    from pathlib import Path
    import os
    require(type(index) is bytes and len(index) <= POLICY['member_bytes'] and digest(index) == sha(expected),
            'page index identity')
    m = decode(index)
    require(type(m) is dict and compact(m) == index and set(m) == {'schema','bytes','pages'}
            and m['schema'] == SCHEMA and type(m['bytes']) is int and 0 < m['bytes'] <= POLICY['total_bytes']
            and type(m['pages']) is list and 0 < len(m['pages']) < POLICY['members'], 'page index schema')
    root=Path(root); require(not root.is_symlink() and root.is_dir(), 'forest page directory')
    found=set()
    with os.scandir(root) as entries:
        for entry in entries:
            work.tick(); require(not entry.is_symlink() and entry.is_file(follow_symlinks=False)
                                 and len(found)<POLICY['members'], 'forest page member type/count')
            found.add(entry.name)
    require(found=={'index.json'}|{f'{i:04d}.bin' for i in range(len(m['pages']))}, 'exact forest page inventory')
    pages = []; total = 0
    for i, ref in enumerate(m['pages']):
        work.tick()
        require(type(ref) is dict and set(ref) == {'file','bytes','sha256'} and ref['file'] == f'{i:04d}.bin'
                and type(ref['bytes']) is int and 0 < ref['bytes'] <= PAGE, 'ordered bounded page')
        total += ref['bytes']; require(total <= m['bytes'], 'page running extent')
        raw = read(root, ref['file'], PAGE, sha(ref['sha256']))
        require(len(raw) == ref['bytes'], 'page exact length'); pages.append(raw)
    require(total == m['bytes'], 'complete page extent')
    packet = Packet(tuple(pages), work)
    require(packet.index == index, 'canonical page partition'); return packet


class Writer:
    def __init__(self, work): self.work=work; self.pages=[]; self.fill=bytearray(); self.size=0
    def put(self, raw):
        require(type(raw) is bytes and self.size+len(raw) <= POLICY['total_bytes'], 'paged preallocation ceiling')
        self.size += len(raw); p=0
        while p<len(raw):
            self.work.tick(); n=min(PAGE-len(self.fill),len(raw)-p)
            self.fill.extend(memoryview(raw)[p:p+n]); p+=n
            if len(self.fill)==PAGE:
                self.pages.append(bytes(self.fill)); self.fill.clear()
    def finish(self):
        if self.fill: self.pages.append(bytes(self.fill)); self.fill.clear()
        return Packet(tuple(self.pages), self.work)


class Reader:
    def __init__(self, packet, work): self.packet=packet; self.work=work; self.p=0
    def read(self, size):
        require(type(size) is int and 0 <= size <= POLICY['header_bytes'] and self.p+size <= self.packet.size,
                'bounded section extent')
        self.work.tick(); out=bytearray()
        while size:
            self.work.tick(); k,j=divmod(self.p,PAGE); n=min(size,len(self.packet.pages[k])-j)
            require(n>0, 'complete page'); out.extend(memoryview(self.packet.pages[k])[j:j+n])
            self.p+=n; size-=n
        return bytes(out)
    def unpack(self, fmt): return fmt.unpack(self.read(fmt.size))


def encode(builder, roles, cut):
    w=builder.work; nodes=builder.nodes; roots=[None]*len(nodes); seqs=[]; lookup={}; views=[]; logical=0
    for v in builder.views:
        row=dict(v)
        for kind in ('output','eq','ine'):
            ids=[]
            for c,n in v[kind]:
                w.visit('row_visits'); require(0<=n<len(nodes), 'source root node')
                require(roots[n] is None or roots[n]==(c,n), 'source root center alias')
                roots[n]=(c,n); ids.append(n)
            logical+=len(ids); require(logical<=POLICY['rows'], 'unchanged expanded membership cap')
            key=tuple(ids)
            if key not in lookup: lookup[key]=len(seqs); seqs.append(key)
            row[kind]=lookup[key]
        views.append(row)
    require(all(r is not None for r in roots), 'complete unique roots')
    aliases=[]; locals_=[]
    for i,n in enumerate(nodes):
        if type(n[1]) is SourceLocal:
            aliases.append([i,*n[1].row.key,n[1].row.width]); locals_.append(n[1].tail)
        else: locals_.append(n[1])
    l=sum(map(len,locals_)); r=sum(len(n[2]) for n in nodes); u=sum(map(len,seqs))
    require(sum(len(n[1]) for n in nodes)<=POLICY['nnz'] and r<=65536 and len(nodes)<=POLICY['rows'],
            'unchanged logical local/reference/node caps')
    spans=[]; offset=0
    for seq in seqs: spans.append([offset,len(seq)]); offset+=len(seq)
    meta=dict(schema=SCHEMA,source=builder.reader.receipt()['source_sha256'],request=identity(builder.reader.request),
              frame=1,cut=cut,policy=dict(POLICY),factors=builder.factors,views=views,roles=roles,
              source_receipt=builder.reader.receipt(),aliases=aliases,sequences=spans)
    header=compact(meta); require(len(header)<=POLICY['header_bytes'], 'shared header cap')
    out=Writer(w); out.put(PREFIX.pack(MAGIC,len(header),len(nodes),l,r,len(roots),u,len(builder.scales))); out.put(header)
    li=ri=0
    for n,local in zip(nodes,locals_,strict=True):
        w.visit('row_visits'); out.put(NODE.pack(n[0],li,len(local),ri,len(n[2]))); li+=len(local); ri+=len(n[2])
    for entries in (locals_,[n[2] for n in nodes]):
        for local in entries:
            for j,q in local: w.visit('coefficient_visits'); out.put(ENTRY.pack(j,number(q)))
    for c,n in roots: w.visit('row_visits'); out.put(ROOT.pack(number(c),n))
    for seq in seqs:
        for j in seq: w.visit('row_visits'); out.put(INDEX.pack(j))
    for q in builder.scales: w.visit('coefficient_visits'); out.put(SCALAR.pack(number(q)))
    return out.finish()


def intake(packet, expected, work, snapshot):
    require(type(packet) is Packet and packet.identity==sha(expected), 'shared source packet identity')
    rd=Reader(packet,work); magic,h,n,l,r,q,u,s=rd.unpack(PREFIX)
    require(magic==MAGIC and 0<h<=POLICY['header_bytes'] and 0<n==q<=POLICY['rows']
            and l<=POLICY['nnz'] and r<=65536 and u<=POLICY['rows'] and 0<s<=POLICY['factors'], 'shared section caps')
    require(PREFIX.size+h+n*NODE.size+(l+r)*ENTRY.size+q*ROOT.size+u*4+s*8==packet.size, 'shared exact extent')
    hr=rd.read(h); m=decode(hr)
    require(type(m) is dict and compact(m)==hr and set(m)=={'schema','source','request','frame','cut','policy',
            'factors','views','roles','source_receipt','aliases','sequences'} and m['schema']==SCHEMA, 'shared metadata')
    descriptors=[]; li=ri=0
    for _ in range(n):
        work.visit('row_visits'); owner,ls,lc,rs,rc=rd.unpack(NODE)
        require(ls==li and rs==ri and lc<=l-li and rc<=r-ri, 'shared contiguous sections')
        descriptors.append((owner,lc,rc)); li+=lc; ri+=rc
    require(li==l and ri==r, 'all node sections')
    parts=[]
    for z in (1,2):
        bank=[]
        for desc in descriptors:
            row=[]
            for _ in range(desc[z]):
                work.visit('coefficient_visits'); j,x=rd.unpack(ENTRY); row.append((j,value(x)))
            bank.append(tuple(row))
        parts.append(bank)
    require(type(m['aliases']) is list and len(m['aliases'])<=n, 'alias inventory')
    last=-1
    for alias in m.pop('aliases'):
        work.visit('row_visits')
        require(type(alias) is list and len(alias)==5 and type(alias[0]) is int and last<alias[0]<n
                and type(alias[1]) is str and all(type(i) is int and i>=0 for i in alias[2:]), 'ordered source alias')
        j,graph,layer,row,width=alias; last=j; source=snapshot.row((graph,layer,row))
        require(width==source.width and not parts[1][j], 'source row width/no references')
        parts[0][j]=SourceLocal(source,parts[0][j],work)
    roots=[]
    for i in range(q):
        work.visit('row_visits'); c,j=rd.unpack(ROOT); require(j==i, 'unique root order'); roots.append((value(c),j))
    roots=RootBank(tuple(roots),work); ids=[]
    for _ in range(u):
        work.visit('row_visits'); j=rd.unpack(INDEX)[0]; require(j<q, 'membership root'); ids.append(j)
    require(set(ids)==set(range(q)), 'all roots used')
    spans=m.pop('sequences'); require(type(spans) is list and 0<len(spans)<=3*POLICY['views'], 'sequence count')
    sequences=[]; offset=0; unique=set()
    for span in spans:
        require(type(span) is list and len(span)==2 and all(type(i) is int for i in span), 'sequence span')
        start,count=span; require(start==offset and 0<=count<=POLICY['view_rows'] and start+count<=u, 'complete sequence span')
        seq=tuple(ids[start:start+count]); require(seq not in unique, 'canonical unique sequences')
        unique.add(seq); sequences.append(RootSequence(roots,seq)); offset+=count
    require(offset==u, 'no unused sequence suffix')
    scales=[]
    for _ in range(s):
        work.visit('coefficient_visits'); x=value(rd.unpack(SCALAR)[0]); require(x>=0, 'nonnegative scale'); scales.append(x)
    require(rd.p==packet.size and type(m['views']) is list and 0<len(m['views'])<=POLICY['views'], 'complete forest/views')
    views=[]; logical=0; seen=set()
    for v in m['views']:
        require(type(v) is dict and set(v)=={'name','scope','parent','op','factors','output','eq','ine'}, 'shared view fields')
        row=dict(v)
        for kind in ('output','eq','ine'):
            sid=v[kind]; require(type(sid) is int and 0<=sid<len(sequences), 'sequence identity')
            if sid not in seen: require(sid==len(seen), 'sequence first-use order'); seen.add(sid)
            seq=sequences[sid]; logical+=len(seq); require(logical<=POLICY['rows'], 'unchanged expanded membership cap')
            for _ in seq: work.visit('row_visits')
            row[kind]=seq
        views.append(row)
    require(len(seen)==len(sequences), 'all sequences used')
    nodes=tuple((d[0],parts[0][i],parts[1][i]) for i,d in enumerate(descriptors))
    counts=dict(unique_roots=q,memberships=logical,stored_memberships=u,sequence_count=len(sequences),
                physical_bytes=packet.size,packed_source_bytes=snapshot.packed_bytes,
                alias_rows=sum(type(a) is SourceLocal for a in parts[0]),explicit_locals=l,
                aliased_locals=sum(a.row.nonzero for a in parts[0] if type(a) is SourceLocal))
    return m,nodes,views,tuple(scales),counts
