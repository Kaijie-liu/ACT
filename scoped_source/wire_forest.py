"""Opt-in lossless V3 wire forest; bounded lazy rows, independent source checks.

Only stored binary64 rationals are encodable. Logical V3 limits/identity remain
in force. This is NOT a bounded source producer or a native/GPU memory contract.
"""
from fractions import Fraction as F
import hashlib
import math
import struct
from types import MappingProxyType

from scoped_source.hz_request_store import POLICY, compact, decode, digest, identity, rational, require, sha
from scoped_source.factored_io import read, BLOCK_LIMIT, HEADER_LIMIT
from scoped_source.tensor_rows import TensorRows, _bounded_inventory

MAGIC=b'HZWF0001'
PREFIX=struct.Struct('<8sIIII')
ROOT=struct.Struct('<dI4s')
NODE=struct.Struct('<IIIIIIIIQQ16s')
ENTRY=struct.Struct('<Id4s')
ZERO4=b'\0'*4
ZERO16=b'\0'*16


def float_exact(q):
    value=float(q)
    require(math.isfinite(value) and F(value)==q, 'wire requires exact finite binary64')
    return value


def unpack_float(value):
    require(math.isfinite(value) and (value!=0 or math.copysign(1,value)>0),
            'nonfinite/noncanonical wire scalar')
    return F(value)


class SourceRows:
    """No tensor bank: reread/hash bounded chunks for a single referenced row.

    No cache survives an access, so subsequent disk pollution cannot silently
    reuse stale parsed values. TensorRows separately checks the entire source.
    """
    def __init__(self,root,source,work):
        self.reader=TensorRows(root,source,deadline=work.deadline)
        self.root=root; self.work=work
        self.tensors=tuple(self.reader._tensors.values())
        self.inventory=tuple((t.name,t.shape,t.sha256) for t in self.tensors)
        self.stats=dict(chunk_reads=0,raw_bytes=0,row_reads=0,max_row_values=0)

    def seal(self):
        """Timely full identity revalidation before issuing an output terminal."""
        w=self.work; r=self.reader; w.tick()
        read(self.root,'manifest.json',HEADER_LIMIT,r._manifest_digest)
        require(_bounded_inventory(r._root,r._files,w.tick)==r._total_bytes,'terminal source inventory')
        for tensor in self.tensors:
            h=hashlib.sha256(b'torch.float64'+compact(list(tensor.shape)))
            for chunk in tensor.chunks:
                w.visit('row_visits'); raw=read(self.root,chunk.file,BLOCK_LIMIT,chunk.sha256)
                require(len(raw)==chunk.bytes,'terminal source chunk length'); h.update(raw)
                self.stats['chunk_reads']+=1; self.stats['raw_bytes']+=len(raw); w.tick()
            require(h.hexdigest()==tensor.sha256,'terminal tensor identity')
        read(self.root,'manifest.json',HEADER_LIMIT,r._manifest_digest)
        require(_bounded_inventory(r._root,r._files,w.tick)==r._total_bytes,'terminal source final inventory')
        w.tick()

    def row(self,tensor,row,width):
        work=self.work; work.tick(); t=self.tensors[tensor]
        require(len(t.shape)==2 and t.shape[1]==width and 0<=row<t.shape[0], 'wire source row dimensions')
        begin=row*width*8; end=begin+width*8; offset=0; values=[]
        for chunk in t.chunks:
            work.tick(); stop=offset+chunk.bytes
            if offset<end and begin<stop:
                raw=read(self.root,chunk.file,BLOCK_LIMIT)
                work.tick(); require(len(raw)==chunk.bytes and digest(raw)==chunk.sha256,'wire source chunk identity')
                self.stats['chunk_reads']+=1; self.stats['raw_bytes']+=len(raw)
                lo=max(begin,offset)-offset; hi=min(end,stop)-offset
                for (value,) in struct.iter_unpack('<d',memoryview(raw)[lo:hi]):
                    work.visit('coefficient_visits'); require(math.isfinite(value),'wire source finite coefficient')
                    values.append(F(value))  # Source signed zero keeps its byte identity.
            offset=stop
        require(len(values)==width,'complete wire source row')
        self.stats['row_reads']+=1; self.stats['max_row_values']=max(self.stats['max_row_values'],len(values))
        return tuple(values)


def source_nodes(doc,source):
    """Reconstruct eligible first-affine nodes from bound graph/view metadata."""
    views=doc['views']; inputs=views[0]['output']; result={}
    tensors={t.name:i for i,t in enumerate(source.tensors)}
    layers={f'{l.graph}/{l.index}':l for l in source.reader.plan if l.kind=='Linear'}
    for owner,v in enumerate(views):
        if v['op']!='Linear' or v['parent'] is None or views[v['parent']]['output']!=inputs: continue
        require(v['name'] in layers,'source descriptor graph/layer')
        layer=layers[v['name']]
        require(len(v['output'])==layer.outputs and layer.inputs==len(inputs),'source descriptor shape')
        for row,(_,n) in enumerate(v['output']):
            require(n not in result,'duplicate source descriptor node')
            result[n]=(owner,tensors[layer.weight],row,layer.inputs)
    return result


def encode(raw,root,source,*,expected,work):
    """Migration encoder consumes old V3 JSON; NOT a bounded live producer."""
    work.tick(); require(type(raw) is bytes and len(raw)<=POLICY['total_bytes'] and digest(raw)==sha(expected),'old forest identity/cap')
    doc=decode(raw); require(compact(doc)==raw,'old canonical forest')
    require(doc['schema']=='SOURCE_SCALED_REFERENCE_FOREST_V3' and doc['source']==source,'wire V3/source')
    rows=SourceRows(root,source,work); refs=source_nodes(doc,rows)
    meta={k:v for k,v in doc.items() if k not in ('nodes','scales','views')}
    views=[]; roots=bytearray()
    for v in doc['views']:
        v=dict(v)
        for kind in ('output','eq','ine'):
            old=v[kind]; v[kind]=len(old)
            for q,n in old:
                work.tick(); roots.extend(ROOT.pack(float_exact(rational(q)),n,ZERO4))
        views.append(v)
    meta.update(views=views,logical=expected,tensors=rows.inventory,
                input_factors=identity(doc['factors'][:len(rows.reader.inputs)]))
    header=compact(meta); require(len(header)<=POLICY['header_bytes'],'wire metadata cap')
    scales=b''.join(struct.pack('<d',float_exact(rational(q))) for q in doc['scales'])
    descriptors=bytearray(); payload=bytearray()
    for ni,(owner,local,parents) in enumerate(doc['nodes']):
        work.tick(); local=[(j,rational(q)) for j,q in local]; parents=[(j,rational(q)) for j,q in parents]
        if ni in refs:
            o,t,row,width=refs[ni]; require(o==owner,'wire source owner')
            values=rows.row(t,row,width)
            require([(j,q) for j,q in local if j<width]==[(j,q) for j,q in enumerate(values) if q],'source row exact coefficients')
            tail=[(j,q) for j,q in local if j>=width]; kind=1
        else: t=row=width=0; tail=local; kind=0
        lo=len(payload)
        for j,q in tail: payload.extend(ENTRY.pack(j,float_exact(q),ZERO4))
        ro=len(payload)
        for j,q in parents: payload.extend(ENTRY.pack(j,float_exact(q),ZERO4))
        descriptors.extend(NODE.pack(owner,len(local),len(parents),kind,t,row,width,len(tail),lo,ro,ZERO16))
    result=PREFIX.pack(MAGIC,len(header),len(doc['nodes']),len(roots)//ROOT.size,len(doc['scales']))+header+scales+roots+descriptors+payload
    work.tick(); require(len(result)<=POLICY['total_bytes'],'wire forest cap'); return result


class NodeTable:
    """Immutable wire bytes, row-at-a-time decoding; never a coefficient bank."""
    __slots__=('raw','source','work','count','base','payload','expected','_sealed')

    def __setattr__(self,k,v):
        if getattr(self,'_sealed',False): raise AttributeError('immutable wire table')
        object.__setattr__(self,k,v)

    def __delattr__(self,k): raise AttributeError('immutable wire table')

    def __init__(self,raw,source,work,count,base,payload,expected):
        self.raw=raw; self.source=source; self.work=work; self.count=count
        self.base=base; self.payload=payload; self.expected=MappingProxyType(expected); self._sealed=True

    def __len__(self): return self.count

    def descriptor(self,i):
        self.work.visit('row_visits'); require(type(i) is int and 0<=i<self.count,'wire node index')
        return NODE.unpack_from(self.raw,self.base+i*NODE.size)

    def __getitem__(self,i):
        if type(i) is int and i==self.count: raise IndexError(i)
        owner,lc,rc,kind,t,row,width,tc,lo,ro,pad=self.descriptor(i)
        local=[]
        if kind:
            local.extend((j,q) for j,q in enumerate(self.source.row(t,row,width)) if q)
        for pos in range(tc):
            self.work.visit('coefficient_visits'); j,value,_=ENTRY.unpack_from(self.raw,self.payload+lo+pos*ENTRY.size)
            local.append((j,unpack_float(value)))
        refs=[]
        for pos in range(rc):
            self.work.visit('coefficient_visits'); j,value,_=ENTRY.unpack_from(self.raw,self.payload+ro+pos*ENTRY.size)
            refs.append((j,unpack_float(value)))
        require(len(local)==lc,'wire logical local count')
        return owner,tuple(local),tuple(refs)

    def document_row(self,i):
        o,local,refs=self[i]
        return [o,[[j,str(q)] for j,q in local],[[j,str(q)] for j,q in refs]]


def logical_chunks(doc,nodes):
    """Canonical old JSON, bounded one-row pieces, never assembled by intake."""
    yield b'{'
    for k,key in enumerate(sorted(set(doc)|{'nodes'})):
        if k: yield b','
        yield compact(key)+b':'
        if key=='nodes':
            yield b'['
            for i in range(len(nodes)):
                if i: yield b','
                yield compact(nodes.document_row(i))
            yield b']'
        elif key=='views':
            yield b'['
            for i,v in enumerate(doc[key]):
                if i: yield b','
                yield compact(v)
            yield b']'
        else: yield compact(doc[key])
    yield b'}'


def intake(raw,root,source,expected,work):
    work.tick(); require(type(raw) is bytes and PREFIX.size<=len(raw)<=POLICY['total_bytes'] and digest(raw)==sha(expected),'wire bytes/hash')
    magic,hs,nc,rc,fc=PREFIX.unpack_from(raw)
    require(magic==MAGIC and 0<hs<=POLICY['header_bytes'] and 0<nc<=POLICY['rows'] and
            rc<=POLICY['rows'] and 0<fc<=POLICY['factors'],'wire fixed header capacities')
    base=PREFIX.size+hs; nodebase=base+fc*8+rc*ROOT.size; payload=nodebase+nc*NODE.size
    require(payload<=len(raw),'truncated wire layout')
    header=raw[PREFIX.size:base]; m=decode(header)
    require(compact(m)==header and type(m) is dict and set(m)=={
        'schema','source','request','frame','cut','policy','factors','views','roles','source_receipt',
        'logical','tensors','input_factors'},'wire metadata schema')
    require(m['source']==sha(source) and len(m['factors'])==fc,'wire source/factors')
    logical=sha(m['logical']); rows=SourceRows(root,source,work)
    require(m['tensors']==decode(compact(rows.inventory)) and m['input_factors']==identity(m['factors'][:len(rows.reader.inputs)]),'wire tensor/factor binding')
    doc={k:v for k,v in m.items() if k not in ('logical','tensors','input_factors')}
    doc['scales']=[]
    for i in range(fc):
        work.tick(); doc['scales'].append(str(unpack_float(struct.unpack_from('<d',raw,base+i*8)[0])))
    require(type(doc['views']) is list and 0<len(doc['views'])<=POLICY['views'],'wire view cap')
    index=0
    for v in doc['views']:
        for kind in ('output','eq','ine'):
            count=v[kind]; require(type(count) is int and 0<=count<=POLICY['view_rows'] and index+count<=rc,'wire root coverage')
            result=[]
            for _ in range(count):
                work.tick(); value,n,pad=ROOT.unpack_from(raw,base+fc*8+index*ROOT.size); index+=1
                require(pad==ZERO4 and n<nc,'wire root padding/index')
                result.append([str(unpack_float(value)),n])
            v[kind]=result
    require(index==rc,'wire complete root roster')
    expected_nodes=source_nodes(doc,rows)
    nodes=NodeTable(raw,rows,work,nc,nodebase,payload,expected_nodes); end=nnz=refs=0
    for ni in range(nc):
        work.tick(); o,lc,pc,kind,t,row,width,tc,lo,ro,pad=nodes.descriptor(ni)
        require(o<len(doc['views']) and kind in (0,1) and pad==ZERO16 and lo==end and
                ro==lo+tc*ENTRY.size,'wire descriptor/contiguous layout')
        if kind:
            require(expected_nodes.get(ni)==(o,t,row,width) and tc<=2 and lc<=width+2,'wire source graph/row/owner')
        else:
            # V3 input=1, affine compensation<=2, ReLU<=3+1 lift,
            # reference guard=0. This is an implied format bound, not a new
            # relaxed gate; refuse before building a malicious giant row.
            require(ni not in expected_nodes and (t,row,width)==(0,0,0) and lc==tc and tc<=4,'wire inline descriptor')
        require(pc<=4096,'wire reference row width')  # TensorRows width bound.
        nnz+=lc; refs+=pc; end=ro+pc*ENTRY.size
        require(nnz<=POLICY['nnz'] and refs<=65536 and payload+end<=len(raw),'wire nnz/reference/byte capacity')
        for offset in range(lo,end,ENTRY.size):
            work.tick(); j,value,padding=ENTRY.unpack_from(raw,payload+offset)
            require(padding==ZERO4,'wire entry padding'); unpack_float(value)
    require(payload+end==len(raw),'wire trailing data')
    h=hashlib.sha256(); size=0
    for chunk in logical_chunks(doc,nodes):
        work.tick(); size+=len(chunk); require(size<=POLICY['total_bytes'],'logical forest cap')
        h.update(chunk)
    require(h.hexdigest()==logical,'logical V3 identity'); work.tick()
    return doc,nodes,size,logical


# Import after the representation definitions to keep the closed intake cycle safe.
from scoped_source.check_scaled_references import Checker, CheckedScaledForest
from scoped_source.forest_endpoints import frozen


class CheckedWireForest(CheckedScaledForest):
    __slots__=('wire_identity','wire_bytes','source_access')

    def __init__(self,raw,root,source,*,expected,work,cut=None):
        doc,nodes,size,logical=intake(raw,root,source,expected,work)
        checker=Checker.__new__(Checker)
        checker.initialize(doc,root,source,work,cut,forest_bytes=size,node_table=nodes)
        result=checker.run()
        self.identity=logical; self.wire_identity=sha(expected); self.wire_bytes=len(raw)
        self.source=sha(source); self.request=result['request']; self.declaration=frozen(checker.reader.request)
        self.factors=checker.factors; self.nodes=nodes; self.views=frozen(checker.views); self.roles=frozen(checker.roles)
        self.norms=tuple(checker.norms); self.scales=checker.scales; self.receipt=frozen(result)
        self.complete=cut is None; self.session=work; self.source_access=nodes.source
        self.factor_identity=identity(dict(factors=self.factors,scales=[str(q) for q in self.scales]))
        self._sealed=True; work.tick()
