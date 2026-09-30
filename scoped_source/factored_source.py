"""Chunked declared-source syntax. No LP constructor, capture, model or solver.

The new manifest identity is NOT the old canonical full-document identity.
Tensor/model-state identities retain the original binary64 byte convention.
Only one affine layer's coefficients are resident while scalar nodes are read.
"""
import base64
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import math
from pathlib import Path
import struct
from scoped_source.factored_io import read, load, write, inventory, required_hash, HEADER_LIMIT, BLOCK_LIMIT
from source_enclosure.format import identity, compact

SCHEMA = 'H2_CHUNKED_SOURCE_V1'
SOURCE_TOTAL_LIMIT = 256*2**20
TENSOR_ELEMENTS = 1_000_000  # unchanged parser admission, not an unbounded blob


def pack(doc, root, tick, chunk_bytes=BLOCK_LIMIT):
    """Untrusted writer from an existing declaration; not streaming live capture."""
    if type(chunk_bytes) is not int or not 8 <= chunk_bytes <= BLOCK_LIMIT or chunk_bytes % 8:
        raise ValueError('aligned bounded chunks')
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    tensors = {'@center':doc['center']}; graphs = []
    for graph in doc['networks']:
        layers = []
        for original in graph['layers']:
            tick(); layer = {k:deepcopy(v) for k,v in original.items() if k not in ('weight','bias')}
            if original['kind'] == 'Linear':
                for role in ('weight','bias'):
                    name = original[role+'_name']
                    if name in tensors: raise ValueError('duplicate logical tensor')
                    tensors[name] = original[role]; layer[role] = {'tensor_ref':name}
            layers.append(layer)
        graphs.append({'name':graph['name'], 'layers':layers})
    header = {k:deepcopy(v) for k,v in doc.items() if k not in ('center','networks')}
    header['center'] = {'tensor_ref':'@center'}; header['networks'] = graphs
    records = []
    for i,(name,t) in enumerate(sorted(tensors.items())):
        tick(); raw = base64.b64decode(t['bytes'], validate=True); chunks = []
        for j,offset in enumerate(range(0,len(raw),chunk_bytes)):
            tick(); rec = write(root,f't/{i:05d}-{j:05d}.bin',raw[offset:offset+chunk_bytes],
                                limit=BLOCK_LIMIT,binary=True)
            chunks.append({**rec,'offset':offset})
        records.append({'name':name, 'dtype':t['dtype'], 'shape':deepcopy(t['shape']),
                        'byte_order':t['byte_order'], 'chunks':chunks})
    manifest = {'schema':SCHEMA, 'declaration':header, 'tensors':records}
    write(root,'manifest.json',manifest,HEADER_LIMIT); tick()
    return identity(manifest)


class Source:
    def __init__(self, root, expected, tick):
        self.root = Path(root); self.tick = tick; tick()
        m = load(root,'manifest.json',HEADER_LIMIT)
        if set(m) != {'schema','declaration','tensors'} or m['schema'] != SCHEMA or identity(m) != expected:
            raise ValueError('external chunked source identity')
        self.identity = expected; self.doc = m['declaration']; self.outputs = {}; self.consumed = False
        if set(self.doc) not in ({'schema','request','center','state_inventory','networks'},
                                {'schema','request','center','state_inventory','networks','trust'}):
            raise ValueError('source declaration fields')
        if self.doc['schema'] != 'SCOPED_DECLARED_TOP2_V1': raise ValueError('declared source schema')
        self.request = self.doc['request']; r = self.request
        if (set(r) != {'experts','classes','label','top_k','gate','tie_policy','training','center',
                       'radius','margin','clip','model_state'} or
                type(r['experts']) is not int or not 2 <= r['experts'] <= 64 or
                type(r['classes']) is not int or not 2 <= r['classes'] <= 1024 or
                type(r['label']) is not int or not 0 <= r['label'] < r['classes'] or
                type(r['top_k']) is not int or r['top_k'] != 2 or r['gate'] != 'SELECTED_SOFTMAX' or
                r['tie_policy'] != 'ANY_LEGAL_TOPK' or r['training'] is not False):
            raise ValueError('request semantics/dimensions')
        inv = self.doc['state_inventory']; self.state = {v['name']:v for v in inv}
        if [v['name'] for v in inv] != sorted(self.state): raise ValueError('state inventory duplicate/order')
        state_hash = hashlib.sha256(); count = 0
        for v in inv:
            if set(v) != {'name','dtype','shape','sha256'}: raise ValueError('state fields')
            self.tensor_shape(v)
            if (type(v['sha256']) is not str or len(v['sha256']) != 64 or
                    any(c not in '0123456789abcdef' for c in v['sha256'])): raise ValueError('tensor hash syntax')
            state_hash.update(v['name'].encode()); state_hash.update(v['sha256'].encode()); count += math.prod(v['shape'])
        if r['model_state'] != {'sha256':state_hash.hexdigest(),'tensor_count':len(inv),'parameter_count':count}:
            raise ValueError('model-state inventory identity')
        self.tensors = {v['name']:v for v in m['tensors']}
        if ([v['name'] for v in m['tensors']] != sorted(self.tensors) or
                set(self.tensors) != set(self.state)|{'@center'}): raise ValueError('tensor inventory')
        files = {'manifest.json'}
        for i,(name,v) in enumerate(self.tensors.items()):
            if set(v) != {'name','dtype','shape','byte_order','chunks'}: raise ValueError('tensor descriptor fields')
            self.tensor_shape(v)
            if v['byte_order'] != 'little': raise ValueError('tensor byte order')
            end = 0
            for j,chunk in enumerate(v['chunks']):
                if (set(chunk) != {'file','bytes','sha256','offset'} or
                        chunk['file'] != f't/{i:05d}-{j:05d}.bin' or
                        type(chunk['offset']) is not int or chunk['offset'] != end or
                        type(chunk['bytes']) is not int or not 0 < chunk['bytes'] <= BLOCK_LIMIT or chunk['bytes']%8):
                    raise ValueError('chunk ordering/coverage/size')
                required_hash(chunk['sha256']); end += chunk['bytes']; files.add(chunk['file'])
            if end != 8*math.prod(v['shape']): raise ValueError('complete tensor byte coverage')
        self.files = frozenset(files)
        self.total_bytes = inventory(root,self.files,SOURCE_TOTAL_LIMIT)
        if self.doc['center'] != {'tensor_ref':'@center'}: raise ValueError('input tensor reference')
        ci,center = self.tensor('@center'); self.shape = ci['shape']
        if ci != r['center'] or len(self.shape) < 2 or self.shape[0] != 1: raise ValueError('center binding')
        radius=F(r['radius']); lo,hi=map(F,r['clip']); margin=F(r['margin'])
        if radius < 0 or lo >= hi or margin < 0 or any(not lo <= x <= hi for x in center):
            raise ValueError('domain/margin')
        self.inputs = [(max(lo,x-radius),min(hi,x+radius)) for x in center]
        names = ['router']+[f'expert{i}' for i in range(r['experts'])]
        if [g['name'] for g in self.doc['networks']] != names: raise ValueError('full ordered graph inventory')
        self.used = set(); tick()

    @staticmethod
    def tensor_shape(v):
        shape=v['shape']
        if (v['dtype'] != 'torch.float64' or type(shape) is not list or not shape or
                any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) > TENSOR_ELEMENTS):
            raise ValueError('tensor dtype/shape limit')

    def tensor(self,name):
        v=self.tensors[name]; h=hashlib.sha256(); h.update(b'torch.float64'); h.update(compact(v['shape']))
        numbers=[]
        for chunk in v['chunks']:
            self.tick(); raw=read(self.root,chunk['file'],BLOCK_LIMIT,chunk['sha256'])
            if len(raw) != chunk['bytes']: raise ValueError('chunk actual byte count')
            h.update(raw)
            for (x,) in struct.iter_unpack('<d',raw):
                if not math.isfinite(x): raise ValueError('nonfinite tensor')
                numbers.append(F.from_float(x))
        ident={'dtype':'torch.float64','shape':v['shape'],'sha256':h.hexdigest()}
        if name != '@center' and self.state[name] != {'name':name,**ident}: raise ValueError('source tensor/state identity')
        self.tick(); return ident,numbers

    def nodes(self):
        """One traversal, no full graph/tensor bank materialization. Exhaust it."""
        if self.consumed: raise ValueError('source iterator already consumed')
        self.consumed=True; inputs=[f'input/{i}' for i in range(len(self.inputs))]
        for name,bounds in zip(inputs,self.inputs):
            self.tick(); yield name,{'kind':'input','bounds':bounds}
        for graph in self.doc['networks']:
            if set(graph) != {'name','layers'} or not graph['layers']: raise ValueError('network schema')
            shape=self.shape[:]; current=inputs[:]
            prefix='router' if graph['name']=='router' else 'experts.'+graph['name'][6:]
            for j,layer in enumerate(graph['layers']):
                self.tick(); kind=layer['kind']
                if type(layer['index']) is not int or layer['index'] != j or layer['training'] is not False:
                    raise ValueError('layer order/mode')
                if kind=='Flatten':
                    if set(layer) != {'index','training','kind','dimensions'} or layer['dimensions'] != [1,-1]:
                        raise ValueError('flatten contract')
                    shape=[1,math.prod(shape[1:])]; continue
                if kind=='ReLU':
                    if set(layer) != {'index','training','kind','inplace'} or layer['inplace'] is not False:
                        raise ValueError('ReLU contract')
                    following=[f"{graph['name']}/layer/{j}/value/{i}" for i in range(len(current))]
                    for key,parent in zip(following,current):
                        self.tick(); yield key,{'kind':'relu','parent':parent}
                elif kind=='Linear':
                    if set(layer) != {'index','training','kind','weight','bias','weight_name','bias_name'}:
                        raise ValueError('affine fields')
                    values={}; identities={}
                    for role in ('weight','bias'):
                        name=f'{prefix}.{j}.{role}'
                        if layer[role+'_name'] != name or layer[role] != {'tensor_ref':name} or name in self.used:
                            raise ValueError('graph parameter binding')
                        self.used.add(name); identities[role],values[role]=self.tensor(name)
                    dims=identities['weight']['shape']
                    if len(shape)!=2 or len(dims)!=2 or dims[1]!=shape[1] or identities['bias']['shape']!=[dims[0]]:
                        raise ValueError('affine dimensions')
                    n,width=dims; following=[]
                    for i in range(n):
                        self.tick(); key=f"{graph['name']}/layer/{j}/value/{i}"; following.append(key)
                        yield key,{'kind':'affine','bias':values['bias'][i],
                                   'terms':{current[k]:v for k,v in enumerate(values['weight'][i*width:(i+1)*width]) if v}}
                    shape=[1,n]; del values,identities
                else: raise ValueError('unsupported source operator')
                current=following
            expected=self.request['experts'] if graph['name']=='router' else self.request['classes']
            if shape != [1,expected]: raise ValueError('network endpoint shape')
            self.outputs[graph['name']]=current
        if self.used != set(self.state): raise ValueError('unused source tensor')
        self.tick()
