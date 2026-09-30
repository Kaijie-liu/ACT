"""Solver-free preparation check: bind the chunked bytes to the declaration.

This checks a source identity, not any output bound or graph/program equivalence.
Executed in an owned process with the same total preparation budget.
"""
import base64
from copy import deepcopy
import hashlib
import math
import struct
from scoped_source.factored_source import Source
from scoped_source.factored_io import read, load, inventory, BLOCK_LIMIT, HEADER_LIMIT, decode
from source_enclosure.format import identity, compact
from scoped_source.capacity_intake import expected_request, protocol


def check_source(root, candidate, case, tick):
    source=Source(root,candidate['source_manifest_sha256'],tick)
    expected_request(case,source.request)
    if source.request!=candidate['request']: raise ValueError('source request binding')
    if case=='full_size':
        for i,graph in enumerate(source.doc['networks']):
            widths=[3072,128,8] if i==0 else [3072,256,128,10]
            kinds=['Flatten']
            for n in range(len(widths)-1):
                kinds+=['Linear']+(['ReLU'] if n<len(widths)-2 else [])
            if [layer['kind'] for layer in graph['layers']]!=kinds: raise ValueError('factory layer recipe')
            affine=[layer for layer in graph['layers'] if layer['kind']=='Linear']
            for layer,ni,no in zip(affine,widths,widths[1:]):
                if source.tensors[layer['weight_name']]['shape']!=[no,ni]:
                    raise ValueError('factory hidden dimensions')
    nodes=0
    for _name,_node in source.nodes(): nodes+=1
    doc=deepcopy(source.doc); tensors={}; chunks=0; raw_bytes=0
    for name,descriptor in source.tensors.items():
        tick(); raw=bytearray(); h=hashlib.sha256()
        h.update(b'torch.float64'); h.update(compact(descriptor['shape']))
        for j,part in enumerate(descriptor['chunks']):
            tick(); data=read(root,part['file'],BLOCK_LIMIT,part['sha256'])
            if len(data)!=part['bytes']: raise ValueError('source chunk actual size')
            remaining=8*math.prod(descriptor['shape'])-part['offset']
            if part['bytes']!=min(protocol()['source_chunk_bytes'],remaining):
                raise ValueError('fixed chunk layout')
            for (x,) in struct.iter_unpack('<d',data):
                if not math.isfinite(x): raise ValueError('nonfinite source coefficient')
            raw.extend(data); h.update(data); chunks+=1; raw_bytes+=len(data)
        ident={'dtype':'torch.float64','shape':descriptor['shape'],'sha256':h.hexdigest()}
        expected=source.request['center'] if name=='@center' else {k:v for k,v in source.state[name].items() if k!='name'}
        if ident!=expected: raise ValueError('source tensor/state identity')
        tensors[name]={'dtype':'torch.float64','shape':descriptor['shape'],'byte_order':'little',
                       'bytes':base64.b64encode(raw).decode()}
    doc['center']=tensors['@center']; used={'@center'}
    for network in doc['networks']:
        for layer in network['layers']:
            tick()
            if layer['kind']!='Linear': continue
            for role in ('weight','bias'):
                name=layer[role+'_name']
                if layer[role]!={'tensor_ref':name} or name in used: raise ValueError('source tensor reference')
                used.add(name); layer[role]=tensors[name]
    if used!=set(tensors): raise ValueError('source unused tensor')
    if identity(doc)!=candidate['declared_source_sha256']: raise ValueError('full declaration identity')
    if case=='tiny_control':
        from scoped_source.endpoint_intake import expected_source
        if doc!=expected_source(): raise ValueError('independent tiny declaration')
    if candidate['chunks']!=chunks or candidate['source_bytes']!=source.total_bytes:
        raise ValueError('source cost inventory')
    if case=='full_size' and (len(tensors)!=53 or chunks!=95 or raw_bytes!=55715520):
        raise ValueError('registered full-size source inventory')
    tick()
    if identity(load(root,'manifest.json',HEADER_LIMIT))!=source.identity:
        raise ValueError('source manifest changed during checking')
    if inventory(root,source.files,256*2**20)!=source.total_bytes:
        raise ValueError('source members changed during checking')
    return {'nodes_checked':nodes,'tensor_count':len(tensors),'chunks':chunks,'source_bytes':source.total_bytes,
            'raw_tensor_bytes':raw_bytes,'declared_source_sha256':identity(doc),
            'source_manifest_sha256':source.identity,'proof_status':'NOT_A_PROOF'}
