"""Exact routing coverage for a declared real AvgPool/Flatten/Linear graph.

No HZ, model runtime or solver. Tensor bytes are matched to the existing
request/state identities; graph correspondence to the registered program is
explicit, not a claim about native floating-point execution.
"""
import base64
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import struct


def compact(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(raw): return hashlib.sha256(raw).hexdigest()


def tensor(value):
    if value['dtype']!='torch.float64' or value['byte_order']!='little':
        raise ValueError('only binary64 little-endian tensors supported')
    shape=value['shape']
    if not shape or any(type(n)is not int or n<=0 for n in shape) or math.prod(shape)>1000000:
        raise ValueError('unsupported tensor dimensions')
    raw=base64.b64decode(value['bytes'],validate=True)
    if len(raw)!=8*math.prod(shape): raise ValueError('tensor byte length')
    h=hashlib.sha256();h.update(b'torch.float64')
    h.update(json.dumps(shape,separators=(',',':')).encode());h.update(raw)
    identity={'dtype':'torch.float64','shape':shape,'sha256':h.hexdigest()}
    numbers=[x[0] for x in struct.iter_unpack('<d',raw)]
    if any(not math.isfinite(x) for x in numbers): raise ValueError('nonfinite tensor')
    return identity,[F.from_float(x) for x in numbers]


def inputs(doc, expected_request):
    if doc['schema']!='AFFINE_ROUTER_SOURCE_V1' or doc['request']!=expected_request:
        raise ValueError('request/source identity mismatch')
    r=expected_request
    if r['top_k']!=2 or r['tie_policy']!='ANY_LEGAL_TOPK' or not 2<=r['experts']<=64:
        raise ValueError('unsupported routing semantics')
    state=doc['state_inventory']; names=[v['name'] for v in state]
    if names!=sorted(set(names)) or len(state)!=r['model_state']['tensor_count']:
        raise ValueError('state inventory missing/duplicate')
    h=hashlib.sha256();count=0
    for item in state:
        if (item['dtype']!='torch.float64' or not item['shape'] or
            any(type(n)is not int or n<=0 for n in item['shape']) or
            len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('state identity format')
        count+=math.prod(item['shape'])
        h.update(item['name'].encode());h.update(item['sha256'].encode())
    if h.hexdigest()!=r['model_state']['sha256'] or count!=r['model_state']['parameter_count']:
        raise ValueError('full parameter identity mismatch')
    image={}
    for name in ('center','lower','upper'):
        ident,numbers=tensor(doc['input'][name])
        if ident!=r[name]: raise ValueError('materialized input changed')
        image[name]=numbers
    shape=r['lower']['shape']
    if (len(shape)!=4 or shape[0]!=1 or r['center']['shape']!=shape or r['upper']['shape']!=shape or
            any(not a<=c<=b for a,c,b in zip(image['lower'],image['center'],image['upper']))):
        raise ValueError('input box/order/shape')
    graph=doc['graph'];pool=graph['pool']
    if (graph['schema']!='REAL_NONOVERLAP_AVGPOOL_FLATTEN_LINEAR_V1' or
            type(pool)is not int or pool<1 or shape[2]%pool or shape[3]%pool or
            graph['pool_stride']!=pool or graph['pool_padding']!=0 or
            graph['ceil_mode'] is not False or graph['count_include_pad'] is not True or
            graph['divisor_override'] is not None or graph['flatten']!=[1,-1] or
            graph['training'] is not False):
        raise ValueError('unsupported graph semantics')
    parameter={}
    inventory={v['name']:{k:v[k] for k in ('dtype','shape','sha256')} for v in state}
    if set(doc['parameters'])!={graph['weight'],graph['bias']}:
        raise ValueError('parameter graph inventory')
    for role in ('weight','bias'):
        name=graph[role];ident,numbers=tensor(doc['parameters'][name])
        if inventory.get(name)!=ident: raise ValueError('router parameters not in model state')
        parameter[role]=numbers
    e=r['experts'];width=shape[1]*(shape[2]//pool)*(shape[3]//pool)
    if (doc['parameters'][graph['weight']]['shape']!=[e,width] or
            doc['parameters'][graph['bias']]['shape']!=[e]):
        raise ValueError('router graph matrix dimensions')
    return shape,pool,parameter,image


def exact_margins(doc, expected_request):
    shape,pool,parameters,image=inputs(doc,expected_request)
    _,channels,height,width=shape
    ph,pw=height//pool,width//pool;features=channels*ph*pw
    weights,bias=parameters['weight'],parameters['bias'];result={}
    # Checker distributes each weight over ORIGINAL pixels, not producer's
    # precomputed pooled intervals or HZ coefficients.
    for a,b in itertools.combinations(range(expected_request['experts']),2):
        lower=upper=bias[a]-bias[b]
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    feature=(c*ph+y//pool)*pw+x//pool
                    coefficient=(weights[a*features+feature]-weights[b*features+feature])/(pool*pool)
                    i=(c*height+y)*width+x
                    v,w=coefficient*image['lower'][i],coefficient*image['upper'][i]
                    lower+=min(v,w);upper+=max(v,w)
        result[(a,b)]=(lower,upper)
    return result


def check(doc, proof, *, expected_request, expected_source_sha256):
    if digest(compact(doc))!=expected_source_sha256 or proof['source_sha256']!=expected_source_sha256:
        raise ValueError('source binding changed')
    margins=exact_margins(doc,expected_request)
    if proof['schema']!='AFFINE_ROUTER_BOX_COVER_V1':raise ValueError('proof schema')
    actual=[{'pair':list(pair),'lower':str(lo),'upper':str(hi)} for pair,(lo,hi) in margins.items()]
    if proof['margins']!=actual: raise ValueError('incorrect exact score bounds')
    rows=proof['routes'];all_pairs=list(itertools.combinations(range(expected_request['experts']),2))
    if [tuple(row['pair']) for row in rows]!=all_pairs: raise ValueError('route inventory missing/duplicate/order')
    covered=[];excluded=[]
    for row in rows:
        pair=row['pair']
        if row['kind']=='covered':
            if set(row)!={'pair','kind'}: raise ValueError('ambiguous cover')
            covered.append(pair);continue
        if row['kind']!='excluded':raise ValueError('unknown route disposition')
        outside,inside=row['witness']
        if inside not in pair or outside in pair or not 0<=outside<expected_request['experts']:
            raise ValueError('invalid exclusion witness')
        key=tuple(sorted((outside,inside)));bounds=margins[key]
        lower=bounds[0] if outside<inside else -bounds[1]
        # A tie is legal: zero never excludes a route, no epsilon/tolerance.
        if lower<=0 or row['strict_lower_bound']!=str(lower):
            raise ValueError('exclusion lacks strict exact dominance')
        excluded.append(pair)
    if not covered:raise ValueError('empty cover inconsistent with nonempty box')
    return {'status':'CHECKED_ROUTER_COVER_FOR_DECLARED_REAL_GRAPH',
        'covered_pairs':covered,'excluded_pairs':excluded,'all_pairs':len(all_pairs),
        'margins':actual,'source_sha256':expected_source_sha256,
        'scope':'All real top2 routes covered on pinned represented box. Exclusion proofs use original router parameters, no HZ/solver. Declared graph correspondence and deployed floating-point semantics are separate.'}
