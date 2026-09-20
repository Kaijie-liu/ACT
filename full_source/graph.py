"""Declared graph semantics and parameter binding, no framework imports."""
import math
from fractions import Fraction as F
try:
    from router_check import tensor
    from local_check import conv_operator
except ImportError:
    from router_source.checker import tensor
    from upstream_source.checker import conv_operator

KINDS = ['Conv2d','ReLU','Conv2d','ReLU','AvgPool2d','Flatten','Linear','ReLU','Linear']


def validate(doc, source, prefix_experts):
    pair = [p['expert'] for p in prefix_experts]
    if doc['schema'] != 'DECLARED_FULL_EXPERTS_V1' or doc['request'] != source['request'] or doc['pair'] != pair:
        raise ValueError('full source request/pair identity')
    if [x['expert'] for x in doc['experts']] != pair: raise ValueError('expert coverage/order')
    inv = {v['name']:{k:v[k] for k in ('dtype','shape','sha256')} for v in source['state_inventory']}
    for e, prefix in zip(doc['experts'], prefix_experts):
        layers = e['layers']
        if [x['kind'] for x in layers] != KINDS or prefix['topology_inspected'] != KINDS:
            raise ValueError('full declared topology')
        for j, layer in enumerate(layers):
            if layer['index'] != j or layer['training'] is not False: raise ValueError('layer index/mode')
            if layer['kind'] in ('Conv2d','Linear'):
                for role in ('weight','bias'):
                    name = f"experts.{e['expert']}.{j}.{role}"
                    if layer[role+'_name'] != name or tensor(layer[role])[0] != inv.get(name):
                        raise ValueError('expert parameter binding')
        first = layers[0]
        if first['graph'] != prefix['graph'] or any(first[r] != prefix[r] for r in ('weight','bias')):
            raise ValueError('prefix-to-full graph/parameter mismatch')
    return pair


def operator(shape, layer):
    """Return shape plus exact sparse affine operator; ReLU/Flatten have no rows."""
    kind = layer['kind']
    if layer['training'] is not False: raise ValueError('eval only')
    if kind == 'Conv2d':
        wi, w = tensor(layer['weight']); bi, b = tensor(layer['bias']); g = layer['graph']
        if bi['shape'] != [wi['shape'][0]]: raise ValueError('conv bias shape')
        rows, bias = conv_operator(shape,g,w,wi['shape'],b)
        dims = [1,wi['shape'][0]] + [(shape[i+2]+2*g['padding'][i]-g['dilation'][i]*(wi['shape'][i+2]-1)-1)//g['stride'][i]+1 for i in (0,1)]
        return dims,rows,bias
    if kind == 'ReLU':
        if layer.get('inplace') is not False: raise ValueError('ReLU mode')
        return shape,None,None
    if kind == 'Flatten':
        if layer['dimensions'] != [1,-1] or shape[0] != 1: raise ValueError('flatten semantics')
        return [1, math.prod(shape[1:])],None,None
    if kind == 'AvgPool2d':
        k = layer['kernel']; stride=layer['stride']; padding=layer['padding']
        if (len(shape)!=4 or k != stride or padding != [0,0] or len(k)!=2 or
                any(type(v)is not int or v<=0 for v in k) or layer['ceil_mode'] is not False or
                layer['count_include_pad'] is not True or layer['divisor_override'] is not None or
                shape[2]%k[0] or shape[3]%k[1]): raise ValueError('pool semantics')
        _,c,h,w=shape; oh,ow=h//k[0],w//k[1]; rows=[]
        for z in range(c):
            for y in range(oh):
                for x in range(ow):
                    rows.append({(z*h+y*k[0]+dy)*w+x*k[1]+dx:F(1,k[0]*k[1])
                                 for dy in range(k[0]) for dx in range(k[1])})
        return [1,c,oh,ow],rows,[F(0)]*len(rows)
    if kind == 'Linear':
        wi,w=tensor(layer['weight']); bi,b=tensor(layer['bias'])
        if len(shape)!=2 or shape[0]!=1 or len(wi['shape'])!=2 or wi['shape'][1]!=shape[1] or bi['shape']!=[wi['shape'][0]]:
            raise ValueError('linear dimensions')
        n,m=wi['shape']; return [1,n],[{j:w[i*m+j] for j in range(m) if w[i*m+j]} for i in range(n)],b
    raise ValueError('unsupported declared operation')
