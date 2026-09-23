"""Finite declared Linear/ReLU/Flatten MoE graph, exact stored coefficients.

Shared parser only. Does not produce HZs, accept certificates or run a model.
Graph/program correspondence remains a separate declared-source assumption.
"""
from fractions import Fraction as F
import hashlib
import math
import time
from router_source.checker import tensor
from source_enclosure.format import identity


def clock(deadline):
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline-time.monotonic()>300:
        raise ValueError('one finite at-most-300s deadline required')
    def tick():
        if time.monotonic()>=deadline:raise TimeoutError('source construction/check deadline')
    tick();return tick


def validate(doc, expected, tick):
    tick()
    if identity(doc)!=expected or doc['schema']!='SCOPED_DECLARED_TOP2_V1':raise ValueError('external source identity')
    r=doc['request'];e,c=r['experts'],r['classes']
    if (type(e)is not int or not 2<=e<=64 or type(c)is not int or c<2 or type(r['label'])is not int or
        not 0<=r['label']<c or r['top_k']!=2 or r['gate']!='SELECTED_SOFTMAX' or
        r['tie_policy']!='ANY_LEGAL_TOPK' or r['training'] is not False):raise ValueError('request semantics')
    ci,center=tensor(doc['center'])
    if ci!=r['center'] or ci['shape'][0]!=1 or len(ci['shape'])<2:raise ValueError('source input binding')
    radius=F(r['radius']);lo,hi=map(F,r['clip']);margin=F(r['margin'])
    if radius<0 or lo>=hi or margin<0 or any(not lo<=x<=hi for x in center):raise ValueError('domain/margin')
    shape=ci['shape'];names=['router']+[f'expert{i}' for i in range(e)]
    if [g['name'] for g in doc['networks']]!=names:raise ValueError('all router/expert graphs required')
    inv=doc['state_inventory'];by_name={v['name']:v for v in inv}
    if [v['name'] for v in inv]!=sorted(by_name):raise ValueError('state inventory duplicate/order')
    state_hash=hashlib.sha256();numel=0;used=set()
    for v in inv:
        state_hash.update(v['name'].encode());state_hash.update(v['sha256'].encode());numel+=math.prod(v['shape'])
    if r['model_state']!={'sha256':state_hash.hexdigest(),'tensor_count':len(inv),'parameter_count':numel}:
        raise ValueError('complete parameter identity')
    for g in doc['networks']:
        dims=shape
        if not g['layers']:raise ValueError('empty network')
        prefix='router' if g['name']=='router' else 'experts.'+g['name'][6:]
        for j,layer in enumerate(g['layers']):
            tick()
            if layer['index']!=j or layer['training'] is not False:raise ValueError('layer index/mode')
            if layer['kind']=='Linear':
                for role in ('weight','bias'):
                    name=f'{prefix}.{j}.{role}';ident,_=tensor(layer[role])
                    if layer[role+'_name']!=name or by_name.get(name)!={'name':name,**ident} or name in used:
                        raise ValueError('parameter graph binding')
                    used.add(name)
            dims,_,_=operator(dims,layer)
        if dims!=[1,e if g['name']=='router' else c]:raise ValueError('incomplete network endpoint')
    if used!=set(by_name):raise ValueError('unused/missing state tensor')
    tick()
    return r,[max(lo,x-radius) for x in center],[min(hi,x+radius) for x in center]


def operator(shape, layer):
    kind=layer['kind']
    if kind=='Flatten':
        if layer['dimensions']!=[1,-1] or shape[0]!=1:raise ValueError('flatten order')
        return [1,math.prod(shape[1:])],None,None
    if kind=='ReLU':
        if layer['inplace'] is not False:raise ValueError('ReLU mode')
        return list(shape),None,None
    if kind!='Linear':raise ValueError('unsupported source operator')
    wi,w=tensor(layer['weight']);bi,b=tensor(layer['bias'])
    if len(shape)!=2 or len(wi['shape'])!=2 or wi['shape'][1]!=shape[1] or bi['shape']!=[wi['shape'][0]]:
        raise ValueError('affine dimensions')
    n,m=wi['shape']
    return [1,n],[{j:w[i*m+j] for j in range(m) if w[i*m+j]} for i in range(n)],b
