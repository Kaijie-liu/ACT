"""Untrusted source/guard/output-LP construction on shared exact factors.

No solver, old certificate, pair exclusion or SAFE-returning entry exists here.
"""
import copy
from fractions import Fraction as F
from itertools import combinations
from source_enclosure.format import unpack,pack,empty,identity,clean
from source_enclosure.produce import box,relu
from full_source.lift import affine
from full_source.obligations import build as lp_build
from scoped_source.graph import validate,operator,clock


def join(base, states):
    s,c0,b0=unpack(base);parts=[unpack(v) for v in states]
    ci=c0+[v for _,c,_ in parts for v in c[len(c0):]]
    bi=b0+[v for _,_,b in parts for v in b[len(b0):]]
    if len(set(ci+bi))!=len(ci+bi):raise ValueError('private factors alias')
    cm={v:j for j,v in enumerate(ci)};bm={v:j for j,v in enumerate(bi)};h=empty(0,s['frame_id'])
    for a,c,b in parts:
        h['c']+=a['c']
        for key,m,ids in [('Gc',cm,c),('Gb',bm,b)]:
            h[key]+=[{m[ids[j]]:v for j,v in row.items()} for row in a[key]]
    for ck,bk,rhs in [('Ac','Ab','b'),('Auc','Aub','ub')]:
        h[ck]=copy.deepcopy(s[ck]);h[bk]=copy.deepcopy(s[bk]);h[rhs]=list(s[rhs]);n=len(s[rhs])
        for a,c,b in parts:
            for key,m,ids in [(ck,cm,c),(bk,bm,b)]:
                h[key]+=[{m[ids[j]]:v for j,v in row.items()} for row in a[key][n:]]
            h[rhs]+=a[rhs][n:]
    return pack(h,ci,bi)


def guards(state,pair,experts):
    h,c,b=unpack(state);h=copy.deepcopy(h)
    for i in pair:
        for k in range(experts):
            if k in pair:continue
            for dst,src in [('Auc','Gc'),('Aub','Gb')]:
                h[dst].append(clean({j:h[src][k].get(j,F(0))-h[src][i].get(j,F(0))
                    for j in h[src][k].keys()|h[src][i].keys()}))
            h['ub'].append(h['c'][i]-h['c'][k])
    return pack(h,c,b)


def project(state,experts,classes,label,margin):
    h,c,b=unpack(state);h=copy.deepcopy(h)
    for k in ('c','Gc','Gb'):h[k]=h[k][experts:]
    if len(h['c'])!=2*classes:raise ValueError('pair output layout')
    for offset in (0,classes):h['c'][offset+label]-=F(margin)
    return pack(h,c,b)


def build(doc,*,expected_source_sha256,deadline):
    tick=clock(deadline);r,lo,hi=validate(doc,expected_source_sha256,tick)
    root=box(lo,hi);trace=[];ends={}
    for graph in doc['networks']:
        state=root;shape=r['center']['shape'];steps=[]
        for layer in graph['layers']:
            tick();before=shape;shape,op,bias=operator(shape,layer);old=state
            tag=graph['name']+'/layer/'+str(layer['index'])
            if layer['kind']=='Flatten':proof=None
            elif layer['kind']=='ReLU':state,proof=relu(old,tag)
            else:state,proof=affine(old,op,bias,tag)
            steps.append({'index':layer['index'],'kind':layer['kind'],'input_shape':before,'output_shape':shape,
                'source_sha256':identity(old),'state':state,'proof':proof})
        ends[graph['name']]=state;trace.append({'name':graph['name'],'steps':steps})
    pairs=[]
    for pair in combinations(range(r['experts']),2):
        tick();pair=list(pair)
        merged=join(root,[ends['router']]+[ends[f'expert{i}'] for i in pair])
        guarded=guards(merged,pair,r['experts'])
        projected=project(guarded,r['experts'],r['classes'],r['label'],r['margin'])
        base,obligations=lp_build(projected,pair,r['classes'],r['label'])
        pairs.append({'pair':pair,'joint':merged,'guarded':guarded,'projected':projected,'base':base,'obligations':obligations})
    tick()
    return {'schema':'SCOPED_SOURCE_CONSTRUCTION_V1','source_sha256':expected_source_sha256,
        'input':root,'networks':trace,'pairs':pairs,'lower_bound_certificates':[]}
