"""Check exact source references, row lifts and all fresh endpoint duties.

No producer or optimizer calls. Declared real semantics, not native FP SAFE.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations

from scoped_source.graph import validate,operator,clock
from source_enclosure.format import unpack,pack,identity,clean
from source_enclosure.check import check_box,check_affine,check_relu
from scoped_source.check_hz_binary64 import REFERENCE
from scoped_source.check_hz_row_enclosure import parse,check as row_check
from scoped_source.hz_source_check import properties,gate_range,state_snapshot
from act.back_end.moe.check_hz_endpoints import check_request

SCHEMA='CHECKED_ROW_LIFTED_SOURCE_ENDPOINT_V1'


def owners(previous,reference,tag):
    _,c,b=parse(previous);_,nc,nb=parse(reference)
    if nc[:len(c)]!=c or nb[:len(b)]!=b:raise ValueError('source original factor prefix')
    expected={k:previous['ownership'][k]+[tag]*(len(new)-len(old))
              for k,new,old in (('continuous',nc,c),('binary',nb,b))}
    if reference['ownership']!=expected:raise ValueError('source reference ownership')


def lift(record,tag,deadline):
    if set(record)!={'reference','enclosure','snapshot'}:raise ValueError('source lift schema')
    reference=record['reference'];parse(reference)
    checked=row_check(reference,record['enclosure'],expected_reference_sha256=identity(reference),
                      expected_owner=tag,deadline=deadline)
    target=record['enclosure']['target'];state_snapshot(target['state'],record['snapshot'])
    return target,checked


def relu_reference(previous,reference,cert,tag):
    s,_,_=unpack(previous['state']);t,c,b=unpack(reference['state'])
    k=sum(v=='unstable' for v in cert['branches']);n=len(s['ub'])
    if len(t['ub'])!=n+2*k:raise ValueError('blocked ReLU row inventory')
    canonical=deepcopy(t);order=list(range(n))+[j for i in range(k) for j in (n+i,n+k+i)]
    for key in ('Auc','Aub','ub'):canonical[key]=[t[key][j] for j in order]
    return check_relu(previous['state'],pack(canonical,c,b),cert,tag)


def network(doc,name,initial,trace,prefix,deadline):
    tick=clock(deadline);g=next(g for g in doc['networks'] if g['name']==name)
    if len(trace)!=len(g['layers']):raise ValueError('all source layers required')
    state=initial;shape=doc['center']['shape'];steps=[];lifts=[]
    for i,(layer,step) in enumerate(zip(g['layers'],trace)):
        tick();tag=f'{prefix}/layer{i}'
        if set(step)!={'index','kind','nominal','certificate','lift'} or (step['index'],step['kind'])!=(i,layer['kind']):
            raise ValueError('source layer schema/order')
        ref=step['lift']['reference'];owners(state,ref,tag)
        shape,op,bias=operator(shape,layer)
        if layer['kind']=='Linear':
            result=check_affine(state['state'],ref['state'],op,bias,step['nominal'],step['certificate'],tag)
        elif layer['kind']=='ReLU':
            if step['nominal'] is not None:raise ValueError('unexpected ReLU nominal')
            result=relu_reference(state,ref,step['certificate'],tag)
        else:
            if step['nominal'] is not None or step['certificate'] is not None or ref!=state:
                raise ValueError('flatten reference changed')
            result={'status':'CHECKED_IDENTITY_FLATTEN'}
        state,l=lift(step['lift'],tag,deadline);steps.append(result);lifts.append(l)
    return state,steps,lifts


def route_reference(initial,router,reference,pair,experts):
    x,xc,xb=parse(initial);r,rc,rb=parse(router);parse(reference)
    if rc[:len(xc)]!=xc or rb[:len(xb)]!=xb or r['frame_id']!=x['frame_id']:
        raise ValueError('conditional input/router identity')
    h=deepcopy(r)
    for k in ('c','Gc','Gb'):h[k]=deepcopy(x[k])
    for selected in pair:
        for outside in range(experts):
            if outside in pair:continue
            for source,dest in (('Gc','Auc'),('Gb','Aub')):
                h[dest].append(clean({j:r[source][outside].get(j,F(0))-r[source][selected].get(j,F(0))
                                     for j in r[source][outside].keys()|r[source][selected].keys()}))
            h['ub'].append(r['c'][selected]-r['c'][outside])
    expected={'schema':REFERENCE,'state':pack(h,rc,rb),'ownership':deepcopy(router['ownership'])}
    if reference!=expected:raise ValueError('exact conditional route reference')


def check(doc,package,*,expected_source_sha256,deadline):
    tick=clock(deadline);anchor=identity(package);r,lower,upper=validate(doc,expected_source_sha256,tick)
    props=properties(r)
    if (set(package)!={'schema','source_sha256','input','router','pairs','endpoint_request','proof'}
            or package['schema']!=SCHEMA or package['source_sha256']!=expected_source_sha256):
        raise ValueError('lifted source package identity')
    ref=package['input']['reference'];h,ci,bi=parse(ref)
    if ref['ownership']!={'continuous':['input']*len(ci),'binary':[]}:
        raise ValueError('input declared factor owners')
    check_box(lower,upper,ref['state']);initial,first=lift(package['input'],'input',deadline)
    router,steps,lifts=network(doc,'router',initial,package['router'],'router',deadline);lifts=[first]+lifts
    roster=[list(p) for p in combinations(range(r['experts']),2)];req=package['endpoint_request']
    if [p['pair'] for p in package['pairs']]!=roster:raise ValueError('all tie legal source pairs required')
    context={'request':expected_source_sha256,'domain':identity(r),'guard':'ALL_TIE_LEGAL_PAIRS_LIFTED_V1'}
    if (req['context']!=context or req['properties']!=props or req['experts']!=r['experts']
            or req['classes']!=r['classes'] or [p['pair'] for p in req['pairs']]!=roster):
        raise ValueError('fresh endpoint source/property binding')
    for p,endpoint in zip(package['pairs'],req['pairs']):
        tick()
        if set(p)!={'pair','entry','a','b','gate_evidence'}:raise ValueError('pair fields')
        a,b=p['pair'];tag=f'pair{a}-{b}'
        route_reference(initial,router,p['entry']['reference'],(a,b),r['experts'])
        entry,l=lift(p['entry'],tag+'/guard',deadline);lifts.append(l)
        left,ls,ll=network(doc,f'expert{a}',entry,p['a'],tag+f'/expert{a}',deadline)
        right,rs,rl=network(doc,f'expert{b}',entry,p['b'],tag+f'/expert{b}',deadline)
        steps+=ls+rs;lifts+=ll+rl
        _,ec,eb=parse(entry);private=[]
        for terminal in (left,right):
            _,tc,tb=parse(terminal,target=True)
            if tc[:len(ec)]!=ec or tb[:len(eb)]!=eb:raise ValueError('expert shared entry prefix')
            private+=tc[len(ec):]+tb[len(eb):]
        if len(set(private))!=len(private) or set(private)&set(ec+eb):raise ValueError('expert private aliases')
        for key,state in (('entry',entry),('a',left),('b',right)):
            state_snapshot(state['state'],endpoint['sources'][key])
        evidence=gate_range(router['state'],(a,b))
        if (p['gate_evidence']!=evidence or endpoint['gate']['bounds']!=evidence['bounds']
                or endpoint['relation_mode']!='shared_input'):raise ValueError('checked lifted router gate binding')
    result=check_request(req,package['proof'],expected_request_sha256=identity(req),deadline=deadline)
    output={'status':'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if result['positive']==result['required'] else result['status'],
            'source_sha256':expected_source_sha256,'package_sha256':anchor,
            **{k:result[k] for k in ('required','positive','checked_endpoints','missing_endpoints','results')},
            'checked_source_steps':len(steps),'checked_row_lifts':sum(l['rows'] for l in lifts),
            'nonzero_lift_compensations':sum(l['added_continuous'] for l in lifts),
            'affine_error_factors':sum(s.get('error_factors',0) for s in steps),
            'source_lowering_checked':True,'deployed_float_SAFE':False,'hard_budget_supervision':False,
            'portable_distribution':False,'real_model_claim':False,
            'remaining_trust':['declaration_corresponds_to_intended_program','exact_checker_implementation']}
    if identity(doc)!=expected_source_sha256 or identity(package)!=anchor:raise ValueError('source/package changed')
    tick();return output
