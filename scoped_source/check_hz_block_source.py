"""Independent original-source leaves plus block endpoint aggregation.

Never calls template specialization, joint construction, or the old full source
checker. Source factor IDs bind the positional maps used by the support checker.
"""
from itertools import combinations

from source_enclosure.format import identity
from source_enclosure.check import check_box
from scoped_source.graph import validate, clock
from scoped_source.check_hz_row_enclosure import parse
from scoped_source.check_hz_lifted_source import lift, network, route_reference
from scoped_source.check_hz_templates import common_entry
from scoped_source.hz_source_check import properties, gate_range, state_snapshot
from act.back_end.moe.check_block_endpoints import check_request

SCHEMA='CHECKED_HZ_BLOCK_SOURCE_V1'
CONTEXT_GUARD='ALL_TIE_LEGAL_BLOCK_TEMPLATES_V1'


def check(doc,package,*,expected_source_sha256,deadline):
    tick=clock(deadline);anchor=identity(package)
    r,lower,upper=validate(doc,expected_source_sha256,tick)
    if (set(package)!={'schema','source_sha256','input','router','common','templates','pairs','endpoint_request','proof'}
            or package['schema']!=SCHEMA or package['source_sha256']!=expected_source_sha256):
        raise ValueError('block source package identity/schema')
    ref=package['input']['reference'];_,ci,bi=parse(ref)
    if ref['ownership']!={'continuous':['input']*len(ci),'binary':[]}:raise ValueError('input ownership')
    check_box(lower,upper,ref['state']);initial,first=lift(package['input'],'input',deadline)
    router,steps,lifts=network(doc,'router',initial,package['router'],'router',deadline);lifts=[first]+lifts
    base=package['common'];common_entry(initial,router,base);_,bc,bb=parse(base,target=True)
    e=r['experts'];saved=package['templates']
    if type(saved) is not list or [t['expert'] for t in saved]!=list(range(e)):
        raise ValueError('all expert templates required in order')
    states=[];private=[]
    for i,item in enumerate(saved):
        if set(item)!={'expert','trace'} or type(item['expert']) is not int:raise ValueError('template fields')
        state,ss,ll=network(doc,f'expert{i}',base,item['trace'],f'template/expert{i}',deadline)
        steps+=ss;lifts+=ll;states.append(state);_,tc,tb=parse(state,target=True)
        if tc[:len(bc)]!=bc or tb[:len(bb)]!=bb:raise ValueError('template shared namespace')
        private+=tc[len(bc):]+tb[len(bb):]
    if len(set(private))!=len(private) or set(private)&set(bc+bb):raise ValueError('expert private factor alias')
    roster=[list(p) for p in combinations(range(e),2)];req=package['endpoint_request'];props=properties(r)
    ctx={'request':expected_source_sha256,'domain':identity(r),'guard':CONTEXT_GUARD}
    if ([p['pair'] for p in package['pairs']]!=roster or req['context']!=ctx or req['properties']!=props
            or req['experts']!=e or req['classes']!=r['classes'] or [p['pair'] for p in req['pairs']]!=roster):
        raise ValueError('all source pairs/properties binding')
    for p,endpoint in zip(package['pairs'],req['pairs']):
        tick()
        if set(p)!={'pair','entry','gate_evidence'}:raise ValueError('source pair fields')
        a,b=p['pair'];tag=f'pair{a}-{b}'
        route_reference(initial,router,p['entry']['reference'],(a,b),e)
        entry,l=lift(p['entry'],tag+'/guard',deadline);lifts.append(l)
        _,ec,eb=parse(entry,target=True)
        if ec!=bc or eb!=bb or entry['ownership']!=base['ownership']:
            raise ValueError('guard factor ownership/identity')
        src=endpoint['batch']['sources']
        for state,key in ((base,'common'),(entry,'entry'),(states[a],'a'),(states[b],'b')):
            state_snapshot(state['state'],src[key])
        evidence=gate_range(router['state'],(a,b))
        if p['gate_evidence']!=evidence or endpoint['gate']['bounds']!=evidence['bounds']:
            raise ValueError('checked source gate binding')
    result=check_request(req,package['proof'],expected_request_sha256=identity(req),deadline=deadline)
    output={'status':'CHECKED_POSITIVE_DECLARED_REAL_SOURCE' if result['positive']==result['required'] else result['status'],
            'source_sha256':expected_source_sha256,'package_sha256':anchor,
            **{k:result[k] for k in ('required','positive','checked_endpoints','missing_endpoints','results')},
            'checked_source_steps':len(steps),'checked_expert_templates':e,
            'checked_row_lifts':sum(l['rows'] for l in lifts),
            'nonzero_lift_compensations':sum(l['added_continuous'] for l in lifts),
            'source_lowering_checked':True,'deployed_float_SAFE':False,'hard_budget_supervision':False,
            'portable_distribution':False,'real_model_claim':False,
            'remaining_trust':['declaration_corresponds_to_intended_program','exact_checker_implementation']}
    if identity(doc)!=expected_source_sha256 or identity(package)!=anchor:raise ValueError('source/package changed')
    tick();return output
