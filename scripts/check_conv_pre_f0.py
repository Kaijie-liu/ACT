"""Independent pre-F0 request check. No builder, floating F0 or solver imports."""
import argparse
from fractions import Fraction
import itertools
from pathlib import Path

from scripts.conv_pre_f0_contract import ROOT,read,save,sha,validate_job
from scripts.check_conv_request_sign_lp import expected_scope,property_vector,check_reuse

TRUSTED=['network_input_to_HZ_interval_sources_and_expert_binding',
         'membership_pair_guard_lowering','router_infeasibility_exclusions']


def order_bounds(lower,negative_upper):
    if lower is not None and negative_upper is not None and lower>-negative_upper:
        raise ValueError('inconsistent order certificates')
    return [Fraction(1,2) if lower is not None and lower>=0 else Fraction(0),
            Fraction(1,2) if negative_upper is not None and negative_upper>=0 else Fraction(1)]


def aggregate(m,snapshot,job,load):
    from act.back_end.solver.lp_certificate import rational,identity
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.back_end.solver.check_rational_mccormick import check_construction
    if (m['schema']!='CONV_PRE_F0_RATIONAL_R1_MANIFEST' or m['request']!=expected_scope(job)
            or m['trusted_base']!=TRUSTED or rational(m['positive_threshold'])!=rational(1e-7)):
        raise ValueError('request/trust/policy mismatch')
    req=m['request'];pairs=m['routes']['feasible'];r=m['routes']
    if sorted(tuple(p) for k in ('feasible','infeasible','unresolved') for p in r[k])!=list(itertools.combinations(range(4),2)):
        raise ValueError('missing or duplicate route')
    if not r['exact'] or r['unresolved']:return {'status':'UNKNOWN','reason':'INCOMPLETE_ROUTE_COVERAGE','trusted_base':TRUSTED}
    if pairs!=job['case']['expected_pairs'] or len(pairs)!=1:raise ValueError('outside frozen pair')
    pair=pairs[0];y=req['clean_prediction'];si=snapshot['identity'];scope=snapshot['scope']
    if (snapshot['feasible_route_sets']!=pairs or not snapshot['route_sets_exact']
            or si['model_state']!=req['model_state'] or any(si[k]!=req[k] for k in ('center','lower','upper'))
            or si['property']!={'classes':10,'clean_prediction':y,'kind':'TOP1_ROBUST'}
            or scope['request_id']!=snapshot['request_id'] or scope['model_state']!=si['model_state']
            or any(scope[k]!=si[k] for k in ('lower','upper','property'))
            or scope['gate']!='selected_softmax_top2' or scope['tie_policy']!='ANY_LEGAL_TOPK'
            or rational(scope['numerical_policy']['safe_positive_margin'])!=rational(1e-7)):
        raise ValueError('snapshot scope mismatch')
    joint=load(m['joint_source']);router=load(m['router_source'])
    for src,dim in ((joint,20),(router,4)):
        if len(src['c'])!=dim or str(src['frame_id'])!=str(scope['frame_id']):raise ValueError('wrong source width/frame')
    if m['expert_order']!=pair:raise ValueError('wrong joint expert order')
    cache={}
    def support(key,kind,index,q,source):
        item=m['supports'][key]
        if (item['kind']!=kind or item['property_index']!=index or item['pair']!=pair
                or item['request']!=req or item['source_sha256']!=identity(source)):
            raise ValueError('support wrong request/source/property')
        record=load(item['export'])
        if record['q']!=q or rational(record['offset'])!=0:raise ValueError('support projection mismatch')
        if item['status'] not in ('PENDING','UNAVAILABLE','PROPOSED'):raise ValueError('invalid support status')
        cert=load(item['certificate']) if item['status']=='PROPOSED' else None
        checked=check_export(record,cert,expected_source_sha256=identity(source))
        value=rational(checked['bound']['checked_lower_bound']) if cert else None
        cache[key]=value
        return value
    qm=[0]*4;qm[pair[0]]=1;qm[pair[1]]=-1
    gl=support('gate_lower','router_order',None,qm,router)
    gu=support('gate_upper','router_order',None,[-v for v in qm],router)
    gate=order_bounds(gl,gu)
    required={(tuple(pair),i) for i in range(9)};rows=m['obligations']
    if len(rows)!=9 or {(tuple(row['pair']),row['property_index']) for row in rows}!=required:
        raise ValueError('missing/duplicate output property')
    expected_supports={'gate_lower','gate_upper'};out=[]
    competitors=[i for i in range(10) if i!=y]
    for row in rows:
        i=row['property_index'];c=competitors[i];q=property_vector(y,c);value=None
        if row['competitor']!=c or row['q']!=q or row['constant']!=0:raise ValueError('wrong output property')
        if row['kind']=='reused':
            value=check_reuse(row['proof'],pair,i,c,snapshot);status='CHECKED_REUSED_POSITIVE'
        elif row['kind']=='residual':
            lower_key=f'p{i}_lower';upper_key=f'p{i}_upper';expected_supports.update((lower_key,upper_key))
            if row['difference_lower']!=lower_key or row['difference_upper']!=upper_key:raise ValueError('wrong difference reference')
            qd=q+[-v for v in q]
            lo=support(lower_key,'difference',i,qd,joint)
            neg=support(upper_key,'difference',i,[-v for v in qd],joint)
            if lo is None or neg is None:
                if row['weighted_status']!='RANGE_UNAVAILABLE' or row['weighted'] is not None:
                    raise ValueError('unproved difference used')
                status='UNKNOWN_RANGE'
            else:
                if lo>-neg:raise ValueError('inverted exact difference range')
                if [rational(v) for v in row['difference_bounds']]!=[lo,-neg] or [rational(v) for v in row['gate_bounds']]!=gate:
                    raise ValueError('range changed or rounded inward')
                if row['weighted_status'] not in ('PROPOSED','UNAVAILABLE'):raise ValueError('weighted query incomplete')
                rec=load(row['weighted']);cert=load(row['certificate']) if row['weighted_status']=='PROPOSED' else None
                checked=check_construction(rec,cert,source_hash=identity(joint),q=q,offset=0,
                    gate=[str(v) for v in gate],difference=[str(lo),str(-neg)])
                value=rational(checked['bound']['checked_lower_bound']) if cert else None
                status='CHECKED_RATIONAL_POSITIVE' if value is not None and value>rational(1e-7) else 'NONPOSITIVE_OR_UNAVAILABLE'
        else:raise ValueError('ungenerated obligation')
        out.append({'pair':pair,'property_index':i,'competitor':c,'status':status,
                    'lower_bound':str(value) if value is not None else None})
    if set(m['supports'])!=expected_supports:raise ValueError('extra/missing support queries')
    positive=[v for v in out if v['status'] in ('CHECKED_REUSED_POSITIVE','CHECKED_RATIONAL_POSITIVE')]
    complete=len(positive)==9 and m['generation_complete']
    return {'status':'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_PRE_F0_LOWERING' if complete else 'UNKNOWN',
        'required_obligations':9,'positive_obligations':len(positive),
        'reused_positive':sum(v['status']=='CHECKED_REUSED_POSITIVE' for v in out),
        'rational_positive':sum(v['status']=='CHECKED_RATIONAL_POSITIVE' for v in out),
        'minimum_lower_bound':str(min(rational(v['lower_bound']) for v in positive)) if complete else None,
        'gate_bounds':[str(v) for v in gate],'support_bounds':{k:str(v) if v is not None else None for k,v in cache.items()},
        'obligations':out,'trusted_base':TRUSTED,'floating_F0_construction_trusted':False,
        'router_exclusions_independently_proved':False,'deployed_float_SAFE':False,'production_verdict_changed':False}


def check_directory(directory):
    job=validate_job(directory);m=read(directory/'manifest.json');cap=read(directory/'capture.json')
    if (sha(directory/'generation.json')!=cap['generation_sha256'] or sha(directory/'budget_journal.jsonl')!=cap['journal_sha256']
            or cap['floating_F0_called'] or cap['weighted_property_solver_called']):raise ValueError('capture drift')
    g=read(directory/'generation.json')
    # Proposal may add only downstream evidence, not change upstream sources.
    for key in ('schema','request','trusted_base','positive_threshold','routes','common_facts','joint_source','router_source','expert_order','generation_complete'):
        if g[key]!=m[key]:raise ValueError('source changed during proposal')
    if len(g['obligations'])!=len(m['obligations']) or set(g['supports'])!=set(m['supports']):raise ValueError('query inventory changed')
    for before,after in zip(g['obligations'],m['obligations']):
        if any(after[k]!=v for k,v in before.items()):raise ValueError('obligation binding changed')
    for key,before in g['supports'].items():
        if any(m['supports'][key][k]!=v for k,v in before.items() if k not in ('status','certificate')):
            raise ValueError('support binding changed')
    if read(directory/'proposal.json')['manifest_sha256']!=sha(directory/'manifest.json'):raise ValueError('proposal identity drift')
    def load(ref):
        path=(directory/ref['file']).resolve()
        if path.parent!=directory.resolve() or sha(path)!=ref['sha256']:raise ValueError('proof path/hash mismatch')
        return read(path)
    from act.back_end.solver.lp_certificate import identity
    snap=load(m['common_facts'])
    if identity(snap['payload'])!=snap['payload_sha256']:raise ValueError('snapshot drift')
    return aggregate(m,snap['payload'],job,load)


if __name__=='__main__':
    from scripts.check_conv_sign_lp import isolate
    isolate();p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):raise ValueError('new local output required')
    save(a.output,check_directory(a.directory))
