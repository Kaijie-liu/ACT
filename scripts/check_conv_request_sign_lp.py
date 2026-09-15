"""Independent full obligation/reuse aggregation; trusted lowering explicit."""
import argparse
from fractions import Fraction
import itertools
import math
from pathlib import Path

from scripts.conv_request_sign_lp_contract import ROOT, read, sha, save, validate_job

TRUSTED=['network_input_to_HZ_and_interval_sources','membership_pair_guard_lowering',
         'router_infeasibility_exclusions','floating_F0_construction_and_scope_binding']


def expected_scope(job):
    sample=job['parent_request']['sample']
    return {'parent_request_sha256':job['parent_request_sha256'], 'dataset_index':sample['dataset_index'],
            'clean_prediction':sample['label'],'classes':10,'experts':4,'top_k':2,
            'tie_policy':'ANY_LEGAL_TOPK','epsilon':job['parent_request']['epsilon'],
            **{k:sample[k] for k in ('center','lower','upper')},
            'model_state':job['parent_request']['subject']['model_state']}


def property_vector(label,competitor):
    return [1 if i==label else -1 if i==competitor else 0 for i in range(10)]


def check_reuse(proof,pair,index,competitor,snapshot):
    from act.back_end.solver.lp_certificate import check,rational,identity
    if (proof['property_index']!=index or proof['reason']!='SAFE_REUSED_TIER1_INTERVAL'
            or proof['containment_rule']!='TOP2_SET_IMPLIES_EACH_MEMBER_TOP2_MEMBERSHIP'
            or len(proof['proof_sources'])!=2):raise ValueError('invalid scoped reuse')
    branches={b['candidate']:b for b in snapshot['branches']}; values=[]
    if len(branches)!=len(snapshot['branches']):raise ValueError('duplicate interval expert')
    for expert,f in zip(pair,proof['proof_sources']):
        if (f['scope']!=snapshot['scope'] or f['expert']!=expert or f['property_index']!=index
                or f['competitor']!=competitor or f['guard_kind']!='TOP2_MEMBERSHIP'):
            raise ValueError('wrong reuse request/domain/property')
        src=branches[expert]['proof_output_bounds'];lo,hi=src['lower'],src['upper']
        if len(lo)!=10 or len(hi)!=10 or any(not math.isfinite(v) for v in lo+hi) or any(a>b for a,b in zip(lo,hi)):
            raise ValueError('invalid source interval')
        y=snapshot['identity']['property']['clean_prediction']
        lp={'c':[1,-1],'lower':[lo[y],lo[competitor]],'upper':[hi[y],hi[competitor]]}
        if f['source_interval']!=src or f['interval_lp']!=lp:raise ValueError('interval source/projection mismatch')
        checked=check(lp,f['interval_certificate']); v=rational(f['lower_bound'])
        if (v!=rational(f['interval_certificate']['claimed_lower_bound'])
                or v>rational(checked['checked_lower_bound']) or v<=rational(1e-7)):
            raise ValueError('unproved/nonpositive interval fact')
        values.append(v)
    if rational(proof['accepted_minimum'])!=min(values):raise ValueError('reuse minimum differs')
    return min(values)


def aggregate(manifest,snapshot,job,load):
    from act.back_end.solver.lp_certificate import identity,rational
    from act.back_end.solver.check_hz_lp_export import check_export
    if (manifest['schema']!='CONV_REQUEST_SIGN_LP_R1_MANIFEST' or manifest['request']!=expected_scope(job)
            or manifest['trusted_base']!=TRUSTED or rational(manifest['positive_threshold'])!=rational(1e-7)):
        raise ValueError('wrong request/threshold/trusted base')
    req=manifest['request'];y=req['clean_prediction'];routes=manifest['routes']
    groups=[routes[k] for k in ('feasible','infeasible','unresolved')]
    if sorted(tuple(p) for g in groups for p in g)!=list(itertools.combinations(range(4),2)):
        raise ValueError('route partition missing, repeated or noncanonical')
    if not routes['exact'] or routes['unresolved']:
        return {'status':'UNKNOWN','reason':'INCOMPLETE_ROUTE_COVERAGE','trusted_base':TRUSTED}
    if not routes['feasible'] or routes['feasible']!=job['case']['expected_pairs']:
        raise ValueError('unexpected feasible routes; do not replace request')
    si=snapshot['identity']
    if (snapshot['feasible_route_sets']!=routes['feasible'] or not snapshot['route_sets_exact']
            or si['model_state']!=req['model_state']
            or any(si[k]!=req[k] for k in ('center','lower','upper'))
            or si['property']!={'classes':10,'clean_prediction':y,'kind':'TOP1_ROBUST'}):
        raise ValueError('common fact snapshot identity mismatch')
    scope=snapshot['scope']
    if (scope['request_id']!=snapshot['request_id'] or scope['model_state']!=si['model_state']
            or scope['property']!=si['property'] or scope['lower']!=si['lower'] or scope['upper']!=si['upper']
            or scope['gate']!='selected_softmax_top2' or scope['tie_policy']!='ANY_LEGAL_TOPK'
            or rational(scope['numerical_policy']['safe_positive_margin'])!=rational(1e-7)):
        raise ValueError('common fact scope mismatch')
    required={(tuple(p),i) for p in routes['feasible'] for i in range(9)}
    rows=manifest['obligations']
    if len(rows)!=len(required) or {(tuple(r['pair']),r['property_index']) for r in rows}!=required:
        raise ValueError('missing/duplicate property obligation')
    competitors=[i for i in range(10) if i!=y];out=[]
    for r in rows:
        i=r['property_index'];comp=competitors[i]; pair=r['pair'];bound=None
        if r['competitor']!=comp or r['q']!=property_vector(y,comp) or r['constant']!=0:
            raise ValueError('wrong property projection')
        if r['kind']=='reused':
            bound=check_reuse(r['proof'],pair,i,comp,snapshot); status='CHECKED_REUSED_POSITIVE'
        elif r['kind']=='lp':
            record=load(r['export'])
            if record['q']!=[1] or rational(record['offset'])!=0:raise ValueError('wrong scalar F0 objective')
            check_export(record,None,expected_source_sha256=r['source_sha256'])
            if str(record['source']['frame_id'])!=str(scope['frame_id']):raise ValueError('F0 factor-frame mismatch')
            if r['proposal_status']=='PROPOSED':
                result=check_export(record,load(r['certificate']),expected_source_sha256=r['source_sha256'])
                bound=rational(result['bound']['checked_lower_bound'])
                status='CHECKED_LP_POSITIVE' if bound>rational(1e-7) else 'CHECKED_LP_NONPOSITIVE'
            elif r['proposal_status'] in ('PENDING','UNAVAILABLE'):status='UNRESOLVED_LP'
            else:raise ValueError('invalid proposal status')
        elif r['kind']=='pending':status='UNGENERATED_OBLIGATION'
        else:raise ValueError('unrecognized proof kind')
        out.append({'pair':pair,'property_index':i,'competitor':comp,'status':status,
                    'lower_bound':str(bound) if bound is not None else None})
    positive=[r for r in out if r['status'] in ('CHECKED_REUSED_POSITIVE','CHECKED_LP_POSITIVE')]
    complete=len(positive)==len(required) and manifest['generation_complete']
    return {'status':'CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_F0_LOWERING' if complete else 'UNKNOWN',
            'required_obligations':len(required),'positive_obligations':len(positive),
            'reused_positive':sum(r['status']=='CHECKED_REUSED_POSITIVE' for r in out),
            'lp_positive':sum(r['status']=='CHECKED_LP_POSITIVE' for r in out),
            'minimum_lower_bound':str(min(rational(r['lower_bound']) for r in positive)) if complete else None,
            'obligations':out,'trusted_base':TRUSTED,'route_partition_complete':True,
            'router_exclusion_bounds_independently_checked':False,'deployed_float_SAFE':False,
            'production_SAFE_verdict_changed':False}


def check_directory(directory):
    job=validate_job(directory);manifest=read(directory/'manifest.json')
    capture=read(directory/'capture.json');generation=read(directory/'generation.json')
    if (sha(directory/'generation.json')!=capture['generation_sha256']
            or sha(directory/'budget_journal.jsonl')!=capture['journal_sha256']
            or capture['weighted_property_milp_called'] is not False):raise ValueError('capture identity changed')
    def source_only(value):
        return {**value,'obligations':[{k:v for k,v in row.items() if k not in
            ('proposal_status','certificate','proposal_seconds','proposal_reason')} for row in value['obligations']]}
    if source_only(generation)!=source_only(manifest):raise ValueError('proposal changed a source obligation')
    if read(directory/'proposal.json')['manifest_sha256']!=sha(directory/'manifest.json'):
        raise ValueError('proposal manifest identity changed')
    def load(ref):
        path=directory/ref['file']
        if path.parent!=directory or sha(path)!=ref['sha256']:raise ValueError('proof reference drift/path escape')
        return read(path)
    snapshot=load(manifest['common_facts'])
    from act.back_end.solver.lp_certificate import identity
    if identity(snapshot['payload'])!=snapshot['payload_sha256']:raise ValueError('snapshot hash mismatch')
    return aggregate(manifest,snapshot['payload'],job,load)


if __name__=='__main__':
    from scripts.check_conv_sign_lp import isolate
    isolate()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):raise ValueError('new project output required')
    save(a.output,check_directory(a.directory))
