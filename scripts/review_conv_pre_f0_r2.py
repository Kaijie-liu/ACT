"""Fresh isolated rational-construction review and immutable evidence inventory."""
import argparse
from fractions import Fraction
import json
from scripts.conv_pre_f0_r2_contract import ROOT,DEFAULT,FREEZE,read,save,sha,git,verify_freeze,validate_job
from scripts.review_conv_request_sign_lp import check_journal
from scripts.review_conv_pre_f0_failure import review as failed_review,OUTPUT as FAILED_REVIEW

OUTPUT=ROOT/'act/pipeline/moe/results/conv_pre_f0_review_20260915_r2.json'


def review():
    from scripts.check_conv_pre_f0_r2 import check_directory
    f=verify_freeze();rt=read(DEFAULT/'runtime.json');d=DEFAULT/f['protocol']['job_id'];j=validate_job(d)
    if (rt['state']!='COMPLETED_CHECKED' or rt['extra_queries_queued'] or rt['freeze_sha256']!=sha(FREEZE)):
        raise ValueError('run incomplete/changed')
    pub=rt['publication']
    if (pub['local_head']!=rt['execution_head'] or pub['remote_head']!=rt['execution_head']
            or pub['remote_ref']!='refs/heads/feat/moe-route-verification'
            or pub['gate']!='REMOTE_EQUALS_LOCAL_BEFORE_RUN_ROOT_CREATED' or pub['confirmed_unix']>=rt['started_unix']):
        raise ValueError('publication gate mismatch')
    if json.loads(git('show',rt['execution_head']+':'+str(FREEZE.relative_to(ROOT))))!=f:
        raise ValueError('execution commit lacks freeze')
    req=j['parent_request']
    for p,h in [(req['subject']['checkpoint'],req['subject']['checkpoint_sha256']),
                (req['tensors']['path'],req['tensors']['sha256']),(req['config']['path'],req['config']['sha256'])]:
        if sha(p)!=h:raise ValueError('model/input/config changed')
    stages=rt['stages'];caps={'capture':300,'proposal':2100,'check':600,'independent':600}
    if set(stages)!=set(caps):raise ValueError('stage missing')
    for name,cap in caps.items():
        v=stages[name]
        if (v!=read(d/(name+'.terminal.json')) or v['state']!='COMPLETED' or v['return_code']!=0
                or v['cap_seconds']!=cap or not 0<v['wall_seconds']<=cap):raise ValueError('stage terminal/cap mismatch')
    fresh=check_directory(d)
    if fresh!=rt['result'] or fresh!=read(d/'check.json') or fresh!=read(d/'independent.json'):
        raise ValueError('fresh exact check differs')
    m=read(d/'manifest.json');r=m['routes'];groups={k:{tuple(p) for p in r[k]} for k in ('feasible','infeasible','unresolved')}
    if (len(r['branches'])!=6 or len({tuple(b['route_set']) for b in r['branches']})!=6
            or any(b['feasibility'] not in groups or tuple(b['route_set']) not in groups[b['feasibility']] for b in r['branches'])):
        raise ValueError('route query terminals differ from partition')
    queries=read(d/'query_log.json');proposal=read(d/'proposal.json')
    if sha(d/'query_log.json')!=proposal['query_log_sha256'] or len(queries)!=proposal['query_count'] or len(queries)>29:
        raise ValueError('LP query count/identity mismatch')
    expected=['gate_lower','gate_upper']+[key for row in m['obligations'] if row['kind']=='residual'
        for key in (row['difference_lower'],row['difference_upper'])]+[
        f"p{row['property_index']}_rational" for row in m['obligations'] if row['kind']=='residual' and row['weighted'] is not None]
    if [v['key'] for v in queries]!=expected or any(v['status'] not in ('PROPOSED','UNAVAILABLE') or v['seconds']<=0 for v in queries):
        raise ValueError('extra/reordered/unaccounted query')
    journal=check_journal(d/'budget_journal.jsonl',sha(d/'job.json'),[[1,2]],0)
    if failed_review()!=read(FAILED_REVIEW):raise ValueError('failed attempt changed')
    details=[]
    for row,result in zip(m['obligations'],fresh['obligations']):
        details.append({**result,'lower_bound_float_descriptive':float(Fraction(result['lower_bound'])) if result['lower_bound'] else None,
            'weighted_export':row.get('weighted'),'certificate':row.get('certificate'),
            'difference_bounds':row.get('difference_bounds'),'gate_bounds':row.get('gate_bounds')})
    return {'schema':'CONV_PRE_F0_RATIONAL_R2_REVIEW','audit_status':'PASS','issues':[],
        'execution_protocol_status':'PASS_NEW_IDENTITY_REMOTE_CONFIRMED_BEFORE_LAUNCH',
        'execution_head':rt['execution_head'],'freeze_sha256':sha(FREEZE),'publication':pub,
        'request':m['request'],'expert_order':m['expert_order'],'joint_source':m['joint_source'],
        'router_source':m['router_source'],'result':fresh,'obligation_details':details,
        'gate_lower_float_descriptive':float(Fraction(fresh['support_bounds']['gate_lower'])),
        'gate_upper_float_descriptive':-float(Fraction(fresh['support_bounds']['gate_upper'])),
        'minimum_float_descriptive':float(Fraction(fresh['minimum_lower_bound'])) if fresh['minimum_lower_bound'] else None,
        'LP_queries':len(queries),'query_records':queries,'stages':stages,'journal':journal,
        'all_stage_seconds':sum(v['wall_seconds'] for v in stages.values()),
        'old_results_and_failed_R1_unchanged':True,'production_verdict_changed':False,
        'failure_description_correction':{
            'earlier_text':'R1 failure archive and frozen R2 comments described Torch property scalars',
            'correct_type':'numpy.float64: linear_safety_rows converts Torch tensors using .numpy() before returning arrays',
            'source':'act/back_end/moe/weighted_top2.py:501',
            'source_sha256':sha(ROOT/'act/back_end/moe/weighted_top2.py'),
            'check':'rational uses type(value) is float; NumPy float64 is rejected, float(value) preserves +/-1/0 exactly',
            'supplemental_test':'scripts/test_conv_pre_f0_scalar_boundary.py',
            'interpretation':'Type description corrected; implementation repair, math, protocol and outcomes unchanged; frozen originals preserved'},
        'scope':'One observed single-pair convolutional request; conditional trusted pre-F0 lowering, not deployed-float or route-changing SAFE',
        'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)),'bytes':p.stat().st_size,'sha256':sha(p)}
                              for p in sorted(DEFAULT.rglob('*')) if p.is_file()]}


if __name__=='__main__':
    from scripts.check_conv_sign_lp import isolate
    isolate();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();r=review()
    if a.check:
        if read(OUTPUT)!=r:raise ValueError('archive differs')
    else:
        if OUTPUT.exists():raise FileExistsError('no overwrite')
        save(OUTPUT,r)
    print(json.dumps({'audit':r['audit_status'],'status':r['result']['status'],
        'positive':r['result']['positive_obligations'],'minimum':r['minimum_float_descriptive'],'LP_queries':r['LP_queries']},indent=2))
