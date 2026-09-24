"""Saved-only exact recheck and full-cost audit, separately timed from requests."""
import argparse
import math
from pathlib import Path
import statistics
import time

from scoped_proof.io import ROOT, load, save, sha
from shared_route_residual.study import CONFIG, OUTPUT, accept
from shared_route_residual.worker import recheck


def comparable(result):
    keys=('higher','lower','checked_lower_bound','residual_box_term','nonzero_residual_coordinates','status')
    return {'bounds':[{k:r[k] for k in keys} for r in result['bounds']],
        'pairs':[{k:v for k,v in r.items() if k not in ('lp_sha256','evidence_sha256')} for r in result['pairs']],
        'needed_experts':result['needed_experts']}


def audit_one(root,*,exact=True):
    plan,term,cost=(load(root/n) for n in ('plan.json','terminal.json','cost.json'))
    if term['plan_sha256']!=sha(root/'plan.json') or cost['terminal_sha256']!=sha(root/'terminal.json'):
        raise ValueError('cost/terminal/plan chain')
    budget=plan['budget_seconds'];stages=term['phases']
    if not 0<budget<=300 or cost['budget_seconds']!=budget:raise ValueError('budget')
    if term['complete_output_positive_proof'] is not False or term['new_real_positive'] is not False:
        raise ValueError('router segment is not output proof')
    if abs(plan['deadline']-plan['started_monotonic']-budget)>1e-7:
        raise ValueError('deadline changed')
    if abs(plan['work_deadline']-(plan['deadline']-min(2.,budget/10)))>1e-7:
        raise ValueError('work deadline changed')
    if [s['phase'] for s in stages]!=['build','check'][:len(stages)] or len(stages)>2:
        raise ValueError('stage order')
    previous=0.
    for s in stages:
        if load(root/(s['phase']+'_stage.json'))!=s:raise ValueError('stage identity')
        for k in ('seconds','start_seconds','end_seconds','sampled_peak_rss'):
            if type(s[k]) not in (int,float) or not math.isfinite(s[k]) or s[k]<0:raise ValueError('finite stage cost')
        if s['start_seconds']<previous or s['end_seconds']<s['start_seconds'] or s['seconds']>s['end_seconds']-s['start_seconds']+1e-6:
            raise ValueError('stage chronology/cost')
        if s.get('deadline_monotonic',plan['work_deadline'])!=plan['work_deadline']:
            raise ValueError('stage budget extension')
        previous=s['end_seconds']
    for k in ('end_to_end_seconds','stage_seconds','overhead_seconds','sampled_peak_rss'):
        if type(cost[k]) not in (int,float) or not math.isfinite(cost[k]) or cost[k]<0:raise ValueError('finite cost')
    if (abs(cost['stage_seconds']-sum(s['seconds'] for s in stages))>1e-7 or
            abs(cost['end_to_end_seconds']-cost['stage_seconds']-cost['overhead_seconds'])>1e-7 or
            cost['sampled_peak_rss']!=max((s['sampled_peak_rss'] for s in stages),default=0) or
            not previous<=term['seconds_before_publication']<=cost['end_to_end_seconds']):
        raise ValueError('complete cost decomposition')
    if cost['status'] not in ('COMPLETED_ROUTER_SEGMENT','ERROR','TIMEOUT','RESOURCE_LIMIT'):
        raise ValueError('terminal status')
    if cost['status']!=term['status'] and not (cost['status']=='TIMEOUT' and cost['end_to_end_seconds']>=budget):
        raise ValueError('terminal status disagreement')
    if term['check_sha256']!=(sha(root/'check.json') if (root/'check.json').exists() else None):
        raise ValueError('check identity')
    result=None
    if cost['status']=='COMPLETED_ROUTER_SEGMENT':
        if ((root/'publication_timeout.json').exists() or len(stages)!=2 or
                any(s['status']!='COMPLETED' for s in stages) or
                cost['end_to_end_seconds']>=budget or
                term['seconds_before_publication']>=plan['work_deadline']-plan['started_monotonic']):
            raise ValueError('late/partial/error completion')
        receipt=accept(root,plan)
        if exact and recheck(root,plan,time.monotonic()+30)!=receipt:
            raise ValueError('fresh independent mathematical recheck differs')
        result=receipt['result']
    return {'status':cost['status'],'cost':cost,'result':result,
            'candidate_bytes':(root/'candidates.json').stat().st_size if (root/'candidates.json').exists() else None}


def audit():
    started=time.monotonic();cfg=load(CONFIG);launch=load(OUTPUT/'launch.json');execution=load(OUTPUT/'execution.json')
    if launch['config_sha256']!=sha(CONFIG) or launch['config']!=cfg:raise ValueError('frozen config')
    for name,digest in cfg['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('frozen implementation '+name)
    if len(execution['rows'])!=len(cfg['calls']):raise ValueError('missing call')
    reviewed={};comparisons=[]
    for call,saved in zip(cfg['calls'],execution['rows']):
        if any(saved[k]!=v for k,v in call.items()):raise ValueError('roster/order')
        root=OUTPUT/call['id'];plan=load(root/'plan.json')
        if (plan['fixture']!=call['fixture'] or plan['mode']!=call['mode'] or
                plan['expected_source_sha256']!=cfg['fixture_sha256'][call['fixture']] or
                plan['budget_seconds']!=cfg['budget_seconds']):raise ValueError('call binding')
        r=audit_one(root)
        if saved['status']!=r['status']:raise ValueError('execution status')
        reviewed[call['id']]={**call,**r}
    for kind in cfg['fixture_sha256']:
        for repeat in range(3):
            a=reviewed[f'{kind}_{repeat}_pairwise'];b=reviewed[f'{kind}_{repeat}_shared']
            same=None if a['result'] is None or b['result'] is None else comparable(a['result'])==comparable(b['result'])
            if same is False:raise ValueError('exact paired differential failed')
            comparisons.append({'fixture':kind,'repeat':repeat,'identical_bounds_and_pairs':same})
    summary={}
    for kind in cfg['fixture_sha256']:
        summary[kind]={}
        for mode in ('pairwise','shared'):
            values=[r for r in reviewed.values() if r['fixture']==kind and r['mode']==mode]
            summary[kind][mode]={
                'statuses':[r['status'] for r in values],
                'median_router_segment_seconds':statistics.median(r['cost']['end_to_end_seconds'] for r in values),
                'median_build_seconds':statistics.median(load(OUTPUT/r['id']/'build_stage.json')['seconds'] for r in values),
                'median_check_seconds':statistics.median(load(OUTPUT/r['id']/'check_stage.json')['seconds'] for r in values if (OUTPUT/r['id']/'check_stage.json').exists()) if any((OUTPUT/r['id']/'check_stage.json').exists() for r in values) else None,
                'candidate_bytes':[r['candidate_bytes'] for r in values],
                'max_sampled_peak_rss':max(r['cost']['sampled_peak_rss'] for r in values),
                'retained_pairs':[None if r['result'] is None else r['result']['retained_pairs'] for r in values]}
    files=[{'path':str(p.relative_to(OUTPUT)),'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(OUTPUT.rglob('*')) if p.is_file()]
    report={'status':'PASS','issues':[],'launch_head':launch['head'],'frozen_sources':len(cfg['sources']),
        'calls':len(reviewed),'comparisons':comparisons,'summary':summary,
        'real_requests':0,'native_solver_calls':0,'new_complete_output_certificates':0,
        'audit_seconds_separate_from_request':time.monotonic()-started,
        'raw_file_count':len(files),'raw_bytes':sum(f['bytes'] for f in files),
        'scope':'synthetic router-proof segment; no expert/output proof or real-model speed claim'}
    save(ROOT/'docs/shared_route_residual_audit_20260924_r1.json',report)
    save(ROOT/'docs/shared_route_residual_archive_20260924_r1.json',{'root':str(OUTPUT),'files':files})
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-frozen-synthetic',action='store_true');a=p.parse_args()
    if not a.audit_frozen_synthetic:p.error('saved-only audit required')
    audit()
