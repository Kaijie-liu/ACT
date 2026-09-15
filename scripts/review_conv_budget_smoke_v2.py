"""Read-only result reconstruction; no new model verification queries."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.conv_budget_smoke_v2 import ROOT, DEFAULT
from scripts.audit_conv_budget_smoke_v2 import audit
from scripts.conv_three_arm_contract import read
from act.pipeline.moe.experiment1 import _sha256
from act.pipeline.moe.conv_training import atomic_json

OUTPUT=ROOT/'act/pipeline/moe/results/conv_budget_smoke_review_20260915_v2.json'


def native_summary(events):
    ready={e['seq']:e for e in events if e['kind']=='NATIVE_READY'}
    returns=[e for e in events if e['kind'] in ('NATIVE_RETURN','NATIVE_RAISE')]
    bad=[e['token'] for e in returns if e['entered']+e['effective']>ready[e['token']]['deadline']+1e-9]
    if bad:raise ValueError('native passed allocation exceeded deadline')
    overruns=[(max(0.,e['clock_elapsed']-ready[e['token']]['deadline']),e) for e in returns]
    worst=max(overruns,key=lambda t:t[0]) if overruns else (0.,None)
    entry=worst[1]
    return {'ready_calls':len(ready),'returned_or_raised_calls':len(returns),'allocation_violations':bad,
        'observed_return_after_deadline_count':sum(t>1e-3 for t,_ in overruns),
        'maximum_observed_return_deadline_overrun_seconds':worst[0],
        'largest_overrun_call':None if entry is None else {'token':entry['token'],
            'entered':entry['entered'],'passed_seconds':entry['effective'],
            'deadline':ready[entry['token']]['deadline'],'return_observed':entry['clock_elapsed'],
            'result':entry.get('result')},
        'meaning':'Passed time limit is compliant; native return may overrun. Observation includes wrapper return logging, not a hard real-time solver guarantee.'}


def review():
    automatic=read(DEFAULT/'audit.final.json');separate=read(DEFAULT/'audit.independent.json')
    fresh=audit(DEFAULT)
    if automatic!=separate or fresh!=automatic or fresh['status']!='PASS':
        raise ValueError('independent/fresh audits differ')
    rt=read(DEFAULT/'runtime.json')
    oldroot=ROOT/'data/moe/results/conv_three_arm_smoke_20260915_r1'
    old={r['job_id']:r for r in map(json.loads,(oldroot/'rows.jsonl').read_text().splitlines())}
    rows=[]
    for d in fresh['details']:
        directory=DEFAULT/d['job_id'];events=[json.loads(l) for l in (directory/'budget_journal.jsonl').read_text().splitlines()]
        props=[e for e in events if e['kind']=='PROPERTY_RESULT']
        parent=old[d['job_id']]
        row={**d,'old':{'status':parent['status'],'outer_timeout':parent['outer_timeout'],
                      'wall_seconds':parent['wall_seconds'],'complete_package':parent['package'] is not None},
            'old_terminal_sha256':_sha256(oldroot/d['job_id']/'terminal.json'),
            'property_status_counts':dict(Counter(e['result'].get('status') for e in props)),
            'property_solver_status_counts':dict(Counter(str(e['result'].get('solver_status')) for e in props)),
            'native_accounting':native_summary(events)}
        rows.append(row)
    return {'schema':'CONV_BUDGET_SMOKE_V2_REVIEW','status':'PASS','issues':[],
        'execution_head':rt['git_head'],'raw_root':str(DEFAULT),'independent_and_fresh_audits_match':True,
        'audit':fresh,'comparison':rows,'tests_passed':74,
        'no_new_SAFE':all(r['status']!='SAFE' for r in rows),
        'old_outer_timeouts':sum(r['old']['outer_timeout'] for r in rows),
        'new_outer_timeouts':sum(r['outer_timeout'] for r in rows),
        'old_complete_packages':sum(r['old']['complete_package'] for r in rows),
        'new_complete_packages':sum(r['complete_package'] for r in rows),
        'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)),'bytes':p.stat().st_size,'sha256':_sha256(p)}
                              for p in sorted(DEFAULT.rglob('*')) if p.is_file()],
        'full_started':False,'scope':'Four old-input budget/terminal controls, not new verification efficacy. HZ numerical policy unchanged; audit is not independent SAFE reproof.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();result=review()
    if args.check:
        if read(OUTPUT)!=result:raise ValueError('compact archive differs')
    else:
        if OUTPUT.exists():raise FileExistsError('archive exists; use --check')
        atomic_json(OUTPUT,result)
    print(json.dumps({'status':result['status'],'smoke_gate':result['audit']['smoke_gate'],
        'packages':result['new_complete_packages'],'old_outer_timeouts':result['old_outer_timeouts'],
        'new_outer_timeouts':result['new_outer_timeouts'],
        'native_max_overruns':{r['job_id']:r['native_accounting']['maximum_observed_return_deadline_overrun_seconds']
                               for r in result['comparison']}},indent=2))
