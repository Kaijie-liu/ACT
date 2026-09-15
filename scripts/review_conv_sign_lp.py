"""Read-only fresh exact check and archival review of the two sign controls."""
import argparse
import hashlib
import json

from scripts.conv_sign_lp_contract import (ROOT, DEFAULT, FREEZE, read, save, sha,
                                           verify_freeze, validate_job)
from scripts.analyze_conv_full_v2_obligations import parse

OUTPUT = ROOT / 'act/pipeline/moe/results/conv_sign_lp_review_20260915_r1.json'


def journal_check(path, job_hash, scope):
    lines=path.read_bytes().splitlines(keepends=True)
    if not lines or not all(l.endswith(b'\n') for l in lines): raise ValueError('partial capture journal')
    previous='0'*64; events=[]; clock=-1.
    for seq,line in enumerate(lines):
        e=json.loads(line); claimed=e.pop('sha256')
        canonical=json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if e['seq']!=seq or e['previous']!=previous or hashlib.sha256(canonical).hexdigest()!=claimed:
            raise ValueError('capture journal hash chain differs')
        if seq and e['clock_elapsed']<clock: raise ValueError('nonmonotonic capture clock')
        clock=e.get('clock_elapsed',clock); previous=claimed; events.append(e)
    if events[0]['identity']!={'proof_control_job_sha256':job_hash}: raise ValueError('wrong capture identity')
    label=scope['row'].index(1)
    properties,natives,stack=parse(events,[tuple(p) for p in scope['pairs']],label)
    if stack or len(properties)!=1 or any(n['end'] is None for n in natives.values()):
        raise ValueError('capture not closed at one fixed obligation')
    p=next(iter(properties.values()))
    if (p['begin']['scope']!=scope or p['end'] is None or p['end']['kind']!='PROPERTY_RAISE'
            or p['end']['exception']!='CapturedProperty' or p['native']
            or any(e['kind']=='WORK_COMPLETE' for e in events)):
        raise ValueError('capture promoted to result or property native solve executed')
    return {'status':'PASS','events':len(events),'pre_capture_native_calls':len(natives),
            'property_queries_captured':1,'property_native_solves':0,'request_verdict_emitted':False,
            'scope':'Capture stop is intentional, not a TIMEOUT or SAFE. Hash/scope/budget accounting is not upstream-bound proof.'}


def review():
    from scripts.check_conv_sign_lp import check_directory
    freeze=verify_freeze(); rt=read(DEFAULT/'runtime.json')
    if (rt['state']!='COMPLETED_CHECKED' or rt['unattempted'] or rt['extra_queries_queued']
            or rt['freeze_sha256']!=sha(FREEZE) or len(rt['cases'])!=len(freeze['jobs'])):
        raise ValueError('incomplete or mismatched run')
    rows=[]
    for case,job in zip(rt['cases'],freeze['jobs']):
        if case['case']!=job['case']:raise ValueError('case order mismatch')
        d=DEFAULT/case['case']['job_id']; validate_job(d)
        req=job['parent_request']
        for path,digest in [(req['subject']['checkpoint'],req['subject']['checkpoint_sha256']),
                            (req['tensors']['path'],req['tensors']['sha256']),
                            (req['config']['path'],req['config']['sha256'])]:
            if sha(path)!=digest:raise ValueError('model/input/config drift')
        stages=case['stages']
        expected={'capture':300,'proposal':180,'check':300,'independent':300}
        if set(stages)!=set(expected):raise ValueError('missing stage')
        for name,cap in expected.items():
            v=stages[name]
            if (read(d/(name+'.terminal.json'))!=v or v['state']!='COMPLETED'
                    or v['return_code']!=0 or v['cap_seconds']!=cap or not 0<v['wall_seconds']<=cap):
                raise ValueError('stage did not close inside cap')
        fresh=check_directory(d)
        if fresh!=case['result'] or fresh!=read(d/'check.json') or fresh!=read(d/'independent.json'):
            raise ValueError('fresh independent proof check differs')
        journal=journal_check(d/'budget_journal.jsonl',sha(d/'job.json'),job['expected_scope'])
        snapshot=read(d/'common_facts.json')['payload']
        if (snapshot['feasible_route_sets']!=job['expected_scope']['pairs'] or not snapshot['route_sets_exact']
                or snapshot['identity']['model_state']!=req['subject']['model_state']
                or any(snapshot['identity'][k]!=req['sample'][k] for k in ('center','lower','upper'))
                or snapshot['identity']['property']!={'kind':'TOP1_ROBUST','classes':10,'clean_prediction':req['sample']['label']}):
            raise ValueError('regenerated request route/model identity differs')
        rows.append({'case':case['case'],'scope':job['expected_scope'],'result':fresh,'stages':stages,
                     'journal':journal,'historical_diagnostic_dual':job['historical_diagnostic_dual'],
                     'historical_lp_coefficient_identity_available':False,
                     'all_stage_seconds':sum(v['wall_seconds'] for v in stages.values()),
                     'artifact_bytes':sum(p.stat().st_size for p in d.rglob('*') if p.is_file())})
    return {'schema':'CONV_SIGN_LP_R1_REVIEW','evidence_audit_status':'PASS','evidence_issues':[],
            'execution_head':rt['execution_head'],'freeze_sha256':sha(FREEZE),'parent_artifacts_unchanged':True,
            'execution_protocol_status':'DECLARED_REMOTE_PUBLICATION_TIMING_DEVIATION',
            'publication_deviation':{
                'description':'Freeze commit e269821cf existed locally before launch. First push was rejected by GitHub Internal Server Error; run was launched before successful push acknowledgement. Immediate retry synchronized the identical commit while controls ran.',
                'source':'Operator tool transcript; exact publication timestamp was not programmatically recorded.',
                'failed_push_reported_utc':'2026-09-15T09:27:14Z',
                'configuration_or_local_commit_changed_during_run':False,
                'interpretation':'Evidence checks pass but prescribed push-before-launch ordering was not satisfied; not a fully conforming confirmatory run.'},
            'rows':rows,'checked_positive_properties':sum(r['result']['status']=='CHECKED_POSITIVE_SUPPLIED_F0_LP' for r in rows),
            'full_requests_certified':0,'statistical_unit':'two post-selected property controls, not an accuracy or coverage cohort',
            'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)),'bytes':p.stat().st_size,'sha256':sha(p)}
                                  for p in sorted(DEFAULT.rglob('*')) if p.is_file()],
            'scope':'Independent rational downstream bounds for supplied floating F0 HZ; no original request relabelled and no upstream F0/network reproof.'}


if __name__=='__main__':
    from scripts.check_conv_sign_lp import isolate
    isolate()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    result=review()
    if a.check:
        if read(OUTPUT)!=result:raise ValueError('archive differs from fresh check')
    else:
        if OUTPUT.exists():raise FileExistsError('no overwrite; use --check')
        save(OUTPUT,result)
    print(json.dumps({'evidence_audit':result['evidence_audit_status'],
                      'protocol':result['execution_protocol_status'],
                      'properties_positive':result['checked_positive_properties'],
                      'full_requests_certified':0,
                      'bounds':[r['result'].get('lower_bound_float_descriptive') for r in result['rows']]},indent=2))
