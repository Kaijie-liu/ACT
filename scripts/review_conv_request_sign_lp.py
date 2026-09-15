"""Fresh exact all-obligation review; no model or solver execution."""
import argparse
from fractions import Fraction
import hashlib
import json

from scripts.conv_request_sign_lp_contract import ROOT,DEFAULT,FREEZE,read,save,sha,git,verify_freeze,validate_job
from scripts.analyze_conv_full_v2_obligations import parse

OUTPUT=ROOT/'act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json'


def check_journal(path,job_hash,pairs,label):
    lines=path.read_bytes().splitlines(keepends=True)
    if not lines or any(not line.endswith(b'\n') for line in lines):raise ValueError('partial journal')
    previous='0'*64;clock=-1;events=[]
    for seq,line in enumerate(lines):
        e=json.loads(line);claimed=e.pop('sha256')
        canonical=json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if (e['seq']!=seq or e['previous']!=previous or hashlib.sha256(canonical).hexdigest()!=claimed
                or (seq and e['clock_elapsed']<clock)):raise ValueError('journal chain/clock drift')
        clock=e.get('clock_elapsed',clock);previous=claimed;events.append(e)
    if events[0]['identity']!={'request_proof_job_sha256':job_hash}:raise ValueError('journal wrong request')
    properties,native,stack=parse(events,[tuple(p) for p in pairs],label)
    if (stack or properties or any(n['end'] is None for n in native.values())
            or any(e['kind']=='WORK_COMPLETE' for e in events)):
        raise ValueError('unfinished native scope, property solve or verifier verdict during generation')
    return {'status':'PASS','events':len(events),'construction_native_calls':len(native),
            'weighted_property_queries':0,'production_verdict_emitted':False,
            'scope':'Hash/allocation/terminal check, NOT a proof of upstream construction bounds'}


def review():
    from scripts.check_conv_request_sign_lp import check_directory,TRUSTED
    f=verify_freeze();rt=read(DEFAULT/'runtime.json');publication=rt['publication']
    if (rt['state']!='COMPLETED_CHECKED' or rt['unattempted'] or rt['extra_queries_queued']
            or rt['freeze_sha256']!=sha(FREEZE) or len(rt['cases'])!=2):raise ValueError('run incomplete')
    if (publication['local_head']!=rt['execution_head'] or publication['remote_head']!=rt['execution_head']
            or publication['gate']!='REMOTE_EQUALS_LOCAL_BEFORE_RUN_ROOT_CREATED'
            or publication['remote_ref']!='refs/heads/feat/moe-route-verification'
            or not publication['confirmed_unix']<rt['started_unix']):raise ValueError('publication ordering not confirmed')
    # The execution commit actually contains the identical freeze, not merely a
    # remote-equality claim for an unrelated revision.
    committed=git('show',rt['execution_head']+':'+str(FREEZE.relative_to(ROOT)))
    if json.loads(committed)!=f:raise ValueError('execution commit missing freeze')
    rows=[]
    for case,job in zip(rt['cases'],f['jobs']):
        if case['case']!=job['case']:raise ValueError('scope/order changed')
        d=DEFAULT/job['case']['job_id'];validate_job(d);req=job['parent_request']
        for p,h in [(req['subject']['checkpoint'],req['subject']['checkpoint_sha256']),
                    (req['tensors']['path'],req['tensors']['sha256']),(req['config']['path'],req['config']['sha256'])]:
            if sha(p)!=h:raise ValueError('model/tensors/config changed')
        stages=case['stages'];expected={'capture':300,'proposal':900,'check':300,'independent':300}
        if set(stages)!=set(expected):raise ValueError('stage missing')
        for name,cap in expected.items():
            v=stages[name]
            if (v!=read(d/(name+'.terminal.json')) or v['state']!='COMPLETED' or v['return_code']!=0
                    or v['cap_seconds']!=cap or not 0<v['wall_seconds']<=cap):raise ValueError('stage cap/terminal mismatch')
        fresh=check_directory(d)
        if fresh!=case['result'] or fresh!=read(d/'check.json') or fresh!=read(d/'independent.json'):
            raise ValueError('independent checks disagree')
        m=read(d/'manifest.json');route=m['routes'];groups={k:{tuple(p) for p in route[k]} for k in ('feasible','infeasible','unresolved')}
        branches=route['branches']
        if (len(branches)!=6 or len({tuple(b['route_set']) for b in branches})!=6
                or any(b['feasibility'] not in groups or tuple(b['route_set']) not in groups[b['feasibility']] for b in branches)):
            raise ValueError('route branch terminals do not match coverage')
        if not m['generation_complete'] or len(m['obligations'])!=9:raise ValueError('required scope not generated')
        for r in m['obligations']:
            if r['kind']=='lp' and not 0<r['proposal_seconds']<900:raise ValueError('invalid proposal accounting')
        journal=check_journal(d/'budget_journal.jsonl',sha(d/'job.json'),route['feasible'],req['sample']['label'])
        details=[]
        for row,checked in zip(m['obligations'],fresh['obligations']):
            v={**checked,'lower_bound_float_descriptive':float(Fraction(checked['lower_bound'])) if checked['lower_bound'] else None,
               'evidence_kind':row['kind']}
            if row['kind']=='lp':
                record=read(d/row['export']['file'])
                v.update(export=row['export'],certificate=row['certificate'],source_sha256=row['source_sha256'],
                         relaxed_binaries=record['n_relaxed_binaries'],factors=len(record['lp']['c']))
            details.append(v)
        rows.append({'case':case['case'],'result':fresh,'obligation_details':details,'stages':stages,'journal':journal,
                     'all_stage_seconds':sum(v['wall_seconds'] for v in stages.values()),
                     'minimum_float_descriptive':float(Fraction(fresh['minimum_lower_bound'])) if fresh['minimum_lower_bound'] else None,
                     'original_production_status':'TIMEOUT','original_status_changed':False,
                     'single_pair':len(route['feasible'])==1})
    return {'schema':'CONV_REQUEST_SIGN_LP_R1_REVIEW','evidence_audit_status':'PASS','evidence_issues':[],
            'execution_protocol_status':'PASS_REMOTE_CONFIRMED_BEFORE_RUN_ROOT','publication':publication,
            'execution_head':rt['execution_head'],'freeze_sha256':sha(FREEZE),'parent_artifacts_unchanged':True,
            'required_obligations':sum(r['result']['required_obligations'] for r in rows),
            'checked_positive_obligations':sum(r['result']['positive_obligations'] for r in rows),
            'conditional_complete_requests':sum(r['result']['status']=='CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_F0_LOWERING' for r in rows),
            'unconditional_or_deployed_float_SAFE':0,'new_route_changing_claims':0,'trusted_base':TRUSTED,
            'rows':rows,'statistical_scope':'Two post-selected observed single-pair controls, no population coverage/accuracy estimate',
            'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)),'bytes':p.stat().st_size,'sha256':sha(p)}
                                  for p in sorted(DEFAULT.rglob('*')) if p.is_file()]}


if __name__=='__main__':
    from scripts.check_conv_sign_lp import isolate
    isolate()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    result=review()
    if a.check:
        if read(OUTPUT)!=result:raise ValueError('archive differs')
    else:
        if OUTPUT.exists():raise FileExistsError('no overwrite')
        save(OUTPUT,result)
    print(json.dumps({k:result[k] for k in ('evidence_audit_status','execution_protocol_status',
        'required_obligations','checked_positive_obligations','conditional_complete_requests')},indent=2))
