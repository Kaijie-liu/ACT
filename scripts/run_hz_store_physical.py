"""One frozen store/block CPU-CUDA attempt; reuse existing bounded execution."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from scoped_proof.io import ROOT, load, save, sha
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import identity, require
from scripts import hz_store_device_supervised as policy
from scripts import run_hz_store_device_supervision as controlled
from scripts import audit_store_device_supervision_v2 as accounting
from scripts import run_hz_physical_device as probe

DESIGN='docs/hz_store_physical_design_20261013.md'
CONTROL_ROOT=ROOT.parent/'baseline_runs/hz_store_device_supervision_20261013_r1'
BATCH_SECONDS=2400
VERSIONS=dict(torch='2.9.1+cu128',cuda_build='12.8',cpu_threads=1)


def roster():
    return [dict(name=case+'_'+('cpu' if dev=='cpu' else 'cuda'),case=case,device=dev)
            for i,case in enumerate(('half','negative','partial'))
            for dev in (('cpu','cuda:0') if i%2==0 else ('cuda:0','cpu'))]


def sources():
    names=(*policy.sources(),*probe.policy.sources(),DESIGN,
        'scripts/run_hz_store_physical.py','scripts/test_hz_store_physical.py',
        'scripts/audit_store_device_supervision_v2.py')
    return {n:sha(ROOT/n) for n in dict.fromkeys(names)}


def prerequisites():
    a=load(CONTROL_ROOT/'audit_v2.json'); r=load(CONTROL_ROOT/'audit_v2_source.json')
    require(a['status']=='PASS' and a['issues']==0 and a['audit_version']==2 and a['extra_tests']==6
        and a['numerical_reruns']==0 and a['cuda_calls']==0 and a['calls']==10 and a['tests']==12
        and a['sources']==policy.sources(),'exact V2 lifecycle outcome')
    require(a['audit_extension_source_sha256']==r['sha256']==sha(ROOT/r['file'])
        ==sha(CONTROL_ROOT/r['archived_file']) and r['file']=='scripts/audit_store_device_supervision_v2.py'
        and a['audit_extension']['sha256']==sha(CONTROL_ROOT/'audit_v2_source.json')
        and a['calls_sha256']==sha(CONTROL_ROOT/'calls.json')
        and a['freeze_sha256']==sha(CONTROL_ROOT/'freeze.json')
        and a['extra_tests_sha256']==sha(CONTROL_ROOT/'audit_v2_tests.json')
        and a['original_audit_failure_sha256']==sha(CONTROL_ROOT/'audit_v1_failure.json'),
        'V2 extension and retained failure binding')
    tests=load(CONTROL_ROOT/'audit_v2_tests.json')
    require(tests['tests']==6 and len(tests['outcomes'])==6 and all(v['status']=='PASS' for v in tests['outcomes']),
            'six complete V2 outcomes')
    controlled.validate_units(CONTROL_ROOT)
    return {n:sha(CONTROL_ROOT/n) for n in ('audit_v2.json','audit_v2_source.json',
        'audit_v2_tests.json','audit_v1_failure.json','unit_outcomes.json','freeze.json','calls.json','implementation.json')}


def make_freeze(wrapper_controls):
    path=Path(wrapper_controls).resolve(); require(path.parent.parent==ROOT.parent/'baseline_runs','controls archive')
    return dict(schema='STORE_DEVICE_PHYSICAL_FREEZE_V1',sources=policy.sources(),
        design_sha256=sha(ROOT/policy.DESIGN),
        spec_sha256_roster=[identity(policy.spec(r['case'],r['device'])) for r in roster()],
        controls_audit=str(CONTROL_ROOT/'audit_v2.json'),controls_audit_sha256=sha(CONTROL_ROOT/'audit_v2.json'),
        prerequisite_bindings=prerequisites(),batch_sources=sources(),roster=roster(),
        wrapper_controls=str(path),wrapper_controls_sha256=sha(path),runtime_versions=VERSIONS,
        request_budget_seconds=300,batch_budget_seconds=BATCH_SECONDS,preflight_attempts=1,
        attempts_per_slot=1,archive_limit_bytes=64*2**20)


def validate_wrapper(f):
    from scripts.test_hz_store_physical import NAMES
    record=load(f['wrapper_controls'],f['wrapper_controls_sha256'])
    require(record['status']=='PASS' and record['sources']==sources() and record['tests']==len(NAMES)
        and record['outcomes']==[dict(name=n,status='PASS') for n in sorted(NAMES)]
        and record['numerical_calls']==record['cuda_calls']==0,'exact wrapper controls')
    require(record['log_sha256']==sha(Path(f['wrapper_controls']).with_name('tests.log')),'wrapper test log')
    archived=Path(f['wrapper_controls']).parent/'implementation'
    require(all(sha(archived/n)==h for n,h in record['sources'].items()),'wrapper control source snapshots')


def validate(permit):
    path=Path(permit['path']); require(path.is_absolute() and path.parent==ROOT/'configs','committed freeze')
    f=load(path,policy.required_hash(permit['sha256']))
    require(f==make_freeze(f['wrapper_controls']),'complete execution freeze equality')
    validate_wrapper(f)
    require(probe.policy.protocol()['gpu_uuid']==policy.UUID,'reused preflight UUID')
    for r in roster(): policy.validate_spec(policy.spec(r['case'],r['device']),permit)
    return f


def inventory(root,*,enforce=True):
    paths=[p for p in root.rglob('*') if p.is_file()]
    require(not any(p.is_symlink() for p in root.rglob('*')),'archive symlink')
    if enforce: require(sum(p.stat().st_size for p in paths)<=64*2**20,'physical batch archive cap')
    return {p.relative_to(root).as_posix():sha(p) for p in paths if p.name!='summary.json' or p.parent!=root}


def check_call(root,obs):
    checked=engine.audit(policy,root,observation=obs,recheck=True)
    t=load(root/'terminal.json'); inv=load(root/'invocation.json'); accepted=t['accepted']
    complete=checked['execution_status']==policy.DONE
    p=policy.payload(root,inv,accepted['inputs']['produce']) if complete else None
    if p is not None: require(p['environment']==VERSIONS,'frozen producer software')
    return dict(execution=checked,mathematics=None if not complete else accepted['checked']['aggregation'],
        operations=accounting.costs(root,t),stage_seconds=t['stage_seconds'],parent_seconds=t['parent_seconds'],
        final_receipt_seconds=obs['end']-inv['start']-t['seconds'],seconds=obs['end']-inv['start'],
        source=inv['spec']['source'],contract=None if p is None else p['contract'],
        store=None if p is None else load(root/'evidence/contract.json',p['contract'])['store'],
        environment=None if p is None else p['environment'],candidate=None if p is None else p['candidate'],
        stage_peak_rss={s['phase']:s['sampled_peak_rss'] for s in t['stages']},
        cuda_intent=(root/'cuda_intent.json').exists())


def execute_slots(root,permit,gate,deadline,*,invoke=engine.supervise,recheck=check_call):
    calls=[]
    if not gate['ready']: return calls
    for i,row in enumerate(roster()):
        if time.monotonic()+300>=deadline:
            save(root/'batch_deadline.json',dict(next_slot=i,at=time.monotonic(),deadline=deadline)); break
        inventory(root); s=policy.spec(row['case'],row['device'])
        if time.monotonic()+300>=deadline:
            save(root/'batch_deadline.json',dict(next_slot=i,at=time.monotonic(),deadline=deadline)); break
        begin=time.monotonic()
        save(root/f'started_{i}.json',dict(slot=i,spec=s,begin=begin,pending=list(range(i+1,6))))
        obs=None; error=None; checked=None; audit_begin=None
        try:
            require(time.monotonic()+300<deadline,'allowance exhausted before owned request launch')
            result=invoke(policy,root/row['name'],s,spec_sha256=identity(s),permit=permit)
            obs=dict(begin=begin,end=time.monotonic(),result=result)
            save(root/f'observed_{i}.json',obs)
            audit_begin=time.monotonic(); checked=recheck(root/row['name'],obs)
        except Exception as exc:
            error=repr(exc)
            if obs is None:
                obs=dict(begin=begin,end=time.monotonic(),result=None,error=error)
                save(root/f'observed_{i}.json',obs)
        end=time.monotonic()
        keep=error is None and obs['result']['status']==policy.DONE and obs['end']-begin<300
        entry=dict(slot=i,request=row,observation=obs,checked=checked,error=error,
            post_return_audit_seconds=0. if audit_begin is None else end-audit_begin,
            audited_at=end,continue_allowed=keep)
        save(root/f'audited_{i}.json',entry); calls.append(entry)
        save(root/f'pending_after_{i}.json',dict(pending=list(range(i+1,6)),continue_allowed=keep))
        print(json.dumps(dict(slot=i,request=row,status=obs['result']['status'] if obs['result'] else 'API_ERROR')),flush=True)
        if not keep: break
    return calls


def compare(cpu,gpu):
    require(cpu['source']==gpu['source'] and cpu['contract']==gpu['contract'] and cpu['store']==gpu['store']
        and cpu['environment']==gpu['environment']==VERSIONS,'paired source/property/software identity')
    a,b=cpu['mathematics'],gpu['mathematics']
    require(all(a[k]==b[k] for k in ('required','required_endpoints','checked_endpoints','missing_endpoints')),
            'complete paired obligation accounting')
    rows=[]
    require(len(a['results'])==len(b['results'])==a['required'],'complete property inventory')
    for x,y in zip(a['results'],b['results']):
        require(all(x[k]==y[k] for k in ('pair','other','endpoints')),'paired obligation order')
        rows.append(dict(pair=x['pair'],other=x['other'],cpu=x['lower'],gpu=y['lower'],
            difference=None if x['lower'] is None or y['lower'] is None else str(Fraction(y['lower'])-Fraction(x['lower'])),
            gained=not x['positive'] and y['positive'],lost=x['positive'] and not y['positive']))
    return dict(rows=rows,gained=sum(r['gained'] for r in rows),lost=sum(r['lost'] for r in rows),
        cpu_mathematical_status=a['status'],gpu_mathematical_status=b['status'],
        cpu_seconds=cpu['seconds'],gpu_seconds=gpu['seconds'],
        zero_loss_control=not any(r['lost'] for r in rows),speedup_established=False)


def batch(root,permit):
    begin=time.monotonic(); f=validate(permit)
    root.mkdir(parents=True,exist_ok=False)
    save(root/'start.json',dict(begin=begin,deadline=begin+BATCH_SECONDS,planned_calls=6))
    save(root/'permit.json',permit); save(root/'freeze.json',f)
    for name in f['batch_sources']:
        target=root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
    error=None; gate=None; calls=[]
    try:
        before=time.monotonic()
        with patch.object(controlled,'costs',accounting.costs): rechecked=controlled.audit(CONTROL_ROOT)
        save(root/'prerequisites_recheck.json',dict(report=rechecked,seconds=time.monotonic()-before))
        gate=probe.preflight(root/'preflight'); gate['api_return_at']=time.monotonic()
        save(root/'preflight_observed.json',gate)
        probe.audit_preflight(root/'preflight',gate)
        calls=execute_slots(root,permit,gate,begin+BATCH_SECONDS)
    except Exception as exc: error=repr(exc)
    if error:
        calls=[load(root/f'audited_{i}.json') for i in range(6) if (root/f'audited_{i}.json').exists()]
    attempted=[i for i in range(6) if (root/f'started_{i}.json').exists()]
    status=('ERROR' if error else 'NOT_ADMITTED' if not gate['ready'] else
            'COMPLETED' if len(calls)==6 and all(c['continue_allowed'] for c in calls) else 'STOPPED_WITH_PREFIX')
    value=dict(schema='STORE_PHYSICAL_BATCH_V1',status=status,error=error,calls=calls,
        attempted_slots=attempted,pending_slots=list(range(len(attempted),6)),preflight=gate,
        seconds_before_final_publication=time.monotonic()-begin,
        inventory=inventory(root,enforce=False),real_model_calls=0,complete_real_model_proofs=0)
    save(root/'summary.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('inventory','calls','preflight')}))


def audit(root):
    begin=time.monotonic(); s=load(root/'summary.json'); permit=load(root/'permit.json'); f=validate(permit)
    require(load(root/'freeze.json')==f and s['inventory']==inventory(root),'whole physical archive identity')
    require(all(sha(root/'implementation'/n)==h for n,h in f['batch_sources'].items()),'archived implementation')
    require(s['schema']=='STORE_PHYSICAL_BATCH_V1' and s['real_model_calls']==s['complete_real_model_proofs']==0,
            'unchanged scope')
    observation=load(root.with_name(root.name+'_observed.json'))
    start=load(root/'start.json')
    for value in (observation['begin'],observation['end'],start['begin'],s['seconds_before_final_publication']): policy.finite(value)
    require(observation['summary_sha256']==sha(root/'summary.json') and observation['begin']<=start['begin']
            and observation['end']>=start['begin']+s['seconds_before_final_publication'],'whole batch observed return')
    if s['error'] is not None:
        return dict(status='FAIL',execution_status=s['status'],error=s['error'],
                    attempted_slots=s['attempted_slots'],pending_slots=s['pending_slots'],
                    observed_batch_seconds=observation['end']-start['begin'],summary_sha256=sha(root/'summary.json'))
    gate=load(root/'preflight_observed.json')
    policy.finite(start['begin']); policy.finite(s['seconds_before_final_publication'])
    require(start['deadline']==start['begin']+BATCH_SECONDS and start['planned_calls']==6
        and s['preflight']==gate and gate['api_return_at']>=start['begin'],'batch/preflight clock')
    pre=probe.audit_preflight(root/'preflight',gate); calls=s['calls']
    launch=load(root/'preflight/launch.json')
    require(launch['start']>=start['begin'] and gate['api_return_at']>=launch['start']+gate['seconds'],
            'preflight return chronology')
    receipt=load(root/'prerequisites_recheck.json'); policy.finite(receipt['seconds'])
    with patch.object(controlled,'costs',accounting.costs): fresh=controlled.audit(CONTROL_ROOT)
    def stable(report):
        import copy
        r=copy.deepcopy(report); r.pop('offline_seconds')
        for row in r['rows'].values(): row.pop('offline_check_seconds')
        return r
    require(stable(receipt['report'])==stable(fresh),'saved prerequisite recheck')
    require(len(calls)<=6 and s['attempted_slots']==list(range(len(calls)))
        and s['pending_slots']==list(range(len(calls),6)) and (gate['ready'] or not calls),
            'complete denominator after admission')
    available={p.name for p in root.iterdir() if p.is_dir()}-{'preflight','implementation'}
    expected_dirs={r['name'] for r in roster()[:len(calls)]}
    optional=set() if not calls or calls[-1]['error'] is None else {calls[-1]['request']['name']}
    require(expected_dirs-optional<=available<=expected_dirs,'all launched call directories')
    for pattern in ('started_*.json','observed_*.json','audited_*.json','pending_after_*.json'):
        stem=pattern.split('*')[0]
        require({p.name for p in root.glob(pattern)}=={stem+str(i)+'.json' for i in range(len(calls))},'complete slot receipts')
    pairs={}; previous=gate['api_return_at']; intents=0; actual=0
    for i,c in enumerate(calls):
        row=roster()[i]; path=root/row['name']; obs=load(root/f'observed_{i}.json')
        require(c==load(root/f'audited_{i}.json') and c['slot']==i and c['request']==row
            and c['observation']==obs and obs['begin']>=previous,'sequential observed slots')
        require(load(root/f'started_{i}.json')==dict(slot=i,spec=policy.spec(row['case'],row['device']),
            begin=obs['begin'],pending=list(range(i+1,6))),'slot start identity')
        policy.finite(c['post_return_audit_seconds']); policy.finite(c['audited_at'])
        require(c['audited_at']>=obs['end'] and c['audited_at']-obs['end']>=c['post_return_audit_seconds']-1e-8,
                'post-return cost')
        previous=c['audited_at']
        if c['error'] is not None:
            require(i==len(calls)-1 and not c['continue_allowed'],'error stopped batch')
            return dict(status='FAIL',execution_status=s['status'],error=c['error'],
                        started_calls=len(calls),pending_slots=s['pending_slots'],summary_sha256=sha(root/'summary.json'))
        inv=load(path/'invocation.json')
        require(inv['spec']==policy.spec(row['case'],row['device']) and inv['permit']==permit and inv['budget']==300,
                'per-request freeze')
        require(inv['start']+300<start['deadline'],'actual request had full batch allowance')
        checked=check_call(path,obs); require(checked==c['checked'],'independent exact result/cost recheck')
        keep=obs['result']['status']==policy.DONE and obs['end']-obs['begin']<300
        require(c['continue_allowed']==keep and (keep or i==len(calls)-1)
            and load(root/f'pending_after_{i}.json')==dict(pending=list(range(i+1,6)),continue_allowed=keep),'stop rule')
        intents+=checked['cuda_intent']
        if keep:
            pairs.setdefault(row['case'],{})[row['device']]=checked
            if row['device']=='cuda:0': actual+=1
    expected='NOT_ADMITTED' if not gate['ready'] else 'COMPLETED' if len(calls)==6 and all(c['continue_allowed'] for c in calls) else 'STOPPED_WITH_PREFIX'
    require(s['status']==expected,'derived batch status')
    if gate['ready'] and len(calls)<6 and (not calls or calls[-1]['continue_allowed']):
        cutoff=load(root/'batch_deadline.json')
        require(cutoff['next_slot']==len(calls) and cutoff['deadline']==start['deadline']
                and cutoff['at']+300>=start['deadline'],'remaining-budget refusal')
    require(start['begin']+s['seconds_before_final_publication']>=previous,'batch complete cost span')
    return dict(status='PASS',execution_status=expected if observation['end']<start['deadline'] else 'BATCH_BUDGET_EXCEEDED',
        recorded_execution_status=expected,planned_calls=6,started_calls=len(calls),
        pending_slots=s['pending_slots'],preflight=pre,cuda_intents=intents,complete_cuda_calls=actual,
        paired={name:compare(arms['cpu'],arms['cuda:0']) for name,arms in pairs.items() if len(arms)==2},
        summary_sha256=sha(root/'summary.json'),freeze_sha256=permit['sha256'],offline_seconds=time.monotonic()-begin,
        observed_batch_seconds=observation['end']-start['begin'],
        batch_within_budget=observation['end']<start['deadline'],
        prerequisite_check_seconds=receipt['seconds'],
        real_model_calls=0,complete_real_model_proofs=0,gpu_speedup_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('batch','audit','freeze-metadata'))
    p.add_argument('path',type=Path); p.add_argument('--freeze',type=Path); p.add_argument('--sha'); p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.action=='freeze-metadata': print(json.dumps(make_freeze(a.path),indent=2))
    else:
        root=a.path.resolve(); require(root.is_relative_to(ROOT.parent/'baseline_runs'),'project archive')
        if a.action=='batch':
            before=time.monotonic(); batch(root,dict(path=str(a.freeze.resolve()),sha256=a.sha))
            after=time.monotonic()
            save(root.with_name(root.name+'_observed.json'),dict(begin=before,end=after,
                 summary_sha256=sha(root/'summary.json')))
        else:
            value=audit(root)
            if a.report:
                require(not a.report.resolve().is_relative_to(root),'audit report outside immutable batch archive')
                save(a.report,value)
            print(json.dumps(value))
