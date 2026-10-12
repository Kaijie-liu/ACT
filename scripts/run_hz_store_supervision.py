"""Freeze, run and independently audit the finite checked-store lifecycle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import identity, require
from scripts import hz_store_supervised as policy
from scripts.run_hz_endpoint_supervision import event_costs as base_costs


def roster():
    return [(k,policy.spec(k)) for k in ('half','negative','partial')]+[
        (k,policy.spec(control=k)) for k in policy.FAULTS]


def execute_calls(root,*,invoke=engine.supervise,publish=save,read=load,admit=None):
    calls={}; fixed=roster()
    for i,(name,s) in enumerate(fixed):
        if admit is not None: admit(root)
        pending=[n for n,_ in fixed[i+1:]]
        publish(root/('started_'+name+'.json'),dict(name=name,spec_sha256=identity(s),pending=pending))
        begin=time.monotonic()
        try:
            r=invoke(policy,root/name,s,spec_sha256=identity(s))
            calls[name]=dict(begin=begin,end=time.monotonic(),result=r)
            publish(root/('observed_'+name+'.json'),calls[name])
            t=read(root/name/'terminal.json')
            clean=r['status']!='CLEANUP_INCOMPLETE' and all(policy.cleanup_confirmed(s) for s in t['stages'])
        except BaseException as exc:
            calls.setdefault(name,dict(begin=begin,end=time.monotonic(),api_error=repr(exc),cleanup_confirmed=False))
            publish(root/('api_error_'+name+'.json'),dict(error=repr(exc),cleanup_confirmed=False))
            clean=False
        publish(root/('pending_after_'+name+'.json'),dict(pending=pending,cleanup_confirmed=clean))
        print(json.dumps(dict(call=name,status=calls[name].get('result',{}).get('status','API_ERROR'))),flush=True)
        if not clean:
            publish(root/'batch_stop.json',dict(after=name,pending=pending)); break
    return calls


def costs(root,t):
    result=base_costs(root,t)
    for stage in t['stages']:
        if stage['status']!='COMPLETED': continue
        phase=stage['phase']; path=root/(phase+'_events.jsonl')
        events=[json.loads(line) for line in path.read_text().splitlines()]
        require(events and events[-1].get('event')=='WORKER_COMPLETE'
                and events[-1].get('phase')==phase, 'complete phase cost receipt')
        required=['final_serialization']+(['imports','source','plan','candidates'] if phase=='produce' else [])
        for name in required:
            require(len([r for r in result[phase] if r['operation']==name and r['status']=='EXIT'])==1,
                    'required completed operation cost')
    return result


def stage_bytes(root):
    value=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(value<=64*2**20, 'frozen stage regular-file payload cap'); return value


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    sources=policy.sources()
    for name,digest in sources.items():
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    save(root/'implementation.json',sources)
    save(root/'freeze.json',dict(protocol=sha(ROOT/policy.DESIGN),sources=sha(root/'implementation.json'),
        roster=[dict(name=n,spec=s) for n,s in roster()],budget_seconds=300,device='cpu',threads=1,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    with (root/'units.log').open('xb') as log:
        tests=subprocess.run([PYTHON,'-B','-S','-m','scripts.test_hz_store_supervised','--report',str(root/'unit_outcomes.json')],
                             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    save(root/'units.json',dict(returncode=tests.returncode,log=sha(root/'units.log')))
    require(tests.returncode==0, 'pure controls must pass before calls')
    validate_units(root)
    stage_bytes(root)
    calls=execute_calls(root,admit=stage_bytes); save(root/'calls.json',calls)
    save(root/'summary.json',dict(calls=len(calls),calls_sha256=sha(root/'calls.json'),
        implementation_sha256=sha(root/'implementation.json'),payload_bytes=stage_bytes(root),
        real_requests=0,cuda_calls=0,performance_claim=False))
    return dict(calls=len(calls),all_started=len(calls)==len(roster()))


def audit(root):
    started=time.monotonic(); summary=load(root/'summary.json'); calls=load(root/'calls.json')
    sources=load(root/'implementation.json'); fixed=roster()
    require(sources==policy.sources() and all(sha(root/'implementation'/n)==h for n,h in sources.items()),
            'frozen transitive implementation')
    require(set(calls)=={n for n,_ in fixed} and summary['calls_sha256']==sha(root/'calls.json')
            and summary['implementation_sha256']==sha(root/'implementation.json'), 'complete execution roster')
    freeze=load(root/'freeze.json')
    require(freeze['sources']==sha(root/'implementation.json') and freeze['protocol']==sha(ROOT/policy.DESIGN)
            and freeze['roster']==[dict(name=n,spec=s) for n,s in fixed], 'pre-execution freeze')
    unit=load(root/'units.json'); require(unit==dict(returncode=0,log=sha(root/'units.log')), 'control gate')
    validate_units(root)
    rows={}
    for i,(name,s) in enumerate(fixed):
        path=root/name; inv=load(path/'invocation.json'); t=load(path/'terminal.json')
        require(inv['spec']==s and load(root/('observed_'+name+'.json'))==calls[name], 'request/observer binding')
        pending=[n for n,_ in fixed[i+1:]]
        require(load(root/('started_'+name+'.json'))==dict(name=name,spec_sha256=identity(s),pending=pending)
                and load(root/('pending_after_'+name+'.json'))==dict(pending=pending,cleanup_confirmed=True),
                'durable start/pending inventory')
        result=engine.audit(policy,path,observation=calls[name],recheck=False)
        expected=policy.DONE if not s['control'] else 'TIMEOUT' if name.endswith('_delay') else 'ERROR'
        require(result['execution_status']==expected, 'frozen execution status')
        operation_costs=costs(path,t)
        if s['control']:
            events=[json.loads(line) for f in path.glob('*_events.jsonl') for line in f.read_text().splitlines()]
            require(any(e.get('event')=='FAULT_REACHED' and e.get('control')==name
                        and e.get('invocation')==inv['invocation'] for e in events), 'registered fault not reached')
            require(t['accepted'] is None, 'failed invocation cannot accept')
            if name=='serialization_delay':
                refs=[e for e in events if e.get('event')=='TILE_PUBLISHED']
                require(len(refs)==1 and refs[0]['index']==0 and not (path/'evidence/proof/manifest.json').exists(),
                        'completed tile prefix, not a complete proof')
                ref=refs[0]['reference']; policy.counts(refs[0]['counts'])
                require(ref['file']=='00000.json' and sha(path/'evidence/proof'/ref['file'])==ref['sha256']
                        and (path/'evidence/proof'/ref['file']).stat().st_size==ref['bytes'], 'published prefix identity')
            aggregate=None; offline=0
        else:
            before=time.monotonic(); received=t['accepted']; inputs=received['inputs']
            checked=policy.check(path,inv,inputs['produce'],time.monotonic()+30)
            require(checked==load(path/'check.json',inputs['check']), 'independent replay exact equality')
            offline=time.monotonic()-before; aggregate=received['checked']['aggregation']
            expected_math={'half':('CHECKED_POSITIVE_DECLARED_SOURCE',3,5,0),
                           'negative':('UNKNOWN_NONPOSITIVE',3,6,0),
                           'partial':('UNKNOWN_MISSING_EVIDENCE',2,4,1)}[name]
            require((aggregate['status'],aggregate['positive'],aggregate['checked_endpoints'],
                     aggregate['missing_endpoints'])==expected_math, 'frozen mathematical outcome')
        rows[name]=dict(execution=result['execution_status'],aggregate=aggregate,
                       seconds=calls[name]['end']-inv['start'],stage_seconds=t['stage_seconds'],
                       parent_seconds=t['parent_seconds'],operations=operation_costs,
                       post_terminal_to_observation_seconds=calls[name]['end']-inv['start']-t['seconds'],
                       cumulative_counts=None if t['accepted'] is None else t['accepted']['counts'],
                       counters_complete=t['accepted'] is not None,offline_check_seconds=offline,
                       nested_files=len(t['evidence_inventory']))
    require(summary['real_requests']==summary['cuda_calls']==0 and summary['performance_claim'] is False,
            'no real/device/performance upgrade')
    return dict(status='PASS',calls=len(rows),rows=rows,offline_audit_seconds=time.monotonic()-started,
                payload_bytes=stage_bytes(root),complete_real_model_proofs=0,gpu_calls=0,
                provenance='synthetic declared-real sources, not checkpoint/native execution')


def validate_units(root):
    from scripts.test_hz_store_supervised import TEST_NAMES
    data=load(root/'unit_outcomes.json')
    require(data==dict(tests=len(TEST_NAMES),outcomes=[dict(name=n,status='PASS') for n in TEST_NAMES],
                      exceptional={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}),
            'exact complete pure-test roster without skips')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('run','audit')); p.add_argument('root',type=Path)
    a=p.parse_args()
    result=run(a.root) if a.mode=='run' else audit(a.root)
    if a.mode=='audit': save(a.root/'audit.json',result)
    print(json.dumps(result if a.mode=='run' else {k:v for k,v in result.items() if k!='rows'}))
