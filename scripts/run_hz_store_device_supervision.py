"""Finite controls and offline audit; physical execution is not a run mode."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import identity, require
from scripts import hz_store_device_supervised as policy
from scripts.run_hz_store_supervision import costs, stage_bytes


def roster():
    return [(c,policy.spec(c)) for c in ('half','negative','partial')]+[
        (c,policy.spec(device='cuda:0' if c in policy.GPU_STUBS else 'cpu',control=c)) for c in policy.FAULTS]


def expected(name):
    if name in ('half','negative','partial'): return policy.DONE
    if name in ('admission_busy_stub','pre_cuda_busy_stub'): return 'RESOURCE_UNAVAILABLE'
    if name in ('release_owned_stays','release_delay'): return 'DEVICE_RELEASE_UNCONFIRMED'
    if name=='readback_delay': return 'TIMEOUT'
    return 'ERROR'


def may_continue(root,name,result,terminal):
    if not all(policy.cleanup_confirmed(s) for s in terminal['stages']): return False
    if result['status']!=expected(name): return False
    if result['status']=='DEVICE_RELEASE_UNCONFIRMED':
        # Only these predeclared no-CUDA simulations may continue after refusal.
        return name in ('release_owned_stays','release_delay') and not (root/'cuda_intent.json').exists()
    return result['status']!='CLEANUP_INCOMPLETE'


def execute_calls(root,invoke=engine.supervise):
    calls={}; fixed=roster()
    for i,(name,s) in enumerate(fixed):
        stage_bytes(root); pending=[n for n,_ in fixed[i+1:]]
        save(root/('started_'+name+'.json'),dict(spec_sha256=identity(s),pending=pending))
        begin=time.monotonic(); clean=False
        try:
            result=invoke(policy,root/name,s,spec_sha256=identity(s))
            calls[name]=dict(begin=begin,end=time.monotonic(),result=result)
            save(root/('observed_'+name+'.json'),calls[name])
            clean=may_continue(root/name,name,result,load(root/name/'terminal.json'))
        except BaseException as exc:
            calls.setdefault(name,dict(begin=begin,end=time.monotonic(),api_error=repr(exc)))
            save(root/('api_error_'+name+'.json'),dict(error=repr(exc),cleanup_confirmed=False))
        save(root/('pending_after_'+name+'.json'),dict(pending=pending,continuation_allowed=clean))
        print(json.dumps(dict(call=name,status=calls[name].get('result',{}).get('status','API_ERROR'))),flush=True)
        if not clean:
            save(root/'batch_stop.json',dict(after=name,pending=pending)); break
    return calls


def validate_units(root):
    from scripts.test_hz_store_device_supervised import TEST_NAMES
    data=load(root/'unit_outcomes.json')
    require(data==dict(tests=12,outcomes=[dict(name=n,status='PASS') for n in TEST_NAMES],
                      exceptional={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}),
            'exact complete pure control inventory')


def run(root):
    root.mkdir(parents=True,exist_ok=False); sources=policy.sources()
    for name in sources:
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    save(root/'implementation.json',sources)
    save(root/'freeze.json',dict(sources=sources,design=sha(ROOT/policy.DESIGN),budget_seconds=300,
        roster=[dict(name=n,spec=s) for n,s in roster()],physical_cuda=False,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    with (root/'units.log').open('xb') as log:
        result=subprocess.run([PYTHON,'-B','-I','-S','-c',
            f'import sys,runpy;sys.path.insert(0,{str(ROOT)!r});runpy.run_module("scripts.test_hz_store_device_supervised",run_name="__main__")',
            '--report',str(root/'unit_outcomes.json')],stdout=log,stderr=subprocess.STDOUT,timeout=60)
    save(root/'units.json',dict(returncode=result.returncode,log=sha(root/'units.log')))
    require(result.returncode==0,'pure controls must pass'); validate_units(root)
    calls=execute_calls(root); save(root/'calls.json',calls)
    save(root/'summary.json',dict(calls=len(calls),calls_sha256=sha(root/'calls.json'),
        sources_sha256=sha(root/'implementation.json'),payload_bytes=stage_bytes(root),physical_cuda=False))
    require(len(calls)==10 and not (root/'batch_stop.json').exists(),'complete once-only roster')


def audit(root):
    start=time.monotonic(); summary=load(root/'summary.json'); calls=load(root/'calls.json')
    sources=load(root/'implementation.json'); fixed=roster(); freeze=load(root/'freeze.json')
    require(sources==policy.sources() and all(sha(root/'implementation'/n)==h for n,h in sources.items()),'implementation identity')
    require(freeze['sources']==sources and freeze['design']==sha(ROOT/policy.DESIGN)
            and freeze['roster']==[dict(name=n,spec=s) for n,s in fixed] and freeze['physical_cuda'] is False,
            'pre-execution freeze identity')
    require(summary['calls']==10 and summary['calls_sha256']==sha(root/'calls.json')
            and summary['sources_sha256']==sha(root/'implementation.json') and summary['physical_cuda'] is False
            and set(calls)=={n for n,_ in fixed},'complete denominator')
    require(load(root/'units.json')==dict(returncode=0,log=sha(root/'units.log')),'unit process outcome')
    validate_units(root); rows={}
    for i,(name,s) in enumerate(fixed):
        path=root/name; inv=load(path/'invocation.json'); terminal=load(path/'terminal.json')
        pending=[n for n,_ in fixed[i+1:]]
        require(inv['spec']==s and calls[name]==load(root/('observed_'+name+'.json'))
                and load(root/('started_'+name+'.json'))==dict(spec_sha256=identity(s),pending=pending)
                and load(root/('pending_after_'+name+'.json'))==dict(pending=pending,continuation_allowed=True),
                'start/pending/observation binding')
        report=engine.audit(policy,path,observation=calls[name],recheck=False)
        require(report['execution_status']==expected(name) and not (path/'cuda_intent.json').exists()
                and all(v['cuda_visible_devices']=='' for v in terminal['stages'])
                and may_continue(path,name,calls[name]['result'],terminal),'terminal/device identity')
        operation_costs=costs(path,terminal)
        events=[json.loads(line) for f in path.glob('*_events.jsonl') for line in f.read_text().splitlines()]
        agg=None; offline=0.
        if s['control']:
            require(any(e.get('event')=='FAULT_REACHED' and e.get('control')==name
                        and e.get('invocation')==inv['invocation'] and e.get('physical_cuda') is False for e in events),
                    'intended fault actually reached')
            require(terminal['accepted'] is None,'fault cannot accept')
            if name=='readback_delay':
                refs=[e for e in events if e.get('event')=='TILE_PUBLISHED']
                require(len(refs)==1 and refs[0]['index']==0 and not (path/'evidence/proof/manifest.json').exists(),
                        'readback deadline preserves exactly first tile')
                ref=refs[0]['reference']; policy.base.counts(refs[0]['counts'])
                require(ref['file']=='00000.json' and sha(path/'evidence/proof'/ref['file'])==ref['sha256']
                        and (path/'evidence/proof'/ref['file']).stat().st_size==ref['bytes'],'partial tile hash/size')
            if name=='wrong_candidate_metadata':
                try: policy.payload(path,inv,load(path/'produce_stage.json')['output_sha256'])
                except ValueError as exc: require(str(exc)=='candidate algorithm/device','specific metadata refusal')
                else: raise ValueError('bad device metadata accepted')
        else:
            inputs=terminal['accepted']['inputs']; before=time.monotonic()
            checked=policy.check(path,inv,inputs['produce'],time.monotonic()+30)
            require(checked==load(path/'check.json',inputs['check']),'independent complete exact recheck')
            offline=time.monotonic()-before; agg=checked['result']['aggregation']
            wanted={'half':(3,3,5,0),'negative':(3,6,6,0),'partial':(2,3,4,1)}[name]
            require((agg['positive'],agg['required'],agg['checked_endpoints'],agg['missing_endpoints'])==wanted,
                    'unchanged full mathematical outcomes')
        rows[name]=dict(status=report['execution_status'],aggregation=agg,release=report['release'],
            seconds=calls[name]['end']-inv['start'],stage_seconds=terminal['stage_seconds'],
            parent_seconds=terminal['parent_seconds'],operations=operation_costs,
            final_receipt_seconds=calls[name]['end']-inv['start']-terminal['seconds'],
            counts=None if terminal['accepted'] is None else terminal['accepted']['counts'],
            offline_check_seconds=offline,nested_files=len(terminal['evidence_inventory']))
    return dict(status='PASS',issues=0,calls=10,tests=12,rows=rows,sources=sources,cuda_calls=0,
                freeze_sha256=sha(root/'freeze.json'),calls_sha256=sha(root/'calls.json'),
                offline_seconds=time.monotonic()-start,payload_bytes=stage_bytes(root),real_requests=0,
                performance_claim=False,complete_real_model_proofs=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('run','audit')); p.add_argument('root',type=Path); a=p.parse_args()
    if a.mode=='run': run(a.root)
    else:
        result=audit(a.root); save(a.root/'audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources')}))
