"""Exactly two frozen full-size synthetic capacity calls; no real admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_proof.io import PYTHON,load,save,sha
from scripts.h2_capacity_supervised import specification,supervise,audit,producer_sources,required_sha
from scripts.archive_h2_capacity_execution import derive as controls

CALL_ROOT=Path('/data1/Kane/MOE/baseline_runs/h2_capacity_full_execution_20261001_r1')
CONTROL_ROOT=Path('/data1/Kane/MOE/baseline_runs/h2_capacity_execution_controls_20261001_r3')
CONTROL_ARCHIVE=ROOT/'docs/h2_capacity_execution_controls_20261001_r3.json'
OBSERVER_ROOT=Path('/data1/Kane/MOE/baseline_runs/h2_capacity_observation_controls_20261001_r1')
ARMS=('endpoints','mccormick')


def gate():
    if controls(CONTROL_ROOT)!=load(CONTROL_ARCHIVE): raise ValueError('capacity controls drift')
    observer=load(OBSERVER_ROOT/'result.json')
    if (observer['status']!='PASS' or observer['cases']!=8 or observer['new_solves']!=0 or
        set(observer['sources'])!={'scripts/test_h2_capacity_observation.py','scripts/h2_capacity_build.py'}):
        raise ValueError('observation control gate')
    for name,digest in observer['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('observer implementation drift')
    return {'controls_sha256':sha(CONTROL_ARCHIVE),'observer_sha256':sha(OBSERVER_ROOT/'result.json')}


def sources():
    result=producer_sources(specification('full_size'))
    for name in ('scripts/h2_capacity_full.py','scripts/archive_h2_capacity_execution.py',
                 'scripts/test_h2_capacity_execution.py','scripts/test_h2_capacity_observation.py'):
        result[name]=sha(ROOT/name)
    return dict(sorted(result.items()))


def run():
    # Administrative gate precedes the two separately charged API budgets.
    g=gate()
    git=lambda *args: subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('branch','--show-current')!='feat/moe-route-verification' or git('status','--porcelain'):
        raise ValueError('clean research branch required')
    head=git('rev-parse','HEAD')
    if head!=git('rev-parse','@{upstream}'): raise ValueError('execution version must already be pushed')
    CALL_ROOT.mkdir(parents=True,exist_ok=False); names=sources()
    for name in names:
        target=CALL_ROOT/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(CALL_ROOT/'implementation.json',names)
    save(CALL_ROOT/'execution.json',{'schema':'H2_CAPACITY_FULL_EXECUTION_V1','head':head,
        **g,'implementation_sha256':sha(CALL_ROOT/'implementation.json'),
        'arms':list(ARMS),'budget_per_arm':300,'rss_limit_per_arm':2**31,
        'required_pairs_per_arm':28,'required_properties_per_arm':252,
        'real_requests':0,'administrative_gate_is_not_free_source_preparation':True})
    for mode in ARMS:
        value=supervise(CALL_ROOT/mode,specification('full_size',mode),budget=300,rss_limit=2**31)
        save(CALL_ROOT/mode/'caller_observation.json',value)
        print(json.dumps({'mode':mode,**value}),flush=True)


def events(path):
    """Diagnostic prefix only. Never upgrades partial output to accepted evidence."""
    if not path.exists(): return {'sha256':None,'records':0,'counts':{},'last':None,'truncated_final_line':False}
    raw=path.read_bytes(); parsed=[]; lines=raw.splitlines(); truncated=False
    for i,line in enumerate(lines):
        try: parsed.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1 or raw.endswith(b'\n'): raise ValueError('nonterminal malformed trace')
            truncated=True
    counts=Counter(e['operation'] for e in parsed if e['event']=='TRACE')
    # Nested EXIT event costs are not added to process costs.
    top=[e for e in parsed if e['event'] in ('ENTER','EXIT','EXIT_ERROR')]
    return {'sha256':sha(path),'records':len(parsed),'counts':dict(counts),
        'last':parsed[-1] if parsed else None,'truncated_final_line':truncated,'operations':top,
        'native_call_boundary_is_not_completed_call':True,
        'property_record_complete_can_have_missing_certificate':True}


def derive():
    execution=load(CALL_ROOT/'execution.json'); names=load(CALL_ROOT/'implementation.json',required_sha(execution['implementation_sha256']))
    if (names!=sources() or execution['arms']!=list(ARMS) or execution['budget_per_arm']!=300 or
        execution['rss_limit_per_arm']!=2**31 or execution['required_pairs_per_arm']!=28 or
        execution['required_properties_per_arm']!=252 or execution['real_requests']!=0):
        raise ValueError('frozen full execution inventory')
    if any(execution[k]!=v for k,v in gate().items()): raise ValueError('executed control gate drift')
    for name,digest in names.items():
        if sha(CALL_ROOT/'implementation'/name)!=digest: raise ValueError('snapshot drift')
    observed_dirs={p.name for p in CALL_ROOT.iterdir() if p.is_dir() and (p/'caller_observation.json').exists()}
    if observed_dirs!=set(ARMS): raise ValueError('missing/extra registered full calls')
    calls=[]
    for mode in ARMS:
        path=CALL_ROOT/mode; call=load(path/'caller_observation.json'); spec=load(path/'spec.json')
        if spec!=specification('full_size',mode): raise ValueError('full arm specification')
        checked=audit(path,call); terminal=load(path/'terminal.json')
        partial={'files':sum(p.is_file() for p in path.rglob('*')),
                 'logical_bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file())}
        math_recheck=None
        if checked['pipeline_complete']:
            b=load(path/'built.json')
            out=subprocess.run([PYTHON,'-B','-I','-S',str(path/'bundle/verify.py'),
                '--manifest-sha',b['sha256'],'--source-sha',spec['source_manifest_sha256'],
                '--proof-sha',b['proof_manifest_sha256'],'--mode',mode],cwd=CALL_ROOT,
                capture_output=True,text=True,timeout=300)
            if out.returncode: raise ValueError('completed full math recheck failed: '+out.stderr)
            math_recheck=json.loads(out.stdout)['result']
            if math_recheck!=load(path/'accepted.json')['result']: raise ValueError('full math drift')
        calls.append({'mode':mode,'call':call,'audit':checked,'terminal_sha256':sha(path/'terminal.json'),
            'caller_observation_sha256':sha(path/'caller_observation.json'),
            'stage_seconds':terminal['stage_seconds'],'parent_publication_seconds':call['seconds']-terminal['stage_seconds'],
            'sampled_peak_rss':max((s['sampled_peak_rss'] for s in terminal['stages']),default=0),
            'partial_files_are_not_accepted_evidence':partial,
            'events':{p:events(path/(p+'_events.jsonl')) for p in ('produce','check','receive')},
            'mathematical_recheck':math_recheck})
    return {'schema':'H2_FULL_CAPACITY_ARCHIVE_V1','execution':execution,'calls':calls,
        'pipeline_completed_arms':sum(c['audit']['pipeline_complete'] for c in calls),
        'fully_checked_arms':sum(c['audit']['all_obligations_checked'] for c in calls),
        'complete_declared_source_positive_arms':sum(c['audit']['positive_execution_accepted'] for c in calls),
        'real_model_admitted':False,'trained_model':False,'native_float_SAFE':False,
        'scope':'single fixed full-size synthetic recipe, not a method efficacy or speed comparison',
        'stop':'no larger limit, rerun, easier source or partial-proof upgrade; inspect retained stop evidence first'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run',action='store_true'); g.add_argument('--output',type=Path); g.add_argument('--check',type=Path)
    a=p.parse_args()
    if a.run: run()
    else:
        result=derive()
        if a.output: save(a.output,result)
        if a.check and load(a.check)!=result: raise ValueError('full archive drift')
        print(json.dumps({k:v for k,v in result.items() if k not in ('calls','execution')},sort_keys=True))
