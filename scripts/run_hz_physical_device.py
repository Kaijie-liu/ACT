"""Archive CPU wiring controls; run a separately frozen tiny physical batch."""
import argparse
import ast
from fractions import Fraction
import io
import os
from pathlib import Path
import time
import unittest
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.owned_bounded import execute
from scoped_proof import device_lifecycle as engine
from source_enclosure.format import identity
from scripts import hz_physical_device as policy
from scripts import hz_device_admission as admission
from scripts.hz_device_lifecycle import cleanup_confirmed
from scripts.hz_propagation_supervised import finite, required_hash


def control_names():
    from scripts.test_hz_physical_device import TEST_NAMES
    tree=ast.parse((ROOT/'scripts/test_hz_physical_device.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PhysicalWiringControls')
    actual=sorted(x.name for x in cls.body if isinstance(x,ast.FunctionDef) and x.name.startswith('test_'))
    if actual!=sorted(TEST_NAMES): raise ValueError('fixed control names')
    return actual


def controls(root):
    os.environ['HZ_PHYSICAL_CONTROLS_ROOT']=str(root)
    from scripts.test_hz_physical_device import PhysicalWiringControls
    stream=io.StringIO(); begin=time.monotonic()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhysicalWiringControls))
    (root/'tests.log').write_text(stream.getvalue())
    passed=result.wasSuccessful() and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
    summary={'status':'PASS' if passed else 'FAIL','tests':result.testsRun,'calls':len(load(root/'calls.json')),
        'names':control_names(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'expected_failures':len(result.expectedFailures),'unexpected_successes':len(result.unexpectedSuccesses),
        'implementation_sha256':sha(root/'implementation.json'),'calls_sha256':sha(root/'calls.json'),
        'tests_sha256':sha(root/'tests.log'),'seconds':time.monotonic()-begin,'physical_cuda_calls':0}
    save(root/'summary.json',summary); print(stream.getvalue()); print(summary)
    return 0 if passed else 1


def audit_controls(root):
    from scripts.test_hz_physical_device import EXTRA,roster,fault_witness
    summary=load(root/'summary.json'); bindings=load(root/'implementation.json',summary['implementation_sha256'])
    if (summary['status']!='PASS' or summary['tests']!=8 or summary['names']!=control_names() or summary['calls']!=6
            or any(summary[k] for k in ('failures','errors','skipped','expected_failures','unexpected_successes','physical_cuda_calls'))
            or summary['calls_sha256']!=sha(root/'calls.json') or summary['tests_sha256']!=sha(root/'tests.log')
            or set(bindings)!=set((*policy.FILES,*EXTRA))): raise ValueError('incomplete controls')
    for name,digest in bindings.items():
        if sha(ROOT/name)!=digest or sha(root/'implementation'/name)!=digest: raise ValueError('controlled source changed: '+name)
    calls=load(root/'calls.json')
    if set(calls)!=set(roster()): raise ValueError('control call roster')
    checked={}
    for name,(spec,b,status) in roster().items():
        path=root/name; inv=load(path/'invocation.json'); obs=calls[name]
        if inv['spec']!=spec or inv['budget']!=b or obs!=load(root/(name+'_observed.json')) or obs['result']['status']!=status:
            raise ValueError('control identity/result')
        checked[name]=engine.audit(policy,path,observation=obs,recheck=status==policy.DONE); fault_witness(path,name)
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),
            'implementation_sha256':summary['implementation_sha256'],'tests':8,'calls':checked,'physical_cuda_calls':0}


def preflight(root):
    root.mkdir(); begin=time.monotonic(); end=begin+3; inv=uuid.uuid4().hex; gpu=policy.protocol()['gpu_uuid']
    launch={'invocation':inv,'gpu_uuid':gpu,'start':begin,'run_deadline':end-.25,'cleanup_deadline':end,
            'cuda_visible_devices':'','executable':PYTHON,'worker':policy.WORKER,'rss_limit':2*2**30}
    lsha=save(root/'launch.json',launch)['sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r=execute([PYTHON,'-B','-S','-m',policy.WORKER,str(root),'--preflight','--deadline',str(end-.25),
               '--invocation',inv,'--gpu-uuid',gpu],root/'probe.log',run_deadline=end-.25,cleanup_deadline=end,env=env,rss_limit=2*2**30)
    ready=False; record=None; digest=None; validation_error=None
    if r['status']=='COMPLETED' and cleanup_confirmed(r):
        try:
            digest=sha(root/'probe.json'); record=load(root/'probe.json',digest)
            admission.validate(record,inv,{'device':'cuda:0','gpu_uuid':gpu,'control':''},end-.25)
            if any(x['start']<begin for x in record['observations']): raise ValueError('stale batch preflight')
            ready=record['status']=='READY'
        except Exception as exc: validation_error=repr(exc)
    seconds=time.monotonic()-begin; ready=ready and seconds<3
    value={'ready':ready,'launch_sha256':lsha,'execution':r,'probe_sha256':digest,'seconds':seconds,'validation_error':validation_error,
           'inventory':{p.name:sha(p) for p in root.iterdir() if p.is_file()}}
    digest=save(root/'result.json',value)['sha256']
    returned=time.monotonic()-begin
    return {'ready':ready and returned<3,'seconds':returned,'result_sha256':digest,'validation_error':validation_error}


def audit_preflight(root,observation):
    value=load(root/'result.json',required_hash(observation['result_sha256']),engine.LIMIT)
    launch=load(root/'launch.json',value['launch_sha256']); r=value['execution']
    names={p.name for p in root.iterdir() if p.is_file()}-{'result.json'}
    if value['inventory']!={n:sha(root/n) for n in names}: raise ValueError('preflight prefix inventory')
    for v in (launch['start'],launch['run_deadline'],launch['cleanup_deadline'],value['seconds'],observation['seconds'],
              r['seconds'],r['execution_seconds'],r['cleanup_seconds'],r['sampled_peak_rss']): finite(v)
    if (launch['gpu_uuid']!=policy.protocol()['gpu_uuid'] or launch['cuda_visible_devices']!=''
            or launch['executable']!=PYTHON or launch['worker']!=policy.WORKER or launch['rss_limit']!=2*2**30
            or launch['cleanup_deadline']!=launch['start']+3 or launch['run_deadline']!=launch['cleanup_deadline']-.25
            or r['run_deadline']!=launch['run_deadline'] or r['cleanup_deadline']!=launch['cleanup_deadline']
            or abs(r['seconds']-r['execution_seconds']-r['cleanup_seconds'])>1e-8
            or value['seconds']<r['seconds'] or observation['seconds']<value['seconds']
            or r['escaped_descendants_or_driver_cleanup'] is not False): raise ValueError('preflight clock/environment')
    if r['cleanup_status']!='CLEANUP_INCOMPLETE' and not cleanup_confirmed(r): raise ValueError('preflight cleanup contradiction')
    ready=False
    if r['status']=='COMPLETED':
        finite(r['exit_observed_at'])
        if (not cleanup_confirmed(r) or r['returncode']!=0 or type(r['pid']) is not int or r['pid']<=0
                or r['descendant_on_leader_exit'] is not False or r['sampled_peak_rss']>2*2**30
                or not launch['start']<=r['exit_observed_at']<launch['run_deadline']): raise ValueError('preflight completion')
        data=load(root/'probe.json',required_hash(value['probe_sha256']),engine.LIMIT)
        admission.validate(data,launch['invocation'],{'device':'cuda:0','gpu_uuid':launch['gpu_uuid'],'control':''},launch['run_deadline'])
        if any(o['start']<launch['start'] for o in data['observations']): raise ValueError('stale preflight')
        ready=data['status']=='READY' and value['seconds']<3
    elif value['probe_sha256'] is not None: raise ValueError('incomplete preflight accepted output')
    if value['ready']!=ready or observation['ready']!=(ready and observation['seconds']<3): raise ValueError('preflight decision')
    return {'status':'PASS','ready':observation['ready'],'seconds':observation['seconds'],
            'cleanup_confirmed':cleanup_confirmed(r),'proves_exclusive_reservation':False}


def batch(root,permit):
    # Parent controls are re-audited before any resource probe or timed call.
    freeze=load(permit['path'],permit['sha256']); audit_controls(Path(freeze['controls_root']))
    for row in policy.physical_roster(): policy.validate_spec(policy.spec(**row),permit)
    root.mkdir(parents=True,exist_ok=False); save(root/'freeze.json',freeze); save(root/'permit.json',permit)
    save(root/'implementation.json',policy.sources()); begin=time.monotonic(); fatal=None
    save(root/'batch_start.json',{'begin':begin,'planned_calls':8,'freeze_sha256':permit['sha256']})
    try: gate=preflight(root/'preflight')
    except Exception as exc:
        fatal=repr(exc); gate={'ready':False,'seconds':time.monotonic()-begin,'error':fatal}
    if gate.get('validation_error'): fatal=gate['validation_error']
    gate['api_return_at']=time.monotonic()
    save(root/'preflight_observed.json',gate); calls=[]; roster=policy.physical_roster()
    if gate['ready']:
        for i,row in enumerate(roster):
            s=policy.spec(**row); begin=time.monotonic()
            save(root/f'call_{i:02}_started.json',{'slot':i,'spec':s,'begin':begin,'preflight_sha256':sha(root/'preflight_observed.json')})
            try:
                result=engine.supervise(policy,root/f'call_{i:02}',s,spec_sha256=identity(s),permit=permit)
                obs={'begin':begin,'end':time.monotonic(),'result':result}
            except Exception as exc:
                fatal=repr(exc); obs={'begin':begin,'end':time.monotonic(),'result':None,'error':fatal}
            save(root/f'call_{i:02}_observed.json',obs)
            audit_begin=time.monotonic()
            try:
                if obs['result'] is None: raise RuntimeError('request did not return a complete receipt')
                checked=engine.audit(policy,root/f'call_{i:02}',observation=obs,recheck=True)
            except Exception as exc:
                fatal=repr(exc); checked={'status':'FAIL','error':fatal}
            calls.append({'slot':i,**row,'observation':obs,'audit':checked,'post_return_audit_seconds':time.monotonic()-audit_begin})
            save(root/f'call_{i:02}_audited.json',calls[-1])
            if fatal or obs['result']['status']!=policy.DONE: break
    summary={'schema':'HZ_PHYSICAL_BATCH_V1','status':'COMPLETED' if len(calls)==8 and not fatal and all(c['observation']['result']['status']==policy.DONE for c in calls)
        else 'NOT_ADMITTED' if not gate['ready'] else 'STOPPED_WITH_PREFIX','planned_calls':8,'calls':calls,
        'pending_slots':list(range(len(calls),8)),'preflight':gate,'fatal_error':fatal,'complete_moe_proofs':0,'real_requests':0,
        'inventory':{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}}
    save(root/'summary.json',summary); print({k:v for k,v in summary.items() if k!='calls'})
    return summary


def audit_batch(root):
    permit=load(root/'permit.json'); freeze=load(permit['path'],required_hash(permit['sha256']))
    audit_controls(Path(freeze['controls_root']))
    if load(root/'freeze.json')!=freeze or load(root/'implementation.json')!=policy.sources(): raise ValueError('batch freeze/source binding')
    for row in policy.physical_roster(): policy.validate_spec(policy.spec(**row),permit)
    summary=load(root/'summary.json'); gate=load(root/'preflight_observed.json')
    if summary['inventory']!={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'summary.json'}:
        raise ValueError('batch full prefix inventory')
    if summary['fatal_error'] is not None:
        # Preserve accounting but never call an unauditable physical prefix PASS.
        return {'status':'FAIL','execution_status':summary['status'],'root':str(root),'fatal_error':summary['fatal_error'],
                'summary_sha256':sha(root/'summary.json'),'started_calls':len(summary['calls']),
                'pending_slots':summary['pending_slots'],'complete_moe_proofs':0,'gpu_speedup_established':False}
    preflight_audit=audit_preflight(root/'preflight',gate); calls=summary['calls']; roster=policy.physical_roster()
    if (summary['schema']!='HZ_PHYSICAL_BATCH_V1' or summary['planned_calls']!=8 or summary['preflight']!=gate
            or summary['pending_slots']!=list(range(len(calls),8)) or len(calls)>8
            or summary['complete_moe_proofs']!=0 or summary['real_requests']!=0): raise ValueError('batch denominator')
    if {p.name for p in root.glob('call_*') if p.is_dir()}!={f'call_{i:02}' for i in range(len(calls))}:
        raise ValueError('unrecorded/deleted call directory')
    if not gate['ready'] and calls: raise ValueError('batch ran without preflight')
    rows=[]; pair={}; intents=0; successes=0; previous=finite(gate['api_return_at'])
    launch=load(root/'preflight'/'launch.json')
    if previous<launch['start']+gate['seconds']: raise ValueError('preflight return chronology')
    for i,c in enumerate(calls):
        path=root/f'call_{i:02}'; inv=load(path/'invocation.json'); obs=load(root/f'call_{i:02}_observed.json')
        start=load(root/f'call_{i:02}_started.json')
        if (start!={'slot':i,'spec':policy.spec(**roster[i]),'begin':obs['begin'],
                    'preflight_sha256':sha(root/'preflight_observed.json')} or obs['begin']<previous):
            raise ValueError('sequential slot/preflight binding')
        previous=obs['end']
        if (c['slot']!=i or {k:c[k] for k in ('case','device')}!=roster[i]
                or inv['spec']!=policy.spec(**roster[i]) or inv['permit']!=permit or inv['budget']!=30
                or c['observation']!=obs or c!=load(root/f'call_{i:02}_audited.json')): raise ValueError('physical slot identity')
        finite(c['post_return_audit_seconds'])
        checked=engine.audit(policy,path,observation=obs,recheck=True)
        if c['audit']!=checked: raise ValueError('batch audit differs')
        if i<len(calls)-1 and obs['result']['status']!=policy.DONE: raise ValueError('continued after failure')
        term=load(path/'terminal.json'); cost=None; bounds=None; hardware=None; allocator=None
        if (path/'cuda_intent.json').exists(): intents+=1
        if obs['result']['status']==policy.DONE:
            p=load(path/'produce.json'); cost=p['candidates']['cost_seconds']; hardware=p['candidates']['hardware']
            allocator=p['candidates']['allocator_memory']; bounds=term['accepted']['checked']['results']
            pair.setdefault(c['case'],{})[c['device']]=bounds
            if c['device']=='cuda:0': successes+=1
        rows.append({'slot':i,**roster[i],'status':obs['result']['status'],'seconds':obs['result']['seconds'],
                     'phase_seconds':{s['phase']:s['seconds'] for s in term['stages']},'nested_candidate_seconds':cost,
                     'post_return_audit_seconds':c['post_return_audit_seconds'],
                     'hardware':hardware,'allocator':allocator,'bounds':bounds,'release':term['release']})
    expected=('NOT_ADMITTED' if not gate['ready'] else 'COMPLETED' if len(calls)==8 and all(c['observation']['result']['status']==policy.DONE for c in calls)
              else 'STOPPED_WITH_PREFIX')
    if summary['status']!=expected: raise ValueError('batch terminal')
    diffs={}
    for case,arms in pair.items():
        if set(arms)=={'cpu','cuda:0'}:
            if [(x['id'],x['side'],x['lp_sha256']) for x in arms['cpu']]!=[(x['id'],x['side'],x['lp_sha256']) for x in arms['cuda:0']]:
                raise ValueError('CPU/GPU mathematical obligation mismatch')
            diffs[case]=[{'id':a['id'],'side':a['side'],'lp_sha256':a['lp_sha256'],'cpu_bound':a['bound'],'gpu_bound':b['bound'],
                         'gpu_minus_cpu':str(Fraction(b['bound'])-Fraction(a['bound']))}
                         for a,b in zip(arms['cpu'],arms['cuda:0'])]
    return {'schema':'HZ_PHYSICAL_BATCH_AUDIT_V1','status':'PASS','execution_status':expected,'root':str(root),
            'freeze_sha256':permit['sha256'],'summary_sha256':sha(root/'summary.json'),'preflight':preflight_audit,
            'planned_calls':8,'started_calls':len(calls),'pending_slots':summary['pending_slots'],'rows':rows,
            'cuda_initialization_intents':intents,'complete_cuda_calls':successes,'paired_bound_differences':diffs,
            'complete_moe_proofs':0,'native_solves':0,'real_requests':0,'gpu_speedup_established':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['controls','audit-controls','batch','audit-batch'])
    p.add_argument('root',type=Path); p.add_argument('--freeze',type=Path); p.add_argument('--freeze-sha'); p.add_argument('--report',type=Path)
    a=p.parse_args(); root=a.root.resolve()
    if not root.is_relative_to(ROOT.parent/'baseline_runs'): raise ValueError('project archive only')
    if a.action=='controls': raise SystemExit(controls(root))
    if a.action in ('audit-controls','audit-batch'):
        value=audit_controls(root) if a.action=='audit-controls' else audit_batch(root)
        if a.report: save(a.report,value)
        print(value)
    else: batch(root,{'path':str(a.freeze.resolve()),'sha256':a.freeze_sha})
