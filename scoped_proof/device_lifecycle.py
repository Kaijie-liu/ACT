"""Explicit-policy lifecycle mechanics; no numerical or CUDA operations here.

The frozen V1 CPU supervisor remains untouched. Policies must supply a fixed
worker, binding, phase deadlines and admission/release/reception contracts.
"""
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.owned_bounded import execute
from scripts.hz_device_lifecycle import cleanup_confirmed, finalize_status
from scripts.hz_propagation_supervised import finite, required_hash
from source_enclosure.format import identity

PHASES=('admit','produce','release','check','receive')
LIMIT=4*2**20


def bind(policy, root, digest):
    inv=load(root/'invocation.json',required_hash(digest),LIMIT)
    policy.validate_spec(inv['spec'],inv['permit'])
    for k in ('start','budget','deadline','work'): finite(inv[k])
    if (inv['sources']!=policy.sources() or inv['budget']!=policy.budget(inv['spec'])
            or inv['deadline']!=inv['start']+inv['budget'] or inv['work']!=inv['deadline']-1
            or inv['rss_limit']!=2*2**30): raise ValueError('request/source/clock binding')
    return inv


def plan(policy,root,inv_sha,plan_sha):
    inv=bind(policy,root,inv_sha); matches=[]
    for phase in PHASES:
        path=root/(phase+'_plan.json')
        if path.is_file() and sha(path)==required_hash(plan_sha): matches.append(load(path,plan_sha,LIMIT))
    if len(matches)!=1: raise ValueError('unique parent launch plan')
    p=matches[0]; finite(p['begin'])
    if (p['invocation_sha256']!=inv_sha or p['phase'] not in PHASES or p['begin']<inv['start']
            or p['cleanup_deadline']!=policy.cutoff(inv,p['phase'],p['begin'])
            or p['run_deadline']!=p['cleanup_deadline']-.25): raise ValueError('launch plan clock')
    return inv,p


def supervise(policy,root,spec,*,spec_sha256,permit=None):
    start=time.monotonic(); policy.validate_spec(spec,permit)
    if identity(spec)!=required_hash(spec_sha256): raise ValueError('caller spec hash')
    root=Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project archive')
    root.mkdir(parents=True,exist_ok=False); budget=policy.budget(spec)
    inv={'spec':spec,'permit':permit,'invocation':uuid.uuid4().hex,'start':start,'budget':budget,
         'deadline':start+budget,'work':start+budget-1,'rss_limit':2*2**30,'sources':policy.sources()}
    inv_sha=save(root/'invocation.json',inv)['sha256']; stages=[]

    def call(phase,inputs):
        begin=time.monotonic(); end=policy.cutoff(inv,phase,begin)
        p={'phase':phase,'invocation_sha256':inv_sha,'begin':begin,'run_deadline':end-.25,
           'cleanup_deadline':end,'inputs':inputs}
        psha=save(root/(phase+'_plan.json'),p)['sha256']; visible=policy.visible(inv,phase)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=visible)
        cmd=[PYTHON,'-B']+(['-S'] if phase in ('admit','release','receive') else [])+[
            '-m',policy.WORKER,str(root),'--invocation-sha',inv_sha,'--plan-sha',psha]
        r=execute(cmd,root/(phase+'.log'),run_deadline=p['run_deadline'],cleanup_deadline=end,
                  env=env,rss_limit=inv['rss_limit'])
        r.update(phase=phase,plan_sha256=psha,start_seconds=begin-start,executable=PYTHON,
                 cpu_threads=1,cuda_visible_devices=visible)
        if r['status']=='COMPLETED':
            try:
                load(root/(phase+'.json'),limit=LIMIT); r['output_sha256']=sha(root/(phase+'.json'))
            except Exception as exc: r['status']='ERROR'; r['output_error']=repr(exc)
        r['end_seconds']=time.monotonic()-start
        if r['status']=='COMPLETED' and start+r['end_seconds']>=end: r['status']='TIMEOUT'
        stages.append(r); save(root/(phase+'_stage.json'),r)
        return r

    status,error,release,accepted='ERROR',None,None,None
    try:
        a=call('admit',{})
        if a['status']=='COMPLETED' and cleanup_confirmed(a) and policy.admitted(root,inv,a):
            p=call('produce',{'admit':a['output_sha256']})
            if cleanup_confirmed(p):
                r=call('release',{'produce':sha(root/'produce_stage.json')})
                release=policy.release_gate(root,inv,{s['phase']:s for s in stages})
                if p['status']=='COMPLETED' and release.get('release_confirmed') and policy.produced(root,inv,p):
                    c=call('check',{'produce':p['output_sha256'],'release':r['output_sha256']})
                    if c['status']=='COMPLETED' and cleanup_confirmed(c):
                        call('receive',{'produce':p['output_sha256'],'release':r['output_sha256'],'check':c['output_sha256']})
        status,release,accepted=policy.decision(root,inv,stages)
    except Exception as exc:
        error=repr(exc)
        if any(not cleanup_confirmed(s) for s in stages): status='CLEANUP_INCOMPLETE'
        elif any(s['phase']=='produce' for s in stages) and not (release or {}).get('release_confirmed'):
            status='DEVICE_RELEASE_UNCONFIRMED'
    inventory={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    now=time.monotonic(); status=finalize_status(status,now,inv['work'])
    term={'invocation_sha256':inv_sha,'status':status,'error':error,'release':release,
          'accepted':accepted if status==policy.DONE else None,'stages':stages,'inventory':inventory,
          'seconds':now-start,'stage_seconds':sum(s['seconds'] for s in stages)}
    term['parent_seconds']=term['seconds']-term['stage_seconds']; tsha=save(root/'terminal.json',term)['sha256']
    now=time.monotonic(); status=finalize_status(status,now,inv['deadline'])
    fsha=save(root/'finish.json',{'terminal_sha256':tsha,'status':status,'seconds':now-start})['sha256']
    now=time.monotonic(); status=finalize_status(status,now,inv['deadline'])
    return {'status':status,'invocation_sha256':inv_sha,'finish_sha256':fsha,'seconds':now-start,
            'complete_moe_proof':False}


def audit(policy,root,*,observation=None,recheck=False):
    root=Path(root); inv_sha=sha(root/'invocation.json'); inv=bind(policy,root,inv_sha)
    f=load(root/'finish.json'); t=load(root/'terminal.json',required_hash(f['terminal_sha256']))
    names={p.name for p in root.iterdir() if p.is_file()}-{'terminal.json','finish.json'}
    if t['invocation_sha256']!=inv_sha or t['inventory']!={n:sha(root/n) for n in names}:
        raise ValueError('complete terminal/prefix inventory')
    stages=t['stages']; phases=[s['phase'] for s in stages]; anchors={}; previous=0
    if not phases or phases!=list(PHASES[:len(phases)]): raise ValueError('phase prefix')
    if {p.name for p in root.glob('*_stage.json')}!={p+'_stage.json' for p in phases}:
        raise ValueError('stage completeness')
    for i,s in enumerate(stages):
        phase=s['phase']; _,p=plan(policy,root,inv_sha,s['plan_sha256'])
        for k in ('seconds','execution_seconds','cleanup_seconds','start_seconds','end_seconds','sampled_peak_rss'): finite(s[k])
        if (load(root/(phase+'_stage.json'))!=s or p['phase']!=phase or s['start_seconds']<previous
                or abs(p['begin']-inv['start']-s['start_seconds'])>1e-8
                or s['end_seconds']<s['start_seconds']+s['seconds']-1e-8
                or abs(s['seconds']-s['execution_seconds']-s['cleanup_seconds'])>1e-8
                or s['run_deadline']!=p['run_deadline'] or s['cleanup_deadline']!=p['cleanup_deadline']
                or s['cuda_visible_devices']!=policy.visible(inv,phase) or s['cpu_threads']!=1
                or s['executable']!=PYTHON or s['escaped_descendants_or_driver_cleanup'] is not False):
            raise ValueError('stage cost/identity/environment')
        expected=({} if phase=='admit' else {'admit':anchors['admit']} if phase=='produce' else
                  {'produce':sha(root/'produce_stage.json')} if phase=='release' else
                  {k:anchors[k] for k in ('produce','release')+(('check',) if phase=='receive' else ())})
        if p['inputs']!=expected: raise ValueError('parent launch input anchor')
        if 'output_sha256' in s:
            load(root/(phase+'.json'),required_hash(s['output_sha256']),LIMIT); anchors[phase]=s['output_sha256']
        if s['cleanup_status']!='CLEANUP_INCOMPLETE' and not cleanup_confirmed(s): raise ValueError('cleanup contradiction')
        if s['status']=='COMPLETED':
            finite(s['exit_observed_at']); required_hash(s.get('output_sha256'))
            if (not cleanup_confirmed(s) or type(s['pid']) is not int or s['pid']<=0 or s['returncode']!=0
                    or s['descendant_on_leader_exit'] is not False or s['sampled_peak_rss']>inv['rss_limit']
                    or not p['begin']<=s['exit_observed_at']<p['run_deadline']
                    or inv['start']+s['end_seconds']>=p['cleanup_deadline']): raise ValueError('successful stage unresolved')
        elif i!=len(stages)-1 and not (phase=='produce' and phases[i+1:]==['release']):
            raise ValueError('continued after failed phase')
        if not cleanup_confirmed(s) and i!=len(stages)-1: raise ValueError('continued after unresolved cleanup')
        previous=s['end_seconds']
    for v in (t['seconds'],t['stage_seconds'],t['parent_seconds'],f['seconds']): finite(v)
    if (t['seconds']<previous or f['seconds']<t['seconds']
            or abs(t['stage_seconds']-sum(s['seconds'] for s in stages))>1e-8
            or abs(t['seconds']-t['stage_seconds']-t['parent_seconds'])>1e-8): raise ValueError('whole cost')
    status,release,accepted=policy.decision(root,inv,stages)
    status=finalize_status(status,inv['start']+t['seconds'],inv['work'])
    if (t['status']!=status or t['release']!=release or t['accepted']!=(accepted if status==policy.DONE else None)
            or f['status']!=finalize_status(status,inv['start']+f['seconds'],inv['deadline'])):
        raise ValueError('terminal derivation')
    bounds=0
    if recheck and status==policy.DONE:
        result=policy.check(root,inv,anchors['produce'],time.monotonic()+30)
        if result!=load(root/'check.json',anchors['check']): raise ValueError('independent exact recheck')
        bounds=len(result['result']['results'])
    observed=None
    if observation is not None:
        r=observation['result']
        for v in (observation['begin'],observation['end'],r['seconds']): finite(v)
        if (r['invocation_sha256']!=inv_sha or r['finish_sha256']!=sha(root/'finish.json')
                or observation['begin']>inv['start'] or observation['end']-inv['start']<r['seconds']
                or r['seconds']<f['seconds'] or r['complete_moe_proof'] is not False): raise ValueError('API anchor/cost')
        observed=finalize_status(f['status'],observation['end'],inv['deadline'])
        if r['status']!=observed: raise ValueError('late API acceptance')
    return {'status':'PASS','execution_status':observed,'bounds_rechecked':bounds,
            'budget_acceptance_observed':observed==policy.DONE,'release':release,'complete_moe_proof':False}
